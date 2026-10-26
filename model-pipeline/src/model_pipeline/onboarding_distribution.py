"""Committed-source, package-only, network-isolated candidate consumer evidence.

This optional stage does not approve a release or adopt protected fixtures. Temporary
checkouts, isolated package caches and model copies are removed after each attempt.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

from model_pipeline import fixtures
from model_pipeline.onboarding_candidate import code_inputs, load_candidate, source_reader, write_json
from model_pipeline.onboarding_e5 import _assert_committed_code, _probe
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command
from model_pipeline.tokenizer_package import TOKENIZER, tokenizer_content_hash, tokenizer_version

# Only this repository's packages are packed; the tokenizer is the published Starlings.Tokenizers (tokenizer_package).
PACKAGES = {
    "Starlings.Acquisition": "ai-dotnet/src/Starlings.Acquisition",
    "Starlings.Runtime.Onnx": "ai-dotnet/src/Starlings.Runtime.Onnx",
    "Starlings.Generation.Abstractions": "ai-dotnet/src/Starlings.Generation.Abstractions",
    "Starlings.Generation.OnnxGenAI": "ai-dotnet/src/Starlings.Generation.OnnxGenAI",
    "Starlings.Models.Bert": "ai-dotnet/src/Starlings.Models.Bert",
    "Starlings.Models.QwenChat": "ai-dotnet/src/Starlings.Models.QwenChat",
    "Starlings.Models.QwenChat.OnnxGenAI": "ai-dotnet/src/Starlings.Models.QwenChat.OnnxGenAI",
}
# The typed model package the candidate consumer references, per backend.
TYPED_MODEL = {"onnx": "Starlings.Models.Bert", "ortgenai": "Starlings.Models.QwenChat.OnnxGenAI"}
TOOLS = {
    "onnx": "ai-dotnet/tools/Starlings.Bert.Fidelity.Tool/Program.cs",
    "ortgenai": "ai-dotnet/tools/Starlings.Generation.Fidelity/Program.cs",
}
# Every fidelity tool links the shared entry point (ai-dotnet/tools/Shared/ToolMain.cs, #37); the consumer compiles
# it next to Program.cs.
TOOL_MAIN = "ai-dotnet/tools/Shared/ToolMain.cs"
NATIVE = {"onnx": ("Microsoft.ML.OnnxRuntime", "1.30.0"), "ortgenai": ("Microsoft.ML.OnnxRuntimeGenAI", "0.17.1")}


def _head(root: Path, timeout: float) -> str:
    result = _probe(["git", "rev-parse", "HEAD"], root=root, timeout=timeout)
    value = result.get("stdout", "")
    if result["status"] != "available" or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise PipelineError("candidate distribution requires a committed Git HEAD")
    return value


def _versions(root: Path) -> dict[str, str]:
    document = ET.parse(root / "ai-dotnet/Directory.Build.props")
    prefix, suffix = document.findtext(".//VersionPrefix"), document.findtext(".//VersionSuffix")
    if not prefix:
        raise PipelineError("missing committed package version for ai-dotnet")
    version = f"{prefix}-{suffix}" if suffix else prefix
    return {**{name: version for name in PACKAGES}, TOKENIZER: tokenizer_version(root)}


def _dependencies(versions: dict[str, str]) -> dict[str, dict[str, str]]:
    exact = {name: f"[{version}]" for name, version in versions.items()}
    return {
        "Starlings.Acquisition": {},
        "Starlings.Runtime.Onnx": {"Microsoft.ML.OnnxRuntime.Managed": "[1.30.0]"},
        "Starlings.Generation.Abstractions": {
            "Microsoft.Extensions.AI.Abstractions": "10.10.1",
            "Starlings.Tokenizers": exact["Starlings.Tokenizers"],
        },
        "Starlings.Generation.OnnxGenAI": {
            "Microsoft.ML.OnnxRuntimeGenAI.Managed": "[0.17.1]",
            "Starlings.Generation.Abstractions": exact["Starlings.Generation.Abstractions"],
            "Starlings.Tokenizers": exact["Starlings.Tokenizers"],
        },
        "Starlings.Models.Bert": {
            name: exact[name]
            for name in (
                "Starlings.Runtime.Onnx",
                "Starlings.Acquisition",
                "Starlings.Tokenizers",
            )
        },
        "Starlings.Models.QwenChat": {
            name: exact[name] for name in ("Starlings.Generation.Abstractions", "Starlings.Tokenizers")
        },
        "Starlings.Models.QwenChat.OnnxGenAI": {
            name: exact[name]
            for name in (
                "Starlings.Acquisition",
                "Starlings.Generation.OnnxGenAI",
                "Starlings.Models.QwenChat",
                "Starlings.Tokenizers",
            )
        },
    }


def _inspect_packages(feed: Path, versions: dict[str, str], commit: str) -> list[dict]:
    expected = _dependencies(versions)
    records = []
    for name in PACKAGES:
        version = versions[name]
        path = feed / f"{name}.{version}.nupkg"
        if not path.is_file():
            raise PipelineError(f"missing candidate package: {path.name}")
        with zipfile.ZipFile(path) as archive:
            manifests = [entry for entry in archive.namelist() if entry.endswith(".nuspec")]
            if len(manifests) != 1:
                raise PipelineError(f"{name}: expected one nuspec")
            document = ET.fromstring(archive.read(manifests[0]))
            namespace = document.tag.partition("}")[0] + "}" if document.tag.startswith("{") else ""
            metadata = document.find(f"{namespace}metadata")
            if (
                metadata is None
                or metadata.findtext(f"{namespace}id") != name
                or metadata.findtext(f"{namespace}version") != version
            ):
                raise PipelineError(f"{name}: package id/version differs")
            repository = metadata.find(f"{namespace}repository")
            if repository is None or repository.attrib.get("commit") != commit:
                raise PipelineError(f"{name}: missing or wrong source commit")
            dependencies = {}
            for item in metadata.iter(f"{namespace}dependency"):
                identifier = item.attrib["id"]
                if identifier in dependencies:
                    raise PipelineError(f"{name}: duplicate dependency")
                dependencies[identifier] = item.attrib["version"]
            if dependencies != expected[name]:
                raise PipelineError(f"{name}: nuspec dependencies differ from exact admitted siblings/managed runtimes")
            if any(entry.endswith((".onnx", ".onnx.data", ".safetensors")) for entry in archive.namelist()):
                raise PipelineError(f"{name}: package contains model weights")
            if not any(entry.startswith("lib/net10.0/") and entry.endswith(".dll") for entry in archive.namelist()):
                raise PipelineError(f"{name}: missing net10.0 assembly")
        records.append(
            {
                "id": name,
                "version": version,
                "sha256": fixtures.sha256_file(path),
                "source_commit": commit,
                "dependencies": dependencies,
            }
        )
    return records


def _consumer_project(model: str, version: str, native: str, native_version: str) -> str:
    return (
        '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>'
        "<TargetFramework>net10.0</TargetFramework><OutputType>Exe</OutputType>"
        "<Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings>"
        "<RestorePackagesWithLockFile>true</RestorePackagesWithLockFile>"
        "<RestoreFallbackFolders></RestoreFallbackFolders></PropertyGroup><ItemGroup>"
        f'<PackageReference Include="{model}" Version="[{version}]" />'
        f'<PackageReference Include="{native}" Version="[{native_version}]" />'
        "</ItemGroup></Project>\n"
    )


def _consumer_graph(path: Path, feed: Path, versions: dict[str, str], backend: str, tokenizer_hash: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    libraries = document["libraries"]
    if not libraries or any(value.get("type") != "package" for value in libraries.values()):
        raise PipelineError("consumer restore graph must contain only packages")
    restored = {}
    for identity, value in libraries.items():
        name, version = identity.rsplit("/", 1)
        lower = name.lower()
        if ("tokenizer" in lower and name != TOKENIZER) or any(
            word in lower for word in (".gpu", ".cuda", "python", "huggingface")
        ):
            raise PipelineError(f"foreign tokenizer/Python/GPU dependency in consumer: {identity}")
        if name == TOKENIZER:
            if versions[TOKENIZER] != version or value.get("sha512") != tokenizer_hash:
                raise PipelineError(f"consumer tokenizer differs from the pinned, locked package: {identity}")
        elif name.startswith("Starlings."):
            if versions.get(name) != version:
                raise PipelineError(f"consumer resolves an unexpected sibling version: {identity}")
            package = feed / f"{name}.{version}.nupkg"
            digest = base64.b64encode(hashlib.sha512(package.read_bytes()).digest()).decode()
            if value.get("sha512") != digest:
                raise PipelineError(f"consumer package bytes differ from packed candidate: {identity}")
        restored[name] = version
    native, version = NATIVE[backend]
    if restored.get(native) != version:
        raise PipelineError("consumer native CPU package identity differs")
    model = TYPED_MODEL[backend]
    if model not in restored or TOKENIZER not in restored:
        raise PipelineError("consumer is missing its typed model or owned tokenizer package")
    return {
        "libraries": restored,
        "only_packages": True,
        "packed_package_bytes_verified": True,
        "tokenizer_package_bytes_verified": True,
    }


def _seed(candidate: dict, source: Path, destination: Path, *, bundle: bool) -> list[dict]:
    destination.mkdir(parents=True)
    if bundle:
        lock = json.loads((source / "bundle-lock.json").read_text(encoding="utf-8"))
        if lock["profile"] != candidate["id"] or lock["origin"] != candidate["origin"]:
            raise PipelineError("consumer bundle belongs to another original candidate")
        records = lock["files"]
    else:
        required = {candidate["behavior"]["graph_file"], candidate["behavior"]["tokenizer_file"]}
        records = [item for item in candidate["origin"]["files"] if item["path"] in required]
        if {item["path"] for item in records} != required:
            raise PipelineError("candidate does not lock both consumer graph and tokenizer")
    seen = set()
    for record in records:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in record["path"] or record["path"] in seen:
            raise PipelineError("unsafe or duplicate consumer model path")
        seen.add(record["path"])
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise PipelineError("consumer model source escapes seed")
        if path.stat().st_size != record["size"] or fixtures.sha256_file(path) != record["sha256"]:
            raise PipelineError(f"consumer model seed mismatch: {record['path']}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if target.stat().st_size != record["size"] or fixtures.sha256_file(target) != record["sha256"]:
            raise PipelineError("consumer model changed during verified copy")
    if bundle:
        shutil.copyfile(source / "bundle-lock.json", destination / "bundle-lock.json")
    return records


def _offline(dotnet: str, app: Path, argv: list[str], *, cwd: Path, log: Path, env: dict, timeout: float) -> dict:
    if shutil.which("unshare") is None:
        raise PipelineError("offline consumer requires unshare -rn; no proxy/env-only fallback is permitted")
    probe = _probe(["unshare", "-rn", "true"], root=cwd, timeout=timeout)
    if probe["status"] != "available":
        raise PipelineError("offline consumer network namespace is unavailable; unshare -rn failed")
    return run_command(["unshare", "-rn", dotnet, str(app), *argv], cwd=cwd, log_path=log, env=env, timeout=timeout)


def _native_files(published: Path, backend: str) -> list[dict]:
    directory = published / "runtimes/linux-x64/native"
    required = {"libonnxruntime.so", *(["libonnxruntime-genai.so"] if backend == "ortgenai" else [])}
    if not directory.is_dir() or not required.issubset({path.name for path in directory.iterdir()}):
        raise PipelineError("consumer is missing the admitted Linux x64 native runtime files")
    return [
        {"path": str(path.relative_to(published)), "sha256": fixtures.sha256_file(path), "size": path.stat().st_size}
        for path in sorted(directory.glob("libonnxruntime*.so"))
    ]


def append_distribution(
    stages: list[Stage], *, recipe: dict, repo_root: Path, snapshot: Path, threads: int, timeout: float
) -> list[Stage]:
    """Insert committed package-consumer evidence and make review depend on it."""
    root, snapshot = repo_root.resolve(), snapshot.resolve()
    names = {stage.name for stage in stages}
    if "distribution" in names or not {"assets", "evaluate", "reference", "dotnet", "review"}.issubset(names):
        raise PipelineError("distribution requires complete candidate stages and a single review")
    candidate_path, corpus = root / recipe["manifest"], root / recipe["reference_corpus"]
    candidate = load_candidate(candidate_path)
    backend = candidate["runtime"]["backend"]
    required_backend_stages = {"bundle", "backend"} if backend == "ortgenai" else {"onnx"}
    if backend not in TOOLS or not required_backend_stages.issubset(names):
        raise PipelineError("distribution requires the candidate backend evidence")
    dependencies = (
        "evaluate",
        "assets",
        "reference",
        "dotnet",
        *(("bundle", "backend") if backend == "ortgenai" else ("onnx",)),
    )
    files = (
        "global.json",
        "Directory.Build.props",
        "ai-dotnet/Directory.Build.props",
        "ai-dotnet/Directory.Build.targets",
        "ai-dotnet/Directory.Packages.props",
        TOOLS[backend],
        TOOL_MAIN,
        "model-pipeline/src/model_pipeline/onboarding_distribution.py",
        "model-pipeline/src/model_pipeline/evaluate_generation_candidate.py",
        "model-pipeline/src/model_pipeline/onboarding_bert.py",
    )
    inputs = code_inputs(root, files=files, directories=tuple(PACKAGES.values()))
    inputs.update(
        {"candidate": candidate_path, "corpus": corpus, "recipe": root / recipe["recipe_path"]}
        if "recipe_path" in recipe
        else {"candidate": candidate_path, "corpus": corpus}
    )
    inputs["reference_requirements"] = root / recipe["reference_environment"]
    commit = _head(root, timeout)
    versions = _versions(root)
    tokenizer_hash = tokenizer_content_hash(root)
    dotnet = shutil.which("dotnet") or "dotnet"

    def execute(out: Path, deps: dict) -> dict:
        _assert_committed_code(root, set(inputs.values()), timeout)
        if _head(root, timeout) != commit:
            raise PipelineError("committed HEAD changed after distribution planning")
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="candidate-consumer-") as temporary:
            work = Path(temporary)
            clone, feed, consumer, model = (work / name for name in ("source", "feed", "consumer", "model"))
            feed.mkdir()
            consumer.mkdir()
            env = {
                "NUGET_PACKAGES": str(work / "nuget-build"),
                "NUGET_HTTP_CACHE_PATH": str(work / "nuget-http-build"),
                "DOTNET_CLI_HOME": str(work / "dotnet-home"),
                "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
                "DOTNET_PROCESSOR_COUNT": str(threads),
                "HF_TOKEN": "",
                "STARLING_MODELS": "",
            }
            run_command(
                ["git", "clone", "--quiet", "--no-hardlinks", str(root), str(clone)],
                cwd=root,
                log_path=out / "clone.log",
                timeout=timeout,
            )
            run_command(
                ["git", "checkout", "--quiet", "--detach", commit],
                cwd=clone,
                log_path=out / "checkout.log",
                timeout=timeout,
            )
            for name, relative in PACKAGES.items():
                run_command(
                    [
                        dotnet,
                        "pack",
                        str(clone / relative / f"{name}.csproj"),
                        "-c",
                        "Release",
                        "-o",
                        str(feed),
                        "-m:1",
                        "--disable-build-servers",
                        "-p:RestoreLockedMode=true",
                        "-p:ContinuousIntegrationBuild=true",
                        f"-p:RepositoryCommit={commit}",
                    ],
                    cwd=clone,
                    log_path=out / f"pack-{name}.log",
                    env=env,
                    timeout=timeout,
                )
            packages = _inspect_packages(feed, versions, commit)
            shutil.copytree(feed, out / "packages")
            native, native_version = NATIVE[backend]
            typed_model = TYPED_MODEL[backend]
            (consumer / "Consumer.csproj").write_text(
                _consumer_project(typed_model, versions[typed_model], native, native_version)
            )
            shutil.copyfile(clone / TOOLS[backend], consumer / "Program.cs")
            shutil.copyfile(clone / TOOL_MAIN, consumer / "ToolMain.cs")
            shutil.copyfile(clone / "global.json", consumer / "global.json")
            config = ET.Element("configuration")
            sources = ET.SubElement(config, "packageSources")
            ET.SubElement(sources, "clear")
            ET.SubElement(sources, "add", key="candidate", value=str(feed))
            ET.SubElement(sources, "add", key="nuget.org", value="https://api.nuget.org/v3/index.json")
            mapping = ET.SubElement(config, "packageSourceMapping")
            ET.SubElement(ET.SubElement(mapping, "packageSource", key="candidate"), "package", pattern="Starlings.*")
            # The tokenizer is not in the candidate feed: the exact pattern sends it to nuget.org (the most specific
            # pattern wins), and _consumer_graph checks its bytes against the lock files.
            nuget_org = ET.SubElement(mapping, "packageSource", key="nuget.org")
            ET.SubElement(nuget_org, "package", pattern=TOKENIZER)
            ET.SubElement(nuget_org, "package", pattern="*")
            ET.ElementTree(config).write(consumer / "NuGet.Config", encoding="unicode")
            env.update(
                {
                    "NUGET_PACKAGES": str(work / "nuget-consumer"),
                    "NUGET_HTTP_CACHE_PATH": str(work / "nuget-http-consumer"),
                }
            )
            run_command(
                [dotnet, "restore", "Consumer.csproj", "--use-lock-file", "--disable-parallel", "-m:1"],
                cwd=consumer,
                log_path=out / "consumer-restore.log",
                env=env,
                timeout=timeout,
            )
            graph = _consumer_graph(consumer / "obj/project.assets.json", feed, versions, backend, tokenizer_hash)
            run_command(
                [dotnet, "restore", "Consumer.csproj", "--locked-mode", "--disable-parallel", "-m:1"],
                cwd=consumer,
                log_path=out / "consumer-locked-restore.log",
                env=env,
                timeout=timeout,
            )
            published = consumer / "app"
            run_command(
                [
                    dotnet,
                    "publish",
                    "Consumer.csproj",
                    "-c",
                    "Release",
                    "--no-restore",
                    "-o",
                    str(published),
                    "-m:1",
                    "--disable-build-servers",
                ],
                cwd=consumer,
                log_path=out / "consumer-build.log",
                env=env,
                timeout=timeout,
            )
            native_files = _native_files(published, backend)
            seeded = _seed(
                candidate, deps["bundle"] if backend == "ortgenai" else snapshot, model, bundle=backend == "ortgenai"
            )
            shutil.copyfile(clone / recipe["manifest"], consumer / "candidate.json")
            shutil.copyfile(clone / recipe["reference_corpus"], consumer / "corpus.jsonl")
            evidence = out / "consumer-evidence"
            evidence.mkdir()
            if backend == "ortgenai":
                argv = [
                    "--candidate",
                    str(consumer / "candidate.json"),
                    "--bundle",
                    str(model),
                    "--corpus",
                    str(consumer / "corpus.jsonl"),
                ]
            else:
                shutil.copyfile(deps["reference"] / "reference.json", consumer / "reference.json")
                argv = [
                    "--profile",
                    str(consumer / "candidate.json"),
                    "--snapshot",
                    str(model),
                    "--reference",
                    str(consumer / "reference.json"),
                ]
            # Execution retains only the published application and explicitly seeded data.
            # The committed checkout and both isolated package caches are no longer present.
            for directory in (
                clone,
                work / "nuget-build",
                work / "nuget-consumer",
                work / "nuget-http-build",
                work / "nuget-http-consumer",
            ):
                if directory.exists():
                    shutil.rmtree(directory)
            command = _offline(
                dotnet,
                published / "Consumer.dll",
                [*argv, "--out", str(evidence), "--threads", str(threads)],
                cwd=consumer,
                log=out / "consumer-offline.log",
                env=env,
                timeout=timeout,
            )
            if backend == "ortgenai":
                from model_pipeline.evaluate_generation_candidate import evaluate

                parity = evaluate(candidate, corpus, deps["reference"], deps["backend"], evidence)
                identities = json.loads((evidence / "environment.json").read_text())
            else:
                from model_pipeline.onboarding_bert import evaluate_development

                parity = evaluate_development(
                    candidate,
                    candidate_path,
                    deps["reference"] / "reference.json",
                    deps["onnx"] / "onnx.json",
                    evidence / "dotnet.json",
                )
                identities = json.loads((evidence / "dotnet.json").read_text())["environment"]
            if parity["status"] != "passed":
                write_json(out / "consumer-parity.json", parity)
                raise PipelineError("package-only consumer development parity failed")
            write_json(out / "consumer-parity.json", parity)
            shutil.copyfile(consumer / "Consumer.csproj", out / "Consumer.csproj")
            shutil.copyfile(consumer / "Program.cs", out / "Program.cs")
            shutil.copyfile(consumer / "ToolMain.cs", out / "ToolMain.cs")
            shutil.copyfile(consumer / "packages.lock.json", out / "consumer.packages.lock.json")
            shutil.copyfile(consumer / "obj/project.assets.json", out / "consumer.project.assets.json")
            shutil.copyfile(published / "Consumer.deps.json", out / "Consumer.deps.json")
            record = {
                "status": "passed",
                "source_commit": commit,
                "profile": candidate["id"],
                "scope": "development distribution; protected release qualification unapproved",
                "offline": "network namespace without interfaces (unshare -rn)",
                "packages": packages,
                "consumer_restore": graph,
                "runtime_and_model_identity": identities,
                "native_runtime_files": native_files,
                "model_seed": seeded,
                "command": command,
                "development_parity": "passed",
                "temporary_caches_removed": True,
                "checkout_and_package_caches_removed_before_execution": True,
            }
            write_json(out / "distribution.json", record)
        return record

    distribution = Stage(
        "distribution",
        dependencies,
        inputs,
        {"candidate_commit": commit, "recipe": recipe, "versions": versions, "threads": threads, "timeout": timeout},
        execute,
    )
    if backend == "onnx":
        distribution = source_reader(distribution, candidate, snapshot)
    result = []
    for stage in stages:
        if stage.name == "review":
            original = stage.execute

            def review(out: Path, deps: dict, callback=original) -> dict:
                metadata = callback(out, deps)
                report = json.loads((deps["distribution"] / "distribution.json").read_text())
                if report["status"] != "passed":
                    raise PipelineError("review requires passed development distribution")
                shutil.copyfile(deps["distribution"] / "distribution.json", out / "distribution.json")
                review_path = out / "review.json"
                document = json.loads(review_path.read_text())
                document["distribution"] = report
                write_json(review_path, document)
                (out / "review.md").write_text(
                    f"# Candidate review: {candidate['id']}\n\n"
                    "Development task parity passed. References and comparison policies remain proposed.\n\n"
                    "Development distribution **passed** using an external package-only consumer "
                    f"built from committed source `{report['source_commit']}` and executed in a network namespace "
                    "without interfaces (`unshare -rn`). The checkout and package caches were removed "
                    "before consumer execution. See `distribution.json` for package, model, native runtime "
                    "and comparison evidence.\n\n"
                    "Protected G1/G2 and release qualification remain **unapproved**. "
                    "No protected fixtures were adopted.\n",
                    encoding="utf-8",
                )
                return {**metadata, "distribution": "passed", "release_qualification": "unapproved"}

            result.append(distribution)
            result.append(
                dataclasses.replace(
                    stage,
                    dependencies=(*stage.dependencies, "distribution"),
                    execute=review,
                    inputs={**stage.inputs, **inputs},
                    parameters={"original": stage.parameters, "distribution_commit": commit},
                )
            )
        else:
            result.append(stage)
    return result
