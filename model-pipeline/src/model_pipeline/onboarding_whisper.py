"""Whisper speech-to-text on ONNX Runtime CPU fp16: transcript parity against a fresh transformers reference.

Explicit allow-list: the pinned onnx-community fp16 graphs, the pinned openai PyTorch checkpoint they were exported
from, and the audio front-end/decoder contract are checked field by field, here and in Starlings.Models.Whisper.
Anything else is refused. Stages:

    assets -> environment -> reference -> onnx -> dotnet -> evaluate -> review

The fp16 ONNX snapshot is `--snapshot`; the PyTorch checkpoint is `--base-snapshot` (a plain snapshot directory).
Correctness is proved at three levels: intermediate tensors (mel, encoder hidden state, teacher-forced decoder logits),
greedy token ids, and transcripts. The fp32 PyTorch reference bounds export fidelity; .NET against Python ONNX Runtime
on the same graphs is the tight same-backend canary. Reference outputs are proposals; nothing here adopts protected
fixtures or changes tolerances.
"""

from __future__ import annotations

import array
import json
import math
import os
import shutil
import sys
import tempfile
import wave
from dataclasses import replace
from pathlib import Path, PurePosixPath

from model_pipeline import fixtures
from model_pipeline.onboarding_candidate import guard_output, repository_input, write_json
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command

PLUGIN = "whisper-onnx-cpu-v1"
# Distribution is appended by onboarding_plugins from onboarding_distribution (and refused below).
SHARED_DISTRIBUTION = True
RUNNER = "model-pipeline/reference/whisper/run_whisper.py"
TOOL = "ai-dotnet/tools/Starlings.Whisper.Fidelity.Tool/Starlings.Whisper.Fidelity.Tool.csproj"
FORMAT = "starlings.whisper-candidate-results"
STAGES = ["assets", "environment", "reference", "onnx", "dotnet", "evaluate", "review"]
RUNTIME = {"backend": "onnx", "version": "1.30.0", "provider": "CPUExecutionProvider", "precision": "fp16"}
ENCODER_FILE = "onnx/encoder_model_fp16.onnx"
DECODER_FILE = "onnx/decoder_model_fp16.onnx"
TOKENIZER_FILE = "tokenizer.json"
SNAPSHOT_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    DECODER_FILE,
    ENCODER_FILE,
    "preprocessor_config.json",
    TOKENIZER_FILE,
)
BASE_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
# The suppression lists of the pinned generation_config.json, restated in WhisperModel.cs.
SUPPRESS_TOKENS = [
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873,
    893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647,
    7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362,
    18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870,
    50254, 50258, 50359, 50360, 50361, 50362, 50363,
]  # fmt: skip
BEGIN_SUPPRESS_TOKENS = [220, 50257]
SPECIAL_TOKENS = {
    "end_of_text": 50257,
    "start_of_transcript": 50258,
    "translate": 50359,
    "transcribe": 50360,
    "start_of_lm": 50361,
    "start_of_prev": 50362,
    "no_speech": 50363,
    "no_timestamps": 50364,
}
LANGUAGES = {"en": 50259}
BEHAVIOR = {
    "architecture": "WhisperForConditionalGeneration",
    "encoder_file": ENCODER_FILE,
    "decoder_file": DECODER_FILE,
    "tokenizer_file": TOKENIZER_FILE,
    "tokenizer_profile": "whisper-bpe",
    "sample_rate": 16000,
    "n_fft": 400,
    "hop_length": 160,
    "num_mel_bins": 128,
    "max_samples": 480000,
    "num_frames": 3000,
    "encoder_positions": 1500,
    "hidden_size": 1280,
    "encoder_layers": 32,
    "decoder_layers": 4,
    "vocabulary_size": 51866,
    "max_target_positions": 448,
    "special_tokens": SPECIAL_TOKENS,
    "languages": LANGUAGES,
    "task": "transcribe",
    "timestamps": False,
    "decoding": "greedy",
    "suppress_tokens": SUPPRESS_TOKENS,
    "begin_suppress_tokens": BEGIN_SUPPRESS_TOKENS,
}
CONTRACTS = {
    "onnx-community/whisper-large-v3-turbo-ONNX": {
        "id": "whisper-large-v3-turbo/onnx-fp16-cpu",
        "revision": "76ba4fb597cca51a329a1820e49282e26248b36b",
        "base": ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
        # The same pins as WhisperModelAssets.cs: the .NET library refuses any other graph or tokenizer.
        "pins": {
            ENCODER_FILE: (1274342603, "b48fd309c51e77ba88324469632fbff8796d2dd37b1658c9516c1c1b1d96428d"),
            DECODER_FILE: (343974281, "15870b95e08e666ea2423738ee12722de965d09cf05f4579fa947357332b5057"),
            TOKENIZER_FILE: (2480645, "b3c8202bbf06d8ee4232c5984baa563784ac4737e2e7fdc42fa180200d3cfcdb"),
        },
        "behavior": BEHAVIOR,
        # Binding tolerances adopted by the maintainer (#142): the candidate policy must equal them exactly.
        "protected_tolerances": "model-pipeline/protected/whisper-large-v3-turbo/onnx-fp16/tolerances.yaml",
    },
}
POLICY = {
    "mel_max_abs": 1e-6,
    "mel_path_max_abs": 1e-4,
    "hidden_max_abs": 0.05,
    "hidden_padding_max_abs": 1.0,
    "logits_max_abs": 0.1,
    "same_backend_max_abs": 1e-5,
    "token_ids": "exact",
    "transcript": "exact",
    "reference_wer_max": 0.0,
    "ground_truth_wer_max": 0.2,
    "non_finite": "fail",
    "required_paths": ["reference", "python-onnx", "dotnet-mel", "dotnet-audio"],
}
# Correctness bounds are fixed: the .NET mel restates the numpy reference, and .NET against Python ONNX Runtime runs the
# same graphs, so both stay near float32 rounding; ids and transcripts must be identical. Only the fp16-export fidelity
# bounds (fp16 graphs against the fp32 PyTorch reference) may be proposed per model, up to these caps.
FIXED_POLICY_KEYS = {
    key: POLICY[key]
    for key in (
        "mel_max_abs",
        "same_backend_max_abs",
        "token_ids",
        "transcript",
        "ground_truth_wer_max",
        "non_finite",
        "required_paths",
    )
}
# The encoder hidden state is bounded separately on the positions that cover audio and on the zero-padded tail: fp16
# drifts most on the padded (log-mel floor) frames, and one bound loose enough for those would hide a regression on the
# frames the transcript depends on.
FIDELITY_CAPS = {
    "mel_path_max_abs": 1e-3,
    "hidden_max_abs": 0.1,
    "hidden_padding_max_abs": 2.0,
    "logits_max_abs": 0.5,
    "reference_wer_max": 0.1,
}
TENSORS = {
    "reference": ("mel", "mel_torch", "hidden", "logits"),
    "onnx": ("hidden", "logits"),
    "dotnet": ("mel", "hidden", "logits"),
}


def _refuse(field: str, detail: str) -> None:
    raise PipelineError(f"unsupported Whisper contract field {field}: {detail}")


def _same(field: str, actual, expected) -> None:
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        _refuse(field, f"expected {expected!r}, found {actual!r}")


def _files(field: str, records, expected: tuple[str, ...]) -> None:
    paths = [record.get("path") for record in records] if isinstance(records, list) else None
    if paths is None or sorted(paths) != sorted(expected) or len(set(paths)) != len(paths):
        _refuse(field, f"pinned files must be exactly {sorted(expected)}")


def _validate_policy(policy: dict) -> None:
    if not isinstance(policy, dict) or set(policy) != set(POLICY):
        _refuse("evaluation.policy", f"fields must be exactly {sorted(POLICY)}")
    for key, value in FIXED_POLICY_KEYS.items():
        _same(f"evaluation.policy.{key}", policy[key], value)
    for key, cap in FIDELITY_CAPS.items():
        value = policy[key]
        lower_ok = value >= 0 if key == "reference_wer_max" else value > 0
        if type(value) not in (int, float) or not math.isfinite(value) or not lower_ok or value > cap:
            _refuse(f"evaluation.policy.{key}", f"must be a finite number up to {cap}")


def protected_policy(text: str, profile: str) -> dict:
    """The `policy` block of a Whisper tolerance file (YAML text) for `profile`; refuses an unexpected layout."""
    import yaml

    try:
        data = yaml.safe_load(text)
        if set(data) != {"profile", "reference", "policy", "report_only"} or data["profile"] != profile:
            raise PipelineError(f"tolerance file must hold profile, reference, policy and report_only for {profile}")
        policy = data["policy"]
        if not isinstance(policy, dict) or set(policy) != set(POLICY):
            raise PipelineError(f"tolerance policy fields must be exactly {sorted(POLICY)}")
        return policy
    except (TypeError, AttributeError, yaml.YAMLError) as exc:
        raise PipelineError(f"unexpected Whisper tolerance file layout: {exc}") from exc


def validate_candidate_behavior(candidate: dict, repo_root: Path | None = None) -> dict:
    """Refuse every contract this route does not implement; returns the matching allow-list entry.

    With `repo_root`, the candidate policy must also equal the binding protected tolerances exactly.
    """
    try:
        origin = candidate["origin"]
        repo = origin["repo"]
        if repo not in CONTRACTS:
            _refuse("origin.repo", f"{repo!r} is not an allow-listed Whisper graph repository")
        contract = CONTRACTS[repo]
        _same("format_version", candidate["format_version"], 1)
        _same("id", candidate["id"], contract["id"])
        _same("task", candidate["task"], "transcription")
        _same("origin.revision", origin["revision"], contract["revision"])
        _files("origin.files", origin["files"], SNAPSHOT_FILES)
        for record in origin["files"]:
            if record["path"] in contract["pins"]:
                _same(
                    f"origin.files[{record['path']}]",
                    [record["size"], record["sha256"]],
                    contract["pins"][record["path"]],
                )
        base = origin["base"]
        _same("origin.base.repo", base["repo"], contract["base"][0])
        _same("origin.base.revision", base["revision"], contract["base"][1])
        _files("origin.base.files", base["files"], BASE_FILES)
        for record in base["files"]:
            path = PurePosixPath(record["path"])
            if path.is_absolute() or len(path.parts) != 1:
                _refuse("origin.base.files", f"unsafe path {record['path']!r}")
        extra = set(origin) - {"repo", "revision", "files", "base"}
        if extra:
            _refuse("origin", f"unknown fields {sorted(extra)}")
        _same("runtime", candidate["runtime"], RUNTIME)
        behavior, expected = candidate["behavior"], contract["behavior"]
        if set(behavior) != set(expected):
            _refuse("behavior", f"fields must be exactly {sorted(expected)}")
        for key, value in expected.items():
            _same(f"behavior.{key}", behavior[key], value)
        _same("evaluation.status", candidate["evaluation"]["status"], "proposed")
        _validate_policy(candidate["evaluation"]["policy"])
        if repo_root is not None:
            relative = contract["protected_tolerances"]
            try:
                text = (repo_root / relative).read_text(encoding="utf-8")
            except OSError as exc:
                raise PipelineError(f"binding tolerances are unreadable: {relative}: {exc}") from exc
            bound = protected_policy(text, contract["id"])
            for key, value in bound.items():
                _same(f"evaluation.policy.{key} (binding {relative})", candidate["evaluation"]["policy"][key], value)
        return contract
    except (KeyError, TypeError, AttributeError) as exc:
        raise PipelineError(f"unsupported or incomplete Whisper contract: {exc!r}") from exc


def check_configs(candidate: dict, snapshot: Path, base_snapshot: Path) -> dict:
    """The pinned configs of both repositories agree with the contract (no upstream code runs)."""
    behavior = candidate["behavior"]
    checked = {}
    for name, directory in (("snapshot", snapshot), ("base", base_snapshot)):
        try:
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            generation = json.loads((directory / "generation_config.json").read_text(encoding="utf-8"))
            preprocessor = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineError(f"{name}: unreadable pinned config: {exc}") from exc
        facts = {
            "config.architectures": (config.get("architectures"), [behavior["architecture"]]),
            "config.d_model": (config.get("d_model"), behavior["hidden_size"]),
            "config.encoder_layers": (config.get("encoder_layers"), behavior["encoder_layers"]),
            "config.decoder_layers": (config.get("decoder_layers"), behavior["decoder_layers"]),
            "config.num_mel_bins": (config.get("num_mel_bins"), behavior["num_mel_bins"]),
            "config.vocab_size": (config.get("vocab_size"), behavior["vocabulary_size"]),
            "config.max_source_positions": (config.get("max_source_positions"), behavior["encoder_positions"]),
            "config.max_target_positions": (config.get("max_target_positions"), behavior["max_target_positions"]),
            "config.decoder_start_token_id": (
                config.get("decoder_start_token_id"),
                SPECIAL_TOKENS["start_of_transcript"],
            ),
            "config.eos_token_id": (config.get("eos_token_id"), SPECIAL_TOKENS["end_of_text"]),
            "generation.suppress_tokens": (generation.get("suppress_tokens"), behavior["suppress_tokens"]),
            "generation.begin_suppress_tokens": (
                generation.get("begin_suppress_tokens"),
                behavior["begin_suppress_tokens"],
            ),
            "generation.no_timestamps_token_id": (
                generation.get("no_timestamps_token_id"),
                SPECIAL_TOKENS["no_timestamps"],
            ),
            "generation.task_to_id": (
                generation.get("task_to_id"),
                {"transcribe": SPECIAL_TOKENS["transcribe"], "translate": SPECIAL_TOKENS["translate"]},
            ),
            "generation.lang_to_id": (
                {code: generation.get("lang_to_id", {}).get(f"<|{code}|>") for code in behavior["languages"]},
                behavior["languages"],
            ),
            "generation.max_length": (generation.get("max_length"), behavior["max_target_positions"]),
            "preprocessor.feature_size": (preprocessor.get("feature_size"), behavior["num_mel_bins"]),
            "preprocessor.sampling_rate": (preprocessor.get("sampling_rate"), behavior["sample_rate"]),
            "preprocessor.n_fft": (preprocessor.get("n_fft"), behavior["n_fft"]),
            "preprocessor.hop_length": (preprocessor.get("hop_length"), behavior["hop_length"]),
            "preprocessor.n_samples": (preprocessor.get("n_samples"), behavior["max_samples"]),
            "preprocessor.nb_max_frames": (preprocessor.get("nb_max_frames"), behavior["num_frames"]),
            "preprocessor.padding_value": (preprocessor.get("padding_value"), 0.0),
        }
        for field, (actual, expected) in facts.items():
            if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise PipelineError(f"{name} {field} is {actual!r}, contract {expected!r}")
        checked[name] = sorted(facts)
    return checked


def verify_base(candidate: dict, base_snapshot: Path) -> list[dict]:
    """Pinned size and SHA-256 of every base file, read only inside the base snapshot."""
    from model_pipeline.onboarding_candidate import verify_files

    return verify_files(candidate["origin"]["base"]["files"], base_snapshot, "base")


def load_corpus(corpus: Path, candidate: dict) -> list[dict]:
    """The ASR corpus index and its audio files, each verified by SHA-256 against the index."""
    behavior = candidate["behavior"]
    rows, seen = [], set()
    for row in fixtures.read_jsonl(corpus):
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise PipelineError("ASR corpus rows require unique nonempty ids")
        seen.add(identifier)
        relative = PurePosixPath(row.get("audio", ""))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".wav" or not relative.parts:
            raise PipelineError(f"{identifier}: unsafe or non-WAV audio path {row.get('audio')!r}")
        path = corpus.parent / relative
        if path.is_symlink() or not path.is_file() or fixtures.sha256_file(path) != row.get("sha256"):
            raise PipelineError(f"{identifier}: audio file is missing or differs from its pinned SHA-256")
        samples = row.get("samples")
        if (
            row.get("sample_rate") != behavior["sample_rate"]
            or type(samples) is not int
            or not 0 < samples <= behavior["max_samples"]
            or row.get("language") not in behavior["languages"]
            or not isinstance(row.get("text"), str)
        ):
            raise PipelineError(f"{identifier}: needs 16 kHz audio up to 30 s, a contract language, and its text")
        # The indexed count decides where the encoder's audio positions end, so it must be the WAV's own.
        try:
            with wave.open(str(path), "rb") as reader:
                header = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes())
        except (OSError, EOFError, wave.Error) as exc:
            raise PipelineError(f"{identifier}: unreadable WAV: {exc}") from exc
        if header != (1, 2, behavior["sample_rate"], samples):
            raise PipelineError(f"{identifier}: WAV is {header} (channels, width, rate, frames), not the indexed clip")
        rows.append({**row, "path": path})
    if len(rows) < 2 or len({row["samples"] for row in rows}) < 2:
        raise PipelineError("ASR corpus must hold at least two differently sized clips")
    return rows


def prompt(candidate: dict, language: str) -> list[int]:
    special = candidate["behavior"]["special_tokens"]
    return [
        special["start_of_transcript"],
        candidate["behavior"]["languages"][language],
        special["transcribe"],
        special["no_timestamps"],
    ]


def words(text: str) -> list[str]:
    """Lower-cased alphanumeric words (apostrophes kept), for word error rate only."""
    cleaned = "".join(char.lower() if char.isalnum() or char == "'" else " " for char in text)
    return cleaned.split()


def edit_distance(hypothesis: list[str], reference: list[str]) -> int:
    previous = list(range(len(reference) + 1))
    for row, word in enumerate(hypothesis, 1):
        current = [row]
        for column, expected in enumerate(reference, 1):
            current.append(
                min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (word != expected))
            )
        previous = current
    return previous[-1]


def word_error_rate(hypothesis: str, reference: str) -> float:
    expected = words(reference)
    errors = edit_distance(words(hypothesis), expected)
    return errors / len(expected) if expected else float(errors > 0)


def _document(path: Path, candidate: dict, digest: str, stage: str) -> tuple[dict, dict[str, dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"missing or malformed required {stage} evidence: {path}: {exc}") from exc
    if not isinstance(data, dict) or type(data.get("format_version")) is not int:
        raise PipelineError(f"{stage}: evidence must be an object with an integer format version")
    if (
        data.get("format"),
        data.get("format_version"),
        data.get("profile"),
        data.get("task"),
        data.get("candidate_sha256"),
        data.get("stage"),
    ) != (FORMAT, 1, candidate["id"], candidate["task"], digest, stage):
        raise PipelineError(f"{stage}: wrong format, profile, task, contract digest, or stage identity")
    if stage == "reference":
        names = {"model.safetensors", "preprocessor_config.json", "tokenizer.json"}
        files = candidate["origin"]["base"]["files"]
    else:
        names = {ENCODER_FILE, DECODER_FILE, TOKENIZER_FILE}
        files = candidate["origin"]["files"]
    expected_assets = {entry["path"]: entry["sha256"] for entry in files if entry["path"] in names}
    behavior = candidate["behavior"]
    try:
        assets = data["assets"]
        if len(assets) != len(names) or {entry["path"]: entry["sha256"] for entry in assets} != expected_assets:
            raise PipelineError(f"{stage}: asset identities differ from the candidate")
        environment = data["environment"]
        if environment["provider"] != "CPUExecutionProvider" or (
            stage != "reference" and environment["onnxruntime"] != candidate["runtime"]["version"]
        ):
            raise PipelineError(f"{stage}: runtime/provider identity differs from the candidate")
        records = data["cases"]
        if not isinstance(records, list) or len(records) < 2:
            raise PipelineError(f"{stage}: missing required case rows")
        cases = {}
        eos = behavior["special_tokens"]["end_of_text"]
        for row in records:
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in cases:
                raise PipelineError(f"{stage}: duplicate or invalid case id {identifier!r}")
            ids, expected_prompt = row["ids"], prompt(candidate, row["language"])
            if (
                not isinstance(ids, list)
                or not len(expected_prompt) < len(ids) <= behavior["max_target_positions"]
                or any(type(value) is not int or not 0 <= value < behavior["vocabulary_size"] for value in ids)
                or ids[: len(expected_prompt)] != expected_prompt
                or eos in ids[:-1]
                or not isinstance(row["transcript"], str)
                or not isinstance(row["audio_sha256"], str)
            ):
                raise PipelineError(f"{stage}/{identifier}: invalid prompt, ids, or transcript")
            if stage == "reference" and (
                row["default_path_ids"][: len(expected_prompt)] != expected_prompt
                or type(row["samples"]) is not int
                or not 0 < row["samples"] <= behavior["max_samples"]
            ):
                raise PipelineError(f"{stage}/{identifier}: invalid default feature-path ids or sample count")
            if stage != "reference" and not isinstance(row["forced_ids"], list):
                raise PipelineError(f"{stage}/{identifier}: the teacher-forced ids are missing")
            if stage == "dotnet" and not isinstance(row["audio_transcript"], str):
                raise PipelineError(f"{stage}/{identifier}: the public audio-path transcript is missing")
            cases[identifier] = row
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"{stage}: incomplete or malformed required evidence: {exc}") from exc
    return data, cases


def _tensors(path: Path, document: dict, cases: dict[str, dict], stage: str, behavior: dict, forced: dict) -> dict:
    """The stage's float32 sidecar, split into named per-case tensors with checked shapes and no unreferenced values."""
    shapes = {
        "mel": [behavior["num_mel_bins"], behavior["num_frames"]],
        "mel_torch": [behavior["num_mel_bins"], behavior["num_frames"]],
        "hidden": [behavior["encoder_positions"], behavior["hidden_size"]],
    }
    try:
        specification = document["tensors"]
        relative = Path(specification["file"])
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {".", ".."}:
            raise PipelineError("unsafe tensor sidecar path")
        tensor_path = path.parent / relative
        if tensor_path.is_symlink() or fixtures.sha256_file(tensor_path) != specification["sha256"]:
            raise PipelineError("tensor sidecar identity mismatch")
        blob = tensor_path.read_bytes()
        if len(blob) % 4:
            raise PipelineError("tensor sidecar has an incomplete float32 value")
        values = array.array("f")
        values.frombytes(blob)
        if sys.byteorder != "little":
            values.byteswap()
        if any(not math.isfinite(value) for value in values):
            raise PipelineError("tensor sidecar contains non-finite values")
        output, position = {}, 0
        for row in specification["cases"]:
            key = (row["id"], row["name"])
            if row["id"] not in cases or row["name"] not in TENSORS[stage] or key in output:
                raise PipelineError(f"unknown or duplicate tensor {key}")
            expected = shapes.get(row["name"]) or [len(forced[row["id"]]) - 1, behavior["vocabulary_size"]]
            if row["shape"] != expected or row["offset"] != position:
                raise PipelineError(f"tensor {key} has shape {row['shape']} at {row['offset']}, expected {expected}")
            end = position + math.prod(expected)
            if end > len(values):
                raise PipelineError("tensor sidecar is truncated")
            output[key] = values[position:end]
            position = end
        if set(output) != {(identifier, name) for identifier in cases for name in TENSORS[stage]}:
            raise PipelineError("tensor sidecar is missing required tensors")
        if position != len(values):
            raise PipelineError("tensor sidecar has unreferenced values")
        return output
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise PipelineError(f"{stage}: missing or malformed tensor evidence: {exc}") from exc


def _difference(actual, expected, bound: float, where: str) -> dict:
    if len(actual) != len(expected):
        raise PipelineError(f"{where}: tensor lengths differ ({len(actual)} against {len(expected)})")
    maximum, total = 0.0, 0.0
    for left, right in zip(actual, expected, strict=True):
        difference = abs(left - right)
        total += difference
        maximum = max(maximum, difference)
    return {
        "id": where,
        "status": "passed" if maximum <= bound else "failed",
        "bound": bound,
        "max_abs": maximum,
        "mean_abs": total / len(expected) if len(expected) else 0.0,
    }


def audio_positions(samples: int, behavior: dict) -> int:
    """Encoder positions whose receptive field holds any audio sample; the rest see only zero padding.

    Mel frame t is a window of n_fft samples centred on sample t * hop, so the last frame touching audio is
    floor((samples - 1 + n_fft / 2) / hop). The encoder's two kernel-3 convolutions (stride 1, then stride 2, both
    padding 1) let position p read mel frames 2p - 2 through 2p + 2, so p sees audio when 2p - 2 <= that frame.
    """
    last_frame = (samples - 1 + behavior["n_fft"] // 2) // behavior["hop_length"]
    return min(behavior["encoder_positions"], (last_frame + 2) // 2 + 1)


def _exact(actual, expected, where: str) -> dict:
    return {"id": where, "status": "passed" if actual == expected else "failed", "bound": "exact", "actual": actual}


def _rate(value: float, bound: float, where: str, **details) -> dict:
    return {"id": where, "status": "passed" if value <= bound else "failed", "bound": bound, "value": value, **details}


def evaluate_development(
    candidate: dict,
    candidate_path: Path,
    reference_path: Path,
    onnx_path: Path,
    dotnet_path: Path,
    corpus: list[dict],
) -> dict:
    """Check complete proposed evidence without reading or promoting protected fixtures.

    `corpus` is the pinned corpus as `load_corpus` verified it: it, not the reference evidence, supplies each clip's
    audio digest, sample count, language and ground-truth text.
    """
    validate_candidate_behavior(candidate)
    behavior, policy = candidate["behavior"], candidate["evaluation"]["policy"]
    digest = fixtures.sha256_file(candidate_path)
    reference, expected = _document(reference_path, candidate, digest, "reference")
    onnx, python_cases = _document(onnx_path, candidate, digest, "onnx")
    dotnet, dotnet_cases = _document(dotnet_path, candidate, digest, "dotnet")
    clips = {row["id"]: row for row in corpus}
    if set(expected) != set(clips):
        raise PipelineError("reference evidence does not cover exactly the pinned corpus clips")
    for identifier, row in expected.items():
        clip = clips[identifier]
        if (row["audio_sha256"], row["samples"], row["language"]) != (
            clip["sha256"],
            clip["samples"],
            clip["language"],
        ):
            raise PipelineError(
                f"reference/{identifier}: audio, sample count or language differs from the pinned corpus"
            )
    machines = [document.get("environment", {}).get("machine") for document in (reference, onnx, dotnet)]
    if any(not isinstance(machine, str) or not machine for machine in machines) or len(set(machines)) != 1:
        raise PipelineError("same-backend comparison requires all stages to name the same machine")
    reference_digest = fixtures.sha256_file(reference_path)
    if onnx.get("reference_sha256") != reference_digest or dotnet.get("reference_sha256") != reference_digest:
        raise PipelineError("backend evidence names a different original reference artifact")
    if set(expected) != set(python_cases) or set(expected) != set(dotnet_cases):
        raise PipelineError("backend evidence has missing or unknown required case rows")
    forced = {identifier: row["ids"] for identifier, row in expected.items()}
    for stage, cases in (("onnx", python_cases), ("dotnet", dotnet_cases)):
        for identifier, row in cases.items():
            if row["audio_sha256"] != expected[identifier]["audio_sha256"] or row["forced_ids"] != forced[identifier]:
                raise PipelineError(f"{stage}/{identifier}: audio or teacher-forced ids differ from the reference")
    tensors = {
        stage: _tensors(path, document, cases, stage, behavior, forced)
        for stage, path, document, cases in (
            ("reference", reference_path, reference, expected),
            ("onnx", onnx_path, onnx, python_cases),
            ("dotnet", dotnet_path, dotnet, dotnet_cases),
        )
    }
    reference_tensors, python_tensors, dotnet_tensors = (tensors[name] for name in ("reference", "onnx", "dotnet"))
    checks = []
    for identifier, row in expected.items():
        python_row, dotnet_row = python_cases[identifier], dotnet_cases[identifier]
        split = audio_positions(row["samples"], behavior) * behavior["hidden_size"]
        checks += [
            _difference(
                reference_tensors[identifier, "mel_torch"],
                reference_tensors[identifier, "mel"],
                policy["mel_path_max_abs"],
                f"reference/{identifier}/mel_torch_vs_numpy",
            ),
            _exact(row["default_path_ids"], row["ids"], f"reference/{identifier}/default_path_ids"),
            _difference(
                dotnet_tensors[identifier, "mel"],
                reference_tensors[identifier, "mel"],
                policy["mel_max_abs"],
                f"dotnet-audio/{identifier}/mel_vs_reference",
            ),
            _difference(
                python_tensors[identifier, "hidden"][:split],
                reference_tensors[identifier, "hidden"][:split],
                policy["hidden_max_abs"],
                f"onnx/{identifier}/hidden_vs_reference",
            ),
            _difference(
                python_tensors[identifier, "hidden"][split:],
                reference_tensors[identifier, "hidden"][split:],
                policy["hidden_padding_max_abs"],
                f"onnx/{identifier}/hidden_padding_vs_reference",
            ),
            _difference(
                dotnet_tensors[identifier, "hidden"],
                python_tensors[identifier, "hidden"],
                policy["same_backend_max_abs"],
                f"dotnet-mel/{identifier}/hidden_vs_python",
            ),
            _difference(
                python_tensors[identifier, "logits"],
                reference_tensors[identifier, "logits"],
                policy["logits_max_abs"],
                f"onnx/{identifier}/logits_vs_reference",
            ),
            _difference(
                dotnet_tensors[identifier, "logits"],
                python_tensors[identifier, "logits"],
                policy["same_backend_max_abs"],
                f"dotnet-mel/{identifier}/logits_vs_python",
            ),
            _exact(dotnet_row["ids"], python_row["ids"], f"dotnet-mel/{identifier}/ids_vs_python"),
            _exact(dotnet_row["transcript"], python_row["transcript"], f"dotnet-mel/{identifier}/transcript_vs_python"),
            _exact(
                dotnet_row["audio_transcript"],
                python_row["transcript"],
                f"dotnet-audio/{identifier}/transcript_vs_python",
            ),
            _rate(
                word_error_rate(python_row["transcript"], row["transcript"]),
                policy["reference_wer_max"],
                f"onnx/{identifier}/wer_vs_reference",
                hypothesis=python_row["transcript"],
                reference=row["transcript"],
            ),
            _rate(
                word_error_rate(dotnet_row["audio_transcript"], row["transcript"]),
                policy["reference_wer_max"],
                f"dotnet-audio/{identifier}/wer_vs_reference",
            ),
        ]
    errors = sum(
        edit_distance(words(row["transcript"]), words(clips[identifier]["text"]))
        for identifier, row in expected.items()
    )
    total = sum(len(words(clip["text"])) for clip in clips.values())
    checks.append(
        _rate(
            errors / total if total else 1.0,
            policy["ground_truth_wer_max"],
            "reference/corpus/wer_vs_ground_truth",
            errors=errors,
            words=total,
        )
    )
    if dotnet.get("checks", {}).get("pre_cancelled_request") is not True:
        raise PipelineError(".NET cancellation evidence is missing or failed")
    failed = [check["id"] for check in checks if check["status"] != "passed"]
    return {
        "format": "starlings.whisper-development-parity",
        "format_version": 1,
        "profile": candidate["id"],
        "task": candidate["task"],
        "status": "failed" if failed else "passed",
        "failed_checks": failed,
        "checks": checks,
        "case_count": len(expected),
        "transcripts": {
            identifier: {
                "ground_truth": clips[identifier]["text"],
                "reference": row["transcript"],
                "python-onnx": python_cases[identifier]["transcript"],
                "dotnet-audio": dotnet_cases[identifier]["audio_transcript"],
            }
            for identifier, row in expected.items()
        },
        "evaluation_status": "proposed",
        "policy": policy,
        "protected_release_qualification": {"status": "unapproved", "G1": "not-run", "G2": "not-run"},
        "inputs": {
            "candidate_sha256": digest,
            "reference_sha256": reference_digest,
            "onnx_sha256": fixtures.sha256_file(onnx_path),
            "dotnet_sha256": fixtures.sha256_file(dotnet_path),
        },
    }


def render_report(report: dict, profile: str) -> str:
    lines = [
        f"# Development parity: {profile}",
        "",
        f"Status: **{report['status']}**; {report['case_count']} clips, {len(report['checks'])} checks. Original "
        "references and comparison policies are **proposed**. Protected G1/G2 release qualification is "
        "**unapproved**. No protected fixtures were adopted.",
        "",
        "| Clip | Ground truth | Reference (PyTorch fp32) | .NET (audio, fp16 ONNX) |",
        "| --- | --- | --- | --- |",
        *(
            f"| {identifier} | {row['ground_truth']} | {row['reference'].strip()} | {row['dotnet-audio'].strip()} |"
            for identifier, row in report["transcripts"].items()
        ),
        "",
    ]
    return "\n".join(lines)


def _profile(recipe: dict, root: Path) -> tuple[Path, dict, dict]:
    from model_pipeline.onboarding_candidate import load_candidate

    if recipe.get("plugin") != PLUGIN or recipe.get("recipe_version") != 1:
        raise PipelineError(f"expected recipe_version 1 and plugin {PLUGIN}")
    path = repository_input(root, recipe["manifest"])
    candidate = load_candidate(path)
    return path, candidate, validate_candidate_behavior(candidate, root)


def describe_plan(recipe: dict, *, repo_root: Path) -> dict:
    root = repo_root.resolve()
    _, candidate, contract = _profile(recipe, root)
    origin = candidate["origin"]
    return {
        "plugin": PLUGIN,
        "recipe": recipe["id"],
        "profile": candidate["id"],
        "task": candidate["task"],
        "revision": origin["revision"],
        "base": {"repo": origin["base"]["repo"], "revision": origin["base"]["revision"]},
        "asset_count": len(origin["files"]) + len(origin["base"]["files"]),
        "stages": list(STAGES),
        "inputs": {
            "snapshot": f"local {origin['repo']} snapshot at {origin['revision']}",
            "base_snapshot": f"local {origin['base']['repo']} snapshot at {origin['base']['revision']}",
            "reference_python": "reference/whisper/requirements.txt",
        },
        "reference_status": "proposed",
        "qualification_status": "unapproved",
        "policy": candidate["evaluation"]["policy"],
        "policy_source": contract["protected_tolerances"],
    }


def build_stages(
    recipe: dict,
    *,
    repo_root: Path,
    snapshot: Path,
    reference_python: Path,
    base_snapshot: Path | None = None,
    threads: int = 2,
    timeout: float = 1800,
    distribution: bool = False,
) -> list[Stage]:
    from model_pipeline.onboarding_candidate import candidate_stages, code_inputs, verify_assets

    if distribution:
        raise PipelineError("Whisper distribution is not implemented by this plugin")
    if base_snapshot is None:
        raise PipelineError("the Whisper route needs --base-snapshot (the pinned openai PyTorch checkpoint)")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise PipelineError("timeout must be finite, positive, and at most 86400 seconds")
    root, snapshot = repo_root.resolve(), snapshot.resolve()
    base_snapshot = base_snapshot.resolve()
    if not base_snapshot.is_dir():
        raise PipelineError(f"base snapshot is not a directory: {base_snapshot}")
    manifest, candidate, contract = _profile(recipe, root)
    corpus = repository_input(root, recipe["reference_corpus"])
    audio = {f"corpus_audio:{row['audio']}": row["path"] for row in load_corpus(corpus, candidate)}
    _, initial, shared = candidate_stages(
        recipe,
        repo_root=root,
        snapshot=snapshot,
        reference_python=reference_python,
        threads=threads,
        timeout=timeout,
        source_files=(
            RUNNER,
            "model-pipeline/src/model_pipeline/onboarding_whisper.py",
            contract["protected_tolerances"],
        ),
    )
    python = Path(os.path.abspath(reference_python))
    env = {
        "PYTHONPATH": str(root / "model-pipeline/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "DOTNET_PROCESSOR_COUNT": str(threads),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }
    parameters = {"plugin": PLUGIN, "candidate": candidate, "threads": threads, "timeout": timeout}

    def guard(out: Path) -> None:
        guard_output(
            out,
            (snapshot, base_snapshot, root / "model-pipeline/protected", corpus.parent, manifest),
            "stage output overlaps original source or protected evidence",
        )

    def verified(stage: Stage, *, base_too: bool) -> Stage:
        """Recheck the pins of everything the stage reads before it runs."""
        original = stage.execute

        def checked(out: Path, deps: dict) -> dict:
            guard(out)
            lock = verify_assets(candidate, snapshot, out)
            records = verify_base(candidate, base_snapshot) if base_too else []
            result = original(out, deps)
            return {
                **result,
                **lock,
                "base_files_verified": len(records),
                "original_source_pins": "verified-before-read",
            }

        return replace(stage, inputs={**stage.inputs, "original_snapshot": snapshot}, execute=checked)

    assets_stage, environment_stage = initial
    original_assets = assets_stage.execute

    def assets(out: Path, deps: dict) -> dict:
        result = original_assets(out, deps)
        lock = json.loads((out / "source-lock.json").read_text(encoding="utf-8"))
        lock["base"] = {
            "repo": candidate["origin"]["base"]["repo"],
            "revision": candidate["origin"]["base"]["revision"],
            "assets": verify_base(candidate, base_snapshot),
        }
        lock["contract_checks"] = check_configs(candidate, snapshot, base_snapshot)
        write_json(out / "source-lock.json", lock)
        return {**result, "base_files_verified": len(lock["base"]["assets"])}

    def runner(out: Path, kind: str, extra: list[str]) -> dict:
        guard(out)
        return run_command(
            [
                str(python),
                str(root / RUNNER),
                "--profile",
                str(manifest),
                "--stage",
                kind,
                "--out",
                str(out),
                "--threads",
                str(threads),
                *extra,
            ],
            cwd=root,
            log_path=out / "command.log",
            env=env,
            timeout=timeout,
        )

    def reference(out: Path, deps: dict) -> dict:
        return runner(out, "reference", ["--snapshot", str(base_snapshot), "--corpus", str(corpus)])

    def onnx(out: Path, deps: dict) -> dict:
        return runner(
            out,
            "onnx",
            [
                "--snapshot",
                str(snapshot),
                "--corpus",
                str(corpus),
                "--reference",
                str(deps["reference"] / "reference.json"),
            ],
        )

    def dotnet(out: Path, deps: dict) -> dict:
        guard(out)
        with tempfile.TemporaryDirectory(prefix="dotnet-build-", dir=out) as build:
            dotnet_executable = shutil.which("dotnet") or "dotnet"
            published = Path(build) / "published"
            build_result = run_command(
                [
                    dotnet_executable,
                    "publish",
                    str(root / TOOL),
                    "-c",
                    "Release",
                    "-m:1",
                    "--disable-build-servers",
                    "--artifacts-path",
                    build,
                    "--output",
                    str(published),
                    "-p:RestoreLockedMode=true",
                ],
                cwd=root,
                log_path=out / "build.log",
                env=env,
                timeout=timeout,
            )
            assembly = published / "Starlings.Whisper.Fidelity.Tool.dll"
            if not assembly.is_file():
                raise PipelineError("Whisper fidelity publish succeeded without producing the required tool assembly")
            result = run_command(
                [
                    dotnet_executable,
                    str(assembly),
                    "--profile",
                    str(manifest),
                    "--snapshot",
                    str(snapshot),
                    "--corpus",
                    str(corpus),
                    "--reference",
                    str(deps["reference"] / "reference.json"),
                    "--out",
                    str(out),
                    "--threads",
                    str(threads),
                ],
                cwd=root,
                log_path=out / "command.log",
                env=env,
                timeout=timeout,
            )
            return {**result, "build": build_result}

    def evaluation(out: Path, deps: dict) -> dict:
        guard(out)
        report = evaluate_development(
            candidate,
            manifest,
            deps["reference"] / "reference.json",
            deps["onnx"] / "onnx.json",
            deps["dotnet"] / "dotnet.json",
            load_corpus(corpus, candidate),
        )
        write_json(out / "report.json", report)
        (out / "report.md").write_text(render_report(report, candidate["id"]), encoding="utf-8")
        if report["status"] != "passed":
            raise PipelineError(
                f"Whisper development parity failed: {', '.join(report['failed_checks'])}; see {out / 'report.json'}"
            )
        return {"status": report["status"], "qualification_status": "unapproved", "reference_status": "proposed"}

    def review(out: Path, deps: dict) -> dict:
        guard(out)
        report = json.loads((deps["evaluate"] / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise PipelineError("review requires complete passed development parity")
        for name in ("report.json", "report.md"):
            shutil.copyfile(deps["evaluate"] / name, out / name)
        document = {
            "profile": candidate["id"],
            "task": candidate["task"],
            "development_parity": report,
            "candidate_commit": fixtures.git_head(root),
            "source_lock": json.loads((deps["assets"] / "source-lock.json").read_text(encoding="utf-8")),
            "support_status": "experimental",
            "release_qualification": "unapproved",
            "reference_status": "proposed",
            "protected_fixture_adoption": False,
            "distribution": "not-run",
        }
        write_json(out / "review.json", document)
        (out / "review.md").write_text(
            f"# Candidate review: {candidate['id']}\n\n"
            "Development transcript parity passed. References/policies remain proposed. "
            "Protected G1/G2 and full release qualification are unapproved; distribution is not run.\n",
            encoding="utf-8",
        )
        return {"development_parity": "passed", "release_qualification": "unapproved"}

    def stage(name: str, dependencies: tuple[str, ...], execute, inputs: dict | None = None, **extra) -> Stage:
        return Stage(name, dependencies, {**shared, **audio, **(inputs or {})}, {**parameters, **extra}, execute)

    dotnet_inputs = code_inputs(
        root,
        files=(
            "global.json",
            "Directory.Build.props",
            "ai-dotnet/Directory.Build.props",
            "ai-dotnet/Directory.Build.targets",
            "ai-dotnet/Directory.Packages.props",
            *(str(path.relative_to(root)) for path in (root / "Directory.Build.targets",) if path.is_file()),
        ),
        directories=(
            "ai-dotnet/tools/Starlings.Whisper.Fidelity.Tool",
            "ai-dotnet/tools/Shared",
            "ai-dotnet/src/Starlings.Models.Whisper",
            "ai-dotnet/src/Starlings.Runtime.Onnx",
            "ai-dotnet/src/Starlings.Acquisition",
        ),
    )
    return [
        replace(assets_stage, inputs={**assets_stage.inputs, "base_snapshot": base_snapshot}, execute=assets),
        environment_stage,
        verified(stage("reference", ("assets", "environment"), reference, {"corpus": corpus}), base_too=True),
        verified(stage("onnx", ("reference", "assets", "environment"), onnx), base_too=False),
        verified(stage("dotnet", ("reference", "assets", "environment"), dotnet, dotnet_inputs), base_too=False),
        stage("evaluate", ("reference", "onnx", "dotnet"), evaluation),
        stage("review", ("assets", "reference", "evaluate"), review, candidate_commit=fixtures.git_head(root)),
    ]
