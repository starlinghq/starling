#Requires -Version 7.2
<#
.SYNOPSIS
    Windows clean-install drill for the Starlings packages (plan 10 G6; production-readiness R9). Mirrors
    tools/release-drill-ai.sh for the Windows x64 CI job (R6).

.DESCRIPTION
    Proves a new Windows user can install and run the release candidate without the source checkout, a hidden cache or
    an upstream tokenizer. From a clean clone of the committed HEAD it
      0. records the inputs (commit, SDK, offline method),
      1. clones the committed HEAD into an empty temporary directory and checks the clone is clean,
      2. reads the package and tokenizer versions and checks the samples and manifests agree,
      3. packs the ai-dotnet packages with a locked restore into an empty package cache,
      4. inspects every nuspec (version, license, readme, repository commit, exact dependency list) and the packed files
         (no model assets, no FFmpeg library) and the symbol packages (SourceLink to the commit),
      5. pushes the packages to a local folder feed only (never nuget.org),
      6. copies the samples and the checkers (tools/release-drill/ai-consumer, mlnet-consumer, bert-consumer) outside the repository,
         with a NuGet.config that knows only that feed and nuget.org and an isolated empty package cache, restores and
         builds them, and proves the restore graph holds only packages, that the Starlings packages are the packed bytes
         (Starlings.Tokenizers: the nuget.org bytes the lock files record) and that no foreign tokenizer, GPU native or
         FFmpeg library is restored or built,
      7. acquires the e5 model into an isolated empty cache through the sample's explicit download step,
      8. runs the e5 sample, the checker (protected fixtures within the protected bounds) and the ML.NET checker with no
         network,
      9. with -SeedGliner2From / -SeedEmbeddingGemma2From / -SeedQwen35From: seeds the graph files that are not
         hosted publicly yet and runs those samples (and the EmbeddingGemma 2 ML.NET checker) with no network,
     11. with -PublicHosting: downloads GLiNER2, EmbeddingGemma 2 and Qwen 3.5 from public hosting with no token and no
         seed (public since 2026-10-14; without the switch it prints "skipped: not requested").

    Differences from the Linux drill, all recorded as untested on Windows rather than skipped silently:
      - No network namespace: "offline" means HTTP_PROXY, HTTPS_PROXY and ALL_PROXY point at a closed local port, which
        .NET's HttpClient honors on Windows; the drill checks that the Hub is unreachable that way first.
      - The EmbeddingGemma 2 video search (user-supplied FFmpeg) and the image search are not run: the FFmpeg adapter
        and SkiaSharp's Windows natives are untested on Windows and outside this drill's scope.
      - The Native AOT check (tools/aot-check-ai.sh) is Linux only; Native AOT on Windows is untested.

    NOT RUN YET: this script was written on Linux and has not been executed on Windows. The first run is the R6
    Windows CI job's; treat its first failures as defects of this script before reading them as package defects.

.EXAMPLE
    pwsh tools/release-drill-ai.ps1 -NoModel
.EXAMPLE
    pwsh tools/release-drill-ai.ps1 -SeedModelFrom C:\models\e5-snapshot -SeedGliner2From C:\models\gliner2
#>
[CmdletBinding()]
param(
    # Copy onnx/model.onnx of e5 from a local snapshot into the isolated cache first (tokenizer.json is still downloaded).
    [string]$SeedModelFrom,
    # CI: no model download; downloads tokenizer.json only and skips the embedding checks.
    [switch]$NoModel,
    # Directory holding onnx/gliner2-entities-fp32.onnx.
    [string]$SeedGliner2From,
    # Directory holding onnx/embeddinggemma2-text-fp32.onnx.
    [string]$SeedEmbeddingGemma2From,
    # A bundle built by build_ortgenai_bundle.py --variant fp32x.
    [string]$SeedQwen35From,
    # Download GLiNER2, EmbeddingGemma 2 and Qwen 3.5 from public hosting.
    [switch]$PublicHosting,
    # Keep the temporary directory for inspection.
    [switch]$Keep
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $false

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$upstreamUrl = 'https://github.com/starlinghq/starling.git'
$e5Repo = 'intfloat/multilingual-e5-small'
$e5Commit = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
$e5ModelSha256 = 'ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665'
$fixturesRel = 'model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu'
$gliner2Repo = 'fastino/gliner2-base-v1'
$gliner2Commit = 'f9634218e53580c56edf0de97ca1a7d3f1c2354e'
$gliner2Graph = 'onnx/gliner2-entities-fp32.onnx'
$eg2Repo = 'google/embeddinggemma-2'
$eg2Commit = '914f7f89142e33e77833254d9c9b90c3cef7303b'
$eg2Graph = 'onnx/embeddinggemma2-text-fp32.onnx'
$skiaVersion = '4.153.1'
$ortVersion = '1.30.0'
$genaiVersion = '0.17.1'
$qwenRepo = 'Qwen/Qwen3.5-0.8B'
$qwenCommit = '2fc06364715b967f1860aea9cf38778875588b17'
$mode = if ($NoModel) { 'no-model' } elseif ($SeedModelFrom) { 'seed' } else { 'download' }

$env:DOTNET_NOLOGO = '1'
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = '1'
# Nothing from the developer's environment may point the apps at an existing model cache or token.
foreach ($name in 'STARLING_MODELS', 'STARLING_EMBEDDINGGEMMA2_SNAPSHOT', 'STARLING_EMBEDDINGGEMMA2_IMAGES',
    'STARLING_EMBEDDINGGEMMA2_VIDEOS', 'STARLING_FFMPEG_DIR', 'STARLING_E5_SNAPSHOT', 'STARLING_GLINER2_SNAPSHOT',
    'STARLING_GLINER2_FIXTURES', 'STARLING_QWEN35_SNAPSHOT', 'HF_TOKEN') {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("starlings-drill-" + [System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $work | Out-Null

function Step([string]$Text) { Write-Host "`n== $Text" }
function Fail([string]$Text) { throw "DRILL FAIL: $Text" }

# Runs a native command, tees its output to a log and fails on a non-zero exit code.
function Run {
    param([string]$Log, [string]$What, [string]$Exe, [string[]]$Arguments)
    $output = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    if ($Log) { $output | Set-Content -Path $Log -Encoding utf8 }
    $output | Write-Host
    if ($code -ne 0) { Fail "$What (exit code $code)" }
    return , $output
}

# Same, with every proxy variable pointing at a closed port (Windows has no unshare; .NET honors these variables).
function RunOffline {
    param([string]$Log, [string]$What, [string]$Exe, [string[]]$Arguments)
    $saved = @{}
    foreach ($name in 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY') { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }
    try {
        $env:HTTP_PROXY = 'http://127.0.0.1:9'; $env:HTTPS_PROXY = 'http://127.0.0.1:9'; $env:ALL_PROXY = 'http://127.0.0.1:9'
        Remove-Item Env:NO_PROXY -ErrorAction SilentlyContinue
        return Run -Log $Log -What $What -Exe $Exe -Arguments $Arguments
    }
    finally {
        foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
    }
}

function Assert-Match([string[]]$Lines, [string]$Pattern, [string]$What) {
    if (-not ($Lines | Where-Object { $_ -match $Pattern })) { Fail $What }
}

# The line after "<prefix><query>" must start with "  top: <top> ".
function Assert-Top([string[]]$Lines, [string]$Header, [string]$Top, [string]$What) {
    $i = [Array]::IndexOf($Lines, $Header)
    if ($i -lt 0 -or $i + 1 -ge $Lines.Count -or -not $Lines[$i + 1].StartsWith("  top: $Top ")) { Fail $What }
}

try {
    Step '0. Inputs'
    $sha = (git -C $root rev-parse HEAD).Trim()
    Write-Host "commit:    $sha"
    if (git -C $root status --porcelain) { Write-Host "note:      working tree has uncommitted changes; the drill ignores them and builds $sha" }
    Push-Location $work; $sdkOutside = (dotnet --version); Pop-Location
    Write-Host "dotnet:    $((Get-Command dotnet).Source) (SDK in the clone: pinned by global.json; outside: $sdkOutside)"
    Write-Host "os:        $([System.Runtime.InteropServices.RuntimeInformation]::OSDescription) $([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)"
    Write-Host "mode:      $mode"
    Write-Host 'offline:   proxy variables pointing at a closed port (no network namespace on Windows)'
    Write-Host "work dir:  $work"

    Step "1. Clean checkout of $sha"
    $src = Join-Path $work 'src'
    git clone --quiet --no-hardlinks $root $src
    if ($LASTEXITCODE -ne 0) { Fail 'clone failed' }
    git -C $src -c advice.detachedHead=false checkout --quiet --detach $sha
    git -C $src remote set-url origin $upstreamUrl
    if (git -C $src status --porcelain --ignored) { Fail 'clone is not clean' }
    Write-Host 'ok:   no build outputs, caches or untracked files in the checkout'

    Step '2. Versions'
    $env:NUGET_PACKAGES = Join-Path $work 'nuget-build'
    $env:NUGET_HTTP_CACHE_PATH = Join-Path $work 'nuget-http-build'
    $props = Get-Content (Join-Path $src 'ai-dotnet/Directory.Packages.props') -Raw
    if ($props -notmatch '<PackageVersion Include="Starlings.Tokenizers" Version="\[([^],]*)\]"') { Fail 'ai-dotnet/Directory.Packages.props does not pin Starlings.Tokenizers exactly' }
    $tokVersion = $Matches[1]
    $tokHashes = @(Get-ChildItem (Join-Path $src 'ai-dotnet/src') -Filter packages.lock.json -Recurse -Depth 1 | ForEach-Object {
            $lock = Get-Content $_.FullName -Raw | ConvertFrom-Json -AsHashtable
            foreach ($deps in $lock['dependencies'].Values) { if ($deps.ContainsKey('Starlings.Tokenizers')) { $deps['Starlings.Tokenizers']['contentHash'] } }
        } | Sort-Object -Unique)
    if ($tokHashes.Count -ne 1) { Fail "lock files record $($tokHashes.Count) Starlings.Tokenizers hashes" }
    $tokSha512 = $tokHashes[0]
    Push-Location $src; $aiVersion = (dotnet msbuild ai-dotnet/src/Starlings.Models.E5 -getProperty:Version).Trim(); Pop-Location
    Write-Host "Starlings.Tokenizers: $tokVersion"
    Write-Host "ai-dotnet packages:   $aiVersion"
    # Version scheme (docs/kickoff.md): 0.MINOR.PATCH, or 0.MINOR.PATCH-preview.N before it; maturity is per profile.
    if ($aiVersion -notmatch '^0\.[0-9]+\.[0-9]+(-preview\.[0-9]+)?$') { Fail "ai-dotnet version $aiVersion is not 0.MINOR.PATCH or 0.MINOR.PATCH-preview.N" }
    if ($tokVersion -notmatch '^0\.\d+\.\d+(-preview\.\d+)?$') { Fail "tokenizer version $tokVersion is not 0.MINOR.PATCH or 0.MINOR.PATCH-preview.N" }
    # NuGet refuses a stable package with a prerelease dependency (NU5104).
    if ($aiVersion -notlike '*-preview.*' -and $tokVersion -like '*-preview.*') { Fail "ai-dotnet $aiVersion is a release but pins the prerelease tokenizer $tokVersion" }
    foreach ($sample in 'E5Search', 'Gliner2Ner', 'EmbeddingGemma2Search', 'QwenChat') {
        $csproj = Get-Content (Join-Path $src "ai-dotnet/samples/$sample/$sample.csproj") -Raw
        if ($csproj -notmatch '<StarlingsVersion>(.*)</StarlingsVersion>' -or $Matches[1] -ne $aiVersion) { Fail "$sample references another version than $aiVersion" }
    }
    foreach ($manifest in Get-ChildItem (Join-Path $src 'model-pipeline/registry') -Filter manifest.yaml -Recurse) {
        $line = Select-String -Path $manifest.FullName -Pattern '^    version: (.*)$' | Select-Object -First 1
        if ($line -and $line.Matches[0].Groups[1].Value -ne $tokVersion) { Fail "$($manifest.FullName) pins tokenizer $($line.Matches[0].Groups[1].Value), the tokenizer is $tokVersion" }
    }
    Write-Host "ok:   samples reference $aiVersion; manifests pin tokenizer $tokVersion"

    Step '3. Pack (locked restore, empty package cache)'
    $pack = Join-Path $work 'pack'
    $projects = @(
        'Starlings.Acquisition', 'Starlings.Runtime.Onnx', 'Starlings.Models.Abstractions', 'Starlings.Models.E5', 'Starlings.Models.E5.Extensions.AI',
        'Starlings.Runtime.OnnxWeb', 'Starlings.Runtime.OnnxWeb.WebGpu', 'Starlings.Models.E5.Browser',
        'Starlings.Models.E5.MLNet', 'Starlings.Models.Gliner2', 'Starlings.Models.EmbeddingGemma2',
        'Starlings.Models.EmbeddingGemma2.Extensions.AI', 'Starlings.Models.EmbeddingGemma2.MLNet',
        'Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp', 'Starlings.Models.EmbeddingGemma2.Video.FFmpeg',
        'Starlings.Generation.Abstractions', 'Starlings.Generation.OnnxGenAI', 'Starlings.Models.Qwen35',
        'Starlings.Models.Qwen35.OnnxGenAI', 'Starlings.Models.Qwen35.Extensions.AI', 'Starlings.Models.Kev',
        'Starlings.Models.Kev.Onnx', 'Starlings.Models.QwenChat', 'Starlings.Models.QwenChat.OnnxGenAI', 'Starlings.Models.Bert',
        'Starlings.Models.Lfm3')
    $env:CI = 'true'
    Push-Location $src
    foreach ($id in $projects) {
        Run -Log (Join-Path $work "pack-$id.log") -What "pack failed: $id" -Exe dotnet -Arguments @('pack', "ai-dotnet/src/$id", '--configuration', 'Release', '--output', $pack) | Out-Null
    }
    Pop-Location
    Remove-Item Env:CI
    Get-ChildItem $pack | ForEach-Object Name

    Step '4. Inspect packages'
    $expectedDeps = @{
        'Starlings.Acquisition'                              = @()
        'Starlings.Runtime.Onnx'                             = @("Microsoft.ML.OnnxRuntime.Managed [$ortVersion]")
        'Starlings.Models.Abstractions'                      = @("Starlings.Acquisition [$aiVersion]")
        'Starlings.Models.E5'                                = @("Starlings.Acquisition [$aiVersion]", "Starlings.Models.Abstractions [$aiVersion]", "Starlings.Runtime.Onnx [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Runtime.OnnxWeb'                          = @()
        'Starlings.Runtime.OnnxWeb.WebGpu'                   = @("Starlings.Runtime.OnnxWeb [$aiVersion]")
        'Starlings.Models.E5.Browser'                        = @("Starlings.Models.E5 [$aiVersion]", "Starlings.Models.E5.Extensions.AI [$aiVersion]", "Starlings.Runtime.OnnxWeb [$aiVersion]")
        'Starlings.Models.E5.Extensions.AI'                  = @("Starlings.Models.E5 [$aiVersion]", 'Microsoft.Extensions.AI.Abstractions 10.10.1', 'Microsoft.Extensions.DependencyInjection.Abstractions 10.0.12')
        'Starlings.Models.E5.MLNet'                          = @("Starlings.Models.E5 [$aiVersion]", 'Microsoft.ML 5.0.0')
        'Starlings.Models.Gliner2'                           = @("Starlings.Acquisition [$aiVersion]", "Starlings.Models.Abstractions [$aiVersion]", "Starlings.Runtime.Onnx [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.EmbeddingGemma2'                   = @("Starlings.Acquisition [$aiVersion]", "Starlings.Models.Abstractions [$aiVersion]", "Starlings.Runtime.Onnx [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.EmbeddingGemma2.Extensions.AI'     = @("Starlings.Models.EmbeddingGemma2 [$aiVersion]", 'Microsoft.Extensions.AI.Abstractions 10.10.1', 'Microsoft.Extensions.DependencyInjection.Abstractions 10.0.12')
        'Starlings.Models.EmbeddingGemma2.MLNet'             = @("Starlings.Models.EmbeddingGemma2 [$aiVersion]", 'Microsoft.ML 5.0.0')
        'Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp' = @("Starlings.Models.EmbeddingGemma2 [$aiVersion]", "SkiaSharp [$skiaVersion]", "SkiaSharp.NativeAssets.Linux.NoDependencies [$skiaVersion]")
        'Starlings.Models.EmbeddingGemma2.Video.FFmpeg'     = @("Starlings.Models.EmbeddingGemma2 [$aiVersion]")
        'Starlings.Generation.Abstractions'                  = @("Starlings.Tokenizers [$tokVersion]", 'Microsoft.Extensions.AI.Abstractions 10.10.1')
        'Starlings.Generation.OnnxGenAI'                     = @("Starlings.Generation.Abstractions [$aiVersion]", "Starlings.Tokenizers [$tokVersion]", "Microsoft.ML.OnnxRuntimeGenAI.Managed [$genaiVersion]")
        'Starlings.Models.Qwen35'                            = @("Starlings.Acquisition [$aiVersion]", "Starlings.Generation.Abstractions [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.Qwen35.OnnxGenAI'                  = @("Starlings.Acquisition [$aiVersion]", "Starlings.Generation.OnnxGenAI [$aiVersion]", "Starlings.Models.Abstractions [$aiVersion]", "Starlings.Models.Qwen35 [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.Qwen35.Extensions.AI'              = @("Starlings.Models.Qwen35.OnnxGenAI [$aiVersion]", 'Microsoft.Extensions.AI.Abstractions 10.10.1', 'Microsoft.Extensions.DependencyInjection.Abstractions 10.0.12')
        'Starlings.Models.Kev'                               = @("Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.Kev.Onnx'                          = @("Starlings.Acquisition [$aiVersion]", "Starlings.Models.Abstractions [$aiVersion]", "Starlings.Models.Kev [$aiVersion]", "Starlings.Runtime.Onnx [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.QwenChat'                          = @("Starlings.Generation.Abstractions [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.QwenChat.OnnxGenAI'                = @("Starlings.Acquisition [$aiVersion]", "Starlings.Generation.OnnxGenAI [$aiVersion]", "Starlings.Models.Abstractions [$aiVersion]", "Starlings.Models.QwenChat [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.Bert'                              = @("Starlings.Acquisition [$aiVersion]", "Starlings.Models.Abstractions [$aiVersion]", "Starlings.Runtime.Onnx [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
        'Starlings.Models.Lfm3'                              = @("Starlings.Acquisition [$aiVersion]", "Starlings.Generation.Abstractions [$aiVersion]", "Starlings.Models.Abstractions [$aiVersion]", "Starlings.Runtime.Onnx [$aiVersion]", "Starlings.Tokenizers [$tokVersion]")
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    function Read-Entry([string]$Zip, [string]$Name) {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
        try {
            $entry = $archive.GetEntry($Name)
            if (-not $entry) { return $null }
            $stream = $entry.Open(); $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::Latin1)
            try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
        }
        finally { $archive.Dispose() }
    }
    function Get-Entries([string]$Zip) {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
        try { return @($archive.Entries | ForEach-Object FullName) } finally { $archive.Dispose() }
    }
    foreach ($id in $projects) {
        $nupkg = Join-Path $pack "$id.$aiVersion.nupkg"
        $snupkg = Join-Path $pack "$id.$aiVersion.snupkg"
        if (-not (Test-Path $nupkg)) { Fail "expected $nupkg" }
        if (-not (Test-Path $snupkg)) { Fail "symbol package missing for $id" }
        $nuspec = Read-Entry $nupkg "$id.nuspec"
        $contents = Get-Entries $nupkg
        if ($nuspec -notmatch "<version>$([regex]::Escape($aiVersion))</version>") { Fail "${id}: nuspec version" }
        if ($id -eq 'Starlings.Models.Lfm3') {
            # The LFM Open License v1.0 is this package's license file, with acceptance required (admissions/lfm3.md).
            if ($nuspec -notmatch '<license type="file">LICENSE.txt</license>') { Fail "${id}: nuspec license file" }
            if ($nuspec -notmatch '<requireLicenseAcceptance>true</requireLicenseAcceptance>') { Fail "${id}: license acceptance not required" }
            if ($nuspec -notmatch '<description>MODEL LICENSE: LFM Open License v1.0') { Fail "${id}: description does not open with the model license" }
            $licenseText = Read-Entry $nupkg 'LICENSE.txt'
            if ($licenseText -notmatch 'USD 10,000,000') { Fail "${id}: packed LICENSE.txt lacks the revenue condition" }
            if ($licenseText -notmatch '(?m)^END OF TERMS AND CONDITIONS') { Fail "${id}: packed LICENSE.txt lacks the full license text" }
            if ($contents -notcontains 'THIRD-PARTY-NOTICES.md') { Fail "${id}: package lacks THIRD-PARTY-NOTICES.md" }
        } elseif ($nuspec -notmatch '<license type="expression">MIT OR Apache-2.0</license>') { Fail "${id}: nuspec license" }
        if ($nuspec -notmatch '<readme>README.md</readme>') { Fail "${id}: nuspec readme" }
        if ($nuspec -notmatch "url=`"https://github.com/starlinghq/starling`".*commit=`"$sha`"") { Fail "${id}: nuspec repository commit" }
        # Maturity is per profile: the description points to the README's generated maturity block. No blanket Preview label.
        if ($nuspec -match '<description>Preview\. ') { Fail "${id}: description starts with a blanket Preview label" }
        if ($nuspec -notmatch 'Maturity is per model profile') { Fail "${id}: description does not state per-profile maturity" }
        $readmeText = Read-Entry $nupkg 'README.md'
        if ($readmeText -notmatch '(?m)^<!-- catalog:maturity profiles=') { Fail "${id}: packed README lacks the generated maturity block" }
        if ($readmeText -notmatch '(?m)^\| Profile \| Maturity \| Platforms \|') { Fail "${id}: packed README maturity block is empty" }
        if ($readmeText -match '\*\*Preview\.\*\*') { Fail "${id}: packed README carries a blanket Preview label" }
        $actual = @([regex]::Matches($nuspec, '<dependency id="([^"]*)" version="([^"]*)"') | ForEach-Object { "$($_.Groups[1].Value) $($_.Groups[2].Value)" } | Sort-Object)
        $expected = @($expectedDeps[$id] | Sort-Object)
        if (($actual -join "`n") -ne ($expected -join "`n")) { Fail "${id}: dependencies`n$($actual -join "`n")`nexpected`n$($expected -join "`n")" }
        foreach ($f in "lib/net10.0/$id.dll", "lib/net10.0/$id.xml", 'README.md') {
            if ($contents -notcontains $f) { Fail "${id}: package lacks $f" }
        }
        if ($contents | Where-Object { $_ -match '(?i)(^|/)(lib)?(avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample|postproc)([-.][0-9]+)*\.(so|dll|dylib)(\.[0-9]+)*$' }) { Fail "${id}: package contains an FFmpeg library" }
        if ($contents | Where-Object { $_ -match '(?i)\.(onnx|safetensors|bin|gguf)$|tokenizer\.json$' }) { Fail "${id}: package contains model assets" }
        $pdb = Read-Entry $snupkg "lib/net10.0/$id.pdb"
        if (-not $pdb -or -not $pdb.Contains("raw.githubusercontent.com/starlinghq/starling/$sha/*")) { Fail "${id}: symbols lack SourceLink to $sha" }
        $licenseLabel = if ($id -eq 'Starlings.Models.Lfm3') { 'LFM Open License v1.0 (license file, acceptance required)' } else { 'MIT OR Apache-2.0' }
        Write-Host "ok:   $id ${aiVersion}: $licenseLabel, per-profile maturity, repository commit, SourceLink; dependencies: $(if ($expected) { $expected -join ', ' } else { 'none' })"
        Write-Host "      sha256 $((Get-FileHash $nupkg -Algorithm SHA256).Hash.ToLowerInvariant())"
    }

    Step '5. Push to a local folder feed (never nuget.org)'
    $feed = Join-Path $work 'feed'
    New-Item -ItemType Directory -Path $feed | Out-Null
    foreach ($nupkg in Get-ChildItem $pack -Filter *.nupkg) {
        dotnet nuget push $nupkg.FullName --source $feed | Out-Null
        if ($LASTEXITCODE -ne 0) { Fail "push failed: $($nupkg.Name)" }
    }
    Get-ChildItem $feed -Recurse -Filter *.nupkg | ForEach-Object Name

    Step '6. Clean consumers outside the repository'
    $apps = Join-Path $work 'apps'
    $fixtures = Join-Path $work 'fixtures'
    $appNames = 'E5Search', 'Gliner2Ner', 'QwenChat', 'EmbeddingGemma2Search', 'AiConsumer', 'MlNetConsumer', 'BertConsumer'
    foreach ($app in $appNames) { New-Item -ItemType Directory -Path (Join-Path $apps $app) -Force | Out-Null }
    New-Item -ItemType Directory -Path $fixtures | Out-Null
    foreach ($sample in 'E5Search', 'Gliner2Ner', 'EmbeddingGemma2Search', 'QwenChat') {
        Copy-Item (Join-Path $src "ai-dotnet/samples/$sample/Program.cs"), (Join-Path $src "ai-dotnet/samples/$sample/$sample.csproj") (Join-Path $apps $sample)
    }
    foreach ($checker in @(@('ai-consumer', 'AiConsumer'), @('mlnet-consumer', 'MlNetConsumer'), @('bert-consumer', 'BertConsumer'))) {
        Copy-Item (Join-Path $src "tools/release-drill/$($checker[0])/Program.cs") (Join-Path $apps $checker[1])
        (Get-Content (Join-Path $src "tools/release-drill/$($checker[0])/$($checker[1]).csproj.in") -Raw).Replace('@VERSION@', $aiVersion) |
            Set-Content (Join-Path $apps "$($checker[1])/$($checker[1]).csproj") -Encoding utf8
    }
    Copy-Item (Join-Path $src "$fixturesRel/embeddings.jsonl"), (Join-Path $src "$fixturesRel/tokenizer.jsonl") $fixtures
    @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="drill-local" value="$feed" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <fallbackPackageFolders>
    <clear />
  </fallbackPackageFolders>
  <packageSourceMapping>
    <packageSource key="drill-local">
      <package pattern="Starlings.*" />
    </packageSource>
    <packageSource key="nuget.org">
      <package pattern="Starlings.Tokenizers" />
      <package pattern="*" />
    </packageSource>
  </packageSourceMapping>
</configuration>
"@ | Set-Content (Join-Path $apps 'NuGet.config') -Encoding utf8
    $env:NUGET_PACKAGES = Join-Path $work 'nuget-consumer'
    $env:NUGET_HTTP_CACHE_PATH = Join-Path $work 'nuget-http-consumer'
    foreach ($app in $appNames) {
        Push-Location (Join-Path $apps $app)
        Run -Log (Join-Path $work "restore-$app.log") -What "$app restore failed" -Exe dotnet -Arguments @('restore') | Out-Null
        Run -Log (Join-Path $work "$app-packages.txt") -What "$app package list failed" -Exe dotnet -Arguments @('list', 'package', '--include-transitive') | Out-Null
        Pop-Location
    }

    # The restore graphs: only packages; Starlings.* are the packed bytes; no foreign tokenizer, no GPU natives,
    # SkiaSharp only in the image sample, no FFmpeg package; the expected model packages exactly once.
    $sha512 = [System.Security.Cryptography.SHA512]::Create()
    foreach ($app in $appNames) {
        $assets = Get-Content (Join-Path $apps "$app/obj/project.assets.json") -Raw | ConvertFrom-Json -AsHashtable
        $libraries = $assets['libraries']
        $names = @($libraries.Keys | Sort-Object)
        Write-Host "${app}: $($names.Count) packages: $($names -join ', ')"
        if ($libraries.Values | Where-Object { $_['type'] -ne 'package' }) { Fail "${app}: non-package library (project reference)" }
        $foreign = @($names | Where-Object { $_ -match '(?i)tokeni[sz]er' -and -not $_.StartsWith('Starlings.Tokenizers/') })
        if ($foreign) { Fail "${app}: foreign tokenizer packages: $($foreign -join ', ')" }
        if ($names | Where-Object { $_ -match '(?i)^Microsoft\.ML\.OnnxRuntime(GenAI\.Cuda|\.Gpu)' }) { Fail "${app}: GPU natives restored" }
        if ($names | Where-Object { $_ -match '(?i)^SixLabors\.' }) { Fail "${app}: ImageSharp restored" }
        if (($names | Where-Object { $_ -match '(?i)^skiasharp' }) -and $app -ne 'EmbeddingGemma2Search') { Fail "${app}: SkiaSharp restored outside the image sample" }
        if ($names | Where-Object { $_ -match '(?i)sdcb\.ffmpeg|ffmpeg\.(native|runtime|redist)|ffmpeg\.autogen|ffmpegcore|xabe\.ffmpeg' }) { Fail "${app}: FFmpeg bindings or binaries restored" }
        $want = switch ($app) {
            'QwenChat' { 'Starlings.Tokenizers', 'Starlings.Acquisition', 'Microsoft.ML.OnnxRuntime', 'Microsoft.ML.OnnxRuntimeGenAI', 'Microsoft.ML.OnnxRuntimeGenAI.Managed', 'Starlings.Generation.Abstractions', 'Starlings.Generation.OnnxGenAI', 'Starlings.Models.Qwen35', 'Starlings.Models.Qwen35.OnnxGenAI', 'Starlings.Models.Qwen35.Extensions.AI', 'Starlings.Models.Abstractions' }
            'Gliner2Ner' { 'Starlings.Tokenizers', 'Starlings.Acquisition', 'Starlings.Runtime.Onnx', 'Starlings.Models.Abstractions', 'Microsoft.ML.OnnxRuntime', 'Starlings.Models.Gliner2' }
            'EmbeddingGemma2Search' { 'Starlings.Tokenizers', 'Starlings.Acquisition', 'Starlings.Runtime.Onnx', 'Starlings.Models.Abstractions', 'Microsoft.ML.OnnxRuntime', 'Starlings.Models.EmbeddingGemma2', 'Starlings.Models.EmbeddingGemma2.Extensions.AI', 'Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp', 'Starlings.Models.EmbeddingGemma2.Video.FFmpeg' }
            'MlNetConsumer' { 'Starlings.Tokenizers', 'Starlings.Acquisition', 'Starlings.Runtime.Onnx', 'Starlings.Models.Abstractions', 'Microsoft.ML.OnnxRuntime', 'Starlings.Models.E5', 'Starlings.Models.E5.MLNet', 'Starlings.Models.EmbeddingGemma2', 'Starlings.Models.EmbeddingGemma2.MLNet', 'Microsoft.ML' }
            'BertConsumer' { 'Starlings.Tokenizers', 'Starlings.Acquisition', 'Starlings.Runtime.Onnx', 'Starlings.Models.Abstractions', 'Microsoft.ML.OnnxRuntime', 'Starlings.Models.Bert' }
            'AiConsumer' { 'Starlings.Tokenizers', 'Starlings.Acquisition', 'Starlings.Runtime.Onnx', 'Starlings.Models.Abstractions', 'Microsoft.ML.OnnxRuntime', 'Starlings.Models.E5', 'Starlings.Models.E5.Extensions.AI', 'Starlings.Models.E5.MLNet', 'Microsoft.ML' }
            default { 'Starlings.Tokenizers', 'Starlings.Acquisition', 'Starlings.Runtime.Onnx', 'Starlings.Models.Abstractions', 'Microsoft.ML.OnnxRuntime', 'Starlings.Models.E5', 'Starlings.Models.E5.Extensions.AI' }
        }
        foreach ($package in $want) {
            $hits = @($names | Where-Object { $_.Split('/')[0] -eq $package })
            if ($hits.Count -ne 1) { Fail "${app}: $package not restored exactly once" }
            $hash = $libraries[$hits[0]]['sha512']
            if ($package -eq 'Starlings.Tokenizers') {
                if ($hash -ne $tokSha512) { Fail "${app}: $($hits[0]) differs from the locked nuget.org package" }
            }
            elseif ($package.StartsWith('Starlings.')) {
                $version = $hits[0].Split('/')[1]
                $packed = [Convert]::ToBase64String($sha512.ComputeHash([System.IO.File]::ReadAllBytes((Join-Path $pack "$package.$version.nupkg"))))
                if ($hash -ne $packed) { Fail "${app}: $($hits[0]) differs from the packed package" }
            }
        }
    }
    Write-Host 'ok:   only packages; Starlings.* sha512 equal the packed .nupkg files and the locked tokenizer; no foreign tokenizer, no GPU natives'
    $foreignFiles = @(Get-ChildItem $env:NUGET_PACKAGES -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '(?i)tokeni[sz]er' -and $_.FullName -notmatch '(?i)[\\/]starlings\.tokenizers' })
    if ($foreignFiles) { Fail "foreign tokenizer files in the consumer package cache: $($foreignFiles[0].FullName)" }
    foreach ($app in $appNames) {
        Push-Location (Join-Path $apps $app)
        Run -Log (Join-Path $work "build-$app.log") -What "$app build failed" -Exe dotnet -Arguments @('build', '--configuration', 'Release', '--no-restore') | Out-Null
        Pop-Location
    }
    if (Get-ChildItem $apps -Recurse | Where-Object { $_.FullName -match '[\\/]bin[\\/]' -and $_.Name -match '(?i)tokeni[sz]er' -and $_.Name -notlike 'Starlings.Tokenizers.*' }) { Fail 'foreign tokenizer files in a build output' }
    if (Get-ChildItem $apps -Recurse -File | Where-Object { $_.Extension -ne '.log' } | Select-String -SimpleMatch -Pattern $root -List) { Fail 'consumer files mention the repository path' }
    if (Get-ChildItem $env:NUGET_PACKAGES, $apps -Recurse -File | Where-Object { $_.Name -match '(?i)^(lib)?(avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample|postproc)([-.][0-9]+)*\.(so|dll|dylib)(\.[0-9]+)*$' }) { Fail 'an FFmpeg library is in the consumer package cache or a build output' }
    Write-Host 'ok:   no repository path in the consumers, no foreign tokenizer, no FFmpeg library in any cache or output'

    $bin = { param($app) Join-Path $apps "$app/bin/Release/net10.0/$app.dll" }
    $models = Join-Path $work 'models'
    New-Item -ItemType Directory -Path $models | Out-Null

    # Every mode: the built-in BERT profiles from the packed package, offline (no model; the snapshot directory is empty).
    $bertEmpty = Join-Path $work 'bert-empty'
    New-Item -ItemType Directory -Path $bertEmpty | Out-Null
    $log = RunOffline -Log (Join-Path $work 'consumer-bert.log') -What 'BERT checker failed' -Exe dotnet -Arguments @((& $bin 'BertConsumer'), $bertEmpty)
    Assert-Match $log '^RESULT: PASS$' 'BERT checker did not pass'
    Write-Host 'ok:   Starlings.Models.Bert: the seven built-in profiles parse from the package alone, offline'

    if ($mode -eq 'no-model') {
        Step '7. Acquisition of tokenizer.json only (real Hub download; no model)'
        $log = Run -Log (Join-Path $work 'consumer-tokenizer.log') -What 'checker failed' -Exe dotnet -Arguments @((& $bin 'AiConsumer'), 'tokenizer', $models, $fixtures)
        Assert-Match $log '^RESULT: PASS$' 'checker did not pass'
        Write-Host 'note: -NoModel: the model download, offline search, embedding and ML.NET checks were skipped'
    }
    else {
        Step '7. Explicit acquisition into an isolated empty cache (real Hub download)'
        $e5Snapshot = Join-Path $models "$e5Repo/$e5Commit"
        if ($mode -eq 'seed') {
            New-Item -ItemType Directory -Path (Join-Path $e5Snapshot 'onnx') -Force | Out-Null
            Copy-Item (Join-Path $SeedModelFrom 'onnx/model.onnx') (Join-Path $e5Snapshot 'onnx/model.onnx')
            Write-Host "seeded: onnx/model.onnx from $SeedModelFrom (the store verifies size and SHA-256 before use)"
        }
        $log = Run -Log (Join-Path $work 'sample-download.log') -What 'sample download failed' -Exe dotnet -Arguments @((& $bin 'E5Search'), 'download', '--cache', $models)
        Assert-Match $log '^tokenizer.json: downloaded,' 'tokenizer.json was not downloaded'
        if ($mode -eq 'download') { Assert-Match $log '^onnx/model.onnx: downloaded,' 'onnx/model.onnx was not downloaded' }
        if ((Get-FileHash (Join-Path $e5Snapshot 'onnx/model.onnx') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $e5ModelSha256) { Fail 'model digest' }

        Step '8. Offline: sample search and fixture checks with no network'
        $probe = RunOffline -Log $null -What 'offline probe' -Exe pwsh -Arguments @('-NoProfile', '-Command', 'try { [System.Net.Http.HttpClient]::new().GetAsync(''https://huggingface.co'').GetAwaiter().GetResult() | Out-Null; ''reachable'' } catch { ''unreachable'' }')
        if ($probe -contains 'reachable') { Fail 'the offline proxy setting still reaches huggingface.co' }
        Write-Host 'ok:   huggingface.co unreachable through the closed-port proxy'
        $log = RunOffline -Log (Join-Path $work 'sample-search.log') -What 'sample search failed' -Exe dotnet -Arguments @((& $bin 'E5Search'), 'search', '--cache', $models)
        foreach ($pair in @(@('how much protein should a female eat', 'protein'), @('Warum rostet Eisen?', 'rust'), @('日本の首都はどこですか', 'tokyo'), @('highest mountain in the world', 'summit'))) {
            Assert-Top $log "query: $($pair[0])" $pair[1] "sample: '$($pair[0])' did not rank '$($pair[1])' first"
        }
        Assert-Match $log 'max abs difference from the typed API 0$' 'Extensions.AI path differs from the typed API'
        $tolerances = Get-Content (Join-Path $src "$fixturesRel/tolerances.yaml")
        $section = ''; $bounds = @{}
        foreach ($line in $tolerances) {
            if ($line -match '^(\S+)') { $section = $Matches[1] }
            if ($section -eq 'float:' -and $line -match '^\s+(embedding_max_abs|padded_batch_vs_single_embedding_max_abs):\s*(\S+)') { $bounds[$Matches[1]] = $Matches[2] }
        }
        if ($bounds.Count -ne 2) { Fail "could not read the bounds from $fixturesRel/tolerances.yaml" }
        Write-Host "bounds (protected): float.embedding_max_abs $($bounds['embedding_max_abs']), float.padded_batch_vs_single_embedding_max_abs $($bounds['padded_batch_vs_single_embedding_max_abs'])"
        $log = RunOffline -Log (Join-Path $work 'consumer-full.log') -What 'checker failed' -Exe dotnet -Arguments @((& $bin 'AiConsumer'), 'full', $models, $fixtures, $bounds['embedding_max_abs'], $bounds['padded_batch_vs_single_embedding_max_abs'])
        Assert-Match $log '^RESULT: PASS$' 'checker did not pass'

        Step '8c. Offline: ML.NET checker on e5 (Starlings.Models.E5.MLNet)'
        $log = RunOffline -Log (Join-Path $work 'consumer-mlnet-e5.log') -What 'ML.NET checker (e5) failed' -Exe dotnet -Arguments @((& $bin 'MlNetConsumer'), 'e5', $models)
        Assert-Match $log '^RESULT: PASS$' 'ML.NET checker (e5) did not pass'
    }

    if ($SeedGliner2From) {
        Step '9. GLiNER2: seeded graph, tokenizer.json from the Hub, offline extraction'
        $target = Join-Path $models "$gliner2Repo/$gliner2Commit/onnx"
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Copy-Item (Join-Path $SeedGliner2From $gliner2Graph) $target
        $log = Run -Log (Join-Path $work 'sample-gliner2-download.log') -What 'Gliner2Ner download failed' -Exe dotnet -Arguments @((& $bin 'Gliner2Ner'), 'download', '--cache', $models)
        Assert-Match $log '^tokenizer.json: downloaded,' 'GLiNER2 tokenizer.json was not downloaded'
        Assert-Match $log "^$([regex]::Escape($gliner2Graph)): already cached," 'the seeded GLiNER2 graph was not verified from the cache'
        $log = RunOffline -Log (Join-Path $work 'sample-gliner2-extract.log') -What 'Gliner2Ner extract failed' -Exe dotnet -Arguments @((& $bin 'Gliner2Ner'), 'extract', '--cache', $models)
        foreach ($expect in 'organization: Apple \[0, 5\)', 'person: Tim Cook \[10, 18\)', 'location: Cupertino \[46, 55\)', 'person: Marie Curie \[0, 11\)', 'person: Pierre Curie \[', 'long input rejected: [0-9]+ tokens > 512') {
            Assert-Match $log "^ *$expect" "Gliner2Ner: expected '$expect'"
        }
        Write-Host 'ok:   GLiNER2 sample extracts the expected entities offline and rejects the 600-word input'
    }
    else { Write-Host 'note: no -SeedGliner2From: the GLiNER2 package was packed, inspected and its sample built, but not run' }

    if ($SeedEmbeddingGemma2From) {
        Step '9b. EmbeddingGemma 2: seeded graph, tokenizer.json and prompt config from the Hub, offline search'
        $target = Join-Path $models "$eg2Repo/$eg2Commit/onnx"
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Copy-Item (Join-Path $SeedEmbeddingGemma2From $eg2Graph) $target
        $log = Run -Log (Join-Path $work 'sample-eg2-download.log') -What 'EmbeddingGemma2Search download failed' -Exe dotnet -Arguments @((& $bin 'EmbeddingGemma2Search'), 'download', '--cache', $models)
        Assert-Match $log '^tokenizer.json: downloaded,' 'EmbeddingGemma 2 tokenizer.json was not downloaded'
        Assert-Match $log "^$([regex]::Escape($eg2Graph)): already cached," 'the seeded EmbeddingGemma 2 graph was not verified from the cache'
        $log = RunOffline -Log (Join-Path $work 'sample-eg2-search.log') -What 'EmbeddingGemma2Search search failed' -Exe dotnet -Arguments @((& $bin 'EmbeddingGemma2Search'), 'search', '--cache', $models)
        foreach ($pair in @(@('how much protein should a female eat', 'protein'), @('Warum rostet Eisen?', 'rust'), @('日本の首都はどこですか', 'tokyo'), @('highest mountain in the world', 'summit'))) {
            Assert-Top $log "query: $($pair[0])" $pair[1] "EmbeddingGemma2Search: '$($pair[0])' did not rank '$($pair[1])' first"
        }
        Assert-Match $log 'max abs difference from the typed API 0$' 'EmbeddingGemma 2 Extensions.AI path differs from the typed API'
        Assert-Match $log '^long input rejected: [0-9]+ tokens > 8192' 'EmbeddingGemma 2 did not reject the long input'
        Step '9e. Offline: ML.NET checker on EmbeddingGemma 2 (Starlings.Models.EmbeddingGemma2.MLNet)'
        $log = RunOffline -Log (Join-Path $work 'consumer-mlnet-eg2.log') -What 'ML.NET checker (EmbeddingGemma 2) failed' -Exe dotnet -Arguments @((& $bin 'MlNetConsumer'), 'eg2', $models)
        Assert-Match $log '^RESULT: PASS$' 'ML.NET checker (EmbeddingGemma 2) did not pass'
        Write-Host 'note: the EmbeddingGemma 2 image and video searches are not part of the Windows drill (SkiaSharp Windows natives and the FFmpeg adapter are untested on Windows)'
    }
    else { Write-Host 'note: no -SeedEmbeddingGemma2From: the EmbeddingGemma 2 packages (including ML.NET) were packed, inspected and the sample and checker built, but not run' }

    if ($SeedQwen35From) {
        Step '10. Qwen3.5: seeded files verified by the download step, offline streamed answer'
        $target = Join-Path $models "$qwenRepo/$qwenCommit"
        New-Item -ItemType Directory -Path (Join-Path $target 'ortgenai-fp32x'), (Join-Path $target 'transformers-5.18.0') -Force | Out-Null
        foreach ($f in 'genai_config.json', 'model.onnx', 'model.onnx.data') { Copy-Item (Join-Path $SeedQwen35From "ortgenai-fp32x/$f") (Join-Path $target 'ortgenai-fp32x') }
        Copy-Item (Join-Path $SeedQwen35From 'transformers-5.18.0/tokenizer.json') (Join-Path $target 'transformers-5.18.0')
        Copy-Item (Join-Path $SeedQwen35From 'chat_template.jinja') $target
        $log = Run -Log (Join-Path $work 'sample-qwen-download.log') -What 'QwenChat download failed' -Exe dotnet -Arguments @((& $bin 'QwenChat'), 'download', '--cache', $models)
        Assert-Match $log '^ortgenai-fp32x/model.onnx.data: already cached,' 'the seeded graph was not verified from the cache'
        $ask = RunOffline -Log (Join-Path $work 'sample-qwen-ask.log') -What 'QwenChat ask failed' -Exe dotnet -Arguments @((& $bin 'QwenChat'), 'ask', '--cache', $models, '--max-new', '64', 'What is the capital of France? Answer in one sentence.')
        Assert-Match $ask '^assistant: .*Paris' 'QwenChat: expected Paris in the answer'
        Assert-Match $ask '^\[Stop; 24 prompt \+ [0-9]+ completion tokens' 'QwenChat: expected a Stop finish with 24 prompt tokens'
        if ($ask | Where-Object { $_ -match 'im_end' }) { Fail 'QwenChat: the stop token leaked into the text' }
        $client = RunOffline -Log (Join-Path $work 'sample-qwen-ask-chat-client.log') -What 'QwenChat ask --chat-client failed' -Exe dotnet -Arguments @((& $bin 'QwenChat'), 'ask', '--cache', $models, '--max-new', '64', '--chat-client', 'What is the capital of France? Answer in one sentence.')
        if ((@($ask | Where-Object { $_ -like 'assistant: *' }) -join "`n") -ne (@($client | Where-Object { $_ -like 'assistant: *' }) -join "`n")) { Fail 'QwenChat --chat-client: the IChatClient answer differs from the typed API' }
        Write-Host 'ok:   QwenChat answers offline from the packages; the IChatClient adapter gives the same answer'
    }
    else { Write-Host 'note: no -SeedQwen35From: the pilot LLM packages were packed, inspected and the QwenChat sample built, but not run' }

    Step '11. Download from public hosting (no token, no seed)'
    $hosting = @(
        @('GLiNER2', 'ai-dotnet/src/Starlings.Models.Gliner2/Gliner2ModelAssets.cs', 'ModelSourceIsPlaceholder = true'),
        @('EmbeddingGemma 2', 'ai-dotnet/src/Starlings.Models.EmbeddingGemma2/EmbeddingGemma2ModelAssets.cs', 'ModelSourceIsPlaceholder = true'),
        @('Qwen 3.5', 'ai-dotnet/src/Starlings.Models.Qwen35.OnnxGenAI/Qwen35ModelAssets.cs', 'HostingIsPrivate = true'))
    $publicRan = @()
    if (-not $PublicHosting) {
        Write-Host 'skipped: not requested (pass -PublicHosting to download the GLiNER2, EmbeddingGemma 2 and Qwen 3.5 files, about 4 GB, from their public hosting)'
    }
    else {
        $publicModels = Join-Path $work 'models-public'
        New-Item -ItemType Directory -Path $publicModels | Out-Null
        foreach ($h in $hosting) {
            if (Select-String -Path (Join-Path $src $h[1]) -SimpleMatch -Pattern $h[2] -Quiet) {
                Write-Host "skipped: hosting private ($($h[0]): $($h[1]) still records $($h[2]))"
                continue
            }
            switch ($h[0]) {
                'GLiNER2' {
                    $log = Run -Log (Join-Path $work 'public-gliner2-download.log') -What 'GLiNER2 public download failed' -Exe dotnet -Arguments @((& $bin 'Gliner2Ner'), 'download', '--cache', $publicModels)
                    Assert-Match $log "^$([regex]::Escape($gliner2Graph)): downloaded," 'GLiNER2 graph was not downloaded'
                    $log = RunOffline -Log (Join-Path $work 'public-gliner2-extract.log') -What 'GLiNER2 extract from public files failed' -Exe dotnet -Arguments @((& $bin 'Gliner2Ner'), 'extract', '--cache', $publicModels)
                    Assert-Match $log '^ *person: Tim Cook \[10, 18\)' "GLiNER2 from public files: expected 'person: Tim Cook [10, 18)'"
                }
                'EmbeddingGemma 2' {
                    $log = Run -Log (Join-Path $work 'public-eg2-download.log') -What 'EmbeddingGemma 2 public download failed' -Exe dotnet -Arguments @((& $bin 'EmbeddingGemma2Search'), 'download', '--cache', $publicModels)
                    Assert-Match $log "^$([regex]::Escape($eg2Graph)): downloaded," 'EmbeddingGemma 2 graph was not downloaded'
                    $log = RunOffline -Log (Join-Path $work 'public-eg2-search.log') -What 'EmbeddingGemma 2 search from public files failed' -Exe dotnet -Arguments @((& $bin 'EmbeddingGemma2Search'), 'search', '--cache', $publicModels)
                    Assert-Top $log 'query: how much protein should a female eat' 'protein' "EmbeddingGemma 2 from public files: expected 'protein' first"
                }
                'Qwen 3.5' {
                    $log = Run -Log (Join-Path $work 'public-qwen-download.log') -What 'Qwen 3.5 public download failed' -Exe dotnet -Arguments @((& $bin 'QwenChat'), 'download', '--cache', $publicModels)
                    Assert-Match $log '^ortgenai-fp32x/model.onnx.data: downloaded,' 'Qwen 3.5 graph was not downloaded'
                    $log = RunOffline -Log (Join-Path $work 'public-qwen-ask.log') -What 'QwenChat from public files failed' -Exe dotnet -Arguments @((& $bin 'QwenChat'), 'ask', '--cache', $publicModels, '--max-new', '64', 'What is the capital of France? Answer in one sentence.')
                    Assert-Match $log '^assistant: .*Paris' 'QwenChat from public files: expected Paris'
                }
            }
            Write-Host "ok:   $($h[0]) downloaded from public hosting with no token and no seed, and ran offline from that cache"
            $publicRan += $h[0]
        }
    }

    $out = Join-Path $root 'artifacts/release-drill-ai-windows'
    if (Test-Path $out) { Remove-Item $out -Recurse -Force }
    New-Item -ItemType Directory -Path $out | Out-Null
    Copy-Item (Join-Path $pack '*.nupkg'), (Join-Path $pack '*.snupkg') $out
    Get-ChildItem $work -File | Where-Object { $_.Name -match '^(sample-|consumer-|public-).*\.log$|-packages\.txt$' } | Copy-Item -Destination $out
    $labels = @($mode)
    if ($SeedGliner2From) { $labels += 'GLiNER2 run' }
    if ($SeedEmbeddingGemma2From) { $labels += 'EmbeddingGemma 2 run' }
    if ($SeedQwen35From) { $labels += 'Qwen3.5 run' }
    if ($publicRan) { $labels += "public hosting: $($publicRan -join ', ')" }
    Step "DRILL PASS on Windows ($($labels -join ', ')): Starlings $aiVersion + Tokenizers $tokVersion from $sha (packages and logs in artifacts/release-drill-ai-windows/)"
}
finally {
    if ($Keep) { Write-Host "kept: $work" }
    else { Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue }
}
