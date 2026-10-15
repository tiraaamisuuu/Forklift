param(
    [switch]$Native,
    [switch]$NoOpen,
    [switch]$CheckOnly,
    [ValidateRange(1, 65535)][int]$Port = 8765,
    [string]$NetworkPath
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $NetworkPath) {
    $NetworkPath = Join-Path $RepositoryRoot "networks\verified-shared-v1\model.nnue"
}
if (-not (Test-Path -LiteralPath $NetworkPath -PathType Leaf)) {
    throw "The approved model is not installed: $NetworkPath. See docs/NNUE.md; this launcher never downloads or substitutes an experimental model."
}
$NetworkPath = (Resolve-Path -LiteralPath $NetworkPath).Path
$ExpectedNetworkSha256 = "ff25cf447dc6137591ecc2522d23f14e2f9bee0a641dda36ec2cd9c67a762c7c"
if ((Get-FileHash -LiteralPath $NetworkPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedNetworkSha256) {
    throw "The network checksum does not match the verified shared V1 model."
}

# Native is an explicit, non-portable local choice; it is never selected just
# because an executable happens to exist. The measured PC supports AVX2.
$BuildName = if ($Native) { "build-nnue-native-ipo-screen" } else { "build-singular" }
$EnginePath = Join-Path $RepositoryRoot "$BuildName\Release\chess-engine-uci.exe"
if (-not (Test-Path -LiteralPath $EnginePath -PathType Leaf)) {
    throw "Build the requested engine first: $EnginePath. See docs/NNUE.md."
}
if ($Native -and -not $CheckOnly) {
    Write-Host "Using the local AVX2 + link-time-optimised build; this executable requires an AVX2-capable CPU."
}
$ServerArguments = @(
    "--verified-nnue", $NetworkPath,
    "--default-profile", "verified-shared-nnue",
    "--port", "$Port"
)
if ($NoOpen) { $ServerArguments += "--no-open" }
if ($CheckOnly) {
    [PSCustomObject]@{
        Engine = $EnginePath
        Model = $NetworkPath
        ModelSha256 = $ExpectedNetworkSha256
        NativeAvx2Required = [bool]$Native
        ServerArguments = $ServerArguments
    } | ConvertTo-Json
    exit 0
}
$PreviousEngineOverride = $env:ENGINE_BIN
try {
    $env:ENGINE_BIN = $EnginePath
    & (Join-Path $RepositoryRoot "scripts\run_web_gui.ps1") @ServerArguments
    $ServerExitCode = $LASTEXITCODE
} finally {
    $env:ENGINE_BIN = $PreviousEngineOverride
}
exit $ServerExitCode
