param(
    [string]$RunDir = 'E:\Dev\Forklift-Research\matches\absolute-calibration-main-20260826',
    [string]$Rungs = '2200,2400,2600,2800,3000,3190',
    [int]$GamesPerRung = 200,
    [string]$TimeControl = '10+0.1',
    [int]$Concurrency = 6,
    [int]$Seed = 1701,
    [int]$Port = 8766,
    [string]$EngineRef = 'HEAD',
    [string[]]$HistoryRunDir = @(),
    [string]$MatchRunDir = '',
    [switch]$EnableMatchControls
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$resolvedRunDir = [System.IO.Path]::GetFullPath($RunDir)
$stockfish = Join-Path $repoRoot '.tools\user-engines\external-stockfish-18\stockfish-windows-x86-64.exe'
$dashboardScript = Join-Path $repoRoot 'scripts\calibration_dashboard.py'
$calibrationScript = Join-Path $repoRoot 'scripts\calibrate_rating.py'
$pythonLauncher = (Get-Command py -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath $stockfish -PathType Leaf)) {
    throw "Stockfish 18 is missing: $stockfish"
}
if ($GamesPerRung -lt 2 -or $GamesPerRung % 2 -ne 0) {
    throw 'GamesPerRung must be a positive even number.'
}
if ($Concurrency -lt 1) {
    throw 'Concurrency must be positive.'
}
if ($Seed -lt 0) {
    throw 'Seed must be non-negative.'
}

New-Item -ItemType Directory -Force -Path $resolvedRunDir | Out-Null

function Test-RecordedProcess([string]$PidPath) {
    if (-not (Test-Path -LiteralPath $PidPath -PathType Leaf)) { return $false }
    $recordedPid = 0
    if (-not [int]::TryParse((Get-Content -Raw -LiteralPath $PidPath).Trim(), [ref]$recordedPid)) {
        return $false
    }
    return $null -ne (Get-Process -Id $recordedPid -ErrorAction SilentlyContinue)
}

$dashboardPidPath = Join-Path $resolvedRunDir 'dashboard.pid'
if (-not (Test-RecordedProcess $dashboardPidPath)) {
    $dashboardArguments = @('-3', $dashboardScript, '--run-dir', $resolvedRunDir, '--port', $Port)
    foreach ($historyDirectory in $HistoryRunDir) {
        $dashboardArguments += @('--history-run-dir', [System.IO.Path]::GetFullPath($historyDirectory))
    }
    if ($MatchRunDir) {
        $dashboardArguments += @('--match-run-dir', [System.IO.Path]::GetFullPath($MatchRunDir))
        if ($EnableMatchControls) {
            $dashboardArguments += '--enable-match-controls'
        }
    }
    $dashboard = Start-Process -FilePath $pythonLauncher `
        -ArgumentList $dashboardArguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput (Join-Path $resolvedRunDir 'dashboard.stdout.log') `
        -RedirectStandardError (Join-Path $resolvedRunDir 'dashboard.stderr.log') `
        -WindowStyle Hidden -PassThru
    [System.IO.File]::WriteAllText($dashboardPidPath, [string]$dashboard.Id)
}

$commit = (& git -C $repoRoot rev-parse "$EngineRef^{commit}").Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to resolve engine ref: $EngineRef" }

$calibrationPidPath = Join-Path $resolvedRunDir 'calibration.pid'
if (-not (Test-RecordedProcess $calibrationPidPath)) {
    $calibrationArguments = @(
        '-3', $calibrationScript,
        '--stockfish-exe', $stockfish,
        '--stockfish-version', '18',
        '--engine-ref', $commit,
        '--engine-name', 'Forklift-calibration',
        '--engine-version', $commit.Substring(0, 10),
        '--rungs', $Rungs,
        '--games', $GamesPerRung,
        '--tc', $TimeControl,
        '--threads', '1',
        '--hash', '256',
        '--concurrency', $Concurrency,
        '--build-jobs', '12',
        '--seed', $Seed,
        '--run-dir', $resolvedRunDir
    )
    $calibration = Start-Process -FilePath $pythonLauncher `
        -ArgumentList $calibrationArguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput (Join-Path $resolvedRunDir 'calibration.stdout.log') `
        -RedirectStandardError (Join-Path $resolvedRunDir 'calibration.stderr.log') `
        -WindowStyle Hidden -PassThru
    [System.IO.File]::WriteAllText($calibrationPidPath, [string]$calibration.Id)
}

[pscustomobject]@{
    RunDirectory = $resolvedRunDir
    LocalDashboard = "http://localhost:$Port"
    LanDashboard = "http://192.168.4.59:$Port"
    TailscaleDashboard = "http://100.105.68.41:$Port"
    Rungs = $Rungs
    GamesPerRung = $GamesPerRung
    TimeControl = $TimeControl
    Concurrency = $Concurrency
    Seed = $Seed
    EngineCommit = $commit
    HistoryRunDirectories = $HistoryRunDir
}
