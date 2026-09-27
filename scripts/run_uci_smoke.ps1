param(
    [Parameter(Position = 0)]
    [string]$EngineBin = ".\build\Release\chess-engine-uci.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $EngineBin -PathType Leaf)) {
    throw "Engine binary not found: $EngineBin"
}

$EngineBin = (Resolve-Path -LiteralPath $EngineBin).Path

$output = & {
    "uci"
    "isready"
    "setoption name Hash value 128"
    "setoption name ProbCut value true"
    "ucinewgame"
    "position startpos moves e2e4 e7e5 g1f3 b8c6"
    "go movetime 300"
    Start-Sleep -Milliseconds 600
    "position startpos"
    "go searchmoves a2a3 movetime 200"
    Start-Sleep -Milliseconds 400
    "position startpos"
    "go searchmoves b2b3 wtime 20 btime 20 winc 20 binc 20"
    Start-Sleep -Milliseconds 300
    "quit"
} | & $EngineBin --uci 2>&1

if ($LASTEXITCODE -ne 0) {
    $output | Write-Error
    throw "Engine exited with code $LASTEXITCODE"
}

function Assert-UciOutput {
    param(
        [string]$Pattern,
        [string]$Failure
    )

    if (-not ($output | Select-String -Pattern $Pattern -Quiet)) {
        $output | Write-Error
        throw $Failure
    }
}

Assert-UciOutput '^uciok$' "Missing uciok"
Assert-UciOutput '^readyok$' "Missing readyok"
Assert-UciOutput '^bestmove [a-h][1-8][a-h][1-8][qrbn]?$' "Missing or invalid bestmove"
Assert-UciOutput '^bestmove a2a3$' "UCI searchmoves restriction was not respected"
Assert-UciOutput '^bestmove b2b3$' "Low-clock UCI search did not complete with its restricted move"
Assert-UciOutput '^option name Clear Hash type button$' "Clear Hash option was not advertised"
Assert-UciOutput '^option name Move Overhead type spin default 25 min 0 max 5000$' `
    "Move Overhead option was not advertised"
Assert-UciOutput '^option name EvalFile type string default <empty>$' `
    "EvalFile option was not advertised"
Assert-UciOutput '^option name Use NNUE type check default false$' `
    "Use NNUE option was not advertised"
Assert-UciOutput '^option name ProbCut type check default false$' `
    "ProbCut option was not advertised"

Write-Output "UCI smoke: PASS"
