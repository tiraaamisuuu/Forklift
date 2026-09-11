#!/usr/bin/env python3
"""Serve a small live dashboard for a Forklift Stockfish calibration ladder."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "web" / "calibration-dashboard.html"


class FILETIME(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("memory_load", ctypes.c_uint32),
        ("total_physical", ctypes.c_uint64),
        ("available_physical", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64),
        ("available_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("available_virtual", ctypes.c_uint64),
        ("available_extended_virtual", ctypes.c_uint64),
    ]


_cpu_lock = threading.Lock()
_last_cpu_times: tuple[int, int, int] | None = None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--history-run-dir",
        type=Path,
        action="append",
        default=[],
        help="completed earlier ladder directory to include (repeatable)",
    )
    parser.add_argument(
        "--match-run-dir",
        type=Path,
        help="direct engine-match run directory to show above the calibration",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--enable-match-controls",
        action="store_true",
        help="allow same-origin web requests to pause and restart the displayed match",
    )
    parser.add_argument("--snapshot", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return 0


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def next_resume_directory(run_dir: Path) -> Path:
    base_name = re.sub(r"-resume-\d{3}$", "", run_dir.name)
    for index in range(1, 1000):
        candidate = run_dir.parent / f"{base_name}-resume-{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("unable to allocate a fresh resume directory")


def latest_control_run(run_dir: Path) -> Path:
    current = run_dir.resolve()
    visited: set[Path] = set()
    while current not in visited:
        visited.add(current)
        state = read_json(current / "dashboard-control.json")
        resumed_as = state.get("resumedAs")
        if not isinstance(resumed_as, str) or not resumed_as:
            break
        next_run = Path(resumed_as).expanduser().resolve()
        if next_run.parent != current.parent or not next_run.is_dir():
            break
        current = next_run
    return current


def champion_resume_command(run_dir: Path, destination: Path) -> tuple[list[str], int]:
    manifest = read_json(run_dir / "gate-manifest.json")
    candidate = manifest.get("candidateCommit")
    contract = manifest.get("contract")
    contract = contract if isinstance(contract, dict) else {}
    if not isinstance(candidate, str) or len(candidate) != 40 or not contract:
        raise RuntimeError("this stopped run is not a resumable champion gate")

    seed = int(contract.get("seed", 0) or 0) + 1
    command = [sys.executable, str(ROOT / "scripts" / "champion_gate.py")]
    champion_registry = manifest.get("championRegistry")
    if isinstance(champion_registry, str) and champion_registry:
        command.extend(["--champion-file", champion_registry])
    command.extend(
        [
            "run",
            "--candidate",
            candidate,
            "--run-dir",
            str(destination),
            "--games",
            str(int(contract.get("games", 0) or 0)),
            "--tc",
            str(contract.get("timeControl", "")),
            "--threads",
            str(int(contract.get("threads", 1) or 1)),
            "--hash",
            str(int(contract.get("hashMb", 256) or 256)),
            "--concurrency",
            str(int(contract.get("concurrency", 1) or 1)),
            "--seed",
            str(seed),
            "--elo0",
            str(float(contract.get("elo0", 0.0) or 0.0)),
            "--elo1",
            str(float(contract.get("elo1", 5.0) or 5.0)),
            "--build-jobs",
            "12",
        ]
    )
    return command, seed


def windows_process_command_line(pid: int) -> str:
    if os.name != "nt":
        return ""
    script = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' "
        "-ErrorAction SilentlyContinue; if($p){$p.CommandLine}"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.stdout.strip()


class MatchController:
    def __init__(self, run_dir: Path, enabled: bool):
        self._run_dir = latest_control_run(run_dir)
        self.enabled = enabled and os.name == "nt"
        self._lock = threading.Lock()
        self.message = ""

    @property
    def run_dir(self) -> Path:
        with self._lock:
            return self._run_dir

    def capabilities(self, snapshot: dict[str, object]) -> dict[str, object]:
        resumable = bool(read_json(self.run_dir / "gate-manifest.json"))
        running = snapshot.get("running") is True
        complete = snapshot.get("state") == "complete"
        return {
            "enabled": self.enabled,
            "canPause": self.enabled and running,
            "canResume": self.enabled and not running and not complete and resumable,
            "resumeStartsFresh": True,
            "message": self.message,
            "runDirectory": str(self.run_dir),
        }

    def pause(self, snapshot: dict[str, object]) -> dict[str, object]:
        if not self.enabled:
            raise RuntimeError("web match controls are disabled")
        with self._lock:
            run_dir = self._run_dir
            pid = read_pid(run_dir / "match.pid")
            if not process_alive(pid):
                raise RuntimeError("the match is not running")
            command_line = windows_process_command_line(pid)
            if (
                str(run_dir).replace("/", "\\").casefold()
                not in command_line.replace("/", "\\").casefold()
                or not re.search(r"(?:champion_gate|compare_engines)\.py", command_line)
            ):
                raise RuntimeError("refusing to stop an unverified process")

            result = subprocess.run(
                ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if result.returncode != 0 and process_alive(pid):
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"unable to stop the match: {detail}")

            state = {
                "schemaVersion": 1,
                "status": "paused",
                "pausedAt": utc_now(),
                "pid": pid,
                "partial": {
                    key: snapshot.get(key)
                    for key in ("games", "wins", "draws", "losses", "scorePercent")
                },
            }
            write_json(run_dir / "dashboard-control.json", state)
            self.message = "Paused; partial evidence preserved"
            return {"ok": True, "state": "paused", "message": self.message}

    def resume(self) -> dict[str, object]:
        if not self.enabled:
            raise RuntimeError("web match controls are disabled")
        with self._lock:
            source = self._run_dir
            pid = read_pid(source / "match.pid")
            if process_alive(pid):
                raise RuntimeError("the match is already running")
            if read_json(source / "match" / "result.json").get("completed") is True:
                raise RuntimeError("a completed match cannot be resumed")

            destination = next_resume_directory(source)
            command, seed = champion_resume_command(source, destination)
            stdout_path = destination.parent / f"{destination.name}.stdout.log"
            stderr_path = destination.parent / f"{destination.name}.stderr.log"
            creation_flags = 0x08000000 | 0x00000200  # no window, new process group
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creation_flags,
                )

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if destination.is_dir():
                    break
                if process.poll() is not None:
                    detail = read_text(stderr_path).strip() or read_text(stdout_path).strip()
                    raise RuntimeError(f"new match failed to start: {detail}")
                time.sleep(0.1)
            if not destination.is_dir():
                process.terminate()
                raise RuntimeError("new match did not create its run directory")

            (destination / "match.pid").write_text(str(process.pid), encoding="ascii")
            (destination / "gate.pid").write_text(str(process.pid), encoding="ascii")
            write_json(
                source / "dashboard-control.json",
                {
                    "schemaVersion": 1,
                    "status": "resumed",
                    "resumedAt": utc_now(),
                    "resumedAs": str(destination),
                    "seed": seed,
                },
            )
            write_json(
                destination / "dashboard-control.json",
                {
                    "schemaVersion": 1,
                    "status": "running",
                    "startedAt": utc_now(),
                    "resumedFrom": str(source),
                    "pid": process.pid,
                    "seed": seed,
                },
            )
            self._run_dir = destination
            self.message = f"Fresh attempt started with seed {seed}"
            return {
                "ok": True,
                "state": "running",
                "message": self.message,
                "runDirectory": str(destination),
                "pid": process.pid,
                "seed": seed,
            }


def parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "calculating"
    rounded = int(seconds)
    days, remainder = divmod(rounded, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _filetime_value(value: FILETIME) -> int:
    return (value.high << 32) | value.low


def system_resources() -> dict[str, float | int | None]:
    global _last_cpu_times

    cpu_percent = None
    ram_percent = None
    ram_used = None
    ram_total = None
    if os.name == "nt":
        idle = FILETIME()
        kernel = FILETIME()
        user = FILETIME()
        if ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            current = (
                _filetime_value(idle),
                _filetime_value(kernel),
                _filetime_value(user),
            )
            with _cpu_lock:
                previous = _last_cpu_times
                _last_cpu_times = current
            if previous:
                idle_delta = current[0] - previous[0]
                total_delta = (
                    current[1] - previous[1] + current[2] - previous[2]
                )
                if total_delta > 0:
                    cpu_percent = 100.0 * (1.0 - idle_delta / total_delta)

        memory = MEMORYSTATUSEX()
        memory.length = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            ram_percent = float(memory.memory_load)
            ram_total = int(memory.total_physical)
            ram_used = ram_total - int(memory.available_physical)

    return {
        "cpuPercent": round(cpu_percent, 1) if cpu_percent is not None else None,
        "ramPercent": round(ram_percent, 1) if ram_percent is not None else None,
        "ramUsed": ram_used,
        "ramTotal": ram_total,
    }


SCORE_PATTERN = re.compile(
    r"^Score of .+? vs .+?:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)"
    r"\s+\[([0-9.]+)]\s+(\d+)\s*$",
    re.MULTILINE,
)
ELO_PATTERN = re.compile(
    r"^Elo difference:\s+([+\-]?(?:[0-9.]+|inf|nan))"
    r"\s+\+/-\s+([+\-]?(?:[0-9.]+|inf|nan))",
    re.IGNORECASE | re.MULTILINE,
)
SPRT_PATTERN = re.compile(
    r"^SPRT:\s+llr\s+([+\-]?(?:[0-9.eE]+|inf|nan))\s+\([^)]*\),"
    r"\s+lbound\s+([+\-]?(?:[0-9.eE]+|inf|nan)),"
    r"\s+ubound\s+([+\-]?(?:[0-9.eE]+|inf|nan))"
    r"(?:\s+-\s+H[01]\s+was\s+accepted)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_live_result(log_text: str) -> dict[str, object]:
    scores = list(SCORE_PATTERN.finditer(log_text))
    elo_rows = list(ELO_PATTERN.finditer(log_text))
    if scores:
        score = scores[-1]
        wins = int(score.group(1))
        losses = int(score.group(2))
        draws = int(score.group(3))
        games = int(score.group(5))
        score_fraction = float(score.group(4))
    else:
        finished = set(
            int(value)
            for value in re.findall(r"(?m)^Finished game (\d+)\s", log_text)
        )
        wins = losses = draws = 0
        games = len(finished)
        score_fraction = None

    relative_elo = None
    uncertainty = None
    if elo_rows:
        relative_elo = finite_float(elo_rows[-1].group(1))
        uncertainty = finite_float(elo_rows[-1].group(2))
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games,
        "score": score_fraction,
        "relativeElo": relative_elo,
        "uncertainty": uncertainty,
    }


def parse_live_sprt(log_text: str, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {"enabled": False, "decision": "disabled"}
    matches = list(SPRT_PATTERN.finditer(log_text))
    if not matches:
        return {"enabled": True, "decision": "collecting"}
    match = matches[-1]
    llr = finite_float(match.group(1))
    lower = finite_float(match.group(2))
    upper = finite_float(match.group(3))
    decision = "inconclusive"
    if llr is not None and lower is not None and upper is not None:
        if llr >= upper:
            decision = "accepted_h1"
        elif llr <= lower:
            decision = "accepted_h0"
    return {
        "enabled": True,
        "decision": decision,
        "llr": llr,
        "lowerBound": lower,
        "upperBound": upper,
    }


def build_match_snapshot(run_dir: Path) -> dict[str, object]:
    match_dir = run_dir / "match"
    if not match_dir.is_dir():
        match_dir = run_dir
    manifest = read_json(match_dir / "manifest.json")
    configuration = manifest.get("configuration")
    configuration = configuration if isinstance(configuration, dict) else {}
    result = read_json(match_dir / "result.json")
    completed = result.get("completed") is True
    gate = read_json(run_dir / "gate-result.json")
    log_text = read_text(match_dir / "match.log")

    if completed:
        score = result.get("score")
        score = score if isinstance(score, dict) else {}
        elo = result.get("elo")
        elo = elo if isinstance(elo, dict) else {}
        live = {
            "wins": int(score.get("candidateWins", 0) or 0),
            "losses": int(score.get("baselineWins", 0) or 0),
            "draws": int(score.get("draws", 0) or 0),
            "games": int(score.get("games", 0) or 0),
            "score": finite_float(score.get("candidateScore")),
            "relativeElo": finite_float(elo.get("difference")),
            "uncertainty": finite_float(elo.get("uncertainty")),
        }
    else:
        live = parse_live_result(log_text)

    sprt_configuration = configuration.get("sprt")
    sprt_configuration = (
        sprt_configuration if isinstance(sprt_configuration, dict) else {}
    )
    sprt_enabled = sprt_configuration.get("enabled") is True
    result_sprt = result.get("sprt")
    sprt = (
        result_sprt
        if completed and isinstance(result_sprt, dict)
        else parse_live_sprt(log_text, sprt_enabled)
    )

    engines = manifest.get("engines")
    engines = engines if isinstance(engines, list) else []
    candidate = next(
        (
            engine
            for engine in engines
            if isinstance(engine, dict) and engine.get("side") == "Candidate"
        ),
        {},
    )
    baseline = next(
        (
            engine
            for engine in engines
            if isinstance(engine, dict) and engine.get("side") == "Baseline"
        ),
        {},
    )
    candidate = candidate if isinstance(candidate, dict) else {}
    baseline = baseline if isinstance(baseline, dict) else {}

    started = parse_datetime(manifest.get("createdAt"))
    completed_at = parse_datetime(result.get("completedAt")) if completed else None
    ended = completed_at or datetime.now(timezone.utc)
    elapsed_seconds = max(0.0, (ended - started).total_seconds()) if started else 0.0
    games = int(live.get("games", 0) or 0)
    total_games = int(configuration.get("games", 0) or 0)
    seconds_per_game = elapsed_seconds / games if games else None
    eta_seconds = (
        0.0
        if completed
        else seconds_per_game * max(0, total_games - games)
        if seconds_per_game is not None
        else None
    )
    pid = read_pid(run_dir / "match.pid")
    running = process_alive(pid)
    if completed:
        state = "complete"
    elif running:
        state = "running"
    elif manifest:
        state = "stopped"
    else:
        state = "starting"

    score_fraction = finite_float(live.get("score"))
    relative_elo = finite_float(live.get("relativeElo"))
    relative_elo_estimated = False
    if relative_elo is None and score_fraction is not None and 0.0 < score_fraction < 1.0:
        relative_elo = 400.0 * math.log10(score_fraction / (1.0 - score_fraction))
        relative_elo_estimated = True
    failures = result.get("failures")
    failures = failures if isinstance(failures, dict) else {}
    recent = [line for line in log_text.splitlines() if line.strip()][-6:]
    return {
        "state": state,
        "running": running,
        "pid": pid,
        "games": games,
        "totalGames": total_games,
        "percent": round(100.0 * games / total_games, 2) if total_games else 0.0,
        "wins": int(live.get("wins", 0) or 0),
        "draws": int(live.get("draws", 0) or 0),
        "losses": int(live.get("losses", 0) or 0),
        "scorePercent": (
            round(score_fraction * 100.0, 1) if score_fraction is not None else None
        ),
        "relativeElo": relative_elo,
        "relativeEloEstimated": relative_elo_estimated,
        "uncertainty": finite_float(live.get("uncertainty")),
        "elapsed": duration(elapsed_seconds),
        "eta": duration(eta_seconds),
        "sprt": sprt,
        "gateDecision": gate.get("decision"),
        "gateReason": gate.get("reason"),
        "failures": {
            key: int(failures.get(key, 0) or 0)
            for key in ("timeForfeits", "crashes", "illegalMoves", "disconnects")
        },
        "candidate": {
            "name": candidate.get("matchName") or "Candidate",
            "commit": candidate.get("commit") or candidate.get("selector"),
        },
        "baseline": {
            "name": baseline.get("matchName") or "Baseline",
            "commit": baseline.get("commit") or baseline.get("selector"),
        },
        "configuration": {
            "timeControl": configuration.get("timeControl"),
            "threads": configuration.get("threads"),
            "hashMb": configuration.get("hashMb"),
            "concurrency": configuration.get("concurrency"),
            "seed": configuration.get("seed"),
            "elo0": sprt_configuration.get("elo0"),
            "elo1": sprt_configuration.get("elo1"),
        },
        "recent": recent,
    }


def latest_attempt(rung_dir: Path) -> Path | None:
    attempts = sorted(path for path in rung_dir.glob("attempt-*") if path.is_dir())
    return attempts[-1] if attempts else None


def attempt_start(attempt: Path) -> datetime | None:
    manifest = read_json(attempt / "match" / "manifest.json")
    started = parse_datetime(manifest.get("createdAt"))
    if started:
        return started
    try:
        return datetime.fromtimestamp(attempt.stat().st_ctime, timezone.utc)
    except OSError:
        return None


def timeout_sensitivity(
    attempt: Path, live: dict[str, object]
) -> dict[str, object] | None:
    match_manifest = read_json(attempt / "match" / "manifest.json")
    engines = match_manifest.get("engines")
    engines = engines if isinstance(engines, list) else []
    candidate_name = next(
        (
            engine.get("matchName")
            for engine in engines
            if isinstance(engine, dict)
            and engine.get("side") == "Candidate"
            and isinstance(engine.get("matchName"), str)
        ),
        None,
    )
    if not candidate_name:
        return None

    timeout_wins = timeout_draws = timeout_losses = 0
    games = re.split(r"(?m)(?=^\[Event )", read_text(attempt / "match" / "games.pgn"))
    for game in games:
        if '[Termination "time forfeit"]' not in game:
            continue
        tags = dict(re.findall(r'^\[([^ ]+) "(.*)"\]$', game, re.MULTILINE))
        white = tags.get("White")
        black = tags.get("Black")
        result = tags.get("Result")
        if candidate_name not in {white, black} or result not in {"1-0", "0-1", "1/2-1/2"}:
            continue
        if result == "1/2-1/2":
            timeout_draws += 1
        elif (white == candidate_name and result == "1-0") or (
            black == candidate_name and result == "0-1"
        ):
            timeout_wins += 1
        else:
            timeout_losses += 1

    excluded = timeout_wins + timeout_draws + timeout_losses
    if not excluded:
        return None
    wins = int(live.get("wins", 0)) - timeout_wins
    draws = int(live.get("draws", 0)) - timeout_draws
    losses = int(live.get("losses", 0)) - timeout_losses
    remaining = wins + draws + losses
    return {
        "excluded": excluded,
        "games": remaining,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "scorePercent": (
            round(100.0 * (wins + 0.5 * draws) / remaining, 1)
            if remaining
            else None
        ),
    }


def rung_snapshot(run_dir: Path, rung: int, games_per_rung: int) -> dict[str, object]:
    rung_dir = run_dir / f"rung-{rung}"
    attempt = latest_attempt(rung_dir)
    if not attempt:
        return {
            "elo": rung,
            "state": "queued",
            "games": 0,
            "totalGames": games_per_rung,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "scorePercent": None,
            "relativeElo": None,
            "uncertainty": None,
            "anchoredPoint": None,
            "failures": {},
            "timeoutSensitivity": None,
            "elapsedSeconds": 0.0,
        }

    result_path = attempt / "match" / "result.json"
    result = read_json(result_path)
    completed = result.get("completed") is True
    if completed:
        score_data = result.get("score")
        elo_data = result.get("elo")
        score_data = score_data if isinstance(score_data, dict) else {}
        elo_data = elo_data if isinstance(elo_data, dict) else {}
        live = {
            "wins": int(score_data.get("candidateWins", 0)),
            "losses": int(score_data.get("baselineWins", 0)),
            "draws": int(score_data.get("draws", 0)),
            "games": int(score_data.get("games", 0)),
            "score": finite_float(score_data.get("candidateScore")),
            "relativeElo": finite_float(elo_data.get("difference")),
            "uncertainty": finite_float(elo_data.get("uncertainty")),
        }
    else:
        live = parse_live_result(read_text(attempt / "driver.log"))

    started = attempt_start(attempt)
    if completed and result_path.is_file():
        ended = datetime.fromtimestamp(result_path.stat().st_mtime, timezone.utc)
    else:
        ended = datetime.now(timezone.utc)
    elapsed = max(0.0, (ended - started).total_seconds()) if started else 0.0
    relative_elo = finite_float(live.get("relativeElo"))
    score_fraction = finite_float(live.get("score"))
    failures = result.get("failures") if completed else {}
    failures = failures if isinstance(failures, dict) else {}
    return {
        "elo": rung,
        "state": "complete" if completed else "active",
        "attempt": attempt.name,
        "games": int(live.get("games", 0)),
        "totalGames": games_per_rung,
        "wins": int(live.get("wins", 0)),
        "draws": int(live.get("draws", 0)),
        "losses": int(live.get("losses", 0)),
        "scorePercent": (
            round(score_fraction * 100.0, 1) if score_fraction is not None else None
        ),
        "relativeElo": round(relative_elo, 1) if relative_elo is not None else None,
        "uncertainty": live.get("uncertainty"),
        "anchoredPoint": (
            round(rung + relative_elo, 1) if relative_elo is not None else None
        ),
        "failures": {
            key: int(failures.get(key, 0) or 0)
            for key in ("timeForfeits", "crashes", "illegalMoves", "disconnects")
        },
        "timeoutSensitivity": timeout_sensitivity(attempt, live) if completed else None,
        "elapsedSeconds": elapsed,
    }


def local_pool_estimate(
    rungs: list[dict[str, object]], *, run_complete: bool = False
) -> dict[str, object]:
    completed = [
        item
        for item in rungs
        if item["state"] == "complete" and item["scorePercent"] is not None
    ]
    if not completed:
        return {"status": "collecting", "estimate": None, "display": "collecting data"}
    completed.sort(key=lambda item: int(item["elo"]))
    for lower, upper in zip(completed, completed[1:]):
        lower_score = float(lower["scorePercent"])
        upper_score = float(upper["scorePercent"])
        if lower_score == 50.0:
            estimate = float(lower["elo"])
        elif (lower_score - 50.0) * (upper_score - 50.0) < 0:
            estimate = float(lower["elo"]) + (
                (50.0 - lower_score)
                * (float(upper["elo"]) - float(lower["elo"]))
                / (upper_score - lower_score)
            )
        else:
            continue
        return {
            "status": "bracketed",
            "estimate": round(estimate, 1),
            "display": f"approximately {estimate:.0f}",
        }
    scores = [float(item["scorePercent"]) for item in completed]
    if all(score > 50.0 for score in scores):
        anchor = int(completed[-1]["elo"])
        return {
            "status": "above_range",
            "estimate": None,
            "display": (
                f"above {anchor}; add a higher rung"
                if run_complete
                else f"above {anchor}; still testing"
            ),
        }
    if all(score < 50.0 for score in scores):
        anchor = int(completed[0]["elo"])
        return {
            "status": "below_range",
            "estimate": None,
            "display": (
                f"below {anchor}; add a lower rung"
                if run_complete
                else f"below {anchor}; still testing"
            ),
        }
    return {"status": "noisy", "estimate": None, "display": "more games required"}


def campaign_signature(configuration: dict[str, object]) -> tuple[object, ...]:
    candidate = configuration.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    stockfish = configuration.get("stockfish")
    stockfish = stockfish if isinstance(stockfish, dict) else {}
    return (
        candidate.get("commit") or candidate.get("selector"),
        stockfish.get("sha256"),
        configuration.get("timeControl"),
        configuration.get("threads"),
        configuration.get("hashMb"),
        configuration.get("concurrency"),
        configuration.get("seed"),
        configuration.get("gamesPerRung"),
    )


def build_snapshot(
    run_dir: Path,
    history_run_dirs: list[Path] | None = None,
    match_run_dir: Path | None = None,
) -> dict[str, object]:
    manifest = read_json(run_dir / "ladder-manifest.json")
    configuration = manifest.get("configuration")
    configuration = configuration if isinstance(configuration, dict) else {}

    source_dirs = [*(history_run_dirs or []), run_dir]
    rows_by_elo: dict[int, dict[str, object]] = {}
    expected_signature = campaign_signature(configuration) if configuration else None
    for source_dir in source_dirs:
        source_manifest = read_json(source_dir / "ladder-manifest.json")
        source_configuration = source_manifest.get("configuration")
        source_configuration = (
            source_configuration if isinstance(source_configuration, dict) else {}
        )
        if (
            expected_signature is not None
            and source_configuration
            and campaign_signature(source_configuration) != expected_signature
        ):
            raise RuntimeError(
                f"history run does not match the active campaign contract: {source_dir}"
            )
        rungs_value = source_configuration.get("rungs", [])
        source_rungs = (
            [int(value) for value in rungs_value]
            if isinstance(rungs_value, list)
            else []
        )
        games_per_rung = int(source_configuration.get("gamesPerRung", 0) or 0)
        for rung in source_rungs:
            rows_by_elo[rung] = rung_snapshot(source_dir, rung, games_per_rung)

    rung_rows = [rows_by_elo[elo] for elo in sorted(rows_by_elo)]
    completed_games = sum(int(item["games"]) for item in rung_rows)
    total_games = sum(int(item["totalGames"]) for item in rung_rows)
    active = next((item for item in rung_rows if item["state"] == "active"), None)
    completed = sum(1 for item in rung_rows if item["state"] == "complete")

    elapsed_seconds = sum(float(item["elapsedSeconds"]) for item in rung_rows)
    seconds_per_game = elapsed_seconds / completed_games if completed_games else None
    eta_seconds = (
        seconds_per_game * (total_games - completed_games)
        if seconds_per_game is not None
        else None
    )

    pid = read_pid(run_dir / "calibration.pid")
    running = process_alive(pid)
    summary = read_json(run_dir / "summary.json")
    is_complete = (
        bool(summary)
        and completed == len(rung_rows)
        and bool(rung_rows)
    )
    stderr_lines = [
        line for line in read_text(run_dir / "calibration.stderr.log").splitlines()
        if line.strip()
    ][-6:]
    if is_complete:
        state = "complete"
    elif running:
        state = "running"
    elif manifest:
        state = "stopped"
    else:
        state = "starting"

    recent_log = read_text(run_dir / "calibration.stdout.log")
    if active and isinstance(active.get("elo"), int):
        attempt = latest_attempt(run_dir / f"rung-{active['elo']}")
        if attempt:
            recent_log = read_text(attempt / "driver.log")
    recent_lines = [line for line in recent_log.splitlines() if line.strip()][-8:]

    candidate = configuration.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    return {
        "state": state,
        "running": running,
        "pid": pid,
        "updatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completedGames": completed_games,
        "totalGames": total_games,
        "completedRungs": completed,
        "totalRungs": len(rung_rows),
        "percent": round(100.0 * completed_games / total_games, 2) if total_games else 0.0,
        "eta": duration(eta_seconds),
        "elapsed": duration(elapsed_seconds),
        "currentRung": active.get("elo") if active else None,
        "estimate": local_pool_estimate(rung_rows, run_complete=is_complete),
        "rungs": rung_rows,
        "resources": system_resources(),
        "configuration": {
            "timeControl": configuration.get("timeControl"),
            "threads": configuration.get("threads"),
            "hashMb": configuration.get("hashMb"),
            "concurrency": configuration.get("concurrency"),
            "candidateCommit": candidate.get("commit") or candidate.get("selector"),
        },
        "recent": recent_lines,
        "errors": stderr_lines,
        "match": build_match_snapshot(match_run_dir) if match_run_dir else None,
    }


def keep_system_awake(run_dir: Path) -> None:
    if os.name != "nt":
        return
    continuous = 0x80000000
    system_required = 0x00000001
    for _ in range(120):
        pid = read_pid(run_dir / "calibration.pid")
        if pid and process_alive(pid):
            break
        time.sleep(1)
    else:
        return
    try:
        while process_alive(pid):
            ctypes.windll.kernel32.SetThreadExecutionState(
                continuous | system_required
            )
            time.sleep(30)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(continuous)


def make_handler(
    run_dir: Path,
    history_run_dirs: list[Path] | None = None,
    match_run_dir: Path | None = None,
    match_controller: MatchController | None = None,
) -> type[BaseHTTPRequestHandler]:
    def snapshot() -> dict[str, object]:
        current_match_dir = (
            match_controller.run_dir if match_controller else match_run_dir
        )
        value = build_snapshot(run_dir, history_run_dirs, current_match_dir)
        match = value.get("match")
        if match_controller and isinstance(match, dict):
            match["controls"] = match_controller.capabilities(match)
        return value

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/api/status":
                payload = json.dumps(snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route in {"/", "/index.html"}:
                payload = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            if route not in {"/api/match/pause", "/api/match/resume"}:
                self.send_error(404)
                return
            if not match_controller or not match_controller.enabled:
                self.send_error(403, "match controls are disabled")
                return
            if self.headers.get("X-Forklift-Control") != "1":
                self.send_error(403, "missing control header")
                return
            origin = self.headers.get("Origin")
            host = self.headers.get("Host")
            if origin and host and urlparse(origin).netloc != host:
                self.send_error(403, "cross-origin control request refused")
                return

            try:
                if route.endswith("/pause"):
                    current = build_match_snapshot(match_controller.run_dir)
                    result = match_controller.pause(current)
                else:
                    result = match_controller.resume()
                status = 200
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                result = {"ok": False, "message": str(error)}
                status = 409
            payload = json.dumps(result).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> int:
    args = arguments()
    run_dir = args.run_dir.expanduser().resolve()
    history_run_dirs = [path.expanduser().resolve() for path in args.history_run_dir]
    match_run_dir = (
        args.match_run_dir.expanduser().resolve() if args.match_run_dir else None
    )
    if args.snapshot:
        print(
            json.dumps(
                build_snapshot(run_dir, history_run_dirs, match_run_dir), indent=2
            )
        )
        return 0
    if not HTML_PATH.is_file():
        raise FileNotFoundError(HTML_PATH)
    match_controller = (
        MatchController(match_run_dir, args.enable_match_controls)
        if match_run_dir
        else None
    )
    threading.Thread(target=keep_system_awake, args=(run_dir,), daemon=True).start()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(run_dir, history_run_dirs, match_run_dir, match_controller),
    )
    print(f"Calibration dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
