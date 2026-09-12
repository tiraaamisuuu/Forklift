#!/usr/bin/env python3
"""Build two Git revisions and run a reproducible paired Cute Chess match."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
OPENINGS_URL = (
    "https://raw.githubusercontent.com/official-stockfish/books/master/"
    "UHO_4060_v4.epd.zip"
)
OPENINGS_SHA256 = "a97424c5b98b42f8c27ff450f0681ad11696148548c975752350e98417ead11d"


@contextmanager
def keep_system_awake() -> object:
    """Keep Windows awake during a match without requiring the display."""
    continuous = 0x80000000
    system_required = 0x00000001
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(
            continuous | system_required
        )
    try:
        yield
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(continuous)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two chess engine revisions with paired opening games."
    )
    parser.add_argument("--baseline", default="v0.4.0", help="Git ref for engine A")
    parser.add_argument("--candidate", default="HEAD", help="Git ref for engine B")
    parser.add_argument("--baseline-exe", type=Path, help="Use an existing UCI executable for engine A")
    parser.add_argument("--candidate-exe", type=Path, help="Use an existing UCI executable for engine B")
    parser.add_argument("--baseline-name", help="Match/display name for engine A")
    parser.add_argument("--candidate-name", help="Match/display name for engine B")
    parser.add_argument("--baseline-version", help="Optional version metadata for engine A")
    parser.add_argument("--candidate-version", help="Optional version metadata for engine B")
    parser.add_argument("--baseline-arg", action="append", default=[],
                        help="Executable argument for engine A; may be repeated")
    parser.add_argument("--candidate-arg", action="append", default=[],
                        help="Executable argument for engine B; may be repeated")
    parser.add_argument("--baseline-option", action="append", default=[], metavar="NAME=VALUE",
                        help="UCI option for engine A; may be repeated")
    parser.add_argument("--candidate-option", action="append", default=[], metavar="NAME=VALUE",
                        help="UCI option for engine B; may be repeated")
    parser.add_argument("--games", type=int, default=200, help="Even number of games")
    parser.add_argument("--tc", default="10+0.1", help="Cute Chess time control")
    parser.add_argument("--threads", type=int, default=1,
                        help="Default threads for both engines")
    parser.add_argument("--hash", type=int, default=256, dest="hash_mb",
                        help="Default hash size for both engines")
    parser.add_argument("--baseline-threads", type=int,
                        help="Override threads for engine A")
    parser.add_argument("--candidate-threads", type=int,
                        help="Override threads for engine B")
    parser.add_argument("--baseline-hash", type=int, dest="baseline_hash_mb",
                        help="Override hash size for engine A")
    parser.add_argument("--candidate-hash", type=int, dest="candidate_hash_mb",
                        help="Override hash size for engine B")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--build-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--openings", type=Path)
    parser.add_argument("--cutechess", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Write artifacts to this new/empty directory")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic Cute Chess RNG seed")
    parser.add_argument("--sfml-prefix", type=Path)
    parser.add_argument("--candidate-eval-file", type=Path)
    parser.add_argument("--baseline-eval-file", type=Path)
    parser.add_argument("--sprt", action="store_true")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=5.0)
    parser.add_argument("--quick", action="store_true", help="Four-game installation check")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command)
    print(f"+ {printable}", flush=True)
    kwargs.setdefault("env", dict(os.environ))
    return subprocess.run(command, check=True, text=True, **kwargs)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, env=dict(os.environ)
    ).strip()


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def extract_ref(ref: str, destination: Path) -> str:
    commit = git_output("rev-parse", "--verify", f"{ref}^{{commit}}")
    marker = destination / ".source-commit"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == commit:
        return commit
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = subprocess.check_output(
        ["git", "archive", commit], cwd=ROOT, env=dict(os.environ)
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        destination_root = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise RuntimeError("Git archive contains an unsafe path")
        tar.extractall(destination)
    marker.write_text(commit + "\n", encoding="utf-8")
    return commit


def default_sfml_prefix() -> Path | None:
    configured = os.environ.get("SFML_PREFIX")
    if configured:
        return Path(configured).expanduser()
    mac_default = Path.home() / ".local" / "sfml-2.6.2"
    return mac_default if mac_default.exists() else None


def find_engine(build: Path) -> Path:
    names = {
        "chess-engine-uci",
        "chess-engine-uci.exe",
        "Forklift",
        "Forklift.exe",
        "gui",
        "gui.exe",
    }
    candidates = [path for path in build.rglob("*") if path.is_file() and path.name in names]
    candidates.sort(key=lambda path: (path.name.startswith("gui"), len(path.parts)))
    if not candidates:
        raise RuntimeError(f"No UCI-capable engine binary found under {build}")
    return candidates[0].resolve()


def build_ref(ref: str, label: str, jobs: int, sfml_prefix: Path | None) -> tuple[Path, str]:
    ref_root = TOOLS / "engine-match" / f"{label}-{safe_name(ref)}"
    source = ref_root / "source"
    build = ref_root / "build"
    commit = extract_ref(ref, source)

    configure = [
        "cmake", "-S", str(source), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", "-DCHESS_BUILD_GUI=OFF", "-DBUILD_TESTING=OFF",
    ]
    # Before v1 the UCI loop lived in the SFML executable, so those refs still
    # need SFML even though the match itself is headless.
    if not (source / "src" / "uci_main.cpp").exists() and sfml_prefix:
        configure.append(f"-DCMAKE_PREFIX_PATH={sfml_prefix.resolve()}")
    run(configure)
    run(["cmake", "--build", str(build), "--config", "Release", "--parallel", str(jobs)])
    engine = find_engine(build)
    print(f"{label}: {ref} ({commit[:10]}) -> {engine}")
    return engine, commit


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cli_options(values: list[str], side: str) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"{side} option must use NAME=VALUE: {value}")
        name, option_value = value.split("=", 1)
        name = name.strip()
        if not name:
            raise RuntimeError(f"{side} option name cannot be empty")
        options.append((name, option_value))
    return options


def parse_uci_options(output: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for line in output.splitlines():
        match = re.match(r"^option name (.+?) type ([^ ]+)(?: (.*))?$", line.strip())
        if not match:
            continue
        option: dict[str, object] = {
            "name": match.group(1),
            "type": match.group(2),
        }
        attributes = match.group(3) or ""
        for segment in re.split(r"\s+(?=(?:default|min|max|var)\s)", attributes):
            if not segment:
                continue
            key, _, value = segment.partition(" ")
            if key == "var":
                option.setdefault("vars", [])
                assert isinstance(option["vars"], list)
                option["vars"].append(value)
            elif key in {"min", "max"}:
                try:
                    option[key] = int(value)
                except ValueError:
                    option[key] = value
            elif key == "default":
                option[key] = value
        parsed.append(option)
    return parsed


def inspect_uci_engine(binary: Path, binary_args: list[str]) -> dict[str, object]:
    command = [str(binary), *binary_args]
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
        )
        output, _ = process.communicate("uci\nquit\n", timeout=15)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise RuntimeError(f"UCI handshake timed out: {binary}") from error
    if process.returncode:
        raise RuntimeError(
            f"UCI handshake failed with code {process.returncode}: {binary}\n{output}"
        )
    if not any(line.strip() == "uciok" for line in output.splitlines()):
        raise RuntimeError(f"UCI handshake did not return uciok: {binary}\n{output}")

    name = ""
    author = ""
    for line in output.splitlines():
        if line.startswith("id name "):
            name = line.removeprefix("id name ").strip()
        elif line.startswith("id author "):
            author = line.removeprefix("id author ").strip()
    return {
        "name": name,
        "author": author,
        "options": parse_uci_options(output),
    }


def configured_options(
    inspection: dict[str, object],
    side: str,
    threads: int,
    hash_mb: int,
    custom_values: list[str],
    eval_file: Path | None,
) -> list[tuple[str, str]]:
    advertised_options = inspection["options"]
    assert isinstance(advertised_options, list)
    advertised = {
        str(option["name"]).casefold(): option
        for option in advertised_options
        if isinstance(option, dict) and "name" in option
    }
    configured: dict[str, tuple[str, str]] = {}

    def set_option(name: str, value: str, required: bool) -> None:
        supported = advertised.get(name.casefold())
        if not supported:
            if required:
                raise RuntimeError(f"{side} engine does not advertise UCI option '{name}'")
            print(f"Warning: {side} engine does not advertise UCI option '{name}'; skipping it")
            return
        advertised_name = str(supported["name"])
        option_type = str(supported.get("type", ""))
        if option_type == "spin":
            try:
                numeric_value = int(value)
            except ValueError as error:
                raise RuntimeError(
                    f"{side} UCI option '{advertised_name}' requires an integer, got '{value}'"
                ) from error
            minimum = supported.get("min")
            maximum = supported.get("max")
            if isinstance(minimum, int) and numeric_value < minimum:
                raise RuntimeError(
                    f"{side} UCI option '{advertised_name}' is below its minimum "
                    f"{minimum}: {numeric_value}"
                )
            if isinstance(maximum, int) and numeric_value > maximum:
                raise RuntimeError(
                    f"{side} UCI option '{advertised_name}' is above its maximum "
                    f"{maximum}: {numeric_value}"
                )
        elif option_type == "check" and value.casefold() not in {"true", "false"}:
            raise RuntimeError(
                f"{side} UCI option '{advertised_name}' requires true or false, got '{value}'"
            )
        elif option_type == "combo":
            allowed = supported.get("vars")
            if isinstance(allowed, list) and value.casefold() not in {
                str(item).casefold() for item in allowed
            }:
                raise RuntimeError(
                    f"{side} UCI option '{advertised_name}' must be one of "
                    f"{', '.join(str(item) for item in allowed)}, got '{value}'"
                )
        configured[advertised_name.casefold()] = (advertised_name, value)

    set_option("Hash", str(hash_mb), required=False)
    set_option("Threads", str(threads), required=False)
    for name, value in parse_cli_options(custom_values, side):
        set_option(name, value, required=True)

    if eval_file:
        network = eval_file.expanduser().resolve()
        if not network.is_file():
            raise FileNotFoundError(f"NNUE network not found: {network}")
        set_option("EvalFile", str(network), required=True)
        set_option("Use NNUE", "true", required=True)

    return list(configured.values())


def per_engine_resources(
    threads: int,
    hash_mb: int,
    baseline_threads: int | None,
    candidate_threads: int | None,
    baseline_hash_mb: int | None,
    candidate_hash_mb: int | None,
) -> dict[str, dict[str, int]]:
    resources = {
        "baseline": {
            "threads": baseline_threads if baseline_threads is not None else threads,
            "hashMb": baseline_hash_mb if baseline_hash_mb is not None else hash_mb,
        },
        "candidate": {
            "threads": candidate_threads if candidate_threads is not None else threads,
            "hashMb": candidate_hash_mb if candidate_hash_mb is not None else hash_mb,
        },
    }
    values = [
        resources[side][field]
        for side in ("baseline", "candidate")
        for field in ("threads", "hashMb")
    ]
    if min(values) < 1:
        raise RuntimeError("per-engine threads and hash sizes must be positive")
    return resources


def resolve_engine(
    side: str,
    ref: str,
    external: Path | None,
    binary_args: list[str],
    match_name: str | None,
    version: str | None,
    custom_options: list[str],
    eval_file: Path | None,
    jobs: int,
    sfml_prefix: Path | None,
    threads: int,
    hash_mb: int,
) -> dict[str, object]:
    if external:
        binary = external.expanduser().resolve()
        if not binary.is_file():
            raise FileNotFoundError(f"{side} engine executable not found: {binary}")
        commit = None
        source = "external"
        effective_args = binary_args
        selector = str(binary)
    else:
        binary, commit = build_ref(ref, side.lower(), jobs, sfml_prefix)
        source = "git"
        effective_args = binary_args or ["--uci"]
        selector = ref

    inspection = inspect_uci_engine(binary, effective_args)
    options = configured_options(
        inspection, side, threads, hash_mb, custom_options, eval_file
    )
    return {
        "side": side,
        "selector": selector,
        "source": source,
        "ref": ref if source == "git" else None,
        "commit": commit,
        "binary": binary,
        "binaryArgs": effective_args,
        "matchName": match_name or side,
        "version": version,
        "identity": inspection,
        "configuredOptions": options,
        "sha256": file_sha256(binary),
    }


def ensure_openings(requested: Path | None, quick: bool) -> Path:
    if requested:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Opening suite not found: {path}")
        return path
    if quick:
        return ROOT / "tests" / "openings.epd"

    destination = TOOLS / "openings"
    book = destination / "UHO_4060_v4.epd"
    if book.exists():
        return book
    print("Downloading the pinned Stockfish UHO opening suite...", flush=True)
    payload = urlopen(OPENINGS_URL, timeout=120).read()
    if hashlib.sha256(payload).hexdigest() != OPENINGS_SHA256:
        raise RuntimeError("Opening archive checksum mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extract("UHO_4060_v4.epd", destination)
    return book


def find_cutechess(requested: Path | None) -> Path:
    windows_install = TOOLS / "cutechess-windows"
    windows_candidates = list(windows_install.rglob("cutechess-cli.exe")) if windows_install.exists() else []
    candidates = [
        requested,
        Path(os.environ["CUTECHESS_BIN"]) if os.environ.get("CUTECHESS_BIN") else None,
        Path(shutil.which("cutechess-cli")) if shutil.which("cutechess-cli") else None,
        TOOLS / "build" / "cutechess" / ("cutechess-cli.exe" if os.name == "nt" else "cutechess-cli"),
        *windows_candidates,
    ]
    for candidate in candidates:
        if candidate and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "cutechess-cli is unavailable. Run scripts/install_cutechess.sh on macOS/Linux "
        "or scripts/install_cutechess.ps1 on Windows."
    )


def engine_definition(
    name: str,
    binary: Path,
    binary_args: list[str],
    options: list[tuple[str, str]],
) -> list[str]:
    definition = [
        f"name={name}", f"cmd={binary}", "proto=uci",
    ]
    definition.extend(f"arg={argument}" for argument in binary_args)
    definition.extend(f"option.{option_name}={value}" for option_name, value in options)
    return definition


def write_web_profiles(baseline_ref: str, baseline: Path, baseline_commit: str,
                       candidate_ref: str, candidate: Path, candidate_commit: str) -> Path:
    """Expose locally built revisions to the web GUI without committing machine paths."""
    destination = TOOLS / "engine-match" / "profiles.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    legacy_baseline = baseline_ref.lower().startswith(("v0.", "v0-"))
    payload = {
        "version": 1,
        "profiles": [
            {
                "id": f"candidate-{safe_name(candidate_ref)}",
                "name": f"Candidate · {candidate_ref}",
                "detail": f"Committed comparison build · revision {candidate_commit[:10]}",
                "role": "candidate",
                "badge": "CANDIDATE · COMMITTED",
                "path": str(candidate.resolve()),
                "args": ["--uci"],
            },
            {
                "id": f"baseline-{safe_name(baseline_ref)}",
                "name": f"{'Legacy' if legacy_baseline else 'Baseline'} · {baseline_ref}",
                "detail": (
                    f"Older release build · revision {baseline_commit[:10]}" if legacy_baseline
                    else f"Reference comparison build · revision {baseline_commit[:10]}"
                ),
                "role": "legacy" if legacy_baseline else "baseline",
                "badge": "LEGACY · BASELINE" if legacy_baseline else "REFERENCE · BASELINE",
                "path": str(baseline.resolve()),
                "args": ["--uci"],
            },
        ],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Web GUI profiles -> {destination}")
    return destination


def engine_manifest(selection: dict[str, object]) -> dict[str, object]:
    identity = selection["identity"]
    assert isinstance(identity, dict)
    configured = selection["configuredOptions"]
    assert isinstance(configured, list)
    return {
        "side": selection["side"],
        "matchName": selection["matchName"],
        "source": selection["source"],
        "selector": selection["selector"],
        "ref": selection["ref"],
        "commit": selection["commit"],
        "version": selection["version"],
        "path": str(selection["binary"]),
        "sha256": selection["sha256"],
        "args": selection["binaryArgs"],
        "uci": identity,
        "configuredOptions": [
            {"name": name, "value": value} for name, value in configured
        ],
    }


def parse_metric(value: str) -> float | None:
    if value.casefold() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
        return None
    return float(value)


def parse_sprt_result(log_text: str, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {"enabled": False, "decision": "disabled"}

    pattern = re.compile(
        r"^SPRT:\s+llr\s+([+\-]?(?:[0-9.eE]+|inf|nan))\s+\([^)]*\),"
        r"\s+lbound\s+([+\-]?(?:[0-9.eE]+|inf|nan)),"
        r"\s+ubound\s+([+\-]?(?:[0-9.eE]+|inf|nan))"
        r"(?:\s+-\s+H[01]\s+was\s+accepted)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(log_text))
    if not matches:
        return {"enabled": True, "decision": "unavailable"}

    match = matches[-1]
    llr = parse_metric(match.group(1))
    lower = parse_metric(match.group(2))
    upper = parse_metric(match.group(3))
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


def parse_match_result(
    log_text: str,
    candidate_name: str,
    baseline_name: str,
    expected_games: int,
    exit_code: int,
    sprt_enabled: bool = False,
) -> dict[str, object]:
    score_pattern = re.compile(
        rf"^Score of {re.escape(candidate_name)} vs {re.escape(baseline_name)}:"
        r"\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)\s+\[([0-9.]+)\]\s+(\d+)\s*$",
        re.MULTILINE,
    )
    score_matches = list(score_pattern.finditer(log_text))
    score: dict[str, object] | None = None
    if score_matches:
        match = score_matches[-1]
        score = {
            "candidateWins": int(match.group(1)),
            "baselineWins": int(match.group(2)),
            "draws": int(match.group(3)),
            "candidateScore": float(match.group(4)),
            "games": int(match.group(5)),
        }

    elo_pattern = re.compile(
        r"^Elo difference:\s+([+\-]?(?:[0-9.]+|inf|nan))"
        r"\s+\+/-\s+([+\-]?(?:[0-9.]+|inf|nan)),"
        r"\s+LOS:\s+([0-9.]+)\s+%,\s+DrawRatio:\s+([0-9.]+)\s+%\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    elo_matches = list(elo_pattern.finditer(log_text))
    elo: dict[str, object] | None = None
    if elo_matches:
        match = elo_matches[-1]
        elo = {
            "difference": parse_metric(match.group(1)),
            "uncertainty": parse_metric(match.group(2)),
            "losPercent": float(match.group(3)),
            "drawRatioPercent": float(match.group(4)),
            "display": f"{match.group(1)} +/- {match.group(2)}",
        }

    terminations: dict[str, int] = {}
    for match in re.finditer(
        r"^Finished game \d+ .*?:\s+(?:1-0|0-1|1/2-1/2|\*)\s+\{(.+)\}\s*$",
        log_text,
        re.MULTILINE,
    ):
        reason = match.group(1).strip()
        terminations[reason] = terminations.get(reason, 0) + 1

    failure_counts = {
        "timeForfeits": 0,
        "crashes": 0,
        "illegalMoves": 0,
        "disconnects": 0,
    }
    for reason, count in terminations.items():
        lowered = reason.casefold()
        if "time" in lowered:
            failure_counts["timeForfeits"] += count
        if "crash" in lowered or "exited" in lowered:
            failure_counts["crashes"] += count
        if "illegal" in lowered:
            failure_counts["illegalMoves"] += count
        if "disconnect" in lowered or "connection" in lowered:
            failure_counts["disconnects"] += count

    finished_games = sum(terminations.values())
    sprt = parse_sprt_result(log_text, sprt_enabled)
    sprt_decision = sprt.get("decision")
    reached_valid_end = finished_games == expected_games or (
        sprt_enabled and sprt_decision in {"accepted_h0", "accepted_h1"}
    )
    return {
        "schemaVersion": 1,
        "completed": (
            exit_code == 0
            and 0 < finished_games <= expected_games
            and reached_valid_end
        ),
        "processExitCode": exit_code,
        "expectedGames": expected_games,
        "finishedGames": finished_games,
        "score": score,
        "elo": elo,
        "sprt": sprt,
        "terminations": terminations,
        "failures": failure_counts,
    }


def main() -> int:
    args = arguments()
    if args.games < 2 or args.games % 2:
        raise SystemExit("--games must be a positive even number so colours remain paired")
    if min(args.threads, args.hash_mb, args.concurrency, args.build_jobs) < 1:
        raise SystemExit("threads, hash, concurrency, and build-jobs must be positive")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.quick:
        args.games = 4
        # Keep the smoke test short without pushing process/protocol overhead
        # into bullet-time forfeits on slower CI or development machines.
        args.tc = "2+0.02"
        args.concurrency = 1

    resources = per_engine_resources(
        args.threads,
        args.hash_mb,
        args.baseline_threads,
        args.candidate_threads,
        args.baseline_hash_mb,
        args.candidate_hash_mb,
    )
    sfml = args.sfml_prefix.expanduser() if args.sfml_prefix else default_sfml_prefix()
    baseline = resolve_engine(
        "Baseline",
        args.baseline,
        args.baseline_exe,
        args.baseline_arg,
        args.baseline_name,
        args.baseline_version,
        args.baseline_option,
        args.baseline_eval_file,
        args.build_jobs,
        sfml,
        resources["baseline"]["threads"],
        resources["baseline"]["hashMb"],
    )
    candidate = resolve_engine(
        "Candidate",
        args.candidate,
        args.candidate_exe,
        args.candidate_arg,
        args.candidate_name,
        args.candidate_version,
        args.candidate_option,
        args.candidate_eval_file,
        args.build_jobs,
        sfml,
        resources["candidate"]["threads"],
        resources["candidate"]["hashMb"],
    )
    if str(candidate["matchName"]).casefold() == str(baseline["matchName"]).casefold():
        raise RuntimeError("Candidate and baseline match names must be different")

    if baseline["source"] == "git" and candidate["source"] == "git":
        baseline_binary = baseline["binary"]
        candidate_binary = candidate["binary"]
        baseline_commit = baseline["commit"]
        candidate_commit = candidate["commit"]
        assert isinstance(baseline_binary, Path)
        assert isinstance(candidate_binary, Path)
        assert isinstance(baseline_commit, str)
        assert isinstance(candidate_commit, str)
        write_web_profiles(
            args.baseline, baseline_binary, baseline_commit,
            args.candidate, candidate_binary, candidate_commit,
        )

    for selection in (candidate, baseline):
        identity = selection["identity"]
        assert isinstance(identity, dict)
        options = identity["options"]
        assert isinstance(options, list)
        uci_elo = next(
            (
                option for option in options
                if isinstance(option, dict)
                and str(option.get("name", "")).casefold() == "uci_elo"
            ),
            None,
        )
        identity_name = identity.get("name") or "Unknown UCI engine"
        print(
            f"{selection['side']} UCI: {identity_name}"
            f" ({len(options)} options, sha256 {str(selection['sha256'])[:12]})"
        )
        if uci_elo:
            print(
                f"  UCI_Elo advertised range: {uci_elo.get('min')}..{uci_elo.get('max')}"
            )

    if args.build_only:
        return 0

    cutechess = find_cutechess(args.cutechess)
    openings = ensure_openings(args.openings, args.quick)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate_artifact_name = (
        args.candidate if candidate["source"] == "git"
        else Path(str(candidate["binary"])).stem
    )
    baseline_artifact_name = (
        args.baseline if baseline["source"] == "git"
        else Path(str(baseline["binary"])).stem
    )
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else ROOT / "artifacts" / "elo"
        / f"{safe_name(candidate_artifact_name)}-vs-{safe_name(baseline_artifact_name)}-{stamp}"
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "match.pid").write_text(str(os.getpid()), encoding="ascii")
    pgn = output / "games.pgn"
    log = output / "match.log"
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"

    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runnerCommit": git_output("rev-parse", "HEAD"),
        "runnerDirty": bool(git_output("status", "--short", "--untracked-files=no")),
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logicalCpuCount": os.cpu_count(),
        },
        "configuration": {
            "games": args.games,
            "timeControl": args.tc,
            "threads": args.threads,
            "hashMb": args.hash_mb,
            "engineResources": resources,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "sprt": {
                "enabled": args.sprt,
                "elo0": args.elo0,
                "elo1": args.elo1,
            },
        },
        "engines": [
            engine_manifest(candidate),
            engine_manifest(baseline),
        ],
        "openings": {
            "path": str(openings.resolve()),
            "sha256": file_sha256(openings),
            "paired": True,
            "order": "random",
        },
        "cutechess": {
            "path": str(cutechess.resolve()),
            "sha256": file_sha256(cutechess),
        },
        "artifacts": {
            "pgn": str(pgn.resolve()),
            "log": str(log.resolve()),
            "result": str(result_path.resolve()),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    command = [str(cutechess)]
    candidate_binary = candidate["binary"]
    baseline_binary = baseline["binary"]
    candidate_binary_args = candidate["binaryArgs"]
    baseline_binary_args = baseline["binaryArgs"]
    candidate_options = candidate["configuredOptions"]
    baseline_options = baseline["configuredOptions"]
    assert isinstance(candidate_binary, Path)
    assert isinstance(baseline_binary, Path)
    assert isinstance(candidate_binary_args, list)
    assert isinstance(baseline_binary_args, list)
    assert isinstance(candidate_options, list)
    assert isinstance(baseline_options, list)
    command += [
        "-engine",
        *engine_definition(
            str(candidate["matchName"]),
            candidate_binary,
            candidate_binary_args,
            candidate_options,
        ),
    ]
    command += [
        "-engine",
        *engine_definition(
            str(baseline["matchName"]),
            baseline_binary,
            baseline_binary_args,
            baseline_options,
        ),
    ]
    command += [
        "-each", f"tc={args.tc}",
        "-openings", f"file={openings}", "format=epd", "order=random",
        "-games", str(args.games), "-repeat", "-recover",
        "-srand", str(args.seed),
        "-concurrency", str(args.concurrency),
        "-resign", "movecount=6", "score=700",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-pgnout", str(pgn),
    ]
    if args.sprt:
        command += [
            "-sprt", f"elo0={args.elo0}", f"elo1={args.elo1}",
            "alpha=0.05", "beta=0.05",
        ]

    print("\nPaired engine match")
    print(f"  Candidate: {candidate['matchName']} [{candidate['selector']}]")
    print(f"  Baseline : {baseline['matchName']} [{baseline['selector']}]")
    print(f"  Games    : {args.games} at {args.tc}")
    print(
        "  Resources: "
        f"candidate {resources['candidate']['threads']}t/"
        f"{resources['candidate']['hashMb']}MB; "
        f"baseline {resources['baseline']['threads']}t/"
        f"{resources['baseline']['hashMb']}MB"
    )
    print(f"  Openings : {openings}")
    print(f"  Manifest : {manifest_path}")
    print(f"  Output   : {output}\n")

    with log.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        exit_code = process.wait()

    result = parse_match_result(
        log.read_text(encoding="utf-8"),
        str(candidate["matchName"]),
        str(baseline["matchName"]),
        args.games,
        exit_code,
        args.sprt,
    )
    result["completedAt"] = datetime.now().astimezone().isoformat(timespec="seconds")
    result["manifest"] = str(manifest_path.resolve())
    result["pgn"] = str(pgn.resolve())
    result["log"] = str(log.resolve())
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nMachine-readable result -> {result_path}")
    return exit_code


if __name__ == "__main__":
    try:
        with keep_system_awake():
            raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
