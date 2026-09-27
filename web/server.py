#!/usr/bin/env python3
"""Local HTTP bridge between chess engine profiles and the web UI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import threading
import webbrowser
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import zipfile

import chess
import chess.engine
import chess.pgn


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
PIECE_ROOT = ROOT / "assets" / "pieces"
PROFILE_MANIFEST = ROOT / ".tools" / "engine-match" / "profiles.json"
USER_ENGINE_ROOT = Path(os.environ.get(
    "CHESS_USER_ENGINE_ROOT", ROOT / ".tools" / "user-engines"
)).expanduser().resolve()
USER_ENGINE_MANIFEST = USER_ENGINE_ROOT / "engines.json"
MAX_ENGINE_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_STOCKFISH_DOWNLOAD_BYTES = 128 * 1024 * 1024
STOCKFISH_RELEASE = "18"
STOCKFISH_RELEASE_TAG = "sf_18"
STOCKFISH_RELEASE_URL = (
    f"https://github.com/official-stockfish/Stockfish/releases/tag/{STOCKFISH_RELEASE_TAG}"
)


@dataclass(frozen=True)
class StockfishAsset:
    filename: str
    sha256: str
    executable: str
    archive_kind: str
    platform_label: str

    @property
    def url(self) -> str:
        override = os.environ.get("CHESS_STOCKFISH_ASSET_URL")
        if override:
            return override
        return (
            "https://github.com/official-stockfish/Stockfish/releases/download/"
            f"{STOCKFISH_RELEASE_TAG}/{self.filename}"
        )


def stockfish_asset_for_host() -> StockfishAsset | None:
    override_url = os.environ.get("CHESS_STOCKFISH_ASSET_URL")
    if override_url:
        digest = os.environ.get("CHESS_STOCKFISH_ASSET_SHA256", "")
        executable = os.environ.get("CHESS_STOCKFISH_EXECUTABLE", "")
        archive_kind = os.environ.get("CHESS_STOCKFISH_ARCHIVE_KIND", "")
        if len(digest) != 64 or not executable or archive_kind not in {"tar", "zip"}:
            return None
        return StockfishAsset(
            Path(urlparse(override_url).path).name or "stockfish-test-archive",
            digest.lower(),
            Path(executable).name,
            archive_kind,
            "test host",
        )

    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return StockfishAsset(
            "stockfish-macos-m1-apple-silicon.tar",
            "4d77c4aa3ad9bd1ea8111f2ac5a4620fe7ebf998d6893bf828d49ccd579c8cb0",
            "stockfish-macos-m1-apple-silicon",
            "tar",
            "macOS · Apple Silicon",
        )
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        return StockfishAsset(
            "stockfish-macos-x86-64.tar",
            "e7d7a2bca13915419d41ac6cb8cedb123dd2ba1c39a22c574df7a2aa3f526592",
            "stockfish-macos-x86-64",
            "tar",
            "macOS · Intel",
        )
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return StockfishAsset(
            "stockfish-ubuntu-x86-64.tar",
            "5c6f38b02a4da5f3ffe763f27da6c3e743eebefd92b50cb3661623b96696adff",
            "stockfish-ubuntu-x86-64",
            "tar",
            "Linux · x64",
        )
    if system == "windows" and machine in {"arm64", "aarch64"}:
        return StockfishAsset(
            "stockfish-windows-armv8.zip",
            "7bc5880c11a58b2fdc4fcd606bf5cb593230026eb501a5a8865dc79fda5ea5fd",
            "stockfish-windows-armv8.exe",
            "zip",
            "Windows · ARM64",
        )
    if system == "windows" and machine in {"x86_64", "amd64"}:
        return StockfishAsset(
            "stockfish-windows-x86-64.zip",
            "40cc975817e7eee270b03f354810d20956df565420d320f6dd37d454dc81a139",
            "stockfish-windows-x86-64.exe",
            "zip",
            "Windows · x64",
        )
    return None


def color_name(color: chess.Color | None) -> str | None:
    if color is None:
        return None
    return "white" if color == chess.WHITE else "black"


def parse_color(value: str) -> chess.Color:
    if value == "random":
        return secrets.choice((chess.WHITE, chess.BLACK))
    if value == "black":
        return chess.BLACK
    return chess.WHITE


def bounded_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return fallback


def slug(value: str) -> str:
    return "-".join(filter(None, "".join(
        character.lower() if character.isalnum() else " " for character in value
    ).split()))


@dataclass(frozen=True)
class EngineProfile:
    profile_id: str
    name: str
    detail: str
    kind: str
    role: str
    badge: str
    command: tuple[str, ...]
    eval_file: Path | None = None
    removable: bool = False
    author: str = ""

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "name": self.name,
            "detail": self.detail,
            "kind": self.kind,
            "role": self.role,
            "badge": self.badge,
            "usesNnue": self.eval_file is not None,
            "available": Path(self.command[0]).is_file(),
            "removable": self.removable,
            "author": self.author,
        }


def classify_nnue_activation(
    output: str, exit_code: int, network: Path
) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines()]
    errors = [
        line
        for line in lines
        if "NNUE load failed:" in line or "Use NNUE requires" in line
    ]
    loaded_message = f"info string NNUE loaded: {network}"
    return {
        "successful": (
            exit_code == 0
            and "uciok" in lines
            and "readyok" in lines
            and loaded_message in lines
            and not errors
        ),
        "exitCode": exit_code,
        "uciOk": "uciok" in lines,
        "readyOk": "readyok" in lines,
        "networkLoaded": loaded_message in lines,
        "errors": errors,
    }


def probe_nnue_activation(command: tuple[str, ...], network: Path) -> dict[str, Any]:
    commands = "\n".join((
        "uci",
        f"setoption name EvalFile value {network}",
        "setoption name NNUE Weight value 100",
        "setoption name Use NNUE value true",
        "isready",
        "quit",
        "",
    ))
    try:
        completed = subprocess.run(
            list(command),
            input=commands,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"NNUE activation probe failed: {error}") from error
    report = classify_nnue_activation(completed.stdout, completed.returncode, network)
    if not report["successful"]:
        details = "; ".join(report["errors"]) or "load confirmation was missing"
        raise RuntimeError(f"NNUE activation probe failed: {details}")
    return report


def load_user_engine_entries() -> list[dict[str, Any]]:
    if not USER_ENGINE_MANIFEST.is_file():
        return []
    try:
        manifest = json.loads(USER_ENGINE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = manifest.get("engines", []) if isinstance(manifest, dict) else []
    return [entry for entry in entries if isinstance(entry, dict)]


def write_user_engine_entries(entries: list[dict[str, Any]]) -> None:
    USER_ENGINE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = USER_ENGINE_MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": 1, "engines": entries}, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(USER_ENGINE_MANIFEST)


def external_engine_profile(entry: dict[str, Any]) -> EngineProfile | None:
    binary = Path(str(entry.get("path", ""))).expanduser().resolve()
    profile_id = slug(str(entry.get("id", "")))
    if not profile_id.startswith("external-") or not binary.is_file():
        return None
    raw_arguments = entry.get("args", [])
    arguments = tuple(str(value) for value in raw_arguments) \
        if isinstance(raw_arguments, list) else ()
    kind = str(entry.get("kind") or "external")
    role = str(entry.get("role") or ("stockfish" if kind == "stockfish" else "external"))
    badge = str(entry.get("badge") or (
        "STOCKFISH · OFFICIAL" if role == "stockfish" else "EXTERNAL · UCI"
    ))
    return EngineProfile(
        profile_id,
        str(entry.get("name") or binary.stem),
        str(entry.get("detail") or "Imported UCI engine"),
        kind,
        role,
        badge,
        (str(binary), *arguments),
        removable=True,
        author=str(entry.get("author") or ""),
    )


def find_built_engine(build: Path) -> Path | None:
    preferred = (
        "chess-engine-uci",
        "chess-engine-uci.exe",
        "Forklift",
        "Forklift.exe",
        "gui",
        "gui.exe",
    )
    candidates = [path for name in preferred for path in build.rglob(name) if path.is_file()]
    return candidates[0].resolve() if candidates else None


def revision_identity(profile_id: str) -> tuple[str, str]:
    if profile_id.startswith("baseline-v0"):
        return "legacy", "LEGACY · BASELINE"
    if profile_id.startswith("baseline-"):
        return "baseline", "REFERENCE · BASELINE"
    if profile_id.startswith("candidate-"):
        return "candidate", "CANDIDATE · COMMITTED"
    return "revision", "BUILT REVISION"


def discover_profiles(engine_path: Path, nnue_path: Path | None) -> list[EngineProfile]:
    profiles = [EngineProfile(
        "current-classical",
        "Development · Classical",
        "Live working-tree build · newest local code",
        "current",
        "development",
        "DEV · NEWEST",
        (str(engine_path), "--uci"),
    )]
    known_commands = {str(engine_path.resolve())}
    used_ids = {profiles[0].profile_id}
    manifest_profiles_loaded = False

    if PROFILE_MANIFEST.is_file():
        try:
            manifest = json.loads(PROFILE_MANIFEST.read_text(encoding="utf-8"))
            entries = manifest.get("profiles", []) if isinstance(manifest, dict) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                binary = Path(str(entry.get("path", ""))).expanduser().resolve()
                if not binary.is_file() or str(binary) in known_commands:
                    continue
                profile_id = slug(str(entry.get("id") or entry.get("name") or binary.stem))
                if not profile_id or profile_id in used_ids:
                    continue
                raw_arguments = entry.get("args", ["--uci"])
                arguments = tuple(str(value) for value in raw_arguments) \
                    if isinstance(raw_arguments, list) else ("--uci",)
                default_role, default_badge = revision_identity(profile_id)
                profiles.append(EngineProfile(
                    profile_id,
                    str(entry.get("name") or profile_id),
                    str(entry.get("detail") or "Built comparison revision"),
                    "revision",
                    str(entry.get("role") or default_role),
                    str(entry.get("badge") or default_badge),
                    (str(binary), *arguments),
                ))
                known_commands.add(str(binary))
                used_ids.add(profile_id)
                manifest_profiles_loaded = True
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    comparison_root = ROOT / ".tools" / "engine-match"
    if comparison_root.is_dir():
        for build in sorted(comparison_root.glob("*/build")):
            binary = find_built_engine(build)
            if binary is None or str(binary) in known_commands:
                continue
            label = build.parent.name
            if manifest_profiles_loaded and not label.startswith("baseline-v0"):
                continue
            profile_id = slug(label)
            if not profile_id or profile_id in used_ids:
                continue
            role, badge = revision_identity(profile_id)
            profiles.append(EngineProfile(
                profile_id,
                label.replace("-", " ").title(),
                "Built comparison revision",
                "revision",
                role,
                badge,
                (str(binary), "--uci"),
            ))
            known_commands.add(str(binary))
            used_ids.add(profile_id)

    for entry in load_user_engine_entries():
        profile = external_engine_profile(entry)
        if profile is None or profile.profile_id in used_ids:
            continue
        command_path = str(Path(profile.command[0]).resolve())
        if command_path in known_commands:
            continue
        profiles.append(profile)
        known_commands.add(command_path)
        used_ids.add(profile.profile_id)

    networks: list[Path] = []
    if nnue_path:
        networks.append(nnue_path.expanduser().resolve())
    network_root = ROOT / "networks"
    if network_root.is_dir():
        networks.extend(sorted(network_root.rglob("*.nnue")))
    seen_networks: set[str] = set()
    for network in networks:
        resolved = str(network.resolve())
        if resolved in seen_networks or not network.is_file():
            continue
        seen_networks.add(resolved)
        profile_id = f"nnue-{slug(network.stem)}"
        suffix = 2
        while profile_id in used_ids:
            profile_id = f"nnue-{slug(network.stem)}-{suffix}"
            suffix += 1
        profiles.append(EngineProfile(
            profile_id,
            f"NNUE · {network.stem}",
            "Development engine with trained evaluation network",
            "nnue",
            "nnue",
            "NNUE · MODEL",
            (str(engine_path), "--uci"),
            network.resolve(),
        ))
        used_ids.add(profile_id)
    return profiles


class EngineSession:
    def __init__(self, profile: EngineProfile, threads: int, hash_mb: int):
        self.profile = profile
        self.engine: chess.engine.SimpleEngine | None = None
        self.engine_name = profile.name
        self.error: str | None = None
        try:
            if profile.eval_file:
                probe_nnue_activation(profile.command, profile.eval_file)
            self.engine = chess.engine.SimpleEngine.popen_uci(list(profile.command), timeout=10.0)
            self.engine_name = self.engine.id.get("name", profile.name)
            configuration: dict[str, Any] = {}
            if "Threads" in self.engine.options:
                configuration["Threads"] = max(1, threads)
            if "Hash" in self.engine.options:
                configuration["Hash"] = max(1, hash_mb)
            if profile.eval_file:
                if "EvalFile" not in self.engine.options:
                    raise RuntimeError(f"{profile.name} does not support an EvalFile option")
                configuration["EvalFile"] = str(profile.eval_file)
                if "Use NNUE" in self.engine.options:
                    configuration["Use NNUE"] = True
            if configuration:
                self.engine.configure(configuration)
        except (OSError, chess.engine.EngineError, TimeoutError, RuntimeError) as error:
            self.error = str(error)
            if self.engine:
                try:
                    self.engine.quit()
                except (OSError, chess.engine.EngineError, TimeoutError):
                    pass
            self.engine = None

    def close(self) -> None:
        if self.engine:
            try:
                self.engine.quit()
            except (OSError, chess.engine.EngineError, TimeoutError):
                pass
            self.engine = None


class GameService:
    def __init__(self, engine_path: Path, threads: int = 1, hash_mb: int = 256,
                 nnue_path: Path | None = None):
        self.lock = threading.RLock()
        discovered = discover_profiles(engine_path, nnue_path)
        self.profiles = {profile.profile_id: profile for profile in discovered}
        self.default_profile_id = discovered[0].profile_id
        self.threads = threads
        self.hash_mb = hash_mb
        self.sessions: dict[str, EngineSession] = {}
        self.board = chess.Board()
        self.start_fen = self.board.fen()
        self.game_started_at = datetime.now().astimezone()
        self.mode = "pvc"
        self.player_color: chess.Color | None = chess.WHITE
        self.controllers: dict[chess.Color, str | None] = {
            chess.WHITE: None,
            chess.BLACK: self.default_profile_id,
        }
        self.engine_time_ms = 650
        self.analysis_time_ms = 500
        self.last_engine_info: dict[str, Any] = {}
        self.last_analysis: dict[str, Any] = {}
        self.game_tokens: dict[str, object] = {}
        self.library_token = secrets.token_urlsafe(24)
        self._session(self.default_profile_id)

    @property
    def engine(self) -> chess.engine.SimpleEngine | None:
        session = self.sessions.get(self.default_profile_id)
        return session.engine if session else None

    def _session(self, profile_id: str) -> EngineSession:
        if profile_id not in self.profiles:
            raise ValueError(f"Unknown engine profile: {profile_id}")
        if profile_id not in self.sessions:
            self.sessions[profile_id] = EngineSession(
                self.profiles[profile_id], self.threads, self.hash_mb
            )
        return self.sessions[profile_id]

    def _profile_id(self, value: Any) -> str:
        profile_id = str(value or self.default_profile_id)
        if profile_id not in self.profiles:
            raise ValueError(f"Unknown engine profile: {profile_id}")
        return profile_id

    def close(self) -> None:
        with self.lock:
            for session in self.sessions.values():
                session.close()
            self.sessions.clear()

    @staticmethod
    def _unique_external_id(name: str, existing: set[str]) -> str:
        base = f"external-{slug(name) or 'engine'}"
        profile_id = base
        suffix = 2
        while profile_id in existing:
            profile_id = f"{base}-{suffix}"
            suffix += 1
        return profile_id

    @staticmethod
    def _probe_engine(command: tuple[str, ...]) -> tuple[str, str]:
        engine: chess.engine.SimpleEngine | None = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(list(command), timeout=10.0)
            name = str(engine.id.get("name") or Path(command[0]).stem).strip()
            author = str(engine.id.get("author") or "").strip()
            return name[:120], author[:120]
        except (OSError, chess.engine.EngineError, TimeoutError) as error:
            raise ValueError(f"This file did not complete a UCI handshake: {error}") from error
        finally:
            if engine:
                try:
                    engine.quit()
                except (OSError, chess.engine.EngineError, TimeoutError):
                    pass

    @staticmethod
    def _download_stockfish(asset: StockfishAsset, destination: Path) -> None:
        digest = hashlib.sha256()
        total = 0
        request = Request(asset.url, headers={"User-Agent": "Forklift-Local-GUI/1.1"})
        try:
            with urlopen(request, timeout=45) as response, destination.open("wb") as output:
                announced = response.headers.get("Content-Length")
                if announced and int(announced) > MAX_STOCKFISH_DOWNLOAD_BYTES:
                    raise ValueError("The official Stockfish archive is unexpectedly large")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_STOCKFISH_DOWNLOAD_BYTES:
                        raise ValueError("The official Stockfish archive exceeded 128 MB")
                    digest.update(chunk)
                    output.write(chunk)
        except (OSError, URLError, TimeoutError) as error:
            raise ValueError(f"Could not download Stockfish: {error}") from error
        if digest.hexdigest() != asset.sha256:
            raise ValueError("Stockfish download checksum did not match the official release")

    @staticmethod
    def _copy_archive_member(archive_path: Path, archive_kind: str,
                             member_name: str, destination: Path,
                             required: bool = True) -> bool:
        if archive_kind == "tar":
            with tarfile.open(archive_path, "r:*") as archive:
                matches = [
                    member for member in archive.getmembers()
                    if member.isfile() and Path(member.name).name == member_name
                ]
                if len(matches) != 1:
                    if not required and not matches:
                        return False
                    raise ValueError(f"Stockfish archive has an invalid {member_name} entry")
                source = archive.extractfile(matches[0])
                if source is None:
                    raise ValueError(f"Could not read {member_name} from Stockfish archive")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                return True
        if archive_kind == "zip":
            with zipfile.ZipFile(archive_path) as archive:
                matches = [
                    member for member in archive.infolist()
                    if not member.is_dir() and Path(member.filename).name == member_name
                ]
                if len(matches) != 1:
                    if not required and not matches:
                        return False
                    raise ValueError(f"Stockfish archive has an invalid {member_name} entry")
                with archive.open(matches[0]) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                return True
        raise ValueError("Unsupported Stockfish archive format")

    def install_stockfish(self) -> dict[str, Any]:
        with self.lock:
            existing = next(
                (profile for profile in self.profiles.values()
                 if profile.role == "stockfish"
                 or "stockfish" in f"{profile.name} {profile.detail}".lower()),
                None,
            )
            if existing:
                return {"profile": existing.serialize(), "state": self._serialize()}

            asset = stockfish_asset_for_host()
            if asset is None:
                raise ValueError(
                    "One-click Stockfish installation is unavailable on this platform; "
                    "use the manual UCI engine importer instead"
                )

            USER_ENGINE_ROOT.mkdir(parents=True, exist_ok=True)
            existing_ids = set(self.profiles)
            existing_ids.update(
                path.name for path in USER_ENGINE_ROOT.iterdir() if path.is_dir()
            )
            profile_id = self._unique_external_id(
                f"stockfish-{STOCKFISH_RELEASE}", existing_ids
            )
            staging_dir = Path(tempfile.mkdtemp(
                prefix=".stockfish-install-", dir=USER_ENGINE_ROOT
            ))
            destination_dir = USER_ENGINE_ROOT / profile_id
            archive_path = staging_dir / asset.filename
            binary_path = staging_dir / asset.executable
            installed = False
            try:
                self._download_stockfish(asset, archive_path)
                try:
                    self._copy_archive_member(
                        archive_path, asset.archive_kind, asset.executable, binary_path
                    )
                    for filename in ("Copying.txt", "AUTHORS", "README.md"):
                        self._copy_archive_member(
                            archive_path, asset.archive_kind, filename,
                            staging_dir / filename, required=False
                        )
                except (tarfile.TarError, zipfile.BadZipFile, OSError) as error:
                    raise ValueError(
                        "The downloaded Stockfish archive could not be read safely"
                    ) from error
                archive_path.unlink()
                if os.name != "nt":
                    binary_path.chmod(
                        binary_path.stat().st_mode
                        | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                    )
                detected_name, author = self._probe_engine((str(binary_path),))
                staging_dir.rename(destination_dir)
                installed = True
                installed_binary = destination_dir / asset.executable
                entry = {
                    "id": profile_id,
                    "name": detected_name or f"Stockfish {STOCKFISH_RELEASE}",
                    "detail": (
                        f"Official Stockfish {STOCKFISH_RELEASE} · {asset.platform_label}"
                    ),
                    "author": author,
                    "path": str(installed_binary.resolve()),
                    "args": [],
                    "kind": "stockfish",
                    "role": "stockfish",
                    "badge": "STOCKFISH · OFFICIAL",
                    "installedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "sourceUrl": asset.url,
                    "releaseUrl": STOCKFISH_RELEASE_URL,
                    "sha256": asset.sha256,
                    "license": "GPL-3.0",
                }
                entries = load_user_engine_entries()
                entries.append(entry)
                write_user_engine_entries(entries)
                profile = external_engine_profile(entry)
                if profile is None:
                    raise ValueError("Installed Stockfish metadata could not be loaded")
                self.profiles[profile.profile_id] = profile
                return {
                    "profile": profile.serialize(),
                    "state": self._serialize(),
                }
            except Exception:
                shutil.rmtree(
                    destination_dir if installed else staging_dir, ignore_errors=True
                )
                raise

    def import_engine(self, filename: str, requested_name: str, content: bytes) -> dict[str, Any]:
        with self.lock:
            clean_filename = Path(filename.replace("\\", "/")).name.strip().replace("\x00", "")
            if not clean_filename:
                clean_filename = "uci-engine.exe" if os.name == "nt" else "uci-engine"
            clean_filename = clean_filename[:180]
            if not content:
                raise ValueError("Choose a non-empty engine executable")
            if len(content) > MAX_ENGINE_UPLOAD_BYTES:
                raise ValueError("Engine executable exceeds the 256 MB import limit")

            provisional_id = self._unique_external_id(
                requested_name or Path(clean_filename).stem, set(self.profiles)
            )
            destination_dir = USER_ENGINE_ROOT / provisional_id
            destination = destination_dir / clean_filename
            destination_dir.mkdir(parents=True, exist_ok=False)
            try:
                destination.write_bytes(content)
                if os.name != "nt":
                    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                detected_name, author = self._probe_engine((str(destination),))
                final_name = requested_name.strip()[:120] or detected_name
                final_id = self._unique_external_id(final_name, set(self.profiles))
                if final_id != provisional_id:
                    final_dir = USER_ENGINE_ROOT / final_id
                    destination_dir.rename(final_dir)
                    destination_dir = final_dir
                    destination = final_dir / clean_filename

                detail = f"Imported UCI engine · {detected_name}"
                if author:
                    detail += f" · {author}"
                entry = {
                    "id": final_id,
                    "name": final_name,
                    "detail": detail,
                    "author": author,
                    "path": str(destination.resolve()),
                    "args": [],
                    "importedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "sourceFilename": clean_filename,
                }
                entries = load_user_engine_entries()
                entries.append(entry)
                write_user_engine_entries(entries)
                profile = external_engine_profile(entry)
                if profile is None:
                    raise ValueError("Imported engine metadata could not be loaded")
                self.profiles[profile.profile_id] = profile
                return {
                    "profile": profile.serialize(),
                    "state": self._serialize(),
                }
            except Exception:
                shutil.rmtree(destination_dir, ignore_errors=True)
                raise

    def remove_engine(self, profile_id: str) -> dict[str, Any]:
        with self.lock:
            profile = self.profiles.get(profile_id)
            if profile is None or not profile.removable:
                raise ValueError("Only imported engines can be removed")
            if profile_id in self.controllers.values():
                raise ValueError("Start a new game without this engine before removing it")

            session = self.sessions.pop(profile_id, None)
            if session:
                session.close()
            entries = [
                entry for entry in load_user_engine_entries()
                if str(entry.get("id")) != profile_id
            ]
            write_user_engine_entries(entries)
            binary = Path(profile.command[0]).resolve()
            try:
                binary.relative_to(USER_ENGINE_ROOT.resolve())
            except ValueError:
                pass
            else:
                shutil.rmtree(binary.parent, ignore_errors=True)
            del self.profiles[profile_id]
            return self._serialize()

    def new_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            requested_mode = str(payload.get("mode", "pvc"))
            legacy_modes = {"engine": "pvc", "local": "pvp"}
            self.mode = legacy_modes.get(requested_mode, requested_mode)
            if self.mode not in {"pvp", "pvc", "cvc"}:
                self.mode = "pvc"

            if self.mode == "pvp":
                self.player_color = None
                self.controllers = {chess.WHITE: None, chess.BLACK: None}
            elif self.mode == "pvc":
                self.player_color = parse_color(str(payload.get("side", "white")))
                profile_id = self._profile_id(payload.get("engineProfile"))
                self.controllers = {
                    self.player_color: None,
                    not self.player_color: profile_id,
                }
            else:
                self.player_color = None
                self.controllers = {
                    chess.WHITE: self._profile_id(payload.get("whiteProfile")),
                    chess.BLACK: self._profile_id(payload.get("blackProfile")),
                }

            self.engine_time_ms = bounded_int(payload.get("engineTimeMs"), 50, 10_000, 650)
            self.analysis_time_ms = bounded_int(payload.get("analysisTimeMs"), 50, 5_000, 500)
            fen = str(payload.get("fen", "")).strip()
            try:
                self.board = chess.Board(fen) if fen else chess.Board()
            except ValueError as error:
                raise ValueError(f"Invalid FEN: {error}") from error
            self.start_fen = self.board.fen()
            self.game_started_at = datetime.now().astimezone()
            self.last_engine_info = {}
            self.last_analysis = {}
            self.game_tokens = {}
            return self._serialize()

    def make_move(self, uci: str) -> dict[str, Any]:
        with self.lock:
            if self.board.is_game_over(claim_draw=True):
                raise ValueError("The game is already over")
            if self.controllers[self.board.turn] is not None:
                raise ValueError("Wait for the engine to move")
            try:
                move = chess.Move.from_uci(uci)
            except ValueError as error:
                raise ValueError("Malformed move") from error
            if move not in self.board.legal_moves:
                raise ValueError("Illegal move")
            self.board.push(move)
            self.last_analysis = {}
            return self._serialize()

    def engine_move(self) -> dict[str, Any]:
        with self.lock:
            profile_id = self.controllers[self.board.turn]
            if profile_id is None or self.board.is_game_over(claim_draw=True):
                return self._serialize()
            session = self._session(profile_id)
            if not session.engine:
                raise RuntimeError(session.error or f"{session.profile.name} is not connected")
            result = session.engine.play(
                self.board,
                chess.engine.Limit(time=self.engine_time_ms / 1000.0),
                game=self.game_tokens.setdefault(profile_id, object()),
                info=chess.engine.INFO_ALL,
            )
            if result.move is None or result.move not in self.board.legal_moves:
                raise RuntimeError(f"{session.profile.name} did not return a legal move")
            self.last_engine_info = self._engine_info(result.info, self.board)
            self.last_engine_info.update({
                "profileId": profile_id,
                "profileName": session.profile.name,
            })
            self.board.push(result.move)
            self.last_analysis = {}
            return self._serialize()

    def analyse(self) -> dict[str, Any]:
        with self.lock:
            if self.board.is_game_over(claim_draw=True):
                self.last_analysis = {}
                return self.last_analysis
            profile_id = self.controllers[self.board.turn]
            if profile_id is None and self.mode == "pvc":
                profile_id = self.controllers[not self.board.turn]
            profile_id = profile_id or self.default_profile_id
            session = self._session(profile_id)
            if not session.engine:
                raise RuntimeError(session.error or f"{session.profile.name} is not connected")
            information = session.engine.analyse(
                self.board,
                chess.engine.Limit(time=self.analysis_time_ms / 1000.0),
                game=self.game_tokens.setdefault(profile_id, object()),
                info=chess.engine.INFO_ALL,
            )
            analysis = self._engine_info(information, self.board)
            principal_variation = information.get("pv", [])[:10]
            replay = self.board.copy(stack=False)
            san_line: list[str] = []
            for move in principal_variation:
                if move not in replay.legal_moves:
                    break
                san_line.append(replay.san(move))
                replay.push(move)
            analysis.update({
                "profileId": profile_id,
                "profileName": session.profile.name,
                "pv": san_line,
                "pvUci": [move.uci() for move in principal_variation],
            })
            self.last_analysis = analysis
            return analysis

    def undo(self) -> dict[str, Any]:
        with self.lock:
            if not self.board.move_stack:
                return self._serialize()
            if (self.mode == "pvc" and self.player_color is not None
                    and self.board.turn == self.player_color and len(self.board.move_stack) == 1):
                return self._serialize()
            self.board.pop()
            if self.mode == "pvc" and self.player_color is not None:
                while self.board.move_stack and self.board.turn != self.player_color:
                    self.board.pop()
            self.last_engine_info = {}
            self.last_analysis = {}
            self.game_tokens = {}
            return self._serialize()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._serialize()

    def export_game(self, payload: dict[str, Any]) -> dict[str, str]:
        with self.lock:
            export_format = str(payload.get("format", "pgn")).strip().lower()
            timestamp = self.game_started_at.strftime("%Y%m%d-%H%M%S")
            if export_format == "pgn":
                content = self._pgn(payload)
                extension = "pgn"
                mime_type = "application/x-chess-pgn"
            elif export_format == "fen":
                content = self.board.fen(en_passant="fen") + "\n"
                extension = "fen"
                mime_type = "text/plain"
            elif export_format == "json":
                content = json.dumps(self._game_log(), indent=2, ensure_ascii=False) + "\n"
                extension = "json"
                mime_type = "application/json"
            else:
                raise ValueError("Export format must be PGN, FEN, or JSON")
            return {
                "format": export_format,
                "filename": f"chess-game-{timestamp}.{extension}",
                "mimeType": mime_type,
                "content": content,
            }

    @staticmethod
    def _header(payload: dict[str, Any], key: str, fallback: str) -> str:
        raw_value = payload.get(key, "")
        value = ("" if raw_value is None else str(raw_value)) \
            .replace("\r", " ").replace("\n", " ").strip()
        return value[:120] or fallback

    def _pgn(self, payload: dict[str, Any]) -> str:
        game = chess.pgn.Game()
        starting_board = chess.Board(self.start_fen)
        if self.start_fen != chess.STARTING_FEN:
            game.setup(starting_board)

        outcome = self.board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "*"
        white = self._controller(chess.WHITE)["name"]
        black = self._controller(chess.BLACK)["name"]
        game.headers.update({
            "Event": self._header(payload, "event", "Local Game"),
            "Site": self._header(payload, "site", "Local"),
            "Date": self.game_started_at.strftime("%Y.%m.%d"),
            "Round": self._header(payload, "round", "-"),
            "White": self._header(payload, "white", white),
            "Black": self._header(payload, "black", black),
            "Result": result,
        })
        if outcome:
            game.headers["Termination"] = outcome.termination.name.replace("_", " ").title()

        node: chess.pgn.GameNode = game
        for move in self.board.move_stack:
            node = node.add_variation(move)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False,
                                            columns=None)
        return game.accept(exporter).strip() + "\n"

    def _game_log(self) -> dict[str, Any]:
        replay = chess.Board(self.start_fen)
        moves: list[dict[str, Any]] = []
        for ply, move in enumerate(self.board.move_stack, start=1):
            moves.append({
                "ply": ply,
                "moveNumber": replay.fullmove_number,
                "color": color_name(replay.turn),
                "uci": move.uci(),
                "san": replay.san(move),
                "fenAfter": None,
            })
            replay.push(move)
            moves[-1]["fenAfter"] = replay.fen(en_passant="fen")
        outcome = self.board.outcome(claim_draw=True)
        return {
            "formatVersion": 1,
            "startedAt": self.game_started_at.isoformat(timespec="seconds"),
            "mode": self.mode,
            "white": self._controller(chess.WHITE),
            "black": self._controller(chess.BLACK),
            "startFen": self.start_fen,
            "currentFen": self.board.fen(en_passant="fen"),
            "result": outcome.result() if outcome else "*",
            "termination": outcome.termination.name if outcome else None,
            "moves": moves,
        }

    def _engine_info(self, information: dict[str, Any], board: chess.Board) -> dict[str, Any]:
        score = information.get("score")
        centipawns: int | None = None
        mate: int | None = None
        if score is not None:
            white_score = score.pov(chess.WHITE)
            mate = white_score.mate()
            centipawns = white_score.score(mate_score=100_000)
        return {
            "depth": information.get("depth"),
            "seldepth": information.get("seldepth"),
            "nodes": information.get("nodes"),
            "nps": information.get("nps"),
            "timeMs": round(float(information.get("time", 0.0)) * 1000),
            "scoreCp": centipawns,
            "mate": mate,
            "turn": color_name(board.turn),
        }

    def _move_history(self) -> list[dict[str, Any]]:
        replay = chess.Board(self.start_fen)
        rows: list[dict[str, Any]] = []
        for ply, move in enumerate(self.board.move_stack):
            san = replay.san(move)
            if ply % 2 == 0:
                rows.append({"number": ply // 2 + 1, "white": san, "black": ""})
            else:
                rows[-1]["black"] = san
            replay.push(move)
        return rows

    def _controller(self, color: chess.Color) -> dict[str, Any]:
        profile_id = self.controllers[color]
        if profile_id is None:
            if self.mode == "pvc":
                name = "You"
            else:
                name = color_name(color).title()
            return {"type": "human", "name": name, "profileId": None}
        profile = self.profiles[profile_id]
        return {
            "type": "engine",
            "name": profile.name,
            "profileId": profile_id,
            "kind": profile.kind,
            "role": profile.role,
            "badge": profile.badge,
            "detail": profile.detail,
        }

    def _serialize(self) -> dict[str, Any]:
        outcome = self.board.outcome(claim_draw=True)
        game_over = outcome is not None
        can_move = not game_over and self.controllers[self.board.turn] is None
        pieces = [{
            "square": chess.square_name(square),
            "color": color_name(piece.color),
            "type": chess.piece_name(piece.piece_type),
            "symbol": piece.symbol(),
        } for square, piece in self.board.piece_map().items()]
        legal_moves = []
        if can_move:
            for move in self.board.legal_moves:
                legal_moves.append({
                    "uci": move.uci(),
                    "from": chess.square_name(move.from_square),
                    "to": chess.square_name(move.to_square),
                    "promotion": chess.piece_name(move.promotion) if move.promotion else None,
                    "capture": self.board.is_capture(move),
                })

        last_move = self.board.peek().uci() if self.board.move_stack else None
        check_square = chess.square_name(self.board.king(self.board.turn)) if self.board.is_check() else None
        needs_engine = not game_over and self.controllers[self.board.turn] is not None
        if game_over:
            status = "Game over"
        elif needs_engine:
            status = f"{self.profiles[self.controllers[self.board.turn]].name} to move"
        elif self.mode == "pvc":
            status = "Your move"
        else:
            status = f"{color_name(self.board.turn).title()} to move"

        active_profile_id = self.controllers[self.board.turn]
        if active_profile_id is None and self.mode == "pvc":
            active_profile_id = self.controllers[not self.board.turn]
        active_profile_id = active_profile_id or self.default_profile_id
        active_profile = self.profiles[active_profile_id]
        active_session = self.sessions.get(active_profile_id)
        default_session = self.sessions.get(self.default_profile_id)
        result = outcome.result() if outcome else None
        reason = outcome.termination.name.replace("_", " ").title() if outcome else None
        stockfish = next(
            (profile for profile in self.profiles.values()
             if profile.role == "stockfish"
             or "stockfish" in f"{profile.name} {profile.detail}".lower()),
            None,
        )
        stockfish_asset = stockfish_asset_for_host()
        return {
            "fen": self.board.fen(en_passant="fen"),
            "turn": color_name(self.board.turn),
            "mode": self.mode,
            "playerColor": color_name(self.player_color),
            "controllers": {
                "white": self._controller(chess.WHITE),
                "black": self._controller(chess.BLACK),
            },
            "profiles": [profile.serialize() for profile in self.profiles.values()],
            "engineLibrary": {
                "token": self.library_token,
                "importedCount": sum(profile.removable for profile in self.profiles.values()),
                "maxUploadMB": MAX_ENGINE_UPLOAD_BYTES // (1024 * 1024),
                "stockfish": {
                    "installed": stockfish is not None,
                    "profileId": stockfish.profile_id if stockfish else None,
                    "installerAvailable": stockfish_asset is not None,
                    "version": STOCKFISH_RELEASE,
                    "platform": stockfish_asset.platform_label if stockfish_asset else None,
                    "releaseUrl": STOCKFISH_RELEASE_URL,
                },
            },
            "canMove": can_move,
            "needsEngineMove": needs_engine,
            "gameOver": game_over,
            "result": result,
            "resultReason": reason,
            "status": status,
            "check": self.board.is_check(),
            "checkSquare": check_square,
            "lastMove": last_move,
            "ply": len(self.board.move_stack),
            "pieces": pieces,
            "legalMoves": legal_moves,
            "moves": self._move_history(),
            "engine": {
                "connected": bool((active_session and active_session.engine)
                                  or (default_session and default_session.engine)),
                "name": active_profile.name,
                "activeProfileId": active_profile_id,
                "activeProfileRole": active_profile.role,
                "activeProfileBadge": active_profile.badge,
                "profileCount": len(self.profiles),
                "moveTimeMs": self.engine_time_ms,
                "error": active_session.error if active_session else None,
                "lastMove": self.last_engine_info,
                "analysis": self.last_analysis,
            },
        }


class WebHandler(BaseHTTPRequestHandler):
    service: GameService
    server_version = "ForkliftWeb/2.0"

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {format_string % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json({"ok": True, "engineConnected": self.service.engine is not None})
            return
        if path == "/api/state":
            self._json(self.service.state())
            return
        self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/engines/import":
                self._verify_library_token()
                length = self._content_length(MAX_ENGINE_UPLOAD_BYTES)
                query = parse_qs(parsed.query)
                filename = query.get("filename", [""])[0]
                name = query.get("name", [""])[0]
                response = self.service.import_engine(filename, name, self.rfile.read(length))
                self._json(response, HTTPStatus.CREATED)
                return
            if path == "/api/engines/install-stockfish":
                self._verify_library_token()
                response = self.service.install_stockfish()
                self._json(response, HTTPStatus.CREATED)
                return
            payload = self._payload()
            if path == "/api/new":
                response = self.service.new_game(payload)
            elif path == "/api/move":
                response = self.service.make_move(str(payload.get("uci", "")))
            elif path == "/api/engine-move":
                response = self.service.engine_move()
            elif path == "/api/analyse":
                response = self.service.analyse()
            elif path == "/api/export":
                response = self.service.export_game(payload)
            elif path == "/api/undo":
                response = self.service.undo()
            elif path == "/api/engines/remove":
                self._verify_library_token()
                response = self.service.remove_engine(str(payload.get("profileId", "")))
            else:
                self._json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
                return
            self._json(response)
        except ValueError as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as error:
            self._json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except (chess.engine.EngineError, TimeoutError, OSError) as error:
            self._json({"error": f"Engine failure: {error}"}, HTTPStatus.SERVICE_UNAVAILABLE)

    def _content_length(self, maximum: int) -> int:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length <= 0:
            raise ValueError("The request body is empty")
        if length > maximum:
            raise ValueError(f"The request exceeds the {maximum // (1024 * 1024)} MB limit")
        return length

    def _verify_library_token(self) -> None:
        supplied = self.headers.get("X-Engine-Library-Token", "")
        if not secrets.compare_digest(supplied, self.service.library_token):
            raise ValueError("Engine library token is missing or invalid")

    def _payload(self) -> dict[str, Any]:
        length = bounded_int(self.headers.get("Content-Length"), 0, 64 * 1024, 0)
        if not length:
            return {}
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON request") from error
        if not isinstance(value, dict):
            raise ValueError("JSON request must be an object")
        return value

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, request_path: str) -> None:
        if request_path in {"/", "/index.html"}:
            path = WEB_ROOT / "index.html"
        elif request_path in {"/styles.css", "/app.js"}:
            path = WEB_ROOT / request_path.removeprefix("/")
        elif request_path.startswith("/assets/pieces/"):
            filename = Path(request_path).name
            path = PIECE_ROOT / filename
            if path.suffix != ".svg" or path.parent != PIECE_ROOT:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache" if path.suffix != ".svg" else "public, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local chess engine web interface")
    parser.add_argument("--engine", type=Path, default=ROOT / "build" / "chess-engine-uci")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=256, dest="hash_mb")
    parser.add_argument("--nnue", type=Path)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    service = GameService(args.engine.expanduser().resolve(), args.threads, args.hash_mb, args.nnue)
    WebHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"Chess engine web GUI: {url}", flush=True)
    default_session = service.sessions.get(service.default_profile_id)
    if default_session and default_session.error:
        print(f"Engine unavailable; PvP remains enabled: {default_session.error}", flush=True)
    if not args.no_open:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    def stop_server(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
