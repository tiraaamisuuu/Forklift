#!/usr/bin/env python3
"""End-to-end smoke test for the local web GUI and a real UCI engine."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

import chess.pgn


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--verified-nnue", type=Path,
                        help="Also exercise the pinned verified model as the explicit startup profile")
    return parser.parse_args()


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def request(base: str, path: str, payload: dict | None = None,
            headers: dict[str, str] | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    call = Request(base + path, data=data, headers=request_headers)
    with urlopen(call, timeout=15) as response:
        return json.load(response)


def upload_engine(base: str, engine: Path, token: str, name: str) -> dict:
    query = urlencode({"filename": engine.name, "name": name})
    call = Request(
        f"{base}/api/engines/import?{query}",
        data=engine.read_bytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Engine-Library-Token": token,
        },
    )
    with urlopen(call, timeout=20) as response:
        assert response.status == 201
        return json.load(response)


def main() -> int:
    args = arguments()
    if not args.engine.is_file():
        raise SystemExit(f"engine not found: {args.engine}")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    engine_library = tempfile.TemporaryDirectory(prefix="chess-user-engines-")
    stockfish_executable = "stockfish-smoke.exe" if os.name == "nt" else "stockfish-smoke"
    stockfish_archive = Path(engine_library.name) / "stockfish-smoke.zip"
    with zipfile.ZipFile(stockfish_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(stockfish_executable, args.engine.read_bytes())
        archive.writestr("Copying.txt", "Stockfish smoke-test licence fixture\n")
        archive.writestr("AUTHORS", "Smoke test\n")
        archive.writestr("README.md", "Smoke test archive\n")
    stockfish_digest = hashlib.sha256(stockfish_archive.read_bytes()).hexdigest()
    environment = dict(os.environ)
    environment["CHESS_USER_ENGINE_ROOT"] = engine_library.name
    environment["CHESS_STOCKFISH_ASSET_URL"] = stockfish_archive.resolve().as_uri()
    environment["CHESS_STOCKFISH_ASSET_SHA256"] = stockfish_digest
    environment["CHESS_STOCKFISH_EXECUTABLE"] = stockfish_executable
    environment["CHESS_STOCKFISH_ARCHIVE_KIND"] = "zip"
    server_command = [sys.executable, str(ROOT / "web" / "server.py"), "--engine", str(args.engine),
                      "--port", str(port), "--no-open"]
    if args.verified_nnue:
        server_command.extend(["--verified-nnue", str(args.verified_nnue.resolve()),
                               "--default-profile", "verified-shared-nnue"])
    server = subprocess.Popen(
        server_command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            if server.poll() is not None:
                output = server.stdout.read() if server.stdout else ""
                raise RuntimeError(f"web server exited early\n{output}")
            try:
                health = request(base, "/api/health")
                break
            except URLError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("web server did not become ready")
                time.sleep(0.05)

        assert health == {"ok": True, "engineConnected": True}
        state = request(base, "/api/state")
        assert len(state["pieces"]) == 32
        assert len(state["legalMoves"]) == 20
        assert state["canMove"] is True
        assert state["mode"] == "pvc"
        assert state["controllers"]["white"]["type"] == "human"
        assert state["controllers"]["black"]["type"] == "engine"
        assert state["profiles"] and state["profiles"][0]["id"] == "current-classical"
        assert state["profiles"][0]["role"] == "development"
        assert state["profiles"][0]["badge"] == "DEV · NEWEST"
        profile_id = "verified-shared-nnue" if args.verified_nnue else "current-classical"
        expected_badge = "NNUE · VERIFIED MODEL" if args.verified_nnue else "DEV · NEWEST"
        assert state["controllers"]["black"]["profileId"] == profile_id
        assert state["controllers"]["black"]["badge"] == expected_badge
        assert state["engine"]["activeProfileId"] == profile_id
        assert state["engine"]["activeProfileRole"] == ("nnue" if args.verified_nnue else "development")
        selected_profile = next(profile for profile in state["profiles"] if profile["id"] == profile_id)
        assert selected_profile["usesNnue"] is bool(args.verified_nnue)
        assert state["engineLibrary"]["stockfish"]["installed"] is False
        assert state["engineLibrary"]["stockfish"]["installerAvailable"] is True

        installed = request(base, "/api/engines/install-stockfish", {}, headers={
            "X-Engine-Library-Token": state["engineLibrary"]["token"],
        })
        stockfish_profile = installed["profile"]
        assert stockfish_profile["role"] == "stockfish"
        assert stockfish_profile["badge"] == "STOCKFISH · OFFICIAL"
        assert stockfish_profile["removable"] is True
        assert installed["state"]["engineLibrary"]["stockfish"]["installed"] is True
        installed_again = request(base, "/api/engines/install-stockfish", {}, headers={
            "X-Engine-Library-Token": installed["state"]["engineLibrary"]["token"],
        })
        assert installed_again["profile"]["id"] == stockfish_profile["id"]
        assert installed_again["state"]["engine"]["profileCount"] \
            == installed["state"]["engine"]["profileCount"]
        state = request(base, "/api/new", {
            "mode": "pvc", "side": "white", "engineProfile": stockfish_profile["id"],
            "engineTimeMs": 100,
        })
        state = request(base, "/api/move", {"uci": "e2e4"})
        state = request(base, "/api/engine-move", {})
        assert state["engine"]["lastMove"]["profileId"] == stockfish_profile["id"]
        state = request(base, "/api/new", {"mode": "pvp"})
        state = request(base, "/api/engines/remove", {"profileId": stockfish_profile["id"]}, headers={
            "X-Engine-Library-Token": state["engineLibrary"]["token"],
        })
        assert state["engineLibrary"]["stockfish"]["installed"] is False

        imported = upload_engine(
            base, args.engine.resolve(), state["engineLibrary"]["token"], "Smoke Test Engine"
        )
        imported_profile = imported["profile"]
        assert imported_profile["role"] == "external"
        assert imported_profile["badge"] == "EXTERNAL · UCI"
        assert imported_profile["removable"] is True
        assert imported["state"]["engine"]["profileCount"] == state["engine"]["profileCount"] + 1
        state = request(base, "/api/new", {
            "mode": "pvc", "side": "white", "engineProfile": imported_profile["id"],
            "engineTimeMs": 100,
        })
        state = request(base, "/api/move", {"uci": "e2e4"})
        state = request(base, "/api/engine-move", {})
        assert state["engine"]["lastMove"]["profileId"] == imported_profile["id"]
        state = request(base, "/api/new", {"mode": "pvp"})
        state = request(base, "/api/engines/remove", {"profileId": imported_profile["id"]}, headers={
            "X-Engine-Library-Token": state["engineLibrary"]["token"],
        })
        assert all(profile["id"] != imported_profile["id"] for profile in state["profiles"])
        state = request(base, "/api/new", {
            "mode": "pvc", "side": "white", "engineProfile": profile_id,
            "engineTimeMs": 100,
        })

        state = request(base, "/api/move", {"uci": "e2e4"})
        assert state["ply"] == 1 and state["needsEngineMove"] is True
        state = request(base, "/api/engine-move", {})
        assert state["ply"] == 2 and state["canMove"] is True
        assert state["engine"]["lastMove"]["depth"] >= 1

        analysis = request(base, "/api/analyse", {})
        assert analysis["depth"] >= 1
        assert isinstance(analysis["pv"], list)
        assert analysis["profileName"]
        assert analysis["profileId"] == profile_id

        exported = request(base, "/api/export", {
            "format": "pgn", "event": "Web Export Test", "white": "Alice", "black": "Engine",
        })
        assert exported["filename"].endswith(".pgn")
        game = chess.pgn.read_game(io.StringIO(exported["content"]))
        assert game is not None
        assert game.headers["Event"] == "Web Export Test"
        assert game.headers["White"] == "Alice"
        assert len(list(game.mainline_moves())) == 2

        fen_export = request(base, "/api/export", {"format": "fen"})
        assert fen_export["content"].strip() == state["fen"]
        json_export = request(base, "/api/export", {"format": "json"})
        game_log = json.loads(json_export["content"])
        assert len(game_log["moves"]) == 2
        assert game_log["currentFen"] == state["fen"]

        state = request(base, "/api/undo", {})
        assert state["ply"] == 0
        assert state["fen"].startswith("rnbqkbnr/pppppppp/")

        state = request(base, "/api/new", {
            "mode": "pvc", "side": "black", "engineProfile": profile_id,
            "engineTimeMs": 100,
        })
        assert state["playerColor"] == "black" and state["needsEngineMove"] is True
        state = request(base, "/api/engine-move", {})
        assert state["ply"] == 1 and state["canMove"] is True
        state = request(base, "/api/undo", {})
        assert state["ply"] == 1, "the engine's opening move is not a human turn to undo"

        state = request(base, "/api/new", {"mode": "pvp"})
        assert state["mode"] == "pvp" and state["playerColor"] is None
        assert state["controllers"]["white"]["type"] == "human"
        assert state["controllers"]["black"]["type"] == "human"
        assert state["canMove"] is True

        state = request(base, "/api/new", {"mode": "pvp", "engineTimeMs": 5000})
        assert state["engine"]["moveTimeMs"] == 5000

        state = request(base, "/api/new", {
            "mode": "cvc", "whiteProfile": profile_id, "blackProfile": profile_id,
            "engineTimeMs": 100,
        })
        assert state["mode"] == "cvc" and state["canMove"] is False
        assert state["controllers"]["white"]["profileId"] == profile_id
        assert state["controllers"]["black"]["profileId"] == profile_id
        state = request(base, "/api/engine-move", {})
        assert state["ply"] == 1 and state["needsEngineMove"] is True
        state = request(base, "/api/engine-move", {})
        assert state["ply"] == 2 and state["needsEngineMove"] is True
        assert state["engine"]["lastMove"]["profileId"] == profile_id

        custom_fen = "8/8/8/8/8/8/4K3/R6k w - - 0 1"
        state = request(base, "/api/new", {"mode": "pvp", "fen": custom_fen})
        state = request(base, "/api/move", {"uci": "e2e3"})
        custom_export = request(base, "/api/export", {"format": "pgn"})
        custom_game = chess.pgn.read_game(io.StringIO(custom_export["content"]))
        assert custom_game is not None
        assert custom_game.headers["SetUp"] == "1"
        assert custom_game.headers["FEN"] == custom_fen
        assert [move.uci() for move in custom_game.mainline_moves()] == ["e2e3"]

        with urlopen(base + "/", timeout=5) as response:
            page = response.read()
            assert b"CHESS ENGINE" in page and b"V1.0 / LOCAL" in page
            assert b"position-status" in page and b'value="cvc"' in page
            assert b"pvc-profile-summary" in page
            assert b"export-modal" in page and b"run-analysis" in page
            assert b"engine-time-slider" in page and b'data-time="5000"' in page
            assert b"thinking-countdown" in page and b"status-countdown-value" in page
            assert b"engine-library-modal" in page and b"engine-import-form" in page
            assert b"setup-engine-action" in page and b"library-install-stockfish" in page
            assert b"official-stockfish/Stockfish/releases/tag/sf_18" in page
        if args.verified_nnue:
            # Explicit opt-out stays available after playing/analyzing with NNUE.
            state = request(base, "/api/new", {
                "mode": "pvc", "side": "white", "engineProfile": "current-classical",
                "engineTimeMs": 100,
            })
            assert state["controllers"]["black"]["profileId"] == "current-classical"
            assert state["engine"]["activeProfileBadge"] == "DEV · NEWEST"
        print("Web GUI smoke: PASS" + (" (verified NNUE default + classical opt-out)" if args.verified_nnue else ""))
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        engine_library.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
