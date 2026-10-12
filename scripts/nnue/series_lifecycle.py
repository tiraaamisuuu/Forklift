"""Read-only liveness checks and a coordinator heartbeat for match series."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import threading
import time


HEARTBEAT_INTERVAL_SECONDS = 5
HEARTBEAT_MAX_AGE_SECONDS = 60


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_uint32()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def series_liveness(run_dir: Path, series: dict, *, now: datetime | None = None) -> dict:
    recorded = series.get("state")
    if recorded == "failed" and series.get("current") in {"network", "classical"}:
        try:
            control = json.loads((run_dir / series["current"] / "dashboard-control.json").read_text(encoding="utf-8-sig"))
            if isinstance(control, dict) and control.get("status") == "paused":
                return {"state": "paused", "recordedState": recorded,
                        "reason": "explicit user pause", "pausedAt": control.get("pausedAt")}
        except (OSError, ValueError):
            pass
    if recorded != "running":
        return {"state": recorded, "recordedState": recorded, "reason": "terminal metadata"}
    heartbeat_path = run_dir / "heartbeat.json"
    if not heartbeat_path.exists():
        # Historical runs recorded only the active match-runner PID.
        stage_id = series.get("current")
        if stage_id not in {"network", "classical"}:
            return {"state": "unverified", "recordedState": recorded, "reason": "no active stage"}
        try:
            pid = int((run_dir / stage_id / "match.pid").read_text(encoding="ascii"))
        except (OSError, ValueError):
            pid = 0
        alive = process_alive(pid)
        return {
            "state": "running" if alive else "interrupted",
            "recordedState": recorded,
            "reason": "legacy match PID is alive" if alive else "legacy match runner is absent",
            "pid": pid,
            "legacy": True,
        }
    try:
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8-sig"))
        at = datetime.fromisoformat(heartbeat["at"])
        if at.tzinfo is None:
            raise ValueError("heartbeat timestamp has no timezone")
        age = ((now or datetime.now(timezone.utc)) - at).total_seconds()
        pid = heartbeat["pid"]
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError("invalid heartbeat PID")
        host = heartbeat["host"]
    except (OSError, ValueError, TypeError, KeyError):
        return {"state": "unverified", "recordedState": recorded, "reason": "invalid heartbeat"}
    details = {"recordedState": recorded, "pid": pid, "host": host, "heartbeatAgeSeconds": round(age, 1)}
    if age < -5:
        return {**details, "state": "unverified", "reason": "heartbeat is in the future"}
    if age > HEARTBEAT_MAX_AGE_SECONDS:
        return {**details, "state": "interrupted", "reason": "coordinator heartbeat expired"}
    if host != socket.gethostname():
        return {**details, "state": "unverified", "reason": "coordinator belongs to another host"}
    alive = process_alive(pid)
    return {
        **details,
        "state": "running" if alive else "interrupted",
        "reason": "live coordinator heartbeat" if alive else "coordinator process is absent",
    }


@contextmanager
def coordinator_heartbeat(run_dir: Path):
    stop = threading.Event()
    errors: list[Exception] = []

    def write():
        payload = {"pid": os.getpid(), "host": socket.gethostname(),
                   "at": datetime.now(timezone.utc).isoformat()}
        path = run_dir / "heartbeat.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
        for attempt in range(6):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.02 * 2 ** attempt)

    def loop():
        while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            try:
                write()
                errors.clear()
            except OSError as error:
                errors.append(error)

    write()
    thread = threading.Thread(target=loop, name="series-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=HEARTBEAT_INTERVAL_SECONDS + 1)
    if errors:
        # Recovered reader/antivirus contention is not a match failure. Require
        # a final successful write only if periodic writes remain unresolved.
        write()
