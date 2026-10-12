#!/usr/bin/env python3
"""Interrupted coordinators must not masquerade as running experiments."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "nnue"))
from series_lifecycle import coordinator_heartbeat, series_liveness


class SeriesLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.now = datetime.now(timezone.utc)
        self.series = {"state": "running", "current": "classical"}

    def tearDown(self):
        self.temporary.cleanup()

    def heartbeat(self, age=0, host=None, pid=None):
        (self.root / "heartbeat.json").write_text(json.dumps({
            "pid": os.getpid() if pid is None else pid,
            "host": socket.gethostname() if host is None else host,
            "at": (self.now - timedelta(seconds=age)).isoformat(),
        }), encoding="utf-8")

    def test_live_coordinator_has_fresh_heartbeat(self):
        with coordinator_heartbeat(self.root):
            self.assertEqual(series_liveness(self.root, self.series)["state"], "running")

    def test_failed_runner_with_user_pause_is_reported_as_paused(self):
        (self.root / "classical").mkdir()
        control = {"status": "paused", "pausedAt": "2026-10-12T19:26:00+00:00"}
        (self.root / "classical" / "dashboard-control.json").write_text(json.dumps(control))
        series = {**self.series, "state": "failed"}
        observed = series_liveness(self.root, series)
        self.assertEqual(observed["state"], "paused")
        self.assertEqual(observed["recordedState"], "failed")
        self.assertEqual(series["state"], "failed")

    def test_heartbeat_retries_transient_windows_reader_contention(self):
        original_replace = Path.replace
        calls = 0

        def transient_denial(path, target):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise PermissionError("temporary file sharing violation")
            return original_replace(path, target)

        with patch.object(Path, "replace", new=transient_denial):
            with coordinator_heartbeat(self.root):
                self.assertEqual(series_liveness(self.root, self.series)["state"], "running")
        self.assertEqual(calls, 2)

    def test_expired_heartbeat_rejects_reused_live_pid(self):
        self.heartbeat(age=61)
        observed = series_liveness(self.root, self.series, now=self.now)
        self.assertEqual(observed["state"], "interrupted")
        self.assertEqual(observed["reason"], "coordinator heartbeat expired")

    def test_fresh_heartbeat_with_dead_process_is_interrupted(self):
        self.heartbeat()
        with patch("series_lifecycle.process_alive", return_value=False):
            self.assertEqual(series_liveness(self.root, self.series)["state"], "interrupted")

    def test_another_host_cannot_be_reported_as_locally_running(self):
        self.heartbeat(host="some-other-host")
        self.assertEqual(series_liveness(self.root, self.series)["state"], "unverified")

    def test_corrupt_heartbeat_fails_closed(self):
        (self.root / "heartbeat.json").write_text('{"pid":', encoding="utf-8")
        self.assertEqual(series_liveness(self.root, self.series)["state"], "unverified")

    def test_future_heartbeat_is_unverified(self):
        self.heartbeat(age=-60)
        self.assertEqual(series_liveness(self.root, self.series, now=self.now)["state"], "unverified")

    def test_legacy_dead_runner_is_detected_without_rewriting_evidence(self):
        self.assertEqual(series_liveness(self.root, self.series)["state"], "interrupted")
        self.assertEqual(self.series["state"], "running")
        self.assertFalse((self.root / "series.json").exists())

    def test_terminal_results_do_not_depend_on_old_process_liveness(self):
        self.heartbeat(age=99999)
        self.assertEqual(series_liveness(self.root, {"state": "complete"})["state"], "complete")


if __name__ == "__main__":
    unittest.main()
