#!/usr/bin/env python3
"""Unit tests for web NNUE profile activation evidence."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "web_server", ROOT / "web" / "server.py"
)
assert SPEC and SPEC.loader
web_server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = web_server
SPEC.loader.exec_module(web_server)


class WebNnueProfileTests(unittest.TestCase):
    def test_accepts_exact_network_load_confirmation(self) -> None:
        network = Path("approved.nnue").resolve()
        output = "\n".join((
            "id name Forklift v1.1.0",
            "uciok",
            f"info string NNUE loaded: {network}",
            "readyok",
        ))

        report = web_server.classify_nnue_activation(output, 0, network)

        self.assertTrue(report["successful"])
        self.assertTrue(report["networkLoaded"])

    def test_rejects_silent_classical_fallback(self) -> None:
        report = web_server.classify_nnue_activation(
            "uciok\nreadyok\n", 0, Path("approved.nnue").resolve()
        )

        self.assertFalse(report["successful"])
        self.assertFalse(report["networkLoaded"])

    def test_rejects_reported_load_and_activation_errors(self) -> None:
        output = "\n".join((
            "uciok",
            "info string NNUE load failed: invalid NNUE magic",
            "info string Use NNUE requires a valid EvalFile",
            "readyok",
        ))

        report = web_server.classify_nnue_activation(
            output, 0, Path("approved.nnue").resolve()
        )

        self.assertFalse(report["successful"])
        self.assertEqual(len(report["errors"]), 2)


class ExplicitVerifiedNnueProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="forklift-web-profile-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.engine = self.directory / "engine-fixture"
        self.engine.write_bytes(b"Not an executable; engine startup is mocked.\n")
        self.network = self.directory / "approved.nnue"
        self.network.write_bytes(b"Tiny approved NNUE identity fixture.\n")
        self.checksum = hashlib.sha256(self.network.read_bytes()).hexdigest()
        for patcher in (
            mock.patch.object(web_server, "ROOT", self.directory),
            mock.patch.object(web_server, "PROFILE_MANIFEST", self.directory / "profiles.json"),
            mock.patch.object(web_server, "load_user_engine_entries", return_value=[]),
            mock.patch.object(web_server, "VERIFIED_SHARED_NNUE_SHA256", self.checksum),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_approved_hash_adds_stable_profile_and_retains_classical_first(self) -> None:
        profiles = web_server.discover_profiles(self.engine, None, self.network)
        self.assertEqual(profiles[0].profile_id, "current-classical")
        verified = next(profile for profile in profiles if profile.profile_id == "verified-shared-nnue")
        self.assertEqual(verified.name, "Verified shared NNUE")
        self.assertEqual(verified.eval_file, self.network.resolve())
        self.assertEqual(verified.command, (str(self.engine), "--uci"))
        self.assertEqual(verified.badge, "NNUE · VERIFIED MODEL")
        self.assertEqual(len(profiles), 2)

    def test_missing_wrong_hash_and_directory_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            web_server.discover_profiles(self.engine, None, self.directory / "missing.nnue")
        with self.assertRaisesRegex(ValueError, "not a file"):
            web_server.discover_profiles(self.engine, None, self.directory)
        self.network.write_bytes(b"Unapproved replacement")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            web_server.discover_profiles(self.engine, None, self.network)

    def test_classical_remains_default_without_explicit_selection(self) -> None:
        with mock.patch.object(web_server, "EngineSession") as session_type:
            service = web_server.GameService(self.engine, verified_nnue_path=self.network)
            self.assertEqual(service.default_profile_id, "current-classical")
            self.assertEqual(service.controllers[web_server.chess.BLACK], "current-classical")
            self.assertEqual(session_type.call_args.args[0].profile_id, "current-classical")
            service.close()

    def test_explicit_default_uses_verified_profile_and_manual_classical_stays_available(self) -> None:
        with mock.patch.object(web_server, "EngineSession") as session_type:
            service = web_server.GameService(
                self.engine, default_profile_id="verified-shared-nnue", verified_nnue_path=self.network,
            )
            self.assertEqual(service.default_profile_id, "verified-shared-nnue")
            self.assertEqual(service.controllers[web_server.chess.BLACK], "verified-shared-nnue")
            self.assertEqual(session_type.call_args.args[0].eval_file, self.network.resolve())
            self.assertIn("current-classical", service.profiles)
            service.close()

    def test_unknown_default_is_rejected_before_starting_engine(self) -> None:
        with mock.patch.object(web_server, "EngineSession") as session_type:
            for unknown in ("not-a-profile", "", "verified-shared-nnue"):
                with self.subTest(unknown=unknown), self.assertRaisesRegex(ValueError, "Unknown default"):
                    web_server.GameService(self.engine, default_profile_id=unknown)
            session_type.assert_not_called()

    def test_manual_nnue_behavior_and_duplicate_suppression_are_preserved(self) -> None:
        profiles = web_server.discover_profiles(self.engine, self.network)
        self.assertEqual([profile.profile_id for profile in profiles], ["current-classical", "nnue-approved"])
        self.assertEqual(profiles[1].badge, "NNUE · MODEL")
        profiles = web_server.discover_profiles(self.engine, self.network, self.network)
        self.assertEqual([profile.profile_id for profile in profiles],
                         ["current-classical", "verified-shared-nnue"])

    def test_revision_manifest_cannot_spoof_reserved_verified_identity(self) -> None:
        import json
        other_engine = self.directory / "other-engine-fixture"
        other_engine.write_bytes(b"Not executable either.\n")
        (self.directory / "profiles.json").write_text(json.dumps({"profiles": [{
            "id": "verified-shared-nnue", "name": "Fake verified model", "path": str(other_engine),
        }]}), encoding="utf-8")
        self.assertEqual([profile.profile_id for profile in web_server.discover_profiles(self.engine, None)],
                         ["current-classical"])
        profiles = web_server.discover_profiles(self.engine, None, self.network)
        self.assertEqual(profiles[1].name, "Verified shared NNUE")
        self.assertEqual(profiles[1].eval_file, self.network.resolve())

    def test_replaced_verified_file_is_rejected_before_activation(self) -> None:
        profile = web_server.discover_profiles(self.engine, None, self.network)[1]
        self.network.write_bytes(b"Changed after initial verification")
        with mock.patch.object(web_server, "probe_nnue_activation") as probe, \
                mock.patch.object(web_server.chess.engine.SimpleEngine, "popen_uci") as start:
            session = web_server.EngineSession(profile, 1, 128)
            self.assertIsNone(session.engine)
            self.assertIn("checksum mismatch", session.error)
            probe.assert_not_called()
            start.assert_not_called()

    def test_cli_flags_are_opt_in_and_unknown_default_has_clear_startup_error(self) -> None:
        with mock.patch.object(sys, "argv", ["server.py"]):
            args = web_server.parse_arguments()
            self.assertIsNone(args.default_profile)
            self.assertIsNone(args.verified_nnue)
        arguments = ["server.py", "--verified-nnue", str(self.network),
                     "--default-profile", "verified-shared-nnue", "--no-open"]
        with mock.patch.object(sys, "argv", arguments):
            args = web_server.parse_arguments()
            self.assertEqual(args.default_profile, "verified-shared-nnue")
            self.assertEqual(args.verified_nnue, self.network)
        with mock.patch.object(sys, "argv", ["server.py", "--default-profile", "unknown"]), \
                mock.patch.object(web_server, "ThreadingHTTPServer") as server:
            with self.assertRaisesRegex(SystemExit, "Unable to start web GUI: Unknown default"):
                web_server.main()
            server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
