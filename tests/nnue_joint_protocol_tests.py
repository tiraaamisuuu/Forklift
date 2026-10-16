#!/usr/bin/env python3
"""Standard-library-only protocol/watchdog fixtures; never start training."""

from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nnue"))

import run_joint_protocol as wrapper


def tiny_inputs(root: Path) -> tuple[wrapper.RunInputs, dict]:
    """A deliberately reduced, in-memory protocol for testing preflight only.

    The public loader still accepts ONLY the full frozen real protocol. Tests
    replacing that loader do not create a way to override the actual CLI.
    """
    protocol = copy.deepcopy(wrapper.load_protocol(wrapper.PROTOCOL))
    protocol["training"]["fullTrainingPositions"] = 4
    protocol["training"]["batchSize"] = 2
    protocol["selection"]["validationPositions"] = 2
    protocol["diagnostic"]["positions"] = 2
    paths = {name: root / name for name in ("train", "validation", "source_weights", "source_network", "cpp_tools")}
    for name, count in (("train", 4), ("validation", 8)):
        paths[name].write_bytes(wrapper.SHARD_HEADER.pack(b"TNNDS1\0\0", 1, 42, count) + bytes(42 * count))
    for name in ("source_weights", "source_network", "cpp_tools"):
        paths[name].write_bytes(("fixture-" + name).encode())
    hashes = {name: wrapper.sha256(path) for name, path in paths.items()}
    protocol.update(trainSha256=hashes["train"], parentValidationSha256=hashes["validation"],
                    sourceWeightsSha256=hashes["source_weights"], sourceNetworkSha256=hashes["source_network"])
    cache_root = root / "cache"
    cache_root.mkdir()
    cache = {"schemaVersion": 1, "status": "complete",
             "sourceWeights": {"sha256": hashes["source_weights"]},
             "sources": {"train": {"sha256": hashes["train"], "count": 4},
                         "validation": {"sha256": hashes["validation"], "count": 8}},
             "configuration": {"hidden": 256, "featureFactorization": True,
                               "targetScale": 600, "wdlScale": 400, "resultWeight": 0},
             "splits": {}}
    for split in ("validation", "test"):
        files = {}
        for name, shape in (("boards", [2, 32]), ("turns", [2]), ("game_ids", [2]),
                            ("source_indices", [2]), ("identity", [2, 32])):
            path = cache_root / (split + "-" + name + ".fixture")
            path.write_bytes((split + name).encode())
            files[name] = {"path": path.name, "sha256": wrapper.sha256(path), "shape": shape}
        cache["splits"][split] = {"count": 2, "files": files}
    cache_path = cache_root / "manifest.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    protocol["metadataCacheSha256"] = wrapper.sha256(cache_path)
    inputs = wrapper.RunInputs(cache=cache_path, output=root / "execution", **paths)
    return inputs, protocol


def completed_report(output: Path, protocol: dict) -> None:
    directory = output / "study"
    directory.mkdir()
    arms = [{"seed": seed, "residualHidden": width, "updates": 4, "trainingPositions": 8,
             "diagnostic": {"samples": 2}}
            for seed in protocol["training"]["seeds"] for width in protocol["architecture"]["residualWidths"]]
    (directory / "study.json").write_text(json.dumps({"state": "complete", "arms": arms}), encoding="utf-8")


class JointProtocolTests(unittest.TestCase):
    def test_canonical_protocol_ignores_whitespace_but_rejects_any_semantic_change(self) -> None:
        protocol = wrapper.load_protocol(wrapper.PROTOCOL)
        with tempfile.TemporaryDirectory(prefix="forklift-protocol-pin-") as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol, indent=7, sort_keys=True) + "\n\n", encoding="utf-8")
            self.assertEqual(wrapper.load_protocol(path), protocol)
            protocol["training"]["headLearningRate"] *= 2
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the frozen"):
                wrapper.load_protocol(path)

    def test_preflight_pins_all_inputs_counts_and_membership_without_torch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-protocol-audit-") as directory:
            inputs, protocol = tiny_inputs(Path(directory))
            result = wrapper.preflight(inputs, protocol)
            self.assertEqual(result["trainingPositions"], 4)
            self.assertEqual(result["selectionPositions"], 2)
            self.assertEqual(set(result["artifacts"]), {"train", "validation", "cache", "source_weights",
                                                        "source_network", "cpp_tools"})
            for name in ("train", "validation", "cache", "source_weights", "source_network"):
                altered = copy.deepcopy(protocol)
                key = {"train": "trainSha256", "validation": "parentValidationSha256", "cache": "metadataCacheSha256",
                       "source_weights": "sourceWeightsSha256", "source_network": "sourceNetworkSha256"}[name]
                altered[key] = "0" * 64
                with self.assertRaisesRegex(ValueError, name + " hash"):
                    wrapper.preflight(inputs, altered)
            wrong_count = copy.deepcopy(protocol)
            wrong_count["training"]["fullTrainingPositions"] += 1
            with self.assertRaisesRegex(ValueError, "frozen 5M"):
                wrapper.preflight(inputs, wrong_count)
            cache = json.loads(inputs.cache.read_text())
            cache["splits"]["validation"]["count"] = 3
            inputs.cache.write_text(json.dumps(cache), encoding="utf-8")
            protocol["metadataCacheSha256"] = wrapper.sha256(inputs.cache)
            with self.assertRaisesRegex(ValueError, "membership.*32k"):
                wrapper.preflight(inputs, protocol)

    def test_child_command_has_exact_protocol_flags_and_no_hyperparameter_overrides(self) -> None:
        root = Path("unused")
        inputs = wrapper.RunInputs(*(root / name for name in
                                      ("train", "validation", "cache", "weights", "network", "tools", "output")))
        protocol = wrapper.load_protocol(wrapper.PROTOCOL)
        command = wrapper.child_command(inputs, protocol)
        self.assertEqual(command[:2], [sys.executable, "-u"])
        for flag, value in (("--epochs", "2"), ("--batch-size", "2048"), ("--transformer-lr", "1e-05"),
                            ("--head-lr", "0.0001"), ("--device", "cuda"), ("--max-seconds", "2700"),
                            ("--diagnostic-count", "32000"), ("--bootstrap-resamples", "1000"),
                            ("--hidden-scale", "1024"), ("--output-scale", "32"),
                            ("--residual-weight-scale", "256")):
            self.assertEqual(command[command.index(flag) + 1], value)
        index = command.index("--seeds")
        self.assertEqual(command[index + 1:index + 3], ["202610161", "202610162"])
        index = command.index("--residual-widths")
        self.assertEqual(command[index + 1:index + 3], ["0", "8"])
        self.assertIn("--feature-factorization", command)
        self.assertNotIn("--resume", command)

    def test_execution_is_written_before_child_and_complete_state_is_checked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-protocol-success-") as directory:
            inputs, protocol = tiny_inputs(Path(directory))
            child = mock.Mock(pid=12345)
            child.poll.return_value = 0
            child.wait.return_value = 0

            def launched(command, **kwargs):
                record = json.loads((inputs.output / "execution.json").read_text())
                self.assertEqual(record["command"], command)
                self.assertFalse(record["childStarted"])
                self.assertEqual(record["state"], "launching")
                self.assertIn("sourceDependencySha256", record["environment"])
                self.assertEqual(record["budget"]["externalChildTimeoutSeconds"], 2730)
                completed_report(inputs.output, protocol)
                return child

            with mock.patch.object(wrapper, "load_protocol", return_value=protocol), \
                    mock.patch.object(wrapper, "capture_environment", return_value={"sourceDependencySha256": {}}), \
                    mock.patch.object(wrapper, "code_hashes", return_value={}), \
                    mock.patch.object(wrapper.subprocess, "Popen", side_effect=launched) as start:
                result = wrapper.run_protocol(inputs)
            start.assert_called_once()
            child.wait.assert_called_once_with(timeout=2730)
            self.assertEqual(result["state"], "complete")
            self.assertFalse(result["promotionAuthorized"])
            final = json.loads((inputs.output / "execution.json").read_text())
            self.assertEqual(final["completedStudy"]["updatesPerArm"], 4)
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                wrapper.run_protocol(inputs)

    def test_timeout_stops_only_the_created_child_and_preserves_incomplete_record(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-protocol-timeout-") as directory:
            inputs, protocol = tiny_inputs(Path(directory))
            child = mock.Mock(pid=34567)
            child.wait.side_effect = subprocess.TimeoutExpired("fixture-child", 2730)
            child.poll.return_value = -9
            with mock.patch.object(wrapper, "load_protocol", return_value=protocol), \
                    mock.patch.object(wrapper, "capture_environment", return_value={"sourceDependencySha256": {}}), \
                    mock.patch.object(wrapper.subprocess, "Popen", return_value=child), \
                    mock.patch.object(wrapper, "stop_timed_out_child") as stop:
                with self.assertRaisesRegex(TimeoutError, "30-second cleanup"):
                    wrapper.run_protocol(inputs)
            stop.assert_called_once_with(child)
            record = json.loads((inputs.output / "execution.json").read_text())
            self.assertEqual(record["state"], "incomplete")
            self.assertEqual(record["childPid"], 34567)
            self.assertTrue(record["watchdogTriggered"])
            self.assertFalse(record["comparable"])
            self.assertFalse(record["automaticExtension"])
            self.assertFalse((inputs.output / "study" / "study.json").exists())

    def test_preflight_and_child_failures_never_claim_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forklift-protocol-failure-") as directory:
            inputs, protocol = tiny_inputs(Path(directory))
            bad = copy.deepcopy(protocol)
            bad["trainSha256"] = "0" * 64
            with mock.patch.object(wrapper, "load_protocol", return_value=bad), \
                    mock.patch.object(wrapper.subprocess, "Popen") as start:
                with self.assertRaisesRegex(ValueError, "train hash"):
                    wrapper.run_protocol(inputs)
                start.assert_not_called()
            self.assertFalse(json.loads((inputs.output / "execution.json").read_text())["childStarted"])
            inputs = replace(inputs, output=inputs.output.parent / "child-failed")
            child = mock.Mock(pid=76543)
            child.wait.return_value = 1
            child.poll.return_value = 1
            with mock.patch.object(wrapper, "load_protocol", return_value=protocol), \
                    mock.patch.object(wrapper, "capture_environment", return_value={"sourceDependencySha256": {}}), \
                    mock.patch.object(wrapper.subprocess, "Popen", return_value=child), \
                    mock.patch.object(wrapper, "stop_timed_out_child") as stop:
                with self.assertRaisesRegex(RuntimeError, "exited 1"):
                    wrapper.run_protocol(inputs)
                stop.assert_not_called()
            record = json.loads((inputs.output / "execution.json").read_text())
            self.assertEqual(record["state"], "incomplete")
            self.assertEqual(record["childExitCode"], 1)


if __name__ == "__main__":
    unittest.main()
