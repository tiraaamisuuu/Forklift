#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_threads", ROOT / "scripts" / "benchmark_threads.py"
)
assert SPEC and SPEC.loader
benchmark_threads = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_threads
SPEC.loader.exec_module(benchmark_threads)


class BenchmarkThreadsTests(unittest.TestCase):
    def test_parse_thread_counts_deduplicates_in_order(self):
        self.assertEqual(benchmark_threads.parse_thread_counts("1,2,4,2,6"), [1, 2, 4, 6])

    def test_parse_benchmark_output(self):
        parsed = benchmark_threads.parse_benchmark_output(
            "Benchmark: depth=64 timeLimit=250ms threads=2 positions=4\n"
            "Start        best=e2e4 depth=10 score=12 nodes=100 qnodes=25 time=250ms nps=400\n"
            "Benchmark summary: nodes=100 time=250ms nps=400\n"
        )
        self.assertEqual(parsed["nodes"], 100)
        self.assertEqual(parsed["nps"], 400)
        self.assertEqual(parsed["positions"][0]["name"], "Start")
        self.assertEqual(parsed["positions"][0]["depth"], 10)

    def test_summary_uses_one_thread_baseline(self):
        summary = benchmark_threads.summarise([
            {"threads": 1, "nps": 100, "time_ms": 1000, "positions": [{"depth": 8}]},
            {"threads": 1, "nps": 120, "time_ms": 1000, "positions": [{"depth": 9}]},
            {"threads": 2, "nps": 200, "time_ms": 1000, "positions": [{"depth": 10}]},
            {"threads": 2, "nps": 240, "time_ms": 1000, "positions": [{"depth": 11}]},
        ])
        self.assertEqual(summary["baselineThreads"], 1)
        self.assertEqual(summary["results"][0]["medianNps"], 110)
        self.assertEqual(summary["results"][1]["speedup"], 2.0)
        self.assertEqual(summary["results"][1]["efficiency"], 1.0)

    def test_parse_search_feature_counters(self):
        for counters in (" probcut=0/0", " probcut=0/0 singular=6/196 singularPathMax=1",
                         " probcut=0/0 singular=6/196 singularPathMax=1 singularProofNodes=27 singularSkipped=3"):
            with self.subTest(counters=counters):
                parsed = benchmark_threads.parse_benchmark_output(
                    "Start best=e2e4 depth=12 score=34 nodes=396533 qnodes=72722"
                    + counters + " time=832ms nps=476602\n"
                    "Benchmark summary: nodes=396533 time=832ms nps=476602\n"
                )
                self.assertEqual(len(parsed["positions"]), 1)
                self.assertEqual(parsed["positions"][0]["depth"], 12)
                self.assertEqual(parsed["positions"][0]["qnodes"], 72722)


if __name__ == "__main__":
    unittest.main()
