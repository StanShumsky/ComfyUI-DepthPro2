import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dp2_node_benchmark", ROOT / "benchmarks" / "benchmark_nodes.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_peak_rss_uses_windows_working_set(monkeypatch):
    monkeypatch.setattr(benchmark, "resource", None)
    monkeypatch.setattr(
        benchmark.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100, peak_wset=250)),
    )

    assert benchmark.peak_rss_bytes() == 250


def test_peak_rss_falls_back_to_current_rss(monkeypatch):
    monkeypatch.setattr(benchmark, "resource", None)
    monkeypatch.setattr(
        benchmark.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100)),
    )

    assert benchmark.peak_rss_bytes() == 100


def test_memory_snapshot_reports_process_memory():
    snapshot = benchmark.memory_snapshot(torch.device("cpu"))

    assert snapshot["rss_bytes"] > 0
    assert snapshot["peak_rss_bytes"] >= snapshot["rss_bytes"]


def test_repeated_aggregates_each_memory_peak_independently(monkeypatch):
    records = iter([
        (None, {"seconds": 0.1, "memory": {
            "rss_bytes": 300, "peak_rss_bytes": 400,
            "accelerator_allocated_bytes": 900, "accelerator_driver_bytes": None,
        }}),
        (None, {"seconds": 0.2, "memory": {
            "rss_bytes": 500, "peak_rss_bytes": 600,
            "accelerator_allocated_bytes": 200, "accelerator_driver_bytes": 1200,
        }}),
    ])
    monkeypatch.setattr(benchmark, "timed_call", lambda *args: next(records))
    args = SimpleNamespace(warmup=0, repeat=2)

    result = benchmark.repeated(lambda: None, args, torch.device("cuda"))

    assert result["memory_peak"] == {
        "rss_bytes": 500,
        "peak_rss_bytes": 600,
        "accelerator_allocated_bytes": 900,
        "accelerator_driver_bytes": 1200,
    }
