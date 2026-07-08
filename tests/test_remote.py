from __future__ import annotations

import time
from typing import Any

import pytest

from slurmwatch import remote
from slurmwatch.model import (
    CpuMetrics,
    GpuMetrics,
    MemoryMetrics,
    TelemetrySnapshot,
)


def _snapshot(host: str = "cn2", node_index: int = 1, node_count: int = 2) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp=time.time(),
        job_id="123",
        step_id="0",
        hostname=host,
        elapsed_seconds=42,
        cpu=CpuMetrics(cores_allocated=4, usage_ns=10**9, usage_percent=99.0, effective_cores=3.9),
        memory=MemoryMetrics(
            current_bytes=3 * 1024**3,
            limit_bytes=8 * 1024**3,
            peak_bytes=4 * 1024**3,
            usage_percent=37.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=3 * 1024**3,
            cache_bytes=0,
        ),
        gpus=[
            GpuMetrics(
                index=0,
                uuid="G",
                name="A100",
                utilization_percent=90.0,
                memory_used_bytes=1,
                memory_total_bytes=2,
                memory_utilization_percent=50.0,
                power_watts=100.0,
                temperature_celsius=40.0,
                throttling=False,
            )
        ],
        node_count=node_count,
        node_index=node_index,
        gpu_count_requested=1,
        gpu_active_count=1,
    )


class TestSnapshotSerialization:
    def test_from_json_round_trip(self) -> None:
        s = _snapshot()
        assert TelemetrySnapshot.from_json(s.to_json()) == s

    def test_from_dict_ignores_unknown_keys(self) -> None:
        # A small version skew between nodes (an extra field) must not crash.
        d = {
            "timestamp": 1.0,
            "job_id": "1",
            "step_id": None,
            "hostname": "cn1",
            "elapsed_seconds": 1,
            "cpu": {"cores_allocated": 1, "usage_ns": 0, "usage_percent": 0.0, "future_field": 9},
            "memory": {
                "current_bytes": 0,
                "limit_bytes": 0,
                "peak_bytes": 0,
                "usage_percent": 0.0,
                "oom_guard_warning": False,
                "oom_guard_critical": False,
            },
            "gpus": [],
            "node_count": 3,
            "node_index": 2,
            "brand_new_top_level_key": True,
        }
        snap = TelemetrySnapshot.from_dict(d)
        assert snap.node_count == 3 and snap.node_index == 2
        assert snap.cpu.cores_allocated == 1  # parsed despite the unknown cpu field


class TestBuildStreamCommand:
    def test_streams_the_node_via_srun_overlap(self) -> None:
        cmd = remote.build_stream_command("456", "cn007", 1.0, python="/venv/bin/python")
        assert cmd[0] == "srun"
        assert "--jobid=456" in cmd and "--overlap" in cmd
        assert cmd[cmd.index("-w") + 1] == "cn007"
        # Runs the same install's headless logger streaming JSONL to stdout.
        assert cmd[-5:-1] == ["456", "--log", "/dev/stdout", "--interval"]
        assert cmd[-1] == "1"  # interval formatted
        assert cmd[cmd.index("-m") + 1] == "slurmwatch"


class TestParseSnapshotLine:
    def test_valid_line(self) -> None:
        snap = remote.parse_snapshot_line(_snapshot(host="cn9", node_index=1).to_json().encode())
        assert snap is not None and snap.hostname == "cn9" and snap.node_index == 1

    def test_garbage_and_empty_are_none(self) -> None:
        assert remote.parse_snapshot_line(b"not json at all\n") is None
        assert remote.parse_snapshot_line(b"   \n") is None
        assert remote.parse_snapshot_line(b"") is None


class TestOpenStream:
    @pytest.mark.asyncio
    async def test_returns_the_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()

        async def fake_exec(*_a: Any, **_k: Any) -> Any:
            return sentinel

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        assert await remote.open_stream("123", "cn9", 1.0) is sentinel

    @pytest.mark.asyncio
    async def test_missing_srun_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*_a: Any, **_k: Any) -> Any:
            raise FileNotFoundError("srun not found")

        monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
        assert await remote.open_stream("123", "cn9", 1.0) is None

    def test_child_env_strips_slurm_and_disables_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_STEP_ID", "7")
        monkeypatch.setenv("SLURM_PROCID", "3")
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        env = remote._child_env()
        assert not any(k.startswith("SLURM_") for k in env)  # step context cleared
        assert env["SLURMWATCH_NO_HOP"] == "1"
        assert "SLURMWATCH_MOCK" not in env
