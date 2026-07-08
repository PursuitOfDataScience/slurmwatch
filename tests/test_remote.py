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


class TestBuildSampleCommand:
    def test_targets_the_node_via_srun_overlap(self) -> None:
        cmd = remote.build_sample_command("456", "cn007", python="/venv/bin/python")
        assert cmd[0] == "srun"
        assert "--jobid=456" in cmd and "--overlap" in cmd
        assert cmd[cmd.index("-w") + 1] == "cn007"
        # Runs the same install's --once --json on the far node.
        assert cmd[-3:] == ["456", "--once", "--json"]
        assert cmd[-5:-3] == ["-m", "slurmwatch"]


class _FakeProc:
    def __init__(self, out: bytes, rc: int) -> None:
        self._out = out
        self.returncode = rc
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._out, b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return 0


def _patch_exec(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> None:
    async def fake_exec(*_a: Any, **_k: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)


class TestSampleNode:
    @pytest.mark.asyncio
    async def test_parses_remote_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = ("some stderr-ish warning\n" + _snapshot(host="cn9", node_index=1).to_json()).encode()
        _patch_exec(monkeypatch, _FakeProc(out, 0))
        snap = await remote.sample_node("123", "cn9")
        assert snap is not None
        assert snap.hostname == "cn9" and snap.node_index == 1

    @pytest.mark.asyncio
    async def test_nonzero_exit_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_exec(monkeypatch, _FakeProc(b"", 1))
        assert await remote.sample_node("123", "cn9") is None

    @pytest.mark.asyncio
    async def test_garbage_output_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_exec(monkeypatch, _FakeProc(b"not json at all\n", 0))
        assert await remote.sample_node("123", "cn9") is None

    @pytest.mark.asyncio
    async def test_missing_srun_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*_a: Any, **_k: Any) -> Any:
            raise FileNotFoundError("srun not found")

        monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
        assert await remote.sample_node("123", "cn9") is None

    def test_child_env_strips_slurm_and_disables_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_STEP_ID", "7")
        monkeypatch.setenv("SLURM_PROCID", "3")
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        env = remote._child_env()
        assert not any(k.startswith("SLURM_") for k in env)  # step context cleared
        assert env["SLURMWATCH_NO_HOP"] == "1"
        assert "SLURMWATCH_MOCK" not in env
