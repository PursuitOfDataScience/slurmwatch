from __future__ import annotations

import asyncio
import json
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

    def test_to_json_sanitizes_non_finite(self) -> None:
        # allow_nan=False + _json_safe: a stray non-finite metric emits spec-compliant
        # JSON (null), not "NaN"/"Infinity" (which jq rejects), and never crashes.
        s = _snapshot()
        s.cpu.usage_percent = float("nan")
        s.memory.usage_percent = float("inf")
        text = s.to_json()
        assert "NaN" not in text and "Infinity" not in text
        d = json.loads(text)  # parses cleanly under a strict parser
        assert d["cpu"]["usage_percent"] is None
        assert d["memory"]["usage_percent"] is None

    def test_remote_flag_round_trips(self) -> None:
        # #34/#35: the remote tag must survive JSON (the node switcher parses a
        # streamed node's JSON back into a snapshot), and default to False when a
        # snapshot from an older version omits it.
        s = _snapshot()
        s.remote = True
        assert TelemetrySnapshot.from_json(s.to_json()).remote is True
        assert _snapshot().remote is False

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
        # Bounded + memory-shared so switching to a node whose GPU is held by the
        # job's own step can't hang the stream (mirrors the login-node hop).
        assert any(a.startswith("--immediate=") for a in cmd)
        assert "--mem=0" in cmd
        assert "--gres=none" not in cmd  # gpu=True (default) -> request the GPU
        # Critical: srun must NOT connect the terminal's stdin to the remote task,
        # or it swallows the user's keystrokes at the live dashboard.
        assert "--input=none" in cmd
        assert cmd[cmd.index("-w") + 1] == "cn007"
        # Runs the same install's headless logger streaming JSONL to stdout.
        assert cmd[-6:-1] == ["456", "--log", "/dev/stdout", "--json", "--interval"]
        assert cmd[-1] == "1"  # interval formatted
        assert cmd[cmd.index("-m") + 1] == "slurmwatch"

    def test_gpu_false_drops_the_gres_request(self) -> None:
        # When the node's GPU is held by the job's own step, the stream must run
        # without requesting a GPU so it still launches (CPU/mem live).
        cmd = remote.build_stream_command("456", "cn007", 1.0, gpu=False)
        assert "--gres=none" in cmd and "--mem=0" in cmd


class TestParseSnapshotLine:
    def test_valid_line(self) -> None:
        snap = remote.parse_snapshot_line(_snapshot(host="cn9", node_index=1).to_json().encode())
        assert snap is not None and snap.hostname == "cn9" and snap.node_index == 1

    def test_garbage_and_empty_are_none(self) -> None:
        assert remote.parse_snapshot_line(b"not json at all\n") is None
        assert remote.parse_snapshot_line(b"   \n") is None
        assert remote.parse_snapshot_line(b"") is None


class _FakeProc:
    """Stands in for a subprocess in open_stream tests; the GPU probe awaits wait()."""

    def __init__(self, rc: int = 0) -> None:
        self._rc = rc
        self.returncode: int | None = None
        self.killed = False

    async def wait(self) -> int:
        self.returncode = self._rc
        return self._rc

    def kill(self) -> None:
        self.killed = True


class _HangingProc:
    """A probe/stream child whose wait() never returns until cancelled — models a
    wedged slurmctld ignoring --immediate."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False

    async def wait(self) -> int:
        await asyncio.Event().wait()  # blocks forever
        return 0  # pragma: no cover

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class TestOpenStream:
    @pytest.mark.asyncio
    async def test_returns_the_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # open_stream now runs a GPU probe (`true`) first, then the stream. The
        # probe returns rc 0 (GPU reachable); the stream returns the real process.
        sentinel = object()

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            return _FakeProc(0) if a and a[-1] == "true" else sentinel

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        assert await remote.open_stream("123", "cn9", 1.0) is sentinel

    @pytest.mark.asyncio
    async def test_detaches_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # stdin must be /dev/null so srun can't read (and steal) the terminal's keys.
        captured: dict[str, Any] = {}

        async def fake_exec(*a: Any, **k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(0)  # GPU probe
            captured.update(k)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        await remote.open_stream("123", "cn9", 1.0)
        assert captured["stdin"] == asyncio.subprocess.DEVNULL

    @pytest.mark.asyncio
    async def test_gpu_held_streams_without_gres(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Probe fails (GPU held by the job's own step) -> the stream must run with
        # --gres=none so switching to that node still shows CPU/mem, not hang.
        stream_cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(1)  # GPU not reachable
            stream_cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert "--gres=none" in stream_cmd

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
        monkeypatch.delenv("SLURM_CONF", raising=False)
        env = remote._child_env()
        assert not any(k.startswith("SLURM_") for k in env)  # step context cleared
        assert env["SLURMWATCH_NO_HOP"] == "1"
        assert "SLURMWATCH_MOCK" not in env

    def test_child_env_keeps_slurm_conf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #51: SLURM_CONF must survive (unlike the rest of SLURM_*) so the nested
        # srun stream can find slurm.conf / reach slurmctld on clusters that export
        # it — matching the login-node hop, which keeps it for the same reason.
        monkeypatch.setenv("SLURM_STEP_ID", "7")
        monkeypatch.setenv("SLURM_CONF", "/etc/slurm/custom.conf")
        env = remote._child_env()
        assert env["SLURM_CONF"] == "/etc/slurm/custom.conf"
        assert "SLURM_STEP_ID" not in env  # the step context is still cleared


class TestStreamSubprocessCleanup:
    """N1: the probe/stream srun must be killed on cancellation (a 25s wait_for
    firing / the user quitting mid-connect), never left running as an orphan."""

    @pytest.mark.asyncio
    async def test_probe_killed_on_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = _HangingProc()

        async def fake_exec(*_a: Any, **_k: Any) -> Any:
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        task = asyncio.create_task(remote._stream_can_get_gpu("123", "cn9"))
        await asyncio.sleep(0.05)  # let it spawn and reach proc.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.killed  # reaped in the finally, not orphaned

    @pytest.mark.asyncio
    async def test_probe_killed_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A wedged controller that ignores --immediate: the Python timeout fires and
        # the probe returns False, having killed its child.
        proc = _HangingProc()

        async def fake_exec(*_a: Any, **_k: Any) -> Any:
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr(remote, "_GPU_PROBE_SECONDS", -2.95)  # -> timeout ~0.05s
        assert await remote._stream_can_get_gpu("123", "cn9") is False
        assert proc.killed

    @pytest.mark.asyncio
    async def test_open_stream_kills_probe_on_cancel_before_launch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Cancelling the whole open_stream while its GPU probe is still awaiting must
        # reap the probe child (the probe's finally runs), so no orphan is left even
        # when the cancel targets open_stream rather than the probe directly.
        probe = _HangingProc()

        async def fake_exec(*_a: Any, **_k: Any) -> Any:
            return probe

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        task = asyncio.create_task(remote.open_stream("123", "cn9", 1.0))
        await asyncio.sleep(0.05)  # let the probe spawn and reach wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert probe.killed
