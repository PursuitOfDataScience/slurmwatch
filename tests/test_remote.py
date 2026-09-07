from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
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

    def test_to_json_refuses_a_non_finite_it_cannot_sanitize(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # allow_nan=False is the second layer behind _json_safe, and it is the one that
        # holds the RFC-8259 contract: _json_safe only walks float/dict/list, so a
        # non-finite reaching json.dumps inside a tuple/set (or as a numpy scalar) would
        # be emitted as a bare NaN, which jq and every strict parser reject — silently
        # corrupting a whole --log file. Neutralise the sanitizer to pin the flag itself:
        # a mutation sweep showed dropping allow_nan=False left the suite green.
        from slurmwatch import model as model_mod

        monkeypatch.setattr(model_mod, "_json_safe", lambda o: o)
        s = _snapshot()
        s.cpu.usage_percent = float("nan")
        with pytest.raises(ValueError):
            s.to_json()

    def test_non_finite_round_trips_as_zero_not_a_dropped_frame(self) -> None:
        # to_json maps a non-finite to `null` to stay valid JSON. Feeding that back used
        # to raise TypeError, and parse_snapshot_line swallows any exception as
        # "unparseable" — so the node switcher showed that node as producing NO data at
        # all, with no diagnostic. Losing one metric beats losing the frame.
        s = _snapshot()
        s.cpu.usage_percent = float("nan")
        s.memory.usage_percent = float("inf")
        back = TelemetrySnapshot.from_json(s.to_json())
        assert back.cpu.usage_percent == 0.0
        assert back.memory.usage_percent == 0.0
        assert back.cpu.cores_allocated == s.cpu.cores_allocated  # the rest survives
        assert remote.parse_snapshot_line(s.to_json().encode()) is not None

    def test_csv_neutralizes_a_formula_style_job_name(self) -> None:
        # csv quoting does NOT stop a spreadsheet evaluating a cell that begins = + - @;
        # a job name is arbitrary user text (`sbatch -J '=cmd|"/bin/sh"!A1'`).
        s = _snapshot()
        header = TelemetrySnapshot.csv_header(0)
        name_col = header.index("job_name")
        for hostile in ('=cmd|"/bin/sh"!A1', "+1", "-2", "@SUM(A1)"):
            s.job_name = hostile
            assert s.to_csv_row(0)[name_col].startswith("'")
        s.job_name = "sweep-run17"  # an ordinary name is untouched
        assert s.to_csv_row(0)[name_col] == "sweep-run17"

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
    async def test_gpu_held_streams_over_ssh_to_get_real_numbers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Probe fails (the job's own step holds the GPUs) -> stream over ssh.

        A ``--gres=none`` step is DENIED /dev/nvidiaN, so it can only ever report
        "GPU unreadable" — useless on a multi-node training job, where an inner
        srun holding every GPU is the normal shape. An adopted ssh session keeps
        the job's cgroups for CPU/memory but sits outside the step's device cgroup,
        so NVML reads real utilization/VRAM/power.
        """
        stream_cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(1)  # GPU not reachable by a step
            stream_cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv("SLURMWATCH_NO_SSH", raising=False)
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert stream_cmd[0] == "ssh"
        assert "cn9" in stream_cmd
        assert "--gres=none" not in stream_cmd
        # No remote tty: it would inject control chars into the JSONL stream.
        assert "-t" not in stream_cmd
        remote_cmd = stream_cmd[-1]
        assert "--log /dev/stdout" in remote_cmd and "--json" in remote_cmd

    @pytest.mark.asyncio
    async def test_falls_back_to_a_gres_none_step_when_ssh_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ssh (absent, or the user opted out) -> the blind step, not nothing.

        Live CPU/memory for the other node still beats a stuck switch.
        """
        stream_cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(1)
            stream_cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setenv("SLURMWATCH_NO_SSH", "1")
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert "--gres=none" in stream_cmd
        assert stream_cmd[0] != "ssh"

    @pytest.mark.asyncio
    async def test_reachable_gpu_still_uses_the_slurm_native_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ssh is for the case a step CAN'T get the GPU — not a replacement.

        When the probe succeeds the step reads the GPUs itself, which costs no ssh
        login (each one leaks threads into the job's .extern stepd).
        """
        stream_cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(0)  # GPU reachable by a step
            stream_cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert stream_cmd[0] != "ssh"
        assert "--gres=none" not in stream_cmd

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
    firing / the user quitting mid-connect), never left running as an orphan.

    The three probe cases below cover the probe. The STREAM child is covered by
    :func:`test_no_orphan_when_the_stream_exec_is_cancelled`, and it is covered
    differently on purpose: `open_stream`'s ``except CancelledError`` cannot see
    that process (its only in-``try`` await is the exec itself, so ``proc`` is
    still unbound), so what has to be pinned is asyncio's transport cleanup —
    the guarantee actually relied on. Asserting on a fake there would only test
    the fake, which is why that one spawns a real child.
    """

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

    @pytest.mark.asyncio
    async def test_no_orphan_when_the_stream_exec_is_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A REAL child, cancelled mid-exec, must not survive.

        This is the case the class docstring claims and the three tests above do
        not reach: past the probe, with the stream child already forked. It uses
        a real subprocess because the protection under test is asyncio's, not
        this module's — ``open_stream``'s handler gets ``proc is None`` here, so
        a fake process would assert nothing about the actual guarantee.

        Both outcomes of the race are acceptable and both are exercised; what may
        never happen is a surviving child. The assertion is an orphan COUNT, not
        a timing, so a loaded node cannot flake it.
        """
        marker = f"slurmwatch_orphan_test_{os.getpid()}"
        # `sleep 45 <marker>` does NOT work: sleep sums its arguments and rejects a
        # non-numeric one outright ("invalid time interval"), so the child exits
        # instantly and the whole test passes without ever testing anything. It did,
        # on the first attempt. `exec -a` puts the marker in argv[0] instead, and
        # because exec REPLACES bash the process asyncio owns is the sleep itself.
        # `test_the_orphan_detector_is_not_vacuous` guards the trap directly.
        command = ["/bin/bash", "-c", f"exec -a {marker} sleep 45"]

        async def _no_gpu(*_a: Any, **_k: Any) -> bool:
            return False

        monkeypatch.setattr(remote, "_stream_can_get_gpu", _no_gpu)
        monkeypatch.setattr(remote, "_ssh_stream_allowed", lambda _node: False)
        monkeypatch.setattr(remote, "build_stream_command", lambda *_a, **_k: command)

        def _alive() -> list[str]:
            out = subprocess.run(
                ["ps", "-o", "pid=,comm=,args="], capture_output=True, text=True
            ).stdout.splitlines()
            return [
                ln.split()[0]
                for ln in out
                if len(ln.split()) > 1 and ln.split()[1] == "sleep" and marker in ln
            ]

        outcomes = set()
        try:
            for delay in (0.0, 0.001, 0.005):
                for _ in range(2):
                    task = asyncio.ensure_future(remote.open_stream("123", "cn9", 1.0))
                    await asyncio.sleep(delay)
                    task.cancel()
                    try:
                        proc = await task
                    except asyncio.CancelledError:
                        outcomes.add("cancelled")
                        continue
                    outcomes.add("launched")  # won the race; the caller owns it now
                    assert proc is not None
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(proc.wait(), timeout=2)
            assert outcomes, "no trial ran"
            assert _alive() == [], "a cancelled stream exec left an orphan child"
        finally:
            for pid in _alive():  # never leave one behind, even on failure
                with contextlib.suppress(Exception):
                    os.kill(int(pid), signal.SIGKILL)

    @pytest.mark.asyncio
    async def test_the_orphan_detector_is_not_vacuous(self) -> None:
        """The control for the test above, and the reason it exists.

        An orphan test whose child dies on its own passes for the wrong reason and
        proves nothing -- which is exactly what the first version did, because
        ``sleep`` rejects a non-numeric argument and exited immediately. So: spawn
        the same shape deliberately, confirm the detector sees it, kill it, confirm
        the detector clears. If this fails, the assertion above is worthless
        however green it looks.
        """
        marker = f"slurmwatch_vacuity_{os.getpid()}"

        def _alive() -> list[str]:
            out = subprocess.run(
                ["ps", "-o", "pid=,comm=,args="], capture_output=True, text=True
            ).stdout.splitlines()
            return [
                ln.split()[0]
                for ln in out
                if len(ln.split()) > 1 and ln.split()[1] == "sleep" and marker in ln
            ]

        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            f"exec -a {marker} sleep 45",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            for _ in range(40):  # let it reach the exec; no wall-clock assertion
                if _alive():
                    break
                await asyncio.sleep(0.05)
            assert _alive(), "the detector cannot see a child that is definitely alive"
        finally:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)
            for pid in _alive():
                with contextlib.suppress(Exception):
                    os.kill(int(pid), signal.SIGKILL)
        for _ in range(40):
            if not _alive():
                break
            await asyncio.sleep(0.05)
        assert _alive() == [], "and it must clear once the child is gone"


class TestASiteThatRefusesLoginToComputeSsh:
    """`shutil.which("ssh")` proves the CLIENT exists, not that the site permits it.

    Measured on a Booth cluster (Slurm 25.11): `ssh mcn57` from the login node
    answers `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password).`
    and exits 255. The ssh rung was still PREFERRED there over a `--gres=none` step
    that works, and "permission denied" is what a REFUSED SLURM STEP also says — so
    `stream_error_is_permanent` retired the node for the rest of the session and the
    banner blamed Slurm for ssh's answer. Two bugs from one missing fact: which rung
    produced the text.
    """

    @staticmethod
    async def _open(monkeypatch: pytest.MonkeyPatch, node: str) -> list[str]:
        """open_stream with a GPU a step can't get; returns the argv it launched."""
        cmd: list[str] = []

        async def fake_exec(*a: Any, **_k: Any) -> Any:
            if a and a[-1] == "true":
                return _FakeProc(1)  # the job's own step holds the GPUs
            cmd.extend(a)
            return object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv("SLURMWATCH_NO_SSH", raising=False)
        assert await remote.open_stream("123", node, 1.0) is not None
        return cmd

    @pytest.mark.asyncio
    async def test_the_transport_actually_used_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert remote.stream_transport("cn9") == ""
        assert (await self._open(monkeypatch, "cn9"))[0] == "ssh"
        assert remote.stream_transport("cn9") == "ssh"

    @pytest.mark.asyncio
    async def test_a_reachable_gpu_records_the_step_rung(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_exec(*a: Any, **_k: Any) -> Any:
            return _FakeProc(0) if a and a[-1] == "true" else object()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        assert await remote.open_stream("123", "cn9", 1.0) is not None
        assert remote.stream_transport("cn9") == "step"

    @pytest.mark.asyncio
    async def test_the_next_launch_takes_the_step_after_ssh_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: the step rung is still there, so USE it."""
        assert (await self._open(monkeypatch, "cn9"))[0] == "ssh"
        assert remote.retry_other_stream_transport("cn9") is True
        second = await self._open(monkeypatch, "cn9")
        assert second[0] != "ssh"
        assert "--gres=none" in second
        assert remote.stream_transport("cn9") == "step"

    @pytest.mark.asyncio
    async def test_giving_up_is_still_right_once_the_step_has_failed_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a retry loop: each rung gets one turn, then the failure is permanent."""
        await self._open(monkeypatch, "cn9")
        assert remote.retry_other_stream_transport("cn9") is True  # ssh -> step
        await self._open(monkeypatch, "cn9")  # now the step
        assert remote.retry_other_stream_transport("cn9") is False

    def test_a_node_never_streamed_has_no_other_rung_to_try(self) -> None:
        """No transport recorded means the launch itself failed — nothing to switch to."""
        assert remote.retry_other_stream_transport("cn-never") is False

    @pytest.mark.asyncio
    async def test_one_node_being_ssh_less_does_not_condemn_another(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sites exist where only some nodes take an adopted login; keep it per node."""
        await self._open(monkeypatch, "cn9")
        assert remote.retry_other_stream_transport("cn9") is True
        assert (await self._open(monkeypatch, "cn10"))[0] == "ssh"

    def test_the_banner_names_ssh_not_slurm(self) -> None:
        text = "youzhi@cn9: Permission denied (publickey,gssapi-keyex,password)."
        # Without the transport this is indistinguishable from a refused step...
        assert "Slurm refused a step" in remote.summarise_stream_error(text, "cn9")
        # ...and with it, the reader is told what actually happened.
        out = remote.summarise_stream_error(text, "cn9", transport="ssh")
        assert "ssh" in out and "cn9" in out
        assert "Slurm refused" not in out

    def test_the_ssh_summary_is_ascii_clean_under_ascii(self) -> None:
        out = remote.summarise_stream_error(
            "cn9: Permission denied (publickey).", "cn9", ascii_mode=True, transport="ssh"
        )
        assert out.isascii(), out

    def test_a_step_failure_is_still_read_as_a_step_failure(self) -> None:
        """The transport must not rewrite a genuine Slurm refusal."""
        out = remote.summarise_stream_error(
            "srun: error: Access/permission denied for job 42", "cn9", transport="step"
        )
        assert "Slurm refused a step" in out


class TestWhyAStreamDiedIsNotDiscarded:
    """The step's stderr was sent to DEVNULL, so the dashboard could only guess.

    Measured on a second cluster: its `/tmp` is node-local, so an install there is
    invisible from the compute node and srun reports
    `execve(): .../python: No such file or directory`. slurmwatch threw that away and
    showed "it may be busy or unreachable - still retrying" — forever, at a failure no
    number of retries can reach. Fifth instance of the misdiagnosis family SW-7, RD-2,
    SW-19 and the missing-Slurm case belong to.
    """

    def test_the_stream_captures_stderr_at_all(self) -> None:
        """A PIPE, not DEVNULL: everything else here depends on it."""
        import asyncio
        import inspect

        src = inspect.getsource(remote.open_stream)
        assert "stderr=asyncio.subprocess.PIPE" in src, src
        assert asyncio  # keep the import meaningful

    @pytest.mark.parametrize(
        ("text", "permanent"),
        [
            ("error: execve(): /tmp/v/bin/python: No such file or directory", True),
            ("srun: error: Access/permission denied for job 42", True),
            ("srun: error: slurm_load_jobs error: Invalid job id specified", True),
            ("srun: error: Invalid user id 4242", True),
            # NOT permanent: this clears when the job's own step releases the CPUs.
            ("srun: error: Unable to create step for job 555: More processors requested", False),
            ("srun: error: Unable to allocate resources: node configuration not available", False),
            ("", False),
        ],
    )
    def test_only_the_unreachable_failures_stop_the_retry(self, text: str, permanent: bool) -> None:
        assert remote.stream_error_is_permanent(text) is permanent

    @pytest.mark.parametrize(
        ("text", "must_contain"),
        [
            ("error: execve(): /x/python: No such file or directory", "compute node can see"),
            ("srun: error: Access/permission denied", "permission denied"),
            ("srun: error: Invalid job id specified", "no longer knows this job"),
            ("srun: error: Unable to create step for job 5", "could not create a step"),
        ],
    )
    def test_the_summary_names_the_cause(self, text: str, must_contain: str) -> None:
        assert must_contain in remote.summarise_stream_error(text, "cn001")

    def test_an_unrecognised_error_is_quoted_not_invented(self) -> None:
        """Better the step's own first line than a guess dressed up as a diagnosis."""
        out = remote.summarise_stream_error("srun: error: something new\nand more", "cn001")
        assert out == "srun: error: something new"

    def test_the_summary_is_ascii_clean_under_ascii(self) -> None:
        out = remote.summarise_stream_error(
            "error: execve(): /x: No such file or directory", "cn001", ascii_mode=True
        )
        assert out.isascii(), out

    @pytest.mark.asyncio
    async def test_reading_the_error_never_blocks_forever(self) -> None:
        """A step that dies without writing anything must not hang the read."""

        class _NoStderr:
            stderr = None

        assert await remote.read_stream_error(_NoStderr()) == ""  # type: ignore[arg-type]


class TestABareNonFiniteTokenFromAnOlderNode:
    """`from_dict` zeroes a `null` in a numeric field; a bare `NaN` got through.

    The two halves of this contract are written down. The producer's half,
    `model._json_safe`, maps NaN/Infinity to `null` so the line stays RFC-8259 and
    `to_json` can pass `allow_nan=False`. The consumer's half, `from_dict._only`,
    coerces that `null` back to zero -- its comment explains that the alternative is
    `remote.parse_snapshot_line` swallowing the exception and the node switcher
    "silently showing that node as producing NO data at all, with no diagnostic".

    But `json.loads` is more permissive than the RFC the producer targets: it reads a
    bare `NaN` token as a real float. A build predating `allow_nan=False` emits exactly
    that, because `json.dumps` defaults to `allow_nan=True`. So the mixed-version hop
    `from_dict` exists to survive was the one case that got through, and `nan` reached
    the dashboard -- `_labeled_bar` printing "nan%" beside a bar clamped to empty, and
    `_area_chart` raising "cannot convert float NaN to integer".
    """

    @staticmethod
    def _wire(value: Any) -> str:
        """A payload as an older node would put it on the wire."""
        d = json.loads(_snapshot().to_json())
        d["cpu"] = {**d["cpu"], "usage_percent": value}
        return json.dumps(d)  # allow_nan defaults to True, as the old build had it

    def test_a_bare_nan_token_becomes_zero_like_a_null(self) -> None:
        assert "NaN" in self._wire(float("nan")), "the fixture must emit a bare token"
        snap = TelemetrySnapshot.from_json(self._wire(float("nan")))
        assert snap.cpu.usage_percent == 0.0

    def test_a_bare_infinity_token_becomes_zero_too(self) -> None:
        for value in (float("inf"), float("-inf")):
            snap = TelemetrySnapshot.from_json(self._wire(value))
            assert snap.cpu.usage_percent == 0.0, value

    def test_the_frame_survives_instead_of_being_dropped(self) -> None:
        """The consequence the `_only` comment names: keep the frame, not lose the node."""
        snap = remote.parse_snapshot_line(self._wire(float("nan")).encode())
        assert snap is not None, "the whole frame was discarded as unparseable"
        assert snap.hostname == "cn2"

    def test_a_null_still_becomes_zero(self) -> None:
        """CONTROL -- the path that already worked, so a regression there is caught."""
        snap = TelemetrySnapshot.from_json(self._wire(None))
        assert snap.cpu.usage_percent == 0.0

    def test_a_finite_value_is_untouched(self) -> None:
        """CONTROL -- passes in both states; a coercion that zeroed real numbers
        would satisfy every test above."""
        snap = TelemetrySnapshot.from_json(self._wire(37.5))
        assert snap.cpu.usage_percent == 37.5
        assert TelemetrySnapshot.from_json(self._wire(0.0)).cpu.usage_percent == 0.0


class TestANegativeElapsedFromAnOlderNode:
    """`elapsed_seconds` was clamped at the producer and taken verbatim here.

    `collector.py:806` computes `max(0, int(now - job_start_time))` and says why:
    *"a just-started job with compute-node clock skew can make now <
    job_start_time, which otherwise rendered 'ran -1:59:56' / '-0%' on the
    dashboard"*. A build predating that clamp streams the raw negative, and
    version skew across the node hop is exactly what `from_dict` documents itself
    as surviving.

    The consumer arithmetic makes it worse than the symptom that motivated the
    original clamp. `min(100.0, elapsed / limit * 100.0)` caps the top of the
    percentage but not the bottom, and `max(0, limit - elapsed)` turns a negative
    elapsed into MORE time remaining than the limit. Measured against a 1h limit:

        elapsed=-100   ->    -3%,  01:01:40 left of 01:00:00 limit
        elapsed=-7196  ->  -200%,  02:59:56 left of 01:00:00 limit

    `_format_duration` clamps internally, so the duration text was always safe --
    which is why this survived: the visibly wrong part was the percentage and the
    impossible "left of" pair, not the "ran" figure the comment named.
    """

    @staticmethod
    def _wire(elapsed: Any) -> str:
        d = json.loads(_snapshot().to_json())
        d["elapsed_seconds"] = elapsed
        return json.dumps(d)

    @pytest.mark.parametrize("value", [-1, -100, -7196])
    def test_a_negative_becomes_zero(self, value: int) -> None:
        snap = TelemetrySnapshot.from_json(self._wire(value))
        assert snap.elapsed_seconds == 0

    def test_the_time_budget_arithmetic_can_no_longer_go_negative(self) -> None:
        """The consequence, computed the way both render sites compute it."""
        snap = TelemetrySnapshot.from_json(self._wire(-7196))
        limit = 3600
        frac = min(100.0, snap.elapsed_seconds / limit * 100.0)
        remaining = max(0, limit - snap.elapsed_seconds)
        assert frac >= 0.0, f"the percentage went negative: {frac}"
        assert remaining <= limit, f"reported {remaining}s remaining against a {limit}s limit"

    def test_the_frame_is_kept_rather_than_dropped(self) -> None:
        """Clamping, not rejecting: `remote.parse_snapshot_line` swallows an
        exception as "unparseable", which would show the node as producing no data
        at all — the outcome `_only`'s comment already argues against."""
        snap = remote.parse_snapshot_line(self._wire(-100).encode())
        assert snap is not None
        assert snap.elapsed_seconds == 0

    @pytest.mark.parametrize("value", [0, 1, 3600, 86400])
    def test_a_non_negative_elapsed_is_untouched(self, value: int) -> None:
        """CONTROL — passes in both states. A clamp that zeroed real elapsed times
        would satisfy every test above and blank the whole time budget."""
        snap = TelemetrySnapshot.from_json(self._wire(value))
        assert snap.elapsed_seconds == value

    def test_the_control_is_not_vacuous(self) -> None:
        """A 1h-elapsed job against a 1h limit must still read 100%, so the
        arithmetic assertion above is not passing on an all-zero snapshot."""
        snap = TelemetrySnapshot.from_json(self._wire(3600))
        assert min(100.0, snap.elapsed_seconds / 3600 * 100.0) == 100.0


# The three stderrs a broken remote install actually produced, measured by running the
# node switcher's own command with `srun --overlap` into a live allocation
# (job 53834744 on midway3-0200, 2026-09-02). They are quoted verbatim because the
# whole point is WHERE in them the cause sits.
#
# 1. The node's python predates this source. Its system python is 3.6.8 and the package
#    floor is 3.10, so any stream whose interpreter resolves to that one hits this.
_SKEW_TRACEBACK = """Traceback (most recent call last):
  File "/usr/lib64/python3.6/runpy.py", line 183, in _run_module_as_main
    mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
  File "/usr/lib64/python3.6/runpy.py", line 142, in _get_module_details
    return _get_module_details(pkg_main_name, error)
  File "/usr/lib64/python3.6/runpy.py", line 109, in _get_module_details
    __import__(pkg_name)
  File "/home/youzhi/slurmwatch/src/slurmwatch/__init__.py", line 1, in <module>
    from ._version import resolve as _resolve_version
  File "/home/youzhi/slurmwatch/src/slurmwatch/_version.py", line 1
    from __future__ import annotations
    ^
SyntaxError: future feature annotations is not defined
srun: error: midway3-0200: task 0: Exited with exit code 1"""
# 2. A python on the node that cannot import the package (a different conda env, a venv
#    that is not the one on PATH there). runpy answers in one line, not a traceback.
_SKEW_NO_MODULE = """/usr/bin/python3: No module named slurmwatch
srun: error: midway3-0200: task 0: Exited with exit code 1"""
# 3. An OLDER slurmwatch on the node, which rejects a flag this build passes. Note that
#    srun's epilogue arrived BEFORE the remote's own line here — the two writers
#    interleave, so position says nothing about which line matters.
_SKEW_OLD_BUILD = """usage: slurmwatch [-h] [--log FILE] [--append] [--once] [--json]
                  [--interval SECONDS] [--verbose] [--version] [--demo]
                  [--ascii] [--format {json,csv}]
                  [job_id]
srun: error: midway3-0200: task 0: Exited with exit code 2
slurmwatch: error: unrecognized arguments: --flag-from-a-newer-build"""
# CONTROL fixture: the remote slurmwatch STARTED and then crashed on a flaky read. Same
# shape (a traceback, srun's epilogue), but a failure the next launch may well not hit.
_TRANSIENT_REMOTE_CRASH = """Traceback (most recent call last):
  File "/opt/sw/lib/python3.11/site-packages/slurmwatch/collector.py", line 71, in _read
    with open(path, "rb") as fh:
OSError: [Errno 5] Input/output error: '/sys/fs/cgroup/memory.current'
srun: error: cn9: task 0: Exited with exit code 1"""


class TestTheNodesPythonIsNotTheOneRunningHere:
    """The step launched; the far side's python/slurmwatch refused the job.

    `stream_error_is_permanent` only knew the wordings srun and slurmstepd use, so all
    three measured version-skew stderrs read as TRANSIENT and the switcher relaunched
    `srun` on a node that can never serve it, on every backoff, for the whole session —
    behind a banner that told the reader it was "still retrying". The sixth instance of
    the misdiagnosis family the `execve()` case above belongs to, and the second where
    the retry itself was the harm.

    The banner was worse than the classification: the summary quotes the FIRST line of
    the stderr, and a traceback's first line is `Traceback (most recent call last):`.
    The line that names the cause was 13 lines further down.
    """

    def test_the_banner_names_the_cause_not_the_traceback_header(self) -> None:
        out = remote.summarise_stream_error(_SKEW_TRACEBACK, "midway3-0200")
        assert "SyntaxError: future feature annotations is not defined" in out
        assert not out.startswith("Traceback")
        assert "midway3-0200" in out

    @pytest.mark.parametrize(
        ("text", "must_contain"),
        [
            (_SKEW_NO_MODULE, "No module named slurmwatch"),
            (_SKEW_OLD_BUILD, "unrecognized arguments"),
        ],
    )
    def test_the_banner_names_the_other_two_shapes_too(self, text: str, must_contain: str) -> None:
        out = remote.summarise_stream_error(text, "midway3-0200")
        # ...and not the argparse `usage:` dump, which is line 1 of shape 3.
        assert must_contain in out and not out.startswith("usage:")

    def test_sruns_own_epilogue_is_never_mistaken_for_the_cause(self) -> None:
        """Every one of the three ends (or, for shape 3, does not end) with
        `srun: error: … task 0: Exited with exit code N`, which names nothing. So the
        line cannot be found by taking the last one either."""
        assert remote._remote_python_failure_line(_SKEW_TRACEBACK).startswith("SyntaxError")
        for text in (_SKEW_TRACEBACK, _SKEW_NO_MODULE, _SKEW_OLD_BUILD):
            assert "task 0: Exited" not in remote._remote_python_failure_line(text)

    @pytest.mark.parametrize("text", [_SKEW_TRACEBACK, _SKEW_NO_MODULE, _SKEW_OLD_BUILD])
    def test_a_node_that_can_never_run_this_build_stops_being_retried(self, text: str) -> None:
        assert remote.stream_error_is_permanent(text) is True

    def test_a_remote_crash_that_may_clear_is_still_retried(self) -> None:
        """CONTROL — passes in both states, and it is the fix's own failure mode:
        condemning every traceback would retire a node for one flaky cgroup read."""
        assert remote.stream_error_is_permanent(_TRANSIENT_REMOTE_CRASH) is False
        assert remote._remote_python_failure_line(_TRANSIENT_REMOTE_CRASH) == ""

    @pytest.mark.parametrize(
        "text",
        [
            "srun: error: Unable to create step for job 555: More processors requested",
            "srun: error: midway3-0200: task 0: Killed",
            "",
        ],
    )
    def test_a_step_level_transient_is_still_transient(self, text: str) -> None:
        """CONTROL — passes in both states. `Killed` was measured on the same node by
        having the streamed task SIGKILL itself mid-stream: the step dying under a
        healthy transport must stay a retry."""
        assert remote.stream_error_is_permanent(text) is False

    def test_the_summary_is_ascii_clean_under_ascii(self) -> None:
        out = remote.summarise_stream_error(_SKEW_TRACEBACK, "midway3-0200", ascii_mode=True)
        assert out.isascii(), out


class TestAnSshRungThatNeverConnected:
    """ssh answering something other than "permission denied" must still hand over.

    The refused case is already handled, but only through its wording: a site that DROPS
    or rejects login->compute logins, or does not resolve compute node names from the
    login node, produces none of the permanent tokens. `stream_error_is_permanent` read
    False, so `retry_other_stream_transport` — the thing that retires that rung for a
    node and lets the `--gres=none` step take over — was never consulted, and the
    switcher retried the one rung that cannot work at that site on every backoff,
    forever.

    The wordings are openssh's own, captured from this login node with
    `-o BatchMode=yes -o ConnectTimeout=3` against unreachable targets.
    """

    NEVER_CONNECTED = [
        "connect to host cn002 port 22: Connection timed out",
        "connect to host cn002 port 22: Connection refused",
        "connect to host cn002 port 22: No route to host",
        "Could not resolve hostname cn002: Name or service not known",
    ]

    @pytest.mark.parametrize("text", NEVER_CONNECTED)
    def test_that_rung_gives_up_its_turn(self, text: str) -> None:
        assert remote.stream_error_is_permanent(text, "ssh") is True

    @pytest.mark.parametrize("text", NEVER_CONNECTED)
    def test_the_same_words_from_a_step_are_not_condemned(self, text: str) -> None:
        """CONTROL — passes in both states. Which rung spoke is a RECORDED fact in this
        module, never an inference from wording (that is the lesson the transport
        bookkeeping exists for), so these must not retire a node on the step rung, where
        the fallback has nothing left to offer and the node would be given up on."""
        assert remote.stream_error_is_permanent(text, "step") is False
        assert remote.stream_error_is_permanent(text) is False

    def test_a_stream_that_worked_and_later_died_is_not_demoted(self) -> None:
        """CONTROL — passes in both states, and it is this fix's own failure mode. That
        rung is the only one that can read the job's GPUs, so a transient death after a
        working session must stay a retry: demoting it would silently swap live GPU
        numbers for "GPU unreadable" for the rest of the session."""
        for text in (
            "Connection to cn002 closed by remote host.",
            "client_loop: send disconnect: Broken pipe",
            "Killed by signal 15.",
        ):
            assert remote.stream_error_is_permanent(text, "ssh") is False
