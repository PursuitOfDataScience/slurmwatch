from __future__ import annotations

import asyncio

import pytest
from rich.text import Text

import slurmwatch.cli as cli
from slurmwatch import pending
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.exceptions import (
    JobNotFoundError,
    JobNotPendingError,
    JobNotRunningError,
    SlurmCommandError,
)
from slurmwatch.pending import (
    PartitionResources,
    PendingJob,
    explain_reason,
    partition_fits_now,
    resolve_cluster_partitions,
    resolve_pending_job,
    resolve_queue_counts,
)

# A realistic `scontrol show job` record for a pending GPU job.
_PENDING_RECORD = (
    "JobId=12345 JobName=train\n"
    "   UserId=demo(1001) GroupId=demo(1001)\n"
    "   Priority=10432 Nice=0 Account=rcc-staff QOS=normal\n"
    "   JobState=PENDING Reason=Resources Dependency=(null)\n"
    "   SubmitTime=2024-01-15T10:00:00 EligibleTime=2024-01-15T10:00:00\n"
    "   StartTime=2024-01-15T11:00:00 EndTime=Unknown\n"
    "   Partition=gpu-highend AllocNode:Sid=login1:42\n"
    "   NumNodes=1 NumCPUs=16 NumTasks=1 CPUs/Task=1\n"
    "   TimeLimit=1-00:00:00\n"
    "   TRES=cpu=16,mem=64G,node=1,billing=16,gres/gpu=2\n"
    "   Gres=gpu:a100:2\n"
)


class TestExplainReason:
    @pytest.mark.parametrize(
        ("reason", "needle"),
        [
            ("Resources", "free nodes"),
            ("Priority", "higher-priority"),
            ("Dependency", "depends on"),
            ("ReqNodeNotAvail", "unavailable"),
            ("QOSMaxJobsPerUserLimit", "QOS limit"),
            ("QOSGrpCpuLimit", "QOS limit"),
            ("AssocMaxCpuPerUserLimit", "account/association"),
            ("BeginTime", "begin time"),
            ("", "Being scheduled"),
            ("None", "Being scheduled"),
        ],
    )
    def test_known_and_prefix_reasons(self, reason: str, needle: str) -> None:
        assert needle.lower() in explain_reason(reason).lower()

    def test_unknown_reason_is_surfaced_verbatim(self) -> None:
        assert "Wibble" in explain_reason("Wibble")

    def test_nodes_down_free_text_is_not_mislabelled_a_partition_limit(self) -> None:
        # #60 review: this common free-text reason contains "partitions" and used
        # to be mislabelled a partition limit; it's really node availability.
        msg = explain_reason(
            "Nodes required for job are DOWN, DRAINED or reserved for jobs in "
            "higher priority partitions"
        )
        assert "unavailable" in msg.lower()
        assert "partition limit" not in msg.lower()


class TestGpuTypeFromGres:
    def test_typed_colon_form(self) -> None:
        assert pending._gpu_type_from_gres("gpu:a100:2") == "a100"

    def test_typed_tres_equals_form(self) -> None:
        # #60 review: a job-level --gpus=a100:2 records the type only as the TRES
        # equals form gres/gpu:a100=2, which must still yield the type.
        assert pending._gpu_type_from_gres("cpu=16,mem=64G,gres/gpu:a100=2") == "a100"

    def test_untyped_and_empty(self) -> None:
        assert pending._gpu_type_from_gres("gpu:2") == ""
        assert pending._gpu_type_from_gres("gres/gpu=2") == ""
        assert pending._gpu_type_from_gres("") == ""

    def test_typed_gpu_request_via_gpus_flag_keeps_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: a --gpus=a100:2 pending record (type only in ReqTRES `=` form,
        # empty Gres/TresPerNode) resolves with req_gpu_type="a100", not "".
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        rec = (
            "JobId=555 JobName=t\n   JobState=PENDING Reason=Resources\n"
            "   Partition=gpu NumNodes=1 NumCPUs=8\n"
            "   ReqTRES=cpu=8,mem=32G,node=1,gres/gpu=2,gres/gpu:a100=2\n"
            "   TresPerNode=(null) Gres=(null)\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: rec)
        job = resolve_pending_job("555")
        assert job.req_gpus == 2 and job.req_gpu_type == "a100"


class TestResolvePendingJob:
    def _no_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

    def test_parses_a_pending_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: _PENDING_RECORD)
        job = resolve_pending_job("12345")
        assert job.job_id == "12345"
        assert job.partition == "gpu-highend"
        assert job.reason == "Resources"
        assert job.account == "rcc-staff" and job.qos == "normal"
        assert job.username == "demo"
        assert job.priority == 10432
        assert job.req_cpus == 16 and job.req_nodes == 1
        assert job.req_mem_bytes == 64 * 1024**3
        assert job.req_gpus == 2 and job.req_gpu_type == "a100"
        assert job.time_limit_seconds == 24 * 3600
        assert job.submit_time is not None and job.start_time_estimate is not None

    def test_min_memory_per_cpu_is_scaled_to_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #60 review: when TRES has no mem= token, MinMemoryCPU is PER CPU — it must
        # be multiplied by NumCPUs to get the whole-job total (not stored verbatim).
        self._no_mock(monkeypatch)
        rec = (
            "JobId=7 JobName=t\n   JobState=PENDING Reason=Resources\n"
            "   Partition=cpu NumNodes=1 NumCPUs=16\n"
            "   TRES=cpu=16,node=1,billing=16\n"
            "   MinMemoryCPU=4G MinMemoryNode=0\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: rec)
        job = resolve_pending_job("7")
        assert job.req_mem_bytes == 16 * 4 * 1024**3  # 4G/cpu x 16 cpus

    def test_min_memory_node_is_scaled_by_node_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)
        rec = (
            "JobId=8 JobName=t\n   JobState=PENDING Reason=Resources\n"
            "   Partition=cpu NumNodes=4 NumCPUs=32\n"
            "   TRES=cpu=32,node=4,billing=32\n"
            "   MinMemoryNode=32G\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: rec)
        job = resolve_pending_job("8")
        assert job.req_mem_bytes == 4 * 32 * 1024**3  # 32G/node x 4 nodes

    def test_running_job_raises_not_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)
        rec = _PENDING_RECORD.replace("JobState=PENDING", "JobState=RUNNING")
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: rec)
        with pytest.raises(JobNotPendingError):
            resolve_pending_job("12345")

    def test_missing_job_raises_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)

        def _boom(cmd: list[str]) -> str:
            raise SlurmCommandError("Invalid job id specified")

        monkeypatch.setattr(pending, "_run_slurm_cmd", _boom)
        with pytest.raises(JobNotFoundError):
            resolve_pending_job("99999")

    def test_array_prefers_the_pending_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_mock(monkeypatch)
        running = _PENDING_RECORD.replace("JobState=PENDING", "JobState=RUNNING")
        both = running + "\n\n" + _PENDING_RECORD
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: both)
        job = resolve_pending_job("12345")
        assert job.reason == "Resources"  # the pending record was selected


class TestResolveClusterPartitions:
    def test_aggregates_sinfo_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        # 8-field format: %R|%a|%D|%t|%C|%G|%l|%m (mem MB is the last column).
        sinfo = (
            "gpu-a100|up|3|idle|0/96/0/96|gpu:a100:8|12:00:00|257000\n"
            "gpu-a100|up|1|mix|16/16/0/32|gpu:a100:8|12:00:00|257000\n"
            "gpu-a100|up|4|alloc|128/0/0/128|gpu:a100:8|12:00:00|257000\n"
            "cpu|up|10|idle|0/320/0/320|(null)|1-00:00:00|192000\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: sinfo)
        parts = {p.name: p for p in resolve_cluster_partitions("gpu-a100")}
        a = parts["gpu-a100"]
        assert a.total_nodes == 8
        assert a.idle_nodes == 3 and a.mix_nodes == 1
        assert a.free_nodes == 4
        assert a.cpus_idle == 112 and a.cpus_total == 256  # 96+16 idle, 96+32+128 total
        assert a.gpu_types == ["a100"]
        assert a.has_gpus is True
        assert a.max_node_mem_bytes == 257000 * 1024**2
        assert a.is_current is True
        assert a.timelimit_seconds == 12 * 3600
        assert parts["cpu"].gpu_types == [] and parts["cpu"].has_gpus is False
        assert parts["cpu"].is_current is False

    def test_untyped_gpu_partition_is_marked_has_gpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #60 review (high): many clusters report GPUs untyped as `gpu:4`; the
        # partition must still register as having GPUs so GPU jobs aren't hidden.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        monkeypatch.setattr(
            pending,
            "_run_slurm_cmd",
            lambda cmd: "gpu|up|11|mix|276/252/0/528|gpu:4|infinite|515000\n",
        )
        p = resolve_cluster_partitions("gpu")[0]
        assert p.has_gpus is True
        assert p.gpu_types == []  # untyped: no model reported
        assert p.cpus_idle == 252

    def test_multi_partition_current_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # audit-3 #4: `sbatch -p a,b` -> Partition="partA,partB". EVERY listed
        # partition must be flagged current (so none is dropped from the table),
        # not compared as a single "parta,partb" string that matches nothing.
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        sinfo = (
            "partA|up|2|idle|0/64/0/64|(null)|1:00:00|192000\n"
            "partB|up|2|idle|0/64/0/64|(null)|1:00:00|192000\n"
            "partC|up|2|idle|0/64/0/64|(null)|1:00:00|192000\n"
        )
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: sinfo)
        parts = {p.name: p for p in resolve_cluster_partitions("partA,partB")}
        assert parts["partA"].is_current and parts["partB"].is_current
        assert parts["partC"].is_current is False

    def test_unavailable_when_sinfo_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

        def _boom(cmd: list[str]) -> str:
            raise SlurmCommandError("down")

        monkeypatch.setattr(pending, "_run_slurm_cmd", _boom)
        assert resolve_cluster_partitions("x") == []


class TestPartitionFits:
    def _job(self, **kw: object) -> PendingJob:
        base: dict[str, object] = {
            "job_id": "1",
            "raw_job_id": "1",
            "name": "j",
            "username": "u",
            "partition": "p",
            "qos": "",
            "account": "",
            "reason": "Resources",
            "submit_time": None,
            "start_time_estimate": None,
            "priority": None,
            "req_cpus": 8,
            "req_nodes": 1,
            "req_mem_bytes": 0,
            "req_gpus": 0,
            "req_gpu_type": "",
            "time_limit_seconds": None,
        }
        base.update(kw)
        return PendingJob(**base)  # type: ignore[arg-type]

    def test_cpu_job_fits_idle_partition(self) -> None:
        p = PartitionResources("p", True, idle_nodes=2, cpus_idle=64)
        assert partition_fits_now(self._job(req_cpus=8), p) is True

    def test_not_enough_cpus(self) -> None:
        p = PartitionResources("p", True, idle_nodes=2, cpus_idle=4)
        assert partition_fits_now(self._job(req_cpus=8), p) is False

    def test_down_partition_never_fits(self) -> None:
        p = PartitionResources("p", False, idle_nodes=9, cpus_idle=999)
        assert partition_fits_now(self._job(req_cpus=1), p) is False

    def test_gpu_type_must_match(self) -> None:
        p = PartitionResources("p", True, idle_nodes=2, cpus_idle=64, gpu_types=["h100"])
        j = self._job(req_gpus=2, req_gpu_type="a100")
        assert partition_fits_now(j, p) is False
        p.gpu_types = ["a100", "h100"]
        assert partition_fits_now(j, p) is True

    def test_gpu_job_needs_some_gpu(self) -> None:
        p = PartitionResources("p", True, idle_nodes=2, cpus_idle=64, gpu_types=[])
        assert partition_fits_now(self._job(req_gpus=1), p) is False

    def test_untyped_gpu_partition_fits_a_gpu_job(self) -> None:
        # #60 review (high): a partition with GPUs but no reported model (has_gpus
        # True, gpu_types []) must still be considered for a GPU job — not hidden.
        p = PartitionResources("p", True, idle_nodes=3, cpus_idle=96, has_gpus=True)
        assert partition_fits_now(self._job(req_gpus=1), p) is True
        # A specific type request can't be excluded when the partition is untyped.
        assert partition_fits_now(self._job(req_gpus=1, req_gpu_type="a100"), p) is True

    def test_memory_that_no_node_can_hold_does_not_fit(self) -> None:
        # #60 review: a per-node memory request larger than the biggest node is a
        # hard no — don't recommend requeuing there.
        p = PartitionResources(
            "p", True, idle_nodes=4, cpus_idle=256, max_node_mem_bytes=128 * 1024**3
        )
        assert partition_fits_now(self._job(req_cpus=8, req_mem_bytes=900 * 1024**3), p) is False
        assert partition_fits_now(self._job(req_cpus=8, req_mem_bytes=64 * 1024**3), p) is True

    def test_memory_unknown_is_not_rejected(self) -> None:
        # When node memory is unknown (max_node_mem_bytes=0) we can't reject on it.
        p = PartitionResources("p", True, idle_nodes=4, cpus_idle=256, max_node_mem_bytes=0)
        assert partition_fits_now(self._job(req_cpus=8, req_mem_bytes=900 * 1024**3), p) is True


class TestQueueCounts:
    def test_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: False)
        out = "RUNNING\nRUNNING\nPENDING\nCOMPLETING\nPENDING\nSUSPENDED\n"
        monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: out)
        running, waiting = resolve_queue_counts("p")
        assert running == 3  # 2 RUNNING + 1 COMPLETING
        assert waiting == 3  # 2 PENDING + 1 SUSPENDED


class TestMockData:
    def test_mock_tells_a_where_could_it_run_story(self) -> None:
        # The demo data is self-consistent: the current partition is full but an
        # alternative fits — so the feature's payoff is visible offline.
        job = pending._mock_pending_job("777")
        parts = pending._mock_partitions(job.partition)
        current = next(p for p in parts if p.is_current)
        alts = [p for p in parts if partition_fits_now(job, p) and not p.is_current]
        assert partition_fits_now(job, current) is False
        assert any(p.name == "gpu-a100" for p in alts)


# ---------------------------------------------------------------------------
# CLI routing + text summary
# ---------------------------------------------------------------------------


class TestCliRouting:
    def test_resolve_routes_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 1 is in state 'PENDING'.")

        pend = pending._mock_pending_job("1")
        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", lambda job_id: pend)
        ctx, got = cli._resolve_running_or_pending("1")
        assert ctx is None and got is pend

    def test_completed_job_still_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A genuinely non-runnable state (not pending) keeps the clean exit(1).
        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 1 is in state 'COMPLETED'.")

        def _not_pending(job_id: str) -> object:
            raise JobNotPendingError("not pending")

        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", _not_pending)
        with pytest.raises(SystemExit) as exc:
            cli._resolve_running_or_pending("1")
        assert exc.value.code == 1

    def test_once_pending_reports_on_stderr_and_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # #60 review: --once is machine-oriented, so a queued job must keep stdout
        # clean (no prose for a jq/CSV reader) — report on STDERR and exit 1.
        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 777 is in state 'PENDING'.")

        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", pending._mock_pending_job)
        monkeypatch.setattr(cli, "resolve_cluster_partitions", pending._mock_partitions)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda p: (12, 5))
        with pytest.raises(SystemExit) as exc:
            cli._run_once("777", SlurmwatchConfig())
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""  # stdout stays clean for machine consumers
        assert "PENDING" in captured.err and "Why" in captured.err and "Where" in captured.err
        assert "gpu-a100" in captured.err and "FITS NOW" in captured.err
        assert "scontrol update JobId=777 Partition=gpu-a100" in captured.err

    def test_demo_pending_sentinel_routes_to_pending_view(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #60 review: `slurmwatch --demo pending` must reach the pending view even
        # though mock resolve_job_context always returns a RUNNING job.
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        ctx, pend = cli._resolve_running_or_pending("pending")
        assert ctx is None and pend is not None and pend.reason == "Resources"

    def test_headless_pending_writes_no_log(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from pathlib import Path

        def _running_raises(job_id: str) -> object:
            raise JobNotRunningError("Job 777 is in state 'PENDING'.")

        monkeypatch.setattr(cli, "resolve_job_context", _running_raises)
        monkeypatch.setattr(cli, "resolve_pending_job", pending._mock_pending_job)
        monkeypatch.setattr(cli, "resolve_cluster_partitions", pending._mock_partitions)
        monkeypatch.setattr(cli, "resolve_queue_counts", lambda p: (0, 0))
        log = Path(str(tmp_path)) / "out.csv"
        cli._run_headless("777", SlurmwatchConfig(), str(log))
        assert not log.exists()  # nothing logged for a queued job
        err = capsys.readouterr().err
        assert "PENDING" in err and "nothing to log yet" in err


# ---------------------------------------------------------------------------
# TUI view
# ---------------------------------------------------------------------------


class TestPendingTui:
    def _view(self) -> object:
        from slurmwatch.tui import PendingView

        job = pending._mock_pending_job("777")
        v = PendingView()
        v.job = job
        v.partitions = pending._mock_partitions(job.partition)
        v.queue_running, v.queue_pending = 12, 5
        v.config = SlurmwatchConfig()
        return v

    def test_render_has_all_sections_and_tip(self) -> None:
        plain = Text.from_markup(self._view().render()).plain  # type: ignore[attr-defined]
        assert "WHY IT'S WAITING" in plain
        assert "WHEN IT MIGHT START" in plain
        assert "WHERE IT COULD RUN" in plain
        assert "PENDING" in plain and "Resources" in plain
        assert "estimated start" in plain
        assert "gpu-a100" in plain and "YES" in plain
        assert "scontrol update JobId=777 Partition=gpu-a100" in plain

    def test_render_no_data(self) -> None:
        from slurmwatch.tui import PendingView

        assert "resolving" in PendingView().render()

    @pytest.mark.asyncio
    async def test_pending_app_mounts_and_refreshes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch.tui import PendingApp, PendingView

        job = pending._mock_pending_job("777")
        monkeypatch.setattr("slurmwatch.tui.resolve_pending_job", lambda jid: job)
        monkeypatch.setattr(
            "slurmwatch.tui.resolve_cluster_partitions",
            lambda p: pending._mock_partitions(p),
        )
        monkeypatch.setattr("slurmwatch.tui.resolve_queue_counts", lambda p: (12, 5))
        app = PendingApp(job)
        async with app.run_test(size=(110, 40)) as pilot:
            for _ in range(30):
                await pilot.pause()
                await asyncio.sleep(0.03)
                if app.screen.query_one(PendingView).partitions:
                    break
            plain = Text.from_markup(app.screen.query_one(PendingView).render()).plain
            assert "WHERE IT COULD RUN" in plain and "gpu-a100" in plain

    @pytest.mark.asyncio
    async def test_pending_screen_notes_when_job_no_longer_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the job leaves the queue (it started), the screen says so and stops.
        from textual.widgets import Static

        from slurmwatch.tui import PendingApp, PendingScreen

        job = pending._mock_pending_job("777")

        def _now_running(jid: str) -> object:
            raise JobNotPendingError("started")

        monkeypatch.setattr("slurmwatch.tui.resolve_pending_job", _now_running)
        app = PendingApp(job)
        async with app.run_test(size=(110, 40)) as pilot:
            scr = app.screen
            assert isinstance(scr, PendingScreen)
            for _ in range(30):
                await pilot.pause()
                await asyncio.sleep(0.03)
                if scr._done:
                    break
            assert scr._done is True
            # The notice is revealed (its exact rendered text is a Textual-version
            # detail; _done + display is the observable contract).
            assert scr.query_one("#pending-notice", Static).display is True
