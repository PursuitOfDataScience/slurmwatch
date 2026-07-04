from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from slurmwatch.collector import TelemetryCollector, _gpu_is_active, _read_meminfo_total
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import GpuMetrics, JobContext, TelemetrySnapshot


@pytest.fixture
def job_ctx() -> JobContext:
    return JobContext(
        job_id="12345",
        username="testuser",
        partition="gpu",
        nodelist="cn001",
        hostname="cn001",
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024 * 1024 * 1024,
        gpu_count_requested=2,
        gpu_indices=[0, 1],
        step_id="0",
        uid=1001,
        job_start_time=time.time() - 3600,
    )


@pytest.fixture
def mock_job_ctx() -> JobContext:
    return JobContext(
        job_id="12345",
        username="testuser",
        partition="gpu",
        nodelist="cn001",
        hostname="cn001",
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024 * 1024 * 1024,
        gpu_count_requested=2,
        gpu_indices=[0, 1],
        step_id="0",
        uid=1001,
        job_start_time=time.time() - 3600,
    )


class TestCollectorMockMode:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_collector_start_stop(self, mock_job_ctx: JobContext) -> None:
        config = SlurmwatchConfig(poll_interval=0.1)
        collector = TelemetryCollector(mock_job_ctx, config)
        await collector.start()
        await asyncio.sleep(0.3)
        assert collector._task is not None
        assert not collector._task.done()
        await collector.stop()
        assert collector._task.done()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_collector_produces_snapshot(self, mock_job_ctx: JobContext) -> None:
        config = SlurmwatchConfig(poll_interval=0.1)
        collector = TelemetryCollector(mock_job_ctx, config)
        await collector.start()
        try:
            snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
            assert isinstance(snapshot, TelemetrySnapshot)
            assert snapshot.job_id == "12345"
            assert snapshot.cpu.cores_allocated == 16
            assert snapshot.hostname == "cn001"
            assert 0 <= snapshot.cpu.usage_percent <= 100
            assert snapshot.memory.limit_bytes == 64 * 1024 * 1024 * 1024
            assert snapshot.cpu.effective_cores > 0
        finally:
            await collector.stop()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_collector_multiple_snapshots(self, mock_job_ctx: JobContext) -> None:
        config = SlurmwatchConfig(poll_interval=0.1)
        collector = TelemetryCollector(mock_job_ctx, config)
        await collector.start()
        try:
            snap1 = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
            snap2 = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
            assert snap2.timestamp >= snap1.timestamp
            assert snap2.cpu.effective_cores >= 0
        finally:
            await collector.stop()


class TestSnapshotSerialization:
    def test_snapshot_json(self) -> None:
        snap = _make_test_snapshot()
        j = snap.to_json()
        parsed = __import__("json").loads(j)
        assert parsed["job_id"] == "12345"
        assert "A100-SXM4-40GB" in j
        assert "effective_cores" in j
        assert "working_set_bytes" in j
        assert "gpu_count_requested" in j
        assert "gpu_active_count" in j

    def test_snapshot_csv_row(self) -> None:
        snap = _make_test_snapshot()
        row = snap.to_csv_row()
        row_str = ",".join(row)
        assert "12345" in row_str
        assert "45.50" in row_str or "45.5" in row_str

    def test_csv_header_length(self) -> None:
        header = TelemetrySnapshot.csv_header(max_gpus=2)
        assert "timestamp" in header
        assert "gpu_0_util_percent" in header
        assert "gpu_1_util_percent" in header
        assert "gpu_2_util_percent" not in header
        assert "cpu_effective_cores" in header
        assert "mem_working_set_bytes" in header
        assert "gpu_count_requested" in header
        assert "gpu_active_count" in header

    def test_csv_row_matches_header_length(self) -> None:
        snap = _make_test_snapshot()
        row = snap.to_csv_row()
        header = TelemetrySnapshot.csv_header(max_gpus=8)
        assert len(row) == len(header)

    def test_csv_row_padded_for_less_gpus(self) -> None:
        snap = _make_test_snapshot()
        snap.gpus = []
        row = snap.to_csv_row()
        header = TelemetrySnapshot.csv_header(max_gpus=8)
        assert len(row) == len(header)
        assert len(row) == 18 + 8 * 12  # 18 fixed + 8 GPUs * 12 cols

    def test_csv_row_has_common_columns(self) -> None:
        snap = _make_test_snapshot()
        row = snap.to_csv_row()
        assert len(row) >= 18
        assert row[0].startswith("1234567890")
        assert row[1] == "12345"


def _make_test_snapshot() -> TelemetrySnapshot:
    from slurmwatch.model import CpuMetrics, GpuMetrics, MemoryMetrics

    return TelemetrySnapshot(
        timestamp=1234567890.0,
        job_id="12345",
        step_id="0",
        hostname="cn001",
        elapsed_seconds=3600,
        cpu=CpuMetrics(
            cores_allocated=16,
            usage_ns=1_000_000_000,
            usage_percent=45.5,
            effective_cores=7.3,
        ),
        memory=MemoryMetrics(
            current_bytes=30 * 1024**3,
            limit_bytes=64 * 1024**3,
            peak_bytes=40 * 1024**3,
            usage_percent=46.9,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=25 * 1024**3,
            cache_bytes=5 * 1024**3,
        ),
        gpus=[
            GpuMetrics(
                index=0,
                uuid="GPU-abc123",
                name="A100-SXM4-40GB",
                utilization_percent=72.5,
                memory_used_bytes=20 * 1024**3,
                memory_total_bytes=40 * 1024**3,
                memory_utilization_percent=50.0,
                power_watts=250.0,
                temperature_celsius=65.0,
                throttling=False,
                process_utilization_percent=60.0,
                process_memory_bytes=18 * 1024**3,
            ),
        ],
        gpu_count_requested=4,
        gpu_active_count=1,
    )


class TestRealCgroupCollector:
    @pytest.fixture
    def cgroup_job_ctx(self, fake_cgroup_v2_job: Path) -> JobContext:  # noqa: F811
        mem_limit = 8 * 1024**3
        return JobContext(
            job_id="12345",
            username="testuser",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=16,
            mem_limit_bytes=mem_limit,
            gpu_count_requested=0,
            gpu_indices=[],
            step_id="0",
            uid=1001,
            job_start_time=1000.0,
            cgroup_v2_path=str(fake_cgroup_v2_job),
        )

    def test_collect_memory_from_cgroup(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(cgroup_job_ctx)
        mem = collector._collect_memory()
        assert mem.current_bytes == 2 * 1024**3
        assert mem.limit_bytes == 8 * 1024**3
        assert mem.peak_bytes == 4 * 1024**3
        assert mem.working_set_bytes > 0
        assert mem.cache_bytes > 0
        assert mem.usage_percent == 25.0

    def test_memory_limit_capped_at_allocation(
        self, cgroup_job_ctx: JobContext, fake_cgroup_v2_job: Path
    ) -> None:
        # Some nodes leave the job cgroup's memory.max at (near) the node's total
        # RAM. The reported limit must be the memory Slurm actually allocated
        # (8 GiB here), not the unconstrained cgroup value, so usage% is
        # meaningful (regression: this reported node RAM -> ~0% usage).
        (fake_cgroup_v2_job / "memory.max").write_text(str(400 * 1024**3))
        collector = TelemetryCollector(cgroup_job_ctx)
        mem = collector._collect_memory()
        assert mem.limit_bytes == 8 * 1024**3
        assert mem.usage_percent == 25.0

    def test_v1_working_set_excludes_page_cache(self, tmp_path: Path) -> None:
        # cgroup v1 (Midway3's version): memory.usage_in_bytes counts reclaimable
        # page cache, so the working set — and the OOM guard — must subtract the
        # inactive file cache. Regression: v1 set working_set == usage, firing
        # false CRITICAL alerts for data-loading jobs and reporting cache as 0.
        v1 = tmp_path / "memory" / "job_1"
        v1.mkdir(parents=True)
        (v1 / "memory.usage_in_bytes").write_text(str(50 * 1024**3))  # incl. cache
        (v1 / "memory.limit_in_bytes").write_text(str(50 * 1024**3))
        (v1 / "memory.max_usage_in_bytes").write_text(str(50 * 1024**3))
        (v1 / "memory.stat").write_text(
            f"total_cache {45 * 1024**3}\n"
            f"total_inactive_file {40 * 1024**3}\n"
            f"total_active_file {4 * 1024**3}\n"
            f"total_rss {5 * 1024**3}\n"
        )
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=50 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            cgroup_v1_mem_path=str(v1),
        )
        mem = TelemetryCollector(ctx)._collect_memory()
        assert mem.current_bytes == 50 * 1024**3
        # working set excludes reclaimable file cache (inactive + active): 50 - 44.
        assert mem.working_set_bytes == 6 * 1024**3
        assert mem.cache_bytes == 44 * 1024**3
        # Guard is driven by the working set (6/50 = 12%), so no false alarm.
        assert mem.oom_guard_warning is False
        assert mem.oom_guard_critical is False

    def test_collect_cpu_from_cgroup(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(cgroup_job_ctx)
        cpu = collector._collect_cpu(time.time())
        assert cpu.cores_allocated == 16
        assert isinstance(cpu.usage_percent, float)

    def test_get_job_pids_from_cgroup(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(cgroup_job_ctx)
        pids = collector._get_job_pids()
        assert 1000 in pids
        assert 1001 in pids

    def test_cpu_falls_back_to_proc_without_cpuacct(self) -> None:
        # On clusters that constrain jobs via cpuset only (no per-job cpuacct
        # cgroup, e.g. Midway3), CPU must be measured from /proc/<pid>/stat.
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            # No cgroup_v1_cpu_path / cgroup_v2_path set.
        )
        collector = TelemetryCollector(ctx)
        pids = {os.getpid()}
        # Two readings with a positive delta produce a real percentage.
        collector._collect_cpu(1000.0, pids)
        collector._prev_cpu_ns = 0  # force a measurable delta on the next read
        cpu = collector._collect_cpu(1001.0, pids)
        assert cpu.usage_ns > 0
        assert cpu.usage_percent > 0.0

    def test_read_cpu_ns_none_without_source(self) -> None:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        collector = TelemetryCollector(ctx)
        assert collector._read_cpu_ns(set()) is None

    def test_memory_oom_guard_uses_working_set(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(cgroup_job_ctx)
        mem = collector._collect_memory()
        assert mem.oom_guard_warning is False
        # Working set should be current - inactive_file
        current = 2 * 1024**3
        inactive_file = 100 * 1024**2  # 100 MiB from fixture
        expected_ws = current - inactive_file
        assert mem.working_set_bytes == expected_ws

    def test_memory_oom_guard_uses_working_set_threshold(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(
            cgroup_job_ctx,
            SlurmwatchConfig(oom_warning_threshold=0.2, oom_critical_threshold=0.3),
        )
        mem = collector._collect_memory()
        # Limit from fixture is 8 GiB, ws is ~1.9 GiB, so ws_pct ~24%
        ws_guard = mem.working_set_bytes or mem.current_bytes
        ws_pct = (ws_guard / mem.limit_bytes) * 100.0
        assert ws_pct > 20
        assert ws_pct < 30
        assert mem.oom_guard_warning is True
        assert mem.oom_guard_critical is False


class TestRemoteCollector:
    def _remote_ctx(self) -> JobContext:
        return JobContext(
            job_id="777",
            username="u",
            partition="gpu",
            nodelist="cn-002",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=200 * 1024**3,
            gpu_count_requested=2,
            gpu_indices=[],
            job_start_time=time.time() - 3600,  # 1h elapsed
            job_state="RUNNING",
            remote=True,
        )

    def test_remote_snapshot_from_sstat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch import slurm

        usage = slurm.RemoteUsage(rss_bytes=100 * 1024**3, cpu_seconds=7200.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id: usage)
        collector = TelemetryCollector(self._remote_ctx())
        cpu, mem = collector._collect_remote(time.time())
        # 100 GiB of a 200 GiB limit.
        assert mem.current_bytes == 100 * 1024**3
        assert mem.usage_percent == 50.0
        # 7200 CPU-seconds over ~3600s elapsed on 4 cores -> ~2 cores, ~50%.
        assert cpu.usage_ns == 7200 * 1_000_000_000
        assert 1.9 <= cpu.effective_cores <= 2.1
        assert 45.0 <= cpu.usage_percent <= 55.0

    def test_remote_multinode_scales_per_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # sstat totals are job-wide; a multi-node job's RSS/CPU must be scaled to
        # a per-node estimate before comparing to the per-node limit/cores, or a
        # healthy job is reported at Nx (false OOM CRITICAL / >100% CPU).
        from slurmwatch import slurm

        usage = slurm.RemoteUsage(rss_bytes=160 * 1024**3, cpu_seconds=14400.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id: usage)
        ctx = JobContext(
            job_id="888",
            username="u",
            partition="gpu",
            nodelist="cn-[01-04]",
            hostname="login-01",
            cpus_allocated=8,  # per node
            mem_limit_bytes=100 * 1024**3,  # per node
            gpu_count_requested=0,
            gpu_indices=[],
            job_start_time=time.time() - 3600,
            job_state="RUNNING",
            nodelist_resolved=["cn-01", "cn-02", "cn-03", "cn-04"],
            remote=True,
        )
        cpu, mem = TelemetryCollector(ctx)._collect_remote(time.time())
        # per-node RSS = 160/4 = 40 GiB of 100 GiB -> 40%, not 160%.
        assert mem.current_bytes == 40 * 1024**3
        assert mem.usage_percent == 40.0
        assert mem.oom_guard_critical is False
        # per-node CPU = (14400/3600)/4 = 1 core of 8.
        assert 0.9 <= cpu.effective_cores <= 1.1

    def test_remote_throttles_sstat_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch import slurm

        calls = {"n": 0}

        def _count(job_id: str) -> slurm.RemoteUsage:
            calls["n"] += 1
            return slurm.RemoteUsage(rss_bytes=1, cpu_seconds=1.0, sampled=True)

        monkeypatch.setattr(slurm, "resolve_remote_usage", _count)
        collector = TelemetryCollector(self._remote_ctx())
        t = time.time()
        collector._collect_remote(t)
        collector._collect_remote(t + 1)  # within the 5s throttle window
        assert calls["n"] == 1  # second call served from cache


class TestReadMeminfo:
    def test_read_meminfo_total(self) -> None:
        total = _read_meminfo_total()
        assert total > 0
        assert isinstance(total, int)


class TestProcCpuParsing:
    def test_parse_stat_with_spaces_and_parens_in_comm(self) -> None:
        from slurmwatch.collector import _parse_stat_cpu_ticks

        # comm "(tmux: server)" contains a space and parentheses; utime and
        # stime are fields 14 and 15 overall.
        fields = ["0"] * 52
        fields[0] = "4242"
        fields[1] = "(tmux: server)"
        fields[2] = "S"
        fields[13] = "150"  # utime (field 14)
        fields[14] = "50"  # stime (field 15)
        line = " ".join(fields)
        assert _parse_stat_cpu_ticks(line) == 200

    def test_parse_stat_malformed(self) -> None:
        from slurmwatch.collector import _parse_stat_cpu_ticks

        assert _parse_stat_cpu_ticks("garbage with no paren") == 0
        assert _parse_stat_cpu_ticks("123 (x) S 1 2 3") == 0  # too few fields

    def test_read_pid_cpu_ticks_self(self) -> None:
        from slurmwatch.collector import _read_pid_cpu_ticks

        # The current process has accumulated some CPU time.
        assert _read_pid_cpu_ticks(os.getpid()) >= 0
        assert _read_pid_cpu_ticks(99_999_999) == 0  # nonexistent PID


class _FakeNVMLError(Exception):
    pass


class _FakeUtil:
    gpu = 75


class _FakeMem:
    used = 20 * 1024**3
    total = 80 * 1024**3


class _FakeProc:
    def __init__(self, pid: int, mem: int | None) -> None:
        # Real pynvml sets usedGpuMemory to None (not a missing attribute)
        # when NVML reports NVML_VALUE_NOT_AVAILABLE, e.g. on MIG devices.
        self.pid = pid
        self.usedGpuMemory = mem


class _FakePUtil:
    def __init__(self, pid: int, sm: int, ts: int = 0) -> None:
        self.pid = pid
        self.smUtil = sm
        self.timeStamp = ts


class _FakePynvml:
    NVMLError = _FakeNVMLError
    NVML_TEMPERATURE_GPU = 0
    # Real constant names from nvidia-ml-py (the invented HwThermal/SwThermal
    # spellings do not exist in any pynvml release).
    nvmlClocksThrottleReasonSwPowerCap = 1
    nvmlClocksThrottleReasonHwThermalSlowdown = 2
    nvmlClocksThrottleReasonSwThermalSlowdown = 4
    nvmlClocksThrottleReasonHwPowerBrakeSlowdown = 8
    nvmlClocksThrottleReasonHwSlowdown = 16

    @staticmethod
    def nvmlInit() -> None:
        return None

    @staticmethod
    def nvmlShutdown() -> None:
        return None

    @staticmethod
    def nvmlDeviceGetCount() -> int:
        return 2

    @staticmethod
    def nvmlDeviceGetHandleByUUID(uuid: bytes) -> object:
        return ("by_uuid", uuid.decode())

    @staticmethod
    def nvmlDeviceGetHandleByIndex(idx: int) -> object:
        return ("by_index", idx)

    @staticmethod
    def nvmlDeviceGetUUID(h: object) -> str:
        return "GPU-test"

    @staticmethod
    def nvmlDeviceGetName(h: object) -> str:
        return "A100-SXM4-80GB"

    @staticmethod
    def nvmlDeviceGetIndex(h: object) -> int:
        return 0

    @staticmethod
    def nvmlDeviceGetUtilizationRates(h: object) -> _FakeUtil:
        return _FakeUtil()

    @staticmethod
    def nvmlDeviceGetMemoryInfo(h: object) -> _FakeMem:
        return _FakeMem()

    @staticmethod
    def nvmlDeviceGetPowerUsage(h: object) -> int:
        return 250_000

    @staticmethod
    def nvmlDeviceGetTemperature(h: object, sensor: int) -> int:
        return 65

    @staticmethod
    def nvmlDeviceGetCurrentClocksThrottleReasons(h: object) -> int:
        return 0

    @staticmethod
    def nvmlDeviceGetComputeRunningProcesses(h: object) -> list[_FakeProc]:
        # Two job processes (one reporting usedGpuMemory=None as on MIG) and
        # one foreign process that must not be attributed to the job.
        return [
            _FakeProc(1000, 18 * 1024**3),
            _FakeProc(1001, None),
            _FakeProc(4321, 10 * 1024**3),
        ]

    @staticmethod
    def nvmlDeviceGetGraphicsRunningProcesses(h: object) -> list[_FakeProc]:
        return []

    @staticmethod
    def nvmlDeviceGetProcessUtilization(h: object, ts: int) -> list[_FakePUtil]:
        # Multiple samples per pid (old ones must be dropped), several job
        # pids (their newest samples must be SUMMED), plus a foreign pid.
        return [
            _FakePUtil(1000, 90, ts=1),
            _FakePUtil(1000, 40, ts=2),
            _FakePUtil(1001, 20, ts=2),
            _FakePUtil(4321, 35, ts=2),
        ]


class TestGpuActive:
    def test_busy_gpu_active_without_process_util_sample(self) -> None:
        # nvmlDeviceGetProcessUtilization returned nothing (process util 0.0),
        # but the device is at 100% and the job owns its VRAM -> ACTIVE, not
        # idle. Regression: the verdict reported "all GPUs idle" on a pegged GPU.
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA H200",
            utilization_percent=100.0,
            memory_used_bytes=140 * 1024**3,
            memory_total_bytes=144 * 1024**3,
            memory_utilization_percent=97.0,
            power_watts=500.0,
            temperature_celsius=45.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=138 * 1024**3,
        )
        assert _gpu_is_active(g, 5.0) is True

    def test_truly_idle_gpu_not_active(self) -> None:
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA H200",
            utilization_percent=2.0,
            memory_used_bytes=0,
            memory_total_bytes=144 * 1024**3,
            memory_utilization_percent=0.0,
            power_watts=60.0,
            temperature_celsius=30.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=0,
        )
        assert _gpu_is_active(g, 5.0) is False

    def test_shared_gpu_not_credited_to_minor_tenant(self) -> None:
        # Shared, non-isolated GPU driven to 100% by another user; this job only
        # holds a sliver of VRAM and has no process-util sample -> NOT active
        # (device util must not be mis-credited to a minor tenant).
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA H200",
            utilization_percent=100.0,
            memory_used_bytes=140 * 1024**3,
            memory_total_bytes=144 * 1024**3,
            memory_utilization_percent=97.0,
            power_watts=500.0,
            temperature_celsius=45.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=1 * 1024**3,  # 1/140 of used VRAM
        )
        assert _gpu_is_active(g, 5.0) is False


class TestCollectGpus:
    def test_collect_gpus_with_fake_nvml(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="12345",
            username="testuser",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=16,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=1,
            gpu_indices=[0],
            step_id="0",
            uid=1001,
            job_start_time=1000.0,
            cgroup_v2_path=str(fake_cgroup_v2_job),
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object()]
        collector._nvml_handle_info = {0: ("GPU-test", "A100-SXM4-80GB")}

        gpus = collector._collect_gpus()
        assert len(gpus) == 1
        g = gpus[0]
        assert g.utilization_percent == 75.0
        assert g.memory_used_bytes == 20 * 1024**3
        assert g.throttling is False
        # PIDs 1000/1001 come from the fixture's leaf cgroup.procs; pid 4321
        # is another user's process and pid 1001 reports usedGpuMemory=None
        # (which must count as 0, not crash and drop the GPU).
        assert g.process_memory_bytes == 18 * 1024**3
        # Newest sample per job pid, summed: 40 (pid 1000) + 20 (pid 1001).
        assert g.process_utilization_percent == 60.0

    def test_init_nvml_selects_by_uuid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=2,
            gpu_indices=[],
            gpu_uuids=["MIG-aaaa", "GPU-bbbb"],
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is True
        # Both job GPUs were resolved via UUID, not raw NVML index.
        assert len(collector._nvml_handles) == 2
        assert collector._nvml_handles[0] == ("by_uuid", "MIG-aaaa")

    def test_init_nvml_constrained_device_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ConstrainDevices: the job holds node-global GPU index 1, but NVML in
        # the job's device cgroup exposes only that one GPU, renumbered to local
        # index 0. The node-global index must not be dropped by an `ordinal <
        # count` bounds check (regression: this returned zero GPUs, so a job on
        # a pegged GPU showed no GPU telemetry at all).
        import sys

        fake = _FakePynvml()
        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 1))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=1,
            gpu_indices=[1],
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is True
        assert len(collector._nvml_handles) == 1
        assert collector._nvml_handles[0] == ("by_index", 0)

    def test_init_nvml_disabled_without_pynvml(self) -> None:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        collector = TelemetryCollector(ctx)
        gpus = collector._collect_gpus()  # not initialized -> empty
        assert gpus == []

    def test_init_nvml_skips_cpu_only_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A job that requested no GPUs must not attach to every device on a
        # shared GPU node (that would display other users' workloads).
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is False
        assert collector._nvml_handles == []

    def test_throttling_detected_with_real_constants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        fake = _FakePynvml()
        # SwPowerCap | HwThermalSlowdown bits set.
        monkeypatch.setattr(
            _FakePynvml,
            "nvmlDeviceGetCurrentClocksThrottleReasons",
            staticmethod(lambda h: 3),
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=1,
            gpu_indices=[0],
        )
        collector = TelemetryCollector(ctx)
        assert collector._check_gpu_throttling(object()) is True

    def test_no_throttling_when_mask_clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=1,
            gpu_indices=[0],
        )
        collector = TelemetryCollector(ctx)
        assert collector._check_gpu_throttling(object()) is False
