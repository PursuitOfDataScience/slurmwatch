from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from slurmwatch.collector import TelemetryCollector, _read_meminfo_total
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import JobContext, TelemetrySnapshot


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


class TestReadMeminfo:
    def test_read_meminfo_total(self) -> None:
        total = _read_meminfo_total()
        assert total > 0
        assert isinstance(total, int)


class _FakeNVMLError(Exception):
    pass


class _FakeUtil:
    gpu = 75


class _FakeMem:
    used = 20 * 1024**3
    total = 80 * 1024**3


class _FakeProc:
    def __init__(self, pid: int, mem: int) -> None:
        self.pid = pid
        self.usedGpuMemory = mem


class _FakePUtil:
    def __init__(self, pid: int, sm: int) -> None:
        self.pid = pid
        self.smUtil = sm


class _FakePynvml:
    NVMLError = _FakeNVMLError
    NVML_TEMPERATURE_GPU = 0
    nvmlClocksThrottleReasonSwPowerCap = 1
    nvmlClocksThrottleReasonHwThermal = 2
    nvmlClocksThrottleReasonSwThermal = 4

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
        return [_FakeProc(1000, 18 * 1024**3)]

    @staticmethod
    def nvmlDeviceGetGraphicsRunningProcesses(h: object) -> list[_FakeProc]:
        return []

    @staticmethod
    def nvmlDeviceGetProcessUtilization(h: object, ts: int) -> list[_FakePUtil]:
        return [_FakePUtil(1000, 60)]


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
        # PIDs 1000/1001 come from the fixture cgroup.procs; proc 1000 is on this GPU.
        assert g.process_memory_bytes == 18 * 1024**3
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
