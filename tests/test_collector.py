from __future__ import annotations

import asyncio
import time

import pytest

from slurmwatch.collector import TelemetryCollector
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import JobContext, TelemetrySnapshot


@pytest.fixture
def job_ctx() -> JobContext:
    return JobContext(
        job_id=12345,
        username="testuser",
        partition="gpu",
        nodelist="cn001",
        hostname="cn001",
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024 * 1024 * 1024,
        gpu_count_requested=2,
        gpu_indices=[0, 1],
        step_id=0,
        uid=1001,
        job_start_time=time.time() - 3600,
    )


@pytest.mark.asyncio
async def test_collector_start_stop(job_ctx: JobContext) -> None:
    config = SlurmwatchConfig(poll_interval=0.1)
    collector = TelemetryCollector(job_ctx, config)
    await collector.start()
    await asyncio.sleep(0.3)
    assert collector._task is not None
    assert not collector._task.done()
    await collector.stop()
    assert collector._task.done()


@pytest.mark.asyncio
async def test_collector_produces_snapshot(job_ctx: JobContext) -> None:
    config = SlurmwatchConfig(poll_interval=0.1)
    collector = TelemetryCollector(job_ctx, config)
    await collector.start()
    try:
        snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
        assert isinstance(snapshot, TelemetrySnapshot)
        assert snapshot.job_id == 12345
        assert snapshot.cpu.cores_allocated == 16
        assert snapshot.hostname == "cn001"
        assert 0 <= snapshot.cpu.usage_percent <= 100
        assert snapshot.memory.limit_bytes == 64 * 1024 * 1024 * 1024
    finally:
        await collector.stop()


@pytest.mark.asyncio
async def test_collector_multiple_snapshots(job_ctx: JobContext) -> None:
    config = SlurmwatchConfig(poll_interval=0.1)
    collector = TelemetryCollector(job_ctx, config)
    await collector.start()
    try:
        snap1 = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
        snap2 = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
        assert snap2.timestamp >= snap1.timestamp
    finally:
        await collector.stop()


def test_snapshot_json_serialization() -> None:
    from slurmwatch.model import CpuMetrics, GpuMetrics, MemoryMetrics

    snap = TelemetrySnapshot(
        timestamp=1234567890.0,
        job_id=12345,
        step_id=0,
        hostname="cn001",
        elapsed_seconds=3600,
        cpu=CpuMetrics(cores_allocated=16, usage_ns=1_000_000_000, usage_percent=45.5),
        memory=MemoryMetrics(
            current_bytes=30 * 1024**3,
            limit_bytes=64 * 1024**3,
            peak_bytes=40 * 1024**3,
            usage_percent=46.9,
            oom_guard_warning=False,
            oom_guard_critical=False,
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
            ),
        ],
    )
    j = snap.to_json()
    assert "12345" in j
    assert "A100-SXM4-40GB" in j
    assert "72.5" in j


def test_snapshot_csv_serialization() -> None:
    from slurmwatch.model import CpuMetrics, GpuMetrics, MemoryMetrics

    snap = TelemetrySnapshot(
        timestamp=1234567890.0,
        job_id=12345,
        step_id=0,
        hostname="cn001",
        elapsed_seconds=3600,
        cpu=CpuMetrics(cores_allocated=16, usage_ns=1_000_000_000, usage_percent=45.5),
        memory=MemoryMetrics(
            current_bytes=30 * 1024**3,
            limit_bytes=64 * 1024**3,
            peak_bytes=40 * 1024**3,
            usage_percent=46.9,
            oom_guard_warning=False,
            oom_guard_critical=False,
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
            ),
        ],
    )
    row = snap.to_csv_row()
    assert "12345" in row
    assert "45.50" in row or "45.5" in row


def test_csv_header() -> None:
    header = TelemetrySnapshot.csv_header(max_gpus=2)
    assert header.startswith("timestamp")
    assert "gpu_0_util_percent" in header
    assert "gpu_1_util_percent" in header
    assert "gpu_2_util_percent" not in header
