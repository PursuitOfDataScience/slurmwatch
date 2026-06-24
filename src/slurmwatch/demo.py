from __future__ import annotations

import asyncio
import math
import time

from .model import (
    CpuMetrics,
    GpuMetrics,
    JobContext,
    MemoryMetrics,
    TelemetrySnapshot,
)


def make_demo_job_context() -> JobContext:
    return JobContext(
        job_id=12345,
        username="demo",
        partition="gpu-highend",
        nodelist="cn-[001-004]",
        hostname="cn001",
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024**3,
        gpu_count_requested=4,
        gpu_indices=[0, 1, 2, 3],
        step_id=0,
        uid=1001,
        job_start_time=time.time() - 7200,
    )


class DemoTelemetryCollector:
    def __init__(self, job_ctx: JobContext) -> None:
        self.job_ctx = job_ctx
        self._demo_start = time.monotonic()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def next_snapshot(self) -> TelemetrySnapshot:
        await asyncio.sleep(0)
        now = time.time()
        elapsed = time.monotonic() - self._demo_start

        cpu_pct = 30 + 40 * (0.5 + 0.5 * math.sin(elapsed * 0.4))
        cpu_ns = int(cpu_pct * self.job_ctx.cpus_allocated * 10_000_000 * max(elapsed, 0.1))

        mem_pct = min(88, 25 + (elapsed / 11) * 63)
        current_bytes = int(mem_pct / 100 * self.job_ctx.mem_limit_bytes)
        peak_bytes = min(int(1.05 * current_bytes), self.job_ctx.mem_limit_bytes)
        oom_warning = mem_pct >= 85
        oom_critical = mem_pct >= 90

        gpus = [
            GpuMetrics(
                index=i,
                uuid=f"GPU-demo-{i}",
                name="NVIDIA A100-SXM4-80GB",
                utilization_percent=round(
                    30 + 50 * (0.5 + 0.5 * math.sin(elapsed * 0.3 + i * 1.5)), 1
                ),
                memory_used_bytes=int(
                    (0.4 + 0.3 * (0.5 + 0.5 * math.sin(elapsed * 0.2 + i))) * 80 * 1024**3
                ),
                memory_total_bytes=80 * 1024**3,
                memory_utilization_percent=round(
                    40 + 40 * (0.5 + 0.5 * math.sin(elapsed * 0.2 + i)), 1
                ),
                power_watts=round(200 + 80 * (0.5 + 0.5 * math.sin(elapsed * 0.25 + i)), 1),
                temperature_celsius=round(
                    55 + 20 * (0.5 + 0.5 * math.sin(elapsed * 0.15 + i)), 1
                ),
                throttling=False,
            )
            for i in range(4)
        ]

        return TelemetrySnapshot(
            timestamp=now,
            job_id=self.job_ctx.job_id,
            step_id=self.job_ctx.step_id,
            hostname=self.job_ctx.hostname,
            elapsed_seconds=int(now - (self.job_ctx.job_start_time or now)),
            cpu=CpuMetrics(
                cores_allocated=self.job_ctx.cpus_allocated,
                usage_ns=cpu_ns,
                usage_percent=round(cpu_pct, 1),
            ),
            memory=MemoryMetrics(
                current_bytes=current_bytes,
                limit_bytes=self.job_ctx.mem_limit_bytes,
                peak_bytes=peak_bytes,
                usage_percent=round(mem_pct, 1),
                oom_guard_warning=oom_warning,
                oom_guard_critical=oom_critical,
            ),
            gpus=gpus,
        )
