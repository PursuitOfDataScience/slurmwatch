#!/usr/bin/env python3
"""Render the README hero GIF (assets/demo.gif).

Drives the real slurmwatch TUI widgets with a representative (synthetic) snapshot
sequence, exports each frame as a Textual SVG screenshot, rasterizes them, and
assembles an animated GIF. The scene intentionally shows one busy and one idle
GPU so the allocation-efficiency verdict ("GPU UNDERUSED — 1/2 active, 1 idle")
is visible.

Usage (from the repo root):

    pip install cairosvg pillow            # in addition to the project deps
    python assets/render_demo.py

Requires the system cairo library (libcairo) for cairosvg.
"""

from __future__ import annotations

import asyncio
import glob
import io
import math
import os
import sys
import tempfile
import time

import cairosvg
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.environ.setdefault("SLURMWATCH_MOCK", "1")

from textual.app import App  # noqa: E402

from slurmwatch.config import SlurmwatchConfig  # noqa: E402
from slurmwatch.model import (  # noqa: E402
    CpuMetrics,
    GpuMetrics,
    JobContext,
    MemoryMetrics,
    TelemetrySnapshot,
)
from slurmwatch.tui import (  # noqa: E402
    CpuPanel,
    DashboardScreen,
    GpuPanel,
    MemoryPanel,
    VerdictPanel,
)

WARMUP, FRAMES = 6, 18
OUTPUT = os.path.join(REPO_ROOT, "assets", "demo.gif")


class _FakeCollector:
    def __init__(self) -> None:
        self.config = SlurmwatchConfig()

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def stop_sync(self) -> None: ...
    async def next_snapshot(self) -> TelemetrySnapshot:  # never resolves; UI is driven manually
        await asyncio.sleep(3600)
        raise RuntimeError


def make_snapshot(t: int) -> TelemetrySnapshot:
    cores = 16
    cpu_pct = 61 + 7 * math.sin(t * 0.5)
    mem_pct = min(71.0, 49 + t * 1.1)
    limit = 64 * 1024**3
    cur = int(mem_pct / 100 * limit)
    ws = int(cur * 0.84)
    gpus = []
    for i, raw in enumerate([92 + 3 * math.sin(t * 0.55), 3 + 1.5 * math.sin(t * 0.4)]):
        u = max(0.0, min(100.0, raw))
        busy = i == 0
        gpus.append(
            GpuMetrics(
                index=i,
                uuid=f"GPU-{i}",
                name="NVIDIA A100-SXM4-80GB",
                utilization_percent=round(u, 1),
                memory_used_bytes=int((0.64 if busy else 0.02) * 80 * 1024**3),
                memory_total_bytes=80 * 1024**3,
                memory_utilization_percent=round(64 if busy else 2, 1),
                power_watts=round((338 if busy else 64) + 7 * math.sin(t * 0.5 + i), 1),
                temperature_celsius=round((63 if busy else 36) + 3 * math.sin(t * 0.3 + i), 1),
                throttling=False,
                process_utilization_percent=round(u if busy else 0.8, 1),
                process_memory_bytes=int((0.62 if busy else 0.0) * 80 * 1024**3),
            )
        )
    active = sum(1 for g in gpus if g.process_utilization_percent > 5)
    return TelemetrySnapshot(
        timestamp=time.time(),
        job_id="4815162",
        step_id="0",
        hostname="gpu-node-07",
        elapsed_seconds=7200 + t * 30,
        cpu=CpuMetrics(
            cores_allocated=cores,
            usage_ns=0,
            usage_percent=round(cpu_pct, 1),
            effective_cores=round(cpu_pct * cores / 100.0, 1),
        ),
        memory=MemoryMetrics(
            current_bytes=cur,
            limit_bytes=limit,
            peak_bytes=int(cur * 1.04),
            usage_percent=round(mem_pct, 1),
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=ws,
            cache_bytes=cur - ws,
        ),
        gpus=gpus,
        node_count=4,
        node_index=0,
        gpu_count_requested=2,
        gpu_active_count=active,
    )


JOB = JobContext(
    job_id="4815162",
    username="ada",
    partition="gpu-a100",
    nodelist="gpu-node-[07-10]",
    hostname="gpu-node-07",
    cpus_allocated=16,
    mem_limit_bytes=64 * 1024**3,
    gpu_count_requested=2,
    gpu_indices=[0, 1],
    step_id="0",
    uid=1001,
    job_start_time=time.time() - 7200,
    nodelist_resolved=["gpu-node-07", "gpu-node-08", "gpu-node-09", "gpu-node-10"],
)


class _ShotApp(App):  # type: ignore[type-arg]
    TITLE = "slurmwatch"

    def __init__(self) -> None:
        super().__init__()
        self.collector = _FakeCollector()
        self.scr: DashboardScreen | None = None

    async def on_mount(self) -> None:
        self.scr = DashboardScreen(self.collector, JOB, self.collector.config)
        await self.push_screen(self.scr)


def _fix_sizes(scr: DashboardScreen) -> None:
    # Size panels to their content (the live app scrolls; a screenshot should not clip).
    scr.query_one("#grid-container").styles.height = "auto"
    scr.query_one("#grid-container").styles.overflow_y = "hidden"
    scr.query_one("#cpu-panel", CpuPanel).styles.height = 5
    scr.query_one("#mem-panel", MemoryPanel).styles.height = 6
    scr.query_one("#gpu-panel", GpuPanel).styles.height = 15
    scr.query_one("#verdict-panel", VerdictPanel).styles.height = 7


async def _capture(tmp: str) -> list[str]:
    app = _ShotApp()
    svgs: list[str] = []
    async with app.run_test(size=(140, 28)) as pilot:
        await pilot.pause()
        assert app.scr is not None
        for t in range(WARMUP + FRAMES):
            app.scr._update_widgets(make_snapshot(t))
            _fix_sizes(app.scr)
            app.scr.refresh(layout=True)
            await pilot.pause()
            if t >= WARMUP:
                path = os.path.join(tmp, f"frame_{t:02d}.svg")
                with open(path, "w") as f:
                    f.write(app.export_screenshot(title="slurmwatch"))
                svgs.append(path)
    return svgs


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svgs = asyncio.run(_capture(tmp))
        frames = [
            Image.open(io.BytesIO(cairosvg.svg2png(url=s, output_width=1300))).convert("RGB")
            for s in sorted(svgs)
        ]
        palette = frames[len(frames) // 2].quantize(colors=255, method=Image.MEDIANCUT)
        paletted = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        paletted[0].save(
            OUTPUT,
            save_all=True,
            append_images=paletted[1:],
            duration=160,
            loop=0,
            optimize=True,
            disposal=2,
        )
    print(f"wrote {OUTPUT} ({os.path.getsize(OUTPUT)} bytes, {len(svgs)} frames)")
    for leftover in glob.glob(os.path.join(REPO_ROOT, "assets", "frame_*.svg")):
        os.remove(leftover)


if __name__ == "__main__":
    main()
