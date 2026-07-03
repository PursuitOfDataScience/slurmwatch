#!/usr/bin/env python3
"""Render the README hero GIF (assets/demo.gif).

Drives the real slurmwatch TUI widgets through a short, scripted scene and
exports each frame as a Textual SVG screenshot, rasterizes them, and assembles
an animated GIF. The scene tells a story a still screenshot can't:

  * a busy GPU 0 next to an idle GPU 1, so the allocation-efficiency verdict
    flags "GPU UNDERUSED - 1/2 active, 1 idle" the whole time;
  * memory that climbs out of the safe band into the OOM guard's WARNING and
    then CRITICAL zones, turning the meter yellow then red and flipping the
    verdict with it;
  * a focus sweep that walks the highlight across CPU -> Memory -> GPU ->
    Verdict, showing off the [c]/[m]/[g]/[v] keyboard navigation.

Usage (from the repo root):

    pip install cairosvg pillow            # in addition to the project deps
    python assets/render_demo.py

Requires the system cairo library (libcairo) for cairosvg. The theme, output
path, and size can be overridden with the SLURMWATCH_DEMO_* environment
variables (handy while iterating).
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

# A trendy, cohesive dark palette; falls back to the default if unavailable.
THEME = os.environ.get("SLURMWATCH_DEMO_THEME", "tokyo-night")
WARMUP, FRAMES = 5, 42
TERM_SIZE = (132, 34)
RENDER_WIDTH = int(os.environ.get("SLURMWATCH_DEMO_WIDTH", "1480"))
FRAME_MS = 110
OUTPUT = os.environ.get("SLURMWATCH_DEMO_OUTPUT", os.path.join(REPO_ROOT, "assets", "demo.gif"))

_SPAN = WARMUP + FRAMES - 1


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
    p = max(0.0, min(1.0, t / _SPAN))
    cores = 16
    # A well-fed CPU (steady green ~ two-thirds busy).
    cpu_pct = 66 + 5 * math.sin(t * 0.45)
    # Memory climbs from a comfortable 60% into the OOM guard's warning band
    # (working-set >= 85% of the limit) and finally the critical band (>= 90%),
    # so the meter and the verdict light up yellow then red on camera.
    limit = 64 * 1024**3
    mem_pct = 60.0 + 37.0 * p
    cur = int(mem_pct / 100 * limit)
    ws = int(cur * 0.99)
    ws_pct = ws / limit * 100.0
    gpus = []
    for i, base in enumerate([94.0, 3.0]):
        busy = i == 0
        raw = base + (3 * math.sin(t * 0.5) if busy else 1.2 * math.sin(t * 0.4))
        u = max(0.0, min(100.0, raw))
        gpus.append(
            GpuMetrics(
                index=i,
                uuid=f"GPU-{i}",
                name="NVIDIA A100-SXM4-80GB",
                utilization_percent=round(u, 1),
                memory_used_bytes=int((0.71 if busy else 0.02) * 80 * 1024**3),
                memory_total_bytes=80 * 1024**3,
                memory_utilization_percent=round(71 if busy else 2, 1),
                power_watts=round((352 if busy else 61) + 7 * math.sin(t * 0.5 + i), 1),
                temperature_celsius=round((68 if busy else 34) + 3 * math.sin(t * 0.3 + i), 1),
                throttling=False,
                process_utilization_percent=round(u if busy else 0.6, 1),
                process_memory_bytes=int((0.69 if busy else 0.0) * 80 * 1024**3),
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
            peak_bytes=int(cur * 1.01),
            usage_percent=round(mem_pct, 1),
            oom_guard_warning=ws_pct >= 85.0,
            oom_guard_critical=ws_pct >= 90.0,
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

# Walk the highlight across the panels so the keyboard navigation is on show,
# lingering on Memory as it climbs and on the Verdict at the critical climax.
_FOCUS_ACTIONS = {
    "cpu": "action_focus_cpu",
    "mem": "action_focus_memory",
    "gpu": "action_focus_gpu",
    "verdict": "action_focus_verdict",
}


def _focus_for(p: float) -> str:
    if p < 0.18:
        return "cpu"
    if p < 0.56:
        return "mem"
    if p < 0.78:
        return "gpu"
    return "verdict"


class _ShotApp(App):  # type: ignore[type-arg]
    TITLE = "slurmwatch"

    def __init__(self) -> None:
        super().__init__()
        self.collector = _FakeCollector()
        self.scr: DashboardScreen | None = None

    async def on_mount(self) -> None:
        try:
            if THEME in self.available_themes:
                self.theme = THEME
        except Exception:
            pass
        self.scr = DashboardScreen(self.collector, JOB, self.collector.config)
        await self.push_screen(self.scr)


def _fix_sizes(scr: DashboardScreen) -> None:
    # Size panels to their content (the live app scrolls; a screenshot should
    # not clip) and hide the scrollbar so no chrome bleeds into the capture.
    grid = scr.query_one("#grid-container")
    grid.styles.height = "auto"
    grid.styles.overflow_y = "hidden"
    grid.styles.scrollbar_size_vertical = 0
    scr.query_one("#cpu-panel", CpuPanel).styles.height = 6
    scr.query_one("#mem-panel", MemoryPanel).styles.height = 6
    scr.query_one("#gpu-panel", GpuPanel).styles.height = 14
    scr.query_one("#verdict-panel", VerdictPanel).styles.height = 6


async def _capture(tmp: str) -> list[str]:
    app = _ShotApp()
    svgs: list[str] = []
    async with app.run_test(size=TERM_SIZE) as pilot:
        await pilot.pause()
        assert app.scr is not None
        for t in range(WARMUP + FRAMES):
            app.scr._update_widgets(make_snapshot(t))
            getattr(app.scr, _FOCUS_ACTIONS[_focus_for(t / _SPAN)])()
            _fix_sizes(app.scr)
            app.scr.refresh(layout=True)
            await pilot.pause()
            if t >= WARMUP:
                path = os.path.join(tmp, f"frame_{t:02d}.svg")
                with open(path, "w") as f:
                    f.write(app.export_screenshot(title="slurmwatch"))
                svgs.append(path)
    return svgs


def _build_palette(frames: list[Image.Image]) -> Image.Image:
    # One shared palette avoids frame-to-frame flicker, but a single frame
    # doesn't contain the whole green -> yellow -> red arc. Stack a few
    # representative frames and quantize the composite so every state's colors
    # survive.
    reps = [frames[0], frames[len(frames) // 2], frames[-1]]
    width = reps[0].width
    strip = Image.new("RGB", (width, sum(f.height for f in reps)))
    y = 0
    for f in reps:
        strip.paste(f, (0, y))
        y += f.height
    return strip.quantize(colors=255, method=Image.MEDIANCUT)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svgs = asyncio.run(_capture(tmp))
        frames = []
        for s in sorted(svgs):
            png = cairosvg.svg2png(url=s, output_width=RENDER_WIDTH)
            frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
        palette = _build_palette(frames)
        paletted = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        # Hold the opening and the critical climax a beat longer than the rest.
        durations = [FRAME_MS] * len(paletted)
        durations[0] = 900
        durations[-1] = 1500
        paletted[0].save(
            OUTPUT,
            save_all=True,
            append_images=paletted[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=2,
        )
    print(f"wrote {OUTPUT} ({os.path.getsize(OUTPUT)} bytes, {len(svgs)} frames)")
    for leftover in glob.glob(os.path.join(REPO_ROOT, "assets", "frame_*.svg")):
        os.remove(leftover)


if __name__ == "__main__":
    main()
