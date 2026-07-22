#!/usr/bin/env python3
"""Render the README hero GIF (assets/demo.gif).

Drives the real slurmwatch TUI through a short, scripted scene and exports each
frame as a Textual SVG screenshot, rasterizes them, and assembles an animated
GIF. The scene tells a story a still screenshot can't:

  * a busy GPU 0 next to an idle GPU 1, so the alarm strip carries a standing
    "1 OF 2 GPUS IDLE" line (facts only — the row's dot and numbers, no verdict);
  * memory that climbs out of the safe band into the OOM guard's warning and
    then critical zones, so the MEM row's dot and the alarm strip light up
    amber then red ("MEMORY ...% of limit") on camera.

The warm "Claude Code" palette is on show throughout: a warm near-black surface
with the coral accent and real card elevation, each resource block in its own
identity hue (CPU deep-cyan, MEM rose, GPU violet) on its bar, its recent-range
tag folded onto the row, and green-amber-red reserved for the health dots /
alarm strip. The JOB card below shows the run's provenance (account/qos/state,
command, workdir, the stdout/stderr log paths, queue wait) packed two per row,
and the job-info + key bar are docked at the foot.

Usage (from the repo root):

    pip install resvg-py pillow            # in addition to the project deps
    python assets/render_demo.py

Rasterizes each SVG frame with resvg (via ``resvg-py``), which honors Textual's
per-cell glyph positioning faithfully — cairosvg mis-advances the wide block and
status glyphs, so percent labels collide with bars and the banner separators
collapse. If ``resvg-py`` isn't installed it falls back to cairosvg (with those
artifacts). A monospace font with block glyphs (DejaVu Sans Mono) must be
available; override its directory with SLURMWATCH_DEMO_FONT_DIR. The theme,
output path, and size can be overridden with the other SLURMWATCH_DEMO_*
environment variables (handy while iterating).
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
from slurmwatch.tui import _CLAUDE_THEME, DashboardScreen  # noqa: E402

# The app's own warm "Claude Code" theme; override with SLURMWATCH_DEMO_THEME
# only to preview a built-in theme.
THEME = os.environ.get("SLURMWATCH_DEMO_THEME", _CLAUDE_THEME.name)
WARMUP, FRAMES = 5, 42
# A tall-ish, wide view so the RESOURCES card (GPU compute+vram merge onto one
# line at >=120 cols) and the JOB card below both show at once above the docked
# bottom bar.
TERM_SIZE = (124, 38)
RENDER_WIDTH = int(os.environ.get("SLURMWATCH_DEMO_WIDTH", "1240"))
FRAME_MS = 110
OUTPUT = os.environ.get("SLURMWATCH_DEMO_OUTPUT", os.path.join(REPO_ROOT, "assets", "demo.gif"))
FONT_DIR = os.environ.get("SLURMWATCH_DEMO_FONT_DIR", "/usr/share/fonts/dejavu")
MONO_FONT = os.environ.get("SLURMWATCH_DEMO_FONT", "DejaVu Sans Mono")

_SPAN = WARMUP + FRAMES - 1


class _FakeCollector:
    def __init__(self) -> None:
        self.config = SlurmwatchConfig()
        # The dashboard's poll loop checks this before awaiting the next snapshot;
        # the scene never ends the job, so it stays False.
        self.job_ended = False

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
    # so the memory row's bar visibly fills toward its limit over the run.
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
            # A plausible lifetime peak above the current, so a regenerated GIF shows
            # the "· peak N" figure (the live collector tracks this as a running max).
            peak_effective_cores=round(min(cores, cpu_pct * cores / 100.0 + 1.5), 1),
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
    time_limit_seconds=24 * 3600,
    nodelist_resolved=["gpu-node-07", "gpu-node-08", "gpu-node-09", "gpu-node-10"],
    # Provenance for the JOB card (account/qos/state, command, workdir, the
    # stdout/stderr log paths, queue wait). Distinct out/err files show the JOB
    # card's two-column path packing — (command | workdir) then (stdout | stderr).
    job_state="RUNNING",
    account="rcc-staff",
    qos="normal",
    command="/home/ada/train/run_a100.sh",
    work_dir="/home/ada/train",
    std_out="/home/ada/train/logs/train_4815162.out",
    std_err="/home/ada/train/logs/train_4815162.err",
    submit_time=time.time() - 7245,  # ~45s queue wait before it started
)


class _ShotApp(App):  # type: ignore[type-arg]
    TITLE = "slurmwatch"

    def __init__(self) -> None:
        super().__init__()
        self.collector = _FakeCollector()
        self.scr: DashboardScreen | None = None

    async def on_mount(self) -> None:
        try:
            self.register_theme(_CLAUDE_THEME)
            if THEME in self.available_themes:
                self.theme = THEME
        except Exception:
            pass
        self.scr = DashboardScreen(self.collector, JOB, self.collector.config)
        await self.push_screen(self.scr)


async def _capture(tmp: str) -> list[str]:
    app = _ShotApp()
    svgs: list[str] = []
    async with app.run_test(size=TERM_SIZE) as pilot:
        await pilot.pause()
        assert app.scr is not None
        for t in range(WARMUP + FRAMES):
            # Each frame appends to the history deques, so the per-row range tag
            # settles on camera as memory climbs into the OOM bands.
            app.scr._update_widgets(make_snapshot(t))
            app.scr.refresh(layout=True)
            await pilot.pause()
            if t >= WARMUP:
                path = os.path.join(tmp, f"frame_{t:02d}.svg")
                with open(path, "w") as f:
                    f.write(app.export_screenshot(title="slurmwatch"))
                svgs.append(path)
    return svgs


def _rasterize(svg_path: str):  # type: ignore[no-untyped-def]
    """SVG file -> RGB PIL image, preferring resvg for faithful glyph metrics."""
    from PIL import Image

    try:
        import resvg_py

        png = resvg_py.svg_to_bytes(
            svg_path=svg_path,
            font_dirs=[FONT_DIR],
            monospace_family=MONO_FONT,
            sans_serif_family=MONO_FONT,
            font_family=MONO_FONT,
        )
        return Image.open(io.BytesIO(bytes(png))).convert("RGB")
    except ImportError:
        import cairosvg

        png = cairosvg.svg2png(url=svg_path, output_width=RENDER_WIDTH)
        return Image.open(io.BytesIO(png)).convert("RGB")


def _build_palette(frames: list):  # type: ignore[no-untyped-def]
    # One shared palette avoids frame-to-frame flicker, but a single frame
    # doesn't contain the whole green -> yellow -> red arc. Stack a few
    # representative frames and quantize the composite so every state's colors
    # survive.
    from PIL import Image

    reps = [frames[0], frames[len(frames) // 2], frames[-1]]
    width = reps[0].width
    strip = Image.new("RGB", (width, sum(f.height for f in reps)))
    y = 0
    for f in reps:
        strip.paste(f, (0, y))
        y += f.height
    return strip.quantize(colors=128, method=Image.MEDIANCUT)


def main() -> None:
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        svgs = asyncio.run(_capture(tmp))
        frames = [_rasterize(s) for s in sorted(svgs)]
        # resvg sizes from the SVG's own dimensions; normalize to RENDER_WIDTH so
        # the GIF width is stable regardless of the rasterizer.
        if frames and frames[0].width != RENDER_WIDTH:
            h = round(frames[0].height * RENDER_WIDTH / frames[0].width)
            frames = [f.resize((RENDER_WIDTH, h), Image.LANCZOS) for f in frames]
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
