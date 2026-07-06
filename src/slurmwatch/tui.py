from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections import deque
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Header, ListItem, ListView, Rule, Static

from .collector import TelemetryCollector, _gpu_is_active
from .config import SlurmwatchConfig
from .model import CpuMetrics, GpuMetrics, JobContext, MemoryMetrics, TelemetrySnapshot
from .slurm import resolve_current_jobs, resolve_job_context


def _format_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def _gib(n: float) -> float:
    return n / 1024**3


def _format_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_cores(n: float) -> str:
    """Cores busy without a pointless trailing '.0' (``1.0`` → ``1``, ``2.8`` → ``2.8``)."""
    return f"{n:.1f}".rstrip("0").rstrip(".")


# A non-breaking space renders like a normal space in the terminal but is not
# collapsed when the TUI is captured to SVG for the README demo, so separators
# that sit right next to Rich markup keep their intended gap.
_NBSP = "\N{NO-BREAK SPACE}"

# Warm "Claude Code" palette. Chrome (borders, titles, section headings) is the
# coral accent; each resource *block* carries its own identity hue on its bar,
# trend line, and label, so blocks read as distinct at a glance. Health stays a
# separate channel — the status dot + word — so a block's colour never has to do
# double duty. The block trio (cyan / rose / violet) was validated against the
# warm surface (#262624): every block clears 5:1 contrast, its worst adjacent
# deuteranopia ΔE is 33 — versus 4.9 for the old warm coral/gold pair, which
# collapsed to a single colour under red-green CVD and on 256-colour terminals —
# and each stays ≥36 ΔE from both the coral chrome and every health colour, so a
# block hue can't be mistaken for the chrome or for a warning.
_INK = "#e8e3da"  # primary text (warm off-white)
_DIM = "#a39b8d"  # secondary text (warm grey)
_FAINT = "#6f685d"  # faint text / the empty portion of a bar track
_ACCENT = "#d97757"  # coral — the one chrome accent
_BG = "#1c1b1a"  # the window background (dark text on a coloured key cap)

_SEP = f"[{_FAINT}]{_NBSP}·{_NBSP}[/]"

# Per-block identity hues: cyan / rose / violet — deliberately spread across the
# wheel (and away from the coral chrome accent) so no two blocks read as the same
# colour, even on a 256-colour terminal or with red-green colour-blindness. Keyed
# by the row label so the bar, trend chart, and label of a resource share a hue.
_CPU_COLOR = "#4fb8cc"  # cyan
_MEM_COLOR = "#e08aa8"  # rose
_GPU_COLOR = "#a884e0"  # violet (GPU identity: label, compute bar, trend line)
# The GPU block shows two bars (compute + vram). They belong to one block so they
# stay in the violet family, but a lighter lilac shade for vram makes the two bars
# distinguishable at a glance instead of two identical stacked bars. ΔE 31 from
# the compute violet (19 under deuteranopia), contrast 9:1 on the surface.
_GPU_VRAM_COLOR = "#d3c0f5"  # pale lilac

# When several GPUs are shown together in the device table, each device gets its
# own colour so identical-looking rows (a job saturating every GPU) are still
# easy to tell apart. Eight distinct hues cover a full DGX-class node (8 GPUs, the
# most the tool tabulates) with no repeat; a 9th+ device cycles. Validated on the
# warm surface: worst pair ΔE 25 (12.5 under deuteranopia), every hue ≥5:1
# contrast and clear of the coral chrome (≥30) and the health colours (≥16). The
# one-line row gap and the explicit GPU-index cell are the primary way rows are
# told apart; colour is a strong secondary aid.
_GPU_CYCLE = [
    "#a884e0",  # violet
    "#45c8b8",  # teal
    "#5aa9f0",  # blue
    "#e07ac8",  # magenta
    "#63c98a",  # green
    "#e6b367",  # amber-sand
    "#97cc47",  # lime
    "#5fe650",  # bright green
]


def _gpu_device_color(index: int) -> str:
    return _GPU_CYCLE[index % len(_GPU_CYCLE)]


# One health vocabulary, everywhere: green = fine, amber = warning/underused,
# red = critical/idle. Kept well clear of every block hue (the closest pair, MEM
# rose ↔ warn amber, is ΔE 36) so a status colour never impersonates a block hue.
_HEALTH_COLOR = {"ok": "#6aa84f", "warn": "#e2bb4c", "crit": "#d1584f", "none": _FAINT}
_HEALTH_GLYPH = {"ok": "●", "warn": "▲", "crit": "✖", "none": "·"}
_HEALTH_GLYPH_ASCII = {"ok": "+", "warn": "!", "crit": "x", "none": "-"}

# The warm "Claude Code" theme: a warm near-black surface with the coral accent,
# replacing the cold blue of an off-the-shelf theme. $primary drives the chrome
# (Header bar, box borders, Rule); success/warning/error mirror the health
# vocabulary so themed widgets agree with our hand-drawn ones.
_CLAUDE_THEME = Theme(
    name="slurmwatch",
    primary=_ACCENT,
    secondary=_MEM_COLOR,
    accent=_GPU_COLOR,
    foreground=_INK,
    background="#1c1b1a",
    surface="#262624",
    panel="#1f1e1d",
    success=_HEALTH_COLOR["ok"],
    warning=_HEALTH_COLOR["warn"],
    error=_HEALTH_COLOR["crit"],
    dark=True,
)

# GPU temperature threshold: above this a device is thermally stressed.
_TEMP_HOT_C = 83.0

_BAR_W = 18
_SPARK_W = 12
# Below this width the bars narrow so the essentials still fit an 80-column
# SSH terminal.
_NARROW_COLS = 100


def _glyph(level: str, ascii_mode: bool) -> str:
    table = _HEALTH_GLYPH_ASCII if ascii_mode else _HEALTH_GLYPH
    return table.get(level, table["none"])


def _dot(level: str, ascii_mode: bool) -> str:
    return f"[{_HEALTH_COLOR[level]}]{_glyph(level, ascii_mode)}[/]"


def _color_bar(
    percent: float, length: int = _BAR_W, ascii_mode: bool = False, color: str = _ACCENT
) -> str:
    """A magnitude bar: filled portion in the block's identity ``color``.

    ``percent`` is clamped to [0, 100] so an over-limit value can't overflow the
    bar's width. The fill colour identifies which block the bar belongs to, not
    its health — only the fill *length* carries the magnitude; health lives in
    the status dot/word beside it. The empty track is a faint neutral.
    """
    filled = max(0, min(length, int(percent / 100 * length)))
    fill_ch, empty_ch = ("#", "-") if ascii_mode else ("█", "░")
    parts = []
    if filled:
        parts.append(f"[{color}]{fill_ch * filled}[/]")
    if length - filled:
        parts.append(f"[{_FAINT}]{empty_ch * (length - filled)}[/]")
    return "".join(parts)


def _render_sparkline(
    values: deque[float],
    length: int = _SPARK_W,
    ascii_mode: bool = False,
    stretch: bool = False,
    lo: float = 0.0,
    hi: float = 100.0,
) -> str:
    """A one-row block sparkline (▁▂▃▄▅▆▇█) of ``length`` cells.

    ``stretch`` spreads all available samples across the full width (oldest left,
    newest right) so a trend always fills the row instead of hugging the right
    edge while history is still filling; the default anchors the newest sample to
    the right edge and blank-pads the left (a fixed time-per-column).

    ``lo``/``hi`` set the value range mapped onto the glyph height. The default
    0–100 shows absolute magnitude; passing a series' own min/max auto-scales it
    so a small-but-real wiggle (e.g. 9–15%) becomes visible instead of a dead
    flat line at the bottom of the 0–100 scale.
    """
    chars = "▁▂▃▄▅▆▇█" if not ascii_mode else "_.,-=+#%"
    span = hi - lo if hi > lo else 1.0
    columns = _stretch_columns(values, length) if stretch else _sample_columns(values, length)
    cells: list[str] = []
    for v in columns:
        if v is None:
            cells.append(" ")
            continue
        frac = (v - lo) / span
        level = int(min(max(frac, 0.0), 1.0) * (len(chars) - 1))
        cells.append(chars[max(0, min(level, len(chars) - 1))])
    return "".join(cells)


def _labeled_bar(metric: str, percent: float, width: int, ascii_mode: bool, color: str) -> str:
    """``metric  ███░░   42%`` — a bar that says what it measures.

    Naming the quantity (dim) removes the "what is this number?" ambiguity a bare
    bar + percent creates when two different quantities share a line — e.g. a
    GPU's compute utilisation sitting next to its VRAM fill.
    """
    bar = _color_bar(percent, width, ascii_mode, color)
    return f"[{_DIM}]{metric:<7}[/] {bar} [{_INK}]{percent:>3.0f}%[/]"


def _sample_columns(values: deque[float], width: int) -> list[float | None]:
    """Down-sample a history deque to exactly ``width`` columns.

    The newest sample is anchored to the right edge; while history is still
    filling, the left is padded with ``None`` (rendered blank). Used by the
    one-row sparkline, where a fixed time-per-column reads naturally.
    """
    vals = list(values)
    if not vals:
        return [None] * width
    step = max(len(vals) / width, 1.0)
    out: list[float | None] = []
    for i in range(width):
        offset = int((width - 1 - i) * step)
        idx = len(vals) - 1 - offset
        out.append(vals[idx] if idx >= 0 else None)
    return out


def _stretch_columns(values: deque[float], width: int) -> list[float | None]:
    """Spread all available samples across the full ``width`` (oldest→newest).

    Unlike :func:`_sample_columns`, this fills the whole width even before history
    is full, so a trend never hugs the right edge with a blank left margin while
    it fills up — the oldest sample sits at the left, the newest at the right.
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return [None] * width
    if n == 1 or width == 1:
        return [vals[-1]] * width
    return [vals[round(i / (width - 1) * (n - 1))] for i in range(width)]


def _escape_markup(text: str) -> str:
    """Neutralize console-markup metacharacters in untrusted text.

    Textual's markup parser (unlike Rich's) treats a *lone/unclosed* ``[`` as a
    tag opener and raises ``MarkupError`` when the rest of the line fails to
    parse — so a job name like ``[experiment`` (``sbatch -J``) would crash the
    job selector. Neither ``rich.markup.escape`` nor ``textual.markup.escape``
    neutralizes an unclosed ``[``, so backslash-escape every ``[`` (after any
    literal backslash) — which both engines render as a literal ``[`` (F1).
    """
    return text.replace("\\", "\\\\").replace("[", "\\[")


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _list_gpus(indices: list[int]) -> str:
    if len(indices) == 1:
        return f"GPU {indices[0]}"
    return "GPUs " + ", ".join(str(i) for i in indices)


# ---------------------------------------------------------------------------
# Health: one function per resource, returning (level, one-word status).
# ---------------------------------------------------------------------------


def _cpu_ratio(cpu: CpuMetrics) -> float:
    if cpu.cores_allocated <= 0:
        return 0.0
    return cpu.effective_cores / cpu.cores_allocated


def _cpu_health(cpu: CpuMetrics, underuse_threshold: float = 0.15) -> tuple[str, str]:
    if cpu.cores_allocated <= 0:
        return "none", "n/a"
    ratio = _cpu_ratio(cpu)
    # A single-core allocation can't be "underused"; for multi-core jobs, flag
    # underuse below the configurable ratio (SLURMWATCH_CPU_UNDERUSE, F4).
    if cpu.cores_allocated > 1 and ratio < underuse_threshold:
        return "warn", "underused"
    return "ok", "healthy"


def _mem_health(mem: MemoryMetrics) -> tuple[str, str]:
    if mem.oom_guard_critical:
        return "crit", "near limit"
    if mem.oom_guard_warning:
        return "warn", "high"
    return "ok", "healthy"


def _gpu_health(gpu: GpuMetrics, idle_threshold: float) -> tuple[str, str]:
    if not _gpu_is_active(gpu, idle_threshold):
        return "crit", "idle"
    if gpu.throttling:
        return "warn", "throttling"
    return "ok", "active"


def _mem_ws_pct(mem: MemoryMetrics) -> float:
    ws = mem.working_set_bytes or mem.current_bytes
    return (ws / mem.limit_bytes * 100.0) if mem.limit_bytes > 0 else 0.0


def _banner_segments(snap: TelemetrySnapshot, config: SlurmwatchConfig) -> list[tuple[str, str]]:
    """The dashboard's headline issues, worst first (crit before warn).

    Returns (level, text) pairs. An empty list means everything is healthy and
    the caller renders the calm all-clear summary instead.
    """
    crit: list[tuple[str, str]] = []
    warn: list[tuple[str, str]] = []

    mem = snap.memory
    ws_pct = _mem_ws_pct(mem)
    if mem.oom_guard_critical:
        crit.append(("crit", f"MEMORY {ws_pct:.0f}% — OOM RISK"))
    elif mem.oom_guard_warning:
        warn.append(("warn", f"MEMORY {ws_pct:.0f}% — APPROACHING LIMIT"))

    gpus = snap.gpus
    idle_threshold = config.gpu_idle_threshold
    if gpus:
        idle = sum(1 for g in gpus if not _gpu_is_active(g, idle_threshold))
        total = len(gpus)
        throttling = sum(1 for g in gpus if g.throttling)
        if idle and idle == total:
            crit.append(("crit", f"ALL {total} GPU{'S' if total > 1 else ''} IDLE"))
        elif idle:
            warn.append(("warn", f"{idle} OF {total} GPUS IDLE"))
        if throttling:
            warn.append(("warn", f"{throttling} GPU{'S' if throttling > 1 else ''} THROTTLING"))

    cpu = snap.cpu
    level, _ = _cpu_health(cpu, config.cpu_underuse_threshold)
    if level == "warn":
        # A plain-language headline; the exact figure ("1 of 8 cores") lives in
        # the CPU row and the Recommendations line right below, so the banner
        # doesn't need to repeat a cryptic "1/8" here.
        warn.append(("warn", "CPU UNDERUSED"))

    return crit + warn


def _banner_line(segments: list[tuple[str, str]], ascii_mode: bool, width: int) -> str:
    """Join the alert segments into one line, worst first.

    When they would overflow ``width`` (0 = unknown / unbounded) and soft-wrap
    mid-phrase, collapse to just the single worst alert plus a ``(+N more)``
    hint, so the headline stays a legible single line (B10).
    """
    parts = [
        f"{_dot(level, ascii_mode)} [bold {_HEALTH_COLOR[level]}]{text}[/]"
        for level, text in segments
    ]
    line = f"{_NBSP}{_NBSP}{_NBSP}".join(parts)
    if width and len(segments) > 1:
        try:
            visible = Text.from_markup(line).cell_len
        except Exception:
            visible = len(line)
        if visible > width:
            level, text = segments[0]
            more = len(segments) - 1
            line = (
                f"{_dot(level, ascii_mode)} [bold {_HEALTH_COLOR[level]}]{text}[/]"
                f"   [dim](+{more} more)[/]"
            )
    return line


class StatusBanner(Static):
    """The single most important line: the worst problem, in plain language."""

    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]connecting…[/]"
        snap = self.snapshot
        cfg = self.config or SlurmwatchConfig()
        ascii_mode = cfg.ascii_mode
        segments = _banner_segments(snap, cfg)

        if segments:
            line = _banner_line(segments, ascii_mode, self.size.width)
        else:
            cpu = snap.cpu
            mem = snap.memory
            bits = [f"CPU {cpu.usage_percent:.0f}%"]
            if mem.limit_bytes > 0:
                bits.append(f"MEM {_mem_ws_pct(mem):.0f}%")
            if snap.gpus:
                bits.append(f"GPU {snap.gpu_active_count}/{len(snap.gpus)} active")
            summary = f"{_NBSP}·{_NBSP}".join(bits)
            line = f"{_dot('ok', ascii_mode)} [bold green]ALL HEALTHY[/] {_SEP} {summary}"

        # A GPU job monitored where NVML can't see the devices isn't an alarm;
        # append a neutral note alongside the healthy summary rather than a
        # red/yellow alert.
        if snap.gpu_count_requested > 0 and not snap.gpus:
            note = f"{_dot('none', ascii_mode)} [dim]GPU telemetry unavailable here[/]"
            line = f"{line}{_NBSP}{_NBSP}{_NBSP}{note}"
        return line


class ResourceRows(Static):
    """One scannable row per live resource: dot · bar · % · numbers · spark · status."""

    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None
    # Set by the dashboard: when the GPU DataTable is showing (3+ devices) the
    # rows don't also render per-GPU lines.
    gpu_table_active: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cpu_history: deque[float] = deque(maxlen=60)
        self.mem_history: deque[float] = deque(maxlen=60)
        self.gpu_history: dict[int, deque[float]] = {}

    def _head(self, label: str, color: str, level: str, status: str, ascii_mode: bool) -> str:
        # Answer-first: resource, health dot, one-word status — then the numbers.
        # Status is padded so the labeled bars line up in a column across rows.
        return (
            f"  [{color}]{label:<5}[/] {_dot(level, ascii_mode)} "
            f"[{_HEALTH_COLOR[level]}]{status:<10}[/]"
        )

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]awaiting telemetry…[/]"
        snap = self.snapshot
        cfg = self.config or SlurmwatchConfig()
        ascii_mode = cfg.ascii_mode
        wide = self.size.width >= _NARROW_COLS or self.size.width == 0
        bar_w = _BAR_W if wide else 12

        # One block per resource, joined with a blank line so the section breathes
        # instead of packing three resources into three tight adjacent rows.
        blocks: list[str] = []

        cpu = snap.cpu
        level, word = _cpu_health(cpu, cfg.cpu_underuse_threshold)
        cpu_bar = _labeled_bar("usage", cpu.usage_percent, bar_w, ascii_mode, _CPU_COLOR)
        cpu_detail = f"{_fmt_cores(cpu.effective_cores)} / {cpu.cores_allocated} cores"
        blocks.append(
            f"{self._head('CPU', _CPU_COLOR, level, word, ascii_mode)}   "
            f"{cpu_bar}   [{_DIM}]{cpu_detail}[/]"
        )

        mem = snap.memory
        level, word = _mem_health(mem)
        ws = mem.working_set_bytes or mem.current_bytes
        mem_head = self._head("MEM", _MEM_COLOR, level, word, ascii_mode)
        if mem.limit_bytes > 0:
            mem_pct = _mem_ws_pct(mem)
            mem_detail = f"{_gib(ws):.0f} / {_gib(mem.limit_bytes):.0f} GiB"
            # Peak is secondary; drop it on a narrow terminal so a big-memory job
            # (3-digit GiB) can't push the line past 80 cols and soft-wrap.
            if wide:
                mem_detail += f" · peak {_gib(mem.peak_bytes):.0f} GiB"
            mem_bar = _labeled_bar("used", mem_pct, bar_w, ascii_mode, _MEM_COLOR)
            blocks.append(f"{mem_head}   {mem_bar}   [{_DIM}]{mem_detail}[/]")
        else:
            # No enforced limit → a 'used 0%' bar would contradict the GiB in
            # use, so show the amount only, with no misleading percentage.
            blocks.append(
                f"{mem_head}   [{_DIM}]{'used':<7}[/] "
                f"[{_INK}]{_format_bytes(ws)}[/] [{_DIM}]· no limit set[/]"
            )

        gpus = snap.gpus
        if not self.gpu_table_active:
            if gpus:
                for gpu in gpus:
                    blocks.append("\n".join(self._gpu_block(gpu, cfg, bar_w, ascii_mode)))
            elif snap.gpu_count_requested > 0:
                blocks.append(
                    f"  [dim]GPU   {snap.gpu_count_requested} requested — "
                    "telemetry unavailable here (run on the compute node)[/]"
                )
            else:
                blocks.append("  [dim]GPU   none requested[/]")
        return "\n\n".join(blocks)

    def _gpu_block(
        self, gpu: GpuMetrics, cfg: SlurmwatchConfig, bar_w: int, ascii_mode: bool
    ) -> list[str]:
        # A GPU has two independent "how busy / how full" axes, so it gets two
        # explicitly-labeled bars — compute (SM/CUDA-core utilisation) and vram
        # (memory fill) — instead of one unlabeled bar that reads as whichever
        # number sits beside it. 'vram' (not 'memory') so it can't blur with the
        # MEM row above.
        level, word = _gpu_health(gpu, cfg.gpu_idle_threshold)
        compute = _labeled_bar("compute", gpu.utilization_percent, bar_w, ascii_mode, _GPU_COLOR)
        vram_bar = _labeled_bar(
            "vram", gpu.memory_utilization_percent, bar_w, ascii_mode, _GPU_VRAM_COLOR
        )
        used_g, tot_g = _gib(gpu.memory_used_bytes), _gib(gpu.memory_total_bytes)
        vram_amt = f"{used_g:.0f} / {tot_g:.0f} GiB"
        pwr = f"{gpu.power_watts:.0f} W"
        deg = "C" if ascii_mode else "°C"
        hot = gpu.temperature_celsius >= _TEMP_HOT_C
        temp_txt = f"{gpu.temperature_celsius:.0f} {deg}" + ("!" if hot else "")
        temp = f"[{_HEALTH_COLOR['warn']}]{temp_txt}[/]" if hot else f"[{_DIM}]{temp_txt}[/]"
        indent = "      "
        return [
            self._head(f"GPU{gpu.index}", _GPU_COLOR, level, word, ascii_mode),
            f"{indent}{compute}",
            f"{indent}{vram_bar}   [{_DIM}]{vram_amt}[/]   [{_DIM}]{pwr}[/] [{_FAINT}]·[/] {temp}",
        ]


class GpuTable(DataTable[Any]):
    """A device-per-row table, used when 3+ GPUs would scroll as stacked rows.

    Two modes, keyed on the widget id: the dashboard copy is a read-only
    overview (no row cursor — nothing is selectable there, so an always-on
    highlight is misleading, U5); the detail-screen copy (``id="detail-table"``)
    is interactive and adds the job's per-device share (JOB% / JOB VRAM), which
    the overview doesn't show, so drilling in reveals something new (F6).
    """

    config: SlurmwatchConfig | None = None

    @property
    def _detailed(self) -> bool:
        return self.id == "detail-table"

    def on_mount(self) -> None:
        self.cursor_type = "row" if self._detailed else "none"
        # Rows are given height 2 in update_gpus for a one-line gap between GPUs
        # (see there); zebra stripes would fill that gap with a background band
        # and defeat the point, so they're off — the gap does the separating.
        self.zebra_stripes = False
        if self._detailed:
            self.add_columns("GPU", "COMPUTE", "VRAM", "JOB%", "JOB VRAM", "PWR", "TEMP", "STATUS")
        else:
            self.add_columns("GPU", "COMPUTE", "VRAM", "PWR", "TEMP", "STATUS")

    def update_gpus(self, gpus: list[GpuMetrics], config: SlurmwatchConfig) -> None:
        ascii_mode = config.ascii_mode
        self.clear()
        for gpu in gpus:
            level, word = _gpu_health(gpu, config.gpu_idle_threshold)
            # Each device wears its own colour (index + compute bar + VRAM) so
            # identical rows stay distinguishable. Health (status) and heat (temp)
            # keep their own colour channel — those are the same across devices.
            dcolor = _gpu_device_color(gpu.index)
            gpu_cell = Text(str(gpu.index), style=f"bold {dcolor}")
            bar = _color_bar(gpu.utilization_percent, 8, ascii_mode, dcolor)
            util = Text.from_markup(f"{gpu.utilization_percent:>3.0f}% {bar}")
            vram = Text(
                f"{_gib(gpu.memory_used_bytes):.0f}/{_gib(gpu.memory_total_bytes):.0f} GiB",
                style=dcolor,
            )
            pwr = f"{gpu.power_watts:.0f}W"
            hot = gpu.temperature_celsius >= _TEMP_HOT_C
            deg = "C" if ascii_mode else "°C"
            temp_mark = ("!" if ascii_mode else "⚠") if hot else ""
            temp = Text(
                f"{temp_mark}{gpu.temperature_celsius:.0f}{deg}", style="yellow" if hot else ""
            )
            status = Text(f"{_glyph(level, ascii_mode)} {word}", style=_HEALTH_COLOR[level])
            # height=2 leaves a blank line under each row so adjacent GPUs (often
            # identical when a job saturates every device) are easy to tell apart.
            if self._detailed:
                job_util = f"{gpu.process_utilization_percent:>3.0f}%"
                job_vram = (
                    f"{_gib(gpu.process_memory_bytes):.1f} GiB" if gpu.process_memory_bytes else "—"
                )
                self.add_row(gpu_cell, util, vram, job_util, job_vram, pwr, temp, status, height=2)
            else:
                self.add_row(gpu_cell, util, vram, pwr, temp, status, height=2)


class EfficiencyPanel(Static):
    """Actionable recommendations only — never a context-free "good/bad" grade.

    Whether a given utilisation is *good* depends on the workload (a data-loading
    stage is meant to be CPU-light; a debug run is meant to idle the GPU), so this
    panel does not grade the rows. It surfaces a concrete suggestion only when
    there is a clear, unambiguous inefficiency or risk — idle GPUs, memory about
    to be OOM-killed, an allocation the job never touches — and otherwise says
    there is nothing to change. The live usage numbers live in the rows above.
    """

    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None
    source: str = ""

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]Recommendations: awaiting data…[/]"
        snap = self.snapshot
        cfg = self.config or SlurmwatchConfig()
        ascii_mode = cfg.ascii_mode

        flags = self._flags(snap, cfg)  # (level, text) — real problems only
        notes = self._notes(snap)  # dim, informational (e.g. telemetry gaps)

        lines: list[str] = [f"[bold {_ACCENT}]Recommendations[/]"]
        for level, text in flags:
            lines.append(f"  {_dot(level, ascii_mode)} [{_HEALTH_COLOR[level]}]{text}[/]")
        for text in notes:
            lines.append(f"  {_dot('none', ascii_mode)} [dim]{text}[/]")
        if not flags and not notes:
            lines.append(
                f"  {_dot('ok', ascii_mode)} "
                "[dim]nothing to change — no idle GPUs, memory within its limit[/]"
            )

        note = self._source_note()
        if note:
            lines.append(f"[dim]{note}[/]")
        return "\n".join(lines)

    def _flags(self, snap: TelemetrySnapshot, cfg: SlurmwatchConfig) -> list[tuple[str, str]]:
        flags: list[tuple[str, str]] = []

        cpu = snap.cpu
        if cpu.cores_allocated > 1 and _cpu_ratio(cpu) < cfg.cpu_underuse_threshold:
            flags.append(
                (
                    "warn",
                    f"CPU barely used — {_fmt_cores(cpu.effective_cores)} of {cpu.cores_allocated} "
                    "cores busy; request fewer --cpus-per-task if this stays low",
                )
            )

        mem = snap.memory
        if mem.limit_bytes > 0:
            limit = _gib(mem.limit_bytes)
            if mem.oom_guard_critical:
                flags.append(
                    (
                        "crit",
                        f"Memory at the {limit:.0f} GiB limit — raise --mem or risk an OOM kill",
                    )
                )
            elif mem.oom_guard_warning:
                flags.append(
                    ("warn", f"Memory near the {limit:.0f} GiB limit — raise --mem or trim usage")
                )

        gpus = snap.gpus
        if gpus:
            idle = [g.index for g in gpus if not _gpu_is_active(g, cfg.gpu_idle_threshold)]
            if idle and len(idle) == len(gpus):
                unit = "GPU" if len(gpus) == 1 else "GPUs"
                flags.append(
                    ("crit", f"All {len(gpus)} {unit} idle — release the allocation or start work")
                )
            elif idle:
                active = len(gpus) - len(idle)
                flags.append(
                    (
                        "warn",
                        f"{_list_gpus(idle)} idle — drop to --gres=gpu:{max(active, 1)} "
                        f"to free {_plural(len(idle), 'device')}",
                    )
                )
        return flags

    @staticmethod
    def _notes(snap: TelemetrySnapshot) -> list[str]:
        # Informational, not actionable-here: a GPU job whose devices NVML can't
        # see (e.g. from a login node) — don't silently imply "all clear".
        if snap.gpu_count_requested > 0 and not snap.gpus:
            return ["GPU telemetry unavailable here — run on the compute node to see it"]
        return []

    def _source_note(self) -> str:
        # Only surface the data source when it changes how to read the numbers:
        # a remote sstat estimate is coarser than live on-node data, and demo
        # data isn't real. On-node cgroup accounting is the norm — no need to
        # explain the plumbing (the raw "cgroup v1/v2" was just confusing).
        s = self.source
        if s.startswith("sstat"):
            return "remote estimate (Slurm accounting) — run on the compute node for live numbers"
        if s == "demo data":
            return "demo data — not a real job"
        return ""


class HistoryPanel(Static):
    """A compact recent-history panel: one labelled sparkline per resource.

    Each row is ``label · current% · sparkline`` — a single-row block sparkline
    (▁▂▃▄▅▆▇█) reads at a glance ("was high, dropped to low") where a multi-row
    line chart looked like two disconnected dotted bands. Each series wears its
    block's identity colour so the trend matches the row above it.
    """

    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None
    cpu_history: deque[float] | None = None
    mem_history: deque[float] | None = None
    gpu_history: dict[int, deque[float]] | None = None
    # Advanced once per telemetry poll (by the dashboard) to travel the ripple on
    # steady lines, so a flat value still visibly "moves".
    frame: int = 0

    def render(self) -> str:
        if self.snapshot is None or self.cpu_history is None:
            return ""
        cfg = self.config or SlurmwatchConfig()
        ascii_mode = cfg.ascii_mode
        # height:auto — so size the sparklines from the width alone (fall back
        # before first layout, like the job bar), never gating on a height the
        # auto-sized widget doesn't have yet.
        w = self.size.width or 100

        snap = self.snapshot
        # Each row names *what* the percentage tracks (matching the labels on the
        # bars above) so "62%" isn't a bare, ambiguous number: CPU = fraction of
        # allocated cores busy, MEM = fraction of the memory limit used, GPU =
        # compute (SM) utilisation. CPU and memory always; add the busiest GPU
        # (not an average, which would hide one hot/idle device) when it exists.
        series: list[tuple[str, deque[float], float, str]] = [
            ("CPU busy", self.cpu_history or deque(), snap.cpu.usage_percent, _CPU_COLOR),
            ("MEM used", self.mem_history or deque(), _mem_ws_pct(snap.memory), _MEM_COLOR),
        ]
        gpu_hist = self.gpu_history or {}
        if snap.gpus and gpu_hist:
            hottest = max(snap.gpus, key=lambda g: g.utilization_percent)
            hh = gpu_hist.get(hottest.index)
            if hh is not None:
                series.append(
                    (f"GPU{hottest.index} compute", hh, hottest.utilization_percent, _GPU_COLOR)
                )

        title = f"[bold {_ACCENT}]TRENDS[/] [{_DIM}]· last {cfg.history_seconds}s[/]"
        label_w = max(len(lbl) for lbl, *_ in series)
        # label + " NNN% " + range column (8) + spaces
        spark_w = max(_SPARK_W, w - label_w - 18)

        # A compact group: title, then one sparkline per series with a single
        # blank line between for legibility. The panel is height:auto, so it takes
        # only the room it needs instead of stretching three lines across a tall
        # 1fr box with big empty gaps between them.
        lines = [title, ""]
        for i, (lbl, hist, cur, color) in enumerate(series):
            if i:
                lines.append("")
            lines.append(self._trend_line(lbl, hist, cur, color, label_w, spark_w, ascii_mode))
        return "\n".join(lines)

    # A window spanning at least this many points is "moving" and auto-scaled to
    # its own range so the real shape shows; below it the line is "steady".
    _MOVE_EPS = 3.0

    def _trend_line(
        self,
        label: str,
        hist: deque[float],
        cur: float,
        color: str,
        label_w: int,
        spark_w: int,
        ascii_mode: bool,
    ) -> str:
        vals = list(hist)
        lo, hi = (min(vals), max(vals)) if vals else (cur, cur)
        head = f"[{color}]{label:<{label_w}}[/] [{_INK}]{cur:>3.0f}%[/]"
        if hi - lo >= self._MOVE_EPS:
            # Moving: scale to the line's own range so the real shape is visible.
            rng = f"{lo:.0f}–{hi:.0f}%"
            spark = _render_sparkline(hist, spark_w, ascii_mode, stretch=True, lo=lo, hi=hi)
            return f"{head} [{_DIM}]{rng:<8}[/] [{color}]{spark}[/]"
        # Steady: draw at the value's ABSOLUTE height (so 13%, 4% and 99% sit at
        # different heights, not all mid), with a small travelling ripple so the
        # band still visibly *moves* rather than being a dead-flat row. The ripple
        # is cosmetic — the real value is the number and the "steady" tag.
        spark = self._steady_band(cur, spark_w, ascii_mode)
        return f"{head} [{_DIM}]{'steady':<8}[/] [{color}]{spark}[/]"

    def _steady_band(self, value: float, width: int, ascii_mode: bool) -> str:
        chars = "▁▂▃▄▅▆▇█" if not ascii_mode else "_.,-=+#%"
        top = len(chars) - 1
        base = min(max(value, 0.0), 100.0) / 100.0 * top  # absolute height of the band
        cells = []
        for i in range(width):
            # A gentle sine ripple (~±1 glyph) that travels with the frame counter,
            # so a steady band shimmers along instead of sitting perfectly flat.
            ripple = 0.9 * math.sin(i * 0.45 + self.frame * 0.7)
            level = int(round(min(max(base + ripple, 0.0), float(top))))
            cells.append(chars[level])
        return "".join(cells)


class JobInfoBar(Static):
    """The bottom info bar: what this job is, and how long it can still run.

    Labels every field (so the header line isn't a cryptic ``a · b · c · d``) and
    turns the otherwise-empty space at the foot of the screen into a live
    time-budget line — elapsed vs. the wall-clock limit, time left, and the
    projected end — which the top header never showed.
    """

    snapshot: TelemetrySnapshot | None = None
    job_ctx: JobContext | None = None
    config: SlurmwatchConfig | None = None

    def render(self) -> str:
        ctx = self.job_ctx
        snap = self.snapshot
        if ctx is None or snap is None:
            return ""
        ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode

        if snap.node_count > 1:
            node = f"{snap.hostname} (node {snap.node_index + 1} of {snap.node_count})"
        else:
            node = ctx.nodelist or snap.hostname
        # Each identity value gets its own hue (dim labels, coloured values) so the
        # bottom bar reads as a lively strip rather than a flat grey line. These
        # are chrome, well below the resource rows, so reusing the palette here
        # ties the UI together without being mistaken for a CPU/MEM/GPU reading.
        ident = (
            f"[{_DIM}]job[/] [{_ACCENT}]{snap.job_id}[/]{_SEP}"
            f"[{_DIM}]user[/] [{_CPU_COLOR}]{ctx.username or '?'}[/]{_SEP}"
            f"[{_DIM}]partition[/] [{_GPU_COLOR}]{ctx.partition or '?'}[/]{_SEP}"
            f"[{_DIM}]node[/] [{_MEM_COLOR}]{node}[/]"
        )

        elapsed = snap.elapsed_seconds
        limit = ctx.time_limit_seconds
        if limit and limit > 0:
            frac = min(100.0, elapsed / limit * 100.0)
            remaining = max(0, limit - elapsed)
            el, rem, lim = (
                _format_duration(elapsed),
                _format_duration(remaining),
                _format_duration(limit),
            )
            ends = time.strftime("%a %H:%M", time.localtime(time.time() + remaining))
            # Colour the bar and the "left" figure by how much time remains — a
            # job about to hit the wall glows red — so the colour is useful, not
            # just decorative: green with room, amber getting low, red near the end.
            frac_left = remaining / limit
            urg_level = "ok" if frac_left > 0.25 else "warn" if frac_left > 0.10 else "crit"
            urg = _HEALTH_COLOR[urg_level]
            # Size the progress bar to whatever width is left after the text, and
            # drop it entirely on a narrow terminal, so the line never wraps past
            # its two rows (the bar was a fixed 20 cells before, overflowing 80).
            text = f"ran {el}  {frac:.0f}%  ·  {rem} left of {lim} limit  ·  ends ~{ends}"
            inner = (self.size.width or 100) - 6  # #jobinfo padding 1 3
            # Leave >=2 cols of right margin so the line never touches the edge.
            bar_w = min(20, inner - len(text) - 3)
            bar = f"{_color_bar(frac, bar_w, ascii_mode, urg)} " if bar_w >= 6 else ""
            time_line = (
                f"[{_DIM}]ran[/] [{_INK}]{el}[/] {bar}[{_INK}]{frac:.0f}%[/]{_SEP}"
                f"[bold {urg}]{rem}[/] [{_DIM}]left of[/] [{_INK}]{lim}[/] [{_DIM}]limit[/]{_SEP}"
                f"[{_DIM}]ends ~[/][{_ACCENT}]{ends}[/]"
            )
        else:
            time_line = (
                f"[{_DIM}]ran[/] [{_INK}]{_format_duration(elapsed)}[/]{_SEP}"
                f"[{_DIM}]no wall-clock time limit[/]"
            )
        return f"{ident}\n{time_line}"


class ResourceDetailScreen(Screen[None]):
    """A real drill-in for one resource, replacing the old inert focus keys.

    Shows a full-width history graph and every number for the resource, plus a
    per-device table for GPUs. Reads the dashboard's live snapshot on a timer so
    it keeps updating while open.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back"),
        Binding("c", "switch('cpu')", "CPU"),
        Binding("m", "switch('mem')", "Memory"),
        Binding("g", "switch('gpu')", "GPU"),
    ]

    CSS = """
    ResourceDetailScreen { align: center middle; }
    #detail-box {
        width: 90%;
        max-width: 130;
        height: auto;
        max-height: 90%;
        border: round $primary;
        padding: 1 2;
    }
    #detail-title { text-style: bold; padding-bottom: 1; }
    #detail-body { height: auto; }
    #detail-chart { height: auto; padding-top: 1; }
    #detail-table { height: auto; margin-top: 1; }
    """

    def __init__(self, dashboard: DashboardScreen, resource: str) -> None:
        super().__init__()
        self._dashboard = dashboard
        self._resource = resource

    def compose(self) -> ComposeResult:
        with Vertical(id="detail-box"):
            yield Static(id="detail-title")
            yield Static(id="detail-body")
            # GPUs also get the per-device table above the chart; every resource
            # gets the tall braille trend line that fills the box (1fr) instead
            # of a single cramped sparkline sized to the wrong width (F2).
            if self._resource == "gpu":
                yield GpuTable(id="detail-table")
            yield Static(id="detail-chart")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(0.5, self._refresh)

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_switch(self, resource: str) -> None:
        if resource == self._resource:
            return
        self.app.pop_screen()
        self.app.push_screen(ResourceDetailScreen(self._dashboard, resource))

    # #detail-box is 90% of the screen (capped at 130) with a round border
    # (1 col/side) and horizontal padding of 2 (2 cols/side) = 6 cols of chrome.
    _BOX_CHROME = 6
    _BOX_MAX_W = 130

    def _chart_width(self, chart_widget: Static) -> int:
        # Size the chart to the widget's real content width, not a
        # screen-relative guess, so it never overflows the padded/bordered box
        # and wraps into broken fragments (F2). Before layout the widget size is
        # 0; fall back to the box geometry (never wider than the box) so even the
        # first pre-layout render can't overflow.
        w = chart_widget.size.width
        if w <= 0:
            w = min(int(self.size.width * 0.9), self._BOX_MAX_W) - self._BOX_CHROME
        return max(w, _SPARK_W)

    def _refresh(self) -> None:
        snap = self._dashboard.latest_snapshot
        cfg = self._dashboard.config
        ascii_mode = cfg.ascii_mode
        with contextlib.suppress(NoMatches):
            title = self.query_one("#detail-title", Static)
            body = self.query_one("#detail-body", Static)
            if snap is None:
                title.update("[dim]awaiting data…[/]")
                body.update("")
                return
            if self._resource == "cpu":
                self._refresh_cpu(title, body, snap, ascii_mode)
            elif self._resource == "mem":
                self._refresh_mem(title, body, snap, ascii_mode)
            else:
                self._refresh_gpu(title, body, snap, cfg)

    def _refresh_cpu(
        self, title: Static, body: Static, snap: TelemetrySnapshot, ascii_mode: bool
    ) -> None:
        title.update("CPU detail  ·  [dim]c/m/g to switch · esc to close[/]")
        cpu = snap.cpu
        util_bar = _color_bar(cpu.usage_percent, 24, ascii_mode, _CPU_COLOR)
        # One bar only: 'effective cores' is just usage% × allocation, so a second
        # bar would be identical every frame — show it as a plain figure instead.
        body.update(
            f"usage         {util_bar}  {cpu.usage_percent:.1f}%\n"
            f"effective     {cpu.effective_cores:.2f} / {cpu.cores_allocated} cores"
            f"  [dim](avg cores kept busy)[/]"
        )
        self._update_chart(self._dashboard.cpu_history, ascii_mode, "usage")

    def _refresh_mem(
        self, title: Static, body: Static, snap: TelemetrySnapshot, ascii_mode: bool
    ) -> None:
        title.update("Memory detail  ·  [dim]c/m/g to switch · esc to close[/]")
        mem = snap.memory
        ws = mem.working_set_bytes or mem.current_bytes
        if mem.limit_bytes > 0:
            limit = mem.limit_bytes

            def pct(v: int) -> float:
                return min(100.0, v / limit * 100.0)

            headroom = max(limit - ws, 0)
            body.update(
                f"working set   {_color_bar(pct(ws), 24, ascii_mode, _MEM_COLOR)}  "
                f"{_format_bytes(ws)}  ({_mem_ws_pct(mem):.1f}% of {_format_bytes(limit)})\n"
                f"peak          {_color_bar(pct(mem.peak_bytes), 24, ascii_mode, _MEM_COLOR)}  "
                f"{_format_bytes(mem.peak_bytes)}\n"
                f"cache         [dim]reclaimable[/] {_format_bytes(mem.cache_bytes)}  ·  "
                f"total used {_format_bytes(mem.current_bytes)}\n"
                f"headroom to the OOM line: [{_MEM_COLOR}]{_format_bytes(headroom)}[/]"
            )
        else:
            body.update(
                f"working set {_format_bytes(ws)} (no limit)\n"
                f"cache (reclaimable) {_format_bytes(mem.cache_bytes)}  ·  "
                f"peak {_format_bytes(mem.peak_bytes)}"
            )
        self._update_chart(self._dashboard.mem_history, ascii_mode, "working set %")

    def _refresh_gpu(
        self, title: Static, body: Static, snap: TelemetrySnapshot, cfg: SlurmwatchConfig
    ) -> None:
        title.update("GPU detail  ·  [dim]c/m/g to switch · esc to close[/]")
        if snap.gpus:
            active = sum(1 for g in snap.gpus if _gpu_is_active(g, cfg.gpu_idle_threshold))
            total = len(snap.gpus)
            body.update(
                f"{total} {'device' if total == 1 else 'devices'} · {active} active"
                f"  ·  [dim]JOB% / JOB VRAM = this job's share of each device[/]"
            )
            with contextlib.suppress(NoMatches):
                self.query_one("#detail-table", GpuTable).update_gpus(snap.gpus, cfg)
            # Utilization trend for the busiest device, sharing the dashboard's
            # per-GPU history so drilling into GPU also gets a live trend line.
            rows_widget = self._dashboard.resource_rows
            hottest = max(snap.gpus, key=lambda g: g.utilization_percent)
            hist = (
                rows_widget.gpu_history.get(hottest.index, deque())
                if rows_widget is not None
                else deque()
            )
            self._update_chart(hist, cfg.ascii_mode, f"GPU{hottest.index} compute %")
        elif snap.gpu_count_requested > 0:
            body.update(
                f"[dim]{_plural(snap.gpu_count_requested, 'GPU')} requested — live telemetry "
                "unavailable here; run on the compute node.[/]"
            )
            self._clear_chart()
        else:
            body.update("[dim]no GPUs requested by this job[/]")
            self._clear_chart()

    def _resource_color(self) -> str:
        return {"cpu": _CPU_COLOR, "mem": _MEM_COLOR, "gpu": _GPU_COLOR}.get(
            self._resource, _ACCENT
        )

    def _clear_chart(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#detail-chart", Static).update("")

    def _update_chart(self, history: deque[float], ascii_mode: bool, label: str) -> None:
        with contextlib.suppress(NoMatches):
            chart = self.query_one("#detail-chart", Static)
            width = self._chart_width(chart)
            color = self._resource_color()
            vals = list(history)
            cur = vals[-1] if vals else 0.0
            head = (
                f"[{_DIM}]{label} · last {self._dashboard.config.history_seconds}s[/] "
                f"[{color}]{cur:>3.0f}%[/]"
            )
            # A full-width one-row sparkline reads clearly; the drill-in's extra
            # value is the min/avg/max line and (for GPUs) the per-device table.
            spark = _render_sparkline(history, width, ascii_mode, stretch=True)
            if vals:
                stats = (
                    f"[{_DIM}]min {min(vals):>3.0f}%   avg {sum(vals) / len(vals):>3.0f}%"
                    f"   max {max(vals):>3.0f}%[/]"
                )
            else:
                stats = "[dim]no history yet[/]"
            chart.update(f"{head}\n[{color}]{spark}[/]\n{stats}")


class KeyFooter(Static):
    """A keybinding bar where each shortcut wears its target's colour.

    Textual's stock Footer paints every key the same accent; here the resource
    keys match their panels — c = CPU cyan, m = MEM rose, g = GPU violet — so the
    shortcut and the thing it opens are colour-linked, and quit stays coral.
    """

    def __init__(self, keys: list[tuple[str, str, str]], **kwargs: Any) -> None:
        # keys: (key, label, colour), all literal → markup-safe.
        super().__init__(**kwargs)
        self._keys = keys

    def render(self) -> str:
        caps = [
            f"[{_BG} on {color}] {key} [/] [{_DIM}]{label}[/]" for key, label, color in self._keys
        ]
        return "  ".join(caps)


class DashboardScreen(Screen[Any]):
    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
        Binding("c", "detail('cpu')", "CPU"),
        Binding("m", "detail('mem')", "Memory"),
        Binding("g", "detail('gpu')", "GPU"),
        Binding("up", "scroll_up", "Up", show=False),
        Binding("down", "scroll_down", "Down", show=False),
        Binding("pageup", "page_up", "PgUp", show=False),
        Binding("pagedown", "page_down", "PgDn", show=False),
    ]

    CSS = """
    DashboardScreen { background: $surface; }

    #banner {
        height: auto;
        min-height: 1;
        padding: 1 2 0 2;
    }

    #body {
        padding: 0 1;
        height: 1fr;
    }

    ResourceRows { height: auto; padding: 1 0; }

    GpuTable { height: auto; margin: 0 1; }

    EfficiencyPanel { height: auto; padding: 0 1; }

    HistoryPanel { height: auto; padding: 1 1 0 1; }

    #jobinfo {
        height: auto;
        padding: 1 3;
        border-top: solid $primary 30%;
        background: $panel;
    }

    #keybar {
        height: 1;
        padding: 0 3;
        background: $panel;
    }

    Rule { margin: 0 1; color: $primary 40%; }
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        job_ctx: JobContext,
        config: SlurmwatchConfig | None = None,
    ) -> None:
        super().__init__()
        self.collector = collector
        self.job_ctx = job_ctx
        self.config = config or SlurmwatchConfig()
        self.latest_snapshot: TelemetrySnapshot | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self.title = "slurmwatch"

    @property
    def resource_rows(self) -> ResourceRows | None:
        try:
            return self.query_one(ResourceRows)
        except NoMatches:
            return None

    @property
    def cpu_history(self) -> deque[float]:
        rows = self.resource_rows
        return rows.cpu_history if rows is not None else deque()

    @property
    def mem_history(self) -> deque[float]:
        rows = self.resource_rows
        return rows.mem_history if rows is not None else deque()

    def compose(self) -> ComposeResult:
        # Blank the header icon: Textual's default "⭘" is a decorative
        # command-palette button that means nothing in this keyboard-driven tool.
        yield Header(show_clock=False, icon=" ")
        yield StatusBanner(id="banner")
        with VerticalScroll(id="body"):
            yield ResourceRows()
            yield GpuTable()
            yield Rule()
            yield EfficiencyPanel()
            yield Rule(id="history-rule")
            yield HistoryPanel()
        yield JobInfoBar(id="jobinfo")
        yield KeyFooter(
            [
                ("q", "Quit", _ACCENT),
                ("c", "CPU", _CPU_COLOR),
                ("m", "Memory", _MEM_COLOR),
                ("g", "GPU", _GPU_COLOR),
            ],
            id="keybar",
        )

    def on_mount(self) -> None:
        self.query_one(GpuTable).display = False
        self._update_header(None)
        self._poll_task = asyncio.create_task(self._poll_loop())

    def on_unmount(self) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()

    async def _poll_loop(self) -> None:
        try:
            while True:
                try:
                    snapshot = await asyncio.wait_for(self.collector.next_snapshot(), timeout=0.3)
                    self.latest_snapshot = snapshot
                    self._update_widgets(snapshot)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
                except Exception:
                    # A transient failure in one update must not silently kill
                    # the whole poll task and freeze the UI (B-C7); log via the
                    # Textual app log and keep polling.
                    self.log.error("dashboard poll iteration failed", exc_info=True)
        except asyncio.CancelledError:
            pass

    def _history_maxlen(self) -> int:
        interval = max(self.config.poll_interval, 0.01)
        return max(int(round(self.config.history_seconds / interval)), 10)

    @staticmethod
    def _resize(hist: deque[float], maxlen: int) -> deque[float]:
        if hist.maxlen != maxlen:
            return deque(hist, maxlen=maxlen)
        return hist

    def _update_widgets(self, snapshot: TelemetrySnapshot) -> None:
        self.latest_snapshot = snapshot
        self._update_header(snapshot)
        maxlen = self._history_maxlen()

        with contextlib.suppress(NoMatches):
            banner = self.query_one(StatusBanner)
            banner.snapshot = snapshot
            banner.config = self.config
            # layout=True so the auto-height widget grows past its initial
            # single "connecting…" line when the content becomes multi-line.
            banner.refresh(layout=True)

        with contextlib.suppress(NoMatches):
            rows = self.query_one(ResourceRows)
            rows.snapshot = snapshot
            rows.config = self.config
            rows.cpu_history = self._resize(rows.cpu_history, maxlen)
            rows.cpu_history.append(snapshot.cpu.usage_percent)
            rows.mem_history = self._resize(rows.mem_history, maxlen)
            rows.mem_history.append(_mem_ws_pct(snapshot.memory))
            for gpu in snapshot.gpus:
                hist = rows.gpu_history.get(gpu.index)
                if hist is None:
                    hist = deque(maxlen=maxlen)
                    rows.gpu_history[gpu.index] = hist
                else:
                    rows.gpu_history[gpu.index] = self._resize(hist, maxlen)
                rows.gpu_history[gpu.index].append(gpu.utilization_percent)
            # 3+ GPUs go in the DataTable; 1-2 stay as scannable rows.
            use_table = len(snapshot.gpus) >= 3
            rows.gpu_table_active = use_table
            rows.refresh(layout=True)

        with contextlib.suppress(NoMatches):
            table = self.query_one(GpuTable)
            if len(snapshot.gpus) >= 3:
                table.display = True
                table.update_gpus(snapshot.gpus, self.config)
            else:
                table.display = False

        with contextlib.suppress(NoMatches):
            eff = self.query_one(EfficiencyPanel)
            eff.snapshot = snapshot
            eff.config = self.config
            eff.source = self._source_label()
            eff.refresh(layout=True)

        with contextlib.suppress(NoMatches):
            rows = self.query_one(ResourceRows)
            panel = self.query_one(HistoryPanel)
            panel.snapshot = snapshot
            panel.config = self.config
            # Share the row widget's history deques (single source of truth,
            # already resized/appended above) so the tall charts and the row
            # sparklines never disagree.
            panel.cpu_history = rows.cpu_history
            panel.mem_history = rows.mem_history
            panel.gpu_history = rows.gpu_history
            panel.frame += 1  # travel the steady-band ripple one step per poll
            panel.refresh(layout=True)

        with contextlib.suppress(NoMatches):
            info = self.query_one(JobInfoBar)
            info.snapshot = snapshot
            info.job_ctx = self.job_ctx
            info.config = self.config
            info.refresh(layout=True)

    def _source_label(self) -> str:
        ctx = self.job_ctx
        if getattr(self.collector, "_mock", False):
            return "demo data"
        if ctx.remote:
            return "sstat (remote)"
        if ctx.cgroup_v2_path:
            return "cgroup v2"
        if ctx.cgroup_v1_mem_path or ctx.cgroup_v1_cpu_path:
            return "cgroup v1"
        return "node-local"

    def _update_header(self, snapshot: TelemetrySnapshot | None) -> None:
        # The full, labelled job identity + time budget live in the JobInfoBar at
        # the bottom; the header just carries a short anchor so it isn't a cryptic
        # unlabelled string.
        if snapshot is None:
            self.sub_title = f"connecting to job {self.job_ctx.job_id}…"
            return
        self.sub_title = f"job {snapshot.job_id} · {self.job_ctx.username}"

    def action_quit(self) -> None:
        self.app.exit()

    def action_detail(self, resource: str) -> None:
        if self.latest_snapshot is not None:
            self.app.push_screen(ResourceDetailScreen(self, resource))

    def action_scroll_up(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#body").scroll_up(animate=False)

    def action_scroll_down(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#body").scroll_down(animate=False)

    def action_page_up(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#body").scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#body").scroll_page_down(animate=False)


class JobSelectorScreen(ModalScreen[str]):
    BINDINGS: ClassVar = [
        Binding("enter", "select_job", "Select"),
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]

    CSS = """
    JobSelectorScreen { align: center middle; }

    #selector-box {
        width: 70;
        height: auto;
        border: round $primary;
        padding: 1;
    }

    #selector-title { text-style: bold; padding-bottom: 1; }

    ListView { height: auto; max-height: 20; }
    ListItem { padding: 0 1; }
    ListItem:hover { background: $accent; }
    """

    def __init__(self, jobs: list[dict[str, object]]) -> None:
        super().__init__()
        self.jobs = jobs

    def compose(self) -> ComposeResult:
        with Vertical(id="selector-box"):
            yield Static(
                f"Select a running job ({len(self.jobs)} found):",
                id="selector-title",
            )
            yield ListView(*[ListItem(Static(self._job_line(j))) for j in self.jobs])

    @staticmethod
    def _job_line(j: dict[str, object]) -> str:
        # The job name (%j) is free-form and user-controlled (`sbatch -J`), so
        # every interpolated value must be neutralized before it reaches the
        # markup parser (F1); only the job-id's [bold] styling is our own markup.
        def field(key: str, default: str = "?") -> str:
            return _escape_markup(str(j.get(key, default)))

        return (
            f"[bold]{_escape_markup(str(j['job_id']))}[/]  "
            f"{field('partition')}  "
            f"{field('name')}  "
            f"nodes={field('nodes')}  "
            f"time={field('wall_time')}"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # ListView has focus, so its own enter binding fires (posting this
        # message) instead of the screen-level binding; mouse clicks arrive
        # here too.
        event.stop()
        self.action_select_job()

    def action_select_job(self) -> None:
        lv = self.query_one(ListView)
        if lv.index is not None and lv.index < len(self.jobs):
            self.dismiss(str(self.jobs[lv.index]["job_id"]))

    def action_cancel(self) -> None:
        self.dismiss("")


class SlurmwatchApp(App[Any]):
    TITLE = "slurmwatch"

    # slurmwatch is driven by its own explicit keys (q/c/m/g); the built-in
    # Ctrl+P command palette isn't part of the design, so disable it — that also
    # stops the header-icon corner from being a hidden click target.
    ENABLE_COMMAND_PALETTE = False

    SCREENS: ClassVar = {}

    CSS = """
    Screen { background: $surface; }

    /* The footer keybindings default to a flat, drab grey. Colour the key cap in
       the coral accent and the label in warm ink so the shortcuts read clearly. */
    Footer { background: $panel; }
    FooterKey { background: $panel; color: $foreground; }
    FooterKey .footer-key--key {
        color: $background;
        background: $primary;
        text-style: bold;
    }
    FooterKey .footer-key--description { color: $foreground; }
    FooterKey:hover { background: $primary 20%; }
    FooterKey:hover .footer-key--description { color: $primary; }
    """

    def __init__(
        self,
        job_ctx: JobContext | None = None,
        collector: Any = None,
        jobs: list[dict[str, object]] | None = None,
        config: SlurmwatchConfig | None = None,
    ) -> None:
        super().__init__()
        self._job_ctx = job_ctx
        self._collector = collector
        self._jobs = jobs
        self._config = config

    def on_mount(self) -> None:
        # The warm "Claude Code" theme. Guarded because register_theme/theme
        # only exist in Textual >= 0.86; older versions keep the default.
        with contextlib.suppress(Exception):
            self.register_theme(_CLAUDE_THEME)
            self.theme = "slurmwatch"
        # push_screen_wait (used by the selector path) requires a Textual worker
        # context; a plain asyncio task would die with NoActiveWorker.
        if self._collector is not None and self._job_ctx is not None:
            self._start_worker = self.run_worker(self._start_monitoring())
        else:
            self._start_worker = self.run_worker(self._run_job_selector())

    def on_unmount(self) -> None:
        if self._collector is not None:
            self._collector.stop_sync()

    async def _start_monitoring(self) -> None:
        assert self._collector is not None
        assert self._job_ctx is not None
        try:
            await self._collector.start()
        except Exception as exc:
            self.exit(message=f"Failed to start collector: {exc}", return_code=1)
            return
        config = self._config or getattr(self._collector, "config", None)
        await self.push_screen(DashboardScreen(self._collector, self._job_ctx, config))

    async def _run_job_selector(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            # resolve_current_jobs shells out to squeue; keep the blocking call
            # off the event loop so the app can still paint and take keys (B-C1).
            jobs = (
                self._jobs
                if self._jobs is not None
                else await loop.run_in_executor(None, resolve_current_jobs)
            )
        except Exception as exc:
            self.exit(message=f"Failed to query Slurm jobs: {exc}", return_code=1)
            return

        if not jobs:
            self.exit(message="No running Slurm jobs found.", return_code=0)
            return

        if len(jobs) == 1:
            job_id = str(jobs[0]["job_id"])
        else:
            result = await self.push_screen_wait(JobSelectorScreen(jobs))
            if not result:
                self.exit(message="No job selected.", return_code=0)
                return
            job_id = result

        try:
            # resolve_job_context runs scontrol + cgroup/uid lookups; also off
            # the event loop so a slow slurmctld can't freeze the UI (B-C1).
            self._job_ctx = await loop.run_in_executor(None, resolve_job_context, job_id)
        except Exception as exc:
            self.exit(message=str(exc), return_code=1)
            return

        self._collector = TelemetryCollector(self._job_ctx, self._config)
        try:
            await self._collector.start()
        except Exception as exc:
            self.exit(message=f"Failed to start collector: {exc}", return_code=1)
            return
        await self.push_screen(DashboardScreen(self._collector, self._job_ctx, self._config))
