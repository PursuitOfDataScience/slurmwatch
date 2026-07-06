from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any, ClassVar

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
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


# A non-breaking space renders like a normal space in the terminal but is not
# collapsed when the TUI is captured to SVG for the README demo, so separators
# that sit right next to Rich markup keep their intended gap.
_NBSP = "\N{NO-BREAK SPACE}"
_SEP = f"[#8a90a6]{_NBSP}·{_NBSP}[/]"

# The single accent used for every bar and sparkline: length is the only signal
# a bar carries. Health lives *only* in the status glyphs and the banner.
_ACCENT = "cyan"

# One health vocabulary, everywhere: green = fine, yellow = warning/underused,
# red = critical/idle. Nothing else uses these colors.
_HEALTH_COLOR = {"ok": "green", "warn": "yellow", "crit": "red", "none": "dim"}
_HEALTH_GLYPH = {"ok": "●", "warn": "▲", "crit": "✖", "none": "·"}
_HEALTH_GLYPH_ASCII = {"ok": "+", "warn": "!", "crit": "x", "none": "-"}

# GPU temperature threshold: above this a device is thermally stressed.
_TEMP_HOT_C = 83.0

_BAR_W = 18
_SPARK_W = 12
_DETAIL_W = 34
# Below this width the sparkline column is dropped so the essentials still fit
# an 80-column SSH terminal.
_NARROW_COLS = 100


def _glyph(level: str, ascii_mode: bool) -> str:
    table = _HEALTH_GLYPH_ASCII if ascii_mode else _HEALTH_GLYPH
    return table.get(level, table["none"])


def _dot(level: str, ascii_mode: bool) -> str:
    return f"[{_HEALTH_COLOR[level]}]{_glyph(level, ascii_mode)}[/]"


def _color_bar(percent: float, length: int = _BAR_W, ascii_mode: bool = False) -> str:
    """A magnitude bar: filled portion in the accent color, empty portion dim.

    ``percent`` is clamped to [0, 100] so an over-limit value can't overflow the
    bar's width. Color never means "good"/"bad" here — only the fill *length*
    carries information.
    """
    filled = max(0, min(length, int(percent / 100 * length)))
    fill_ch, empty_ch = ("#", "-") if ascii_mode else ("█", "░")
    parts = []
    if filled:
        parts.append(f"[{_ACCENT}]{fill_ch * filled}[/]")
    if length - filled:
        parts.append(f"[dim]{empty_ch * (length - filled)}[/]")
    return "".join(parts)


def _render_sparkline(
    values: deque[float], length: int = _SPARK_W, ascii_mode: bool = False
) -> str:
    # Anchor to the newest sample (right edge), blank-padding the left while
    # history fills — the shared sampler that _area_chart's stretch variant sits
    # beside.
    chars = "▁▂▃▄▅▆▇█" if not ascii_mode else "_.,-=+#%"
    cells: list[str] = []
    for v in _sample_columns(values, length):
        if v is None:
            cells.append(" ")
            continue
        level = int(min(v / 100.0, 1.0) * (len(chars) - 1))
        cells.append(chars[max(0, min(level, len(chars) - 1))])
    return "".join(cells)


def _spark(values: deque[float], length: int = _SPARK_W, ascii_mode: bool = False) -> str:
    body = _render_sparkline(values, length, ascii_mode)
    return f"[{_ACCENT}]{body}[/]"


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

    Unlike :func:`_sample_columns`, this fills the whole width even before
    history is full, so the tall area chart never shows an awkward blank left
    margin while it fills up — the oldest sample sits at the left edge, the
    newest at the right.
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return [None] * width
    if n == 1 or width == 1:
        return [vals[-1]] * width
    return [vals[round(i / (width - 1) * (n - 1))] for i in range(width)]


def _area_chart(
    values: deque[float], width: int, height: int, ascii_mode: bool = False
) -> list[str]:
    """Render history as a filled area chart: ``height`` rows of ``width`` cells.

    Each column's value (a percentage in [0, 100]) is drawn as a vertical bar
    using the eight sub-cell block levels for smooth height resolution, so a tall
    panel shows the trend at far higher fidelity than a one-row sparkline. Empty
    cells are blank; the caller wraps the result in the single accent color.
    """
    blocks = " ▁▂▃▄▅▆▇█" if not ascii_mode else " ...:-=+#"
    height = max(height, 1)
    cols = _stretch_columns(values, width)
    grid = [[" "] * width for _ in range(height)]
    for col, v in enumerate(cols):
        if v is None:
            continue
        sub = int(round(min(max(v, 0.0), 100.0) / 100.0 * height * 8))
        for r in range(height):
            row_from_bottom = height - 1 - r
            units = min(8, max(0, sub - row_from_bottom * 8))
            grid[r][col] = blocks[units]
    return ["".join(row) for row in grid]


def _pad(text: str, width: int) -> str:
    if len(text) > width:
        return text[:width]
    return text.ljust(width)


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
        warn.append(("warn", f"CPU UNDERUSED — {cpu.effective_cores:.1f}/{cpu.cores_allocated}"))

    return crit + warn


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
            parts = []
            for level, text in segments:
                parts.append(f"{_dot(level, ascii_mode)} [bold {_HEALTH_COLOR[level]}]{text}[/]")
            line = f"{_NBSP}{_NBSP}{_NBSP}".join(parts)
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

    def _row(
        self,
        label: str,
        level: str,
        percent: float,
        details: str,
        history: deque[float] | None,
        status: str,
        wide: bool,
        ascii_mode: bool,
    ) -> str:
        # Narrower bar + details when the sparkline is dropped, so the row
        # (dot · bar · % · numbers · status) still fits an 80-column terminal.
        bar_w = _BAR_W if wide else 12
        det_w = _DETAIL_W if wide else 26
        dot = _dot(level, ascii_mode)
        bar = _color_bar(percent, bar_w, ascii_mode)
        pct = f"{percent:.0f}%".rjust(4)
        det = _pad(details, det_w)
        status_txt = f"[{_HEALTH_COLOR[level]}]{status}[/]"
        spark = ""
        if wide and history is not None:
            spark = f"{_spark(history, _SPARK_W, ascii_mode)}  "
        return f"  {label:<5} {dot}  {bar}  {pct}  {det}  {spark}{status_txt}"

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]awaiting telemetry…[/]"
        snap = self.snapshot
        cfg = self.config or SlurmwatchConfig()
        ascii_mode = cfg.ascii_mode
        wide = self.size.width >= _NARROW_COLS or self.size.width == 0

        lines: list[str] = []

        cpu = snap.cpu
        level, word = _cpu_health(cpu, cfg.cpu_underuse_threshold)
        cpu_details = f"{cpu.effective_cores:.1f} / {cpu.cores_allocated} cores"
        lines.append(
            self._row(
                "CPU",
                level,
                cpu.usage_percent,
                cpu_details,
                self.cpu_history,
                word,
                wide,
                ascii_mode,
            )
        )

        mem = snap.memory
        level, word = _mem_health(mem)
        ws = mem.working_set_bytes or mem.current_bytes
        if mem.limit_bytes > 0:
            mem_pct = _mem_ws_pct(mem)
            mem_details = (
                f"{_gib(ws):.0f} / {_gib(mem.limit_bytes):.0f} GiB "
                f"· peak {_gib(mem.peak_bytes):.0f}"
            )
        else:
            mem_pct = 0.0
            mem_details = f"{_format_bytes(ws)} (unlimited)"
        lines.append(
            self._row("MEM", level, mem_pct, mem_details, self.mem_history, word, wide, ascii_mode)
        )

        gpus = snap.gpus
        if not self.gpu_table_active:
            if gpus:
                for gpu in gpus:
                    level, word = _gpu_health(gpu, cfg.gpu_idle_threshold)
                    # details must stay plain text (no markup): it is padded to a
                    # fixed width, and padding a markup string would truncate a
                    # tag mid-way and corrupt the render. The threshold marker is
                    # a plain "!" so hot temp is visible without color here; the
                    # amber colouring lives in the GPU table / detail view.
                    # Fixed sub-column widths so power/temp line up across GPU
                    # rows regardless of VRAM magnitude, instead of drifting with
                    # the value width (U6).
                    used_g, tot_g = _gib(gpu.memory_used_bytes), _gib(gpu.memory_total_bytes)
                    vram = f"{used_g:.0f}/{tot_g:.0f}G"
                    pwr = f"{gpu.power_watts:>4.0f}W"
                    temp = _row_temp(gpu.temperature_celsius, ascii_mode)
                    details = f"VRAM {vram:>7}  {pwr}  {temp:>5}"
                    lines.append(
                        self._row(
                            f"GPU{gpu.index}",
                            level,
                            gpu.utilization_percent,
                            details,
                            self.gpu_history.get(gpu.index),
                            word,
                            wide,
                            ascii_mode,
                        )
                    )
            elif snap.gpu_count_requested > 0:
                lines.append(
                    f"  [dim]GPU   {snap.gpu_count_requested} requested — "
                    "telemetry unavailable here (run on the compute node)[/]"
                )
            else:
                lines.append("  [dim]GPU   none requested[/]")
        return "\n".join(lines)


def _row_temp(temp_c: float, ascii_mode: bool) -> str:
    """Plain (markup-free) temperature for a padded row; '!' once thermally hot."""
    deg = "C" if ascii_mode else "°C"
    hot = "!" if temp_c >= _TEMP_HOT_C else ""
    return f"{temp_c:.0f}{deg}{hot}"


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
        self.zebra_stripes = True
        if self._detailed:
            self.add_columns("GPU", "UTIL", "VRAM", "JOB%", "JOB VRAM", "PWR", "TEMP", "STATUS")
        else:
            self.add_columns("GPU", "UTIL", "VRAM", "PWR", "TEMP", "STATUS")

    def update_gpus(self, gpus: list[GpuMetrics], config: SlurmwatchConfig) -> None:
        ascii_mode = config.ascii_mode
        self.clear()
        for gpu in gpus:
            level, word = _gpu_health(gpu, config.gpu_idle_threshold)
            bar = _color_bar(gpu.utilization_percent, 8, ascii_mode)
            util = Text.from_markup(f"{gpu.utilization_percent:>3.0f}% {bar}")
            vram = f"{_gib(gpu.memory_used_bytes):.0f}/{_gib(gpu.memory_total_bytes):.0f} GiB"
            pwr = f"{gpu.power_watts:.0f}W"
            hot = gpu.temperature_celsius >= _TEMP_HOT_C
            deg = "C" if ascii_mode else "°C"
            temp_mark = ("!" if ascii_mode else "⚠") if hot else ""
            temp = Text(
                f"{temp_mark}{gpu.temperature_celsius:.0f}{deg}", style="yellow" if hot else ""
            )
            status = Text(f"{_glyph(level, ascii_mode)} {word}", style=_HEALTH_COLOR[level])
            if self._detailed:
                job_util = f"{gpu.process_utilization_percent:>3.0f}%"
                job_vram = (
                    f"{_gib(gpu.process_memory_bytes):.1f} GiB" if gpu.process_memory_bytes else "—"
                )
                self.add_row(str(gpu.index), util, vram, job_util, job_vram, pwr, temp, status)
            else:
                self.add_row(str(gpu.index), util, vram, pwr, temp, status)


class EfficiencyPanel(Static):
    """The de-duplicated verdict: one place for the actionable recommendation."""

    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None
    source: str = ""

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]Allocation efficiency: awaiting data…[/]"
        snap = self.snapshot
        cfg = self.config or SlurmwatchConfig()
        lines: list[str] = ["[bold]Allocation efficiency[/]"]
        # One grade vocabulary (good / underused / warning / critical) in an
        # aligned column, then the *actionable* advice. The advice deliberately
        # avoids re-printing the headline % the banner and rows already show
        # (U3); it names concrete figures (cores, GiB, GPU indices) instead.
        for res, (grade, advice) in (
            ("CPU", self._cpu_verdict(snap.cpu, cfg)),
            ("MEM", self._mem_verdict(snap.memory)),
            ("GPU", self._gpu_verdict(snap, cfg)),
        ):
            lines.append(f"  {res:<4} {grade:<10} {advice}")
        if self.source:
            lines.append(f"[dim]source: {self.source}[/]")
        return "\n".join(lines)

    @staticmethod
    def _cpu_verdict(cpu: CpuMetrics, cfg: SlurmwatchConfig) -> tuple[str, str]:
        if cpu.cores_allocated <= 0:
            return "n/a", "[dim]no CPU allocation reported[/]"
        used, tot = cpu.effective_cores, cpu.cores_allocated
        if cpu.cores_allocated > 1 and _cpu_ratio(cpu) < cfg.cpu_underuse_threshold:
            return "underused", f"using {used:.1f} of {tot} cores — request fewer --cpus-per-task"
        return "good", f"using {used:.1f} of {tot} cores"

    @staticmethod
    def _mem_verdict(mem: MemoryMetrics) -> tuple[str, str]:
        if mem.limit_bytes <= 0:
            return "good", f"{_format_bytes(mem.working_set_bytes)} working set (no limit)"
        limit = _gib(mem.limit_bytes)
        if mem.oom_guard_critical:
            return "critical", f"at the {limit:.0f} GiB ceiling — raise --mem or risk an OOM kill"
        if mem.oom_guard_warning:
            return "warning", f"approaching the {limit:.0f} GiB limit — raise --mem or trim usage"
        ws = _gib(mem.working_set_bytes or mem.current_bytes)
        return "good", f"{ws:.0f} of {limit:.0f} GiB working set — comfortable headroom"

    @staticmethod
    def _gpu_verdict(snap: TelemetrySnapshot, cfg: SlurmwatchConfig) -> tuple[str, str]:
        gpus = snap.gpus
        req = snap.gpu_count_requested
        if req > 0 and not gpus:
            return "n/a", "[dim]telemetry unavailable here — run on the compute node[/]"
        if not gpus:
            return "n/a", "[dim]no GPUs requested[/]"
        idle_idx = [g.index for g in gpus if not _gpu_is_active(g, cfg.gpu_idle_threshold)]
        total, idle = len(gpus), len(idle_idx)
        active = total - idle
        unit = "GPU" if total == 1 else "GPUs"
        if idle == 0:
            return "good", f"all {total} {unit} busy"
        if idle == total:
            return "critical", f"all {total} {unit} idle — release the allocation"
        return (
            "underused",
            f"{_list_gpus(idle_idx)} idle — drop to --gres=gpu:{max(active, 1)} "
            f"to free {_plural(idle, 'device')}",
        )


class HistoryPanel(Static):
    """Fills the dashboard's lower half with a tall per-resource history chart.

    Each resource row carries a one-row sparkline for a glance; this panel uses
    the otherwise-blank space below the fold for a high-resolution area chart of
    the same series, so a trend (memory climbing toward the limit, a GPU that
    just went idle) is legible instead of the screen sitting mostly empty (U2).
    It sizes itself to whatever height the layout gives it.
    """

    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None
    cpu_history: deque[float] | None = None
    mem_history: deque[float] | None = None
    gpu_history: dict[int, deque[float]] | None = None

    def render(self) -> str:
        if self.snapshot is None or self.cpu_history is None:
            return ""
        cfg = self.config or SlurmwatchConfig()
        ascii_mode = cfg.ascii_mode
        w, h = self.size.width, self.size.height
        if w < 8 or h < 4:
            # Too small to add anything the row sparklines don't already show.
            return ""

        snap = self.snapshot
        # CPU and memory always; add the busiest GPU (not an average, which would
        # hide a single hot or idle device) when GPU history exists.
        series: list[tuple[str, deque[float], float]] = [
            ("CPU", self.cpu_history or deque(), snap.cpu.usage_percent),
            ("MEM", self.mem_history or deque(), _mem_ws_pct(snap.memory)),
        ]
        gpu_hist = self.gpu_history or {}
        if snap.gpus and gpu_hist:
            hottest = max(snap.gpus, key=lambda g: g.utilization_percent)
            hh = gpu_hist.get(hottest.index)
            if hh is not None:
                series.append((f"GPU{hottest.index}", hh, hottest.utilization_percent))

        # Fit as many series as the height allows (each needs a label + >=1 chart
        # row); drop the lowest-priority extras first, so the total never
        # overflows the panel and scrolls.
        n = max(1, min(len(series), h // 2))
        series = series[:n]
        chart_h = max((h - n) // n, 1)

        lines: list[str] = []
        for label, hist, cur in series:
            lines.append(
                f"[dim]{label} history[/] "
                f"[{_ACCENT}]{cur:>3.0f}%[/] "
                f"[dim]· last {cfg.history_seconds}s[/]"
            )
            chart = _area_chart(hist, w, chart_h, ascii_mode)
            lines.extend(f"[{_ACCENT}]{row}[/]" for row in chart)
        return "\n".join(lines)


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
        height: 90%;
        border: round $primary;
        padding: 1 2;
    }
    #detail-title { text-style: bold; padding-bottom: 1; }
    #detail-body { height: auto; }
    #detail-chart { height: 1fr; min-height: 0; padding-top: 1; }
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
            if self._resource == "gpu":
                yield GpuTable(id="detail-table")
            else:
                # A tall history area chart that fills the box (1fr) instead of a
                # single cramped sparkline sized to the wrong width (F2).
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
        cores = cpu.cores_allocated or 1
        util_bar = _color_bar(cpu.usage_percent, 24, ascii_mode)
        core_pct = min(100.0, cpu.effective_cores / cores * 100.0) if cores else 0.0
        core_bar = _color_bar(core_pct, 24, ascii_mode)
        body.update(
            f"utilization   {util_bar}  {cpu.usage_percent:.1f}%\n"
            f"effective     {core_bar}  {cpu.effective_cores:.2f} / "
            f"{cpu.cores_allocated} cores"
        )
        self._update_chart(self._dashboard.cpu_history, ascii_mode, "utilization")

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
                f"working set   {_color_bar(pct(ws), 24, ascii_mode)}  "
                f"{_format_bytes(ws)}  ({_mem_ws_pct(mem):.1f}% of {_format_bytes(limit)})\n"
                f"peak          {_color_bar(pct(mem.peak_bytes), 24, ascii_mode)}  "
                f"{_format_bytes(mem.peak_bytes)}\n"
                f"cache         [dim]reclaimable[/] {_format_bytes(mem.cache_bytes)}  ·  "
                f"total used {_format_bytes(mem.current_bytes)}\n"
                f"headroom to the OOM line: [{_ACCENT}]{_format_bytes(headroom)}[/]"
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
        elif snap.gpu_count_requested > 0:
            body.update(
                f"[dim]{_plural(snap.gpu_count_requested, 'GPU')} requested — live telemetry "
                "unavailable here; run on the compute node.[/]"
            )
        else:
            body.update("[dim]no GPUs requested by this job[/]")

    def _update_chart(self, history: deque[float], ascii_mode: bool, label: str) -> None:
        with contextlib.suppress(NoMatches):
            chart = self.query_one("#detail-chart", Static)
            width = self._chart_width(chart)
            height = max(chart.size.height - 1, 6)
            cur = history[-1] if history else 0.0
            head = (
                f"[dim]{label} · last {self._dashboard.config.history_seconds}s[/] "
                f"[{_ACCENT}]{cur:>3.0f}%[/]"
            )
            rows = _area_chart(history, width, height, ascii_mode)
            chart.update(head + "\n" + "\n".join(f"[{_ACCENT}]{r}[/]" for r in rows))


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

    HistoryPanel { height: 1fr; min-height: 0; padding: 1 1 0 1; }

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
        yield Header(show_clock=False)
        yield StatusBanner(id="banner")
        with VerticalScroll(id="body"):
            yield ResourceRows()
            yield GpuTable()
            yield Rule()
            yield EfficiencyPanel()
            yield Rule(id="history-rule")
            yield HistoryPanel()
        yield Footer()

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
            panel.refresh(layout=True)

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
        if snapshot is None:
            self.sub_title = f"connecting to job {self.job_ctx.job_id}…"
            return
        parts = [
            f"job {snapshot.job_id}",
            self.job_ctx.username,
            self.job_ctx.partition,
        ]
        if snapshot.node_count > 1:
            parts.append(
                f"{snapshot.hostname} (node {snapshot.node_index + 1}/{snapshot.node_count})"
            )
        else:
            parts.append(self.job_ctx.nodelist or snapshot.hostname)
        parts.append(_format_duration(snapshot.elapsed_seconds))
        self.sub_title = " · ".join(parts)

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
        # The job name (%j) is free-form and user-controlled (`sbatch -J`), so a
        # bracketed name like "sweep[3]" or "[red]x" must be escaped before it
        # reaches Rich's markup parser — otherwise it is silently dropped, or an
        # unbalanced tag ("run[/]done") raises MarkupError and crashes the whole
        # selector screen (F1). Only the job-id styling is our own markup.
        def field(key: str, default: str = "?") -> str:
            return escape(str(j.get(key, default)))

        return (
            f"[bold]{escape(str(j['job_id']))}[/]  "
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

    SCREENS: ClassVar = {}

    CSS = """
    Screen { background: $surface; }
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
        # A modern built-in theme (matches the README demo). Guarded because
        # themes only exist in Textual >= 0.86; older versions keep the default.
        with contextlib.suppress(Exception):
            if "tokyo-night" in self.available_themes:
                self.theme = "tokyo-night"
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
