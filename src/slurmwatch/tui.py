from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any, ClassVar

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
    if not values:
        return " " * length
    chars = "▁▂▃▄▅▆▇█" if not ascii_mode else "_.,-=+#%"
    max_val = 100.0
    vals = list(values)
    # Anchor sampling to the newest sample so the right edge is always current;
    # blank-pad on the left while history is still filling up.
    step = max(len(vals) / length, 1.0)
    cells: list[str] = []
    for i in range(length):
        offset = int((length - 1 - i) * step)
        idx = len(vals) - 1 - offset
        if idx < 0:
            cells.append(" ")
            continue
        level = int(min(vals[idx] / max_val, 1.0) * (len(chars) - 1))
        cells.append(chars[max(0, min(level, len(chars) - 1))])
    return "".join(cells)


def _spark(values: deque[float], length: int = _SPARK_W, ascii_mode: bool = False) -> str:
    body = _render_sparkline(values, length, ascii_mode)
    return f"[{_ACCENT}]{body}[/]"


def _pad(text: str, width: int) -> str:
    if len(text) > width:
        return text[:width]
    return text.ljust(width)


# ---------------------------------------------------------------------------
# Health: one function per resource, returning (level, one-word status).
# ---------------------------------------------------------------------------


def _cpu_ratio(cpu: CpuMetrics) -> float:
    if cpu.cores_allocated <= 0:
        return 0.0
    return cpu.effective_cores / cpu.cores_allocated


def _cpu_health(cpu: CpuMetrics) -> tuple[str, str]:
    if cpu.cores_allocated <= 0:
        return "none", "n/a"
    ratio = _cpu_ratio(cpu)
    if cpu.cores_allocated > 1 and ratio < 0.15:
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
    level, _ = _cpu_health(cpu)
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
        level, word = _cpu_health(cpu)
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
                    details = (
                        f"VRAM {_gib(gpu.memory_used_bytes):.0f}/"
                        f"{_gib(gpu.memory_total_bytes):.0f}G  "
                        f"{gpu.power_watts:.0f}W  {_row_temp(gpu.temperature_celsius, ascii_mode)}"
                    )
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
    """A device-per-row table, used when 3+ GPUs would scroll as stacked rows."""

    config: SlurmwatchConfig | None = None

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
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

        cpu = snap.cpu
        if cpu.cores_allocated > 0:
            ratio = _cpu_ratio(cpu)
            grade = "good" if ratio >= 0.5 else "ok" if ratio >= 0.15 else "underused"
            detail = f"{cpu.usage_percent:.0f}% of {cpu.cores_allocated} cores in use"
            if grade == "underused":
                detail += " — consider fewer --cpus-per-task"
            lines.append(f"  CPU   {grade:<11} {detail}")
        else:
            lines.append("  CPU   [dim]n/a[/]")

        mem = snap.memory
        if mem.limit_bytes > 0:
            ws_pct = _mem_ws_pct(mem)
            if mem.oom_guard_critical:
                detail = (
                    f"working set {ws_pct:.0f}% of {_gib(mem.limit_bytes):.0f} GiB "
                    "— lower --mem or the job risks being OOM-killed"
                )
                grade = "critical"
            elif mem.oom_guard_warning:
                detail = (
                    f"working set {ws_pct:.0f}% of {_gib(mem.limit_bytes):.0f} GiB "
                    "— approaching limit"
                )
                grade = "warning"
            else:
                detail = f"working set {ws_pct:.0f}% of {_gib(mem.limit_bytes):.0f} GiB"
                grade = "good"
            lines.append(f"  MEM   {grade:<11} {detail}")
        else:
            lines.append(f"  MEM   {'good':<11} {_format_bytes(mem.working_set_bytes)} (unlimited)")

        lines.append(self._gpu_line(snap, cfg))

        if self.source:
            lines.append(f"[dim]source: {self.source}[/]")
        return "\n".join(lines)

    @staticmethod
    def _gpu_line(snap: TelemetrySnapshot, cfg: SlurmwatchConfig) -> str:
        gpus = snap.gpus
        req = snap.gpu_count_requested
        if req > 0 and not gpus:
            return f"  GPU   [dim]{req} requested — telemetry unavailable here[/]"
        if not gpus:
            return "  GPU   [dim]none requested[/]"
        idle = sum(1 for g in gpus if not _gpu_is_active(g, cfg.gpu_idle_threshold))
        total = len(gpus)
        active = total - idle
        if idle == 0:
            return f"  GPU   {'good':<11} {active}/{total} active"
        if idle == total:
            return f"  GPU   {'idle':<11} all {total} GPU(s) unused — release the allocation"
        keep = max(active, 1)
        return (
            f"  GPU   {f'{idle} of {total} idle':<11} "
            f"drop to --gres=gpu:{keep} to free {idle} device(s)"
        )


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
        max-width: 120;
        height: auto;
        max-height: 90%;
        border: round $primary;
        padding: 1 2;
    }
    #detail-title { text-style: bold; padding-bottom: 1; }
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

    def _refresh(self) -> None:
        snap = self._dashboard.latest_snapshot
        rows = self._dashboard.resource_rows
        cfg = self._dashboard.config
        ascii_mode = cfg.ascii_mode
        with contextlib.suppress(NoMatches):
            title = self.query_one("#detail-title", Static)
            body = self.query_one("#detail-body", Static)
            if snap is None or rows is None:
                title.update("[dim]awaiting data…[/]")
                body.update("")
                return
            width = max(self.size.width - 12, _SPARK_W)
            if self._resource == "cpu":
                title.update("CPU detail")
                cpu = snap.cpu
                spark = _spark(rows.cpu_history, width, ascii_mode)
                body.update(
                    f"utilization {cpu.usage_percent:.1f}%\n"
                    f"effective {cpu.effective_cores:.2f} of "
                    f"{cpu.cores_allocated} cores allocated\n\n"
                    f"{spark}"
                )
            elif self._resource == "mem":
                title.update("Memory detail")
                mem = snap.memory
                ws = mem.working_set_bytes or mem.current_bytes
                head = f"working set {_format_bytes(ws)}" + (
                    f" ({_mem_ws_pct(mem):.1f}% of {_format_bytes(mem.limit_bytes)})"
                    if mem.limit_bytes > 0
                    else " (unlimited)"
                )
                spark = _spark(rows.mem_history, width, ascii_mode)
                body.update(
                    f"{head}\n"
                    f"cache (reclaimable) {_format_bytes(mem.cache_bytes)}\n"
                    f"total used {_format_bytes(mem.current_bytes)} · "
                    f"peak {_format_bytes(mem.peak_bytes)}\n\n"
                    f"{spark}"
                )
            else:
                title.update("GPU detail")
                body.update(f"{len(snap.gpus)} device(s)")
                with contextlib.suppress(NoMatches):
                    table = self.query_one("#detail-table", GpuTable)
                    table.update_gpus(snap.gpus, cfg)


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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield StatusBanner(id="banner")
        with VerticalScroll(id="body"):
            yield ResourceRows()
            yield GpuTable()
            yield Rule()
            yield EfficiencyPanel()
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
            yield ListView(
                *[
                    ListItem(
                        Static(
                            f"[bold]{j['job_id']}[/]  "
                            f"{j.get('partition', '?')}  "
                            f"{j.get('name', '?')}  "
                            f"nodes={j.get('nodes', '?')}  "
                            f"time={j.get('wall_time', '?')}"
                        )
                    )
                    for j in self.jobs
                ]
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
