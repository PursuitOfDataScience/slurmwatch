from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.widgets import ListItem, ListView, Static

from .collector import TelemetryCollector
from .config import SlurmwatchConfig
from .model import JobContext, TelemetrySnapshot
from .slurm import resolve_current_jobs, resolve_job_context


def _format_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def _format_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


_CHAR_BAR = "█"
_CHAR_EMPTY = "░"
_CHAR_WARN = "!"
_CHAR_CRIT = "X"
_CHAR_THROTTLE = "!"
_CHAR_BULLET = "*"
_CHAR_PIPE = "|"
_CHAR_DASH = "-"


def _render_bar(percent: float, length: int = 12, ascii_mode: bool = False) -> str:
    filled = max(0, min(length, int(percent / 100 * length)))
    if ascii_mode:
        return "#" * filled + "-" * (length - filled)
    return "█" * filled + "░" * (length - filled)


def _render_sparkline(values: deque[float], length: int = 10, ascii_mode: bool = False) -> str:
    if not values:
        return " " * length
    chars = "▁▂▃▄▅▆▇█" if not ascii_mode else "_.,-=+#%"
    max_val = 100.0
    vals = list(values)
    # Anchor sampling to the newest sample so the right edge is always
    # current; blank-pad on the left while history is still filling up.
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


class CpuPanel(Static):
    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: deque[float] = deque(maxlen=60)

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]CPU: awaiting data...[/]"
        cpu = self.snapshot.cpu
        cfg = self.config
        ascii_mode = cfg.ascii_mode if cfg else False
        bar = _render_bar(cpu.usage_percent, 16, ascii_mode)
        spark = _render_sparkline(self.history, 16, ascii_mode)
        verdict = ""
        underuse = cfg.cpu_underuse_threshold if cfg else 0.5
        if cpu.cores_allocated > 1 and cpu.effective_cores < underuse:
            verdict = (
                f"\n      [!] ~{cpu.effective_cores:.1f} core used of {cpu.cores_allocated} "
                f"\u2014 consider --cpus-per-task=2"
            )

        return (
            f"[bold]CPU[/]  {bar}  {cpu.usage_percent:.1f}%\n"
            f"      {cpu.cores_allocated} cores allocated, "
            f"~{cpu.effective_cores:.1f} effective\n"
            f"      {spark}{verdict}"
        )


class MemoryPanel(Static):
    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: deque[float] = deque(maxlen=60)

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]Memory: awaiting data...[/]"
        mem = self.snapshot.memory
        ascii_mode = self.config.ascii_mode if self.config else False
        pct = mem.usage_percent
        bar = _render_bar(pct, 16, ascii_mode)
        spark = _render_sparkline(self.history, 16, ascii_mode)
        show_pct = mem.limit_bytes > 0

        if mem.oom_guard_critical:
            style = "[bold red]"
            close = "[/]"
            guard = f" {_CHAR_CRIT} CRITICAL"
        elif mem.oom_guard_warning:
            style = "[bold yellow]"
            close = "[/]"
            guard = f" {_CHAR_WARN} WARNING"
        else:
            style = ""
            close = ""
            guard = ""

        used = _format_bytes(mem.current_bytes)
        working = _format_bytes(mem.working_set_bytes)
        limit = _format_bytes(mem.limit_bytes) if show_pct else "(unlimited)"
        peak = _format_bytes(mem.peak_bytes)

        pct_str = f"{pct:.1f}%" if show_pct else "N/A"
        ws_pct = ""
        if show_pct and mem.limit_bytes > 0:
            ws_pct_val = (mem.working_set_bytes / mem.limit_bytes) * 100.0
            ws_pct = f" (ws: {ws_pct_val:.1f}%)"

        return (
            f"{style}[bold]MEMORY[/]  {bar}  {pct_str}{ws_pct}{guard}{close}\n"
            f"      {used} / {limit}  peak: {peak}\n"
            f"      working set: {working}\n"
            f"      {spark}"
        )


class GpuPanel(Static):
    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: dict[int, deque[float]] = {}

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]GPU: awaiting data...[/]"
        gpus = self.snapshot.gpus
        if not gpus:
            return "[dim]GPU: no GPUs detected[/]"
        ascii_mode = self.config.ascii_mode if self.config else False

        lines: list[str] = []
        for gpu in gpus:
            util_bar = _render_bar(gpu.utilization_percent, 12, ascii_mode)
            mem_bar = _render_bar(gpu.memory_utilization_percent, 12, ascii_mode)
            mem_used = _format_bytes(gpu.memory_used_bytes)
            mem_total = _format_bytes(gpu.memory_total_bytes)
            proc_used = _format_bytes(gpu.process_memory_bytes)
            throttle = f" {_CHAR_THROTTLE}" if gpu.throttling else ""

            spark = ""
            if gpu.index in self.history:
                spark = _render_sparkline(self.history[gpu.index], 12, ascii_mode)

            proc_util_str = ""
            if gpu.process_utilization_percent > 0:
                proc_util_str = f"proc: {gpu.process_utilization_percent:.1f}%"

            idle_note = ""
            idle_pct = self.config.gpu_idle_threshold if self.config else 5.0
            proc_idle = idle_pct * 0.4 if self.config else 2.0
            if gpu.utilization_percent < idle_pct and gpu.process_utilization_percent < proc_idle:
                idle_note = " [dim]IDLE[/]"

            lines.append(
                f"[bold]GPU {gpu.index}: {gpu.name}[/]{throttle}{idle_note}\n"
                f"  Util:  {util_bar}  {gpu.utilization_percent:.1f}%  {proc_util_str}\n"
                f"  VRAM:  {mem_bar}  {gpu.memory_utilization_percent:.1f}%"
                f" ({mem_used}/{mem_total}) \u2014 job: {proc_used}\n"
                f"  Power: {gpu.power_watts:.0f}W  Temp: {gpu.temperature_celsius:.0f}C\n"
                f"  {spark}"
            )
        req = self.snapshot.gpu_count_requested
        active = self.snapshot.gpu_active_count
        idle_count = len(gpus) - active
        gpu_verdict = ""
        if req > 0 and idle_count > 0:
            gpu_verdict = f"\n  [bold]{active}/{req}[/] GPUs active, {idle_count} idle"
        elif req > 0:
            gpu_verdict = f"\n  [bold]{active}/{req}[/] GPUs active"

        return "\n\n".join(lines) + gpu_verdict


class VerdictPanel(Static):
    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]Allocation Efficiency: awaiting data...[/]"
        snap = self.snapshot
        lines: list[str] = ["[bold]Allocation Efficiency[/]"]

        cpu = snap.cpu
        eff = cpu.effective_cores
        cores = cpu.cores_allocated
        if cores > 0:
            ratio = eff / cores
            if ratio < 0.15:
                cpu_v = f"UNDERUSED \u2014 {eff:.1f}/{cores} cores"
            elif ratio < 0.5:
                cpu_v = f"OK \u2014 {eff:.1f}/{cores} cores"
            else:
                cpu_v = f"GOOD \u2014 {eff:.1f}/{cores} cores"
        else:
            cpu_v = "N/A"
        lines.append(f"  CPU     {cpu_v}")

        mem = snap.memory
        if mem.limit_bytes > 0:
            ws_pct = (mem.working_set_bytes / mem.limit_bytes) * 100.0
            if mem.oom_guard_critical:
                mem_v = "CRITICAL \u2014 near limit"
            elif mem.oom_guard_warning:
                mem_v = "WARNING \u2014 approaching limit"
            else:
                mem_v = f"OK \u2014 {ws_pct:.0f}% of limit"
        else:
            used = _format_bytes(mem.working_set_bytes)
            mem_v = f"{used} (unlimited)"
        lines.append(f"  Memory  {mem_v}")

        gpus = snap.gpus
        req = snap.gpu_count_requested
        active = snap.gpu_active_count
        idle_count = len(gpus) - active
        if req > 0:
            if idle_count == len(gpus):
                gpu_v = f"IDLE \u2014 all {len(gpus)} GPU(s) idle"
            elif idle_count > 0:
                gpu_v = f"UNDERUSED \u2014 {active}/{req} active, {idle_count} idle"
            else:
                gpu_v = f"GOOD \u2014 {active}/{req} GPUs active"
        elif gpus:
            gpu_v = f"{len(gpus)} GPU(s) detected (requested: unknown)"
        else:
            gpu_v = "N/A"
        lines.append(f"  GPU     {gpu_v}")

        return "\n".join(lines)


class DashboardScreen(Screen[Any]):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("m", "focus_memory", "Memory"),
        Binding("c", "focus_cpu", "CPU"),
        Binding("g", "focus_gpu", "GPU"),
        Binding("v", "focus_verdict", "Verdict"),
        Binding("up", "scroll_line_up", "Up"),
        Binding("down", "scroll_line_down", "Down"),
        Binding("pageup", "scroll_page_up", "PgUp"),
        Binding("pagedown", "scroll_page_down", "PgDn"),
    ]

    CSS = """
    DashboardScreen {
        background: $surface;
    }

    #header {
        dock: top;
        height: 3;
        padding: 0 1;
        background: $primary;
        color: $text;
    }

    #grid-container {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        grid-rows: auto;
        padding: 0 1;
        height: 100%;
        overflow-y: auto;
    }

    .panel {
        border: solid $primary;
        padding: 0 1;
        min-height: 5;
    }

    .panel-focused {
        border: solid yellow;
        background: $boost;
    }

    #footer {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text-muted;
    }

    GpuPanel {
        column-span: 2;
    }

    VerdictPanel {
        column-span: 2;
    }
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
        self._snapshot: TelemetrySnapshot | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._poll_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Container(id="grid-container"):
            yield CpuPanel(id="cpu-panel", classes="panel")
            yield MemoryPanel(id="mem-panel", classes="panel")
            yield GpuPanel(id="gpu-panel", classes="panel")
            yield VerdictPanel(id="verdict-panel", classes="panel")
        yield Static(id="footer")

    def on_mount(self) -> None:
        self._update_header(None)
        self._update_footer()
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._tasks.append(self._poll_task)

    def on_unmount(self) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()

    async def _poll_loop(self) -> None:
        try:
            while True:
                try:
                    snapshot = await asyncio.wait_for(self.collector.next_snapshot(), timeout=0.3)
                    self._snapshot = snapshot
                    self._update_widgets(snapshot)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
        except asyncio.CancelledError:
            pass

    def _update_widgets(self, snapshot: TelemetrySnapshot) -> None:
        self._update_header(snapshot)
        # One sample arrives per poll_interval, so convert the configured
        # history duration (seconds) into a sample count.
        if self.config:
            interval = max(self.config.poll_interval, 0.01)
            hist_maxlen = max(int(round(self.config.history_seconds / interval)), 10)
        else:
            hist_maxlen = 60
        try:
            cpu_w = self.query_one("#cpu-panel", CpuPanel)
            cpu_w.snapshot = snapshot
            cpu_w.config = self.config
            if cpu_w.history.maxlen != hist_maxlen:
                cpu_w.history = deque(cpu_w.history, maxlen=hist_maxlen)
            cpu_w.history.append(snapshot.cpu.usage_percent)
            cpu_w.refresh()
        except NoMatches:
            pass
        try:
            mem_w = self.query_one("#mem-panel", MemoryPanel)
            mem_w.snapshot = snapshot
            mem_w.config = self.config
            if mem_w.history.maxlen != hist_maxlen:
                mem_w.history = deque(mem_w.history, maxlen=hist_maxlen)
            mem_w.history.append(snapshot.memory.usage_percent)
            mem_w.refresh()
        except NoMatches:
            pass
        try:
            gpu_w = self.query_one("#gpu-panel", GpuPanel)
            gpu_w.snapshot = snapshot
            gpu_w.config = self.config
            for gpu in snapshot.gpus:
                if gpu.index not in gpu_w.history:
                    gpu_w.history[gpu.index] = deque(maxlen=hist_maxlen)
                elif gpu_w.history[gpu.index].maxlen != hist_maxlen:
                    gpu_w.history[gpu.index] = deque(gpu_w.history[gpu.index], maxlen=hist_maxlen)
                gpu_w.history[gpu.index].append(gpu.utilization_percent)
            gpu_w.refresh()
        except NoMatches:
            pass
        try:
            verdict_w = self.query_one("#verdict-panel", VerdictPanel)
            verdict_w.snapshot = snapshot
            verdict_w.config = self.config
            verdict_w.refresh()
        except NoMatches:
            pass

    def _update_header(self, snapshot: TelemetrySnapshot | None) -> None:
        try:
            header = self.query_one("#header", Static)
            if snapshot:
                elapsed = _format_duration(snapshot.elapsed_seconds)
                node_info = ""
                if snapshot.node_count > 1:
                    hostname = snapshot.hostname
                    node_info = (
                        f" | Node [bold]{hostname}[/] "
                        f"({snapshot.node_index + 1} of {snapshot.node_count})"
                    )
                header.update(
                    f"[bold]slurmwatch[/] | "
                    f"Job [bold]{snapshot.job_id}[/] | "
                    f"User [bold]{self.job_ctx.username}[/] | "
                    f"Partition [bold]{self.job_ctx.partition}[/] | "
                    f"Nodes [bold]{self.job_ctx.nodelist}[/]{node_info} | "
                    f"Elapsed [bold]{elapsed}[/]"
                )
            else:
                msg = f"[bold]slurmwatch[/] connecting to job {self.job_ctx.job_id}..."
                header.update(msg)
        except NoMatches:
            pass

    def _update_footer(self) -> None:
        try:
            footer = self.query_one("#footer", Static)
            footer.update(" [c] CPU  [m] Memory  [g] GPU  [v] Verdict  [q] Quit")
        except NoMatches:
            pass

    def action_quit(self) -> None:
        self.app.exit()

    def action_focus_cpu(self) -> None:
        try:
            self.query_one("#cpu-panel", CpuPanel).classes = "panel panel-focused"
            self.query_one("#mem-panel", MemoryPanel).classes = "panel"
            self.query_one("#gpu-panel", GpuPanel).classes = "panel"
            self.query_one("#verdict-panel", VerdictPanel).classes = "panel"
        except NoMatches:
            pass

    def action_focus_memory(self) -> None:
        try:
            self.query_one("#mem-panel", MemoryPanel).classes = "panel panel-focused"
            self.query_one("#cpu-panel", CpuPanel).classes = "panel"
            self.query_one("#gpu-panel", GpuPanel).classes = "panel"
            self.query_one("#verdict-panel", VerdictPanel).classes = "panel"
        except NoMatches:
            pass

    def action_focus_gpu(self) -> None:
        try:
            self.query_one("#gpu-panel", GpuPanel).classes = "panel panel-focused"
            self.query_one("#cpu-panel", CpuPanel).classes = "panel"
            self.query_one("#mem-panel", MemoryPanel).classes = "panel"
            self.query_one("#verdict-panel", VerdictPanel).classes = "panel"
        except NoMatches:
            pass

    def action_focus_verdict(self) -> None:
        try:
            self.query_one("#verdict-panel", VerdictPanel).classes = "panel panel-focused"
            self.query_one("#cpu-panel", CpuPanel).classes = "panel"
            self.query_one("#mem-panel", MemoryPanel).classes = "panel"
            self.query_one("#gpu-panel", GpuPanel).classes = "panel"
        except NoMatches:
            pass

    def action_scroll_line_up(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#grid-container").scroll_up(animate=False)

    def action_scroll_line_down(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#grid-container").scroll_down(animate=False)

    def action_scroll_page_up(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#grid-container").scroll_page_up(animate=False)

    def action_scroll_page_down(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#grid-container").scroll_page_down(animate=False)


class JobSelectorScreen(ModalScreen[str]):
    BINDINGS = [
        Binding("enter", "select_job", "Select"),
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]

    CSS = """
    JobSelectorScreen {
        align: center middle;
    }

    #selector-box {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1;
    }

    #selector-title {
        text-style: bold;
        padding-bottom: 1;
    }

    ListView {
        height: auto;
        max-height: 20;
    }

    ListItem {
        padding: 0 1;
    }

    ListItem:hover {
        background: $accent;
    }
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
    Screen {
        background: $surface;
    }
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
        # push_screen_wait (used by the selector path) requires a Textual
        # worker context; a plain asyncio task would die with NoActiveWorker.
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
        try:
            jobs = resolve_current_jobs() if self._jobs is None else self._jobs
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
            self._job_ctx = resolve_job_context(job_id)
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
