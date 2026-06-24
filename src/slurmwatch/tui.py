from __future__ import annotations

import asyncio
import contextlib
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.widgets import ListItem, ListView, Static

from .collector import TelemetryCollector
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


def _render_bar(percent: float, length: int = 12) -> str:
    filled = int(percent / 100 * length)
    return "█" * filled + "░" * (length - filled)


class CpuPanel(Static):
    snapshot: TelemetrySnapshot | None = None

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]CPU: awaiting data…[/]"
        cpu = self.snapshot.cpu
        bar = _render_bar(cpu.usage_percent, 16)
        return (
            f"[bold]CPU[/]  {bar}  {cpu.usage_percent:.1f}%\n"
            f"      {cpu.cores_allocated} cores allocated"
        )


class MemoryPanel(Static):
    snapshot: TelemetrySnapshot | None = None

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]Memory: awaiting data…[/]"
        mem = self.snapshot.memory
        pct = mem.usage_percent
        bar = _render_bar(pct, 16)

        if mem.oom_guard_critical:
            style = "[bold red]"
            guard = "⚠ CRITICAL"
        elif mem.oom_guard_warning:
            style = "[bold yellow]"
            guard = "⚠ WARNING"
        else:
            style = ""
            guard = ""

        used = _format_bytes(mem.current_bytes)
        limit = _format_bytes(mem.limit_bytes)
        peak = _format_bytes(mem.peak_bytes)

        return (
            f"{style}[bold]MEMORY[/]  {bar}  {pct:.1f}%[/]\n"
            f"      {used} / {limit}\n"
            f"      Peak: {peak}\n"
            f"      {guard}"
        )


class GpuPanel(Static):
    snapshot: TelemetrySnapshot | None = None

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]GPU: awaiting data…[/]"
        gpus = self.snapshot.gpus
        if not gpus:
            return "[dim]GPU: no GPUs detected[/]"

        lines: list[str] = []
        for gpu in gpus:
            util_bar = _render_bar(gpu.utilization_percent)
            mem_bar = _render_bar(gpu.memory_utilization_percent)
            mem_used = _format_bytes(gpu.memory_used_bytes)
            mem_total = _format_bytes(gpu.memory_total_bytes)
            throttle = " ⚠" if gpu.throttling else ""
            lines.append(
                f"[bold]GPU {gpu.index}: {gpu.name}[/]{throttle}\n"
                f"  Util:  {util_bar}  {gpu.utilization_percent:.1f}%\n"
                f"  VRAM:  {mem_bar}  {gpu.memory_utilization_percent:.1f}%"
                f" ({mem_used}/{mem_total})\n"
                f"  Power: {gpu.power_watts:.0f}W  Temp: {gpu.temperature_celsius:.0f}°C"
            )
        return "\n\n".join(lines)


class DashboardScreen(Screen):  # type: ignore[type-arg]
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("m", "focus_memory", "Memory"),
        Binding("c", "focus_cpu_gpu", "CPU/GPU"),
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
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        job_ctx: JobContext,
    ) -> None:
        super().__init__()
        self.collector = collector
        self.job_ctx = job_ctx
        self._snapshot: TelemetrySnapshot | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Container(id="grid-container"):
            yield CpuPanel(id="cpu-panel", classes="panel")
            yield MemoryPanel(id="mem-panel", classes="panel")
            yield GpuPanel(id="gpu-panel", classes="panel")
        yield Static(id="footer")

    def on_mount(self) -> None:
        self._update_header(None)
        self._update_footer()
        self.set_interval(0.25, self._poll_queue)

    async def _poll_queue(self) -> None:
        try:
            snapshot = await asyncio.wait_for(self.collector.next_snapshot(), timeout=0.3)
            self._snapshot = snapshot
            self._update_widgets(snapshot)
        except asyncio.TimeoutError:
            pass

    def _update_widgets(self, snapshot: TelemetrySnapshot) -> None:
        self._update_header(snapshot)
        try:
            cpu_w = self.query_one("#cpu-panel", CpuPanel)
            cpu_w.snapshot = snapshot
            cpu_w.refresh()
        except NoMatches:
            pass
        try:
            mem_w = self.query_one("#mem-panel", MemoryPanel)
            mem_w.snapshot = snapshot
            mem_w.refresh()
        except NoMatches:
            pass
        try:
            gpu_w = self.query_one("#gpu-panel", GpuPanel)
            gpu_w.snapshot = snapshot
            gpu_w.refresh()
        except NoMatches:
            pass

    def _update_header(self, snapshot: TelemetrySnapshot | None) -> None:
        try:
            header = self.query_one("#header", Static)
            if snapshot:
                elapsed = _format_duration(snapshot.elapsed_seconds)
                header.update(
                    f"[bold]slurmwatch[/] │ "
                    f"Job [bold]{snapshot.job_id}[/] │ "
                    f"User [bold]{self.job_ctx.username}[/] │ "
                    f"Partition [bold]{self.job_ctx.partition}[/] │ "
                    f"Nodes [bold]{self.job_ctx.nodelist}[/] │ "
                    f"Elapsed [bold]{elapsed}[/]"
                )
            else:
                header.update(f"[bold]slurmwatch[/] — connecting to job {self.job_ctx.job_id}…")
        except NoMatches:
            pass

    def _update_footer(self) -> None:
        try:
            footer = self.query_one("#footer", Static)
            footer.update(" [m] Memory  [c] CPU/GPU  [q] Quit")
        except NoMatches:
            pass

    def action_quit(self) -> None:
        self.app.exit()

    def action_focus_memory(self) -> None:
        try:
            self.query_one("#mem-panel", MemoryPanel).classes = "panel panel-focused"
            self.query_one("#cpu-panel", CpuPanel).classes = "panel"
            self.query_one("#gpu-panel", GpuPanel).classes = "panel"
        except NoMatches:
            pass

    def action_focus_cpu_gpu(self) -> None:
        try:
            self.query_one("#cpu-panel", CpuPanel).classes = "panel panel-focused"
            self.query_one("#gpu-panel", GpuPanel).classes = "panel panel-focused"
            self.query_one("#mem-panel", MemoryPanel).classes = "panel"
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


class JobSelectorScreen(ModalScreen[int]):
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

    def action_select_job(self) -> None:
        lv = self.query_one(ListView)
        if lv.index is not None and lv.index < len(self.jobs):
            val = self.jobs[lv.index]["job_id"]
            if isinstance(val, int):
                self.dismiss(val)

    def action_cancel(self) -> None:
        self.dismiss(-1)


class SlurmwatchApp(App):  # type: ignore[type-arg]
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
    ) -> None:
        super().__init__()
        self._job_ctx = job_ctx
        self._collector = collector

    def on_mount(self) -> None:
        if self._job_ctx is not None and self._collector is not None:
            asyncio.create_task(self._start_monitoring())
        else:
            asyncio.create_task(self._run_job_selector())

    async def _start_monitoring(self) -> None:
        assert self._collector is not None
        assert self._job_ctx is not None
        await self._collector.start()
        await self.push_screen(DashboardScreen(self._collector, self._job_ctx))

    async def _run_job_selector(self) -> None:
        try:
            jobs = resolve_current_jobs()
        except Exception:
            self.exit(message="Failed to query Slurm jobs", return_code=1)
            return

        if not jobs:
            self.exit(message="No running Slurm jobs found.", return_code=0)
            return

        if len(jobs) == 1:
            jid = jobs[0]["job_id"]
            job_id = int(jid)  # type: ignore[call-overload]
        else:
            result = await self.push_screen_wait(JobSelectorScreen(jobs))
            if result is None or result == -1:
                self.exit(message="No job selected.", return_code=0)
                return
            job_id = result

        try:
            self._job_ctx = resolve_job_context(job_id)
        except Exception as exc:
            self.exit(message=str(exc), return_code=1)
            return

        self._collector = TelemetryCollector(self._job_ctx)
        await self._collector.start()
        await self.push_screen(DashboardScreen(self._collector, self._job_ctx))
