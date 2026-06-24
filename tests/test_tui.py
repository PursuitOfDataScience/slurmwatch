from __future__ import annotations

import time

from slurmwatch.model import CpuMetrics, GpuMetrics, MemoryMetrics, TelemetrySnapshot
from slurmwatch.tui import (
    CpuPanel,
    GpuPanel,
    MemoryPanel,
    VerdictPanel,
    _format_bytes,
    _format_duration,
    _render_bar,
    _render_sparkline,
)


class TestHelpers:
    def test_format_bytes_bytes(self) -> None:
        assert _format_bytes(0) == "0.0 B"
        assert _format_bytes(500) == "500.0 B"

    def test_format_bytes_kib(self) -> None:
        assert _format_bytes(1024) == "1.0 KiB"
        assert _format_bytes(2048) == "2.0 KiB"

    def test_format_bytes_mib(self) -> None:
        assert _format_bytes(1024 * 1024) == "1.0 MiB"

    def test_format_bytes_gib(self) -> None:
        assert _format_bytes(1024**3) == "1.0 GiB"

    def test_format_bytes_tib(self) -> None:
        assert _format_bytes(1024**4) == "1.0 TiB"

    def test_format_bytes_pib(self) -> None:
        assert _format_bytes(1024**5) == "1.0 PiB"

    def test_format_duration(self) -> None:
        assert _format_duration(0) == "00:00:00"
        assert _format_duration(3661) == "01:01:01"
        assert _format_duration(86399) == "23:59:59"

    def test_render_bar_full(self) -> None:
        assert _render_bar(100, 4) == "████"

    def test_render_bar_half(self) -> None:
        assert _render_bar(50, 4) == "██░░"

    def test_render_bar_empty(self) -> None:
        assert _render_bar(0, 4) == "░░░░"

    def test_render_bar_rounded(self) -> None:
        bar = _render_bar(33, 6)
        assert len(bar) == 6
        assert bar.count("█") == 1
        assert bar.count("░") == 5

    def test_render_bar_ascii(self) -> None:
        assert _render_bar(50, 4, ascii_mode=True) == "##--"

    def test_render_sparkline(self) -> None:
        from collections import deque

        vals: deque[float] = deque([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        result = _render_sparkline(vals, 5)
        assert len(result) == 5

    def test_render_sparkline_empty(self) -> None:
        from collections import deque

        result = _render_sparkline(deque(), 5)
        assert result == " " * 5

    def test_render_sparkline_scaled_to_100(self) -> None:
        from collections import deque

        vals: deque[float] = deque([5.0])
        result_5 = _render_sparkline(vals, 3)
        assert len(result_5) == 3
        vals2: deque[float] = deque([95.0])
        result_95 = _render_sparkline(vals2, 3)
        assert result_5 != result_95  # different values produce different patterns


class TestCpuPanel:
    def test_render_no_data(self) -> None:
        panel = CpuPanel()
        rendered = panel.render()
        assert "awaiting data" in rendered

    def test_render_with_data(self) -> None:
        panel = CpuPanel()
        panel.snapshot = _make_snapshot()
        rendered = panel.render()
        assert "CPU" in rendered
        assert "16 cores" in rendered
        assert "50.0%" in rendered
        assert "effective" in rendered


class TestMemoryPanel:
    def test_render_no_data(self) -> None:
        panel = MemoryPanel()
        rendered = panel.render()
        assert "awaiting data" in rendered

    def test_render_normal(self) -> None:
        panel = MemoryPanel()
        snap = _make_snapshot()
        snap.memory.usage_percent = 50.0
        snap.memory.oom_guard_warning = False
        snap.memory.oom_guard_critical = False
        panel.snapshot = snap
        rendered = panel.render()
        assert "MEMORY" in rendered
        assert "WARNING" not in rendered
        assert "CRITICAL" not in rendered
        assert "working set" in rendered

    def test_render_warning_threshold(self) -> None:
        panel = MemoryPanel()
        snap = _make_snapshot()
        snap.memory.usage_percent = 86.0
        snap.memory.oom_guard_warning = True
        snap.memory.oom_guard_critical = False
        panel.snapshot = snap
        rendered = panel.render()
        assert "WARNING" in rendered
        assert "CRITICAL" not in rendered

    def test_render_critical_threshold(self) -> None:
        panel = MemoryPanel()
        snap = _make_snapshot()
        snap.memory.usage_percent = 95.0
        snap.memory.oom_guard_warning = False
        snap.memory.oom_guard_critical = True
        panel.snapshot = snap
        rendered = panel.render()
        assert "CRITICAL" in rendered


class TestGpuPanel:
    def test_render_no_data(self) -> None:
        panel = GpuPanel()
        rendered = panel.render()
        assert "awaiting data" in rendered

    def test_render_no_gpus(self) -> None:
        panel = GpuPanel()
        snap = _make_snapshot()
        snap.gpus = []
        panel.snapshot = snap
        rendered = panel.render()
        assert "no GPUs" in rendered

    def test_render_with_gpus(self) -> None:
        panel = GpuPanel()
        snap = _make_snapshot()
        panel.snapshot = snap
        rendered = panel.render()
        assert "GPU 0" in rendered
        assert "A100" in rendered
        assert "72.5%" in rendered

    def test_render_throttling(self) -> None:
        panel = GpuPanel()
        snap = _make_snapshot()
        snap.gpus[0].throttling = True
        panel.snapshot = snap
        rendered = panel.render()
        assert "!" in rendered or "⚠" in rendered

    def test_render_process_util(self) -> None:
        panel = GpuPanel()
        snap = _make_snapshot()
        snap.gpus[0].process_utilization_percent = 60.0
        snap.gpus[0].process_memory_bytes = 18 * 1024**3
        panel.snapshot = snap
        rendered = panel.render()
        assert "proc:" in rendered


class TestVerdictPanel:
    def test_render_no_data(self) -> None:
        panel = VerdictPanel()
        rendered = panel.render()
        assert "awaiting data" in rendered

    def test_render_with_snapshot(self) -> None:
        panel = VerdictPanel()
        panel.snapshot = _make_snapshot()
        rendered = panel.render()
        assert "Allocation Efficiency" in rendered
        assert "CPU" in rendered
        assert "Memory" in rendered
        assert "GPU" in rendered


def _make_snapshot() -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp=time.time(),
        job_id="12345",
        step_id="0",
        hostname="cn001",
        elapsed_seconds=3600,
        cpu=CpuMetrics(
            cores_allocated=16,
            usage_ns=1_000_000_000,
            usage_percent=50.0,
            effective_cores=8.0,
        ),
        memory=MemoryMetrics(
            current_bytes=32 * 1024**3,
            limit_bytes=64 * 1024**3,
            peak_bytes=40 * 1024**3,
            usage_percent=50.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=28 * 1024**3,
            cache_bytes=4 * 1024**3,
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
                process_utilization_percent=60.0,
                process_memory_bytes=18 * 1024**3,
            ),
        ],
        gpu_count_requested=4,
        gpu_active_count=1,
    )
