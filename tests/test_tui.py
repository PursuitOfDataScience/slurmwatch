from __future__ import annotations

from slurmwatch.model import CpuMetrics, GpuMetrics, MemoryMetrics, TelemetrySnapshot
from slurmwatch.tui import (
    CpuPanel,
    GpuPanel,
    MemoryPanel,
    _format_bytes,
    _format_duration,
    _render_bar,
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
        assert "⚠" in rendered


def _make_snapshot() -> TelemetrySnapshot:
    import time

    return TelemetrySnapshot(
        timestamp=time.time(),
        job_id=12345,
        step_id=0,
        hostname="cn001",
        elapsed_seconds=3600,
        cpu=CpuMetrics(cores_allocated=16, usage_ns=1_000_000_000, usage_percent=50.0),
        memory=MemoryMetrics(
            current_bytes=32 * 1024**3,
            limit_bytes=64 * 1024**3,
            peak_bytes=40 * 1024**3,
            usage_percent=50.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
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
            ),
        ],
    )
