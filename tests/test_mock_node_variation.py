"""Every figure on a `--demo` node card varies when you switch nodes.

`_vary_mock_for_node` exists for one reason, which it states: the per-node mock
frames "were byte-identical apart from the hostname: every node read 50.0% CPU
and 25.0% memory ... pressing the key changed nothing on screen. That is
indistinguishable from a switch that silently failed."

Two fields survived that fix. The memory peaks -- `peak_bytes` and
`peak_working_set_bytes` -- were not scaled, while `current_bytes`,
`working_set_bytes`, both percentages, the whole CPU triple (including
`peak_effective_cores`) and every GPU figure were. Measured on the mock's own
frame before the fix:

    node  mem_used   mem_peak   ws        ws_peak
    0     16.00      16.80      16.00     16.80
    1     13.28      16.80      13.28     16.80
    4      5.12      16.80       5.12     16.80

so node 4 showed `used 5.12 GiB` beside `peak 16.80 GiB` -- a 3.3x ratio where
`_collect_memory`'s mock branch generates 1.05x -- and the two peaks read the
same bytes on all five nodes.

The scaled peaks are floored at the scaled reading, because `peak >= used` is
the invariant `_apply_peaks` states and every producer upholds. The floor is
load-bearing rather than decorative: integer truncation at factor 0.32 can pull
a peak only 1.05x above its reading down to or below it.
"""

from __future__ import annotations

import pytest

from slurmwatch.collector import _vary_mock_for_node
from slurmwatch.model import CpuMetrics, GpuMetrics, MemoryMetrics

GIB = 1 << 30

#: The factor sequence the function documents: 0.83, 0.66, 0.49, 0.32, repeat.
NODES = [1, 2, 3, 4]


def _cpu() -> CpuMetrics:
    return CpuMetrics(
        cores_allocated=16,
        usage_ns=0,
        usage_percent=50.0,
        effective_cores=8.0,
        peak_effective_cores=8.0,
        source="mock",
    )


def _mem(peak_ratio: float = 1.05) -> MemoryMetrics:
    """The mock's own shape: a peak a little above the reading."""
    current = 16 * GIB
    return MemoryMetrics(
        current_bytes=current,
        limit_bytes=64 * GIB,
        peak_bytes=int(peak_ratio * current),
        usage_percent=25.0,
        oom_guard_warning=False,
        oom_guard_critical=False,
        working_set_bytes=current,
        cache_bytes=0,
        peak_working_set_bytes=int(peak_ratio * current),
        working_set_percent=25.0,
        source="mock",
    )


def _vary(
    node_index: int, peak_ratio: float = 1.05
) -> tuple[CpuMetrics, MemoryMetrics, list[GpuMetrics]]:
    return _vary_mock_for_node(_cpu(), _mem(peak_ratio), [], node_index)


class TestSwitchingNodesMovesEveryFigure:
    def test_both_memory_peaks_vary(self) -> None:
        """The two fields that did not. Byte-identical across nodes was the bug."""
        peaks = {_vary(i)[1].peak_bytes for i in [0] + NODES}
        ws_peaks = {_vary(i)[1].peak_working_set_bytes for i in [0] + NODES}
        assert len(peaks) == 5, sorted(peaks)
        assert len(ws_peaks) == 5, sorted(ws_peaks)

    @pytest.mark.parametrize("node", NODES)
    def test_the_peak_moves_with_the_reading_beside_it(self, node: int) -> None:
        """Not merely different -- proportionate. The mock generates 1.05x."""
        _, mem, _ = _vary(node)
        assert mem.current_bytes > 0
        ratio = mem.peak_bytes / mem.current_bytes
        assert 1.0 <= ratio <= 1.10, (node, ratio)
        ws_ratio = mem.peak_working_set_bytes / mem.working_set_bytes
        assert 1.0 <= ws_ratio <= 1.10, (node, ws_ratio)

    def test_the_floor_actually_bites_on_a_tight_peak(self) -> None:
        """Why `max(...)` and not a bare multiply.

        With a peak just above its reading, `int()` truncation at the smallest
        factor lands on or below the scaled reading, so the floor is what keeps
        the invariant rather than arithmetic luck.
        """
        _, mem, _ = _vary(4, peak_ratio=1.0)
        assert mem.peak_bytes == mem.current_bytes
        assert mem.peak_working_set_bytes == mem.working_set_bytes


class TestControls:
    """Each passes with the two peak lines removed as well as with them.

    They cover the figures the fix did not touch and the guarantee node 0 has,
    so a neuter that reddens one of them means the change went further than the
    two memory peaks -- verified by running it.
    """

    def test_the_primary_node_is_returned_untouched(self) -> None:
        """The docstring's own guarantee: "node 0 is left EXACTLY as it was"."""
        cpu, mem, gpus = _vary(0)
        assert cpu == _cpu()
        assert mem == _mem()
        assert gpus == []

    @pytest.mark.parametrize("node", NODES)
    def test_the_cpu_triple_still_scales(self, node: int) -> None:
        """Including its peak, which this function already handled."""
        cpu, _, _ = _vary(node)
        assert cpu.effective_cores < 8.0, node
        assert cpu.peak_effective_cores < 8.0, node
        assert cpu.usage_percent < 50.0, node
        assert cpu.cores_allocated == 16, "the allocation is not a measurement"

    @pytest.mark.parametrize("node", NODES)
    def test_the_readings_still_scale(self, node: int) -> None:
        _, mem, _ = _vary(node)
        assert mem.current_bytes < 16 * GIB, node
        assert mem.working_set_bytes < 16 * GIB, node
        assert mem.usage_percent < 25.0, node
        assert mem.limit_bytes == 64 * GIB, "the limit is not a measurement"

    @pytest.mark.parametrize("node", [0] + NODES)
    def test_the_peak_never_falls_below_its_reading(self, node: int) -> None:
        """`peak >= used`, the invariant `_apply_peaks` states -- and a CONTROL
        rather than a finding test, established by running the neuter.

        It held before the fix too, for a reason worth writing down: leaving the
        peaks UNSCALED left them above every scaled reading, so the invariant
        survived the defect. What it guards is the other direction -- that
        scaling them down did not pull one under its own reading, which is what
        the `max(...)` floor is for.
        """
        _, mem, _ = _vary(node)
        assert mem.peak_bytes >= mem.current_bytes, node
        assert mem.peak_working_set_bytes >= mem.working_set_bytes, node

    def test_the_factor_sequence_is_what_the_comment_says(self) -> None:
        """0.83, 0.66, 0.49, 0.32, then repeats -- so node 5 equals node 0."""
        got = [round(_vary(i)[1].current_bytes / (16 * GIB), 2) for i in NODES]
        assert got == [0.83, 0.66, 0.49, 0.32], got
        assert _vary(5)[1].current_bytes == 16 * GIB

    @pytest.mark.parametrize("node", NODES)
    def test_gpu_figures_still_scale(self, node: int) -> None:
        """The third block, untouched by this fix."""
        gpu = GpuMetrics(
            index=0,
            uuid="GPU-0",
            name="A100",
            utilization_percent=80.0,
            memory_used_bytes=40 * GIB,
            memory_total_bytes=80 * GIB,
            memory_utilization_percent=50.0,
            power_watts=240.0,
            temperature_celsius=65.0,
            throttling=False,
            process_utilization_percent=80.0,
        )
        _, _, gpus = _vary_mock_for_node(_cpu(), _mem(), [gpu], node)
        assert gpus[0].utilization_percent < 80.0, node
        assert gpus[0].memory_used_bytes < 40 * GIB, node
