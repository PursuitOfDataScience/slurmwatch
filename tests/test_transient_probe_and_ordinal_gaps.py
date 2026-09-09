"""Two latches that recorded a failure as if it were an answer.

Both were on the deferred list in ``issues.md`` (D11, D12) and both were
re-measured against the code before being touched, because nine rows in that
table have been found already fixed while still reading as open.

**D11 — the interconnect probe latched before it ran.**
``_interconnect_probed = True`` was the first statement inside the guard, so a
single transient NVML error on the first frame left ``_interconnect_static`` at
``None`` for the life of the run, and ``static is None`` returns early -- the
whole interconnect section stayed hidden for a multi-day job because of one bad
read. The guard cannot simply move to the success path, which is why the row was
deferred: ``_build_topology`` returns ``None`` legitimately when fewer than two
devices are visible, and re-probing that every frame is exactly what the guard
exists to prevent. So it latches on any RETURN, and only the exception path
retries, up to ``_INTERCONNECT_PROBE_ATTEMPTS``.

**D12 — a device NVML could not identify shifted every later CUDA ordinal.**
``cuda_ordinal`` came from the device's position in ``_nvml_handles``. In the
``gpu_uuids`` path a uuid that ``_handle_by_uuid`` cannot resolve -- a card that
has fallen off the bus -- is skipped and never attached, so everything after it
sits one position lower: a three-GPU job whose middle card is gone labels its
``cuda:2`` as ``CUDA 1`` on every surface. The position in the list the JOB asked
for is now recorded at attach. Note the comment already at the metrics site
addressed the *other* moment -- a device dropped during collection -- which is
why the defect survived it.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

from slurmwatch import collector as _collector_mod
from slurmwatch.collector import TelemetryCollector
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import GpuInterconnect, JobContext


def _min_ctx(**kw: object) -> JobContext:
    base: dict[str, object] = {
        "job_id": "1",
        "username": "u",
        "partition": "p",
        "nodelist": "n",
        "hostname": "n",
        "cpus_allocated": 1,
        "mem_limit_bytes": 1,
        "gpu_count_requested": 0,
        "gpu_indices": [],
    }
    base.update(kw)
    return JobContext(**base)  # type: ignore[arg-type]


class _BareNvml:
    """Only what `_init_nvml` and `_attach_handle` reach."""

    class NVMLError(Exception):
        pass

    def __init__(self, count: int = 3) -> None:
        self._count = count

    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        return None

    def nvmlDeviceGetCount(self) -> int:
        return self._count

    def nvmlDeviceGetHandleByIndex(self, idx: int) -> tuple[str, int]:
        return ("h", idx)

    def nvmlDeviceGetIndex(self, handle: tuple[str, int]) -> int:
        return handle[1]

    def nvmlDeviceGetUUID(self, handle: tuple[str, int]) -> bytes:
        return f"GPU-{handle[1]}".encode()

    def nvmlDeviceGetName(self, handle: tuple[str, int]) -> bytes:
        return b"FakeGPU"


def _wired() -> GpuInterconnect:
    return GpuInterconnect(fabric="pcie", devices=[0, 1])


def _ready(monkeypatch: pytest.MonkeyPatch) -> TelemetryCollector:
    monkeypatch.setitem(sys.modules, "pynvml", _BareNvml())
    collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig())
    collector._nvml_initialized = True
    return collector


class TestATransientProbeFailureIsRetried:
    def test_a_first_frame_failure_no_longer_hides_the_section_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collector = _ready(monkeypatch)
        calls: list[int] = []

        def flaky() -> GpuInterconnect | None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient NVML error on the first frame")
            return _wired()

        monkeypatch.setattr(collector, "_build_topology", flaky)
        assert collector._collect_interconnect([]) is None  # the bad frame
        second = collector._collect_interconnect([])
        assert second is not None, "the probe was never retried"
        assert second.fabric == "pcie"
        assert len(calls) == 2

    def test_the_retries_are_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collector = _ready(monkeypatch)
        calls: list[int] = []

        def always_fails() -> GpuInterconnect | None:
            calls.append(1)
            raise RuntimeError("this node's topology cannot be read")

        monkeypatch.setattr(collector, "_build_topology", always_fails)
        for _ in range(_collector_mod._INTERCONNECT_PROBE_ATTEMPTS + 4):
            assert collector._collect_interconnect([]) is None
        assert len(calls) == _collector_mod._INTERCONNECT_PROBE_ATTEMPTS, calls


class TestControlsOnTheProbeLatch:
    """These hold whether or not the latch was moved."""

    def test_a_successful_none_still_latches_on_the_first_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason the row was deferred rather than fixed in one line.

        Fewer than two visible devices is a real answer, not a failure, and
        re-probing it every frame is what this guard exists to prevent.
        """
        collector = _ready(monkeypatch)
        calls: list[int] = []

        def no_fabric() -> GpuInterconnect | None:
            calls.append(1)
            return None

        monkeypatch.setattr(collector, "_build_topology", no_fabric)
        for _ in range(5):
            assert collector._collect_interconnect([]) is None
        assert len(calls) == 1, calls

    def test_a_successful_topology_is_still_probed_only_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collector = _ready(monkeypatch)
        calls: list[int] = []

        def wired() -> GpuInterconnect | None:
            calls.append(1)
            return _wired()

        monkeypatch.setattr(collector, "_build_topology", wired)
        for _ in range(5):
            assert collector._collect_interconnect([]) is not None
        assert len(calls) == 1, calls


class TestAnUnresolvableDeviceDoesNotShiftTheOrdinals:
    def _attached(
        self, monkeypatch: pytest.MonkeyPatch, uuids: list[str], missing: set[str]
    ) -> TelemetryCollector:
        monkeypatch.setitem(sys.modules, "pynvml", _BareNvml())
        monkeypatch.setattr(_collector_mod, "_nvidia_node_gpu_models", lambda: {})
        collector = TelemetryCollector(
            _min_ctx(gpu_uuids=uuids, gpu_count_requested=len(uuids)),
            SlurmwatchConfig(),
        )
        monkeypatch.setattr(
            collector,
            "_handle_by_uuid",
            lambda _p, uuid, _n: None if uuid in missing else ("h", uuids.index(uuid)),
        )
        assert collector._init_nvml() is True
        return collector

    def test_the_middle_card_falling_off_the_bus_leaves_a_gap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uuids = ["GPU-a", "GPU-b", "GPU-c"]
        collector = self._attached(monkeypatch, uuids, {"GPU-b"})
        assert len(collector._nvml_handles) == 2
        # 0 and 2, not 0 and 1: the third device is still CUDA 2 to the job.
        assert collector._cuda_ordinals == [0, 2]


class TestControlsOnTheOrdinals:
    """These hold whether or not the ordinal is carried from the requested list."""

    def test_with_every_device_resolvable_the_ordinals_are_dense(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uuids = ["GPU-a", "GPU-b", "GPU-c"]
        collector = TestAnUnresolvableDeviceDoesNotShiftTheOrdinals()._attached(
            monkeypatch, uuids, set()
        )
        assert collector._cuda_ordinals == [0, 1, 2]

    def test_the_lists_stay_aligned_with_the_handles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-P7's invariant, extended to the new list: a per-device lookup that
        # reads position `i` of one list and position `i` of another must be
        # reading the same device.
        uuids = ["GPU-a", "GPU-b", "GPU-c"]
        collector = TestAnUnresolvableDeviceDoesNotShiftTheOrdinals()._attached(
            monkeypatch, uuids, {"GPU-b"}
        )
        assert len(collector._cuda_ordinals) == len(collector._nvml_handles)
        assert len(collector._nvml_indices) == len(collector._nvml_handles)

    def test_the_pci_ordered_path_still_uses_the_append_position(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No requested list, so the append position IS the ordinal.

        Guards against "carry an ordinal" being applied where there is nothing
        to carry: here every visible device belongs to the job and none is
        skipped, so the two numbers must agree.
        """
        monkeypatch.setitem(sys.modules, "pynvml", _BareNvml(count=3))
        monkeypatch.setattr(_collector_mod, "_nvidia_node_gpu_models", lambda: {})
        collector = TelemetryCollector(_min_ctx(gpu_count_requested=3), SlurmwatchConfig())
        assert collector._init_nvml() is True
        assert collector._cuda_ordinals == list(range(len(collector._nvml_handles)))


class TestThePublishedOrdinalReadsTheRecord:
    """The observable figure, not just the bookkeeping list behind it."""

    def test_a_gap_in_the_record_reaches_the_published_ordinal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig())
        collector._cuda_ordinals = [0, 2]
        assert [collector._cuda_ordinal_for(i) for i in (0, 1)] == [0, 2]

    def test_it_falls_back_to_the_position_where_nothing_was_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The PCI-ordered attach paths record nothing, and a `pos` past the end
        # must not raise -- the reader is on the per-frame metrics path.
        collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig())
        assert collector._cuda_ordinal_for(0) == 0
        collector._cuda_ordinals = [5]
        assert collector._cuda_ordinal_for(0) == 5
        assert collector._cuda_ordinal_for(3) == 3


class TestTheIndexPathCarriesThePositionToo:
    """`gpu_indices` has the same gap, and two numbers that are easy to confuse.

    The CUDA ordinal is the POSITION in the job's list; the value at that
    position is the node-global device id used to look the handle up. An id that
    is out of range is skipped, so the position has to be carried rather than
    inferred from what attached.
    """

    def _attached(self, monkeypatch: pytest.MonkeyPatch, indices: list[int]) -> TelemetryCollector:
        # Four devices on the node, never as many as the job asks for. Equality
        # means ConstrainDevices -- NVML already shows only the job's GPUs, so
        # `_init_nvml` attaches them all and never consults the id list at all.
        # A first draft of this test used three and silently exercised that
        # branch instead of the mapping one.
        monkeypatch.setitem(sys.modules, "pynvml", _BareNvml(count=4))
        monkeypatch.setattr(_collector_mod, "_nvidia_node_gpu_models", lambda: {})
        collector = TelemetryCollector(
            _min_ctx(gpu_indices=indices, gpu_count_requested=len(indices)),
            SlurmwatchConfig(),
        )
        assert collector._init_nvml() is True
        return collector

    def test_an_out_of_range_id_leaves_a_gap_rather_than_shifting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Node has 4 devices; the job asks for ids 0, 5 and 2. The 5 cannot be
        # looked up, so without carrying the position the device the job calls
        # CUDA 2 would report CUDA 1.
        collector = self._attached(monkeypatch, [0, 5, 2])
        assert len(collector._nvml_handles) == 2
        assert collector._cuda_ordinals == [0, 2]

    def test_a_fully_resolvable_index_list_is_dense(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The control: nothing skipped, so position and append order agree.
        collector = self._attached(monkeypatch, [0, 2])
        assert collector._cuda_ordinals == [0, 1]


class TestTheMetricsSiteActuallyReadsTheRecord:
    """The wiring between the reader and the published field.

    `_cuda_ordinal_for` has teeth of its own above, but nothing proved the
    per-GPU metrics call it: reverting that one keyword back to `pos` left all
    twelve tests green (measured). Driving the real path would mean standing up
    every NVML call `_collect_gpu_metrics` makes, so it is pinned at the source
    instead -- by AST rather than by string, so reformatting cannot break it.
    """

    def _cuda_ordinal_arguments(self) -> set[str]:
        tree = ast.parse(pathlib.Path(_collector_mod.__file__).read_text())
        return {
            ast.unparse(kw.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "cuda_ordinal"
        }

    def test_the_real_path_goes_through_the_reader(self) -> None:
        assert "self._cuda_ordinal_for(pos)" in self._cuda_ordinal_arguments()

    def test_no_site_publishes_the_bare_handle_position(self) -> None:
        # `pos` is the position in `_nvml_handles`, which is exactly what a
        # skipped device makes wrong. The mock path's `i` is fine -- it invents
        # its own devices and never skips one.
        assert "pos" not in self._cuda_ordinal_arguments()


class TestTeardownForgetsTheOrdinals:
    """A re-attach must not inherit the previous attach's ordinals.

    `_cuda_ordinals` is aligned with `_nvml_handles` by position, so a teardown
    that clears one and not the other leaves a per-device lookup reading another
    device's ordinal -- the same alignment failure B-P7 exists for.
    """

    def test_the_sync_shutdown_clears_it_with_the_handles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", _BareNvml(count=3))
        monkeypatch.setattr(_collector_mod, "_nvidia_node_gpu_models", lambda: {})
        collector = TelemetryCollector(_min_ctx(gpu_count_requested=3), SlurmwatchConfig())
        assert collector._init_nvml() is True
        assert collector._cuda_ordinals, "nothing attached, so this proves nothing"
        collector._shutdown_nvml_sync()
        assert collector._nvml_handles == []
        assert collector._cuda_ordinals == []
