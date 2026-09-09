"""An NVML string that is not valid UTF-8 must not misalign the handle lists.

`_attach_handle` appends to `_nvml_handles` first and `_nvml_indices` last, and its
docstring says why: the two are "appended together so that a later, transient
nvmlDeviceGetIndex failure during collection can fall back to the index cached
here instead of dropping the GPU (B-P7)". Everything raised between those two
appends breaks that pairing for the life of the process, and a later index lookup
then reads a DIFFERENT GPU's cached index.

Three sites in this file open-coded a strict `raw.decode()` while the class already
had a tolerant `_decode`. A strict decode raises `UnicodeDecodeError`, which is not
an `nv.NVMLError`, so it walked straight through the `except nv.NVMLError` that
every other failure in that block lands in.

NVML device names and UUIDs are ASCII in practice, so this is a narrow path -- but
the cost if it is ever taken is silent and permanent, and the tolerant decoder was
already sitting in the same class.
"""

from __future__ import annotations

import sys

import pytest

from slurmwatch.collector import TelemetryCollector


class _Nvml:
    """The three calls `_attach_handle` makes, and nothing else."""

    class NVMLError(Exception):
        pass

    uuid: object = b"GPU-0123"
    name: object = b"A100-SXM4-80GB"
    index: object = 0

    @classmethod
    def nvmlDeviceGetIndex(cls, _h: object) -> object:
        if isinstance(cls.index, Exception):
            raise cls.index
        return cls.index

    @classmethod
    def nvmlDeviceGetUUID(cls, _h: object) -> object:
        return cls.uuid

    @classmethod
    def nvmlDeviceGetName(cls, _h: object) -> object:
        return cls.name


@pytest.fixture
def collector(monkeypatch: pytest.MonkeyPatch) -> TelemetryCollector:
    """Just the attributes `_attach_handle` touches.

    `object.__new__` on purpose: the invariant under test is about the parallel
    lists and one method, and building a real collector would drag in a job
    context, a scheduler and a sampling thread that have nothing to do with it.

    The cost of that is this fixture: it mirrors the method's touched set by
    hand, so the set growing breaks every test in the file with an
    `AttributeError` rather than a useful message. `_cuda_ordinals` (D12) is the
    third list, and it is aligned with the other two by position for the same
    reason they are aligned with each other.
    """
    monkeypatch.setitem(sys.modules, "pynvml", _Nvml)
    obj = object.__new__(TelemetryCollector)
    obj._nvml_handles = []
    obj._nvml_indices = []
    obj._cuda_ordinals = []
    obj._nvml_handle_info = {}
    return obj


def _reset(**kw: object) -> None:
    _Nvml.uuid = kw.get("uuid", b"GPU-0123")
    _Nvml.name = kw.get("name", b"A100-SXM4-80GB")
    _Nvml.index = kw.get("index", 0)


class TestTheHandleListsStayAligned:
    def test_an_undecodable_uuid_does_not_break_the_pairing(
        self, collector: TelemetryCollector
    ) -> None:
        _reset(uuid=b"GPU-\xff\xfe")
        collector._attach_handle(None, ("h", 0))
        assert len(collector._nvml_handles) == len(collector._nvml_indices) == 1

    def test_an_undecodable_name_does_not_break_the_pairing(
        self, collector: TelemetryCollector
    ) -> None:
        _reset(name=b"A100\xff")
        collector._attach_handle(None, ("h", 0))
        assert len(collector._nvml_handles) == len(collector._nvml_indices) == 1

    def test_the_bad_bytes_are_replaced_not_dropped(self, collector: TelemetryCollector) -> None:
        _reset(uuid=b"GPU-\xff\xfe", name=b"A100\xff")
        collector._attach_handle(None, ("h", 0))
        recorded_uuid, recorded_name = collector._nvml_handle_info[0]
        assert recorded_uuid.startswith("GPU-") and "\ufffd" in recorded_uuid
        assert recorded_name.startswith("A100") and "\ufffd" in recorded_name

    def test_several_gpus_stay_index_aligned(self, collector: TelemetryCollector) -> None:
        # The consequence the docstring names: with the lists out of step, an index
        # lookup returns another GPU's cached index.
        _reset(uuid=b"GPU-\xff")
        for i in range(4):
            _Nvml.index = i
            collector._attach_handle(None, ("h", i))
        assert collector._nvml_indices == [0, 1, 2, 3]
        assert len(collector._nvml_handles) == 4


class TestTheOrdinaryAndTheAlreadyHandledCases:
    def test_valid_bytes_decode_exactly(self, collector: TelemetryCollector) -> None:
        # The control: the tolerant decoder must not alter ASCII.
        _reset()
        collector._attach_handle(None, ("h", 0))
        assert collector._nvml_handle_info[0] == ("GPU-0123", "A100-SXM4-80GB")

    def test_a_str_is_passed_through(self, collector: TelemetryCollector) -> None:
        # Newer pynvml returns str, which is why the `isinstance` check existed.
        _reset(uuid="GPU-str", name="H100")
        collector._attach_handle(None, ("h", 0))
        assert collector._nvml_handle_info[0] == ("GPU-str", "H100")

    def test_an_nvml_error_still_records_the_minus_one_placeholder(
        self, collector: TelemetryCollector
    ) -> None:
        # Unchanged behaviour, and the reason the lists can stay aligned at all.
        _reset(index=_Nvml.NVMLError("no index"))
        collector._attach_handle(None, ("h", 0))
        assert collector._nvml_indices == [-1]
        assert 0 not in collector._nvml_handle_info
