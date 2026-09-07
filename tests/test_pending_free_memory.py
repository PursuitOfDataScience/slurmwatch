"""D7: "FITS NOW" measured memory against hardware someone else was using.

The WHERE table's CPU half has used *idle* `%C` since M5. Its memory half used
`sinfo`'s aggregate `%m`, which is a node's **configured** size — so a partition
whose nodes were all busy still advertised their full memory.

Measured live on this cluster before the fix, on `vitelli-amd` (2 nodes, neither
idle, largest mix node holding **6,160 MB free of 250,000 MB configured**)::

    request    5 GiB/node -> fit_blocker = ''
    request  240 GiB/node -> fit_blocker = ''      <- "plausibly fits"

and after::

    request    5 GiB/node -> ''
    request  240 GiB/node -> 'no room'

`'no room'` and not `'node too small'`: the hardware really is 250 GB, it is
merely occupied, and `_shape_blocker` draws exactly that line by comparing the
request against `max_config_node_mem_bytes`. A blank blocker is what puts the
copy-pasteable requeue command on screen, so this was advice to move a job onto a
partition that could not start it — losing the priority it had accrued.

**The row that recorded this said it "needs a new `sinfo -O AllocMem,MemSpecLimit`
query". Both halves of that are wrong**, and measuring is what showed it:

* no new query is needed — `_fetch_free_gpus_by_partition` already visits every
  node line for `GresUsed`, so `Memory` and `AllocMem` ride along on it;
* `MemSpecLimit` is **not a valid `-O` field on this controller** (Slurm 20.11.8):
  `sinfo: error: Invalid job format specification: MemSpecLimit` on stderr, rc=0,
  and the column renders EMPTY — the exact unsupported-field trap the `GresUsed`
  comment in that function documents. Asking for it would have silently produced a
  column of blanks. It is also moot here: 0 of 1,248 node lines report one.

Two ordering details are load-bearing and are pinned below. The memory read
happens BEFORE the GPU-specific `continue`s, because those skip every node without
GPUs and such nodes still have memory — on this cluster only 464 of 1,248 node
lines have GPUs, and none of the CPU partitions where the over-report was found.
And `isdigit` is what separates "this node has 0 MB free" from "this Slurm does
not know the field": the first renders `0`, the second renders ``.
"""

from __future__ import annotations

from typing import Any

import pytest

from slurmwatch import pending
from slurmwatch.pending import PartitionResources, fit_blocker

#: `sinfo -a -h -N -O "Partition|,StateLong|,Gres|,GresUsed|,Memory|,AllocMem|"`,
#: in the shape this controller emits: two mix nodes of a CPU partition, most of
#: whose memory is already handed out.
BUSY_CPU = "\n".join(
    [
        "busy-cpu            |mixed               |(null)|(null)|250000|243840|",
        "busy-cpu            |mixed               |(null)|(null)|250000|243840|",
    ]
)

#: The same partition with a GPU node beside it, to pin that memory is collected
#: for BOTH (the GPU reads `continue` past a node with no GPUs).
MIXED_SHAPES = "\n".join(
    [
        "both                |mixed               |(null)|(null)|250000|243840|",
        "both                |mixed               |gpu:a100:4|gpu:a100:1|500000|100000|",
    ]
)

#: An `AllocMem` column this Slurm does not know: empty, not zero.
UNSUPPORTED = "\n".join(
    [
        "busy-cpu            |mixed               |(null)|(null)|250000||",
        "busy-cpu            |mixed               |(null)|(null)|250000||",
    ]
)


def _fetch(monkeypatch: pytest.MonkeyPatch, out: str) -> tuple[Any, ...]:
    monkeypatch.setattr(pending, "_is_mock", lambda: False)
    monkeypatch.setattr(pending, "_run_slurm_cmd", lambda cmd: out)
    return pending._fetch_free_gpus_by_partition()


def _job(mem_gib: float, **kw: Any) -> Any:
    job = pending._mock_pending_job("1")
    job.exclusive = False
    job.req_gpus = 0
    job.req_nodes = 1
    job.req_cpus = 4
    job.req_mem_bytes = int(mem_gib * 1024**3)
    for name, value in kw.items():
        setattr(job, name, value)
    return job


def _busy_partition(**kw: Any) -> PartitionResources:
    """A partition with no idle nodes, 6.2 GB free per node, 250 GB configured."""
    part = PartitionResources("busy-cpu", True, idle_nodes=0, mix_nodes=2, cpus_idle=64)
    part.max_node_cpus = 128
    part.max_config_node_cpus = 128
    part.max_node_mem_bytes = 250_000 * 1024**2
    part.max_config_node_mem_bytes = 250_000 * 1024**2
    part.free_node_mem_mb = [6_160, 6_160]
    part.mem_detail = True
    for name, value in kw.items():
        setattr(part, name, value)
    return part


class TestAJobIsMeasuredAgainstFreeMemory:
    def test_a_request_the_free_memory_cannot_hold_is_refused(self) -> None:
        assert fit_blocker(_job(240), _busy_partition()) == "no room"

    def test_the_refusal_is_transient_not_permanent(self) -> None:
        """The hardware is 250 GB; it is busy, which clears by itself."""
        part = _busy_partition()
        assert fit_blocker(_job(240), part) == "no room"
        assert not pending.blocker_is_permanent(fit_blocker(_job(240), part))

    def test_a_request_that_fits_the_free_memory_still_fits(self) -> None:
        assert fit_blocker(_job(5), _busy_partition()) == ""

    def test_a_request_beyond_the_hardware_is_still_permanent(self) -> None:
        # Bigger than the node was BUILT with: no waiting fixes that.
        assert fit_blocker(_job(400), _busy_partition()) == "node too small"

    @pytest.mark.parametrize(
        ("gib", "expected"), [(5, ""), (6, ""), (7, "no room"), (240, "no room")]
    )
    def test_the_boundary_is_the_free_figure(self, gib: float, expected: str) -> None:
        assert fit_blocker(_job(gib), _busy_partition()) == expected


class TestTheQueryCarriesTheFieldsWithoutASecondCall:
    def test_one_call_asks_for_memory_and_allocmem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(pending, "_is_mock", lambda: False)

        def record(cmd: list[str]) -> str:
            calls.append(cmd)
            return BUSY_CPU

        monkeypatch.setattr(pending, "_run_slurm_cmd", record)
        pending._fetch_free_gpus_by_partition()
        assert len(calls) == 1, calls
        fields = calls[0][calls[0].index("-O") + 1]
        assert "Memory:" in fields and "AllocMem:" in fields, fields
        assert "MemSpecLimit" not in fields, "invalid -O field on Slurm 20.11.8"

    def test_free_memory_is_read_per_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _gpus, _ok, free_mem, mem_ok = _fetch(monkeypatch, BUSY_CPU)
        assert mem_ok is True
        assert free_mem == {"busy-cpu": [6_160, 6_160]}, free_mem

    def test_a_node_without_gpus_still_reports_its_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ordering that matters: the GPU reads skip such a node entirely."""
        _gpus, _ok, free_mem, mem_ok = _fetch(monkeypatch, MIXED_SHAPES)
        assert mem_ok is True
        assert sorted(free_mem["both"]) == [6_160, 400_000], free_mem

    def test_an_unsupported_field_is_unknown_not_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _gpus, _ok, free_mem, mem_ok = _fetch(monkeypatch, UNSUPPORTED)
        assert mem_ok is False
        assert free_mem == {}, free_mem


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_without_the_detail_it_falls_back_to_configured_memory(self) -> None:
        # An unreadable field must not refuse a job the old code would have allowed.
        part = _busy_partition(free_node_mem_mb=[], mem_detail=False)
        assert fit_blocker(_job(240), part) == ""

    def test_a_whole_node_job_still_uses_the_idle_only_figure(self) -> None:
        # An idle node must actually exist for an exclusive job to be placed on it —
        # setting the idle MEMORY on a partition with `idle_nodes=0` describes no
        # cluster, and the node-count test rightly answers "no room" there.
        part = _busy_partition(idle_nodes=1, idle_node_cpus=128)
        part.max_idle_node_mem_bytes = 250_000 * 1024**2
        part.max_idle_node_cpus = 128
        assert fit_blocker(_job(240, exclusive=True), part) == ""

    def test_an_exclusive_job_with_no_idle_node_is_still_refused(self) -> None:
        part = _busy_partition(idle_nodes=0, idle_node_cpus=0)
        assert fit_blocker(_job(240, exclusive=True), part) != ""

    def test_a_down_partition_is_still_down_first(self) -> None:
        assert fit_blocker(_job(240), _busy_partition(available=False)) == "down"

    def test_the_cpu_half_is_untouched(self) -> None:
        part = _busy_partition()
        assert fit_blocker(_job(1, req_cpus=999), part) == "node too small"
        # Within the 64 idle cores this fixture has — 100 would be "no room" on its
        # own merits, which says nothing about the memory half.
        assert fit_blocker(_job(1, req_cpus=50), part) == ""

    def test_the_gpu_reads_still_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpus, ok, _mem, _mem_ok = _fetch(monkeypatch, MIXED_SHAPES)
        assert ok is True
        assert gpus == {"both": [3]}, gpus

    def test_a_partition_with_no_memory_figure_at_all_is_unchanged(self) -> None:
        # A caller-built PartitionResources (no sinfo): 0 means unknown, and an
        # unknown must not become a refusal.
        part = PartitionResources("p", True, idle_nodes=2, cpus_idle=64)
        assert fit_blocker(_job(240), part) == ""

    def test_the_mock_path_still_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pending, "_is_mock", lambda: True)
        gpus, ok, mem, mem_ok = pending._fetch_free_gpus_by_partition()
        assert (gpus, ok, mem, mem_ok) == ({}, False, {}, False)
