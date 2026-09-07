"""The replay round-trip was pinned on a snapshot with the nested halves empty.

`from_dict`'s docstring names the contract: "Used by the node switcher to turn
another node's ``--once --json`` output back into a snapshot." So a field that
serialises but is not reconstructed does not raise -- it silently comes back at
its default, and the reader sees another node's frame with a figure quietly reset.

`test_remote.py::TestSnapshotSerialization::test_from_json_round_trip` already
asserts whole-object equality, which is the right shape, but its fixture has
**no `fabric` and no `interconnect`**, and its single GPU leaves every
later-added field at its default. So the collections where a `from_dict`
omission is most likely -- a list of lists (`interconnect.matrix`), lists of
floats (`nvlink_rx_gbps`/`_tx_gbps`), lists of strings (`devices`,
`throttle_reasons`) -- were never round-tripped with a value in them.

Measured before writing this: a fully populated snapshot flattens to **122**
leaves and round-trips exactly, so there is no defect here today. What was
missing is the guard. `NodeFabric` (8 fields) and `GpuInterconnect` (12) were
reachable by no round-trip test at all.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from slurmwatch import model
from slurmwatch.model import (
    CpuMetrics,
    GpuInterconnect,
    GpuMetrics,
    MemoryMetrics,
    NodeFabric,
    TelemetrySnapshot,
)


def _populated() -> TelemetrySnapshot:
    """A snapshot with every dataclass present and no field left at its default.

    Non-default values throughout on purpose: a field reconstructed as its
    default is indistinguishable from a correctly round-tripped one when the
    fixture's value IS the default, which is how the nested halves stayed
    uncovered.
    """
    return TelemetrySnapshot(
        timestamp=1_700_000_000.5,
        job_id="4711",
        step_id="0",
        hostname="cn042",
        elapsed_seconds=7200,
        cpu=CpuMetrics(
            cores_allocated=16,
            usage_ns=800_174_117,
            usage_percent=50.0,
            effective_cores=8.0,
            peak_effective_cores=9.5,
            source="mock",
        ),
        memory=MemoryMetrics(
            current_bytes=17_183_944_449,
            limit_bytes=68_719_476_736,
            peak_bytes=18_043_141_671,
            usage_percent=25.0,
            oom_guard_warning=True,
            oom_guard_critical=False,
            working_set_bytes=17_000_000_000,
            cache_bytes=183_944_449,
            peak_working_set_bytes=18_000_000_000,
            working_set_percent=24.7,
            source="sstat",
            cache_measured=True,
            peak_is_lifetime=True,
        ),
        gpus=[
            GpuMetrics(
                index=i,
                uuid=f"GPU-demo-{i}",
                name="NVIDIA A100-SXM4-80GB",
                utilization_percent=55.0 + i,
                memory_used_bytes=47_248_244_503 + i,
                memory_total_bytes=85_899_345_920,
                memory_utilization_percent=55.0 + i,
                power_watts=240.0 + i,
                temperature_celsius=65.0 + i,
                throttling=bool(i),
                process_utilization_percent=54.0 + i,
                process_memory_bytes=42_523_420_052 + i,
                utilization_available=True,
                utilization_supported=True,
                power_limit_watts=400.0,
                throttle_reasons=["sw_power_cap", "thermal"][: i + 1],
                memory_available=True,
                power_available=True,
                temperature_available=True,
                process_utilization_available=True,
                cuda_ordinal=i,
            )
            for i in range(2)
        ],
        fabric=NodeFabric(
            kind="InfiniBand",
            link_rate_gbps=200.0,
            link_rate_total_gbps=400.0,
            ports=2,
            rx_gbps=132.52,
            tx_gbps=165.0,
        ),
        interconnect=GpuInterconnect(
            # Types read off the dataclass, not guessed: `devices` is device
            # INDICES ("in matrix order") and `matrix` cells are STRINGS -- "self"
            # on the diagonal, "NV<k>" for k NVLinks, or a PCIe class -- while
            # `nvlink_version` is the marketing GENERATION as an int. mypy caught
            # all three when this fixture guessed otherwise.
            fabric="nvlink",
            nvlink_version=4,
            links_per_gpu=12,
            link_speed_gbps=25.0,
            per_gpu_gbps=600.0,
            nvswitch=True,
            devices=[0, 1],
            matrix=[["self", "NV12"], ["NV12", "self"]],
            nvlink_rx_gbps=[770.3, 12.5],
            nvlink_tx_gbps=[510.7, 9.25],
            pcie_rx_gbps=[3.5, 3.25],
            pcie_tx_gbps=[2.75, 2.5],
        ),
        # Every remaining field set away from its default too, so the round trip
        # cannot pass because a value happened to equal what a dropped field would
        # reconstruct as. `test_no_field_is_left_at_its_default` enforces this.
        node_count=4,
        node_index=1,
        gpu_count_requested=2,
        usage_age_seconds=0.25,
        usage_sampled=False,
        gpu_active_count=2,
        job_name="train-llama-8b",
        time_limit_seconds=86_400,
        partition="gpu-highend",
        owner="ada",
        account="rcc-staff",
        qos="normal",
        array_job_id="4700",
        array_task_id="11",
        remote=True,
        mock=True,
        gpu_monitoring_available=False,
        gpu_unavailable_reason="devices_denied",
        gpu_node_count=2,
        gpu_node_model="NVIDIA A100-SXM4-80GB",
        gpu_allocated_indices=[0, 1],
    )


def _leaves(obj: Any, prefix: str = "") -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            yield from _leaves(
                getattr(obj, field.name),
                (prefix + "." + field.name) if prefix else field.name,
            )
    elif isinstance(obj, list):
        yield prefix + "[]", len(obj)
        for index, item in enumerate(obj):
            yield from _leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, obj


class TestTheFixtureCanProveSomething:
    def test_it_reaches_every_dataclass_in_the_model(self) -> None:
        """Vacuity guard, and the one that would have caught the gap: a dataclass
        no fixture populates is a dataclass no round-trip test covers."""
        declared = {
            name
            for name in dir(model)
            if isinstance(getattr(model, name), type)
            and dataclasses.is_dataclass(getattr(model, name))
            and name != "JobContext"  # not part of a telemetry payload
        }
        reached = {type(snap).__name__ for snap, _ in [(_populated(), None)]}
        snap = _populated()
        reached |= {type(snap.cpu).__name__, type(snap.memory).__name__}
        reached |= {type(g).__name__ for g in snap.gpus}
        reached |= {type(snap.fabric).__name__, type(snap.interconnect).__name__}
        assert declared - reached == set(), sorted(declared - reached)

    def test_it_is_as_full_as_the_docstring_claims(self) -> None:
        leaves = dict(_leaves(_populated()))
        assert len(leaves) >= 120, len(leaves)

    def test_no_field_is_left_at_its_default(self) -> None:
        """The point of a populated fixture: a default value round-trips even when
        the field is dropped, so a fixture full of defaults proves nothing."""
        snap = _populated()
        bare = TelemetrySnapshot(
            timestamp=0.0,
            job_id="",
            step_id=None,
            hostname="",
            elapsed_seconds=0,
            cpu=CpuMetrics(cores_allocated=0, usage_ns=0, usage_percent=0.0),
            memory=MemoryMetrics(
                current_bytes=0,
                limit_bytes=0,
                peak_bytes=0,
                usage_percent=0.0,
                oom_guard_warning=False,
                oom_guard_critical=False,
            ),
        )
        same = [
            f.name
            for f in dataclasses.fields(TelemetrySnapshot)
            if f.name not in ("cpu", "memory", "gpus", "fabric", "interconnect")
            and getattr(snap, f.name) == getattr(bare, f.name)
        ]
        assert same == [], same


class TestAPopulatedSnapshotSurvivesReplay:
    def test_the_whole_object_round_trips(self) -> None:
        """The invariant the node switcher depends on."""
        snap = _populated()
        assert TelemetrySnapshot.from_json(snap.to_json()) == snap

    def test_every_leaf_survives_with_its_value(self) -> None:
        """Equality above is the assertion; this one names WHICH leaf moved when it
        fails, because a 122-field `==` failure says nothing useful."""
        snap = _populated()
        back = TelemetrySnapshot.from_json(snap.to_json())
        before, after = dict(_leaves(snap)), dict(_leaves(back))
        moved = {
            key: (before[key], after.get(key, "<ABSENT>"))
            for key in before
            if before[key] != after.get(key, "<ABSENT>")
        }
        assert moved == {}, moved

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            ("interconnect.matrix", [["self", "NV12"], ["NV12", "self"]]),
            ("interconnect.nvlink_rx_gbps", [770.3, 12.5]),
            ("interconnect.nvlink_tx_gbps", [510.7, 9.25]),
            ("interconnect.devices", [0, 1]),
        ],
    )
    def test_the_nested_collections_come_back_intact(self, path: str, value: Any) -> None:
        """The fields no round-trip test reached: a list of lists, two lists of
        floats, and a list of strings. `from_dict` rebuilds `interconnect` from a
        sub-mapping, which is where a dropped key would land."""
        back = TelemetrySnapshot.from_json(_populated().to_json())
        section, attr = path.split(".")
        assert getattr(getattr(back, section), attr) == value

    def test_a_per_gpu_list_field_comes_back_intact(self) -> None:
        back = TelemetrySnapshot.from_json(_populated().to_json())
        assert [g.throttle_reasons for g in back.gpus] == [
            ["sw_power_cap"],
            ["sw_power_cap", "thermal"],
        ]


class TestControls:
    def test_control_the_minimal_snapshot_still_round_trips(self) -> None:
        """CONTROL. The shape `test_remote.py` already pins -- no fabric, no
        interconnect, one defaulted GPU -- must keep working, so this file adds
        coverage rather than changing the contract. Holds in both states."""
        snap = TelemetrySnapshot(
            timestamp=1.0,
            job_id="1",
            step_id=None,
            hostname="n",
            elapsed_seconds=1,
            cpu=CpuMetrics(cores_allocated=4, usage_ns=10**9, usage_percent=99.0),
            memory=MemoryMetrics(
                current_bytes=1,
                limit_bytes=2,
                peak_bytes=1,
                usage_percent=50.0,
                oom_guard_warning=False,
                oom_guard_critical=False,
            ),
        )
        assert TelemetrySnapshot.from_json(snap.to_json()) == snap

    def test_control_an_absent_section_stays_absent(self) -> None:
        """CONTROL. `fabric` and `interconnect` are optional, and reconstructing
        them as empty objects instead of `None` would be its own defect -- the UI
        branches on their absence."""
        snap = TelemetrySnapshot(
            timestamp=1.0,
            job_id="1",
            step_id=None,
            hostname="n",
            elapsed_seconds=1,
            cpu=CpuMetrics(cores_allocated=1, usage_ns=0, usage_percent=0.0),
            memory=MemoryMetrics(
                current_bytes=0,
                limit_bytes=0,
                peak_bytes=0,
                usage_percent=0.0,
                oom_guard_warning=False,
                oom_guard_critical=False,
            ),
        )
        back = TelemetrySnapshot.from_json(snap.to_json())
        assert back.fabric is None and back.interconnect is None
        assert back.gpus == []

    def test_control_an_unknown_key_is_still_ignored(self) -> None:
        """CONTROL. "Unknown keys are ignored so a small version skew between nodes
        can't crash the parse." Holds in both states."""
        import copy
        import json

        clean = json.loads(_populated().to_json())
        extra = copy.deepcopy(clean)
        extra["a_field_from_a_newer_build"] = 1
        extra["cpu"]["also_new"] = 2
        # Compared against the parse of the SAME payload without the extra keys,
        # not against `_populated()`: an earlier version did the latter, so a
        # neuter that dropped a nested field reddened this for the round trip's
        # reason rather than its own and stopped being a control.
        assert TelemetrySnapshot.from_dict(extra) == TelemetrySnapshot.from_dict(clean)
