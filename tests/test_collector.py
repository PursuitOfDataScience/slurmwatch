from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from slurmwatch import collector as _collector_mod
from slurmwatch.collector import (
    TelemetryCollector,
    _any_launcher_pid,
    _gpu_is_active,
    _read_meminfo_total,
    _read_pid_comm,
)
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import CpuMetrics, GpuMetrics, JobContext, MemoryMetrics, TelemetrySnapshot


@pytest.fixture
def job_ctx() -> JobContext:
    return JobContext(
        job_id="12345",
        username="testuser",
        partition="gpu",
        nodelist="cn001",
        hostname="cn001",
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024 * 1024 * 1024,
        gpu_count_requested=2,
        gpu_indices=[0, 1],
        step_id="0",
        uid=1001,
        job_start_time=time.time() - 3600,
    )


class TestLauncherDetection:
    """Best-effort 'a new srun/mpirun is stuck behind our monitor step' signal."""

    def test_detects_top_level_launcher_clients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        comms = {1: "python", 2: "mpirun", 3: "bash"}
        monkeypatch.setattr(_collector_mod, "_read_pid_comm", lambda pid: comms.get(pid, ""))
        assert _any_launcher_pid({1, 3}) is False
        assert _any_launcher_pid({1, 2, 3}) is True

    def test_ignores_running_step_daemons(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # hydra_pmi_proxy / orted only exist once a step is RUNNING, so seeing them
        # means the launch succeeded — never flag those as a stuck client.
        comms = {5: "hydra_pmi_proxy", 6: "orted", 7: "prted"}
        monkeypatch.setattr(_collector_mod, "_read_pid_comm", lambda pid: comms.get(pid, ""))
        assert _any_launcher_pid({5, 6, 7}) is False

    def test_read_pid_comm_missing_pid_is_empty(self) -> None:
        assert _read_pid_comm(2_147_483_000) == ""

    def test_detection_is_gated_by_env(
        self, job_ctx: JobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SLURMWATCH_MONITOR_STEP", raising=False)
        assert TelemetryCollector(job_ctx, SlurmwatchConfig())._detect_launchers is False
        monkeypatch.setenv("SLURMWATCH_MONITOR_STEP", "1")
        assert TelemetryCollector(job_ctx, SlurmwatchConfig())._detect_launchers is True


@pytest.fixture
def mock_job_ctx() -> JobContext:
    return JobContext(
        job_id="12345",
        username="testuser",
        partition="gpu",
        nodelist="cn001",
        hostname="cn001",
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024 * 1024 * 1024,
        gpu_count_requested=2,
        gpu_indices=[0, 1],
        step_id="0",
        uid=1001,
        job_start_time=time.time() - 3600,
        job_name="argonne35-pretrain",
    )


class TestCollectorMockMode:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_collector_start_stop(self, mock_job_ctx: JobContext) -> None:
        config = SlurmwatchConfig(poll_interval=0.1)
        collector = TelemetryCollector(mock_job_ctx, config)
        await collector.start()
        await asyncio.sleep(0.3)
        assert collector._task is not None
        assert not collector._task.done()
        await collector.stop()
        assert collector._task.done()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_collector_produces_snapshot(self, mock_job_ctx: JobContext) -> None:
        config = SlurmwatchConfig(poll_interval=0.1)
        collector = TelemetryCollector(mock_job_ctx, config)
        await collector.start()
        try:
            snapshot = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
            assert isinstance(snapshot, TelemetrySnapshot)
            assert snapshot.job_id == "12345"
            # The collector must COPY the name onto every snapshot, not just carry it
            # on the context — otherwise --json/--log/the node switcher all lose it
            # while the plumbing tests (which hand-set the field) still pass.
            assert snapshot.job_name == "argonne35-pretrain"
            assert snapshot.cpu.cores_allocated == 16
            assert snapshot.hostname == "cn001"
            assert 0 <= snapshot.cpu.usage_percent <= 100
            assert snapshot.memory.limit_bytes == 64 * 1024 * 1024 * 1024
            assert snapshot.cpu.effective_cores > 0
        finally:
            await collector.stop()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_collector_multiple_snapshots(self, mock_job_ctx: JobContext) -> None:
        config = SlurmwatchConfig(poll_interval=0.1)
        collector = TelemetryCollector(mock_job_ctx, config)
        await collector.start()
        try:
            snap1 = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
            snap2 = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
            assert snap2.timestamp >= snap1.timestamp
            assert snap2.cpu.effective_cores >= 0
        finally:
            await collector.stop()

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_mock_snapshot_for_node_stamps_each_node(self) -> None:
        # Demo node-switching: mock_snapshot_for_node synthesizes a frame stamped
        # for the requested node (no srun), so switching in --demo is instant and
        # never triggers the "still reaching / unreachable" watchdog on a fake node.
        from slurmwatch import slurm

        ctx = slurm._make_mock_job_context("12345")
        assert len(ctx.nodelist_resolved) >= 2  # multi-node, so a switch is possible
        collector = TelemetryCollector(ctx)
        assert collector.is_mock is True
        for i, node in enumerate(ctx.nodelist_resolved):
            snap = collector.mock_snapshot_for_node(node)
            assert snap.hostname == node
            assert snap.node_index == i
            assert snap.node_count == len(ctx.nodelist_resolved)

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_mock_memory_never_trips_oom_warning(self) -> None:
        # The demo must never fire a (false) amber "MEMORY nn% of limit" alarm —
        # the mock memory plateau stays below the 85% OOM-warn threshold.
        from slurmwatch import slurm

        collector = TelemetryCollector(slurm._make_mock_job_context("12345"))
        collector._mock_start = time.monotonic() - 300  # well past the ramp/plateau
        mem = collector._collect_memory()
        assert mem.usage_percent < 85
        assert mem.oom_guard_warning is False
        assert mem.oom_guard_critical is False


class TestSnapshotSerialization:
    def test_snapshot_json(self) -> None:
        snap = _make_test_snapshot()
        j = snap.to_json()
        parsed = __import__("json").loads(j)
        assert parsed["job_id"] == "12345"
        assert "A100-SXM4-40GB" in j
        assert "effective_cores" in j
        assert "working_set_bytes" in j
        assert "gpu_count_requested" in j
        assert "gpu_active_count" in j

    def test_the_json_says_unknown_even_if_the_object_says_zero(self) -> None:
        """Assert on the RAW payload, not on a round trip.

        The reader re-derives this, so a round-trip test passes whether or not the
        WRITER is honest — and the wire is what another tool reads. A snapshot whose
        field still holds a stale summed count (a replayed log line, an older
        producer) must not publish it as a measurement.
        """
        snap = _make_test_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 4
        snap.gpu_monitoring_available = False
        snap.gpu_active_count = 0  # what a build predating the distinction would set
        assert json.loads(snap.to_json())["gpu_active_count"] is None

    def test_a_payload_from_an_older_build_is_normalised_on_read(self) -> None:
        """The reader cannot trust a 0 it did not compute.

        A hop is mixed-version by nature: the node runs the site's module while the
        login side runs a newer wheel. Hand-built dict on purpose — routing it through
        ``to_json`` first would launder the very field under test.
        """
        payload = {
            "timestamp": 1.0,
            "job_id": "1",
            "hostname": "n",
            "elapsed_seconds": 1,
            "cpu": {"cores_allocated": 1, "usage_ns": 0, "usage_percent": 0.0},
            "memory": {
                "current_bytes": 0,
                "limit_bytes": 1,
                "peak_bytes": 0,
                "usage_percent": 0.0,
                "oom_guard_warning": False,
                "oom_guard_critical": False,
            },
            "gpus": [],
            "gpu_count_requested": 4,
            "gpu_active_count": 0,
            "gpu_monitoring_available": False,
            "gpu_unavailable_reason": "devices_denied",
        }
        assert TelemetrySnapshot.from_dict(payload).gpu_active_count is None
        # An explicit null survives where the unreadable rule does NOT fire — a
        # payload that says "unknown" is not silently rounded down to zero. Requested
        # 0 so the rule above cannot be what produces the None.
        explicit_null = {**payload, "gpu_count_requested": 0, "gpu_active_count": None}
        assert TelemetrySnapshot.from_dict(explicit_null).gpu_active_count is None
        # ... and a CPU-only job keeps its measured zero: no GPU was asked for, so
        # "none active" is a fact. Without this the reader calls every CPU job unknown.
        cpu_only = {**payload, "gpu_count_requested": 0, "gpu_unavailable_reason": "no_devices"}
        assert TelemetrySnapshot.from_dict(cpu_only).gpu_active_count == 0

    def test_snapshot_csv_row(self) -> None:
        snap = _make_test_snapshot()
        row = snap.to_csv_row()
        row_str = ",".join(row)
        assert "12345" in row_str
        assert "45.50" in row_str or "45.5" in row_str

    def test_gpu_unavailable_reason_survives_json_and_csv(self) -> None:
        """The CAUSE must reach a consumer, not just the boolean.

        A right-sizing script reading ``gpu_monitoring_available=0`` cannot otherwise
        tell "this node has no GPU, drop the request" from "the GPUs are allocated
        and busy, I just could not read them from here" — opposite advice from the
        same row. Same reason ``gpu_monitoring_available`` itself was added to CSV.
        """
        snap = _make_test_snapshot()
        snap.gpus = []
        snap.gpu_count_requested = 2
        snap.gpu_monitoring_available = False
        snap.gpu_unavailable_reason = "devices_denied"
        snap.gpu_node_count = 4
        snap.gpu_node_model = "NVIDIA A100-PCIE-40GB"
        snap.gpu_allocated_indices = [0, 2]

        back = TelemetrySnapshot.from_json(snap.to_json())
        assert back.gpu_unavailable_reason == "devices_denied"
        # "0 of 2 active" about cards nothing could open is a measurement the reader
        # would act on. Re-derived on READ as well as write, so a snapshot forwarded
        # by a node running an older build is normalised here rather than believed.
        assert back.gpu_active_count is None
        assert back.gpu_node_count == 4
        assert back.gpu_node_model == "NVIDIA A100-PCIE-40GB"
        assert back.gpu_allocated_indices == [0, 2]

        header = TelemetrySnapshot.csv_header(max_gpus=0)
        row = snap.to_csv_row(max_gpus=0)
        assert len(header) == len(row), "header/row drifted apart"
        cells = dict(zip(header, row, strict=True))
        assert cells["gpu_unavailable_reason"] == "devices_denied"
        assert cells["gpu_active_count"] == "", "an unread device set is not 0 active"
        assert cells["gpu_node_count"] == "4"
        assert cells["gpu_node_model"] == "NVIDIA A100-PCIE-40GB"
        assert cells["gpu_allocated_indices"] == "0;2"

    def test_snapshot_from_an_older_node_defaults_the_new_gpu_fields(self) -> None:
        """A node streaming a pre-change build omits the fields; that must not throw.

        The node-switcher parses JSONL produced by whatever slurmwatch is installed
        on the *other* node, so a mixed-version job is a normal state.
        """
        payload = json.loads(_make_test_snapshot().to_json())
        for key in (
            "gpu_unavailable_reason",
            "gpu_node_count",
            "gpu_node_model",
            "gpu_allocated_indices",
        ):
            payload.pop(key, None)
        back = TelemetrySnapshot.from_dict(payload)
        assert back.gpu_unavailable_reason == ""
        assert back.gpu_node_count == 0
        assert back.gpu_node_model == ""
        assert back.gpu_allocated_indices == []

    def test_an_older_payload_never_claims_its_peak_is_a_lifetime_figure(self) -> None:
        """SW-3's rule applied to the new field: a payload that does not state its
        provenance must not have provenance invented for it. A build predating
        `peak_is_lifetime` said nothing about which reading its `peak_bytes` was, so
        the reader must not label it "lifetime" on that build's behalf."""
        payload = json.loads(_make_test_snapshot().to_json())
        assert "peak_is_lifetime" in payload["memory"], "current builds state it"
        payload["memory"].pop("peak_is_lifetime")
        assert TelemetrySnapshot.from_dict(payload).memory.peak_is_lifetime is False
        # And a payload that DOES state it is believed, either way.
        for stated in (True, False):
            payload["memory"]["peak_is_lifetime"] = stated
            assert TelemetrySnapshot.from_dict(payload).memory.peak_is_lifetime is stated

    def test_csv_header_length(self) -> None:
        header = TelemetrySnapshot.csv_header(max_gpus=2)
        assert "timestamp" in header
        assert "gpu_0_util_percent" in header
        assert "gpu_1_util_percent" in header
        assert "gpu_2_util_percent" not in header
        assert "cpu_effective_cores" in header
        assert "mem_working_set_bytes" in header
        assert "gpu_count_requested" in header
        assert "gpu_active_count" in header
        # #38: node identity + the remote-estimate flag are in the CSV, so a
        # multi-node log is interpretable and matches the JSON output.
        assert "node_count" in header
        assert "node_index" in header
        assert "remote" in header

    def test_csv_row_matches_header_length(self) -> None:
        snap = _make_test_snapshot()
        row = snap.to_csv_row()
        header = TelemetrySnapshot.csv_header(max_gpus=8)
        assert len(row) == len(header)

    def test_csv_row_padded_for_less_gpus(self) -> None:
        snap = _make_test_snapshot()
        snap.gpus = []
        row = snap.to_csv_row()
        header = TelemetrySnapshot.csv_header(max_gpus=8)
        assert len(row) == len(header)
        # Derived from the schema, not hardcoded, so adding a column can't rot this.
        fixed = len(TelemetrySnapshot.csv_header(0))
        assert len(row) == fixed + 8 * TelemetrySnapshot._GPU_COLS

    def test_csv_row_has_common_columns(self) -> None:
        snap = _make_test_snapshot()
        row = snap.to_csv_row()
        assert len(row) >= 21
        assert row[0].startswith("1234567890")
        assert row[1] == "12345"

    def test_csv_columns_size_to_gpu_count(self) -> None:
        # #38: a 16-GPU node isn't clipped at 8 — the header + row size to the
        # requested count, and every device is present.
        snap = _make_test_snapshot()
        snap.gpus = snap.gpus * 16  # 16 device rows
        header = TelemetrySnapshot.csv_header(max_gpus=16)
        row = snap.to_csv_row(max_gpus=16)
        fixed = len(TelemetrySnapshot.csv_header(0))
        assert len(row) == len(header) == fixed + 16 * TelemetrySnapshot._GPU_COLS
        assert "gpu_15_index" in header

    def test_csv_gpu_count_is_real_and_signals_truncation(self) -> None:
        # The gpu_count column reports the REAL device count, not a value capped
        # at the column width — so a row truncated to the default 8 groups still
        # advertises that 16 devices existed (#38).
        snap = _make_test_snapshot()
        snap.gpus = snap.gpus * 16
        header = TelemetrySnapshot.csv_header()  # default 8 groups
        row = snap.to_csv_row()  # default 8 groups
        gpu_count = row[header.index("gpu_count")]
        assert gpu_count == "16"  # not clipped to 8
        assert len(row) == len(header)  # still self-consistent

    def test_csv_node_columns_carry_identity(self) -> None:
        snap = _make_test_snapshot()
        snap.node_count = 4
        snap.node_index = 2
        snap.remote = True
        header = TelemetrySnapshot.csv_header()
        row = snap.to_csv_row()
        assert row[header.index("node_count")] == "4"
        assert row[header.index("node_index")] == "2"
        assert row[header.index("remote")] == "1"

    def test_working_set_percent_in_json_and_csv(self) -> None:
        # Note 1: the cache-EXCLUDED working-set % is emitted so a --json/CSV consumer
        # sizing --mem sees it, not only the cache-inclusive usage_percent.
        import json

        snap = _make_test_snapshot()
        snap.memory.working_set_percent = 42.5
        assert json.loads(snap.to_json())["memory"]["working_set_percent"] == 42.5
        header = TelemetrySnapshot.csv_header()
        row = snap.to_csv_row()
        assert "mem_working_set_percent" in header
        assert row[header.index("mem_working_set_percent")] == "42.50"

    def test_job_name_in_json_and_csv(self) -> None:
        # The name belongs wherever the id is: a --json capture or a CSV log has to say
        # WHICH experiment it measured, not just which numeric record. Beside job_id.
        snap = _make_test_snapshot()
        snap.job_name = "argonne35-pretrain"
        header = TelemetrySnapshot.csv_header()
        row = snap.to_csv_row()
        assert header.index("job_name") == header.index("job_id") + 1
        assert row[header.index("job_name")] == "argonne35-pretrain"
        assert json.loads(snap.to_json())["job_name"] == "argonne35-pretrain"
        # It survives the node switcher's JSON round-trip, and a node running an older
        # build (no such key) parses as "" instead of crashing.
        assert TelemetrySnapshot.from_json(snap.to_json()).job_name == "argonne35-pretrain"
        payload = json.loads(snap.to_json())
        del payload["job_name"]
        assert TelemetrySnapshot.from_dict(payload).job_name == ""

    def test_cuda_ordinal_in_json_and_csv(self) -> None:
        # C2: `index` is NVML's device index (what nvidia-smi prints); the ordinal is
        # what the job's own code addresses (cuda:0). They differ on a cluster without
        # device-cgroup isolation, so both have to reach machine output by NAME.
        snap = _make_test_snapshot()
        snap.gpus[0].index = 2  # node-global index
        snap.gpus[0].cuda_ordinal = 0  # ...but the job's first device
        header = TelemetrySnapshot.csv_header()
        row = snap.to_csv_row()
        assert "gpu_0_cuda_ordinal" in header
        assert row[header.index("gpu_0_cuda_ordinal")] == "0"
        assert row[header.index("gpu_0_index")] == "2"
        assert json.loads(snap.to_json())["gpus"][0]["cuda_ordinal"] == 0

    def test_gpu_util_flags_in_csv(self) -> None:
        # P6/A7: a CSV consumer must be able to tell "util unreadable" from a genuine
        # 0%, and (A7) an unsupported device from a transient miss — so both flag
        # columns are part of the schema by NAME, not just by column count.
        snap = _make_test_snapshot()
        snap.gpus[0].utilization_available = False
        snap.gpus[0].utilization_supported = False
        header = TelemetrySnapshot.csv_header()
        row = snap.to_csv_row()
        assert "gpu_0_util_available" in header and "gpu_0_util_supported" in header
        assert row[header.index("gpu_0_util_available")] == "0"
        assert row[header.index("gpu_0_util_supported")] == "0"
        snap.gpus[0].utilization_available = True
        snap.gpus[0].utilization_supported = True
        row = snap.to_csv_row()
        assert row[header.index("gpu_0_util_available")] == "1"
        assert row[header.index("gpu_0_util_supported")] == "1"

    def test_cpu_peak_in_json_and_csv(self) -> None:
        # The high-water mark to size --cpus-per-task against reached --json (via
        # asdict) but was MISSING from the CSV schema, so a CSV consumer couldn't do
        # the tool's headline right-sizing — the same gap Note 1 closed for memory.
        import json

        snap = _make_test_snapshot()
        snap.cpu.effective_cores = 6.25
        snap.cpu.peak_effective_cores = 9.5
        assert json.loads(snap.to_json())["cpu"]["peak_effective_cores"] == 9.5
        header = TelemetrySnapshot.csv_header()
        row = snap.to_csv_row()
        assert "cpu_peak_effective_cores" in header
        assert row[header.index("cpu_peak_effective_cores")] == "9.50"
        # It sits beside the live figure, not in place of it.
        assert row[header.index("cpu_effective_cores")] == "6.25"

    def test_csv_round_trips_through_from_dict(self) -> None:
        # The node switcher rebuilds a snapshot from another node's --once --json, so
        # every field a peak/percent consumer reads must survive the round trip.
        snap = _make_test_snapshot()
        snap.cpu.peak_effective_cores = 7.5
        snap.memory.working_set_percent = 33.25
        back = TelemetrySnapshot.from_json(snap.to_json())
        assert back.cpu.peak_effective_cores == 7.5
        assert back.memory.working_set_percent == 33.25

    def test_interconnect_json_round_trip(self) -> None:
        from slurmwatch.model import GpuInterconnect

        snap = _make_test_snapshot()
        snap.interconnect = GpuInterconnect(
            fabric="nvlink",
            nvlink_version=3,
            links_per_gpu=12,
            link_speed_gbps=25.0,
            per_gpu_gbps=600.0,
            nvswitch=True,
            devices=[0, 1],
            matrix=[["self", "NV12"], ["NV12", "self"]],
            nvlink_rx_gbps=[100.0, 120.0],
            nvlink_tx_gbps=[90.0, 110.0],
            pcie_rx_gbps=[3.0, 4.0],
            pcie_tx_gbps=[2.0, 5.0],
        )
        back = TelemetrySnapshot.from_json(snap.to_json())
        assert back.interconnect is not None
        ic = back.interconnect
        assert ic.fabric == "nvlink"
        assert ic.nvlink_version == 3
        assert ic.per_gpu_gbps == 600.0
        assert ic.nvswitch is True
        assert ic.matrix == [["self", "NV12"], ["NV12", "self"]]
        assert ic.nvlink_rx_gbps == [100.0, 120.0]
        assert ic.pcie_tx_gbps == [2.0, 5.0]

    def test_interconnect_absent_round_trips_as_none(self) -> None:
        # A CPU-only / single-GPU / off-node snapshot carries no interconnect; it
        # must survive a JSON round-trip as None, never a crash.
        snap = _make_test_snapshot()
        assert snap.interconnect is None
        assert TelemetrySnapshot.from_json(snap.to_json()).interconnect is None

    def test_interconnect_missing_key_parses(self) -> None:
        # A peer node running an older sw may omit the key entirely (the node
        # switcher parses another node's --once --json); from_dict must tolerate it.
        import json

        payload = json.loads(_make_test_snapshot().to_json())
        payload.pop("interconnect", None)
        assert TelemetrySnapshot.from_dict(payload).interconnect is None


def _make_test_snapshot() -> TelemetrySnapshot:
    from slurmwatch.model import CpuMetrics, GpuMetrics, MemoryMetrics

    return TelemetrySnapshot(
        timestamp=1234567890.0,
        job_id="12345",
        step_id="0",
        hostname="cn001",
        elapsed_seconds=3600,
        cpu=CpuMetrics(
            cores_allocated=16,
            usage_ns=1_000_000_000,
            usage_percent=45.5,
            effective_cores=7.3,
        ),
        memory=MemoryMetrics(
            current_bytes=30 * 1024**3,
            limit_bytes=64 * 1024**3,
            peak_bytes=40 * 1024**3,
            usage_percent=46.9,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=25 * 1024**3,
            cache_bytes=5 * 1024**3,
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


class TestRealCgroupCollector:
    @pytest.fixture
    def cgroup_job_ctx(self, fake_cgroup_v2_job: Path) -> JobContext:
        mem_limit = 8 * 1024**3
        return JobContext(
            job_id="12345",
            username="testuser",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=16,
            mem_limit_bytes=mem_limit,
            gpu_count_requested=0,
            gpu_indices=[],
            step_id="0",
            uid=1001,
            job_start_time=1000.0,
            cgroup_v2_path=str(fake_cgroup_v2_job),
        )

    def test_collect_memory_from_cgroup(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(cgroup_job_ctx)
        mem = collector._collect_memory()
        assert mem.current_bytes == 2 * 1024**3
        assert mem.limit_bytes == 8 * 1024**3
        assert mem.peak_bytes == 4 * 1024**3
        assert mem.working_set_bytes > 0
        assert mem.cache_bytes > 0
        assert mem.usage_percent == 25.0

    def test_v2_memory_falls_back_to_proc_rss_when_controller_absent(
        self, cgroup_job_ctx: JobContext, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # F4: no memory.current (memory controller not delegated into the job
        # cgroup) must fall back to summing the job's /proc RSS, not report 0.
        (fake_cgroup_v2_job / "memory.current").unlink()
        collector = TelemetryCollector(cgroup_job_ctx)
        monkeypatch.setattr(collector, "_proc_rss_bytes", lambda: 3 * 1024**3)
        mem = collector._collect_memory()
        assert mem.current_bytes == 3 * 1024**3

    def test_proc_rss_bytes_sums_real_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The fallback helper reads real /proc/<pid>/statm; our own pid has RSS > 0.
        collector = TelemetryCollector(_min_ctx())
        monkeypatch.setattr(collector, "_get_job_pids", lambda: {os.getpid()})
        assert collector._proc_rss_bytes() > 0

    def test_memory_limit_caps_huge_cgroup_to_allocation(
        self, cgroup_job_ctx: JobContext, fake_cgroup_v2_job: Path
    ) -> None:
        # When the cgroup's memory.max is the whole node's RAM (e.g.
        # ConstrainRAMSpace=no), report the memory Slurm allocated (8 GiB) rather
        # than "196 of 200 GiB requested".
        (fake_cgroup_v2_job / "memory.max").write_text(str(400 * 1024**3))
        mem = TelemetryCollector(cgroup_job_ctx)._collect_memory()
        assert mem.limit_bytes == 8 * 1024**3  # the allocation, not node RAM
        assert mem.usage_percent == 25.0  # current 2 GiB of the 8 GiB allocation

    def test_memory_guard_uses_tighter_cgroup_limit(
        self, cgroup_job_ctx: JobContext, fake_cgroup_v2_job: Path
    ) -> None:
        # F5: when the cgroup enforces a *lower* ceiling than the allocation, the
        # kernel OOM-kills at the cgroup limit, so report and guard against that
        # tighter value. Otherwise the guard measures % against a too-generous
        # allocation and stays silent while the job is about to be killed.
        (fake_cgroup_v2_job / "memory.max").write_text(str(6 * 1024**3))
        (fake_cgroup_v2_job / "memory.current").write_text(str(int(5.7 * 1024**3)))
        mem = TelemetryCollector(cgroup_job_ctx)._collect_memory()
        assert mem.limit_bytes == 6 * 1024**3  # the real enforced ceiling, not 8
        # ~5.6 GiB working set of 6 GiB -> critical; against the 8 GiB allocation
        # it would be ~70% and the guard would have stayed silent.
        assert mem.oom_guard_critical

    def test_guard_ignores_allocation_overrun_on_unconstrained_node(
        self, cgroup_job_ctx: JobContext, fake_cgroup_v2_job: Path
    ) -> None:
        # P3, the mirror image of the F5 test above: with ConstrainRAMSpace=no the
        # cgroup ceiling is the whole node's RAM, so a job that merely exceeds its
        # REQUEST is nowhere near the kernel's kill point and must NOT trip a
        # "near limit, raise --mem" critical. The guard therefore measures against
        # the real cgroup limit, not the reported min(alloc, cgroup).
        (fake_cgroup_v2_job / "memory.max").write_text(str(400 * 1024**3))  # node RAM
        (fake_cgroup_v2_job / "memory.current").write_text(str(12 * 1024**3))  # > 8 GiB alloc
        (fake_cgroup_v2_job / "memory.stat").write_text("inactive_file 0\n")
        mem = TelemetryCollector(cgroup_job_ctx)._collect_memory()
        assert mem.limit_bytes == 8 * 1024**3  # still reported against the allocation
        assert mem.working_set_bytes == 12 * 1024**3  # genuinely over the request
        assert mem.oom_guard_warning is False  # 12 GiB of 400 GiB: nowhere near OOM
        assert mem.oom_guard_critical is False
        # And the percent a consumer sizes --mem from stays clamped, never >100.
        assert mem.usage_percent == 100.0
        assert mem.working_set_percent == 100.0

    def test_guard_uses_node_ram_when_cgroup_v2_declares_no_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cgroup_job_ctx: JobContext,
        fake_cgroup_v2_job: Path,
    ) -> None:
        # cgroup v2 spells "no enforced limit" as the literal string "max"; v1 spells it
        # as a huge sentinel. Only the sentinel was recognised (via the `> 10**16`
        # branch), so on v2 the guard basis silently stayed the Slurm REQUEST and a job
        # merely over its own request tripped a false "near limit, raise --mem" critical
        # — precisely what P3 removed, and the OPPOSITE verdict from v1 for the identical
        # machine state. Same physical situation, so both branches must agree.
        node_ram = 400 * 1024**3
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: node_ram)
        (fake_cgroup_v2_job / "memory.max").write_text("max\n")
        # 7.9 of the 8 GiB allocation: 98% of the REQUEST, 2% of the node.
        (fake_cgroup_v2_job / "memory.current").write_text(str(7900 * 1024**2))
        (fake_cgroup_v2_job / "memory.stat").write_text("inactive_file 0\n")
        mem = TelemetryCollector(cgroup_job_ctx)._collect_memory()
        assert mem.limit_bytes == 8 * 1024**3  # display still reads against the request
        assert mem.oom_guard_warning is False  # but the kernel kills at 400 GiB, not 8
        assert mem.oom_guard_critical is False

    @pytest.mark.parametrize("how", ["absent", "unreadable"])
    def test_guard_uses_node_ram_when_the_v1_limit_file_cannot_be_read(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, how: str
    ) -> None:
        # The v1 sibling of the test above, for the THIRD way "no enforced cap" can
        # arrive: the file does not answer at all. v1 spells unlimited as a huge
        # sentinel, which `> 10**16` catches — but `_read_int_file` returns None for
        # both an absent and an unreadable file, and `... or limit_bytes` then
        # substituted the Slurm REQUEST, so `cgroup_limit` became the allocation and
        # the guard measured the job against its own request. Measured: 7.9 of an
        # 8 GiB allocation reported oom_guard_critical=True on v1 while the identical
        # v2 shape (memory.max absent) correctly reported False. Same physical
        # situation, so both branches must agree.
        node_ram = 400 * 1024**3
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: node_ram)
        v1 = tmp_path / how
        v1.mkdir()
        (v1 / "memory.usage_in_bytes").write_text(str(7900 * 1024**2))
        (v1 / "memory.stat").write_text("total_inactive_file 0\ntotal_active_file 0\n")
        if how == "unreadable":
            # A path that exists but cannot be read as an int — the same
            # `_read_int_file() is None` the EACCES case produces, without depending
            # on the euid the suite happens to run as (mode 000 is readable by root).
            (v1 / "memory.limit_in_bytes").mkdir()
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v1_mem_path=str(v1))
        )._collect_memory()
        assert mem.limit_bytes == 8 * 1024**3  # display still reads against the request
        assert mem.oom_guard_warning is False  # nothing caps the cgroup: 8 of 400 GiB
        assert mem.oom_guard_critical is False

    def test_a_real_v1_cap_at_the_allocation_still_trips_the_guard(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The control on the fix above, and not its mirror.

        Here ``memory.limit_in_bytes`` IS readable and IS the allocation, so the
        kernel really does kill at 8 GiB and 98% of it must warn AND crit. A fix that
        reaches for node RAM whenever the reported limit equals the allocation — or
        that simply stops consulting the file — passes the test above and silences the
        one alarm this guard exists to raise.
        """
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: 400 * 1024**3)
        v1 = tmp_path / "enforced"
        v1.mkdir()
        (v1 / "memory.usage_in_bytes").write_text(str(7900 * 1024**2))
        (v1 / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
        (v1 / "memory.stat").write_text("total_inactive_file 0\ntotal_active_file 0\n")
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v1_mem_path=str(v1))
        )._collect_memory()
        assert mem.limit_bytes == 8 * 1024**3
        assert mem.oom_guard_warning is True
        assert mem.oom_guard_critical is True

    def test_peak_and_percent_exclude_page_cache(
        self, cgroup_job_ctx: JobContext, fake_cgroup_v2_job: Path
    ) -> None:
        # P2 + Note 1 at the COMPUTATION (not the JSON plumbing): a cache-heavy job
        # must not have its --mem sizing figures inflated by reclaimable page cache.
        # peak_working_set_bytes / working_set_percent are cache-EXCLUDED, while
        # peak_bytes stays the kernel's cache-INCLUSIVE lifetime total.
        limit = 8 * 1024**3
        (fake_cgroup_v2_job / "memory.current").write_text(str(6 * 1024**3))
        (fake_cgroup_v2_job / "memory.peak").write_text(str(7 * 1024**3))
        # 4 GiB of the 6 GiB "current" is reclaimable file cache -> 2 GiB working set.
        (fake_cgroup_v2_job / "memory.stat").write_text(
            f"inactive_file {3 * 1024**3}\nactive_file {1024**3}\n"
        )
        mem = TelemetryCollector(cgroup_job_ctx)._collect_memory()
        assert mem.cache_bytes == 4 * 1024**3
        assert mem.working_set_bytes == 2 * 1024**3
        # The sizing peak follows the WORKING SET, not the cache-inclusive counter.
        assert mem.peak_working_set_bytes == 2 * 1024**3
        assert mem.peak_bytes == 7 * 1024**3  # kernel lifetime total, cache included
        # ... and so does the sizing percent: 2/8 = 25%, not the 75% current/limit.
        assert mem.working_set_percent == 25.0
        assert mem.usage_percent == 75.0
        assert mem.limit_bytes == limit

    def test_working_set_peak_is_monotonic_across_polls(
        self, cgroup_job_ctx: JobContext, fake_cgroup_v2_job: Path
    ) -> None:
        # P2: the cache-excluded peak has no kernel counter behind it, so it must be
        # a running max — a working set that rises then falls keeps the high-water
        # mark (that's the number --mem is sized against).
        collector = TelemetryCollector(cgroup_job_ctx)
        (fake_cgroup_v2_job / "memory.stat").write_text("inactive_file 0\n")
        (fake_cgroup_v2_job / "memory.current").write_text(str(5 * 1024**3))
        assert collector._collect_memory().peak_working_set_bytes == 5 * 1024**3
        (fake_cgroup_v2_job / "memory.current").write_text(str(2 * 1024**3))
        mem = collector._collect_memory()
        assert mem.working_set_bytes == 2 * 1024**3  # live value dropped
        assert mem.peak_working_set_bytes == 5 * 1024**3  # peak did not

    def test_v1_working_set_excludes_page_cache(self, tmp_path: Path) -> None:
        # cgroup v1 (Midway3's version): memory.usage_in_bytes counts reclaimable
        # page cache, so the working set — and the OOM guard — must subtract the
        # inactive file cache. Regression: v1 set working_set == usage, firing
        # false CRITICAL alerts for data-loading jobs and reporting cache as 0.
        v1 = tmp_path / "memory" / "job_1"
        v1.mkdir(parents=True)
        (v1 / "memory.usage_in_bytes").write_text(str(50 * 1024**3))  # incl. cache
        (v1 / "memory.limit_in_bytes").write_text(str(50 * 1024**3))
        (v1 / "memory.max_usage_in_bytes").write_text(str(50 * 1024**3))
        (v1 / "memory.stat").write_text(
            f"total_cache {45 * 1024**3}\n"
            f"total_inactive_file {40 * 1024**3}\n"
            f"total_active_file {4 * 1024**3}\n"
            f"total_rss {5 * 1024**3}\n"
        )
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=50 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            cgroup_v1_mem_path=str(v1),
        )
        mem = TelemetryCollector(ctx)._collect_memory()
        assert mem.current_bytes == 50 * 1024**3
        # working set excludes reclaimable file cache (inactive + active): 50 - 44.
        assert mem.working_set_bytes == 6 * 1024**3
        assert mem.cache_bytes == 44 * 1024**3
        # Guard is driven by the working set (6/50 = 12%), so no false alarm.
        assert mem.oom_guard_warning is False
        assert mem.oom_guard_critical is False

    def test_v1_memory_falls_back_to_proc_rss_when_controller_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The v1 analogue of the v2 F4 test above: a v1 memory-controller cgroup
        # discovered but with no memory.usage_in_bytes file (controller not
        # actually delegated there) must fall back to /proc RSS, not report 0 —
        # the F4 fix was mirrored to the v2 branch only, not this one.
        v1 = tmp_path / "memory" / "job_1"
        v1.mkdir(parents=True)
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            cgroup_v1_mem_path=str(v1),
        )
        collector = TelemetryCollector(ctx)
        monkeypatch.setattr(collector, "_proc_rss_bytes", lambda: 3 * 1024**3)
        mem = collector._collect_memory()
        assert mem.current_bytes == 3 * 1024**3

    def test_collect_cpu_from_cgroup(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(cgroup_job_ctx)
        cpu = collector._collect_cpu()
        assert cpu.cores_allocated == 16
        assert isinstance(cpu.usage_percent, float)

    def test_get_job_pids_from_cgroup(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(cgroup_job_ctx)
        pids = collector._get_job_pids()
        assert 1000 in pids
        assert 1001 in pids

    def test_job_pids_exclude_the_monitor_itself(
        self, cgroup_job_ctx: JobContext, fake_cgroup_v2_job: Path
    ) -> None:
        # On a cpuacct-less cluster the /proc PID-sum IS the production CPU path, and
        # after an `srun --overlap` hop slurmwatch runs INSIDE the job's own cgroup — so
        # its pid really does appear in cgroup.procs. Counting it would bill the
        # monitor's CPU to the job, inflating effective_cores and permanently latching
        # peak_effective_cores: exactly the numbers a user sizes --cpus-per-task from.
        task_cg = fake_cgroup_v2_job / "step_0" / "user" / "task_0"
        task_cg.joinpath("cgroup.procs").write_text(f"1000\n1001\n{os.getpid()}\n")
        pids = TelemetryCollector(cgroup_job_ctx)._get_job_pids()
        assert {1000, 1001} <= pids
        assert os.getpid() not in pids

    def test_cpu_falls_back_to_proc_without_cpuacct(self) -> None:
        # On clusters that constrain jobs via cpuset only (no per-job cpuacct
        # cgroup, e.g. Midway3), CPU must be measured from /proc/<pid>/stat.
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            # No cgroup_v1_cpu_path / cgroup_v2_path set.
        )
        collector = TelemetryCollector(ctx)
        pids = {os.getpid()}
        # Two readings with a positive delta produce a real percentage. The rate
        # window uses a monotonic clock now, so simulate a 1s gap via _prev_timestamp.
        collector._collect_cpu(pids)
        collector._prev_cpu_ns = 0  # force a measurable delta on the next read
        collector._prev_timestamp = time.monotonic() - 1.0
        cpu = collector._collect_cpu(pids)
        assert cpu.usage_ns > 0
        assert cpu.usage_percent > 0.0

    def test_effective_cores_uncapped_reveals_oversubscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A3: on a ConstrainCores=no node a job can burn more CPU-time than its
        # allocation; effective_cores must REVEAL that (not clamp at cores) so the
        # "raise --cpus-per-task" signal survives. The bar percent stays clamped.
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=2,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        c = TelemetryCollector(ctx)
        monkeypatch.setattr(c, "_read_cpu_ns", lambda pids=None: 4_000_000_000)
        monkeypatch.setattr(time, "monotonic", lambda: 1001.0)
        c._prev_cpu_ns = 0
        c._prev_timestamp = 1000.0  # dt = 1.0s
        cpu = c._collect_cpu(set())
        assert cpu.effective_cores == 4.0  # 4 cores busy on a 2-core alloc, NOT capped
        assert cpu.usage_percent == 100.0  # the bar % is still clamped to 100

    def _rate_ctx(self) -> JobContext:
        return JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )

    def test_unreadable_cpu_frame_keeps_the_baseline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A frame where NO counter is readable used to overwrite a perfectly good
        # baseline with None, so the NEXT frame had nothing to difference against and
        # also read 0% — two consecutive frames of "0 cores" on a fully busy job. Every
        # counter it can come from is monotonic, so the baseline must simply be kept and
        # the next good read measures a correct two-interval rate.
        c = TelemetryCollector(self._rate_ctx())
        reads: list[int | None] = [1_000_000_000, None, 3_000_000_000]

        def next_read(pids: object = None) -> int | None:
            return reads.pop(0)

        monkeypatch.setattr(c, "_read_cpu_ns", next_read)
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        c._collect_cpu(set())  # seeds the baseline at 1.0 CPU-s
        monkeypatch.setattr(time, "monotonic", lambda: 1001.0)
        blind = c._collect_cpu(set())  # unreadable frame
        assert blind.effective_cores == 0.0
        # The emitted counter must report the last known value, not a literal 0 —
        # a CSV consumer differencing this column would otherwise see a fake
        # reset on exactly the frame the rate calc is careful to skip over.
        assert blind.usage_ns == 1_000_000_000
        assert c._prev_cpu_ns == 1_000_000_000, "a good baseline must survive a blind frame"
        monkeypatch.setattr(time, "monotonic", lambda: 1002.0)
        recovered = c._collect_cpu(set())
        # 2.0 CPU-s of work over the 2 s since the baseline = 1.0 core, not a second 0.0.
        assert recovered.effective_cores == 1.0

    def test_cpu_source_change_does_not_spike_the_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _read_cpu_ns can answer from three MUTUALLY NON-COMPARABLE counters (v2
        # cpu.stat, v1 cpuacct.usage, the /proc accumulator). Differencing one against
        # another produced effective_cores=982 on a 4-core job when a cgroup read failed
        # for one frame and the /proc accumulator (which only counts CPU seen since
        # attach) answered instead — and that spike latched into peak_effective_cores for
        # the rest of the session. A source change must re-seed, not subtract.
        c = TelemetryCollector(self._rate_ctx())
        plan = [
            ("v2", 1_000_000_000_000),  # cgroup counter: 1000 CPU-s since job start
            ("proc", 20_000_000_000),  # cgroup read failed; /proc accumulator: 20 CPU-s
            ("v2", 1_002_000_000_000),  # cgroup recovers — 982 s "ahead" of the /proc one
            ("v2", 1_004_000_000_000),  # and now two comparable reads in a row
        ]

        def next_read(pids: object = None) -> int:
            source, value = plan.pop(0)
            c._cpu_source = source
            return value

        monkeypatch.setattr(c, "_read_cpu_ns", next_read)
        # Frames 0-2 each re-seed (no baseline, then two source changes), so none of them
        # can invent a rate; frame 3 is the first pair drawn from the SAME counter.
        seen: list[float] = []
        for i, expected in enumerate((0.0, 0.0, 0.0, 2.0)):
            monkeypatch.setattr(time, "monotonic", lambda t=1000.0 + i: t)
            cpu = c._collect_cpu(set())
            seen.append(cpu.effective_cores)
            assert cpu.effective_cores == expected, f"frame {i}"
        # No frame ever carried the ~982-core delta, so there is nothing for the
        # monotonic peak to latch onto downstream.
        assert max(seen) == 2.0

    def test_tiny_dt_does_not_spike_effective_cores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A3 guard: an anomalously short window (dt below the floor) must NOT turn a
        # normal CPU delta into a huge uncapped rate — that spike would latch into
        # the monotonic peak. Below the floor the frame reads 0 and the baseline is
        # kept so the delta accumulates into the next adequately-spaced frame.
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        c = TelemetryCollector(ctx)  # poll_interval 0.5 -> min_dt 0.1
        monkeypatch.setattr(c, "_read_cpu_ns", lambda pids=None: 20_000_000)  # 0.02 CPU-s
        monkeypatch.setattr(time, "monotonic", lambda: 1000.001)
        c._prev_cpu_ns = 0
        c._prev_timestamp = 1000.0  # dt = 0.001 s, far below the 0.1 floor
        cpu = c._collect_cpu(set())
        assert cpu.effective_cores == 0.0  # no bogus spike (would have been ~20.0)
        assert c._prev_cpu_ns == 0  # baseline kept so the delta accumulates next tick

    def test_read_cpu_ns_none_without_source(self) -> None:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        collector = TelemetryCollector(ctx)
        assert collector._read_cpu_ns(set()) is None

    def test_proc_cpu_fallback_is_monotonic_across_child_churn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # audit-3 #2: on a cpuset-only node (no cpuacct cgroup), the /proc CPU
        # counter must ACCUMULATE — a child that exits between polls must not erase
        # its ticks (which made a busy fork-churn job read 0%). The value must only
        # ever climb, like the cgroup counter it substitutes for.
        from slurmwatch import collector as collector_mod

        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )  # no cgroup paths -> forces the /proc accumulator
        coll = TelemetryCollector(ctx)
        ticks: dict[int, int] = {}
        # ppid 1: these PIDs are orphans (reparented), so nothing inside the job will
        # ever re-report their CPU and the accumulator must keep it. The in-job-parent
        # case is the separate reconciliation test below.
        monkeypatch.setattr(
            collector_mod,
            "_read_pid_cpu",
            lambda pid: (
                collector_mod._PidCpu(own=ticks[pid], children=0, ppid=1) if pid in ticks else None
            ),
        )

        ticks = {100: 200}  # pid 100 has burned 200 ticks
        a = coll._read_cpu_ns({100})
        ticks = {101: 150}  # pid 100 exited; a fresh pid 101 burned 150
        b = coll._read_cpu_ns({101})
        ticks = {101: 300}  # pid 101 kept working (now 300 total)
        c = coll._read_cpu_ns({101})

        assert a is not None and b is not None and c is not None
        assert a <= b <= c  # monotonic despite pid 100's 200 ticks "vanishing"
        # 200 (pid100) + 150 (pid101 fresh) + 150 (pid101 forward delta) = 500 ticks
        assert coll._proc_cpu_accum_ticks == 500

    def _proc_only_collector(self) -> TelemetryCollector:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )  # no cgroup paths -> forces the /proc accumulator
        return TelemetryCollector(ctx)

    def test_proc_cpu_no_double_count_when_pid_briefly_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A still-live PID (present in /proc) can briefly fall out of the sampled set
        # (an enumeration race, a one-off /proc read miss). It must NOT be forgotten
        # and re-added whole on return — that re-added its entire history as if
        # brand-new, producing a spurious ~2x CPU spike (a double-count). Eviction is
        # now keyed on /proc liveness, so a live-but-unsampled PID is never evicted.
        from slurmwatch import collector as collector_mod

        coll = self._proc_only_collector()
        ticks: dict[int, int] = {}
        monkeypatch.setattr(
            collector_mod,
            "_read_pid_cpu",
            lambda pid: (
                collector_mod._PidCpu(own=ticks[pid], children=0, ppid=1) if pid in ticks else None
            ),
        )
        monkeypatch.setattr(collector_mod, "_pid_alive", lambda pid: pid in {100, 200})

        ticks = {100: 500, 200: 300}
        coll._read_cpu_ns({100, 200})
        ticks = {100: 500, 200: 310}  # pid 100 briefly missing from the sample (still alive)
        coll._read_cpu_ns({200})
        assert 100 in coll._proc_cpu_seen  # still alive -> retained, not forgotten
        ticks = {100: 520, 200: 320}  # pid 100 back (same process, a little more work)
        coll._read_cpu_ns({100, 200})

        # True cumulative is 520 (pid100) + 320 (pid200); the old code re-added
        # pid100's 520 on top of its earlier 500.
        assert coll._proc_cpu_accum_ticks == 520 + 320

    def test_proc_cpu_evicts_dead_pids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The retain-across-absence rule must still bound memory: a PID that has truly
        # exited (gone from /proc) is dropped from the seen map on the very next poll.
        # Eviction never changes the total (the exited PID's ticks already live in the
        # accumulator), and evicting only a /proc-confirmed-dead PID can't re-trigger
        # the double-count above.
        from slurmwatch import collector as collector_mod

        coll = self._proc_only_collector()
        ticks: dict[int, int] = {}
        alive: set[int] = {100, 200}
        monkeypatch.setattr(
            collector_mod,
            "_read_pid_cpu",
            lambda pid: (
                collector_mod._PidCpu(own=ticks[pid], children=0, ppid=1) if pid in ticks else None
            ),
        )
        monkeypatch.setattr(collector_mod, "_pid_alive", lambda pid: pid in alive)

        ticks = {100: 500}
        coll._read_cpu_ns({100})
        assert 100 in coll._proc_cpu_seen
        ticks = {200: 10}  # pid 100 exits (gone from /proc); another PID keeps the job alive
        alive = {200}
        coll._read_cpu_ns({200})
        assert 100 not in coll._proc_cpu_seen  # dead -> evicted, memory bounded
        # Eviction leaves the total alone HERE because pid 100 was an orphan (ppid 1):
        # nothing inside the job will re-report its CPU. When the parent IS in the
        # job, eviction must subtract instead — see
        # TestReapedChildCpu.test_no_double_count_when_an_in_job_parent_reaps.
        assert coll._proc_cpu_accum_ticks == 500 + 10

    def test_memory_oom_guard_uses_working_set(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(cgroup_job_ctx)
        mem = collector._collect_memory()
        assert mem.oom_guard_warning is False
        # Working set should be current - inactive_file
        current = 2 * 1024**3
        inactive_file = 100 * 1024**2  # 100 MiB from fixture
        expected_ws = current - inactive_file
        assert mem.working_set_bytes == expected_ws

    def test_memory_oom_guard_uses_working_set_threshold(self, cgroup_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(
            cgroup_job_ctx,
            SlurmwatchConfig(oom_warning_threshold=0.2, oom_critical_threshold=0.3),
        )
        mem = collector._collect_memory()
        # Limit from fixture is 8 GiB, ws is ~1.9 GiB, so ws_pct ~24%
        ws_guard = mem.working_set_bytes or mem.current_bytes
        ws_pct = (ws_guard / mem.limit_bytes) * 100.0
        assert ws_pct > 20
        assert ws_pct < 30
        assert mem.oom_guard_warning is True
        assert mem.oom_guard_critical is False

    def test_oom_guard_keyed_on_working_set_not_usage(self, cgroup_job_ctx: JobContext) -> None:
        # B-T9: put the threshold strictly BETWEEN the working-set % and the
        # cache-inclusive usage %. Fixture: current 2 GiB, inactive_file 100 MiB,
        # limit 8 GiB -> usage 25.0%, working set ~23.8%. A 24.5% warn threshold
        # fires only if the guard is (wrongly) keyed on cache-inclusive usage.
        collector = TelemetryCollector(
            cgroup_job_ctx,
            SlurmwatchConfig(oom_warning_threshold=0.245, oom_critical_threshold=0.5),
        )
        mem = collector._collect_memory()
        ws_pct = mem.working_set_bytes / mem.limit_bytes * 100.0
        usage_pct = mem.current_bytes / mem.limit_bytes * 100.0
        assert ws_pct < 24.5 < usage_pct  # the threshold is genuinely between them
        assert mem.oom_guard_warning is False  # keyed on the working set


class TestRemoteCollector:
    def _remote_ctx(self) -> JobContext:
        return JobContext(
            job_id="777",
            username="u",
            partition="gpu",
            nodelist="cn-002",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=200 * 1024**3,
            gpu_count_requested=2,
            gpu_indices=[],
            job_start_time=time.time() - 3600,  # 1h elapsed
            job_state="RUNNING",
            remote=True,
        )

    def test_remote_snapshot_from_sstat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch import slurm

        usage = slurm.RemoteUsage(rss_bytes=100 * 1024**3, cpu_seconds=7200.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        collector = TelemetryCollector(self._remote_ctx())
        cpu, mem = collector._collect_remote(time.time())
        # 100 GiB of a 200 GiB limit.
        assert mem.current_bytes == 100 * 1024**3
        assert mem.usage_percent == 50.0
        # 7200 CPU-seconds over ~3600s elapsed on 4 cores -> ~2 cores, ~50%.
        assert cpu.usage_ns == 7200 * 1_000_000_000
        assert 1.9 <= cpu.effective_cores <= 2.1
        assert 45.0 <= cpu.usage_percent <= 55.0

    def test_remote_effective_cores_uncapped_reveals_oversubscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A3 applies to BOTH paths: on a ConstrainCores=no node a job can burn more
        # CPU-time than its allocation, and capping effective_cores at cores_allocated
        # would erase the "raise --cpus-per-task" signal off-node exactly as it did
        # on-node. Only usage_percent (the bar) is clamped.
        from slurmwatch import slurm

        # 28800 CPU-seconds over ~3600s elapsed = 8 cores busy on a 4-core alloc.
        usage = slurm.RemoteUsage(rss_bytes=1024, cpu_seconds=28800.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        collector = TelemetryCollector(self._remote_ctx())
        cpu, _mem = collector._collect_remote(time.time())
        assert cpu.effective_cores >= 7.9, "off-node over-subscription was capped away"
        assert cpu.usage_percent == 100.0  # the bar still can't exceed full

    def test_remote_memory_percent_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # F1: rss = MaxRSS x tasks_per_node can exceed the limit for an imbalanced
        # step; the remote path must clamp to 100 like the on-node one, not emit an
        # impossible >100% in --json / the remote summary.
        from slurmwatch import slurm

        usage = slurm.RemoteUsage(rss_bytes=300 * 1024**3, cpu_seconds=0.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        collector = TelemetryCollector(self._remote_ctx())
        _cpu, mem = collector._collect_remote(time.time())
        assert mem.usage_percent == 100.0  # 300 GiB of a 200 GiB limit -> clamped
        assert mem.current_bytes == 300 * 1024**3  # raw value preserved, only % clamped

    def test_remote_transient_failure_keeps_last_good_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # S2-class: a transient sstat failure returns sampled=False zeros. The
        # collector must keep the last REAL sample rather than collapsing the
        # remote CPU/MEM bars to a fabricated 0% / 0 GiB for a busy-controller blip.
        from slurmwatch import slurm

        good = slurm.RemoteUsage(rss_bytes=100 * 1024**3, cpu_seconds=7200.0, sampled=True)
        fail = slurm.RemoteUsage(rss_bytes=0, cpu_seconds=0.0, sampled=False)
        seq = [good, fail]
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: seq.pop(0))
        collector = TelemetryCollector(self._remote_ctx())
        t = time.time()
        cpu1, mem1 = collector._collect_remote(t)
        assert mem1.current_bytes == 100 * 1024**3  # first (good) sample
        # Second fetch fails; step past the cache min-interval so it re-queries.
        cpu2, mem2 = collector._collect_remote(t + 100)
        assert mem2.current_bytes == 100 * 1024**3  # kept the good sample, not 0
        assert cpu2.usage_ns == cpu1.usage_ns  # CPU not collapsed to 0 either

    def test_remote_avg_cores_does_not_decay_during_outage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # N9: while sstat transiently fails, the frozen cpu_seconds must be divided
        # by the elapsed captured WITH it, not a growing now-based elapsed — else the
        # "avg cores" figure slides downward every frame until the next real sample.
        from slurmwatch import slurm

        good = slurm.RemoteUsage(rss_bytes=50 * 1024**3, cpu_seconds=7200.0, sampled=True)
        fail = slurm.RemoteUsage(rss_bytes=0, cpu_seconds=0.0, sampled=False)
        seq = [good, fail, fail]
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: seq.pop(0))
        ctx = self._remote_ctx()
        collector = TelemetryCollector(ctx)
        start = ctx.job_start_time or 0.0
        t = start + 3600.0  # exactly 1h elapsed -> 7200/3600 = 2.0 avg cores
        cpu1, _ = collector._collect_remote(t)
        assert cpu1.effective_cores == 2.0
        # Outage: time advances an hour each frame; a now-based elapsed would halve
        # the figure. It must stay frozen at the last real sample instead.
        cpu2, _ = collector._collect_remote(t + 3600.0)
        cpu3, _ = collector._collect_remote(t + 7200.0)
        assert cpu2.effective_cores == 2.0
        assert cpu3.effective_cores == 2.0

    def _remote_mem(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rss_gib: float,
        config: SlurmwatchConfig | None = None,
    ) -> MemoryMetrics:
        from slurmwatch import slurm

        usage = slurm.RemoteUsage(rss_bytes=int(rss_gib * 1024**3), cpu_seconds=1.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        collector = TelemetryCollector(self._remote_ctx(), config)  # 200 GiB limit
        _, mem = collector._collect_remote(time.time())
        return mem

    def test_the_oom_guard_is_evaluated_off_node_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SW-15: hardcoded False disabled the guard in the place it is most used —
        `sw <jobid>` from a login node — so a job creeping to 90% of its --mem
        reported "healthy" all the way into the OOM kill.

        This REVERSES #34 (which set both flags False because MaxRSS only climbs, so
        an alarm on it can't clear). What makes the alarm honest instead of sticky is
        that the reading is labelled a peak wherever it is shown: "this job came
        within 10% of its limit" stays true afterwards, and raising --mem stays the
        right advice.
        """
        mem = self._remote_mem(monkeypatch, 180)  # 90% of 200 GiB
        assert mem.usage_percent == 90.0
        assert mem.oom_guard_warning is True
        assert mem.oom_guard_critical is True
        assert mem.source == "sstat", "the flag is only honest while the reading is labelled"

    def test_the_off_node_guard_honours_the_configured_thresholds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`False` at 89% with a 30% threshold is only possible if nothing is
        evaluated — which is how the report proved it was hardcoded."""
        cfg = SlurmwatchConfig(oom_warning_threshold=0.3, oom_critical_threshold=0.4)
        mem = self._remote_mem(monkeypatch, 100, cfg)  # 50% of 200 GiB
        assert mem.oom_guard_warning is True and mem.oom_guard_critical is True
        raised = SlurmwatchConfig(oom_warning_threshold=0.95, oom_critical_threshold=0.99)
        mem = self._remote_mem(monkeypatch, 180, raised)  # 90%
        assert mem.oom_guard_warning is False and mem.oom_guard_critical is False

    def test_a_quiet_job_stays_quiet_off_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mem = self._remote_mem(monkeypatch, 40)  # 20% of 200 GiB
        assert mem.oom_guard_warning is False and mem.oom_guard_critical is False

    def test_remote_snapshot_is_tagged_remote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # #34/#35: the assembled snapshot carries remote=True so the UI labels the
        # memory bar "peak" (not "used") and structured-output readers know it's a
        # job-wide sstat estimate, not live per-node telemetry.
        from slurmwatch import slurm

        usage = slurm.RemoteUsage(rss_bytes=10 * 1024**3, cpu_seconds=1.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        collector = TelemetryCollector(self._remote_ctx())
        snap = collector._collect_snapshot_sync()
        assert snap.remote is True

    def test_remote_cpu_peak_is_populated_not_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Off-node, peak_effective_cores used to stay 0.0 next to a non-zero
        # effective_cores — breaking the "peak >= current" invariant every other peak
        # upholds, so a --json consumer sizing --cpus-per-task read "used no CPU".
        # (The memory path already reports MaxRSS as both current and peak here.)
        from slurmwatch import slurm

        usage = slurm.RemoteUsage(rss_bytes=10 * 1024**3, cpu_seconds=7200.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        collector = TelemetryCollector(self._remote_ctx())
        snap = collector._collect_snapshot_sync()
        assert snap.cpu.effective_cores == 2.0
        assert snap.cpu.peak_effective_cores == 2.0
        assert snap.cpu.peak_effective_cores >= snap.cpu.effective_cores
        assert snap.memory.peak_working_set_bytes >= snap.memory.working_set_bytes

    def test_remote_cpu_peak_keeps_high_water_mark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The off-node figure is an AVERAGE, which can fall; the peak must not.
        from slurmwatch import slurm

        state = {"cpu": 7200.0}
        monkeypatch.setattr(
            slurm,
            "resolve_remote_usage",
            lambda job_id, node_count=1: slurm.RemoteUsage(
                rss_bytes=10 * 1024**3, cpu_seconds=state["cpu"], sampled=True
            ),
        )
        collector = TelemetryCollector(self._remote_ctx())
        collector._remote_min_interval = 0.0  # don't serve the throttled cache
        first = collector._collect_snapshot_sync()
        assert first.cpu.peak_effective_cores == 2.0
        state["cpu"] = 3600.0  # the average halves
        second = collector._collect_snapshot_sync()
        assert second.cpu.effective_cores == 1.0
        assert second.cpu.peak_effective_cores == 2.0

    def test_remote_usage_scales_balanced_per_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # sstat totals are job-wide; a balanced 4-node/32-task step must be
        # scaled to per-node (8 tasks/node) so it matches the per-node limit.
        from slurmwatch import slurm

        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda cmd: "51|2097152K|01:00:00|32\n")
        u = slurm.resolve_remote_usage("51", node_count=4)
        assert u.rss_bytes == 16 * 1024**3  # 2 GiB/task x (32 // 4) tasks/node
        assert u.cpu_seconds == 28800.0  # 3600s/task x 8 tasks/node

    def test_remote_usage_concentrated_not_diluted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A single-task head step using 90 GiB on one node of a 2-node alloc must
        # report 90 GiB (its real per-node footprint), not 45 GiB — otherwise the
        # OOM guard stays green while a node is near its limit.
        from slurmwatch import slurm

        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda cmd: "51.batch|94371840K|00:30:00|1\n")
        u = slurm.resolve_remote_usage("51", node_count=2)
        assert u.rss_bytes == 90 * 1024**3  # max(1, 1 // 2) = 1 task/node

    def test_remote_usage_ceil_for_unbalanced_tasks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A4: 6 tasks on 4 nodes places 2,2,1,1 — the busiest node holds 2, so a
        # 90 GiB/task step is 180 GiB there. ceil(6/4)=2, not floor(6/4)=1 (which
        # would under-report to 90 GiB and keep the OOM guard falsely green).
        from slurmwatch import slurm

        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda cmd: "51|94371840K|00:30:00|6\n")
        u = slurm.resolve_remote_usage("51", node_count=4)
        assert u.rss_bytes == 180 * 1024**3  # 90 GiB x ceil(6/4)=2 tasks/node

    def test_remote_usage_survives_zero_node_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A4's defensive divisor: the sole in-tree caller passes >= 1, but this is a
        # public function — node_count=0 must not ZeroDivisionError out of a live
        # dashboard poll (it degrades to "everything on one node").
        from slurmwatch import slurm

        monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda cmd: "51|94371840K|00:30:00|6\n")
        u = slurm.resolve_remote_usage("51", node_count=0)
        assert u.sampled is True
        assert u.rss_bytes == 6 * 90 * 1024**3  # all 6 tasks attributed to one node

    def test_remote_queries_sstat_with_raw_numeric_job_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #30: `sstat -j 12345` and `sstat -j 12345_3` both expand to every
        # running task of an array, summing their steps and over-reporting CPU
        # N-fold. Only the underlying numeric JobId scopes to this one task, so
        # _collect_remote must query with raw_job_id, not the user-facing job_id.
        from slurmwatch import slurm

        seen = {}

        def _capture(job_id: str, node_count: int = 1) -> slurm.RemoteUsage:
            seen["job_id"] = job_id
            return slurm.RemoteUsage(rss_bytes=1, cpu_seconds=1.0, sampled=True)

        monkeypatch.setattr(slurm, "resolve_remote_usage", _capture)
        ctx = self._remote_ctx()
        ctx.job_id = "12345_3"  # user-facing array-task form
        ctx.raw_job_id = "12348"  # the underlying numeric JobId
        TelemetryCollector(ctx)._collect_remote(time.time())
        assert seen["job_id"] == "12348"

    def test_remote_falls_back_to_job_id_when_raw_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # raw_job_id is unset for a demo/mock context; the query must still work.
        from slurmwatch import slurm

        seen = {}

        def _capture(job_id: str, node_count: int = 1) -> slurm.RemoteUsage:
            seen["job_id"] = job_id
            return slurm.RemoteUsage(rss_bytes=1, cpu_seconds=1.0, sampled=True)

        monkeypatch.setattr(slurm, "resolve_remote_usage", _capture)
        ctx = self._remote_ctx()  # raw_job_id defaults to ""
        assert ctx.raw_job_id == ""
        TelemetryCollector(ctx)._collect_remote(time.time())
        assert seen["job_id"] == "777"

    def test_remote_throttles_sstat_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch import slurm

        calls = {"n": 0}

        def _count(job_id: str, node_count: int = 1) -> slurm.RemoteUsage:
            calls["n"] += 1
            return slurm.RemoteUsage(rss_bytes=1, cpu_seconds=1.0, sampled=True)

        monkeypatch.setattr(slurm, "resolve_remote_usage", _count)
        collector = TelemetryCollector(self._remote_ctx())
        t = time.time()
        collector._collect_remote(t)
        collector._collect_remote(t + 1)  # within the 5s throttle window
        assert calls["n"] == 1  # second call served from cache


class TestReadMeminfo:
    def test_read_meminfo_total(self) -> None:
        total = _read_meminfo_total()
        assert total > 0
        assert isinstance(total, int)


class TestProcCpuParsing:
    def test_parse_stat_with_spaces_and_parens_in_comm(self) -> None:
        from slurmwatch.collector import _parse_stat_cpu_ticks

        # comm "(tmux: server)" contains a space and parentheses; utime and
        # stime are fields 14 and 15 overall.
        fields = ["0"] * 52
        fields[0] = "4242"
        fields[1] = "(tmux: server)"
        fields[2] = "S"
        fields[13] = "150"  # utime (field 14)
        fields[14] = "50"  # stime (field 15)
        line = " ".join(fields)
        assert _parse_stat_cpu_ticks(line) == 200

    def test_parse_stat_malformed(self) -> None:
        from slurmwatch.collector import _parse_stat_cpu_ticks

        assert _parse_stat_cpu_ticks("garbage with no paren") == 0
        assert _parse_stat_cpu_ticks("123 (x) S 1 2 3") == 0  # too few fields

    def test_read_pid_cpu_ticks_self(self) -> None:
        from slurmwatch.collector import _read_pid_cpu_ticks

        # The current process has accumulated some CPU time.
        assert _read_pid_cpu_ticks(os.getpid()) >= 0
        assert _read_pid_cpu_ticks(99_999_999) == 0  # nonexistent PID


class _FakeNVMLError(Exception):
    # Real pynvml errors carry a numeric `.value` (NVML_ERROR_*); the collector reads
    # it to tell a durable NOT_SUPPORTED apart from a transient failure (A7), so the
    # fake declares it too. Absent by default, like a bare non-NVML exception.
    value: int


class _FakeUtil:
    gpu = 75


class _FakeMem:
    used = 20 * 1024**3
    total = 80 * 1024**3


class _FakeProc:
    def __init__(self, pid: int, mem: int | None) -> None:
        # Real pynvml sets usedGpuMemory to None (not a missing attribute)
        # when NVML reports NVML_VALUE_NOT_AVAILABLE, e.g. on MIG devices.
        self.pid = pid
        self.usedGpuMemory = mem


class _FakePUtil:
    def __init__(self, pid: int, sm: int, ts: int = 0) -> None:
        self.pid = pid
        self.smUtil = sm
        self.timeStamp = ts


class _FakePci:
    def __init__(self, bus_id: bytes) -> None:
        self.busId = bus_id


class _FakePynvml:
    NVMLError = _FakeNVMLError
    NVML_TEMPERATURE_GPU = 0
    # Real constant names from nvidia-ml-py (the invented HwThermal/SwThermal
    # spellings do not exist in any pynvml release).
    nvmlClocksThrottleReasonSwPowerCap = 1
    nvmlClocksThrottleReasonHwThermalSlowdown = 2
    nvmlClocksThrottleReasonSwThermalSlowdown = 4
    nvmlClocksThrottleReasonHwPowerBrakeSlowdown = 8
    nvmlClocksThrottleReasonHwSlowdown = 16

    @staticmethod
    def nvmlInit() -> None:
        return None

    @staticmethod
    def nvmlShutdown() -> None:
        return None

    @staticmethod
    def nvmlDeviceGetCount() -> int:
        return 2

    @staticmethod
    def nvmlDeviceGetHandleByUUID(uuid: bytes) -> object:
        return ("by_uuid", uuid.decode())

    @staticmethod
    def nvmlDeviceGetHandleByIndex(idx: int) -> object:
        return ("by_index", idx)

    @staticmethod
    def nvmlDeviceGetUUID(h: object) -> str:
        return "GPU-test"

    @staticmethod
    def nvmlDeviceGetName(h: object) -> str:
        return "A100-SXM4-80GB"

    @staticmethod
    def nvmlDeviceGetIndex(h: object) -> int:
        return 0

    @staticmethod
    def nvmlDeviceGetPciInfo(h: object) -> _FakePci:
        # h == ("by_index", idx). Give a bus id that *decreases* as the index
        # increases, so PCI-bus order is the reverse of NVML index order and a
        # test can tell "sorted by bus id" apart from "kept the first N".
        idx = h[1] if isinstance(h, tuple) and len(h) == 2 else 0
        return _FakePci(f"0000:{100 - int(idx):02x}:00.0".encode())

    @staticmethod
    def nvmlDeviceGetUtilizationRates(h: object) -> _FakeUtil:
        return _FakeUtil()

    @staticmethod
    def nvmlDeviceGetMemoryInfo(h: object) -> _FakeMem:
        return _FakeMem()

    @staticmethod
    def nvmlDeviceGetPowerUsage(h: object) -> int:
        return 250_000

    @staticmethod
    def nvmlDeviceGetEnforcedPowerLimit(h: object) -> int:
        # Milliwatts, like nvmlDeviceGetPowerUsage — the enforced cap of an
        # A100-SXM4-80GB. Present so the mW->W conversion and the "used / cap W"
        # display are actually exercised (#7); a fake WITHOUT this attribute is
        # covered separately, since AttributeError is the older-pynvml path.
        return 400_000

    @staticmethod
    def nvmlDeviceGetTemperature(h: object, sensor: int) -> int:
        return 65

    @staticmethod
    def nvmlDeviceGetCurrentClocksThrottleReasons(h: object) -> int:
        return 0

    @staticmethod
    def nvmlDeviceGetComputeRunningProcesses(h: object) -> list[_FakeProc]:
        # Two job processes (one reporting usedGpuMemory=None as on MIG) and
        # one foreign process that must not be attributed to the job.
        return [
            _FakeProc(1000, 18 * 1024**3),
            _FakeProc(1001, None),
            _FakeProc(4321, 10 * 1024**3),
        ]

    @staticmethod
    def nvmlDeviceGetGraphicsRunningProcesses(h: object) -> list[_FakeProc]:
        return []

    @staticmethod
    def nvmlDeviceGetProcessUtilization(h: object, ts: int) -> list[_FakePUtil]:
        # Multiple samples per pid (old ones must be dropped), several job
        # pids (their newest samples must be SUMMED), plus a foreign pid.
        return [
            _FakePUtil(1000, 90, ts=1),
            _FakePUtil(1000, 40, ts=2),
            _FakePUtil(1001, 20, ts=2),
            _FakePUtil(4321, 35, ts=2),
        ]


class _FVUnion:
    """The tagged-union member of an nvmlFieldValue — every accessor returns the
    same seeded number, so the reader picks whichever the valueType names."""

    def __init__(self, val: int) -> None:
        self.dVal = float(val)
        self.uiVal = val
        self.ulVal = val
        self.ullVal = val
        self.sllVal = val


class _FakeFieldValue:
    def __init__(self, field_id: int, val: int, vtype: int = 3, ret: int = 0) -> None:
        self.fieldId = field_id
        self.nvmlReturn = ret  # 0 = NVML_SUCCESS; 3 = NOT_SUPPORTED
        self.valueType = vtype  # default UNSIGNED_LONG_LONG
        self.value = _FVUnion(val)


class _FakeTopoPynvml:
    """A two-GPU node directly wired by 4 NVLinks (NVLink 3, 25 GB/s/link), with
    readable per-link speed and cumulative throughput counters — exercises the real
    ``_build_topology`` path (state → remote busId → peer match) without hardware.

    Set ``links = 0`` to model a PCIe-only node (falls through to the topology
    common-ancestor, returning ``pcie_level``)."""

    NVMLError = _FakeNVMLError
    NVML_NVLINK_MAX_LINKS = 18
    NVML_FEATURE_ENABLED = 1
    NVML_NVLINK_DEVICE_TYPE_SWITCH = 2
    NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON = 90
    NVML_FI_DEV_NVLINK_LINK_COUNT = 91
    NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX = 139
    NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX = 138
    NVML_VALUE_TYPE_DOUBLE = 0
    NVML_VALUE_TYPE_UNSIGNED_INT = 1
    NVML_VALUE_TYPE_UNSIGNED_LONG = 2
    NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3
    NVML_VALUE_TYPE_SIGNED_LONG_LONG = 4
    NVML_PCIE_UTIL_TX_BYTES = 0
    NVML_PCIE_UTIL_RX_BYTES = 1

    _BUS = {0: b"0000:07:00.0", 1: b"0000:0a:00.0"}
    _RX = {0: 1_000_000, 1: 2_000_000}  # cumulative NVLink KiB
    _TX = {0: 500_000, 1: 900_000}
    _PCIE = {0: (12_000_000, 8_000_000), 1: (10_000_000, 6_000_000)}  # (tx, rx) KB/s live

    def __init__(
        self,
        links: int = 4,
        version: int = 3,
        speed_mbps: int = 25000,
        pcie_level: int = 50,
        aggregate_scope: bool = True,
    ) -> None:
        # Every (fieldId, scopeId) this fake was asked for, so a test can assert the
        # collector requests the all-links aggregate rather than link 0.
        self.scopes_seen: list[tuple[int, int]] = []
        # False emulates a driver that rejects the UINT_MAX aggregate scope, forcing the
        # collector's per-link summation fallback.
        self.aggregate_scope = aggregate_scope
        # Instance attributes (not class) so a per-test config (e.g. links=0 for a
        # PCIe-only node) sticks — the collector calls these on the passed instance.
        self.links = links
        # NVML's driver-internal link-version code. Only ever logged now: a live H200
        # returns 7 here, so it can't be read as the marketing generation.
        self.version = version
        # speed_mbps = 0 models the real driver-535 behaviour, where
        # SPEED_MBPS_COMMON answers NOT_SUPPORTED and the model table must supply it.
        self.speed_mbps = speed_mbps
        self.pcie_level = pcie_level  # NVML_TOPOLOGY_SYSTEM → "SYS" (used when links == 0)

    @staticmethod
    def _idx(h: object) -> int:
        return h[1] if isinstance(h, tuple) and len(h) == 2 else 0

    def nvmlDeviceGetPciInfo(self, h: object) -> _FakePci:
        return _FakePci(self._BUS[self._idx(h)])

    def nvmlDeviceGetNvLinkState(self, h: object, link: int) -> int:
        if link < self.links:
            return self.NVML_FEATURE_ENABLED
        raise self.NVMLError("NOT_SUPPORTED")

    def nvmlDeviceGetNvLinkVersion(self, h: object, link: int) -> int:
        return self.version

    def nvmlDeviceGetNvLinkRemoteDeviceType(self, h: object, link: int) -> int:
        return 0  # NVML_NVLINK_DEVICE_TYPE_GPU

    def nvmlDeviceGetNvLinkRemotePciInfo(self, h: object, link: int) -> _FakePci:
        return _FakePci(self._BUS[1 - self._idx(h)])  # the other GPU in a 2-GPU node

    def nvmlDeviceGetFieldValues(
        self, h: object, field_ids: list[int | tuple[int, int]]
    ) -> list[_FakeFieldValue]:
        """Mirror real NVML's scope semantics for the NVLink throughput counters.

        Those two fields are PER-LINK, selected by ``nvmlFieldValue_t.scopeId``, and
        ``scopeId = UINT_MAX`` means "summed across all links" (nvml.h). pynvml only
        populates scopeId when the caller passes a ``(fieldId, scopeId)`` tuple, so a
        bare int arrives here as scope 0 — link 0 alone. ``_RX``/``_TX`` are the
        all-links totals, so a single link's share is that over ``links``: asking with
        the wrong scope therefore under-reports by exactly the link count, which is what
        the real bug did. Scopes past the device's link count are NOT_SUPPORTED.
        """
        idx = self._idx(h)
        throughput = {
            self.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX: self._RX[idx],
            self.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX: self._TX[idx],
        }
        table = {
            self.NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON: self.speed_mbps,
            self.NVML_FI_DEV_NVLINK_LINK_COUNT: self.links,
        }
        out: list[_FakeFieldValue] = []
        for req in field_ids:
            fid, scope = req if isinstance(req, tuple) else (req, 0)
            self.scopes_seen.append((fid, scope))
            if fid in throughput:
                total = throughput[fid]
                if scope == 0xFFFFFFFF:
                    value = total if self.aggregate_scope else 0
                    ret = 0 if self.aggregate_scope else 3
                elif scope < self.links:
                    # Split the total evenly across the links, remainder on link 0, so
                    # summing every per-link scope reproduces the aggregate exactly.
                    per = total // self.links if self.links else 0
                    value = per + (total - per * self.links if scope == 0 else 0)
                    ret = 0
                else:
                    value, ret = 0, 3
                out.append(_FakeFieldValue(fid, value, ret=ret))
                continue
            # speed_mbps = 0 means "the driver won't say" — a NOT_SUPPORTED return,
            # not a successful read of zero (that's how driver 535 answers on H200).
            unsupported = fid == self.NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON and not self.speed_mbps
            out.append(_FakeFieldValue(fid, table.get(fid, 0), ret=3 if unsupported else 0))
        return out

    def nvmlDeviceGetTopologyCommonAncestor(self, h1: object, h2: object) -> int:
        return self.pcie_level

    def nvmlDeviceGetPcieThroughput(self, h: object, counter: int) -> int:
        tx, rx = self._PCIE[self._idx(h)]
        return tx if counter == self.NVML_PCIE_UTIL_TX_BYTES else rx


class TestNvlinkModelSpec:
    """The model→(generation, per-link GB/s) table that replaced NVML's version code."""

    def test_real_product_names_map_to_their_generation(self) -> None:
        from slurmwatch.collector import _nvlink_model_spec

        assert _nvlink_model_spec("NVIDIA H200") == (4, 25.0)
        assert _nvlink_model_spec("NVIDIA H100 80GB HBM3") == (4, 25.0)
        assert _nvlink_model_spec("NVIDIA A100-SXM4-80GB") == (3, 25.0)
        assert _nvlink_model_spec("Tesla V100-SXM2-16GB") == (2, 25.0)
        assert _nvlink_model_spec("Tesla P100-SXM2-16GB") == (1, 20.0)
        assert _nvlink_model_spec("NVIDIA B200") == (5, 50.0)

    def test_matching_is_whole_token_so_families_do_not_collide(self) -> None:
        from slurmwatch.collector import _nvlink_model_spec

        # GH200 and GB200 are their own entries — a substring match would read them as
        # H200 / B200, which happens to agree for GH200 but NOT for GB200 (gen 5, 50
        # GB/s per link vs gen 5 too — so assert the tokens resolve on their own terms).
        assert _nvlink_model_spec("NVIDIA GH200 480GB") == (4, 25.0)
        assert _nvlink_model_spec("NVIDIA GB200") == (5, 50.0)
        # A name that merely contains a model string must not match it.
        assert _nvlink_model_spec("NVIDIA XH200Z") == (0, 0.0)
        assert _nvlink_model_spec("NVIDIA RTX A6000") == (0, 0.0)
        assert _nvlink_model_spec("") == (0, 0.0)

    def test_every_entry_matches_the_published_aggregate_bandwidth(self) -> None:
        # The per-link figure is only meaningful through the aggregate the UI shows
        # (links x speed x 2). NVML supplies the link count, so pin each entry against
        # the bandwidth NVIDIA publishes for that card — the table's self-check.
        from slurmwatch.collector import _nvlink_model_spec

        for name, links, aggregate in (
            ("Tesla P100-SXM2-16GB", 4, 160.0),
            ("Tesla V100-SXM2-16GB", 6, 300.0),
            ("NVIDIA A100-SXM4-80GB", 12, 600.0),
            ("NVIDIA H100 80GB HBM3", 18, 900.0),
            ("NVIDIA H200", 18, 900.0),
            ("NVIDIA B200", 18, 1800.0),
        ):
            _gen, per_link = _nvlink_model_spec(name)
            assert links * per_link * 2 == aggregate, name


class TestInterconnect:
    def _collector(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake: object,
        model: str = "NVIDIA A100-SXM4-40GB",
    ) -> TelemetryCollector:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=2,
            gpu_indices=[0, 1],
        )
        c = TelemetryCollector(ctx)
        c._nvml_initialized = True
        c._nvml_handles = [("by_index", 0), ("by_index", 1)]
        c._nvml_indices = [0, 1]
        # The NVLink generation / per-link speed are derived from the MODEL, so the
        # names cached at attach time have to be present for the topology probe.
        c._nvml_handle_info = {0: ("GPU-0", model), 1: ("GPU-1", model)}
        return c

    def test_direct_nvlink_topology(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = self._collector(monkeypatch, _FakeTopoPynvml())
        ic = c._collect_interconnect(
            [_gpu(0), _gpu(1)]  # two devices → interconnect is meaningful
        )
        assert ic is not None
        assert ic.fabric == "nvlink"
        assert ic.nvlink_version == 3
        assert ic.links_per_gpu == 4
        assert ic.link_speed_gbps == 25.0
        assert ic.per_gpu_gbps == 200.0  # 4 links * 25 GB/s * 2 (bidirectional)
        assert ic.nvswitch is False
        assert ic.devices == [0, 1]
        assert ic.matrix == [["self", "NV4"], ["NV4", "self"]]

    def test_pcie_only_topology(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No NVLink → every pair falls through to the PCIe common-ancestor (SYS).
        c = self._collector(monkeypatch, _FakeTopoPynvml(links=0))
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.fabric == "pcie"
        assert ic.links_per_gpu == 0
        assert ic.per_gpu_gbps == 0.0
        assert ic.matrix == [["self", "SYS"], ["SYS", "self"]]
        # A PCIe fabric reports live PCIe traffic (KB/s → GB/s), no NVLink counters.
        assert ic.nvlink_rx_gbps == [] and ic.nvlink_tx_gbps == []
        assert ic.pcie_tx_gbps == [12.0, 10.0]  # 12e6 / 10e6 KB/s → GB/s
        assert ic.pcie_rx_gbps == [8.0, 6.0]

    def test_static_topology_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = self._collector(monkeypatch, _FakeTopoPynvml())
        first = c._collect_interconnect([_gpu(0), _gpu(1)])
        # A second sweep must reuse the cached matrix object (topology can't change
        # under a running job), only re-reading live throughput.
        c._build_topology = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("topology re-probed")
        )
        second = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert first is not None and second is not None
        assert second.matrix == first.matrix

    def test_live_throughput_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Freeze the clock so the counter delta maps to an exact rate. The collector
        # uses a MONOTONIC clock for the rate window (A1), so patch time.monotonic.
        monkeypatch.setattr(time, "monotonic", lambda: 2000.0)
        c = self._collector(monkeypatch, _FakeTopoPynvml())
        # Seed each device's prior counter reading 1s earlier at 0, so the delta over
        # 1s is the full cumulative KiB → GB/s (KiB * 1024 / 1e9), kept to 3 decimals
        # so per-device values stay accurate before the UI sums them: dev0 rx 1e6 KiB
        # → 1.024, tx 5e5 → 0.512; dev1 rx 2e6 → 2.048, tx 9e5 → 0.9216 → 0.922.
        c._nvlink_prev = {0: (1999.0, 0, 0), 1: (1999.0, 0, 0)}
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.nvlink_rx_gbps == [1.024, 2.048]
        assert ic.nvlink_tx_gbps == [0.512, 0.922]

    def test_topology_matches_devices_across_differing_pci_domain_widths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _norm_bus exists because nvmlDeviceGetPciInfo and
        # nvmlDeviceGetNvLinkRemotePciInfo can render the SAME device's busId with
        # different domain widths. If that normalization stops, every remote-endpoint
        # lookup misses, so a genuine NVLink node silently reports fabric="pcie" with no
        # generation and no bandwidth. The fake previously used one bus dict for both
        # calls, so the widths always agreed and nothing could catch it.
        fake = _FakeTopoPynvml()

        def wide_remote(h: object, link: int) -> object:
            narrow = fake._BUS[1 - fake._idx(h)].decode()

            class _Info:
                busId = f"0000{narrow}".encode()  # 8-digit domain vs the 4-digit local

            return _Info()

        monkeypatch.setattr(fake, "nvmlDeviceGetNvLinkRemotePciInfo", wide_remote)
        c = self._collector(monkeypatch, fake)
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.fabric == "nvlink"
        assert ic.matrix[0][1] == "NV4"

    def test_nvlink_throughput_asks_for_the_all_links_aggregate_not_link_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The THROUGHPUT_DATA_* fields are PER-LINK, chosen by nvmlFieldValue_t.scopeId;
        # nvml.h documents scopeId=UINT_MAX as the sum across all links. pynvml sets
        # scopeId ONLY for a (fieldId, scopeId) tuple, so passing a bare int silently
        # measured link 0 alone — 1/links of the real traffic (18x under-report on an
        # 18-link H200), making a saturated fabric look nearly idle.
        monkeypatch.setattr(time, "monotonic", lambda: 2000.0)
        fake = _FakeTopoPynvml()
        c = self._collector(monkeypatch, fake)
        c._nvlink_prev = {0: (1999.0, 0, 0), 1: (1999.0, 0, 0)}
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        throughput_scopes = {
            scope
            for fid, scope in fake.scopes_seen
            if fid
            in (
                fake.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX,
                fake.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
            )
        }
        assert throughput_scopes == {0xFFFFFFFF}, "must request the all-links aggregate"
        # And the value is the WHOLE fabric, not one link's quarter of it (links=4).
        assert ic.nvlink_rx_gbps == [1.024, 2.048]

    def test_nvlink_throughput_falls_back_to_summing_per_link_scopes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A driver that rejects the UINT_MAX aggregate scope must not silently lose the
        # fabric rate: sum the per-link scopes instead, which totals the same traffic.
        monkeypatch.setattr(time, "monotonic", lambda: 2000.0)
        c = self._collector(monkeypatch, _FakeTopoPynvml(aggregate_scope=False))
        c._nvlink_prev = {0: (1999.0, 0, 0), 1: (1999.0, 0, 0)}
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.nvlink_rx_gbps == [1.024, 2.048]
        assert ic.nvlink_tx_gbps == [0.512, 0.922]

    def test_links_per_gpu_counts_enabled_links_not_the_present_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NVML_FI_DEV_NVLINK_LINK_COUNT is "NVLinks PRESENT on the device"; only the
        # per-link state says which are UP. Folding them together with max() made a
        # degraded fabric advertise spec bandwidth while the grid beside it said NV3.
        fake = _FakeTopoPynvml(links=4)
        real_state = fake.nvmlDeviceGetNvLinkState

        def one_link_down(h: object, link: int) -> int:
            return 0 if link == 3 else int(real_state(h, link))

        monkeypatch.setattr(fake, "nvmlDeviceGetNvLinkState", one_link_down)
        c = self._collector(monkeypatch, fake)
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.links_per_gpu == 3, "must report the 3 links that are actually up"
        assert ic.matrix[0][1] == "NV3"
        # ...and the bandwidth follows the measured links, not the spec sheet.
        assert ic.per_gpu_gbps == pytest.approx(3 * ic.link_speed_gbps * 2)

    def test_nvlink_rate_hidden_until_every_device_has_a_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The UI SUMS these per-device rates, so a device that only just seeded its
        # baseline must not contribute a hard 0.0 beside a sibling that has a real rate:
        # that halved the reported fabric traffic on a symmetric 2-GPU ring and drew the
        # second GPU as idle. "Not yet knowable" is per-device — suppress the whole line.
        monkeypatch.setattr(time, "monotonic", lambda: 2000.0)
        c = self._collector(monkeypatch, _FakeTopoPynvml())
        c._nvlink_prev = {0: (1999.0, 0, 0)}  # device 1 has no baseline yet
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.nvlink_rx_gbps == [] and ic.nvlink_tx_gbps == []
        # The suppressed frame still SEEDED device 1, so the next frame — where both
        # devices have a baseline — publishes a full-length list again. (The fake's
        # counters are constant, so the rate itself is a true 0.0 here.)
        monkeypatch.setattr(time, "monotonic", lambda: 2001.0)
        second = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert second is not None
        assert len(second.nvlink_rx_gbps) == 2 and len(second.nvlink_tx_gbps) == 2

    def test_live_throughput_uses_monotonic_not_wall_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A1: the rate window uses time.monotonic, so a backward wall-clock step
        # (NTP/leap/VM migration) must NOT spike the rate. Freeze monotonic at a 1s
        # gap and jump time.time far into the past — the rate is unchanged.
        monkeypatch.setattr(time, "monotonic", lambda: 2000.0)
        monkeypatch.setattr(time, "time", lambda: 1.0)  # hostile wall-clock jump
        c = self._collector(monkeypatch, _FakeTopoPynvml())
        c._nvlink_prev = {0: (1999.0, 0, 0), 1: (1999.0, 0, 0)}  # monotonic, 1s ago
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        # Identical to the monotonic dt=1s rates — the wall-clock jump had no effect.
        assert ic.nvlink_rx_gbps == [1.024, 2.048]
        assert ic.nvlink_tx_gbps == [0.512, 0.922]

    def test_generation_and_speed_come_from_the_model_not_the_nvml_version_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # D1/D2, reproduced from live hardware: on a 3x H200 node (driver 535.216.03)
        # nvmlDeviceGetNvLinkVersion returns 7 on every link and SPEED_MBPS_COMMON
        # answers NOT_SUPPORTED. Reading that 7 as the generation printed "NVLink 7"
        # (there is no such generation; CUDA 12.7's enum calls 7 NVLink 5.0, and an
        # H200 is Hopper/NVLink 4), and the gen-keyed speed fallback had no key 7, so
        # the per-link and per-GPU bandwidth silently vanished from the drill-in.
        fake = _FakeTopoPynvml(links=18, version=7, speed_mbps=0)
        c = self._collector(monkeypatch, fake, model="NVIDIA H200")
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.fabric == "nvlink"
        assert ic.nvlink_version == 4, "H200 is NVLink 4, whatever NVML's code says"
        assert ic.links_per_gpu == 18
        assert ic.link_speed_gbps == 25.0
        # 18 x 25 x 2 — exactly the 900 GB/s NVIDIA publishes for an H200.
        assert ic.per_gpu_gbps == 900.0

    def test_measured_per_link_speed_still_wins_over_the_model_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The table is a FALLBACK. When the driver does expose SPEED_MBPS_COMMON that
        # is ground truth for this link and must be preferred, even where it disagrees
        # with the model's nominal rate (NVML reports the raw signalling rate).
        fake = _FakeTopoPynvml(links=18, version=7, speed_mbps=26562)
        c = self._collector(monkeypatch, fake, model="NVIDIA H200")
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.link_speed_gbps == 26.6  # rounded to 1 decimal for display
        assert ic.nvlink_version == 4  # generation is still the model's, not the code's

    def test_unknown_model_reports_no_generation_rather_than_a_guess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A card the table doesn't know (a workstation part with bridge NVLink, or one
        # newer than this build) must degrade to a bare "NVLink" — the wiring facts NVML
        # really measured (fabric, link count, matrix) stay, the invented ones don't.
        fake = _FakeTopoPynvml(links=4, version=7, speed_mbps=0)
        c = self._collector(monkeypatch, fake, model="NVIDIA FutureCard 1234")
        ic = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert ic is not None
        assert ic.fabric == "nvlink" and ic.links_per_gpu == 4
        assert ic.nvlink_version == 0
        assert ic.link_speed_gbps == 0.0 and ic.per_gpu_gbps == 0.0

    def test_first_nvlink_sample_reports_unknown_not_a_hard_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # D3: the DATA counters are cumulative, so one sample can't give a rate. It used
        # to emit 0.0 anyway — and `--once` takes exactly one sample, so it reported an
        # idle fabric on a live H200 job whose counters already held 90 GB. Unknown must
        # read as unknown (empty → the UI hides the line), and the rate appears once a
        # second sample exists.
        monkeypatch.setattr(time, "monotonic", lambda: 2000.0)
        fake = _FakeTopoPynvml()
        c = self._collector(monkeypatch, fake)
        first = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert first is not None
        assert first.nvlink_rx_gbps == [] and first.nvlink_tx_gbps == []
        # That first read seeded the baseline at the counters' current value. Advance
        # them by the same amount again over 1s (instance attrs, so no class-level
        # leak into other tests) — now there are two samples and the rate is real.
        fake._RX = {0: 2_000_000, 1: 4_000_000}
        fake._TX = {0: 1_000_000, 1: 1_800_000}
        monkeypatch.setattr(time, "monotonic", lambda: 2001.0)
        second = c._collect_interconnect([_gpu(0), _gpu(1)])
        assert second is not None
        assert second.nvlink_rx_gbps == [1.024, 2.048]
        assert second.nvlink_tx_gbps == [0.512, 0.922]

    def test_single_gpu_has_no_interconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = self._collector(monkeypatch, _FakeTopoPynvml())
        c._nvml_handles = [("by_index", 0)]
        c._nvml_indices = [0]
        assert c._collect_interconnect([_gpu(0)]) is None

    def test_mock_mode_reports_nvlink(self) -> None:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=4,
            gpu_indices=[0, 1, 2, 3],
        )
        c = TelemetryCollector(ctx)
        c._mock = True
        c._mock_start = time.monotonic()
        snap = c._collect_snapshot_sync()
        assert snap.interconnect is not None
        assert snap.interconnect.fabric == "nvlink"
        assert len(snap.interconnect.matrix) == 4


def _gpu(index: int) -> GpuMetrics:
    return GpuMetrics(
        index=index,
        uuid=f"GPU-{index}",
        name="A100-SXM4-80GB",
        utilization_percent=50.0,
        memory_used_bytes=10 * 1024**3,
        memory_total_bytes=80 * 1024**3,
        memory_utilization_percent=12.5,
        power_watts=200.0,
        temperature_celsius=60.0,
        throttling=False,
    )


class TestGpuActive:
    def test_busy_gpu_active_without_process_util_sample(self) -> None:
        # nvmlDeviceGetProcessUtilization returned nothing (process util 0.0),
        # but the device is at 100% and the job owns its VRAM -> ACTIVE, not
        # idle. Regression: the verdict reported "all GPUs idle" on a pegged GPU.
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA H200",
            utilization_percent=100.0,
            memory_used_bytes=140 * 1024**3,
            memory_total_bytes=144 * 1024**3,
            memory_utilization_percent=97.0,
            power_watts=500.0,
            temperature_celsius=45.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=138 * 1024**3,
        )
        assert _gpu_is_active(g, 5.0) is True

    def test_busy_gpu_active_when_process_vram_unreadable(self) -> None:
        # Containerized/vGPU jobs: device util is high and VRAM is used, but NVML
        # withholds per-process VRAM (0) because the PIDs are namespaced. The
        # ≥50%-own-VRAM guard must NOT then score a pegged GPU idle (false "GPU
        # IDLE"); an unreadable per-process figure is not evidence of an idle job.
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA A100",
            utilization_percent=95.0,
            memory_used_bytes=40 * 1024**3,
            memory_total_bytes=80 * 1024**3,
            memory_utilization_percent=50.0,
            power_watts=300.0,
            temperature_celsius=55.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=0,  # NVML withheld per-process VRAM
        )
        assert _gpu_is_active(g, 5.0) is True

    def test_active_via_job_util_when_device_looks_idle(self) -> None:
        # E1: the PREFERRED branch — the job's own per-process SM utilization is
        # above the threshold, so the GPU is active even though device-wide util
        # is below it and the job holds a minority of VRAM. Every other test feeds
        # process_utilization_percent=0.0, so without this the branch is untested
        # (it could be inverted/deleted with the suite still green).
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA H200",
            utilization_percent=2.0,  # device-wide below the idle threshold
            memory_used_bytes=140 * 1024**3,
            memory_total_bytes=144 * 1024**3,
            memory_utilization_percent=97.0,
            power_watts=120.0,
            temperature_celsius=40.0,
            throttling=False,
            process_utilization_percent=42.0,  # this job's own SM util, above threshold
            process_memory_bytes=1 * 1024**3,  # a minority of used VRAM (< 50%)
        )
        assert _gpu_is_active(g, 5.0) is True

    def test_busy_gpu_with_unreadable_vram_is_not_scored_idle(self) -> None:
        # The activity heuristic vetoes on `memory_used_bytes > 0` as a sanity check that
        # a busy-looking device has something resident. But when the VRAM QUERY failed,
        # that 0 is not a measurement, and vetoing on it called a 99%-utilized GPU idle:
        # amber "idle", gpu_active_count=0, and "0% HBM" on a full card. Reachable and
        # persistent, not transient — nvidia-ml-py >= 11.510 raises FunctionNotFound from
        # nvmlDeviceGetMemoryInfo_v2 against a pre-510 driver.
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA H200",
            utilization_percent=99.0,  # pegged
            memory_used_bytes=0,  # not a reading...
            memory_total_bytes=0,
            memory_utilization_percent=0.0,
            power_watts=600.0,
            temperature_celsius=70.0,
            throttling=False,
            memory_available=False,  # ...which is what this says
        )
        assert _gpu_is_active(g, 5.0) is True
        # A genuine zero-VRAM reading still vetoes, so the sanity check is intact.
        g.memory_available = True
        assert _gpu_is_active(g, 5.0) is False

    def test_unreadable_power_renders_as_na_not_zero_watts(self) -> None:
        # "0 W" beside an active device described an unpowered card. A MIG slice returns
        # NOT_SUPPORTED from the power API, so this is the normal case there, not an edge.
        from slurmwatch.tui import _gpu_power_text

        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA A100",
            utilization_percent=0.0,
            memory_used_bytes=3 * 1024**3,
            memory_total_bytes=5 * 1024**3,
            memory_utilization_percent=60.0,
            power_watts=0.0,
            temperature_celsius=0.0,
            throttling=False,
            power_available=False,
        )
        assert "n/a" in _gpu_power_text(g, 3)
        assert "0 W" not in _gpu_power_text(g, 3)
        # A real 0 W reading (idle, powered) still prints as a number.
        g.power_available = True
        assert "0 W" in _gpu_power_text(g, 3)

    def test_truly_idle_gpu_not_active(self) -> None:
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA H200",
            utilization_percent=2.0,
            memory_used_bytes=0,
            memory_total_bytes=144 * 1024**3,
            memory_utilization_percent=0.0,
            power_watts=60.0,
            temperature_celsius=30.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=0,
        )
        assert _gpu_is_active(g, 5.0) is False

    def test_shared_gpu_not_credited_to_minor_tenant(self) -> None:
        # Shared, non-isolated GPU driven to 100% by another user; this job only
        # holds a sliver of VRAM and has no process-util sample -> NOT active
        # (device util must not be mis-credited to a minor tenant).
        g = GpuMetrics(
            index=0,
            uuid="GPU-x",
            name="NVIDIA H200",
            utilization_percent=100.0,
            memory_used_bytes=140 * 1024**3,
            memory_total_bytes=144 * 1024**3,
            memory_utilization_percent=97.0,
            power_watts=500.0,
            temperature_celsius=45.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=1 * 1024**3,  # 1/140 of used VRAM
        )
        assert _gpu_is_active(g, 5.0) is False


class TestCollectGpus:
    def test_collect_gpus_with_fake_nvml(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="12345",
            username="testuser",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=16,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=1,
            gpu_indices=[0],
            step_id="0",
            uid=1001,
            job_start_time=1000.0,
            cgroup_v2_path=str(fake_cgroup_v2_job),
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object()]
        collector._nvml_handle_info = {0: ("GPU-test", "A100-SXM4-80GB")}

        gpus = collector._collect_gpus()
        assert len(gpus) == 1
        g = gpus[0]
        assert g.utilization_percent == 75.0
        assert g.memory_used_bytes == 20 * 1024**3
        assert g.throttling is False
        # PIDs 1000/1001 come from the fixture's leaf cgroup.procs; pid 4321
        # is another user's process and pid 1001 reports usedGpuMemory=None
        # (which must count as 0, not crash and drop the GPU).
        assert g.process_memory_bytes == 18 * 1024**3
        # Newest sample per job pid, summed: 40 (pid 1000) + 20 (pid 1001).
        assert g.process_utilization_percent == 60.0
        # #7: the ENFORCED power cap is read (nvmlDeviceGetEnforcedPowerLimit) and
        # converted mW -> W, so the UI can show headroom-to-cap. It used never to be
        # read at all — only the SwPowerCap throttle bit was.
        assert g.power_watts == 250.0
        assert g.power_limit_watts == 400.0

    def test_power_cap_absent_on_older_pynvml(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #7's other branch: a pynvml without nvmlDeviceGetEnforcedPowerLimit raises
        # AttributeError, which must degrade to 0.0 (the UI then shows a bare "W")
        # rather than dropping the whole device.
        import sys

        class _NoCapPynvml:
            """_FakePynvml with nvmlDeviceGetEnforcedPowerLimit genuinely ABSENT.

            A subclass can't express this (it would inherit the method), so proxy
            every other attribute and let this one raise AttributeError the way an
            older nvidia-ml-py does."""

            def __init__(self, inner: object) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                if name == "nvmlDeviceGetEnforcedPowerLimit":
                    raise AttributeError(name)
                return getattr(self._inner, name)

        monkeypatch.setitem(sys.modules, "pynvml", _NoCapPynvml(_FakePynvml()))
        ctx = JobContext(
            job_id="12345",
            username="testuser",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=16,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=1,
            gpu_indices=[0],
            step_id="0",
            uid=1001,
            job_start_time=1000.0,
            cgroup_v2_path=str(fake_cgroup_v2_job),
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object()]
        collector._nvml_handle_info = {0: ("GPU-test", "A100-SXM4-80GB")}
        gpus = collector._collect_gpus()
        assert len(gpus) == 1
        assert gpus[0].power_watts == 250.0
        assert gpus[0].power_limit_watts == 0.0

    def test_mock_gpus_carry_a_power_cap(self) -> None:
        # --demo drives the same renderer as a live run, so the mock must supply an
        # enforced cap; without one the demo showed a bare "240 W" while the README
        # GIF (its own synthetic data) advertised the "used / cap W" headroom figure.
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1 << 30,
            gpu_count_requested=4,
            gpu_indices=[0, 1, 2, 3],
        )
        collector = TelemetryCollector(ctx)
        collector._mock = True
        gpus = collector._collect_gpus()
        assert gpus, "mock must synthesize devices"
        for g in gpus:
            assert g.power_limit_watts > 0
            assert 0 < g.power_watts <= g.power_limit_watts

    def test_init_nvml_selects_by_uuid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=2,
            gpu_indices=[],
            gpu_uuids=["MIG-aaaa", "GPU-bbbb"],
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is True
        # Both job GPUs were resolved via UUID, not raw NVML index.
        assert len(collector._nvml_handles) == 2
        assert collector._nvml_handles[0] == ("by_uuid", "MIG-aaaa")

    def test_init_nvml_constrained_device_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ConstrainDevices: the job holds node-global GPU index 1, but NVML in
        # the job's device cgroup exposes only that one GPU, renumbered to local
        # index 0. The node-global index must not be dropped by an `ordinal <
        # count` bounds check (regression: this returned zero GPUs, so a job on
        # a pegged GPU showed no GPU telemetry at all).
        import sys

        fake = _FakePynvml()
        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 1))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=1,
            gpu_indices=[1],
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is True
        assert len(collector._nvml_handles) == 1
        assert collector._nvml_handles[0] == ("by_index", 0)

    def test_init_nvml_caps_unresolved_indices_to_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A GPU job whose indices/UUIDs couldn't be resolved must NOT attach to
        # every device on a shared node (that shows other users' GPUs). It must
        # cap to the requested count. Regression: this attached all devices.
        import sys

        fake = _FakePynvml()
        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 8))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=2,  # asked for 2 of the node's 8 GPUs
            gpu_indices=[],
            gpu_uuids=[],  # but none could be resolved
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is True
        assert len(collector._nvml_handles) == 2  # not all 8

    def test_init_nvml_whole_node_attaches_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When the request covers the whole node (requested >= device_count),
        # attaching every device is correct.
        import sys

        fake = _FakePynvml()
        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 2))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=2,
            gpu_indices=[],
            gpu_uuids=[],
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is True
        assert len(collector._nvml_handles) == 2

    def test_init_nvml_disabled_without_pynvml(self) -> None:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        collector = TelemetryCollector(ctx)
        gpus = collector._collect_gpus()  # not initialized -> empty
        assert gpus == []

    def test_init_nvml_missing_driver_is_quiet_info_not_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # On a node with no NVIDIA driver, nvmlInit raises NVMLError_LibraryNotFound.
        # For a GPU job that's a normal environment condition (login/CPU node), so
        # it should be a quiet INFO, never a scary WARNING with a cryptic message.
        import logging
        import sys

        # Same class name real pynvml uses; the code detects it by type name.
        class NVMLError_LibraryNotFound(Exception):  # noqa: N801, N818
            pass

        class _NoDriverPynvml:
            @staticmethod
            def nvmlInit() -> None:
                raise NVMLError_LibraryNotFound("NVML Shared Library Not Found")

        monkeypatch.setitem(sys.modules, "pynvml", _NoDriverPynvml())
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=2,  # a GPU job, but no driver here
            gpu_indices=[0, 1],
        )
        collector = TelemetryCollector(ctx)
        with caplog.at_level(logging.INFO, logger="slurmwatch"):
            assert collector._init_nvml() is False
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        assert any("GPU monitoring off" in r.message for r in caplog.records)

    def test_init_nvml_skips_cpu_only_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A job that requested no GPUs must not attach to every device on a
        # shared GPU node (that would display other users' workloads).
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=4,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is False
        assert collector._nvml_handles == []

    def test_throttling_detected_with_real_constants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        fake = _FakePynvml()
        # SwPowerCap | HwThermalSlowdown bits set.
        monkeypatch.setattr(
            _FakePynvml,
            "nvmlDeviceGetCurrentClocksThrottleReasons",
            staticmethod(lambda h: 3),
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=1,
            gpu_indices=[0],
        )
        collector = TelemetryCollector(ctx)
        throttling, reasons = collector._check_gpu_throttling(object())
        assert throttling is True
        # bits=3 = SwPowerCap | HwThermalSlowdown; both are meaningful reasons.
        assert "sw_power_cap" in reasons
        assert "hw_thermal" in reasons

    def test_no_throttling_when_mask_clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=1,
            gpu_indices=[0],
        )
        collector = TelemetryCollector(ctx)
        assert collector._check_gpu_throttling(object()) == (False, [])


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


class TestGpuActiveMig:
    def test_mig_active_when_util_unavailable_but_holds_vram(self) -> None:
        # B-P3: on a MIG slice the rate APIs return NOT_SUPPORTED, so util reads
        # 0 with utilization_available=False. The job clearly holds VRAM here, so
        # it must count as active, not idle.
        g = GpuMetrics(
            index=0,
            uuid="MIG-x",
            name="A100 MIG",
            utilization_percent=0.0,
            memory_used_bytes=10 * 1024**3,
            memory_total_bytes=20 * 1024**3,
            memory_utilization_percent=50.0,
            power_watts=0.0,
            temperature_celsius=0.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=9 * 1024**3,
            utilization_available=False,
        )
        assert _gpu_is_active(g, 5.0) is True

    def test_mig_idle_when_util_unavailable_and_no_vram(self) -> None:
        g = GpuMetrics(
            index=0,
            uuid="MIG-x",
            name="A100 MIG",
            utilization_percent=0.0,
            memory_used_bytes=0,
            memory_total_bytes=20 * 1024**3,
            memory_utilization_percent=0.0,
            power_watts=0.0,
            temperature_celsius=0.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=0,
            utilization_available=False,
        )
        assert _gpu_is_active(g, 5.0) is False

    def test_mig_active_when_util_and_proc_vram_unreadable_but_slice_holds_vram(self) -> None:
        # #36: NVML frequently reports usedGpuMemory as NOT_AVAILABLE on MIG, so
        # process_memory_bytes collapses to 0 even on an actively-used slice.
        # Falling back to the slice's own used VRAM avoids a spurious "idle" (crit)
        # verdict when both per-process VRAM and device util are unreadable.
        g = GpuMetrics(
            index=0,
            uuid="MIG-x",
            name="A100 MIG",
            utilization_percent=0.0,
            memory_used_bytes=8 * 1024**3,  # slice VRAM is occupied
            memory_total_bytes=20 * 1024**3,
            memory_utilization_percent=40.0,
            power_watts=0.0,
            temperature_celsius=0.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=0,  # per-process VRAM withheld by NVML on MIG
            utilization_available=False,
            utilization_supported=False,  # MIG: the rate API is genuinely NOT_SUPPORTED
        )
        assert _gpu_is_active(g, 5.0) is True

    def test_transient_util_fail_does_not_overcredit_shared_gpu(self) -> None:
        # A7: a TRANSIENT util-read failure (utilization_supported=True) on a shared
        # GPU must NOT credit device-wide VRAM as this job's — only positive majority
        # ownership counts, so another tenant's load can't inflate gpu_active_count.
        # (Contrast the MIG case above, where util is genuinely NOT_SUPPORTED.)
        from dataclasses import replace

        minority = GpuMetrics(
            index=0,
            uuid="u",
            name="H100",
            utilization_percent=0.0,
            memory_used_bytes=40 * 1024**3,
            memory_total_bytes=80 * 1024**3,
            memory_utilization_percent=50.0,
            power_watts=0.0,
            temperature_celsius=0.0,
            throttling=False,
            process_utilization_percent=0.0,
            process_memory_bytes=5 * 1024**3,  # the job holds only 12.5% of used VRAM
            utilization_available=False,
            utilization_supported=True,  # transient read failure, NOT a MIG slice
        )
        assert _gpu_is_active(minority, 5.0) is False  # not over-credited
        # If the job owns the majority of the used VRAM, it IS active.
        assert _gpu_is_active(replace(minority, process_memory_bytes=30 * 1024**3), 5.0) is True


class TestCollectGpusDedup:
    def test_pid_in_compute_and_graphics_counted_once(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B-P5: a PID present in BOTH the compute and graphics process lists must
        # have its VRAM counted once, not doubled.
        import sys

        fake = _FakePynvml()
        monkeypatch.setattr(
            _FakePynvml,
            "nvmlDeviceGetComputeRunningProcesses",
            staticmethod(lambda h: [_FakeProc(1000, 18 * 1024**3)]),
        )
        monkeypatch.setattr(
            _FakePynvml,
            "nvmlDeviceGetGraphicsRunningProcesses",
            staticmethod(lambda h: [_FakeProc(1000, 18 * 1024**3)]),
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = _min_ctx(
            cgroup_v2_path=str(fake_cgroup_v2_job), gpu_count_requested=1, gpu_indices=[0]
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object()]
        collector._nvml_indices = [0]
        collector._nvml_handle_info = {0: ("GPU-test", "A100-SXM4-80GB")}
        gpus = collector._collect_gpus()
        assert gpus[0].process_memory_bytes == 18 * 1024**3  # once, not 36


class TestCollectGpusUtilSupported:
    """A7, collector side: WHY the util read failed decides which fallback is safe."""

    def _collect_with_util_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cgroup: Path,
        err: Exception,
    ) -> GpuMetrics:
        import sys

        fake = _FakePynvml()

        def _boom(h: object) -> object:
            raise err

        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetUtilizationRates", staticmethod(_boom))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = _min_ctx(cgroup_v2_path=str(cgroup), gpu_count_requested=1, gpu_indices=[0])
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object()]
        collector._nvml_indices = [0]
        collector._nvml_handle_info = {0: ("GPU-test", "A100")}
        gpus = collector._collect_gpus()
        assert len(gpus) == 1
        return gpus[0]

    def test_not_supported_marks_util_unsupported(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NOT_SUPPORTED is the durable MIG case: _gpu_is_active may then fall back to
        # device VRAM, because a slice's VRAM is isolated to this job.
        err = _FakeNVMLError()
        err.value = 3  # NVML_ERROR_NOT_SUPPORTED
        gpu = self._collect_with_util_error(monkeypatch, fake_cgroup_v2_job, err)
        assert gpu.utilization_available is False
        assert gpu.utilization_supported is False

    def test_other_nvml_error_keeps_util_supported(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Any OTHER failure is a transient miss on a util-capable device. Marking it
        # "unsupported" would hand a shared GPU the lenient VRAM fallback and let
        # another tenant's VRAM inflate gpu_active_count for that frame (A7).
        err = _FakeNVMLError()
        err.value = 15  # e.g. NVML_ERROR_TIMEOUT
        gpu = self._collect_with_util_error(monkeypatch, fake_cgroup_v2_job, err)
        assert gpu.utilization_available is False
        assert gpu.utilization_supported is True
        # An exception carrying no .value at all is treated the same (safe) way.
        plain = self._collect_with_util_error(monkeypatch, fake_cgroup_v2_job, _FakeNVMLError())
        assert plain.utilization_available is False
        assert plain.utilization_supported is True


class TestCudaOrdinal:
    """C2: the ordinal is the device's position in the job's own device list."""

    def test_ordinal_is_position_not_nvml_index(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A cluster WITHOUT device-cgroup isolation: NVML sees the whole node, so the
        # job's two GPUs carry node-global indices 2 and 3 while its code addresses
        # them as cuda:0 and cuda:1. The label has to be the latter.
        import sys

        fake = _FakePynvml()
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        h2, h3 = object(), object()
        node_index = {id(h2): 2, id(h3): 3}
        monkeypatch.setattr(
            _FakePynvml,
            "nvmlDeviceGetIndex",
            staticmethod(lambda h: node_index.get(id(h), 0)),
        )
        ctx = _min_ctx(
            cgroup_v2_path=str(fake_cgroup_v2_job), gpu_count_requested=2, gpu_indices=[2, 3]
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [h2, h3]
        collector._nvml_indices = [2, 3]
        collector._nvml_handle_info = {2: ("GPU-2", "A100"), 3: ("GPU-3", "A100")}
        gpus = collector._collect_gpus()
        assert [g.index for g in gpus] == [2, 3]
        assert [g.cuda_ordinal for g in gpus] == [0, 1]

    def test_ordinal_survives_a_dropped_device(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The ordinal comes from the handle's POSITION, not from how many devices made
        # it into the list — so a device the per-handle guard drops must not renumber
        # the ones after it (which would silently re-map every "CUDA N" and the
        # positional gpu_<N>_* CSV groups for that frame).
        import sys

        fake = _FakePynvml()
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        bad = object()

        def _mem(h: object) -> object:
            # Not an NVMLError, so it escapes the per-query suppress and trips the
            # whole-device guard — the real "this handle is unusable" path.
            if h is bad:
                raise RuntimeError("device fell off the bus")
            return _FakeMem()

        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetMemoryInfo", staticmethod(_mem))
        ctx = _min_ctx(
            cgroup_v2_path=str(fake_cgroup_v2_job), gpu_count_requested=3, gpu_indices=[0, 1, 2]
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object(), bad, object()]
        collector._nvml_indices = [0, 1, 2]
        collector._nvml_handle_info = {i: (f"GPU-{i}", "A100") for i in range(3)}
        gpus = collector._collect_gpus()
        # The middle device dropped out; the third keeps ordinal 2, not 1.
        assert [g.cuda_ordinal for g in gpus] == [0, 2]


class TestCollectGpusIndexFallback:
    def test_transient_get_index_uses_cached_index(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B-P7: a transient nvmlDeviceGetIndex failure must not drop the GPU for
        # the cycle; it falls back to the index cached at attach time.
        import sys

        fake = _FakePynvml()

        def _boom(h: object) -> int:
            raise _FakeNVMLError()

        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetIndex", staticmethod(_boom))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        ctx = _min_ctx(
            cgroup_v2_path=str(fake_cgroup_v2_job), gpu_count_requested=1, gpu_indices=[3]
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object()]
        collector._nvml_indices = [3]
        collector._nvml_handle_info = {3: ("GPU-x", "A100")}
        gpus = collector._collect_gpus()
        assert len(gpus) == 1  # not dropped despite the getIndex failure
        assert gpus[0].index == 3  # fell back to the cached index
        assert gpus[0].name == "A100"


class TestInitNvmlLifecycle:
    def test_flag_set_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-C5: the initialized flag is set from inside _init_nvml (not only via
        # the awaiting task's assignment), so cleanup can't be skipped.
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        collector = TelemetryCollector(_min_ctx(gpu_count_requested=1, gpu_indices=[0]))
        assert collector._init_nvml() is True
        assert collector._nvml_initialized is True

    def test_enumeration_failure_shuts_nvml_back_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B-P6: if enumeration raises after nvmlInit(), NVML must be shut down
        # again rather than left initialized.
        import sys

        fake = _FakePynvml()
        shutdowns = {"n": 0}

        def _boom() -> int:
            raise _FakeNVMLError()

        def _count_shutdown() -> None:
            shutdowns["n"] += 1

        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(_boom))
        monkeypatch.setattr(_FakePynvml, "nvmlShutdown", staticmethod(_count_shutdown))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        collector = TelemetryCollector(_min_ctx(gpu_count_requested=1, gpu_indices=[0]))
        assert collector._init_nvml() is False
        assert shutdowns["n"] >= 1


class TestInitNvmlPciOrdering:
    def test_caps_to_requested_in_pci_bus_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-T3: with unresolved indices on an 8-GPU node the cap keeps the
        # requested count in PCI-bus order. The fake's bus ids decrease as the
        # NVML index rises, so the two lowest-bus devices are indices 7 and 6 —
        # asserting *which* two proves the sort, not merely that two were kept.
        import sys

        fake = _FakePynvml()
        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 8))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        collector = TelemetryCollector(
            _min_ctx(gpu_count_requested=2, gpu_indices=[], gpu_uuids=[])
        )
        assert collector._init_nvml() is True
        assert collector._nvml_handles == [("by_index", 7), ("by_index", 6)]


class TestCollectorLoopResilience:
    @pytest.mark.asyncio
    async def test_loop_survives_one_bad_cycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # B-T5: a single raising collection must not end telemetry; the next
        # cycle recovers and next_snapshot() still yields.
        collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig(poll_interval=0.02))
        monkeypatch.setattr(collector, "_init_nvml", lambda: False)
        monkeypatch.setattr(collector, "_prime_cpu_baseline", lambda: None)
        good = _make_test_snapshot()
        calls = {"n": 0}

        def _collect() -> TelemetrySnapshot:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("cgroup vanished mid-read")
            return good

        monkeypatch.setattr(collector, "_collect_snapshot_sync", _collect)
        await collector.start()
        try:
            snap = await asyncio.wait_for(collector.next_snapshot(), timeout=2.0)
            assert snap is good
            assert calls["n"] >= 2  # the first cycle raised, a later one succeeded
        finally:
            await collector.stop()

    @pytest.mark.asyncio
    async def test_stop_awaits_inflight_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # C1: stop() must capture the in-flight collection BEFORE cancelling the
        # poll task (whose `finally` nulls _inflight_collect), then await it — so
        # the executor's result is retrieved gracefully instead of orphaned. With
        # a collection genuinely in flight, stop() completes cleanly and the
        # future ends up done with no unretrieved exception.
        import threading

        collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig(poll_interval=0.01))
        monkeypatch.setattr(collector, "_init_nvml", lambda: False)
        monkeypatch.setattr(collector, "_prime_cpu_baseline", lambda: None)
        started = threading.Event()
        release = threading.Event()
        snap = _make_test_snapshot()

        def _collect() -> TelemetrySnapshot:
            started.set()
            release.wait(timeout=5.0)  # block so the collection is truly in-flight
            return snap

        monkeypatch.setattr(collector, "_collect_snapshot_sync", _collect)
        await collector.start()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, started.wait, 5.0)  # a collection reached the executor
        fut = collector._inflight_collect
        assert fut is not None and not fut.done()  # a collection is genuinely in flight
        # Let the blocked collection finish, then tear down. With the capture-
        # before-cancel fix, stop() references this future and settles it (rather
        # than reading a nulled attribute); teardown stays clean and bounded.
        release.set()
        await asyncio.wait_for(collector.stop(), timeout=3.0)
        assert collector._task is not None and collector._task.done()
        assert fut.done()  # the in-flight future was settled, not left pending/orphaned

    def test_enqueue_drops_oldest_when_full(self) -> None:
        # B-T5: a full 32-slot queue evicts the oldest so the freshest sample
        # always survives.
        collector = TelemetryCollector(_min_ctx())
        snaps = [_make_test_snapshot() for _ in range(33)]
        for i, s in enumerate(snaps):
            s.job_id = str(i)  # make them distinguishable (dataclasses compare by value)
            collector._enqueue(s)
        assert collector.queue.qsize() == 32
        drained = [collector.queue.get_nowait() for _ in range(32)]
        assert drained[-1] is snaps[-1]  # newest survived
        assert all(s is not snaps[0] for s in drained)  # oldest was dropped


class TestJobEndedDetection:
    """#28: the collector latches job_ended from a dedicated Slurm-liveness task so
    the dashboard/headless logger can stop instead of freezing forever — and so a
    slow squeue never stalls the live telemetry feed."""

    @pytest.mark.asyncio
    async def test_latches_job_ended_when_slurm_reports_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from slurmwatch import slurm

        collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig(poll_interval=0.01))
        monkeypatch.setattr(collector, "_init_nvml", lambda: False)
        monkeypatch.setattr(collector, "_prime_cpu_baseline", lambda: None)
        monkeypatch.setattr(collector, "_collect_snapshot_sync", _make_test_snapshot)
        collector._liveness_min_interval = 0.01  # fast interval for the test
        monkeypatch.setattr(slurm, "is_job_active", lambda job_id: False)
        assert collector.job_ended is False
        await collector.start()
        try:
            for _ in range(50):
                if collector.job_ended:
                    break
                await asyncio.sleep(0.02)
            assert collector.job_ended is True
        finally:
            await collector.stop()

    @pytest.mark.asyncio
    async def test_unknown_liveness_does_not_latch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A transient squeue failure (is_job_active -> None) must NOT end the job.
        from slurmwatch import slurm

        collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig(poll_interval=0.01))
        monkeypatch.setattr(collector, "_init_nvml", lambda: False)
        monkeypatch.setattr(collector, "_prime_cpu_baseline", lambda: None)
        monkeypatch.setattr(collector, "_collect_snapshot_sync", _make_test_snapshot)
        collector._liveness_min_interval = 0.01
        monkeypatch.setattr(slurm, "is_job_active", lambda job_id: None)
        await collector.start()
        try:
            await asyncio.sleep(0.2)
            assert collector.job_ended is False
        finally:
            await collector.stop()

    @pytest.mark.asyncio
    async def test_remote_view_latches_job_ended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A1: the remote (login-node) dashboard must detect job end too. is_job_active
        # polls Slurm STATE (squeue/sacct), which is independent of whether sstat has
        # sampled usage — so the liveness task MUST run remotely and latch job_ended,
        # rather than retry srun against a dead job forever.
        from slurmwatch import slurm

        monkeypatch.setattr(slurm, "is_job_active", lambda job_id: False)
        collector = TelemetryCollector(
            _min_ctx(remote=True, nodelist_resolved=["cn1"]),
            SlurmwatchConfig(poll_interval=0.01),
        )
        collector._liveness_min_interval = 0.01
        await collector.start()
        try:
            assert collector._liveness_task is not None
            for _ in range(50):
                if collector.job_ended:
                    break
                await asyncio.sleep(0.02)
            assert collector.job_ended is True
        finally:
            await collector.stop()

    @pytest.mark.asyncio
    async def test_mock_never_starts_liveness_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig(poll_interval=0.01))
        await collector.start()
        try:
            await asyncio.sleep(0.1)
            assert collector._liveness_task is None
            assert collector.job_ended is False
        finally:
            await collector.stop()

    @pytest.mark.asyncio
    async def test_query_uses_raw_job_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Liveness must be checked with the raw numeric JobId (scopes to the task).
        from slurmwatch import slurm

        collector = TelemetryCollector(
            _min_ctx(job_id="12345_3"), SlurmwatchConfig(poll_interval=0.01)
        )
        monkeypatch.setattr(collector, "_init_nvml", lambda: False)
        monkeypatch.setattr(collector, "_prime_cpu_baseline", lambda: None)
        monkeypatch.setattr(collector, "_collect_snapshot_sync", _make_test_snapshot)
        collector.job_ctx.raw_job_id = "12348"
        collector._liveness_min_interval = 0.01
        seen: dict[str, str] = {}

        def _capture(job_id: str) -> bool:
            seen["job_id"] = job_id
            return True  # stays active so the loop keeps running

        monkeypatch.setattr(slurm, "is_job_active", _capture)
        await collector.start()
        try:
            for _ in range(50):
                if seen:
                    break
                await asyncio.sleep(0.02)
            assert seen["job_id"] == "12348"
        finally:
            await collector.stop()

    @pytest.mark.asyncio
    async def test_liveness_does_not_stall_snapshots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The regression the decoupling prevents: a slow squeue must NOT delay the
        # telemetry feed. With is_job_active blocking ~0.3s, snapshots must keep
        # flowing on their own faster cadence.
        from slurmwatch import slurm

        collector = TelemetryCollector(_min_ctx(), SlurmwatchConfig(poll_interval=0.01))
        monkeypatch.setattr(collector, "_init_nvml", lambda: False)
        monkeypatch.setattr(collector, "_prime_cpu_baseline", lambda: None)
        monkeypatch.setattr(collector, "_collect_snapshot_sync", _make_test_snapshot)
        collector._liveness_min_interval = 0.01

        def _slow(job_id: str) -> bool:
            time.sleep(0.3)  # a slow controller
            return True

        monkeypatch.setattr(slurm, "is_job_active", _slow)
        await collector.start()
        try:
            # Several snapshots should arrive well within one slow squeue call.
            for _ in range(3):
                await asyncio.wait_for(collector.next_snapshot(), timeout=0.2)
        finally:
            await collector.stop()


class TestPeakFallback:
    def test_running_max_used_when_memory_peak_absent(self, tmp_path: Path) -> None:
        # B-T6: on kernels without memory.peak (< 5.19, incl. this cluster's
        # 4.18) the running max is the only peak source and must be retained
        # across polls even when current memory later drops.
        v2 = tmp_path / "cg"
        v2.mkdir()
        (v2 / "memory.max").write_text(str(8 * 1024**3))
        (v2 / "memory.stat").write_text("inactive_file 0\nactive_file 0\n")
        collector = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v2_path=str(v2))
        )
        (v2 / "memory.current").write_text(str(4 * 1024**3))
        m1 = collector._collect_memory()
        assert m1.peak_bytes == 4 * 1024**3
        (v2 / "memory.current").write_text(str(2 * 1024**3))
        m2 = collector._collect_memory()
        assert m2.peak_bytes == 4 * 1024**3  # retained, not reset to the lower current

    def test_the_fallback_peak_does_not_claim_to_be_a_lifetime_figure(self, tmp_path: Path) -> None:
        """Same reading, weaker claim: a running max has no pre-session history and a
        restart resets it, so no surface may call it the job's lifetime peak. Every
        display of it is worded off this flag."""
        v2 = tmp_path / "cg"
        v2.mkdir()
        (v2 / "memory.max").write_text(str(8 * 1024**3))
        (v2 / "memory.stat").write_text("inactive_file 0\nactive_file 0\n")
        (v2 / "memory.current").write_text(str(4 * 1024**3))
        collector = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v2_path=str(v2))
        )
        assert collector._collect_memory().peak_is_lifetime is False
        # With the counter present (kernel >= 5.19) it IS a lifetime figure.
        (v2 / "memory.peak").write_text(str(6 * 1024**3))
        assert collector._collect_memory().peak_is_lifetime is True

    def test_the_v1_counter_is_a_lifetime_figure_and_its_absence_is_not(
        self, tmp_path: Path
    ) -> None:
        v1 = tmp_path / "v1"
        v1.mkdir()
        (v1 / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
        (v1 / "memory.usage_in_bytes").write_text(str(4 * 1024**3))
        (v1 / "memory.stat").write_text("total_inactive_file 0\ntotal_active_file 0\n")
        collector = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v1_mem_path=str(v1))
        )
        assert collector._collect_memory().peak_is_lifetime is False
        (v1 / "memory.max_usage_in_bytes").write_text(str(6 * 1024**3))
        assert collector._collect_memory().peak_is_lifetime is True

    def test_sstat_maxrss_is_a_lifetime_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MaxRSS never falls precisely because it IS a job-lifetime high-water."""
        from slurmwatch import slurm

        collector = TelemetryCollector(_min_ctx(mem_limit_bytes=8 * 1024**3, remote=True))
        usage = slurm.RemoteUsage(rss_bytes=3 * 1024**3, cpu_seconds=10.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        _cpu, mem = collector._collect_remote(time.time())
        assert mem.peak_is_lifetime is True
        assert mem.cache_measured is False  # and cache-EXCLUDED, so no gap note

    def _v2_tree(self, tmp_path: Path) -> Path:
        cg = tmp_path / "v2"
        cg.mkdir()
        (cg / "memory.current").write_text(str(4 * 1024**3))
        (cg / "memory.max").write_text(str(8 * 1024**3))
        (cg / "memory.peak").write_text(str(6 * 1024**3))
        (cg / "memory.stat").write_text("inactive_file 1073741824\nactive_file 0\n")
        return cg

    def _v1_tree(self, tmp_path: Path) -> Path:
        cg = tmp_path / "v1"
        cg.mkdir()
        (cg / "memory.usage_in_bytes").write_text(str(4 * 1024**3))
        (cg / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
        (cg / "memory.max_usage_in_bytes").write_text(str(6 * 1024**3))
        (cg / "memory.stat").write_text("total_inactive_file 1073741824\ntotal_active_file 0\n")
        return cg

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_the_peak_survives_the_cgroup_vanishing_mid_sample(
        self, tmp_path: Path, version: str
    ) -> None:
        """A job that ends between the liveness poll and the cgroup read took the
        lifetime peak with it: usage AND max_usage stop answering together, so
        `current` collapsed to the /proc fallback's 0 and the running-max fallback —
        which only ever saw `current` — republished a 6 GiB high-water as 0. Lower
        than the cache-EXCLUDED `peak_working_set_bytes` beside it, which is
        impossible for a cache-INCLUSIVE figure, and --once/--log write that frame.
        Both readers, because the fallback is written twice."""
        if version == "v2":
            cg = self._v2_tree(tmp_path)
            key = "cgroup_v2_path"
        else:
            cg = self._v1_tree(tmp_path)
            key = "cgroup_v1_mem_path"
        collector = TelemetryCollector(_min_ctx(mem_limit_bytes=8 * 1024**3, **{key: str(cg)}))
        live = collector._collect_memory()
        assert live.peak_bytes == 6 * 1024**3
        assert live.peak_is_lifetime is True

        for f in list(cg.iterdir()):  # the job ended; the whole cgroup is gone
            f.unlink()
        ended = collector._collect_memory()
        assert ended.current_bytes == 0, "nothing left to read"
        assert ended.peak_bytes == 6 * 1024**3, "a peak may not read below one reported"
        assert ended.peak_bytes >= ended.peak_working_set_bytes, "cache-incl. >= cache-excl."

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_a_retained_peak_is_a_floor_not_a_latch(self, tmp_path: Path, version: str) -> None:
        """The control on the fix above: flooring the reading must not freeze it.
        A kernel counter that climbs still has to be reported (a cached whole
        reading, or a max() taken the wrong way round, passes the test above and
        fails this one), and the frame that lost the counter must not go on
        claiming to hold a kernel LIFETIME figure."""
        if version == "v2":
            cg = self._v2_tree(tmp_path)
            key, counter = "cgroup_v2_path", "memory.peak"
        else:
            cg = self._v1_tree(tmp_path)
            key, counter = "cgroup_v1_mem_path", "memory.max_usage_in_bytes"
        collector = TelemetryCollector(_min_ctx(mem_limit_bytes=8 * 1024**3, **{key: str(cg)}))
        assert collector._collect_memory().peak_bytes == 6 * 1024**3
        (cg / counter).write_text(str(7 * 1024**3))  # the job grew
        risen = collector._collect_memory()
        assert risen.peak_bytes == 7 * 1024**3, "the floor must not cap a rising peak"
        assert risen.peak_is_lifetime is True
        (cg / counter).unlink()
        kept = collector._collect_memory()
        assert kept.peak_bytes == 7 * 1024**3
        assert kept.peak_is_lifetime is False, "no counter answered THIS frame"


class TestLifetimePeaks:
    """`_apply_peaks` folds the CPU high-water mark (the peak cores ever busy at
    once, a monotonic running max) into a local snapshot and keeps the memory peak
    self-consistent. The memory peak itself is the cgroup's own lifetime counter
    (read in `_collect_memory`), so `_apply_peaks` PRESERVES it rather than
    recomputing it from the live working set, and only guarantees `max >= used`."""

    def _cpu(self, effective: float) -> CpuMetrics:
        return CpuMetrics(
            cores_allocated=8, usage_ns=0, usage_percent=0.0, effective_cores=effective
        )

    def _mem(self, ws: int) -> MemoryMetrics:
        return MemoryMetrics(
            current_bytes=ws,
            limit_bytes=64 * 1024**3,
            peak_bytes=ws,  # placeholder; tests set the cgroup's lifetime peak explicitly
            usage_percent=0.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
            working_set_bytes=ws,
            cache_bytes=0,
        )

    def test_cpu_peak_is_the_lifetime_max_cores(self) -> None:
        c = TelemetryCollector(_min_ctx(cpus_allocated=8))
        cpu = self._cpu(3.0)
        c._apply_peaks(cpu, self._mem(10 * 1024**3))
        assert cpu.peak_effective_cores == 3.0
        cpu2 = self._cpu(6.5)  # a busier moment
        c._apply_peaks(cpu2, self._mem(10 * 1024**3))
        assert cpu2.peak_effective_cores == 6.5
        cpu3 = self._cpu(2.0)  # cores drop back
        c._apply_peaks(cpu3, self._mem(10 * 1024**3))
        assert cpu3.peak_effective_cores == 6.5  # peak retained, not the lower current

    def test_mem_peak_preserved_from_cgroup_lifetime_counter(self) -> None:
        # U1: the lifetime peak comes from _collect_memory (the cgroup's own
        # memory.max_usage_in_bytes / memory.peak). _apply_peaks must PRESERVE it,
        # never shrink it to the smaller live working set — else a late-attached job
        # would under-report its true high-water mark for --mem sizing.
        c = TelemetryCollector(_min_ctx())
        mem = self._mem(20 * 1024**3)
        mem.peak_bytes = 50 * 1024**3  # the lifetime peak the cgroup counter reported
        c._apply_peaks(self._cpu(1.0), mem)
        assert mem.peak_bytes == 50 * 1024**3  # preserved, not dragged down to 20

    def test_mem_peak_never_reads_below_current(self) -> None:
        # Defensive invariant: max >= used always holds, even if the reported peak
        # briefly lags the current usage.
        c = TelemetryCollector(_min_ctx())
        mem = self._mem(20 * 1024**3)
        mem.peak_bytes = 10 * 1024**3  # a stale/low reading
        mem.current_bytes = 30 * 1024**3
        c._apply_peaks(self._cpu(1.0), mem)
        assert mem.peak_bytes == 30 * 1024**3  # raised to current, never below it

    def test_peaks_are_wired_into_the_emitted_snapshot(self) -> None:
        # Guard the integration point: _collect_snapshot_sync must actually call
        # _apply_peaks, or the peaks would silently stay 0 in production. Drive the
        # real local (mock) snapshot path and assert the peak lands on the snapshot.
        c = TelemetryCollector(_min_ctx(cpus_allocated=8))
        c._mock = True
        c._mock_start = time.monotonic()
        snap = c._collect_snapshot_sync()
        assert snap.cpu.effective_cores > 0
        # First sample: the running max equals the current — proving _apply_peaks ran
        # (a dropped call would leave peak_effective_cores at its 0.0 default).
        assert snap.cpu.peak_effective_cores == snap.cpu.effective_cores


class TestElapsedClamp:
    def test_elapsed_never_negative_under_clock_skew(self) -> None:
        # N10: a job whose start time is in the future (compute-node clock skew) must
        # report elapsed 0 — not a negative value that rendered "ran -1:59:56" / "-0%"
        # and wrote a negative elapsed_seconds to CSV.
        ctx = _min_ctx()
        ctx.job_start_time = time.time() + 3600  # "starts" an hour from now
        c = TelemetryCollector(ctx)
        snap = c._collect_snapshot_sync()
        assert snap.elapsed_seconds == 0


class TestCsvGpuCountCap:
    def test_gpu_count_and_columns_track_the_device_count(self) -> None:
        # #38 (supersedes B-P9's silent cap): sizing the columns to the actual
        # device count means a 10-GPU node emits 10 groups and gpu_count=10 —
        # nothing is dropped. When a caller uses the default 8 groups on a 10-GPU
        # snapshot, gpu_count still reports the true 10 so the truncation is
        # visible (a reader compares gpu_count against the groups present) rather
        # than hidden behind a capped "8".
        snap = _make_test_snapshot()
        snap.gpus = [snap.gpus[0] for _ in range(10)]

        # Sized to fit: all 10 present, count matches.
        row = snap.to_csv_row(max_gpus=10)
        header = TelemetrySnapshot.csv_header(max_gpus=10)
        assert len(row) == len(header)
        assert row[header.index("gpu_count")] == "10"
        assert "gpu_9_index" in header

        # Default (8) groups: still self-consistent, and gpu_count signals 10.
        row8 = snap.to_csv_row()
        header8 = TelemetrySnapshot.csv_header()
        assert len(row8) == len(header8)
        assert row8[header8.index("gpu_count")] == "10"  # truncation signalled, not hidden


class TestTheCgroupV2CpuSourceAndItsFallbacks:
    """On a cgroup v2 cluster `cpu.stat` IS the CPU source, and nothing asserted it.

    This cluster is v1 with no per-job cpuacct, so its production CPU path is the
    /proc PID sum — meaning the v2 read runs only somewhere else, which is exactly
    why it needed a test. `usage_usec` appeared in no test in the suite; the fixture
    writes one, but no assertion followed it through, and the fallback ORDER
    (v2 -> v1 -> /proc) was unasserted too.

    Verified against a synthetic v2 tree and a real child process before writing
    these: 149 clock ticks of child CPU came back as 1.5s through the /proc rung, and
    the child's VmRSS came back to the byte through the memory rung.
    """

    @staticmethod
    def _ctx(**over: object) -> JobContext:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=4,
            mem_limit_bytes=16 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        for k, v in over.items():
            setattr(ctx, k, v)
        return ctx

    def test_usage_usec_is_read_as_microseconds(self, tmp_path: Path) -> None:
        """A factor of 1000 here would misreport every v2 cluster's CPU by 1000x."""
        v2 = tmp_path / "job"
        v2.mkdir()
        (v2 / "cpu.stat").write_text("usage_usec 90000000\nuser_usec 80000000\n")
        collector = TelemetryCollector(self._ctx(cgroup_v2_path=str(v2)))
        assert collector._read_cpu_ns(set()) == 90_000_000_000
        assert collector._cpu_source == "v2"

    def test_a_cpu_stat_without_usage_usec_falls_through(self, tmp_path: Path) -> None:
        """Some kernels expose cpu.stat with only the user/system split."""
        v2 = tmp_path / "job"
        v2.mkdir()
        (v2 / "cpu.stat").write_text("user_usec 1\nsystem_usec 2\n")
        v1 = tmp_path / "v1"
        v1.mkdir()
        (v1 / "cpuacct.usage").write_text("7000000000\n")
        collector = TelemetryCollector(
            self._ctx(cgroup_v2_path=str(v2), cgroup_v1_cpu_path=str(v1))
        )
        assert collector._read_cpu_ns(set()) == 7_000_000_000
        assert collector._cpu_source == "v1", "must not stop at a cpu.stat it cannot use"

    def test_no_cgroup_counter_falls_back_to_the_proc_sum(self, tmp_path: Path) -> None:
        """The rung this cluster runs on in production, reached here from a v2 path
        whose cpu.stat is missing entirely."""
        v2 = tmp_path / "job"
        v2.mkdir()  # no cpu.stat at all
        collector = TelemetryCollector(self._ctx(cgroup_v2_path=str(v2)))
        collector._read_cpu_ns({os.getpid()})
        assert collector._cpu_source == "proc"

    def test_the_v2_rung_wins_when_both_exist(self, tmp_path: Path) -> None:
        """Order matters: the two counters are not comparable, and differencing one
        against the other is what test_cpu_source_change_does_not_spike_the_rate
        exists to prevent."""
        v2 = tmp_path / "job"
        v2.mkdir()
        (v2 / "cpu.stat").write_text("usage_usec 1000000\n")
        v1 = tmp_path / "v1"
        v1.mkdir()
        (v1 / "cpuacct.usage").write_text("9000000000\n")
        collector = TelemetryCollector(
            self._ctx(cgroup_v2_path=str(v2), cgroup_v1_cpu_path=str(v1))
        )
        assert collector._read_cpu_ns(set()) == 1_000_000_000
        assert collector._cpu_source == "v2"


class TestTheRowSaysHowOldItsMeasurementIs:
    """Off-node a row can be a re-serialisation of a measurement taken seconds ago.

    Measured on a real off-node log at 1s: 17 of 21 consecutive rows were identical
    in cpu and memory, then the 18th jumped by 20 core-seconds, because sstat is
    queried at most every 5s. A consumer cannot detect that by diffing — an
    unchanged cpu_usage_ns is also what an idle job produces — so the row has to
    carry the age itself.
    """

    @staticmethod
    def _remote_ctx() -> JobContext:
        return JobContext(
            job_id="9",
            username="u",
            partition="p",
            nodelist="cn001",
            hostname="login-01",
            cpus_allocated=4,
            mem_limit_bytes=64 * 1024**3,
            gpu_count_requested=0,
            gpu_indices=[],
            job_start_time=1000.0,
            remote=True,
        )

    def _collector(self, monkeypatch: pytest.MonkeyPatch, calls: list[int]) -> Any:
        from slurmwatch import slurm

        def _usage(job_id: str, node_count: int = 1) -> Any:
            calls.append(1)
            return slurm.RemoteUsage(rss_bytes=8 * 1024**3, cpu_seconds=100.0, sampled=True)

        monkeypatch.setattr(slurm, "resolve_remote_usage", _usage)
        return TelemetryCollector(self._remote_ctx())

    def test_a_fresh_query_reports_age_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        collector = self._collector(monkeypatch, calls)
        snap = collector._collect_snapshot_sync()
        assert len(calls) == 1
        assert snap.usage_age_seconds == 0.0

    def test_a_cached_row_reports_the_real_age(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second sample re-serialises the first, and must say so."""
        calls: list[int] = []
        collector = self._collector(monkeypatch, calls)
        collector._collect_snapshot_sync()
        # Age the cache by hand rather than sleeping: 3s is inside the 5s window, so
        # no new query happens and the row is a repeat of the first measurement.
        touched, usage, sample_elapsed, measured = collector._remote_cache
        collector._remote_cache = (touched - 3.0, usage, sample_elapsed, measured - 3.0)
        snap = collector._collect_snapshot_sync()
        assert len(calls) == 1, "no new sstat query inside the window"
        assert 2.5 <= snap.usage_age_seconds <= 3.5, snap.usage_age_seconds

    def test_a_transient_sstat_failure_keeps_ageing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The retry clock moves; the measurement does not.

        The failure branch bumps the cache timestamp so retries are paced, and it used
        to be the only timestamp there — so a sample kept alive across repeated sstat
        failures would have reported itself freshly measured.
        """
        from slurmwatch import slurm

        calls: list[int] = []
        collector = self._collector(monkeypatch, calls)
        collector._collect_snapshot_sync()
        touched, usage, sample_elapsed, measured = collector._remote_cache
        collector._remote_cache = (touched - 9.0, usage, sample_elapsed, measured - 9.0)
        monkeypatch.setattr(
            slurm,
            "resolve_remote_usage",
            lambda job_id, node_count=1: slurm.RemoteUsage(
                rss_bytes=0, cpu_seconds=0.0, sampled=False
            ),
        )
        snap = collector._collect_snapshot_sync()
        assert snap.memory.current_bytes == 8 * 1024**3, "kept the last real sample"
        assert snap.usage_age_seconds >= 8.5, snap.usage_age_seconds
        # A SECOND sample after the failure: the failure branch rewrites the cache,
        # and writing `now` as the measurement time there would reset the age for
        # every later row while the numbers stayed frozen. One sample cannot see
        # that — the row being retried still carries the old stamp — so take another.
        again = collector._collect_snapshot_sync()
        assert again.memory.current_bytes == 8 * 1024**3
        assert again.usage_age_seconds >= 8.5, again.usage_age_seconds

    def test_on_node_rows_are_always_fresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On-node every sample re-reads the cgroup, so the age is 0 by construction."""
        ctx = self._remote_ctx()
        ctx.remote = False
        snap = TelemetryCollector(ctx)._collect_snapshot_sync()
        assert snap.usage_age_seconds == 0.0

    def test_a_payload_without_the_field_is_unknown_not_fresh(self) -> None:
        """A build that never reported it cannot be read as "measured just now"."""
        snap = _make_test_snapshot()
        payload = json.loads(snap.to_json())
        del payload["usage_age_seconds"]
        assert TelemetrySnapshot.from_dict(payload).usage_age_seconds == -1.0

    def test_the_age_reaches_csv_in_its_own_column(self) -> None:
        snap = _make_test_snapshot()
        snap.usage_age_seconds = 3.25
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        cells = dict(zip(header, snap.to_csv_row(max_gpus=0), strict=True))
        assert cells["usage_age_seconds"] == "3.25"


class TestGpuUnavailableReason:
    """Why GPU telemetry is missing must be reported, not guessed at.

    A monitor step beside a job that holds all its GPUs sees NVML succeed (via
    ``/dev/nvidiactl``, which Slurm leaves open to every step) and then enumerate
    zero devices — indistinguishable, from the boolean alone, from a node with no
    NVIDIA driver at all. slurmwatch used to report both as "no driver/pynvml, or a
    non-NVIDIA GPU", which on a GPU node is simply false.
    """

    @staticmethod
    def _gpu_ctx() -> JobContext:
        return JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=2,
            gpu_indices=[0, 2],
            gpu_uuids=[],
        )

    @staticmethod
    def _fake_procfs(tmp_path: Path, models: list[str]) -> Path:
        """A stand-in for /proc/driver/nvidia/gpus, mirroring the real layout.

        The driver writes one directory per device named by PCI address, each with
        an ``information`` file whose "Model:" line is tab-padded — reproduce that
        shape (not a tidied version of it) so the parser is tested against what it
        will actually meet.
        """
        root = tmp_path / "gpus"
        for i, model in enumerate(models):
            d = root / f"0000:{i:02x}:00.0"
            d.mkdir(parents=True)
            (d / "information").write_text(
                f"Model: \t\t {model}\nIRQ:   \t\t 18\nGPU UUID: \t GPU-{i}\n"
            )
        return root

    def test_zero_devices_on_a_gpu_node_is_devices_denied(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """NVML sees 0 devices but procfs lists 4 ⇒ withheld, not missing."""
        import sys

        from slurmwatch import collector as collector_mod

        fake = _FakePynvml()
        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 0))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        monkeypatch.setattr(
            collector_mod,
            "_NVIDIA_PROC_GPUS",
            self._fake_procfs(tmp_path, ["NVIDIA A100-PCIE-40GB"] * 4),
        )
        collector = TelemetryCollector(self._gpu_ctx())
        assert collector._init_nvml() is False
        assert collector._gpu_unavailable_reason == "devices_denied"
        assert collector._gpu_node_models == ["NVIDIA A100-PCIE-40GB"] * 4

    def test_an_unread_device_set_has_no_active_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``gpu_active_count`` is a SUM over the devices, so nothing read means 0.

        Measured live off-node against a 4 x A100 job: the payload carried
        ``gpu_count_requested: 4``, the model, the allocated indices 0-3 and
        ``gpu_active_count: 0`` — a right-sizing consumer reads that as four idle
        cards and advises dropping them, while the job was using all four. Nothing in
        the suite covered it: the neighbouring wire test builds exactly this state and
        asserted every other field. Producer-level on purpose — setting the attribute
        by hand would pass against a hardcoded 0.
        """
        import sys

        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 0))
        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        monkeypatch.setattr(
            collector_mod,
            "_NVIDIA_PROC_GPUS",
            self._fake_procfs(tmp_path, ["NVIDIA A100-PCIE-40GB"] * 4),
        )
        collector = TelemetryCollector(self._gpu_ctx())
        assert collector._init_nvml() is False
        snap = collector._collect_snapshot_sync()
        assert snap.gpus == []
        assert snap.gpu_count_requested == 2
        assert snap.gpu_active_count is None

    def test_a_job_that_asked_for_no_gpu_still_reports_zero_active(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """None means "unknown", so it must not swallow a real zero.

        A CPU-only job on a GPU node has nothing to be active: 0 is the fact, and
        turning it into an empty cell would make every CPU job's row look unreadable.
        """
        import sys

        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 0))
        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        monkeypatch.setattr(
            collector_mod,
            "_NVIDIA_PROC_GPUS",
            self._fake_procfs(tmp_path, ["NVIDIA A100-PCIE-40GB"] * 4),
        )
        ctx = self._gpu_ctx()
        ctx.gpu_count_requested = 0
        ctx.gpu_indices = []
        snap = TelemetryCollector(ctx)._collect_snapshot_sync()
        assert snap.gpu_active_count == 0
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        cells = dict(zip(header, snap.to_csv_row(max_gpus=0), strict=True))
        assert cells["gpu_active_count"] == "0"

    def test_readable_but_idle_gpus_are_a_measured_zero(self) -> None:
        """The distinction only helps if a genuine "read them, all idle" stays 0."""
        snap = _make_test_snapshot()
        snap.gpu_count_requested = 2
        snap.gpu_monitoring_available = True
        snap.gpu_active_count = 0  # read all of them, none busy
        snap.gpus = [
            GpuMetrics(
                index=i,
                uuid=f"GPU-{i}",
                name="A100",
                utilization_percent=0.0,
                memory_used_bytes=0,
                memory_total_bytes=42949672960,
                memory_utilization_percent=0.0,
                power_watts=41.0,
                temperature_celsius=32.0,
                throttling=False,
            )
            for i in range(2)
        ]
        back = TelemetrySnapshot.from_json(snap.to_json())
        assert back.gpu_active_count == 0
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        cells = dict(zip(header, snap.to_csv_row(max_gpus=0), strict=True))
        assert cells["gpu_active_count"] == "0", "a measured zero must not read as unknown"

    def test_reason_and_indices_reach_the_SNAPSHOT_not_just_the_collector(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The collector must COPY the cause and the indices onto every snapshot.

        Asserting on ``collector._gpu_unavailable_reason`` only proves the collector
        worked it out; the TUI, ``--json``, ``--log`` and the node switcher all read
        the *snapshot*. Without this test the whole plumbing could be replaced by a
        hardcoded ""/[] and every other test here would still pass — the same gap the
        job_name plumbing assertion elsewhere in this file exists to close. Note this
        deliberately does NOT use ``mock_slurm_env``: SLURMWATCH_MOCK makes ``start()``
        skip ``_init_nvml`` and synthesize GPU data, so the wire under test would
        never be exercised.
        """
        import sys

        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 0))
        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        monkeypatch.setattr(
            collector_mod,
            "_NVIDIA_PROC_GPUS",
            self._fake_procfs(tmp_path, ["NVIDIA A100-PCIE-40GB"] * 4),
        )
        ctx = self._gpu_ctx()
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is False
        snap = collector._collect_snapshot_sync()
        assert snap.gpu_monitoring_available is False
        assert snap.gpu_unavailable_reason == "devices_denied"
        assert snap.gpu_node_count == 4
        assert snap.gpu_node_model == "NVIDIA A100-PCIE-40GB"
        # The job's own allocation on this node, not the node's full device list.
        assert snap.gpu_allocated_indices == [0, 2]

    def test_zero_devices_with_no_nvidia_driver_stays_no_devices(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The honest "nothing here" case must NOT be relabelled as denied."""
        import sys

        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(_FakePynvml, "nvmlDeviceGetCount", staticmethod(lambda: 0))
        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        monkeypatch.setattr(collector_mod, "_NVIDIA_PROC_GPUS", tmp_path / "absent")
        collector = TelemetryCollector(self._gpu_ctx())
        assert collector._init_nvml() is False
        assert collector._gpu_unavailable_reason == "no_devices"
        assert collector._gpu_node_models == []

    def test_driver_not_loaded_is_no_driver(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys

        from slurmwatch import collector as collector_mod

        # The production code keys off the EXCEPTION CLASS NAME that pynvml raises,
        # so the fake's class must carry pynvml's exact name — a renamed stand-in
        # would test a branch the real library can never reach.
        class NVMLError_DriverNotLoaded(Exception):  # noqa: N801, N818
            pass

        class _NoDriver(_FakePynvml):
            @staticmethod
            def nvmlInit() -> None:
                raise NVMLError_DriverNotLoaded()

        monkeypatch.setitem(sys.modules, "pynvml", _NoDriver())
        monkeypatch.setattr(collector_mod, "_NVIDIA_PROC_GPUS", tmp_path / "absent")
        collector = TelemetryCollector(self._gpu_ctx())
        assert collector._init_nvml() is False
        assert collector._gpu_unavailable_reason == "no_driver"

    def test_missing_pynvml_is_no_pynvml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import builtins

        from slurmwatch import collector as collector_mod

        real_import = builtins.__import__

        def _no_pynvml(name: str, *a: object, **k: object) -> object:
            if name == "pynvml":
                raise ImportError("no pynvml")
            return real_import(name, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _no_pynvml)
        monkeypatch.setattr(
            collector_mod,
            "_NVIDIA_PROC_GPUS",
            self._fake_procfs(tmp_path, ["NVIDIA A100-PCIE-40GB"]),
        )
        collector = TelemetryCollector(self._gpu_ctx())
        assert collector._init_nvml() is False
        assert collector._gpu_unavailable_reason == "no_pynvml"
        # Still name the hardware: the GPUs are there, only the reader is absent.
        assert collector._gpu_node_models == ["NVIDIA A100-PCIE-40GB"]

    def test_cpu_only_job_reports_no_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job that asked for no GPU has nothing to explain."""
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            gpu_uuids=[],
        )
        collector = TelemetryCollector(ctx)
        assert collector._init_nvml() is False
        assert collector._gpu_unavailable_reason == ""


class TestNvidiaProcfsProbe:
    def test_parses_model_per_device_in_pci_order(self, tmp_path: Path) -> None:
        from slurmwatch.collector import _nvidia_node_gpu_models

        root = tmp_path / "gpus"
        for addr, model in (
            ("0000:31:00.0", "NVIDIA A100-PCIE-40GB"),
            ("0000:17:00.0", "NVIDIA H100"),
        ):
            (root / addr).mkdir(parents=True)
            (root / addr / "information").write_text(f"Model: \t\t {model}\nIRQ: \t 18\n")
        assert _nvidia_node_gpu_models(root) == ["NVIDIA H100", "NVIDIA A100-PCIE-40GB"]

    def test_absent_procfs_is_empty_not_an_error(self, tmp_path: Path) -> None:
        from slurmwatch.collector import _nvidia_node_gpu_models

        assert _nvidia_node_gpu_models(tmp_path / "nope") == []

    def test_unreadable_information_still_counts_the_device(self, tmp_path: Path) -> None:
        """The COUNT is what the "denied" message leans on; a nameless GPU is still one."""
        from slurmwatch.collector import _nvidia_node_gpu_models

        (tmp_path / "gpus" / "0000:17:00.0").mkdir(parents=True)
        assert _nvidia_node_gpu_models(tmp_path / "gpus") == [""]

    def test_common_model_collapses_only_when_unanimous(self) -> None:
        from slurmwatch.collector import _common_gpu_model

        assert _common_gpu_model(["A100", "A100"]) == "A100"
        assert _common_gpu_model(["A100", "H100"]) == ""
        assert _common_gpu_model([]) == ""
        assert _common_gpu_model(["", ""]) == ""


class TestNodeFabric:
    """The inter-NODE fabric — the number that explains a slow multi-node step.

    The GPU interconnect is intra-node only, and NVML's PCIe counters never see
    the all-reduce (GPUDirect RDMA moves data GPU->NIC without appearing as host
    PCIe traffic), so on a multi-node job neither says anything about the network
    the job actually depends on.
    """

    @staticmethod
    def _hca(
        tmp_path: Path,
        *,
        rx: int,
        tx: int,
        state: str = "4: ACTIVE",
        rate: str = "100 Gb/sec (2X HDR)",
        link_layer: str = "InfiniBand",
        dev: str = "mlx5_0",
        port: str = "1",
    ) -> Path:
        """A stand-in for /sys/class/infiniband mirroring the real layout."""
        root = tmp_path / "infiniband"
        pdir = root / dev / "ports" / port
        (pdir / "counters").mkdir(parents=True, exist_ok=True)
        (pdir / "state").write_text(state + "\n")
        (pdir / "rate").write_text(rate + "\n")
        (pdir / "link_layer").write_text(link_layer + "\n")
        (pdir / "counters" / "port_rcv_data").write_text(f"{rx}\n")
        (pdir / "counters" / "port_xmit_data").write_text(f"{tx}\n")
        return root

    def test_counters_are_four_octet_units_not_bytes(self, tmp_path: Path) -> None:
        """IBTA counts port_xmit_data/port_rcv_data in FOUR-OCTET units.

        Reading them as bytes under-reports the fabric by exactly 4x — the classic
        InfiniBand counter bug, and invisible without an explicit check because the
        number still looks plausible.
        """
        from slurmwatch.collector import _ib_ports

        root = self._hca(tmp_path, rx=1000, tx=250)
        (port,) = _ib_ports(root)
        assert port.rx_bytes == 4000
        assert port.tx_bytes == 1000

    def test_inactive_ports_are_skipped(self, tmp_path: Path) -> None:
        """A DOWN port's counters are stale and would dilute the rate."""
        from slurmwatch.collector import _ib_ports

        root = self._hca(tmp_path, rx=1, tx=1, state="1: DOWN")
        assert _ib_ports(root) == []

    def test_absent_sysfs_is_no_fabric(self, tmp_path: Path) -> None:
        from slurmwatch.collector import _ib_ports

        assert _ib_ports(tmp_path / "nope") == []

    def test_parses_rate_and_transport(self, tmp_path: Path) -> None:
        from slurmwatch.collector import _ib_ports

        (ib,) = _ib_ports(self._hca(tmp_path, rx=0, tx=0))
        assert ib.rate_gbps == 100.0
        assert ib.rate_label == "100 Gb/sec (2X HDR)"
        assert ib.kind == "InfiniBand"
        (roce,) = _ib_ports(
            self._hca(tmp_path / "b", rx=0, tx=0, link_layer="Ethernet", rate="25 Gb/sec")
        )
        assert roce.kind == "RoCE"
        assert roce.rate_gbps == 25.0

    def _collector(self) -> TelemetryCollector:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            gpu_uuids=[],
        )
        return TelemetryCollector(ctx)

    def test_first_sample_reports_no_rate_rather_than_a_fake_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cumulative counters need two samples; frame one must not claim 0 Gb/s."""
        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(collector_mod, "_IB_SYSFS", self._hca(tmp_path, rx=100, tx=100))
        c = self._collector()
        first = c._collect_fabric(1000.0)
        assert first is not None
        assert first.rates_known is False
        assert first.rx_gbps == 0.0 and first.tx_gbps == 0.0
        assert first.link_rate_gbps == 100.0  # the LINK is known immediately

    def test_rate_is_the_delta_in_gigabits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from slurmwatch import collector as collector_mod

        root = self._hca(tmp_path, rx=0, tx=0)
        monkeypatch.setattr(collector_mod, "_IB_SYSFS", root)
        c = self._collector()
        c._collect_fabric(1000.0)
        # +1e9 four-octet units = 4e9 bytes over 2s = 16 Gbit/s.
        self._hca(tmp_path, rx=1_000_000_000, tx=500_000_000)
        second = c._collect_fabric(1002.0)
        assert second is not None and second.rates_known is True
        assert second.rx_gbps == pytest.approx(16.0, abs=0.01)
        assert second.tx_gbps == pytest.approx(8.0, abs=0.01)

    def test_counter_reset_clamps_to_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An HCA reset / bounced port makes the counter go BACKWARDS."""
        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(collector_mod, "_IB_SYSFS", self._hca(tmp_path, rx=10**9, tx=10**9))
        c = self._collector()
        c._collect_fabric(1000.0)
        self._hca(tmp_path, rx=5, tx=5)  # counters reset
        after = c._collect_fabric(1001.0)
        assert after is not None
        assert after.rx_gbps == 0.0 and after.tx_gbps == 0.0

    def test_no_hca_yields_none_and_forgets_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(collector_mod, "_IB_SYSFS", tmp_path / "absent")
        c = self._collector()
        assert c._collect_fabric(1000.0) is None
        assert c._fabric_prev is None

    def test_fabric_survives_json_round_trip(self, tmp_path: Path) -> None:
        from slurmwatch.model import NodeFabric

        snap = _make_test_snapshot()
        snap.fabric = NodeFabric(
            ports=1,
            link_rate_gbps=100.0,
            kind="InfiniBand",
            rate_label="100 Gb/sec (2X HDR)",
            rx_gbps=50.435,
            tx_gbps=45.623,
            rates_known=True,
        )
        back = TelemetrySnapshot.from_json(snap.to_json())
        assert back.fabric is not None
        assert back.fabric.rx_gbps == 50.435
        assert back.fabric.link_rate_gbps == 100.0
        assert back.fabric.rates_known is True
        # A node streaming a build from before the field existed omits it.
        payload = json.loads(snap.to_json())
        payload.pop("fabric")
        assert TelemetrySnapshot.from_dict(payload).fabric is None


class TestFabricCsv:
    """--once DEFAULTS to CSV, so a JSON-only field is invisible to the common path.

    A multi-node right-sizing sweep that logs CSV would otherwise see no network
    at all and read the job as compute-bound while its all-reduce sits at 95% of
    the link. Same blind spot that once hid gpu_monitoring_available from CSV.
    """

    def _snap(self) -> TelemetrySnapshot:
        return _make_test_snapshot()

    def test_columns_exist_and_stay_aligned(self) -> None:
        from slurmwatch.model import NodeFabric

        snap = self._snap()
        snap.fabric = NodeFabric(
            ports=2,
            link_rate_gbps=200.0,
            kind="InfiniBand",
            rate_label="200 Gb/sec (4X NDR)",
            rx_gbps=88.5,
            tx_gbps=77.25,
            rates_known=True,
        )
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        row = snap.to_csv_row(max_gpus=0)
        assert len(header) == len(row), "header/row drifted apart"
        cells = dict(zip(header, row, strict=True))
        assert cells["fabric_kind"] == "InfiniBand"
        assert cells["fabric_link_rate_gbps"] == "200"
        assert cells["fabric_ports"] == "2"
        assert cells["fabric_rx_gbps"] == "88.5"
        assert cells["fabric_tx_gbps"] == "77.25"

    def test_no_hca_writes_empty_not_zero(self) -> None:
        """A hard 0 would read as a measured idle fabric — a different claim."""
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        cells = dict(zip(header, self._snap().to_csv_row(max_gpus=0), strict=True))
        for col in ("fabric_kind", "fabric_rx_gbps", "fabric_tx_gbps", "fabric_ports"):
            assert cells[col] == "", col

    def test_first_frame_reports_the_link_but_not_a_rate(self) -> None:
        from slurmwatch.model import NodeFabric

        snap = self._snap()
        snap.fabric = NodeFabric(ports=1, link_rate_gbps=100.0, kind="InfiniBand")
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        cells = dict(zip(header, snap.to_csv_row(max_gpus=0), strict=True))
        # The LINK is known immediately; the RATE needs two samples.
        assert cells["fabric_kind"] == "InfiniBand"
        assert cells["fabric_link_rate_gbps"] == "100"
        assert cells["fabric_rx_gbps"] == ""
        assert cells["fabric_tx_gbps"] == ""


class TestInterconnectCsv:
    """The GPU interconnect in CSV: enough to answer "is this job fabric-bound".

    The NxN topology matrix and per-device lists are not table-shaped and stay
    --json-only; flattening them badly would be worse than omitting them. What a
    right-sizing sweep needs is the fabric kind, the per-GPU ceiling, and traffic
    summed across devices.
    """

    def test_scalar_facts_and_summed_traffic(self) -> None:
        from slurmwatch.model import GpuInterconnect

        snap = _make_test_snapshot()
        snap.interconnect = GpuInterconnect(
            fabric="nvlink",
            per_gpu_gbps=300.0,
            devices=[0, 1],
            matrix=[["self", "NV6"], ["NV6", "self"]],
            nvlink_rx_gbps=[1.5, 2.5],
            nvlink_tx_gbps=[1.0, 1.0],
            pcie_rx_gbps=[0.01, 0.02],
            pcie_tx_gbps=[0.005, 0.005],
        )
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        row = snap.to_csv_row(max_gpus=0)
        assert len(header) == len(row)
        cells = dict(zip(header, row, strict=True))
        assert cells["gpu_interconnect"] == "nvlink"
        assert cells["gpu_interconnect_per_gpu_gbps"] == "300"
        # SUMMED across devices, not just the first one.
        assert cells["gpu_nvlink_rx_gbps"] == "4"
        assert cells["gpu_nvlink_tx_gbps"] == "2"
        assert cells["gpu_pcie_rx_gbps"] == "0.03"

    def test_unreadable_counters_write_blank_not_zero(self) -> None:
        """A PCIe-only node has no NVLink counters; 0 would claim idle NVLink."""
        from slurmwatch.model import GpuInterconnect

        snap = _make_test_snapshot()
        snap.interconnect = GpuInterconnect(
            fabric="pcie",
            devices=[1, 2],
            nvlink_rx_gbps=[],
            nvlink_tx_gbps=[],
            pcie_rx_gbps=[0.027, 0.03],
            pcie_tx_gbps=[0.007, 0.007],
        )
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        cells = dict(zip(header, snap.to_csv_row(max_gpus=0), strict=True))
        assert cells["gpu_nvlink_rx_gbps"] == ""
        assert cells["gpu_nvlink_tx_gbps"] == ""
        assert cells["gpu_pcie_rx_gbps"] == "0.057"

    def test_no_interconnect_at_all_is_blank(self) -> None:
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        cells = dict(zip(header, _make_test_snapshot().to_csv_row(max_gpus=0), strict=True))
        for col in ("gpu_interconnect", "gpu_pcie_rx_gbps", "gpu_nvlink_rx_gbps"):
            assert cells[col] == "", col


class TestPerNodeGpuMapOffNode:
    def test_map_is_derived_from_the_record_so_it_works_off_node(self) -> None:
        """It needs only `scontrol show job -d` — no cgroup, no NVML, no local state.

        It used to be set only on the on-node resolution path, which returns much
        later, so a login-node --json (and the cross-node GPU view) silently got {}
        for a multi-node job.
        """
        from slurmwatch.slurm import parse_gres_idx_by_node

        record = (
            "     Nodes=beagle3-0006 CPU_IDs=1-2,4-5 Mem=53248 GRES=gpu:2(IDX:1-2)\n"
            "     Nodes=beagle3-0020 CPU_IDs=2-5 Mem=53248 GRES=gpu:2(IDX:1-2)\n"
        )
        out = parse_gres_idx_by_node(record)
        assert out == {"beagle3-0006": [1, 2], "beagle3-0020": [1, 2]}


class TestMockFabric:
    """Demo mode must synthesize the fabric, never read the recording host's.

    Same rule the interconnect already followed. Without a mock branch the NET row
    either vanished (a login node has no HCA, so `--demo` silently lacked the
    feature) or published the host's REAL InfiniBand counters as the fake job's —
    including into the README GIF, which is rendered from demo mode.
    """

    @staticmethod
    def _ctx() -> JobContext:
        return JobContext(
            job_id="12345",
            username="u",
            partition="gpu",
            nodelist="cn[001-002]",
            hostname="cn001",
            cpus_allocated=8,
            mem_limit_bytes=64 * 1024**3,
            gpu_count_requested=4,
            gpu_indices=[0, 1, 2, 3],
            gpu_uuids=[],
            nodelist_resolved=["cn001", "cn002"],
        )

    def test_mock_synthesizes_a_fabric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        fab = TelemetryCollector(self._ctx())._collect_fabric(1000.0)
        assert fab is not None
        assert fab.kind == "InfiniBand"
        assert fab.link_rate_gbps == 200.0
        # A rate immediately: the demo has no second sample to wait for.
        assert fab.rates_known is True
        assert fab.rx_gbps > 0 and fab.tx_gbps > 0

    def test_mock_never_reads_real_sysfs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Point the reader at a populated tree and prove mock ignores it."""
        from slurmwatch import collector as collector_mod

        root = tmp_path / "infiniband"
        pdir = root / "mlx5_9" / "ports" / "1"
        (pdir / "counters").mkdir(parents=True)
        (pdir / "state").write_text("4: ACTIVE\n")
        (pdir / "rate").write_text("400 Gb/sec (8X NDR)\n")
        (pdir / "link_layer").write_text("InfiniBand\n")
        (pdir / "counters" / "port_rcv_data").write_text("7\n")
        (pdir / "counters" / "port_xmit_data").write_text("7\n")
        monkeypatch.setattr(collector_mod, "_IB_SYSFS", root)
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        fab = TelemetryCollector(self._ctx())._collect_fabric(1000.0)
        assert fab is not None
        # The synthetic link, NOT the 400 Gb/s tree above.
        assert fab.link_rate_gbps == 200.0
        assert "400" not in fab.rate_label

    def test_real_mode_still_reads_sysfs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The guard must not disable the real path when mock is off."""
        from slurmwatch import collector as collector_mod

        root = tmp_path / "infiniband"
        pdir = root / "mlx5_0" / "ports" / "1"
        (pdir / "counters").mkdir(parents=True)
        (pdir / "state").write_text("4: ACTIVE\n")
        (pdir / "rate").write_text("400 Gb/sec (8X NDR)\n")
        (pdir / "link_layer").write_text("InfiniBand\n")
        (pdir / "counters" / "port_rcv_data").write_text("7\n")
        (pdir / "counters" / "port_xmit_data").write_text("7\n")
        monkeypatch.setattr(collector_mod, "_IB_SYSFS", root)
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        fab = TelemetryCollector(self._ctx())._collect_fabric(1000.0)
        assert fab is not None
        assert fab.link_rate_gbps == 400.0


class TestMemoryReadingProvenance:
    """SW-3: off-node, the same field NAMES mean different things.

    `current_bytes` is sstat's MaxRSS — a lifetime high-water that never falls —
    `peak_bytes` is a copy of it, and nothing measures the page cache at all. The
    snapshot's `remote` flag said the reading came from elsewhere but not that the
    SEMANTICS changed, so a reader sizing --mem off `peak_bytes` could not tell a
    real high-water from a copy of one instantaneous sample.
    """

    def _remote_ctx(self) -> JobContext:
        return TestRemoteCollector()._remote_ctx()

    def test_the_off_node_reading_names_its_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch import slurm

        usage = slurm.RemoteUsage(rss_bytes=100 * 1024**3, cpu_seconds=1.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        _cpu, mem = TelemetryCollector(self._remote_ctx())._collect_remote(time.time())
        assert mem.source == "sstat"
        assert mem.cache_measured is False, "0 cache off-node is 'not measured'"
        assert mem.current_bytes == mem.peak_bytes, "both are the same MaxRSS high-water"

    def test_the_on_node_reading_says_cgroup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default has to stay right too, or every consumer reads "sstat"."""
        from slurmwatch.model import MemoryMetrics

        fresh = MemoryMetrics(
            current_bytes=1,
            limit_bytes=2,
            peak_bytes=1,
            usage_percent=50.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
        )
        assert fresh.source == "cgroup"
        assert fresh.cache_measured is True

    def test_csv_carries_the_source_beside_the_figures(self) -> None:
        from slurmwatch.model import TelemetrySnapshot

        header = TelemetrySnapshot.csv_header(1)
        assert "mem_source" in header and "mem_cache_measured" in header
        # Beside the cache figure they qualify, not appended after the GPU groups.
        assert header.index("mem_source") == header.index("mem_cache_bytes") + 1

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_an_unread_memory_stat_is_not_a_measured_zero_cache(
        self, tmp_path: Path, version: str
    ) -> None:
        """The same SW-3 rule, on the ON-node path. memory.stat is the only thing
        that measures cache, and two real paths reach the payload without it — a
        cgroup with no memory controller delegated (usage absent, so the /proc-RSS
        fallback answers) and a cgroup that vanished as the job ended. Both used to
        publish the untouched `cache_bytes: 0` as measured, which the TUI renders as
        "0.0 B" of reclaimable cache. Both readers, because the branch is written
        twice."""
        cg = tmp_path / version
        cg.mkdir()
        if version == "v2":
            (cg / "memory.current").write_text(str(4 * 1024**3))
            (cg / "memory.max").write_text(str(8 * 1024**3))
            key = "cgroup_v2_path"
        else:
            (cg / "memory.usage_in_bytes").write_text(str(4 * 1024**3))
            (cg / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
            key = "cgroup_v1_mem_path"
        # No memory.stat: the file the cache figure comes from never answered.
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, **{key: str(cg)})
        )._collect_memory()
        assert mem.current_bytes == 4 * 1024**3, "the rest of the reading is fine"
        assert mem.cache_bytes == 0
        assert mem.cache_measured is False

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_a_memory_stat_reporting_no_cache_is_a_measured_zero(
        self, tmp_path: Path, version: str
    ) -> None:
        """The control on the fix above: "measured zero" is a real answer and must
        keep saying so. Deriving the flag from the VALUE (`cache_bytes > 0`) passes
        the test above and fails this one — it would relabel every job that genuinely
        holds no page cache as unmeasured, and drop the peak-gap note that only
        appears where cache IS accounted for."""
        cg = tmp_path / version
        cg.mkdir()
        if version == "v2":
            (cg / "memory.current").write_text(str(4 * 1024**3))
            (cg / "memory.max").write_text(str(8 * 1024**3))
            (cg / "memory.stat").write_text("inactive_file 0\nactive_file 0\n")
            key = "cgroup_v2_path"
        else:
            (cg / "memory.usage_in_bytes").write_text(str(4 * 1024**3))
            (cg / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
            (cg / "memory.stat").write_text("total_inactive_file 0\ntotal_active_file 0\n")
            key = "cgroup_v1_mem_path"
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, **{key: str(cg)})
        )._collect_memory()
        assert mem.cache_bytes == 0
        assert mem.cache_measured is True

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_the_proc_rss_fallback_does_not_claim_to_be_a_cgroup_reading(
        self, tmp_path: Path, version: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The SW-3 rule applied to the FIGURE, not just the cache flag beside it.

        When no memory controller is delegated the F4 fallback sums
        /proc/<pid>/statm instead — ``_proc_rss_bytes``'s own docstring calls it
        "the memory analogue of the CPU /proc fallback", and CpuMetrics.source
        already publishes "proc" for that counter precisely because the two "are
        not comparable to each other". The memory side published the statm sum as
        ``source: "cgroup"``, so a --json/CSV consumer sizing --mem could not tell
        a memcg counter from a PID-sum that counts shared pages and sees nothing of
        processes that already exited. Both branches, because it is written twice.
        """
        cg = tmp_path / version
        cg.mkdir()
        if version == "v2":
            # memory.current absent = controller not delegated (F4).
            (cg / "memory.max").write_text(str(8 * 1024**3))
            key = "cgroup_v2_path"
        else:
            (cg / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
            key = "cgroup_v1_mem_path"
        collector = TelemetryCollector(_min_ctx(mem_limit_bytes=8 * 1024**3, **{key: str(cg)}))
        monkeypatch.setattr(collector, "_proc_rss_bytes", lambda: 3 * 1024**3)
        mem = collector._collect_memory()
        assert mem.current_bytes == 3 * 1024**3, "the F4 fallback still answers"
        assert mem.source == "proc", "a statm sum is not a cgroup reading"

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_a_readable_counter_says_cgroup_even_when_memory_stat_is_missing(
        self, tmp_path: Path, version: str
    ) -> None:
        """The control on the fix above, and it is not its mirror.

        The counter here IS readable, so the label must stay "cgroup" — a fix that
        relabels the whole on-node path, or derives the label from something that
        happens to be false on the fallback (``cache_measured``, or a zero figure),
        passes the test above and fails this one. memory.stat is deliberately absent
        so ``cache_measured`` is False while the reading is still a real memcg one:
        the two flags answer different questions and must not be wired together.
        """
        cg = tmp_path / version
        cg.mkdir()
        if version == "v2":
            (cg / "memory.current").write_text(str(4 * 1024**3))
            (cg / "memory.max").write_text(str(8 * 1024**3))
            key = "cgroup_v2_path"
        else:
            (cg / "memory.usage_in_bytes").write_text(str(4 * 1024**3))
            (cg / "memory.limit_in_bytes").write_text(str(8 * 1024**3))
            key = "cgroup_v1_mem_path"
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, **{key: str(cg)})
        )._collect_memory()
        assert mem.current_bytes == 4 * 1024**3
        assert mem.cache_measured is False, "no memory.stat, so the cache is unmeasured"
        assert mem.source == "cgroup", "the counter answered; only the cache did not"


class TestDemoHonoursOomThresholds:
    """SW-15 (secondary): three sites computed the OOM guard and only one honoured
    SLURMWATCH_OOM_WARN/_CRIT — the demo path hardcoded 85/90, so a user who lowered
    the thresholds had them silently ignored there."""

    def _mem(self, monkeypatch: pytest.MonkeyPatch, cfg: SlurmwatchConfig) -> MemoryMetrics:
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        from slurmwatch.slurm import _make_mock_job_context

        return TelemetryCollector(_make_mock_job_context("12345"), cfg)._collect_memory()

    def test_a_lowered_threshold_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The demo curve climbs to ~72%, deliberately under the 85% default — so
        only a threshold BELOW that can tell an evaluated guard from a hardcoded one."""
        cfg = SlurmwatchConfig(oom_warning_threshold=0.2, oom_critical_threshold=0.25)
        mem = self._mem(monkeypatch, cfg)
        assert mem.usage_percent >= 25.0, mem.usage_percent
        assert mem.oom_guard_warning is True
        assert mem.oom_guard_critical is True

    def test_the_default_demo_stays_quiet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The showcase must not open on a false amber alarm."""
        mem = self._mem(monkeypatch, SlurmwatchConfig())
        assert mem.oom_guard_warning is False and mem.oom_guard_critical is False


class TestDemoGpuFlag:
    """SW-5: `--demo --once --json` emitted four fully-populated GPUs beside a
    top-level `gpu_monitoring_available: false`, so a consumer that gates on the
    flag read the demo as GPU-less."""

    def test_demo_does_not_deny_the_gpus_it_synthesized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        from slurmwatch.slurm import _make_mock_job_context

        collector = TelemetryCollector(_make_mock_job_context("12345"))
        snap = collector._collect_snapshot_sync()
        assert snap.gpus, "the demo synthesizes devices"
        assert snap.gpu_monitoring_available is True


class TestLauncherContentionWiring:
    """Round 19 characterised `SLURMWATCH_MONITOR_STEP` correctly and left its
    detector untested: two aimed-wrong attempts, then a recipe. Both HALVES had unit
    coverage (`_any_launcher_pid`, `_detect_launchers`) — the WIRING between them did
    not, which is the part that decides whether a user ever sees the warning.

    What it answers: "is the srun/mpirun you just started stuck behind the step
    slurmwatch itself is holding?" — self-interference, not CPU attribution, and
    deliberately not a payload field. The underlying situation is real: round 19
    measured a 4-CPU `srun` blocking indefinitely inside a fully-subscribed 4-CPU
    allocation, with no explanation from Slurm.
    """

    def _collector(self, monkeypatch: pytest.MonkeyPatch, *, pids: set[int]) -> TelemetryCollector:
        from slurmwatch.slurm import _make_mock_job_context

        ctx = _make_mock_job_context("12345")
        ctx.remote = False
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        collector = TelemetryCollector(ctx, SlurmwatchConfig())
        monkeypatch.setattr(collector, "_get_job_pids", lambda: pids)
        monkeypatch.setattr(collector, "_collect_cpu", lambda *a, **k: CpuMetrics(1, 0, 0.0, 0.0))
        monkeypatch.setattr(collector, "_collect_gpus", lambda *a, **k: [])
        return collector

    def _saw_launcher(self, monkeypatch: pytest.MonkeyPatch, pids: set[int]) -> bool:
        collector = self._collector(monkeypatch, pids=pids)
        collector._collect_snapshot_sync()
        return collector.launcher_present

    def test_a_launcher_in_the_jobs_cgroup_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURMWATCH_MONITOR_STEP", "1")
        monkeypatch.setattr(
            "slurmwatch.collector._read_pid_comm", lambda pid: "srun" if pid == 42 else "python3"
        )
        assert self._saw_launcher(monkeypatch, {41, 42}) is True

    def test_the_scan_only_runs_for_the_hop_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A normal on-node run must pay nothing for a warning about a step it
        isn't holding — the env var is what marks slurmwatch's own hop."""
        monkeypatch.delenv("SLURMWATCH_MONITOR_STEP", raising=False)
        monkeypatch.setattr("slurmwatch.collector._read_pid_comm", lambda pid: "srun")
        assert self._saw_launcher(monkeypatch, {42}) is False

    def test_no_launcher_means_no_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_MONITOR_STEP", "1")
        monkeypatch.setattr("slurmwatch.collector._read_pid_comm", lambda pid: "python3")
        assert self._saw_launcher(monkeypatch, {41, 42}) is False

    def test_the_demo_never_fabricates_contention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--demo has no cgroup to scan, so the flag stays down. Note the `not
        self._mock` clause in the collector is REDUNDANT with the empty pid set on
        that path (a mutation that removes it changes nothing) — it is kept as
        defence in depth, not because it is the mechanism."""
        monkeypatch.setenv("SLURMWATCH_MONITOR_STEP", "1")
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        from slurmwatch.slurm import _make_mock_job_context

        collector = TelemetryCollector(_make_mock_job_context("12345"), SlurmwatchConfig())
        collector._collect_snapshot_sync()
        assert collector.launcher_present is False
        assert collector._mock is True

    def test_the_off_node_path_never_claims_contention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Off-node there are no PIDs to scan, and the step being contended for
        wouldn't be ours — so the flag must clear, not go stale."""
        from slurmwatch import slurm

        monkeypatch.setenv("SLURMWATCH_MONITOR_STEP", "1")
        usage = slurm.RemoteUsage(rss_bytes=1024, cpu_seconds=1.0, sampled=True)
        monkeypatch.setattr(slurm, "resolve_remote_usage", lambda job_id, node_count=1: usage)
        collector = TelemetryCollector(TestRemoteCollector()._remote_ctx(), SlurmwatchConfig())
        collector.launcher_present = True  # a stale True from an earlier on-node frame
        collector._collect_snapshot_sync()
        assert collector.launcher_present is False


class TestFabricCapacityIsTheWholeNode:
    """Found by audit: `_collect_fabric` sums rx/tx across every active port, so the
    only honest denominator is the ports' summed rate. Reporting the sum against ONE
    port's rate made a busy 2-HCA node read "180% of 100 Gb/s link" — impossible on
    its face, and only visible on the multi-HCA sites this has never run on."""

    def _fabric(self, monkeypatch: pytest.MonkeyPatch, rates: list[float]) -> object:
        from slurmwatch.collector import _IbPort
        from slurmwatch.slurm import _make_mock_job_context

        ports = [
            _IbPort(
                device=f"mlx5_{i}",
                port="1",
                rx_bytes=0,
                tx_bytes=0,
                rate_gbps=rate,
                rate_label=f"{rate:g} Gb/sec (4X NDR)",
                kind="InfiniBand",
            )
            for i, rate in enumerate(rates)
        ]
        monkeypatch.setattr("slurmwatch.collector._ib_ports", lambda *a, **k: ports)
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        collector = TelemetryCollector(_make_mock_job_context("12345"), SlurmwatchConfig())
        collector._mock = False
        return collector._collect_fabric(time.time())

    def test_two_hcas_report_the_summed_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fab = self._fabric(monkeypatch, [100.0, 100.0])
        assert fab is not None
        assert fab.ports == 2  # type: ignore[attr-defined]
        assert fab.link_rate_gbps == 100.0  # type: ignore[attr-defined]
        assert fab.link_rate_total_gbps == 200.0  # type: ignore[attr-defined]

    def test_one_hca_keeps_both_figures_equal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fab = self._fabric(monkeypatch, [100.0])
        assert fab is not None
        assert fab.link_rate_gbps == fab.link_rate_total_gbps == 100.0  # type: ignore[attr-defined]

    def test_mixed_rates_sum_rather_than_multiply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ports x max would overstate a node whose HCAs differ (200+100, not 400)."""
        fab = self._fabric(monkeypatch, [200.0, 100.0])
        assert fab is not None
        assert fab.link_rate_gbps == 200.0  # type: ignore[attr-defined]
        assert fab.link_rate_total_gbps == 300.0  # type: ignore[attr-defined]

    def test_the_csv_carries_both(self) -> None:
        from slurmwatch.model import TelemetrySnapshot

        header = TelemetrySnapshot.csv_header(1)
        assert "fabric_link_rate_gbps" in header
        assert header.index("fabric_link_rate_total_gbps") == (
            header.index("fabric_link_rate_gbps") + 1
        )


def _audit_gpu(**over: object) -> GpuMetrics:
    """A GpuMetrics with every required field filled, for the audit tests below."""
    base: dict[str, object] = {
        "index": 0,
        "uuid": "u",
        "name": "A100",
        "utilization_percent": 0.0,
        "memory_used_bytes": 0,
        "memory_total_bytes": 40 * 1024**3,
        "memory_utilization_percent": 0.0,
        "power_watts": 0.0,
        "temperature_celsius": 0.0,
        "throttling": False,
    }
    base.update(over)
    return GpuMetrics(**base)  # type: ignore[arg-type]


def _make_snapshot_with_gpus(n: int) -> TelemetrySnapshot:
    """A snapshot with ``n`` devices, for CSV width checks."""
    from slurmwatch.model import CpuMetrics, MemoryMetrics, TelemetrySnapshot

    return TelemetrySnapshot(
        timestamp=1234567890.0,
        job_id="12345",
        step_id="0",
        hostname="cn001",
        elapsed_seconds=1,
        cpu=CpuMetrics(cores_allocated=1, usage_ns=0, usage_percent=0.0),
        memory=MemoryMetrics(
            current_bytes=0,
            limit_bytes=1,
            peak_bytes=0,
            usage_percent=0.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
        ),
        gpus=[_audit_gpu(index=i, uuid=f"u{i}") for i in range(n)],
    )


class TestPerProcessGpuShareProvenance:
    """Found by audit: `nvmlDeviceGetProcessUtilization` is optional — NOT_SUPPORTED
    on MIG slices and old drivers, NO_PERMISSION where the process APIs are
    restricted, and absent entirely on some pynvml builds. Its failure left
    `process_utilization_percent` at 0.0 with no flag, so "this job used none of the
    GPU" and "NVML wouldn't say" were the same value — the exact confusion the four
    sibling flags (util/memory/power/temperature_available) exist to prevent, and the
    one a right-sizing consumer would read as "drop the GPU"."""

    def test_the_flag_defaults_to_measured(self) -> None:
        assert _audit_gpu().process_utilization_available is True

    def test_the_share_line_shows_no_number_when_it_was_not_measured(self) -> None:
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        unread = _audit_gpu(
            utilization_percent=99.0,
            process_utilization_percent=0.0,
            process_utilization_available=False,
        )
        line = screen._gpu_share_line(unread, SlurmwatchConfig())
        assert "0% compute" not in line, line
        assert "— compute" in line

    def test_a_genuine_zero_is_still_shown(self) -> None:
        from slurmwatch.tui import ResourceDetailScreen

        screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
        idle = _audit_gpu(process_utilization_percent=0.0)
        assert "0% compute" in screen._gpu_share_line(idle, SlurmwatchConfig())

    def test_the_csv_carries_the_flag_beside_the_figure(self) -> None:
        from slurmwatch.model import TelemetrySnapshot

        header = TelemetrySnapshot.csv_header(1)
        assert "gpu_0_proc_util_available" in header
        assert header.index("gpu_0_proc_util_available") > header.index("gpu_0_proc_util_percent")

    def _collect(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch, nvml: object
    ) -> GpuMetrics:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", nvml)
        ctx = JobContext(
            job_id="12345",
            username="testuser",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=16,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=1,
            gpu_indices=[0],
            step_id="0",
            uid=1001,
            job_start_time=1000.0,
            cgroup_v2_path=str(fake_cgroup_v2_job),
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object()]
        collector._nvml_handle_info = {0: ("GPU-test", "A100-SXM4-80GB")}
        (gpu,) = collector._collect_gpus()
        return gpu

    def test_a_readable_share_is_flagged_measured(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gpu = self._collect(fake_cgroup_v2_job, monkeypatch, _FakePynvml())
        assert gpu.process_utilization_percent == 60.0
        assert gpu.process_utilization_available is True

    def test_an_unsupported_query_is_flagged_unmeasured(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The MIG / old-driver / NO_PERMISSION case: NVML raises, so the 0.0 that
        lands in the payload is not a reading."""

        class _NoProcUtil(_FakePynvml):
            @staticmethod
            def nvmlDeviceGetProcessUtilization(h: object, ts: int) -> list[_FakePUtil]:
                raise _FakePynvml.NVMLError("NOT_SUPPORTED")

        gpu = self._collect(fake_cgroup_v2_job, monkeypatch, _NoProcUtil())
        assert gpu.process_utilization_percent == 0.0
        assert gpu.process_utilization_available is False
        # The rest of the device must still be reported — one optional query
        # failing may not cost the whole card.
        assert gpu.utilization_percent == 75.0
        assert gpu.process_memory_bytes == 18 * 1024**3

    def test_no_job_pids_is_not_a_measured_zero(
        self, fake_cgroup_v2_job: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing to attribute is not "the job used 0%"."""
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", _FakePynvml())
        ctx = JobContext(
            job_id="12345",
            username="testuser",
            partition="gpu",
            nodelist="cn001",
            hostname="cn001",
            cpus_allocated=16,
            mem_limit_bytes=8 * 1024**3,
            gpu_count_requested=1,
            gpu_indices=[0],
            step_id="0",
            uid=1001,
            job_start_time=1000.0,
            cgroup_v2_path=str(fake_cgroup_v2_job),
        )
        collector = TelemetryCollector(ctx)
        collector._nvml_initialized = True
        collector._nvml_handles = [object()]
        collector._nvml_handle_info = {0: ("GPU-test", "A100-SXM4-80GB")}
        (gpu,) = collector._collect_gpus(job_pids=set())
        assert gpu.process_utilization_available is False

    def test_the_csv_row_still_matches_its_header(self) -> None:
        """The per-GPU block has a fixed width; adding a column without updating
        _GPU_COLS silently shifts every later field."""
        from slurmwatch.model import TelemetrySnapshot

        for n in (0, 1, 4):
            snap = _make_snapshot_with_gpus(n)
            assert len(snap.to_csv_row(n)) == len(TelemetrySnapshot.csv_header(n)), n


class TestFromDictDoesNotInventProvenance:
    """`from_dict` is the node switcher's entry point and is documented as tolerant
    of version skew between nodes. Tolerating a MISSING field is not the same as
    asserting its flattering default: three fields say whether the figure beside
    them is a measurement, and their dataclass defaults are written for the
    collector (where the answer is yes). A node streaming from an older build omits
    them — taking the defaults on faith would present that node's sstat reading as
    live cgroup data, its unmeasured 0 cache as "no cache", and an unreadable GPU
    share as a measured 0%."""

    OLD_PAYLOAD: dict[str, object] = {
        "timestamp": 1.0,
        "job_id": "12345",
        "step_id": "0",
        "hostname": "cn001",
        "elapsed_seconds": 10,
        "cpu": {"cores_allocated": 4, "usage_ns": 1, "usage_percent": 5.0},
        "memory": {
            "current_bytes": 1,
            "limit_bytes": 2,
            "peak_bytes": 1,
            "usage_percent": 50.0,
            "oom_guard_warning": False,
            "oom_guard_critical": False,
            "cache_bytes": 0,
        },
        "gpus": [
            {
                "index": 0,
                "uuid": "u",
                "name": "A100",
                "utilization_percent": 50.0,
                "memory_used_bytes": 1,
                "memory_total_bytes": 2,
                "memory_utilization_percent": 50.0,
                "power_watts": 1.0,
                "temperature_celsius": 40.0,
                "throttling": False,
                "process_utilization_percent": 10.0,
            }
        ],
    }

    def test_an_unstated_source_is_unknown_not_cgroup(self) -> None:
        from slurmwatch.model import TelemetrySnapshot

        snap = TelemetrySnapshot.from_dict(dict(self.OLD_PAYLOAD))
        assert snap.memory.source == "unknown"
        assert snap.memory.cache_measured is False
        assert snap.gpus[0].process_utilization_available is False

    def test_a_stated_value_is_preserved_exactly(self) -> None:
        from slurmwatch.model import TelemetrySnapshot

        payload = json.loads(_make_snapshot_with_gpus(1).to_json())
        payload["memory"]["source"] = "sstat"
        payload["memory"]["cache_measured"] = False
        payload["gpus"][0]["process_utilization_available"] = True
        snap = TelemetrySnapshot.from_dict(payload)
        assert snap.memory.source == "sstat"
        assert snap.memory.cache_measured is False
        assert snap.gpus[0].process_utilization_available is True

    def test_the_collector_still_defaults_to_measured(self) -> None:
        """The dataclass defaults serve the collector, where the answer IS yes —
        only the deserializer treats absence as unknown."""
        from slurmwatch.model import MemoryMetrics

        fresh = MemoryMetrics(
            current_bytes=1,
            limit_bytes=2,
            peak_bytes=1,
            usage_percent=50.0,
            oom_guard_warning=False,
            oom_guard_critical=False,
        )
        assert fresh.source == "cgroup" and fresh.cache_measured is True

    def test_an_unknown_future_key_is_still_ignored(self) -> None:
        """The skew tolerance this method exists for must survive the change."""
        from slurmwatch.model import TelemetrySnapshot

        payload = json.loads(_make_snapshot_with_gpus(1).to_json())
        payload["future_top_level"] = True
        payload["memory"]["future_field"] = 7
        payload["gpus"][0]["future_gpu_field"] = "x"
        assert TelemetrySnapshot.from_dict(payload).job_id == "12345"


class TestReapedChildCpu:
    """SW-24: on a cluster with no per-job cpuacct cgroup, the /proc sum read only
    `utime`/`stime` — so a job whose work happens in short-lived children lost it the
    moment each child was reaped. Measured on a real R `furrr multisession` job whose
    8 workers `system()` a 1 s shell in a loop: a true 8.00 of 8 cores read 5.75 at
    the TUI's 0.5 s default, 2.96 at the headless 1 s default, and 0.1 at 30 s.

    The naive fix (also read `cutime`/`cstime`) double-counts, which would be worse
    than the bug: a live child's CPU is credited directly, then again when its parent
    reaps it and the parent's `cutime` jumps. These tests pin both halves.
    """

    def _coll(self) -> TelemetryCollector:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=8,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )  # no cgroup paths -> forces the /proc accumulator
        return TelemetryCollector(ctx)

    def _patch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        table: dict[int, tuple[int, int, int]],
        alive: set[int],
    ) -> None:
        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(
            collector_mod,
            "_read_pid_cpu",
            lambda pid: collector_mod._PidCpu(*table[pid]) if pid in table else None,
        )
        monkeypatch.setattr(collector_mod, "_pid_alive", lambda pid: pid in alive)

    def test_a_reaped_childs_cpu_is_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The headline: a child born and reaped entirely between two polls is never
        sighted, and used to contribute nothing at all."""
        coll = self._coll()
        # Worker 100 (own 10) has reaped children worth 400 ticks. No child is live.
        self._patch(monkeypatch, {100: (10, 400, 1)}, {100})
        coll._read_cpu_ns({100})
        assert coll._proc_cpu_accum_ticks == 410

    def test_no_double_count_when_an_in_job_parent_reaps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trap. Child 200 is credited 100 ticks while alive; when worker 100
        reaps it, 100's cutime jumps by the same 100. The total must be 100 + the
        worker's own, not 200."""
        coll = self._coll()
        table = {100: (5, 0, 1), 200: (100, 0, 100)}  # 200's parent is 100, in the job
        self._patch(monkeypatch, table, {100, 200})
        coll._read_cpu_ns({100, 200})
        assert coll._proc_cpu_accum_ticks == 105

        # Child exits; the parent's cutime absorbs its whole subtree total.
        table = {100: (5, 100, 1)}
        self._patch(monkeypatch, table, {100})
        coll._read_cpu_ns({100})
        assert coll._proc_cpu_accum_ticks == 105, "the child's ticks were counted twice"

    def test_an_orphans_cpu_survives_its_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other half of the same workload: a PSOCK worker reparented to PID 1.
        Nothing inside the job will ever report its CPU, so the accumulator must keep
        it — this is what makes the `LONG` variant read exactly 8.00."""
        coll = self._coll()
        self._patch(monkeypatch, {100: (5, 0, 1), 900: (800, 0, 1)}, {100, 900})
        coll._read_cpu_ns({100, 900})
        assert coll._proc_cpu_accum_ticks == 805

        self._patch(monkeypatch, {100: (5, 0, 1)}, {100})  # orphan 900 exits
        coll._read_cpu_ns({100})
        assert coll._proc_cpu_accum_ticks == 805, "an orphan's CPU was discarded"

    def test_the_total_never_goes_backwards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Monotonicity is what the accumulator exists for; the reconciliation must
        not be able to drive it negative."""
        coll = self._coll()
        self._patch(monkeypatch, {100: (0, 0, 1), 200: (50, 0, 100)}, {100, 200})
        coll._read_cpu_ns({100, 200})
        # Parent 100 exits WITHOUT its cutime ever being observed, and its own parent
        # (1) is outside the job; child 200 then exits with 100 in the job's set.
        self._patch(monkeypatch, {}, set())
        coll._read_cpu_ns(set())
        assert coll._proc_cpu_accum_ticks >= 0

    def test_the_stat_parser_reads_all_four_fields(self) -> None:
        from slurmwatch.collector import _parse_stat_cpu

        # A real-shaped line: comm contains a space AND parentheses.
        fields = ["S", "7", "7", "7", "-1", "0", "0", "0", "0", "0", "0", "11", "12", "13", "14"]
        line = "42 (R (worker) ) " + " ".join(fields)
        parsed = _parse_stat_cpu(line)
        assert parsed is not None
        assert parsed.own == 11 + 12
        assert parsed.children == 13 + 14
        assert parsed.ppid == 7

    def test_a_truncated_stat_line_is_refused(self) -> None:
        from slurmwatch.collector import _parse_stat_cpu

        assert _parse_stat_cpu("42 (sh) S 1 2 3") is None
        assert _parse_stat_cpu("garbage") is None


class TestEffectiveCoresRespectsTheCpusetCeiling:
    """SW-25: `effective_cores` read 8.1-8.2 on a job confined to
    `cpuset.cpus=14-21` — eight CPUs, so more than 8.00 is not physically available
    and the figure was a sampling artifact. It also contradicted `usage_percent`,
    which IS clamped, so one --json record said 100.0% and 8.2/8 at once.

    The uncapped behaviour is deliberate and must survive where there is no cpuset
    confinement: an over-subscribed job on a ConstrainCores=no node still has to show
    that it used more than it asked for."""

    def _coll(self, cores: int = 8) -> TelemetryCollector:
        ctx = JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=cores,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
        )
        return TelemetryCollector(ctx, SlurmwatchConfig())

    def test_a_confined_job_is_capped_at_its_cpuset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        coll = self._coll()
        monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(14, 22)))  # 8 CPUs
        monkeypatch.setattr("os.cpu_count", lambda: 48)  # of a 48-CPU node
        assert coll._cpu_affinity_ceiling({4242}) == 8

    def test_an_unconfined_job_stays_uncapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Affinity covering the whole node is not a constraint — keep the
        over-subscription signal."""
        coll = self._coll()
        monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(48)))
        monkeypatch.setattr("os.cpu_count", lambda: 48)
        assert coll._cpu_affinity_ceiling({4242}) is None

    def test_the_ceiling_is_resolved_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job's cpuset does not change while it runs; this must not add a syscall
        per PID per frame."""
        coll = self._coll()
        calls: list[int] = []

        def _aff(pid: int) -> set[int]:
            calls.append(pid)
            return set(range(8))

        monkeypatch.setattr("os.sched_getaffinity", _aff)
        monkeypatch.setattr("os.cpu_count", lambda: 48)
        for _ in range(5):
            coll._cpu_affinity_ceiling({1, 2, 3})
        assert len(calls) == 1, calls

    def test_an_unreadable_affinity_does_not_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        coll = self._coll()

        def _boom(pid: int) -> set[int]:
            raise ProcessLookupError(pid)

        monkeypatch.setattr("os.sched_getaffinity", _boom)
        assert coll._cpu_affinity_ceiling({4242}) is None

    def test_the_reported_figure_cannot_exceed_the_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: a delta implying 8.2 cores on an 8-CPU cpuset reports 8.0, and
        agrees with the clamped percent instead of contradicting it."""
        from slurmwatch import collector as collector_mod

        coll = self._coll()
        monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(8)))
        monkeypatch.setattr("os.cpu_count", lambda: 48)
        monkeypatch.setattr(collector_mod, "_pid_alive", lambda pid: True)
        ticks = {100: 0}
        monkeypatch.setattr(
            collector_mod,
            "_read_pid_cpu",
            lambda pid: collector_mod._PidCpu(own=ticks[pid], children=0, ppid=1),
        )
        coll._collect_cpu({100})
        # 8.2 cores' worth of ticks over a 1s window.
        ticks = {100: int(8.2 * collector_mod._CLK_TCK)}
        assert coll._prev_timestamp is not None
        coll._prev_timestamp -= 1.0
        cpu = coll._collect_cpu({100})
        assert cpu.effective_cores <= 8.0, cpu.effective_cores
        assert cpu.usage_percent == 100.0


class TestSimulatedTelemetryIsFlagged:
    """SW-30: `--demo` / SLURMWATCH_MOCK=1 produced 3438 bytes of plausible CPU/memory/
    GPU figures for a job that does not exist, and the payload had nothing in it to say
    so. The flag is documented and the dashboard is obviously a demo to a human, but
    `--once --json` is the form a pipeline consumes with nobody reading it — and the env
    var is a documented equivalent, so a wrapper or a leftover export turns simulated
    figures into ingested measurement."""

    def test_the_mock_collector_marks_its_snapshots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        collector = TelemetryCollector(_min_ctx(mem_limit_bytes=8 * 1024**3))
        snap = collector._collect_snapshot_sync()
        assert snap.mock is True
        assert json.loads(snap.to_json())["mock"] is True, "top level, not nested"

    def test_a_real_snapshot_says_false(self) -> None:
        snap = _make_test_snapshot()
        assert snap.mock is False
        assert json.loads(snap.to_json())["mock"] is False

    def test_a_real_COLLECTOR_snapshot_says_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Through the collector, not a hand-built snapshot: the mutation that hardcodes
        the flag to True survives a test whose object never went through the code that
        sets it."""
        monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
        collector = TelemetryCollector(_min_ctx(mem_limit_bytes=8 * 1024**3))
        assert collector._collect_snapshot_sync().mock is False

    def test_the_csv_carries_it_too(self) -> None:
        """The --log half, which is what a long-running pipeline reads."""
        header = TelemetrySnapshot.csv_header(max_gpus=0)
        assert "mock" in header
        snap = _make_test_snapshot()
        snap.mock = True
        row = dict(zip(header, snap.to_csv_row(0), strict=True))
        assert row["mock"] == "1"

    def test_an_older_payload_is_not_assumed_simulated(self) -> None:
        """A build with no marker says nothing; False is the safe reading, and such a
        payload still carries memory.source == "mock" if it was a demo."""
        payload = json.loads(_make_test_snapshot().to_json())
        assert "mock" in payload
        payload.pop("mock")
        assert TelemetrySnapshot.from_dict(payload).mock is False

    def test_a_stated_marker_is_believed_either_way(self) -> None:
        payload = json.loads(_make_test_snapshot().to_json())
        for stated in (True, False):
            payload["mock"] = stated
            assert TelemetrySnapshot.from_dict(payload).mock is stated

    def test_appending_to_a_log_written_before_the_column_existed(self) -> None:
        """Their note: "a new column is a schema change, so the CSV half should go
        through the same drift path SW-25 covers." It does — by name, so the 100
        columns after the insertion point do not shift."""
        from slurmwatch.cli import _conform_csv_row, _csv_append_layout

        snap = _make_test_snapshot()
        current = TelemetrySnapshot.csv_header(0)
        row = snap.to_csv_row(0)
        truth = dict(zip(current, row, strict=True))
        older = [c for c in current if c != "mock"]
        assert len(older) == len(current) - 1
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "old.csv"
            with log.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(older)
                writer.writerow([truth[c] for c in older])
            cmap = _csv_append_layout(str(log), "excel", 0, 0)
            assert cmap is not None
            with log.open("a", newline="") as handle:
                csv.writer(handle).writerow(_conform_csv_row(row, cmap))
            with log.open() as handle:
                assert {len(r) for r in csv.reader(handle)} == {len(older)}
            with log.open() as handle:
                parsed = list(csv.DictReader(handle))
            assert parsed[0] == parsed[1], "nothing after the insertion point shifted"


class TestThePayloadCarriesTheTimeBudgetDenominator:
    """Found by checking slurmwatch's numbers against independent Slurm sources: the
    elapsed figure is exact (282279 s == squeue's 3-06:24:39 == scontrol's RunTime), but
    the payload had no wall-clock LIMIT to compare it against — while the machine
    surface for somebody ELSE's job (`_foreign_facts`) did carry `time_limit_seconds`.
    "How much of my budget is gone" is the most common reason to look at a running job,
    and it was the one arithmetic a consumer could not do."""

    def test_the_collector_passes_the_limit_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx = _min_ctx(mem_limit_bytes=8 * 1024**3)
        ctx.time_limit_seconds = 864000  # 10-00:00:00, as scontrol prints it
        collector = TelemetryCollector(ctx)
        assert collector._collect_snapshot_sync().time_limit_seconds == 864000

    def test_a_job_with_no_limit_says_none_not_zero(self) -> None:
        """0 would read as "no time left"; None is "no limit"."""
        ctx = _min_ctx(mem_limit_bytes=8 * 1024**3)
        ctx.time_limit_seconds = None
        assert TelemetryCollector(ctx)._collect_snapshot_sync().time_limit_seconds is None

    def test_json_and_csv_both_carry_it(self) -> None:
        snap = _make_test_snapshot()
        snap.time_limit_seconds = 864000
        assert json.loads(snap.to_json())["time_limit_seconds"] == 864000
        header = TelemetrySnapshot.csv_header(0)
        assert "time_limit_seconds" in header
        row = dict(zip(header, snap.to_csv_row(0), strict=True))
        assert row["time_limit_seconds"] == "864000"
        assert row["elapsed_seconds"], "and the numerator is still beside it"

    def test_the_csv_leaves_it_empty_when_there_is_no_limit(self) -> None:
        snap = _make_test_snapshot()
        snap.time_limit_seconds = None
        row = dict(zip(TelemetrySnapshot.csv_header(0), snap.to_csv_row(0), strict=True))
        assert row["time_limit_seconds"] == "", "empty, not 0"

    def test_from_dict_round_trips_both_states(self) -> None:
        for limit in (864000, None):
            snap = _make_test_snapshot()
            snap.time_limit_seconds = limit
            back = TelemetrySnapshot.from_dict(json.loads(snap.to_json()))
            assert back.time_limit_seconds == limit

    def test_an_older_payload_without_the_key_is_accepted(self) -> None:
        payload = json.loads(_make_test_snapshot().to_json())
        payload.pop("time_limit_seconds")
        assert TelemetrySnapshot.from_dict(payload).time_limit_seconds is None


class TestThePayloadCanAttributeItsRows:
    """Found by diffing the fact sets of slurmwatch's own surfaces against each other:
    `partition`, `owner`, `account` and `qos` are all resolved into JobContext, the
    foreign-job payload already carries `owner`/`partition`, and the JOB card shows
    account/QOS to a human — but the telemetry payload had none of them. So a `--log`
    file accumulating rows across jobs could not be grouped by the two axes every
    cluster reports on."""

    @staticmethod
    def _ctx() -> Any:
        ctx = _min_ctx(mem_limit_bytes=8 * 1024**3)
        ctx.partition = "build"
        ctx.username = "youzhi"
        ctx.account = "rcc-staff"
        ctx.qos = "build"
        return ctx

    def test_the_collector_carries_all_four(self) -> None:
        snap = TelemetryCollector(self._ctx())._collect_snapshot_sync()
        assert (snap.partition, snap.owner, snap.account, snap.qos) == (
            "build",
            "youzhi",
            "rcc-staff",
            "build",
        )

    def test_json_carries_them(self) -> None:
        payload = json.loads(TelemetryCollector(self._ctx())._collect_snapshot_sync().to_json())
        assert payload["partition"] == "build"
        assert payload["owner"] == "youzhi", "named as the sibling foreign payload names it"
        assert payload["account"] == "rcc-staff"
        assert payload["qos"] == "build"

    def test_csv_carries_them(self) -> None:
        snap = TelemetryCollector(self._ctx())._collect_snapshot_sync()
        header = TelemetrySnapshot.csv_header(0)
        row = dict(zip(header, snap.to_csv_row(0), strict=True))
        assert [row[c] for c in ("partition", "owner", "account", "qos")] == [
            "build",
            "youzhi",
            "rcc-staff",
            "build",
        ]

    def test_nothing_reported_is_empty_not_missing(self) -> None:
        """A job with no account/QOS still emits the columns, so the schema is stable."""
        ctx = _min_ctx(mem_limit_bytes=8 * 1024**3)
        snap = TelemetryCollector(ctx)._collect_snapshot_sync()
        row = dict(zip(TelemetrySnapshot.csv_header(0), snap.to_csv_row(0), strict=True))
        assert row["account"] == "" and row["qos"] == ""

    def test_from_dict_round_trips_them(self) -> None:
        snap = TelemetryCollector(self._ctx())._collect_snapshot_sync()
        back = TelemetrySnapshot.from_dict(json.loads(snap.to_json()))
        assert (back.partition, back.owner, back.account, back.qos) == (
            "build",
            "youzhi",
            "rcc-staff",
            "build",
        )

    def test_an_older_payload_without_them_is_accepted(self) -> None:
        payload = json.loads(_make_test_snapshot().to_json())
        for key in ("partition", "owner", "account", "qos"):
            payload.pop(key)
        back = TelemetrySnapshot.from_dict(payload)
        assert (back.partition, back.owner, back.account, back.qos) == ("", "", "", "")


class TestTheReportedCpuTotalNeverGoesBackwards:
    """Found by checking a real 100-sample `--log` file against the invariants the code
    documents, rather than by reading the code: `cpu_usage_ns` dropped 118.7
    CPU-seconds in one row when three burners exited.

    The cause is SW-24's own fix, and it is not a bug in the RATE: the /proc accumulator
    deliberately hands a process's ticks back when an in-job ancestor will re-report
    them, which is what makes the rate exact (measured +/-0.0% against ground truth).
    But `usage_ns` is published as a cumulative counter, and the CSV comment says in so
    many words that a consumer differencing it must never see a fake reset."""

    def _collector(self) -> Any:
        return TelemetryCollector(_min_ctx(mem_limit_bytes=8 * 1024**3))

    def test_a_regressing_accumulator_reports_its_previous_high(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collector = self._collector()
        series = [100, 200, 300, 181, 182, 400]  # the 300 -> 181 step is the hand-back
        reads = iter(series)
        monkeypatch.setattr(collector, "_read_cpu_ns", lambda pids: next(reads))
        monkeypatch.setattr(collector, "_get_job_pids", lambda: {1})
        reported = [collector._collect_cpu({1}).usage_ns for _ in series]
        assert reported == [100, 200, 300, 300, 300, 400], reported
        assert all(b >= a for a, b in zip(reported, reported[1:], strict=False))

    def test_the_rate_still_uses_the_raw_accumulator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The clamp must not change what the rate measures — that is the number SW-24
        got exact, and re-deriving it from a monotone total would double-count the
        handed-back ticks on the next frame."""
        collector = self._collector()
        monkeypatch.setattr(collector, "_get_job_pids", lambda: {1})
        # A real interval between frames: below the min_dt floor the rate branch
        # deliberately leaves the baseline alone (so the delta lands in the next
        # adequately-spaced frame), which would hide what this test is checking.
        clock = iter([100.0, 101.0, 102.0])
        monkeypatch.setattr("time.monotonic", lambda: next(clock))
        monkeypatch.setattr(collector, "_read_cpu_ns", lambda pids: 1_000_000_000)
        collector._collect_cpu({1})  # seeds the baseline
        monkeypatch.setattr(collector, "_read_cpu_ns", lambda pids: 500_000_000)
        metrics = collector._collect_cpu({1})  # the hand-back frame
        assert collector._prev_cpu_ns == 500_000_000, "the raw baseline follows the drop"
        assert collector._reported_cpu_ns == 1_000_000_000, "the reported total does not"
        assert metrics.usage_ns == 1_000_000_000
        assert metrics.effective_cores == 0.0, "and the rate clamps rather than going negative"

    def test_a_failed_read_reports_the_last_known_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-existing contract: a frame whose read failed must not publish 0."""
        collector = self._collector()
        monkeypatch.setattr(collector, "_get_job_pids", lambda: {1})
        monkeypatch.setattr(collector, "_read_cpu_ns", lambda pids: 700)
        assert collector._collect_cpu({1}).usage_ns == 700
        monkeypatch.setattr(collector, "_read_cpu_ns", lambda pids: None)
        assert collector._collect_cpu({1}).usage_ns == 700

    def test_a_fresh_collector_starts_at_zero(self) -> None:
        assert self._collector()._reported_cpu_ns == 0


class TestTeardownIsBounded:
    """`stop()` joined its own cancelled tasks with no bound (SW-76).

    A join is only as prompt as the task's willingness to unwind. `cancel()` raises
    at a suspension point, and anything the task does on the way out — a `finally`
    that awaits, a caught CancelledError, a submission queued behind a saturated
    default executor — happens on the canceller's clock. `stop()` runs on the path a
    Ctrl-C takes, so an unbounded join there is the "slow is indistinguishable from
    hung" failure the dashboard's own loops were fixed for.

    Motivating observation (not a claimed cause): a CI job sat for the full 120s
    pytest-timeout in test_a_refused_writer_does_not_truncate_the_first_ones_file,
    the event loop idle in `selector.poll` with nothing left to schedule — the shape
    of a join that will not finish rather than one that is merely slow. The two
    joins here and the NVML shutdown below are now bounded, and the test that hung
    bounds its own join so a recurrence reports in seconds instead of at the job
    timeout.
    """

    @staticmethod
    def _stubborn_task(release: asyncio.Event) -> asyncio.Task[None]:
        """A task that does not die when cancelled.

        Which is what an uninterruptible executor thread looks like from the
        awaiter's side: cancel() returns, and the await goes on regardless.
        """

        async def _body() -> None:
            while not release.is_set():
                try:
                    await asyncio.wait_for(release.wait(), timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    continue

        return asyncio.create_task(_body())

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    @pytest.mark.parametrize("attr", ["_task", "_liveness_task"])
    async def test_stop_gives_up_on_a_task_that_will_not_die(
        self, mock_job_ctx: JobContext, monkeypatch: pytest.MonkeyPatch, attr: str
    ) -> None:
        monkeypatch.setattr(_collector_mod, "_TEARDOWN_JOIN_SECONDS", 0.05)
        collector = TelemetryCollector(mock_job_ctx, SlurmwatchConfig(poll_interval=0.01))
        release = asyncio.Event()
        task = self._stubborn_task(release)
        setattr(collector, attr, task)
        loop = asyncio.get_running_loop()
        try:
            # It has to be RUNNING and suspended before the cancel: a task cancelled
            # before its first step never enters the body, so it dies at once and the
            # join is trivially bounded whether or not the bound exists. That mistake
            # made an earlier version of this test pass against the bug it describes.
            await asyncio.sleep(0.05)
            assert not task.done(), "the stubborn task never got going"
            began = loop.time()
            stopping = asyncio.create_task(collector.stop())
            await asyncio.wait({stopping}, timeout=5.0)
            assert stopping.done(), "stop() never returned: the join is unbounded"
            assert loop.time() - began < 5.0, loop.time() - began
            await stopping
        finally:
            release.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=2.0)

    def test_the_bound_is_a_real_bound(self) -> None:
        assert 0 < _collector_mod._TEARDOWN_JOIN_SECONDS <= 5.0


class TestFabricRateWindowUsesAMonotonicClock:
    """SW-80: the inter-node rate was measured against the WALL clock.

    `_collect_snapshot_sync` computes `now = time.time()` for elapsed — correct, since
    elapsed is compared against Slurm's start time — and handed that same value to
    `_collect_fabric` as its rate window. The CPU rate and the NVLink rate both read
    `time.monotonic()` instead, each with a comment saying why: a wall-clock step (NTP
    correction, leap second, VM migration) divides a real byte delta by a fraction of a
    second, and neither rate has a high-side clamp, so the result is an arbitrarily
    large throughput. The fabric path was the one that still used the clock that jumps
    — the same defect the other two document, in the third place it occurs.

    A backward step of 0.9s on a 1s poll turns a 100 Gb/s link into a reported
    ~1000 Gb/s, which is not a rate the hardware can produce.
    """

    @staticmethod
    def _ctx() -> JobContext:
        return JobContext(
            job_id="1",
            username="u",
            partition="p",
            nodelist="n",
            hostname="n",
            cpus_allocated=1,
            mem_limit_bytes=1,
            gpu_count_requested=0,
            gpu_indices=[],
            gpu_uuids=[],
        )

    def test_the_snapshot_measures_the_window_with_monotonic_not_wall_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End to end: the wall clock steps BACKWARDS between the two frames while
        monotonic advances by a second. The rate must reflect the second, not the
        step."""
        from slurmwatch import collector as collector_mod

        hca = TestNodeFabric._hca(tmp_path, rx=0, tx=0)
        monkeypatch.setattr(collector_mod, "_IB_SYSFS", hca)
        mono = iter([1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
        # A clock that jumps back 0.9s: dt would be 0.1s instead of 1.0s, inflating
        # the reported rate tenfold.
        wall = iter([5000.0, 4999.1, 4998.2, 4997.3, 4996.4])
        monkeypatch.setattr("slurmwatch.collector.time.monotonic", lambda: next(mono))
        monkeypatch.setattr("slurmwatch.collector.time.time", lambda: next(wall))

        c = TelemetryCollector(self._ctx(), SlurmwatchConfig(poll_interval=1.0))
        c._collect_fabric(next(mono))  # seed, through the same clock the caller uses
        TestNodeFabric._hca(tmp_path, rx=250_000_000, tx=0)
        # 250e6 four-octet units = 1e9 bytes = 8 Gbit over the 1s monotonic window.
        fab = c._collect_fabric(next(mono))
        assert fab is not None and fab.rates_known is True
        assert fab.rx_gbps == pytest.approx(8.0, abs=0.01), fab.rx_gbps
        # and nowhere near the ~80 Gb/s a 0.1s window would have produced
        assert fab.rx_gbps < fab.link_rate_gbps

    def test_the_call_site_passes_a_monotonic_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pins the wiring, not the arithmetic: a revert to `now` would hand this a
        wall-clock timestamp again, and every rate assertion above would still pass
        because the tests call _collect_fabric directly."""
        seen: list[float] = []
        c = TelemetryCollector(self._ctx(), SlurmwatchConfig())
        monkeypatch.setattr(c, "_remote", False)

        def _record(t: float) -> None:
            seen.append(t)

        monkeypatch.setattr(c, "_collect_fabric", _record)
        monkeypatch.setattr("slurmwatch.collector.time.time", lambda: 1_700_000_000.0)
        monkeypatch.setattr("slurmwatch.collector.time.monotonic", lambda: 42.5)
        with contextlib.suppress(Exception):
            c._collect_snapshot_sync()
        assert seen, "the fabric collector was never called"
        assert seen[0] == 42.5, f"got a wall-clock timestamp: {seen[0]}"

    def test_a_window_shorter_than_the_floor_keeps_its_baseline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A sliver of a window is not measured — and the bytes are not thrown away
        either: the next properly spaced frame reports the WHOLE delta. Overwriting
        the baseline on a skipped frame would silently halve it."""
        from slurmwatch import collector as collector_mod

        monkeypatch.setattr(collector_mod, "_IB_SYSFS", TestNodeFabric._hca(tmp_path, rx=0, tx=0))
        c = TelemetryCollector(self._ctx(), SlurmwatchConfig(poll_interval=1.0))
        c._collect_fabric(100.0)
        TestNodeFabric._hca(tmp_path, rx=125_000_000, tx=0)  # 5e8 bytes = 4 Gbit
        too_soon = c._collect_fabric(100.01)  # 10ms: below min(0.1, 0.5*1.0)
        assert too_soon is not None and too_soon.rates_known is False
        TestNodeFabric._hca(tmp_path, rx=250_000_000, tx=0)  # 1e9 bytes total = 8 Gbit
        good = c._collect_fabric(101.0)  # 1s since the RETAINED baseline
        assert good is not None and good.rates_known is True
        assert good.rx_gbps == pytest.approx(8.0, abs=0.01), good.rx_gbps


class TestTheDemoPayloadDoesNotInventIdentityFields:
    """SW-81: `--demo` emitted a `step_id` no real run ever produces.

    Found by diffing a demo payload against a live one field by field, which is the
    check for a class of defect that reading cannot reach: not a wrong value, a value
    where production has none. The mock context set `step_id=step_id or "0"`, so
    `--demo --once --json` reported `"step_id": "0"` while every real invocation —
    including the step form `12345.0`, which the CLI deliberately strips because
    slurmwatch is a job-level monitor — reports `null`.

    `--demo` exists so someone can explore the dashboard and the payload without a
    job, and the module's own usage text advertises it. A consumer who builds on a
    field that is populated only there gets `null` in production. It is the inverse of
    SW-30 (demo telemetry the payload didn't mark as simulated), and the same rule
    settles both: the demo payload may differ in VALUES, never in which fields it
    claims to know.
    """

    def test_the_mock_context_leaves_step_id_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        from slurmwatch.slurm import resolve_job_context

        assert resolve_job_context("12345").step_id is None

    def test_an_explicit_step_is_still_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The parameter is functional, not decorative — it scopes cgroup discovery to
        `step_<id>` — so a caller that passes one must still get it back."""
        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        from slurmwatch.slurm import resolve_job_context

        assert resolve_job_context("12345", step_id="7").step_id == "7"

    def test_no_identity_field_is_populated_only_in_demo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The general rule, so the next fabricated field fails here rather than in a
        consumer's parser. Compares the demo context against one built from a scontrol
        record with the same identity fields absent."""
        from slurmwatch.slurm import resolve_job_context

        monkeypatch.setenv("SLURMWATCH_MOCK", "1")
        demo = resolve_job_context("12345")
        # The identity axes a log is grouped by. A demo value here would be a fact
        # invented by the simulator.
        for field in ("step_id", "array_job_id", "array_task_id"):
            value = getattr(demo, field, None)
            assert value in (None, "", "12345"), f"{field} fabricated as {value!r}"


class TestTheCpuFigureSaysWhichCounterProducedIt:
    """SW-85: `mem_source` was published and `cpu_source` was not.

    `_cumulative_cpu_ns` picks between three counters — the cgroup's own (`v2`/`v1`,
    which also captures children that already exited) and a sum over the job's live
    PIDs (`proc`, the only option where a site constrains with cpuset but creates no
    per-job cpuacct) — and its docstring has always said their values "are not
    comparable to each other". `MemoryMetrics` has published its `source` since SW-3 for
    exactly that reason. The CPU counter kept its provenance internal.

    Why that matters, measured on a live 4-day reservation rather than argued:

        slurmwatch cpu.usage_ns   127,053 CPU-s   (source: proc)
        sacct TotalCPU                612 CPU-s
        sstat AveCPU              00:00.000

    jobacct_gather polls the step's TASK TREE; the `proc` sum counts every PID in the
    job's cgroup, and on a reservation whose processes joined the cgroup outside that
    tree Slurm sees almost none of them. Being the larger number is the point of a
    node-local monitor — but a consumer who cannot see which counter answered cannot
    tell that apart from a bug, and the field's own comment used to point them at "SU
    accounting" for corroboration, which is doubly wrong: SUs bill ALLOCATED core-time.
    """

    def test_the_column_and_the_json_field_both_exist(self) -> None:
        assert "cpu_source" in TelemetrySnapshot.csv_header(0)
        assert "source" in CpuMetrics(cores_allocated=1, usage_ns=0, usage_percent=0.0).to_dict()

    def test_it_sits_beside_the_counter_it_describes(self) -> None:
        """Provenance next to the figure, like mem_source: a reader scanning the CPU
        block should not have to hunt for it."""
        header = TelemetrySnapshot.csv_header(0)
        assert header.index("cpu_source") == header.index("cpu_usage_ns") + 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_slurm_env")
    async def test_the_mock_says_mock(self, mock_job_ctx: JobContext) -> None:
        collector = TelemetryCollector(mock_job_ctx, SlurmwatchConfig(poll_interval=0.01))
        await collector.start()
        try:
            snap = await collector.next_snapshot()
        finally:
            await collector.stop()
        assert snap.cpu.source == "mock"

    def test_each_on_node_counter_labels_itself(
        self, mock_job_ctx: JobContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """v2 and v1 come from different files; the label must follow whichever was
        readable, not the first one tried."""
        collector = TelemetryCollector(mock_job_ctx, SlurmwatchConfig())
        monkeypatch.setattr(collector, "_mock", False)

        v2 = tmp_path / "v2"
        v2.mkdir()
        (v2 / "cpu.stat").write_text("usage_usec 5000000\n")
        collector.job_ctx = replace(mock_job_ctx, cgroup_v2_path=str(v2))
        assert collector._read_cpu_ns() == 5_000_000_000
        assert collector._cpu_source == "v2"

        v1 = tmp_path / "v1"
        v1.mkdir()
        (v1 / "cpuacct.usage").write_text("7000000000\n")
        collector.job_ctx = replace(mock_job_ctx, cgroup_v2_path="", cgroup_v1_cpu_path=str(v1))
        assert collector._read_cpu_ns() == 7_000_000_000
        assert collector._cpu_source == "v1"

    def test_the_published_field_carries_the_on_node_label(
        self, mock_job_ctx: JobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gap a mutant found: asserting the internal `_cpu_source` proves the
        counter labelled itself, not that the label reaches the payload. Blanking the
        published field passed every other test in this class."""
        collector = TelemetryCollector(mock_job_ctx, SlurmwatchConfig())
        monkeypatch.setattr(collector, "_mock", False)

        def _reads_from_proc(job_pids: object = None) -> int:
            collector._cpu_source = "proc"
            return 3_000_000_000

        monkeypatch.setattr(collector, "_read_cpu_ns", _reads_from_proc)
        assert collector._collect_cpu().source == "proc"

    def test_every_metric_block_with_several_sources_publishes_one(self) -> None:
        """The rule this finding came from: if a block can be measured more than one
        way, the payload says which way. Both blocks that qualify now do."""
        header = TelemetrySnapshot.csv_header(0)
        assert {"cpu_source", "mem_source"} <= set(header)


class TestTheDemoNodeSwitcherShowsSomething:
    """SW-87: every --demo node reported identical telemetry, so switching looked inert.

    The per-node mock frames differed only in `hostname` and `node_index`: all four read
    50.0% CPU and 25.0% memory. The node switcher is an entire feature — a banner, a
    watchdog, digit-and-Enter selection, arrow keys — and `--demo` is the only place
    anyone can try it without a multi-node allocation. Pressing the key changed the name
    and nothing else, which is indistinguishable from a switch that silently failed: the
    confusion the switch banner was built to prevent.

    Values may vary; the field set may not. That is the rule SW-81 settled when the demo
    was found FABRICATING a field (`step_id`), and it cuts both ways — a demo that
    under-represents variation misleads as surely as one that invents data.
    """

    @staticmethod
    def _collector() -> tuple[TelemetryCollector, list[str]]:
        from slurmwatch.slurm import resolve_job_context

        ctx = resolve_job_context("12345")
        c = TelemetryCollector(ctx, SlurmwatchConfig())
        c._loop = asyncio.new_event_loop()
        return c, list(ctx.nodelist_resolved)

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_each_node_reports_its_own_numbers(self) -> None:
        c, nodes = self._collector()
        seen = [c.mock_snapshot_for_node(n) for n in nodes]
        assert len({s.hostname for s in seen}) == len(nodes), "stamps must differ"
        cpus = {s.cpu.usage_percent for s in seen}
        assert len(cpus) == len(nodes), f"a switch must change the numbers: {cpus}"
        mems = {s.memory.usage_percent for s in seen}
        assert len(mems) == len(nodes), mems

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_the_primary_node_is_untouched(self) -> None:
        """Node 0 keeps exactly the values it had, so every existing expectation about
        the demo's own node still holds — the variation applies only where a switch
        goes."""
        c, nodes = self._collector()
        first = c.mock_snapshot_for_node(nodes[0])
        assert first.cpu.usage_percent == 50.0
        assert first.memory.usage_percent == 25.0

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_the_variation_is_deterministic(self) -> None:
        """Same node, same numbers: a demo that jittered per call would make the
        switcher look broken in the other direction."""
        c, nodes = self._collector()
        a = c.mock_snapshot_for_node(nodes[2])
        b = c.mock_snapshot_for_node(nodes[2])
        assert (a.cpu.usage_percent, a.memory.usage_percent) == (
            b.cpu.usage_percent,
            b.memory.usage_percent,
        )

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_no_field_appears_or_disappears_between_nodes(self) -> None:
        """The SW-81 rule: values may differ between demo nodes, the schema may not."""
        c, nodes = self._collector()
        shapes = {tuple(sorted(json.loads(c.mock_snapshot_for_node(n).to_json()))) for n in nodes}
        assert len(shapes) == 1, "the payload's field set must not depend on the node"

    @pytest.mark.usefixtures("mock_slurm_env")
    def test_the_numbers_stay_plausible(self) -> None:
        """Varied, not corrupted: percentages stay in range and memory stays under its
        limit, or the demo would teach a reader to expect impossible readings."""
        c, nodes = self._collector()
        for n in nodes:
            s = c.mock_snapshot_for_node(n)
            assert 0.0 <= s.cpu.usage_percent <= 100.0
            assert 0.0 <= s.memory.usage_percent <= 100.0
            assert s.memory.current_bytes <= s.memory.limit_bytes
            for g in s.gpus:
                assert 0.0 <= g.utilization_percent <= 100.0
                assert g.memory_used_bytes <= g.memory_total_bytes


class TestMemoryWhenNoMemoryCgroupIsDelegated:
    """The missing ``else`` on ``_collect_memory``'s cgroup chain.

    ``_collect_cgroup_paths`` reports failure only when v2, v1_mem and v1_cpu are
    ALL absent, so a v1 node that delegated cpuacct but not memory is a discovery
    SUCCESS — and a success is exactly what stops the caller degrading to sstat.
    The chain was ``if cgroup_v2_path: ... elif cgroup_v1_mem_path: ...`` with no
    ``else``, so nothing ran and the zeros initialised at the top of the method
    were published as a measured ``cgroup`` reading.
    """

    def _cpuacct_only(self, tmp_path: Path, pid: int | None = None) -> Path:
        cpuacct = tmp_path / "cpuacct"
        cpuacct.mkdir()
        (cpuacct / "cpuacct.usage").write_text("0")
        if pid is not None:
            # `_get_job_pids` reads `cgroup.procs` on every path it is given,
            # including the v1 cpuacct one — that is what lets the /proc fallback
            # answer at all here.
            (cpuacct / "cgroup.procs").write_text(f"{pid}\n")
        return cpuacct

    def test_the_proc_sum_answers_instead_of_a_confident_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: 400 * 1024**3)
        monkeypatch.setattr(TelemetryCollector, "_proc_rss_bytes", lambda self: 5 * 1024**3)
        mem = TelemetryCollector(
            _min_ctx(
                mem_limit_bytes=8 * 1024**3,
                cgroup_v1_cpu_path=str(self._cpuacct_only(tmp_path)),
            )
        )._collect_memory()
        assert mem.current_bytes == 5 * 1024**3  # was 0
        # And it says which counter answered: statm counts shared pages and sees
        # only pids alive this instant, so calling it "cgroup" misdescribes it.
        assert mem.source == "proc"  # was "cgroup", a claim nothing measured
        assert mem.cache_measured is False
        assert mem.peak_is_lifetime is False

    def test_the_oom_guard_measures_against_node_ram_not_the_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """7.9 GiB of an 8 GiB request, with nothing capping the job.

        No cgroup means no enforced cap, so the kernel kills at node RAM and this
        must not alarm.

        The guard basis is a SECOND site of the fix, and it only becomes
        load-bearing once the first one works: with no ``else`` arm at all the
        usage is 0, so nothing can trip and this assertion is vacuous (measured —
        it passes with the whole arm deleted). Its teeth are the branch-local
        neuter: keep ``current_bytes``/``mem_source`` and drop only
        ``limit_bytes = _read_meminfo_total()``, and the now-real 7.9 GiB is
        measured against the job's own 8 GiB request — 98.75%, tripping both
        guards. That is the false "near limit, raise --mem" critical P3 removed
        and that both branches above avoid by name.
        """
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: 400 * 1024**3)
        monkeypatch.setattr(TelemetryCollector, "_proc_rss_bytes", lambda self: 7900 * 1024**2)
        mem = TelemetryCollector(
            _min_ctx(
                mem_limit_bytes=8 * 1024**3,
                cgroup_v1_cpu_path=str(self._cpuacct_only(tmp_path)),
            )
        )._collect_memory()
        assert mem.oom_guard_warning is False
        assert mem.oom_guard_critical is False
        # Display still reads against the request, as the v1 branch does.
        assert mem.limit_bytes == 8 * 1024**3

    def test_the_pids_really_come_from_the_cpuacct_cgroup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No ``_proc_rss_bytes`` stub — the read must work end to end.

        Uses this process's PARENT: ``_get_job_pids`` discards ``os.getpid()`` so
        the monitor never counts itself, and the parent is alive for the duration
        with a readable ``/proc/<pid>/statm``.
        """
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: 400 * 1024**3)
        cpuacct = self._cpuacct_only(tmp_path, pid=os.getppid())
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v1_cpu_path=str(cpuacct))
        )._collect_memory()
        assert mem.current_bytes > 0
        assert mem.source == "proc"


class TestControlsOnTheDelegationFallback:
    """Controls: these pass whether or not the ``else`` arm is present.

    A new final arm on an if/elif chain is exactly the kind of change that can
    shadow the branches above it, so each of those still has to answer for itself.
    """

    def test_a_v1_memory_cgroup_still_reads_its_own_counter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: 400 * 1024**3)
        v1 = tmp_path / "memory"
        v1.mkdir()
        (v1 / "memory.usage_in_bytes").write_text(str(3 * 1024**3))
        (v1 / "memory.stat").write_text("total_inactive_file 0\ntotal_active_file 0\n")
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v1_mem_path=str(v1))
        )._collect_memory()
        assert mem.current_bytes == 3 * 1024**3
        assert mem.source == "cgroup"

    def test_a_v2_cgroup_still_reads_its_own_counter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: 400 * 1024**3)
        v2 = tmp_path / "v2"
        v2.mkdir()
        (v2 / "memory.current").write_text(str(2 * 1024**3))
        (v2 / "memory.stat").write_text("inactive_file 0\nactive_file 0\n")
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v2_path=str(v2))
        )._collect_memory()
        assert mem.current_bytes == 2 * 1024**3
        assert mem.source == "cgroup"

    def test_a_delegated_cgroup_with_no_controller_still_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The pre-existing F4 fallback INSIDE the v1 branch, not the new arm.

        Keeps ``source == "proc"`` from being reachable only through the new
        ``else``: a v1 memory cgroup with no ``memory.usage_in_bytes`` has always
        answered from /proc.
        """
        monkeypatch.setattr(_collector_mod, "_read_meminfo_total", lambda: 400 * 1024**3)
        monkeypatch.setattr(TelemetryCollector, "_proc_rss_bytes", lambda self: 1024**3)
        v1 = tmp_path / "memory"
        v1.mkdir()  # exists, but no memory.usage_in_bytes
        mem = TelemetryCollector(
            _min_ctx(mem_limit_bytes=8 * 1024**3, cgroup_v1_mem_path=str(v1))
        )._collect_memory()
        assert mem.current_bytes == 1024**3
        assert mem.source == "proc"
