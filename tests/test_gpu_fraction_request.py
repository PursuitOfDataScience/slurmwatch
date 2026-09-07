"""D18: a job that asked for a FRACTION of a GPU is never told it asked for none.

`gres/shard` (a device split into N shards) and `gres/mps` (a percentage of one) were
matched by neither `_parse_tres_gpus` nor `_parse_gpu_count`, so off the node — where
there is no NVML and no CUDA_VISIBLE_DEVICES to fall back on — `gpu_count_requested`
came out 0 for a `--gres=shard:2` job. Measured before the fix, from one login-node
`scontrol` record: the dashboard row read `● GPU     none requested`, the `g` drill-in
read `no GPUs requested by this job`, and the plain report omitted GPU entirely. The
first two are the defect: a figure the tool could not measure was published as a
measured zero, and stating "no GPUs requested" about a job holding two shards is worse
than saying nothing at all.

The fix withholds the negative rather than inventing a count (`slurm.py`'s
`_parse_gpu_fraction_request`), because a shard count is NOT a device count: two
shards can be two slices of one physical GPU and `mps:100` is 100% of one, so folding
either into `gpu_count_requested` would trade a false zero for a false device count.
`gpu_count_requested` is therefore asserted to stay 0 here — that is the fix's shape,
not a leftover.

Both surfaces are covered separately, because "fixed only on one side" is this repo's
recurring defect: the dashboard (`tui.py`, two sites — the resource row and the
drill-in headline) and the plain report (`cli.py`), which a reader falls back to when
they cannot have the dashboard at all.
"""

from __future__ import annotations

import asyncio
import io
import socket
from contextlib import redirect_stdout
from dataclasses import replace

import pytest

import slurmwatch.cli as cli
from slurmwatch import slurm
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.exceptions import CgroupNotFoundError
from slurmwatch.model import JobContext, TelemetrySnapshot
from slurmwatch.slurm import resolve_job_context
from tests.test_tui import _dashboard_surfaces, _sstat_ctx, _sstat_snapshot


def _scontrol(*, tres: str, per_node: str, detail: str) -> str:
    """One `scontrol show job -d` record, varying only in how the GRES is spelled.

    The shapes are Slurm's own: `gres/<name>[:type]=N` in TRES/AllocTRES, the
    `gres:<name>:N` of `TresPerNode`, and the per-node `GRES=...` detail line that
    `-d` adds. Everything else is held constant so a difference between cases can
    only come from the GRES.
    """
    return (
        "JobId=901 JobState=RUNNING Partition=gpu Account=rcc QOS=normal\n"
        "JobName=infer\n"
        "NodeList=cn-001 NumCPUs=4 NumNodes=1\n"
        f"TRES=cpu=4,mem=16G,node=1,billing=4{tres}\n"
        f"AllocTRES=cpu=4,mem=16G,node=1,billing=4{tres}\n"
        "RunTime=00:10:00 TimeLimit=01:00:00\n"
        "SubmitTime=2024-01-15T10:29:00 MinMemoryNode=16G "
        "StartTime=2024-01-15T10:30:00 UserId=user(1001)\n"
        f"{per_node}"
        f"   Nodes=cn-001 CPU_IDs=0-3 Mem=16384{detail}\n"
    )


_SHARD = _scontrol(
    tres=",gres/shard=2", per_node="TresPerNode=gres:shard:2\n", detail=" GRES=shard:2(IDX:0)"
)
_MPS = _scontrol(
    tres=",gres/mps=100", per_node="TresPerNode=gres:mps:100\n", detail=" GRES=mps:100(IDX:0)"
)
# The two controls' records. Neither is any finding's input: one asks for whole
# devices, the other asks for no GPU at all.
_WHOLE_GPU = _scontrol(
    tres=",gres/gpu=2", per_node="TresPerNode=gres:gpu:2\n", detail=" GRES=gpu:2(IDX:0-1)"
)
_CPU_ONLY = _scontrol(tres="", per_node="", detail="")


def _resolve_off_node(monkeypatch: pytest.MonkeyPatch, record: str) -> JobContext:
    """The context a login node builds: the record parses, the cgroups do not exist."""

    def _no_cgroup(*_a: object, **_k: object) -> dict[str, object]:
        raise CgroupNotFoundError("no cgroup on a login node")

    monkeypatch.setattr(slurm, "_run_slurm_cmd", lambda *a, **k: record)
    monkeypatch.setattr(slurm, "_resolve_uid", lambda u: 1001)
    monkeypatch.setattr(socket, "gethostname", lambda: "login1")
    monkeypatch.setattr(slurm, "_discover_cgroup_paths", _no_cgroup)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return resolve_job_context("901")


def _snapshot(gpu_count: int = 0, remote: bool = True) -> TelemetrySnapshot:
    """The sstat frame that goes with these contexts: no devices readable off-node."""
    snap = _sstat_snapshot(rss=8 * 1024**3, limit=16 * 1024**3, cpu_seconds=3200.0)
    return replace(snap, gpu_count_requested=gpu_count, remote=remote)


def _gpu_row(ctx: JobContext, cfg: SlurmwatchConfig | None = None, gpu_count: int = 0) -> str:
    """The dashboard's GPU resource row (markup stripped)."""
    got = asyncio.run(_dashboard_surfaces(ctx, _snapshot(gpu_count), cfg or SlurmwatchConfig()))
    hits = [ln for ln in got["rows"].splitlines() if "GPU" in ln]
    assert len(hits) == 1, (hits, got["rows"])
    return hits[0]


def _gpu_headline(ctx: JobContext, cfg: SlurmwatchConfig | None = None, remote: bool = True) -> str:
    """The `g` drill-in's headline for the same context."""
    snap = _snapshot(remote=remote)
    got = asyncio.run(_dashboard_surfaces(ctx, snap, cfg or SlurmwatchConfig(), drill="g"))
    return got["headline"]


def _report(
    monkeypatch: pytest.MonkeyPatch, ctx: JobContext, cfg: SlurmwatchConfig | None = None
) -> str:
    """The plain-text summary the off-node (sstat) path prints for the same context."""
    # Pinned rather than left to the machine: it shells out to `scontrol show config`
    # and caches the answer process-wide.
    monkeypatch.setattr(cli, "acct_gather_disabled", lambda: False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._print_remote_summary(ctx, _snapshot(), cfg or SlurmwatchConfig())
    return buf.getvalue()


class TestTheRequestIsReadFromTheRecord:
    """slurm.py: the fact the renderers were missing is in the record all along."""

    def test_a_shard_request_is_read_off_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx = _resolve_off_node(monkeypatch, _SHARD)
        assert ctx.remote is True
        assert ctx.gpu_fraction_request == "shard:2"
        # Not folded into the count, on purpose: two shards can be two slices of ONE
        # device, so 2 would be a false device count and the collector would go
        # looking for two GPUs.
        assert ctx.gpu_count_requested == 0
        assert ctx.gpu_indices == []

    def test_an_mps_request_is_read_off_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx = _resolve_off_node(monkeypatch, _MPS)
        # `mps:100` is a PERCENTAGE of one device — the clearest case for keeping this
        # out of a device count.
        assert ctx.gpu_fraction_request == "mps:100"
        assert ctx.gpu_count_requested == 0

    def test_every_spelling_slurm_uses_is_matched(self) -> None:
        parse = slurm._parse_gpu_fraction_request
        # TRES only (no `-d` per-node detail), typed and untyped.
        assert parse("TRES=cpu=4,mem=16G,gres/shard=2\n") == "shard:2"
        assert parse("TRES=cpu=4,mem=16G,gres/shard:a100=2\n") == "shard:2"
        assert parse("AllocTRES=cpu=4,gres/mps:a100=50\n") == "mps:50"
        # The per-node fields, with and without the `gres:` prefix and a type.
        assert parse("TresPerNode=gres:shard:2\n") == "shard:2"
        assert parse("Gres=shard:a100:8\n") == "shard:8"
        # Per-node wins over the job-wide total, so this says the same thing
        # `gpu_count_requested` does: THIS node's request.
        assert parse("TRES=gres/shard=8\nTresPerNode=gres:shard:2\n") == "shard:2"
        # Whole devices are not fractions, and neither are the GPU-adjacent TRES that
        # merely share the prefix.
        assert parse("TRES=cpu=4,gres/gpu=2\nTresPerNode=gres:gpu:2\n") == ""
        assert parse("TRES=cpu=4,gres/gpumem=8G,gres/gpuutil=50\n") == ""
        assert parse("TRES=cpu=4,mem=16G,node=1\n") == ""


class TestTheDashboardStopsClaimingNoneWasRequested:
    """tui.py, both sites: the resource row and the drill-in headline."""

    def test_the_gpu_row_names_the_request_instead(self) -> None:
        row = _gpu_row(_sstat_ctx(gpu_fraction_request="shard:2"))
        assert "shard:2 requested" in row, row
        assert "a fraction of a device" in row, row
        assert "none requested" not in row, row

    def test_the_drill_in_headline_names_the_request_instead(self) -> None:
        head = _gpu_headline(_sstat_ctx(gpu_fraction_request="shard:2"))
        assert "shard:2 requested" in head, head
        assert "no device count to report" in head, head
        assert "no GPUs requested by this job" not in head, head

    def test_the_drill_in_still_points_off_node_readers_at_the_node(self) -> None:
        """Off-node the device count is unknowable here but the utilization is not:
        the on-node path resolves shards through CUDA_VISIBLE_DEVICES. Said only when
        the reader is not already there — the mistake the `remote` branch beside this
        one exists to avoid."""
        ctx = _sstat_ctx(gpu_fraction_request="shard:2")
        off_node = _gpu_headline(ctx)
        assert "Run on the compute node" in off_node, off_node
        on_node = _gpu_headline(ctx, remote=False)
        assert "shard:2 requested" in on_node, on_node
        assert "Run on the compute node" not in on_node, on_node

    def test_the_row_folds_its_dash_under_ascii_mode(self) -> None:
        """A terminal that asked for ASCII must not be handed an em dash — the leak
        class this repo has now fixed on four other lines."""
        ctx = _sstat_ctx(gpu_fraction_request="shard:2")
        row = _gpu_row(ctx, SlurmwatchConfig(ascii_mode=True))
        assert "shard:2 requested -" in row, row
        assert "—" not in row, row


class TestThePlainReportSaysTheSameThing:
    """cli.py: the surface a reader falls back to, which said nothing at all."""

    def test_the_report_names_the_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = _report(monkeypatch, _sstat_ctx(gpu_fraction_request="shard:2"))
        assert "  GPU      shard:2 requested" in out, out
        assert "not a whole GPU" in out, out
        # The one thing off-node prose can still usefully say about a GPU.
        assert "live GPU utilization" in out, out

    def test_the_report_folds_its_dash_under_ascii_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _report(
            monkeypatch,
            _sstat_ctx(gpu_fraction_request="mps:100"),
            SlurmwatchConfig(ascii_mode=True),
        )
        assert "  GPU      mps:100 requested -" in out, out
        gpu_line = [ln for ln in out.splitlines() if ln.startswith("  GPU")][0]
        assert "—" not in gpu_line, gpu_line


class TestControls:
    """Both pass with the fix in AND with it neutered, and neither shares an input
    with a finding above: one job asks for whole devices, the other for no GPU."""

    def test_control_a_whole_gpu_request_reads_the_same_as_before(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = _resolve_off_node(monkeypatch, _WHOLE_GPU)
        assert ctx.gpu_count_requested == 2
        assert ctx.gpu_fraction_request == ""
        surface_ctx = _sstat_ctx(gpu_count_requested=2)
        assert "2 requested" in _gpu_row(surface_ctx, gpu_count=2)
        assert "  GPU      2 allocated" in _report(monkeypatch, surface_ctx)

    def test_control_a_cpu_only_job_is_still_told_it_asked_for_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The claim is TRUE for this job, so it has to survive: the fix withholds the
        negative, it does not delete it."""
        ctx = _resolve_off_node(monkeypatch, _CPU_ONLY)
        assert ctx.gpu_count_requested == 0
        assert ctx.gpu_fraction_request == ""
        surface_ctx = _sstat_ctx()
        assert "none requested" in _gpu_row(surface_ctx)
        assert _gpu_headline(surface_ctx) == "no GPUs requested by this job"
        # No GPU row at all in the prose report (the label is "  GPU      ...");
        # "live GPU utilization" in the sstat caveat is not one.
        assert "  GPU " not in _report(monkeypatch, surface_ctx)
