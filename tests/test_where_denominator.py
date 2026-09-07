"""D17: the WHERE table's free-capacity figures carry the denominator they are a
fraction of.

`sinfo` already gives `total_nodes` and `cpus_total`, `resolve_cluster_partitions`
already sums them, and before this round nothing in `src/` read either one: the table
printed "free nodes 6   idle cores 240" for a 256-core partition and for a 3200-core
one and the two rows came out character-identical, so 94%-free and 7%-free looked the
same and the requeue tip (which ranks by ABSOLUTE free cores) had nothing on screen to
justify its pick.

Both renderers are covered, separately, because "fixed only on one side" is this repo's
recurring defect and this very table has been an instance of it.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
from rich.text import Text

import slurmwatch.cli as cli
from slurmwatch import pending
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.pending import PartitionResources, PendingJob, capacity_cell


def _part(name: str, **kw: object) -> PartitionResources:
    """A schedulable CPU partition big enough that only capacity varies between rows."""
    gib = 1024**3
    defaults: dict[str, object] = {
        "timelimit_seconds": 24 * 3600,
        "max_node_mem_bytes": 256 * gib,
        "max_node_cpus": 32,
        "max_idle_node_cpus": 32,
        "max_idle_node_mem_bytes": 256 * gib,
        "assoc_verified": True,
    }
    defaults.update(kw)
    return PartitionResources(name=name, available=True, **defaults)  # type: ignore[arg-type]


def _parts() -> list[PartitionResources]:
    """Two alternatives with the SAME free capacity and very different totals.

    6/8 nodes and 240/256 cores is a partition about to take the job; 6/100 and
    240/3200 is a busy one that merely has a couple of holes in it. The numerators
    match exactly so that anything distinguishing the rows has to be the denominator.
    """
    return [
        _part("cur-part", total_nodes=4, cpus_total=128, is_current=True),
        _part(
            "small-part",
            total_nodes=8,
            idle_nodes=4,
            mix_nodes=2,
            cpus_idle=240,
            cpus_total=256,
            idle_node_cpus=240,
        ),
        _part(
            "big-part",
            total_nodes=100,
            idle_nodes=4,
            mix_nodes=2,
            cpus_idle=240,
            cpus_total=3200,
            idle_node_cpus=240,
        ),
    ]


def _job() -> PendingJob:
    job = pending._mock_pending_job("777")
    job.reason = "Resources"
    job.partition = "cur-part"
    job.req_gpus = 0
    job.req_cpus = 4
    job.req_nodes = 1
    job.req_mem_bytes = 8 * 1024**3
    job.exclusive = False
    job.time_limit_seconds = 3600
    return job


def _report(
    monkeypatch: pytest.MonkeyPatch, job: PendingJob, parts: list[PartitionResources]
) -> str:
    """The plain-text `_print_pending_summary` surface."""
    monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: parts)
    monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
    monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)
    monkeypatch.setattr(cli, "resolve_user_associations", lambda *a, **k: None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._print_pending_summary(job, stream=buf)
    return buf.getvalue()


def _card(monkeypatch: pytest.MonkeyPatch, job: PendingJob, parts: list[PartitionResources]) -> str:
    """The dashboard surface (PendingView), markup stripped."""
    from slurmwatch.tui import PendingView

    monkeypatch.setattr("slurmwatch.tui.resolve_user_associations", lambda *a, **k: None)
    view = PendingView()
    view.job = job
    view.config = SlurmwatchConfig()
    view.resolved = True
    view.partitions = parts
    return Text.from_markup(view.render()).plain


def _row(text: str, name: str) -> str:
    """The table row for one partition (the tip mentions names too, so match the row)."""
    hits = [ln for ln in text.splitlines() if ln.lstrip().startswith(name)]
    assert len(hits) == 1, (name, hits, text)
    return hits[0]


def _cells(row: str) -> str:
    """A row minus its partition name, i.e. what the reader compares between rows."""
    return row.split(None, 1)[1]


class TestTheDenominatorIsOnScreen:
    def test_report_idle_cores_name_their_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = _report(monkeypatch, _job(), _parts())
        assert "240/256" in _row(out, "small-part"), out
        assert "240/3200" in _row(out, "big-part"), out

    def test_card_idle_cores_name_their_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        card = _card(monkeypatch, _job(), _parts())
        assert "240/256" in _row(card, "small-part"), card
        assert "240/3200" in _row(card, "big-part"), card

    def test_report_free_nodes_name_their_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = _report(monkeypatch, _job(), _parts())
        assert "6/8" in _row(out, "small-part"), out
        assert "6/100" in _row(out, "big-part"), out

    def test_card_free_nodes_name_their_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        card = _card(monkeypatch, _job(), _parts())
        assert "6/8" in _row(card, "small-part"), card
        assert "6/100" in _row(card, "big-part"), card

    def test_the_two_rows_no_longer_render_identically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The actual complaint: same numerators, so pre-fix the numeric half of a
        94%-free row and a 7%-free row were the same string on both surfaces."""
        for text in (_report(monkeypatch, _job(), _parts()), _card(monkeypatch, _job(), _parts())):
            small, big = _row(text, "small-part"), _row(text, "big-part")
            assert _cells(small) != _cells(big), (small, big)

    def test_the_current_partition_gets_one_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The row the reader starts from: "0 idle cores" says nothing about whether
        the queue it is stuck in is 4 nodes or 4000."""
        for text in (_report(monkeypatch, _job(), _parts()), _card(monkeypatch, _job(), _parts())):
            row = _row(text, "cur-part")
            assert "0/4" in row and "0/128" in row, row

    def test_the_helper_pairs_free_with_total(self) -> None:
        assert capacity_cell(240, 3200) == "240/3200"
        assert capacity_cell(6, 8) == "6/8"


class TestControls:
    """Each of these passes with the fix in AND with it neutered — they pin the
    behaviour the fix must not have changed, and none of them reads a denominator."""

    def test_control_the_free_capacity_figure_is_still_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The numerator is the number the table was always about; it stays."""
        for text in (_report(monkeypatch, _job(), _parts()), _card(monkeypatch, _job(), _parts())):
            for name in ("small-part", "big-part"):
                row = _row(text, name)
                assert "240" in row and "6" in row, row

    def test_control_both_columns_are_still_labelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for text in (_report(monkeypatch, _job(), _parts()), _card(monkeypatch, _job(), _parts())):
            assert "free nodes" in text, text
            assert "idle cores" in text, text

    def test_control_a_fitting_alternative_is_still_marked_and_tipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _report(monkeypatch, _job(), _parts())
        assert "FITS NOW" in _row(out, "small-part"), out
        assert "small-part has room" in out, out
        card = _card(monkeypatch, _job(), _parts())
        assert "YES" in _row(card, "small-part"), card
        assert "small-part has room" in card, card

    def test_control_an_unknown_total_prints_the_bare_figure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller-built PartitionResources (or an `sinfo` with no %D/%C) leaves the
        totals at 0, and "240/0" would claim more than the bare figure does."""
        parts = [
            _part("cur-part", is_current=True),
            _part("no-totals", idle_nodes=4, mix_nodes=2, cpus_idle=240, idle_node_cpus=240),
        ]
        for text in (
            _report(monkeypatch, _job(), parts),
            _card(monkeypatch, _job(), parts),
        ):
            row = _row(text, "no-totals")
            assert "240" in row, row
            assert "/" not in _cells(row).split("   ")[0], row

    def test_control_the_verdict_column_stays_aligned_across_magnitudes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Right-aligned numerics keep the marker in line whether a row reads 646 or
        10458 (the misalignment a user hit), with or without totals."""
        for totals in ({}, {"total_nodes": 400, "cpus_total": 12800}):
            parts = [
                _part("cur-part", is_current=True, **totals),
                _part("small", idle_nodes=6, cpus_idle=646, idle_node_cpus=646, **totals),
                _part("big", idle_nodes=300, cpus_idle=10458, idle_node_cpus=10458, **totals),
            ]
            out = _report(monkeypatch, _job(), parts)
            fits = [ln for ln in out.splitlines() if "FITS NOW" in ln]
            assert len(fits) == 2, out
            assert len({ln.index("FITS NOW") for ln in fits}) == 1, fits
            card = _card(monkeypatch, _job(), parts)
            yes = [ln for ln in card.splitlines() if "YES" in ln and "partition" not in ln]
            assert len(yes) == 2, card
            assert len({ln.index("YES") for ln in yes}) == 1, yes
