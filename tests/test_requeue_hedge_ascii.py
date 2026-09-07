"""The requeue hedge kept its em dash on a terminal that had asked for none.

`_print_pending_summary` opens by saying what it owes the reader:

    # Honour --ascii here too (a non-UTF-8 terminal / pipe): no stray Unicode.

It then folds its own `dash`/`dot`/`dots` and passes `ascii_mode` into
`explain_reason` -- and printed `partition_move_caveat` raw, which hardcoded an em
dash. The two lines land together, so `--ascii` produced a report that contradicted
itself one line apart:

    Tip    broadwl has room for this request now - requeue with: scontrol update ...
           add QOS=<name> if that partition needs its own — the QOS does not move ...

The fold now lives in the producer, where :func:`explain_reason` already keeps its
own. The usual "both renderers, since fixing one is half a fix" rule turned out NOT
to apply: the dashboard folds its entire render (`PendingView.render` ends in
``_asciify(out) if ascii_mode else out``), so it never leaked and its call site is
left alone. Verified by neuter rather than by reading -- dropping the flag at the
dashboard's site changes nothing. `TestControls` pins that asymmetry.

Why no existing test caught it: the one `ascii_mode=True` test of the plain report
stubs `resolve_cluster_partitions` to `[]`, so the Tip branch never runs, and the
suite's only `isascii()` assertion covers the dashboard's *loading* state.
"""

import io

import pytest
from rich.text import Text

from slurmwatch import cli, pending, tui
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.pending import PartitionResources

# broadwl fits and is permitted, but names no QOS of its own -> the hedge fires.
ASSOC_WITHOUT_A_QOS = {"build": ["build"], "broadwl": ["short", "long"]}
HEDGE_MARK = "QOS=<name>"


def _parts() -> list[PartitionResources]:
    return [
        PartitionResources(
            "build", True, idle_nodes=4, cpus_idle=128, max_node_cpus=48, is_current=True
        ),
        PartitionResources("broadwl", True, idle_nodes=53, cpus_idle=1101, max_node_cpus=48),
    ]


def _capacity_wait() -> pending.PendingJob:
    job = pending._mock_pending_job("777")
    job.reason, job.req_gpus, job.req_cpus = "Priority", 0, 4
    return job


@pytest.fixture
def _stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for mod in (cli, tui):
        monkeypatch.setattr(
            mod, "resolve_user_associations", lambda *a, **k: ASSOC_WITHOUT_A_QOS, raising=False
        )
    monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a, **k: _parts())
    monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a, **k: None)
    monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a, **k: None)


def _plain_report(ascii_mode: bool) -> str:
    buf = io.StringIO()
    cli._print_pending_summary(_capacity_wait(), stream=buf, ascii_mode=ascii_mode)
    return buf.getvalue()


def _dashboard(ascii_mode: bool) -> str:
    view = tui.PendingView()
    view.job = _capacity_wait()
    view.config = SlurmwatchConfig(ascii_mode=ascii_mode)
    view.resolved = True
    view.partitions = _parts()
    # The table is an input the screen's poll fills in (D23), as well as being what
    # `_stubs` patches the resolver to return; set both so this file keeps measuring
    # the fold and not the plumbing.
    view.assoc = ASSOC_WITHOUT_A_QOS
    return Text.from_markup(view.render()).plain


class TestTheHedgeFoldsOnBothSurfaces:
    def test_the_plain_report_carries_no_stray_unicode(self, _stubs: None) -> None:
        out = _plain_report(ascii_mode=True)
        assert HEDGE_MARK in out, out  # the branch really was reached
        assert out.isascii(), sorted({c for c in out if not c.isascii()})

    def test_the_tip_and_its_hedge_agree_within_the_report(self, _stubs: None) -> None:
        """The defect was local: a folded dash and an unfolded one, one line apart."""
        lines = [
            ln for ln in _plain_report(ascii_mode=True).splitlines() if " - " in ln or "—" in ln
        ]
        assert lines, _plain_report(ascii_mode=True)
        assert all("—" not in ln for ln in lines), lines

    def test_the_producer_folds_when_asked(self) -> None:
        asked = pending.partition_move_caveat("gpu", {"gpu": ["short", "long"]}, True)
        assert "—" not in asked and " - " in asked, asked


class TestControls:
    """None of these calls the fold, so they hold with the fix in or out.

    Each uses the two-argument form deliberately: before the fix there was no
    third parameter at all, so passing one would have been a TypeError rather
    than a measurement.
    """

    def test_the_dash_survives_when_nobody_asked_to_fold(self) -> None:
        assert "—" in pending.partition_move_caveat("gpu", {"gpu": ["short", "long"]})
        assert "—" in pending.partition_move_caveat("gpu", None)

    def test_the_wording_is_frozen_not_merely_dash_swapped(self) -> None:
        # Pinned as literals so a fold that mangled the sentence would show up
        # here rather than passing an "is it ASCII" check.
        assert pending.partition_move_caveat("gpu", {"gpu": ["short", "long"]}) == (
            "add QOS=<name> if that partition needs its own — the QOS does not move with it"
        )
        assert pending.partition_move_caveat("gpu", None) == (
            "check your QOS for it first — the QOS moves with the job, not the partition"
        )

    def test_the_dashboard_was_never_leaking_and_still_is_not(self, _stubs: None) -> None:
        """The asymmetry, pinned: this holds with the fix in OR out.

        `PendingView.render` ends in ``_asciify(out) if ascii_mode else out``, so
        the dashboard folds everything it has built and its call site deliberately
        does not pass the flag. Measured by neuter: removing the flag there changed
        nothing, which is why only the plain report's site carries it.
        """
        out = _dashboard(ascii_mode=True)
        assert HEDGE_MARK in out, out
        assert out.isascii(), sorted({c for c in out if not c.isascii()})

    def test_a_knowable_qos_still_yields_no_hedge_at_all(self) -> None:
        assert (
            pending.partition_move_caveat("astroplasmas", {"astroplasmas": ["astroplasmas"]}) == ""
        )

    def test_the_unfolded_report_still_reads_naturally(self, _stubs: None) -> None:
        # The default surface must be untouched: an em dash is correct there.
        out = _plain_report(ascii_mode=False)
        assert HEDGE_MARK in out and "—" in out, out
