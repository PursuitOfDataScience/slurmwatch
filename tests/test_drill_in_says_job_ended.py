"""The resource drill-in called a frozen snapshot "live" (issues.md D14).

`ResourceDetailScreen._refresh` built its title as::

    live = "live" if not snap.remote else f"{int(time.time() - snap.timestamp)}s old"

`DashboardScreen._show_job_ended` sets `_job_ended = True` and says at that very site
that "the poll loop has stopped", so no further frame arrives. This screen keeps
re-rendering the last snapshot on its OWN timer — so the word "live" outlived the
thing it described, and the class docstring promised a "live figure" that "keeps
updating while open".

**`_job_ended` was reachable here the whole time**, through the `self._dashboard`
handed to `__init__`; nothing asked. The one read of it anywhere near this code is
`LogViewScreen`, whose comment cites *this* defect as its precedent:

    if self._dashboard._job_ended:
        # D14's mistake in the resource drill-in was calling a frozen snapshot
        # live; a tail that has stopped growing must say why.
        bits.append("job ended")

So the sibling was fixed on the strength of D14 and D14 itself was left — which is the
"fixed only on one side" shape this repo's issues.md names.

An off-node snapshot keeps its age: `0s old · job ended`. "42s old" stays true and
useful; it is only "live" that becomes false.
"""

import types
from typing import Any

import pytest

from slurmwatch.config import SlurmwatchConfig
from slurmwatch.tui import ResourceDetailScreen
from tests.test_tui import _make_snapshot


class _Recorder:
    """Stands in for the `#detail-title` Static, capturing what was written."""

    def __init__(self) -> None:
        self.text = ""

    def update(self, markup: Any) -> None:
        self.text = str(markup)


def _title(*, ended: bool, remote: bool = False, have_snapshot: bool = True) -> str:
    """The drill-in's title line for one dashboard state.

    `_refresh_cpu` is stubbed out because the title is written BEFORE the
    per-resource dispatch, and the chart path needs a live Textual app for
    `Screen.size`. The subject here is the title, not the graph.
    """
    snap = None
    if have_snapshot:
        snap = _make_snapshot()
        snap.remote = remote
    screen = ResourceDetailScreen.__new__(ResourceDetailScreen)
    screen._resource = "cpu"
    screen._dashboard = types.SimpleNamespace(  # type: ignore[assignment]
        latest_snapshot=snap, config=SlurmwatchConfig(), _job_ended=ended
    )
    recorder = _Recorder()
    screen.query_one = lambda sel, kind=None: recorder  # type: ignore[assignment]
    screen._refresh_cpu = lambda *a, **k: None  # type: ignore[method-assign]
    screen._refresh()
    return recorder.text


class TestAFrozenSnapshotIsNotCalledLive:
    def test_an_ended_job_says_so_instead_of_live(self) -> None:
        title = _title(ended=True)
        assert "job ended" in title, title
        assert "live" not in title, title

    def test_an_ended_off_node_job_keeps_its_age_and_adds_the_notice(self) -> None:
        """The age is still true; only "live" becomes false."""
        title = _title(ended=True, remote=True)
        assert "job ended" in title, title
        assert "s old" in title, title

    @pytest.mark.parametrize("remote", [False, True], ids=["on-node", "off-node"])
    def test_the_two_states_do_not_render_identically(self, remote: bool) -> None:
        # The whole point: before this, a running job and an ended one produced
        # the same title on the on-node path.
        assert _title(ended=False, remote=remote) != _title(ended=True, remote=remote)


class TestControls:
    """None of these depends on the ended branch, so each holds with the fix in or out."""

    def test_a_running_on_node_job_still_says_live(self) -> None:
        title = _title(ended=False)
        assert "live" in title, title
        assert "job ended" not in title, title

    def test_a_running_off_node_job_still_reports_its_age(self) -> None:
        title = _title(ended=False, remote=True)
        assert "s old" in title, title
        assert "job ended" not in title, title

    def test_the_resource_still_names_itself(self) -> None:
        for ended in (False, True):
            assert "CPU" in _title(ended=ended)

    def test_no_snapshot_still_says_awaiting(self) -> None:
        title = _title(ended=False, have_snapshot=False)
        assert "awaiting telemetry" in title, title

    def test_the_log_view_sibling_still_discloses_it(self) -> None:
        # The precedent this fix follows, asserted against the source so the
        # docstring's account of it cannot go stale.
        import pathlib

        import slurmwatch.tui as tui_mod

        text = pathlib.Path(tui_mod.__file__).read_text()
        assert 'bits.append("job ended")' in text
