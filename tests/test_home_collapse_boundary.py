"""The `~` collapse must match a path COMPONENT boundary, and nothing checked that.

`_shorten_path` collapses a home-relative path to `~`, guarded by::

    if home and (path == home or path.startswith(home + os.sep)):

The `+ os.sep` is the whole subtlety. Drop it and `startswith(home)` still passes
every existing test, while `/home/alicex/work/run.sh` renders as `~x/work/run.sh` --
another user's path relabelled as the reader's own home, in the one field a reader
uses to confirm they are looking at their own job.

`issues.md` D21 says this test reads the developer's real `$HOME` and "breaks at
HOME=/". Measured: it does read it, but it does not break -- `/`, empty, unset, a
trailing slash and a relative value all pass, because `expanduser` falls back to the
passwd entry and because at `HOME=/` the test's own `home + "/work/run.sh"` yields a
doubled slash that the guard then matches. So the finding as written is withdrawn.
What was really missing is the negative, which is what this file adds; `HOME` is set
explicitly here so the assertions measure the rule rather than this machine.
"""

import os

import pytest

from slurmwatch.tui import _shorten_path

HOME = "/home/alice"


@pytest.fixture(autouse=True)
def _fixed_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """A known home, so nothing here depends on who ran it."""
    monkeypatch.setenv("HOME", HOME)


class TestTheCollapseStopsAtAComponentBoundary:
    @pytest.mark.parametrize(
        "path",
        [
            "/home/alicex/work/run.sh",  # longer name, same prefix
            "/home/alice-backup/w.sh",  # a sibling that starts with the home name
            "/home/alice.old/w.sh",
        ],
    )
    def test_a_prefix_that_is_not_a_component_does_not_collapse(self, path: str) -> None:
        out = _shorten_path(path, budget=40)
        assert out == path, out
        assert not out.startswith("~"), out

    def test_a_shorter_sibling_is_untouched(self) -> None:
        assert _shorten_path("/home/alic/work/run.sh", budget=40) == "/home/alic/work/run.sh"

    def test_the_positive_case_still_collapses(self) -> None:
        # The loud branch, restated against a KNOWN home rather than the runner's.
        assert _shorten_path(HOME + "/work/run.sh", budget=40) == "~/work/run.sh"

    def test_home_itself_collapses_to_a_bare_tilde(self) -> None:
        assert _shorten_path(HOME, budget=40) == "~"

    def test_the_guard_is_what_makes_these_differ(self) -> None:
        """Vacuity guard, not a control: the two inputs differ by one character, so
        a build where nothing collapsed at all would fail here rather than pass."""
        inside = _shorten_path(HOME + "/work/run.sh", budget=40)
        beside = _shorten_path(HOME + "x/work/run.sh", budget=40)
        assert inside != beside
        assert inside.startswith("~") and not beside.startswith("~")


class TestControls:
    """Paths the boundary rule cannot reach, so these hold with the guard either way."""

    def test_a_path_outside_home_is_left_alone(self) -> None:
        # Shares no prefix with home, so no spelling of the guard can touch it.
        assert _shorten_path("/scratch/u/run.sh", budget=40) == "/scratch/u/run.sh"

    def test_a_short_path_is_returned_unchanged(self) -> None:
        assert _shorten_path("/a/b/c.sh", budget=40) == "/a/b/c.sh"

    def test_a_long_path_outside_home_still_elides_its_middle(self) -> None:
        out = _shorten_path("/scratch/u/" + "deep/" * 12 + "run.sh", budget=40)
        assert len(out) <= 40 and out.endswith("run.sh"), out
        assert "…" in out, out

    def test_the_env_fixture_really_took(self) -> None:
        # If the monkeypatch silently failed, every assertion above would be
        # measuring this machine's home instead of the rule.
        assert os.path.expanduser("~") == HOME
