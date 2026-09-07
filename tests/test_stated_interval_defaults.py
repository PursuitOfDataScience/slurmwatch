"""`--interval`'s help names four numbers, and all four live in the code too.

    Polling interval in seconds (default: 0.5 for TUI, 1.0 for headless;
    raised to a 0.1s floor on the node, 1.0s off it)

Every one of those is a copy:

* `0.5`  -> `SlurmwatchConfig().poll_interval`
* `1.0`  -> `SlurmwatchConfig().headless_interval`
* `0.1`  -> `config.MIN_INTERVAL`
* `1.0`  -> `config.MIN_REMOTE_INTERVAL`

A copy drifts silently: lower the floor and the sentence still promises the old
one. Nothing checked them, and this is the flag most likely to be tuned -- the
memory of this package records the interval being forwarded, floored and
re-floored across three code paths.

Sibling packages have the same check where their parsers are reachable
(`nodetop/tests/test_stated_defaults.py`, `slurmpast/tests/test_stated_defaults.py`)
and rapidu sidesteps it by interpolating `%(default)s`. Those compare
`action.default` to a `(default: X)` regex; this one cannot, because
`--interval`'s own default is `None` (the two real values are chosen later by
mode) -- so the comparison here is against the CONFIG the modes read, which is
where the numbers actually live.

**The expected strings are generated from the constants, not typed**, so this
file cannot drift either. The format spec has to match the help's own spelling:
`:.1f`, not `:g`. `f"{1.0:g}"` is `"1"` while the sentence says `"1.0"`, so the
first version of this file failed on correct code -- the same trailing-zero
distinction `units.format_cores` exists to make.

`headless_interval` and `MIN_REMOTE_INTERVAL` are both
`1.0`, so each assertion anchors on the words around the number rather than on
the number alone -- otherwise one of the two could move and the test would still
find "1.0" somewhere in the sentence.
"""

from __future__ import annotations

import argparse

import pytest

from slurmwatch.cli import _build_parser
from slurmwatch.config import MIN_INTERVAL, MIN_REMOTE_INTERVAL, SlurmwatchConfig


def _action(flag: str) -> argparse.Action:
    for action in _build_parser()._actions:
        if flag in action.option_strings:
            return action
    raise AssertionError(f"{flag} is gone from the parser")


def _interval_help() -> str:
    return _action("--interval").help or ""


class TestTheHelpNamesTheRealNumbers:
    def test_the_tui_interval_is_the_configs(self) -> None:
        want = f"{SlurmwatchConfig().poll_interval:.1f} for TUI"
        assert want in _interval_help(), (want, _interval_help())

    def test_the_headless_interval_is_the_configs(self) -> None:
        want = f"{SlurmwatchConfig().headless_interval:.1f} for headless"
        assert want in _interval_help(), (want, _interval_help())

    def test_the_on_node_floor_is_the_constant(self) -> None:
        want = f"{MIN_INTERVAL:.1f}s floor"
        assert want in _interval_help(), (want, _interval_help())

    def test_the_off_node_floor_is_the_constant(self) -> None:
        want = f"{MIN_REMOTE_INTERVAL:.1f}s off it"
        assert want in _interval_help(), (want, _interval_help())


class TestTheComparisonItself:
    """Guards -- a check that passes because it compares nothing is worse than none."""

    def test_the_flag_still_has_a_help_string(self) -> None:
        assert len(_interval_help()) > 40, _interval_help()

    def test_the_two_ones_are_told_apart_by_their_words(self) -> None:
        """`headless_interval` and `MIN_REMOTE_INTERVAL` are both 1.0, so a bare
        number search would accept either standing in for the other."""
        assert SlurmwatchConfig().headless_interval == MIN_REMOTE_INTERVAL
        text = _interval_help()
        assert "for headless" in text and "off it" in text, text

    def test_a_drifted_number_would_be_caught(self) -> None:
        """The comparison, exercised on a planted sentence rather than on the
        real one -- so it holds whatever the real values become."""
        planted = "interval (default: 0.5 for TUI, 1.0 for headless)"
        assert f"{0.6:.1f} for TUI" not in planted
        assert f"{0.5:.1f} for TUI" in planted


class TestControls:
    """Each passes whatever the constants are, so they survive any neuter of a
    number or of the sentence -- verified by running those neuters."""

    def test_the_flag_takes_a_value(self) -> None:
        assert _action("--interval").metavar == "SECONDS"
        assert _action("--interval").default is None

    def test_the_help_still_says_what_the_flag_is(self) -> None:
        assert "Polling interval in seconds" in _interval_help()

    def test_the_floors_are_ordered(self) -> None:
        """Off-node polling is the expensive one, so its floor is the higher."""
        assert MIN_REMOTE_INTERVAL > MIN_INTERVAL

    @pytest.mark.parametrize("flag", ["--once", "--log", "--json", "--format"])
    def test_the_neighbouring_flags_are_still_there(self, flag: str) -> None:
        assert _action(flag) is not None
