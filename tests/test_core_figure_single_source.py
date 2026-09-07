"""One core figure, two spellings, in the same sentence.

The CPU-underuse advisory is drawn by both surfaces and the ADVICE half was
already single-sourced -- `model.CPU_UNDERUSE_ADVICE`, with a comment in `tui`
saying so. The FIGURE was not: `tui` used a helper that drops a pointless
trailing ``.0``, `cli` wrote ``:.1f``. On one snapshot with
``effective_cores == 1.0``, both surfaces driven at once:

    dashboard   only ~1 of 8 cores are doing work — …
    plain       only ~1.0 of 8 cores are doing work — …

and the plain summary's own CPU line read ``~1.0 of 8 cores (avg, running
steps)`` against the card's ``1 of 8``. An integer is the common case rather than
a corner: a single-threaded process on an 8-core allocation reads exactly 1.0, and
a saturating one reads exactly the core count.

Nothing caught it because **each surface had a test pinning its own spelling** --
`test_tui.py` asserted ``_fmt_cores(1.0) == "1"  # not "1.0"``, and
`test_cli.py::TestDegradedSummaryCarriesTheAdvisory` asserted
``"only ~1.0 of 8 cores" in out``. The second was incidental: that test's subject
is the advisory being present at all (SW-18), and its own docstring quotes
``~1.0 of 8 cores`` as the PRE-fix symptom. It now expects ``~1``.

Found by intersecting the two modules' string literals, the probe that also found
rapidu's three shared strings. `format_cores` now lives in `units`, which both
modules already import.

**Where each surface shows the figure**, measured, because the gating is not
symmetric and a test that assumes it is compares nothing:

* the plain summary's CPU line carries it on EVERY snapshot;
* the advisory carries it only when the job is underused (threshold x cores) --
  and in `cli` only when the snapshot is not `remote`, which is why the
  advisory half below builds an on-node snapshot.
"""

from __future__ import annotations

import asyncio
import pathlib
import re
import sys
from typing import Any

import pytest

from slurmwatch.model import CPU_UNDERUSE_ADVICE
from slurmwatch.units import format_cores

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

#: The figure as either surface prints it: ``~N of M cores``.
_FIGURE = re.compile(r"~(\S+) of (\d+) cores")

ALLOCATED = 8
#: Below `threshold * ALLOCATED` the advisory fires; above it, only the CPU line
#: carries a figure. Asserted in `test_the_gating_is_what_this_file_assumes`.
#: Measured, not assumed -- I guessed this boundary twice and was wrong twice.
#: `cpu_is_underused` is `cores_allocated > 1 and cpu_ratio(cpu) < threshold`, and
#: `config.cpu_underuse_threshold` defaults to **0.15**, so on 8 cores the
#: advisory fires strictly below 1.2. A wrong boundary here does not fail loudly:
#: it makes the cross-surface assertion compare a figure against an empty set.
UNDERUSED = (0.0, 0.4, 1.0)
WELL_USED = (1.2, 2.8, 6.1, 8.0)


def _snapshot(effective: float, *, remote: bool) -> tuple[Any, Any]:
    import test_tui as harness

    snap = harness._sstat_snapshot(rss=8 * 1024**3, limit=64 * 1024**3, cpu_seconds=1.0)
    snap.cpu.effective_cores = effective
    snap.cpu.peak_effective_cores = float(ALLOCATED)
    snap.cpu.cores_allocated = ALLOCATED
    snap.remote = remote
    return harness, snap


def _plain(effective: float, *, remote: bool = True) -> str:
    harness, snap = _snapshot(effective, remote=remote)
    cfg = harness.SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
    return str(harness._plain_summary(harness._sstat_ctx(), snap, cfg))


def _card(effective: float) -> str:
    harness, snap = _snapshot(effective, remote=True)
    cfg = harness.SlurmwatchConfig(poll_interval=0.05, headless_interval=0.05)
    surfaces = asyncio.run(harness._dashboard_surfaces(harness._sstat_ctx(), snap, cfg, drill="c"))
    return str(surfaces["body"])


def _figures(text: str) -> set[str]:
    return {m.group(1) for m in _FIGURE.finditer(text)}


class TestTheGatingIsWhatThisFileAssumes:
    """Vacuity guards. Every comparison below depends on which surface shows a
    figure for which input, so that is asserted rather than assumed."""

    @pytest.mark.parametrize("effective", UNDERUSED)
    def test_an_underused_job_shows_a_figure_on_both(self, effective: float) -> None:
        assert _figures(_plain(effective)), effective
        assert _figures(_card(effective)), effective

    @pytest.mark.parametrize("effective", WELL_USED)
    def test_a_well_used_job_shows_one_only_on_the_plain_cpu_line(self, effective: float) -> None:
        assert _figures(_plain(effective)), effective
        assert _figures(_card(effective)) == set(), effective


class TestBothSurfacesPrintOneFigure:
    @pytest.mark.parametrize("effective", UNDERUSED + WELL_USED)
    def test_the_plain_cpu_line_uses_the_shared_formatter(self, effective: float) -> None:
        text = _plain(effective)
        assert f"~{format_cores(effective)} of {ALLOCATED} cores" in text, text

    @pytest.mark.parametrize("effective", UNDERUSED)
    def test_the_two_surfaces_agree(self, effective: float) -> None:
        """The drift, end to end: whatever ``~N`` each prints, the two must match."""
        from_plain = _figures(_plain(effective))
        from_card = _figures(_card(effective))
        assert from_plain == from_card == {format_cores(effective)}, (
            effective,
            from_plain,
            from_card,
        )

    @pytest.mark.parametrize("effective", UNDERUSED)
    def test_the_advisory_agrees_too_on_node(self, effective: float) -> None:
        """The ADVISORY's own figure, which the comparison above cannot reach.

        `cli` prints the advisory only when the snapshot is not `remote`, so the
        default off-node plain text carries the CPU line alone -- and reverting
        just the advisory's formatter left every behavioural assertion green,
        with only the source pin catching it. Measured, then closed.
        """
        plain = _plain(effective, remote=False)
        assert "Advice" in plain, plain
        advisory = [ln for ln in plain.splitlines() if "Advice" in ln]
        assert advisory, plain
        assert f"~{format_cores(effective)} of {ALLOCATED} cores" in advisory[0], advisory
        # Only where the two spellings actually differ: for a fraction like 0.4
        # `format_cores` and `:.1f` produce the same string, so the negative would
        # contradict the positive above.
        if format_cores(effective) != f"{effective:.1f}":
            assert f"~{effective:.1f} of" not in advisory[0], advisory

    @pytest.mark.parametrize("effective", [1.0, 2.0, 8.0])  # only 1.0 is underused
    def test_an_integer_loses_its_trailing_zero_on_both(self, effective: float) -> None:
        """The case that differed. `~1` not `~1.0`."""
        assert f"~{effective:.1f} of" not in _plain(effective), effective
        if effective in UNDERUSED:
            assert f"~{effective:.1f} of" not in _card(effective), effective

    @pytest.mark.parametrize("effective", [0.4, 6.1, 2.8])
    def test_control_a_fraction_reads_as_it_always_did(self, effective: float) -> None:
        """CONTROL. A real fraction keeps its decimal under either spelling, so
        this passed before the fix and must still pass -- if it reddens the change
        went further than the trailing zero."""
        assert f"~{effective:.1f} of {ALLOCATED} cores" in _plain(effective)


class TestTheFormatterHasOneHome:
    def test_neither_surface_formats_the_figure_itself(self) -> None:
        """The source pin. `cli` spelled it ``:.1f``, which is what drifted."""
        import slurmwatch.cli as cli_mod
        import slurmwatch.tui as tui_mod

        for module in (cli_mod, tui_mod):
            source = pathlib.Path(str(module.__file__)).read_text()
            assert "effective_cores:.1f" not in source, module.__name__
            assert "format_cores" in source, module.__name__

    def test_the_rule_it_states_is_the_rule_it_keeps(self) -> None:
        """Vacuity guard: `format_cores` must actually differ from ``:.1f``, or
        every assertion above would hold against any implementation."""
        assert format_cores(1.0) == "1" and f"{1.0:.1f}" == "1.0"
        assert format_cores(8.0) == "8"
        assert format_cores(0.0) == "0"
        assert format_cores(2.8) == "2.8"
        assert format_cores(0.04) == "0"

    def test_control_the_advice_half_is_still_shared(self) -> None:
        """CONTROL. The words after the dash were single-sourced by an earlier
        round; sharing the figure must not disturb them. The plain advisory is
        gated on the snapshot NOT being remote, so this one is on-node."""
        head, _, tail = CPU_UNDERUSE_ADVICE.partition("--cpus-per-task")
        for surface in (_plain(1.0, remote=False), _card(1.0)):
            assert head.strip() in surface, surface
            assert "--cpus-per-task" in surface, surface
            assert tail.strip(" .") in surface, surface

    def test_control_a_well_used_job_still_gets_no_advisory(self) -> None:
        """CONTROL. The advisory is threshold-gated and the figure change must not
        make it fire more often. Holds in both states."""
        assert "Advice" not in _plain(7.0, remote=False)
