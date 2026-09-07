"""The whole underuse sentence has one home, not just its two ends.

`tui.py` says this at the call site, and has for several rounds:

    # The tail comes from model.CPU_UNDERUSE_ADVICE so this line and the
    # plain-text summary's cannot drift; only the ink on the flag differs.

That was true of the tail. It was not true of *this line*: the subject half --
``only ~N of M cores are doing work`` -- was spelled once in `cli.py` and once
in `tui.py`, and the subject is the half that actually drifted. `format_cores`'s
own docstring records the incident it caused ("the card said ``only ~1 of 8
cores are doing work`` and the plain summary said ``only ~1.0 of 8 ...``"), and
the previous round fixed it by sharing the *formatter* -- which repaired that
instance and left the sentence in two places.

`model.cpu_underuse_subject` is now the one home, so the comment describes what
the code does. Its companion `test_core_figure_single_source.py` pins the figure
inside the sentence; this file pins the words around it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from slurmwatch.model import CPU_UNDERUSE_ADVICE, cpu_underuse_subject

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_core_figure_single_source import (
    ALLOCATED,
    UNDERUSED,
    WELL_USED,
    _card,
    _plain,
    _snapshot,
)


def _subject(effective: float) -> str:
    """The clause as the shared helper spells it for this reading."""
    _, snap = _snapshot(effective, remote=True)
    return cpu_underuse_subject(snap.cpu)


class TestBothSurfacesDrawTheSharedSentence:
    @pytest.mark.parametrize("effective", UNDERUSED)
    def test_the_plain_advisory_is_the_shared_clause(self, effective: float) -> None:
        """On-node, because `cli` refuses the advisory off-node by design."""
        plain = _plain(effective, remote=False)
        assert _subject(effective) in plain, (effective, plain)

    @pytest.mark.parametrize("effective", UNDERUSED)
    def test_the_dashboard_insight_is_the_shared_clause(self, effective: float) -> None:
        assert _subject(effective) in _card(effective), effective

    @pytest.mark.parametrize("effective", UNDERUSED)
    def test_the_two_surfaces_spell_it_identically(self, effective: float) -> None:
        """The drift this prevents, stated as the reader would meet it."""
        clause = _subject(effective)
        assert clause in _plain(effective, remote=False)
        assert clause in _card(effective)


class TestTheSentenceHasOneHome:
    def test_neither_renderer_spells_the_clause(self) -> None:
        """The source pin. Both modules used to carry these words verbatim."""
        import slurmwatch.cli as cli_mod
        import slurmwatch.tui as tui_mod

        for module in (cli_mod, tui_mod):
            source = pathlib.Path(str(module.__file__)).read_text()
            assert "cores are doing work" not in source, module.__name__
            assert "cpu_underuse_subject" in source, module.__name__

    def test_the_home_spells_it_exactly_once(self) -> None:
        """One occurrence, not zero -- the helper is allowed to say it."""
        import slurmwatch.model as model_mod

        source = pathlib.Path(str(model_mod.__file__)).read_text()
        assert source.count("cores are doing work") == 1, source.count("cores are doing work")

    def test_the_clause_is_words_only(self) -> None:
        """Vacuity guard: markup in the shared clause would make the dashboard
        assertion above pass for the wrong reason, and would leak Rich tags into
        the plain summary."""
        clause = _subject(1.0)
        for markup in ("[", "]", "\x1b"):
            assert markup not in clause, clause
        assert clause.startswith("only ~")
        assert clause.endswith("cores are doing work")


class TestControls:
    """These pass with either call site reverted to its own spelling.

    Verified by neutering `cli` and `tui` separately -- the fix has two call
    sites and a per-surface neuter is the only way to see that each is wired.
    """

    def test_the_advice_half_is_still_shared(self) -> None:
        """The end an earlier round single-sourced; untouched by this one."""
        head, _, tail = CPU_UNDERUSE_ADVICE.partition("--cpus-per-task")
        for surface in (_plain(1.0, remote=False), _card(1.0)):
            assert head.strip() in surface, surface
            assert "--cpus-per-task" in surface, surface
            assert tail.strip(" .") in surface, surface

    @pytest.mark.parametrize("effective", WELL_USED)
    def test_a_well_used_job_still_gets_no_advisory(self, effective: float) -> None:
        """The threshold gate. Sharing a sentence must not make it fire more."""
        assert "Advice" not in _plain(effective, remote=False), effective

    def test_the_plain_cpu_line_is_not_the_advisory(self) -> None:
        """The CPU row carries a figure on every snapshot and is a different
        sentence -- so a surface can hold the figure and not the clause."""
        plain = _plain(8.0, remote=False)
        assert f"of {ALLOCATED} cores" in plain
        assert "cores are doing work" not in plain

    def test_the_helper_tracks_the_reading_it_is_given(self) -> None:
        """Vacuity guard: a constant clause would satisfy every containment
        assertion above regardless of which surface drew it."""
        assert _subject(1.0) != _subject(0.4)
        assert "8" in _subject(1.0)
