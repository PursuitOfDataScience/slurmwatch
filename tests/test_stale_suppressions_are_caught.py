"""A `# noqa` claims the line has a known violation, and the gate now checks it.

Two directives in this repo had gone stale, and both were BARE -- the code with
no rationale after it:

* `tests/test_collector.py`'s `# noqa: F811` on a fixture ruff no longer flags;
* a `# noqa: E402` on an import following a `sys.path.insert`.

`F811` and `E402` are both in the select list and neither is exempted for
`tests/*` (that entry is `ARG`, `N802`, `N815`), so nothing was suppressing
them -- the directives simply told a reader those lines were known exceptions
when they are not.

**The bareness decided the treatment.** The same sweep found stale directives in
`nodetop` and `rapidu` where every one carries a reviewer's note -- `# noqa: S603
- fixed argv, never a shell`, `# noqa: BLE001  (a hang is worse than a report)`
-- and this rule would demand deleting the note to satisfy the linter. It was
not enabled there, deliberately.

**Measured on both ends of the dev bound before removing anything**, because a
directive one ruff calls unused can be load-bearing on another and CI resolves
the upper end: 0.15.18 and 0.16.6 agree on both. The two directives that REMAIN
in `test_collector.py` are now proven load-bearing by the same rule.

With `RUF100` selected the gate keeps them honest, so nothing here
re-implements `ruff check`. What is pinned is that the rule stays selected, plus
one end-to-end check that it fires.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The codes selected before this round, which must survive it.
ESTABLISHED = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "ARG", "C4", "T20"]


def _codes(pattern: str) -> list[str]:
    """The string list on the single `pyproject.toml` line matching `pattern`.

    Read with a regex rather than `tomllib`, which is stdlib only from 3.11
    while this package's floor is `requires-python = ">=3.10"`. slurmpast's
    `TestNothingImportsPastTheDeclaredPythonFloor` catches exactly that and it
    caught this file; slurmwatch has no such test, so the same import passed
    its gate while being unrunnable on the floor it declares. Every value
    needed here is a single line, so a parser is not required to read one.
    """
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(pattern + r"\s*=\s*\[([^\]]*)\]", text)
    assert match is not None, pattern
    return re.findall(r'"([^"]+)"', match.group(1))


class TestTheRuleIsSelected:
    def test_ruf100_is_in_the_select_list(self) -> None:
        selected = _codes("select")
        assert "RUF100" in selected, selected

    def test_no_bare_noqa_survives_in_the_tree(self) -> None:
        """The two that were removed, as a property rather than a count.

        Guards only the CLEANUP half: it reddens if a bare directive comes back,
        and it holds with `RUF100` dropped from the select list, so for the
        config half it is a control. Established by running that neuter.
        """
        offenders = []
        here = pathlib.Path(__file__).resolve()
        for path in sorted(ROOT.glob("tests/*.py")) + sorted(ROOT.glob("src/slurmwatch/*.py")):
            # THIS file quotes the directive it scans for. A source-scanning test
            # that does not exclude itself finds its own explanation.
            if path.resolve() == here:
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if "# noqa:" not in line:
                    continue
                tail = line.split("# noqa:", 1)[1].strip()
                if tail and tail == tail.split()[0]:  # nothing but the code
                    offenders.append(f"{path.name}:{number}")
        assert offenders == [], offenders

    @pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")
    def test_a_planted_stale_directive_is_reported(self, tmp_path: pathlib.Path) -> None:
        """The behavioural half: the rule actually fires, under this config."""
        planted = tmp_path / "planted.py"
        planted.write_text("x = 1  # noqa: E402\n")
        result = subprocess.run(
            [
                "ruff",
                "check",
                "--config",
                str(ROOT / "pyproject.toml"),
                "--output-format",
                "concise",
                str(planted),
            ],
            capture_output=True,
            text=True,
        )
        assert "RUF100" in result.stdout, (result.stdout, result.stderr)


class TestControls:
    """Each passes with `RUF100` removed from the select list as well as with it.

    They cover the configuration this round did not touch -- verified by running
    that neuter.
    """

    def test_every_previously_selected_code_is_still_there(
        self,
    ) -> None:
        for code in ESTABLISHED:
            assert code in _codes("select"), code

    def test_the_tests_exemption_is_unchanged(self) -> None:
        """And it is what proves `F811`/`E402` were never exempted for tests."""
        assert _codes(r'"tests/\*"') == ["ARG", "N802", "N815"]

    def test_the_print_exemptions_are_unchanged(self) -> None:
        """A CLI writes to stdout; that stays allowed."""
        assert _codes(r'"src/slurmwatch/cli\.py"') == ["T201"]

    def test_a_noted_directive_would_be_allowed(self) -> None:
        """The distinction the sweep turned on, asserted on the classifier."""
        for line, bare in [
            ("x = 1  # noqa: E402", True),
            ("x = 1  # noqa: S603 - fixed argv, never a shell", False),
            ("x = 1  # noqa: BLE001  (a hang is worse than a report)", False),
        ]:
            tail = line.split("# noqa:", 1)[1].strip()
            assert (tail == tail.split()[0]) is bare, line

    def test_the_config_reader_actually_finds_the_list(self) -> None:
        """Vacuity guard: every assertion above rests on `_codes`, and a
        regex that matched nothing would make them all pass trivially -- the
        helper asserts a match, and this checks the match is the real list."""
        selected = _codes("select")
        assert len(selected) >= 11, selected
        assert "E" in selected and "F" in selected
