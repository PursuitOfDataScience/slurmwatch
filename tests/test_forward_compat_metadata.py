"""What breaks on a newer interpreter, and what this package advertises.

`requires-python = ">=3.10"` has no upper bound, so pip installs this on whatever
comes next. Nothing held the source to that: `src/slurmwatch` uses none of the APIs
that a newer interpreter removes or deprecates, but only by accident of nobody having
reached for one.

slurmate carries this scan and has since the round its CHANGELOG describes; slurmpast
gained it in its round sixty-five. slurmwatch had neither half.

**This file deliberately does NOT claim 3.14.** slurmpast added a `Programming
Language :: Python :: 3.14` classifier because its own `issues.md` records the
published artefact installed from PyPI onto midway2 at Python 3.14.6 and exercised
against real accounting history. slurmwatch has no such run recorded anywhere, and a
classifier is a promise to installers — so the invariant pinned here is the one that
is actually true: **the versions advertised are exactly the versions CI runs.** If a
3.14 run is ever recorded, `test_the_classifiers_are_exactly_the_tested_versions` is
what will fail, which makes adding the claim a deliberate act rather than a drift.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "slurmwatch"
CI = ROOT / ".github" / "workflows" / "ci.yml"

#: What actually breaks on a newer interpreter. `distutils` and `imp` are gone,
#: `utcnow` and `getdefaultlocale` are deprecated, `find_loader` was removed.
#: Same list slurmate and slurmpast scan for, so a fix in one transfers.
BANNED = re.compile(
    r"\b(distutils|import imp\b|utcnow|getdefaultlocale|find_loader"
    r"|pkg_resources|typing\.ByteString)\b"
)


def _pyproject() -> str:
    """Read as text, deliberately.

    `tomllib` is 3.11+ and 3.10 is the oldest version this file asserts support
    for, so importing it would make the test unrunnable on the interpreter it most
    needs to run on. The siblings' equivalents record the same reason, and
    `release.yml` avoids tomllib for it too.
    """
    return (ROOT / "pyproject.toml").read_text()


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def declared_versions() -> list[str]:
    return re.findall(r'"Programming Language :: Python :: ([0-9]+\.[0-9]+)"', _pyproject())


def matrix_versions() -> list[str]:
    found = re.search(r"python-version:\s*\[([^\]]+)\]", CI.read_text())
    assert found, "ci.yml should still declare a python matrix"
    return [v.strip().strip('"').strip("'") for v in found.group(1).split(",")]


class TestTheClaimStaysTrue:
    def test_no_removed_or_deprecated_stdlib_apis(self) -> None:
        offenders = [
            f"{path.name}:{n}"
            for path in sorted(SRC.glob("*.py"))
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if BANNED.search(line) and not line.lstrip().startswith("#")
        ]
        assert offenders == [], offenders

    def test_there_are_sources_to_scan(self) -> None:
        # A silent glob miss would make the scan vacuously clean.
        assert len(list(SRC.glob("*.py"))) >= 8

    def test_the_scanner_would_notice_a_real_offender(self) -> None:
        """A guard that cannot fail is not a guard."""
        for planted in (
            "from distutils.util import strtobool",
            "import imp",
            "datetime.datetime.utcnow()",
            "locale.getdefaultlocale()",
            "importlib.find_loader('x')",
            "import pkg_resources",
            "typing.ByteString",
        ):
            assert BANNED.search(planted), planted


class TestWhatIsAdvertisedIsWhatIsRun:
    def test_the_classifiers_are_exactly_the_tested_versions(self) -> None:
        assert sorted(declared_versions(), key=_version_key) == sorted(
            matrix_versions(), key=_version_key
        ), {"classifiers": declared_versions(), "matrix": matrix_versions()}

    def test_the_declared_floor_is_the_lowest_advertised_version(self) -> None:
        found = re.search(r'requires-python\s*=\s*">=([0-9]+\.[0-9]+)"', _pyproject())
        assert found, "requires-python should still be declared"
        assert min(declared_versions(), key=_version_key) == found.group(1)

    def test_both_lists_were_actually_found(self) -> None:
        assert len(declared_versions()) >= 2
        assert len(matrix_versions()) >= 2


class TestControls:
    """None of these compares the tree against the metadata, so each holds either way."""

    def test_the_version_key_orders_numerically_not_lexically(self) -> None:
        versions = ["3.9", "3.10", "3.13"]
        assert min(versions, key=_version_key) == "3.9"
        assert min(versions) == "3.10", "string ordering really does differ here"

    def test_the_source_tree_is_where_this_expects(self) -> None:
        assert (SRC / "cli.py").is_file() and (SRC / "tui.py").is_file()

    def test_the_generic_python_3_classifier_is_not_counted_as_a_minor(self) -> None:
        # `Python :: 3` must not slip into the minor-version list and break the
        # comparison; the regex requires a dotted pair.
        assert "3" not in declared_versions()

    def test_requires_python_is_still_declared_at_all(self) -> None:
        assert re.search(r"^requires-python\s*=", _pyproject(), re.M)
