"""The release gate's Python endpoints track ci.yml's, and nothing checked that.

`release.yml` runs the suite itself rather than trusting ci.yml, and says why:

    GitHub cannot express `needs:` across workflow files, so a tag push went
    straight to build-and-publish with no dependency on the suite at all -- and a
    PyPI upload cannot be taken back: the version is burned whether or not the
    code works.

It gates on the ENDPOINTS of ci.yml's matrix rather than the whole thing, and runs
`pytest -q` rather than ci.yml's coverage invocation. Both differences are
deliberate -- the identical shape appears in all five sibling packages, which is
design, not drift -- and this file pins the part that must not drift: if ci.yml's
floor moves (say 3.10 -> 3.11, or rapidu's 3.9 -> 3.10) and `release.yml` is not
moved with it, the irreversible path would gate on a version the package no longer
claims to support, or stop gating on the lowest one it does.

What is deliberately NOT pinned here: the pytest invocation, and the middle of the
matrix. Those are the release gate being a smoke test on purpose.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"

#: The checks whose spelling must be IDENTICAL on both paths. Not pytest: the
#: release gate drops the coverage flags on purpose (see the module docstring).
SHARED_CHECKS = ("ruff check", "ruff format", "mypy")


def matrix_versions(text: str) -> list[str]:
    """The `python-version:` list of the first matrix in a workflow, in order."""
    found = re.search(r"python-version:\s*\[([^\]]+)\]", text)
    assert found, "no python-version matrix found"
    return [v.strip().strip('"').strip("'") for v in found.group(1).split(",")]


def gate_commands(text: str) -> list[str]:
    """Every `run:` line that invokes a gate tool, normalised to one line."""
    out = []
    for raw in re.findall(r"^\s*(?:- )?run:\s*(.+)$", text, flags=re.M):
        cmd = raw.strip()
        if cmd.startswith(("ruff", "mypy", "python -m pytest", "pytest")):
            out.append(re.sub(r"\s+", " ", cmd))
    return out


def _as_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


class TestTheReleaseGateTracksCi:
    def test_the_endpoints_are_ci_s_endpoints(self) -> None:
        ci = matrix_versions(CI.read_text())
        release = matrix_versions(RELEASE.read_text())
        lowest, highest = min(ci, key=_as_tuple), max(ci, key=_as_tuple)
        assert release == [lowest, highest], (
            f"release.yml gates on {release}; ci.yml's endpoints are "
            f"{[lowest, highest]}. A tag push cannot be taken back, so the "
            "irreversible path must gate on the versions the package claims."
        )

    def test_the_lint_and_type_checks_are_spelled_identically(self) -> None:
        ci = [c for c in gate_commands(CI.read_text()) if c.startswith(SHARED_CHECKS)]
        release = [c for c in gate_commands(RELEASE.read_text()) if c.startswith(SHARED_CHECKS)]
        assert ci == release, {"ci.yml": ci, "release.yml": release}

    def test_the_declared_floor_is_the_gated_floor(self) -> None:
        """`requires-python` must not promise a version nothing runs."""
        declared = re.search(
            r'requires-python\s*=\s*"[><=~^]*([0-9.]+)"', (ROOT / "pyproject.toml").read_text()
        )
        assert declared, "no requires-python"
        gated = min(matrix_versions(CI.read_text()), key=_as_tuple)
        assert _as_tuple(gated) >= _as_tuple(declared.group(1)), (declared.group(1), gated)

    def test_both_workflows_really_were_parsed(self) -> None:
        # Guards against the regexes matching nothing and every check passing.
        assert len(matrix_versions(CI.read_text())) >= 2
        assert len(gate_commands(CI.read_text())) >= 3
        assert len(gate_commands(RELEASE.read_text())) >= 3


class TestControls:
    """The parsers, on planted YAML, so no edit to the real workflows moves them."""

    def test_the_matrix_parser_keeps_order_and_strips_quotes(self) -> None:
        planted = (
            'jobs:\n  x:\n    strategy:\n      matrix:\n        python-version: ["3.9", "3.13"]\n'
        )
        assert matrix_versions(planted) == ["3.9", "3.13"]

    def test_the_command_parser_ignores_non_gate_steps(self) -> None:
        planted = (
            "      - run: python -m pip install --upgrade pip\n"
            "      - run: ruff check .\n"
            "        run: echo hello\n"
            "      - run: mypy src/ tests/\n"
        )
        assert gate_commands(planted) == ["ruff check .", "mypy src/ tests/"]

    def test_version_ordering_is_numeric_not_lexical(self) -> None:
        # "3.9" > "3.10" as strings; the endpoints check would pick the wrong floor.
        versions = ["3.9", "3.10", "3.13"]
        assert min(versions, key=_as_tuple) == "3.9"
        assert max(versions, key=_as_tuple) == "3.13"
        assert min(versions) == "3.10", "string ordering really is different here"
