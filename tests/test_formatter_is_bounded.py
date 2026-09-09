"""The tools the gates run must have an upper bound.

`ruff format --check .` is a gate in both workflows, so ruff decides whether CI
passes. With no ceiling on it, a minor release that reflows one line reddens a
tree nobody touched — and the failure arrives on somebody else's push, with no
change to blame. rapidu's pin states that rule in a comment and slurmpast's
records a minor ruff release changing its rule defaults as what broke its first
CI run; this repo ran the format gate with no ceiling over it.

Shaped after `test_nvml_spelling_coverage.py`: an implication that only fires
while the hazard exists, plus a control asserting the hazard is really there, so
the pin cannot be satisfied by quietly deleting the gate. `pyproject.toml` is
read as TEXT on purpose — `tomllib` is 3.11+ and this package supports 3.10.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS = ["ci.yml", "release.yml"]


def _requirement(name: str) -> str:
    """The version spec of `name` in the `dev` extra, e.g. `>=0.15,<0.17`."""
    text = (ROOT / "pyproject.toml").read_text()
    found = re.findall(rf'^\s*"{name}([^"]*)",\s*$', text, re.MULTILINE)
    assert found, f"{name} is not pinned in pyproject.toml at all"
    assert len(found) == 1, f"{name} is pinned {len(found)} times: {found}"
    return str(found[0])


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text()


class TestTheGatedToolsAreBounded:
    @pytest.mark.parametrize("tool", ["ruff", "mypy"])
    def test_a_tool_a_gate_runs_has_an_upper_bound(self, tool: str) -> None:
        gated = [w for w in WORKFLOWS if re.search(rf"run:\s*{tool}\b", _workflow(w))]
        assert gated, f"no workflow runs {tool}; this test's premise is gone"
        spec = _requirement(tool)
        assert "<" in spec, (
            f"{tool} is run as a gate by {gated} but pinned {tool!r}{spec!r} with no "
            f"ceiling: a minor release can redden a tree nobody touched"
        )


class TestControls:
    """These pass whether or not the ceilings are present."""

    def test_the_format_gate_really_runs_so_the_pin_is_not_vacuous(self) -> None:
        # If this ever fails, the implication above went quiet for the wrong
        # reason — the gate was removed rather than the pin fixed.
        running = [w for w in WORKFLOWS if "ruff format --check" in _workflow(w)]
        assert running == WORKFLOWS, f"only {running} run `ruff format --check`"

    @pytest.mark.parametrize("tool", ["ruff", "mypy"])
    def test_a_floor_is_still_declared(self, tool: str) -> None:
        # A ceiling is not a substitute for a floor: the rule set has to be
        # bounded at both ends to be reproducible.
        assert ">=" in _requirement(tool)

    @pytest.mark.parametrize("tool", ["ruff", "mypy"])
    def test_the_tool_is_still_a_dev_dependency(self, tool: str) -> None:
        assert _requirement(tool)  # raises with a clear message if absent
