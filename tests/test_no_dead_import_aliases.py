"""A module-level `alias = imported_name` that nothing reads is invisible to ruff.

`tui.py` carried `_cpu_ratio = cpu_ratio` — one occurrence in the whole package,
zero in the tests. It was a leftover: `_cpu_health` moved its rule into
`model.cpu_is_underused`, and the comment six lines below the alias still says
so ("The rule lives in model.cpu_is_underused so the plain-text summary reaches
the same verdict"). The alias went unnoticed because it keeps the *import*
used, so `F401` sees nothing to report, and a zero-reader grep for `cpu_ratio(`
finds the definition and the one real call and moves on.

Found by mining partial-branch coverage: `model.py:125` — `cpu_ratio`'s
zero-allocation guard — was uncovered, which led to asking who calls it. The
answer was "one caller that short-circuits first, plus a dead alias".

The guard itself is left alone: `cpu_is_underused` is
`cores_allocated > 1 and cpu_ratio(cpu) < threshold`, so the `<= 0` branch is
unreachable through the only real caller and is defensive by design.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "slurmwatch"


def _dead_aliases(source: str) -> dict[str, int]:
    """`{alias: lineno}` for module-level `alias = imported` nothing reads."""
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
    aliases: dict[str, int] = {}
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
            and stmt.value.id in imported
        ):
            aliases[stmt.targets[0].id] = stmt.lineno
    # Loads only: the alias's own target is a Store, so it cannot count as its
    # own reader. Names in `__all__` are strings and do not appear here.
    read = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return {name: line for name, line in aliases.items() if name not in read}


class TestNoModuleKeepsAnUnreadAlias:
    @pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
    def test_every_alias_has_a_reader(self, path: pathlib.Path) -> None:
        dead = _dead_aliases(path.read_text())
        assert dead == {}, f"{path.name}: alias with no reader {dead}"

    def test_there_are_modules_to_scan(self) -> None:
        """Vacuity guard: an empty glob makes the sweep pass trivially."""
        paths = sorted(SRC.glob("*.py"))
        assert len(paths) >= 10, len(paths)
        assert any(p.name == "tui.py" for p in paths)


class TestTheDetectorItself:
    """Guards -- the sweep above passes on a clean tree either way, so the
    detector is exercised on planted sources rather than on the package."""

    def test_a_dead_alias_is_reported(self) -> None:
        planted = "from .model import cpu_ratio\n\n_alias = cpu_ratio\n"
        assert _dead_aliases(planted) == {"_alias": 3}

    def test_an_alias_with_a_reader_is_not_reported(self) -> None:
        planted = (
            "from .model import cpu_ratio\n\n_alias = cpu_ratio\n\n"
            "def f(c):\n    return _alias(c)\n"
        )
        assert _dead_aliases(planted) == {}

    def test_a_plain_constant_is_not_an_alias(self) -> None:
        """Only `name = imported_name` counts; a literal is not an alias."""
        assert _dead_aliases("X = 3\n") == {}

    def test_an_alias_of_something_not_imported_is_left_alone(self) -> None:
        """A rename of a local definition is a different judgement call."""
        planted = "def g():\n    pass\n\n_alias = g\n"
        assert _dead_aliases(planted) == {}

    def test_a_nested_assignment_is_not_module_level(self) -> None:
        planted = "from .model import cpu_ratio\n\ndef f():\n    _alias = cpu_ratio\n    return 1\n"
        assert _dead_aliases(planted) == {}


class TestControls:
    """Each passes with the alias restored as well as removed -- they cover the
    import that alias was keeping alive, and the helper it aliased."""

    def test_the_model_helper_still_exists_and_works(self) -> None:
        from slurmwatch.model import CpuMetrics, cpu_ratio

        cpu = CpuMetrics(
            cores_allocated=8,
            usage_ns=0,
            usage_percent=0.0,
            effective_cores=2.0,
            peak_effective_cores=2.0,
            source="cgroup",
        )
        assert cpu_ratio(cpu) == pytest.approx(0.25)

    def test_the_defensive_guard_is_still_there(self) -> None:
        """Unreachable through `cpu_is_underused`, kept as a division guard."""
        from slurmwatch.model import CpuMetrics, cpu_ratio

        cpu = CpuMetrics(
            cores_allocated=0,
            usage_ns=0,
            usage_percent=0.0,
            effective_cores=0.0,
            peak_effective_cores=0.0,
            source="cgroup",
        )
        assert cpu_ratio(cpu) == 0.0

    def test_the_cpu_health_verdict_still_reads_the_shared_rule(self) -> None:
        """What `_cpu_health` uses instead of the alias."""
        from slurmwatch.model import CpuMetrics, cpu_is_underused

        cpu = CpuMetrics(
            cores_allocated=8,
            usage_ns=0,
            usage_percent=0.0,
            effective_cores=0.4,
            peak_effective_cores=0.4,
            source="cgroup",
        )
        assert cpu_is_underused(cpu, 0.15) is True
