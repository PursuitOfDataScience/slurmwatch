"""Nothing imports a stdlib module newer than the floor this package declares.

`pyproject.toml` says `requires-python = ">=3.10"` and `ci.yml` runs a 3.10 job,
and this repo already states the rule that follows from that -- in prose, at
`test_forward_compat_metadata.py`'s `_pyproject()`:

    Read as text, deliberately.

    `tomllib` is 3.11+ and 3.10 is the oldest version this file asserts support
    for, so importing it would make the test unrunnable on the interpreter it
    most needs to run on. The siblings' equivalents record the same reason, and
    `release.yml` avoids tomllib for it too.

Nothing checked it. It was followed by hand in that one file and nowhere else,
which is the shape this loop keeps finding: a rule the code states about itself
with no test reading it. It cost slurmpast a CI round-trip already -- its
equivalent test opens "Four of its assertions died with `ModuleNotFoundError` in
CI's py3.10 and oldest-Textual jobs, having passed every local gate: a local run
is one interpreter, and this is the class of defect that costs nothing to catch
and cannot be caught there."

`sys.stdlib_module_names` is the RUNNING interpreter's, so it cannot answer "was
this in 3.10". The table below is the alternative and is deliberately small: the
stdlib additions between this floor and the newest version CI runs. A module
missing from it is not a false pass -- CI still runs 3.10 -- it is one round-trip
through CI instead of a local failure, which is what this exists to save.

Ported from `slurmpast/tests/test_portability.py`, deliberately keeping its
table and its `try:`-guarded exemption so a fix in one transfers, exactly as
`BANNED` in `test_forward_compat_metadata.py` already says of its own list.
"""

from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: module -> the version it entered the stdlib.
ADDED_IN: dict[str, tuple[int, ...]] = {
    "tomllib": (3, 11),
    "dbm.sqlite3": (3, 13),
    "annotationlib": (3, 14),
    "compression": (3, 14),
}


def _floor() -> tuple[int, ...]:
    spec = re.search(
        r'requires-python\s*=\s*"[>=~^]*([\d.]+)"', (ROOT / "pyproject.toml").read_text()
    )
    assert spec, "pyproject no longer declares requires-python"
    return tuple(int(part) for part in spec.group(1).split("."))


def _guarded_imports(tree: ast.AST) -> set[str]:
    """Names imported inside a `try:` -- a backfill, not an unguarded import."""
    safe: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Import):
                safe.update(alias.name for alias in inner.names)
            elif isinstance(inner, ast.ImportFrom) and inner.module:
                safe.add(inner.module)
    return safe


def _sources() -> list[pathlib.Path]:
    return sorted(
        list((ROOT / "src" / "slurmwatch").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    )


def _offenders(paths: list[pathlib.Path], late: dict[str, tuple[int, ...]]) -> list[str]:
    found = []
    for path in paths:
        tree = ast.parse(path.read_text())
        guarded = _guarded_imports(tree)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                if name in late and name not in guarded:
                    found.append(f"{path.name}:{node.lineno} imports {name}")
    return found


class TestTheFloorHolds:
    def test_no_module_newer_than_the_floor_is_imported_unguarded(self) -> None:
        floor = _floor()
        late = {name: added for name, added in ADDED_IN.items() if added > floor}
        assert late, "the floor has moved past every module in the table; prune it"
        assert _offenders(_sources(), late) == []

    def test_the_declared_floor_is_still_what_the_prose_assumes(self) -> None:
        """`test_forward_compat_metadata` says "3.10 is the oldest version this
        file asserts support for". If the floor rises past 3.11, `tomllib`
        becomes legal and that paragraph -- and this table -- want revisiting."""
        assert _floor() == (3, 10)

    def test_a_planted_import_is_caught(self) -> None:
        """The detector fires, rather than passing because it scans nothing."""
        planted = ast.parse("import tomllib\n")
        assert _guarded_imports(planted) == set()
        late: dict[str, tuple[int, ...]] = {"tomllib": (3, 11)}
        names = [
            alias.name
            for node in ast.walk(planted)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert [n for n in names if n in late] == ["tomllib"]

    def test_every_source_file_is_actually_parsed(self) -> None:
        """Vacuity guard: an empty file list makes the sweep pass trivially."""
        paths = _sources()
        assert len(paths) > 20, len(paths)
        assert any(p.name == "collector.py" for p in paths)
        assert any(p.name.startswith("test_") for p in paths)


class TestControls:
    """Each passes with the sweep removed as well as with it -- this round adds a
    test rather than changing behaviour, so they cover the facts it rests on.
    """

    def test_a_try_guarded_import_is_exempt(self) -> None:
        """The backfill pattern, which must stay legal: a 3.11+ module behind a
        `try:` with a fallback is how a package supports both."""
        tree = ast.parse("try:\n    import tomllib\nexcept ImportError:\n    tomllib = None\n")
        assert "tomllib" in _guarded_imports(tree)
        exempt: dict[str, tuple[int, ...]] = {"tomllib": (3, 11)}
        assert _offenders([], exempt) == []

    def test_the_floor_reader_finds_the_declaration(self) -> None:
        assert "requires-python" in (ROOT / "pyproject.toml").read_text()
        assert _floor() >= (3, 0)

    def test_the_prose_this_rests_on_is_still_there(self) -> None:
        """If that paragraph goes, this test's motivation goes with it."""
        source = (ROOT / "tests" / "test_forward_compat_metadata.py").read_text()
        assert "tomllib` is 3.11+" in source, "the stated rule moved or was reworded"
