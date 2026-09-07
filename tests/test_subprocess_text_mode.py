"""Every text-mode subprocess call names its codec explicitly.

`text=True` (or `universal_newlines=True`) with no `encoding=` decodes the child's
output using the PARENT's preferred encoding. Under `LC_ALL=C` with coercion
disabled -- an ordinary cron or batch environment, and the default on older cluster
images -- that is `ANSI_X3.4-1968`, pure ASCII, and one accented byte anywhere in
the output ends the read. The bytes are not ours to predict: job names, a node's
`Reason`, fileset names and account descriptions are free text somebody typed.

Three sibling packages already passed `encoding="utf-8", errors="replace"` on their
scheduler runners, and three call sites did not -- one of them the single funnel
every Slurm query in its package went through. That is the shape this asserts
against: not "is this one line right" but "did a new call site skip the rule".

Scanned from the AST rather than by importing, so a site that is only reached on
another platform still counts.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

#: Calls that hand back str rather than bytes.
_TEXT_KWARGS = ("text", "universal_newlines")

#: `asyncio.create_subprocess_*` has no text mode at all -- it is bytes-only, and
#: the caller decodes explicitly -- so it is not in scope here.
_SPAWNERS = (
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.check_output",
    "subprocess.call",
    "subprocess.check_call",
)


def _text_mode_calls() -> list[tuple[str, int, set[str]]]:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = ast.unparse(node.func)
            if not any(
                target.endswith(s.split(".")[-1]) and "subprocess" in target for s in _SPAWNERS
            ):
                continue
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
            textish = any(kw.get(name) == "True" for name in _TEXT_KWARGS)
            if textish or "encoding" in kw:
                rel = str(path.relative_to(SRC))
                found.append((rel, node.lineno, set(kw)))
    return found


def test_there_is_at_least_one_call_to_check() -> None:
    # Guards against the scan silently matching nothing, which would make every
    # assertion below vacuously true.
    assert _text_mode_calls(), (
        "found no text-mode subprocess calls; either the package stopped shelling "
        "out or this scan no longer recognises how it does"
    )


@pytest.mark.parametrize("site", _text_mode_calls(), ids=lambda s: f"{s[0]}:{s[1]}")
def test_a_text_mode_call_names_its_encoding_and_error_handler(
    site: tuple[str, int, set[str]],
) -> None:
    path, line, kwargs = site
    missing = [k for k in ("encoding", "errors") if k not in kwargs]
    assert not missing, (
        f"{path}:{line} decodes the child's output in text mode without {missing}. "
        f'Pass encoding="utf-8", errors="replace": without them the decode uses '
        f"the parent's locale, which is ASCII under LC_ALL=C, and a single accented "
        f"byte from the scheduler raises UnicodeDecodeError."
    )
