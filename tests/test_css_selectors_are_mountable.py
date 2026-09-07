"""Every widget-class selector in the app's CSS is a class the app can mount.

`issues.md` D19: the dashboard carried ten lines of CSS styling Textual's stock
`Footer` and `FooterKey`, which this package never mounts -- the keybinding bar is
its own `KeyFooter(Static)`, "precisely because the stock Footer paints every key
the same accent". Dead CSS is invisible: Textual does not warn about a selector that
matches nothing, so it reads as working styling for a widget that isn't there.

The same D19 noted the `textual>=0.86` floor comment claimed the floor "covers every
widget the dashboard uses (Sparkline, DataTable, Digits, Rule, Header/Footer)". Three
of those five are not imported anywhere in the package. The floor's real reason -- the
theme system -- was already the comment's first sentence.

Both are pinned here: the CSS against what the module can mount, and the comment
against what the module imports.
"""

import ast
import pathlib
import re

from textual.widgets import Static

import slurmwatch.tui as tui_mod

SRC = pathlib.Path(tui_mod.__file__).resolve()
ROOT = SRC.parent.parent.parent
PYPROJECT = ROOT / "pyproject.toml"

#: Textual resolves this without any import or local class.
CSS_BUILTINS = {"Screen"}


def _tree() -> ast.Module:
    return ast.parse(SRC.read_text())


def css_blobs(tree: ast.Module) -> list[tuple[str, str]]:
    """Every string assigned to a ``CSS``-ish name, as (name, text)."""
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if not (name and "CSS" in name.upper()):
                    continue
                blob = node.value
                # A CSS attribute must be a plain string literal; anything else
                # (an f-string, a join) is not something this scan can read, and
                # mypy is right to refuse the unchecked `.value`.
                if isinstance(blob, ast.Constant) and isinstance(blob.value, str):
                    out.append((name, blob.value))
    return out


def css_type_selectors(tree: ast.Module) -> set[str]:
    """Bare ``CapitalisedName`` selectors, which is what a widget TYPE looks like.

    Ids (`#foo`), classes (`.bar`) and pseudo-classes are excluded by the lookbehind;
    comments are stripped first so a widget named only in prose is not counted.
    """
    found: set[str] = set()
    for _name, blob in css_blobs(tree):
        blob = re.sub(r"/\*.*?\*/", " ", blob, flags=re.S)
        for line in blob.splitlines():
            head = line.split("{")[0]
            found.update(re.findall(r"(?<![.#\w-])([A-Z][A-Za-z0-9_]*)", head))
    return found


def mountable_names(tree: ast.Module) -> set[str]:
    """Widget names this module could actually put on screen."""
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("textual"):
            imported |= {a.asname or a.name for a in node.names}
    local = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    return imported | local | CSS_BUILTINS


class TestTheCssStylesOnlyWidgetsThatExist:
    def test_no_selector_is_unmountable(self) -> None:
        tree = _tree()
        orphans = sorted(css_type_selectors(tree) - mountable_names(tree))
        assert orphans == [], (
            f"CSS styles widget classes this module never mounts: {orphans} -- Textual "
            "does not warn about a selector that matches nothing, so it reads as live "
            "styling (issues.md D19)"
        )

    def test_the_scan_found_selectors_to_check(self) -> None:
        # Guards against passing because the regex matched nothing at all.
        found = css_type_selectors(_tree())
        assert len(found) >= 8, sorted(found)
        assert "Screen" in found, sorted(found)


class TestTheDependencyCommentNamesOnlyWhatIsUsed:
    def test_no_widget_is_claimed_that_the_package_never_imports(self) -> None:
        tree = _tree()
        usable = mountable_names(tree)
        comment = "\n".join(
            ln for ln in PYPROJECT.read_text().splitlines() if ln.strip().startswith("#")
        )
        # Only the names the comment presents as widgets the dashboard uses.
        claimed = {"Sparkline", "DataTable", "Digits", "Rule", "Header", "Footer"}
        wrong = sorted(w for w in claimed if w not in usable and re.search(rf"\b{w}\b", comment))
        assert wrong == [], (
            f"pyproject's textual floor comment cites widgets nothing imports: {wrong}"
        )


class TestControls:
    """Independent of the removed CSS and the reworded comment."""

    def test_the_key_bar_is_this_packages_own_widget(self) -> None:
        # The reason the stock Footer CSS was dead. True before and after.
        assert issubclass(tui_mod.KeyFooter, Static)

    def test_the_widgets_the_dashboard_really_mounts_are_importable(self) -> None:
        for name in ("Digits", "Header", "ListItem", "ListView", "RichLog", "Static"):
            assert hasattr(tui_mod, name), name

    def test_the_theme_is_registered_which_is_what_the_floor_is_for(self) -> None:
        # The floor's actual reason, asserted against the code rather than prose.
        assert "register_theme" in SRC.read_text()

    def test_the_css_is_not_empty(self) -> None:
        assert sum(len(b.splitlines()) for _n, b in css_blobs(_tree())) > 50
