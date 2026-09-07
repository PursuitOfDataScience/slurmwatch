"""The ``B`` tier prints whole bytes, like both sibling tools.

``units.format_bytes`` applied ``:.1f`` to every tier, so a byte count came out
as "512.0 B" and an empty file as "0.0 B" -- a tenth of a byte is not a quantity.
Both siblings spell it bare and one of them says why:
``slurmpast.duration.format_bytes`` returns ``"%d B" % int(value)`` and its
comment notes the promotion bound "leaves '1000 B' and '1023 B' exactly", and
``rapidu.fmt.human_bytes`` agrees. slurmwatch was the only one of the three
printing a decimal there.

Reachable, not theoretical: the log viewer's status bar prints ``chunk.size`` for
any file with ``size >= 0``, so a job's ``.out`` before its first write read
"0.0 B", and ``headroom``/``cache_bytes``/a memory ``gap`` can all land under a
kibibyte too.

Truncated rather than rounded, and that is load-bearing rather than taste:
``%.0f`` of 1023.6 is "1024 B" -- one kibibyte spelled in the unit below it,
which is the A5 defect the tier test exists to prevent, reintroduced at the
bottom of the ladder by the fix for the top of it.
"""

import pytest

from slurmwatch.units import format_bytes, mem_figure, mem_pair, mem_scale


def test_a_byte_count_has_no_fractional_part() -> None:
    assert format_bytes(0) == "0 B"
    assert format_bytes(1) == "1 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1023) == "1023 B"


def test_the_b_tier_can_never_print_1024_b() -> None:
    """The reason for ``int()``. Rounding would spell a kibibyte as "1024 B"."""
    assert format_bytes(1023.4) == "1023 B"
    assert format_bytes(1023.6) == "1023 B", "rounding here would read '1024 B'"
    assert format_bytes(1023.9) == "1023 B"
    # and the KiB tier still claims anything that would round to 1.0 KiB
    assert format_bytes(1023.95) == "1.0 KiB"
    assert format_bytes(1024) == "1.0 KiB"


def test_it_now_agrees_with_both_siblings() -> None:
    """The three tools format the same quantity and must not disagree about it.

    Skipped rather than failed where the siblings are not installed: this package
    ships on its own and must not acquire a test dependency on the other two.
    """
    human_bytes = pytest.importorskip("rapidu.fmt").human_bytes
    slurmpast_bytes = pytest.importorskip("slurmpast.duration").format_bytes

    for value in (0, 1, 512, 1000, 1023, 1024, 1024**2, 1024**3):
        mine = format_bytes(float(value))
        assert mine == slurmpast_bytes(value), (value, mine)
        assert mine == human_bytes(value), (value, mine)


def test_control_every_tier_above_bytes_is_unchanged() -> None:
    """CONTROL, passing with the change present or absent.

    Only the ``B`` tier moved. One decimal everywhere else, including the top
    tier's fall-through, is what the rest of the dashboard is built on.
    """
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(1024**2) == "1.0 MiB"
    assert format_bytes(1024**3) == "1.0 GiB"
    assert format_bytes(1024**4) == "1.0 TiB"
    assert format_bytes(1024**5) == "1.0 PiB"
    assert format_bytes(1024**6) == "1024.0 PiB"
    assert format_bytes(1536) == "1.5 KiB"
    assert format_bytes(1610612736) == "1.5 GiB"


def test_control_the_a5_promotion_rule_still_holds() -> None:
    """CONTROL, in both states. The defect this function was last fixed for.

    A value just under a power of 1024 must promote rather than print
    "1024.0 MiB". Asserted at both the boundaries A5 named.
    """
    assert format_bytes(1073741800) == "1.0 GiB"
    assert format_bytes(1099460000000) == "1.0 TiB"


# ---------------------------------------------------------------------------
# `mem_figure` renders the same tier and kept the decimal this module dropped
# ---------------------------------------------------------------------------


def test_mem_figure_agrees_with_format_bytes_in_the_b_tier() -> None:
    """Two functions, one module, one tier -- they must not disagree about it.

    ``mem_figure`` has its own ``:.1f``/``:.0f`` branch for values at or above
    one unit of the limit's scale. In the ``B`` tier that scale is 1.0, so a
    single-digit figure took the decimal path and read "5.0 B" after
    ``format_bytes`` had stopped doing exactly that. ``mem_pair`` then produced
    ``('5.0', '512 B')``.

    HARDENING rather than a user-visible fix, and the measurement says so:
    reaching the ``B`` tier needs ``mem_scale`` to see a limit of 1-1023 bytes,
    which Slurm's ``--mem`` cannot express, or a live process under 10 bytes via
    tui.py's scale-by-working-set. Both callers already guard a zero limit. It is
    closed because one rule with one home beats two that happen to agree.
    """
    unit, size = mem_scale(1023.0)
    assert (unit, size) == ("B", 1.0), "otherwise this is not the tier under test"
    for value in (0, 1, 5, 9, 9.5, 10, 512, 1023):
        assert mem_figure(float(value), unit, size) == format_bytes(float(value)), value


def test_the_pair_no_longer_shows_a_fractional_byte() -> None:
    """What a reader would have seen: "5.0 / 512 B"."""
    assert mem_pair(5.0, 512.0) == ("5", "512 B")
    assert mem_pair(9.0, 100.0) == ("9", "100 B")


def test_control_the_scaled_tiers_keep_their_decimal() -> None:
    """CONTROL, passing with the change present or absent.

    Only the ``B`` tier lost its decimal. A single-digit GiB figure keeps one --
    that is the whole point of the ``< 10`` branch -- and a two-digit one does
    not. A fix that dropped the decimal everywhere would pass the tests above
    and fail this one.
    """
    gib = float(1024**3)
    assert mem_figure(5.5 * gib, "GiB", gib) == "5.5 GiB"
    assert mem_figure(26 * gib, "GiB", gib) == "26 GiB"
    assert mem_figure(51 * gib, "GiB", gib) == "51 GiB"


def test_control_the_sw4_path_still_keeps_its_own_unit() -> None:
    """CONTROL, in both states. The defect ``mem_figure`` exists for.

    A ``--mem=200M`` job at 18% rendered ``0 / 0 GiB``. A figure below one unit
    of the limit's scale must keep its own unit instead of rounding away, and
    that branch is reached before the one this change touched.
    """
    gib = float(1024**3)
    assert mem_figure(20 * 1024**2, "GiB", gib) == "20.0 MiB"
    assert mem_pair(200 * 1024**2, 64 * gib) == ("200.0 MiB", "64 GiB")


def test_control_the_familiar_shared_unit_pair_is_untouched() -> None:
    """CONTROL, in both states. ``26 / 51 GiB`` is what most jobs show."""
    gib = float(1024**3)
    assert mem_pair(26 * gib, 51 * gib) == ("26", "51 GiB")
