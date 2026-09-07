"""Byte/size rendering shared by every surface that shows a memory figure.

The same numbers reach the dashboard gauge, the memory drill-in and the plain-text
summary the degraded (sstat) path prints, and they were formatted independently —
so a fix to one left the others rounding a 400 MiB limit to ``0.4 GiB`` or ``0 / 0
GiB``. One implementation, imported by both renderers, is what keeps them honest
together (SW-4, and its second sighting in the summary renderer).
"""

from __future__ import annotations

_MEM_SCALES = (
    ("TiB", float(1024**4)),
    ("GiB", float(1024**3)),
    ("MiB", float(1024**2)),
    ("KiB", 1024.0),
)


def format_bytes(n: float) -> str:
    """A byte count in the largest unit that keeps it above 1, one decimal.

    **Except the ``B`` tier, which has no fraction to report.** A byte is the
    unit of account here; "512.0 B" claims a precision that does not exist, and
    both sibling tools spell it bare -- ``slurmpast.duration.format_bytes``
    returns ``"%d B" % int(value)`` and says in as many words that its bound
    "leaves '1000 B' and '1023 B' exactly", and ``rapidu.fmt.human_bytes`` agrees.
    This was the only one of the three printing a decimal, and it is reachable:
    the log viewer's status bar prints ``chunk.size`` for any file with
    ``size >= 0``, so a job's `.out` before its first write read ``0.0 B``.

    Truncated rather than rounded, which is the same choice ``slurmpast`` makes
    and it is load-bearing: ``%.0f`` of 1023.6 is "1024 B", i.e. one kibibyte
    spelled in the unit below it -- the A5 defect described below, reintroduced at
    the bottom of the ladder by the fix for the top of it. ``int()`` cannot
    produce 1024 from a value the KiB tier did not already claim.
    """
    # Compare the ROUNDED value against 1024 so a number just under a boundary
    # promotes to the next unit instead of printing "1024.0 MiB": e.g. 1073741800
    # is < 1024 MiB but rounds to 1024.0 at one decimal, so it must read 1.0 GiB (A5).
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if round(abs(n), 1) < 1024.0:
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def pct_text(percent: float) -> str:
    """A magnitude percentage that must not claim a boundary it has not reached.

    ``:.0f`` reaches ``100`` from 99.5 up, so every gauge drawn with it -- CPU
    used, memory used, GPU compute, VRAM, and the plain report's ``peak x / y
    (n%)`` -- printed ``100%`` for a job with headroom left. Measured on the
    dashboard renderer before this existed: **99.6% came out ``████████ 100%``**,
    a solid bar and a boundary claim, where 99.4% correctly read ``███████▉ 99%``.

    It matters for the same reason it mattered for the elapsed figure, which
    ``tui._time_frac_text`` already fixed one gauge over -- and that function cites
    the family's spelling for it: ``rapidu.fmt.ratio_x`` returns ``<0.01x``,
    ``nodetop.core.duration`` returns ``<1m``. The memory figure is what the
    off-node OOM guard fires on and what a reader consults before raising
    ``--mem``, so "at your limit" and "a fraction under it" are not one sentence.

    It lives here rather than in either renderer because BOTH draw this figure and
    a second copy is how the two surfaces come to disagree -- the reason
    ``mem_pair`` is in this module too.

    Only the TOP boundary moves. A sub-0.5% value keeps ``0%``: the bar beside it
    is empty and the two are deliberately kept in step, and "0%" for a nearly idle
    job is the reading its own underuse advisory gives. At or above 100 is
    untouched -- a job over its request really is at 100% of it -- and so is
    anything negative, which a clock-skewed reading can be.

    Four characters either way (``100%`` / ``>99%``), which is what the gauge's
    ``:>4`` slot allows.
    """
    if percent < 100.0 and round(percent) >= 100:
        return ">99%"
    return f"{percent:.0f}%"


def format_cores(n: float) -> str:
    """Cores busy without a pointless trailing ``.0`` (``1.0`` -> ``1``, ``2.8`` -> ``2.8``).

    Here rather than in `tui`, because BOTH surfaces render this figure and they
    rendered it differently. `tui` had this rule; `cli` wrote ``:.1f``, so one
    measurement reached the reader two ways in the same sentence -- the dashboard
    said ``only ~1 of 8 cores are doing work`` and the plain summary said
    ``only ~1.0 of 8 cores are doing work``, with the ADVICE half already shared
    from `model.CPU_UNDERUSE_ADVICE` for exactly this reason. Measured on one
    snapshot with `effective_cores == 1.0`, both surfaces driven at once.

    An integer is the common case, not a corner: a single-threaded process on an
    8-core allocation reads exactly 1.0, and a saturating one reads exactly the
    core count. The trailing ``.0`` is the whole difference, which is why nothing
    caught it -- each surface had a test pinning its own spelling.
    """
    return f"{n:.1f}".rstrip("0").rstrip(".")


def mem_scale(limit_bytes: float) -> tuple[str, float]:
    """The unit a memory LIMIT reads naturally in — the one ``--mem`` was written in."""
    for unit, size in _MEM_SCALES:
        if abs(limit_bytes) >= size:
            return unit, size
    return "B", 1.0


def mem_figure(n: float, unit: str, size: float) -> str:
    """``n`` rendered in ``unit``, or in its OWN unit when it would round to nothing.

    A hard ``:.0f`` of GiB is right for the tens-of-GiB jobs a GPU cluster runs and
    silently wrong for everything under half a GiB: a ``--mem=200M`` job at 18%
    rendered ``0 / 0 GiB`` — a zero-byte limit, zero used — next to a bar reading
    18%, and every array task / preprocessing / eval job on a cluster lives down
    there. So a figure below one unit of the limit's scale keeps its own unit
    instead of rounding away, and a single-digit one keeps a decimal. SW-4.
    """
    if abs(n) < size:
        return format_bytes(n)
    if unit == "B":
        # Delegated rather than re-derived, so the two cannot disagree about a
        # tier they both render. `format_bytes` stopped putting a decimal on a
        # byte count -- a tenth of a byte is not a quantity, and both sibling
        # tools spell it bare -- and this branch kept doing it: a single-digit
        # figure took the `:.1f` path below and read "5.0 B", so `mem_pair`
        # produced "5.0 / 512 B".
        #
        # HARDENING, not a user-visible fix, and measured as such: reaching it
        # needs `mem_scale` to return the B tier, i.e. a memory limit of 1-1023
        # bytes -- which Slurm's `--mem` cannot express -- or, via tui.py's
        # scale-by-working-set, a live process under 10 bytes. Both callers
        # already guard a zero limit (`cli.py` on `limit_bytes > 0`, `tui.py` by
        # blanking the pair), so the wrong output was computed and discarded.
        # Closed anyway: one rule with one home beats two that agree today.
        return format_bytes(n)
    scaled = n / size
    return f"{scaled:.0f} {unit}" if abs(scaled) >= 10 else f"{scaled:.1f} {unit}"


def mem_pair(used: float, limit: float) -> tuple[str, str]:
    """``(used, limit)`` texts for a "N / M GiB" pair, sharing one unit when they can.

    The unit is printed once, on the limit, while both sides are in it — so the
    familiar ``26 / 51 GiB`` is untouched — and on both sides when the used figure
    had to keep its own to stay non-zero (``20.0 MiB / 64 GiB``)."""
    unit, size = mem_scale(limit)
    used_txt = mem_figure(used, unit, size)
    limit_txt = mem_figure(limit, unit, size)
    if used_txt.endswith(f" {unit}"):
        used_txt = used_txt[: -len(unit) - 1]
    return used_txt, limit_txt


def per_node_suffix(node_count: int) -> str:
    """``"/node"`` on a multi-node job, ``""`` on a single-node one.

    ``cpus_allocated`` / ``mem_limit_bytes`` are PER-NODE (slurm.py scales the
    job-wide NumCPUs and memory down to one node, since slurmwatch watches a single
    node), so a multi-node allocation has to say so or "4 nodes  16 CPU" reads as the
    whole-job total (N8). Shared because two renderers show this pair — the foreign
    job card and the plain-text foreign summary — and the summary was missing the
    figures entirely while the card had them.
    """
    return "/node" if node_count > 1 else ""


def fabric_rate_text(rate_label: str, kind: str) -> str:
    """The port's rate, with an InfiniBand speed grade dropped on an Ethernet link.

    ``rate_label`` is the HCA's own words, read straight from
    ``/sys/class/infiniband/<dev>/ports/<n>/rate``, and on RoCE the mlx5 driver fills
    that file with an INFINIBAND grade regardless of the link layer: a 25 GbE port on
    a real RoCE cluster reports ``25 Gb/sec (1X EDR)``. Verified against the kernel
    there — the string is genuinely the driver's, not ours — but printing it beside
    "inter-node RoCE" tells a reader their Ethernet is EDR InfiniBand, which is a
    speed grade that does not apply to it. The bandwidth is the part that means
    something on either fabric, so keep it and drop the parenthetical when the link
    is not InfiniBand. The raw field is left untouched for machine consumers.
    """
    label = (rate_label or "").strip()
    if not label or kind.lower() == "infiniband":
        return label
    head, _, rest = label.partition("(")
    return head.strip() if rest else label


def printable_text(text: str) -> str:
    """Render control characters visibly instead of letting them execute.

    A job name is arbitrary user text — this package already treats it as untrusted
    twice over: ``_csv_text`` prefixes a quote so a spreadsheet cannot evaluate
    ``=cmd|"/bin/sh"!A1`` as a formula, and ``_escape_markup`` neutralizes ``[`` so
    Textual's parser cannot be steered by it. The TERMINAL is the third interpreter of
    that same field, and it was the one left unguarded: ``sbatch -J $'\\e[2J'`` puts a
    clear-screen sequence in the name, which `cat`ing the CSV log executes, and which
    Rich passes through to the dashboard verbatim (measured — Rich strips CR but not
    ESC). A bare ``\r`` mid-field has the milder version of the same effect on
    ``awk``/``cut`` output, which is the audience SW-31's line endings were fixed for.

    Escaped rather than dropped, for the reason the encoding fallback is:
    ``train\x1b[2J`` still tells the reader what the job was called and what was in
    the name. Silently deleting bytes from an identity field would make two different
    jobs look like the same one.

    Legitimate non-ASCII is preserved — ``str.isprintable()`` is true for ``é`` and
    ``中`` — so this does not mangle a name in a language other than English.
    """
    return "".join(c if c.isprintable() or c == " " else repr(c)[1:-1] for c in text)
