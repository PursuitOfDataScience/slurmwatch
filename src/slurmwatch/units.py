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
    """A byte count in the largest unit that keeps it above 1, one decimal."""
    # Compare the ROUNDED value against 1024 so a number just under a boundary
    # promotes to the next unit instead of printing "1024.0 MiB": e.g. 1073741800
    # is < 1024 MiB but rounds to 1024.0 at one decimal, so it must read 1.0 GiB (A5).
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if round(abs(n), 1) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


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
