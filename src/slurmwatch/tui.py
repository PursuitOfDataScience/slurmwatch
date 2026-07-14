from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections import deque
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import DataTable, Digits, Header, ListItem, ListView, Static

from .collector import TelemetryCollector, _gpu_is_active
from .config import SlurmwatchConfig
from .exceptions import JobNotPendingError
from .model import (
    CpuMetrics,
    GpuMetrics,
    JobContext,
    MemoryMetrics,
    TelemetrySnapshot,
    local_node_name,
    short_host,
)
from .pending import (
    PartitionResources,
    PendingJob,
    available_node_count,
    explain_reason,
    format_gpu_types,
    is_held_like,
    partition_fits_now,
    requeue_could_help,
    resolve_cluster_partitions,
    resolve_pending_job,
    resolve_priority_rank,
    resolve_queue_counts,
)
from .remote import open_stream, parse_snapshot_line
from .slurm import resolve_current_jobs, resolve_job_context


def _format_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def _gib(n: float) -> float:
    return n / 1024**3


def _format_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_cores(n: float) -> str:
    """Cores busy without a pointless trailing '.0' (``1.0`` → ``1``, ``2.8`` → ``2.8``)."""
    return f"{n:.1f}".rstrip("0").rstrip(".")


def _format_wait(seconds: int) -> str:
    """A compact queue-wait duration: ``45s`` / ``3m`` / ``1h 5m`` / ``2d 3h``."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        h, m = divmod(seconds, 3600)
        return f"{h}h {m // 60}m"
    d, rem = divmod(seconds, 86400)
    return f"{d}d {rem // 3600}h"


def _format_clock(ts: float) -> str:
    """A wall-clock timestamp as ``Jul 08 09:38`` in local time."""
    return time.strftime("%b %d %H:%M", time.localtime(ts))


def _shorten_path(path: str, budget: int, keep: int = 2, ell: str = "…") -> str:
    """Fit a long filesystem path into ~``budget`` columns so it reads on one line
    instead of wrapping mid-word.

    Keeps the two things worth reading — the root (for orientation) and the last
    ``keep`` components (the file / leaf you actually care about) — and elides the
    noisy middle with ``…``. A home-relative path is collapsed to ``~`` first. A
    path (or command) already within budget is returned unchanged, so short values
    are never mangled. Examples (budget ~40)::

        /project/rcc/.../scratchpad/run.sbatch  ->  /project/…/scratchpad/run.sbatch
        /project/rcc/.../abc123.../scratchpad   ->  /project/…/scratchpad   (keep=1)
    """
    home = os.path.expanduser("~")
    if home and (path == home or path.startswith(home + os.sep)):
        path = "~" + path[len(home) :]
    if budget <= 3 or len(path) <= budget:
        return path
    parts = [p for p in path.split("/") if p]
    if not parts:
        return path
    anchor = "/" if path.startswith("/") else ("~/" if path.startswith("~/") else "")
    first = parts[0] if parts[0] != "~" else (parts[1] if len(parts) > 1 else "")
    for k in range(min(keep, len(parts)), 0, -1):
        tail = "/".join(parts[-k:])
        candidates = (
            f"{anchor}{first}/{ell}/{tail}" if first and first not in parts[-k:] else None,
            f"{anchor}{ell}/{tail}",
        )
        for cand in candidates:
            if cand and len(cand) <= budget:
                return cand
    # Even the bare filename overflows: truncate its end, keeping the extension side.
    name = parts[-1]
    keepn = max(1, budget - len(ell))
    return ell + name[-keepn:]


# A non-breaking space renders like a normal space in the terminal but is not
# collapsed when the TUI is captured to SVG for the README demo, so separators
# that sit right next to Rich markup keep their intended gap.
_NBSP = "\N{NO-BREAK SPACE}"

# Warm "Claude Code" palette. Chrome (borders, titles, section headings) is the
# coral accent; each resource *block* carries its own identity hue on its bar and
# label, so blocks read as distinct at a glance. Health stays a separate channel
# (the status dot's colour), so a block's colour never has to do double duty. The
# block trio (deep cyan / rose / violet) is deeper and more saturated than the
# old pastels so it pops off the dark card instead of washing into it; validated
# against the lifted card (#262320): each clears >=4:1 contrast, worst adjacent
# CVD ΔE is 12.9 (protanopia) — up from 11.1 for the old washed trio, whose CPU
# cyan sat at the grey chroma floor — and each stays well clear of both the coral
# chrome and every health colour, so a block hue can't be mistaken for either.
_INK = "#ede7dd"  # primary text (warm off-white)
_DIM = "#b3a998"  # secondary text (warm grey, ~7:1 on the card)
_FAINT = "#857d70"  # faint text / the empty portion of a bar track (a visible groove)
_ACCENT = "#d97757"  # coral — the one chrome accent
_BG = "#141312"  # the darkest plane / dark ink on a coloured key cap


def _sep(ascii_mode: bool = False) -> str:
    """A ``·`` field separator. ASCII ``-`` when the middle dot can't render, so
    ``--ascii`` / a non-UTF-8 terminal never leaks a stray Unicode glyph."""
    return f"[{_FAINT}] {'-' if ascii_mode else '·'} [/]"


def _pack_chips(chips: list[str], sep: str, width: int) -> str:
    """Join ``chips`` with ``sep``, wrapping only *between* chips (never inside
    one), so a labelled value is never split from its label across a line break.

    Rich's word-wrap breaks at any space — including the space inside a
    ``label value`` chip and even a non-breaking space (it treats U+00A0 as
    whitespace) — so on a narrow terminal a bare ``sep``-joined strip orphans
    values onto the next line ("partition" on one row, its value on the next).
    Packing at chip granularity, measuring each chip's rendered width, keeps every
    chip intact; a single chip wider than ``width`` still wraps (unavoidable).
    """
    if width <= 0:
        return sep.join(chips)
    sep_w = Text.from_markup(sep).cell_len
    lines: list[str] = []
    cur, cur_w = "", 0
    for chip in chips:
        w = Text.from_markup(chip).cell_len
        if cur and cur_w + sep_w + w > width:
            lines.append(cur)
            cur, cur_w = chip, w
        elif cur:
            cur, cur_w = f"{cur}{sep}{chip}", cur_w + sep_w + w
        else:
            cur, cur_w = chip, w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# Per-block identity hues: deep cyan / rose / violet — deliberately spread across
# the wheel (and away from the coral chrome accent) so no two blocks read as the
# same colour, even on a 256-colour terminal or with red-green colour-blindness.
# Deeper and more saturated than the old pastels (the old CPU cyan sat on the grey
# chroma floor), so they pop off the dark card. Keyed by the row label so a
# resource's bar and label share a hue.
_CPU_COLOR = "#159fc0"  # deep cyan
_MEM_COLOR = "#df5f97"  # rose
_GPU_COLOR = "#8a6ee6"  # violet (GPU identity: label + compute bar)
# The GPU block shows two bars (compute + vram). They belong to one block so they
# stay in the violet family, but a lighter lilac shade for vram makes the two bars
# distinguishable at a glance instead of two identical stacked bars.
_GPU_VRAM_COLOR = "#cbb8f5"  # pale lilac

# When several GPUs are shown together in the device table, each device gets its
# own colour so identical-looking rows (a job saturating every GPU) are still easy
# to tell apart. Eight CVD-distinct hues on a dark surface is over-constrained (the
# old all-green tail collapsed to ΔE 4.3 under deuteranopia), so identity is
# encoded by hue AND lightness: four base hues, each a bright and a deep shade.
# Validated: worst all-pairs ΔE 12.0. The explicit GPU-index cell is the primary
# way rows are told apart; colour is a strong secondary aid.
_GPU_CYCLE = [
    "#a98ff0",  # violet bright
    "#7658d8",  # violet deep
    "#3fc9d6",  # teal bright
    "#1c8a97",  # teal deep
    "#e6b24a",  # amber bright
    "#b07d1e",  # amber deep
    "#ef8fc0",  # pink bright
    "#c04f86",  # pink deep
]


def _gpu_device_color(index: int) -> str:
    return _GPU_CYCLE[index % len(_GPU_CYCLE)]


# One health vocabulary, everywhere: green = fine, amber = warning/underused,
# red = critical/idle. Kept well clear of every block hue (the closest pair, MEM
# rose ↔ warn amber, is ΔE 36) so a status colour never impersonates a block hue.
_HEALTH_COLOR = {"ok": "#6aa84f", "warn": "#e2bb4c", "crit": "#d1584f", "none": _FAINT}
_HEALTH_GLYPH = {"ok": "●", "warn": "●", "crit": "●", "none": "·"}
_HEALTH_GLYPH_ASCII = {"ok": "+", "warn": "!", "crit": "x", "none": "-"}

# The resource-row marker is DECORATIVE: a bullet in the resource's own hue
# (matching its label), never a health grade. slurmwatch reports facts — the bar,
# the %, the recent range, the temperature — and lets the reader decide whether
# the job is running well; it deliberately does NOT colour a row green/amber/red
# to assert a verdict (two resources that happened to share a grade then wore the
# same colour, which read as "these are the same" and imposed a judgement).
_MARKER = "●"
_MARKER_ASCII = "*"

# The warm "Claude Code" theme: a warm near-black surface with the coral accent,
# replacing the cold blue of an off-the-shelf theme. $primary drives the chrome
# (Header bar, box borders, Rule); success/warning/error mirror the health
# vocabulary so themed widgets agree with our hand-drawn ones. Three distinct
# planes give real elevation (the old three sat within ~10 sRGB units and read as
# one flat slab): a deep page behind a slightly-lifted screen behind the lifted
# cards / bottom bar.
_CLAUDE_THEME = Theme(
    name="slurmwatch",
    primary=_ACCENT,
    secondary=_MEM_COLOR,
    accent=_GPU_COLOR,
    foreground=_INK,
    background="#141312",  # deepest page plane
    surface="#1e1c1b",  # the screen
    panel="#262320",  # lifted card / bottom bar
    success=_HEALTH_COLOR["ok"],
    warning=_HEALTH_COLOR["warn"],
    error=_HEALTH_COLOR["crit"],
    dark=True,
)

# GPU temperature threshold: above this a device is thermally stressed.
_TEMP_HOT_C = 83.0

_BAR_W = 18
_SPARK_W = 12
# Below this width the bars narrow so the essentials still fit an 80-column
# SSH terminal.
_NARROW_COLS = 100
# At/above this width a GPU's compute and vram bars ride a single line (one row
# per device) instead of stacking; below it they stack so nothing wraps.
_GPU_MERGE_COLS = 120
# At/above this terminal width the GPU drill-in table can fit its per-device TREND
# sparkline column without pushing STATUS off-screen behind a horizontal scroll;
# below it the sparkline is dropped so the essentials (incl. health) stay visible.
_GPU_TREND_MIN_COLS = 108

# TRENDS "steady" threshold: a series whose 60s range spans fewer than this many
# points is labelled "steady" instead of an "X–Y%" range. This only controls the
# text tag — the bar's *length* always reflects the real level, so two different
# values (e.g. 0% and 3%) never look identical.
_TREND_STEADY_SPAN = 1.0

# Node-switch feedback timing. A switch normally lands in ~1.5-3s (a Slurm step
# launch on the target node); after _SWITCH_SLOW_S the banner adds a reassuring
# "this can take a moment" note, and after _SWITCH_STUCK_S it stops blocking (the
# body un-dims and the banner turns into an amber "still reaching / retrying"
# warning) so an unreachable node never freezes the session — the poll loop keeps
# retrying, and a frame that finally arrives clears it normally.
_SWITCH_SLOW_S = 4.0
_SWITCH_STUCK_S = 12.0

# Below this many terminal rows the docked bottom bar collapses to a single line
# (drops the time-budget line, blank padding, and border) so the primary RESOURCES
# gauges keep their rows instead of scrolling below the fold on a small split-pane.
_COMPACT_HEIGHT = 20

# Spinner frames (braille at ~8fps, |/-\ under --ascii) — reused for the pending
# view's "calculating…" estimate animation so it reads as actively working.
_SPIN_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_SPIN_FRAMES_ASCII = ("|", "/", "-", "\\")


def _glyph(level: str, ascii_mode: bool) -> str:
    table = _HEALTH_GLYPH_ASCII if ascii_mode else _HEALTH_GLYPH
    return table.get(level, table["none"])


def _dot(level: str, ascii_mode: bool) -> str:
    return f"[{_HEALTH_COLOR[level]}]{_glyph(level, ascii_mode)}[/]"


def _bar_cells(percent: float, width: int) -> int:
    """How many cells of a ``width``-wide magnitude bar are filled at ``percent``.

    The single source of truth for bar length, shared by the RESOURCES gauge
    (``_color_bar``) and the TRENDS bar (``_trend_bar``) so the *same* value can
    never render empty in one and filled in the other. We **round** to the
    nearest cell (not floor), and a value that *displays* as ≥1% keeps at least
    one filled cell — a visible sliver, even on a narrow bar — while a sub-0.5%
    value that shows as "0%" draws empty, so the bar always matches the whole
    percent printed beside it. (A bare floor with no minimum is what made a 4%
    row read as an empty ``░░░`` gauge next to its own "4%".)
    """
    percent = min(max(percent, 0.0), 100.0)
    n = min(width, round(percent / 100.0 * width))
    return max(1, n) if round(percent) >= 1 else n


def _color_bar(
    percent: float, length: int = _BAR_W, ascii_mode: bool = False, color: str = _ACCENT
) -> str:
    """A magnitude bar: filled portion in the block's identity ``color``.

    ``percent`` is clamped to [0, 100] so an over-limit value can't overflow the
    bar's width. The fill colour identifies which block the bar belongs to, not
    its health — only the fill *length* carries the magnitude; health lives in
    the status dot/word beside it. The empty track is a faint neutral.
    """
    filled = _bar_cells(percent, length)
    fill_ch, empty_ch = ("#", "-") if ascii_mode else ("█", "░")
    parts = []
    if filled:
        parts.append(f"[{color}]{fill_ch * filled}[/]")
    if length - filled:
        parts.append(f"[{_FAINT}]{empty_ch * (length - filled)}[/]")
    return "".join(parts)


def _render_sparkline(
    values: deque[float],
    length: int = _SPARK_W,
    ascii_mode: bool = False,
    stretch: bool = False,
    lo: float = 0.0,
    hi: float = 100.0,
) -> str:
    """A one-row block sparkline (▁▂▃▄▅▆▇█) of ``length`` cells.

    ``stretch`` spreads all available samples across the full width (oldest left,
    newest right) so a trend always fills the row instead of hugging the right
    edge while history is still filling; the default anchors the newest sample to
    the right edge and blank-pads the left (a fixed time-per-column).

    ``lo``/``hi`` set the value range mapped onto the glyph height. The default
    0–100 shows absolute magnitude; passing a series' own min/max auto-scales it
    so a small-but-real wiggle (e.g. 9–15%) becomes visible instead of a dead
    flat line at the bottom of the 0–100 scale.
    """
    chars = "▁▂▃▄▅▆▇█" if not ascii_mode else "_.,-=+#%"
    span = hi - lo if hi > lo else 1.0
    columns = _stretch_columns(values, length) if stretch else _sample_columns(values, length)
    cells: list[str] = []
    for v in columns:
        if v is None:
            cells.append(" ")
            continue
        frac = (v - lo) / span
        level = int(min(max(frac, 0.0), 1.0) * (len(chars) - 1))
        cells.append(chars[max(0, min(level, len(chars) - 1))])
    return "".join(cells)


# Filled-area chart glyphs: index 0 = empty, 1..8 = increasing fill (a partial top
# cell). Used by the resource drill-in's tall history graph.
_AREA_GLYPHS = " ▁▂▃▄▅▆▇█"
_AREA_GLYPHS_ASCII = " .:-=+*x#"


def _area_chart(
    values: deque[float],
    width: int,
    height: int,
    ascii_mode: bool = False,
    lo: float = 0.0,
    hi: float = 100.0,
) -> list[str]:
    """A filled area chart of ``values`` — ``height`` rows of exactly ``width`` cells.

    Oldest sample at the left, newest at the right (stretched to fill the width).
    Each column is a vertical bar rising from the baseline to its value on a fixed
    ``lo``–``hi`` scale (default 0–100, so the height honestly shows the level),
    with a partial top cell for sub-row precision — ``height`` rows give
    ``height × 8`` levels, so even small movements are visible. Returns the rows
    top-to-bottom; the caller adds the axis, colour, and labels.
    """
    glyphs = _AREA_GLYPHS_ASCII if ascii_mode else _AREA_GLYPHS
    width = max(1, width)
    height = max(1, height)
    span = hi - lo if hi > lo else 1.0
    columns = _stretch_columns(values, width)
    grid: list[list[str]] = [[] for _ in range(height)]
    for v in columns:
        if v is None:
            for r in range(height):
                grid[r].append(" ")
            continue
        frac = min(max((v - lo) / span, 0.0), 1.0)
        sub = frac * height * 8  # total eighth-cells filled from the bottom
        for r in range(height):
            base = (height - 1 - r) * 8  # eighth-cells below this row
            level = int(round(sub - base))
            level = 0 if level < 0 else 8 if level > 8 else level
            grid[r].append(glyphs[level])
    return ["".join(row) for row in grid]


def _labeled_bar(metric: str, percent: float, width: int, ascii_mode: bool, color: str) -> str:
    """``metric  ███░░   42%`` — a bar that says what it measures.

    Naming the quantity (dim) removes the "what is this number?" ambiguity a bare
    bar + percent creates when two different quantities share a line — e.g. a
    GPU's compute utilisation sitting next to its VRAM fill.
    """
    bar = _color_bar(percent, width, ascii_mode, color)
    return f"[{_DIM}]{metric:<7}[/] {bar} [{_INK}]{percent:>3.0f}%[/]"


def _sample_columns(values: deque[float], width: int) -> list[float | None]:
    """Down-sample a history deque to exactly ``width`` columns.

    The newest sample is anchored to the right edge; while history is still
    filling, the left is padded with ``None`` (rendered blank). Used by the
    one-row sparkline, where a fixed time-per-column reads naturally.
    """
    vals = list(values)
    if not vals:
        return [None] * width
    step = max(len(vals) / width, 1.0)
    out: list[float | None] = []
    for i in range(width):
        offset = int((width - 1 - i) * step)
        idx = len(vals) - 1 - offset
        out.append(vals[idx] if idx >= 0 else None)
    return out


def _stretch_columns(values: deque[float], width: int) -> list[float | None]:
    """Spread all available samples across the full ``width`` (oldest→newest).

    Unlike :func:`_sample_columns`, this fills the whole width even before history
    is full, so a trend never hugs the right edge with a blank left margin while
    it fills up — the oldest sample sits at the left, the newest at the right.
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return [None] * width
    if n == 1 or width == 1:
        return [vals[-1]] * width
    return [vals[round(i / (width - 1) * (n - 1))] for i in range(width)]


def _escape_markup(text: str) -> str:
    """Neutralize console-markup metacharacters in untrusted text.

    Textual's markup parser (unlike Rich's) treats a *lone/unclosed* ``[`` as a
    tag opener and raises ``MarkupError`` when the rest of the line fails to
    parse — so a job name like ``[experiment`` (``sbatch -J``) would crash the
    job selector. Neither ``rich.markup.escape`` nor ``textual.markup.escape``
    neutralizes an unclosed ``[``, so backslash-escape every ``[`` (after any
    literal backslash) — which both engines render as a literal ``[`` (F1).
    """
    return text.replace("\\", "\\\\").replace("[", "\\[")


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


# ---------------------------------------------------------------------------
# Health: one function per resource, returning (level, one-word status).
# ---------------------------------------------------------------------------


def _cpu_ratio(cpu: CpuMetrics) -> float:
    if cpu.cores_allocated <= 0:
        return 0.0
    return cpu.effective_cores / cpu.cores_allocated


def _cpu_health(cpu: CpuMetrics, underuse_threshold: float = 0.15) -> tuple[str, str]:
    if cpu.cores_allocated <= 0:
        return "none", "n/a"
    ratio = _cpu_ratio(cpu)
    # A single-core allocation can't be "underused"; for multi-core jobs, flag
    # underuse below the configurable ratio (SLURMWATCH_CPU_UNDERUSE, F4).
    if cpu.cores_allocated > 1 and ratio < underuse_threshold:
        return "warn", "underused"
    return "ok", "healthy"


def _mem_health(mem: MemoryMetrics) -> tuple[str, str]:
    if mem.oom_guard_critical:
        return "crit", "near limit"
    if mem.oom_guard_warning:
        return "warn", "high"
    return "ok", "healthy"


def _gpu_health(gpu: GpuMetrics, idle_threshold: float) -> tuple[str, str]:
    if not _gpu_is_active(gpu, idle_threshold):
        return "crit", "idle"
    if gpu.throttling:
        return "warn", "throttling"
    return "ok", "active"


# Slurm job-state → colour. This describes the state itself (a fact from Slurm),
# not a judgement of the user's choices: green while it runs / finishes, amber
# while it waits, red when it ended badly.
_STATE_OK = {"RUNNING", "COMPLETING", "COMPLETED"}
_STATE_WARN = {"PENDING", "CONFIGURING", "RESIZING", "REQUEUED", "SUSPENDED"}
_STATE_CRIT = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
}


def _job_state_color(state: str) -> str:
    s = state.upper()
    if s in _STATE_OK:
        return _HEALTH_COLOR["ok"]
    if s in _STATE_WARN:
        return _HEALTH_COLOR["warn"]
    if s in _STATE_CRIT:
        return _HEALTH_COLOR["crit"]
    return _INK


def _mem_ws_pct(mem: MemoryMetrics) -> float:
    ws = mem.working_set_bytes or mem.current_bytes
    return (ws / mem.limit_bytes * 100.0) if mem.limit_bytes > 0 else 0.0


def _banner_segments(snap: TelemetrySnapshot, config: SlurmwatchConfig) -> list[tuple[str, str]]:
    """The dashboard's headline issues, worst first (crit before warn).

    Returns (level, text) pairs. An empty list means everything is healthy and
    the caller renders the calm all-clear summary instead.
    """
    crit: list[tuple[str, str]] = []
    warn: list[tuple[str, str]] = []

    mem = snap.memory
    ws_pct = _mem_ws_pct(mem)
    # State the fact (how full memory is against its limit); the colour already
    # says crit/warn, so no "OOM RISK"/"APPROACHING LIMIT" verdict is needed.
    if mem.oom_guard_critical:
        crit.append(("crit", f"MEMORY {ws_pct:.0f}% of limit"))
    elif mem.oom_guard_warning:
        warn.append(("warn", f"MEMORY {ws_pct:.0f}% of limit"))

    gpus = snap.gpus
    idle_threshold = config.gpu_idle_threshold
    if gpus:
        active = [g for g in gpus if _gpu_is_active(g, idle_threshold)]
        idle = len(gpus) - len(active)
        total = len(gpus)
        # A GPU's throttle state only matters when the job is actually using it —
        # an idle GPU is reported idle, not *also* "throttling" (which on an idle
        # device is a neighbour's load on a shared card or a benign clocked-down
        # flag). Counting throttling only among active GPUs keeps a single GPU
        # from being flagged idle AND throttling at once, matching _gpu_health.
        throttling = sum(1 for g in active if g.throttling)
        if idle and idle == total:
            # Read naturally for one GPU ("GPU IDLE") vs. many ("ALL 4 GPUS IDLE").
            crit.append(("crit", "GPU IDLE" if total == 1 else f"ALL {total} GPUS IDLE"))
        elif idle:
            warn.append(("warn", f"{idle} OF {total} GPUS IDLE"))
        if throttling:
            warn.append(("warn", f"{throttling} GPU{'S' if throttling > 1 else ''} THROTTLING"))

    # CPU underuse is deliberately NOT a banner alarm: it's often intentional (a
    # debug shell, a data-loading stage) and the CPU row already carries its own
    # amber dot, so a headline here just nagged and duplicated the row. The
    # banner is reserved for things that need action — memory near OOM, GPUs idle
    # or throttling.
    return crit + warn


def _banner_line(segments: list[tuple[str, str]], ascii_mode: bool, width: int) -> str:
    """Join the alert segments into one line, worst first.

    When they would overflow ``width`` (0 = unknown / unbounded) and soft-wrap
    mid-phrase, collapse to just the single worst alert plus a ``(+N more)``
    hint, so the headline stays a legible single line (B10).
    """
    parts = [
        f"{_dot(level, ascii_mode)} [bold {_HEALTH_COLOR[level]}]{text}[/]"
        for level, text in segments
    ]
    # NBSP keeps a chip from wrapping apart, but it's non-ASCII; under --ascii use
    # plain spaces so nothing leaks on a non-UTF-8 terminal.
    gap = "   " if ascii_mode else f"{_NBSP}{_NBSP}{_NBSP}"
    line = gap.join(parts)
    if width and len(segments) > 1:
        try:
            visible = Text.from_markup(line).cell_len
        except Exception:
            visible = len(line)
        if visible > width:
            level, text = segments[0]
            more = len(segments) - 1
            line = (
                f"{_dot(level, ascii_mode)} [bold {_HEALTH_COLOR[level]}]{text}[/]"
                f"   [dim](+{more} more)[/]"
            )
    return line


class StatusBanner(Static):
    """The single most important line: the worst problem, in plain language."""

    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]connecting…[/]"
        snap = self.snapshot
        cfg = self.config or SlurmwatchConfig()
        ascii_mode = cfg.ascii_mode
        segments = _banner_segments(snap, cfg)

        parts: list[str] = []
        if segments:
            parts.append(_banner_line(segments, ascii_mode, self.size.width))
        # NB: a GPU job where NVML can't see the devices is NOT surfaced here — the
        # RESOURCES GPU row already says "N requested — telemetry unavailable here
        # (run on the compute node)", which is more informative and actionable, so
        # a second banner note would just be the same message twice on one screen.
        # Everything healthy shows no banner at all (the rows tell the story).
        return f"{_NBSP}{_NBSP}{_NBSP}".join(parts)


# Monitor-step contention note: when slurmwatch is the login-node hop's own job
# step it reserves the allocation's cores, so a NEW srun/mpirun the user starts
# will wait until slurmwatch quits (Slurm won't create a second, non-overlapping
# step while ours holds them). The note is quiet by default and escalates to an
# amber line once a launcher has sat in the job at ~idle CPU for _STUCK_POLLS
# consecutive polls (the fingerprint of a launch stuck at step creation).
_STUCK_CPU_PCT = 5.0
_STUCK_POLLS = 3


class MonitorNote(Static):
    """Quiet-by-default note, shown only when this process is the hop's monitor step."""

    escalated: bool = False
    node: str = ""
    ascii_mode: bool = False

    def render(self) -> str:
        dash = "-" if self.ascii_mode else "\N{EM DASH}"
        if self.escalated:
            dot = _dot("warn", self.ascii_mode)
            return (
                f"{dot} [bold {_HEALTH_COLOR['warn']}]a launch looks stuck waiting on cores "
                f"this monitor holds {dash} press q to quit slurmwatch and let it start[/]"
            )
        node = self.node or "this node"
        # A faint neutral dot (font-safe, with an ASCII variant) — NOT a health
        # glyph; this is an informational note, not an alarm.
        return (
            f"{_dot('none', self.ascii_mode)} [{_FAINT}]monitoring via a job step on {node} "
            f"{dash} a new srun/mpirun you start now waits until you quit; run slurmwatch on "
            f"the node to avoid[/]"
        )


class SwitchBanner(Static):
    """The 'now changing node' indicator, shown while a newly-selected node's
    first live snapshot is still on its way.

    Unlike a toast (which vanishes on a timer, often *before* the data lands, so
    a slow ``srun`` step launch reads as "nothing happened / stuck"), this stays
    up until the dashboard has real data for the target node, and it animates —
    a spinning glyph plus the destination node — so a multi-second attach never
    looks frozen. The dashboard shows it on a key press and hides it the instant
    the node's data arrives.
    """

    # Braille spinner (10 frames) reads as smooth motion at ~8fps; the ASCII
    # fallback is the classic |/-\ so a non-UTF-8 terminal still animates.
    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _FRAMES_ASCII = ("|", "/", "-", "\\")

    target_label: str = ""  # e.g. "node 2 of 2"
    node: str = ""  # destination hostname
    frame: int = 0
    ascii: bool = False
    slow: bool = False  # set once the attach is taking a while, for a reassuring note
    stuck: bool = False  # set once it's taking long enough to look unreachable
    prompt: str = ""  # digits typed so far for a "go to node N" jump (big jobs)
    total: str = ""  # node count, shown as "of N" in the prompt (own field, not `node`)
    ended: bool = False  # the monitored job has finished; a static, final notice
    ended_job: str = ""  # job id, shown in the ended notice

    def render(self) -> str:
        # The job has ended: a static, final notice that outranks everything else
        # (no spinner, no switch prompt — telemetry has stopped for good). #28.
        if self.ended:
            mark = "x" if self.ascii else "⚑"
            cap = f"[bold {_BG} on {_HEALTH_COLOR['crit']}] {mark} [/]"
            job = f" [{_INK}]{self.ended_job}[/]" if self.ended_job else ""
            head = f"[bold {_HEALTH_COLOR['crit']}]JOB{job} ENDED[/]"
            dash = "-" if self.ascii else "—"
            note = (
                f"[{_DIM}]{dash} telemetry stopped (last values shown) "
                f"{_sep(self.ascii)} press q to quit[/]"
            )
            return f"{cap} {head} {note}"
        # "Go to node" input takes precedence: while the user is typing a node
        # number (multi-digit jump on a big job), echo it so they see what they're
        # entering before it commits.
        if self.prompt:
            cursor = "_"
            of = f" [{_DIM}]of {self.total}[/]" if self.total else ""
            cap = f"[bold {_BG} on {_GPU_COLOR}] # [/]"
            head = f"[bold {_GPU_COLOR}]go to node[/] [{_INK}]{self.prompt}[/][{_DIM}]{cursor}[/]"
            return f"{cap} {head}{of}"
        if not self.target_label:
            return ""
        arrow = "->" if self.ascii else "→"
        tail = "..." if self.ascii else "…"
        # Once a switch has waited long enough to look unreachable, the banner
        # turns into an amber warning (the dashboard also un-dims the body) so the
        # session never sits frozen on a node that may be down or refusing the
        # attach — it keeps retrying in the background, and this says so plainly.
        if self.stuck:
            mark = "!" if self.ascii else "⚠"
            cap = f"[bold {_BG} on {_HEALTH_COLOR['warn']}] {mark} [/]"
            head = f"[bold {_HEALTH_COLOR['warn']}]still reaching {self.target_label}[/]"
            dash = "-" if self.ascii else "—"
            where = (
                f"[{_DIM}]{arrow} {self.node} {tail} "
                f"(it may be busy or unreachable {dash} still retrying; or switch nodes)[/]"
            )
            return f"{cap} {head} {where}"
        frames = self._FRAMES_ASCII if self.ascii else self._FRAMES
        glyph = frames[self.frame % len(frames)]
        # A filled violet cap carries the animated spinner; the destination reads
        # in the same node/violet family as the "Node" footer key so the eye ties
        # the key press to what it's doing.
        cap = f"[bold {_BG} on {_GPU_COLOR}] {glyph} [/]"
        head = f"[bold {_GPU_COLOR}]switching to {self.target_label}[/]"
        where = f"[{_DIM}]{arrow} {self.node} {tail}[/]"
        line = f"{cap} {head} {where}"
        if self.slow:
            line += f"   [{_FAINT}](Slurm can take a few seconds to attach)[/]"
        return line


class ResourceRows(Static):
    """One scannable row per live resource: dot · bar · % · numbers · spark · status."""

    snapshot: TelemetrySnapshot | None = None
    config: SlurmwatchConfig | None = None
    # Set by the dashboard: when the GPU DataTable is showing (3+ devices) the
    # rows don't also render per-GPU lines.
    gpu_table_active: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cpu_history: deque[float] = deque(maxlen=60)
        self.mem_history: deque[float] = deque(maxlen=60)
        self.gpu_history: dict[int, deque[float]] = {}

    def _head(self, label: str, color: str, ascii_mode: bool) -> str:
        # A decorative marker dot in the resource's own hue, then the label in the
        # same hue — so the row reads as "this is the CPU / MEM / GPU line", an
        # attractive identity colour, NOT a health verdict. The row reports facts
        # (the bar, the %, the recent range, the figures) and lets the reader judge
        # whether the job is running well; it never appends a verdict word, and the
        # marker's colour never asserts one either.
        marker = _MARKER_ASCII if ascii_mode else _MARKER
        return f"  [{color}]{marker}[/] [{color}]{label:<5}[/]"

    @staticmethod
    def _trend_tag(hist: deque[float], window_s: int, ascii_mode: bool = False) -> str:
        """The series' recent min–max, as a dim trailing tag on the row.

        This is the recent-range the standalone TRENDS panel used to draw, folded
        onto the resource's own row: the current level (bar + %) and how much it
        moved over the window now live in one place, so nothing is a second panel
        repeating the same value. A series that barely moved reads as ``steady``;
        otherwise the observed span, e.g. ``9–15% over 60s``. Empty history (no
        sample yet) prints nothing.
        """
        vals = list(hist)
        if not vals:
            return ""
        lo, hi = min(vals), max(vals)
        dot = "-" if ascii_mode else "·"
        if hi - lo < _TREND_STEADY_SPAN:
            return f"   [{_DIM}]{dot} steady[/]"
        dash = "-" if ascii_mode else "–"
        return f"   [{_DIM}]{dot} {lo:.0f}{dash}{hi:.0f}% over {window_s}s[/]"

    def render(self) -> str:
        if self.snapshot is None:
            return "[dim]awaiting telemetry…[/]"
        snap = self.snapshot
        cfg = self.config or SlurmwatchConfig()
        ascii_mode = cfg.ascii_mode
        wide = self.size.width >= _NARROW_COLS or self.size.width == 0
        bar_w = _BAR_W if wide else 12
        # The recent-range tag (folding in what the old TRENDS panel showed) is
        # secondary; drop it on a narrow terminal so a row can't wrap, exactly as
        # the memory peak is dropped below.
        window_s = cfg.history_seconds

        # One block per resource, joined with a blank line so the section breathes
        # instead of packing three resources into three tight adjacent rows.
        blocks: list[str] = []

        cpu = snap.cpu
        cpu_bar = _labeled_bar("usage", cpu.usage_percent, bar_w, ascii_mode, _CPU_COLOR)
        cpu_detail = f"{_fmt_cores(cpu.effective_cores)} / {cpu.cores_allocated} cores"
        cpu_tag = self._trend_tag(self.cpu_history, window_s, ascii_mode) if wide else ""
        blocks.append(
            f"{self._head('CPU', _CPU_COLOR, ascii_mode)}   "
            f"{cpu_bar}   [{_DIM}]{cpu_detail}[/]{cpu_tag}"
        )

        mem = snap.memory
        ws = mem.working_set_bytes or mem.current_bytes
        mem_head = self._head("MEM", _MEM_COLOR, ascii_mode)
        if mem.limit_bytes > 0:
            mem_pct = _mem_ws_pct(mem)
            mem_detail = f"{_gib(ws):.0f} / {_gib(mem.limit_bytes):.0f} GiB"
            # Off-node (sstat) the figure is a lifetime peak, not a live "used", so
            # label the bar "peak" — matching the text summary — and skip the "·
            # peak N" suffix (it would just repeat the same number). #34.
            mem_metric = "peak" if snap.remote else "used"
            # Peak is secondary; drop it on a narrow terminal so a big-memory job
            # (3-digit GiB) can't push the line past 80 cols and soft-wrap.
            if wide and not snap.remote:
                mem_detail += f" {'-' if ascii_mode else '·'} peak {_gib(mem.peak_bytes):.0f} GiB"
            mem_bar = _labeled_bar(mem_metric, mem_pct, bar_w, ascii_mode, _MEM_COLOR)
            mem_tag = self._trend_tag(self.mem_history, window_s, ascii_mode) if wide else ""
            blocks.append(f"{mem_head}   {mem_bar}   [{_DIM}]{mem_detail}[/]{mem_tag}")
        else:
            # No enforced limit → a 'used 0%' bar would contradict the GiB in
            # use, so show the amount only, with no misleading percentage.
            dot = "-" if ascii_mode else "·"
            blocks.append(
                f"{mem_head}   [{_DIM}]{'used':<7}[/] "
                f"[{_INK}]{_format_bytes(ws)}[/] [{_DIM}]{dot} no limit set[/]"
            )

        gpus = snap.gpus
        if self.gpu_table_active and gpus:
            # 3+ GPUs render in the DataTable below. Give the group the same
            # marker · label section head the CPU/MEM rows carry (GPU violet), so
            # GPU reads as a first-class resource aligned with the others, not a
            # header-less table floating to the left. Device/active counts are
            # facts; the reader judges from them and the per-device rows below.
            active = sum(1 for g in gpus if _gpu_is_active(g, cfg.gpu_idle_threshold))
            dot = "-" if ascii_mode else "·"
            blocks.append(
                f"{self._head('GPU', _GPU_COLOR, ascii_mode)}   "
                f"[{_DIM}]{_plural(len(gpus), 'device')} {dot} {active} active[/]"
            )
        elif not self.gpu_table_active:
            if gpus:
                for gpu in gpus:
                    blocks.append("\n".join(self._gpu_block(gpu, cfg, bar_w, ascii_mode, wide)))
            elif snap.gpu_count_requested > 0:
                # Reuse _head so "GPU" lines up with the CPU/MEM labels; ASCII dash
                # off-node. Off-node the fix is "go to the node"; ON the node with
                # no readable GPU we got here via the --gres=none fallback (the
                # GPU is held by the job's own step), so don't tell the user to do
                # what they've already done.
                dash = "-" if ascii_mode else "—"
                if snap.remote:
                    note = "telemetry unavailable here (run on the compute node)"
                else:
                    note = (
                        "GPU locked by this job's own srun step; Slurm can't share it "
                        "with a monitor (launch the program without srun for live GPU)"
                    )
                blocks.append(
                    f"{self._head('GPU', _GPU_COLOR, ascii_mode)}   "
                    f"[dim]{snap.gpu_count_requested} requested {dash} {note}[/]"
                )
            else:
                blocks.append(
                    f"{self._head('GPU', _GPU_COLOR, ascii_mode)}   [dim]none requested[/]"
                )
        return "\n\n".join(blocks)

    def _gpu_block(
        self, gpu: GpuMetrics, cfg: SlurmwatchConfig, bar_w: int, ascii_mode: bool, wide: bool
    ) -> list[str]:
        # A GPU has two independent "how busy / how full" axes, so it gets two
        # explicitly-labeled bars — compute (SM/CUDA-core utilisation) and vram
        # (memory fill) — instead of one unlabeled bar that reads as whichever
        # number sits beside it. 'vram' (not 'memory') so it can't blur with the
        # MEM row above. The marker is the GPU identity colour, not a verdict.
        compute = _labeled_bar("compute", gpu.utilization_percent, bar_w, ascii_mode, _GPU_COLOR)
        vram_bar = _labeled_bar(
            "vram", gpu.memory_utilization_percent, bar_w, ascii_mode, _GPU_VRAM_COLOR
        )
        used_g, tot_g = _gib(gpu.memory_used_bytes), _gib(gpu.memory_total_bytes)
        vram_amt = f"{used_g:.0f} / {tot_g:.0f} GiB"
        pwr = f"{gpu.power_watts:.0f} W"
        deg = "C" if ascii_mode else "°C"
        hot = gpu.temperature_celsius >= _TEMP_HOT_C
        # Same hot marker as the GPU table ("⚠" / ASCII "!"), so the two views agree.
        mark = (" !" if ascii_mode else " ⚠") if hot else ""
        temp_txt = f"{gpu.temperature_celsius:.0f} {deg}{mark}"
        temp = f"[{_HEALTH_COLOR['warn']}]{temp_txt}[/]" if hot else f"[{_DIM}]{temp_txt}[/]"
        head = self._head(f"GPU{gpu.index}", _GPU_COLOR, ascii_mode)
        tail = f"[{_DIM}]{vram_amt}[/]   [{_DIM}]{pwr}[/]{_sep(ascii_mode)}{temp}"

        # On a wide-enough terminal the two bars (they describe one device) ride a
        # single line, so a multi-GPU job is one row per device instead of three.
        if self.size.width >= _GPU_MERGE_COLS or self.size.width == 0:
            return [f"{head}   {compute}   {vram_bar}   {tail}"]

        # Otherwise stack them, and fold the compute series' recent range (what the
        # old TRENDS panel tracked per GPU) onto the compute line.
        gpu_hist = self.gpu_history.get(gpu.index, deque())
        compute_tag = self._trend_tag(gpu_hist, cfg.history_seconds, ascii_mode) if wide else ""
        # Indent the stacked bars to the SAME column the CPU/MEM bars start at, so
        # every gauge in the card lines up. That lead is the _head width (2 + dot +
        # space + 5-wide label = 9) plus the 3-space gap before the bar = 12.
        indent = " " * 12
        return [
            head,
            f"{indent}{compute}{compute_tag}",
            f"{indent}{vram_bar}   {tail}",
        ]


class GpuTable(DataTable[Any]):
    """A device-per-row table, used when 3+ GPUs would scroll as stacked rows.

    Two modes, keyed on the widget id: the compact dashboard overview, and the
    detail-screen copy (``id="detail-table"``) which adds the job's per-device
    share (JOB% / JOB VRAM) and a per-device compute TREND sparkline the overview
    doesn't show, so drilling in reveals something new (F6). Neither copy has a
    row cursor — nothing is selectable in either (every device is shown, and the
    detail table charts each inline), so an always-on highlight would mislead (U5).
    """

    config: SlurmwatchConfig | None = None
    # Whether the per-device TREND sparkline column is currently shown (detail
    # table only, and only when the terminal is wide enough — see _want_trend).
    _trend_on: bool = False

    @property
    def _detailed(self) -> bool:
        return self.id == "detail-table"

    def on_mount(self) -> None:
        # No row cursor in either mode: the overview isn't interactive (U5), and
        # the detail table now charts *every* device inline (a per-row TREND
        # sparkline), so there's nothing to "select" — an always-on highlight
        # would just imply a selection that does nothing.
        self.cursor_type = "none"
        # Zebra stripes off: the per-device colour on each row already separates
        # them, and a background band would clutter the compact one-line rows.
        self.zebra_stripes = False
        self._sync_columns()

    def _want_trend(self) -> bool:
        """Whether the detail table should carry the per-device TREND sparkline.

        It's the widest "nice to have" column, so it's dropped on a terminal too
        narrow to fit the essentials — otherwise STATUS (the health word) gets
        pushed off-screen behind a horizontal scrollbar. The overview never shows
        it (that view is compact by design)."""
        if not self._detailed:
            return False
        try:
            width = self.app.size.width
        except Exception:
            return True  # size not known yet → assume wide; corrected on next update
        return width == 0 or width >= _GPU_TREND_MIN_COLS

    def _sync_columns(self) -> None:
        """(Re)build the column set, adding/dropping TREND as the width crosses the
        threshold. A no-op once built at a steady width, so the in-place cell
        update path below keeps working."""
        want = self._want_trend()
        if self.columns and want == self._trend_on:
            return
        self.clear(columns=True)
        cols = ["#", "COMPUTE"]
        if self._detailed:
            if want:
                cols.append("TREND")
            cols += ["VRAM", "JOB%", "JOB VRAM", "PWR", "TEMP", "STATUS"]
        else:
            cols += ["VRAM", "PWR", "TEMP", "STATUS"]
        self.add_columns(*cols)
        self._trend_on = want

    def _row_cells(
        self,
        gpu: GpuMetrics,
        config: SlurmwatchConfig,
        history: dict[int, deque[float]] | None = None,
    ) -> list[Text | str]:
        """The cells for one device's row (fixed widths so in-place updates don't
        shift columns). Column order matches :meth:`on_mount`."""
        ascii_mode = config.ascii_mode
        _, word = _gpu_health(gpu, config.gpu_idle_threshold)
        # Each device wears its own colour (index + compute bar + VRAM) so identical
        # rows stay distinguishable. Health (status) and heat (temp) keep their own
        # colour channel — those are the same across devices.
        dcolor = _gpu_device_color(gpu.index)
        gpu_cell = Text(str(gpu.index), style=f"bold {dcolor}")
        bar = _color_bar(gpu.utilization_percent, 8, ascii_mode, dcolor)
        util = Text.from_markup(f"{gpu.utilization_percent:>3.0f}% {bar}")
        vram = Text(
            f"{_gib(gpu.memory_used_bytes):>3.0f}/{_gib(gpu.memory_total_bytes):>3.0f} GiB",
            style=dcolor,
        )
        pwr = f"{gpu.power_watts:>4.0f}W"
        hot = gpu.temperature_celsius >= _TEMP_HOT_C
        deg = "C" if ascii_mode else "°C"
        temp_mark = ("!" if ascii_mode else "⚠") if hot else " "
        temp = Text(
            f"{temp_mark}{gpu.temperature_celsius:>3.0f}{deg}", style="yellow" if hot else ""
        )
        # STATUS is the plain fact — the word active / idle / throttling — in the
        # device's own (decorative) colour, NOT a health grade: the reader sees
        # what the GPU is doing and decides for themselves. Padded to a FIXED width
        # ("throttling" is the widest) so the in-place cell update (update_width=
        # False) never has to grow the column and clip a later "throttling".
        status = Text(word.ljust(12), style=dcolor)
        if self._detailed:
            job_util = f"{gpu.process_utilization_percent:>3.0f}%"
            # Fixed 9-wide too, for the same reason: "999.9 GiB" is the widest, and
            # the no-VRAM case pads to match so it can't shrink the column.
            job_vram = (
                f"{_gib(gpu.process_memory_bytes):>5.1f} GiB"
                if gpu.process_memory_bytes
                else f"{('-' if ascii_mode else '—'):>9}"
            )
            cells: list[Text | str] = [gpu_cell, util]
            if self._trend_on:
                # A per-device compute sparkline, in the device's own hue, so every
                # GPU's recent trend is visible AT ONCE — no cursor to move, nothing
                # gated behind a selection. Absolute 0–100 scale (honest height =
                # real level); stretch fills the cell even while history fills.
                hist = (history or {}).get(gpu.index, deque())
                cells.append(
                    Text(_render_sparkline(hist, _SPARK_W, ascii_mode, stretch=True), style=dcolor)
                )
            cells += [vram, job_util, job_vram, pwr, temp, status]
            return cells
        return [gpu_cell, util, vram, pwr, temp, status]

    def update_gpus(
        self,
        gpus: list[GpuMetrics],
        config: SlurmwatchConfig,
        history: dict[int, deque[float]] | None = None,
    ) -> None:
        # A resize may have crossed the TREND-column width threshold; rebuild the
        # column set if so (a no-op at steady width). This also clears the rows,
        # forcing the full rebuild below when the column count changed.
        self._sync_columns()
        rows = [self._row_cells(gpu, config, history) for gpu in gpus]
        # Update cells IN PLACE when the device count is unchanged (the normal
        # case): clearing + re-adding every ~0.5s would reset the scroll position,
        # yanking a device the user scrolled to back to the top. Cells are
        # fixed-width, so update_width is off to avoid a column reflow flicker.
        # Only a changed device count (a GPU appearing/vanishing — rare) rebuilds.
        if self.row_count == len(rows) and len(self.columns) == (len(rows[0]) if rows else 0):
            for r, cells in enumerate(rows):
                for c, value in enumerate(cells):
                    self.update_cell_at(Coordinate(r, c), value, update_width=False)
            return
        self.clear()
        for cells in rows:
            self.add_row(*cells)


class JobDetailsPanel(Static):
    """Job provenance the rest of the UI doesn't carry — account/QOS/state,
    command, workdir, and submit→start (queue wait).

    Deliberately excludes anything already visible elsewhere: the RESOURCES rows
    already show allocated cores / memory-limit / used / peak, and the bottom bar
    shows id/user/partition/node + the time budget — so this card never restates
    an allocation number or the requested TRES (that would just be the same
    facts twice). A line is omitted when its field is absent.
    """

    job_ctx: JobContext | None = None
    config: SlurmwatchConfig | None = None
    # Toggled by the dashboard's "p" key: show command/workdir in full (hard-
    # wrapped) instead of the elided root/…/leaf form, so the whole path is
    # readable/selectable on demand.
    full_paths: bool = False

    def render(self) -> str:
        ctx = self.job_ctx
        if ctx is None:
            return "[dim]awaiting job info…[/]"
        ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode
        # A roomy, colourful card: dim labels, values in the palette hues (so it
        # reads lively like the bottom bar, not a flat grey block), and the three
        # logical groups — identity / where / when — separated by a blank line so
        # it breathes. Each group is packed at the card width so a chip wraps as a
        # unit (a value is never stranded from its label) and every wrapped line
        # keeps the 2-space indent.
        gap = f"  {_sep(ascii_mode)}  "
        inner_w = max(20, (self.size.width or 100) - 2)

        def _group(items: list[str]) -> str:
            return "\n".join("  " + ln for ln in _pack_chips(items, gap, inner_w).split("\n"))

        groups: list[str] = []

        chips = []
        if ctx.account:
            chips.append(f"[{_DIM}]account[/] [{_CPU_COLOR}]{_escape_markup(ctx.account)}[/]")
        if ctx.qos:
            chips.append(f"[{_DIM}]qos[/] [{_GPU_COLOR}]{_escape_markup(ctx.qos)}[/]")
        if ctx.job_state:
            color = _job_state_color(ctx.job_state)
            chips.append(f"[{_DIM}]state[/] [{color}]{_escape_markup(ctx.job_state)}[/]")
        if chips:
            groups.append(_group(chips))

        # Long paths are elided to one line (root + …/ + leaf) so a deep working
        # directory or script path doesn't wrap mid-word into a cluttered block —
        # unless the user pressed "p" to reveal them in full (self.full_paths),
        # in which case the whole value is shown, hard-wrapped to the card width
        # with a hanging indent so even a path with no break points can't overflow.
        ell = "..." if ascii_mode else "…"
        card_w = self.size.width or 100
        budget = max(24, card_w - 12)
        truncated = False

        def _path_row(label: str, value: str, color: str, *, keep: int) -> str:
            nonlocal truncated
            lead = 2 + len(label) + 2  # "  command  " / "  workdir  "
            if self.full_paths:
                avail = max(8, card_w - lead)
                chunks = [value[i : i + avail] for i in range(0, len(value), avail)] or [""]
                head = f"  [{_DIM}]{label}[/]  [{color}]{_escape_markup(chunks[0])}[/]"
                cont = [f"{' ' * lead}[{color}]{_escape_markup(c)}[/]" for c in chunks[1:]]
                return "\n".join([head, *cont])
            # Elided: a full command line (has spaces/args) is left intact so its
            # arguments aren't mistaken for directories; a bare path is shortened.
            is_cmdline = keep == 2 and " " in value
            shown = value if is_cmdline else _shorten_path(value, budget, keep, ell)
            if shown != value:
                truncated = True
            return f"  [{_DIM}]{label}[/]  [{color}]{_escape_markup(shown)}[/]"

        paths = []
        if ctx.command:  # the headline "what is this job running" — coral pops
            paths.append(_path_row("command", ctx.command, _ACCENT, keep=2))
        if ctx.work_dir:
            paths.append(_path_row("workdir", ctx.work_dir, _MEM_COLOR, keep=1))
        if paths:
            # A quiet hint right under the paths — only when it's useful: something
            # was shortened (press p to reveal it), or paths are already expanded
            # (press p to collapse). Never shown when everything already fits.
            if self.full_paths:
                paths.append(f"  [{_FAINT}]press [/][{_INK}]p[/][{_FAINT}] to collapse[/]")
            elif truncated:
                paths.append(f"  [{_FAINT}]press [/][{_INK}]p[/][{_FAINT}] for the full path[/]")
            groups.append("\n".join(paths))

        if ctx.submit_time or ctx.job_start_time:
            times = []
            if ctx.submit_time:
                times.append(f"[{_DIM}]submitted[/] [{_INK}]{_format_clock(ctx.submit_time)}[/]")
            if ctx.job_start_time:
                times.append(f"[{_DIM}]started[/] [{_INK}]{_format_clock(ctx.job_start_time)}[/]")
            if ctx.submit_time and ctx.job_start_time:
                wait = max(0, int(ctx.job_start_time - ctx.submit_time))
                times.append(f"[{_DIM}]queue wait[/] [{_CPU_COLOR}]{_format_wait(wait)}[/]")
            groups.append(_group(times))

        if not groups:
            return "[dim]no job details available[/]"
        # Blank line between groups for vertical breathing room.
        return "\n\n".join(groups)


class JobInfoBar(Static):
    """The bottom info bar: what this job is, and how long it can still run.

    Labels every field (so the header line isn't a cryptic ``a · b · c · d``) and
    turns the otherwise-empty space at the foot of the screen into a live
    time-budget line — elapsed vs. the wall-clock limit, time left, and the
    latest possible end (when the wall-clock limit is reached) — which the top
    header never showed.
    """

    snapshot: TelemetrySnapshot | None = None
    job_ctx: JobContext | None = None
    config: SlurmwatchConfig | None = None
    # Set by the dashboard on a short terminal: drop the second (time-budget) line
    # so the docked bar doesn't starve the RESOURCES gauges of screen rows.
    compact: bool = False

    def render(self) -> str:
        ctx = self.job_ctx
        snap = self.snapshot
        if ctx is None or snap is None:
            return ""
        ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode
        sep = _sep(ascii_mode)
        inner = (self.size.width or 100) - 6  # #jobinfo padding 1 3

        if snap.node_count > 1:
            node = f"{snap.hostname} (node {snap.node_index + 1} of {snap.node_count})"
        else:
            node = ctx.nodelist or snap.hostname
        node_style = _MEM_COLOR
        # When the shown node is streamed from another node (the switcher), its
        # snapshot is a few seconds old — surface that age honestly instead of
        # implying it's live. A live local node is always sub-second fresh, so
        # this hides. (Plain "Ns old", not "sampled" — the switcher no longer
        # calls it "sampling".)
        age = time.time() - snap.timestamp
        _d = "-" if ascii_mode else "·"
        freshness = f" [{_FAINT}]{_d} {int(age)}s old[/]" if age >= 3 else ""
        # Each identity value gets its own hue (dim labels, coloured values) so the
        # bottom bar reads as a lively strip rather than a flat grey line. Packing
        # wraps the strip only between chips, so on a narrow terminal a value is
        # never orphaned from its label. These are chrome, well below the resource
        # rows, so reusing the palette ties the UI together without being mistaken
        # for a CPU/MEM/GPU reading.
        # Every interpolated value is user-controllable (job name can smuggle a
        # `Partition=` token into scontrol's first line, poisoning ctx.partition;
        # names/nodes are free-form) — escape them, or a stray `[/]` crashes the
        # whole TUI via Textual's markup parser, exactly as the other panels guard.
        ident_chips = [
            f"[{_DIM}]job[/] [{_ACCENT}]{_escape_markup(str(snap.job_id))}[/]",
            f"[{_DIM}]user[/] [{_CPU_COLOR}]{_escape_markup(ctx.username or '?')}[/]",
            f"[{_DIM}]partition[/] [{_GPU_COLOR}]{_escape_markup(ctx.partition or '?')}[/]",
            f"[{_DIM}]node[/] [{node_style}]{_escape_markup(node)}[/]{freshness}",
        ]
        ident = _pack_chips(ident_chips, sep, inner)
        # On a short terminal the docked bar is capped to a single line so the
        # RESOURCES gauges keep their rows — drop the secondary time-budget line.
        if self.compact:
            return ident

        elapsed = snap.elapsed_seconds
        limit = ctx.time_limit_seconds
        if limit and limit > 0:
            frac = min(100.0, elapsed / limit * 100.0)
            remaining = max(0, limit - elapsed)
            el, rem, lim = (
                _format_duration(elapsed),
                _format_duration(remaining),
                _format_duration(limit),
            )
            # The wall-clock deadline (start + limit): the LATEST the job can run,
            # not a prediction — a job that finishes early stops sooner. Hence
            # "ends by", not "ends ~".
            ends = time.strftime("%a %H:%M", time.localtime(time.time() + remaining))
            # Colour the bar and the "left" figure by how much time remains — a
            # job about to hit the wall glows red — so the colour is useful, not
            # just decorative: green with room, amber getting low, red near the end.
            frac_left = remaining / limit
            urg_level = "ok" if frac_left > 0.25 else "warn" if frac_left > 0.10 else "crit"
            urg = _HEALTH_COLOR[urg_level]
            # Size the progress bar to whatever width is left after the text, and
            # drop it entirely on a narrow terminal, so the line never wraps past
            # its two rows (the bar was a fixed 20 cells before, overflowing 80).
            text = f"ran {el}  {frac:.0f}%  {_d}  {rem} left of {lim} limit  {_d}  ends by {ends}"
            # Leave >=2 cols of right margin so the line never touches the edge.
            bar_w = min(20, inner - len(text) - 3)
            bar = f"{_color_bar(frac, bar_w, ascii_mode, urg)} " if bar_w >= 6 else ""
            # Chip-pack the time budget too, so on a narrow terminal it wraps only
            # between fields (never orphaning "limit" from "02:00:00"), like ident.
            time_line = _pack_chips(
                [
                    f"[{_DIM}]ran[/] [{_INK}]{el}[/] {bar}[{_INK}]{frac:.0f}%[/]",
                    f"[bold {urg}]{rem}[/] [{_DIM}]left of[/] [{_INK}]{lim}[/] [{_DIM}]limit[/]",
                    f"[{_DIM}]ends by[/] [{_ACCENT}]{ends}[/]",
                ],
                sep,
                inner,
            )
        else:
            time_line = _pack_chips(
                [
                    f"[{_DIM}]ran[/] [{_INK}]{_format_duration(elapsed)}[/]",
                    f"[{_DIM}]no wall-clock time limit[/]",
                ],
                sep,
                inner,
            )
        # A blank line between the identity and the time-budget line so the docked
        # bar doesn't read as two cramped rows crushed against the footer — it
        # breathes like the RESOURCES card's gauge blocks do. (Compact mode above
        # returns the single identity line, so this only adds a row when there's
        # room for the time budget anyway.)
        return f"{ident}\n\n{time_line}"


class ResourceDetailScreen(Screen[None]):
    """A full-screen drill-in for one resource: a big live figure, a tall filled
    history graph, the full set of numbers, and (for GPUs) the per-device table.

    Reads the dashboard's live snapshot on a timer so it keeps updating while
    open. The three resources share one layout so ``c``/``m``/``g`` flip between
    them instantly, and each wears its own accent (CPU cyan / MEM rose / GPU
    violet) with a health-aware figure colour.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back"),
        Binding("c", "switch('cpu')", "CPU"),
        Binding("m", "switch('mem')", "Memory"),
        Binding("g", "switch('gpu')", "GPU"),
    ]

    CSS = """
    ResourceDetailScreen { align: center middle; background: $surface; }
    #detail-box {
        width: 92%;
        max-width: 132;
        height: auto;
        max-height: 94%;
        border: round $primary;
        background: $panel;
        padding: 1 2 0 2;
    }
    #detail-title { height: auto; text-style: bold; padding-bottom: 1; }
    #detail-hero { height: auto; }
    #detail-figure { width: auto; height: auto; }
    #detail-headline { width: 1fr; height: auto; padding: 1 0 0 3; }
    #detail-chart { height: auto; padding-top: 1; }
    #detail-table { height: auto; margin-top: 1; }
    #detail-body { height: auto; padding: 1 0; }
    #detail-keybar { height: 1; dock: bottom; padding: 0 3; background: $panel; }
    """

    def __init__(self, dashboard: DashboardScreen, resource: str) -> None:
        super().__init__()
        self._dashboard = dashboard
        self._resource = resource

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="detail-box") as box:
            box.border_title = f" {self._resource_name()} "
            yield Static(id="detail-title")
            if self._resource == "gpu":
                yield Static(id="detail-headline")
                yield GpuTable(id="detail-table")
            else:
                # A big Digits figure with the health-aware headline beside it.
                with Horizontal(id="detail-hero"):
                    yield Digits("", id="detail-figure")
                    yield Static(id="detail-headline")
            yield Static(id="detail-chart")
            yield Static(id="detail-body")
        keys = [
            ("q", "Back", _ACCENT),
            ("c", "CPU", _CPU_COLOR),
            ("m", "Memory", _MEM_COLOR),
            ("g", "GPU", _GPU_COLOR),
        ]
        yield KeyFooter(keys, id="detail-keybar")

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(0.5, self._refresh)
        # Focus the device table (it has no row cursor now — every device is
        # charted inline). ←/→ then scroll the table horizontally to reveal a
        # column clipped on a narrow terminal (e.g. STATUS on an 80-col SSH
        # session), while ↑/↓ · PgUp/PgDn bubble up to scroll the box vertically to
        # reach devices below the fold. Focusing the box instead would lose the
        # horizontal scroll (the box is overflow-x: hidden).
        if self._resource == "gpu":
            with contextlib.suppress(NoMatches):
                self.query_one("#detail-table", GpuTable).focus()

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_switch(self, resource: str) -> None:
        if resource == self._resource:
            return
        self.app.pop_screen()
        self.app.push_screen(ResourceDetailScreen(self._dashboard, resource))

    _BOX_CHROME = 6
    _BOX_MAX_W = 132

    def _resource_name(self) -> str:
        return {"cpu": "CPU", "mem": "MEMORY", "gpu": "GPU"}.get(self._resource, "")

    def _resource_color(self) -> str:
        return {"cpu": _CPU_COLOR, "mem": _MEM_COLOR, "gpu": _GPU_COLOR}.get(
            self._resource, _ACCENT
        )

    def _figure_color(self, level: str) -> str:
        # Identity hue when healthy; escalate to the health colour when it needs
        # attention, so the big number itself signals a problem at a glance.
        return _HEALTH_COLOR[level] if level in ("warn", "crit") else self._resource_color()

    def _chart_width(self, chart_widget: Static) -> int:
        w = chart_widget.size.width
        if w <= 0:
            w = min(int(self.size.width * 0.92), self._BOX_MAX_W) - self._BOX_CHROME
        return max(w, _SPARK_W)

    def _chart_height(self) -> int:
        # Fill the vertical space, but leave room for the figure/table/footer. The
        # GPU table is the hero there, so its chart is shorter.
        h = self.size.height
        base = 7 if h <= 0 else max(4, min(10, int(h * 0.94) - 14))
        return min(base, 5) if self._resource == "gpu" else base

    def _refresh(self) -> None:
        snap = self._dashboard.latest_snapshot
        cfg = self._dashboard.config
        with contextlib.suppress(NoMatches):
            title = self.query_one("#detail-title", Static)
            if snap is None:
                title.update("[dim]awaiting telemetry…[/]")
                return
            live = "live" if not snap.remote else f"{int(time.time() - snap.timestamp)}s old"
            dot = "-" if cfg.ascii_mode else "·"
            title.update(
                f"[{self._resource_color()}]{self._resource_name()}[/]  [{_FAINT}]{dot} {live}[/]"
            )
            if self._resource == "cpu":
                self._refresh_cpu(snap, cfg)
            elif self._resource == "mem":
                self._refresh_mem(snap, cfg)
            else:
                self._refresh_gpu(snap, cfg)

    def _set_figure(self, text: str, level: str) -> None:
        with contextlib.suppress(NoMatches):
            fig = self.query_one("#detail-figure", Digits)
            fig.update(text)
            fig.styles.color = self._figure_color(level)

    def _set_headline(self, markup: str) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#detail-headline", Static).update(markup)

    def _set_body(self, markup: str) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#detail-body", Static).update(markup)

    def _refresh_cpu(self, snap: TelemetrySnapshot, cfg: SlurmwatchConfig) -> None:
        cpu = snap.cpu
        level, word = _cpu_health(cpu, cfg.cpu_underuse_threshold)
        self._set_figure(f"{cpu.usage_percent:.0f}%", level)
        cores = f"{_fmt_cores(cpu.effective_cores)} of {cpu.cores_allocated}"
        self._set_headline(
            f"[{_HEALTH_COLOR[level]}]{_glyph(level, cfg.ascii_mode)} {word}[/]\n"
            f"[{_DIM}]cores busy[/] [{_INK}]{cores}[/]\n"
            f"[{_DIM}]allocated[/] [{_INK}]{_plural(cpu.cores_allocated, 'core')}[/]"
        )
        insight = ""
        if level == "warn":  # underused
            insight = (
                f"[{_HEALTH_COLOR['warn']}]{_glyph('warn', cfg.ascii_mode)}[/] "
                f"[{_DIM}]only ~{_fmt_cores(cpu.effective_cores)} of "
                f"{cpu.cores_allocated} cores are doing work — a smaller [/]"
                f"[{_INK}]--cpus-per-task[/][{_DIM}] would schedule faster and free the rest.[/]"
            )
        self._set_body(insight)
        self._render_chart(self._dashboard.cpu_history, cfg)

    def _refresh_mem(self, snap: TelemetrySnapshot, cfg: SlurmwatchConfig) -> None:
        mem = snap.memory
        level, word = _mem_health(mem)
        ws = mem.working_set_bytes or mem.current_bytes
        if mem.limit_bytes > 0:
            pct = _mem_ws_pct(mem)
            self._set_figure(f"{pct:.0f}%", level)
            headroom = max(mem.limit_bytes - ws, 0)
            used = f"{_gib(ws):.0f} / {_gib(mem.limit_bytes):.0f} GiB"
            self._set_headline(
                f"[{_HEALTH_COLOR[level]}]{_glyph(level, cfg.ascii_mode)} {word}[/]\n"
                f"[{_DIM}]working set[/] [{_INK}]{used}[/]\n"
                f"[{_DIM}]headroom[/] [{_INK}]{_format_bytes(headroom)}[/]"
            )
            sep = _sep(cfg.ascii_mode)
            body = (
                f"[{_DIM}]peak[/] [{_INK}]{_gib(mem.peak_bytes):.0f} GiB[/]  {sep}  "
                f"[{_DIM}]reclaimable cache[/] [{_INK}]{_format_bytes(mem.cache_bytes)}[/]  {sep}  "
                f"[{_DIM}]total (incl. cache)[/] [{_INK}]{_format_bytes(mem.current_bytes)}[/]"
            )
            if level == "crit":
                body += (
                    f"\n[{_HEALTH_COLOR['crit']}]{_glyph('crit', cfg.ascii_mode)}[/] "
                    f"[{_DIM}]working set is {pct:.0f}% of the limit — a higher [/]"
                    f"[{_INK}]--mem[/][{_DIM}] would cut the OOM-kill risk.[/]"
                )
            self._set_body(body)
        else:
            self._set_figure(f"{_gib(ws):.0f}", "none")
            self._set_headline(
                f"[{_FAINT}]{_glyph('none', cfg.ascii_mode)} no limit set[/]\n"
                f"[{_DIM}]working set[/] [{_INK}]{_format_bytes(ws)}[/]\n"
                f"[{_DIM}]GiB in use[/]"
            )
            sep = _sep(cfg.ascii_mode)
            self._set_body(
                f"[{_DIM}]peak[/] [{_INK}]{_format_bytes(mem.peak_bytes)}[/]  {sep}  "
                f"[{_DIM}]reclaimable cache[/] [{_INK}]{_format_bytes(mem.cache_bytes)}[/]"
            )
        self._render_chart(self._dashboard.mem_history, cfg)

    def _refresh_gpu(self, snap: TelemetrySnapshot, cfg: SlurmwatchConfig) -> None:
        sep_ch = "-" if cfg.ascii_mode else "·"
        dash = "-" if cfg.ascii_mode else "—"
        if snap.gpus:
            active = sum(1 for g in snap.gpus if _gpu_is_active(g, cfg.gpu_idle_threshold))
            total = len(snap.gpus)
            rows_widget = self._dashboard.resource_rows
            history = rows_widget.gpu_history if rows_widget is not None else None
            self._set_headline(
                f"[{_GPU_COLOR}]{_plural(total, 'device')}[/] [{_DIM}]{sep_ch}[/] "
                f"[{_INK}]{active} active[/]   "
                f"[{_FAINT}]TREND = compute % over last {cfg.history_seconds}s  {sep_ch}  "
                f"JOB% / JOB VRAM = this job's share[/]"
            )
            with contextlib.suppress(NoMatches):
                self.query_one("#detail-table", GpuTable).update_gpus(snap.gpus, cfg, history)
            self._set_body("")
            # Every device is charted inline (its own per-row TREND sparkline), so
            # there's no single selected-device graph below the table to render.
            self._clear_chart()
        elif snap.gpu_count_requested > 0:
            if snap.remote:
                note = "live telemetry unavailable here; run on the compute node."
            else:
                note = (
                    "GPU locked by this job's own srun step — Slurm can't share a GPU "
                    "with a separate monitor step. Launch the program without an inner "
                    "srun (run it directly in the batch script) to see live GPU util."
                )
            self._set_headline(
                f"[{_DIM}]{_plural(snap.gpu_count_requested, 'GPU')} requested {dash} {note}[/]"
            )
            self._set_body("")
            self._clear_chart()
        else:
            self._set_headline("[dim]no GPUs requested by this job[/]")
            self._set_body("")
            self._clear_chart()

    def _clear_chart(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#detail-chart", Static).update("")

    def _render_chart(self, history: deque[float], cfg: SlurmwatchConfig, label: str = "") -> None:
        """The tall filled area graph — the drill-in's headline feature over the
        dashboard's one-row sparkline — plus a labelled axis and the summary
        stats (min / avg / max / now) the overview can't fit."""
        ascii_mode = cfg.ascii_mode
        with contextlib.suppress(NoMatches):
            chart = self.query_one("#detail-chart", Static)
            width = self._chart_width(chart)
            color = self._resource_color()
            gutter = 4  # "100 " / " 50 " / "  0 " left axis labels
            # Reserve 2 cols so a row (gutter + area) can never equal-or-exceed the
            # widget width and soft-wrap — covers the scrollbar the VerticalScroll
            # box shows on a short terminal, which narrows the content mid-layout.
            area_w = max(width - gutter - 2, _SPARK_W)
            height = self._chart_height()
            vals = list(history)

            lines: list[str] = []
            if label:
                sep_ch = "-" if ascii_mode else "·"
                lines.append(f"[{_DIM}]{label} {sep_ch} last {cfg.history_seconds}s[/]")
            rows = _area_chart(history, area_w, height, ascii_mode)
            for i, row in enumerate(rows):
                if i == 0:
                    lab = "100"
                elif i == height // 2:
                    lab = " 50"
                else:
                    lab = "   "
                lines.append(f"[{_FAINT}]{lab} [/][{color}]{row}[/]")
            rule = "-" if ascii_mode else "─"
            lines.append(f"[{_FAINT}]  0 {rule * area_w}[/]")

            # A time caption: oldest on the left, newest on the right.
            larr, rarr = ("<-", "->") if ascii_mode else ("←", "→")
            left, right = f"{larr} {cfg.history_seconds}s", f"now {rarr}"
            pad = area_w - len(left) - len(right)
            cap = left + " " * pad + right if pad >= 1 else left[:area_w]
            lines.append(f"[{_FAINT}]    {cap}[/]")

            if vals:
                mn, mx = min(vals), max(vals)
                av, cur = sum(vals) / len(vals), vals[-1]
                stats = (
                    f"min {mn:>3.0f}%   avg {av:>3.0f}%   max {mx:>3.0f}%   "
                    f"[{color}]now {cur:>3.0f}%[/]"
                )
            else:
                stats = "no history yet"
            lines.append(f"[{_DIM}]    {stats}[/]")
            chart.update("\n".join(lines))


class KeyFooter(Static):
    """A keybinding bar where each shortcut wears its target's colour.

    Textual's stock Footer paints every key the same accent; here the resource
    keys match their panels — c = CPU cyan, m = MEM rose, g = GPU violet — so the
    shortcut and the thing it opens are colour-linked, and quit stays coral.
    """

    def __init__(self, keys: list[tuple[str, str, str]], **kwargs: Any) -> None:
        # keys: (key, label, colour), all literal → markup-safe.
        super().__init__(**kwargs)
        self._keys = keys

    def render(self) -> str:
        # Full form: a coloured key cap + its label, e.g. "[ q ] Quit". When the
        # row would overrun a narrow terminal, drop labels ONE AT A TIME from the
        # left (q/c/m/g are self-evident from the letter) so the least-obvious cap
        # — the node switcher's "1-9 ◂▸" — keeps its "Node" word longest, and the
        # bar never wraps a label onto a hidden second line.
        avail = (self.size.width or 200) - 6  # #keybar padding 0 3
        n = len(self._keys)
        show = [True] * n  # whether each key still shows its word label

        def width() -> int:
            w = 2 * (n - 1)  # 2-space join between caps
            for i, (k, lbl, _c) in enumerate(self._keys):
                w += len(k) + 2 + (1 + len(lbl) if show[i] else 0)
            return w

        for i in range(n):  # drop the most-obvious labels first (left to right)
            if width() <= avail:
                break
            show[i] = False
        caps = []
        for i, (key, label, color) in enumerate(self._keys):
            cap = f"[{_BG} on {color}] {key} [/]"
            if show[i]:
                cap += f" [{_DIM}]{label}[/]"
            caps.append(cap)
        return "  ".join(caps)


class DashboardScreen(Screen[Any]):
    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
        Binding("c", "detail('cpu')", "CPU"),
        Binding("m", "detail('mem')", "Memory"),
        Binding("g", "detail('gpu')", "GPU"),
        # Toggle the JOB card's command/workdir between the elided root/…/leaf form
        # and the full path (so a deep path is readable/selectable on demand).
        Binding("p", "toggle_paths", "Full path", show=False),
        # Node switcher (multi-node jobs): TYPE the node's number to jump straight
        # to it — one digit for a small job, several for a big one (e.g. "199" on a
        # 200-node job), committing as soon as the number is unambiguous (or on
        # Enter / a brief pause). Left/Right step to the adjacent node. Digits
        # arrive as their own key name, so they bind cleanly.
        *[Binding(str(i), f"node_digit('{i}')", show=False) for i in range(10)],
        Binding("enter", "commit_node_input", show=False),
        Binding("backspace", "node_backspace", show=False),
        Binding("right", "next_node", "Next node", show=False),
        Binding("left", "prev_node", "Prev node", show=False),
        Binding("up", "scroll_up", "Up", show=False),
        Binding("down", "scroll_down", "Down", show=False),
        Binding("pageup", "page_up", "PgUp", show=False),
        Binding("pagedown", "page_down", "PgDn", show=False),
    ]

    CSS = """
    DashboardScreen { background: $surface; }

    #banner {
        height: auto;
        padding: 1 2 0 2;
    }

    /* The login-node-hop contention note (see MonitorNote): one line under the
       banner, shown only when this process is the monitor step. */
    #monitornote {
        height: auto;
        padding: 1 2 0 2;
    }

    /* The node-switch indicator sits at the very top so a switch is impossible
       to miss; it's hidden (display toggled in code) whenever no switch is in
       flight. */
    #switch {
        height: auto;
        padding: 1 2 0 2;
    }

    /* While a switch is in flight the body + bottom bar dim, so the previous
       node's still-on-screen numbers visibly recede behind the bright switch
       banner — no mistaking stale figures for the incoming node's. They pop
       back to full brightness the instant the new node's data lands. */
    #body.switching, #bottombar.switching {
        opacity: 55%;
    }

    /* The body fills the space between the banner and the docked bottom bar and
       scrolls internally when a many-GPU job overflows a short terminal, so the
       job-info + key bar stay pinned to the terminal floor (and visible) instead
       of floating mid-screen above dead space. */
    #body {
        padding: 1 2 0 2;
        height: 1fr;
        /* Keep the scrollbar (it's genuinely needed when a full 8-GPU node or a
           short/split terminal overflows the fold — otherwise rows below would be
           invisible with no hint), but make it a QUIET thin hint rather than a
           chunky bright bar: 1 cell wide, thumb dim until hovered, track blended
           into the surface. It stays hidden entirely (overflow-y: auto) whenever
           everything fits. */
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        scrollbar-color: $primary 30%;
        scrollbar-color-hover: $primary 65%;
        scrollbar-color-active: $primary;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
    }

    /* Titled, rounded cards give the dashboard structure — a lifted plane
       ($panel) + a hairline border reads as a raised card, not a flat slab. Each
       card wears its OWN frame hue so the sections read as distinct at a glance:
       RESOURCES the warm coral chrome accent (the live-data card), JOB a cool
       steel-blue (provenance). Both are chrome (a 55% hairline + a bold,
       full-strength title); steel-blue is deliberately NOT one of the data hues
       (CPU cyan / MEM rose / GPU violet) or a health colour (green/amber/red), so
       a frame is never mistaken for a reading — the JOB frame used to be violet,
       which collided with the GPU series. */
    #resources-panel, #job-panel {
        height: auto;
        background: $panel;
        border-title-style: bold;
        padding: 1 2;
    }
    #resources-panel {
        border: round $primary 55%;
        border-title-color: $primary;
    }
    #job-panel {
        margin-top: 1;
        border: round #6d8fce 55%;
        border-title-color: #6d8fce;
    }

    ResourceRows { height: auto; }

    /* Match the card plane ($panel), else the DataTable paints its own $surface
       and reads as a darker band across the RESOURCES card behind the GPU rows. */
    /* Sits directly under the "● GPU" section head ResourceRows now renders, as
       its per-device breakdown, so no top margin; a small left pad tucks the rows
       under that head instead of hugging the panel edge. */
    GpuTable { height: auto; margin-top: 0; padding-left: 2; background: $panel; }
    GpuTable > .datatable--header { background: $panel; }

    /* Docked to the terminal floor (wrapping job-info + key bar together so the
       first-in-DOM sits on top, not inverted) so the bottom bar is always where
       the eye expects it. */
    #bottombar {
        dock: bottom;
        height: auto;
        background: $panel;
    }

    #jobinfo {
        height: auto;
        padding: 1 3;
        border-top: solid $primary 30%;
        background: $panel;
    }

    /* On a short terminal the docked bar shrinks (no blank padding rows, no
       border, single line — see JobInfoBar.compact) so the RESOURCES gauges keep
       their rows instead of scrolling below the fold. */
    #bottombar.compact #jobinfo {
        padding: 0 3;
        border-top: none;
    }

    #keybar {
        height: 1;
        padding: 0 3;
        background: $panel;
    }
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        job_ctx: JobContext,
        config: SlurmwatchConfig | None = None,
    ) -> None:
        super().__init__()
        self.collector = collector
        self.job_ctx = job_ctx
        self.config = config or SlurmwatchConfig()
        self.latest_snapshot: TelemetrySnapshot | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self.title = "slurmwatch"
        # Node switcher: the list of nodes and which one the dashboard shows. The
        # local collector serves the node this process runs on; other nodes are
        # sampled on demand via srun (see slurmwatch.remote). Default to the local
        # node so the common single-node case never shells out.
        self._node_list: list[str] = list(job_ctx.nodelist_resolved) or (
            [job_ctx.hostname] if job_ctx.hostname else []
        )
        # Identify the local node tolerantly: Slurm's NodeName can differ from the
        # node's own gethostname by case or a kept domain suffix. Canonicalise
        # `_local_node` to the matching resolved-list entry so the poll loop's
        # "is this the live local node?" check (node == self._local_node) holds and
        # we use the fast collector instead of streaming our own node over srun.
        local = local_node_name()
        match = next((n for n in self._node_list if short_host(n) == short_host(local)), None)
        self._local_node = match or local
        self._selected_node = match or (self._node_list[0] if self._node_list else local)
        # Node switcher plumbing: a per-node cache of the last snapshot (so
        # switching back to a node shows instantly while it re-streams), plus the
        # single persistent stream for the node currently on screen (only one at a
        # time → O(1) in node count).
        self._node_cache: dict[str, TelemetrySnapshot] = {}
        self._stream_proc: asyncio.subprocess.Process | None = None
        self._stream_node: str | None = None
        # Consecutive stream failures for the current node, for exponential
        # backoff — an unreachable node must not respawn srun every tick.
        self._stream_fails = 0
        # Node-switch feedback: while a switch is in flight `_switch_target` names
        # the node we're waiting on, `_switch_started` stamps when (to nudge the
        # banner to a "still attaching" note if Slurm is slow), and a paused
        # interval drives the spinner. The switch clears the instant that node's
        # first real snapshot is shown (see `_show`), not on a timer.
        self._switch_target: str | None = None
        self._switch_started: float | None = None
        self._spinner_timer: Any = None
        # Latched once the job has ended (#28): the poll loop has stopped, the last
        # numbers are frozen on screen, so node switching is disabled — otherwise a
        # switch would dim the frozen screen with no way to un-dim (nothing delivers
        # a frame to clear it) and could swap in another node's stale data (#50).
        self._job_ended = False
        # "p" toggles the JOB card's command/workdir between elided and full.
        self._paths_full = False
        # Login-node hop: this process IS the monitor step (env set by the hop),
        # so it holds a job step that can block a new srun/mpirun. When set, show
        # the contention note and run the best-effort stalled-launch detector.
        self._monitor_step = os.environ.get("SLURMWATCH_MONITOR_STEP") == "1"
        self._stuck_polls = 0
        # Digits typed toward a "go to node N" jump, plus the pause-timer that
        # commits an ambiguous prefix (e.g. "1" on a 200-node job) if no more
        # digits follow. Cleared on commit/cancel.
        self._node_input = ""
        self._node_input_timer: Any = None

    @property
    def resource_rows(self) -> ResourceRows | None:
        try:
            return self.query_one(ResourceRows)
        except NoMatches:
            return None

    @property
    def cpu_history(self) -> deque[float]:
        rows = self.resource_rows
        return rows.cpu_history if rows is not None else deque()

    @property
    def mem_history(self) -> deque[float]:
        rows = self.resource_rows
        return rows.mem_history if rows is not None else deque()

    def compose(self) -> ComposeResult:
        # Blank the header icon: Textual's default "⭘" is a decorative
        # command-palette button that means nothing in this keyboard-driven tool.
        yield Header(show_clock=False, icon=" ")
        yield SwitchBanner(id="switch")
        yield StatusBanner(id="banner")
        yield MonitorNote(id="monitornote")
        # Titled, rounded cards stacked in the scrolling body. RESOURCES carries
        # the live gauges (each row has its own recent-range tag, so there's no
        # separate TRENDS card); JOB adds only the provenance the rest of the UI
        # doesn't show (account/qos/state/command/workdir/queue-wait) — never an
        # allocation number, which the rows + bottom bar already carry.
        with VerticalScroll(id="body"):
            with Vertical(id="resources-panel") as res:
                res.border_title = "RESOURCES"
                yield ResourceRows()
                yield GpuTable()
            with Vertical(id="job-panel") as job:
                job.border_title = f"JOB · {_escape_markup(str(self.job_ctx.job_id))}"
                yield JobDetailsPanel()
        keys = [
            ("q", "Quit", _ACCENT),
            ("c", "CPU", _CPU_COLOR),
            ("m", "Memory", _MEM_COLOR),
            ("g", "GPU", _GPU_COLOR),
        ]
        # NB: the "p" full-path toggle is NOT advertised here — it's only useful
        # when a path is actually elided, so its hint lives inline in the JOB card
        # next to the truncated path (see JobDetailsPanel), not always at the foot.
        # Only a multi-node job can switch nodes, so only then advertise the keys.
        # You TYPE a node number to jump straight there (1-N, any length — "199"
        # on a 200-node job), and ◂ ▸ step to the adjacent node; the cap shows the
        # full typeable range so no node ever looks unreachable.
        n_nodes = len(self._node_list)
        if n_nodes > 1:
            ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode
            arrows = " <>" if ascii_mode else " ◂▸"
            cap = f"1-{n_nodes}{arrows if n_nodes > 9 else ''}"
            keys.append((cap, "Node", _GPU_VRAM_COLOR))
        with Vertical(id="bottombar"):
            yield JobInfoBar(id="jobinfo")
            yield KeyFooter(keys, id="keybar")

    def on_mount(self) -> None:
        self.query_one(GpuTable).display = False
        self.query_one(SwitchBanner).display = False
        note = self.query_one(MonitorNote)
        note.display = self._monitor_step
        note.node = short_host(self._local_node) if self._local_node else ""
        note.ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode
        # Runs only while a switch is in flight (resumed in `_begin_switch`,
        # paused in `_end_switch`); ~8fps reads as smooth spinner motion.
        self._spinner_timer = self.set_interval(0.12, self._tick_switch, pause=True)
        self._apply_compact(self.app.size.height)
        self._update_header(None)
        self._poll_task = asyncio.create_task(self._poll_loop())

    def on_resize(self, event: Any) -> None:
        # Collapse the docked bottom bar on a short terminal so the RESOURCES
        # gauges keep their rows (see `_apply_compact`).
        self._apply_compact(event.size.height)

    def _apply_compact(self, height: int) -> None:
        compact = 0 < height < _COMPACT_HEIGHT
        with contextlib.suppress(NoMatches):
            bar = self.query_one(JobInfoBar)
            if bar.compact != compact:
                bar.compact = compact
                bar.refresh(layout=True)
        with contextlib.suppress(NoMatches):
            self.query_one("#bottombar").set_class(compact, "compact")

    async def on_unmount(self) -> None:
        # Await the cancelled poll task, then stop the stream, so the remote
        # streaming subprocess (slurmwatch.remote) is killed and reaped *before*
        # the event loop tears down — otherwise its transport is GC'd after the
        # loop closes and prints a spurious "Event loop is closed" on exit.
        task = self._poll_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._stop_stream()

    async def _stop_stream(self) -> None:
        """Kill and reap the current remote stream, if any."""
        proc = self._stream_proc
        self._stream_proc = None
        self._stream_node = None
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()

    async def _read_remote(self, node: str) -> TelemetrySnapshot | None:
        """One snapshot from ``node``'s stream, launching/replacing it as needed."""
        if self._stream_node != node or self._stream_proc is None:
            await self._stop_stream()
            interval = max(self.config.poll_interval, 1.0)
            self._stream_proc = await open_stream(
                self.job_ctx.raw_job_id or self.job_ctx.job_id, node, interval
            )
            self._stream_node = node if self._stream_proc is not None else None
        proc = self._stream_proc
        if proc is None or proc.stdout is None:
            await self._stream_backoff(node)  # couldn't launch; back off before retry
            return None
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
        except TimeoutError:
            return None
        if not line:  # EOF — the stream died
            await self._stop_stream()
            # Back off before the next relaunch: a node that keeps dying
            # immediately (draining, --overlap denied, gone) would otherwise
            # respawn srun every tick and storm the scheduler. The stuck-switch
            # watchdog still surfaces the amber "still reaching" warning at 12s.
            await self._stream_backoff(node)
            return None
        self._stream_fails = 0  # a real frame arrived — reset the backoff
        return parse_snapshot_line(line)

    async def _stream_backoff(self, node: str) -> None:
        """Sleep with exponential backoff (1→8s cap) after a failed/short-lived
        stream, so an unreachable node can't respawn srun on every poll tick.

        Sleeps in short slices and returns early the moment the user switches away
        (``_selected_node`` changes), so leaving a dead node is never blocked for
        the full backoff — switching off an unreachable node stays responsive.
        """
        self._stream_fails += 1
        # Cap the EXPONENT, not just the result: after ~1000 failures on a node
        # that keeps dying, `2.0 ** (fails - 1)` would raise OverflowError before
        # the min() could clamp it, killing the backoff. The cap of 3 already
        # reaches the 8s ceiling (2**3 == 8).
        remaining = min(2.0 ** min(self._stream_fails - 1, 3), 8.0)
        while remaining > 0 and self._selected_node == node:
            await asyncio.sleep(min(0.1, remaining))
            remaining -= 0.1

    async def _poll_loop(self) -> None:
        try:
            while True:
                if self.collector.job_ended:
                    # The job left Slurm while we were attached: show the final
                    # notice, keep the last numbers on screen, and stop polling.
                    # The app stays open until the user quits (#28).
                    self._show_job_ended()
                    break
                try:
                    node = self._selected_node
                    if node == self._local_node:
                        # The node this process runs on: live, from the collector.
                        await self._stop_stream()
                        snapshot = await asyncio.wait_for(
                            self.collector.next_snapshot(), timeout=0.3
                        )
                        self._show(snapshot, node)
                    else:
                        # A different node: streamed via srun (~1 snapshot/s once
                        # launched). Ignore a frame that arrives after the user
                        # switched away.
                        snap = await self._read_remote(node)
                        if snap is not None and self._selected_node == node:
                            self._show(snap, node)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
                except Exception:
                    # A transient failure in one update must not silently kill
                    # the whole poll task and freeze the UI (B-C7); log via the
                    # Textual app log and keep polling.
                    self.log.error("dashboard poll iteration failed", exc_info=True)
        except asyncio.CancelledError:
            pass

    def _show(self, snap: TelemetrySnapshot, node: str) -> None:
        """Cache a frame under the node it represents and render it — if on screen.

        ``node`` is the node the poll loop *requested* this frame for, which is
        authoritative: ``snapshot.hostname`` is the serving host's own
        ``gethostname`` and can differ from Slurm's NodeName (aliases, a kept
        domain, case) on some clusters, so keying/gating on ``node`` — not on the
        self-reported hostname — is what keeps the dashboard from blanking there.

        A frame is always cached (so a re-visit is instant), but only *rendered*
        when it belongs to the currently-selected node. This drops a stale frame
        already in flight for the node we just switched away from — otherwise it
        would land a beat after the switch and overwrite the new node's view with
        the old node's numbers (the bug that made a switch look like it got
        stuck). Arrival of the selected node's own frame also *ends* a pending
        switch.
        """
        self._node_cache[node] = snap
        if node != self._selected_node:
            return
        if self._switch_target is not None:
            self._end_switch()
        self.latest_snapshot = snap
        self._update_widgets(snap)

    def _set_node(self, node: str) -> None:
        """Switch the dashboard to ``node``, with immediate, unmistakable feedback."""
        # Once the job has ended the poll loop is gone, so a switch could never be
        # un-dimmed and would only corrupt the frozen final view — ignore it (#50).
        if self._job_ended:
            return
        if node not in self._node_list or node == self._selected_node:
            return
        # Any switch (arrow, or the commit of a typed number) cancels a
        # half-typed "go to node" buffer + its pause timer, so a stale timer can't
        # fire later and yank the view to the abandoned number.
        self._clear_node_input()
        self._selected_node = node
        self._stream_fails = 0  # a fresh node gets fresh stream attempts (no carried backoff)
        # A fresh node starts a fresh 60s history so the row range tags reflect
        # only the node now on screen, and drop any queued local frames so a
        # later switch back to the local node shows a current one, not a backlog.
        rows = self.resource_rows
        if rows is not None:
            rows.cpu_history.clear()
            rows.mem_history.clear()
            rows.gpu_history.clear()
        with contextlib.suppress(Exception):
            while True:
                self.collector.queue.get_nowait()
        # Show the animated "switching to node N" banner for EVERY switch, so the
        # key press is confirmed the same way in both directions (a switch back to
        # the local node used to show nothing, which read as "did my key work?").
        # It clears the instant that node's real data lands (see `_show`) — for
        # the local node that's sub-second, so it's a brief, symmetric flash, not
        # a wait; for a remote node it stays up through the Slurm step launch.
        self._begin_switch(node)
        # A re-visit is instant: show the node's last-seen snapshot from cache
        # (correct node, dimmed while the banner reads "switching") while its
        # stream re-attaches. A first visit has no cache, so the previous view
        # just dims until the first frame lands.
        cached = self._node_cache.get(node)
        if cached is not None:
            self.latest_snapshot = cached
            self._update_widgets(cached)

    def _begin_switch(self, node: str) -> None:
        """Enter the 'switching' state: show + animate the banner, dim the body."""
        self._switch_target = node
        self._switch_started = time.monotonic()
        idx = self._node_list.index(node) + 1
        with contextlib.suppress(NoMatches):
            banner = self.query_one(SwitchBanner)
            banner.prompt = ""  # a switch supersedes any half-typed "go to node" input
            banner.target_label = f"node {idx} of {len(self._node_list)}"
            banner.node = node
            banner.frame = 0
            banner.slow = False
            banner.stuck = False
            banner.ascii = (self.config or SlurmwatchConfig()).ascii_mode
            banner.display = True
            banner.refresh(layout=True)
        for sel in ("#body", "#bottombar"):
            with contextlib.suppress(NoMatches):
                self.query_one(sel).add_class("switching")
        if self._spinner_timer is not None:
            with contextlib.suppress(Exception):
                self._spinner_timer.resume()

    def _end_switch(self) -> None:
        """Leave the 'switching' state: hide the banner, undim, stop the spinner."""
        self._switch_target = None
        self._switch_started = None
        if self._spinner_timer is not None:
            with contextlib.suppress(Exception):
                self._spinner_timer.pause()
        with contextlib.suppress(NoMatches):
            self.query_one(SwitchBanner).display = False
        for sel in ("#body", "#bottombar"):
            with contextlib.suppress(NoMatches):
                self.query_one(sel).remove_class("switching")

    def _show_job_ended(self) -> None:
        """The monitored job has finished: show a final, static banner and stop.

        Any in-flight switch state is cleared, the spinner is stopped, and the
        banner turns into a persistent "JOB ENDED" notice. The last real
        snapshot stays on screen (frozen), and the poll loop exits — the app
        stays open so the user can read the final numbers and press q (#28).
        """
        # Disable node switching from here on: the poll loop has stopped, so a
        # switch's dim could never be cleared and would only corrupt the frozen
        # final view (#50).
        self._job_ended = True
        # Cancel any half-typed "go to node" input + its pause timer, so a timer
        # that fires ~0.9s later can't hide the JOB ENDED notice it's replacing.
        self._clear_node_input()
        self._end_switch()
        with contextlib.suppress(NoMatches):
            banner = self.query_one(SwitchBanner)
            banner.ended = True
            banner.ended_job = str(self.job_ctx.job_id)
            # The ended notice can render outside the node-switch path (which is the
            # only place that otherwise sets banner.ascii), so set it here too — else
            # its glyphs (⚑, dash, dot) leak under --ascii.
            banner.ascii = (self.config or SlurmwatchConfig()).ascii_mode
            banner.display = True
            banner.refresh(layout=True)

    def _tick_switch(self) -> None:
        """Advance the switch banner's spinner and escalate if the attach stalls."""
        if self._switch_target is None:
            return
        waited = time.monotonic() - self._switch_started if self._switch_started else 0.0
        # Past the "stuck" threshold, stop blocking: un-dim the body and flip the
        # banner to an amber "still reaching / retrying" warning, so an unreachable
        # or busy node never leaves the session frozen on a dim, spinning screen.
        if waited > _SWITCH_STUCK_S:
            self._mark_switch_stuck()
            return
        with contextlib.suppress(NoMatches):
            banner = self.query_one(SwitchBanner)
            banner.frame += 1
            # After a few seconds a slow Slurm attach gets a reassuring note so
            # the wait reads as "still working", not "wedged". The note widens the
            # line (and may wrap on a narrow terminal), so recompute height once
            # when it first appears — a plain refresh would clip it.
            if waited > _SWITCH_SLOW_S and not banner.slow:
                banner.slow = True
                banner.refresh(layout=True)
            else:
                banner.refresh()

    def _mark_switch_stuck(self) -> None:
        """The attach is taking long enough to look unreachable: stop blocking.

        Un-dim the body (so its last-known data is readable again) and turn the
        banner into a static warning, but keep ``_switch_target`` set — the poll
        loop keeps retrying and a frame that finally lands still clears it via
        `_show`. Acts only on the transition (the warning doesn't animate, so the
        spinner timer pauses until the next switch resumes it).
        """
        already_stuck = True
        with contextlib.suppress(NoMatches):
            banner = self.query_one(SwitchBanner)
            already_stuck = banner.stuck
            if not banner.stuck:
                banner.stuck = True
                banner.refresh(layout=True)
        if already_stuck:
            return
        for sel in ("#body", "#bottombar"):
            with contextlib.suppress(NoMatches):
                self.query_one(sel).remove_class("switching")
        if self._spinner_timer is not None:
            with contextlib.suppress(Exception):
                self._spinner_timer.pause()

    def _switch_node(self, step: int) -> None:
        """Move the shown node by ``step`` (wrapping); no-op unless multi-node."""
        if len(self._node_list) <= 1:
            return
        cur = (
            self._node_list.index(self._selected_node)
            if self._selected_node in self._node_list
            else 0
        )
        self._set_node(self._node_list[(cur + step) % len(self._node_list)])

    def action_node_digit(self, d: str) -> None:
        """Type a digit toward a "go to node N" jump.

        Accumulates digits and commits the moment the number can't be a prefix of
        any larger valid node (so "5" jumps at once on a 6-node job, "199" jumps on
        the third digit of a 200-node job); an ambiguous prefix (e.g. "1" on a
        200-node job) commits on Enter or after a brief pause. A digit that would
        overshoot the node count restarts the buffer, so a fat-finger can't wedge it.
        """
        if self._job_ended:
            return  # node switching is disabled after the job ends (#50)
        n_nodes = len(self._node_list)
        if n_nodes <= 1:
            return
        candidate = self._node_input + d
        if int(candidate) == 0 or int(candidate) > n_nodes:
            candidate = d  # overshoot / leading zero → start fresh from this digit
            if int(candidate) == 0 or int(candidate) > n_nodes:
                return  # even this digit alone is out of range — ignore it
        self._node_input = candidate
        # Unambiguous once no larger number could still be forming (n*10 > count).
        if int(candidate) * 10 > n_nodes:
            self.action_commit_node_input()
        else:
            self._show_node_prompt()
            self._cancel_node_timer()  # replace, don't stack, the pause-commit timer
            with contextlib.suppress(Exception):
                self._node_input_timer = self.set_timer(0.9, self.action_commit_node_input)

    def _cancel_node_timer(self) -> None:
        if self._node_input_timer is not None:
            with contextlib.suppress(Exception):
                self._node_input_timer.stop()
            self._node_input_timer = None

    def action_node_backspace(self) -> None:
        if not self._node_input:
            return
        self._node_input = self._node_input[:-1]
        if self._node_input:
            self._show_node_prompt()
            # Re-arm the pause timer from the correction, not the earlier keystroke,
            # so editing doesn't get pre-empted by a premature auto-commit.
            self._cancel_node_timer()
            with contextlib.suppress(Exception):
                self._node_input_timer = self.set_timer(0.9, self.action_commit_node_input)
        else:
            self._clear_node_input()

    def action_commit_node_input(self) -> None:
        """Jump to the typed node number (if any), then clear the input."""
        buf = self._node_input
        self._clear_node_input()
        if buf and 1 <= int(buf) <= len(self._node_list):
            self._set_node(self._node_list[int(buf) - 1])

    def _show_node_prompt(self) -> None:
        with contextlib.suppress(NoMatches):
            banner = self.query_one(SwitchBanner)
            banner.prompt = self._node_input
            banner.total = str(len(self._node_list))  # "go to node N of <count>"
            banner.display = True
            banner.refresh(layout=True)

    def _clear_node_input(self) -> None:
        self._node_input = ""
        self._cancel_node_timer()
        # Only hide the banner if it's showing the prompt (not an in-flight switch
        # and not the terminal JOB ENDED notice, which must stay up — #50/#3).
        with contextlib.suppress(NoMatches):
            banner = self.query_one(SwitchBanner)
            if banner.prompt:
                banner.prompt = ""
                if self._switch_target is None and not banner.ended:
                    banner.display = False
                banner.refresh(layout=True)

    def action_next_node(self) -> None:
        self._switch_node(1)

    def action_prev_node(self) -> None:
        self._switch_node(-1)

    def _effective_interval(self) -> float:
        """Seconds between the samples currently filling the history deques.

        The local node is served by the collector at ``poll_interval``; a remote
        node is streamed at ``max(poll_interval, 1.0)`` (see ``_read_remote``).
        Sizing the history to the *displayed* node's cadence keeps the deque
        holding exactly ``history_seconds`` of data, so the row trend range tag
        ("… over 60s") and the drill-in chart ("last 60s") aren't mislabelled — a
        remote 1s stream in a 0.5s-sized (120-slot) deque spanned ~120s while the
        UI claimed 60s (#55)."""
        base = max(self.config.poll_interval, 0.01)
        if self._selected_node != self._local_node:
            return max(base, 1.0)
        return base

    def _history_maxlen(self) -> int:
        return max(int(round(self.config.history_seconds / self._effective_interval())), 10)

    @staticmethod
    def _resize(hist: deque[float], maxlen: int) -> deque[float]:
        if hist.maxlen != maxlen:
            return deque(hist, maxlen=maxlen)
        return hist

    def _update_widgets(self, snapshot: TelemetrySnapshot) -> None:
        self.latest_snapshot = snapshot
        self._update_header(snapshot)
        maxlen = self._history_maxlen()

        with contextlib.suppress(NoMatches):
            banner = self.query_one(StatusBanner)
            banner.snapshot = snapshot
            banner.config = self.config
            # The banner is now an alarm-only strip: hide it entirely when there's
            # nothing wrong, so a healthy job doesn't carry an empty padded row at
            # the top. layout=True so the auto-height widget resizes when it does
            # have content.
            banner.display = bool(str(banner.render()).strip())
            banner.refresh(layout=True)

        # Escalate the monitor-step note when a launch looks stuck behind our own
        # held step (a launcher present in the job while CPU stays idle). Gated to
        # the local node — a switched-to remote node's frame says nothing about
        # the step we hold here.
        if self._monitor_step:
            with contextlib.suppress(NoMatches):
                note = self.query_one(MonitorNote)
                local_view = self._selected_node == self._local_node
                launcher = local_view and getattr(self.collector, "launcher_present", False)
                if launcher and snapshot.cpu.usage_percent < _STUCK_CPU_PCT:
                    self._stuck_polls += 1
                else:
                    self._stuck_polls = 0
                escalated = self._stuck_polls >= _STUCK_POLLS
                if note.escalated != escalated:
                    note.escalated = escalated
                    note.refresh(layout=True)

        with contextlib.suppress(NoMatches):
            rows = self.query_one(ResourceRows)
            rows.snapshot = snapshot
            rows.config = self.config
            rows.cpu_history = self._resize(rows.cpu_history, maxlen)
            rows.cpu_history.append(snapshot.cpu.usage_percent)
            rows.mem_history = self._resize(rows.mem_history, maxlen)
            rows.mem_history.append(_mem_ws_pct(snapshot.memory))
            for gpu in snapshot.gpus:
                hist = rows.gpu_history.get(gpu.index)
                if hist is None:
                    hist = deque(maxlen=maxlen)
                    rows.gpu_history[gpu.index] = hist
                else:
                    rows.gpu_history[gpu.index] = self._resize(hist, maxlen)
                rows.gpu_history[gpu.index].append(gpu.utilization_percent)
            # 3+ GPUs go in the DataTable; 1-2 stay as scannable rows.
            use_table = len(snapshot.gpus) >= 3
            rows.gpu_table_active = use_table
            rows.refresh(layout=True)

        with contextlib.suppress(NoMatches):
            table = self.query_one(GpuTable)
            if len(snapshot.gpus) >= 3:
                table.display = True
                rr = self.resource_rows
                table.update_gpus(
                    snapshot.gpus, self.config, rr.gpu_history if rr is not None else None
                )
            else:
                table.display = False

        with contextlib.suppress(NoMatches):
            job = self.query_one(JobDetailsPanel)
            job.job_ctx = self.job_ctx
            job.config = self.config
            job.full_paths = self._paths_full
            job.refresh(layout=True)

        with contextlib.suppress(NoMatches):
            info = self.query_one(JobInfoBar)
            info.snapshot = snapshot
            info.job_ctx = self.job_ctx
            info.config = self.config
            info.refresh(layout=True)

    def _update_header(self, snapshot: TelemetrySnapshot | None) -> None:
        # The full, labelled job identity + time budget live in the JobInfoBar at
        # the bottom; the header just carries a short anchor so it isn't a cryptic
        # unlabelled string.
        ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode
        if snapshot is None:
            dots = "..." if ascii_mode else "…"
            self.sub_title = f"connecting to job {self.job_ctx.job_id}{dots}"
            return
        sep = "-" if ascii_mode else "·"
        self.sub_title = f"job {snapshot.job_id} {sep} {self.job_ctx.username}"

    def action_quit(self) -> None:
        self.app.exit()

    def action_toggle_paths(self) -> None:
        """Toggle the JOB card's command/workdir between elided and full."""
        self._paths_full = not self._paths_full
        with contextlib.suppress(NoMatches):
            job = self.query_one(JobDetailsPanel)
            job.full_paths = self._paths_full
            job.refresh(layout=True)

    def action_detail(self, resource: str) -> None:
        if self.latest_snapshot is not None:
            self.app.push_screen(ResourceDetailScreen(self, resource))

    # The body fills the space above the docked bottom bar and scrolls internally
    # when a many-GPU job overflows a short terminal, so these keys drive the body
    # (not the screen) and the bottom bar stays pinned and visible.
    def _scroll_body(self) -> VerticalScroll | None:
        try:
            return self.query_one("#body", VerticalScroll)
        except NoMatches:
            return None

    def action_scroll_up(self) -> None:
        if (body := self._scroll_body()) is not None:
            body.scroll_up(animate=False)

    def action_scroll_down(self) -> None:
        if (body := self._scroll_body()) is not None:
            body.scroll_down(animate=False)

    def action_page_up(self) -> None:
        if (body := self._scroll_body()) is not None:
            body.scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        if (body := self._scroll_body()) is not None:
            body.scroll_page_down(animate=False)


class JobSelectorScreen(ModalScreen[str]):
    BINDINGS: ClassVar = [
        Binding("enter", "select_job", "Select"),
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]

    CSS = """
    JobSelectorScreen { align: center middle; }

    #selector-box {
        width: 70;
        height: auto;
        border: round $primary;
        padding: 1;
    }

    #selector-title { text-style: bold; padding-bottom: 1; }

    ListView { height: auto; max-height: 20; }
    ListItem { padding: 0 1; }
    ListItem:hover { background: $accent; }
    """

    def __init__(self, jobs: list[dict[str, object]]) -> None:
        super().__init__()
        self.jobs = jobs

    def compose(self) -> ComposeResult:
        with Vertical(id="selector-box"):
            yield Static(
                f"Select a job ({len(self.jobs)} found):",
                id="selector-title",
            )
            yield ListView(*[ListItem(Static(self._job_line(j))) for j in self.jobs])

    @staticmethod
    def _job_line(j: dict[str, object]) -> str:
        # The job name (%j) is free-form and user-controlled (`sbatch -J`), so
        # every interpolated value must be neutralized before it reaches the
        # markup parser (F1); only our own [colour] styling is trusted markup.
        def field(key: str, default: str = "?") -> str:
            return _escape_markup(str(j.get(key, default)))

        pending = str(j.get("state", "")).upper() in ("PD", "PENDING")
        # A coloured state tag so running vs pending is obvious at a glance; for a
        # pending job the elapsed "time" is 0, so show its scheduler reason instead.
        tag_color = _HEALTH_COLOR["warn"] if pending else _HEALTH_COLOR["ok"]
        tag = "PENDING" if pending else "RUNNING"
        tail = f"why={field('reason')}" if pending else f"time={field('wall_time')}"
        return (
            f"[bold]{_escape_markup(str(j['job_id']))}[/]  "
            f"[{tag_color}]{tag}[/]  "
            f"{field('partition')}  "
            f"{field('name')}  "
            f"nodes={field('nodes')}  "
            f"{tail}"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # ListView has focus, so its own enter binding fires (posting this
        # message) instead of the screen-level binding; mouse clicks arrive
        # here too.
        event.stop()
        self.action_select_job()

    def action_select_job(self) -> None:
        lv = self.query_one(ListView)
        if lv.index is not None and lv.index < len(self.jobs):
            self.dismiss(str(self.jobs[lv.index]["job_id"]))

    def action_cancel(self) -> None:
        self.dismiss("")


class PendingView(Static):
    """The pending-job report — why it's waiting, when it might start, and where
    in the cluster it could run.

    A pure render over the resolved data (so it's easy to test); the screen
    re-feeds it on a refresh timer. Never touches the running-job dashboard — this
    is a separate, read-only view shown only for a PENDING job (#60).
    """

    job: PendingJob | None = None
    partitions: list[PartitionResources] = []
    queue_running: int = 0
    queue_pending: int = 0
    # (rank, total) among the partition's pending jobs by priority; None = unknown.
    queue_rank: tuple[int, int] | None = None
    # Advances the "calculating…" spinner while there's no estimate yet (driven by
    # PendingScreen's timer).
    frame: int = 0
    config: SlurmwatchConfig | None = None

    # Cap the WHERE table so a pathological (unfiltered) list can't flood the
    # screen — but high enough that a normal account's access-filtered set shows
    # in full (no silly "… and 1 more"). Current partition + fits always kept.
    _MAX_ROWS = 24

    def render(self) -> str:
        job = self.job
        if job is None:
            return "[dim]resolving pending job…[/]"
        ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode
        arrow = "->" if ascii_mode else "▸"
        return "\n\n".join(
            [
                self._why(job, ascii_mode),
                self._when(job, ascii_mode),
                self._where(job, ascii_mode, arrow),
            ]
        )

    def _why(self, job: PendingJob, ascii_mode: bool) -> str:
        # Each section wears its own hue so the three read as distinct bands at a
        # glance (coral / cyan / violet); Title Case, not shouty all-caps.
        head = f"[bold {_ACCENT}]Why It's Waiting[/]"
        reason = job.reason.strip()
        code = ""
        if reason and reason not in ("None", "(null)"):
            code = f"  {_sep(ascii_mode)}  [{_INK}]{_escape_markup(reason)}[/]"
        state = f"  {_dot('warn', ascii_mode)} [bold {_HEALTH_COLOR['warn']}]PENDING[/]{code}"
        why = f"  [{_DIM}]{_escape_markup(explain_reason(job.reason))}[/]"
        # The job's own request, colour-coded by resource, right where the reason is
        # — so the user can read "what I asked for" against WHERE's "what's free".
        # "(total)" spells out that these are whole-job totals, not per-node (the
        # ambiguity users hit: "20 CPU per node or in total?").
        req = f"  [{_DIM}]requested (total)[/]  {self._req_chips(job, ascii_mode)}"
        return f"{head}\n{state}\n{why}\n{req}"

    def _when(self, job: PendingJob, ascii_mode: bool) -> str:
        head = f"[bold {_CPU_COLOR}]When It Might Start[/]"
        now = time.time()
        # Each row's key figure gets its own colour so the four don't read as one
        # grey wall: start time cyan, wait rose, queue spot violet, running/pending
        # green/amber.
        ok, warn = _HEALTH_COLOR["ok"], _HEALTH_COLOR["warn"]
        dash = "-" if ascii_mode else "—"
        dots = "..." if ascii_mode else "…"
        # A held / dependency / begin-time / reservation job isn't being
        # priority-scheduled, so a "calculating" estimate and a queue position are
        # both meaningless (a held job has priority 0 → a bogus "everyone ahead").
        held = is_held_like(job.reason)
        lines: list[str] = []
        est = job.start_time_estimate
        if est is not None and est >= now - 1:
            rel = _format_wait(max(0, int(est - now)))
            lines.append(
                f"  [{_DIM}]estimated start[/]  [bold {_CPU_COLOR}]{_format_clock(est)}[/] "
                f"[{_DIM}](in ~{rel} {dash} scheduler estimate, may change)[/]"
            )
        elif held:
            lines.append(
                f"  [{_DIM}]estimated start[/]  "
                f"[{_FAINT}]{dash} not scheduled while blocked (see the reason above)[/]"
            )
        else:
            # No estimate yet: an animated spinner says the scheduler is still
            # working on it, not that it's a permanent dead end.
            spin = _SPIN_FRAMES_ASCII if ascii_mode else _SPIN_FRAMES
            glyph = spin[self.frame % len(spin)]
            lines.append(
                f"  [{_DIM}]estimated start[/]  [{_CPU_COLOR}]{glyph}[/] "
                f"[{_FAINT}]calculating{dots} "
                f"(the scheduler estimates a start once the job has waited a few minutes)[/]"
            )
        if job.submit_time is not None:
            waited = _format_wait(max(0, int(now - job.submit_time)))
            lines.append(
                f"  [{_DIM}]submitted[/]  [{_INK}]{_format_clock(job.submit_time)}[/] "
                f"[{_DIM}]{_sep(ascii_mode)} waiting[/] [{_MEM_COLOR}]{waited}[/] [{_DIM}]so far[/]"
            )
        # A queue position people actually understand — "#266 of 475, 265 ahead" —
        # not the opaque raw priority score. Held-like jobs aren't priority-ordered,
        # so the position would be bogus; skip it for them.
        if self.queue_rank is not None and not held:
            rk, tot = self.queue_rank
            ahead = max(0, rk - 1)
            job_word = "job" if ahead == 1 else "jobs"
            lines.append(
                f"  [{_DIM}]in line[/]  [bold {_GPU_COLOR}]#{rk} of {tot}[/] "
                f"[{_DIM}]{_sep(ascii_mode)} {ahead} higher-priority {job_word} ahead of yours[/]"
            )
        lines.append(
            f"  [{_DIM}]queue on[/] [{_INK}]{_escape_markup(job.partition)}[/]  "
            f"[{ok}]{self.queue_running}[/] [{_DIM}]running[/] {_sep(ascii_mode)} "
            f"[{warn}]{self.queue_pending}[/] [{_DIM}]pending[/]"
        )
        return head + "\n" + "\n".join(lines)

    def _req_chips(self, job: PendingJob, ascii_mode: bool = False) -> str:
        """The job's request, each resource in its dashboard identity colour (nodes
        ink, CPU cyan, memory rose, GPU violet) so it maps to the live gauges."""
        # For an --exclusive job say "whole nodes": it gets every core on each node,
        # so the CPU floor below isn't the whole story.
        node_txt = (
            _plural(job.req_nodes, "whole node")
            if job.exclusive
            else _plural(job.req_nodes, "node")
        )
        bits = [
            f"[{_INK}]{node_txt}[/]",
            f"[{_CPU_COLOR}]{job.req_cpus} CPU[/]",
        ]
        if job.req_mem_bytes > 0:
            # One decimal so a sub-GiB request (e.g. 512 MiB) doesn't render "0 GiB".
            bits.append(f"[{_MEM_COLOR}]{_gib(job.req_mem_bytes):.1f} GiB[/]")
        if job.req_gpus > 0:
            bits.append(
                f"[{_GPU_COLOR}]{job.req_gpus}x {_escape_markup(job.req_gpu_type or 'GPU')}[/]"
            )
        return f"  {_sep(ascii_mode)}  ".join(bits)

    def _where(self, job: PendingJob, ascii_mode: bool, arrow: str) -> str:
        # The request now lives in the WHY section; WHERE is just the header + the
        # capacity list to read against it.
        head = f"[bold {_GPU_COLOR}]Where It Could Run[/]"
        parts = self.partitions
        if not parts:
            return head + f"\n  [{_FAINT}]cluster partition info unavailable[/]"

        # Keep the current partition + any fitting alternatives, then fill up to the
        # cap with the rest (already sorted by free capacity). Note if truncated.
        # The current partition is where the job is PENDING, so by definition it does
        # NOT fit now (else it'd be running) — force it False so we never print the
        # self-contradictory "FITS NOW (current)".
        fits = {p.name: (False if p.is_current else partition_fits_now(job, p)) for p in parts}
        kept: list[PartitionResources] = []
        for p in parts:
            if p.is_current or fits[p.name]:
                kept.append(p)
        for p in parts:
            if len(kept) >= self._MAX_ROWS:
                break
            if p not in kept:
                kept.append(p)
        dropped = len(parts) - len(kept)

        ok, faint = _HEALTH_COLOR["ok"], _FAINT
        dash = "-" if ascii_mode else "—"
        # A labelled header so no number is a mystery ("0 empty" -> "empty nodes:
        # 0"). A whole-node (--exclusive) or GPU job needs fully-EMPTY nodes; a
        # plain job, nodes with room. Right-aligned numerics keep columns aligned.
        node_hdr = "empty nodes" if (job.exclusive or job.req_gpus > 0) else "free nodes"
        rows: list[str] = [
            f"  [{_DIM}]{'partition':<16}{node_hdr:>12}  {'idle cores':>10}   "
            f"{'gpu':<12}can run now?[/]"
        ]
        ell = "..." if ascii_mode else "…"
        for p in kept:
            full = p.name + ("*" if p.is_current else "")
            # Reserve room for the ellipsis so the elided name is EXACTLY 16 cells
            # in both modes ("…" is 1 char, "..." is 3) — else ascii overflowed the
            # column and shoved every following column out of alignment.
            name_plain = full if len(full) <= 16 else full[: 16 - len(ell)] + ell
            ncolor = _MEM_COLOR if p.is_current else _INK
            gpus = format_gpu_types(p.gpu_types, 12, ascii_mode, has_gpus=p.has_gpus)
            navail = available_node_count(job, p)
            if p.is_current:
                # Pending here → not "fits now"; say it's where the job waits.
                mark = f"[{_DIM}]waiting (current)[/]"
            elif fits[p.name]:
                mark = f"[{ok}]{'YES' if not ascii_mode else 'yes'} {arrow}[/]"
            elif not p.available:
                mark = f"[{faint}]down[/]"
            else:
                # "no room" not "full": an exclusive/GPU job can see idle cores yet
                # no empty node, so "full" read as a contradiction.
                mark = f"[{faint}]no room[/]"
            rows.append(
                f"  [{ncolor}]{name_plain:<16}[/][{_DIM}]{navail:>12}  {p.cpus_idle:>10}   "
                f"{gpus:<12}[/]{mark}"
            )
        table = "\n".join(rows)
        if dropped > 0:
            table += f"\n  [{_FAINT}]… and {dropped} more partition(s)[/]"

        # Actionable suggestion — but only when a requeue could actually help.
        alts = [p for p in kept if fits[p.name] and not p.is_current]
        if not requeue_could_help(job.reason):
            # Held / dependency / begin-time / reservation: a partition change can't
            # start it, so don't suggest one.
            tip = (
                f"\n  [{_FAINT}]moving to another partition won't start this job {dash} it "
                f"isn't waiting on free capacity (see the reason above)[/]"
            )
        elif alts:
            best = alts[0]
            tip = (
                f"\n  [{ok}]{arrow}[/] [{_INK}]{_escape_markup(best.name)}[/] "
                f"[{_DIM}]has room for this request right now {dash} requeue with[/]  "
                f"[{_INK}]scontrol update JobId={_escape_markup(job.job_id)} "
                f"Partition={_escape_markup(best.name)}[/]"
            )
        elif not any(fits[p.name] for p in parts if p.is_current):
            # None of the job's own partition(s) can take it right now.
            tip = (
                f"\n  [{_FAINT}]no partition currently has enough free capacity for this "
                f"request {dash} it will start once resources free up[/]"
            )
        else:
            tip = ""
        return f"{head}\n{table}{tip}"


class PendingScreen(Screen[None]):
    """Hosts the :class:`PendingView`, refreshing it so the user can watch the
    job move toward starting. If the job stops being pending (it started, or
    ended), a notice says so and the refresh stops."""

    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
        Binding("r", "refresh", "Refresh"),
    ]

    CSS = """
    PendingScreen { background: $surface; }
    #pending-notice { height: auto; padding: 1 2 0 2; }
    #pending-body { padding: 1 2 0 2; height: 1fr; }
    #pending-card {
        height: auto;
        background: $panel;
        border: round $primary 55%;
        border-title-style: bold;
        border-title-color: $primary;
        padding: 1 2;
    }
    #pending-keybar { height: 1; padding: 0 3; background: $panel; dock: bottom; }
    """

    def __init__(self, job: PendingJob, config: SlurmwatchConfig | None = None) -> None:
        super().__init__()
        self._job = job
        self.config = config or SlurmwatchConfig()
        self.title = "slurmwatch"
        self.sub_title = f"pending job {job.job_id}"
        self._done = False  # set once the job is no longer pending

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon=" ")
        yield Static(id="pending-notice")
        with VerticalScroll(id="pending-body"), Vertical(id="pending-card") as card:
            card.border_title = f"PENDING · {_escape_markup(str(self._job.job_id))}"
            yield PendingView()
        keys = [("q", "Quit", _ACCENT), ("r", "Refresh", _CPU_COLOR)]
        yield KeyFooter(keys, id="pending-keybar")

    def on_mount(self) -> None:
        self.query_one("#pending-notice", Static).display = False
        # Seed with what the CLI already resolved so there's no blank first frame,
        # then pull live partition/queue data (and re-check the job) in the worker.
        view = self.query_one(PendingView)
        view.job = self._job
        view.config = self.config
        view.refresh(layout=True)
        self.run_worker(self._refresh(), exclusive=True)
        # Re-poll on a gentle cadence (scontrol + sinfo + squeue each tick).
        self.set_interval(10.0, self._kick_refresh)
        # Animate the "calculating…" estimate spinner (~8fps) — a no-op frame once
        # an estimate exists, so it costs nothing after the scheduler plans the job.
        self.set_interval(0.12, self._tick_spinner)

    def _kick_refresh(self) -> None:
        if not self._done:
            self.run_worker(self._refresh(), exclusive=True)

    def _tick_spinner(self) -> None:
        if self._done:
            return
        with contextlib.suppress(NoMatches):
            view = self.query_one(PendingView)
            job = view.job
            if job is None:
                return
            est = job.start_time_estimate
            if est is not None and est >= time.time() - 1:
                return  # an estimate exists — nothing to animate
            if is_held_like(job.reason):
                return  # held/blocked jobs show a static note, not the spinner
            view.frame += 1
            view.refresh()

    async def _refresh(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            job = await loop.run_in_executor(None, resolve_pending_job, self._job.job_id)
        except JobNotPendingError:
            self._mark_started()
            return
        except Exception:
            return  # transient failure: keep the last view
        try:
            parts = await loop.run_in_executor(
                None, resolve_cluster_partitions, job.partition, job.account, job.username
            )
            counts = await loop.run_in_executor(None, resolve_queue_counts, job.partition)
            rank = await loop.run_in_executor(
                None, resolve_priority_rank, job.partition, job.priority
            )
        except Exception:
            view = self.query_one(PendingView)
            parts, counts, rank = view.partitions, (0, 0), view.queue_rank
        self._job = job
        with contextlib.suppress(NoMatches):
            view = self.query_one(PendingView)
            view.job = job
            view.partitions = parts
            view.queue_running, view.queue_pending = counts
            view.queue_rank = rank
            view.config = self.config
            view.refresh(layout=True)

    def _mark_started(self) -> None:
        """The job left the queue (started or ended): say so and stop refreshing."""
        self._done = True
        ascii_mode = self.config.ascii_mode
        mark = "->" if ascii_mode else "▸"
        with contextlib.suppress(NoMatches):
            notice = self.query_one("#pending-notice", Static)
            notice.update(
                f"  [{_HEALTH_COLOR['ok']}]{mark} job {_escape_markup(self._job.job_id)} is no "
                f"longer pending[/] [{_DIM}]— it may have started. Press[/] [{_INK}]q[/] "
                f"[{_DIM}]and run[/] [{_INK}]slurmwatch {_escape_markup(self._job.job_id)}[/] "
                f"[{_DIM}]to monitor it live.[/]"
            )
            notice.display = True
            notice.refresh(layout=True)

    def action_quit(self) -> None:
        self.app.exit()

    def action_refresh(self) -> None:
        if not self._done:
            self.run_worker(self._refresh(), exclusive=True)


class PendingApp(App[Any]):
    """A tiny app that shows the pending-job view for one queued job."""

    TITLE = "slurmwatch"
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, job: PendingJob, config: SlurmwatchConfig | None = None) -> None:
        super().__init__()
        self._job = job
        self._config = config

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            self.register_theme(_CLAUDE_THEME)
            self.theme = "slurmwatch"
        self.push_screen(PendingScreen(self._job, self._config))


class SlurmwatchApp(App[Any]):
    TITLE = "slurmwatch"

    # slurmwatch is driven by its own explicit keys (q/c/m/g); the built-in
    # Ctrl+P command palette isn't part of the design, so disable it — that also
    # stops the header-icon corner from being a hidden click target.
    ENABLE_COMMAND_PALETTE = False

    SCREENS: ClassVar = {}

    CSS = """
    Screen { background: $surface; }

    /* The footer keybindings default to a flat, drab grey. Colour the key cap in
       the coral accent and the label in warm ink so the shortcuts read clearly. */
    Footer { background: $panel; }
    FooterKey { background: $panel; color: $foreground; }
    FooterKey .footer-key--key {
        color: $background;
        background: $primary;
        text-style: bold;
    }
    FooterKey .footer-key--description { color: $foreground; }
    FooterKey:hover { background: $primary 20%; }
    FooterKey:hover .footer-key--description { color: $primary; }
    """

    def __init__(
        self,
        job_ctx: JobContext | None = None,
        collector: Any = None,
        jobs: list[dict[str, object]] | None = None,
        config: SlurmwatchConfig | None = None,
    ) -> None:
        super().__init__()
        self._job_ctx = job_ctx
        self._collector = collector
        self._jobs = jobs
        self._config = config

    def on_mount(self) -> None:
        # The warm "Claude Code" theme. Guarded because register_theme/theme
        # only exist in Textual >= 0.86; older versions keep the default.
        with contextlib.suppress(Exception):
            self.register_theme(_CLAUDE_THEME)
            self.theme = "slurmwatch"
        # Exit cleanly on SIGTERM. When we run under `srun --pty` on a compute node
        # and the job is cancelled, slurmstepd SIGTERMs the step; without this the
        # process is killed mid-draw and the terminal is left in the alt-screen/raw
        # state (garbled, leaked mode-query responses). Handling SIGTERM lets
        # Textual tear the screen down and restore the terminal; return_code 143
        # tells the login-node hop the job ended (so it won't dump a stale summary).
        with contextlib.suppress(Exception):
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGTERM, lambda: self.exit(return_code=143)
            )
        # push_screen_wait (used by the selector path) requires a Textual worker
        # context; a plain asyncio task would die with NoActiveWorker.
        if self._collector is not None and self._job_ctx is not None:
            self._start_worker = self.run_worker(self._start_monitoring())
        else:
            self._start_worker = self.run_worker(self._run_job_selector())

    def on_unmount(self) -> None:
        if self._collector is not None:
            self._collector.stop_sync()

    async def _start_monitoring(self) -> None:
        assert self._collector is not None
        assert self._job_ctx is not None
        try:
            await self._collector.start()
        except Exception as exc:
            self.exit(message=f"Failed to start collector: {exc}", return_code=1)
            return
        config = self._config or getattr(self._collector, "config", None)
        await self.push_screen(DashboardScreen(self._collector, self._job_ctx, config))

    async def _run_job_selector(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            # resolve_current_jobs shells out to squeue; keep the blocking call
            # off the event loop so the app can still paint and take keys (B-C1).
            jobs = (
                self._jobs
                if self._jobs is not None
                else await loop.run_in_executor(None, resolve_current_jobs)
            )
        except Exception as exc:
            self.exit(message=f"Failed to query Slurm jobs: {exc}", return_code=1)
            return

        if not jobs:
            self.exit(message="No running Slurm jobs found.", return_code=0)
            return

        if len(jobs) == 1:
            job_id = str(jobs[0]["job_id"])
        else:
            result = await self.push_screen_wait(JobSelectorScreen(jobs))
            if not result:
                self.exit(message="No job selected.", return_code=0)
                return
            job_id = result

        # A PENDING pick can't take a live collector — route it to the why/when/
        # where view. If it started between listing and selection, fall through to
        # the live monitor below.
        state = next(
            (str(j.get("state", "")).upper() for j in jobs if str(j["job_id"]) == job_id), ""
        )
        if state in ("PD", "PENDING"):
            try:
                pend = await loop.run_in_executor(None, resolve_pending_job, job_id)
                await self.push_screen(PendingScreen(pend, self._config))
                return
            except JobNotPendingError:
                pass  # started since the list was built → attach live below
            except Exception as exc:
                self.exit(message=str(exc), return_code=1)
                return

        try:
            # resolve_job_context runs scontrol + cgroup/uid lookups; also off
            # the event loop so a slow slurmctld can't freeze the UI (B-C1).
            self._job_ctx = await loop.run_in_executor(None, resolve_job_context, job_id)
        except Exception as exc:
            self.exit(message=str(exc), return_code=1)
            return

        self._collector = TelemetryCollector(self._job_ctx, self._config)
        try:
            await self._collector.start()
        except Exception as exc:
            self.exit(message=f"Failed to start collector: {exc}", return_code=1)
            return
        await self.push_screen(DashboardScreen(self._collector, self._job_ctx, self._config))
