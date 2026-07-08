from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import time
from collections import deque
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Header, ListItem, ListView, Static

from .collector import TelemetryCollector, _gpu_is_active
from .config import SlurmwatchConfig
from .model import CpuMetrics, GpuMetrics, JobContext, MemoryMetrics, TelemetrySnapshot
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
_HEALTH_GLYPH = {"ok": "●", "warn": "▲", "crit": "✖", "none": "·"}
_HEALTH_GLYPH_ASCII = {"ok": "+", "warn": "!", "crit": "x", "none": "-"}

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
    line = f"{_NBSP}{_NBSP}{_NBSP}".join(parts)
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

    def render(self) -> str:
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
            where = (
                f"[{_DIM}]{arrow} {self.node} {tail} "
                f"(it may be busy or unreachable — still retrying; or switch to another node)[/]"
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

    def _head(self, label: str, color: str, level: str, ascii_mode: bool) -> str:
        # Health dot first (its colour is the only status channel), then the
        # resource label. The row reports facts — the bar, the %, the raw
        # figures, the recent range — and lets the reader judge; it never appends
        # a verdict word like "underused"/"idle" (the dot's colour is the signal,
        # and the banner still names anything that actually needs action).
        return f"  {_dot(level, ascii_mode)} [{color}]{label:<5}[/]"

    @staticmethod
    def _trend_tag(hist: deque[float], window_s: int) -> str:
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
        if hi - lo < _TREND_STEADY_SPAN:
            return f"   [{_DIM}]· steady[/]"
        return f"   [{_DIM}]· {lo:.0f}–{hi:.0f}% over {window_s}s[/]"

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
        level, _ = _cpu_health(cpu, cfg.cpu_underuse_threshold)
        cpu_bar = _labeled_bar("usage", cpu.usage_percent, bar_w, ascii_mode, _CPU_COLOR)
        cpu_detail = f"{_fmt_cores(cpu.effective_cores)} / {cpu.cores_allocated} cores"
        cpu_tag = self._trend_tag(self.cpu_history, window_s) if wide else ""
        blocks.append(
            f"{self._head('CPU', _CPU_COLOR, level, ascii_mode)}   "
            f"{cpu_bar}   [{_DIM}]{cpu_detail}[/]{cpu_tag}"
        )

        mem = snap.memory
        level, _ = _mem_health(mem)
        ws = mem.working_set_bytes or mem.current_bytes
        mem_head = self._head("MEM", _MEM_COLOR, level, ascii_mode)
        if mem.limit_bytes > 0:
            mem_pct = _mem_ws_pct(mem)
            mem_detail = f"{_gib(ws):.0f} / {_gib(mem.limit_bytes):.0f} GiB"
            # Peak is secondary; drop it on a narrow terminal so a big-memory job
            # (3-digit GiB) can't push the line past 80 cols and soft-wrap.
            if wide:
                mem_detail += f" · peak {_gib(mem.peak_bytes):.0f} GiB"
            mem_bar = _labeled_bar("used", mem_pct, bar_w, ascii_mode, _MEM_COLOR)
            mem_tag = self._trend_tag(self.mem_history, window_s) if wide else ""
            blocks.append(f"{mem_head}   {mem_bar}   [{_DIM}]{mem_detail}[/]{mem_tag}")
        else:
            # No enforced limit → a 'used 0%' bar would contradict the GiB in
            # use, so show the amount only, with no misleading percentage.
            blocks.append(
                f"{mem_head}   [{_DIM}]{'used':<7}[/] "
                f"[{_INK}]{_format_bytes(ws)}[/] [{_DIM}]· no limit set[/]"
            )

        gpus = snap.gpus
        if not self.gpu_table_active:
            if gpus:
                for gpu in gpus:
                    blocks.append("\n".join(self._gpu_block(gpu, cfg, bar_w, ascii_mode, wide)))
            elif snap.gpu_count_requested > 0:
                blocks.append(
                    f"  [dim]GPU   {snap.gpu_count_requested} requested — "
                    "telemetry unavailable here (run on the compute node)[/]"
                )
            else:
                blocks.append("  [dim]GPU   none requested[/]")
        return "\n\n".join(blocks)

    def _gpu_block(
        self, gpu: GpuMetrics, cfg: SlurmwatchConfig, bar_w: int, ascii_mode: bool, wide: bool
    ) -> list[str]:
        # A GPU has two independent "how busy / how full" axes, so it gets two
        # explicitly-labeled bars — compute (SM/CUDA-core utilisation) and vram
        # (memory fill) — instead of one unlabeled bar that reads as whichever
        # number sits beside it. 'vram' (not 'memory') so it can't blur with the
        # MEM row above. The health dot alone carries status; no verdict word.
        level, _ = _gpu_health(gpu, cfg.gpu_idle_threshold)
        compute = _labeled_bar("compute", gpu.utilization_percent, bar_w, ascii_mode, _GPU_COLOR)
        vram_bar = _labeled_bar(
            "vram", gpu.memory_utilization_percent, bar_w, ascii_mode, _GPU_VRAM_COLOR
        )
        used_g, tot_g = _gib(gpu.memory_used_bytes), _gib(gpu.memory_total_bytes)
        vram_amt = f"{used_g:.0f} / {tot_g:.0f} GiB"
        pwr = f"{gpu.power_watts:.0f} W"
        deg = "C" if ascii_mode else "°C"
        hot = gpu.temperature_celsius >= _TEMP_HOT_C
        temp_txt = f"{gpu.temperature_celsius:.0f} {deg}" + ("!" if hot else "")
        temp = f"[{_HEALTH_COLOR['warn']}]{temp_txt}[/]" if hot else f"[{_DIM}]{temp_txt}[/]"
        head = self._head(f"GPU{gpu.index}", _GPU_COLOR, level, ascii_mode)
        tail = f"[{_DIM}]{vram_amt}[/]   [{_DIM}]{pwr}[/] [{_FAINT}]·[/] {temp}"

        # On a wide-enough terminal the two bars (they describe one device) ride a
        # single line, so a multi-GPU job is one row per device instead of three.
        if self.size.width >= _GPU_MERGE_COLS or self.size.width == 0:
            return [f"{head}   {compute}   {vram_bar}   {tail}"]

        # Otherwise stack them, and fold the compute series' recent range (what the
        # old TRENDS panel tracked per GPU) onto the compute line.
        compute_tag = (
            self._trend_tag(self.gpu_history.get(gpu.index, deque()), cfg.history_seconds)
            if wide
            else ""
        )
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

    Two modes, keyed on the widget id: the dashboard copy is a read-only
    overview (no row cursor — nothing is selectable there, so an always-on
    highlight is misleading, U5); the detail-screen copy (``id="detail-table"``)
    is interactive and adds the job's per-device share (JOB% / JOB VRAM), which
    the overview doesn't show, so drilling in reveals something new (F6).
    """

    config: SlurmwatchConfig | None = None

    @property
    def _detailed(self) -> bool:
        return self.id == "detail-table"

    def on_mount(self) -> None:
        self.cursor_type = "row" if self._detailed else "none"
        # Zebra stripes off: the per-device colour on each row already separates
        # them, and a background band would clutter the compact one-line rows.
        self.zebra_stripes = False
        if self._detailed:
            self.add_columns("GPU", "COMPUTE", "VRAM", "JOB%", "JOB VRAM", "PWR", "TEMP", "STATUS")
        else:
            self.add_columns("GPU", "COMPUTE", "VRAM", "PWR", "TEMP", "STATUS")

    def update_gpus(self, gpus: list[GpuMetrics], config: SlurmwatchConfig) -> None:
        ascii_mode = config.ascii_mode
        self.clear()
        for gpu in gpus:
            level, _ = _gpu_health(gpu, config.gpu_idle_threshold)
            # Each device wears its own colour (index + compute bar + VRAM) so
            # identical rows stay distinguishable. Health (status) and heat (temp)
            # keep their own colour channel — those are the same across devices.
            dcolor = _gpu_device_color(gpu.index)
            gpu_cell = Text(str(gpu.index), style=f"bold {dcolor}")
            bar = _color_bar(gpu.utilization_percent, 8, ascii_mode, dcolor)
            util = Text.from_markup(f"{gpu.utilization_percent:>3.0f}% {bar}")
            vram = Text(
                f"{_gib(gpu.memory_used_bytes):.0f}/{_gib(gpu.memory_total_bytes):.0f} GiB",
                style=dcolor,
            )
            pwr = f"{gpu.power_watts:.0f}W"
            hot = gpu.temperature_celsius >= _TEMP_HOT_C
            deg = "C" if ascii_mode else "°C"
            temp_mark = ("!" if ascii_mode else "⚠") if hot else ""
            temp = Text(
                f"{temp_mark}{gpu.temperature_celsius:.0f}{deg}", style="yellow" if hot else ""
            )
            # The status cell is the health glyph alone (facts-only: no
            # "idle"/"throttling" verdict word) — its colour and shape carry the
            # level, and the numbers in the row let the reader judge.
            status = Text(_glyph(level, ascii_mode), style=_HEALTH_COLOR[level])
            # One line per device (no blank-row gap): the coloured index cell and
            # per-device hue already tell otherwise-identical rows apart, so the
            # gap was just wasted vertical space.
            if self._detailed:
                job_util = f"{gpu.process_utilization_percent:>3.0f}%"
                job_vram = (
                    f"{_gib(gpu.process_memory_bytes):.1f} GiB" if gpu.process_memory_bytes else "—"
                )
                self.add_row(gpu_cell, util, vram, job_util, job_vram, pwr, temp, status)
            else:
                self.add_row(gpu_cell, util, vram, pwr, temp, status)


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
        # directory or script path doesn't wrap mid-word into a cluttered block.
        # Budget = the card's text width minus the label lead ("  command  ").
        ell = "..." if ascii_mode else "…"
        budget = max(24, (self.size.width or 100) - 12)
        paths = []
        if ctx.command:
            # A bare script path is elided like a path; a full command line (has
            # spaces/args) is left intact so its arguments aren't mistaken for dirs.
            cmd = (
                ctx.command
                if " " in ctx.command
                else _shorten_path(ctx.command, budget, keep=2, ell=ell)
            )
            # The command is the headline "what is this job running" — coral pops.
            paths.append(f"  [{_DIM}]command[/]  [{_ACCENT}]{_escape_markup(cmd)}[/]")
        if ctx.work_dir:
            wd = _shorten_path(ctx.work_dir, budget, keep=1, ell=ell)
            paths.append(f"  [{_DIM}]workdir[/]  [{_MEM_COLOR}]{_escape_markup(wd)}[/]")
        if paths:
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

    def render(self) -> str:
        ctx = self.job_ctx
        snap = self.snapshot
        if ctx is None or snap is None:
            return ""
        ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode
        sep = _sep(ascii_mode)

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
        freshness = f" [{_FAINT}]· {int(age)}s old[/]" if age >= 3 else ""
        # Each identity value gets its own hue (dim labels, coloured values) so the
        # bottom bar reads as a lively strip rather than a flat grey line. Packing
        # wraps the strip only between chips, so on a narrow terminal a value is
        # never orphaned from its label. These are chrome, well below the resource
        # rows, so reusing the palette ties the UI together without being mistaken
        # for a CPU/MEM/GPU reading.
        ident_chips = [
            f"[{_DIM}]job[/] [{_ACCENT}]{snap.job_id}[/]",
            f"[{_DIM}]user[/] [{_CPU_COLOR}]{ctx.username or '?'}[/]",
            f"[{_DIM}]partition[/] [{_GPU_COLOR}]{ctx.partition or '?'}[/]",
            f"[{_DIM}]node[/] [{node_style}]{node}[/]{freshness}",
        ]
        ident = _pack_chips(ident_chips, sep, (self.size.width or 100) - 6)

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
            text = f"ran {el}  {frac:.0f}%  ·  {rem} left of {lim} limit  ·  ends by {ends}"
            inner = (self.size.width or 100) - 6  # #jobinfo padding 1 3
            # Leave >=2 cols of right margin so the line never touches the edge.
            bar_w = min(20, inner - len(text) - 3)
            bar = f"{_color_bar(frac, bar_w, ascii_mode, urg)} " if bar_w >= 6 else ""
            time_line = (
                f"[{_DIM}]ran[/] [{_INK}]{el}[/] {bar}[{_INK}]{frac:.0f}%[/]{sep}"
                f"[bold {urg}]{rem}[/] [{_DIM}]left of[/] [{_INK}]{lim}[/] [{_DIM}]limit[/]{sep}"
                f"[{_DIM}]ends by[/] [{_ACCENT}]{ends}[/]"
            )
        else:
            time_line = (
                f"[{_DIM}]ran[/] [{_INK}]{_format_duration(elapsed)}[/]{sep}"
                f"[{_DIM}]no wall-clock time limit[/]"
            )
        return f"{ident}\n{time_line}"


class ResourceDetailScreen(Screen[None]):
    """A real drill-in for one resource, replacing the old inert focus keys.

    Shows a full-width history graph and every number for the resource, plus a
    per-device table for GPUs. Reads the dashboard's live snapshot on a timer so
    it keeps updating while open.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back"),
        Binding("c", "switch('cpu')", "CPU"),
        Binding("m", "switch('mem')", "Memory"),
        Binding("g", "switch('gpu')", "GPU"),
    ]

    CSS = """
    ResourceDetailScreen { align: center middle; }
    #detail-box {
        width: 90%;
        max-width: 130;
        height: auto;
        max-height: 90%;
        border: round $primary;
        padding: 1 2;
    }
    #detail-title { text-style: bold; padding-bottom: 1; }
    #detail-body { height: auto; }
    #detail-chart { height: auto; padding-top: 1; }
    #detail-table { height: auto; margin-top: 1; }
    """

    def __init__(self, dashboard: DashboardScreen, resource: str) -> None:
        super().__init__()
        self._dashboard = dashboard
        self._resource = resource

    def compose(self) -> ComposeResult:
        with Vertical(id="detail-box"):
            yield Static(id="detail-title")
            yield Static(id="detail-body")
            # GPUs also get the per-device table above the chart; every resource
            # gets the tall braille trend line that fills the box (1fr) instead
            # of a single cramped sparkline sized to the wrong width (F2).
            if self._resource == "gpu":
                yield GpuTable(id="detail-table")
            yield Static(id="detail-chart")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(0.5, self._refresh)

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_switch(self, resource: str) -> None:
        if resource == self._resource:
            return
        self.app.pop_screen()
        self.app.push_screen(ResourceDetailScreen(self._dashboard, resource))

    # #detail-box is 90% of the screen (capped at 130) with a round border
    # (1 col/side) and horizontal padding of 2 (2 cols/side) = 6 cols of chrome.
    _BOX_CHROME = 6
    _BOX_MAX_W = 130

    def _chart_width(self, chart_widget: Static) -> int:
        # Size the chart to the widget's real content width, not a
        # screen-relative guess, so it never overflows the padded/bordered box
        # and wraps into broken fragments (F2). Before layout the widget size is
        # 0; fall back to the box geometry (never wider than the box) so even the
        # first pre-layout render can't overflow.
        w = chart_widget.size.width
        if w <= 0:
            w = min(int(self.size.width * 0.9), self._BOX_MAX_W) - self._BOX_CHROME
        return max(w, _SPARK_W)

    def _refresh(self) -> None:
        snap = self._dashboard.latest_snapshot
        cfg = self._dashboard.config
        ascii_mode = cfg.ascii_mode
        with contextlib.suppress(NoMatches):
            title = self.query_one("#detail-title", Static)
            body = self.query_one("#detail-body", Static)
            if snap is None:
                title.update("[dim]awaiting data…[/]")
                body.update("")
                return
            if self._resource == "cpu":
                self._refresh_cpu(title, body, snap, ascii_mode)
            elif self._resource == "mem":
                self._refresh_mem(title, body, snap, ascii_mode)
            else:
                self._refresh_gpu(title, body, snap, cfg)

    def _refresh_cpu(
        self, title: Static, body: Static, snap: TelemetrySnapshot, ascii_mode: bool
    ) -> None:
        title.update("CPU detail  ·  [dim]c/m/g to switch · esc to close[/]")
        cpu = snap.cpu
        util_bar = _color_bar(cpu.usage_percent, 24, ascii_mode, _CPU_COLOR)
        # One bar only: 'effective cores' is just usage% × allocation, so a second
        # bar would be identical every frame — show it as a plain figure instead.
        body.update(
            f"usage         {util_bar}  {cpu.usage_percent:.1f}%\n"
            f"effective     {cpu.effective_cores:.2f} / {cpu.cores_allocated} cores"
            f"  [dim](avg cores kept busy)[/]"
        )
        self._update_chart(self._dashboard.cpu_history, ascii_mode, "usage")

    def _refresh_mem(
        self, title: Static, body: Static, snap: TelemetrySnapshot, ascii_mode: bool
    ) -> None:
        title.update("Memory detail  ·  [dim]c/m/g to switch · esc to close[/]")
        mem = snap.memory
        ws = mem.working_set_bytes or mem.current_bytes
        if mem.limit_bytes > 0:
            limit = mem.limit_bytes

            def pct(v: int) -> float:
                return min(100.0, v / limit * 100.0)

            headroom = max(limit - ws, 0)
            body.update(
                f"working set   {_color_bar(pct(ws), 24, ascii_mode, _MEM_COLOR)}  "
                f"{_format_bytes(ws)}  ({_mem_ws_pct(mem):.1f}% of {_format_bytes(limit)})\n"
                f"peak          {_color_bar(pct(mem.peak_bytes), 24, ascii_mode, _MEM_COLOR)}  "
                f"{_format_bytes(mem.peak_bytes)}\n"
                f"cache         [dim]reclaimable[/] {_format_bytes(mem.cache_bytes)}  ·  "
                f"total used {_format_bytes(mem.current_bytes)}\n"
                f"headroom to the OOM line: [{_MEM_COLOR}]{_format_bytes(headroom)}[/]"
            )
        else:
            body.update(
                f"working set {_format_bytes(ws)} (no limit)\n"
                f"cache (reclaimable) {_format_bytes(mem.cache_bytes)}  ·  "
                f"peak {_format_bytes(mem.peak_bytes)}"
            )
        self._update_chart(self._dashboard.mem_history, ascii_mode, "working set %")

    def _refresh_gpu(
        self, title: Static, body: Static, snap: TelemetrySnapshot, cfg: SlurmwatchConfig
    ) -> None:
        title.update("GPU detail  ·  [dim]c/m/g to switch · esc to close[/]")
        if snap.gpus:
            active = sum(1 for g in snap.gpus if _gpu_is_active(g, cfg.gpu_idle_threshold))
            total = len(snap.gpus)
            body.update(
                f"{total} {'device' if total == 1 else 'devices'} · {active} active"
                f"  ·  [dim]JOB% / JOB VRAM = this job's share of each device[/]"
            )
            with contextlib.suppress(NoMatches):
                self.query_one("#detail-table", GpuTable).update_gpus(snap.gpus, cfg)
            # Utilization trend for the busiest device, sharing the dashboard's
            # per-GPU history so drilling into GPU also gets a live trend line.
            rows_widget = self._dashboard.resource_rows
            hottest = max(snap.gpus, key=lambda g: g.utilization_percent)
            hist = (
                rows_widget.gpu_history.get(hottest.index, deque())
                if rows_widget is not None
                else deque()
            )
            self._update_chart(hist, cfg.ascii_mode, f"GPU{hottest.index} compute %")
        elif snap.gpu_count_requested > 0:
            body.update(
                f"[dim]{_plural(snap.gpu_count_requested, 'GPU')} requested — live telemetry "
                "unavailable here; run on the compute node.[/]"
            )
            self._clear_chart()
        else:
            body.update("[dim]no GPUs requested by this job[/]")
            self._clear_chart()

    def _resource_color(self) -> str:
        return {"cpu": _CPU_COLOR, "mem": _MEM_COLOR, "gpu": _GPU_COLOR}.get(
            self._resource, _ACCENT
        )

    def _clear_chart(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#detail-chart", Static).update("")

    def _update_chart(self, history: deque[float], ascii_mode: bool, label: str) -> None:
        with contextlib.suppress(NoMatches):
            chart = self.query_one("#detail-chart", Static)
            width = self._chart_width(chart)
            color = self._resource_color()
            vals = list(history)
            cur = vals[-1] if vals else 0.0
            head = (
                f"[{_DIM}]{label} · last {self._dashboard.config.history_seconds}s[/] "
                f"[{color}]{cur:>3.0f}%[/]"
            )
            # A full-width one-row sparkline reads clearly; the drill-in's extra
            # value is the min/avg/max line and (for GPUs) the per-device table.
            spark = _render_sparkline(history, width, ascii_mode, stretch=True)
            if vals:
                stats = (
                    f"[{_DIM}]min {min(vals):>3.0f}%   avg {sum(vals) / len(vals):>3.0f}%"
                    f"   max {max(vals):>3.0f}%[/]"
                )
            else:
                stats = "[dim]no history yet[/]"
            chart.update(f"{head}\n[{color}]{spark}[/]\n{stats}")


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
        # Full form: coloured key cap + its label, e.g. "[ q ] Quit". When that
        # would overrun the bar (a narrow terminal), drop the labels and keep just
        # the coloured caps ("q c m g 1-2") — self-documenting enough — so the
        # footer never wraps or clips a label off the right edge.
        avail = (self.size.width or 200) - 6  # #keybar padding 0 3
        full_w = sum(len(k) + 3 + len(lbl) for k, lbl, _ in self._keys) + 2 * (len(self._keys) - 1)
        if full_w <= avail:
            caps = [
                f"[{_BG} on {color}] {key} [/] [{_DIM}]{label}[/]"
                for key, label, color in self._keys
            ]
            return "  ".join(caps)
        caps = [f"[{_BG} on {color}] {key} [/]" for key, _label, color in self._keys]
        return " ".join(caps)


class DashboardScreen(Screen[Any]):
    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
        Binding("c", "detail('cpu')", "CPU"),
        Binding("m", "detail('mem')", "Memory"),
        Binding("g", "detail('gpu')", "GPU"),
        # Node switcher (multi-node jobs): press the node's number to jump to it
        # (matches the "node K of N" label). Left/Right also step prev/next, for
        # jobs with more than 9 nodes. Digits arrive as their own key name, so
        # these bind cleanly (unlike the bracket keys they replace).
        *[Binding(str(i), f"select_node({i})", show=False) for i in range(1, 10)],
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
    }

    /* Titled, rounded cards give the dashboard structure. An explicit lifted
       plane ($panel, a step above the screen surface) + a coral hairline border
       frame each section so it reads as a raised card, not a flat slab; the
       title wears the violet accent. The RESOURCES card carries the live gauges;
       ALLOCATION (allocated vs used) and JOB (provenance) fill the space below
       with facts we already collect. */
    #resources-panel, #job-panel {
        height: auto;
        background: $panel;
        border: round $primary 55%;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 1 2;
    }
    #job-panel { margin-top: 1; }

    ResourceRows { height: auto; }

    /* Match the card plane ($panel), else the DataTable paints its own $surface
       and reads as a darker band across the RESOURCES card behind the GPU rows. */
    GpuTable { height: auto; margin-top: 1; background: $panel; }
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
        local = socket.gethostname().split(".")[0]
        self._local_node = local
        self._selected_node = (
            local
            if local in self._node_list
            else (self._node_list[0] if self._node_list else local)
        )
        # Node switcher plumbing: a per-node cache of the last snapshot (so
        # switching back to a node shows instantly while it re-streams), plus the
        # single persistent stream for the node currently on screen (only one at a
        # time → O(1) in node count).
        self._node_cache: dict[str, TelemetrySnapshot] = {}
        self._stream_proc: asyncio.subprocess.Process | None = None
        self._stream_node: str | None = None
        # Node-switch feedback: while a switch is in flight `_switch_target` names
        # the node we're waiting on, `_switch_started` stamps when (to nudge the
        # banner to a "still attaching" note if Slurm is slow), and a paused
        # interval drives the spinner. The switch clears the instant that node's
        # first real snapshot is shown (see `_show`), not on a timer.
        self._switch_target: str | None = None
        self._switch_started: float | None = None
        self._spinner_timer: Any = None

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
                job.border_title = f"JOB · {self.job_ctx.job_id}"
                yield JobDetailsPanel()
        keys = [
            ("q", "Quit", _ACCENT),
            ("c", "CPU", _CPU_COLOR),
            ("m", "Memory", _MEM_COLOR),
            ("g", "GPU", _GPU_COLOR),
        ]
        # Only a multi-node job can switch nodes, so only then advertise the keys.
        # Up to 9 nodes: "press the node's number" (1-N). Beyond 9, the number keys
        # only reach 1-9, so also advertise the ◂ ▸ arrows (which step through ALL
        # nodes) — otherwise nodes 10-N look unreachable.
        n_nodes = len(self._node_list)
        if n_nodes > 1:
            ascii_mode = (self.config or SlurmwatchConfig()).ascii_mode
            arrows = "1-9 <>" if ascii_mode else "1-9 ◂▸"
            cap = f"1-{n_nodes}" if n_nodes <= 9 else arrows
            keys.append((cap, "Node", _GPU_VRAM_COLOR))
        with Vertical(id="bottombar"):
            yield JobInfoBar(id="jobinfo")
            yield KeyFooter(keys, id="keybar")

    def on_mount(self) -> None:
        self.query_one(GpuTable).display = False
        self.query_one(SwitchBanner).display = False
        # Runs only while a switch is in flight (resumed in `_begin_switch`,
        # paused in `_end_switch`); ~8fps reads as smooth spinner motion.
        self._spinner_timer = self.set_interval(0.12, self._tick_switch, pause=True)
        self._update_header(None)
        self._poll_task = asyncio.create_task(self._poll_loop())

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
            await asyncio.sleep(1.0)  # couldn't launch the stream; back off before retry
            return None
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
        except TimeoutError:
            return None
        if not line:  # EOF — the stream died; drop it so the next tick relaunches
            await self._stop_stream()
            return None
        return parse_snapshot_line(line)

    async def _poll_loop(self) -> None:
        try:
            while True:
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
        if node not in self._node_list or node == self._selected_node:
            return
        self._selected_node = node
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
        # The local node is the always-live collector, so switching to it needs no
        # "connecting" ceremony — the banner + dim would just flash for the ~1s
        # until its first frame lands, which reads as jarring. Only a *remote*
        # node (a Slurm step launch that can take seconds) shows the animated
        # banner; it stays up until that node's real data arrives (see `_show`),
        # never a timer. Clearing any in-flight switch keeps state consistent.
        if node == self._local_node:
            self._end_switch()
        else:
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

    def action_select_node(self, n: int) -> None:
        """Jump straight to node ``n`` (1-based), matching the "node K of N" label."""
        if 1 <= n <= len(self._node_list):
            self._set_node(self._node_list[n - 1])

    def action_next_node(self) -> None:
        self._switch_node(1)

    def action_prev_node(self) -> None:
        self._switch_node(-1)

    def _history_maxlen(self) -> int:
        interval = max(self.config.poll_interval, 0.01)
        return max(int(round(self.config.history_seconds / interval)), 10)

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
                table.update_gpus(snapshot.gpus, self.config)
            else:
                table.display = False

        with contextlib.suppress(NoMatches):
            job = self.query_one(JobDetailsPanel)
            job.job_ctx = self.job_ctx
            job.config = self.config
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
        if snapshot is None:
            self.sub_title = f"connecting to job {self.job_ctx.job_id}…"
            return
        self.sub_title = f"job {snapshot.job_id} · {self.job_ctx.username}"

    def action_quit(self) -> None:
        self.app.exit()

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
                f"Select a running job ({len(self.jobs)} found):",
                id="selector-title",
            )
            yield ListView(*[ListItem(Static(self._job_line(j))) for j in self.jobs])

    @staticmethod
    def _job_line(j: dict[str, object]) -> str:
        # The job name (%j) is free-form and user-controlled (`sbatch -J`), so
        # every interpolated value must be neutralized before it reaches the
        # markup parser (F1); only the job-id's [bold] styling is our own markup.
        def field(key: str, default: str = "?") -> str:
            return _escape_markup(str(j.get(key, default)))

        return (
            f"[bold]{_escape_markup(str(j['job_id']))}[/]  "
            f"{field('partition')}  "
            f"{field('name')}  "
            f"nodes={field('nodes')}  "
            f"time={field('wall_time')}"
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
