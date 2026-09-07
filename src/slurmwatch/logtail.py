"""Bounded, restartable tail of a job's stdout/stderr log file.

Framework-free on purpose: everything here is plain bytes → text, so the awkward
parts (a 1 MB tqdm log, a half-written last line, a file that vanishes and comes
back) are unit-testable against a real file without a terminal, and the Textual
screen in ``tui.py`` is left with nothing but layout.

Three properties the dashboard's logs actually need on this cluster:

* **Bounded.** A ``.err`` here routinely reaches 1+ MB and the interesting part is
  always the end, so the reader seeks to ``size - window`` and reads forward from
  there. It never reads the whole file, and its memory does not scale with the
  file. See ``_TAIL_WINDOW_BYTES``.
* **Incremental.** After the first window, each read consumes only the bytes
  appended since the last one, so following a live job costs one ``stat`` plus the
  new bytes per tick rather than re-reading the tail.
* **Honest about the last line.** A job appending to its log has, most of the
  time, a final line with no newline yet. That line is reported separately
  (``TailChunk.partial``) so a caller can show it as "still being written" instead
  of committing a half-line to the scrollback — and the bytes stay in the
  assembler, so nothing is lost when the rest of the line arrives.
"""

from __future__ import annotations

import codecs
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field

# How far back from EOF the first read seeks. 256 KiB is chosen against the shape
# of the logs on this cluster, not as a round number: a tqdm-heavy `.err` is 1+ MB
# but collapses (see `_LineAssembler`) to a handful of lines, while a plain text
# log at ~80 bytes a line puts ~3k lines in this window — several screens of
# scrollback either way, one bounded read, and no dependence whatsoever on how
# large the file grew. Reading more would buy scrollback nobody scrolls to at the
# cost of a read that gets slower as the job runs.
_TAIL_WINDOW_BYTES = 256 * 1024

# A second, independent cap: how many complete lines one read may hand back. The
# byte window bounds the FIRST read; this bounds a later one, because a job can
# emit more in a single 0.5 s tick than a viewer can usefully draw. Whatever is
# dropped is the oldest, and the count is reported (`TailChunk.dropped`) so the UI
# can say so rather than silently skipping output.
_MAX_TAIL_LINES = 4000

# The widest a single physical line may get, in characters. A line rewritten with
# `\r` is unbounded in bytes but bounded in COLUMNS, so this only trips on genuinely
# pathological output (a program that writes megabytes with no newline and no
# carriage return); dropping the overflow keeps the assembler O(columns) instead of
# O(bytes written).
_MAX_LINE_CELLS = 8192

# Slurm's filename patterns, restricted to the letters Slurm actually defines
# (`%%`, %A array job, %a array task, %J jobid.step, %j jobid, %N node, %n node
# rank, %s step, %t task, %u user, %x job name). Deliberately NOT `[A-Za-z]`: a
# real path may contain a bare percent — `/scratch/100%off/run.out` — and matching
# `%o` there would make a perfectly good path look like an unresolvable pattern.
_LOG_PATTERN = re.compile(r"%(\d*)([%AaJjNnstux])")


@dataclass
class TailChunk:
    """One read's worth of tail: what to append, and what state the file is in."""

    # Complete lines (newline seen), oldest first, carriage returns already
    # collapsed. Never includes the line still being written.
    lines: list[str] = field(default_factory=list)
    # The line currently being written — no trailing newline yet. Replaced (not
    # appended to) on every read until its newline arrives.
    partial: str = ""
    # The caller must discard everything it was showing before appending `lines`:
    # this is either the first read or the file went backwards (truncated/rotated).
    reset: bool = False
    # Why nothing could be read, in words a user can act on. Empty on success.
    error: str = ""
    # The file's size at read time, or -1 when it could not be stat'ed.
    size: int = -1
    # The window starts mid-file, so the reader is showing the end and not the
    # whole thing.
    tail_only: bool = False
    # Complete lines discarded by `_MAX_TAIL_LINES` on this read.
    dropped: int = 0


class _LineAssembler:
    """Bytes-as-they-arrive → complete lines, with a terminal's ``\\r`` semantics.

    This is the difference between a viewer that is pleasant on this cluster and
    one that is unusable. A tqdm progress bar is ONE physical line rewritten
    thousands of times: ``\\r 1%|… \\r 2%|… \\r 3%|…`` with a single ``\\n`` at the
    very end. Treating each ``\\r`` chunk as its own line turns a 15-line training
    log into 40 000 lines of near-identical progress bars, and the actual output —
    the loss numbers, the traceback — is buried.

    So ``\\r`` is given the meaning the terminal gives it: return to column 0 and
    overwrite from there. A reader then sees what they would have seen live — the
    bar's final state, on one line. Overwrite, not "keep the last chunk", because
    a shorter rewrite genuinely leaves the tail of the longer one behind on a real
    terminal (``\\rabcdef\\rxy`` reads ``xycdef``), and a viewer that disagrees with
    the terminal the log was written for is telling a small lie about the output.

    Stateful because the input arrives in arbitrary chunks: the column has to
    survive a read boundary, or ``"abc\\r"`` followed by ``"XY"`` would render
    ``abcXY`` instead of ``XYc``.
    """

    __slots__ = ("_cells", "_col")

    def __init__(self) -> None:
        self._cells: list[str] = []
        self._col = 0

    def feed(self, text: str) -> list[str]:
        """Consume ``text``; return the complete lines it finished."""
        # Fast path: no carriage return in this chunk AND the cursor is at the end
        # of the line (nothing to overwrite), which is every ordinary log. A
        # per-character loop over a 256 KiB first window is ~50 ms of pure Python;
        # `str.split` is ~1 ms, and the result is identical when neither `\r` nor a
        # pending overwrite is in play.
        if "\r" not in text and self._col == len(self._cells):
            return self._feed_plain(text)
        out: list[str] = []
        cells = self._cells
        for ch in text:
            if ch == "\n":
                out.append("".join(cells))
                cells = self._cells = []
                self._col = 0
            elif ch == "\r":
                self._col = 0
            elif self._col < len(cells):
                cells[self._col] = ch
                self._col += 1
            elif len(cells) < _MAX_LINE_CELLS:
                cells.append(ch)
                self._col += 1
            # else: past the column cap — drop it (see _MAX_LINE_CELLS).
        return out

    def _feed_plain(self, text: str) -> list[str]:
        parts = text.split("\n")
        if len(parts) == 1:
            self._append(parts[0])
            return []
        room = max(0, _MAX_LINE_CELLS - len(self._cells))
        out = ["".join(self._cells) + parts[0][:room]]
        out.extend(mid[:_MAX_LINE_CELLS] for mid in parts[1:-1])
        self._cells = []
        self._col = 0
        self._append(parts[-1])
        return out

    def _append(self, text: str) -> None:
        room = _MAX_LINE_CELLS - len(self._cells)
        if room > 0:
            self._cells.extend(text[:room])
        self._col = len(self._cells)

    def value(self) -> str:
        """The line currently being written, as it would look on a terminal."""
        return "".join(self._cells)

    def reset(self) -> None:
        self._cells = []
        self._col = 0


class LogTail:
    """A restartable, bounded tail of one path.

    Construct once per file and call `read` on a timer. Every failure mode is a
    ``TailChunk.error`` string rather than an exception, because all of them are
    ordinary states of a Slurm log — the job hasn't created the file yet, the file
    belongs to another user, the filesystem hiccuped — and the caller's job in each
    case is to say so and try again on the next tick, not to unwind.
    """

    def __init__(
        self,
        path: str,
        *,
        window_bytes: int = _TAIL_WINDOW_BYTES,
        max_lines: int = _MAX_TAIL_LINES,
    ) -> None:
        self.path = path
        self._window = max(1, window_bytes)
        self._max_lines = max(1, max_lines)
        self._pos = 0  # byte offset of the next unread byte
        # (st_dev, st_ino) of the file `_pos` is an offset INTO. A byte offset only
        # means something relative to one particular file; see `read`.
        self._ident: tuple[int, int] | None = None
        self._started = False
        self._line = _LineAssembler()
        self._decoder = self._new_decoder()

    @property
    def max_lines(self) -> int:
        """The most complete lines one read will hand back. Public because a caller
        with its own scrollback cap has to keep it clear of this number, or a single
        burst pushes out the very lines it just delivered."""
        return self._max_lines

    @property
    def window_bytes(self) -> int:
        """How far back from EOF the first read looks. Public so the UI can say so:
        "last 256 KiB only" is the difference between a reader trusting the view and
        a reader wondering where the start of their log went."""
        return self._window

    @staticmethod
    def _new_decoder() -> codecs.IncrementalDecoder:
        # Incremental, not `bytes.decode`: a multi-byte character straddling a read
        # boundary would otherwise decode to two replacement characters — the reader
        # would corrupt text that is perfectly valid, purely because of where it
        # happened to stop reading. `errors="replace"` covers the genuinely non-UTF-8
        # case (a binary blob accidentally written to a log) without raising.
        return codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _restart(self) -> None:
        self._pos = 0
        self._line.reset()
        self._decoder = self._new_decoder()

    def read(self) -> TailChunk:
        """Read whatever is new (or, first time, the tail window)."""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            # The normal state of a job that has not written anything yet, and of a
            # file that was rotated away — so reset, and let the next read pick the
            # new file up from its start.
            self._restart()
            self._started = False
            return TailChunk(error="no output file at this path yet")
        except OSError as exc:
            # PermissionError included: reading another user's job (or a `--jobid`
            # whose output lives somewhere this account cannot see) is an ordinary
            # outcome here, and `exc.strerror` already says "permission denied".
            return TailChunk(error=_why("cannot stat", exc))

        mode = st.st_mode
        if stat.S_ISDIR(mode):
            return TailChunk(error="this path is a directory, not a file")
        if not (stat.S_ISREG(mode) or stat.S_ISCHR(mode)):
            # A FIFO/socket would block `open` forever (Slurm never writes one, but
            # `--output=` takes any path), and a block device is not a log.
            return TailChunk(error="this path is not a regular file")

        size = st.st_size
        ident = (st.st_dev, st.st_ino)
        # `size < self._pos` means the file went backwards: truncated in place, or a
        # new file at the same path. Either way what we are showing is no longer
        # this file's content, so start over and tell the caller to clear.
        #
        # The size test alone only catches the second case when the replacement
        # happens to be SHORTER. A same-size or larger new file at the same path
        # left `_pos` pointing into a file it was never an offset into, so the
        # reader resumed mid-content: measured, a 12-byte log replaced by a
        # 24-byte one showed only its last two lines, skipped its first two
        # entirely, and reported `reset=False`, so the viewer did not clear either.
        # Silently showing the wrong bytes with no diagnostic is the failure this
        # module's error strings exist to avoid, so identity is compared as well —
        # which is what the rule above already said it was doing.
        #
        # The limit, stated rather than papered over: an in-place rewrite that
        # keeps the inode and does not shrink the file is indistinguishable from an
        # append through `stat`, so it is not caught and cannot be. Every way a
        # path is really swapped (`mv` in, rm+create on a filesystem that does not
        # recycle the inode, a retargeted symlink) does change identity, and a
        # rewrite that shortens the file is caught by the size test.
        fresh = not self._started or size < self._pos or ident != self._ident
        if fresh:
            self._restart()
        self._ident = ident

        try:
            with open(self.path, "rb") as fh:
                if fresh:
                    start = max(0, size - self._window)
                    if start:
                        fh.seek(start)
                else:
                    fh.seek(self._pos)
                    start = self._pos
                data = fh.read()
                # `tell()`, not `start + len(data)` conceptually — the file may have
                # grown between the stat and the read, and `read()` took all of it.
                self._pos = fh.tell()
        except OSError as exc:
            if fresh:
                # `_restart` above already threw `_pos` away, so this read WAS meant
                # to be the start of a new view of the file — and it never happened.
                # Leaving `_started` set makes the NEXT read look like a resume at
                # `_pos == 0`, which loses both of this module's promises at once:
                # measured, a file truncated in place whose next `open` failed came
                # back `reset=False` on recovery (so the viewer appended the new
                # file's lines underneath the old file's, with no "restarted" note),
                # and the read was unbounded — all 40 lines through a 64-byte window,
                # because a resume never seeks to `size - window`. Forget the start
                # instead, and let the next read be the fresh one this one was.
                self._started = False
            return TailChunk(error=_why("cannot read", exc), size=size)

        self._started = True
        tail_only = False
        if fresh and start > 0:
            data, tail_only = _drop_leading_fragment(data)

        lines = self._line.feed(self._decoder.decode(data))
        dropped = 0
        if len(lines) > self._max_lines:
            dropped = len(lines) - self._max_lines
            lines = lines[-self._max_lines :]
            tail_only = True
        return TailChunk(
            lines=lines,
            partial=self._line.value(),
            reset=fresh,
            size=size,
            tail_only=tail_only,
            dropped=dropped,
        )


def _drop_leading_fragment(data: bytes) -> tuple[bytes, bool]:
    """Trim the partial line the window's start byte landed in the middle of.

    Returns ``(data, tail_only)``. Prefer a newline boundary; failing that a
    carriage return, which is where a terminal would have put the cursor back to
    column 0 anyway. If the window contains neither, the file is one unbroken blob
    — keep it, because showing its end is far better than showing nothing.
    """
    idx = data.find(b"\n")
    if idx >= 0:
        return data[idx + 1 :], True
    idx = data.find(b"\r")
    if idx >= 0:
        return data[idx + 1 :], True
    return data, True


def _why(what: str, exc: OSError) -> str:
    """An OS error as a phrase a user can act on, never a bare errno."""
    reason = exc.strerror or exc.__class__.__name__
    return f"{what}: {reason.lower()}"


def expand_log_pattern(path: str, values: Mapping[str, str]) -> tuple[str, list[str]]:
    """Substitute Slurm's ``%j``-style filename patterns in ``path``.

    ``scontrol show job`` normally reports StdOut/StdErr already expanded, but not
    always: the per-task and per-node patterns (``%t``, ``%n``) name a file only one
    of the job's tasks can identify, and Slurm hands them back verbatim. A viewer
    that pasted such a path into ``open()`` would report "no such file" and blame
    the job for it.

    So: substitute every pattern we can actually resolve, and RETURN the ones we
    cannot alongside the result, so the caller can say which pattern defeated it
    instead of showing a misleading filesystem error.
    """
    unresolved: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        width, letter = m.group(1), m.group(2)
        if letter == "%":
            return "%"
        value = values.get(letter, "")
        if not value:
            unresolved.append(m.group(0))
            return m.group(0)
        # Slurm's `%<n>j` zero-pads a numeric field to n digits; a non-numeric
        # value (a job name) has no padding to apply.
        return value.zfill(int(width)) if width and value.isdigit() else value

    return _LOG_PATTERN.sub(_sub, path), unresolved
