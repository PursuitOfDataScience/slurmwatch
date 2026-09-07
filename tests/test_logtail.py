"""`slurmwatch.logtail`: the bounded tail behind the dashboard's log viewer.

Every test writes a REAL file and appends to it, because the awkward cases are all
filesystem behaviour — a byte offset that has to survive a read boundary, a file
whose size goes backwards, a last line with no newline yet. Mocking the filesystem
here would only prove the mock agrees with itself.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from slurmwatch.logtail import (
    _MAX_LINE_CELLS,
    LogTail,
    _LineAssembler,
    expand_log_pattern,
)


class TestCarriageReturnsCollapse:
    """A tqdm bar is ONE line rewritten thousands of times, and must read as one.

    The failure mode this guards is not subtle: rendering each `\\r` chunk as its own
    line turns a 15-line training log into 40 000 lines of near-identical progress
    bars, and buries the loss numbers and the traceback the user opened the file for.
    """

    def test_a_progress_bar_is_one_line_not_thousands(self) -> None:
        bar = "".join(f"\r{p:3d}%|{'#' * (p // 10):10s}| {p}/100" for p in range(0, 101, 10))
        lines = _LineAssembler().feed(f"training\n{bar}\ndone\n")
        assert lines == ["training", "100%|##########| 100/100", "done"]

    def test_a_shorter_rewrite_leaves_the_residue_a_terminal_would_leave(self) -> None:
        # Overwrite semantics, not "keep the last chunk": on a real terminal the tail
        # of the longer earlier write is still on screen, and a viewer that disagrees
        # with the terminal the log was written for is telling a small lie.
        assert _LineAssembler().feed("\rabcdef\rxy\n") == ["xycdef"]

    def test_the_column_survives_a_read_boundary(self) -> None:
        # The reader consumes arbitrary chunks, so "abc\r" then "XY" must render the
        # same as "abc\rXY" in one piece. A stateless collapse gets this wrong.
        one = _LineAssembler()
        one.feed("abc\r")
        one.feed("XY")
        assert one.value() == "XYc"
        whole = _LineAssembler()
        whole.feed("abc\rXY")
        assert whole.value() == one.value()

    def test_a_bare_cr_at_the_end_does_not_emit_a_line(self) -> None:
        # `\r` is a cursor move, never a line terminator: emitting on it is exactly
        # the bug that multiplies a progress bar into thousands of rows.
        a = _LineAssembler()
        assert a.feed("50%\r") == []
        assert a.value() == "50%"

    def test_one_absurdly_wide_line_is_capped(self) -> None:
        # A program that writes megabytes with neither newline nor carriage return
        # must not make the assembler grow without bound.
        a = _LineAssembler()
        a.feed("z" * (_MAX_LINE_CELLS * 2))
        assert len(a.value()) == _MAX_LINE_CELLS

    def test_the_fast_path_and_the_slow_path_agree(self) -> None:
        # `feed` short-circuits when a chunk has no `\r`; the two paths must produce
        # identical output or the optimisation is a behaviour change.
        text = "alpha\nbeta\ngamma"
        plain = _LineAssembler()
        forced = _LineAssembler()
        assert plain.feed(text) == forced.feed(text.replace("alpha", "\ralpha"))
        assert plain.value() == forced.value()


class TestTheFirstReadIsBounded:
    """A `.err` on this cluster reaches 1+ MB; the reader must never slurp it."""

    def _big(self, tmp_path: Path, lines: int = 20000) -> Path:
        p = tmp_path / "big.out"
        p.write_text("".join(f"line {i:06d} {'y' * 60}\n" for i in range(lines)))
        return p

    def test_only_the_window_is_read_and_it_is_the_END_of_the_file(self, tmp_path: Path) -> None:
        path = self._big(tmp_path)
        assert path.stat().st_size > 1_000_000, "the case this test is about"
        tail = LogTail(str(path), window_bytes=8192)
        chunk = tail.read()
        # Bounded by the window, not by the file: 8 KiB of ~68-byte lines.
        assert 100 < len(chunk.lines) < 130, len(chunk.lines)
        assert chunk.lines[-1].startswith("line 019999")
        assert chunk.tail_only is True
        assert chunk.size == path.stat().st_size

    def test_the_line_the_window_landed_inside_is_dropped_not_shown_as_a_line(
        self, tmp_path: Path
    ) -> None:
        # Seeking to `size - window` lands mid-line; showing that fragment as a
        # complete line would put a truncated, meaningless first row on screen.
        path = self._big(tmp_path)
        chunk = LogTail(str(path), window_bytes=8192).read()
        assert all(ln.startswith("line 0") for ln in chunk.lines), chunk.lines[0]

    def test_a_file_inside_the_window_is_shown_whole(self, tmp_path: Path) -> None:
        path = tmp_path / "small.out"
        path.write_text("a\nb\nc\n")
        chunk = LogTail(str(path), window_bytes=8192).read()
        assert chunk.lines == ["a", "b", "c"]
        assert chunk.tail_only is False  # nothing was hidden, so say nothing

    def test_a_window_with_no_line_break_at_all_is_kept(self, tmp_path: Path) -> None:
        # One unbroken blob (a progress bar that never got a newline): there is no
        # boundary to trim to, and showing the end beats showing nothing.
        path = tmp_path / "blob.out"
        path.write_text("q" * 50_000)
        chunk = LogTail(str(path), window_bytes=1024).read()
        assert chunk.lines == []
        assert len(chunk.partial) == 1024
        assert chunk.tail_only is True

    def test_a_single_read_cannot_hand_back_unbounded_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "many.out"
        path.write_text("".join(f"{i}\n" for i in range(500)))
        chunk = LogTail(str(path), max_lines=10).read()
        assert len(chunk.lines) == 10
        assert chunk.lines[-1] == "499"  # the newest survive
        assert chunk.dropped == 490  # ...and the loss is reported, not hidden

    def test_window_bytes_is_readable_so_the_ui_can_name_it(self, tmp_path: Path) -> None:
        assert LogTail(str(tmp_path / "x"), window_bytes=4096).window_bytes == 4096


class TestFollowingALiveFile:
    def test_new_bytes_only_are_returned_on_the_next_read(self, tmp_path: Path) -> None:
        path = tmp_path / "live.out"
        path.write_text("one\ntwo\n")
        tail = LogTail(str(path))
        assert tail.read().lines == ["one", "two"]
        with path.open("a") as fh:
            fh.write("three\n")
        chunk = tail.read()
        assert chunk.lines == ["three"], "the earlier lines must not repeat"
        assert chunk.reset is False

    def test_a_last_line_with_no_newline_is_partial_never_a_line(self, tmp_path: Path) -> None:
        # The normal state of a file being appended to. Committing it would print a
        # half-line as finished — and then print it again when the rest arrives.
        path = tmp_path / "half.out"
        path.write_text("done\nhalf")
        tail = LogTail(str(path))
        chunk = tail.read()
        assert chunk.lines == ["done"]
        assert chunk.partial == "half"

    def test_the_rest_of_a_half_line_completes_it_without_losing_the_start(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "half.out"
        path.write_text("half")
        tail = LogTail(str(path))
        assert tail.read().partial == "half"
        with path.open("a") as fh:
            fh.write("-and-half\n")
        chunk = tail.read()
        assert chunk.lines == ["half-and-half"]
        assert chunk.partial == ""

    def test_nothing_new_is_an_empty_chunk_not_a_repeat(self, tmp_path: Path) -> None:
        path = tmp_path / "idle.out"
        path.write_text("only\n")
        tail = LogTail(str(path))
        tail.read()
        chunk = tail.read()
        assert chunk.lines == [] and chunk.partial == "" and chunk.error == ""

    def test_a_live_progress_bar_is_visible_before_its_newline_arrives(
        self, tmp_path: Path
    ) -> None:
        # The single most interesting line in a training log is usually the one with
        # no newline yet, so `partial` has to carry the collapsed CURRENT state.
        path = tmp_path / "tqdm.err"
        path.write_text("\r 10%|#   | 10/100")
        tail = LogTail(str(path))
        assert tail.read().partial == " 10%|#   | 10/100"
        with path.open("a") as fh:
            fh.write("\r 90%|### | 90/100")
        assert tail.read().partial == " 90%|### | 90/100"


class TestAFileThatGoesBackwards:
    def test_truncation_in_place_resets_the_view(self, tmp_path: Path) -> None:
        path = tmp_path / "rot.out"
        path.write_text("old one\nold two\n")
        tail = LogTail(str(path))
        tail.read()
        path.write_text("new\n")  # size went backwards
        chunk = tail.read()
        assert chunk.reset is True, "the caller must clear what it was showing"
        assert chunk.lines == ["new"]

    def test_rotation_away_and_back_picks_the_new_file_up(self, tmp_path: Path) -> None:
        path = tmp_path / "gone.out"
        path.write_text("before\n")
        tail = LogTail(str(path))
        tail.read()
        path.unlink()
        assert tail.read().error != ""
        path.write_text("after\n")
        chunk = tail.read()
        assert chunk.reset is True and chunk.lines == ["after"]

    def test_a_half_line_from_the_old_file_is_not_glued_onto_the_new_one(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "glue.out"
        path.write_text("stale-fragment")
        tail = LogTail(str(path))
        assert tail.read().partial == "stale-fragment"
        path.write_text("fresh\n")
        chunk = tail.read()
        assert chunk.lines == ["fresh"], "the old fragment must not survive the reset"


class TestEveryWayThereIsNothingToRead:
    """All of these are ordinary states of a Slurm log, so all of them are strings.

    A job that has produced no output HAS no `.out` file; another user's job has one
    this account cannot open. Raising on either would make the viewer's normal case
    an error path.
    """

    def test_a_file_that_does_not_exist_yet(self, tmp_path: Path) -> None:
        chunk = LogTail(str(tmp_path / "nothing.out")).read()
        assert "no output file" in chunk.error
        assert chunk.lines == [] and chunk.size == -1

    def test_it_starts_working_the_moment_the_job_creates_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "later.out"
        tail = LogTail(str(path))
        assert tail.read().error != ""
        path.write_text("first output\n")
        chunk = tail.read()
        assert chunk.error == "" and chunk.lines == ["first output"]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read a 000 file")
    def test_an_unreadable_file_says_why(self, tmp_path: Path) -> None:
        path = tmp_path / "secret.out"
        path.write_text("someone else's job\n")
        path.chmod(0)
        try:
            chunk = LogTail(str(path)).read()
        finally:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        assert "permission denied" in chunk.error.lower(), chunk.error
        assert "someone else" not in chunk.error

    def test_a_directory_is_named_as_such(self, tmp_path: Path) -> None:
        assert "directory" in LogTail(str(tmp_path)).read().error

    def test_a_fifo_is_refused_rather_than_blocking_forever(self, tmp_path: Path) -> None:
        # `--output=` takes any path, and `open()` on a FIFO with no writer blocks
        # indefinitely — which, on the UI thread, is the whole app.
        path = tmp_path / "pipe.out"
        os.mkfifo(path)
        assert "not a regular file" in LogTail(str(path)).read().error

    def test_an_empty_file_is_not_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.out"
        path.touch()
        chunk = LogTail(str(path)).read()
        assert chunk.error == "" and chunk.lines == [] and chunk.size == 0


class TestBytesThatAreNotText:
    def test_non_utf8_bytes_are_replaced_not_raised(self, tmp_path: Path) -> None:
        path = tmp_path / "binary.out"
        path.write_bytes(b"good\n\xff\xfe garbage\n")
        chunk = LogTail(str(path)).read()
        assert chunk.lines[0] == "good"
        assert "garbage" in chunk.lines[1]
        assert "�" in chunk.lines[1]

    def test_a_multibyte_character_split_across_two_reads_is_not_mangled(
        self, tmp_path: Path
    ) -> None:
        # The reader stops wherever the file currently ends, which can be halfway
        # through a UTF-8 sequence. A plain `bytes.decode` would corrupt text that is
        # perfectly valid, purely because of where the job happened to have flushed.
        path = tmp_path / "utf8.out"
        snowman = "\N{SNOWMAN}".encode()
        assert len(snowman) == 3
        path.write_bytes(b"x" + snowman[:1])
        tail = LogTail(str(path))
        first = tail.read()
        assert "�" not in first.partial, first.partial
        with path.open("ab") as fh:
            fh.write(snowman[1:] + b"\n")
        chunk = tail.read()
        assert chunk.lines == ["x\N{SNOWMAN}"]


class TestSlurmFilenamePatterns:
    def test_the_common_patterns_resolve(self) -> None:
        path, unresolved = expand_log_pattern(
            "/scratch/%u/slurm-%j.out", {"u": "ada", "j": "52330903"}
        )
        assert path == "/scratch/ada/slurm-52330903.out"
        assert unresolved == []

    def test_an_array_task_resolves_from_its_own_two_fields(self) -> None:
        path, unresolved = expand_log_pattern("%A_%a.out", {"A": "999", "a": "7"})
        assert (path, unresolved) == ("999_7.out", [])

    def test_a_width_zero_pads_a_numeric_field(self) -> None:
        assert expand_log_pattern("%4j.out", {"j": "42"})[0] == "0042.out"

    def test_a_doubled_percent_is_a_literal_percent(self) -> None:
        assert expand_log_pattern("100%%.out", {"j": "1"})[0] == "100%.out"

    def test_a_pattern_only_a_task_can_resolve_is_REPORTED_not_guessed(self) -> None:
        # Guessing `%t` = 0 would hand back a path that does not exist and blame the
        # filesystem for it; the caller needs to be able to say which pattern beat us.
        path, unresolved = expand_log_pattern("run-%j-%t.out", {"j": "5"})
        assert unresolved == ["%t"]
        assert path == "run-5-%t.out", "the unresolved pattern is left visible"

    def test_a_bare_percent_in_a_real_path_is_left_alone(self) -> None:
        # `%o` is not a Slurm pattern, so a directory literally called "100%off" must
        # not be reported as an unresolvable pattern.
        path, unresolved = expand_log_pattern("/scratch/100%off/run.out", {"j": "1"})
        assert (path, unresolved) == ("/scratch/100%off/run.out", [])

    def test_a_path_with_no_patterns_is_returned_untouched(self) -> None:
        plain = "/project/rcc/youzhi/2026-08.out"
        assert expand_log_pattern(plain, {"j": "1"}) == (plain, [])


class TestANewFileAtTheSamePathIsNotResumedIntoMidway:
    """A byte offset only means something relative to one particular file.

    `read`'s own comment says `size < self._pos` covers "truncated in place, **or a
    new file at the same path**". It covered the second case only when the
    replacement happened to be shorter. A same-size or larger new file left `_pos`
    pointing into a file it was never an offset into, so the reader resumed
    mid-content, never showed the new file's first bytes, and reported
    `reset=False` — so the viewer did not clear what it was already showing.

    Slurm does not rotate job output, but a user moving a `.out` aside while
    watching it does exactly this, and the failure is the silent kind: the wrong
    bytes with no diagnostic.
    """

    @staticmethod
    def _replace(path: str, text: str) -> None:
        """A genuinely new inode at the same path, via rename.

        Deliberately NOT `os.remove` + create: that reuses the freed inode on
        local ext4/xfs (measured on this host: 8 of 8 trials on `/tmp`, 0 of 8 on
        GPFS), so a test written that way passes or fails depending on where
        `tmp_path` happens to live. Renaming a file that existed at the same time
        as the old one guarantees a different inode on every filesystem, and it is
        also what a user actually does — `mv fixed.out job.out`.
        """
        tmp = path + ".incoming"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.rename(tmp, path)

    def test_a_larger_replacement_is_not_skipped_into(self, tmp_path: Any) -> None:
        p = tmp_path / "j.out"
        p.write_text("old-1\nold-2\n")  # 12 bytes
        tail = LogTail(str(p))
        assert tail.read().lines == ["old-1", "old-2"]

        self._replace(str(p), "NEW-a\nNEW-b\nNEW-c\nNEW-d\n")  # 24 bytes, larger
        chunk = tail.read()
        assert chunk.lines == ["NEW-a", "NEW-b", "NEW-c", "NEW-d"], chunk.lines
        assert chunk.reset is True, "the viewer must clear the old file's lines"

    def test_a_same_size_replacement_is_noticed(self, tmp_path: Any) -> None:
        """The case the size comparison can never see, by construction.

        Note the honest limit this does NOT claim: an in-place rewrite that keeps
        the inode and does not shrink the file is indistinguishable from an append
        using `stat` alone, so it is not detected and cannot be. That is why the
        replacement here arrives by rename, which is how a file at a path is
        really swapped.
        """
        p = tmp_path / "j.out"
        p.write_text("one\ntwo\n")
        tail = LogTail(str(p))
        tail.read()
        self._replace(str(p), "AAA\nBBB\n")  # byte-for-byte the same length
        chunk = tail.read()
        assert chunk.lines == ["AAA", "BBB"], chunk.lines
        assert chunk.reset is True

    def test_a_truncation_in_place_still_resets(self, tmp_path: Any) -> None:
        """CONTROL — the case the size comparison always caught, same inode.

        Passes before and after: a fix that only compared identity and dropped the
        size test would break this, because truncating in place keeps the inode.
        """
        p = tmp_path / "j.out"
        p.write_text("alpha\nbravo\ncharlie\n")
        tail = LogTail(str(p))
        tail.read()
        p.write_text("hi\n")  # same inode, shorter
        chunk = tail.read()
        assert chunk.lines == ["hi"]
        assert chunk.reset is True

    def test_an_ordinary_append_is_still_incremental(self, tmp_path: Any) -> None:
        """CONTROL — the hot path. Same inode, growing: only the new bytes, no reset.

        This is what stops the fix from being "reset on every read", which would
        satisfy both tests above and re-render the whole window every 0.5 s.
        """
        p = tmp_path / "j.out"
        p.write_text("one\n")
        tail = LogTail(str(p))
        assert tail.read().lines == ["one"]
        with open(p, "a") as fh:
            fh.write("two\n")
        chunk = tail.read()
        assert chunk.lines == ["two"], "an append must not re-deliver old lines"
        assert chunk.reset is False, "an append is not a restart"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can open a 000 file")
class TestAReadThatNeverHappenedIsNotAStart:
    """A fresh read that could not open the file must not leave a resume behind it.

    `read` decides "this is a new view of the file" — a first read, a truncation, a
    replacement — and throws `_pos` away *before* it opens anything. If that open then
    fails, `_started` was left set, so the NEXT read saw `_started` true with
    `_pos == 0` and treated it as an ordinary resume from byte zero. Both of this
    module's promises went with it: `reset` came back False, so the viewer appended the
    new file's lines underneath the old file's with no "restarted" note, and the read
    was unbounded, because a resume never seeks to ``size - window``.

    Not a hypothetical pairing: `os.stat` succeeding while `open` fails is what a
    replaced file looks like through an NFS/GPFS `ESTALE`, and what a 000-mode file
    swapped in at the path looks like on any filesystem — which is how these tests
    produce it.
    """

    def test_a_restart_whose_read_failed_still_resets_the_next_one(self, tmp_path: Path) -> None:
        path = tmp_path / "j.out"
        path.write_text("old-1\nold-2\n")
        tail = LogTail(str(path))
        assert tail.read().lines == ["old-1", "old-2"]

        path.write_text("new\n")  # truncated in place: the next read is a restart
        path.chmod(0)  # ...and it cannot be opened
        assert "permission denied" in tail.read().error

        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        chunk = tail.read()
        assert chunk.lines == ["new"]
        assert chunk.reset is True, "the viewer is still showing the old file's lines"

    def test_a_restart_whose_read_failed_is_still_bounded_when_it_lands(
        self, tmp_path: Path
    ) -> None:
        # The other half: a resume reads from `_pos`, so a lost restart also reads the
        # whole file instead of the last window — the one thing this module promises
        # never to do.
        path = tmp_path / "j.out"
        path.write_text("".join(f"old {i:05d}\n" for i in range(20)))
        tail = LogTail(str(path), window_bytes=64)
        tail.read()

        path.write_text("short\n")  # restart
        path.chmod(0)
        assert tail.read().error != ""
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.write_text("".join(f"new {i:05d}\n" for i in range(40)))  # 400 bytes

        chunk = tail.read()
        assert chunk.tail_only is True, "a 64-byte window over 400 bytes hid something"
        assert len(chunk.lines) < 10, f"{len(chunk.lines)} lines is the whole file"
        assert chunk.lines[-1] == "new 00039", "and it is the END that survived"

    def test_a_failed_read_mid_stream_resumes_where_it_left_off(self, tmp_path: Path) -> None:
        """CONTROL — the same failure on a read that was NOT a restart.

        Passes before and after: nothing was thrown away, so `_pos` is still a valid
        offset into this file and recovery must be an ordinary incremental read. This
        is what stops the fix from being "forget the position whenever a read fails",
        which would re-deliver the whole window on every filesystem hiccup.
        """
        path = tmp_path / "j.out"
        path.write_text("one\n")
        tail = LogTail(str(path))
        assert tail.read().lines == ["one"]

        with path.open("a") as fh:
            fh.write("two\n")
        path.chmod(0)
        assert tail.read().error != ""
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)

        chunk = tail.read()
        assert chunk.lines == ["two"], "the earlier lines must not repeat"
        assert chunk.reset is False, "a hiccup is not a restart"

    def test_an_unreadable_file_that_becomes_readable_is_read_from_its_tail(
        self, tmp_path: Path
    ) -> None:
        """CONTROL — a FIRST read that failed. Passes before and after.

        `_started` was already False here, so this path was never broken; it is in the
        suite because the fix's mechanism is `_started`, and reaching the same verdict
        by a different route is what says the fix generalised rather than special-cased.
        """
        path = tmp_path / "j.out"
        path.write_text("".join(f"line {i:05d}\n" for i in range(20)))
        path.chmod(0)
        tail = LogTail(str(path), window_bytes=64)
        assert tail.read().error != ""
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)

        chunk = tail.read()
        assert chunk.reset is True
        assert chunk.tail_only is True
        assert chunk.lines[-1] == "line 00019"
        assert len(chunk.lines) < 10, "the window, not the file"
