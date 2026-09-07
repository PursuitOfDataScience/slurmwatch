"""What happens to the exit code when nobody is reading stdout.

Two conditions, both of which a cron job or a `| head` reaches, and both of which
replaced this tool's exit code with something that is not one of its codes.

**fd 1 closed** (`sw --once --json >&-`, and what a daemon started with closed
descriptors gives). CPython sets `sys.stdout` to `None` and makes `print()` a
silent no-op, so the run completed and only the explicit flush assumed a stream:

    File ".../slurmwatch/cli.py", line 1209, in _once_loop
      sys.stdout.flush()
    AttributeError: 'NoneType' object has no attribute 'flush'

`AttributeError` is not an `OSError`, so it passed through every handler.

The same condition split the two output formats apart, which is how a third
defect hid behind the first: `print()` is a no-op with no stdout, but
`csv.writer(None)` raises `TypeError: argument 1 must have a "write" method`, so
`--json` came back cleanly from a closed fd 1 while `--format csv` died in
`_emit_no_telemetry_facts` on the same input.

**A reader that closes the pipe.** `_once_loop` already flushed inside its `try`
so that the `BrokenPipeError` handler could turn it into a quiet 0 -- but the
early-out paths (`job_unknown`, pending, foreign, usage-not-sampled) print a
facts payload and *return*, and their flush only **suppressed** the error.
Suppressing leaves the unwritten bytes in the buffer, the interpreter's shutdown
flush retries them, fails again, and replaces the code with **120**. Measured
before this fix:

    sw --once --json 999999999 | head -n 0    -> 120
    sw --once --json 999999999 > /dev/null    -> 1

120 tells a right-sizing script neither "measured" nor "no such job".  Both are
now `_flush_quietly`, which points fd 1 at `/dev/null` on failure so the retry
has nothing to fail on.

The control is the case that must NOT change: after a *successful* measurement a
broken pipe is still a quiet 0, which is a separate promise made by a separate
handler, and the one a "just suppress everything" fix would break.
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import shutil
import signal
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: A job id no cluster has issued, so the `job_unknown` early-out is reached
#: without depending on what is queued.
ABSENT_JOB = "999999999"

pytestmark = pytest.mark.skipif(
    shutil.which("scontrol") is None,
    reason="needs Slurm's client commands to reach the emit paths",
)


def _env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
        "COLUMNS": "200",
    }


def _shell(script: str) -> subprocess.CompletedProcess[str]:
    """Run one shell line with slurmwatch's own exit code preserved.

    `bash`, and `PIPESTATUS`, because the code under test is the one at the head
    of a pipeline -- `$?` there is the *reader's*, which is how a first pass at
    this measurement read 0 where the tool had actually exited 120.
    """
    return subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=280,
        cwd=str(ROOT),
        env=_env(),
    )


SW = '"$PYTHON" -m slurmwatch'


def _cmd(script: str) -> str:
    return f"PYTHON={sys.executable!r}; {script}"


class TestFd1Closed:
    def test_a_closed_fd_1_is_not_a_crash(self) -> None:
        done = _shell(_cmd(f"exec {SW} --once --json {ABSENT_JOB} >&-"))
        assert "Traceback" not in done.stderr, done.stderr[-600:]
        assert "AttributeError" not in done.stderr, done.stderr[-600:]

    def test_and_the_exit_code_is_the_one_the_job_earns(self) -> None:
        # 1 = "no telemetry", the same code the reader-attached run gives. Who is
        # listening is not a fact about the job.
        done = _shell(_cmd(f"exec {SW} --once --json {ABSENT_JOB} >&-"))
        assert done.returncode == 1, done.stderr[-600:]

    def test_the_csv_format_too(self) -> None:
        done = _shell(_cmd(f"exec {SW} --once --format csv {ABSENT_JOB} >&-"))
        assert "Traceback" not in done.stderr, done.stderr[-600:]
        assert done.returncode == 1, done.stderr[-600:]


class TestAReaderThatClosesThePipe:
    def test_an_early_out_no_longer_exits_120(self) -> None:
        done = _shell(
            _cmd(
                f"{SW} --once --json {ABSENT_JOB} 2>/dev/null | head -n 0; exit ${{PIPESTATUS[0]}}"
            )
        )
        assert done.returncode != 120, "the shutdown flush replaced the exit code again"
        assert done.returncode == 1

    def test_the_same_command_with_a_reader_attached_is_unchanged(self) -> None:
        # The comparison that made 120 visible as a defect rather than a choice.
        done = _shell(_cmd(f"{SW} --once --json {ABSENT_JOB} >/dev/null 2>&1"))
        assert done.returncode == 1

    def test_no_ignored_exception_is_printed(self) -> None:
        done = _shell(_cmd(f"{SW} --once --json {ABSENT_JOB} 2>&1 >/dev/null | cat"))
        assert "Exception ignored" not in done.stdout, done.stdout[-400:]


class TestTheQuietZeroAfterASuccessfulMeasurementSurvives:
    """The control. A separate handler, a separate promise.

    `_once_loop`'s own flush must keep RAISING `BrokenPipeError` so that handler
    can exit 0 -- so this is the assertion that fails if `_flush_quietly` is
    applied there too, which is the tempting one-line version of this fix.
    """

    def test_the_measured_path_still_exits_0_on_a_broken_pipe(self) -> None:
        import getpass

        listed = subprocess.run(
            ["squeue", "-h", "-u", getpass.getuser(), "-t", "RUNNING", "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        running = [j for j in listed.stdout.split() if j.isdigit()]
        if not running:
            pytest.skip("no running job of this user to measure")
        done = _shell(
            _cmd(
                f"{SW} --once --json {running[0]} 2>/dev/null | head -n 0; exit ${{PIPESTATUS[0]}}"
            )
        )
        assert done.returncode == 0, done.stderr[-400:]

    def test_the_flush_in_once_loop_is_not_the_quiet_helper(self) -> None:
        """Read from the source, because no cluster state can prove a negative.

        Both flushes look identical at the call site; only one of them may
        swallow the error, and swapping them is silent.
        """
        source = (ROOT / "src" / "slurmwatch" / "cli.py").read_text()
        head, _, tail = source.partition("async def _once_loop(")
        assert tail, "_once_loop moved; this test needs re-pointing"
        body = tail.partition("\ndef ")[0]
        assert "if sys.stdout is not None:\n            sys.stdout.flush()" in body, (
            "the post-measurement flush in `_once_loop` must stay a plain guarded "
            "flush; its BrokenPipeError is what the handler below it converts to 0"
        )


class TestTheHelperItself:
    def test_a_stream_that_cannot_flush_is_not_reraised(self) -> None:
        from slurmwatch.cli import _flush_quietly

        class _Broken(io.StringIO):
            def flush(self) -> None:
                raise BrokenPipeError(32, "Broken pipe")

        _flush_quietly(_Broken())  # must not raise

    def test_none_is_accepted(self) -> None:
        from slurmwatch.cli import _flush_quietly

        _flush_quietly(None)

    def test_a_healthy_stream_is_actually_flushed(self) -> None:
        from slurmwatch.cli import _flush_quietly

        class _Counting(io.StringIO):
            flushes = 0

            def flush(self) -> None:
                type(self).flushes += 1

        stream = _Counting()
        _flush_quietly(stream)
        assert _Counting.flushes == 1, "the helper stopped flushing altogether"


class TestATerminalOnStdinAndAClosedStdout:
    """The configuration that hid this for a whole test matrix.

    Thirteen callers asked `sys.stdout.isatty()`, and most were reached only
    through `sys.stdin.isatty() and ...` -- so a harness whose stdin is a pipe
    short-circuits every one of them and the bug is invisible. `sw 12345 >&-`
    typed at a prompt has a *terminal* on stdin and `None` for stdout, and raised
    `AttributeError: 'NoneType' object has no attribute 'isatty'` from the line
    deciding whether to draw a TUI. Hence a pty here rather than a pipe.

    `--demo`, so this needs no cluster and no live job.
    """

    #: Long enough for the TUI to mount and paint, short enough not to dominate
    #: the suite. The closed-stdout child exits on its own well before this.
    WINDOW = 14.0

    @staticmethod
    def _drive(argv: list[str], *, close_stdout: bool, window: float) -> tuple[int, str]:
        """Run slurmwatch on a pty, optionally with fd 1 closed. Returns (rc, output).

        Teardown is `SIGKILL`, and it keeps draining the master the whole way.
        A first version asked politely -- `q`, then `SIGTERM` -- and blocked in
        `waitpid` for ten minutes: a TUI mid-write to a pty nobody is reading
        blocks in `write`, so the signal changed nothing, and a test that needs the
        app's signal handling to end is testing that instead of what it claims to.
        Whether `SIGTERM` leaves the terminal clean is a real property with its own
        tests elsewhere. Here the child is simply not allowed to outlive the
        assertion.
        """
        import fcntl
        import pty
        import select
        import struct
        import termios
        import time

        pid, master = pty.fork()
        if pid == 0:  # pragma: no cover - the child execs
            try:
                fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
                if close_stdout:
                    os.close(1)
                env = _env()
                env["TERM"] = "xterm-256color"
                if not close_stdout:
                    env.pop("NO_COLOR", None)
                os.chdir(str(ROOT))
                os.execve(sys.executable, [sys.executable, "-m", "slurmwatch", *argv], env)
            finally:
                os._exit(127)

        chunks: list[bytes] = []
        status = 0
        reaped = False

        def pump(seconds: float) -> bool:
            """Drain the pty for `seconds`. True once the child has been reaped."""
            nonlocal status, reaped
            end = time.time() + seconds
            while time.time() < end:
                ready, _, _ = select.select([master], [], [], 0.2)
                if ready:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        data = b""
                    if not data:
                        break
                    chunks.append(data)
                got, state = os.waitpid(pid, os.WNOHANG)
                if got:
                    status, reaped = state, True
                    return True
            return reaped

        try:
            if not pump(window):
                os.kill(pid, signal.SIGKILL)
                pump(10.0)
        except (OSError, ChildProcessError):
            pass
        finally:
            with contextlib.suppress(OSError):
                os.close(master)
            if not reaped:
                with contextlib.suppress(ChildProcessError, OSError):
                    _, status = os.waitpid(pid, 0)

        rc = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 128 + os.WTERMSIG(status)
        return rc, b"".join(chunks).decode("utf-8", "replace")

    def test_a_tty_on_stdin_and_no_stdout_is_not_a_crash(self) -> None:
        rc, text = self._drive(["--demo"], close_stdout=True, window=self.WINDOW)
        assert "AttributeError" not in text, text[-700:]
        assert "Traceback" not in text, text[-700:]
        assert rc == 0, f"rc={rc}\n{text[-700:]}"

    def test_a_terminal_on_both_still_draws_the_dashboard(self) -> None:
        """The control, and the one that matters.

        A predicate that answered `False` unconditionally would pass every test
        above and silently turn the TUI off for everyone.
        """
        # Killed at the end of the window, so the exit code is 128+9 and says
        # nothing here; the painted bytes are the measurement.
        _rc, text = self._drive(["--demo"], close_stdout=False, window=self.WINDOW)
        assert "\033[?1049h" in text, "never entered the alternate screen"
        drawn = sum(text.count(ch) for ch in "─│╭╮╰╯━┃")
        assert drawn > 100, f"only {drawn} box-drawing characters; the TUI did not paint"
