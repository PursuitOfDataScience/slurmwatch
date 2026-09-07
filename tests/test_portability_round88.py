"""SW-90 — auto-discovery reported "no jobs" while six of the user's jobs ran.

Round 87 audited this surface and passed it, and its conclusion was right about
the code::

    `resolve_current_jobs` correctly RAISES on a `squeue` failure rather than
    returning an empty list, so "couldn't ask" is already distinct from
    "nothing there".

It does.  The guard is ``returncode != 0`` — and on a node whose name service
cannot resolve the user, ``squeue`` fails **with exit 0**:

===================  ========================================  ======
command              stderr                                    exit
===================  ========================================  ======
``squeue -u youzhi``  ``squeue: error: Invalid user: youzhi``   **0**
``squeue --me``       ``squeue: error: Invalid user: 940...``   **0**
``sacct -u youzhi``   ``sacct: error: Invalid user id: ...``    1
===================  ========================================  ======

So nothing about the error handling was careless: it trusted a convention Slurm
does not hold to.  That is also why the sibling package surfaces the same
underlying failure correctly — ``sacct`` sets an exit code and ``squeue`` does
not.

Three guards, LAYERED — detect the exit-0 failure, rephrase the question by uid,
and only then read the job this process is standing in.  The first attempt made the
last of those the FIRST thing tried, which is a different bug: `$SLURM_JOB_ID` is
inherited by every child of an allocation, so a long-lived tmux inside a
reservation job made bare `slurmwatch` monitor the reservation instead of the
user's work, with no override.  This is the entry point a new user meets first.
"""

import os
import subprocess
from typing import TypedDict

import pytest

import slurmwatch.slurm as slurm
from slurmwatch.cli import _job_id_from_environment
from slurmwatch.exceptions import SlurmCommandError
from slurmwatch.slurm import _run_slurm_cmd, resolve_current_jobs


class _DriveResult(TypedDict):
    """What `_drive` observes on the pty. Counts are ints; the child's fate is not
    known until it exits, so `code` and `signalled` are Optional -- mypy caught a
    `dict[str, int]` here hiding exactly that."""

    entered: int
    left: int
    code: int | None
    signalled: bool | None
    traceback: bool


class _StartupResult(TypedDict):
    """The startup-window variant: same shape, without the traceback probe."""

    opened: int
    closed: int
    signalled: bool | None
    code: int | None


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _not_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURMWATCH_MOCK", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)


# --------------------------------------------------------------------------
# (1) a zero exit is not proof the command worked
# --------------------------------------------------------------------------
class TestAZeroExitIsNotProofOfSuccess:
    def test_the_measured_case_is_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verbatim from a compute node, inside an allocation."""
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: _Result(0, "", "squeue: error: Invalid user: youzhi\n"),
        )
        with pytest.raises(SlurmCommandError) as exc:
            _run_slurm_cmd(["squeue", "-u", "youzhi", "-h", "-o", "%i"])
        # The scheduler's own words, kept: this is the sentence that says what is
        # wrong, and it was being discarded along with the failure.
        assert "Invalid user" in str(exc.value)

    @pytest.mark.parametrize(
        "stderr",
        [
            "squeue: error: Invalid user: youzhi\n",
            "squeue: error: Invalid user: 940740166\n",
            "sacct: error: Invalid user id: youzhi\n",
            "squeue: error: Unknown user nosuchuser\n",
            "prefix line\nsqueue: error: Invalid user: x\n",
        ],
    )
    def test_an_unresolvable_subject_with_no_output_is_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, stderr: str
    ) -> None:
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(0, "", stderr))
        with pytest.raises(SlurmCommandError):
            _run_slurm_cmd(["squeue", "-h", "-o", "%i"])

    @pytest.mark.parametrize(
        "stderr",
        [
            # THE case the first version of this guard got wrong. Measured on this
            # login node: rc=0, stdout empty, and this on stderr. A job with no
            # live steps is the ordinary state of a job, and "nothing to report"
            # is the right answer to it, not an exception.
            "sstat: error: couldn't get steps for job 54117243\n",
            "sstat: error: No steps running for job 54117243\n",
            # Not this surface's business either: a bad job id sets a nonzero exit
            # status and puts its sentence on STDOUT (SW-7), so the rc!=0 path
            # already handles it and `_is_missing_job_error` already classifies it.
            "scontrol: error: Invalid job id specified\n",
            "squeue: error: Unable to contact slurm controller\n",
        ],
    )
    def test_a_legitimately_empty_answer_is_not_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, stderr: str
    ) -> None:
        """The narrowing. Only the identity/lookup class makes rc=0 a failure.

        `sstat --allsteps --noheader -P -j <id>` answers exit 0 / empty stdout /
        `sstat: error: …` for every job that has no live steps — which is most
        jobs, most of the time.
        """
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(0, "", stderr))
        assert _run_slurm_cmd(["sstat", "--allsteps", "-j", "54117243"]) == ""

    def test_a_genuinely_empty_queue_is_still_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The control, and the one that matters most.

        An empty queue is the ordinary state of a cluster account, and turning it
        into an exception would replace one wrong answer with a louder one.
        """
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(0, "", ""))
        assert _run_slurm_cmd(["squeue", "-h", "-o", "%i"]) == ""

    def test_output_with_a_warning_beside_it_is_still_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both halves of the condition must hold. A command that answered rows
        # keeps its answer whatever it wrote to stderr.
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: _Result(
                0, "9001|R\n", "squeue: error: some partitions were not returned\n"
            ),
        )
        assert _run_slurm_cmd(["squeue", "-h", "-o", "%i|%t"]) == "9001|R\n"

    @pytest.mark.parametrize(
        "stderr",
        [
            "a job named error: do not match this\n",
            "ERROR: shouting is not the format\n",
            "srun: Job step created\n",
            "  squeue: error:\n",
        ],
    )
    def test_prose_that_merely_contains_the_word_is_not_a_diagnostic(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stderr: str,
    ) -> None:
        # Anchored to a line start and a bare tool name, so a job name or a site
        # banner cannot make an empty queue look broken.
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(0, "", stderr))
        assert _run_slurm_cmd(["squeue", "-h", "-o", "%i"]) == ""

    def test_a_nonzero_exit_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: _Result(1, "Job 12345_9 not found", ""),
        )
        with pytest.raises(SlurmCommandError) as exc:
            _run_slurm_cmd(["scontrol", "show", "job", "12345_9"])
        # SW-7: the sentence is on STDOUT here, and losing it made a permanent
        # failure read as a transient one.
        assert "not found" in str(exc.value)


# --------------------------------------------------------------------------
# (2) the job this process is inside — a FALLBACK, after squeue has been asked
# --------------------------------------------------------------------------
class TestTheOwnJobIdIsTheLastResort:
    """``srun --jobid=<id> --overlap ... slurmwatch`` is the documented way to run
    this on a node, and Slurm sets ``$SLURM_JOB_ID`` for exactly that invocation.

    With an explicit job id everything on-node was already correct, including the
    owner name SW-1 used to get wrong.  Only discovery was affected — so
    discovery falls back to the answer it is standing in.

    A FALLBACK, and only that.  The first version read the variable at the top of
    `_auto_discover_job_id`, which pre-empted `squeue` everywhere it worked: a
    long-lived tmux started inside a reservation job exports it for the rest of the
    session, so bare `slurmwatch` on the login node monitored the reservation (job
    53834744) instead of the user's real work, with the picker unreachable and no
    way to override.  Discovery-first still closes SW-90, because asking the
    controller is precisely what fails on a node with no name service.
    """

    @pytest.mark.parametrize("name", ["SLURM_JOB_ID", "SLURM_JOBID"])
    def test_either_spelling_is_read(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        monkeypatch.setenv(name, "48853294")
        assert _job_id_from_environment() == "48853294"

    def test_an_array_task_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "48853297_1")
        assert _job_id_from_environment() == "48853297_1"

    @pytest.mark.parametrize(
        "value",
        ["", "   ", "notajob", "12345;rm -rf /", "$(id)", "12345 67", "-1"],
    )
    def test_anything_that_is_not_a_job_id_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        # A stale export, or a site wrapper setting it to something else, must not
        # become a job id nobody asked about.
        monkeypatch.setenv("SLURM_JOB_ID", value)
        assert _job_id_from_environment() is None

    def test_whitespace_is_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "  53834744  ")
        assert _job_id_from_environment() == "53834744"

    def test_nothing_set_means_nothing_claimed(self) -> None:
        assert _job_id_from_environment() is None

    def test_it_is_used_when_squeue_finds_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(0, "", ""))
        monkeypatch.setenv("SLURM_JOB_ID", "48853294")
        got = _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        assert got == "48853294"

    def test_it_is_used_when_squeue_cannot_be_asked_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SW-90 itself: on a node with no name service, discovery RAISES.

        Nothing else in this file exercises the raise-then-fall-back edge, and it
        is the one the finding is actually about — the previous behaviour was
        `sys.exit(1)` with "launch a job first".
        """
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: _Result(0, "", "squeue: error: Invalid user: youzhi\n"),
        )
        monkeypatch.setattr(slurm, "_own_uid", lambda: None)
        monkeypatch.setenv("SLURM_JOB_ID", "48853294")
        assert _auto_discover_job_id(SlurmwatchConfig(), interactive=False) == "48853294"

    def test_where_it_came_from_is_said_on_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`logger.info` is invisible without `--verbose`.

        A stale export therefore produced a bare "Job 54117243 has finished" with
        nothing anywhere connecting that id to the environment it came out of.
        """
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(0, "", ""))
        monkeypatch.setenv("SLURM_JOB_ID", "54117243")
        _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        err = capsys.readouterr().err
        assert "SLURM_JOB_ID=54117243" in err
        assert "no running or pending job" in err

    def test_an_empty_queue_and_a_failed_query_do_not_say_the_same_thing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The distinction is the whole of SW-90, and the MESSAGE lost it.

        Both cases fall back to the environment id, and both printed "no job
        found via squeue" -- but on the node this finding is from, `squeue` does
        not come back empty. It errors, with exit 0. Saying "no job found" there
        states something false about a query that never answered, which is the
        same conflation one level up, and the real error was `logger.info`, so
        invisible without `--verbose`.
        """
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        monkeypatch.setenv("SLURM_JOB_ID", "48853294")
        said = {}
        for label, fn in (
            ("empty", lambda username=None, **k: []),
            (
                "raised",
                lambda username=None, **k: (_ for _ in ()).throw(
                    SlurmCommandError("squeue: error: Invalid user: youzhi")
                ),
            ),
        ):
            monkeypatch.setattr("slurmwatch.cli.resolve_current_jobs", fn)
            monkeypatch.setattr(
                "slurmwatch.cli.resolve_unmonitorable_jobs", lambda username=None, **k: []
            )
            got = _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
            assert got == "48853294", label
            said[label] = capsys.readouterr().err

        assert said["empty"] != said["raised"], (
            "the two cases print the same sentence, so the one distinction that "
            "matters on the node SW-90 is about is unreportable"
        )
        assert "no running or pending job" in said["empty"]
        # The scheduler's own words, on stderr where they can be seen.
        assert "could not ask" in said["raised"]
        assert "Invalid user" in said["raised"]

    def test_a_shell_inside_a_reservation_does_not_hijack_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression this fix undoes.

        `squeue` works fine on a login node, and the user's real jobs are right
        there; the reservation job whose tmux they happen to be typing in is not
        the thing they asked about.
        """
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: _Result(0, "9001|R|gpu|2|1:23|4:00:00|cn1|real work\n", ""),
        )
        monkeypatch.setenv("SLURM_JOB_ID", "53834744")
        got = _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        assert got == "9001", "the holder job must not pre-empt the real one"

    def test_squeue_is_asked_before_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked: list[tuple[object, ...]] = []

        def _record(*a: object, **k: object) -> _Result:
            asked.append(a)
            return _Result(0, "", "")

        monkeypatch.setattr(subprocess, "run", _record)
        monkeypatch.setenv("SLURM_JOB_ID", "48853294")
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        assert asked, "discovery is first; the environment is the fallback"


# --------------------------------------------------------------------------
# (3) discover by numeric uid, which needs no name resolution anywhere
# --------------------------------------------------------------------------
#: `squeue -h -o "%i|%t|%P|%D|%M|%l|%R|%U|%j"`, with `%u` deliberately absent:
#: on-node it prints `nobody` for every row, and `%U` carries the numeric uid.
_ROWS = (
    "9001|R|gpu|2|1:23|4:00:00|cn[001-002]|940740146|my training|job\n"
    "9002|R|gpu|1|0:10|1:00:00|cn003|999999|somebody else\n"
    "9003|PD|gpu|1|0:00|1:00:00|Resources|940740146|mine, queued\n"
)


class TestDiscoveryFallsBackToTheNumericUid:
    def _broken_name_service(
        self, monkeypatch: pytest.MonkeyPatch, rows: str = _ROWS, uid: int = 940740146
    ) -> None:
        def _run(cmd: list[str], **kwargs: object) -> _Result:
            if "-u" in cmd:
                return _Result(0, "", "squeue: error: Invalid user: youzhi\n")
            return _Result(0, rows, "")

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setattr(slurm, "_own_uid", lambda: uid)

    def test_the_users_jobs_are_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The whole finding: six running jobs, and "Launch a job first".
        self._broken_name_service(monkeypatch)
        jobs = resolve_current_jobs("youzhi")
        assert [j["job_id"] for j in jobs] == ["9001", "9003"]

    def test_another_users_job_is_not_adopted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._broken_name_service(monkeypatch)
        assert all(j["job_id"] != "9002" for j in resolve_current_jobs("youzhi"))

    def test_a_pipe_in_a_job_name_still_survives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The uid column goes BEFORE the name, not after it.

        The job name is the only free-form field, so it is placed last and a
        literal `|` inside it falls into the final `split()` field. Appending the
        uid after it put the uid in the ninth field, split the name at its own
        pipe, and `my training|job` came back as `my training`. Caught by the test
        that already pinned that property, which is what it is for.
        """
        self._broken_name_service(monkeypatch)
        jobs = resolve_current_jobs("youzhi")
        assert jobs[0]["name"] == "my training|job"

    def test_a_row_with_an_unreadable_uid_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A wrong owner here is somebody else's job on your screen.
        self._broken_name_service(monkeypatch, rows="9001|R|gpu|1|1:23|4:00:00|cn1|nobody|mine\n")
        assert resolve_current_jobs("youzhi") == []

    def test_only_an_identity_failure_triggers_the_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timeout is not helped by rephrasing the question.

        Retrying it unfiltered would ask a busy controller for the whole queue
        for nothing, so the fallback is keyed on the one error a numeric-uid
        query can actually get past. (rc=1 here, because rc=0 plus a non-identity
        diagnostic is now a legal empty answer — see the sstat case above.)
        """
        calls = []

        def _run(cmd: list[str], **kwargs: object) -> _Result:
            calls.append(cmd)
            return _Result(1, "", "squeue: error: Unable to contact slurm controller\n")

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setattr(slurm, "_own_uid", lambda: 940740146)
        with pytest.raises(SlurmCommandError) as exc:
            resolve_current_jobs("youzhi")
        assert "contact slurm controller" in str(exc.value)
        assert len(calls) == 1, "no second, unfiltered query"

    def test_the_name_query_is_still_tried_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The fallback gives up the `-u` filter, so the controller returns every
        # job in the queue. Fine for the tens of thousands a real cluster holds,
        # and the reason this is second rather than first.
        calls = []

        def _run(cmd: list[str], **kwargs: object) -> _Result:
            calls.append(cmd)
            return _Result(0, "9001|R|gpu|2|1:23|4:00:00|cn1|mine\n", "")

        monkeypatch.setattr(subprocess, "run", _run)
        jobs = resolve_current_jobs("youzhi")
        assert len(calls) == 1
        assert "-u" in calls[0]
        assert "%U" not in " ".join(calls[0])
        assert [j["job_id"] for j in jobs] == ["9001"]

    def test_the_numeric_uid_is_tried_as_a_FILTERED_query_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`squeue -u` documents accepting a numeric uid, so try that before
        giving up the filter entirely.

        Dropping `-u` straight away made the controller return the WHOLE queue —
        6,206 rows / 480 KB measured on this login node — for a question about one
        user, and it removed the bound that kept `_squeue_rows`' phantom-row
        residual inside the user's own jobs.
        """
        calls = []

        def _run(cmd: list[str], **kwargs: object) -> _Result:
            calls.append(cmd)
            if "youzhi" in cmd:
                return _Result(0, "", "squeue: error: Invalid user: youzhi\n")
            return _Result(0, "9001|R|gpu|2|1:23|4:00:00|cn1|mine\n", "")

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setattr(slurm, "_own_uid", lambda: 940740146)
        jobs = resolve_current_jobs("youzhi")
        assert [j["job_id"] for j in jobs] == ["9001"]
        assert len(calls) == 2, "name, then uid — and no unfiltered scan"
        assert calls[1][:3] == ["squeue", "-u", "940740146"]
        assert "%U" not in " ".join(calls[1]), "still filtered, so no local uid column"

    def test_the_unfiltered_scan_is_only_reached_when_both_filters_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def _run(cmd: list[str], **kwargs: object) -> _Result:
            calls.append(cmd)
            if "-u" in cmd:
                return _Result(0, "", "squeue: error: Invalid user: youzhi\n")
            return _Result(0, _ROWS, "")

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setattr(slurm, "_own_uid", lambda: 940740146)
        assert [j["job_id"] for j in resolve_current_jobs("youzhi")] == ["9001", "9003"]
        assert len(calls) == 3, "name, uid, then the whole queue"
        assert "-u" not in calls[2], "the last resort has no -u"

    def test_the_field_count_is_derived_from_the_format(self) -> None:
        """It was hand-written as `8` and the uid form has 9 fields.

        `_squeue_rows` decides where a record STARTS by counting pipes against this
        number, so drift here shows up as phantom entries in the job picker rather
        than as an error anywhere.
        """
        assert slurm._SQUEUE_FORMAT.count("|") + 1 == slurm._SQUEUE_FIELD_COUNT
        assert slurm._SQUEUE_UID_FORMAT.count("|") + 1 == slurm._SQUEUE_UID_FIELD_COUNT
        assert slurm._SQUEUE_UID_FIELD_COUNT == slurm._SQUEUE_FIELD_COUNT + 1
        assert slurm._SQUEUE_FORMAT.endswith("|%j")
        assert slurm._SQUEUE_UID_FORMAT.endswith("|%U|%j")

    def test_no_uid_means_no_guess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Without a uid there is nothing to filter on, so the original failure
        # stands rather than being replaced by every job on the cluster.
        def _run(cmd: list[str], **kwargs: object) -> _Result:
            return _Result(0, "", "squeue: error: Invalid user: youzhi\n")

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setattr(slurm, "_own_uid", lambda: None)
        with pytest.raises(SlurmCommandError):
            resolve_current_jobs("youzhi")


def test_the_three_fixes_layer_rather_than_overlap() -> None:
    """Three guards, in the order they are reached, not three interchangeable ones.

    This is the entry point a new user meets first, and the failure mode was a
    tool telling somebody to do the thing they had already done six times. So:
    detect the exit-0 failure, rephrase the question by uid, and only if there is
    still no answer read the job this process is standing in. Calling the last of
    those a peer of the other two is what put it FIRST and broke bare `slurmwatch`
    on a login shell that had merely inherited `$SLURM_JOB_ID`.
    """
    import inspect

    # Via `_identity_failure_line`, which applies BOTH stderr tests to the same
    # line -- scanning the whole stream for each independently let an unrelated
    # diagnostic and an unrelated phrase combine to trip the guard.
    assert "_identity_failure_line" in inspect.getsource(slurm._run_slurm_cmd)
    assert "_SLURM_ERROR_LINE" in inspect.getsource(slurm._identity_failure_line)
    # In `_resolve_filtered`, which `resolve_current_jobs` delegates the three
    # attempts to so that step 3 can also be entered directly (`scan_uid=`) when
    # the CLI defers it to try `$SLURM_JOB_ID` first.
    assert "_own_uid" in inspect.getsource(slurm._resolve_filtered)
    assert "_resolve_filtered" in inspect.getsource(slurm.resolve_current_jobs)
    from slurmwatch.cli import _auto_discover_job_id

    src = inspect.getsource(_auto_discover_job_id)
    assert src.index("resolve_current_jobs") < src.index("_job_id_from_environment_source")
    assert os.environ is not None  # the third lives in cli, covered above


class TestTheEnvironmentComesBeforeTheClusterWideScan:
    """SW-90 review, 2026-08-27: the fallback order within the chain.

    Reaching step 3 means both filtered queries failed **with an identity
    error** -- which is the no-name-service compute node and nothing else. There
    `$SLURM_JOB_ID` is exact, free, and needs no controller round trip, while the
    unfiltered scan asks for the entire queue. As written, a bare `slurmwatch` on
    such a node ran the cluster-wide query every time with the authoritative
    answer sitting in its own environment.

    The queue size that makes it matter is per-cluster, which is why the figure
    now travels with its site: midway3 login on 2026-08-27 was **4,228 rows
    (9,551 with `-r`), 324 KB**; midway2 the same day was **288 / 321, 28 KB**.
    A 20x spread between two clusters of one site.
    """

    #: One job, in the `%U`-bearing format step 3 asks for. One, so
    #: `_auto_discover_job_id` returns it rather than needing a picker it has no
    #: terminal for.
    SCAN_ROWS = "9001|R|gpu|1|1:23|4:00:00|cn1|940740146|mine\n"
    #: The same job in the NAME format steps 1 and 2 ask for -- no uid column.
    FILTERED_ROWS = "9001|R|gpu|1|1:23|4:00:00|cn1|mine\n"

    @staticmethod
    def _no_name_service(
        monkeypatch: pytest.MonkeyPatch, uid: int = 940740146, rows: str | None = None
    ) -> list[list[str]]:
        """Every `-u` form refused, so only step 3 remains. Records each call."""
        calls: list[list[str]] = []
        body = (
            rows
            if rows is not None
            else (TestTheEnvironmentComesBeforeTheClusterWideScan.SCAN_ROWS)
        )

        def _run(cmd: list[str], **kwargs: object) -> _Result:
            calls.append(list(cmd))
            if "-u" in cmd:
                return _Result(0, "", "squeue: error: Invalid user id\n")
            return _Result(0, body, "")

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setattr(slurm, "_own_uid", lambda: uid)
        return calls

    def test_the_variable_is_used_and_the_scan_is_not_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        calls = self._no_name_service(monkeypatch)
        monkeypatch.setenv("SLURM_JOB_ID", "9001")
        monkeypatch.setattr("slurmwatch.cli.current_username", lambda: "youzhi")

        assert _auto_discover_job_id(SlurmwatchConfig(), interactive=False) == "9001"
        unfiltered = [c for c in calls if "squeue" in c[0] and "-u" not in c]
        assert unfiltered == [], (
            f"queried the whole queue with the answer in $SLURM_JOB_ID: {unfiltered}"
        )

    def test_with_nothing_in_the_environment_the_scan_still_happens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: this is a reordering, not a removal.

        Step 3 is the only thing left for a Slurm that takes neither a name nor a
        numeric uid, and dropping it would strand that cluster.
        """
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        calls = self._no_name_service(monkeypatch)
        for var in ("SLURM_JOB_ID", "SLURM_JOBID"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("slurmwatch.cli.current_username", lambda: "youzhi")

        got = _auto_discover_job_id(SlurmwatchConfig(), interactive=False)
        unfiltered = [c for c in calls if "squeue" in c[0] and "-u" not in c]
        assert len(unfiltered) == 1, calls
        assert got == "9001", got

    def test_the_two_failed_attempts_are_not_repeated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resumed at step 3 with the uid the deferral carried.

        Re-entering `resolve_current_jobs` from the top would re-run the two `-u`
        queries that just failed, to arrive at the same place -- two round trips
        to learn nothing that has changed.
        """
        from slurmwatch.cli import _auto_discover_job_id
        from slurmwatch.config import SlurmwatchConfig

        calls = self._no_name_service(monkeypatch)
        for var in ("SLURM_JOB_ID", "SLURM_JOBID"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("slurmwatch.cli.current_username", lambda: "youzhi")
        _auto_discover_job_id(SlurmwatchConfig(), interactive=False)

        filtered = [c for c in calls if "-u" in c]
        assert len(filtered) == 2, f"expected one `-u <name>` and one `-u <uid>`, got {filtered}"

    def test_a_login_node_never_reaches_the_fallback_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tmux regression this ordering must not re-open.

        On a login node `squeue -u <name>` succeeds, so discovery returns before
        any fallback is consulted -- and a shell that merely INHERITED
        `$SLURM_JOB_ID` (a tmux inside a reservation job, this account's
        documented workflow) keeps the picker and its real jobs.
        """
        seen = []

        def _run(cmd: list[str], **kwargs: object) -> _Result:
            seen.append(list(cmd))
            return _Result(0, self.FILTERED_ROWS, "")

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setenv("SLURM_JOB_ID", "8888888")
        jobs = resolve_current_jobs("youzhi", allow_unfiltered_scan=False)
        assert [j["job_id"] for j in jobs] == ["9001"]
        assert all("-u" in c for c in seen if "squeue" in c[0]), seen
        assert "8888888" not in [j["job_id"] for j in jobs]

    def test_a_timeout_is_not_turned_into_a_deferral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The deferral is keyed on an identity error, like the retry above it.

        An unreachable controller is not a node-identity problem, and answering it
        with `$SLURM_JOB_ID` would monitor whatever job this shell happens to sit
        in while the real failure went unreported.
        """

        def _run(cmd: list[str], **kwargs: object) -> _Result:
            return _Result(1, "", "squeue: error: Unable to contact slurm controller\n")

        monkeypatch.setattr(subprocess, "run", _run)
        monkeypatch.setattr(slurm, "_own_uid", lambda: 940740146)
        with pytest.raises(SlurmCommandError):
            resolve_current_jobs("youzhi", allow_unfiltered_scan=False)

    def test_the_deferral_carries_the_uid_it_would_have_scanned_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._no_name_service(monkeypatch, uid=12345)
        with pytest.raises(slurm.UnfilteredScanDeferredError) as raised:
            resolve_current_jobs("youzhi", allow_unfiltered_scan=False)
        assert raised.value.uid == 12345


class TestNoReasonExplanationIsUnreachable:
    """Polish pass, 2026-08-27: `_REASON_EXPLANATIONS["None"]` could never be read.

    `_explain_reason` answers `""`, `None`, `(null)` and `N/A` with a literal
    before it consults the table, and the table carried a `"None"` key holding
    that same sentence. Two copies, one of them unreachable — so the dict copy
    could have been reworded to say anything at all and no reader would ever have
    seen it. The sentence is one constant now and the dead key is gone.
    """

    def test_every_key_in_the_table_can_actually_be_reached(self) -> None:
        """A mechanism check, so a future early return cannot orphan a key again."""
        from slurmwatch.pending import (
            _NOT_A_REASON,
            _REASON_EXPLANATIONS,
            _explain_reason,
        )

        shadowed = sorted(k for k in _REASON_EXPLANATIONS if k in _NOT_A_REASON or not k.strip())
        assert not shadowed, (
            f"{shadowed} are answered by the early return before the table is "
            f"consulted, so those entries are unreachable"
        )
        # And every surviving key really does come back from the table.
        for key, value in _REASON_EXPLANATIONS.items():
            assert _explain_reason(key) == value, key

    def test_the_no_reason_answer_is_unchanged_for_every_spelling(self) -> None:
        # The control: consolidating must not change what any of the four inputs
        # produces. These are the spellings Slurm actually emits.
        from slurmwatch.pending import _NO_BLOCKING_REASON, _explain_reason

        for probe in ("", "   ", "None", "(null)", "N/A"):
            assert _explain_reason(probe) == _NO_BLOCKING_REASON, probe
        assert "no blocking reason" in _NO_BLOCKING_REASON

    def test_a_real_reason_still_gets_its_own_explanation(self) -> None:
        from slurmwatch.pending import _explain_reason

        assert "higher-priority" in _explain_reason("Priority")
        assert _explain_reason("Resources") != _explain_reason("Priority")


class TestEveryThirdPartyImportIsDeclared:
    """`tui.py` imported `rich` while `pyproject.toml` declared only textual.

    It installed and ran, because textual depends on rich and so pulled it in.
    That is the dependency working by luck of somebody else's requirements:
    textual's rich floor has moved before (8.2.8 asks for `rich>=14.2`), and a
    version that dropped or vendored rich would break `import slurmwatch.tui` on a
    fresh install with nothing in this package's metadata to explain why.

    A transitive edge that happens to hold is the hardest kind of packaging bug to
    catch, because every local environment already has the module.
    """

    #: Import name -> distribution name, where they differ.
    ALIASES = {"pil": "pillow", "yaml": "pyyaml", "attr": "attrs"}

    @staticmethod
    def _declared() -> set[str]:
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        text = (root / "pyproject.toml").read_text()
        names = set()
        for block in re.findall(r"(?:^dependencies\s*=\s*\[|=\s*\[)(.*?)^\]", text, re.S | re.M):
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                spec = line.strip('",')
                name = re.split(r"[><=!~\[;\s]", spec, maxsplit=1)[0].strip().lower()
                if name:
                    names.add(name)
        return names

    @staticmethod
    def _imported() -> dict[str, set[str]]:
        import ast
        import pathlib
        import sys

        import slurmwatch

        src = pathlib.Path(slurmwatch.__file__).parent
        local = {p.stem for p in src.rglob("*.py")} | {"slurmwatch"}
        stdlib = set(sys.stdlib_module_names)
        found: dict[str, set[str]] = {}
        for path in sorted(src.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                roots = []
                if isinstance(node, ast.Import):
                    roots = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    roots = [node.module.split(".")[0]]
                for r in roots:
                    if r not in stdlib and r not in local:
                        found.setdefault(r.lower(), set()).add(path.name)
        return found

    def test_nothing_is_imported_without_being_declared(self) -> None:
        declared = self._declared()
        missing = {
            name: sorted(where)
            for name, where in self._imported().items()
            if self.ALIASES.get(name, name) not in declared
        }
        assert not missing, (
            f"{missing} imported but not in pyproject's dependencies — it installs "
            f"today only because another dependency happens to require it"
        )

    def test_the_detector_sees_the_imports_it_should(self) -> None:
        # The control: every assertion above is a negative, so a detector that
        # found nothing would pass it.
        found = self._imported()
        assert {"rich", "textual", "pynvml"} <= set(found), sorted(found)
        assert "tui.py" in found["rich"]

    def test_the_declared_list_is_read_correctly(self) -> None:
        declared = self._declared()
        assert {"textual", "rich", "pynvml"} <= declared, sorted(declared)


class TestASuppliedEmptyJobIdSaysSo:
    """Polish pass, 2026-08-28: an empty id fell through to discovery, silently.

    The fall-through itself is right and is NOT changed here -- the comment at that
    line records what the alternative cost (`scontrol show job -d ""` means "every
    job", so slurmwatch monitored whatever the resolver landed on and emitted a
    blank `job_id` primary key on every `--log` row). Reopening that would need an
    argument against the recorded reason, and there isn't one.

    What was missing is that the two cases are not the same. An ABSENT id means
    "find my job". A SUPPLIED empty one means the caller believes they named one --
    `sw "$JOBID" --once` with `JOBID` unset -- and they then get telemetry for
    whichever job discovery lands on, at rc=0, with nothing tying it to the id they
    meant to pass.
    """

    @staticmethod
    def _stderr(
        argv: list[str], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> str:
        """Run `main` with discovery stubbed, returning what it logged.

        `main` RETURNS an exit code rather than raising `SystemExit`, so the run is
        simply allowed to finish; what is under test is the message, not the code.
        """
        import contextlib

        import slurmwatch.cli as cli

        monkeypatch.setattr(cli, "_auto_discover_job_id", lambda *a, **k: None)
        # `SystemExit` is suppressed rather than asserted: some paths return a code
        # and some exit, and neither is what this test is about.
        with caplog.at_level("WARNING", logger="slurmwatch"), contextlib.suppress(SystemExit):
            cli.main(argv)
        return " | ".join(r.getMessage() for r in caplog.records)

    def test_a_supplied_empty_id_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        said = self._stderr(["--once", ""], monkeypatch, caplog)
        assert "empty job id" in said, said
        assert "discovering your job instead" in said
        assert "check the variable you passed" in said

    def test_an_absent_id_says_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The control, and the reason this is keyed on `args.job_id` not `job_id`.

        Omitting the argument is the ordinary way to ask for auto-discovery -- a
        note there would fire on the commonest invocation there is.
        """
        said = self._stderr(["--once"], monkeypatch, caplog)
        assert "empty job id" not in said, said

    def test_a_whitespace_only_id_counts_as_empty(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert "empty job id" in self._stderr(["--once", "   "], monkeypatch, caplog)


class TestTheReadmeShowsCommandsThatWork:
    """slurmwatch was the last of the five with no test touching its README.

    Its README is deliberately thin -- four examples and then
    `slurmwatch --help  # everything else` -- so this guard is small by design. It
    is not nothing: those four lines are what pins the *positional* job id and the
    `sw` alias, and a rename of either would leave the front page telling people to
    type something that no longer parses.

    A parse check, not a run: `slurmwatch 12345` names a job that exists on nobody
    else's cluster, and what a README can promise is the shape of the command.
    """

    @staticmethod
    def _invocations() -> list[tuple[str, list[str]]]:
        import pathlib
        import re
        import shlex

        readme = pathlib.Path(__file__).resolve().parent.parent / "README.md"
        found = []
        for raw in readme.read_text().splitlines():
            line = raw.strip().lstrip("$ ").strip()
            if not re.match(r"^(slurmwatch|sw)(\s|$)", line):
                continue
            try:
                parts = shlex.split(line, comments=True)
            except ValueError:
                continue
            found.append((line, parts[1:]))
        return found

    def test_the_readme_shows_some(self) -> None:
        # A silent extraction failure would make the test below vacuously pass.
        assert len(self._invocations()) >= 3, self._invocations()

    def test_every_shown_command_parses(self) -> None:
        import contextlib
        import io

        from slurmwatch.cli import _build_parser

        broken = []
        for shown, argv in self._invocations():
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    _build_parser().parse_args(argv)
            except SystemExit as exc:
                # `--help` exits 0 by design; anything else is a rejection.
                if exc.code not in (0, None):
                    broken.append((shown, err.getvalue().strip().splitlines()[-1:]))
        assert not broken, broken

    def test_the_alias_the_readme_uses_is_the_installed_one(self) -> None:
        """`sw 12345` is shown; the console script has to provide `sw`.

        A README teaching an alias the package does not install is the same defect
        as one teaching a removed flag.
        """
        import pathlib
        import re

        pyproject = (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        scripts = re.search(r"\[project\.scripts\](.*?)(?=\n\[|\Z)", pyproject, re.S)
        assert scripts, "pyproject declares no console scripts"
        assert re.search(r"^\s*sw\s*=", scripts.group(1), re.M), scripts.group(1)


class TestTheDashboardReallyRestoresTheTerminal:
    """The end-to-end half of the SW-26 signal handlers, in a real pty.

    The handlers themselves are asserted elsewhere at the unit level. What nothing
    checked is the OUTCOME: that after a real signal to a real dashboard the
    terminal is out of the alternate screen and the exit code says "signalled".

    Both are separately breakable. Restoring the screen while exiting 0 makes
    `kill -TERM` indistinguishable from a clean `q` to a supervisor — which is
    half of what SW-26 was about — and exiting 143 while leaving the alternate
    screen open hands the user a shell showing a dead dashboard.

    Two sibling packages grew this defect and were fixed this week; each now has a
    pty test like this one. slurmwatch was already correct, which is exactly why
    its correctness had nothing guarding it.
    """

    EXPECTED = {"TERM": 143, "HUP": 129, "INT": 130}

    @staticmethod
    def _drive(signame: str, settle: float = 25.0) -> _DriveResult:
        import contextlib
        import os
        import pathlib
        import pty
        import select
        import signal
        import sys
        import time

        root = pathlib.Path(__file__).resolve().parent.parent
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - the child execs immediately
            os.environ.update(
                {
                    "PYTHONPATH": str(root / "src"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TERM": "xterm-256color",
                    "COLUMNS": "120",
                    "LINES": "40",
                }
            )
            os.chdir(str(root))
            os.execv(sys.executable, [sys.executable, "-m", "slurmwatch"])

        seen = bytearray()

        def pump(seconds: float) -> None:
            end = time.time() + seconds
            while time.time() < end:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    return
                if not chunk:
                    return
                seen.extend(chunk)

        # Poll for the alternate screen rather than sleeping a fixed interval:
        # how long the first Slurm query takes is not this test's business.
        deadline = time.time() + settle
        while time.time() < deadline and b"\x1b[?1049h" not in seen:
            pump(0.3)
        # ...then let `on_mount` run. Textual writes the alt-screen sequence when
        # the app starts, and the signal handlers are installed in `on_mount` a
        # moment later, so signalling the instant the sequence appears lands in a
        # window where the default disposition still applies -- measured: killed by
        # the signal with the screen left open. That window is real and only
        # milliseconds wide; this test is about the steady state a running
        # dashboard is in, which is where a cancelled job or a closed pane
        # actually arrives.
        pump(3.0)
        entered = seen.count(b"\x1b[?1049h")
        os.kill(pid, getattr(signal, f"SIG{signame}"))

        code = signalled = None
        end = time.time() + 25
        while time.time() < end:
            pump(0.4)
            try:
                done, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            if done:
                signalled = os.WIFSIGNALED(status)
                code = None if signalled else os.waitstatus_to_exitcode(status)
                break
        else:  # pragma: no cover - only on a hang
            os.kill(pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
            pytest.fail(f"the dashboard never exited after SIG{signame}")

        text = seen.decode("utf-8", "replace")
        return {
            "entered": entered,
            "left": text.count("\x1b[?1049l"),
            "code": code,
            "signalled": signalled,
            "traceback": "Traceback" in text,
        }

    @staticmethod
    def _drive_startup_window(signame: str) -> _StartupResult:
        """Signal the INSTANT the alternate screen appears, then drain to EOF.

        Draining to EOF rather than for a fixed time is the whole point: the pty
        slave closes only when the child is gone, so this cannot miss bytes the
        child already wrote. A time-bounded read lost them and made a restored
        terminal look like a leaking one — about one run in four — which is exactly
        the observation that invites defensive code for a bug that is not there.
        """
        import contextlib
        import os
        import pathlib
        import pty
        import select
        import signal
        import sys
        import time

        root = pathlib.Path(__file__).resolve().parent.parent
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - the child execs immediately
            os.environ.update(
                {
                    "PYTHONPATH": str(root / "src"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TERM": "xterm-256color",
                    "COLUMNS": "120",
                    "LINES": "40",
                }
            )
            os.chdir(str(root))
            os.execv(sys.executable, [sys.executable, "-m", "slurmwatch"])

        seen = bytearray()

        def read_some(timeout: float) -> bool:
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                return True
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return False
            if not chunk:
                return False
            seen.extend(chunk)
            return True

        deadline = time.time() + 40
        while time.time() < deadline and b"\x1b[?1049h" not in seen:
            if not read_some(0.05):
                break
        opened = seen.count(b"\x1b[?1049h")
        os.kill(pid, getattr(signal, f"SIG{signame}"))
        while read_some(3.0):
            pass
        signalled = code = None
        with contextlib.suppress(ChildProcessError):
            _done, status = os.waitpid(pid, 0)
            signalled = os.WIFSIGNALED(status)
            code = None if signalled else os.waitstatus_to_exitcode(status)
        text = seen.decode("utf-8", "replace")
        return {
            "opened": opened,
            "closed": text.count("\x1b[?1049l"),
            "signalled": signalled,
            "code": code,
        }

    @pytest.mark.parametrize("signame", ["TERM", "HUP"])
    def test_a_signal_in_the_startup_window_still_restores(self, signame: str) -> None:
        """The window between the screen opening and `on_mount` installing handlers.

        Textual writes the alternate-screen sequence as its very FIRST output — at
        the moment it appears only 8 bytes have been emitted, i.e. just that
        sequence — and `on_mount` runs a moment later. A signal landing in between
        used to kill the process outright:

            without `_TerminalGuard`   6/6  killed by signal, screen left open
            with it                    6/6  exit 143, screen restored

        `_TerminalGuard` already existed for this on the two hop paths, and its own
        docstring names the same window. It had never been applied to the four local
        `app.run()` sites.
        """
        got = self._drive_startup_window(signame)
        if not got["opened"]:
            pytest.skip("the dashboard never reached the alternate screen here")
        assert got["signalled"] is False, (
            f"the default disposition still applied — the guard is not covering "
            f"the startup window: {got}"
        )
        assert got["closed"] >= got["opened"], f"screen left open: {got}"
        assert got["code"] == 128 + int(getattr(__import__("signal"), f"SIG{signame}")), got

    @pytest.mark.parametrize("signame", ["TERM", "HUP", "INT"])
    def test_the_screen_is_left_and_the_code_says_signalled(self, signame: str) -> None:
        got = self._drive(signame)
        if not got["entered"]:
            pytest.skip("the dashboard never reached the alternate screen here")
        assert got["left"] >= got["entered"], f"alternate screen left open: {got}"
        assert not got["traceback"], got
        assert got["signalled"] is False, f"killed instead of handled: {got}"
        assert got["code"] == self.EXPECTED[signame], got


class TestTheDocumentedExitCodesAreTheRealOnes:
    """`--help` promises an exit-status contract; scripts are written against it.

    The `128+N` line used to read "stopped by signal N **(--log**; ...)", which
    scopes signalled exits to the headless logger. Measured: the dashboard with no
    `--log` exits **143** on SIGTERM as well — its own handlers (SW-26) and, since
    the startup guard, the window before them. So a script author reading that help
    would conclude the dashboard returns only 0/1/2 and treat 143 as a crash.

    This ties each documented code to something that produces it, so the help
    cannot drift from the behaviour again.
    """

    @staticmethod
    def _help_text() -> str:
        import contextlib
        import io

        from slurmwatch.cli import _build_parser

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
            _build_parser().parse_args(["--help"])
        return out.getvalue()

    def test_the_exit_status_section_exists(self) -> None:
        # A silent rename would make every assertion below vacuous.
        assert "exit status" in self._help_text()

    def test_every_documented_code_is_listed(self) -> None:
        text = self._help_text()
        section = text[text.index("exit status") :]
        for code in ("0", "1", "2", "128+N"):
            assert f"  {code}" in section, f"{code} is no longer documented"

    def test_the_signal_line_is_not_scoped_to_log(self) -> None:
        """The specific correction, asserted by MECHANISM not wording.

        What must hold is that the line does not present signalled exits as a
        `--log`-only phenomenon. Checked by requiring it to say otherwise, so a
        future rewording that re-narrows it fails here.
        """
        text = self._help_text()
        line_start = text.index("128+N")
        clause = text[line_start : line_start + 220]
        assert "any mode" in clause, clause
        assert "not only --log" in clause, clause

    @pytest.mark.parametrize(
        "argv,expected",
        [
            (["--interval", "0", "1"], 2),  # bad usage
            (["--format", "bogus", "1"], 2),
        ],
    )
    def test_code_2_is_what_bad_usage_gives(self, argv: list[str], expected: int) -> None:
        import os
        import pathlib
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        done = subprocess.run(
            [sys.executable, "-m", "slurmwatch", *argv],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "200",
            },
        )
        assert done.returncode == expected, done.stderr[-200:]

    def test_code_1_is_what_a_job_with_no_telemetry_gives(self) -> None:
        """A job id that is not one: the doc's "an id that isn't one" case."""
        import os
        import pathlib
        import subprocess
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        done = subprocess.run(
            [sys.executable, "-m", "slurmwatch", "--once", "--json", "999999999"],
            capture_output=True,
            text=True,
            timeout=280,
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "200",
            },
        )
        assert done.returncode == 1, done.stderr[-200:]
        # And the doc's promise about WHY: the object is still printed with a reason.
        import json

        payload = json.loads(done.stdout)
        assert payload.get("telemetry_unavailable_reason"), payload
