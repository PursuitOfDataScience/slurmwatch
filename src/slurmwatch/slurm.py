from __future__ import annotations

import contextlib
import getpass
import logging
import os
import pwd
import re
import socket
import subprocess
import time
from pathlib import Path

from .exceptions import (
    CgroupNotFoundError,
    CgroupPermissionError,
    JobNotFoundError,
    JobNotRunningError,
    SlurmCommandError,
)
from .model import JobContext, local_node_name, short_host

SLURM_CMD_TIMEOUT = 15
_CGROUP_V2_BASE = Path("/sys/fs/cgroup")
_MOCK_ENV_VAR = "SLURMWATCH_MOCK"
_MAX_SANE_CPU_SECONDS = 3650 * 86400  # 10 years; guards against NO_VAL sentinels
# Cap total hostlist expansion so a corrupt/garbage NodeList (e.g. cn[1-1e8] or a
# cartesian blow-up a[1-10000]b[1-10000]) can't exhaust memory / hang. Far above
# any real allocation, so a legitimate nodelist is never truncated (#audit3-10).
_MAX_HOSTLIST_NODES = 65536
# Cap GPU IDX-range expansion the same way — far above any real node's GPU count,
# so a crafted `IDX:0-<huge>` can't exhaust memory, but a real list never truncates.
_MAX_GPU_IDX = 4096

logger = logging.getLogger("slurmwatch")

# A het job's scontrol records carry JobId=<leader>+<component>, e.g. 12345+0.
_HET_JOBID_RE = re.compile(r"^(\S+\+\d+)$")


def _count_het_components(scontrol_output: str) -> int:
    """How many distinct het-job components are in a ``scontrol show job`` dump.

    >1 means the job is heterogeneous (one record per component); slurmwatch
    monitors only the selected component, so the caller warns.

    Reads each RECORD's own ``JobId`` field via :func:`_parse_scontrol_field` —
    which is field-shadow protected — rather than regexing the raw text for any
    ``JobId=<leader>+<n>``-shaped substring. The naive scan matched that pattern
    anywhere at all, including inside a free-text field's value (a JobName,
    Comment, or Command containing the literal text "JobId=1+1"), which would
    spuriously call an ordinary job heterogeneous.
    """
    records = [r for r in re.split(r"\n\s*\n", scontrol_output) if "JobId=" in r]
    het_ids = set()
    for record in records:
        jid = _parse_scontrol_field(record, "JobId")
        match = _HET_JOBID_RE.match(jid) if jid else None
        if match:
            het_ids.add(match.group(1))
    return len(het_ids)


def _is_mock() -> bool:
    return os.environ.get(_MOCK_ENV_VAR) == "1"


# A squeue row begins with %i — a job id, with an optional array element and/or het
# component. The delimiters of the requested format must be there too, so a stray
# fragment can't be mistaken for the start of a record.
#
# The array element is `_<task>` for a task that has started AND `_[<range>]` for a
# PENDING array — the single row squeue prints for a whole unstarted range,
# concurrency throttle and all (`56814401_[1-28%4]`). Only `_\d+` was accepted, so
# every pending-array row failed the start-of-record test and was glued onto the row
# before it as a continuation of THAT job's name: the arrays vanished from the picker
# — which lists PD jobs on purpose, and routes a pending pick to the why/when/where
# view — and the preceding job's name came back with the swallowed row appended. It
# is the form `--help` promises ("a pending array's range 12345_[1-9%3]"), and 105 of
# the rows in this controller's queue were that shape (Slurm 20.11.8).
#
# The bracket is tolerated exactly as `ARRAY_RANGE_RE` above tolerates it, for the
# same reason: squeue TRUNCATES %i (`SLURM_BITSTR_LEN`), so the closing `]` may be
# absent and the contents may end in `...`. Live ids are cut mid-number
# (`56843185_[1-30,32-44,47-83,85-8|PD|...`). The contents stay MANDATORY and
# numeric, so a name fragment must still open with digits and a real `_[` to be
# misread as a row — and it must carry the format's delimiters as well.
#: A pending array's id AS SQUEUE PRINTS IT: one row for the whole unstarted
#: range, concurrency throttle and all. The bracket may be unclosed and the
#: contents may end in `...` because squeue truncates `%i` (`SLURM_BITSTR_LEN`);
#: the contents stay mandatory and numeric so a mistyped id is never rewritten
#: into a silent monitor of some other job. See `_SQUEUE_ROW_START` below, which
#: tolerates the same shapes for the same reason.
ARRAY_RANGE_RE = re.compile(r"^(?P<base>\d+)_\[(?P<range>[\d,\-%]+)(?:\.\.\.)?\]?$")


def array_range_base(job_id: str) -> str | None:
    """``54222358_[1-9%3]`` -> ``54222358``; ``None`` when it is not a range.

    Pure, so both surfaces can use it: `cli` prints a note to stderr on the
    command-line path, and the job picker cannot (a TUI owns the screen). It used
    to live only in `cli`, and the picker therefore handed the bracket string
    straight to `scontrol`, which answers `Invalid job id specified` -- so a row
    the picker had just DRAWN could not be opened: `sw: Job
    57902634_[31-48%18] not found`. Measured on this controller.

    That is the same closed loop `cli._job_id_without_array_range` was written to
    break, one surface later: the id came from squeue, and squeue prints only this
    form for an unstarted array. A range names no single task and an unstarted
    array has no per-task telemetry, so the useful target is the array's own job,
    whose pending reason and queue position are what the reader was asking about.
    """
    match = ARRAY_RANGE_RE.match(job_id)
    return match.group("base") if match else None


_SQUEUE_ROW_START = re.compile(r"^\d+(?:_(?:\d+|\[[\d,\-%]+(?:\.\.\.)?\]?))?(?:\+\d+)?\|")

# The two formats `resolve_current_jobs` asks for, and their field counts DERIVED
# from the format strings rather than written out again. A hand-maintained `8`
# silently went out of step with the 9-field uid form the moment that form was
# added, and the row-start heuristic below is the thing that count feeds — so the
# drift would have shown up as phantom entries in the job picker, not as an error.
# `%j` is LAST in both: the job name is the only free-form field, so a literal
# `|` inside it must land in the final `split()` field instead of shifting every
# column after it (B-P10). Appending `%U` after the name instead put the uid in
# the ninth field, split the name at its own pipe, and `my training|job` came back
# as `my training`. The uid is machine-generated and pipe-free, so it is safe
# anywhere before the name.
_SQUEUE_FIXED_FIELDS = "%i|%t|%P|%D|%M|%l|%R"
_SQUEUE_FORMAT = _SQUEUE_FIXED_FIELDS + "|%j"
_SQUEUE_UID_FORMAT = _SQUEUE_FIXED_FIELDS + "|%U|%j"
_SQUEUE_FIELD_COUNT = _SQUEUE_FORMAT.count("|") + 1
_SQUEUE_UID_FIELD_COUNT = _SQUEUE_UID_FORMAT.count("|") + 1


def _squeue_rows(output: str, field_count: int = _SQUEUE_FIELD_COUNT) -> list[str]:
    """``squeue`` output as logical rows, with a multi-line job name reassembled.

    The last field is the job NAME — the one free-text value — and a name holding a
    NEWLINE breaks its own row across physical lines. Splitting on "\n" then made
    each tail fragment a row of its own, so the job picker listed entries for jobs
    that do not exist. This is slurmpast's SP-1 in our own parser (the shared root
    cause the cross-cluster report's cross-cutting section names): a subprocess's
    output trusted to be one-record-per-line.

    So a line that does not START a record is joined onto the one before it, which
    is where its text belongs — inside the name — with the newline flattened to a
    space. Residual: a name that deliberately imitates a whole row, newline and
    all, can still add one phantom entry to the picker; the id is re-resolved when
    they pick it, so a fabricated one fails there.

    That residual is NOT bounded to the user's own jobs on the unfiltered fallback
    path in `resolve_current_jobs` — see the note there. `-u` is what used to make
    the input the user's own rows only, and the last-resort query has no `-u`.
    """
    rows: list[str] = []
    for raw in output.split("\n"):
        line = raw.strip()
        if not line:
            continue
        starts_record = (
            _SQUEUE_ROW_START.match(line) is not None and line.count("|") >= field_count - 1
        )
        if rows and not starts_record:
            rows[-1] = f"{rows[-1]} {line}"
        else:
            rows.append(line)
    return rows


#: One of Slurm's own diagnostics, as its clients write them: `squeue: error: …`,
#: `sacct: error: …`, `scontrol: error: …`.  Anchored to the start of a line and
#: to a bare tool name so a job name or a site banner containing the word "error"
#: cannot be read as one.
_SLURM_ERROR_LINE = re.compile(r"^[a-z_][a-z0-9_-]*:\s*error:\s*\S", re.IGNORECASE | re.MULTILINE)

#: What Slurm says when it cannot map a name or a uid to a user.  Distinguished
#: from every other failure because it is the one a numeric-uid query can get
#: past: a timeout or an unreachable controller is not helped by rephrasing the
#: question, and retrying it unfiltered would ask a busy controller for the whole
#: queue for nothing.
#:
#: It is also the ONLY class for which an exit-0-with-empty-stdout counts as a
#: failure — see the guard at the end of `_run_slurm_cmd`.
_IDENTITY_ERROR_MARKERS = (
    "invalid user",
    "unknown user",
    "no such user",
    "invalid user id",
)


def _is_identity_error(exc: Exception) -> bool:
    """Whether a ``SlurmCommandError`` is about WHO was asked about."""
    return _mentions_identity_error(str(exc))


def _mentions_identity_error(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _IDENTITY_ERROR_MARKERS)


def _run_slurm_cmd(cmd: list[str], timeout: int = SLURM_CMD_TIMEOUT) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # A job name/comment can carry non-UTF-8 bytes; without errors= the
            # strict decode raises UnicodeDecodeError and breaks every Slurm call.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            # Normalize the environment so parsing is deterministic regardless of
            # the user's shell: a user's SLURM_TIME_FORMAT would otherwise reformat
            # scontrol timestamps out of the ISO form _parse_scontrol_time expects
            # (→ StartTime unparsed → elapsed 0 → CPU% 0), and a non-C locale can
            # shift number/date formatting. PATH etc. are preserved.
            env={**os.environ, "SLURM_TIME_FORMAT": "standard", "LC_ALL": "C"},
        )
    except FileNotFoundError as exc:
        raise SlurmCommandError(f"Slurm binary not found: {cmd[0]}. Is Slurm installed?") from exc
    except subprocess.TimeoutExpired as exc:
        raise SlurmCommandError(f"Command {' '.join(cmd)} timed out after {timeout}s") from exc
    except OSError as exc:
        # The rest of the OSError family — e.g. fork failing with EAGAIN at
        # RLIMIT_NPROC on a busy login node (BlockingIOError), ENOMEM, or a
        # PermissionError — so it surfaces as a SlurmCommandError every caller
        # already handles, not a raw traceback (N4). FileNotFoundError is caught
        # above; TimeoutExpired is not an OSError.
        raise SlurmCommandError(f"Command {' '.join(cmd)} could not run: {exc}") from exc

    if result.returncode != 0:
        # stderr FIRST (a real controller/infra error goes there), but fall back to
        # stdout: `scontrol show job <bad-task-of-a-real-array>` exits 1 having
        # written "Job 12345_9 not found" to STDOUT and nothing to stderr, so a
        # stderr-only message loses the one sentence that says the job is gone for
        # good — and _is_missing_job_error, matching on the message, then reads a
        # permanent failure as a transient one ("try again in a moment"). SW-7.
        detail = result.stderr.strip() or result.stdout.strip()
        raise SlurmCommandError(
            f"Command {' '.join(cmd)} failed (rc={result.returncode}): {detail}"
        )
    # A zero exit is not proof the command worked.
    #
    # Slurm's own clients do not agree about this. On a node whose name service
    # cannot map the uid, all three fail the same query and only one says so in
    # its exit status:
    #
    #     squeue -u youzhi   "squeue: error: Invalid user: youzhi"       exit 0
    #     squeue --me        "squeue: error: Invalid user: 940740146"    exit 0
    #     sacct  -u youzhi   "sacct: error: Invalid user id: youzhi"     exit 1
    #
    # So `returncode != 0` returned "" for a working query, `resolve_current_jobs`
    # parsed zero rows, and slurmwatch told a user with **six running jobs** to
    # "launch a job first" — while executing inside one of them. That is the
    # "could not ask" ≠ "nothing there" distinction SW-89 rests on, defeated by a
    # convention Slurm does not hold to.
    #
    # Narrow on purpose, so a legitimately-empty answer cannot become a failure:
    # THREE things must hold. Empty stdout, a stderr line that is one of Slurm's
    # own `<tool>: error:` lines, AND that line being about the SUBJECT of the
    # query not being resolvable — the identity class above, which is the one
    # SW-90 needs and the one a rephrased query can get past.
    #
    # The first version of this guard fired on any `<tool>: error:` line, and that
    # premise is simply false for `sstat`. Measured on this login node:
    #
    #     $ sstat --allsteps --noheader -P -j 54117243 --format=MaxRSS
    #     sstat: error: couldn't get steps for job 54117243     <- stderr
    #     rc=0, stdout empty
    #
    # A job with no live steps is the ordinary state of a job, not a broken query,
    # and "nothing to report" is the correct answer to it. Turning that into an
    # exception replaced one wrong answer with a louder one in exactly the surface
    # SW-89 is about. A command that answered rows keeps its answer whatever it
    # wrote to stderr, and a command that wrote nothing at all is still a legal
    # empty result.
    if not result.stdout.strip() and _identity_failure_line(result.stderr or ""):
        raise SlurmCommandError(
            f"Command {' '.join(cmd)} failed (rc=0, but wrote only an error): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _identity_failure_line(stderr: str) -> str:
    """A ``<tool>: error:`` line that is ABOUT who was asked, or ``""``.

    Both halves are tested on the SAME line, deliberately. Scanning the whole of
    stderr for each independently let an unrelated diagnostic combine with an
    unrelated phrase to trip the guard -- `sstat: error: couldn't get steps for
    job N` on one line plus the words "invalid user" on another are two true
    statements about different lines making a false one about the command, and
    the cost is an exception where the honest answer is an empty result.
    """
    for line in stderr.splitlines():
        if _SLURM_ERROR_LINE.match(line) and _mentions_identity_error(line):
            return line.strip()
    return ""


def _is_missing_job_error(exc: Exception) -> bool:
    """Whether a ``SlurmCommandError`` means the job id is genuinely invalid/unknown.

    Only an explicit "invalid job id" is taken as "the job doesn't exist". Every
    other failure (timeout, "Socket timed out on send/recv", "Unable to contact
    slurm controller", "connect failure", "Zero Bytes were transmitted") is a
    transient/infra problem — NOT proof the job is gone — so callers must treat it
    as retryable/unknown rather than reporting a live job as finished, missing, or
    started.
    """
    msg = str(exc).lower()
    if "invalid job id" in msg or "invalid job" in msg:
        return True
    # `scontrol show job 12345_9` (a task a real array never had) says "Job
    # 12345_9 not found" instead — a different channel AND a different wording for
    # the same permanent condition, so matching only "invalid job id" reports a
    # job that can never exist as a busy controller. Anchored on "job … not
    # found" within one line so "Slurm binary not found: squeue" (no job token) and
    # a multi-line controller error can't match. SW-7.
    return re.search(r"\bjob\b[^\n]*\bnot found\b", msg) is not None


def _is_missing_binary_error(exc: Exception) -> bool:
    """Whether a ``SlurmCommandError`` means Slurm's client tools are not here at all.

    ``FileNotFoundError`` on the exec becomes "Slurm binary not found: scontrol",
    which is permanent and is not a controller problem — there is no controller in the
    picture. It reached the generic "couldn't reach the Slurm controller … it may be
    busy — try again in a moment", so on a machine with no Slurm (a PBS/Flux/
    Kubernetes site, a laptop, a login node whose module isn't loaded) slurmwatch
    blamed a controller that does not exist and told the reader to keep retrying.
    Fourth trigger of the misdiagnosis family SW-7, RD-2 and SW-19 belong to; the
    string is already recognised one function above, but only well enough to keep it
    OUT of the "job not found" match.
    """
    return "slurm binary not found" in str(exc).lower()


def _is_malformed_query_error(exc: Exception) -> bool:
    """Whether a ``SlurmCommandError`` means WE asked for something Slurm doesn't have.

    Field names move between Slurm releases (23.02 renamed ``Reserved`` to
    ``Planned``), and one unknown field makes ``sacct``/``sstat`` reject the ENTIRE
    query — not just that column — with "Invalid field requested". That is permanent
    and it is our bug, so it must not be reported as a busy controller the user
    should retry. Third trigger for that same misdiagnosis after SW-7 and rapidu's
    RD-2, which is why the DIAGNOSIS is what gets fixed here, not just the trigger.
    SW-19.
    """
    msg = str(exc).lower()
    return "invalid field" in msg or "invalid entity" in msg


# Accounting fields, and the tools that must accept them. Requested blind is what
# SW-19 is about: a rename on some site's Slurm would return nothing at all, so a
# rejected query is retried against what that Slurm actually supports.
_HELPFORMAT_CACHE: dict[str, frozenset[str]] = {}


def _supported_fields(tool: str) -> frozenset[str]:
    """Field names ``tool --helpformat`` admits to, lowercased; empty if unknown.

    Cached per process: this is a fallback path, so it costs one subprocess only on
    a cluster that actually rejected a field.
    """
    if tool not in _HELPFORMAT_CACHE:
        try:
            out = _run_slurm_cmd([tool, "--helpformat"])
        except SlurmCommandError:
            out = ""
        _HELPFORMAT_CACHE[tool] = frozenset(tok.strip().lower() for tok in out.split() if tok)
    return _HELPFORMAT_CACHE[tool]


_ACCT_GATHER_CACHE: list[bool] | None = None


def acct_gather_disabled() -> bool:
    """True when this Slurm gathers no per-job accounting, so ``sstat`` never samples.

    ``JobAcctGatherType=jobacct_gather/none`` is Slurm's DEFAULT when a site does not
    set it, and with it sstat has nothing to report for a running job — not "yet",
    ever. Off-node slurmwatch then shows zeros and says "not yet sampled by Slurm …
    try again shortly", which is a false promise: the reader retries forever for a
    figure that cannot exist on their cluster. Read the config once and say the true
    thing instead.

    Answers False when the config cannot be read: claiming a site's accounting is off
    on the strength of a failed subprocess would be a worse error than staying quiet.
    """
    global _ACCT_GATHER_CACHE
    if _ACCT_GATHER_CACHE is None:
        try:
            out = _run_slurm_cmd(["scontrol", "show", "config"])
        except SlurmCommandError:
            out = ""
        disabled = False
        for line in out.splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().lower() == "jobacctgathertype":
                # Match the TYPE, not the plugin path: sites write
                # "jobacct_gather/none" but the bare "none" is also accepted.
                disabled = value.strip().lower().rsplit("/", 1)[-1] == "none"
                break
        _ACCT_GATHER_CACHE = [disabled]
    return _ACCT_GATHER_CACHE[0]


def _drop_unsupported_fields(tool: str, fields: list[str]) -> list[str]:
    """``fields`` minus the ones this Slurm has never heard of.

    Losing one column beats losing the whole query: a site whose Slurm dropped
    ``AveCPU`` still gets MaxRSS out of sstat, where the blind request returned
    nothing at all. Returns the list unchanged when ``--helpformat`` says nothing —
    guessing would be worse than trying.
    """
    supported = _supported_fields(tool)
    if not supported:
        return fields
    kept = [f for f in fields if f.lower() in supported]
    dropped = [f for f in fields if f.lower() not in supported]
    if dropped:
        logger.warning(
            "%s on this cluster does not support the field(s) %s; continuing without them.",
            tool,
            ", ".join(dropped),
        )
    return kept


def _parse_mem_to_bytes(mem_str: str, default_unit: str = "M") -> int | None:
    """A Slurm memory spelling in bytes, or ``None`` when it cannot be read.

    ``None`` rather than ``0`` because downstream a limit of ``0`` means "no limit
    is enforced" (see the tui "no limit set" branch), so answering ``0`` for a
    spelling this doesn't understand silently promotes a capped job to unlimited —
    a wrong answer that looks like a fact. Callers with only a number to show still
    fall back to their own ``0``, but they do it knowingly. SW-12.

    Spellings handled, across the Slurm versions a site might be running:
    ``4G``, the two-letter ``64GB``, fractional ``1.5T``, and the per-node /
    per-cpu qualified ``4Gn`` / ``500Mc`` that Slurm <= 20.11 wrote into
    ``ReqMem``. The ``n``/``c`` says WHOSE limit it is, which the caller already
    knows from the field it read, so it is stripped rather than interpreted.
    A unit-less integer takes ``default_unit``, and the right default DIFFERS by
    field: a bare ``--mem``/``ReqMem``/``MinMemory*`` is MEGABYTES (Slurm's own
    convention — reading it as bytes is off by 1,048,576x, silently), while a bare
    ``sstat``/``sacct`` ``MaxRSS`` counts in KILOBYTES. One blanket default would
    inflate one of them by 1024x, so the caller names the field's unit.
    """
    mem_str = mem_str.strip().upper()
    # "4Gn" / "500Mc" — strip the per-node/per-cpu qualifier before the unit.
    if mem_str.endswith(("N", "C")):
        mem_str = mem_str[:-1]
    # Tolerate a trailing "B" (the two-letter form "64GB"/"512MB"); Slurm emits the
    # single-letter form, but failing "64GB" would silently drop a limit.
    if mem_str.endswith("B"):
        mem_str = mem_str[:-1]
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    unit_scale = multipliers.get(default_unit.upper(), 1024**2)
    if mem_str.isdigit():
        return int(mem_str) * unit_scale
    for suffix, mult in multipliers.items():
        if mem_str.endswith(suffix):
            try:
                scaled = float(mem_str[:-1]) * mult
            except ValueError:
                return None
            # A negative count is not a limit anyone can act on: unreadable.
            return int(scaled) if scaled >= 0 else None
    try:
        plain = float(mem_str)
    except ValueError:
        return None
    return int(plain * unit_scale) if plain >= 0 else None


def _expand_range_group(content: str) -> list[str]:
    """Expand the inside of one bracket group into padded strings.

    Handles the forms Slurm emits: single values ('007'), ranges ('001-003'),
    and stepped ranges ('001-007:2' -> 001,003,005,007), comma-joined
    ('001-003,007'). An unparseable or reversed range is kept verbatim rather
    than silently dropped, so a malformed hostlist never quietly collapses the
    node count (#39)."""
    out: list[str] = []
    for rng in content.split(","):
        rng = rng.strip()
        if not rng:
            continue
        if "-" not in rng:
            out.append(rng)
            continue
        start_str, rest = rng.split("-", 1)
        # A stepped range is 'start-end:step' (Slurm's hostlist syntax).
        step_str = "1"
        if ":" in rest:
            end_str, step_str = rest.split(":", 1)
        else:
            end_str = rest
        try:
            pad = len(start_str)
            start_n, end_n, step_n = int(start_str), int(end_str), int(step_str)
        except ValueError:
            out.append(rng)  # not numeric (e.g. 'a-c') -> keep verbatim
            continue
        if step_n < 1 or start_n > end_n:
            out.append(rng)  # bad step or reversed -> keep verbatim, don't drop
            continue
        for i in range(start_n, end_n + 1, step_n):
            if len(out) >= _MAX_HOSTLIST_NODES:  # bound a pathological range (#audit3-10)
                return out
            out.append(str(i).zfill(pad))
    return out


def _expand_hostlist_part(part: str) -> list[str]:
    """Expand one hostlist element, including multi-dimensional bracket groups.

    Slurm hostlists can carry more than one bracket group per element
    (``rack[1-2]node[3-4]`` → rack1node3, rack1node4, rack2node3, rack2node4);
    the earlier single-trailing-bracket parser silently dropped the extra
    dimensions and undercounted nodes (B-P12). Splits the element into literal
    and bracket tokens in order, then takes the cartesian product.
    """
    tokens: list[tuple[str, str]] = []  # (kind, value); kind is "lit" or "brk"
    i = 0
    n = len(part)
    while i < n:
        if part[i] == "[":
            j = part.find("]", i)
            if j == -1:
                tokens.append(("lit", part[i:]))
                break
            tokens.append(("brk", part[i + 1 : j]))
            i = j + 1
        else:
            j = part.find("[", i)
            if j == -1:
                tokens.append(("lit", part[i:]))
                break
            tokens.append(("lit", part[i:j]))
            i = j

    result = [""]
    for kind, value in tokens:
        options = [value] if kind == "lit" else _expand_range_group(value)
        # Build the product incrementally and stop AT the cap, so a cartesian
        # blow-up (a[1-10000]b[1-10000] = 100M) can't materialise the full list
        # before a post-hoc size check (#audit3-10).
        combined: list[str] = []
        for prefix in result:
            for opt in options:
                if len(combined) >= _MAX_HOSTLIST_NODES:
                    return combined
                combined.append(prefix + opt)
        result = combined
    return result


def _parse_nodelist(nodelist: str) -> list[str]:
    if not nodelist or nodelist == "(null)":
        return []

    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in nodelist:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))

    nodes: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            nodes.extend(_expand_hostlist_part(part))
        if len(nodes) >= _MAX_HOSTLIST_NODES:  # bound total expansion (#audit3-10)
            return nodes[:_MAX_HOSTLIST_NODES]
    return nodes


# squeue's short state codes for the states the picker deliberately leaves out, spelled
# so a message can name what the user is looking at in `squeue`. Not exhaustive by
# design — an unknown code falls back to the raw letters, which is still better than
# claiming there is nothing there.
_UNMONITORABLE_STATE_NAMES = {
    "CG": "COMPLETING",
    "CF": "CONFIGURING",
    "S": "SUSPENDED",
    "ST": "STOPPED",
    "PR": "PREEMPTED",
    "RQ": "REQUEUED",
    "RS": "RESIZING",
    "RV": "REVOKED",
    "SI": "SIGNALING",
    "SO": "STAGE_OUT",
    "CD": "COMPLETED",
    "CA": "CANCELLED",
    "F": "FAILED",
    "TO": "TIMEOUT",
    "NF": "NODE_FAIL",
    "OOM": "OUT_OF_MEMORY",
}


def resolve_unmonitorable_jobs(username: str | None = None) -> list[tuple[str, str]]:
    """``(job_id, state)`` for the user's jobs that ``resolve_current_jobs`` filters out.

    Called only when the monitorable list came back EMPTY, to tell "you have nothing
    queued" apart from "your job is there, in a state with no live telemetry". Those two
    got the same message — *"No running or pending Slurm jobs found … Launch a job
    first"* — while `squeue` sat there showing a COMPLETING job, so the tool was
    contradicting the command the user had just run.

    A second ``squeue`` rather than widening ``resolve_current_jobs``: that function's
    result feeds the picker, and a state the picker cannot monitor must not be able to
    reach it. The cost is one extra call on a path that has already decided to exit, and
    the only race — the job finishing in between — degrades to the original message,
    which is then correct.
    """
    if _is_mock():
        return []
    if username is None:
        username = current_username()
    try:
        output = _run_slurm_cmd(["squeue", "-u", username, "-h", "-o", "%i|%t"])
    except Exception:
        return []  # best-effort context for a message; never turn it into a failure
    out: list[tuple[str, str]] = []
    # Plain newline split, NOT _squeue_rows: that helper reassembles a record whose
    # trailing job NAME contains a newline, and it decides where a record starts by
    # counting pipes against the WIDE format's field count. This query asks for `%i|%t`
    # — one pipe — so every row after the first failed that test and was glued onto the
    # previous one, collapsing five suspended jobs into one line whose state read
    # "CG 54999001 CG ...". Splitting directly is safe here precisely BECAUSE this
    # format has no free-text field: with no name, a row cannot span lines.
    for line in output.split("\n"):
        # Split FULLY and take the state as its own field: with maxsplit=1 anything
        # after the first pipe lands in the state and gets printed into the message.
        # We ask for exactly `%i|%t`, so a longer line means the response was not what
        # was requested — and then the state is the second field, not the remainder.
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[1] and parts[1] not in ("R", "PD"):
            out.append((parts[0], _UNMONITORABLE_STATE_NAMES.get(parts[1], parts[1])))
    return out


class UnfilteredScanDeferredError(Exception):
    """Both filtered `squeue` attempts failed on identity; the scan is what's left.

    Raised only when the caller passed ``allow_unfiltered_scan=False``, so it can
    put something cheaper in front of the cluster-wide query -- and it carries the
    uid, so resuming costs no repeat of the two attempts that just failed.

    Reaching this point is diagnostic in itself: it means the node has no name
    service AND this Slurm will not take a numeric uid, which is exactly the
    situation in which ``$SLURM_JOB_ID`` is exact, free, and needs no controller.
    """

    def __init__(self, uid: int, cause: Exception) -> None:
        super().__init__(str(cause))
        self.uid = uid
        self.cause = cause


def _resolve_filtered(username: str, allow_unfiltered_scan: bool) -> tuple[str, int | None]:
    """Steps 1-3 of the chain. Returns the output, and the uid when unfiltered.

    Its own function so `resolve_current_jobs` can be entered directly at step 3
    (see ``scan_uid``) without the three attempts and the one parser having to
    interleave.
    """
    wanted_uid: int | None = None
    try:
        output = _run_slurm_cmd(["squeue", "-u", username, "-h", "-o", _SQUEUE_FORMAT])
    except SlurmCommandError as exc:
        own = _own_uid()
        if own is None or not _is_identity_error(exc):
            # A timeout or an unreachable controller is not helped by rephrasing
            # the question, and without a uid there is nothing to rephrase it to.
            raise
        logger.info(
            "squeue could not resolve %r (%s); retrying by uid %d",
            username,
            exc,
            own,
        )
        try:
            output = _run_slurm_cmd(["squeue", "-u", str(own), "-h", "-o", _SQUEUE_FORMAT])
        except SlurmCommandError as by_uid:
            if not _is_identity_error(by_uid):
                raise
            # LAST RESORT. Step 3 exists only for a Slurm that will not take a
            # numeric uid at all, and it is last because of what it costs: without
            # `-u` the controller returns EVERY job in the queue, and the `-u`
            # filter was also what bounded `_squeue_rows`' phantom-row residual to the user's own
            # jobs. It is not bounded here: another user's job NAME containing a
            # newline plus a forged continuation carrying THIS uid can still put
            # one fabricated entry in the picker. The id is re-resolved when the
            # user picks it, so a fabricated one fails there and the owner check
            # rejects a real foreign job — but the entry can appear, and saying
            # otherwise would be untrue.
            #
            # HOW BIG the queue is varies by cluster and by the hour, so the
            # figure belongs with the site it was taken at rather than stated as
            # a property of "here":
            #
            #     midway3 login, 2026-08-27:  4,228 rows (9,551 with `-r`), 324 KB
            #     midway2 login, 2026-08-27:    288 rows (  321 with `-r`),  28 KB
            #
            # A 20x spread between two clusters of one site, which is the reason
            # this directory exists. The design does not turn on the number --
            # unfiltered is the last resort at either size -- but a measurement
            # quoted without its cluster invites exactly the wrong conclusion.
            if not allow_unfiltered_scan:
                # The caller has something cheaper to try first. See
                # `UnfilteredScanDeferredError`.
                raise UnfilteredScanDeferredError(own, by_uid) from by_uid
            wanted_uid = own
            logger.info(
                "squeue would not take a numeric uid either (%s); scanning the "
                "whole queue and filtering on uid %d locally",
                by_uid,
                own,
            )
            output = _run_slurm_cmd(["squeue", "-h", "-o", _SQUEUE_UID_FORMAT])
    return output, wanted_uid


def resolve_current_jobs(
    username: str | None = None,
    *,
    allow_unfiltered_scan: bool = True,
    scan_uid: int | None = None,
) -> list[dict[str, object]]:
    if _is_mock():
        return [
            {
                "job_id": "12345",
                "state": "R",
                "partition": "gpu-highend",
                "name": "train",
                "nodes": "4",
                "wall_time": "2:00:00",
                "time_limit": "4:00:00",
                "reason": "None",
            },
        ]
    if username is None and scan_uid is None:
        username = current_username()
    # Pipe-delimited so job names with spaces don't shift columns; the field order
    # and the derived field counts are pinned at `_SQUEUE_FORMAT` above.
    #
    # Three attempts, cheapest and narrowest first, because each step down gives
    # something up:
    #
    #   1. `-u <name>`   — filtered by the controller. Needs a name service.
    #   2. `-u <uid>`    — still filtered by the controller. `squeue -u` documents
    #                      accepting a numeric uid (and `current_username` already
    #                      relies on that), so this needs no name service either
    #                      and keeps the query scoped to one user.
    #   3. no `-u`, `%U` — unfiltered, narrowed by `os.getuid()` on this side.
    #
    # `-u <name>` breaks because a compute node routinely has no name service:
    # `getent passwd <uid>` fails, `whoami` fails, and `scontrol` prints
    # `UserId=nobody(940740146)`. Any site that does not run a name-service client
    # on its compute nodes lands here — a normal hardening choice, not one
    # cluster's quirk. `%u` is useless there (every row prints `nobody`); `%U`
    # carries the numeric uid and is correct.
    wanted_uid: int | None = None
    if scan_uid is not None:
        # Resuming a deferred step 3, straight into the same code path it would
        # have taken. The two filtered attempts already failed on this node;
        # repeating them to arrive back here would cost two round trips to learn
        # nothing that has changed.
        wanted_uid = scan_uid
        output = _run_slurm_cmd(["squeue", "-h", "-o", _SQUEUE_UID_FORMAT])
    else:
        # Non-None on this branch: the guard above resolves `username` whenever
        # `scan_uid` was not given, which is exactly when we are here.
        assert username is not None
        output, wanted_uid = _resolve_filtered(username, allow_unfiltered_scan)
    jobs: list[dict[str, object]] = []
    # One more field in the uid form, and the name is still the last of them.
    fields = _SQUEUE_FIELD_COUNT if wanted_uid is None else _SQUEUE_UID_FIELD_COUNT
    for line in _squeue_rows(output, fields):
        parts = line.split("|", fields - 1)
        parts = [p.strip() for p in parts]
        if wanted_uid is not None:
            # Field 7, between the machine-generated columns and the name. A row
            # whose uid is not a number is dropped rather than attributed to this
            # user: a wrong owner here is somebody else's job on your screen.
            if len(parts) < 8:
                continue
            try:
                row_uid = int(parts[7])
            except ValueError:
                continue
            if row_uid != wanted_uid:
                continue
            parts = parts[:7] + parts[8:]
        # Include running AND pending jobs so the picker offers both (a pending
        # pick routes to the why/when/where view). Other transient states
        # (completing/configuring) aren't monitorable, so they're left out.
        if len(parts) >= 2 and parts[1] in ("R", "PD"):
            job: dict[str, object] = {"job_id": parts[0], "state": parts[1]}
            if len(parts) > 2:
                job["partition"] = parts[2]
            if len(parts) > 3:
                job["nodes"] = parts[3]
            if len(parts) > 4:
                job["wall_time"] = parts[4]
            if len(parts) > 5:
                job["time_limit"] = parts[5]
            if len(parts) > 6:
                job["reason"] = parts[6]
            if len(parts) > 7:
                job["name"] = parts[7]
            jobs.append(job)
    return jobs


def _sacct_final_state(job_id: str) -> tuple[str, str] | None:
    """``(State, End)`` from the accounting DB for a job that has LEFT the node.

    Returns ``None`` if sacct has no record (the job truly never existed), the
    query fails, OR *any* row is still active/requeued — so a transient controller
    failure on a *running* job (or one live task of an array) is never mistaken for
    "finished". ``-X`` limits to top-level job records (no steps); a bare array-base
    id still yields one row per task, so all rows are scanned. A state like
    ``CANCELLED by 1234`` is reduced to its first word.
    """
    if _is_mock():
        return None
    fields = ["State", "End"]
    try:
        out = _run_slurm_cmd(
            ["sacct", "-n", "-P", "-X", "-j", job_id, f"--format={','.join(fields)}"]
        )
    except SlurmCommandError as exc:
        # A rejected FIELD is not "no record" — without this, a version skew made
        # every finished job look still-running and the caller reported a busy
        # controller (SW-19). State alone still answers the question this asks.
        if not _is_malformed_query_error(exc):
            return None
        kept = _drop_unsupported_fields("sacct", fields)
        if not kept or kept == fields:
            return None
        try:
            out = _run_slurm_cmd(
                ["sacct", "-n", "-P", "-X", "-j", job_id, f"--format={','.join(kept)}"]
            )
        except SlurmCommandError:
            return None
    # Scan EVERY row before deciding. Only a genuinely TERMINAL state means the job
    # has left the node; a row still active or requeued (RUNNING/PENDING/SUSPENDED/…)
    # means the caller's *scontrol* call failed transiently — NOT that the job
    # finished — so never report terminal. Crucially, a bare array-base id makes
    # `sacct -X` emit one row PER TASK, so a terminal task ordered first must not
    # classify the whole array as finished: if ANY row is active/requeued the job is
    # still live (return None); only when none are do we return a terminal state
    # (M1). (Returning "RUNNING"/a premature terminal here made resolve_job_context
    # raise "has finished (State: RUNNING)" on a live job when `scontrol show job -d`
    # timed out on a busy controller — surfaced as the bogus "… is in state
    # 'RUNNING', not PENDING.".)
    terminal: tuple[str, str] | None = None
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        state = parts[0].strip().split(" ")[0] if parts[0].strip() else ""
        if not state:
            continue
        if state.upper() in _ACTIVE_JOB_STATES | _REQUEUED_JOB_STATES:
            return None
        if terminal is None:
            end = parts[1].strip() if len(parts) > 1 else ""
            terminal = (state, end)
    return terminal


def resolve_job_context(
    job_id: str,
    step_id: str | None = None,
) -> JobContext:
    if _is_mock():
        return _make_mock_job_context(job_id, step_id)

    try:
        # -d adds the per-node allocation detail lines (GRES=gpu:N(IDX:...)),
        # the only node-global source of allocated GPU indices.
        output = _run_slurm_cmd(["scontrol", "show", "job", "-d", job_id])
    except SlurmCommandError as exc:
        # scontrol purges finished jobs after MinJobAge, but the accounting DB
        # keeps them: distinguish a job that *finished* (still in sacct) from one
        # that never existed, so we don't tell the user a completed job "does not
        # exist" (which reads like a typo'd id).
        finished = _sacct_final_state(job_id)
        if finished is not None:
            state, end = finished
            when = f", ended {end}" if end and end not in ("Unknown", "") else ""
            raise JobNotRunningError(
                f"Job {job_id} has finished (State: {state}{when}). "
                "slurmwatch shows live telemetry for running jobs only.",
                # sacct already told us the state; carry it so the machine row
                # says COMPLETED/CANCELLED/TIMEOUT instead of null.
                {"state": state},
            ) from exc
        # sacct doesn't confirm the job is terminal (no record, or still active).
        # Only an explicit "invalid job id" means the job truly doesn't exist; any
        # other failure (timeout, socket, "Unable to contact slurm controller",
        # connect failure) is transient — surface it as retryable rather than the
        # misleading "not found" (which reads like a mistyped job id).
        if _is_malformed_query_error(exc):
            # Permanent and ours: "try again in a moment" would send the reader
            # after a controller that is working fine (SW-19).
            raise SlurmCommandError(
                f"slurmwatch asked Slurm for a field this version doesn't have "
                f"while looking up job {job_id} ({exc}). This is a slurmwatch bug, "
                "not a cluster problem — retrying won't help.",
                kind="unsupported",
            ) from exc
        if _is_missing_binary_error(exc):
            # Permanent, and nothing to do with the controller: retry advice here
            # sends someone on a non-Slurm cluster round a loop forever.
            raise SlurmCommandError(
                f"{exc} slurmwatch reads Slurm's own tools (scontrol, squeue, sstat), "
                "so they have to be on PATH. If Slurm IS installed here, its bin "
                "directory isn't in this shell's PATH (try `module load slurm`); if "
                "this cluster runs PBS, Flux or Kubernetes instead, slurmwatch cannot "
                "monitor it. Retrying will not help.",
                kind="unavailable",
            ) from exc
        if not _is_missing_job_error(exc):
            raise SlurmCommandError(
                f"Couldn't reach the Slurm controller for job {job_id} ({exc}). "
                "It may be busy — try again in a moment."
            ) from exc
        raise JobNotFoundError(f"Job {job_id} not found") from exc

    hostname = local_node_name()
    record = _select_job_record(output, hostname)

    # Het jobs return one record per component (JobId=<leader>+N); we resolve only
    # the selected component, so its sibling components' nodes/GPUs aren't shown.
    # Warn rather than silently misrepresent the job's scope (full het aggregation
    # is not implemented).
    het = _count_het_components(output)
    if het > 1:
        logger.warning(
            "Job %s is heterogeneous (%d components); slurmwatch is showing only "
            "one component's node(s) — the other components aren't monitored.",
            job_id,
            het,
        )

    # squeue first, for every field a forged line could poison. `scontrol show job`
    # prints JobName BEFORE these and Comment AFTER them, both accept a newline, and
    # `_parse_scontrol_field` takes the FIRST match — so an injected copy of any of
    # them wins. The one anchor no injected line can precede is the record's own
    # first line, so the id read from it is what we query with (SW-1 feedback,
    # generalised from the owner to every load-bearing field).
    raw_for_facts = _parse_scontrol_field(record, "JobId") or job_id
    facts = _authoritative_job_facts(raw_for_facts, _squeue_id_for_record(record))

    def _fact(key: str, field: str) -> str:
        """squeue's answer when it has one, else the record's."""
        return facts.get(key) or (_parse_scontrol_field(record, field) or "")

    job_state = _fact("state", "JobState")
    if job_state and job_state.upper() not in ("RUNNING", "CONFIGURING", "COMPLETING"):
        raise JobNotRunningError(
            f"Job {job_id} is in state '{job_state}'. Only running jobs can be monitored.",
            # Everything squeue/scontrol already answered for this job. The state is
            # the point, but owner/partition/name cost nothing here and a poller that
            # loses them at the last frame has to re-query to label its own row.
            {
                "state": job_state,
                "job_name": _fact("name", "JobName") or None,
                "owner": _owner_name_for_facts(_fact("username", "UserId")),
                "partition": _fact("partition", "Partition") or None,
            },
        )

    username, uid = ("", None)
    if facts.get("uid", "").isdigit():
        uid = int(facts["uid"])
        name = facts.get("username", "")
        username = "" if name.lower() in _PLACEHOLDER_NAMES else name
        username = username or _name_for_uid(uid)
    if uid is None:
        # No squeue answer: the record parse, which refuses to guess when the record
        # disagrees with itself.
        username, uid = _owner_from_record(record)

    partition = _fact("partition", "Partition") or "unknown"
    nodelist_raw = _fact("nodelist", "NodeList")
    cpus = _parse_leading_int(_fact("cpus", "NumCPUs"))
    num_nodes = max(_parse_leading_int(_fact("nodes", "NumNodes")), 1)

    resolved_nodes = _parse_nodelist(nodelist_raw)

    # Array-task membership: scontrol carries ArrayJobId (the array's base id) and
    # ArrayTaskId (this task's index) on each task's record; both absent on a
    # non-array job. We only resolve a running task, whose ArrayTaskId is a single
    # index — guard against a still-pending meta-record's range ("20-22").
    array_job_id = _parse_scontrol_field(record, "ArrayJobId") or ""
    array_task_id = _parse_scontrol_field(record, "ArrayTaskId") or ""
    is_array_task = bool(array_job_id) and array_task_id.isdigit()
    if not is_array_task:
        array_job_id = array_task_id = ""
    # A bare-base request ("52353625") resolves to one arbitrary running task via
    # _select_job_record; label the context with the task we actually picked
    # ("52353625_19") so the header isn't a misleading array-wide id. A request
    # that already names its element ("_15" / het "+0") is left untouched.
    display_job_id = job_id
    if is_array_task and "_" not in job_id and "+" not in job_id:
        display_job_id = f"{array_job_id}_{array_task_id}"
        # ...and SAY so when the choice was ambiguous. With tasks 1, 2 and 3 all
        # running, `sw <base-id>` monitors _3 every time: deterministic, but nothing
        # told the user which of the three they were looking at or how to ask for
        # another, and --help documents only the `12345_3` spelling. One line, and
        # only when there is more than one running task to choose between. SW-9.
        counts = resolve_array_task_counts(array_job_id)
        if counts is not None and counts[0] > 1:
            logger.warning(
                "Array %s has %d running tasks; showing %s — pass a task id "
                "(e.g. %s_<n>) to watch a different one.",
                array_job_id,
                counts[0],
                display_job_id,
                array_job_id,
            )

    # The node whose per-node detail we read. On the compute node that's this
    # host; viewed off-node (login node / --once / --log) the host is in no
    # detail line, so scope to the node the collector will actually represent —
    # nodelist[0], the hop/stream/remote-summary target — instead of falling
    # back to a job-wide // NumNodes average that matches no real node on a
    # heterogeneous allocation (#31).
    if _host_in_nodelist(hostname, resolved_nodes) or not resolved_nodes:
        detail_host = hostname
    else:
        detail_host = resolved_nodes[0]

    tres = _parse_scontrol_field(record, "TRES") or ""
    alloc_tres = _parse_scontrol_field(record, "AllocTRES") or ""
    tres_str = alloc_tres or tres

    mem_bytes = 0
    gpu_count = 0
    if tres_str:
        for token in tres_str.split(","):
            token = token.strip()
            if token.startswith("mem="):
                raw_mem = token.split("=", 1)[1]
                parsed_mem = _parse_mem_to_bytes(raw_mem)
                if parsed_mem is None:
                    # Leaves mem_bytes at its "not known" 0 — which the collector
                    # then reads as "no cgroup cap" and replaces with the node's
                    # whole RAM, so the gauge silently measures against the wrong
                    # ceiling. Say it out loud rather than let a spelling we don't
                    # know become a fact (SW-12).
                    logger.warning(
                        "Could not read the job's memory limit from TRES 'mem=%s'; "
                        "the MEM gauge will fall back to the node's capacity.",
                        raw_mem,
                    )
                mem_bytes = parsed_mem or 0
        gpu_count = _parse_tres_gpus(tres_str)

    min_memory_node = 0
    min_mem_str = _parse_scontrol_field(record, "MinMemoryNode") or ""
    if min_mem_str:
        min_memory_node = _parse_mem_to_bytes(min_mem_str) or 0
    min_mem_per_cpu = _parse_mem_to_bytes(_parse_scontrol_field(record, "MinMemoryCPU") or "") or 0

    # slurmwatch monitors one node, so limits must be node-local. Prefer the
    # exact per-node figures on the `scontrol -d` detail line (CPU_IDs / Mem) for
    # the target node; fall back to the job-wide totals only when the detail line
    # is absent (B-P4). The fallback rounds UP (ceil): on a job that doesn't
    # divide evenly (30 CPUs over 4 nodes) it matches the largest real node
    # instead of truncating to 7 and inflating every % against a too-small limit,
    # which could trip a false OOM-critical (#32).
    node_cpus, node_mem = _parse_node_detail(record, detail_host)
    if node_cpus > 0:
        cpus = node_cpus
    elif num_nodes > 1:
        cpus = max(-(-cpus // num_nodes), 1)
    if min_memory_node == 0 and min_mem_per_cpu:
        # Slurm prints MinMemoryNode *or* MinMemoryCPU, never both — checked across 60
        # running jobs on this cluster (53 node-only, 7 cpu-only, 0 both, 0 neither),
        # so the claim this fallback rests on is measured rather than read. A site with
        # DefMemPerCPU set (or a user passing --mem-per-cpu) only ever gets the
        # latter. Reading only the per-node field left this fallback at 0, which the
        # collector reads as "no cap" and replaces with the node's whole RAM — so a
        # 24 GiB job's gauge measured against 192 GiB. The pending view already did
        # this multiplication; the running-job path did not. Multiplied by the
        # PER-NODE cpu count resolved just above, because that is what
        # min_memory_node means everywhere it is used below.
        min_memory_node = min_mem_per_cpu * max(cpus, 1)
    if node_mem > 0:
        mem_bytes = node_mem
    elif num_nodes > 1:
        mem_bytes = min_memory_node if min_memory_node > 0 else -(-mem_bytes // num_nodes)
    if mem_bytes == 0:
        mem_bytes = min_memory_node

    # Per-node GPU count: prefer the exact IDX list on the target node's detail
    # line (node-local and precise), so an uneven GPU-per-node allocation shows
    # the right number instead of a job-wide // NumNodes average that contradicts
    # the GPU rows actually rendered (#33). Fall back to a per-node Gres field,
    # then to division.
    detail_gpu_indices = _parse_gres_idx(record, detail_host)
    per_node_gpus = 0
    for field in ("TresPerNode", "Gres"):
        gres = _parse_scontrol_field(record, field) or ""
        per_node_gpus = _parse_gpu_count(gres)
        if per_node_gpus:
            break
    if detail_gpu_indices:
        gpu_count = len(detail_gpu_indices)
    elif num_nodes > 1:
        # TRES gres/gpu=N is the job-wide total; the panel wants this node's.
        gpu_count = per_node_gpus if per_node_gpus else -(-gpu_count // num_nodes)
    elif gpu_count == 0:
        gpu_count = per_node_gpus

    def _parse_scontrol_time(field: str) -> float | None:
        raw = _parse_scontrol_field(record, field)
        if not raw or raw in ("Unknown", "N/A"):
            return None
        try:
            return time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OSError, OverflowError):
            return None

    job_start_time = _parse_scontrol_time("StartTime")
    submit_time = _parse_scontrol_time("SubmitTime")

    # Job provenance from the same record — for the dashboard's JOB card. These
    # are single-token scontrol fields (Command is the script path; args aren't
    # in this field), so the existing key=value parser captures them cleanly.
    # scontrol prints "(null)" for an unset field (e.g. Command on an interactive
    # salloc job) — normalize that (and N/A/Unknown) to "" so the card omits the
    # line instead of showing a useless "(null)".
    def _clean_field(field: str) -> str:
        val = _parse_scontrol_field(record, field) or ""
        return "" if val in ("(null)", "(none)", "N/A", "Unknown") else val

    # JobName can contain spaces, so it's in _MULTIWORD_FIELDS and the parser reads
    # it to the end of its line. Slurm defaults it to the script's basename (or
    # "bash"/"interactive" for a salloc), so it's essentially always present.
    job_name = _clean_field("JobName") or _clean_field("Name")
    account = _clean_field("Account")
    qos = _clean_field("QOS")
    command = _clean_field("Command")
    work_dir = _clean_field("WorkDir")
    std_out = _clean_field("StdOut")
    std_err = _clean_field("StdErr")

    # TimeLimit is 'D-HH:MM:SS' / 'HH:MM:SS' — or 'UNLIMITED'/'Partition_Limit'
    # when there's no fixed wall-clock cap (leave it None then).
    time_limit_str = _parse_scontrol_field(record, "TimeLimit") or ""
    time_limit_seconds: int | None = None
    if time_limit_str and time_limit_str.upper() not in ("UNLIMITED", "PARTITION_LIMIT", "N/A"):
        secs = _parse_slurm_duration(time_limit_str)
        if secs > 0:
            time_limit_seconds = int(secs)

    ctx = JobContext(
        job_id=display_job_id,
        username=username,
        partition=partition,
        nodelist=",".join(resolved_nodes) if resolved_nodes else nodelist_raw,
        nodelist_compact=nodelist_raw,
        hostname=hostname,
        cpus_allocated=cpus,
        mem_limit_bytes=mem_bytes,
        gpu_count_requested=gpu_count,
        # A `--gres=shard:2` / `--gres=mps:100` job requests GPU that no device count
        # can express; carry the request itself so no surface has to publish the
        # uncountable as a measured zero (D18).
        gpu_fraction_request=_parse_gpu_fraction_request(record),
        # Left empty here on purpose: these mean "the devices THIS process can
        # attach", which off-node is none. The job-wide map below is a different
        # question and is answerable from anywhere.
        gpu_indices=[],
        gpu_uuids=[],
        # Derived from the scontrol record alone — no cgroup, no NVML, no local
        # state — so it is available OFF-node too. Set here rather than only on the
        # on-node path, which returns much later: the cross-node GPU view and any
        # --json consumer running from a login node were getting {} and silently
        # showing nothing for a multi-node job.
        gpu_indices_by_node=parse_gres_idx_by_node(record),
        step_id=step_id,
        uid=uid,
        job_start_time=job_start_time,
        job_state=job_state,
        time_limit_seconds=time_limit_seconds,
        nodelist_resolved=resolved_nodes,
        min_memory_node=min_memory_node,
        tres=tres_str,
        job_name=job_name,
        account=account,
        qos=qos,
        command=command,
        work_dir=work_dir,
        std_out=std_out,
        std_err=std_err,
        submit_time=submit_time,
        array_job_id=array_job_id,
        array_task_id=array_task_id,
    )

    # Cgroups are named after the task's raw JobId (array tasks and het
    # components have their own), not the user-facing 12345_3 / 123+1 form.
    raw_job_id = _parse_scontrol_field(record, "JobId") or job_id
    # Persist it before the possible remote early-return: `srun --jobid=` only
    # accepts this numeric id, so the login-node hop needs it too.
    ctx.raw_job_id = raw_job_id
    try:
        cgroup_paths = _discover_cgroup_paths(raw_job_id, uid, step_id)
    except CgroupNotFoundError:
        # Not on the job's compute node (e.g. a login node): fall back to
        # remote usage via sstat instead of erroring. GPU utilization is
        # unavailable this way, but memory and CPU are.
        ctx.remote = True
        return ctx
    ctx.cgroup_v2_path = str(cgroup_paths.get("v2")) if cgroup_paths.get("v2") else None
    ctx.cgroup_v1_mem_path = str(cgroup_paths.get("v1_mem")) if cgroup_paths.get("v1_mem") else None
    ctx.cgroup_v1_cpu_path = str(cgroup_paths.get("v1_cpu")) if cgroup_paths.get("v1_cpu") else None

    job_pids = _cgroup_pids(
        [p for p in (cgroup_paths.get("v2"), cgroup_paths.get("v1_cpu")) if p is not None]
    )
    gpu_indices, gpu_uuids = _resolve_gpu_indices(record, hostname, job_pids)
    ctx.gpu_indices = gpu_indices
    ctx.gpu_uuids = gpu_uuids
    # Job-wide GPU layout, so the GPU view can show every node at once instead of
    # making the user hop node by node. Static for the job's life and tiny, so it
    # rides on the context (fetched once) rather than on every streamed snapshot.
    ctx.gpu_indices_by_node = parse_gres_idx_by_node(record)
    if ctx.gpu_count_requested == 0 and gpu_indices:
        ctx.gpu_count_requested = len(gpu_indices)

    return ctx


def _job_owner_differs(job_ctx: JobContext) -> bool:
    """True when the job belongs to a *different* user than the one running ``sw``.

    Slurm only lets a job's owner (or root/SlurmUser) create job steps in its
    allocation or read its ``sstat`` usage. So both ways slurmwatch gets live
    data off a login node — the ``srun --overlap`` hop and the remote sstat
    summary — fail for someone else's job: the hop errors with
    "Unable to create step for job <id>: Access/permission denied" and sstat
    returns nothing. There is simply no live telemetry to be had for another
    user's job from a login node, so detect it up front and show an honest
    read-only summary instead of attempting the doomed hop.

    Conservative by design: returns True only when we positively know both names
    AND they differ. An unknown owner (scontrol parse gap), an unknown caller, or
    root (which *can* attach to any job) all fall through to the existing
    behavior, so the owner's own-job path is never regressed.
    """
    # root / SlurmUser can create steps and read sstat for any job — never treat
    # another user's job as unreachable for them.
    try:
        my_uid: int | None = os.getuid()
    except AttributeError:  # pragma: no cover - non-POSIX; getuid always exists on Linux
        my_uid = None
    if my_uid == 0:
        return False
    # Prefer a UID comparison (N11). getpass.getuser() below trusts $USER/$LOGNAME,
    # which a `su otheruser` (no dash) or `sudo -E` shell leaves stale — that would
    # make `sw <your-own-job>` look foreign and silently skip the live hop for your
    # OWN job. os.getuid() and the job's scontrol-derived uid can't be spoofed by
    # the environment, so compare those whenever both are known.
    if my_uid is not None and job_ctx.uid is not None:
        return my_uid != job_ctx.uid
    # Fall back to login names only when a uid isn't available on either side.
    owner = (job_ctx.username or "").strip()
    if not owner:
        return False
    try:
        me = getpass.getuser()
    except Exception:
        # No resolvable login name (odd env) → can't be sure; don't override.
        return False
    return bool(me) and me != owner


def resolve_array_task_counts(array_job_id: str) -> tuple[int, int] | None:
    """``(running, pending)`` task counts for an array, or ``None`` if unavailable.

    ``squeue -r`` lists one row per array element. Only currently-queued tasks are
    counted — finished tasks have left the queue — so this reflects the array's
    *live* state, not its original size. Best-effort: any failure (or a mock/empty
    result, or an array with nothing live) returns ``None`` so the caller simply
    omits the counts rather than showing a fabricated 0/0.
    """
    if not array_job_id or _is_mock():
        return None
    try:
        output = _run_slurm_cmd(["squeue", "-j", array_job_id, "-r", "-h", "-o", "%T"])
    except SlurmCommandError:
        return None
    running = pending = 0
    # Count the transient active states as "running" too, or a mid-transition array
    # undercounts (and one with only such tasks wrongly returns None and omits the
    # line entirely). SIGNALING/REQUEUED are included so they aren't silently dropped
    # into neither bucket — the same undercount fixed for the queue counter (A6).
    running_ish = {
        "RUNNING",
        "COMPLETING",
        "SUSPENDED",
        "CONFIGURING",
        "RESIZING",
        "SIGNALING",
        "REQUEUED",
    }
    for line in output.strip().split("\n"):
        state = line.strip().upper()
        if state in running_ish:
            running += 1
        elif state == "PENDING":
            pending += 1
    if running + pending == 0:
        return None
    return (running, pending)


class RemoteUsage:
    """Live job usage sampled remotely via sstat (works from any node)."""

    def __init__(self, rss_bytes: int, cpu_seconds: float, sampled: bool) -> None:
        self.rss_bytes = rss_bytes
        self.cpu_seconds = cpu_seconds
        self.sampled = sampled  # False until Slurm has taken its first sample


def _parse_slurm_duration(text: str) -> float:
    """Parse a Slurm duration to seconds.

    Handles 'D-HH:MM:SS', 'HH:MM:SS', 'MM:SS.mmm', and the short 'D-HH' / 'D-HH:MM'
    day forms — after a day component the fields run HH[:MM[:SS]] (LEFT-aligned to
    hours), whereas without a day they are right-aligned to seconds.
    """
    text = text.strip()
    if not text:
        return 0.0
    days = 0
    # Day form is `<digits>-HH...`; require the dash to FOLLOW leading digits so a
    # stray leading minus ("-5") isn't misread as a day separator (→ 5 hours). Real
    # Slurm durations are never negative, so this is defensive, but cheap and exact.
    has_day = bool(re.match(r"\d+-", text))
    if has_day:
        day_str, _, text = text.partition("-")
        with contextlib.suppress(ValueError):
            days = int(day_str)
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if has_day:
        # HH[:MM[:SS]] — pad on the RIGHT to [HH, MM, SS] so 'D-HH' / 'D-HH:MM' don't
        # mis-read the last field as seconds.
        nums = (nums + [0.0, 0.0])[:3]
        seconds = nums[0] * 3600 + nums[1] * 60 + nums[2]
    else:
        seconds = 0.0
        for n in nums:
            seconds = seconds * 60 + n
    return days * 86400 + seconds


_ACCT_GATHER_WARNED = False


def _warn_if_acct_gather_disabled() -> None:
    """Say once why every off-node figure will be zero on this cluster.

    The prose summary and the dashboard both explain it in place, but ``--json``,
    ``--csv`` and ``--log`` show only the numbers: a consumer logging zeros for an
    hour has no way to learn that this Slurm gathers nothing.
    """
    global _ACCT_GATHER_WARNED
    if _ACCT_GATHER_WARNED or not acct_gather_disabled():
        return
    _ACCT_GATHER_WARNED = True
    logger.warning(
        "This Slurm has JobAcctGatherType=none, so sstat reports no CPU/memory for a "
        "running job: every off-node figure will read zero. Run slurmwatch ON the "
        "compute node (or let --once hop there) for real measurements."
    )


def resolve_remote_usage(job_id: str, node_count: int = 1) -> RemoteUsage:
    """Query sstat for a running job's per-node peak RSS and CPU time.

    sstat totals are job-wide, but slurmwatch compares against per-node limits,
    so each step is scaled by an estimated per-node task count
    (max(1, ceil(NTasks / node_count)) — the busiest node's share). Using at least
    one task means a concentrated step (NTasks < nodes, or a single-task head step)
    reports its real single-node footprint rather than being diluted by node_count.
    Returns zeros with sampled=False when Slurm has not yet produced a sample.
    """
    if _is_mock():
        return RemoteUsage(rss_bytes=32 * 1024**3, cpu_seconds=3600.0, sampled=True)
    _warn_if_acct_gather_disabled()
    fields = ["JobID", "MaxRSS", "AveCPU", "NTasks"]

    def _sstat(cols: list[str]) -> str:
        return _run_slurm_cmd(
            [
                "sstat",
                "--allsteps",
                "--noheader",
                "-P",
                "-j",
                job_id,
                f"--format={','.join(cols)}",
            ]
        )

    cols = fields
    try:
        output = _sstat(fields)
    except SlurmCommandError as exc:
        # One unknown field rejects the WHOLE query, so a rename on this site's
        # Slurm would cost every figure rather than one column. Ask what it does
        # support and try once more — the parser below tolerates missing columns
        # (a short row is skipped), so partial data beats none. SW-19.
        kept = _drop_unsupported_fields("sstat", fields) if _is_malformed_query_error(exc) else []
        if not kept or kept == fields:
            return RemoteUsage(rss_bytes=0, cpu_seconds=0.0, sampled=False)
        try:
            output = _sstat(kept)
        except SlurmCommandError:
            return RemoteUsage(rss_bytes=0, cpu_seconds=0.0, sampled=False)
        cols = kept

    peak_rss = 0
    cpu_seconds = 0.0
    sampled = False
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        values = line.split("|")
        if len(values) < len(cols):
            continue
        # By NAME, not by position: when a field was dropped because this Slurm
        # rejected it, the remaining columns shift, and reading NTasks as AveCPU
        # would produce a confidently wrong number — worse than the missing data
        # the retry was added to avoid (SW-19).
        row = dict(zip(cols, values, strict=False))
        job_field = row.get("JobID", "")
        max_rss = row.get("MaxRSS", "")
        ave_cpu = row.get("AveCPU", "")
        ntasks = row.get("NTasks", "")
        if not job_field:
            continue  # without the id we cannot scope the row to this job
        # Scope to the requested job. `sstat -j <ArrayJobId>` widens to EVERY
        # running array task (the representative task's raw id equals the
        # ArrayJobId), so summing every row would over-count CPU N-fold — the #30
        # fix's blind spot for the base task. Keep only steps whose base id matches.
        if job_field.strip().split(".")[0] != job_id:
            continue
        if not max_rss.strip():
            # No RSS sample -> no valid CPU sample either (skips extern step).
            continue
        sampled = True
        try:
            tasks = int(ntasks.strip())
        except ValueError:
            tasks = 1
        tasks = max(tasks, 1)
        # MaxRSS/AveCPU are single-task figures; scale by the per-node task count
        # for a per-node total (exact for balanced tasks such as MPI ranks). Use
        # CEIL, not floor: MaxRSS is compared against a PER-NODE limit, so the
        # multiplier must be the BUSIEST node's task count. For NTasks not divisible
        # by node_count, floor picks the least-loaded node and under-reports RSS/CPU
        # — the OOM-dangerous direction for --mem sizing (A4). ceil==floor for
        # balanced or single-task steps, so those are unchanged.
        tasks_per_node = max(1, -(-tasks // max(1, node_count)))
        # MaxRSS: a bare figure here is KILOBYTES, not the megabytes a bare --mem
        # means. sstat normally suffixes it ("523508K"), but a site that doesn't
        # would be read 1024x too high under the request-field default.
        peak_rss = max(peak_rss, (_parse_mem_to_bytes(max_rss, "K") or 0) * tasks_per_node)
        step_cpu = _parse_slurm_duration(ave_cpu)
        # Steps Slurm hasn't sampled report a NO_VAL sentinel
        # (e.g. AveCPU "213503982334-14:25:51"); ignore anything absurd.
        if step_cpu >= _MAX_SANE_CPU_SECONDS:
            continue
        cpu_seconds += step_cpu * tasks_per_node
    return RemoteUsage(rss_bytes=peak_rss, cpu_seconds=cpu_seconds, sampled=sampled)


# States in which a job is still on a compute node and worth monitoring. SUSPENDED
# and STOPPED are held allocations (gang scheduling, PreemptMode=SUSPEND, `scontrol
# suspend`) that resume on the SAME node, so they are NOT ended. Anything else
# (COMPLETED, FAILED, CANCELLED, TIMEOUT, ...) means the job has left the node.
_ACTIVE_JOB_STATES = frozenset(
    {"RUNNING", "COMPLETING", "CONFIGURING", "RESIZING", "SIGNALING", "SUSPENDED", "STOPPED"}
)
# States where the job is back in the queue but NOT finished — it will run again
# under the same JobId (preemption with PreemptMode=REQUEUE, `scontrol requeue`,
# NODE_FAIL with --requeue, a held requeue). Treating these as "ended" would tear
# the dashboard down on a job that's merely waiting to resume, so they count as
# alive (there's just no node telemetry until it runs again). PENDING covers a
# requeued job that has settled back to the pending queue.
# SPECIAL_EXIT belongs here for the same reason REQUEUE_HOLD does: Slurm sets it
# when a job exits with a code the site's `--requeue` policy treats specially, and
# the job is then requeued AND held — still in the queue, still the same JobId,
# still going to run. Missing from both sets, `is_job_active` fell through to its
# "not listed as active -> ended" answer and slurmwatch announced "job ended" for a
# job that had not. Found auditing the state vocabulary rather than from a report:
# which states a site can produce depends on its requeue/preemption policy, so a
# gap here is invisible until you run somewhere that uses it.
_REQUEUED_JOB_STATES = frozenset(
    {
        "PENDING",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESV_DEL_HOLD",
        "SPECIAL_EXIT",
    }
)


def is_job_active(job_id: str) -> bool | None:
    """Whether ``job_id`` is still on a node, for a mid-flight liveness recheck.

    ``True`` = still allocated (running or a resumable suspend); ``False`` =
    gone/terminal (so the dashboard can show "job ended" and stop); ``None`` =
    couldn't tell, so the caller must NOT treat it as ended — a transient squeue
    hiccup or a slow controller should never tear down a live dashboard.

    ``squeue`` only lists active jobs, so an empty result means the job has left
    the queue (ended). A purged job is instead *rejected* with "Invalid job id
    specified"; that exact message is the only failure taken as ended — every
    other error (timeouts, socket errors, controller unreachable) is unknown, so
    we never mistake a slow/flaky controller for a finished job. Pass the raw
    numeric JobId: ``squeue -j 12345`` and ``12345_3`` both widen to the whole
    array, but any still-active task keeps it True regardless.
    """
    if _is_mock():
        return True
    try:
        output = _run_slurm_cmd(["squeue", "-h", "-j", job_id, "-o", "%T"])
    except SlurmCommandError as exc:
        # Only an explicit "invalid/unknown job id" means the job is truly gone.
        # Anything else (timeout, "Socket timed out on send/recv", "Unable to
        # contact slurm controller") is a transient/infra failure -> unknown.
        if _is_missing_job_error(exc):
            return False
        return None
    states = [line.strip().upper() for line in output.strip().split("\n") if line.strip()]
    if not states:
        return False  # not listed among active jobs -> ended
    # A requeued/preempted job is still queued (will rerun under the same id), so
    # it's alive even though it's momentarily off the node — don't declare "ended".
    return any(state in _ACTIVE_JOB_STATES or state in _REQUEUED_JOB_STATES for state in states)


def _host_in_nodelist(hostname: str, nodes: list[str]) -> bool:
    """Whether ``hostname`` is one of ``nodes``, tolerant of case and domain.

    A node's own ``gethostname`` and Slurm's ``NodeName`` can differ by case or a
    kept domain suffix on some clusters (e.g. ``gpu01`` vs ``gpu01.cluster.edu``).
    Comparing the short forms on *both* sides keeps host matching working there
    (#29); an exact ``in`` test silently failed, discarding the per-node CPU/mem/
    GPU detail, picking the wrong array-task record, and mis-flagging a live local
    job as remote. ``short_host`` also lower-cases, so it covers case mismatch.
    This matches how ``collector.py`` and ``tui.py`` already resolve the local
    node, so the whole codebase agrees on what "this host" means.

    Slurm ``NodeName`` values are unique short names within a cluster, so the
    short forms don't collide in practice. The only shapes that could over-match
    are degenerate (two nodes named identically apart from their domain, or
    bare-IP node names collapsing to a first octet) and are not produced by a
    normal single-domain Slurm config."""
    target = short_host(hostname)
    return any(short_host(n) == target for n in nodes)


def _select_job_record(output: str, hostname: str) -> str:
    """Pick the right record when scontrol returns several (job arrays).

    Prefers a RUNNING record whose nodelist contains this host, then any
    RUNNING record, then the first record.
    """
    records = [r for r in re.split(r"\n\s*\n", output) if "JobId=" in r]
    if len(records) <= 1:
        return records[0] if records else output
    running = [
        r
        for r in records
        if (_parse_scontrol_field(r, "JobState") or "").upper()
        in ("RUNNING", "CONFIGURING", "COMPLETING")
    ]
    for record in running:
        nodes = _parse_nodelist(_parse_scontrol_field(record, "NodeList") or "")
        if _host_in_nodelist(hostname, nodes):
            return record
    return running[0] if running else records[0]


def _parse_leading_int(value: str | None) -> int:
    if not value:
        return 0
    m = re.match(r"\d+", value)
    return int(m.group(0)) if m else 0


def _parse_tres_gpus(tres_str: str) -> int:
    """GPU count from a TRES string.

    Only `gres/gpu=N` and typed `gres/gpu:type=N` count; `gres/gpumem=...` and
    `gres/gpuutil=...` share the prefix but are different TRES. The generic
    entry is the total; typed entries are summed only when it is absent.
    """
    generic: int | None = None
    typed_total = 0
    for token in tres_str.split(","):
        m = re.match(r"gres/gpu(?::([^=]+))?=(\d+)$", token.strip())
        if not m:
            continue
        if m.group(1) is None:
            generic = int(m.group(2))
        else:
            typed_total += int(m.group(2))
    return generic if generic is not None else typed_total


# A ``key=`` token: a key of any non-space, non-'=' chars (so colon/slash keys
# like ``AllocNode:Sid`` and ``gres/gpu`` are single tokens), at a start-of-line
# or whitespace boundary. Each field's value runs up to the NEXT such token.
_SCONTROL_KEY_RE = re.compile(r"(?:^|\s)([^\s=]+)=")

# Free-form fields whose value is user-controlled and may itself contain a
# ``<space>Key=value`` sequence. Once one starts, the rest of ITS line is its
# value — so a job name like ``x Partition=[/]`` can't shadow the real
# ``Partition=`` on a later line, corrupt the reported field, or (before the
# panels escaped) smuggle markup that crashes the TUI (#audit3-9). Each is the
# last field on its own line in ``scontrol show job`` output.
_SCONTROL_FREE_TEXT_KEYS = frozenset(
    {"JobName", "Name", "Command", "WorkDir", "Comment", "StdOut", "StdErr", "StdIn"}
)


def _parse_scontrol_field(output: str, field: str) -> str | None:
    """Value of ``field`` from ``scontrol`` key=value output.

    A value may contain spaces (Command, WorkDir, JobName, Comment, Std* paths),
    so it must not be truncated at the first space (#37); it runs from just after
    ``field=`` to the next ``key=`` token (or end of line). Splitting on the next
    key token — rather than a lazy ``.*?`` regex — keeps values with spaces intact
    AND correctly bounds a field whose neighbour has a colon/slash key: real
    scontrol prints ``Partition=gpu AllocNode:Sid=login1:42`` on one line, and a
    ``\\w+=`` lookahead would over-capture that whole tail into Partition. It is
    also linear, avoiding the O(n^2) backtracking a lazy regex has on long input.
    ``key=`` inside a comma-joined value (``TRES=cpu=4,mem=8G``) isn't matched: it
    follows ``=``/``,``, not a whitespace/line boundary.
    """
    for line in output.split("\n"):
        tokens = list(_SCONTROL_KEY_RE.finditer(line))
        # Once a free-text key starts, the rest of the line is ITS value — drop
        # any later ``key=`` tokens so an embedded ``Partition=…`` inside a job
        # name can't be read as a real field (#audit3-9).
        for j, m in enumerate(tokens):
            if m.group(1) in _SCONTROL_FREE_TEXT_KEYS:
                tokens = tokens[: j + 1]
                break
        for i, m in enumerate(tokens):
            if m.group(1) == field:
                start = m.end()
                end = tokens[i + 1].start() if i + 1 < len(tokens) else len(line)
                return line[start:end].strip()
    return None


def _parse_gpu_count(gres: str) -> int:
    """GPU count from a Gres/TresPerNode value like 'gpu:2' or 'gres/gpu:a100:2'."""
    if not gres:
        return 0
    total = 0
    for part in gres.split(","):
        part = part.strip()
        gpu_match = re.match(r"(?:gres/)?gpu(?::[\w.\-]+)?:(\d+)", part)
        if gpu_match:
            total += int(gpu_match.group(1))
    return total


# Slurm's two SHARED-GPU GRES: ``gres/shard`` (a device split into N shards) and
# ``gres/mps`` (a percentage of a device). Both spellings appear in two places — the
# TRES form ``gres/shard[:type]=N`` (job-wide) and the per-node
# ``[gres:]shard[:type]:N`` of TresPerNode/Gres.
_TRES_FRACTION_RE = re.compile(r"gres/(shard|mps)(?::[^=]+)?=(\d+)$")
_GRES_FRACTION_RE = re.compile(r"(?:gres:)?(shard|mps)(?::[\w.\-]+)?:(\d+)$")


def _parse_gpu_fraction_request(record: str) -> str:
    """The shared-GPU GRES this job asked for, as ``"shard:2"`` / ``"mps:100"``.

    ``""`` when it asked for none. This is a REQUEST, not a device count, and it is
    kept out of :func:`_parse_tres_gpus` on purpose: a shard is a slice of a GPU and
    an mps figure is a percentage of one, so summing either into
    ``gpu_count_requested`` would replace a false zero with a false device count —
    and the collector would then try to attach NVML to that many devices. What the
    count CANNOT express, this field says in Slurm's own words, so a renderer can
    stop asserting "no GPUs requested" about a job that asked for a fraction of one
    (D18).

    Read from the per-node fields first (TresPerNode/Gres), so it means the same
    thing ``gpu_count_requested`` does — this node's request — falling back to the
    job-wide TRES when the record carries no per-node form.
    """
    for field in ("TresPerNode", "Gres"):
        for part in (_parse_scontrol_field(record, field) or "").split(","):
            m = _GRES_FRACTION_RE.match(part.strip())
            if m:
                return f"{m.group(1)}:{m.group(2)}"
    tres = _parse_scontrol_field(record, "AllocTRES") or _parse_scontrol_field(record, "TRES") or ""
    for token in tres.split(","):
        m = _TRES_FRACTION_RE.match(token.strip())
        if m:
            return f"{m.group(1)}:{m.group(2)}"
    return ""


def _resolve_uid(username: str) -> int | None:
    try:
        return pwd.getpwnam(username).pw_uid
    except (KeyError, OSError):
        return None


# Names a name service hands back when it CANNOT map a uid. They are real passwd
# entries (`nobody` is uid 99 / 65534), so getpwnam succeeds on them and nothing
# raises — which is exactly why they have to be recognised by name.
_PLACEHOLDER_NAMES = frozenset({"nobody", "nfsnobody"})


def _owner_name_for_facts(raw: str) -> str | None:
    """A login name from squeue's ``%u`` or scontrol's ``UserId=name(uid)``.

    ``None`` (not a guess, not the placeholder) when the name is one of the
    stand-ins an unresolvable uid produces — the same distinction
    :func:`resolve_job_context` makes for a live job, so a finished job's row does
    not report an owner called "nobody".
    """
    name = raw.split("(")[0].strip()
    if not name or name.lower() in _PLACEHOLDER_NAMES:
        return None
    return name


def _name_for_uid(uid: int) -> str:
    """A display name for ``uid``, or ``""`` — never a placeholder."""
    try:
        name = pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        name = ""
    if name.lower() in _PLACEHOLDER_NAMES:
        name = ""
    if not name and uid == _own_uid():
        # It's us. $USER can be stale, but it is labelling a uid that came from
        # getuid(), so it can only be wrong about the spelling of our own name.
        name = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    return name


def _own_uid() -> int | None:
    try:
        return os.getuid()
    except AttributeError:  # pragma: no cover - POSIX always has getuid
        return None


def _parse_user_id(raw: str) -> tuple[str, int | None]:
    """``UserId=youzhi(940740146)`` → ``("youzhi", 940740146)``.

    Take the NUMBER, not the name. The uid in the parentheses is what Slurm
    itself recorded and the one part of this field a thin name service cannot
    corrupt: on a compute node whose passwd lookup doesn't resolve the uid
    (normal on diskless/imaged compute images, and any site whose nodes run a
    thinner nsswitch than its login nodes), scontrol prints
    ``UserId=nobody(940740146)``. Re-resolving that NAME through getpwnam
    succeeds and returns 99 — a real uid belonging to someone else — so nothing
    raises, `_job_owner_differs` says your own job is another user's, cgroup
    discovery looks for ``uid_99``, and the header reads ``user nobody``. SW-1.

    Falls back to getpwnam only when Slurm printed no uid at all.
    """
    raw = raw.strip()
    match = re.search(r"\((\d+)\)\s*$", raw)
    name = raw.split("(")[0].split("@")[0].strip()
    if name.lower() in _PLACEHOLDER_NAMES:
        name = ""
    uid = int(match.group(1)) if match else (_resolve_uid(name) if name else None)
    if not name and uid is not None:
        name = _name_for_uid(uid)
    return name, uid


# The fields slurmwatch will not take from `scontrol show job`, because a newline in
# a job name or comment can plant a forged copy of any of them ahead of the real one
# and `_parse_scontrol_field` takes the FIRST match. Every one of these decides
# either the data path or a denominator: a forged NodeList sends the hop at the wrong
# node and makes a local job look remote; NumCPUs/NumNodes are the CPU-percent
# denominators; JobState decides whether slurmwatch will monitor at all. squeue can
# answer all of them in ONE query whose fields are all machine-generated -- no free
# text, so nothing in it can contain the delimiter or a newline. Owner included, per
# the SW-1 feedback.
# ``%i`` leads so a row can be MATCHED to the job it was asked about. Without it the
# function took squeue's first row on trust, which is wrong for exactly one shape and
# badly wrong there: an array task that has ENDED while its siblings run. Slurm prints
# `JobId=<the array base>` for a task with no allocation left (measured on Slurm
# 25.11: `scontrol show job 563321_3` -> `JobId=563321 ArrayTaskId=3
# JobState=CANCELLED`), so the query widened to the whole array and the first live
# sibling answered for it — slurmwatch reported a CANCELLED task as RUNNING, on the
# sibling's node, with the sibling's CPU and node counts as its denominators.
_SQUEUE_FACTS_FORMAT = "%i|%U|%u|%T|%P|%N|%C|%D"


def _squeue_id_for_record(record: str) -> str:
    """How ``squeue -o %i`` spells the job this ``scontrol`` record describes.

    ``<ArrayJobId>_<ArrayTaskId>`` for an array task, "" otherwise (the caller then
    uses the record's own JobId). Read from the record's FIRST LINE only, for the
    SW-1 reason: ``JobName`` is free text on that same line and a newline inside it
    plants forged lines AFTER it, so every field printed before the name — JobId,
    ArrayJobId, ArrayTaskId — is on the one stretch of output no injected line can
    precede.

    A pending array's ``ArrayTaskId`` is a RANGE (``1-9%2``), which is not one job
    and not a ``%i`` any row will equal; only a plain integer task index yields an id.
    """
    first = record.split("\n", 1)[0]
    base = (_parse_scontrol_field(first, "ArrayJobId") or "").strip()
    task = (_parse_scontrol_field(first, "ArrayTaskId") or "").strip()
    if base.isdigit() and task.isdigit():
        return f"{base}_{task}"
    return ""


def _authoritative_job_facts(raw_job_id: str, want_id: str = "") -> dict[str, str]:
    """The forgery-proof view of a job: uid, name, state, partition, nodes, sizes.

    Keyed by field name; empty when squeue can't answer (no Slurm, a purged job, a
    controller hiccup), in which case the caller falls back to the record and its
    documented weaknesses. `raw_job_id` must be the id from the record's FIRST line,
    which is the one part of `scontrol show job` output no injected line can precede.

    ``want_id`` is how *this* job is spelled in ``squeue``'s ``%i`` — for an array
    task, ``<base>_<task>``, which is neither the base nor the task's own numeric
    JobId. It is what the query asks for and what a row must match when the answer
    has more than one.

    A SINGLE row is taken as given, exactly as before: for a plain job, a het
    component and any version that spells ``%i`` its own way there is no sibling to
    confuse it with, so tightening that case would only trade a right answer for the
    record fallback. Several rows means the id was a whole array, and then only an
    exact match will do — a near-miss is a different job.
    """
    query = want_id or raw_job_id
    if not query or _is_mock():
        return {}
    try:
        out = _run_slurm_cmd(["squeue", "-j", query, "-h", "-o", _SQUEUE_FACTS_FORMAT])
    except SlurmCommandError:
        return {}
    keys = ("job_id", "uid", "username", "state", "partition", "nodelist", "cpus", "nodes")
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) != len(keys) or not parts[1].isdigit():
            continue
        rows.append(dict(zip(keys, parts, strict=True)))
    if len(rows) == 1:
        return rows[0]
    for row in rows:
        if row["job_id"] == query:
            return row
    return {}


def _owner_from_record(record: str) -> tuple[str, int | None]:
    """``(username, uid)`` from a ``scontrol show job`` record — best-effort fallback.

    Only reached when ``squeue`` can't answer (see
    :func:`_authoritative_job_facts`, which is the forgery-proof source).
    Positional rules do not work here: `JobName` is
    printed BEFORE `UserId=` and `Comment` AFTER it, both accept a newline, so a
    forged `UserId=root(0) GroupId=root(0)` line can be planted on either side of
    the real one. First-wins loses to a job name; last-wins loses to a comment.

    So this does not guess. Every `UserId=` carrying a numeric uid is collected, and
    the answer is used only when the record AGREES with itself, or when one of the
    candidates is our own uid (an attacker gains nothing by forging the uid of the
    person reading). When they disagree and none is ours, the uid is left ``None``:
    `_job_owner_differs` then falls back to comparing names, which yields the honest
    read-only view rather than a confident wrong owner.
    """
    numeric: list[tuple[str, int]] = []
    any_value: list[str] = []
    for line in record.split("\n"):
        value = _parse_scontrol_field(line, "UserId")
        if value is None:
            continue
        any_value.append(value)
        match = re.search(r"\((\d+)\)\s*$", value.strip())
        if match:
            numeric.append((value, int(match.group(1))))
    if not numeric:
        return _parse_user_id(any_value[0]) if any_value else ("", None)
    uids = {uid for _v, uid in numeric}
    if len(uids) == 1:
        return _parse_user_id(numeric[0][0])
    mine = _own_uid()
    for value, uid in numeric:
        if mine is not None and uid == mine:
            return _parse_user_id(value)
    logger.warning(
        "scontrol reported %d different owners for this job (%s) — a newline in a "
        "job name or comment can forge one, so the owner is being left unresolved.",
        len(uids),
        ", ".join(str(u) for u in sorted(uids)),
    )
    name, _uid = _parse_user_id(numeric[0][0])
    return name, None


def current_username() -> str:
    """The user to ask Slurm about, resolved from the uid before the environment.

    ``$USER``/``$LOGNAME`` are simply absent under cron, systemd units, ``env -i``
    and minimal containers, and stale after ``su otheruser`` (no dash) or
    ``sudo -E``. An empty name is the harmful case: ``squeue -u ""`` returns zero
    rows, so "you have no jobs" is printed while the job is running. getuid()
    can't be spoofed by the environment, so its name wins; the env vars are the
    fallback for a uid the name service can't map, and the uid itself is the last
    resort (``squeue -u`` documents accepting a numeric uid). SW-11.
    """
    uid = _own_uid()
    if uid is not None:
        name = _name_for_uid(uid)
        if name:
            return name
    env_name = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if env_name:
        return env_name
    return str(uid) if uid is not None else ""


def _split_cuda_visible(cuda_visible: str) -> tuple[list[int], list[str]]:
    """Split a CUDA_VISIBLE_DEVICES value into integer ordinals and UUID/MIG tokens."""
    idxs: list[int] = []
    uuids: list[str] = []
    for tok in cuda_visible.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idxs.append(int(tok))
        except ValueError:
            uuids.append(tok)
    return idxs, uuids


_GRES_IDX_RE = re.compile(r"gpu[^(,]*\(IDX:([0-9,\-]+)\)")


def _resolve_gpu_indices(
    record: str, hostname: str, job_pids: list[int]
) -> tuple[list[int], list[str]]:
    """Resolve which node-local GPUs belong to the job.

    Priority: the IDX list from `scontrol show job -d` (node-global, exact),
    then CUDA_VISIBLE_DEVICES read from the job's own processes, then this
    process's environment. Integer ordinals from process environments are a
    last resort: with ConstrainDevices they are renumbered relative to the
    job's device cgroup and may not match node-global indices. UUID/MIG
    tokens are absolute, so they are kept whenever found.
    """
    indices = _parse_gres_idx(record, hostname)

    # Union CUDA_VISIBLE_DEVICES across the job's processes: with per-task GPU
    # binding each rank sees only its own device(s), so keeping just the first
    # PID's value under-reports the node's allocation (B-P13).
    ordinals: list[int] = []
    uuids: list[str] = []
    for pid in job_pids[:8]:
        env = _read_pid_environ(pid)
        cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            continue
        o, u = _split_cuda_visible(cuda_visible)
        for ordinal in o:
            if ordinal not in ordinals:
                ordinals.append(ordinal)
        for uuid in u:
            if uuid not in uuids:
                uuids.append(uuid)

    if not (indices or ordinals or uuids):
        env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if env_gpus:
            ordinals, uuids = _split_cuda_visible(env_gpus)

    if indices:
        return indices, uuids
    return sorted(ordinals), uuids


def _parse_node_detail(record: str, hostname: str) -> tuple[int, int]:
    """(cpus, mem_bytes) for this host from the ``scontrol show job -d`` detail.

    The detail output carries per-node lines such as
    ``Nodes=cn[001-002] CPU_IDs=0-15 Mem=64000 GRES=gpu:a100:2(IDX:0-1)``. The
    CPU_IDs count and Mem are the exact per-node allocation — more precise than
    dividing the job-wide totals by the node count. ``Mem`` is megabytes when
    unsuffixed. Returns (0, 0) when no single line can be attributed to this
    host (same fallback rule as :func:`_parse_gres_idx`)."""
    matching: list[str] = []
    fallback: list[str] = []
    for line in record.split("\n"):
        if "CPU_IDs=" not in line:
            continue
        nodes_str = _parse_scontrol_field(line, "Nodes") or ""
        if _host_in_nodelist(hostname, _parse_nodelist(nodes_str)):
            matching.append(line)
        else:
            fallback.append(line)
    lines = matching or (fallback if len(fallback) == 1 else [])
    if not lines:
        return 0, 0
    line = lines[0]

    cpu_ids = _parse_scontrol_field(line, "CPU_IDs") or ""
    cpus = len(_expand_idx_list(cpu_ids)) if cpu_ids else 0

    mem_str = _parse_scontrol_field(line, "Mem") or ""
    mem_bytes = 0
    if mem_str:
        if mem_str[-1:].isalpha():
            mem_bytes = _parse_mem_to_bytes(mem_str) or 0
        else:
            # An unsuffixed Mem in the -d node detail is in megabytes.
            mem_bytes = _parse_leading_int(mem_str) * 1024**2
    return cpus, mem_bytes


def _parse_gres_idx(record: str, hostname: str) -> list[int]:
    """Extract this node's allocated GPU indices from `scontrol show job -d`.

    The detail output contains per-node lines like
    `Nodes=cn[001-002] CPU_IDs=0-15 Mem=64000 GRES=gpu:a100:2(IDX:0-1)`.
    """
    matching_lines: list[str] = []
    fallback_lines: list[str] = []
    for line in record.split("\n"):
        if "IDX:" not in line or "GRES" not in line:
            continue
        nodes_str = _parse_scontrol_field(line, "Nodes") or ""
        node_names = _parse_nodelist(nodes_str)
        if _host_in_nodelist(hostname, node_names):
            matching_lines.append(line)
        else:
            fallback_lines.append(line)
    # If no line names this host (single-node job or hostname mismatch),
    # only trust the detail when there is exactly one allocation line.
    lines = matching_lines or (fallback_lines if len(fallback_lines) == 1 else [])

    indices: list[int] = []
    for line in lines:
        for idx_list in _GRES_IDX_RE.findall(line):
            indices.extend(_expand_idx_list(idx_list))
    return sorted(set(indices))


def parse_gres_idx_by_node(record: str) -> dict[str, list[int]]:
    """Every node's allocated GPU indices, from one ``scontrol show job -d`` record.

    :func:`_parse_gres_idx` deliberately answers only "which GPUs on *this* host",
    because that is what NVML attachment needs. But the same record already names
    the whole allocation::

        Nodes=beagle3-0015 CPU_IDs=1-2,4-5 Mem=53248 GRES=gpu:2(IDX:0,2)
        Nodes=beagle3-0020 CPU_IDs=2-5     Mem=53248 GRES=gpu:2(IDX:1-2)

    so a multi-node job's full GPU layout costs no extra Slurm call and no ``srun``
    hop — it is readable from a login node. That matters because a monitor step
    cannot read GPU *utilization* at all when the job holds every GPU, making the
    allocation the only GPU fact there is; answering "which GPUs did my job get, on
    every node" without walking the nodes one at a time is the whole point.

    Returns ``{node: sorted indices}``, ``{}`` when the record carries no per-node
    GRES detail (a CPU-only job, or ``scontrol`` without ``-d``). Node names are
    expanded, so ``Nodes=cn[001-002]`` yields an entry per node.
    """
    by_node: dict[str, list[int]] = {}
    for line in record.split("\n"):
        if "IDX:" not in line or "GRES" not in line:
            continue
        nodes_str = _parse_scontrol_field(line, "Nodes") or ""
        indices: list[int] = []
        for idx_list in _GRES_IDX_RE.findall(line):
            indices.extend(_expand_idx_list(idx_list))
        if not indices:
            continue
        for node in _parse_nodelist(nodes_str):
            # A range line (Nodes=cn[001-002]) states ONE index set shared by every
            # node it names, so each gets the same list rather than the union.
            by_node.setdefault(node, []).extend(indices)
    return {node: sorted(set(idx)) for node, idx in by_node.items()}


def _expand_idx_list(idx_list: str) -> list[int]:
    """Expand an IDX range list like '0-1,3' into [0, 1, 3].

    Capped like the NodeList expansion (#audit3-10): this parses untrusted
    ``scontrol show job -d`` text, and a crafted GPU ``IDX:0-2000000000`` (e.g. via
    a hostile JobName that the raw GRES regex doesn't field-shadow) would otherwise
    materialize billions of ints and exhaust memory / hang the monitor. No real
    node has more GPUs than this bound, so a legitimate list never hits it.
    """
    out: list[int] = []
    for rng in idx_list.split(","):
        rng = rng.strip()
        if not rng:
            continue
        if "-" in rng:
            start_str, _, end_str = rng.partition("-")
            with contextlib.suppress(ValueError):
                start, end = int(start_str), int(end_str)
                # Bound the span before materializing range() (a huge end would OOM).
                end = min(end, start + _MAX_GPU_IDX)
                out.extend(range(start, end + 1))
        else:
            with contextlib.suppress(ValueError):
                out.append(int(rng))
        if len(out) >= _MAX_GPU_IDX:
            return out[:_MAX_GPU_IDX]
    return out


def _cgroup_pids(paths: list[Path]) -> list[int]:
    """Union of PIDs from cgroup.procs files anywhere under the given cgroups.

    On cgroup v2 processes live only in leaf cgroups (job_X/step_Y/user/task_Z),
    so every descendant must be visited.
    """
    pids: set[int] = set()
    for base in paths:
        files = [base / "cgroup.procs"]
        with contextlib.suppress(OSError):
            files.extend(base.rglob("cgroup.procs"))
        for procs_file in files:
            try:
                data = procs_file.read_text()
            except OSError:
                continue
            for token in data.split():
                if token.isdigit():
                    pids.add(int(token))
    return sorted(pids)


def _read_pid_environ(pid: int) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        data = Path(f"/proc/{pid}/environ").read_bytes()
        for entry in data.split(b"\x00"):
            if not entry:
                continue
            if b"=" in entry:
                key, _, val = entry.partition(b"=")
                env[key.decode("utf-8", errors="replace")] = val.decode("utf-8", errors="replace")
    except (PermissionError, FileNotFoundError, OSError):
        pass
    return env


def _make_mock_job_context(
    job_id: str,
    step_id: str | None = None,
) -> JobContext:
    hostname = socket.gethostname().split(".")[0]
    # Node 1 must be *this* host. The dashboard serves the node it runs on from
    # the local collector and streams every other node over srun; a mock nodelist
    # of purely fictional names therefore matched no local node, so `--demo`
    # selected an unreachable node[0] and sat on "awaiting telemetry…" forever
    # while the mock collector's frames went nowhere (#27). Keeping the remaining
    # names fictional still exercises the node switcher in the demo. Any filler
    # name that collides with this host's own is dropped (a machine actually
    # called cn-002 would otherwise be listed twice in the switcher), so the demo
    # always shows exactly four distinct nodes.
    filler = [
        n for n in ("cn-002", "cn-003", "cn-004", "cn-005") if short_host(n) != short_host(hostname)
    ]
    resolved_nodes = [hostname, *filler[:3]]
    return JobContext(
        job_id=job_id,
        username="demo",
        partition="gpu-highend",
        nodelist=",".join(resolved_nodes),
        hostname=hostname,
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024**3,
        gpu_count_requested=4,
        gpu_indices=[0, 1, 2, 3],
        # NOT `step_id or "0"`: production leaves this None (the CLI is a job-level
        # monitor and strips a step form, saying so), so fabricating "0" here made the
        # demo payload carry an identity value no real run ever emits — a consumer
        # developing against --demo would build on it and get null in production. The
        # inverse of SW-30, which was demo telemetry the payload didn't mark.
        step_id=step_id,
        uid=1001,
        job_start_time=time.time() - 7200,
        time_limit_seconds=24 * 3600,
        nodelist_resolved=resolved_nodes,
        job_state="RUNNING",
        tres="cpu=16,mem=64G,gres/gpu=4",
        job_name="train-llama-8b",
        # Generic, like every other value in this fixture: a real site's
        # allocation name has no business shipping in --demo (SW-6).
        account="demo-alloc",
        qos="normal",
        command="/home/demo/proj/train.py",
        work_dir="/home/demo/proj/runs/2026-07",
        std_out=f"/home/demo/proj/runs/2026-07/logs/train-{job_id}.out",
        std_err=f"/home/demo/proj/runs/2026-07/logs/train-{job_id}.err",
        submit_time=time.time() - 7500,
    )


def _cgroup_name_matches_job(name: str, base_job_id: str) -> bool:
    """Whether a cgroup directory name belongs to ``base_job_id``.

    Matches ``job_<id>`` only at a numeric boundary, so ``job_123`` no longer
    matches ``job_1234``/``job_12345`` and attaches to the wrong job (B-P11).
    Still tolerates suffixed forms such as ``job_123.scope`` or ``job_123_0``.
    """
    target = f"job_{base_job_id}"
    pos = name.find(target)
    if pos == -1:
        return False
    # Reject a leading alphanumeric run-on ("xjob_123" -> not job 123) as well as a
    # trailing-digit one ("job_1234"): match only at a token boundary on both sides.
    before = name[pos - 1] if pos > 0 else ""
    if before.isalnum():
        return False
    after = name[pos + len(target) :]
    return not after[:1].isdigit()


def _read_self_cgroup() -> str:
    """Contents of ``/proc/self/cgroup`` ('' if unreadable).

    Split out so the parsers below are unit-testable without a real ``/proc`` and
    so the read itself is mockable.
    """
    try:
        return Path("/proc/self/cgroup").read_text(errors="replace")
    except OSError:
        return ""


def _v2_job_dir_from_cgroup_content(content: str) -> Path | None:
    """The job-level cgroup dir named in a cgroup/v2 ``/proc/<pid>/cgroup`` body.

    cgroup v2 has a single unified line ``0::<path>`` giving the process's exact
    cgroup, e.g. ``0::/system.slice/slurmstepd.scope/<jobdir>/step_0/user/task_0``.
    The job-level directory is the component directly under ``slurmstepd.scope`` —
    returned WITHOUT interpreting its name, so it works whether Slurm named it
    ``job_<id>`` (<=25.05 / ``CgroupJobIdPaths=yes``) or an opaque SLUID such as
    ``sEKNKTV3WPV500`` (the Slurm 26.05 default). ``None`` when the process isn't
    under a slurmstepd job cgroup (e.g. on a login node).
    """
    marker = "/slurmstepd.scope/"
    for line in content.splitlines():
        if not line.startswith("0::"):
            continue
        rel = line[3:].strip()
        idx = rel.find(marker)
        if idx == -1:
            return None
        after = rel[idx + len(marker) :]
        job_component = after.split("/", 1)[0]
        if not job_component:
            return None
        job_rel = rel[: idx + len(marker)] + job_component
        return _CGROUP_V2_BASE / job_rel.lstrip("/")
    return None


def _v1_job_dir_from_cgroup_content(content: str, controller: str) -> Path | None:
    """The job-level v1 cgroup for ``controller`` from a ``/proc/<pid>/cgroup`` body.

    v1 lines are ``<hierarchy>:<controllers>:<path>``; the job level is the path
    with any trailing ``step_*``/``task_*`` stripped. Name/layout-agnostic, so it
    also covers a non-standard mountpoint or ``uid_``/``job_`` layout. ``None`` if
    the controller isn't listed or the process isn't in a per-job cgroup.
    """
    for line in content.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3 or controller not in parts[1].split(","):
            continue
        rel = parts[2].strip()
        if not rel or rel == "/":
            return None
        marker = "/step_"
        if marker in rel:
            rel = rel[: rel.index(marker)]
        return _CGROUP_V2_BASE / controller / rel.lstrip("/")
    return None


def _cgroup_belongs_to_job(cgdir: Path, base_job_id: str, max_pids: int = 16) -> bool:
    """Whether any process under ``cgdir`` reports ``SLURM_JOB_ID == base_job_id``.

    An identity check by process environment, so it recognises the job's cgroup
    regardless of the directory's NAME — the key to surviving the Slurm 26.05
    SLUID rename and any non-standard slice. Only the caller's own job has
    readable ``/proc/<pid>/environ`` (same uid), which is exactly the case
    slurmwatch monitors on-node.
    """
    target = _parse_leading_int(base_job_id)
    if target <= 0:
        return False
    for pid in _cgroup_pids([cgdir])[:max_pids]:
        env = _read_pid_environ(pid)
        jid = env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID") or ""
        if jid and _parse_leading_int(jid) == target:
            return True
    return False


def _v2_from_self_cgroup(base_job_id: str) -> Path | None:
    """Locate the job's v2 cgroup from our OWN ``/proc/self/cgroup``.

    Fires whenever slurmwatch runs inside the allocation — the srun hop, an
    ssh-to-node session adopted into the job cgroup (pam_slurm_adm), or a batch
    step. Name-agnostic, so it is the primary fix for Slurm 26.05 SLUID names.
    """
    job_dir = _v2_job_dir_from_cgroup_content(_read_self_cgroup())
    if job_dir is None or not job_dir.exists():
        return None
    # Trust it only if it is really the requested job: our own SLURM_JOB_ID is the
    # fast accept (the hop/batch set it); an ssh-adopted session may carry none of
    # its own, so fall back to confirming via a member PID's environ.
    env_jid = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or ""
    if env_jid and _parse_leading_int(env_jid) == _parse_leading_int(base_job_id):
        return job_dir
    if _cgroup_belongs_to_job(job_dir, base_job_id):
        return job_dir
    return None


def _v2_by_membership(base_job_id: str) -> Path | None:
    """Find the job's v2 cgroup by PROCESS MEMBERSHIP, ignoring the dir name.

    The universal last resort: scans ``slurmstepd.scope`` (then ``system.slice``
    for older layouts) and returns the child whose processes belong to the job —
    handling Slurm 26.05 SLUID directories and custom slices when slurmwatch is
    on the node but not itself inside the cgroup.
    """
    scope = _CGROUP_V2_BASE / "system.slice" / "slurmstepd.scope"
    for parent in (scope, _CGROUP_V2_BASE / "system.slice"):
        try:
            children = sorted(parent.iterdir())
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        for child in children:
            if child.is_dir() and _cgroup_belongs_to_job(child, base_job_id):
                return child
    return None


def _discover_cgroup_paths(
    job_id: str,
    uid: int | None = None,
    step_id: str | None = None,
) -> dict[str, Path | None]:
    result: dict[str, Path | None] = {"v2": None, "v1_mem": None, "v1_cpu": None}

    # Strip the array "_task" AND the heterogeneous-component "+offset" separators
    # so the exact-path candidates below become job_<numeric> (which can exist),
    # not job_12345+0 (which never does).
    base_job_id = re.split(r"[_+]", job_id, maxsplit=1)[0]

    if detect_cgroup_version() == 2:
        v2_candidates = [
            _CGROUP_V2_BASE / "system.slice" / "slurmstepd.scope" / f"job_{base_job_id}",
        ]
        if step_id is not None:
            step_path = (
                _CGROUP_V2_BASE
                / "system.slice"
                / "slurmstepd.scope"
                / f"job_{base_job_id}"
                / f"step_{step_id}"
            )
            v2_candidates.insert(0, step_path)
            v2_candidates.append(
                _CGROUP_V2_BASE / "system.slice" / "slurmstepd.scope" / f"step_{step_id}",
            )

        for path in v2_candidates:
            if path.exists():
                result["v2"] = path
                break

        # Name-agnostic discovery via our own /proc/self/cgroup — works whenever
        # slurmwatch runs inside the allocation (the srun hop, an ssh-adopted
        # session, or a batch step) and, crucially, survives the Slurm 26.05
        # default that names the job cgroup with an opaque SLUID instead of
        # `job_<id>` (see cgroup.conf `CgroupJobIdPaths`).
        if result["v2"] is None:
            result["v2"] = _v2_from_self_cgroup(base_job_id)

        if result["v2"] is None:
            # Fallback scan for a job cgroup whose exact path didn't match a
            # candidate above (e.g. a `.scope`-suffixed or otherwise non-standard
            # name). Slurm's job cgroups live *inside* slurmstepd.scope, so scan
            # there first; older layouts put them directly under system.slice, so
            # scan that too (F3 — the fallback previously only looked in
            # system.slice and never descended into slurmstepd.scope).
            scope = _CGROUP_V2_BASE / "system.slice" / "slurmstepd.scope"
            for parent in (scope, _CGROUP_V2_BASE / "system.slice"):
                try:
                    children = list(parent.iterdir())
                except (PermissionError, FileNotFoundError, NotADirectoryError):
                    continue
                for child in children:
                    if _cgroup_name_matches_job(child.name, base_job_id):
                        result["v2"] = child
                        break
                if result["v2"] is not None:
                    break

        # Universal last resort: identify the job cgroup by PROCESS MEMBERSHIP,
        # ignoring its name entirely — the path that handles Slurm 26.05 SLUID
        # directories (and any custom slice) when slurmwatch isn't itself in the
        # cgroup but is on the node running the job.
        if result["v2"] is None:
            result["v2"] = _v2_by_membership(base_job_id)

    v1_mem_base = _CGROUP_V2_BASE / "memory"
    v1_cpu_base = _CGROUP_V2_BASE / "cpuacct"

    if uid is not None:
        for base, key in [(v1_mem_base, "v1_mem"), (v1_cpu_base, "v1_cpu")]:
            if not base.exists():
                continue
            # The v1 root is a site-configurable PREFIX, not a fixed name: the el8
            # default is a bare `slurm`, while other sites emit `slurm_<nodename>`
            # (…/memory/slurm_midway2-0300/uid_940740146/job_48818838). Only the bare
            # form was tried by exact path, so on a node-suffixed site the job cgroup
            # was found ONLY through the /proc/self/cgroup fallback below — which
            # requires slurmwatch to be running INSIDE that job's cgroup. Watching
            # another of your jobs on the same node therefore found nothing and
            # degraded to sstat for no reason. Bare name first (no directory listing
            # on the common path), then the suffixed forms.
            roots = [base / "slurm"]
            with contextlib.suppress(OSError):
                roots += sorted(d for d in base.glob("slurm_*") if d.is_dir())
            paths_to_check: list[Path] = []
            for root in roots:
                job_dir = root / f"uid_{uid}" / f"job_{base_job_id}"
                if step_id is not None:
                    paths_to_check.append(job_dir / f"step_{step_id}")
                paths_to_check.append(job_dir)
            for path in paths_to_check:
                if path.exists():
                    result[key] = path
                    break

    # Name/layout-agnostic v1 fallback via /proc/self/cgroup (a custom mountpoint
    # or non-standard uid_/job_ layout). Additive — only fills a gap the exact
    # path above missed, and needs no uid. Verify ownership exactly like the v2
    # self path does (A2): this fallback binds the cgroup slurmwatch *itself* runs
    # in, so without a SLURM_JOB_ID / membership check, `sw <other-job>` launched
    # from inside our own allocation would silently report THIS shell's job under
    # the requested job's label. If neither confirms, leave it None and degrade to
    # the remote view rather than serve a different job's numbers.
    env_jid = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or ""
    env_owns = bool(env_jid) and _parse_leading_int(env_jid) == _parse_leading_int(base_job_id)
    for key, controller, base in (
        ("v1_mem", "memory", v1_mem_base),
        ("v1_cpu", "cpuacct", v1_cpu_base),
    ):
        if result[key] is None and base.exists():
            cand = _v1_job_dir_from_cgroup_content(_read_self_cgroup(), controller)
            # Require the candidate to be JOB-SCOPED (a `job_<id>` component). On a
            # node that delegates cpuset/memory/… per job but NOT cpuacct, our own
            # /proc/self/cgroup names a non-job ANCESTOR for that controller —
            # `/system.slice/slurmd.service`, the daemon's node-wide counter. The
            # membership check below would wrongly accept it (every job step rolls up
            # into slurmd.service, so it "contains" our job), and reading that counter
            # adds every co-tenant job's CPU to this job's usage — over-reporting on a
            # shared node, hidden by the [0,cores] clamp (P1). A real per-job v1 cgroup
            # always has a `job_<id>` component; when it doesn't, leave the path None
            # so the reader falls through to the job-scoped per-PID /proc sum.
            job_scoped = cand is not None and any(
                _cgroup_name_matches_job(part, base_job_id) for part in cand.parts
            )
            if (
                cand is not None
                and job_scoped
                and cand.exists()
                and (env_owns or _cgroup_belongs_to_job(cand, base_job_id))
            ):
                result[key] = cand

    if result["v2"] is None and result["v1_mem"] is None and result["v1_cpu"] is None:
        if uid is not None:
            raise CgroupNotFoundError(
                f"No cgroup hierarchy found for job {job_id} (uid={uid}). "
                "This host may not be a Slurm compute node, or the job's cgroups "
                "have been cleaned up. Try running from within the job allocation."
            )
        raise CgroupNotFoundError(
            f"No cgroup hierarchy found for job {job_id}. "
            "Unable to determine UID for path resolution."
        )

    if result["v2"] is not None:
        try:
            _check_cgroup_readable(result["v2"])
        except PermissionError as exc:
            raise CgroupPermissionError(
                f"Cgroup path {result['v2']} exists but is not readable. "
                "Try running slurmwatch from within a Slurm job allocation."
            ) from exc

    return result


def _check_cgroup_readable(path: Path) -> None:
    if not path.is_dir():
        return
    try:
        next(iter(path.iterdir()))
    except PermissionError:
        raise
    except StopIteration:
        pass


def detect_cgroup_version() -> int:
    if (_CGROUP_V2_BASE / "cgroup.controllers").exists():
        return 2
    return 1
