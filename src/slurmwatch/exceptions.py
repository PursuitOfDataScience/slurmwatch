from __future__ import annotations


class SlurmwatchError(Exception):
    """Base exception for all slurmwatch errors."""


class JobNotFoundError(SlurmwatchError):
    """The requested job ID does not exist in the Slurm accounting database."""


class JobNotRunningError(SlurmwatchError):
    """The requested job exists but is not currently in a running state.

    ``known`` carries the facts the raiser had ALREADY parsed when it decided the
    job was not runnable — above all the STATE. Without it the machine-readable
    row for a finished job came out with every field ``null`` while the prose
    beside it read "Job 562683 is in state 'CANCELLED'": a poller watching a job
    through its lifecycle saw ``state: "RUNNING"`` and then ``state: null`` at
    exactly the moment the terminal state became the one fact worth having. Only
    keys the facts schema already defines are merged, so a raiser cannot invent a
    column by passing one.
    """

    def __init__(self, message: str, known: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.known: dict[str, object] = dict(known or {})


class JobNotPendingError(SlurmwatchError):
    """The requested job exists but is not currently pending (queued)."""


class CgroupAccessError(SlurmwatchError):
    """Generic failure when reading the control-group filesystem."""


class CgroupNotFoundError(CgroupAccessError):
    """No matching cgroup hierarchy could be located for the target job."""


class CgroupPermissionError(CgroupAccessError):
    """The cgroup path exists but the process lacks read permissions."""


class SlurmCommandError(SlurmwatchError):
    """A Slurm CLI binary returned a non-zero exit code.

    ``kind`` classifies WHY, because the three causes call for opposite actions and a
    machine consumer should not have to read prose to tell them apart:

    * ``"unavailable"`` — Slurm's client tools aren't here (a non-Slurm cluster, or a
      PATH without them). Permanent, environmental.
    * ``"unsupported"`` — we asked for a field this Slurm version doesn't have.
      Permanent, and OUR bug, not the site's.
    * ``"transient"`` (the default) — the controller was busy, unreachable, or slow.
      Retrying is the right advice for this one only.
    """

    def __init__(self, *args: object, kind: str = "transient") -> None:
        super().__init__(*args)
        self.kind = kind
