from __future__ import annotations


class SlurmwatchError(Exception):
    """Base exception for all slurmwatch errors."""


class JobNotFoundError(SlurmwatchError):
    """The requested job ID does not exist in the Slurm accounting database."""


class JobNotRunningError(SlurmwatchError):
    """The requested job exists but is not currently in a running state."""


class CgroupAccessError(SlurmwatchError):
    """Generic failure when reading the control-group filesystem."""


class CgroupNotFoundError(CgroupAccessError):
    """No matching cgroup hierarchy could be located for the target job."""


class CgroupPermissionError(CgroupAccessError):
    """The cgroup path exists but the process lacks read permissions."""


class SlurmCommandError(SlurmwatchError):
    """A Slurm CLI binary returned a non-zero exit code."""
