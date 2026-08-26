from __future__ import annotations

# `importlib.metadata` is the single most expensive import in the startup path: it
# pulls in email, zipfile and importlib.resources for what is, on every run but
# `--version`, a string nobody reads. Measured at ~34 ms of the ~130 ms it takes to
# import slurmwatch.cli — a quarter of the tool's cold start spent parsing package
# metadata. So resolve it on FIRST ACCESS instead of at import (PEP 562), which keeps
# `slurmwatch.__version__` and `from ._version import VERSION` working unchanged while
# the paths that never touch it pay nothing.
_cached: str | None = None


def resolve() -> str:
    """The installed distribution's version, resolved once per process."""
    global _cached
    if _cached is None:
        try:
            from importlib.metadata import PackageNotFoundError, version

            _cached = version("slurmwatch")
        except PackageNotFoundError:
            _cached = "0.0.0+unknown"
    return _cached


def __getattr__(name: str) -> str:
    if name in ("VERSION", "__version__"):
        return resolve()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), "VERSION", "__version__"])
