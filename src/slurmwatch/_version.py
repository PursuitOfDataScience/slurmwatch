from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("slurmwatch")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

VERSION = __version__
