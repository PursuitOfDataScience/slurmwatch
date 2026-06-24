import importlib.metadata

try:
    __version__ = importlib.metadata.version("slurmwatch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

VERSION = __version__
