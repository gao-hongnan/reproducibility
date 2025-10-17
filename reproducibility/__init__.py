from __future__ import annotations

from pathlib import Path

from .git import get_commit_sha
from .seed import configure_deterministic_mode, get_numpy_rng, seed_all, seed_worker
from .system import MemoryInfo, SystemInfo
from .version import get_package_version

__all__ = [
    "seed_all",
    "seed_worker",
    "configure_deterministic_mode",
    "get_numpy_rng",
    "SystemInfo",
    "MemoryInfo",
    "get_package_version",
    "get_commit_sha",
]

__version__ = get_package_version(
    "reproducibility",
    fallback_pyproject=Path(__file__).parent.parent / "pyproject.toml",
)
