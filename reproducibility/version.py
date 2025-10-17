from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .types import UNKNOWN, PackageName, PackageVersion


def _read_version_from_pyproject(pyproject_path: Path) -> PackageVersion | None:
    if not pyproject_path.exists():
        return None

    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        if isinstance(data.get("project"), dict):
            version_value = data["project"].get("version")
            return version_value if isinstance(version_value, str) else None
        return None
    except (OSError, tomllib.TOMLDecodeError):
        return None


def get_package_version(package_name: PackageName, *, fallback_pyproject: Path | None = None) -> PackageVersion:
    try:
        return version(package_name)
    except PackageNotFoundError:
        if fallback_pyproject is not None and (pyproject_version := _read_version_from_pyproject(fallback_pyproject)):
            return pyproject_version
        return UNKNOWN
