from __future__ import annotations

from typing import Final, Literal

type FullSha = str
type ShortSha = str
type ShaStyle = Literal["short", "full"]

type PackageName = str
type PackageVersion = str

UNKNOWN: Final[str] = "unknown"
