"""Simple wrapper for git command execution.

```python
from pydantic import BaseModel

class GitInfo(BaseModel):
    commit: str | None = None
    branch: str | None = None
    is_dirty: bool = False

    @property
    def short_commit(self) -> str | None:
        return self.commit[:7] if self.commit else None

    @classmethod
    def current(cls: type[GitInfo], cwd: Path | None = None) -> GitInfo:
        commit_result = run("rev-parse", "HEAD", cwd=cwd)
        branch_result = run("branch", "--show-current", cwd=cwd)
        status_result = run("status", "--porcelain", cwd=cwd)

        return cls(
            commit=commit_result.stdout if commit_result.success else None,
            branch=branch_result.stdout if branch_result.success else None,
            is_dirty=bool(status_result.stdout) if status_result.success else False,
        )
```
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Literal, overload

from pydantic import BaseModel

from .types import UNKNOWN, FullSha, ShaStyle, ShortSha


def _strip_if_needed(text: str, strip_output: bool) -> str:
    return text.strip() if strip_output else text


class GitResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0

    @property
    def success(self) -> bool:
        return self.returncode == 0


# TODO: add a whitelist = {"READONLY", ...} to whitelist some git commands.
def run(
    *args: str,
    cwd: Path | None = None,
    timeout: float = 30,
    capture_output: Literal[True] = True,
    text: Literal[True] = True,
    strip_output: bool = True,
    **kwargs: Any,
) -> GitResult:
    """Run git command and return result.

    Parameters
    ----------
    *args : str
        Git command arguments (e.g., "status", "--porcelain")
    cwd : Path | None, optional
        Working directory for the command
    timeout : float, optional
        Command timeout in seconds, by default 30
    strip_output : bool, optional
        Whether to strip whitespace from output, by default True.
        Set to False to preserve exact git output (useful for diffs).
    **kwargs : Any
        Additional keyword arguments passed to subprocess.run()
        (e.g., env, encoding, errors). Note that overriding
        capture_output or text may break GitResult expectations.

    Returns
    -------
    GitResult
        Result object with stdout, stderr, and returncode

    Examples
    --------
    >>> result = run("status", "--porcelain")
    >>> if result.success:
    ...     print(result.stdout)

    >>> result = run("rev-parse", "HEAD")
    >>> commit = result.stdout if result.success else None

    >>> # Preserve whitespace for diffs
    >>> result = run("diff", strip_output=False)

    >>> # With custom environment
    >>> result = run("status", env={"GIT_DIR": ".git"})
    """
    cmd: list[str] = ["git", *args]
    result: CompletedProcess[str] = subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        capture_output=capture_output,
        text=text,
        **kwargs,
    )
    return GitResult(
        stdout=_strip_if_needed(result.stdout or "", strip_output),
        stderr=_strip_if_needed(result.stderr or "", strip_output),
        returncode=result.returncode,
    )


@overload
def get_commit_sha(style: Literal["full"], cwd: Path | None = None) -> FullSha: ...


@overload
def get_commit_sha(style: Literal["short"] = ..., cwd: Path | None = None) -> ShortSha: ...


def get_commit_sha(style: ShaStyle = "short", cwd: Path | None = None) -> str:
    result = run("rev-parse", "HEAD", cwd=cwd) if style == "full" else run("rev-parse", "--short", "HEAD", cwd=cwd)
    return result.stdout if result.success else UNKNOWN
