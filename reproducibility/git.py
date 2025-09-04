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
from typing import Any, Literal

from pydantic import BaseModel


class GitResult(BaseModel):
    """Simple result from git command."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.returncode == 0


def run(
    *args: str,
    cwd: Path | None = None,
    timeout: float = 30,
    capture_output: Literal[True] = True,
    text: Literal[True] = True,
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
        stdout=result.stdout.strip() if result.stdout else "",
        stderr=result.stderr.strip() if result.stderr else "",
        returncode=result.returncode,
    )
