from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from typing import Any
from unittest.mock import Mock, patch

import pytest

from reproducibility.git import GitResult, run


class TestGitResult:
    def test_success_true_when_returncode_zero(self) -> None:
        result = GitResult(stdout="output", stderr="", returncode=0)
        assert result.success is True

    def test_success_false_when_returncode_nonzero(self) -> None:
        result = GitResult(stdout="", stderr="error", returncode=1)
        assert result.success is False

    def test_default_values(self) -> None:
        result = GitResult()
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.returncode == 0
        assert result.success is True

    def test_with_values(self) -> None:
        result = GitResult(stdout="commit abc123", stderr="warning: something", returncode=0)
        assert result.stdout == "commit abc123"
        assert result.stderr == "warning: something"
        assert result.returncode == 0


class TestRun:
    @pytest.fixture
    def mock_subprocess_run(self) -> Any:
        with patch("subprocess.run") as mock_run:
            yield mock_run

    def test_run_status_success(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "  test output  \n"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("status", "--porcelain")

        assert isinstance(result, GitResult)
        assert result.stdout == "test output"
        assert result.stderr == ""
        assert result.returncode == 0
        assert result.success is True

        mock_subprocess_run.assert_called_once_with(
            ["git", "status", "--porcelain"],
            cwd=None,
            timeout=30,
            capture_output=True,
            text=True,
        )

    def test_run_with_cwd(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "abc123def\n"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        test_path = Path("/test/repo")
        result = run("rev-parse", "HEAD", cwd=test_path)

        assert result.stdout == "abc123def"
        mock_subprocess_run.assert_called_with(
            ["git", "rev-parse", "HEAD"],
            cwd=test_path,
            timeout=30,
            capture_output=True,
            text=True,
        )

    def test_run_with_custom_timeout(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "output"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("log", "--oneline", timeout=60)

        assert result.success is True
        mock_subprocess_run.assert_called_with(
            ["git", "log", "--oneline"],
            cwd=None,
            timeout=60,
            capture_output=True,
            text=True,
        )

    def test_run_with_error(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = ""
        mock_process.stderr = "fatal: not a git repository\n"
        mock_process.returncode = 128
        mock_subprocess_run.return_value = mock_process

        result = run("status")

        assert result.stdout == ""
        assert result.stderr == "fatal: not a git repository"
        assert result.returncode == 128
        assert result.success is False

    def test_run_strips_whitespace(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "  \n\toutput with spaces\t\n  "
        mock_process.stderr = "\n  error message  \n"
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("test")

        assert result.stdout == "output with spaces"
        assert result.stderr == "error message"

    def test_run_handles_empty_output(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = None
        mock_process.stderr = None
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("test")

        assert result.stdout == ""
        assert result.stderr == ""
        assert result.success is True

    def test_run_with_kwargs(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "output"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        custom_env = {"GIT_DIR": ".git"}
        result = run("status", env=custom_env, encoding="utf-8")

        assert result.success is True
        mock_subprocess_run.assert_called_with(
            ["git", "status"],
            cwd=None,
            timeout=30,
            capture_output=True,
            text=True,
            env=custom_env,
            encoding="utf-8",
        )

    def test_run_multiple_args(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "abc123"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("rev-parse", "--short=7", "HEAD")

        assert result.stdout == "abc123"
        mock_subprocess_run.assert_called_with(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=None,
            timeout=30,
            capture_output=True,
            text=True,
        )

    def test_run_strip_output_false(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "  output with spaces  \n\t"
        mock_process.stderr = "\n  error with spaces  \n"
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("diff", strip_output=False)

        assert result.stdout == "  output with spaces  \n\t"
        assert result.stderr == "\n  error with spaces  \n"
        assert result.success is True

    def test_run_strip_output_false_empty(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = None
        mock_process.stderr = None
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("diff", strip_output=False)

        assert result.stdout == ""
        assert result.stderr == ""
        assert result.success is True

    def test_run_strip_output_true_default(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "  output  \n"
        mock_process.stderr = "  error  \n"
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        # Test default behavior (strip_output=True)
        result = run("status")

        assert result.stdout == "output"
        assert result.stderr == "error"


class TestGitExamples:
    @pytest.fixture
    def mock_subprocess_run(self) -> Any:
        with patch("subprocess.run") as mock_run:
            yield mock_run

    def test_get_commit_hash(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "abc123def456789\n"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("rev-parse", "HEAD")
        commit = result.stdout if result.success else None

        assert commit == "abc123def456789"

    def test_get_branch_name(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = "main\n"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("rev-parse", "--abbrev-ref", "HEAD")
        branch = result.stdout if result.success else None

        assert branch == "main"

    def test_check_dirty_status(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = " M file.txt\n?? new.py\n"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("status", "--porcelain")
        is_dirty = bool(result.stdout) if result.success else False

        assert is_dirty is True

    def test_check_clean_status(self, mock_subprocess_run: Mock) -> None:
        mock_process = Mock(spec=CompletedProcess)
        mock_process.stdout = ""
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_subprocess_run.return_value = mock_process

        result = run("status", "--porcelain")
        is_dirty = bool(result.stdout) if result.success else False

        assert is_dirty is False
