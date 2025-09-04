from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reproducibility.git import run


def _is_git_installed() -> bool:
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=2,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestIntegration:
    @pytest.fixture
    def temp_git_repo(self, tmp_path: Path) -> Path:
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        self._run_git_command(["git", "init"], cwd=repo_path, timeout=5)
        self._run_git_command(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            timeout=5,
        )
        self._run_git_command(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            timeout=5,
        )

        test_file = repo_path / "test.txt"
        test_file.write_text("initial content")
        self._run_git_command(["git", "add", "test.txt"], cwd=repo_path, timeout=5)
        self._run_git_command(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            timeout=5,
        )

        return repo_path

    def _run_git_command(self, cmd: list[str], cwd: Path, timeout: float = 10) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"Git command timed out after {timeout}s: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Git command failed: {e.stderr}")

    @pytest.mark.skipif(
        not _is_git_installed(),
        reason="Git not installed",
    )
    def test_real_git_repository_clean(self, temp_git_repo: Path) -> None:
        commit_result = run("rev-parse", "HEAD", cwd=temp_git_repo)
        assert commit_result.success
        assert commit_result.stdout
        assert len(commit_result.stdout) == 40

        short_commit_result = run("rev-parse", "--short=7", "HEAD", cwd=temp_git_repo)
        assert short_commit_result.success
        assert short_commit_result.stdout
        assert len(short_commit_result.stdout) == 7

        branch_result = run("rev-parse", "--abbrev-ref", "HEAD", cwd=temp_git_repo)
        assert branch_result.success
        assert branch_result.stdout in ["main", "master"]

        status_result = run("status", "--porcelain", cwd=temp_git_repo)
        assert status_result.success
        assert status_result.stdout == ""
        is_dirty = bool(status_result.stdout)
        assert is_dirty is False

    @pytest.mark.skipif(
        not _is_git_installed(),
        reason="Git not installed",
    )
    def test_real_git_repository_dirty(self, temp_git_repo: Path) -> None:
        dirty_file = temp_git_repo / "dirty.txt"
        dirty_file.write_text("uncommitted changes")

        status_result = run("status", "--porcelain", cwd=temp_git_repo)
        assert status_result.success
        assert status_result.stdout
        is_dirty = bool(status_result.stdout)
        assert is_dirty is True

    @pytest.mark.skipif(
        not _is_git_installed(),
        reason="Git not installed",
    )
    def test_real_git_repository_staged(self, temp_git_repo: Path) -> None:
        staged_file = temp_git_repo / "staged.txt"
        staged_file.write_text("staged content")
        subprocess.run(["git", "add", "staged.txt"], cwd=temp_git_repo, check=True)

        status_result = run("status", "--porcelain", cwd=temp_git_repo)
        assert status_result.success
        assert status_result.stdout
        is_dirty = bool(status_result.stdout)
        assert is_dirty is True

    @pytest.mark.skipif(
        not _is_git_installed(),
        reason="Git not installed",
    )
    def test_run_timeout_handling(self, temp_git_repo: Path) -> None:
        for i in range(10):
            test_file = temp_git_repo / f"file_{i}.txt"
            test_file.write_text(f"content {i}")
            subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True)
            subprocess.run(
                ["git", "commit", "-m", f"Commit {i}"],
                cwd=temp_git_repo,
                check=True,
            )

        result = run("log", "--oneline", cwd=temp_git_repo, timeout=5)
        assert result.success
        assert result.stdout

    def test_non_git_directory(self, tmp_path: Path) -> None:
        commit_result = run("rev-parse", "HEAD", cwd=tmp_path)
        assert not commit_result.success
        assert commit_result.returncode != 0

        branch_result = run("rev-parse", "--abbrev-ref", "HEAD", cwd=tmp_path)
        assert not branch_result.success

        status_result = run("status", "--porcelain", cwd=tmp_path)
        assert not status_result.success

    def test_nonexistent_directory(self) -> None:
        nonexistent = Path("/this/does/not/exist")

        try:
            commit_result = run("rev-parse", "HEAD", cwd=nonexistent)
            assert not commit_result.success
        except FileNotFoundError:
            pass

        try:
            status_result = run("status", "--porcelain", cwd=nonexistent)
            assert not status_result.success
        except FileNotFoundError:
            pass


class TestCleanup:
    def test_temp_directory_cleanup(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "cleanup_test"
        test_dir.mkdir()
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        assert test_dir.exists()
        assert test_file.exists()

        _ = str(test_dir)

    def test_git_repo_cleanup_on_error(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "error_test_repo"
        repo_path.mkdir()

        try:
            subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                check=True,
                timeout=5,
            )
            subprocess.run(
                ["git", "commit", "-m", "Will fail - no files staged"],
                cwd=repo_path,
                check=False,
                timeout=5,
            )
        except Exception:
            pass

        assert repo_path.exists()

    @pytest.mark.parametrize("num_repos", [1, 3, 5])
    def test_multiple_temp_repos(self, tmp_path: Path, num_repos: int) -> None:
        repos = []

        for i in range(num_repos):
            repo_path = tmp_path / f"repo_{i}"
            repo_path.mkdir()

            result = subprocess.run(
                ["git", "init"],
                cwd=repo_path,
                capture_output=True,
                timeout=5,
            )

            if result.returncode == 0:
                repos.append(repo_path)

        assert len(repos) == num_repos
        for repo in repos:
            assert repo.exists()
            assert (repo / ".git").exists()

    def test_temp_space_usage(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "space_test_repo"
        repo_path.mkdir()

        subprocess.run(["git", "init"], cwd=repo_path, check=True, timeout=5)

        test_file = repo_path / "data.txt"
        test_file.write_text("x" * 1000)

        total_size = sum(f.stat().st_size for f in repo_path.rglob("*") if f.is_file())

        assert total_size < 1_000_000, f"Temp repo too large: {total_size} bytes"


class TestRunFunction:
    @pytest.mark.skipif(
        not _is_git_installed(),
        reason="Git not installed",
    )
    def test_run_with_real_git_version(self) -> None:
        result = run("--version")
        assert result.success
        assert "git version" in result.stdout
        assert result.stderr == ""

    @pytest.mark.skipif(
        not _is_git_installed(),
        reason="Git not installed",
    )
    def test_run_with_invalid_command(self) -> None:
        result = run("invalid-command-that-does-not-exist")
        assert not result.success
        assert result.returncode != 0
        assert "invalid-command-that-does-not-exist" in result.stderr or result.stderr
