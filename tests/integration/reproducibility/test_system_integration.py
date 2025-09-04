from __future__ import annotations

import importlib.util
import platform

import pytest

from reproducibility.system import MemoryInfo, SystemInfo


def _has_psutil() -> bool:
    return importlib.util.find_spec("psutil") is not None


class TestIntegration:
    @pytest.mark.skipif(
        not _has_psutil(),
        reason="psutil not installed",
    )
    def test_real_memory_info(self) -> None:
        memory = MemoryInfo.current()

        assert memory.total_mb is not None
        assert memory.available_mb is not None
        assert memory.total_mb > 0
        assert memory.available_mb > 0
        assert memory.available_mb <= memory.total_mb

    def test_real_system_info(self) -> None:
        info = SystemInfo()

        assert info.platform
        assert info.platform in ["Darwin", "Linux", "Windows"]

        assert info.platform_release
        assert info.platform_version

        assert info.architecture
        assert info.architecture in ["x86_64", "arm64", "aarch64", "AMD64"]

        assert info.hostname

        if info.cpu_count is not None:
            assert info.cpu_count > 0
            assert info.cpu_count <= 256  # Reasonable upper bound

        if _has_psutil():
            assert info.memory.total_mb is not None
            assert info.memory.available_mb is not None

    def test_system_info_matches_platform(self) -> None:
        info = SystemInfo()

        assert info.platform == platform.system()
        assert info.platform_release == platform.release()
        assert info.platform_version == platform.version()
        assert info.architecture == platform.machine()
        assert info.hostname == platform.node()

    @pytest.mark.skipif(
        not _has_psutil(),
        reason="psutil not installed",
    )
    def test_memory_info_reasonable_values(self) -> None:
        memory = MemoryInfo.current()

        assert memory.total_mb is not None
        assert memory.available_mb is not None
        assert memory.total_mb > 512
        assert 0 < memory.available_mb < memory.total_mb
        assert 512 <= memory.total_mb <= 1048576

    def test_cpu_count_reasonable_values(self) -> None:
        info = SystemInfo()

        if info.cpu_count is not None:
            assert 1 <= info.cpu_count <= 256

    def test_multiple_calls_consistent(self) -> None:
        info1 = SystemInfo()
        info2 = SystemInfo()

        assert info1.platform == info2.platform
        assert info1.platform_release == info2.platform_release
        assert info1.platform_version == info2.platform_version
        assert info1.architecture == info2.architecture
        assert info1.hostname == info2.hostname
        assert info1.cpu_count == info2.cpu_count

    @pytest.mark.skipif(
        not _has_psutil(),
        reason="psutil not installed",
    )
    def test_memory_changes_over_time(self) -> None:
        memory1 = MemoryInfo.current()
        memory2 = MemoryInfo.current()

        assert memory1.total_mb == memory2.total_mb
        if memory1.available_mb and memory2.available_mb:
            diff = abs(memory1.available_mb - memory2.available_mb)
            assert diff < 1024
