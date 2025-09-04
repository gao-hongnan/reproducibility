from __future__ import annotations

from unittest.mock import Mock, patch

from reproducibility.system import MemoryInfo, SystemInfo, fallback


class TestFallback:
    def test_fallback_returns_result(self) -> None:
        func = Mock(return_value="value")
        result = fallback(func)
        assert result == "value"
        func.assert_called_once()

    def test_fallback_returns_default_on_exception(self) -> None:
        func = Mock(side_effect=Exception("error"))
        result = fallback(func, "default")
        assert result == "default"

    def test_fallback_returns_default_on_none(self) -> None:
        func = Mock(return_value=None)
        result = fallback(func, "default")
        assert result == "default"

    def test_fallback_returns_none_default(self) -> None:
        func = Mock(side_effect=Exception())
        result = fallback(func)
        assert result is None

    def test_fallback_returns_empty_string(self) -> None:
        func = Mock(return_value="")
        result = fallback(func, "default")
        assert result == "default"


class TestMemoryInfo:
    def test_default_values(self) -> None:
        memory = MemoryInfo()
        assert memory.total_mb is None
        assert memory.available_mb is None

    def test_with_values(self) -> None:
        memory = MemoryInfo(total_mb=16384, available_mb=8192)
        assert memory.total_mb == 16384
        assert memory.available_mb == 8192

    @patch("psutil.virtual_memory")
    def test_current_success(self, mock_vm: Mock) -> None:
        mock_vm.return_value = Mock(
            total=17179869184,
            available=8589934592,
        )

        memory = MemoryInfo.current()
        assert memory.total_mb == 16384
        assert memory.available_mb == 8192

    @patch("psutil.virtual_memory")
    def test_current_import_error(self, mock_vm: Mock) -> None:
        mock_vm.side_effect = ImportError()

        memory = MemoryInfo.current()
        assert memory.total_mb is None
        assert memory.available_mb is None

    @patch("psutil.virtual_memory")
    def test_current_attribute_error(self, mock_vm: Mock) -> None:
        mock_vm.side_effect = AttributeError()

        memory = MemoryInfo.current()
        assert memory.total_mb is None
        assert memory.available_mb is None


class TestSystemInfo:
    @patch("psutil.virtual_memory")
    @patch("os.cpu_count")
    @patch("platform.node")
    @patch("platform.machine")
    @patch("platform.version")
    @patch("platform.release")
    @patch("platform.system")
    def test_default_factory_success(
        self,
        mock_system: Mock,
        mock_release: Mock,
        mock_version: Mock,
        mock_machine: Mock,
        mock_node: Mock,
        mock_cpu: Mock,
        mock_vm: Mock,
    ) -> None:
        mock_system.return_value = "Darwin"
        mock_release.return_value = "23.0.0"
        mock_version.return_value = "Darwin Kernel Version 23.0.0"
        mock_machine.return_value = "arm64"
        mock_node.return_value = "MacBook.local"
        mock_cpu.return_value = 8
        mock_vm.return_value = Mock(
            total=17179869184,
            available=8589934592,
        )

        info = SystemInfo()

        assert info.platform == "Darwin"
        assert info.platform_release == "23.0.0"
        assert info.platform_version == "Darwin Kernel Version 23.0.0"
        assert info.architecture == "arm64"
        assert info.hostname == "MacBook.local"
        assert info.cpu_count == 8
        assert info.memory.total_mb == 16384
        assert info.memory.available_mb == 8192

    @patch("psutil.virtual_memory")
    @patch("os.cpu_count")
    @patch("platform.node")
    @patch("platform.machine")
    @patch("platform.version")
    @patch("platform.release")
    @patch("platform.system")
    def test_default_factory_failures(
        self,
        mock_system: Mock,
        mock_release: Mock,
        mock_version: Mock,
        mock_machine: Mock,
        mock_node: Mock,
        mock_cpu: Mock,
        mock_vm: Mock,
    ) -> None:
        mock_system.side_effect = Exception()
        mock_release.side_effect = Exception()
        mock_version.side_effect = Exception()
        mock_machine.side_effect = Exception()
        mock_node.side_effect = Exception()
        mock_cpu.side_effect = Exception()
        mock_vm.side_effect = ImportError()

        info = SystemInfo()

        assert info.platform == ""
        assert info.platform_release == ""
        assert info.platform_version == ""
        assert info.architecture == ""
        assert info.hostname == ""
        assert info.cpu_count is None
        assert info.memory.total_mb is None
        assert info.memory.available_mb is None

    def test_manual_initialization(self) -> None:
        memory = MemoryInfo(total_mb=32768, available_mb=16384)
        info = SystemInfo(
            platform="Linux",
            platform_release="5.15.0",
            platform_version="Ubuntu 22.04",
            architecture="x86_64",
            hostname="server.local",
            cpu_count=16,
            memory=memory,
        )

        assert info.platform == "Linux"
        assert info.platform_release == "5.15.0"
        assert info.platform_version == "Ubuntu 22.04"
        assert info.architecture == "x86_64"
        assert info.hostname == "server.local"
        assert info.cpu_count == 16
        assert info.memory.total_mb == 32768
        assert info.memory.available_mb == 16384
