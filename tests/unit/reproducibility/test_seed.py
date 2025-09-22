from __future__ import annotations

import os
import random
from unittest.mock import Mock, patch

import pytest

from reproducibility.seed import configure_deterministic_mode, seed_all, seed_worker
from reproducibility.system import is_numpy_available, is_torch_available

TORCH_AVAILABLE = is_torch_available()
NUMPY_AVAILABLE = is_numpy_available()


class TestModuleImport:
    def test_module_imports_without_error(self) -> None:
        import reproducibility
        import reproducibility.seed

        assert hasattr(reproducibility, "seed_all")
        assert hasattr(reproducibility, "seed_worker")
        assert hasattr(reproducibility, "configure_deterministic_mode")


class TestSeedAll:
    def test_seed_all_sets_pythonhashseed(self) -> None:
        result = seed_all(seed=123, deterministic=False)
        assert os.environ["PYTHONHASHSEED"] == "123"
        assert result == 123

    def test_seed_all_python_seeding(self) -> None:
        initial_state = random.getstate()
        seed_all(seed=456, python=True, deterministic=False)
        new_state = random.getstate()
        assert initial_state != new_state

    def test_seed_all_skip_python_seeding(self) -> None:
        random.seed()
        initial_state = random.getstate()
        seed_all(seed=456, python=False, deterministic=False)
        new_state = random.getstate()
        assert initial_state == new_state

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_seed_all_numpy_seeding(self) -> None:
        import reproducibility.seed as seed_module

        seed_all(seed=789, numpy=True, deterministic=False)
        if seed_module._numpy_rng is not None:
            state1 = seed_module._numpy_rng.bit_generator.state
            seed_module._numpy_rng.random()
            state2 = seed_module._numpy_rng.bit_generator.state
            assert state1["state"] != state2["state"]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_seed_all_torch_seeding(self) -> None:
        import torch

        seed_all(seed=111, pytorch=True, deterministic=False)
        tensor1 = torch.rand(3)
        tensor2 = torch.rand(3)
        assert not torch.equal(tensor1, tensor2)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_seed_all_deterministic_mode(self) -> None:
        with patch("reproducibility.seed.configure_deterministic_mode") as mock_configure:
            seed_all(seed=222, deterministic=True)
            mock_configure.assert_called_once()

    @patch("reproducibility.seed.is_numpy_available", return_value=False)
    @patch("reproducibility.seed.is_torch_available", return_value=False)
    def test_seed_all_without_libraries(self, _mock_torch: Mock, _mock_numpy: Mock) -> None:
        result = seed_all()
        assert result == 42

    def test_seed_validation(self) -> None:
        with pytest.raises(ValueError, match=r"Seed must be within the range \[0, 4294967295\]"):
            seed_all(seed=-1)
        with pytest.raises(ValueError, match=r"Seed must be within the range \[0, 4294967295\]"):
            seed_all(seed=2**32)


class TestConfigureDeterministicMode:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch("warnings.warn")
    @patch("reproducibility.seed.is_torch_available", return_value=True)
    def test_with_torch_module(self, _mock_is_available: Mock, mock_warn: Mock) -> None:
        with patch.dict("sys.modules", {"torch": Mock()}):
            import sys

            mock_torch = sys.modules["torch"]
            mock_torch.backends.cudnn = Mock()
            mock_torch.backends.cuda.matmul = Mock()
            mock_torch.use_deterministic_algorithms = Mock()  # type: ignore[attr-defined]

            configure_deterministic_mode()
            mock_torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=True)
            assert mock_torch.backends.cudnn.benchmark is False
            assert mock_torch.backends.cudnn.deterministic is True
            assert mock_torch.backends.cudnn.enabled is True
            mock_warn.assert_called_once()

    @patch("reproducibility.seed.is_torch_available", return_value=False)
    @patch("warnings.warn")
    def test_without_torch_installed(self, mock_warn: Mock, _mock_is_available: Mock) -> None:
        configure_deterministic_mode()
        mock_warn.assert_called_once()
        assert "PyTorch not installed" in mock_warn.call_args[0][0]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch("reproducibility.seed.is_torch_available", return_value=True)
    def test_with_custom_params(self, _mock_is_available: Mock) -> None:
        with patch.dict("sys.modules", {"torch": Mock()}):
            import sys

            mock_torch = sys.modules["torch"]
            mock_torch.backends.cudnn = Mock()
            mock_torch.backends.cuda.matmul = Mock()
            mock_torch.use_deterministic_algorithms = Mock()  # type: ignore[attr-defined]

            configure_deterministic_mode(
                use_deterministic_algorithms=False,
                warn_only=True,
                cudnn_benchmark=True,
                cudnn_deterministic=False,
                cudnn_enabled=False,
            )
            mock_torch.use_deterministic_algorithms.assert_called_once_with(False, warn_only=True)
            assert mock_torch.backends.cudnn.benchmark is True
            assert mock_torch.backends.cudnn.deterministic is False
            assert mock_torch.backends.cudnn.enabled is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch("warnings.warn")
    @patch("reproducibility.seed.is_torch_available", return_value=True)
    def test_cudnn_enabled_default_true(self, _mock_is_available: Mock, _mock_warn: Mock) -> None:
        with patch.dict("sys.modules", {"torch": Mock()}):
            import sys

            mock_torch = sys.modules["torch"]
            mock_torch.backends.cudnn = Mock()
            mock_torch.backends.cuda.matmul = Mock()
            mock_torch.use_deterministic_algorithms = Mock()  # type: ignore[attr-defined]

            configure_deterministic_mode()

            assert mock_torch.backends.cudnn.enabled is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch("warnings.warn")
    @patch("reproducibility.seed.is_torch_available", return_value=True)
    def test_cudnn_enabled_configurable(self, _mock_is_available: Mock, _mock_warn: Mock) -> None:
        with patch.dict("sys.modules", {"torch": Mock()}):
            import sys

            mock_torch = sys.modules["torch"]
            mock_torch.backends.cudnn = Mock()
            mock_torch.backends.cuda.matmul = Mock()
            mock_torch.use_deterministic_algorithms = Mock()  # type: ignore[attr-defined]

            configure_deterministic_mode(cudnn_enabled=False)

            assert mock_torch.backends.cudnn.enabled is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch("warnings.warn")
    @patch("reproducibility.seed.is_torch_available", return_value=True)
    def test_deterministic_warning(self, _mock_is_available: Mock, mock_warn: Mock) -> None:
        with patch.dict("sys.modules", {"torch": Mock()}):
            import sys

            mock_torch = sys.modules["torch"]
            mock_torch.backends.cudnn = Mock()
            mock_torch.backends.cuda.matmul = Mock()
            mock_torch.use_deterministic_algorithms = Mock()  # type: ignore[attr-defined]

            configure_deterministic_mode(use_deterministic_algorithms=True)
        assert any("Deterministic mode activated" in str(call) for call in mock_warn.call_args_list)


class TestSeedWorker:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_with_torch(self) -> None:
        import torch

        with patch.object(torch, "initial_seed", return_value=12345678):
            initial_random_state = random.getstate()
            seed_worker(0)
            new_random_state = random.getstate()
            assert initial_random_state != new_random_state

    @patch("reproducibility.seed.is_torch_available", return_value=False)
    @patch("warnings.warn")
    def test_without_torch(self, mock_warn: Mock, _mock_is_available: Mock) -> None:
        seed_worker(0)
        mock_warn.assert_called_once()
        assert "PyTorch not available for worker seeding" in mock_warn.call_args[0][0]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_worker_seed_determinism(self) -> None:
        import torch

        with patch.object(torch, "initial_seed", return_value=98765432):
            initial_python_state = random.getstate()

            seed_worker(0)
            state1 = random.getstate()
            values1 = [random.random() for _ in range(5)]

            random.setstate(initial_python_state)
            seed_worker(0)
            state2 = random.getstate()
            values2 = [random.random() for _ in range(5)]

            assert state1 == state2
            assert values1 == values2
