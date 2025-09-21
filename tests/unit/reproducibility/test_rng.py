from __future__ import annotations

import pickle
import random
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from reproducibility.models import RNGSaveConfig, RNGStateData
from reproducibility.rng import RNGStateManager, load_rng_state, save_rng_state
from reproducibility.system import is_numpy_available, is_torch_available

TORCH_AVAILABLE = is_torch_available()
NUMPY_AVAILABLE = is_numpy_available()


class TestRNGStateData:
    def test_default_values(self) -> None:
        state = RNGStateData()
        assert state.python_state is None
        assert state.numpy_state is None
        assert state.torch_cpu_state is None
        assert state.torch_cuda_states is None
        assert state.epoch is None

    def test_with_values(self) -> None:
        state = RNGStateData(
            python_state=(3, (1, 2, 3), None),
            epoch=10,
        )
        assert state.python_state == (3, (1, 2, 3), None)
        assert state.epoch == 10


class TestRNGSaveConfig:
    def test_default_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RNGSaveConfig(save_dir=Path(tmpdir))
            assert config.save_dir == Path(tmpdir)
            assert config.filename_prefix == "rng_state"
            assert config.save_format == "pickle"
            assert config.epoch == 0

    def test_custom_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RNGSaveConfig(
                save_dir=Path(tmpdir),
                filename_prefix="custom",
                save_format="pickle",
                epoch=5,
            )
            assert config.filename_prefix == "custom"
            assert config.epoch == 5


class TestRNGStateManager:
    def test_init(self) -> None:
        manager = RNGStateManager()
        assert manager.numpy_available == NUMPY_AVAILABLE
        assert manager.torch_available == TORCH_AVAILABLE

    @patch("reproducibility.system.importlib.util.find_spec")
    def test_numpy_availability(self, mock_find_spec: Mock) -> None:
        from reproducibility.system import is_numpy_available

        mock_find_spec.return_value = None
        assert is_numpy_available() is False
        mock_find_spec.return_value = Mock()
        assert is_numpy_available() is True

    @patch("reproducibility.system.importlib.util.find_spec")
    def test_torch_availability(self, mock_find_spec: Mock) -> None:
        from reproducibility.system import is_torch_available

        mock_find_spec.return_value = None
        assert is_torch_available() is False
        mock_find_spec.return_value = Mock()
        assert is_torch_available() is True

    def test_get_current_state_python(self) -> None:
        manager = RNGStateManager()
        random.seed(42)
        state = manager.get_current_state()
        assert state.python_state is not None
        assert isinstance(state.python_state, tuple)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_get_current_state_numpy(self) -> None:
        import numpy as np

        import reproducibility.rng as rng_module

        manager = RNGStateManager()
        rng_module._numpy_rng = np.random.default_rng(42)
        state = manager.get_current_state()
        assert state.numpy_state is not None
        assert "bit_generator" in state.numpy_state
        assert "state" in state.numpy_state

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_get_current_state_torch(self) -> None:
        import torch

        manager = RNGStateManager()
        torch.manual_seed(42)
        state = manager.get_current_state()
        assert state.torch_cpu_state is not None
        assert isinstance(state.torch_cpu_state, torch.ByteTensor)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @pytest.mark.skipif(
        not TORCH_AVAILABLE
        or not hasattr(__import__("torch").cuda, "is_available")
        or not __import__("torch").cuda.is_available(),
        reason="CUDA not available",
    )
    def test_get_current_state_cuda(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            manager = RNGStateManager()
            state = manager.get_current_state()
            assert state.torch_cuda_states is not None
            assert len(state.torch_cuda_states) == torch.cuda.device_count()

    def test_set_state_python(self) -> None:
        manager = RNGStateManager()
        random.seed(42)
        original_state = random.getstate()

        random.seed(100)
        different_state = random.getstate()

        state_data = RNGStateData(python_state=original_state)
        manager.set_state(state_data)

        restored_state = random.getstate()
        assert restored_state == original_state
        assert restored_state != different_state

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_set_state_numpy(self) -> None:
        import numpy as np

        import reproducibility.rng as rng_module

        manager = RNGStateManager()
        rng_module._numpy_rng = np.random.default_rng(42)
        original_state = manager.get_current_state()

        rng_module._numpy_rng = np.random.default_rng(100)
        rng_module._numpy_rng.random()

        manager.set_state(original_state)
        restored = rng_module._numpy_rng.bit_generator.state

        assert original_state.numpy_state is not None
        assert restored["bit_generator"] == original_state.numpy_state["bit_generator"]
        assert restored["state"] == original_state.numpy_state["state"]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_set_state_torch(self) -> None:
        import torch

        manager = RNGStateManager()
        torch.manual_seed(42)
        original_state = manager.get_current_state()

        torch.manual_seed(100)
        torch.rand(3)

        manager.set_state(original_state)
        restored = torch.get_rng_state()

        assert original_state.torch_cpu_state is not None
        assert torch.equal(restored, original_state.torch_cpu_state)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    @patch("warnings.warn")
    def test_set_state_cuda_device_mismatch(self, mock_warn: Mock) -> None:
        import torch

        manager = RNGStateManager()
        mock_cuda_states = [torch.ByteTensor([1, 2, 3])] * 3

        state_data = RNGStateData(torch_cuda_states=mock_cuda_states)

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            manager.set_state(state_data)
            mock_warn.assert_called_once()
            assert "CUDA device count mismatch" in mock_warn.call_args[0][0]


class TestSaveLoadRNGState:
    def test_save_rng_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RNGSaveConfig(
                save_dir=Path(tmpdir),
                filename_prefix="test",
                epoch=5,
            )

            save_path = save_rng_state(config)

            assert save_path.exists()
            assert "test_epoch_5.pkl" in save_path.name
            assert save_path.parent == Path(tmpdir)

    def test_save_rng_state_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "dir"
            config = RNGSaveConfig(save_dir=nested_dir)

            save_path = save_rng_state(config)

            assert nested_dir.exists()
            assert save_path.exists()

    def test_save_rng_state_unsupported_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RNGSaveConfig(
                save_dir=Path(tmpdir),
                save_format="json",
            )

            with pytest.raises(NotImplementedError, match="Format json not yet supported"):
                save_rng_state(config)

    def test_load_rng_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            random.seed(42)
            config = RNGSaveConfig(save_dir=Path(tmpdir))
            save_path = save_rng_state(config)

            random.seed(100)
            different_state = random.getstate()

            loaded_state = load_rng_state(save_path)

            restored_state = random.getstate()
            assert restored_state != different_state
            assert loaded_state.python_state is not None

    def test_load_rng_state_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="RNG state file not found"):
            load_rng_state("nonexistent_file.pkl")

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RNGStateManager()

            random.seed(42)
            if NUMPY_AVAILABLE:
                import numpy as np

                import reproducibility.rng as rng_module

                rng_module._numpy_rng = np.random.default_rng(42)
            if TORCH_AVAILABLE:
                import torch

                torch.manual_seed(42)

            original_state = manager.get_current_state()
            config = RNGSaveConfig(save_dir=Path(tmpdir), epoch=10)
            save_path = save_rng_state(config)

            random.seed(999)
            if NUMPY_AVAILABLE:
                import numpy as np

                import reproducibility.rng as rng_module

                rng_module._numpy_rng = np.random.default_rng(999)
            if TORCH_AVAILABLE:
                import torch

                torch.manual_seed(999)

            loaded_state = load_rng_state(save_path)

            assert loaded_state.epoch == 10
            assert random.getstate() == original_state.python_state

            if NUMPY_AVAILABLE:
                import numpy as np

                import reproducibility.rng as rng_module

                current_np_state = rng_module._numpy_rng.bit_generator.state
                assert original_state.numpy_state is not None
                assert current_np_state["bit_generator"] == original_state.numpy_state["bit_generator"]
                assert current_np_state["state"] == original_state.numpy_state["state"]

            if TORCH_AVAILABLE:
                import torch

                assert original_state.torch_cpu_state is not None
                assert torch.equal(torch.get_rng_state(), original_state.torch_cpu_state)

    def test_pickle_serialization(self) -> None:
        state = RNGStateData(
            python_state=random.getstate(),
            epoch=5,
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(state.model_dump(), f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                loaded_data = pickle.load(f)

            loaded_state = RNGStateData(**loaded_data)
            assert loaded_state.python_state == state.python_state
            assert loaded_state.epoch == 5
        finally:
            Path(temp_path).unlink()
