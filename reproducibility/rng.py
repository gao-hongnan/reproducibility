from __future__ import annotations

import pickle
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .models import RNGSaveConfig, RNGStateData
from .system import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    import numpy as np

__all__ = ["save_rng_state", "load_rng_state", "RNGStateManager"]

_numpy_rng: np.random.Generator | None = None


class RNGStateManager:
    def __init__(self) -> None:
        self.numpy_available = is_numpy_available()
        self.torch_available = is_torch_available()
        self._ensure_numpy_rng()

    def _ensure_numpy_rng(self) -> None:
        global _numpy_rng
        if self.numpy_available and _numpy_rng is None:
            import numpy as np

            _numpy_rng = np.random.default_rng()

    def _get_numpy_rng(self) -> np.random.Generator | None:
        global _numpy_rng
        self._ensure_numpy_rng()
        return _numpy_rng

    def get_current_state(self) -> RNGStateData:
        state_data = RNGStateData()

        state_data.python_state = random.getstate()

        if self.numpy_available:
            rng = self._get_numpy_rng()
            if rng is not None:
                rng_state = rng.bit_generator.state
                state_data.numpy_state = dict(rng_state)

        if self.torch_available:
            import torch

            state_data.torch_cpu_state = torch.get_rng_state()

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                cuda_states = torch.cuda.get_rng_state_all()
                state_data.torch_cuda_states = cast(list[torch.ByteTensor], cuda_states)

        return state_data

    def set_state(self, state_data: RNGStateData) -> None:
        if state_data.python_state is not None:
            random.setstate(state_data.python_state)

        if self.numpy_available and state_data.numpy_state is not None:
            rng = self._get_numpy_rng()
            if rng is not None:
                rng.bit_generator.state = state_data.numpy_state

        if self.torch_available:
            import torch

            if state_data.torch_cpu_state is not None:
                torch.set_rng_state(state_data.torch_cpu_state)

            if state_data.torch_cuda_states is not None and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                saved_count = len(state_data.torch_cuda_states)

                if device_count != saved_count:
                    warnings.warn(
                        f"CUDA device count mismatch: saved {saved_count}, current {device_count}. "
                        f"Skipping CUDA state restoration.",
                        stacklevel=2,
                    )
                else:
                    torch.cuda.set_rng_state_all(state_data.torch_cuda_states)


def save_rng_state(config: RNGSaveConfig) -> Path:
    manager = RNGStateManager()
    state = manager.get_current_state()
    state.epoch = config.epoch

    save_path = config.save_dir / f"{config.filename_prefix}_epoch_{config.epoch}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if config.save_format == "pickle":
        with open(save_path, "wb") as f:
            pickle.dump(state.model_dump(), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise NotImplementedError(f"Format {config.save_format} not yet supported")

    return save_path


def load_rng_state(path: Path | str) -> RNGStateData:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"RNG state file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    state = RNGStateData(**data)

    manager = RNGStateManager()
    manager.set_state(state)

    return state
