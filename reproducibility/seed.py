from __future__ import annotations

import os
import random
import warnings
from typing import TYPE_CHECKING

from .system import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    import numpy as np


_MIN_SEED_VALUE = 0  # np.iinfo(np.uint32).min
_MAX_SEED_VALUE = 2**32 - 1  # np.iinfo(np.uint32).max


def _raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer(seed: int) -> None:
    if not (_MIN_SEED_VALUE <= seed <= _MAX_SEED_VALUE):
        raise ValueError(f"Seed must be within the range [{_MIN_SEED_VALUE}, {_MAX_SEED_VALUE}], got {seed}")


"""
Global numpy random generator instance.
This is intentionally global to maintain a single RNG state across the module.
NumPy's new random API (numpy>=1.17) recommends using explicit Generator objects
rather than the legacy global random state. We maintain one generator instance
here that can be accessed and modified by multiple functions in this module.
"""
_numpy_rng: np.random.Generator | None = None


def seed_all(
    seed: int = 42,
    python: bool = True,
    numpy: bool = False,
    pytorch: bool = False,
    deterministic: bool = False,
) -> int:
    global _numpy_rng

    _raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if python:
        random.seed(seed)

    if numpy and is_numpy_available():
        import numpy as np

        _numpy_rng = np.random.default_rng(seed)

    if pytorch and is_torch_available():
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if deterministic and is_torch_available():
        configure_deterministic_mode()

    return seed


def configure_deterministic_mode(
    use_deterministic_algorithms: bool = True,
    warn_only: bool = True,
    cudnn_benchmark: bool = False,
    cudnn_deterministic: bool = True,
    cudnn_enabled: bool = True,
    cublas_workspace_config: str = ":4096:8",
    allow_tf32: bool = False,
    allow_fp16_reduction: bool = False,
) -> None:
    if not is_torch_available():
        warnings.warn("PyTorch not installed, skipping deterministic mode", stacklevel=2)
        return

    import torch

    torch.use_deterministic_algorithms(use_deterministic_algorithms, warn_only=warn_only)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.enabled = cudnn_enabled

    if hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_fp16_reduction

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", cublas_workspace_config)

    if use_deterministic_algorithms:
        warnings.warn(
            "Deterministic mode activated. This may impact performance and increase CUDA memory usage.",
            stacklevel=2,
        )


def seed_worker(worker_id: int) -> None:
    _ = worker_id

    if not is_torch_available():
        warnings.warn("PyTorch not available for worker seeding", stacklevel=2)
        return

    import torch

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)

    if is_numpy_available():
        import numpy as np

        global _numpy_rng
        _numpy_rng = np.random.default_rng(worker_seed)
