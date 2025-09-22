"""Utilities for seeding random number generators to ensure reproducibility.

Notes
-----
Reproducibility in deep learning ensures that experiments can be repeated with
identical results, critical for verifying research findings and deploying
reliable models. This module provides utilities to seed various random number
generators (Python's built-in random, NumPy, and PyTorch) and configure
deterministic behavior in PyTorch.

Distributed training introduces complexity because it involves multiple
computation units which may not synchronize their random states perfectly.
If training is paused and resumed, ensuring each unit starts with the correct
seed to reproduce the exact computational path becomes challenging.

For more sophisticated examples of handling reproducibility in distributed
environments, see libraries like Composer, where the whole library's core is
built around training deep neural nets in any environment (distributed or not)
with reproducibility in mind.

References
----------
.. [1] PyTorch Reproducibility Guide
   https://pytorch.org/docs/stable/notes/randomness.html
.. [2] PyTorch Deterministic Algorithms
   https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
.. [3] CUBLAS Reproducibility
   https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
.. [4] PyTorch DataLoader Workers
   https://pytorch.org/docs/stable/notes/randomness.html#dataloader
.. [5] MosaicML Composer
   https://github.com/mosaicml/composer/blob/dev/composer/utils/reproducibility.py
"""

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


def get_numpy_rng() -> np.random.Generator | None:
    """Get the global numpy random number generator instance.

    Returns
    -------
    np.random.Generator | None
        The global numpy random generator instance if initialized,
        None otherwise.
    """
    return _numpy_rng


def seed_all(
    seed: int = 42,
    python: bool = True,
    seed_numpy: bool = True,
    seed_torch: bool = True,
    set_torch_deterministic: bool = False,
) -> int:
    """Seed all relevant random number generators to ensure reproducibility.

    Seeds multiple random number generators including Python's built-in random,
    NumPy, and PyTorch. Also sets the PYTHONHASHSEED environment variable for
    hash reproducibility. Optionally configures PyTorch for deterministic
    behavior.

    Parameters
    ----------
    seed : int, default=42
        The seed value to use. Must be within the range [0, 2^32-1]
        (valid unsigned 32-bit integer range).
    python : bool, default=True
        Whether to seed Python's built-in random module.
    seed_numpy : bool, default=True
        Whether to seed NumPy's random number generators. Seeds both
        the legacy global state and creates a new Generator instance.
    seed_torch : bool, default=True
        Whether to seed PyTorch's random number generators on both
        CPU and CUDA devices. Also disables cudnn.benchmark.
    set_torch_deterministic : bool, default=False
        Whether to configure PyTorch for fully deterministic behavior.
        This may impact performance and increase memory usage.

    Returns
    -------
    int
        The seed value that was used.

    Raises
    ------
    ValueError
        If seed is not within the valid range [0, 2^32-1].

    Notes
    -----
    The function sets the following:

    - ``PYTHONHASHSEED`` environment variable to ensure hash reproducibility
    - Python's built-in ``random`` module seed
    - NumPy's legacy global random state via ``np.random.seed()``
    - NumPy's new random generator via ``np.random.default_rng()``
    - PyTorch CPU seed via ``torch.manual_seed()``
    - PyTorch CUDA seeds via ``torch.cuda.manual_seed_all()``
    - Disables cudnn.benchmark for consistent convolution algorithms

    When ``deterministic=True``, additional settings are configured to ensure
    fully reproducible results at the cost of performance.

    References
    ----------
    .. [1] PyTorch Reproducibility
       https://pytorch.org/docs/stable/notes/randomness.html

    Examples
    --------
    >>> seed = seed_all(42)
    >>> # All random number generators are now seeded with 42

    >>> # For fully deterministic PyTorch operations
    >>> seed = seed_all(42, set_torch_deterministic=True)
    """
    global _numpy_rng

    _raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if python:
        random.seed(seed)

    if seed_numpy and is_numpy_available():
        import numpy as np

        np.random.seed(seed)  # noqa: NPY002
        _numpy_rng = np.random.default_rng(seed)

    if seed_torch and is_torch_available():
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if set_torch_deterministic and is_torch_available():
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
    """Configure PyTorch for deterministic behavior.

    Activates deterministic mode in PyTorch and CUDA to ensure reproducible
    results. This typically comes at the cost of performance and may increase
    CUDA memory usage.

    Parameters
    ----------
    use_deterministic_algorithms : bool, default=True
        Whether to use deterministic algorithms in PyTorch operations.
    warn_only : bool, default=True
        If True, operations that don't have deterministic implementations
        will produce warnings instead of errors.
    cudnn_benchmark : bool, default=False
        Whether to enable cudnn auto-tuner. Should be False for reproducibility.
    cudnn_deterministic : bool, default=True
        Whether to use deterministic convolution algorithms.
    cudnn_enabled : bool, default=True
        Whether cudnn is enabled. Can be disabled for debugging.
    cublas_workspace_config : str, default=":4096:8"
        CUBLAS workspace configuration for deterministic algorithms.
        Format is ":size:count" where size is in bytes.
    allow_tf32 : bool, default=False
        Whether to allow TF32 tensor cores on Ampere GPUs.
        Should be False for exact reproducibility.
    allow_fp16_reduction : bool, default=False
        Whether to allow FP16 reduction in GEMM operations.
        Should be False for exact reproducibility.

    Notes
    -----
    This function configures multiple PyTorch and CUDA settings:

    - ``torch.use_deterministic_algorithms``: Forces deterministic algorithms
    - ``torch.backends.cudnn.benchmark``: Disables cudnn auto-tuner
    - ``torch.backends.cudnn.deterministic``: Uses deterministic convolutions
    - ``torch.backends.cuda.matmul.allow_tf32``: Controls TF32 usage
    - ``CUBLAS_WORKSPACE_CONFIG``: Sets workspace for deterministic CUBLAS

    Enabling deterministic mode may significantly impact performance,
    particularly for convolution operations, and may increase memory usage.

    References
    ----------
    .. [1] PyTorch Deterministic Algorithms
       https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    .. [2] CUBLAS Reproducibility
       https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

    Warnings
    --------
    Deterministic mode is not available for all PyTorch operations.
    Some operations may still be non-deterministic even with these settings.

    Examples
    --------
    >>> configure_deterministic_mode()
    >>> # PyTorch is now configured for deterministic behavior

    >>> # Allow warnings but don't error on non-deterministic ops
    >>> configure_deterministic_mode(warn_only=True)
    """
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
    """Initialize random seeds for a DataLoader worker.

    This function should be used as the ``worker_init_fn`` parameter in
    PyTorch's DataLoader to ensure that each worker process has a unique
    but reproducible random state. This prevents workers from generating
    identical random sequences.

    Parameters
    ----------
    worker_id : int
        The unique identifier for the DataLoader worker process.
        This is automatically provided by PyTorch's DataLoader.

    Notes
    -----
    Each worker process needs its own random state to:

    - Ensure different workers generate different random augmentations
    - Maintain reproducibility when using the same seed
    - Prevent data loading bias from identical random sequences

    The function uses PyTorch's ``initial_seed()`` which is set based on
    the base seed and worker ID to generate a unique seed for each worker.

    References
    ----------
    .. [1] PyTorch DataLoader Random Seed
       https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    Examples
    --------
    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # Create a generator with a fixed seed
    >>> g = torch.Generator()
    >>> g.manual_seed(42)
    >>>
    >>> # Use seed_worker to ensure reproducible data loading
    >>> dataloader = DataLoader(
    ...     dataset,
    ...     batch_size=32,
    ...     num_workers=4,
    ...     worker_init_fn=seed_worker,
    ...     generator=g,
    ... )
    """
    _ = worker_id

    if not is_torch_available():
        warnings.warn("PyTorch not available for worker seeding", stacklevel=2)
        return

    import torch

    worker_seed = torch.initial_seed() % (_MAX_SEED_VALUE + 1)

    random.seed(worker_seed)

    if is_numpy_available():
        import numpy as np

        np.random.seed(worker_seed)  # noqa: NPY002
        global _numpy_rng
        _numpy_rng = np.random.default_rng(worker_seed)
