from __future__ import annotations

import os
import random
import warnings
from typing import TYPE_CHECKING

from .models import DeterministicConfig, SeedConfig
from .system import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "seed_all",
    "seed_worker",
    "configure_deterministic_mode",
    "Seeder",
]

_numpy_rng: np.random.Generator | None = None


class Seeder:
    def __init__(self, config: SeedConfig | None = None) -> None:
        self.config = config or SeedConfig()
        self.numpy_available = is_numpy_available()
        self.torch_available = is_torch_available()

    def seed_all(self) -> int:
        global _numpy_rng
        seed = self.config.seed

        os.environ["PYTHONHASHSEED"] = str(seed)

        if self.config.seed_python:
            random.seed(seed)

        if self.config.seed_numpy and self.numpy_available:
            import numpy as np

            _numpy_rng = np.random.default_rng(seed)

        if self.config.seed_torch and self.torch_available:
            import torch

            torch.manual_seed(seed)
            torch.backends.cudnn.benchmark = False

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if self.config.enable_deterministic and self.torch_available:
            self.configure_deterministic_mode()

        return seed

    @staticmethod
    def configure_deterministic_mode(
        config: DeterministicConfig | None = None,
    ) -> None:
        config = config or DeterministicConfig()

        if not is_torch_available():
            warnings.warn("PyTorch not installed, skipping deterministic mode", stacklevel=2)
            return

        import torch

        torch.use_deterministic_algorithms(
            config.use_deterministic_algorithms,
            warn_only=config.warn_only,
        )
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.cudnn_deterministic

        if hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config.allow_fp16_reduction

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", config.cublas_workspace_config)

        if config.use_deterministic_algorithms:
            warnings.warn(
                "Deterministic mode activated. This may impact performance and increase CUDA memory usage.",
                stacklevel=2,
            )

    @staticmethod
    def seed_worker(worker_id: int) -> None:
        _ = worker_id  # NOTE: DataLoader provides worker_id but seed is derived from torch.initial_seed()
        numpy_available = is_numpy_available()
        torch_available = is_torch_available()

        if not torch_available:
            warnings.warn("PyTorch not available for worker seeding", stacklevel=2)
            return

        import torch

        worker_seed = torch.initial_seed() % 2**32

        random.seed(worker_seed)

        if numpy_available:
            import numpy as np

            global _numpy_rng
            _numpy_rng = np.random.default_rng(worker_seed)


def seed_all(
    seed: int = 42,
    seed_torch: bool = True,
    set_torch_deterministic: bool = True,
) -> int:
    config = SeedConfig(
        seed=seed,
        seed_torch=seed_torch,
        enable_deterministic=set_torch_deterministic,
    )
    seeder = Seeder(config)
    return seeder.seed_all()


def configure_deterministic_mode() -> None:
    Seeder.configure_deterministic_mode()


def seed_worker(worker_id: int) -> None:
    Seeder.seed_worker(worker_id)
