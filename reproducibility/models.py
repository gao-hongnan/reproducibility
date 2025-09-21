from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SeedConfig(BaseModel):
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    enable_deterministic: bool = Field(default=True)
    seed_numpy: bool = Field(default=True)
    seed_torch: bool = Field(default=True)
    seed_python: bool = Field(default=True)

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        if not 0 <= v <= 2**32 - 1:
            raise ValueError(f"Seed must be between 0 and {2**32 - 1}")
        return v


class RNGStateData(BaseModel):
    python_state: Any | None = None
    numpy_state: dict[str, Any] | None = None
    torch_cpu_state: Any | None = None
    torch_cuda_states: list[Any] | None = None
    epoch: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeterministicConfig(BaseModel):
    use_deterministic_algorithms: bool = Field(default=True)
    warn_only: bool = Field(default=True)
    cudnn_benchmark: bool = Field(default=False)
    cudnn_deterministic: bool = Field(default=True)
    cublas_workspace_config: str = Field(default=":4096:8")
    allow_tf32: bool = Field(default=False)
    allow_fp16_reduction: bool = Field(default=False)


class RNGSaveConfig(BaseModel):
    save_dir: Path
    epoch: int = 0
    filename_prefix: str = Field(default="rng_state")
    save_format: str = Field(default="pickle", pattern="^(pickle|json|safetensors)$")
    compress: bool = Field(default=False)

    @field_validator("save_dir")
    @classmethod
    def validate_save_dir(cls, v: Path) -> Path:
        return Path(v)
