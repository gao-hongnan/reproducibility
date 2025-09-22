from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from reproducibility import get_numpy_rng, seed_all
from reproducibility.system import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    pass

NUMPY_AVAILABLE = is_numpy_available()
TORCH_AVAILABLE = is_torch_available()


class TestActualReproducibility:
    def test_python_random_reproducibility(self) -> None:
        seed_all(seed=42, python=True, numpy=False, pytorch=False)
        results1 = [random.random() for _ in range(10)]
        results2 = [random.randint(0, 100) for _ in range(10)]

        seed_all(seed=42, python=True, numpy=False, pytorch=False)
        results1_repeat = [random.random() for _ in range(10)]
        results2_repeat = [random.randint(0, 100) for _ in range(10)]

        assert results1 == results1_repeat
        assert results2 == results2_repeat

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_numpy_global_reproducibility(self) -> None:
        import numpy as np

        seed_all(seed=123, numpy=True, pytorch=False)
        arr1 = np.random.random((5, 5))  # noqa: NPY002
        arr2 = np.random.randint(0, 100, size=(3, 3))  # noqa: NPY002
        arr3 = np.random.choice([1, 2, 3, 4, 5], size=10)  # noqa: NPY002

        seed_all(seed=123, numpy=True, pytorch=False)
        arr1_repeat = np.random.random((5, 5))  # noqa: NPY002
        arr2_repeat = np.random.randint(0, 100, size=(3, 3))  # noqa: NPY002
        arr3_repeat = np.random.choice([1, 2, 3, 4, 5], size=10)  # noqa: NPY002

        assert np.array_equal(arr1, arr1_repeat)
        assert np.array_equal(arr2, arr2_repeat)
        assert np.array_equal(arr3, arr3_repeat)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_numpy_generator_reproducibility(self) -> None:
        import numpy as np

        seed_all(seed=456, numpy=True, pytorch=False)
        rng = get_numpy_rng()
        assert rng is not None

        arr1 = rng.random((5, 5))
        arr2 = rng.integers(0, 100, size=(3, 3))
        arr3 = rng.choice([1, 2, 3, 4, 5], size=10)

        seed_all(seed=456, numpy=True, pytorch=False)
        rng2 = get_numpy_rng()
        assert rng2 is not None

        arr1_repeat = rng2.random((5, 5))
        arr2_repeat = rng2.integers(0, 100, size=(3, 3))
        arr3_repeat = rng2.choice([1, 2, 3, 4, 5], size=10)

        assert np.array_equal(arr1, arr1_repeat)
        assert np.array_equal(arr2, arr2_repeat)
        assert np.array_equal(arr3, arr3_repeat)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_get_numpy_rng_function(self) -> None:
        import numpy as np

        seed_all(seed=789, numpy=True)
        rng_after = get_numpy_rng()

        assert rng_after is not None
        assert isinstance(rng_after, np.random.Generator)

        values = rng_after.random(10)
        assert len(values) == 10
        assert all(0 <= v <= 1 for v in values)

        if NUMPY_AVAILABLE:
            seed_all(seed=789, numpy=False)
            rng_unchanged = get_numpy_rng()
            assert rng_unchanged is rng_after

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_pytorch_reproducibility(self) -> None:
        import torch

        seed_all(seed=999, pytorch=True, deterministic=True)

        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randint(0, 10, (2, 4))
        tensor3 = torch.rand(5)

        seed_all(seed=999, pytorch=True, deterministic=True)

        tensor1_repeat = torch.randn(3, 3)
        tensor2_repeat = torch.randint(0, 10, (2, 4))
        tensor3_repeat = torch.rand(5)

        assert torch.equal(tensor1, tensor1_repeat)
        assert torch.equal(tensor2, tensor2_repeat)
        assert torch.equal(tensor3, tensor3_repeat)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_pytorch_cuda_reproducibility(self) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        seed_all(seed=555, pytorch=True, deterministic=True)

        cuda_tensor1 = torch.randn(3, 3, device="cuda")
        cuda_tensor2 = torch.randint(0, 10, (2, 4), device="cuda")

        seed_all(seed=555, pytorch=True, deterministic=True)

        cuda_tensor1_repeat = torch.randn(3, 3, device="cuda")
        cuda_tensor2_repeat = torch.randint(0, 10, (2, 4), device="cuda")

        assert torch.equal(cuda_tensor1, cuda_tensor1_repeat)
        assert torch.equal(cuda_tensor2, cuda_tensor2_repeat)

    @pytest.mark.skipif(not (NUMPY_AVAILABLE and TORCH_AVAILABLE), reason="numpy or torch not installed")
    def test_combined_reproducibility(self) -> None:
        import numpy as np
        import torch

        seed_all(seed=1337, python=True, numpy=True, pytorch=True, deterministic=True)

        py_random = [random.random() for _ in range(5)]
        np_global = np.random.random((3, 3))  # noqa: NPY002

        rng = get_numpy_rng()
        assert rng is not None
        np_gen = rng.random((2, 2))

        torch_tensor = torch.randn(4, 4)

        seed_all(seed=1337, python=True, numpy=True, pytorch=True, deterministic=True)

        py_random_repeat = [random.random() for _ in range(5)]
        np_global_repeat = np.random.random((3, 3))  # noqa: NPY002

        rng2 = get_numpy_rng()
        assert rng2 is not None
        np_gen_repeat = rng2.random((2, 2))

        torch_tensor_repeat = torch.randn(4, 4)

        assert py_random == py_random_repeat
        assert np.array_equal(np_global, np_global_repeat)
        assert np.array_equal(np_gen, np_gen_repeat)
        assert torch.equal(torch_tensor, torch_tensor_repeat)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_different_seeds_produce_different_results(self) -> None:
        import numpy as np

        seed_all(seed=100, numpy=True)
        arr1 = np.random.random((5, 5))  # noqa: NPY002

        seed_all(seed=200, numpy=True)
        arr2 = np.random.random((5, 5))  # noqa: NPY002

        assert not np.array_equal(arr1, arr2)

    def test_get_numpy_rng_without_initialization(self) -> None:
        import reproducibility.seed as seed_module

        original_rng = seed_module._numpy_rng
        try:
            seed_module._numpy_rng = None
            rng = get_numpy_rng()
            assert rng is None
        finally:
            seed_module._numpy_rng = original_rng

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_numpy_rng_persistence_across_calls(self) -> None:
        seed_all(seed=333, numpy=True)
        rng1 = get_numpy_rng()
        assert rng1 is not None

        val1 = rng1.random()

        rng2 = get_numpy_rng()
        assert rng2 is rng1

        val2 = rng2.random()
        assert val1 != val2

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
    def test_global_and_generator_independence(self) -> None:
        import numpy as np

        seed_all(seed=444, numpy=True)

        global_vals = [np.random.random() for _ in range(5)]  # noqa: NPY002

        rng = get_numpy_rng()
        assert rng is not None
        gen_vals = [rng.random() for _ in range(5)]

        assert global_vals != gen_vals

        seed_all(seed=444, numpy=True)
        global_vals_repeat = [np.random.random() for _ in range(5)]  # noqa: NPY002
        assert global_vals == global_vals_repeat
