from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reproducibility import seed_all, seed_worker
from reproducibility.system import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    from typing import Any

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

TORCH_AVAILABLE = is_torch_available()
NUMPY_AVAILABLE = is_numpy_available()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestTrainingReproducibility:
    @staticmethod
    def create_simple_network() -> nn.Module:
        import torch.nn as nn

        class SimpleNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(20, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        return SimpleNet()

    @staticmethod
    def create_synthetic_dataset(size: int = 100) -> Dataset[Any]:
        import torch
        from torch.utils.data import TensorDataset

        torch.manual_seed(42)
        X = torch.randn(size, 10)
        y = torch.randn(size, 1)
        return TensorDataset(X, y)

    def train_one_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader[Any],
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> list[float]:
        model.train()
        losses: list[float] = []

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    def test_simple_training_reproducibility(self) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader

        seed_all(seed=123, pytorch=True, deterministic=True)

        model1 = self.create_simple_network()
        dataset1 = self.create_synthetic_dataset()
        dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True)
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
        criterion1 = nn.MSELoss()

        losses1 = self.train_one_epoch(model1, dataloader1, optimizer1, criterion1)

        seed_all(seed=123, pytorch=True, deterministic=True)

        model2 = self.create_simple_network()
        dataset2 = self.create_synthetic_dataset()
        dataloader2 = DataLoader(dataset2, batch_size=16, shuffle=True)
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
        criterion2 = nn.MSELoss()

        losses2 = self.train_one_epoch(model2, dataloader2, optimizer2, criterion2)

        assert len(losses1) == len(losses2)
        for loss1, loss2 in zip(losses1, losses2, strict=True):
            assert abs(loss1 - loss2) < 1e-6

        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            assert torch.allclose(p1, p2)

    def test_multi_epoch_training_reproducibility(self) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader

        num_epochs = 3

        seed_all(seed=456, pytorch=True, deterministic=True)

        model1 = self.create_simple_network()
        dataset1 = self.create_synthetic_dataset()
        dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True)
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
        criterion1 = nn.MSELoss()

        all_losses1: list[list[float]] = []
        for _ in range(num_epochs):
            losses = self.train_one_epoch(model1, dataloader1, optimizer1, criterion1)
            all_losses1.append(losses)

        seed_all(seed=456, pytorch=True, deterministic=True)

        model2 = self.create_simple_network()
        dataset2 = self.create_synthetic_dataset()
        dataloader2 = DataLoader(dataset2, batch_size=16, shuffle=True)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        criterion2 = nn.MSELoss()

        all_losses2: list[list[float]] = []
        for _ in range(num_epochs):
            losses = self.train_one_epoch(model2, dataloader2, optimizer2, criterion2)
            all_losses2.append(losses)

        assert len(all_losses1) == len(all_losses2)
        for epoch_losses1, epoch_losses2 in zip(all_losses1, all_losses2, strict=True):
            assert len(epoch_losses1) == len(epoch_losses2)
            for loss1, loss2 in zip(epoch_losses1, epoch_losses2, strict=True):
                assert abs(loss1 - loss2) < 1e-6

        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_dataloader_with_workers_reproducibility(self) -> None:
        import torch
        from torch.utils.data import DataLoader

        seed_all(seed=789, pytorch=True, deterministic=True)

        g1 = torch.Generator()
        g1.manual_seed(789)

        dataset1 = self.create_synthetic_dataset(size=200)
        dataloader1 = DataLoader(
            dataset1,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=g1,
        )

        batches1 = []
        for batch_x, batch_y in dataloader1:
            batches1.append((batch_x.clone(), batch_y.clone()))

        seed_all(seed=789, pytorch=True, deterministic=True)

        g2 = torch.Generator()
        g2.manual_seed(789)

        dataset2 = self.create_synthetic_dataset(size=200)
        dataloader2 = DataLoader(
            dataset2,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=g2,
        )

        batches2 = []
        for batch_x, batch_y in dataloader2:
            batches2.append((batch_x.clone(), batch_y.clone()))

        assert len(batches1) == len(batches2)
        for (x1, y1), (x2, y2) in zip(batches1, batches2, strict=True):
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)

    def test_model_initialization_reproducibility(self) -> None:
        import torch

        seed_all(seed=999, pytorch=True, deterministic=True)
        model1 = self.create_simple_network()

        seed_all(seed=999, pytorch=True, deterministic=True)
        model2 = self.create_simple_network()

        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_different_seeds_produce_different_models(self) -> None:
        import torch

        seed_all(seed=111, pytorch=True, deterministic=True)
        model1 = self.create_simple_network()

        seed_all(seed=222, pytorch=True, deterministic=True)
        model2 = self.create_simple_network()

        different = False
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            if not torch.equal(p1, p2):
                different = True
                break

        assert different

    @pytest.mark.skipif(not (NUMPY_AVAILABLE and TORCH_AVAILABLE), reason="numpy or torch not installed")
    def test_numpy_in_training_loop_reproducibility(self) -> None:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader

        from reproducibility import get_numpy_rng

        seed_all(seed=555, numpy=True, pytorch=True, deterministic=True)

        model1 = self.create_simple_network()
        dataset1 = self.create_synthetic_dataset()
        dataloader1 = DataLoader(dataset1, batch_size=16, shuffle=True)
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
        criterion1 = nn.MSELoss()

        rng1 = get_numpy_rng()
        assert rng1 is not None

        noise_samples1 = []
        for batch_x, batch_y in dataloader1:
            noise = rng1.normal(0, 0.01, size=batch_x.shape)
            noise_samples1.append(noise.copy())

            noise_tensor = torch.from_numpy(noise).float()
            batch_x_noisy = batch_x + noise_tensor

            optimizer1.zero_grad()
            outputs = model1(batch_x_noisy)
            loss = criterion1(outputs, batch_y)
            loss.backward()
            optimizer1.step()

        seed_all(seed=555, numpy=True, pytorch=True, deterministic=True)

        model2 = self.create_simple_network()
        dataset2 = self.create_synthetic_dataset()
        dataloader2 = DataLoader(dataset2, batch_size=16, shuffle=True)
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
        criterion2 = nn.MSELoss()

        rng2 = get_numpy_rng()
        assert rng2 is not None

        noise_samples2 = []
        for batch_x, batch_y in dataloader2:
            noise = rng2.normal(0, 0.01, size=batch_x.shape)
            noise_samples2.append(noise.copy())

            noise_tensor = torch.from_numpy(noise).float()
            batch_x_noisy = batch_x + noise_tensor

            optimizer2.zero_grad()
            outputs = model2(batch_x_noisy)
            loss = criterion2(outputs, batch_y)
            loss.backward()
            optimizer2.step()

        assert len(noise_samples1) == len(noise_samples2)
        for n1, n2 in zip(noise_samples1, noise_samples2, strict=True):
            assert np.allclose(n1, n2)

        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            assert torch.allclose(p1, p2, atol=1e-6)
