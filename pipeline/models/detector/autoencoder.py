from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.models.base import PreprocessingSpec


class _AE(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class MLPAutoencoder:
    label = "MLP Autoencoder"
    preprocessing = PreprocessingSpec(needs_scaling=True, preferred_scalers=("standard",))

    def __init__(
        self,
        device: str = "cpu",
        epochs: int = 20,
        batch_size: int = 512,
        lr: float = 1e-3,
    ) -> None:
        self._device_str = device
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = lr
        self._network: _AE | None = None
        self._threshold: float = 0.0
        self._n_features: int = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    def _resolve_device(self) -> torch.device:
        return torch.device(self._device_str)

    def fit(self, X: np.ndarray, **kwargs) -> None:
        device = self._resolve_device()
        self._n_features = X.shape[1]
        self._network = _AE(self._n_features).to(device)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)
        criterion = nn.MSELoss()

        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self._batch_size, shuffle=True)

        self._network.train()
        for _ in range(self._epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstruction = self._network(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._network is not None, "fit() first"
        device = self._resolve_device()
        self._network.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            reconstruction = self._network(tensor)
        mse = ((tensor - reconstruction) ** 2).mean(dim=1).cpu().numpy()
        return mse

    def score_per_feature(self, X: np.ndarray) -> np.ndarray:
        """Per-feature squared reconstruction errors, shape (n_samples, n_features)."""
        assert self._network is not None, "fit() first"
        device = self._resolve_device()
        self._network.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            reconstruction = self._network(tensor)
        return ((tensor - reconstruction) ** 2).cpu().numpy()

    def calibrate_threshold(self, X_val_benign: np.ndarray, percentile: float = 95.0) -> float:
        errors = self.score(X_val_benign)
        self._threshold = float(np.percentile(errors, percentile))
        return self._threshold

    def fine_tune(self, X: np.ndarray, freeze_epochs: int = 5) -> None:
        """Fine-tune with encoder frozen for `freeze_epochs`, then fully unfrozen."""
        assert self._network is not None, "fit() or load() first"
        device = self._resolve_device()
        self._network.to(device)

        for param in self._network.encoder.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            lr=self._learning_rate,
        )
        criterion = nn.MSELoss()
        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self._batch_size, shuffle=True)

        self._network.train()
        for epoch in range(self._epochs):
            if epoch == freeze_epochs:
                for param in self._network.encoder.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.Adam(self._network.parameters(), lr=self._learning_rate)
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstruction = self._network(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()

    def save(self, path: str) -> None:
        assert self._network is not None, "fit() first"
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._network.state_dict(), "n_features": self._n_features, "threshold": self._threshold},
            Path(path) / "detector.pt",
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(Path(path) / "detector.pt", map_location="cpu", weights_only=True)
        self._n_features = checkpoint["n_features"]
        self._threshold = checkpoint["threshold"]
        self._network = _AE(self._n_features)
        self._network.load_state_dict(checkpoint["state_dict"])
        self._network.eval()
