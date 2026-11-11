from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.base import PreprocessingSpec


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
        self._lr = lr
        self._net: _AE | None = None
        self._threshold: float = 0.0
        self._n_features: int = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    def _resolve_device(self) -> torch.device:
        if self._device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device_str)

    def fit(self, X: np.ndarray, **kwargs) -> None:
        device = self._resolve_device()
        self._n_features = X.shape[1]
        self._net = _AE(self._n_features).to(device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self._batch_size, shuffle=True)

        self._net.train()
        for _ in range(self._epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = self._net(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "fit() first"
        device = self._resolve_device()
        self._net.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            recon = self._net(tensor)
        mse = ((tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        return mse

    def calibrate_threshold(self, X_val_benign: np.ndarray, percentile: float = 95.0) -> float:
        errors = self.score(X_val_benign)
        self._threshold = float(np.percentile(errors, percentile))
        return self._threshold

    def save(self, path: str) -> None:
        assert self._net is not None, "fit() first"
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(), "n_features": self._n_features, "threshold": self._threshold},
            Path(path) / "detector.pt",
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(Path(path) / "detector.pt", map_location="cpu", weights_only=True)
        self._n_features = ckpt["n_features"]
        self._threshold = ckpt["threshold"]
        self._net = _AE(self._n_features)
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
