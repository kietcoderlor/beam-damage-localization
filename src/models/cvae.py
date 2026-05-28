"""
Conditional Variational Autoencoder (CVAE) for synthetic class-0 generation.

Architecture
------------
Input x : 4 modal frequencies + 4 x resample_len resampled mode vectors
           = 4 + 4*32 = 132 features (default)
Condition c : one-hot class label  [0, 1, 2, 4]  →  4 dims

Encoder : [x ; c]  →  FC stack  →  (mu, log_var)   latent_dim each
Decoder : [z ; c]  →  FC stack  →  x_reconstructed

Loss = MSE(x, x_hat) + beta * KL(q(z|x,c) || N(0,I))

Beta annealing: beta ramps linearly from 0 to beta_max over anneal_epochs,
then stays constant. This prevents KL collapse early in training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.baseline_features import (
    FREQ_COLS,
    MODE_VECTOR_JSON_COLS,
    parse_mode_vector_json,
    resample_1d,
)


# Canonical class label ordering (must match one-hot encoding)
CLASS_ORDER: list[int] = [0, 1, 2, 4]


@dataclass(frozen=True)
class CVAEConfig:
    resample_len: int = 32
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (64, 32)
    beta_max: float = 0.01
    anneal_epochs: int = 200
    epochs: int = 500
    lr: float = 1e-3
    batch_size: int = 16
    random_state: int = 42
    device: str = "cpu"


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dims: Sequence[int], latent_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim + cond_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_features, h), nn.ReLU()]
            in_features = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = latent_dim + cond_dim
        for h in reversed(list(hidden_dims)):
            layers += [nn.Linear(in_features, h), nn.ReLU()]
            in_features = h
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, c], dim=-1))


class CVAE(nn.Module):
    def __init__(self, cfg: CVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CVAEConfig()

        self._cond_dim = len(CLASS_ORDER)
        self._input_dim = len(FREQ_COLS) + len(MODE_VECTOR_JSON_COLS) * self.cfg.resample_len

        self.encoder = _Encoder(
            input_dim=self._input_dim,
            cond_dim=self._cond_dim,
            hidden_dims=self.cfg.hidden_dims,
            latent_dim=self.cfg.latent_dim,
        )
        self.decoder = _Decoder(
            latent_dim=self.cfg.latent_dim,
            cond_dim=self._cond_dim,
            hidden_dims=self.cfg.hidden_dims,
            output_dim=self._input_dim,
        )

        # Feature statistics for normalisation (set during fit)
        self._x_mean: torch.Tensor | None = None
        self._x_std: torch.Tensor | None = None
        self._is_fit: bool = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, c)
        z = self._reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _onehot(labels: np.ndarray) -> np.ndarray:
        """Map num_damages labels → one-hot using CLASS_ORDER."""
        n = len(labels)
        oh = np.zeros((n, len(CLASS_ORDER)), dtype=np.float32)
        for i, lbl in enumerate(labels):
            idx = CLASS_ORDER.index(int(lbl))
            oh[i, idx] = 1.0
        return oh

    def _df_to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract feature tensor from a scenario-level DataFrame."""
        freqs = df[FREQ_COLS].astype(float).to_numpy(dtype=np.float32)  # (N, 4)

        n = len(df)
        vec_mat = np.zeros((n, len(MODE_VECTOR_JSON_COLS) * self.cfg.resample_len), dtype=np.float32)
        for i, col in enumerate(MODE_VECTOR_JSON_COLS):
            start = i * self.cfg.resample_len
            end = start + self.cfg.resample_len
            for j, val in enumerate(df[col].tolist()):
                raw = parse_mode_vector_json(val)
                vec_mat[j, start:end] = resample_1d(raw, self.cfg.resample_len).astype(np.float32)

        x = np.concatenate([freqs, vec_mat], axis=1)  # (N, input_dim)
        return torch.from_numpy(x)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        assert self._x_mean is not None
        return (x - self._x_mean) / (self._x_std + 1e-8)

    def _denormalise(self, x: torch.Tensor) -> torch.Tensor:
        assert self._x_mean is not None
        return x * (self._x_std + 1e-8) + self._x_mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> list[float]:
        """
        Train the CVAE on all classes in train_df.

        Args:
            train_df: scenario-level DataFrame (all classes, num_modes_found==4).

        Returns:
            List of per-epoch total losses.
        """
        cfg = self.cfg
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)
        device = torch.device(cfg.device)

        X = self._df_to_tensor(train_df)
        C = torch.from_numpy(self._onehot(train_df["num_damages"].to_numpy()))

        # Compute normalisation stats on training data
        self._x_mean = X.mean(dim=0)
        self._x_std = X.std(dim=0)

        X_norm = self._normalise(X)
        dataset = TensorDataset(X_norm.to(device), C.to(device))
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        self.to(device)
        optimiser = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        epoch_losses: list[float] = []

        for epoch in range(1, cfg.epochs + 1):
            # Linear beta annealing
            beta = min(cfg.beta_max, cfg.beta_max * epoch / max(cfg.anneal_epochs, 1))

            self.train()
            total_loss = 0.0
            for x_batch, c_batch in loader:
                x_hat, mu, logvar = self(x_batch, c_batch)
                recon_loss = nn.functional.mse_loss(x_hat, x_batch, reduction="mean")
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(x_batch)

            epoch_losses.append(total_loss / len(dataset))

            if epoch % 100 == 0:
                print(f"  Epoch {epoch:4d}/{cfg.epochs}  loss={epoch_losses[-1]:.6f}  beta={beta:.5f}")

        self._is_fit = True
        self.to("cpu")
        return epoch_losses

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_class0(self, n_samples: int, random_state: int | None = None) -> pd.DataFrame:
        """
        Sample n_samples synthetic class-0 rows from the learned latent prior.

        Returns a DataFrame with the same schema as scenario_dataset.csv
        (no damage metadata, but with freq_mode_* and mode_*_vector_json columns).
        """
        if not self._is_fit:
            raise RuntimeError("Call fit() before generate_class0().")

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.cfg.latent_dim)
            c0_idx = CLASS_ORDER.index(0)
            c0 = torch.zeros(n_samples, len(CLASS_ORDER))
            c0[:, c0_idx] = 1.0

            x_hat_norm = self.decoder(z, c0)
            x_hat = self._denormalise(x_hat_norm).numpy()

        rows = []
        for i in range(n_samples):
            row: dict = {}
            row["config_id"] = f"cvae_normal_{i + 1:03d}__pos_na_na_na_na__sev_na"
            row["scenario_name"] = f"cvae_normal_{i + 1:03d}"
            row["num_damages"] = 0
            row["damage_pos_1"] = float("nan")
            row["damage_pos_2"] = float("nan")
            row["damage_pos_3"] = float("nan")
            row["damage_pos_4"] = float("nan")
            row["damage_severity"] = float("nan")
            row["num_modes_found"] = 4

            # Frequencies: first 4 values, sorted to preserve ordering
            freqs = np.sort(np.abs(x_hat[i, :4]))
            for j, col in enumerate(FREQ_COLS):
                row[col] = float(freqs[j])

            # Mode vectors: upsample resampled 32D → 191D
            for j, col in enumerate(MODE_VECTOR_JSON_COLS):
                start = 4 + j * self.cfg.resample_len
                end = start + self.cfg.resample_len
                vec32 = x_hat[i, start:end]
                vec191 = resample_1d(vec32.astype(float), 191)
                row[col] = json.dumps(vec191.tolist())

            rows.append(row)

        return pd.DataFrame(rows)
