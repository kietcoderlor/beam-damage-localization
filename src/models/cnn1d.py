from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.baseline_features import MODE_VECTOR_JSON_COLS, FREQ_COLS, parse_mode_vector_json, resample_1d


POSITION_COLS = ["damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


@dataclass(frozen=True)
class Cnn1dBaselineConfig:
    random_state: int = 42
    resample_len: int = 191
    batch_size: int = 32
    max_epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    hidden_dim: int = 64
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    device: str = "cpu"


class _CnnBackbone(nn.Module):
    def __init__(self, seq_len: int, freq_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        pooled_dim = 64 * 2  # avg + max
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim + freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(hidden_dim, n_classes)
        self.reg_head = nn.Linear(hidden_dim, len(POSITION_COLS))

    def forward(self, x_modes: torch.Tensor, x_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x_modes)
        h_avg = torch.mean(h, dim=-1)
        h_max = torch.amax(h, dim=-1)
        h = torch.cat([h_avg, h_max, x_freq], dim=1)
        h = self.mlp(h)
        logits = self.cls_head(h)
        pos = self.reg_head(h)
        return logits, pos


class Cnn1dDamageBaseline:
    """
    1D-CNN baseline on raw mode shapes:
    - input: 4 mode-shape channels + 4 normalized frequencies
    - output 1: multiclass num_damages
    - output 2: 4-slot position regression
    """

    def __init__(self, cfg: Cnn1dBaselineConfig | None = None) -> None:
        self.cfg = cfg or Cnn1dBaselineConfig()
        self._is_fit = False
        self._class_values: np.ndarray | None = None
        self._freq_scaler = StandardScaler()
        self._pos_scaler = StandardScaler()
        self.model: _CnnBackbone | None = None

    def _build_inputs(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        modes_per_col: list[np.ndarray] = []
        for col in MODE_VECTOR_JSON_COLS:
            vecs = [resample_1d(parse_mode_vector_json(v), self.cfg.resample_len) for v in df[col].tolist()]
            modes_per_col.append(np.stack(vecs, axis=0))

        x_modes = np.stack(modes_per_col, axis=1).astype(np.float32)  # (n, 4, L)
        x_freq = df[FREQ_COLS].astype(float).to_numpy(dtype=np.float32)
        return x_modes, x_freq

    def _to_loader(
        self,
        x_modes: np.ndarray,
        x_freq: np.ndarray,
        y_cls: np.ndarray | None = None,
        y_pos: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        tensors: list[torch.Tensor] = [
            torch.from_numpy(x_modes),
            torch.from_numpy(x_freq),
        ]
        if y_cls is not None:
            tensors.append(torch.from_numpy(y_cls.astype(np.int64)))
        if y_pos is not None:
            tensors.append(torch.from_numpy(y_pos.astype(np.float32)))
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> None:
        torch.manual_seed(self.cfg.random_state)
        np.random.seed(self.cfg.random_state)

        y_num_raw_train = y_train["num_damages"].astype(int).to_numpy()
        y_num_raw_val = None if y_val is None else y_val["num_damages"].astype(int).to_numpy()
        if y_num_raw_val is None:
            self._class_values = np.unique(y_num_raw_train)
        else:
            self._class_values = np.unique(np.concatenate([y_num_raw_train, y_num_raw_val]))

        y_cls_train = np.searchsorted(self._class_values, y_num_raw_train).astype(np.int64)
        y_cls_val = None if y_num_raw_val is None else np.searchsorted(self._class_values, y_num_raw_val).astype(np.int64)

        x_modes_train, x_freq_train = self._build_inputs(X_train)
        x_freq_train = self._freq_scaler.fit_transform(x_freq_train).astype(np.float32)

        y_pos_train = y_train[POSITION_COLS].astype(float).to_numpy()
        y_pos_train = np.where(np.isfinite(y_pos_train), y_pos_train, -1.0)
        y_pos_train = self._pos_scaler.fit_transform(y_pos_train).astype(np.float32)

        train_loader = self._to_loader(
            x_modes_train,
            x_freq_train,
            y_cls=y_cls_train,
            y_pos=y_pos_train,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None and y_cls_val is not None:
            x_modes_val, x_freq_val = self._build_inputs(X_val)
            x_freq_val = self._freq_scaler.transform(x_freq_val).astype(np.float32)
            y_pos_val = y_val[POSITION_COLS].astype(float).to_numpy()
            y_pos_val = np.where(np.isfinite(y_pos_val), y_pos_val, -1.0)
            y_pos_val = self._pos_scaler.transform(y_pos_val).astype(np.float32)
            val_loader = self._to_loader(x_modes_val, x_freq_val, y_cls=y_cls_val, y_pos=y_pos_val, shuffle=False)

        device = torch.device(self.cfg.device)
        self.model = _CnnBackbone(
            seq_len=self.cfg.resample_len,
            freq_dim=len(FREQ_COLS),
            hidden_dim=self.cfg.hidden_dim,
            n_classes=len(self._class_values),
        ).to(device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        cls_loss_fn = nn.CrossEntropyLoss()
        reg_loss_fn = nn.MSELoss()

        best_state = None
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for _epoch in range(self.cfg.max_epochs):
            self.model.train()
            for xb_modes, xb_freq, yb_cls, yb_pos in train_loader:
                xb_modes = xb_modes.to(device)
                xb_freq = xb_freq.to(device)
                yb_cls = yb_cls.to(device)
                yb_pos = yb_pos.to(device)

                opt.zero_grad()
                logits, pred_pos = self.model(xb_modes, xb_freq)
                loss_cls = cls_loss_fn(logits, yb_cls)
                loss_reg = reg_loss_fn(pred_pos, yb_pos)
                loss = self.cfg.cls_loss_weight * loss_cls + self.cfg.reg_loss_weight * loss_reg
                loss.backward()
                opt.step()

            if val_loader is None:
                continue

            self.model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb_modes, xb_freq, yb_cls, yb_pos in val_loader:
                    xb_modes = xb_modes.to(device)
                    xb_freq = xb_freq.to(device)
                    yb_cls = yb_cls.to(device)
                    yb_pos = yb_pos.to(device)
                    logits, pred_pos = self.model(xb_modes, xb_freq)
                    loss_cls = cls_loss_fn(logits, yb_cls)
                    loss_reg = reg_loss_fn(pred_pos, yb_pos)
                    loss = self.cfg.cls_loss_weight * loss_cls + self.cfg.reg_loss_weight * loss_reg
                    val_loss_total += float(loss.item())
                    val_batches += 1

            val_loss = val_loss_total / max(val_batches, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.patience:
                    break

        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)

        self._is_fit = True

    def _predict_internal(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fit or self.model is None or self._class_values is None:
            raise RuntimeError("Model is not fit yet.")

        x_modes, x_freq = self._build_inputs(X)
        x_freq = self._freq_scaler.transform(x_freq).astype(np.float32)
        loader = self._to_loader(x_modes, x_freq, shuffle=False)

        device = torch.device(self.cfg.device)
        self.model.eval()
        pred_cls_all = []
        pred_pos_all = []

        with torch.no_grad():
            for xb_modes, xb_freq in loader:
                xb_modes = xb_modes.to(device)
                xb_freq = xb_freq.to(device)
                logits, pred_pos = self.model(xb_modes, xb_freq)
                pred_cls = torch.argmax(logits, dim=1).cpu().numpy()
                pred_pos = pred_pos.cpu().numpy()
                pred_cls_all.append(pred_cls)
                pred_pos_all.append(pred_pos)

        pred_cls_enc = np.concatenate(pred_cls_all, axis=0)
        pred_pos_scaled = np.concatenate(pred_pos_all, axis=0)
        pred_pos = self._pos_scaler.inverse_transform(pred_pos_scaled)
        pred_cls = self._class_values[pred_cls_enc]
        return pred_cls, pred_pos

    def predict_num_damages(self, X: pd.DataFrame) -> np.ndarray:
        pred_cls, _ = self._predict_internal(X)
        return pred_cls.astype(int)

    def predict_positions(self, X: pd.DataFrame) -> pd.DataFrame:
        _, pred_pos = self._predict_internal(X)
        return pd.DataFrame(pred_pos, columns=POSITION_COLS)

