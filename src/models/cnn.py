#!/usr/bin/env python3
"""基于 PyTorch 的 1D CNN 模型适配器，支持谓词约束（LUSI）。

将 138 维特征视为 1D 信号，通过三层 Conv1d 提取局部模式，输出单个
logit（二分类）。支持 LUSI 谓词约束：无约束时用 BCEWithLogitsLoss，
有约束时用 sigmoid 概率上的 LUSI 加权 MSE（与 MLP 一致，复用共享训练循环）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..core.contracts import OptionalDependencyError, TrainingData
from ..utils.logging import get_logger
from .registry import MODEL_REGISTRY

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise OptionalDependencyError("CNN model requires PyTorch. Install it with: pip install torch")


if _TORCH_AVAILABLE:

    class MPMCnn1D(nn.Module):
        """面向 MPM 表格数据的 1D 卷积神经网络，输出单个 logit（二分类）。"""

        def __init__(
            self,
            n_features: int,
            conv_channels: list[int] | None = None,
            kernel_size: int = 3,
            fc_units: int = 64,
            dropout: float = 0.3,
        ):
            super().__init__()
            if conv_channels is None:
                conv_channels = [32, 64, 128]
            self.n_features = n_features
            self.conv_channels = conv_channels
            self.kernel_size = kernel_size
            self.fc_units = fc_units
            self.dropout = dropout

            layers: list[nn.Module] = []
            in_channels = 1
            for out_channels in conv_channels:
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                in_channels = out_channels
            self.conv = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(conv_channels[-1], fc_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_units, 1),
            )
            # LUSI 谓词约束：可学习 τ（sigmoid(alpha)）
            self.alpha = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x.unsqueeze(1)  # (B, 1, n_features)
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)  # (B, conv_channels[-1])
            return self.fc(x)  # (B, 1)


class CnnClassifier(BaseEstimator, ClassifierMixin):
    """sklearn 兼容的 1D CNN 分类器，支持 sample_weight 与谓词约束。"""

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        conv_channels: list[int] | None = None,
        kernel_size: int = 3,
        fc_units: int = 64,
        dropout: float = 0.3,
        random_state: int | None = None,
        device: str = "cpu",
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.conv_channels = conv_channels if conv_channels is not None else [32, 64, 128]
        self.kernel_size = kernel_size
        self.fc_units = fc_units
        self.dropout = dropout
        self.random_state = random_state
        self.device = device

    def fit(self, X, y, sample_weight=None, constraints=None):
        _require_torch()
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32, ensure_min_samples=2)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        y = np.asarray(y, dtype=np.float32)

        log = get_logger("cnn")
        log.info("CNN training: %d samples, %d features, %d epochs, batch=%d, lr=%.5f",
                 len(X), X.shape[1], self.n_epochs, self.batch_size, self.learning_rate)

        self.model_ = MPMCnn1D(
            n_features=X.shape[1],
            conv_channels=list(self.conv_channels),
            kernel_size=self.kernel_size,
            fc_units=self.fc_units,
            dropout=self.dropout,
        ).to(self.device)

        from ..training.trainer import TorchTrainingConfig, TorchTrainingLoop

        has_constraints = constraints is not None and ("phi_vector" in constraints or "phi_vectors" in constraints)
        if has_constraints:
            log.info("Using predicate constraints for weighted MSE loss")
        else:
            log.info("No predicate constraints, using standard BCEWithLogitsLoss")

        training_config = TorchTrainingConfig(
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
            random_state=self.random_state,
        )
        TorchTrainingLoop.fit(
            self.model_, X, y, training_config,
            sample_weight=sample_weight,
            constraints=constraints,
            logger=log,
        )
        return self

    def predict_proba(self, X):
        _require_torch()
        check_is_fitted(self, "model_")
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze(-1)
            proba_pos = torch.sigmoid(logits).cpu().numpy()
            proba_neg = 1 - proba_pos
        return np.column_stack((proba_neg, proba_pos))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "conv_channels": self.conv_channels,
            "kernel_size": self.kernel_size,
            "fc_units": self.fc_units,
            "dropout": self.dropout,
            "random_state": self.random_state,
            "device": self.device,
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def __getstate__(self):
        _require_torch()
        state = self.__dict__.copy()
        if "model_" in state and state["model_"] is not None:
            state["_model_state_dict"] = {k: v.cpu().clone() for k, v in state["model_"].state_dict().items()}
            del state["model_"]
        return state

    def __setstate__(self, state):
        _require_torch()
        model_state = state.pop("_model_state_dict", None)
        self.__dict__.update(state)
        if model_state:
            model = MPMCnn1D(
                n_features=getattr(self, "n_features_in_", 138),
                conv_channels=list(self.conv_channels),
                kernel_size=self.kernel_size,
                fc_units=self.fc_units,
                dropout=self.dropout,
            )
            model.load_state_dict(model_state)
            self.model_ = model


@MODEL_REGISTRY.decorator("cnn")
class CnnAdapter:
    """CNN 模型适配器，支持谓词约束（LUSI）。"""

    name = "cnn"
    artifact_filename = "model_cnn.pkl"
    supports_constraints = True

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        n_epochs = int(params.get("n_epochs", 100))
        if n_epochs < 1:
            raise ValueError("cnn n_epochs must be positive")
        batch_size = int(params.get("batch_size", 64))
        if batch_size < 1:
            raise ValueError("cnn batch_size must be positive")
        learning_rate = float(params.get("learning_rate", 1e-3))
        if learning_rate <= 0:
            raise ValueError("cnn learning_rate must be positive")
        dropout = float(params.get("dropout", 0.3))
        if not 0 <= dropout < 1:
            raise ValueError("cnn dropout must be in [0, 1)")

    def build(self, params: Mapping[str, Any], seed: int) -> CnnClassifier:
        return CnnClassifier(
            n_epochs=int(params.get("n_epochs", 100)),
            batch_size=int(params.get("batch_size", 64)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
            conv_channels=list(params.get("conv_channels", [32, 64, 128])),
            kernel_size=int(params.get("kernel_size", 3)),
            fc_units=int(params.get("fc_units", 64)),
            dropout=float(params.get("dropout", 0.3)),
            random_state=seed,
            device=str(params.get("device", "cpu")),
        )

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {"sample_weight": data.sample_weight}
        if data.constraints:
            kwargs["constraints"] = dict(data.constraints)
        return kwargs
