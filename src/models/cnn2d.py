#!/usr/bin/env python3
"""基于 patch 的空间二维卷积模型（grid 型模型）。

与 1D CNN 不同，本模型不是对 138 维特征向量做卷积，而是把整个研究区
当作一张 (H, W, C) 的图像：围绕每个训练点取一个 P×P 的局部空间窗口
（patch），用 Conv2d 判断窗口中心是否矿化。因此它需要网格数据，是
``grid_model``（区别于逐点的 point 模型）。

由于网格数据不在逐点模型接口（fit/predict_proba 的 X 仅为特征矩阵）中，
``Experiment`` 在 ``train_from_archive_features`` 模式下会通过 fit_params 把
``grid`` / ``inside`` / ``train_cells`` 注入，并在评估时用 ``set_predict_cells``
指定点级预测的单元。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..core.contracts import OptionalDependencyError, TrainingData
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
        raise OptionalDependencyError("cnn2d requires PyTorch. Install it with: pip install torch")


if _TORCH_AVAILABLE:

    class _PatchConv(nn.Module):
        def __init__(self, in_ch: int, dropout: float = 0.3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 1))
            # LUSI 谓词约束：可学习 τ（sigmoid(alpha)）
            self.alpha = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            h = self.features(x).flatten(1)
            return self.classifier(h).squeeze(-1)


class Cnn2dClassifier(BaseEstimator, ClassifierMixin):
    """sklearn 兼容的空间 patch CNN 分类器。

    训练用 ``grid``（(H,W,C)）围绕 ``train_cells`` 取 patch；
    预测时若设置了 ``_predict_cells`` 则按点预测，否则在整幅有效单元上滑动。
    """

    def __init__(
        self,
        patch: int = 15,
        n_epochs: int = 150,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        random_state: int | None = None,
        device: str = "cpu",
    ):
        self.patch = patch
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.random_state = random_state
        self.device = device

    def _extract_patches(self, cells):
        grid = self.grid_  # (H, W, C)
        H, W, C = grid.shape
        pad = self.patch // 2
        gt = torch.from_numpy(grid.transpose(2, 0, 1)).float()  # (C, H, W)
        padded = torch.nn.functional.pad(gt, (pad, pad, pad, pad), mode="reflect")
        out = []
        for r, c in cells:
            out.append(padded[:, r:r + self.patch, c:c + self.patch])
        return torch.stack(out)

    def fit(self, X, y, sample_weight=None, grid=None, inside=None, train_cells=None, constraints=None):
        _require_torch()
        if grid is None or train_cells is None:
            raise ValueError("cnn2d requires grid/train_cells injected via fit_params")
        self.grid_ = np.asarray(grid, dtype=np.float32)
        self.inside_ = np.asarray(inside, dtype=bool) if inside is not None else np.ones(self.grid_.shape[:2], dtype=bool)
        self.classes_ = np.array([0, 1])
        self._predict_cells = None

        y = np.asarray(y, dtype=np.float32)
        # 训练张量和模型统一放置在配置的 device 上。
        Xp = self._extract_patches(train_cells).to(self.device)
        yt = torch.from_numpy(y).float().to(self.device)
        sw = (
            torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)).float().to(self.device)
            if sample_weight is not None
            else None
        )

        from ..training.losses import _build_combined_phi, weighted_mse_with_predicate

        has_constraints = constraints is not None and ("phi_vector" in constraints or "phi_vectors" in constraints)
        if has_constraints:
            phi = _build_combined_phi(constraints, len(y), torch.device(self.device))
        else:
            phi = None

        pos = int(y.sum())
        neg = len(y) - pos
        pos_weight = torch.tensor(neg / max(pos, 1))
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        # 与普通 CNN/MLP 一致：BatchNorm 初始化和训练均使用模型 seed。
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        model = _PatchConv(
            in_ch=self.grid_.shape[2], dropout=self.dropout
        ).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # 实际按 batch_size 分批训练（此前 batch_size 只做校验、训练仍全批，
        # 使该研究变量静默失效）；每批独立随机翻转做增强。
        n = len(Xp)
        optimizer_steps = 0
        for ep in range(self.n_epochs):
            model.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                x = Xp[idx]
                if torch.rand(1) > 0.5:
                    x = torch.flip(x, dims=[2])
                if torch.rand(1) > 0.5:
                    x = torch.flip(x, dims=[3])
                opt.zero_grad()
                logits = model(x)
                if has_constraints:
                    pred = torch.sigmoid(logits)
                    tau_hat = torch.sigmoid(model.alpha)
                    tau = 1 - tau_hat
                    batch_w = sw[idx] if sw is not None else None
                    batch_phi = phi[idx]
                    loss = weighted_mse_with_predicate(
                        pred, yt[idx], batch_phi, tau_hat, tau, sample_weight=batch_w
                    )["total"]
                else:
                    # baseline 分支也使用样本权重，与谓词分支一致，避免配置的
                    # 权重实验在 CNN2D 上静默失效。
                    losses = bce(logits, yt[idx])
                    loss = (losses * sw[idx]).mean() if sw is not None else losses.mean()
                loss.backward()
                opt.step()
                optimizer_steps += 1

        # 保护：即使绕过配置校验直接构建估计器，也不允许零优化步训练被
        # 当作已拟合模型保存/预测（与共享 trainer 的零步保护对齐）。
        if optimizer_steps == 0:
            raise RuntimeError(
                "cnn2d training completed zero optimizer steps; set n_epochs >= 1 "
                "and batch_size <= n_train_samples."
            )
        self.model_ = model
        return self

    def set_predict_cells(self, cells):
        self._predict_cells = cells

    def _forward(self, cells):
        _require_torch()
        model = self.model_
        model.eval()
        with torch.no_grad():
            patches = self._extract_patches(cells)
            out = []
            for i in range(0, len(patches), 256):
                out.append(torch.sigmoid(model(patches[i:i + 256].to(self.device))).cpu().numpy())
        return np.concatenate(out)

    def predict_proba(self, X):
        _require_torch()
        if self._predict_cells is not None:
            cells = self._predict_cells
        else:
            cells = list(zip(*np.where(self.inside_)))
        p = self._forward(cells)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            "patch": self.patch, "n_epochs": self.n_epochs, "batch_size": self.batch_size,
            "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
            "dropout": self.dropout, "random_state": self.random_state, "device": self.device,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state and state["model_"] is not None:
            state["_model_state_dict"] = {k: v.cpu().clone() for k, v in state["model_"].state_dict().items()}
            del state["model_"]
        return state

    def __setstate__(self, state):
        model_state = state.pop("_model_state_dict", None)
        self.__dict__.update(state)
        if model_state and hasattr(self, "grid_"):
            model = _PatchConv(
                in_ch=self.grid_.shape[2], dropout=getattr(self, "dropout", 0.3)
            )
            model.load_state_dict(model_state)
            self.model_ = model


@MODEL_REGISTRY.decorator("cnn2d")
class Cnn2dAdapter:
    """空间 patch CNN 适配器。grid_model=True 表示需要网格数据。"""

    name = "cnn2d"
    artifact_filename = "model_cnn2d.pkl"
    supports_constraints = True
    grid_model = True

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("patch", 15)) < 3:
            raise ValueError("cnn2d patch must be >= 3 (odd recommended)")
        n_epochs = int(params.get("n_epochs", 150))
        if n_epochs < 1:
            raise ValueError("cnn2d n_epochs must be positive")
        batch_size = int(params.get("batch_size", 64))
        if batch_size < 2:
            raise ValueError("cnn2d batch_size must be at least 2 because CNN2D uses BatchNorm")
        dropout = float(params.get("dropout", 0.3))
        if not 0 <= dropout < 1:
            raise ValueError("cnn2d dropout must be in [0, 1)")

    def build(self, params: Mapping[str, Any], seed: int) -> Cnn2dClassifier:
        return Cnn2dClassifier(
            patch=int(params.get("patch", 15)),
            n_epochs=int(params.get("n_epochs", 150)),
            batch_size=int(params.get("batch_size", 64)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
            dropout=float(params.get("dropout", 0.3)),
            random_state=seed,
            device=str(params.get("device", "cpu")),
        )

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {
            "sample_weight": data.sample_weight,
            "grid": data.metadata["grid"],
            "inside": data.metadata["inside"],
            "train_cells": data.metadata["train_cells"],
        }
        if data.constraints:
            kwargs["constraints"] = dict(data.constraints)
        return kwargs
