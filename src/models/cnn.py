#!/usr/bin/env python3
"""基于 PyTorch 的 1D CNN 模型适配器，用于 MPM 表格数据二分类。

将 138 维特征视为 1D 信号，通过三层 Conv1d 提取局部模式，
再经全连接层输出二分类概率。通过 sklearn 兼容包装器与
BayesSearchCV 超参数搜索无缝协作。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..core.contracts import OptionalDependencyError, TrainingData
from ..utils.logging import get_logger
from .registry import MODEL_REGISTRY

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise OptionalDependencyError(
            "CNN model requires PyTorch. Install it with: pip install torch"
        )


# ============================================================================
# PyTorch 1D CNN 模块 — 仅在 torch 可用时定义
# ============================================================================

if _TORCH_AVAILABLE:

    class MPMCnn1D(nn.Module):
        """面向 MPM 表格数据的 1D 卷积神经网络。

        将 (batch_size, n_features) 的表格数据视为单通道 1D 信号，
        通过三层 Conv1d → BatchNorm → ReLU → Dropout 提取层次化特征，
        最后经 AdaptiveAvgPool1d 压缩时序维度，由全连接层输出类别 logits。

        Parameters
        ----------
        n_features : int
            输入特征维度（列数）。
        n_classes : int
            输出类别数，默认 2。
        conv_channels : list[int]
            每层卷积的输出通道数，同时定义网络深度。默认 [32, 64, 128]。
        kernel_size : int
            卷积核大小，默认 3。
        fc_units : int
            全连接隐藏层单元数，默认 64。
        dropout : float
            Dropout 比率，默认 0.3。
        """

        def __init__(
            self,
            n_features: int,
            n_classes: int = 2,
            conv_channels: list[int] | None = None,
            kernel_size: int = 3,
            fc_units: int = 64,
            dropout: float = 0.3,
        ):
            super().__init__()
            if conv_channels is None:
                conv_channels = [32, 64, 128]

            self.n_features = n_features
            self.n_classes = n_classes
            self.conv_channels = conv_channels
            self.kernel_size = kernel_size
            self.fc_units = fc_units
            self.dropout = dropout

            # 构建卷积层
            layers: list[nn.Module] = []
            in_channels = 1  # 表格数据在 unsqueeze 后为单通道
            for out_channels in conv_channels:
                layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
                )
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                in_channels = out_channels
            self.conv = nn.Sequential(*layers)

            # 全局池化 → 全连接
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(conv_channels[-1], fc_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_units, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """前向传播。

            Parameters
            ----------
            x : torch.Tensor, shape (batch_size, n_features)
                输入特征。

            Returns
            -------
            torch.Tensor, shape (batch_size, n_classes)
                类别 logits。
            """
            x = x.unsqueeze(1)  # (B, 1, n_features)
            x = self.conv(x)     # (B, conv_channels[-1], n_features)
            x = self.pool(x)     # (B, conv_channels[-1], 1)
            x = x.squeeze(-1)    # (B, conv_channels[-1])
            return self.fc(x)    # (B, n_classes)


# ============================================================================
# sklearn 兼容分类器包装器
# ============================================================================


class CnnClassifier(BaseEstimator, ClassifierMixin):
    """sklearn 兼容的 CNN 分类器，封装 PyTorch 训练循环。

    支持 sample_weight、多分类（通过 CrossEntropyLoss）以及
    get_params/set_params 用于 BayesSearchCV 超参数搜索。

    Parameters
    ----------
    n_epochs : int
        训练轮数，默认 100。
    batch_size : int
        批大小，默认 64。
    learning_rate : float
        Adam 学习率，默认 1e-3。
    weight_decay : float
        L2 正则化系数，默认 1e-4。
    conv_channels : list[int]
        卷积层通道数列表，默认 [32, 64, 128]。
    kernel_size : int
        卷积核大小，默认 3。
    fc_units : int
        全连接隐藏层单元数，默认 64。
    dropout : float
        Dropout 比率，默认 0.3。
    random_state : int or None
        随机种子。
    device : str
        训练设备，默认 "cpu"。

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        训练后发现的类别标签。
    model_ : MPMCnn1D
        训练后的 PyTorch 模型。
    label_encoder_ : LabelEncoder
        将原始标签映射到 [0, 1, ...] 的编码器。
    """

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

    def fit(self, X, y, sample_weight=None):
        """训练 CNN 模型。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练特征。
        y : array-like of shape (n_samples,)
            训练标签。
        sample_weight : array-like of shape (n_samples,), optional
            样本权重，将乘以每个样本的交叉熵损失。

        Returns
        -------
        self : CnnClassifier
        """
        _require_torch()

        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32, ensure_min_samples=2)
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        log = get_logger("cnn")
        log.info("CNN training: %d samples, %d features, %d classes, %d epochs, batch=%d, lr=%.5f",
                 len(X), n_features, n_classes, self.n_epochs, self.batch_size, self.learning_rate)

        # 设置随机种子
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # 构建模型
        self.model_ = MPMCnn1D(
            n_features=n_features,
            n_classes=n_classes,
            conv_channels=list(self.conv_channels),
            kernel_size=self.kernel_size,
            fc_units=self.fc_units,
            dropout=self.dropout,
        )
        self.model_.to(self.device)

        # 转换为 tensor
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y_encoded).long()
        if sample_weight is not None:
            sw_tensor = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)).float()
        else:
            sw_tensor = None

        dataset = TensorDataset(X_tensor, y_tensor, sw_tensor if sw_tensor is not None else torch.ones(len(X_tensor)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(reduction="none")

        self.model_.train()
        log_interval = max(1, self.n_epochs // 10)
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_samples = 0
            for batch_X, batch_y, batch_w in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_w = batch_w.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(batch_X)
                losses = criterion(logits, batch_y)
                weighted_loss = (losses * batch_w).mean()
                weighted_loss.backward()
                optimizer.step()

                epoch_loss += weighted_loss.item() * len(batch_X)
                n_samples += len(batch_X)

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                log.info("Epoch %d/%d, loss=%.6f", epoch + 1, self.n_epochs, epoch_loss / n_samples)

        return self

    def predict_proba(self, X):
        """返回类别概率。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        _require_torch()
        check_is_fitted(self, "model_")

        X = check_array(X, accept_sparse=False, dtype=np.float32)
        X_tensor = torch.from_numpy(X).float().to(self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            proba = F.softmax(logits, dim=1).cpu().numpy()

        return proba

    def predict(self, X):
        """返回预测类别标签。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(indices)

    def get_params(self, deep=True):
        """获取模型参数，用于 BayesSearchCV。"""
        params = {
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
        return params

    def set_params(self, **params):
        """设置模型参数，用于 BayesSearchCV。"""
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def __getstate__(self):
        """序列化支持：将 PyTorch state_dict 转换为 CPU tensor。"""
        _require_torch()
        state = self.__dict__.copy()
        if "model_" in state and state["model_"] is not None:
            state["_model_state_dict"] = {k: v.cpu().clone() for k, v in state["model_"].state_dict().items()}
            del state["model_"]
        return state

    def __setstate__(self, state):
        """反序列化支持：从 state_dict 恢复 PyTorch 模型。"""
        _require_torch()
        # 先恢复简单属性（不含 _model_state_dict）
        model_state = state.pop("_model_state_dict", None)
        self.__dict__.update(state)
        if model_state:
            # 从 state_dict 推断 n_features
            n_features = 138  # 默认
            for key in model_state:
                if "conv.0.weight" in key:
                    n_features = model_state[key].shape[2]  # weight shape: (out_ch, 1, kernel_size)
                    break
            model = MPMCnn1D(
                n_features=n_features,
                n_classes=getattr(self, "n_classes", 2),
                conv_channels=list(self.conv_channels),
                kernel_size=self.kernel_size,
                fc_units=self.fc_units,
                dropout=self.dropout,
            )
            model.load_state_dict(model_state)
            self.model_ = model


# ============================================================================
# CnnAdapter — MPM 框架适配器
# ============================================================================


@MODEL_REGISTRY.decorator("cnn")
class CnnAdapter:
    """CNN 模型适配器，将 PyTorch 1D CNN 集成到 MPM 实验框架。

    通过 ComponentRegistry 机制注册为 "cnn" 模型，与 Experiment、
    Tuner、Splitter 等组件无缝协作。

    Attributes
    ----------
    name : str
        注册名称 "cnn"。
    artifact_filename : str
        序列化模型文件名。
    supports_constraints : bool
        CNN 不支持谓词约束。
    """

    name = "cnn"
    artifact_filename = "model_cnn.pkl"
    supports_constraints = False

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        """校验 CNN 配置参数的合法性。"""
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
        """根据配置参数构建 CnnClassifier 实例。

        Parameters
        ----------
        params : Mapping[str, Any]
            模型配置参数（来自 YAML 配置的 model.params 节）。
        seed : int
            实验级随机种子。

        Returns
        -------
        CnnClassifier
            配置好的 CNN 分类器实例。
        """
        resolved = dict(params)

        return CnnClassifier(
            n_epochs=int(resolved.get("n_epochs", 100)),
            batch_size=int(resolved.get("batch_size", 64)),
            learning_rate=float(resolved.get("learning_rate", 1e-3)),
            weight_decay=float(resolved.get("weight_decay", 1e-4)),
            conv_channels=list(resolved.get("conv_channels", [32, 64, 128])),
            kernel_size=int(resolved.get("kernel_size", 3)),
            fc_units=int(resolved.get("fc_units", 64)),
            dropout=float(resolved.get("dropout", 0.3)),
            random_state=seed,
            device=str(resolved.get("device", "cpu")),
        )

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        """返回传递给模型 fit() 方法的额外参数。

        CNN 通过 sample_weight 在 CrossEntropyLoss 中加权，
        以反映各地质点的置信度。

        Parameters
        ----------
        data : TrainingData
            训练数据包。

        Returns
        -------
        Mapping[str, Any]
            包含 sample_weight 的字典。
        """
        return {"sample_weight": data.sample_weight}