#!/usr/bin/env python3
"""基于 PyTorch 的 MLP 模型适配器，支持 LUSI 谓词约束加权损失。

将表格特征通过多层全连接网络进行二分类，支持通过谓词约束
（Predicate constraints）在损失函数中引入样本间的统计不变量。
理论框架为 Vapnik & Izmailov 的 LUSI（Learning Using Statistical
Invariants），实现形式参考 TSIL。

LUSI 加权损失：
    loss = τ̂ · MSE + τ · (1/N) · (φ̃ᵀ e)²

其中 φ̃ 是 L2 归一化后的谓词描述向量，e = sigmoid(logits) - target 是
概率残差向量。τ̂ + τ = 1，τ 通过可学习参数 α 经 sigmoid 得到。

当没有谓词约束时，使用标准的 BCEWithLogitsLoss 进行二分类训练。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
            "MLP model requires PyTorch. Install it with: pip install torch"
        )


# ============================================================================
# LUSI 谓词约束加权损失函数 — 从共享 training 模块重新导出
# ============================================================================

# 重新导出以保持向后兼容性；规范位置在 src.training.losses
from ..training.losses import (  # noqa: F401
    _build_combined_phi,
    _normalize_phi,
    weighted_mse_with_predicate,
)


# ============================================================================
# PyTorch MLP 模块
# ============================================================================

if _TORCH_AVAILABLE:

    class MPM_MLP(nn.Module):
        """面向 MPM 表格数据的多层感知机。

        Parameters
        ----------
        n_features : int
            输入特征维度。
        hidden_layers : list[int]
            隐藏层维度列表，默认 [64, 32]。
        dropout : float
            Dropout 比率，默认 0.3。
        use_batch_norm : bool
            是否使用 BatchNorm，默认 True。
        """

        def __init__(
            self,
            n_features: int,
            hidden_layers: list[int] | None = None,
            dropout: float = 0.3,
            use_batch_norm: bool = True,
        ):
            super().__init__()
            if hidden_layers is None:
                hidden_layers = [64, 32]

            self.n_features = n_features
            self.hidden_layers = hidden_layers
            self.dropout = dropout
            self.use_batch_norm = use_batch_norm

            layers: list[nn.Module] = []
            in_dim = n_features
            for out_dim in hidden_layers:
                layers.append(nn.Linear(in_dim, out_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            # 输出层：单个 logit（二分类）
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

            # 可学习的 τ 参数（控制 P 矩阵权重）
            # α 初始化为 0 → sigmoid(0) = 0.5 → τ̂ = 0.5, τ = 0.5
            self.alpha = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """前向传播。

            Args:
                x: 输入特征，shape (batch_size, n_features)

            Returns:
                logits，shape (batch_size, 1)
            """
            return self.net(x)


# ============================================================================
# sklearn 兼容的 MLP 分类器
# ============================================================================


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """sklearn 兼容的 MLP 分类器，支持 LUSI 谓词约束。

    封装 PyTorch 训练循环，支持 sample_weight 和谓词约束。
    与 get_params/set_params 兼容，可用于 BayesSearchCV 超参数搜索。

    训练模式：
    - 无谓词约束：使用 BCEWithLogitsLoss（标准二分类）
    - 有谓词约束：使用 LUSI 加权 MSE（在 sigmoid 概率上计算）

    Parameters
    ----------
    hidden_layers : list[int]
        隐藏层维度列表，默认 [64, 32]。
    n_epochs : int
        训练轮数，默认 100。
    batch_size : int
        批大小，默认 64。
    learning_rate : float
        Adam 学习率，默认 1e-3。
    weight_decay : float
        L2 正则化系数，默认 1e-4。
    dropout : float
        Dropout 比率，默认 0.3。
    use_batch_norm : bool
        是否使用 BatchNorm，默认 True。
    tau_init : float
        τ 初始值（P 矩阵权重），范围 [0, 1]，默认 0.5。
    learn_tau : bool
        τ 是否可学习，默认 True。
    random_state : int or None
        随机种子。
    device : str
        训练设备，默认 "cpu"。

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        训练后发现的类别标签 [0, 1]。
    model_ : MPM_MLP
        训练后的 PyTorch 模型。
    scaler_ : StandardScaler
        特征标准化器。
    """

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        n_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        tau_init: float = 0.5,
        learn_tau: bool = True,
        random_state: int | None = None,
        device: str = "cpu",
    ):
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.tau_init = tau_init
        self.learn_tau = learn_tau
        self.random_state = random_state
        self.device = device

    def fit(self, X, y, sample_weight=None, constraints=None):
        """训练 MLP 模型。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练特征。
        y : array-like of shape (n_samples,)
            训练标签（0/1 二分类）。
        sample_weight : array-like of shape (n_samples,), optional
            样本权重。
        constraints : Mapping[str, Any], optional
            谓词约束字典，包含 phi_vector(s) 用于构建加权损失。

        Returns
        -------
        self : MLPClassifier
        """
        _require_torch()

        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32, ensure_min_samples=2)
        self.n_features_in_ = X.shape[1]

        # 标准化特征
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        # 标签编码
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        log = get_logger("mlp")
        log.info(
            "MLP training: %d samples, %d features, epochs=%d, batch=%d, lr=%.5f",
            len(X), self.n_features_in_, self.n_epochs, self.batch_size, self.learning_rate,
        )

        # 构建模型
        self.model_ = MPM_MLP(
            n_features=self.n_features_in_,
            hidden_layers=list(self.hidden_layers),
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
        )
        self.model_.to(self.device)

        # 设置 τ 初始值
        if not self.learn_tau:
            self.model_.alpha.requires_grad_(False)
        import math
        tau_clamped = max(0.01, min(0.99, self.tau_init))
        with torch.no_grad():
            self.model_.alpha.fill_(math.log(tau_clamped / (1 - tau_clamped)))

        # 委托给共享训练循环
        from ..training.trainer import TorchTrainingConfig, TorchTrainingLoop

        has_constraints = constraints is not None and (
            "phi_vector" in constraints or "phi_vectors" in constraints
        )
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
            self.model_,
            X,
            y_encoded,
            training_config,
            sample_weight=sample_weight,
            constraints=constraints,
            logger=log,
        )

        return self

    def predict_proba(self, X):
        """返回类别概率。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        _require_torch()
        check_is_fitted(self, "model_")

        X = check_array(X, accept_sparse=False, dtype=np.float32)
        X = self.scaler_.transform(X)
        X_tensor = torch.from_numpy(X).float().to(self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze(-1)
            proba_pos = torch.sigmoid(logits).cpu().numpy()
            proba_neg = 1 - proba_pos
            proba = np.column_stack((proba_neg, proba_pos))

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
        return {
            "hidden_layers": self.hidden_layers,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "tau_init": self.tau_init,
            "learn_tau": self.learn_tau,
            "random_state": self.random_state,
            "device": self.device,
        }

    def set_params(self, **params):
        """设置模型参数，用于 BayesSearchCV。"""
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def __getstate__(self):
        """序列化：将 PyTorch state_dict 转为 CPU tensor。"""
        _require_torch()
        state = self.__dict__.copy()
        if "model_" in state and state["model_"] is not None:
            state["_model_state_dict"] = {
                k: v.cpu().clone() for k, v in state["model_"].state_dict().items()
            }
            del state["model_"]
        if "scaler_" in state and state["scaler_"] is not None:
            state["_scaler_mean"] = state["scaler_"].mean_.copy()
            state["_scaler_scale"] = state["scaler_"].scale_.copy()
            del state["scaler_"]
        return state

    def __setstate__(self, state):
        """反序列化：从 state_dict 恢复 PyTorch 模型。"""
        _require_torch()
        model_state = state.pop("_model_state_dict", None)
        scaler_mean = state.pop("_scaler_mean", None)
        scaler_scale = state.pop("_scaler_scale", None)
        self.__dict__.update(state)
        if model_state:
            # 从 state_dict 推断 n_features
            first_weight = model_state.get("net.0.weight")
            n_features = first_weight.shape[1] if first_weight is not None else 138
            hidden_layers = getattr(self, "hidden_layers", [64, 32])
            model = MPM_MLP(
                n_features=n_features,
                hidden_layers=list(hidden_layers),
                dropout=getattr(self, "dropout", 0.3),
                use_batch_norm=getattr(self, "use_batch_norm", True),
            )
            model.load_state_dict(model_state)
            self.model_ = model
        if scaler_mean is not None and scaler_scale is not None:
            self.scaler_ = StandardScaler()
            self.scaler_.mean_ = scaler_mean
            self.scaler_.scale_ = scaler_scale


# ============================================================================
# MLPAdapter — MPM 框架适配器
# ============================================================================


@MODEL_REGISTRY.decorator("mlp")
class MLPAdapter:
    """MLP 模型适配器，将 PyTorch MLP 集成到 MPM 实验框架。

    支持谓词约束（predicate constraints），通过 LUSI 谓词约束加权损失
    在训练中引入样本间的统计不变量。

    Attributes
    ----------
    name : str
        注册名称 "mlp"。
    artifact_filename : str
        序列化模型文件名。
    supports_constraints : bool
        MLP 支持谓词约束。
    """

    name = "mlp"
    artifact_filename = "model_mlp.pkl"
    supports_constraints = True

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        """校验 MLP 配置参数的合法性。"""
        n_epochs = int(params.get("n_epochs", 100))
        if n_epochs < 1:
            raise ValueError("mlp n_epochs must be positive")
        batch_size = int(params.get("batch_size", 64))
        if batch_size < 1:
            raise ValueError("mlp batch_size must be positive")
        learning_rate = float(params.get("learning_rate", 1e-3))
        if learning_rate <= 0:
            raise ValueError("mlp learning_rate must be positive")
        dropout = float(params.get("dropout", 0.3))
        if not 0 <= dropout < 1:
            raise ValueError("mlp dropout must be in [0, 1)")
        hidden_layers = params.get("hidden_layers", [64, 32])
        if not isinstance(hidden_layers, list) or len(hidden_layers) == 0:
            raise ValueError("mlp hidden_layers must be a non-empty list of integers")
        for dim in hidden_layers:
            if not isinstance(dim, int) or dim < 1:
                raise ValueError(f"mlp hidden layer dimension must be positive, got {dim}")
        tau_init = float(params.get("tau_init", 0.5))
        if not 0 < tau_init < 1:
            raise ValueError("mlp tau_init must be in (0, 1)")

    def build(self, params: Mapping[str, Any], seed: int) -> MLPClassifier:
        """根据配置参数构建 MLPClassifier 实例。

        Parameters
        ----------
        params : Mapping[str, Any]
            模型配置参数（来自 YAML 配置的 model.params 节）。
        seed : int
            实验级随机种子。

        Returns
        -------
        MLPClassifier
        """
        resolved = dict(params)

        return MLPClassifier(
            hidden_layers=list(resolved.get("hidden_layers", [64, 32])),
            n_epochs=int(resolved.get("n_epochs", 100)),
            batch_size=int(resolved.get("batch_size", 64)),
            learning_rate=float(resolved.get("learning_rate", 1e-3)),
            weight_decay=float(resolved.get("weight_decay", 1e-4)),
            dropout=float(resolved.get("dropout", 0.3)),
            use_batch_norm=bool(resolved.get("use_batch_norm", True)),
            tau_init=float(resolved.get("tau_init", 0.5)),
            learn_tau=bool(resolved.get("learn_tau", True)),
            random_state=seed,
            device=str(resolved.get("device", "cpu")),
        )

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        """返回传递给模型 fit() 方法的额外参数。

        包含 sample_weight 和谓词约束（如果存在）。

        Parameters
        ----------
        data : TrainingData
            训练数据包。

        Returns
        -------
        Mapping[str, Any]
            包含 sample_weight 和可选的 constraints 的字典。
        """
        kwargs: dict[str, Any] = {"sample_weight": data.sample_weight}
        if data.constraints:
            kwargs["constraints"] = dict(data.constraints)
        return kwargs