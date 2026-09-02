"""共享 PyTorch 训练循环。

封装 DataLoader → epoch → batch → loss → backward 的通用流程，
供所有 sklearn 兼容的 PyTorch 模型复用。

设计原则：
- 无状态工具类：每次 fit() 调用是独立的
- 不包装模型：接收 nn.Module 并在其上就地训练，保持 sklearn 兼容性
- 双模式：标准模式（BCE/CrossEntropy）和谓词模式（TSIL 加权 MSE）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .losses import _build_combined_phi, weighted_mse_with_predicate


@dataclass
class TorchTrainingConfig:
    """PyTorch 训练配置，独立于具体模型架构。

    Attributes:
        n_epochs: 训练轮数。
        batch_size: 批大小。
        learning_rate: Adam 学习率。
        weight_decay: L2 正则化系数。
        device: 训练设备。
        random_state: 随机种子。
        log_interval: 日志输出间隔（epoch），默认 n_epochs // 10。
    """

    n_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    random_state: int | None = None
    log_interval: int | None = None


class TorchTrainingLoop:
    """共享 PyTorch 训练循环。

    封装了所有深度学习模型通用的训练流程：
    - DataLoader 构建
    - Optimizer 初始化
    - Epoch / batch 迭代
    - Loss 计算（标准模式 vs 谓词模式）
    - Backward + optimizer step
    - 日志输出

    使用方式（在 sklearn 兼容的 fit() 方法中）:

        config = TorchTrainingConfig(n_epochs=100, batch_size=64, ...)
        TorchTrainingLoop.fit(
            self.model_, X, y_encoded, config,
            sample_weight=sample_weight,
            constraints=constraints,
            logger=log,
        )

    模型要求:
    - model.forward(x) 返回 logits
    - 谓词模式下 model.alpha 必须是 nn.Parameter（用于 tau 计算）
    """

    @staticmethod
    def fit(
        model: "nn.Module",
        X: np.ndarray,
        y: np.ndarray,
        config: TorchTrainingConfig,
        *,
        sample_weight: np.ndarray | None = None,
        constraints: Mapping[str, Any] | None = None,
        criterion: "nn.Module | None" = None,
        logger: Any = None,
    ) -> None:
        """在 model 上就地训练。

        Args:
            model: PyTorch nn.Module，必须已有 forward() 方法。
            X: 训练特征，shape (n_samples, n_features)。
            y: 训练标签，shape (n_samples,)。
            config: 训练配置。
            sample_weight: 样本权重，shape (n_samples,)。
            constraints: 谓词约束字典，包含 phi_vector(s)。
            criterion: 标准模式下的 loss 函数（默认 BCEWithLogitsLoss）。
            logger: 可选的 logger 实例。
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("TorchTrainingLoop requires PyTorch.")

        n_samples = len(X)

        # 设置随机种子
        if config.random_state is not None:
            torch.manual_seed(config.random_state)
            np.random.seed(config.random_state)

        # 准备谓词 φ 数据
        has_constraints = constraints is not None and (
            "phi_vector" in constraints or "phi_vectors" in constraints
        )
        if has_constraints:
            phi_full = _build_combined_phi(constraints, n_samples, config.device)
        else:
            phi_full = None

        # 转换为 tensor
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        if sample_weight is not None:
            sw_tensor = torch.from_numpy(
                np.asarray(sample_weight, dtype=np.float32)
            ).float()
        else:
            sw_tensor = torch.ones(len(X_tensor))

        # 构建 DataLoader
        if phi_full is not None:
            dataset = TensorDataset(X_tensor, y_tensor, sw_tensor, phi_full)
        else:
            dataset = TensorDataset(X_tensor, y_tensor, sw_tensor)

        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # 构建 optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 标准模式：默认使用 BCEWithLogitsLoss
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss(reduction="none")

        # 日志间隔
        log_interval = config.log_interval or max(1, config.n_epochs // 10)

        model.train()

        for epoch in range(config.n_epochs):
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_p_loss = 0.0
            n_batches = 0

            for batch in loader:
                if has_constraints:
                    batch_X, batch_y, batch_w, batch_phi = batch
                    batch_phi = batch_phi.to(config.device)
                else:
                    batch_X, batch_y, batch_w = batch
                    batch_phi = None

                batch_X = batch_X.to(config.device)
                batch_y = batch_y.to(config.device)
                batch_w = batch_w.to(config.device)

                optimizer.zero_grad()
                logits = model(batch_X).squeeze(-1)  # [B]

                if has_constraints and batch_phi is not None:
                    # TSIL 谓词模式：sigmoid 概率 → 加权 MSE
                    pred_prob = torch.sigmoid(logits)
                    tau_hat = torch.sigmoid(model.alpha)
                    tau = 1 - tau_hat
                    loss_dict = weighted_mse_with_predicate(
                        pred_prob, batch_y, batch_phi, tau_hat, tau, sample_weight=batch_w
                    )
                    loss = loss_dict["total"]
                else:
                    # 标准模式：BCEWithLogitsLoss（或其他 criterion）
                    losses = criterion(logits, batch_y)
                    loss = (losses * batch_w).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if has_constraints and batch_phi is not None:
                    epoch_mse += loss_dict["mse"].item()
                    epoch_p_loss += loss_dict["p_loss"].item()
                n_batches += 1

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                if has_constraints and logger is not None:
                    logger.info(
                        "Epoch %d/%d, loss=%.6f, mse=%.6f, p_loss=%.6f",
                        epoch + 1,
                        config.n_epochs,
                        avg_loss,
                        epoch_mse / max(n_batches, 1),
                        epoch_p_loss / max(n_batches, 1),
                    )
                elif logger is not None:
                    logger.info(
                        "Epoch %d/%d, loss=%.6f",
                        epoch + 1,
                        config.n_epochs,
                        avg_loss,
                    )

        # 记录最终的 τ 值
        if has_constraints and logger is not None:
            final_tau = torch.sigmoid(model.alpha).item()
            logger.info("Training complete. Final tau=%.4f", final_tau)