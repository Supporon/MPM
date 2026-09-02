"""共享训练模块 — TSIL 风格损失函数与 PyTorch 训练循环。

提供所有深度学习模型可复用的：
- TSIL 加权 MSE 损失函数
- 谓词 φ 向量构建与归一化
- PyTorch 训练循环（DataLoader → epoch → batch → loss → backward）
"""

from .losses import _build_combined_phi, _normalize_phi, weighted_mse_with_predicate
from .trainer import TorchTrainingConfig, TorchTrainingLoop

__all__ = [
    "_normalize_phi",
    "weighted_mse_with_predicate",
    "_build_combined_phi",
    "TorchTrainingConfig",
    "TorchTrainingLoop",
]