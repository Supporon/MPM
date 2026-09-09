"""LUSI（Learning Using Statistical Invariants）谓词约束加权损失函数。

理论框架是 Vapnik & Izmailov 的 LUSI——通过谓词（predicate）将数据中
应满足的统计不变量引入学习目标；实现形式参考 TSIL。

提供所有深度学习模型可复用的谓词约束损失函数：
- φ 向量 L2 归一化
- LUSI 加权 MSE 损失（MSE + τ·P_loss，τ 为固定约束系数）
- 从 TrainingData constraints 构建合并的 φ 向量

数学原理（Vapnik & Izmailov 的 LUSI；实现参考 TSIL）:
    φ 向量 → L2 归一化 φ̃ = φ / ||φ||
    P = φ̃ φ̃ᵀ（投影矩阵）
    loss = MSE + τ · (1/N) · ||φ̃ᵀ e||²

P 矩阵将具有相似谓词特征的样本聚合在一起，在损失计算中引入
样本间的统计不变量。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _normalize_phi(phi: "torch.Tensor") -> "torch.Tensor":
    """L2 归一化 φ 向量，匹配 LUSI 中的 Φ̃ = Φ / ||Φ||。"""
    return phi / (torch.linalg.norm(phi) + 1e-8)


def weighted_mse_with_predicate(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    phi: "torch.Tensor",
    tau: float,
    sample_weight: "torch.Tensor | None" = None,
) -> dict[str, "torch.Tensor"]:
    """LUSI 谓词约束加权 MSE 损失（固定约束系数）。

    数学形式：
        loss = MSE + τ · (1/N) · (φ̃ᵀ e)²

    其中 P_loss = (1/N) · (φ̃ᵀ e)² 等价于 (1/N) · eᵀ P e，
    P = φ̃ φ̃ᵀ 是投影矩阵。``sample_weight`` 应用于个体误差项 MSE。

    ``tau`` 是**固定的**谓词（统计不变量）项系数，由调用方作为超参数传入，
    不参与梯度优化。早期实现把约束系数做成可学习参数 ``sigmoid(alpha)``，
    由于约束项 ``P_loss ≤ MSE``（对等权样本、归一化谓词），梯度下降倾向
    通过减小监督项系数来降低总目标，而非改善预测（P0-03 退化方向）；固定
    系数消除了这一退化方向，使 lambda=0 与有谓词分支的监督项完全一致。

    Args:
        pred: 预测概率值（已通过 sigmoid），shape [N]
        target: 目标值，shape [N]
        phi: 谓词描述向量，shape [N] 或 [N, d]
        tau: 谓词项系数（统计不变量权重），>= 0，固定不参与优化
        sample_weight: 逐样本权重，shape [N]；None 表示等权。

    Returns:
        dict with 'total', 'mse', 'v_loss', 'p_loss', 'p_loss_raw'
    """
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    e = pred - target  # [N]
    N = e.shape[0]

    if sample_weight is not None:
        w = sample_weight
        mse = (w * e**2).sum() / w.sum()
    else:
        mse = (e**2).mean()

    # 归一化 φ 并计算 P 项
    if phi.dim() == 1:
        phi = phi.unsqueeze(1)  # [N, 1]
    phi_norm = _normalize_phi(phi)  # [N, d]

    # P_loss = (1/N) * ||φ̃ᵀ e||² = (1/N) * Σ_d (φ̃_{:,d} · e)²
    phi_T_e = torch.mv(phi_norm.T, e)  # [d]
    p_loss_raw = (phi_T_e**2).sum() / N

    v_loss = mse
    p_term = tau * p_loss_raw
    total = v_loss + p_term

    return {
        "total": total,
        "mse": mse,
        "v_loss": v_loss,
        "p_loss": p_term,
        "p_loss_raw": p_loss_raw,
    }


def _build_combined_phi(
    constraints: Mapping[str, Any], n_samples: int, device: "torch.device"
) -> "torch.Tensor":
    """从 TrainingData.constraints 构建合并的 φ 向量。

    支持两种约束格式：
    1. 简单格式: {"phi_vector": np.ndarray}
    2. 多谓词格式: {"phi_vectors": [{"name": ..., "vector": np.ndarray}, ...]}

    多个 φ 向量通过拼接并在 L2 归一化后返回。
    """
    phi_list = []

    # 接受旧版单向量输入以及规范化后的 phi_vectors。
    if "phi_vector" in constraints and "phi_vectors" in constraints:
        raise ValueError("constraints cannot contain both phi_vector and phi_vectors")

    # 格式 1: 直接的 phi_vector
    if "phi_vector" in constraints:
        vec = np.asarray(constraints["phi_vector"], dtype=np.float32)
        if len(vec) != n_samples:
            raise ValueError(
                f"phi_vector length {len(vec)} != n_samples {n_samples}"
            )
        phi_list.append(torch.from_numpy(vec).to(device))

    # 格式 2: 多个 phi_vectors
    if "phi_vectors" in constraints:
        for entry in constraints["phi_vectors"]:
            vec = np.asarray(entry["vector"], dtype=np.float32)
            if len(vec) != n_samples:
                raise ValueError(
                    f"phi_vector '{entry.get('name', 'unknown')}' length "
                    f"{len(vec)} != n_samples {n_samples}"
                )
            phi_list.append(torch.from_numpy(vec).to(device))

    if not phi_list:
        # 默认：全 1 谓词（Φ_ones 控制谓词）
        return torch.ones(n_samples, device=device)

    # 拼接所有 φ 向量
    if len(phi_list) == 1:
        return phi_list[0]
    else:
        return torch.stack(phi_list, dim=1)