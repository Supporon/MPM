"""Model 与可选标签细化组件的注册表。"""

from __future__ import annotations

from typing import Any

from ..core.registry import ComponentRegistry

MODEL_REGISTRY = ComponentRegistry("model")
LABEL_REFINER_REGISTRY = ComponentRegistry("label_refiner")


def fit_params_for(adapter: Any, data) -> dict[str, Any]:
    """解析估计器拟合参数，包括模型特定的约束载荷。"""
    if data.constraints and not getattr(adapter, "supports_constraints", False):
        raise ValueError(
            f"Model '{adapter.name}' does not support predicate constraints: {sorted(data.constraints)}"
        )
    if hasattr(adapter, "fit_params"):
        return dict(adapter.fit_params(data))
    if data.constraints:
        raise ValueError(
            f"Model '{adapter.name}' declares constraint support but does not implement fit_params(data)"
        )
    return {"sample_weight": data.sample_weight}
