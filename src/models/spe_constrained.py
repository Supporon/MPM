#!/usr/bin/env python3
"""约束重加权的 SPE（自步集成 + 谓词约束，方案 B）。

用通用组件 ``ConstrainedReweighting`` 包装 SelfPacedEnsemble，把谓词约束翻译成
样本权重迭代重训练。SPE 本身是处理不平衡的集成方法，这里验证"约束重加权"这一
通用机制在非梯度集成方法上的可复用性。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.contracts import TrainingData
from ..training.constrained import ConstrainedReweighting
from .registry import MODEL_REGISTRY
from .spe import SPEAdapter


@MODEL_REGISTRY.decorator("spe_constrained")
class SpeConstrainedAdapter:
    """约束重加权 SPE 适配器。"""

    name = "spe_constrained"
    artifact_filename = "model_spe_constrained.pkl"
    supports_constraints = True

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        SPEAdapter.validate_config(params)
        if int(params.get("n_iter", 5)) < 1:
            raise ValueError("spe_constrained n_iter must be positive")

    def build(self, params: Mapping[str, Any], seed: int) -> ConstrainedReweighting:
        base = SPEAdapter().build(params, seed)
        return ConstrainedReweighting(
            base_estimator=base,
            n_iter=int(params.get("n_iter", 5)),
            eta=float(params.get("eta", 0.5)),
            tol=float(params.get("tol", 1e-3)),
        )

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {"sample_weight": data.sample_weight}
        if data.constraints:
            kwargs["constraints"] = dict(data.constraints)
        return kwargs
