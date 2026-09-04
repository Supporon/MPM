#!/usr/bin/env python3
"""约束重加权的随机森林（RF + 谓词约束，方案 B）。

RF 无梯度，无法直接加 LUSI 约束项；这里用通用组件
``ConstrainedReweighting`` 包装 RandomForestClassifier，把约束翻译成样本权重
迭代重训练。
"""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier

from ..core.contracts import TrainingData
from ..training.constrained import ConstrainedReweighting
from .registry import MODEL_REGISTRY


@MODEL_REGISTRY.decorator("rf_constrained")
class RfConstrainedAdapter:
    """约束重加权随机森林适配器。"""

    name = "rf_constrained"
    artifact_filename = "model_rf_constrained.pkl"
    supports_constraints = True

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        rf_params = {k: v for k, v in params.items() if k not in {"n_iter", "eta", "tol"}}
        RandomForestClassifier(**rf_params)

    def build(self, params: Mapping[str, Any], seed: int) -> ConstrainedReweighting:
        rf_params = {k: v for k, v in params.items() if k not in {"n_iter", "eta", "tol"}}
        return ConstrainedReweighting(
            base_estimator=RandomForestClassifier(**rf_params, random_state=seed),
            n_iter=int(params.get("n_iter", 5)),
            eta=float(params.get("eta", 0.5)),
            tol=float(params.get("tol", 1e-3)),
        )

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {"sample_weight": data.sample_weight}
        if data.constraints:
            kwargs["constraints"] = dict(data.constraints)
        return kwargs
