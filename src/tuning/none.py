"""不执行搜索的 Tuner：直接拟合已配置模型。"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.contracts import TrainingData
from ..models.registry import fit_params_for
from .registry import TUNER_REGISTRY


@TUNER_REGISTRY.decorator("none")
class NoSearchTuner:
    """跳过参数搜索，直接构建并拟合所选模型。"""

    name = "none"
    uses_cross_validation = False

    def fit(
        self,
        model_adapter,
        data: TrainingData,
        model_params: Mapping[str, Any],
        tuning_params: Mapping[str, Any],
        cv_config: Mapping[str, Any],
        scoring: str,
        seed: int,
    ):
        model = model_adapter.build(model_params, seed)
        model.fit(data.features, data.labels, **fit_params_for(model_adapter, data))
        return model
