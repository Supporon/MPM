"""用于验证扩展契约且不改变科学方法的最小内置 Predicate。"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.contracts import TrainingData
from .registry import PREDICATE_REGISTRY


@PREDICATE_REGISTRY.decorator("identity")
class IdentityPredicate:
    """不执行实际变换，适用于连线测试和新插件模板。"""

    name = "identity"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        metadata = dict(data.metadata)
        applied = list(metadata.get("predicates", []))
        applied.append(self.name)
        return TrainingData(
            data.features,
            data.labels,
            data.sample_weight,
            constraints=dict(data.constraints),
            metadata={**metadata, "predicates": applied},
        )
