"""顺序执行的 Predicate 管线。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..core.bootstrap import load_builtin_components
from ..core.contracts import TrainingData
from ..core.spec import ComponentSpec
from .registry import PREDICATE_REGISTRY


class PredicatePipeline:
    """依次应用 Predicate，并保证训练数据行数契约不被破坏。"""

    def __init__(self, specs: Iterable[ComponentSpec]):
        load_builtin_components()
        self.specs = tuple(specs)
        self.items = [PREDICATE_REGISTRY.create(spec.name, spec.params) for spec in self.specs]

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        result = data
        for predicate in self.items:
            result = predicate.apply(result, context)
            if len(result.features) != len(data.features):
                raise ValueError(
                    f"Predicate '{predicate.name}' changed row count; row-changing predicates require an explicit resampling contract"
                )
        return result
