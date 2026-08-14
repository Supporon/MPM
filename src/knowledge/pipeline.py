"""为 Predicate 或 Model 构建带命名空间的知识工件，避免混淆两类概念。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..core.bootstrap import load_builtin_components
from ..core.contracts import TrainingData
from ..core.spec import ComponentSpec
from .registry import KNOWLEDGE_REGISTRY


class KnowledgePipeline:
    """按配置构建知识工件，并以 Provider 名称组织结果。"""

    def __init__(self, specs: Iterable[ComponentSpec]):
        load_builtin_components()
        self.specs = tuple(specs)
        self.providers = [KNOWLEDGE_REGISTRY.create(spec.name, spec.params) for spec in self.specs]

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def build(self, data: TrainingData, context: Mapping[str, Any]) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        for spec, provider in zip(self.specs, self.providers):
            if spec.name in artifacts:
                raise ValueError(f"Duplicate knowledge provider name: {spec.name}")
            artifacts[spec.name] = dict(provider.build(data, context))
        return artifacts
