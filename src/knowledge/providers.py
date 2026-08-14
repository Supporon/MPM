"""内置 Knowledge Provider；后续领域知识模块可在此注册。"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.contracts import TrainingData
from .registry import KNOWLEDGE_REGISTRY


@KNOWLEDGE_REGISTRY.decorator("empty")
class EmptyKnowledgeProvider:
    """用于验证 Knowledge 扩展链路的空知识提供器。"""

    name = "empty"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    def build(self, data: TrainingData, context: Mapping[str, Any]) -> Mapping[str, Any]:
        return {}
