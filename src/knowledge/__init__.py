"""外部知识与领域知识扩展点。"""

from .pipeline import KnowledgePipeline
from .registry import KNOWLEDGE_REGISTRY

__all__ = ["KnowledgePipeline", "KNOWLEDGE_REGISTRY"]
