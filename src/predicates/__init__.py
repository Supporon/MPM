"""独立于外部 Knowledge Provider 的 Predicate 扩展点。"""

from .pipeline import PredicatePipeline
from .registry import PREDICATE_REGISTRY

__all__ = ["PredicatePipeline", "PREDICATE_REGISTRY"]
