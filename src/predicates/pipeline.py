"""顺序执行的 Predicate 管线，按 kind 分离数据变换与模型约束。

管线执行顺序：
    1. data_transform 谓词：修改 TrainingData 的标签/权重/特征
    2. constraint 谓词：生成 phi 向量，添加到 constraints 中

所有谓词必须保持行数不变（row-count contract）。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..core.bootstrap import load_builtin_components
from ..core.contracts import TrainingData
from ..core.spec import ComponentSpec
from .registry import PREDICATE_REGISTRY


class PredicatePipeline:
    """按 kind 分离数据变换与约束生成，依次应用。

    谓词分为两类：
    - ``data_transform``：修改标签、权重或特征（如 positive_constraint、weight_adjustment）
    - ``constraint``：生成 phi 向量，通过 constraints 传递给支持 TSIL 的模型
      （如 all_ones、spatial_box、spatial_distance、combined）

    执行顺序保证 data_transform 先于 constraint，确保约束基于最终变换后的数据。
    """

    def __init__(self, specs: Iterable[ComponentSpec]):
        load_builtin_components()
        self.specs = tuple(specs)
        self.items = [PREDICATE_REGISTRY.create(spec.name, spec.params) for spec in self.specs]

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    @property
    def data_transform_predicates(self) -> list:
        """data_transform 谓词 + 未声明 kind 的谓词（向后兼容，默认视为 data_transform）。"""
        return [p for p in self.items if getattr(p, "kind", "data_transform") == "data_transform"]

    @property
    def constraint_predicates(self) -> list:
        return [p for p in self.items if getattr(p, "kind", None) == "constraint"]

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        result = data
        # Phase 1: 数据变换谓词（修改标签/权重）
        for predicate in self.data_transform_predicates:
            result = predicate.apply(result, context)
            if len(result.features) != len(data.features):
                raise ValueError(
                    f"Predicate '{predicate.name}' changed row count; "
                    f"row-changing predicates require an explicit resampling contract"
                )
        # Phase 2: 约束谓词（生成 phi 向量）
        for predicate in self.constraint_predicates:
            result = predicate.apply(result, context)
            if len(result.features) != len(data.features):
                raise ValueError(
                    f"Predicate '{predicate.name}' changed row count; "
                    f"row-changing predicates require an explicit resampling contract"
                )
        return result