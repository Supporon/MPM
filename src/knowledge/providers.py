"""内置 Knowledge Provider；后续领域知识模块可在此注册。

知识提供器（Knowledge Provider）读取并版本化地质知识来源，产出带
provenance（来源、CRS、空间支撑、有效范围）的知识工件。命名空间约定为
``knowledge[provider_name][artifact_name]``，谓词通过 ``context["knowledge"]``
消费这些工件。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

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


def _extract_coordinates(data: TrainingData) -> np.ndarray | None:
    """从 TrainingData 中提取 X/Y 坐标，优先 metadata 的 units 表。

    与 ``src/predicates/builtins._get_coordinates`` 的提取策略一致，
    但独立实现以避免 knowledge 模块反向依赖 predicates。

    Returns:
        shape (N, 2) 的坐标数组 [X, Y]，或 None（无坐标来源）。
    """
    import pandas as pd

    units = data.metadata.get("units")
    if isinstance(units, pd.DataFrame):
        x_col = next((c for c in units.columns if c.upper() == "X"), None)
        y_col = next((c for c in units.columns if c.upper() == "Y"), None)
        if x_col and y_col:
            return np.column_stack([units[x_col].to_numpy(), units[y_col].to_numpy()])

    features = data.features
    x_col = next((c for c in features.columns if c.upper() == "X"), None)
    y_col = next((c for c in features.columns if c.upper() == "Y"), None)
    if x_col and y_col:
        return np.column_stack(
            [features[x_col].to_numpy(), features[y_col].to_numpy()]
        )

    return None


@KNOWLEDGE_REGISTRY.decorator("spatial_extent")
class SpatialExtentKnowledgeProvider:
    """从研究单元坐标提取研究区域空间范围的知识提供器。

    产出带 provenance 的知识工件：
        - ``bounds``：{min_x, min_y, max_x, max_y} 有效范围
        - ``center``：{x, y} 区域中心（中位数，对离群坐标稳健）
        - ``span``：{x, y} 范围跨度
        - ``n_units``：参与统计的研究单元数量（空间支撑）
        - ``source``：坐标来源（``research_unit_metadata`` / ``feature_columns``）

    该工件可供 spatial_box / spatial_distance 等空间谓词消费，
    用研究区域的先验范围替代从数据重新估计。

    Parameters
    ----------
    params : Mapping[str, Any]
        可选参数：
        - center_statistic : "median"（默认）或 "mean"，决定中心点统计量。
        - crs : str | None，研究单元坐标的 CRS 标识（仅记录 provenance，不重投影）。
    """

    name = "spatial_extent"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        center_statistic = str(params.get("center_statistic", "median"))
        if center_statistic not in {"median", "mean"}:
            raise ValueError(
                f"spatial_extent center_statistic must be 'median' or 'mean', "
                f"got {center_statistic!r}"
            )

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)
        self.validate_config(self.params)
        self.center_statistic = str(self.params.get("center_statistic", "median"))

    def build(self, data: TrainingData, context: Mapping[str, Any]) -> Mapping[str, Any]:
        coords = _extract_coordinates(data)
        if coords is None:
            # 无可用的坐标来源时返回空工件，由消费方决定是否回退。
            return {}

        if coords.shape[0] == 0:
            return {}

        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        if self.center_statistic == "median":
            center_x, center_y = np.median(coords, axis=0)
        else:
            center_x, center_y = coords.mean(axis=0)

        units = data.metadata.get("units")
        source = "research_unit_metadata" if units is not None else "feature_columns"

        return {
            "bounds": {
                "min_x": float(min_x),
                "min_y": float(min_y),
                "max_x": float(max_x),
                "max_y": float(max_y),
            },
            "center": {"x": float(center_x), "y": float(center_y)},
            "span": {
                "x": float(max_x - min_x),
                "y": float(max_y - min_y),
            },
            "n_units": int(coords.shape[0]),
            "source": source,
            "crs": self.params.get("crs"),
        }
