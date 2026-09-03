"""研究单元构建器：将原始 GIS 数据转换为带 X/Y 坐标的点单元。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .label_strategies import PositiveUnlabeledAsZeroStrategy
from .registries import LABEL_STRATEGY_REGISTRY, RESEARCH_UNIT_REGISTRY
from .samplers import _boundary_union, load_vector


@RESEARCH_UNIT_REGISTRY.decorator("point_local_environment")
class PointLocalEnvironmentUnit:
    """基于点的局部环境研究单元。

    正样本来自矿点（occurrence），预测单元来自规则网格（经边界筛选）。
    该研究单元类型对应 Notebook 的 ``point_local_environment`` 语义。
    """

    name = "point_local_environment"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        pass

    def build_positive_units(
        self, occurrence_path: str, label_config: Mapping[str, Any]
    ) -> pd.DataFrame:
        occurrences = load_vector(occurrence_path)
        label_strategy = LABEL_STRATEGY_REGISTRY.create(
            label_config.get("strategy", "positive_unlabeled_as_zero"), {}
        )
        labelled = label_strategy.apply_positive(occurrences, label_config)
        return pd.DataFrame(
            {
                "X": labelled.geometry.x,
                "Y": labelled.geometry.y,
                "label": labelled["label"],
                "sample_weight": labelled["sample_weight"],
            }
        ).reset_index(drop=True)

    def build_prediction_units(
        self, boundary_path: str, grid_size: float
    ) -> tuple[pd.DataFrame, np.ndarray]:
        boundary = load_vector(boundary_path)
        geometry = _boundary_union(boundary)
        min_x, min_y, max_x, max_y = geometry.bounds
        range_x = np.arange(min_x, max_x, grid_size)
        range_y = np.arange(min_y, max_y, grid_size)
        mesh_x, mesh_y = np.meshgrid(range_x, range_y)
        grid = pd.DataFrame({"X": mesh_x.ravel(), "Y": mesh_y.ravel()})
        try:
            from shapely.geometry import Point
        except ImportError as error:  # pragma: no cover - 依赖 GeoPandas
            raise RuntimeError("raw_gis research-unit construction requires shapely") from error
        mask = np.array(
            [Point(x, y).within(geometry) for x, y in zip(grid["X"], grid["Y"])],
            dtype=bool,
        )
        return grid.loc[mask].reset_index(drop=True), np.column_stack(
            (grid["X"], grid["Y"], mask)
        )
