"""构造当前基于点的研究单元和预测单元。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .labels import positive_labels_from_size_code, unlabeled_labels


def load_vector(path: str):
    """使用 GeoPandas 加载矢量数据集，或说明缺失的依赖。"""
    try:
        import geopandas as gpd
    except ImportError as error:  # pragma: no cover - 与环境相关
        raise RuntimeError("raw_gis research-unit construction requires geopandas") from error
    return gpd.read_file(path)


def build_positive_units(occurrence_path: str, label_config: Mapping[str, Any]) -> pd.DataFrame:
    """将已标注矿点几何转换为基于点的训练单元。"""
    occurrences = load_vector(occurrence_path)
    labelled = positive_labels_from_size_code(occurrences, label_config)
    return pd.DataFrame(
        {
            "X": labelled.geometry.x,
            "Y": labelled.geometry.y,
            "label": labelled["label"],
            "sample_weight": labelled["sample_weight"],
        }
    ).reset_index(drop=True)


def sample_unlabeled_units(
    boundary_path: str, count: int, label_config: Mapping[str, Any], seed: int | None = None
) -> pd.DataFrame:
    """在边界内执行均匀采样，返回未标注点。

    Parameters
    ----------
    boundary_path : str
        边界 Shapefile 路径。
    count : int
        待生成的未标注样本数量。
    label_config : Mapping[str, Any]
        标签配置，包含 sample_weight 映射。
    seed : int or None
        随机种子，用于确保可复现性。若为 None，使用全局 numpy 随机状态。

    TODO(science)：未使用正例排除缓冲区或经验证的无矿区掩膜。
    """
    boundary = load_vector(boundary_path)
    if boundary.empty:
        raise ValueError("Boundary vector contains no geometry")
    geometry = boundary.geometry.iloc[0]
    min_x, min_y, max_x, max_y = geometry.bounds
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    candidates_x = rng.uniform(low=min_x, high=max_x, size=count * 2)
    candidates_y = rng.uniform(low=min_y, high=max_y, size=count * 2)
    accepted_x: list[float] = []
    accepted_y: list[float] = []
    try:
        from shapely.geometry import Point
    except ImportError as error:  # pragma: no cover - 依赖 GeoPandas
        raise RuntimeError("raw_gis research-unit construction requires shapely") from error
    for x, y in zip(candidates_x, candidates_y):
        if len(accepted_x) == count:
            break
        if Point(x, y).within(geometry):
            accepted_x.append(float(x))
            accepted_y.append(float(y))
    return unlabeled_labels(pd.DataFrame({"X": accepted_x, "Y": accepted_y}), label_config)


def build_prediction_grid(boundary_path: str, grid_size: float) -> tuple[pd.DataFrame, np.ndarray]:
    """创建与 Notebook 相同的经多边形筛选的规则坐标网格。"""
    boundary = load_vector(boundary_path)
    if boundary.empty:
        raise ValueError("Boundary vector contains no geometry")
    geometry = boundary.geometry.iloc[0]
    min_x, min_y, max_x, max_y = geometry.bounds
    range_x = np.arange(min_x, max_x, grid_size)
    range_y = np.arange(min_y, max_y, grid_size)
    mesh_x, mesh_y = np.meshgrid(range_x, range_y)
    grid = pd.DataFrame({"X": mesh_x.ravel(), "Y": mesh_y.ravel()})
    try:
        from shapely.geometry import Point
    except ImportError as error:  # pragma: no cover - 依赖 GeoPandas
        raise RuntimeError("raw_gis research-unit construction requires shapely") from error
    mask = np.array([Point(x, y).within(geometry) for x, y in zip(grid["X"], grid["Y"])], dtype=bool)
    return grid.loc[mask].reset_index(drop=True), np.column_stack((grid["X"], grid["Y"], mask))
