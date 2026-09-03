"""未标注/背景采样器：在边界内生成未标注点。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .registries import BACKGROUND_SAMPLER_REGISTRY


def load_vector(path: str):
    """使用 GeoPandas 加载矢量数据集，或说明缺失的依赖。"""
    try:
        import geopandas as gpd
    except ImportError as error:  # pragma: no cover - 与环境相关
        raise RuntimeError("raw_gis research-unit construction requires geopandas") from error
    return gpd.read_file(path)


def _require_crs(frame, what: str):
    """要求矢量数据带有 CRS，缺失时直接失败。"""
    if frame.crs is None:
        raise ValueError(
            f"{what} is missing a CRS. Assign a CRS (e.g. reproject to a metric "
            "projection) before constructing research units."
        )
    return frame.crs


def _boundary_union(boundary):
    """返回边界全部要素的几何并集（dissolve），而不是只取第一个要素。"""
    if boundary.empty:
        raise ValueError("Boundary vector contains no geometry")
    _require_crs(boundary, "Boundary vector")
    if hasattr(boundary.geometry, "union_all"):
        geometry = boundary.geometry.union_all()
    else:  # pragma: no cover - 兼容旧版 geopandas
        geometry = boundary.geometry.unary_union
    if geometry is None or geometry.is_empty:
        raise ValueError("Boundary vector contains no valid geometry")
    return geometry


@BACKGROUND_SAMPLER_REGISTRY.decorator("random_points_in_nsw_boundary")
class UniformBoundarySampler:
    """在边界内均匀采样未标注点。

    通过拒绝采样（rejection sampling）生成落在边界内的点，直到达到请求
    数量；达到最大尝试次数仍不足时明确失败，避免静默返回少于请求数量。
    """

    name = "random_points_in_nsw_boundary"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    def sample(self, boundary_path: str, count: int, seed: int | None = None) -> pd.DataFrame:
        boundary = load_vector(boundary_path)
        geometry = _boundary_union(boundary)
        min_x, min_y, max_x, max_y = geometry.bounds
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        try:
            from shapely.geometry import Point
        except ImportError as error:  # pragma: no cover - 依赖 GeoPandas
            raise RuntimeError("raw_gis research-unit construction requires shapely") from error

        accepted_x: list[float] = []
        accepted_y: list[float] = []
        max_attempts = max(count * 20, 1000)
        attempts = 0
        while len(accepted_x) < count and attempts < max_attempts:
            batch = min(count - len(accepted_x), 1000)
            candidates_x = rng.uniform(low=min_x, high=max_x, size=batch)
            candidates_y = rng.uniform(low=min_y, high=max_y, size=batch)
            for x, y in zip(candidates_x, candidates_y):
                attempts += 1
                if Point(x, y).within(geometry):
                    accepted_x.append(float(x))
                    accepted_y.append(float(y))
                    if len(accepted_x) == count:
                        break
        if len(accepted_x) < count:
            raise ValueError(
                f"Could not sample {count} unlabeled points within the boundary "
                f"after {attempts} attempts (only {len(accepted_x)} accepted). "
                "Check that the boundary is a valid polygon covering the requested extent."
            )
        return pd.DataFrame({"X": accepted_x, "Y": accepted_y})
