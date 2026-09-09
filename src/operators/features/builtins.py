"""对现有 ``lib_mpm`` 函数进行封装的内置 Feature Operator。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ...utils.crs import crs_identifier
from .context import LegacyFeatureContext
from .registry import FEATURE_OPERATOR_REGISTRY


def _unit_crs_string(units: pd.DataFrame) -> str | None:
    """从研究单元 DataFrame 的 attrs 中解析 CRS 标识字符串。

    研究单元构建器（``research_unit_builtins``）把输入的矢量 CRS 记录在
    ``units.attrs["crs"]``；Feature Operator 必须把它显式传给底层 ``lib_mpm``
    的 ``input_crs``，否则底层会退回经纬度默认值（P0-04：投影坐标被当作
    经纬度，从而提取错误距离/类别特征）。返回 ``EPSG:xxxx`` 或 WKT；无 CRS
    时返回 None，由调用方退回旧版默认行为（仅兼容旧的经纬度数据）。
    """
    crs = getattr(units, "attrs", {}).get("crs")
    if crs is None:
        return None
    return crs_identifier(crs)


class _BaseOperator:
    name = "base"
    # raw_gis 配置校验据此按所选算子收紧数据集要求（顶层键 + geology 子键）。
    REQUIRED_DATASET_KEYS: tuple[str, ...] = ()
    REQUIRED_GEOLOGY_KEYS: tuple[str, ...] = ()

    def __init__(self, context: LegacyFeatureContext, params: Mapping[str, Any]):
        self.context = context
        self.params = dict(params)

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if "buffer_size" in params and int(params["buffer_size"]) < 1:
            raise ValueError("feature operator buffer_size must be a positive integer")


@FEATURE_OPERATOR_REGISTRY.decorator("line_distance")
class LineDistanceOperator(_BaseOperator):
    name = "line_distance"
    REQUIRED_DATASET_KEYS = ("seismic",)
    REQUIRED_GEOLOGY_KEYS = ("line_files",)

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        line_files = list(self.context.dataset_config["geology"]["line_files"])
        line_files.append(self.context.dataset_config["seismic"])
        kwargs: dict[str, Any] = {
            "distance_type": str(self.params.get("distance_type", "geodesic")),
        }
        input_crs = _unit_crs_string(units)
        if input_crs is not None:
            kwargs["input_crs"] = input_crs
        return self.context.legacy().get_dist_line(
            units["X"],
            units["Y"],
            line_files,
            **kwargs,
        )


@FEATURE_OPERATOR_REGISTRY.decorator("categorical_geology")
class CategoricalGeologyOperator(_BaseOperator):
    name = "categorical_geology"
    REQUIRED_DATASET_KEYS = ("geology",)
    REQUIRED_GEOLOGY_KEYS = ("metamorphic_facies", "intrusions", "rock_units")

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        operators = self.context.legacy()
        geology = self.context.dataset_config["geology"]
        input_crs = _unit_crs_string(units)
        common: dict[str, Any] = {}
        if input_crs is not None:
            common["input_crs"] = input_crs
        # 类别字段名可配置，缺省沿用 NSW 归档字段（MetFacies / Dominant_L）。
        metamorphic = operators.get_cat_data(
            units["X"], units["Y"], geology["metamorphic_facies"],
            field=str(self.params.get("metamorphic_field", "MetFacies")),
            **common,
        )
        intrusions = operators.get_cat_data(
            units["X"], units["Y"], geology["intrusions"],
            field=str(self.params.get("intrusions_field", "Dominant_L")),
            **common,
        )
        rock_units = operators.get_cat_data(
            units["X"], units["Y"], geology["rock_units"],
            field=str(self.params.get("rock_units_field", "Dominant_L")),
            **common,
        )
        return pd.concat([metamorphic, intrusions, rock_units], axis=1)


@FEATURE_OPERATOR_REGISTRY.decorator("raster_statistics")
class RasterStatisticsOperator(_BaseOperator):
    name = "raster_statistics"
    REQUIRED_DATASET_KEYS = ("magnetic", "gravity", "radiometric", "remote_sensing")

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        return self.context.legacy().get_grid_stat_features(
            units["X"],
            units["Y"],
            self.context.raster_files(),
            buffer_shape=str(self.params.get("buffer_shape", "square")),
            buffer_size=int(self.params.get("buffer_size", 10)),
        )


@FEATURE_OPERATOR_REGISTRY.decorator("texture")
class TextureOperator(_BaseOperator):
    name = "texture"
    REQUIRED_DATASET_KEYS = ("magnetic", "gravity", "radiometric", "remote_sensing")

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        return self.context.legacy().get_grid_tex_features(
            units["X"],
            units["Y"],
            self.context.raster_files(),
            buffer_size=int(self.params.get("buffer_size", 10)),
        )


@FEATURE_OPERATOR_REGISTRY.decorator("elevation_gradient")
class ElevationGradientOperator(_BaseOperator):
    name = "elevation_gradient"
    REQUIRED_DATASET_KEYS = ("elevation",)

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        return self.context.legacy().get_grid_grad_stat_features(
            units["X"],
            units["Y"],
            self.context.dataset_config["elevation"],
            buffer_shape=str(self.params.get("buffer_shape", "square")),
            buffer_size=int(self.params.get("buffer_size", 10)),
        )
