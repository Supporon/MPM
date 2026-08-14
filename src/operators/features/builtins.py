"""对现有 ``lib_mpm`` 函数进行封装的内置 Feature Operator。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .context import LegacyFeatureContext
from .registry import FEATURE_OPERATOR_REGISTRY


class _BaseOperator:
    name = "base"

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

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        line_files = list(self.context.dataset_config["geology"]["line_files"])
        line_files.append(self.context.dataset_config["seismic"])
        return self.context.legacy().get_dist_line(
            units["X"],
            units["Y"],
            line_files,
            distance_type=str(self.params.get("distance_type", "geodesic")),
        )


@FEATURE_OPERATOR_REGISTRY.decorator("categorical_geology")
class CategoricalGeologyOperator(_BaseOperator):
    name = "categorical_geology"

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        operators = self.context.legacy()
        geology = self.context.dataset_config["geology"]
        metamorphic = operators.get_cat_data(
            units["X"], units["Y"], geology["metamorphic_facies"], field="MetFacies"
        )
        intrusions = operators.get_cat_data(
            units["X"], units["Y"], geology["intrusions"], field="Dominant_L"
        )
        rock_units = operators.get_cat_data(
            units["X"], units["Y"], geology["rock_units"], field="Dominant_L"
        )
        return pd.concat([metamorphic, intrusions, rock_units], axis=1)


@FEATURE_OPERATOR_REGISTRY.decorator("raster_statistics")
class RasterStatisticsOperator(_BaseOperator):
    name = "raster_statistics"

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

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        return self.context.legacy().get_grid_grad_stat_features(
            units["X"],
            units["Y"],
            self.context.dataset_config["elevation"],
            buffer_shape=str(self.params.get("buffer_shape", "square")),
            buffer_size=int(self.params.get("buffer_size", 10)),
        )
