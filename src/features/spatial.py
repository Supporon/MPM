"""二阶段 Feature Operator 管线的向后兼容包装器。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..core.spec import ComponentSpec
from ..operators.features.pipeline import FeaturePipeline


def legacy_feature_specs(config: Mapping[str, Any]) -> list[ComponentSpec]:
    """将一期布尔特征开关转换为有序的算子规格。"""
    if "operators" in config:
        return [ComponentSpec.from_mapping(item) for item in config["operators"]]
    buffer_size = int(config.get("buffer_size", 10))
    specs: list[ComponentSpec] = []
    if config.get("raster_statistics", False):
        specs.append(ComponentSpec("raster_statistics", {"buffer_size": buffer_size, "buffer_shape": "square"}))
    if config.get("texture", False):
        specs.append(ComponentSpec("texture", {"buffer_size": buffer_size}))
    if config.get("gradient", False):
        specs.append(ComponentSpec("elevation_gradient", {"buffer_size": buffer_size, "buffer_shape": "square"}))
    if config.get("line_distance", False):
        specs.append(ComponentSpec("line_distance", {"distance_type": "geodesic"}))
    if config.get("categorical", False):
        specs.append(ComponentSpec("categorical_geology", {}))
    return specs


class SpatialFeatureExtractor:
    """一期 API 兼容层；新代码应直接使用 ``FeaturePipeline``。"""

    def __init__(self, dataset_config: Mapping[str, Any], features_config: Mapping[str, Any]):
        self.pipeline = FeaturePipeline(dataset_config, legacy_feature_specs(features_config))

    @property
    def _legacy(self):
        return self.pipeline.context._legacy

    @_legacy.setter
    def _legacy(self, value):
        self.pipeline.context._legacy = value

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.extract(units)

    def _raster_files(self) -> list[str]:
        return self.pipeline.context.raster_files()
