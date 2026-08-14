"""可组合的 Feature Operator 管线。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from ...core.bootstrap import load_builtin_components
from ...core.spec import ComponentSpec
from .context import LegacyFeatureContext
from .registry import FEATURE_OPERATOR_REGISTRY


class FeaturePipeline:
    """按顺序运行已配置算子，同时强制执行行数与 schema 契约。"""

    def __init__(self, dataset_config: Mapping[str, Any], operator_specs: Iterable[ComponentSpec]):
        load_builtin_components()
        self.context = LegacyFeatureContext(dataset_config)
        self.specs = tuple(operator_specs)
        self.operators = [
            FEATURE_OPERATOR_REGISTRY.create(spec.name, self.context, spec.params) for spec in self.specs
        ]

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        if units.columns.duplicated().any():
            duplicates = sorted(set(units.columns[units.columns.duplicated()].tolist()))
            raise ValueError(f"Research units contain duplicate columns: {duplicates}")
        frames: list[pd.DataFrame] = [units.reset_index(drop=True)]
        seen = set(frames[0].columns)
        for spec, operator in zip(self.specs, self.operators):
            frame = operator.extract(units).reset_index(drop=True)
            if len(frame) != len(units):
                raise ValueError(
                    f"Feature operator '{spec.name}' returned {len(frame)} rows for {len(units)} units"
                )
            if frame.columns.duplicated().any():
                duplicates = sorted(set(frame.columns[frame.columns.duplicated()].tolist()))
                raise ValueError(
                    f"Feature operator '{spec.name}' emitted duplicate columns: {duplicates}"
                )
            duplicates = seen.intersection(frame.columns)
            if duplicates:
                raise ValueError(
                    f"Feature operator '{spec.name}' emitted duplicate columns: {sorted(duplicates)}"
                )
            seen.update(frame.columns)
            frames.append(frame)
        return pd.concat(frames, axis=1)
