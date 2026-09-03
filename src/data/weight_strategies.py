"""样本权重策略：将属性（如 SIZE_CODE）映射为样本权重。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .registries import WEIGHT_STRATEGY_REGISTRY


@WEIGHT_STRATEGY_REGISTRY.decorator("size_code")
class SizeCodeWeightStrategy:
    """将 SIZE_CODE 映射为配置的样本权重。

    Notebook 基线语义：不同矿床规模（VLG/LGE/MED/SML/OCC）对应不同权重。
    """

    name = "size_code"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        # 权重映射在 label.sample_weight 中配置，这里无额外参数。
        pass

    def compute(self, attributes: pd.DataFrame, weight_config: Mapping[str, Any]) -> pd.Series:
        if "SIZE_CODE" not in attributes.columns:
            raise ValueError("Occurrence data must include SIZE_CODE")
        weights = weight_config
        result = attributes["SIZE_CODE"].map(weights)
        if result.isna().any():
            missing = sorted(result.loc[result.isna()].unique())
            raise ValueError(f"No configured sample weight for SIZE_CODE values: {missing}")
        return result


@WEIGHT_STRATEGY_REGISTRY.decorator("uniform")
class UniformWeightStrategy:
    """所有样本等权。"""

    name = "uniform"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    def compute(self, attributes: pd.DataFrame, weight_config: Mapping[str, Any]) -> pd.Series:
        value = float(weight_config.get("value", 1.0))
        return pd.Series(value, index=attributes.index)
