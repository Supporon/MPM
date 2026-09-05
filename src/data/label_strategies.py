"""标签策略：将原始观测编码为模型训练目标与样本权重。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .registries import LABEL_STRATEGY_REGISTRY, WEIGHT_STRATEGY_REGISTRY


@LABEL_STRATEGY_REGISTRY.decorator("positive_unlabeled_as_zero")
class PositiveUnlabeledAsZeroStrategy:
    """正样本标注 + 未标注样本临时编码为零类的 PU 标签策略。

    这是 Notebook 基线的语义：正样本（矿点）标注为 ``positive_value``，
    随机未标注点编码为 ``unlabeled_value``（当前为 0）。样本权重由
    ``label.sample_weight`` 中的 SIZE_CODE 映射决定。

    注意：随机未标注点并非经验证的无矿负样本；将未标注编码为 0 只是
    当前训练策略，不改变其数据事实（观测状态应为 unlabeled）。
    """

    name = "positive_unlabeled_as_zero"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        # 标签值在 label 配置节中声明，这里无额外参数。
        pass

    def apply_positive(self, occurrences: pd.DataFrame, label_config: Mapping[str, Any]) -> pd.DataFrame:
        result = occurrences.copy()
        result["label"] = label_config["positive_value"]
        # 权重策略可配置：默认 size_code（按 SIZE_CODE 映射），亦可用 uniform。
        weight_cfg = label_config.get("weight_strategy", {"name": "size_code", "params": {}})
        if isinstance(weight_cfg, str):
            weight_cfg = {"name": weight_cfg, "params": {}}
        weight_strategy = WEIGHT_STRATEGY_REGISTRY.create(
            weight_cfg["name"], weight_cfg.get("params", {})
        )
        result["sample_weight"] = weight_strategy.compute(
            result, label_config["sample_weight"]
        )
        return result

    def apply_unlabeled(self, points: pd.DataFrame, label_config: Mapping[str, Any]) -> pd.DataFrame:
        result = points.copy()
        result["label"] = label_config["unlabeled_value"]
        result["sample_weight"] = label_config["sample_weight"]["unlabeled"]
        return result
