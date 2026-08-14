"""当前标签与样本权重规则，保持显式定义并进行版本管理。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def positive_labels_from_size_code(
    occurrences: pd.DataFrame,
    label_config: Mapping[str, Any],
) -> pd.DataFrame:
    """应用 Notebook 中的 SIZE_CODE -> sample_weight 映射。"""
    if "SIZE_CODE" not in occurrences.columns:
        raise ValueError("Occurrence data must include SIZE_CODE")
    result = occurrences.copy()
    weights = label_config["sample_weight"]
    result["label"] = label_config["positive_value"]
    result["sample_weight"] = result["SIZE_CODE"].map(weights)
    if result["sample_weight"].isna().any():
        missing = sorted(result.loc[result["sample_weight"].isna(), "SIZE_CODE"].unique())
        raise ValueError(f"No configured sample weight for SIZE_CODE values: {missing}")
    return result


def unlabeled_labels(points: pd.DataFrame, label_config: Mapping[str, Any]) -> pd.DataFrame:
    """保留当前对随机未标注样本采用的硬编码零类。

    TODO(science)：随机未标注点并非经验证的无矿负样本。
    本阶段为保持基线兼容性而保留现有编码。
    """
    result = points.copy()
    result["label"] = label_config["unlabeled_value"]
    result["sample_weight"] = label_config["sample_weight"]["unlabeled"]
    return result
