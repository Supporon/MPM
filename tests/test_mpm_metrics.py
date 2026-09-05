"""MPM 面积捕获指标与评估链集成的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.validation.metrics import evaluate_classifier
from src.validation.mpm_metrics import (
    capture_rate_at_area_fraction,
    evaluate_mpm_metrics,
    prediction_rate_auc,
)


class _FixedModel:
    def __init__(self, proba: np.ndarray):
        self._proba = proba

    def predict(self, features):
        return (self._proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, features):
        return self._proba


def test_evaluate_mpm_metrics_keys_and_values():
    load_builtin_components()
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0])
    area = np.array([1.0, 1.0, 1.0, 1.0])

    result = evaluate_mpm_metrics(scores, labels, area)
    assert "capture_rate_at_area_0_01" in result
    assert "capture_rate_at_area_0_05" in result
    assert "capture_rate_at_area_0_10" in result
    assert "area_fraction_at_capture_0_50" in result
    assert "area_fraction_at_capture_0_80" in result
    assert "area_fraction_at_capture_0_90" in result
    assert "prediction_rate_auc" in result
    # 正类面积占比 0.5 的理想模型：50% 面积捕获全部正类，AUC=0.75
    assert abs(result["prediction_rate_auc"] - 0.75) < 1e-3
    # 10% 面积落在第一个正类（占 25% 面积）内 → 捕获 10%/25% * 50% = 0.2
    assert abs(result["capture_rate_at_area_0_10"] - 0.2) < 1e-3
    assert abs(result["capture_rate_at_area_0_05"] - 0.1) < 1e-3  # 5% 面积 → 0.1


def test_evaluate_classifier_computes_configured_mpm_metrics_when_unit_area_provided():
    load_builtin_components()
    model = _FixedModel(np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]))
    result = evaluate_classifier(
        model,
        pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        pd.Series([1, 1, 0, 0]),
        pd.Series([1.0, 1.0, 1.0, 1.0]),
        ["f1", "prediction_rate_auc"],
        unit_area=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    assert "prediction_rate_auc" in result
    assert "f1" in result

    # 只请求标准指标时，不隐式追加 MPM 指标（配置闭环：显式配置才计算）。
    result_standard_only = evaluate_classifier(
        model,
        pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        pd.Series([1, 1, 0, 0]),
        pd.Series([1.0, 1.0, 1.0, 1.0]),
        ["f1"],
        unit_area=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    assert "prediction_rate_auc" not in result_standard_only
    assert "f1" in result_standard_only

    # 请求 MPM 指标但未提供 unit_area 时记录明确不可计算原因。
    result_no_area = evaluate_classifier(
        model,
        pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        pd.Series([1, 1, 0, 0]),
        pd.Series([1.0, 1.0, 1.0, 1.0]),
        ["f1", "prediction_rate_auc"],
    )
    assert result_no_area["prediction_rate_auc"] is None
    assert result_no_area["prediction_rate_auc_not_computed_reason"] == "unit_area not provided"
    assert "f1" in result_no_area


def test_prediction_rate_auc_random_is_half():
    load_builtin_components()
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, 2000).astype(float)
    area = np.ones(2000)
    assert abs(prediction_rate_auc(scores, labels, area) - 0.5) < 0.05


if __name__ == "__main__":
    test_evaluate_mpm_metrics_keys_and_values()
    test_evaluate_classifier_computes_configured_mpm_metrics_when_unit_area_provided()
    test_prediction_rate_auc_random_is_half()
    print("MPM 指标测试全部通过")
