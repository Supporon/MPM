"""Metric 注册表与共享的分类器评估逻辑。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .registry import METRIC_REGISTRY
from .mpm_metrics import MPM_METRIC_NAMES


MetricFunction = Callable[[Any, Any, Any, Any], Any]


def _register(name: str):
    return METRIC_REGISTRY.decorator(name)


@_register("accuracy")
def metric_accuracy(labels, predictions, probabilities, sample_weight):
    return float(accuracy_score(labels, predictions, sample_weight=sample_weight))


@_register("precision")
def metric_precision(labels, predictions, probabilities, sample_weight):
    return float(precision_score(labels, predictions, sample_weight=sample_weight, zero_division=0))


@_register("recall")
def metric_recall(labels, predictions, probabilities, sample_weight):
    return float(recall_score(labels, predictions, sample_weight=sample_weight, zero_division=0))


@_register("f1")
def metric_f1(labels, predictions, probabilities, sample_weight):
    return float(f1_score(labels, predictions, sample_weight=sample_weight, zero_division=0))


@_register("roc_auc")
def metric_roc_auc(labels, predictions, probabilities, sample_weight):
    if len(set(labels)) != 2:
        return None
    return float(roc_auc_score(labels, probabilities, sample_weight=sample_weight))


@_register("confusion_matrix")
def metric_confusion_matrix(labels, predictions, probabilities, sample_weight):
    return confusion_matrix(labels, predictions, sample_weight=sample_weight).tolist()


@_register("average_precision")
def metric_average_precision(labels, predictions, probabilities, sample_weight):
    if len(set(labels)) != 2:
        return None
    return float(average_precision_score(labels, probabilities, sample_weight=sample_weight))


@_register("balanced_accuracy")
def metric_balanced_accuracy(labels, predictions, probabilities, sample_weight):
    return float(balanced_accuracy_score(labels, predictions, sample_weight=sample_weight))


@_register("mcc")
def metric_mcc(labels, predictions, probabilities, sample_weight):
    return float(matthews_corrcoef(labels, predictions, sample_weight=sample_weight))


def _mpm_metric_requires_unit_area(*args, **kwargs):
    """MPM 面积捕获指标需要 unit_area，不能作为标准 4 参数 metric 直接调用。"""
    raise ValueError(
        "MPM area-capture metrics require per-unit area (unit_area) and are "
        "computed via evaluate_classifier, not called directly."
    )


# 注册 MPM 指标键，使 validation.metrics 可配置（配置校验 loop 通过
# METRIC_REGISTRY.require 识别）；实际计算在 evaluate_classifier 的 unit_area 分支。
for _name in sorted(MPM_METRIC_NAMES):
    METRIC_REGISTRY.register(_name, _mpm_metric_requires_unit_area)


def build_metric_scorer(name: str, data):
    """从 Metric 注册表构建 SearchCV 评分器，并保留各折对应的样本权重。"""
    if name in MPM_METRIC_NAMES:
        # MPM 面积捕获指标需要每个预测单元的面积（unit_area），其签名不满足
        # sklearn 的 4 参数 scorer 约定，无法在 BayesSearchCV 内层折上计算。
        # 这里在构建评分器时即拒绝，避免每次评分调用才抛出占位错误（P1-03）。
        raise ValueError(
            f"Primary metric '{name}' is an MPM area-capture metric and cannot be "
            "used as a cross-validation scoring function: it requires per-unit "
            "area (unit_area). Choose a standard scalar metric (e.g. f1, roc_auc) "
            "for tuning, or use tuning.name=none and report the MPM metric in "
            "evaluation."
        )
    metric = METRIC_REGISTRY.get(name)

    def scorer(estimator, features, labels):
        predictions = estimator.predict(features)
        probabilities = estimator.predict_proba(features)[:, 1]
        try:
            sample_weight = data.sample_weight.loc[features.index]
        except (AttributeError, KeyError):
            sample_weight = None
        value = metric(labels, predictions, probabilities, sample_weight)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Primary metric '{name}' must return a scalar for tuning")
        return float(value)

    return scorer


def evaluate_classifier(
    model: Any,
    features,
    labels,
    sample_weight,
    metric_names: list[str] | tuple[str, ...] | None = None,
    unit_area=None,
) -> dict[str, Any]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    names = tuple(metric_names or ("accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"))
    result: dict[str, Any] = {"row_count": int(len(labels))}
    mpm_requested = [name for name in names if name in MPM_METRIC_NAMES]
    standard_names = [name for name in names if name not in MPM_METRIC_NAMES]
    for name in standard_names:
        metric = METRIC_REGISTRY.get(name)
        value = metric(labels, predictions, probabilities, sample_weight)
        if value is not None:
            result[name] = value
    # MPM 面积捕获指标：需要每个预测单元的面积；缺失时记录明确不可计算原因。
    if mpm_requested:
        if unit_area is not None:
            from .mpm_metrics import evaluate_mpm_metrics

            computed = evaluate_mpm_metrics(probabilities, labels, unit_area)
            for name in mpm_requested:
                result[name] = computed[name]
        else:
            for name in mpm_requested:
                result[name] = None
                result[f"{name}_not_computed_reason"] = "unit_area not provided"
    return result
