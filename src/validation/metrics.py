"""Metric 注册表与共享的分类器评估逻辑。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .registry import METRIC_REGISTRY


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


def build_metric_scorer(name: str, data):
    """从 Metric 注册表构建 SearchCV 评分器，并保留各折对应的样本权重。"""
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
) -> dict[str, Any]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    names = tuple(metric_names or ("accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"))
    result: dict[str, Any] = {"row_count": int(len(labels))}
    for name in names:
        metric = METRIC_REGISTRY.get(name)
        value = metric(labels, predictions, probabilities, sample_weight)
        if value is not None:
            result[name] = value
    return result
