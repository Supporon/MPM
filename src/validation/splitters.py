"""通过统一 Splitter 注册表选择留出法与交叉验证策略。"""

from __future__ import annotations

import warnings
from inspect import signature
from typing import Any, Mapping

from sklearn.model_selection import StratifiedKFold, train_test_split

from ..core.contracts import SplitData, TrainingData
from .registry import SPLITTER_REGISTRY


@SPLITTER_REGISTRY.decorator("random_holdout")
class RandomHoldoutSplitter:
    kind = "holdout"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        test_size = float(params.get("test_size", 0.25))
        if not 0 < test_size < 1:
            raise ValueError("random_holdout test_size must be in (0, 1)")

    def split(self, data: TrainingData, params: Mapping[str, Any], seed: int) -> SplitData:
        warnings.warn(
            "Baseline compatibility: using a random point-level train/test split; spatial leakage remains possible.",
            UserWarning,
            stacklevel=2,
        )
        indices = list(range(len(data.features)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(params.get("test_size", 0.25)),
            random_state=seed,
            stratify=None,
        )
        return SplitData(_take(data, train_idx), _take(data, test_idx))


@SPLITTER_REGISTRY.decorator("stratified_kfold")
class StratifiedKFoldSplitter:
    kind = "cross_validation"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("n_splits", 5)) < 2:
            raise ValueError("stratified_kfold n_splits must be at least 2")

    def build_cv(self, params: Mapping[str, Any], seed: int, data: TrainingData | None = None):
        shuffle = bool(params.get("shuffle", False))
        kwargs = {
            "n_splits": int(params.get("n_splits", 5)),
            "shuffle": shuffle,
        }
        if shuffle:
            kwargs["random_state"] = seed
        return StratifiedKFold(**kwargs)


def _take(data: TrainingData, indices: list[int]) -> TrainingData:
    return data.take(indices)


def create_holdout(name: str, params: Mapping[str, Any], seed: int):
    splitter = SPLITTER_REGISTRY.create(name)
    if getattr(splitter, "kind", None) != "holdout":
        raise ValueError(f"Splitter '{name}' is not a holdout splitter")
    return splitter


def build_cv(name: str, params: Mapping[str, Any], seed: int, data: TrainingData | None = None):
    splitter = SPLITTER_REGISTRY.create(name)
    if getattr(splitter, "kind", None) != "cross_validation":
        raise ValueError(f"Splitter '{name}' is not a cross-validation splitter")
    build_parameters = signature(splitter.build_cv).parameters
    if "data" in build_parameters or len(build_parameters) >= 3:
        return splitter.build_cv(params, seed, data)
    return splitter.build_cv(params, seed)


def random_split(features, labels, sample_weight, test_size: float, seed: int):
    """向后兼容一期接口的辅助类，对外保留旧字段名。"""
    data = TrainingData(
        features.reset_index(drop=True), labels.reset_index(drop=True), sample_weight.reset_index(drop=True)
    )
    split = RandomHoldoutSplitter().split(data, {"test_size": test_size}, seed)

    class LegacySplit:
        x_train = split.train.features
        x_test = split.test.features
        y_train = split.train.labels
        y_test = split.test.labels
        weight_train = split.train.sample_weight
        weight_test = split.test.sample_weight

    return LegacySplit()
