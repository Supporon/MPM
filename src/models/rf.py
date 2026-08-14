"""随机森林模型适配器；超参数搜索位于 ``src.tuning``。"""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier

from ..core.contracts import OptionalDependencyError, TrainingData
from .registry import MODEL_REGISTRY


ModelCapabilityError = OptionalDependencyError


@MODEL_REGISTRY.decorator("rf")
class RandomForestAdapter:
    """根据配置构建随机森林，并提供统一的拟合参数。"""

    name = "rf"
    artifact_filename = "model_rf.pkl"
    supports_constraints = False
    archive_metric_aliases = ("RF (Post-PUB)", "rf", "RF")

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        RandomForestClassifier(**dict(params))

    def build(self, params: Mapping[str, Any], seed: int) -> RandomForestClassifier:
        resolved = dict(params)
        resolved.setdefault("n_jobs", -1)
        resolved.setdefault("random_state", seed)
        return RandomForestClassifier(**resolved)

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        return {"sample_weight": data.sample_weight}


def build_rf(params: Mapping[str, Any], seed: int) -> RandomForestClassifier:
    """向后兼容一期接口的辅助函数。"""
    return RandomForestAdapter().build(params, seed)


def train_rf(
    features,
    labels,
    sample_weight,
    model_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    seed: int,
):
    """向后兼容的辅助函数，将调优委托给二阶段 Tuner 注册表。"""
    from ..core.bootstrap import load_builtin_components
    from ..tuning.registry import TUNER_REGISTRY

    load_builtin_components()
    data = TrainingData(
        features=features.reset_index(drop=True),
        labels=labels.reset_index(drop=True),
        sample_weight=sample_weight.reset_index(drop=True),
    )
    search = dict(model_config.get("search", {}))
    if search.get("enabled", False):
        tuner = TUNER_REGISTRY.create("bayes")
        tuning_params = {
            "n_iter": int(search.get("n_iter", 50)),
            "search_space": search.get("space", {}),
        }
    else:
        tuner = TUNER_REGISTRY.create("none")
        tuning_params = {}
    cv = {
        "name": "stratified_kfold",
        "params": {"n_splits": int(validation_config.get("cv", 5)), "shuffle": False},
    }
    return tuner.fit(
        RandomForestAdapter(),
        data,
        model_config.get("params", {}),
        tuning_params,
        cv,
        str(validation_config.get("scoring", "f1")),
        seed,
    )
