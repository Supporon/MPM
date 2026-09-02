"""与主模型契约分离保留的旧版 PUB 标签细化适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier

from ..core.contracts import OptionalDependencyError, TrainingData
from .registry import LABEL_REFINER_REGISTRY


@dataclass(frozen=True)
class LabelRefinementResult:
    model: Any
    data: TrainingData


@LABEL_REFINER_REGISTRY.decorator("pub")
class LegacyPubRelabeler:
    name = "pub"
    artifact_filename = "model_pub.pkl"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("n_iter", 100)) < 1:
            raise ValueError("pub n_iter must be positive")
        required = {
            "bootstrap",
            "max_depth",
            "max_features",
            "min_samples_leaf",
            "min_samples_split",
            "n_estimators",
        }
        missing = sorted(required.difference(params.get("search_space", {})))
        if missing:
            raise ValueError(f"pub search_space is missing parameters: {missing}")

    def refine(
        self,
        data: TrainingData,
        params: Mapping[str, Any],
        validation_config: Mapping[str, Any],
        seed: int,
    ) -> LabelRefinementResult:
        try:
            from pulearn import BaggingPuClassifier
            from skopt import BayesSearchCV
            from skopt.space import Categorical, Integer, Real
        except ImportError as error:
            raise OptionalDependencyError(
                "PUB label refinement requires pulearn and scikit-optimize (skopt)."
            ) from error

        base = RandomForestClassifier(n_jobs=-1, random_state=seed)
        pub = BaggingPuClassifier(base, n_jobs=-1, random_state=seed)
        space = params["search_space"]
        search_space = {
            "estimator__bootstrap": Categorical(space["bootstrap"]["values"]),
            "estimator__max_depth": Integer(space["max_depth"]["low"], space["max_depth"]["high"]),
            "estimator__max_features": Categorical(space["max_features"]["values"]),
            "estimator__min_samples_leaf": Integer(
                space["min_samples_leaf"]["low"], space["min_samples_leaf"]["high"]
            ),
            "estimator__min_samples_split": Integer(
                space["min_samples_split"]["low"], space["min_samples_split"]["high"]
            ),
            "estimator__n_estimators": Integer(
                space["n_estimators"]["low"], space["n_estimators"]["high"]
            ),
            "max_samples": Real(
                float(params.get("max_samples_low", 0.4)),
                float(params.get("max_samples_high", 0.9)),
                prior="uniform",
            ),
        }
        cv_config = validation_config["cross_validation"]
        from ..validation.splitters import build_cv

        searcher = BayesSearchCV(
            pub,
            search_space,
            n_iter=int(params.get("n_iter", 100)),
            scoring=validation_config["primary_metric"],
            cv=build_cv(cv_config["name"], cv_config.get("params", {}), seed, data),
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=seed,
            verbose=0,
        )
        searcher.fit(data.features, data.labels, sample_weight=data.sample_weight)
        relabelled = searcher.best_estimator_.predict(data.features)
        relabelled[data.labels.to_numpy() == 1] = 1
        labels = data.labels.copy()
        labels.iloc[:] = relabelled
        refined = data.with_labels(labels, label_refiner="pub")
        return LabelRefinementResult(searcher.best_estimator_, refined)


def train_and_relabel(
    features,
    labels,
    sample_weight,
    model_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    seed: int,
):
    """向后兼容一期接口的包装器。"""
    legacy_space = model_config["search"]["space"]
    explicit_space = {
        "bootstrap": {"type": "categorical", "values": legacy_space["bootstrap"]},
        "max_depth": {"type": "integer", "low": legacy_space["max_depth"][0], "high": legacy_space["max_depth"][1]},
        "max_features": {"type": "categorical", "values": legacy_space["max_features"]},
        "min_samples_leaf": {"type": "integer", "low": legacy_space["min_samples_leaf"][0], "high": legacy_space["min_samples_leaf"][1]},
        "min_samples_split": {"type": "integer", "low": legacy_space["min_samples_split"][0], "high": legacy_space["min_samples_split"][1]},
        "n_estimators": {"type": "integer", "low": legacy_space["n_estimators"][0], "high": legacy_space["n_estimators"][1]},
    }
    data = TrainingData(features.reset_index(drop=True), labels.reset_index(drop=True), sample_weight.reset_index(drop=True))
    v2_validation = {
        "cross_validation": {"name": "stratified_kfold", "params": {"n_splits": int(validation_config["cv"]), "shuffle": False}},
        "primary_metric": validation_config["scoring"],
    }
    result = LegacyPubRelabeler().refine(
        data,
        {"n_iter": model_config["search"]["n_iter"], "search_space": explicit_space},
        v2_validation,
        seed,
    )
    return result.model, result.data.labels.to_numpy()
