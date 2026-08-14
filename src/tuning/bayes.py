"""使用可选依赖 scikit-optimize 的贝叶斯超参数 Tuner。"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.contracts import OptionalDependencyError, TrainingData
from ..models.registry import fit_params_for
from ..validation.metrics import build_metric_scorer
from ..validation.splitters import build_cv
from .registry import TUNER_REGISTRY


def build_bayes_search_space(space: Mapping[str, Any]):
    """将配置中的搜索空间声明转换为 scikit-optimize 维度对象。"""

    try:
        from skopt.space import Categorical, Integer, Real
    except ImportError as error:
        raise OptionalDependencyError("Bayesian tuning requires scikit-optimize (skopt).") from error

    result: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, Mapping):
            kind = spec.get("type")
            if kind == "categorical":
                result[name] = Categorical(list(spec["values"]))
            elif kind == "integer":
                result[name] = Integer(int(spec["low"]), int(spec["high"]), prior=str(spec.get("prior", "uniform")))
            elif kind == "real":
                result[name] = Real(float(spec["low"]), float(spec["high"]), prior=str(spec.get("prior", "uniform")))
            else:
                raise ValueError(f"Unknown Bayes search-space type for '{name}': {kind}")
            continue
        if isinstance(spec, list):
            if len(spec) == 2 and all(isinstance(value, int) and not isinstance(value, bool) for value in spec):
                result[name] = Integer(int(spec[0]), int(spec[1]))
            else:
                result[name] = Categorical(spec)
            continue
        raise ValueError(f"Invalid Bayes search-space specification for '{name}': {spec!r}")
    return result


@TUNER_REGISTRY.decorator("bayes")
class BayesTuner:
    """使用注册的 Splitter 和 Metric 执行贝叶斯参数搜索。"""

    name = "bayes"
    uses_cross_validation = True

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("n_iter", 50)) < 1:
            raise ValueError("bayes n_iter must be positive")
        space = params.get("search_space", {})
        if not isinstance(space, Mapping):
            raise ValueError("bayes search_space must be a mapping")
        for name, spec in space.items():
            if not isinstance(spec, (Mapping, list)):
                raise ValueError(f"Invalid Bayes search-space specification for '{name}': {spec!r}")

    def fit(
        self,
        model_adapter,
        data: TrainingData,
        model_params: Mapping[str, Any],
        tuning_params: Mapping[str, Any],
        cv_config: Mapping[str, Any],
        scoring: str,
        seed: int,
    ):
        try:
            from skopt import BayesSearchCV
        except ImportError as error:
            raise OptionalDependencyError("Bayesian tuning requires scikit-optimize (skopt).") from error
        model = model_adapter.build(model_params, seed)
        searcher = BayesSearchCV(
            model,
            build_bayes_search_space(tuning_params.get("search_space", {})),
            n_iter=int(tuning_params.get("n_iter", 50)),
            scoring=build_metric_scorer(scoring, data),
            cv=build_cv(cv_config["name"], cv_config.get("params", {}), seed, data),
            n_jobs=int(tuning_params.get("n_jobs", model_params.get("n_jobs", -1))),
            random_state=seed,
            verbose=int(tuning_params.get("verbose", 0)),
        )
        searcher.fit(data.features, data.labels, **fit_params_for(model_adapter, data))
        return searcher.best_estimator_
