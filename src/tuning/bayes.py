"""使用可选依赖 scikit-optimize 的贝叶斯超参数 Tuner。"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.contracts import OptionalDependencyError, TrainingData
from ..features.fold_safe import FoldSafePreprocessor
from ..features.preprocess import BaselinePreprocessor
from ..models.registry import fit_params_for
from ..utils.logging import get_logger
from ..validation.metrics import build_metric_scorer
from ..validation.splitters import build_cv
from .registry import TUNER_REGISTRY


def _to_json_value(value: Any) -> Any:
    """将 numpy 标量/列表/映射递归转为 JSON 可序列化值。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "item"):  # numpy 标量
        return _to_json_value(value.item())
    return str(value)


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
        *,
        model_seed: int | None = None,
        dataloader_seed: int | None = None,
        preprocessor: BaselinePreprocessor | None = None,
        raw_data: TrainingData | None = None,
    ):
        log = get_logger("bayes")
        try:
            from skopt import BayesSearchCV
        except ImportError as error:
            raise OptionalDependencyError("Bayesian tuning requires scikit-optimize (skopt).") from error
        if data.constraints:
            raise ValueError(
                "Bayesian tuning with predicate constraints is not yet supported: "
                "BayesSearchCV slices X/y per fold but cannot slice the constraint "
                "phi vectors to match each fold. Use tuning.name=none with "
                "constraint-supporting models, or a constraint-aware CV splitter."
            )
        model = model_adapter.build(
            model_params,
            model_seed if model_seed is not None else seed,
        )
        if dataloader_seed is not None and hasattr(model, "dataloader_seed"):
            model.dataloader_seed = dataloader_seed
        search_space = build_bayes_search_space(tuning_params.get("search_space", {}))
        n_iter = int(tuning_params.get("n_iter", 50))

        # raw_gis + 内层 CV 的折内隔离：预处理作为 Pipeline 首步，每个内层
        # fold 只重拟合 scaler，避免验证折参与缩放统计（P0-03）。
        if preprocessor is not None:
            if raw_data is None:
                raise ValueError(
                    "fold-safe bayes tuning requires raw_data (untransformed "
                    "training features) alongside the preprocessor"
                )
            from sklearn.pipeline import Pipeline

            estimator = Pipeline(
                [
                    ("fold_safe_preprocessor", FoldSafePreprocessor(preprocessor)),
                    ("model", model),
                ]
            )
            prefixed_space = {f"model__{name}": dim for name, dim in search_space.items()}
            fit_params = {
                f"model__{name}": value
                for name, value in fit_params_for(model_adapter, raw_data).items()
            }
            search_features = raw_data.features
            search_labels = raw_data.labels
            scoring_data = raw_data
        else:
            estimator = model
            prefixed_space = search_space
            fit_params = fit_params_for(model_adapter, data)
            search_features = data.features
            search_labels = data.labels
            scoring_data = data

        cv = build_cv(cv_config["name"], cv_config.get("params", {}), seed, scoring_data)
        log.info(
            "Bayesian search: n_iter=%d, cv=%s(%d folds), scoring=%s, params=%s, fold_safe=%s",
            n_iter, cv_config["name"], cv.get_n_splits(), scoring, sorted(search_space),
            preprocessor is not None,
        )
        searcher = BayesSearchCV(
            estimator,
            prefixed_space,
            n_iter=n_iter,
            scoring=build_metric_scorer(scoring, scoring_data),
            cv=cv,
            n_jobs=int(tuning_params.get("n_jobs", model_params.get("n_jobs", -1))),
            random_state=seed,
            verbose=int(tuning_params.get("verbose", 0)),
        )
        searcher.fit(search_features, search_labels, **fit_params)
        best = searcher.best_estimator_
        # 折内隔离时解包 Pipeline，返回裸模型；后续 evaluate/predict 由
        # Experiment 使用外层 preprocessor 对验证/目标特征做 transform。
        if preprocessor is not None:
            best = best.named_steps["model"]
        log.info("Bayesian search complete. Best score=%.4f, best params=%s",
                 searcher.best_score_, searcher.best_params_)
        # 附上调参轨迹摘要（best_score/best_params/逐候选 cv_results），供
        # Experiment 写入 manifest（P1-07 工件 provenance）。
        cv_results = searcher.cv_results_
        params_trace = cv_results.get("params", []) if cv_results is not None else []
        ranks = cv_results.get("rank_test_score", []) if cv_results is not None else []
        means = cv_results.get("mean_test_score", []) if cv_results is not None else []
        summary = {
            "tuner": "bayes",
            "n_iter": n_iter,
            "n_splits": int(cv.get_n_splits()),
            "scoring": scoring,
            "best_score": _to_json_value(searcher.best_score_),
            "best_params": _to_json_value(searcher.best_params_),
            "cv_results": [
                {
                    "rank": _to_json_value(ranks[i]),
                    "mean_test_score": _to_json_value(means[i]),
                    "params": _to_json_value(params_trace[i]),
                }
                for i in range(len(params_trace))
            ],
        }
        try:
            best._tuning_summary = summary
        except Exception:  # pragma: no cover - 部分估计器禁止任意属性
            log.warning("Could not attach tuning summary to the fitted model")
        return best
