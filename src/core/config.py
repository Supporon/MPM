"""YAML 配置加载、一期兼容迁移、路径解析与校验。"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .spec import ExperimentSpec


class ConfigError(ValueError):
    """实验配置不完整或无效时抛出的异常。"""


_RF_SEARCH_SPACE = {
    "bootstrap": {"type": "categorical", "values": [True, False]},
    "max_depth": {"type": "integer", "low": 5, "high": 20},
    "max_features": {"type": "categorical", "values": [None, "sqrt", "log2"]},
    "min_samples_leaf": {"type": "integer", "low": 2, "high": 20},
    "min_samples_split": {"type": "integer", "low": 2, "high": 30},
    "n_estimators": {"type": "integer", "low": 10, "high": 200},
}

DEFAULTS: dict[str, Any] = {
    "plugins": [],
    "experiment": {
        "seed": 42,
        "output_dir": "outputs/unnamed_experiment",
        "execution_mode": "archive_replay",
        "variant": "baseline",
    },
    "task": {"name": "target_area_prediction", "region": "unknown", "params": {}},
    "research_unit": {
        "type": "point_local_environment",
        "train_positive": "occurrence_points",
        "train_unlabeled": "random_points_in_nsw_boundary",
        "prediction_grid_size": 0.1,
    },
    "label": {
        "strategy": "positive_unlabeled_as_zero",
        "positive_value": 1,
        "unlabeled_value": 0,
        "sample_weight": {
            "VLG": 0.5,
            "LGE": 0.4,
            "MED": 0.3,
            "SML": 0.2,
            "OCC": 0.1,
            "unlabeled": 0.5,
        },
    },
    "features": {
        "operators": [
            {"name": "raster_statistics", "params": {"buffer_size": 10, "buffer_shape": "square"}},
            {"name": "texture", "params": {"buffer_size": 10}},
            {"name": "elevation_gradient", "params": {"buffer_size": 10, "buffer_shape": "square"}},
            {"name": "line_distance", "params": {"distance_type": "geodesic"}},
            {"name": "categorical_geology", "params": {}},
        ]
    },
    "preprocess": {
        "correlation_threshold": 0.7,
        "categorical_encoding": "onehot_ignore_unknown",
        "scaling": "standard",
    },
    "model": {"name": "rf", "params": {}},
    "label_refinement": {
        "enabled": False,
        "name": "pub",
        "params": {},
    },
    "knowledge": {"enabled": False, "items": []},
    "predicates": {"enabled": False, "combine": "sequential", "items": []},
    "tuning": {
        "name": "none",
        "params": {},
    },
    "validation": {
        "holdout": {"name": "random_holdout", "params": {"test_size": 0.25}},
        "cross_validation": {"name": "stratified_kfold", "params": {"n_splits": 10, "shuffle": False}},
        "metrics": ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"],
        "primary_metric": "f1",
    },
    "prediction": {"score_type": "probability", "normalization": "minmax", "export_geotiff": True},
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _legacy_space_to_explicit(space: Mapping[str, Any]) -> dict[str, Any]:
    explicit: dict[str, Any] = {}
    categorical_names = {"bootstrap", "max_features"}
    for name, value in space.items():
        if name in categorical_names:
            explicit[name] = {"type": "categorical", "values": list(value)}
        elif isinstance(value, list) and len(value) == 2 and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            explicit[name] = {"type": "integer", "low": int(value[0]), "high": int(value[1])}
        else:
            explicit[name] = {"type": "categorical", "values": list(value)}
    return explicit


def _migrate_phase1_config(loaded: Mapping[str, Any]) -> dict[str, Any]:
    """仅在缺少二阶段等价配置时，转换旧版布尔、搜索和划分配置键。"""
    result = copy.deepcopy(dict(loaded))

    features = result.get("features")
    if isinstance(features, Mapping) and "operators" not in features:
        buffer_size = int(features.get("buffer_size", 10))
        operators: list[dict[str, Any]] = []
        if features.get("raster_statistics", False):
            operators.append({"name": "raster_statistics", "params": {"buffer_size": buffer_size, "buffer_shape": "square"}})
        if features.get("texture", False):
            operators.append({"name": "texture", "params": {"buffer_size": buffer_size}})
        if features.get("gradient", False):
            operators.append({"name": "elevation_gradient", "params": {"buffer_size": buffer_size, "buffer_shape": "square"}})
        if features.get("line_distance", False):
            operators.append({"name": "line_distance", "params": {"distance_type": "geodesic"}})
        if features.get("categorical", False):
            operators.append({"name": "categorical_geology", "params": {}})
        result["features"] = {"operators": operators}

    model = result.get("model")
    if isinstance(model, Mapping):
        search = model.get("search")
        if "tuning" not in result and isinstance(search, Mapping):
            enabled = bool(search.get("enabled", False))
            result["tuning"] = {
                "name": "bayes" if enabled else "none",
                "params": {
                    "n_iter": int(search.get("n_iter", 100)),
                    "n_jobs": int(model.get("params", {}).get("n_jobs", -1)),
                    "search_space": _legacy_space_to_explicit(search.get("space", {})),
                },
            }
        pu = model.get("pu")
        if "label_refinement" not in result and isinstance(pu, Mapping):
            search_space = _legacy_space_to_explicit(search.get("space", {})) if isinstance(search, Mapping) else copy.deepcopy(_RF_SEARCH_SPACE)
            result["label_refinement"] = {
                "enabled": bool(pu.get("enabled", True)),
                "name": "pub",
                "params": {
                    "n_iter": int(search.get("n_iter", 100)) if isinstance(search, Mapping) else 100,
                    "n_jobs": int(model.get("params", {}).get("n_jobs", -1)),
                    "search_space": search_space,
                },
            }
        result["model"] = {"name": model.get("name", "rf"), "params": dict(model.get("params", {}))}

    validation = result.get("validation")
    if isinstance(validation, Mapping) and "holdout" not in validation:
        split_name = validation.get("split", "random")
        if split_name != "random":
            holdout_name = str(split_name)
        else:
            holdout_name = "random_holdout"
        result["validation"] = {
            "holdout": {
                "name": holdout_name,
                "params": {"test_size": float(validation.get("test_size", 0.25))},
            },
            "cross_validation": {
                "name": "stratified_kfold",
                "params": {"n_splits": int(validation.get("cv", 10)), "shuffle": False},
            },
            "metrics": ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"],
            "primary_metric": str(validation.get("scoring", "f1")),
        }
    return result


def _require(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing required configuration key: {path}")
        current = current[part]
    if current in (None, ""):
        raise ConfigError(f"Configuration key cannot be empty: {path}")
    return current


def _resolve_path(config_root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (config_root / path).resolve())


def _resolve_dataset_paths(config: dict[str, Any], config_root: Path) -> None:
    dataset = config["dataset"]
    for key in (
        "root",
        "archive_dir",
        "occurrence",
        "boundary",
        "training_boundary",
        "magnetic",
        "gravity",
        "radiometric",
        "remote_sensing",
        "elevation",
        "seismic",
    ):
        if isinstance(dataset.get(key), str):
            dataset[key] = _resolve_path(config_root, dataset[key])
    geology = dataset.get("geology", {})
    if isinstance(geology, Mapping):
        for key, value in list(geology.items()):
            if isinstance(value, str):
                geology[key] = _resolve_path(config_root, value)
            elif isinstance(value, list):
                geology[key] = [
                    _resolve_path(config_root, item) if isinstance(item, str) else item for item in value
                ]
    config["experiment"]["output_dir"] = _resolve_path(config_root, config["experiment"]["output_dir"])


@dataclass(frozen=True)
class ExperimentConfig:
    """保存已解析配置、源文件位置和供运行器使用的实验规格。"""

    values: dict[str, Any]
    source_path: Path

    @property
    def output_dir(self) -> Path:
        return Path(self.values["experiment"]["output_dir"])

    @property
    def name(self) -> str:
        return str(self.values["experiment"]["name"])

    @property
    def spec(self) -> ExperimentSpec:
        return ExperimentSpec.from_mapping(self.values)

    def section(self, name: str) -> dict[str, Any]:
        return self.values[name]


def validate_config(config: Mapping[str, Any]) -> None:
    """校验配置结构和已注册组件名称，不使用硬编码的实现枚举。"""
    from .bootstrap import load_builtin_components
    from ..knowledge.registry import KNOWLEDGE_REGISTRY
    from ..models.registry import LABEL_REFINER_REGISTRY, MODEL_REGISTRY
    from ..operators.features.registry import FEATURE_OPERATOR_REGISTRY
    from ..predicates.registry import PREDICATE_REGISTRY
    from ..tasks.registry import TASK_REGISTRY
    from ..tuning.registry import TUNER_REGISTRY
    from ..validation.registry import METRIC_REGISTRY, SPLITTER_REGISTRY

    load_builtin_components()
    plugins = config.get("plugins", [])
    if not isinstance(plugins, list) or not all(isinstance(item, str) and item.strip() for item in plugins):
        raise ConfigError("plugins must be a list of importable module names")
    for module_name in plugins:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            raise ConfigError(f"Unable to load plugin module '{module_name}': {error}") from error
    for key in ("experiment.name", "experiment.output_dir", "task.name", "dataset.root"):
        _require(config, key)

    if not isinstance(config["experiment"]["seed"], int):
        raise ConfigError("experiment.seed must be an integer")
    if config["experiment"]["execution_mode"] not in {"archive_replay", "train_from_archive_features", "raw_gis"}:
        raise ConfigError("experiment.execution_mode must be archive_replay, train_from_archive_features, or raw_gis")

    try:
        TASK_REGISTRY.require(config["task"]["name"])
        TASK_REGISTRY.validate(
            config["task"]["name"],
            config["task"],
            config["research_unit"],
            config["prediction"],
            config["dataset"],
            config["experiment"]["execution_mode"],
        )
        MODEL_REGISTRY.require(config["model"]["name"])
        MODEL_REGISTRY.validate(config["model"]["name"], config["model"].get("params", {}))
        TUNER_REGISTRY.require(config["tuning"]["name"])
        TUNER_REGISTRY.validate(config["tuning"]["name"], config["tuning"].get("params", {}))
        SPLITTER_REGISTRY.require(config["validation"]["holdout"]["name"])
        SPLITTER_REGISTRY.validate(
            config["validation"]["holdout"]["name"],
            config["validation"]["holdout"].get("params", {}),
        )
        SPLITTER_REGISTRY.require(config["validation"]["cross_validation"]["name"])
        SPLITTER_REGISTRY.validate(
            config["validation"]["cross_validation"]["name"],
            config["validation"]["cross_validation"].get("params", {}),
        )
        for name in config["validation"]["metrics"]:
            METRIC_REGISTRY.require(name)
        METRIC_REGISTRY.require(config["validation"]["primary_metric"])
        for operator in config["features"].get("operators", []):
            FEATURE_OPERATOR_REGISTRY.require(operator["name"])
            FEATURE_OPERATOR_REGISTRY.validate(operator["name"], operator.get("params", {}))
        if config.get("label_refinement", {}).get("enabled", False):
            LABEL_REFINER_REGISTRY.require(config["label_refinement"]["name"])
            LABEL_REFINER_REGISTRY.validate(
                config["label_refinement"]["name"],
                config["label_refinement"].get("params", {}),
            )
        if config.get("knowledge", {}).get("enabled", False):
            knowledge_names = [item["name"] for item in config["knowledge"].get("items", [])]
            if len(knowledge_names) != len(set(knowledge_names)):
                raise ValueError("knowledge provider names must be unique within an experiment")
            for item in config["knowledge"].get("items", []):
                KNOWLEDGE_REGISTRY.require(item["name"])
                KNOWLEDGE_REGISTRY.validate(item["name"], item.get("params", {}))
        if config.get("predicates", {}).get("enabled", False):
            for item in config["predicates"].get("items", []):
                PREDICATE_REGISTRY.require(item["name"])
                PREDICATE_REGISTRY.validate(item["name"], item.get("params", {}))
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error

    if not 0 < float(config["preprocess"]["correlation_threshold"]) <= 1:
        raise ConfigError("preprocess.correlation_threshold must be in (0, 1]")
    if config["preprocess"]["categorical_encoding"] != "onehot_ignore_unknown":
        raise ConfigError("Only categorical_encoding=onehot_ignore_unknown is currently implemented")
    if config["preprocess"]["scaling"] != "standard":
        raise ConfigError("Only scaling=standard is currently implemented")

    if config.get("predicates", {}).get("enabled", False) and config["predicates"].get("combine", "sequential") != "sequential":
        raise ConfigError("Only predicates.combine=sequential is currently implemented")
    if config["validation"]["primary_metric"] not in config["validation"]["metrics"]:
        raise ConfigError("validation.primary_metric must also be listed in validation.metrics")

    if config["experiment"]["execution_mode"] in {"archive_replay", "train_from_archive_features"}:
        _require(config, "dataset.archive_dir")


def load_config(path: str | Path) -> ExperimentConfig:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ConfigError(f"Configuration file not found: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ConfigError("Experiment YAML must contain a mapping at its root")
    migrated = _migrate_phase1_config(loaded)
    resolved = _deep_merge(DEFAULTS, migrated)
    # 沿用仓库约定：configs/experiments/*.yaml 中的路径相对于项目根目录解析。
    config_root = source_path.parents[2] if len(source_path.parents) >= 3 else source_path.parent
    _resolve_dataset_paths(resolved, config_root)
    validate_config(resolved)
    return ExperimentConfig(values=resolved, source_path=source_path)
