"""YAML 配置加载、一期兼容迁移、路径解析与校验。"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..utils.logging import get_logger
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
        # 默认不声明算子：archive_replay / train_from_archive_features 禁止算子，
        # raw_gis 配置需显式声明。避免默认值在“不生效”的执行模式下被静默忽略。
        "operators": [],
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
    "prediction": {"score_type": "probability", "normalization": "none", "export_geotiff": True},
}


# 顶层允许的配置节（严格 schema 用于拒绝未知/拼写错误的顶层键）。
_KNOWN_SECTIONS = set(DEFAULTS) | {"dataset"}

# 已实现的研究变量枚举。注册化（P1-1）完成后这些将被注册表取代，
# 目前先用于拒绝 "does_not_exist" 一类的无效值，避免静默忽略。
_KNOWN_RESEARCH_UNIT_TYPES = {"point_local_environment"}
_KNOWN_TRAIN_POSITIVE = {"occurrence_points"}
_KNOWN_TRAIN_UNLABELED = {"random_points_in_nsw_boundary"}
_KNOWN_LABEL_STRATEGIES = {"positive_unlabeled_as_zero"}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_value(value: str) -> int | float | bool | str:
    """将 CLI 字符串值智能转换为 int / float / bool，否则保持字符串。"""
    if not isinstance(value, str):
        return value
    # 检测列表/JSON 值，CLI --set 不支持
    stripped = value.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        raise ConfigError(
            f"CLI --set does not support list/JSON values: '{value}'. "
            "Use a dedicated YAML config file instead."
        )
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    # 尝试 int，再尝试 float
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value


def _dot_to_nested(overrides: dict[str, Any]) -> dict[str, Any]:
    """将 ``{"a.b.c": v}`` 展开为 ``{"a": {"b": {"c": v}}}``。

    冲突处理：若同时存在 ``a.b = 1`` 和 ``a.b.c = 2``，后者覆盖前者。
    """
    result: dict[str, Any] = {}
    for key, value in overrides.items():
        parts = key.split(".")
        for part in parts:
            if part.isdigit():
                raise ConfigError(
                    f"CLI --set does not support list index in path: '{key}'. "
                    "Use a dedicated YAML config file instead."
                )
        current = result
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        # 如果中间路径上已有非 dict 值，强制替换为 dict（后设置的覆盖前的）
        if parts[-1] in current and isinstance(current[parts[-1]], dict) and not isinstance(value, dict):
            # 保留 dict 叶子不变，仅当 value 不是 dict 时覆盖
            pass
        current[parts[-1]] = value
    return result


def apply_cli_overrides(config: ExperimentConfig, raw_overrides: list[str]) -> ExperimentConfig:
    """解析 CLI ``--set key=value`` 参数，合并到配置中并重新校验。

    返回一个新的 ``ExperimentConfig``，原配置不受影响。
    所有覆盖值会记录在 ``config_resolved.yaml`` 中。
    """
    if not raw_overrides:
        return config
    parsed: dict[str, str] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid --set value: '{item}' (expected KEY=VALUE)")
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"Invalid --set value: '{item}' (key is empty)")
        parsed[key] = _coerce_value(value.strip())
    overrides_nested = _dot_to_nested(parsed)
    merged = _deep_merge(copy.deepcopy(config.values), overrides_nested)
    validate_config(merged)
    return ExperimentConfig(values=merged, source_path=config.source_path)


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

    _validate_strict_schema(config)

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

    _validate_mode_capabilities(config)


def _reject_unknown_keys(
    mapping: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    """拒绝未知键，避免拼写错误或未实现字段被静默忽略。"""
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown configuration key(s) in {context}: {unknown}. "
            f"Allowed keys: {sorted(allowed)}"
        )


def _validate_strict_schema(config: Mapping[str, Any]) -> None:
    """严格 schema：拒绝顶层、model、research_unit、label 的未知键。"""
    _reject_unknown_keys(config, _KNOWN_SECTIONS, "top level")
    _reject_unknown_keys(config["model"], {"name", "params"}, "model")
    _reject_unknown_keys(
        config["research_unit"],
        {"type", "train_positive", "train_unlabeled", "prediction_grid_size"},
        "research_unit",
    )
    _reject_unknown_keys(
        config["label"],
        {"strategy", "positive_value", "unlabeled_value", "sample_weight"},
        "label",
    )

    research_unit = config["research_unit"]
    if research_unit["type"] not in _KNOWN_RESEARCH_UNIT_TYPES:
        raise ConfigError(
            f"research_unit.type={research_unit['type']!r} is not implemented; "
            f"available: {sorted(_KNOWN_RESEARCH_UNIT_TYPES)}"
        )
    if research_unit["train_positive"] not in _KNOWN_TRAIN_POSITIVE:
        raise ConfigError(
            f"research_unit.train_positive={research_unit['train_positive']!r} is not implemented; "
            f"available: {sorted(_KNOWN_TRAIN_POSITIVE)}"
        )
    if research_unit["train_unlabeled"] not in _KNOWN_TRAIN_UNLABELED:
        raise ConfigError(
            f"research_unit.train_unlabeled={research_unit['train_unlabeled']!r} is not implemented; "
            f"available: {sorted(_KNOWN_TRAIN_UNLABELED)}"
        )
    if config["label"]["strategy"] not in _KNOWN_LABEL_STRATEGIES:
        raise ConfigError(
            f"label.strategy={config['label']['strategy']!r} is not implemented; "
            f"available: {sorted(_KNOWN_LABEL_STRATEGIES)}"
        )


def _validate_mode_capabilities(config: Mapping[str, Any]) -> None:
    """按执行模式拒绝不生效的配置（配置必须产生行为）。"""
    mode = config["experiment"]["execution_mode"]

    if mode == "archive_replay":
        if config["tuning"]["name"] != "none":
            raise ConfigError(
                "archive_replay mode does not use tuning; set tuning.name=none "
                "or use a training execution mode."
            )
        if config["label_refinement"]["enabled"]:
            raise ConfigError(
                "archive_replay mode replays archived artifacts and cannot re-run "
                "label refinement (PUB); disable label_refinement."
            )
        if config["knowledge"]["enabled"]:
            raise ConfigError(
                "archive_replay mode does not support knowledge injection; disable knowledge."
            )
        if config["predicates"]["enabled"]:
            raise ConfigError(
                "archive_replay mode does not support predicate constraints; disable predicates."
            )
        if config["features"]["operators"]:
            raise ConfigError(
                "archive_replay mode does not run feature operators; set features.operators=[]."
            )
        return

    if mode == "train_from_archive_features":
        if config["label_refinement"]["enabled"]:
            raise ConfigError(
                "train_from_archive_features mode uses precomputed archive labels; "
                "PUB label refinement does not execute. Disable label_refinement "
                "or use raw_gis mode."
            )
        if config["features"]["operators"]:
            raise ConfigError(
                "train_from_archive_features mode uses archived features; feature "
                "operators do not execute. Set features.operators=[] or use raw_gis mode."
            )
        if config["validation"]["holdout"]["name"] != "random_holdout":
            raise ConfigError(
                "train_from_archive_features mode uses the archive's precomputed "
                "train/test split; the holdout is fixed and cannot be changed. "
                "Set validation.holdout.name=random_holdout or use raw_gis mode."
            )
        return


def load_config(path: str | Path) -> ExperimentConfig:
    log = get_logger("config")
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ConfigError(f"Configuration file not found: {source_path}")
    log.info("Loading config: %s", source_path)
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
    log.info("Config loaded and validated: experiment=%s, model=%s, mode=%s",
             resolved["experiment"]["name"], resolved["model"]["name"], resolved["experiment"]["execution_mode"])
    return ExperimentConfig(values=resolved, source_path=source_path)
