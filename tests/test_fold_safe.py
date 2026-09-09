"""P0-01: 折内隔离预处理（完整重拟合相关性筛选 + OHE + scaler）测试。"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.config import ConfigError, load_config, validate_config
from src.core.contracts import TrainingData
from src.features.fold_safe import FoldSafePreprocessor
from src.features.preprocess import BaselinePreprocessor


def _raw_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f1": rng.normal(0.0, 1.0, n),
            "f2": rng.normal(10.0, 5.0, n),
            "f3": rng.normal(100.0, 20.0, n),
            "cat": np.where(rng.random(n) < 0.5, "A", "B"),
            "label": np.tile([1, 0], n // 2),
            "sample_weight": np.ones(n),
        }
    )


def _fit_schema(frame: pd.DataFrame) -> BaselinePreprocessor:
    pp = BaselinePreprocessor(
        {"correlation_threshold": 0.7, "categorical_encoding": "onehot_ignore_unknown", "scaling": "standard"},
        seed=42,
    )
    pp.fit(frame, unit_columns=("X", "Y"))
    return pp


def test_fold_safe_preprocessor_refits_full_schema_per_fold():
    """折内独立重拟合相关性筛选 + OHE + scaler，不复用外层 schema 的编码器。"""
    frame = _raw_frame()
    schema = _fit_schema(frame)
    raw_columns = list(schema._all_numerical_columns) + list(schema._all_categorical_columns)

    subset = frame.iloc[:20].reset_index(drop=True)
    fold = FoldSafePreprocessor(schema).fit(subset[raw_columns])

    # 编码器与 scaler 都基于折内数据重新拟合，而非复用外层实例。
    assert fold.pp_.encoder is not schema.encoder
    assert fold.pp_.numerical_columns != schema.numerical_columns or (
        fold.pp_.scaler_fitted
    )

    # transform 输出维度与该折自己的 feature_columns 一致。
    out = fold.transform(subset[raw_columns])
    assert list(out.columns) == fold.pp_.feature_columns
    assert len(out) == len(subset)


def test_fold_safe_isolates_validation_only_category():
    """P0-01 反例：验证折专属类别不进入训练折的 OHE schema，映射为全零而非 1。"""
    rng = np.random.default_rng(0)
    n = 60
    a = rng.normal(size=n)
    b = a + rng.normal(0, 1e-3, n)  # 与 a 高度相关
    cat = np.where(rng.random(n) < 0.5, "cat_a", "validation_only")
    frame = pd.DataFrame(
        {
            "a": a,
            "b": b,
            "cat": cat,
            "label": np.tile([1, 0], n // 2),
            "sample_weight": np.ones(n),
        }
    )
    schema = _fit_schema(frame)
    # 外层 schema：b 被相关性筛选剔除；OHE 含 validation_only。
    assert "b" not in schema.numerical_columns
    assert any("validation_only" in c for c in schema._encoded_columns)

    # 内层训练折只有 cat_a 类别（不含 validation_only）。
    fold_frame = frame.loc[cat == "cat_a"].reset_index(drop=True)
    fold = FoldSafePreprocessor(schema).fit(fold_frame)
    assert not any("validation_only" in c for c in fold.pp_._encoded_columns)

    # 验证折出现 validation_only → handle_unknown 映射为全零，而非编码为 1。
    val_frame = frame.loc[cat == "validation_only"].reset_index(drop=True)
    out = fold.transform(val_frame)
    cat_cols = [c for c in out.columns if c.startswith("cat_")]
    assert cat_cols
    assert np.allclose(out[cat_cols].to_numpy(), 0.0)


def test_bayes_tuner_fold_safe_returns_bare_model():
    skopt = pytest.importorskip("skopt")
    load_builtin_components()

    from src.models.registry import MODEL_REGISTRY
    from src.tuning.bayes import BayesTuner

    frame = _raw_frame()
    schema = _fit_schema(frame)
    raw_columns = list(schema._all_numerical_columns) + list(schema._all_categorical_columns)

    raw_data = TrainingData(
        frame[raw_columns].reset_index(drop=True),
        frame["label"].reset_index(drop=True),
        frame["sample_weight"].reset_index(drop=True),
    )
    transformed = TrainingData(
        schema.transform(frame[raw_columns]).reset_index(drop=True),
        frame["label"].reset_index(drop=True),
        frame["sample_weight"].reset_index(drop=True),
    )

    adapter = MODEL_REGISTRY.create("rf")
    model = BayesTuner().fit(
        adapter,
        transformed,
        {},  # model_params
        {
            "n_iter": 1,
            "search_space": {"n_estimators": {"type": "integer", "low": 5, "high": 8}},
        },
        {"name": "stratified_kfold", "params": {"n_splits": 3, "shuffle": False}},
        "f1",
        42,
        model_seed=42,
        dataloader_seed=43,
        preprocessor=schema,
        raw_data=raw_data,
    )

    # 解包后返回裸模型（而非 Pipeline），便于后续 evaluate/predict 复用外层 preprocessor
    assert not hasattr(model, "named_steps")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "predict")


def test_config_rejects_pub_with_cv_tuner_in_raw_gis():
    load_builtin_components()
    from src.core.config import _RF_SEARCH_SPACE

    gis = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_raw_gis.yaml")
    invalid = copy.deepcopy(gis.values)
    invalid["tuning"] = {
        "name": "bayes",
        "params": {
            "n_iter": 3,
            "search_space": {"n_estimators": {"type": "integer", "low": 5, "high": 10}},
        },
    }
    invalid["label_refinement"] = {
        "enabled": True,
        "name": "pub",
        "params": {"n_iter": 10, "search_space": copy.deepcopy(_RF_SEARCH_SPACE)},
    }
    with pytest.raises(ConfigError, match="leaking validation labels"):
        validate_config(invalid)


def test_config_allows_raw_gis_bayes_without_pub():
    load_builtin_components()
    gis = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_raw_gis.yaml")
    valid = copy.deepcopy(gis.values)
    valid["tuning"] = {
        "name": "bayes",
        "params": {
            "n_iter": 3,
            "search_space": {"n_estimators": {"type": "integer", "low": 5, "high": 10}},
        },
    }
    valid["label_refinement"] = {"enabled": False, "name": "pub", "params": {}}
    validate_config(valid)  # 不抛异常
