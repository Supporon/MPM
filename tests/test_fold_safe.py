"""P0-03: 折内隔离预处理的单元与集成测试。"""

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


def test_fold_safe_preprocessor_refits_scaler_but_keeps_schema_fixed():
    frame = _raw_frame()
    schema = _fit_schema(frame)
    raw_columns = list(schema.numerical_columns) + list(schema.categorical_columns)

    subset = frame.iloc[:20].reset_index(drop=True)
    fold = FoldSafePreprocessor(schema).fit(subset[raw_columns])

    # schema（列选择 + OHE 类别）固定，与完整训练集一致
    assert fold.pp_.numerical_columns == schema.numerical_columns
    assert fold.pp_.categorical_columns == schema.categorical_columns
    assert fold.pp_._encoded_columns == schema._encoded_columns
    assert fold.pp_.feature_columns == schema.feature_columns
    # OHE 编码器共享（handle_unknown=ignore 类别集合固定）
    assert fold.pp_.encoder is schema.encoder

    # scaler 只基于折内样本重拟合，统计量应不同于完整训练集
    if schema.numerical_columns:
        assert not np.allclose(
            fold.pp_.scaler.mean_, schema.scaler.mean_, equal_nan=True
        )

    # transform 输出维度与固定 feature_columns 一致
    out = fold.transform(subset[raw_columns])
    assert list(out.columns) == schema.feature_columns
    assert len(out) == len(subset)


def test_fold_safe_preprocessor_shares_encoder_categories_across_folds():
    frame = _raw_frame()
    schema = _fit_schema(frame)
    raw_columns = list(schema.numerical_columns) + list(schema.categorical_columns)

    fold_a = FoldSafePreprocessor(schema).fit(frame.iloc[:30][raw_columns])
    fold_b = FoldSafePreprocessor(schema).fit(frame.iloc[30:][raw_columns])
    assert fold_a.pp_.encoder is fold_b.pp_.encoder
    # 两个折 transform 出的列名一致（折间无 schema 漂移）
    assert list(fold_a.transform(frame.iloc[:30][raw_columns]).columns) == list(
        fold_b.transform(frame.iloc[30:][raw_columns]).columns
    )


def test_bayes_tuner_fold_safe_returns_bare_model():
    skopt = pytest.importorskip("skopt")
    load_builtin_components()

    from src.models.registry import MODEL_REGISTRY
    from src.tuning.bayes import BayesTuner

    frame = _raw_frame()
    schema = _fit_schema(frame)
    raw_columns = list(schema.numerical_columns) + list(schema.categorical_columns)

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
