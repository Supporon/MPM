#!/usr/bin/env python3
"""MLP 模型和 TSIL 风格谓词的验证测试。

测试覆盖：
1. MLP 模型适配器注册、构建、配置校验
2. MLP 分类器训练和预测（标准 BCE 模式）
3. MLP 分类器 + 全 1 谓词约束加权损失
4. MLP 分类器 + 空间选框谓词约束加权损失
5. 谓词管线（PredicatePipeline）与 MLP 集成
6. TrainingData 谓词约束表示
7. 端到端：谓词 → 约束 → MLP 训练

运行方式：
    python tests/test_mlp_predicates.py
或
    python -m pytest tests/test_mlp_predicates.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.contracts import TrainingData
from src.core.spec import ComponentSpec
from src.models.mlp import (
    MLPClassifier,
    MLPAdapter,
    weighted_mse_with_predicate,
    _normalize_phi,
)
from src.models.registry import MODEL_REGISTRY, fit_params_for
from src.predicates.pipeline import PredicatePipeline
from src.predicates.registry import PREDICATE_REGISTRY

import pytest

pytest.importorskip("torch")


def _make_synthetic_data(
    n_samples: int = 200, n_features: int = 10, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """生成合成二分类数据。"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        weights=[0.8, 0.2],
        random_state=seed,
    )
    return X, y


def _make_training_data(
    X: np.ndarray, y: np.ndarray, with_coords: bool = False
) -> TrainingData:
    """构建 TrainingData，可选地包含空间坐标。"""
    n = len(X)
    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    features = pd.DataFrame(X, columns=feature_cols)

    if with_coords:
        rng = np.random.default_rng(42)
        features["X"] = rng.uniform(140, 155, size=n)
        features["Y"] = rng.uniform(-40, -25, size=n)

    return TrainingData(
        features=features,
        labels=pd.Series(y, index=features.index),
        sample_weight=pd.Series(np.ones(n), index=features.index),
    )


# ============================================================================
# 测试 1-3: 适配器注册、构建、校验
# ============================================================================


def test_mlp_adapter_registration():
    load_builtin_components()
    adapter = MODEL_REGISTRY.create("mlp")
    assert adapter.name == "mlp"
    assert adapter.artifact_filename == "model_mlp.pkl"
    assert adapter.supports_constraints == True
    print("  ✓ MLP 适配器注册成功")


def test_mlp_adapter_build():
    load_builtin_components()
    adapter = MODEL_REGISTRY.create("mlp")
    model = adapter.build(
        {"hidden_layers": [32, 16], "n_epochs": 5, "batch_size": 32, "device": "cpu"},
        seed=42,
    )
    assert isinstance(model, MLPClassifier)
    assert model.hidden_layers == [32, 16]
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    print("  ✓ build() 返回正确的 MLPClassifier 实例")


def test_mlp_config_validation():
    load_builtin_components()
    adapter = MODEL_REGISTRY.create("mlp")

    adapter.validate_config({
        "n_epochs": 10, "batch_size": 32, "learning_rate": 1e-3,
        "dropout": 0.3, "hidden_layers": [64, 32], "tau_init": 0.5,
    })
    print("  ✓ 有效配置校验通过")

    for bad, desc in [
        ({"n_epochs": 0}, "n_epochs=0"),
        ({"dropout": 1.0}, "dropout=1.0"),
        ({"hidden_layers": []}, "空的 hidden_layers"),
        ({"tau_init": 0.0}, "tau_init=0.0"),
    ]:
        try:
            adapter.validate_config(bad)
            assert False, f"应抛出异常: {desc}"
        except ValueError:
            pass
    print("  ✓ 无效配置正确拒绝")


# ============================================================================
# 测试 4-6: 训练、数学验证、归一化
# ============================================================================


def test_mlp_train_predict_standard():
    load_builtin_components()
    X, y = _make_synthetic_data(200, 10, 42)
    model = MLPClassifier(
        hidden_layers=[64, 32],
        n_epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        random_state=42,
        device="cpu",
    )
    model.fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert pred.shape == (200,)
    assert proba.shape == (200, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    accuracy = np.mean(pred == y)
    # BCEWithLogitsLoss 应该收敛到合理的准确率
    assert accuracy > 0.55, f"Accuracy too low: {accuracy:.3f}"
    print(f"  ✓ MLP 标准 BCE 训练成功 (accuracy={accuracy:.3f})")


def test_weighted_mse_math():
    import torch

    pred = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    phi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    tau_hat = torch.tensor(0.5, dtype=torch.float64)
    tau = torch.tensor(0.5, dtype=torch.float64)

    result = weighted_mse_with_predicate(pred, target, phi, tau_hat, tau)

    e = pred - target
    N = 3
    mse_expected = (e**2).mean().item()
    phi_norm = phi / (torch.linalg.norm(phi) + 1e-8)
    phi_T_e = torch.dot(phi_norm, e)
    p_loss_raw_expected = (phi_T_e**2).item() / N

    assert abs(result["mse"].item() - mse_expected) < 1e-5, \
        f"mse: {result['mse'].item()} vs {mse_expected}"
    assert abs(result["p_loss_raw"].item() - p_loss_raw_expected) < 1e-5, \
        f"p_loss_raw: {result['p_loss_raw'].item()} vs {p_loss_raw_expected}"
    assert abs(result["total"].item() - (0.5 * mse_expected + 0.5 * p_loss_raw_expected)) < 1e-5
    print("  ✓ 加权 MSE 损失数学正确性验证通过")


def test_phi_normalization():
    import torch
    phi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    phi_norm = _normalize_phi(phi)
    norm = torch.linalg.norm(phi_norm)
    assert abs(norm.item() - 1.0) < 1e-5
    print("  ✓ φ 向量 L2 归一化: ||φ̃|| = 1.0")


# ============================================================================
# 测试 7-9: 带谓词的 MLP 训练
# ============================================================================


def test_mlp_with_all_ones_predicate():
    load_builtin_components()
    X, y = _make_synthetic_data(200, 10, 42)
    phi_vector = np.ones(200, dtype=np.float32)

    model = MLPClassifier(
        hidden_layers=[64, 32],
        n_epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        tau_init=0.5,
        learn_tau=True,
        random_state=42,
        device="cpu",
    )
    model.fit(X, y, constraints={"phi_vector": phi_vector})

    pred = model.predict(X)
    accuracy = np.mean(pred == y)
    assert accuracy > 0.55, f"Accuracy too low: {accuracy:.3f}"

    import torch
    final_tau = torch.sigmoid(model.model_.alpha).item()
    assert 0.0 < final_tau < 1.0
    print(f"  ✓ MLP + 全 1 谓词训练成功 (accuracy={accuracy:.3f}, final_tau={final_tau:.4f})")


def test_mlp_with_spatial_box_predicate():
    load_builtin_components()
    X, y = _make_synthetic_data(200, 10, 42)

    rng = np.random.default_rng(42)
    coords_x = rng.uniform(140, 155, size=200)
    coords_y = rng.uniform(-40, -25, size=200)
    center_x, center_y = np.median(coords_x), np.median(coords_y)
    half = min(coords_x.max() - coords_x.min(), coords_y.max() - coords_y.min()) * 0.25
    in_box = (coords_x >= center_x - half) & (coords_x <= center_x + half) & \
             (coords_y >= center_y - half) & (coords_y <= center_y + half)
    phi_vector = np.where(in_box, 1.0, 0.0).astype(np.float32)
    assert int(in_box.sum()) > 0
    print(f"  - 空间选框内: {int(in_box.sum())}/{len(X)}")

    model = MLPClassifier(
        hidden_layers=[64, 32],
        n_epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        tau_init=0.5,
        learn_tau=True,
        random_state=42,
        device="cpu",
    )
    model.fit(X, y, constraints={"phi_vector": phi_vector})
    accuracy = np.mean(model.predict(X) == y)
    assert accuracy > 0.55, f"Accuracy too low: {accuracy:.3f}"
    print(f"  ✓ MLP + 空间选框谓词训练成功 (accuracy={accuracy:.3f})")


def test_mlp_with_multi_predicate():
    load_builtin_components()
    X, y = _make_synthetic_data(200, 10, 42)
    rng = np.random.default_rng(42)
    phi_vectors = [
        {"name": "all_ones", "vector": np.ones(200, dtype=np.float32)},
        {"name": "random", "vector": rng.uniform(0, 1, size=200).astype(np.float32)},
    ]

    model = MLPClassifier(
        hidden_layers=[64, 32],
        n_epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        tau_init=0.5,
        learn_tau=True,
        random_state=42,
        device="cpu",
    )
    model.fit(X, y, constraints={"phi_vectors": phi_vectors})
    accuracy = np.mean(model.predict(X) == y)
    assert accuracy > 0.55, f"Accuracy too low: {accuracy:.3f}"
    print(f"  ✓ MLP + 多谓词组合训练成功 (accuracy={accuracy:.3f})")


# ============================================================================
# 测试 10-16: 谓词对 TrainingData 的变换
# ============================================================================


def test_predicate_all_ones_on_training_data():
    load_builtin_components()
    X, y = _make_synthetic_data(50, 10, 42)
    data = _make_training_data(X, y)
    predicate = PREDICATE_REGISTRY.create("all_ones", {})
    result = predicate.apply(data, {})

    assert "phi_vector" in result.constraints
    phi = result.constraints["phi_vector"]
    assert len(phi) == 50
    assert np.all(phi == 1.0)
    assert "all_ones" in result.metadata.get("predicates_applied", [])
    print("  ✓ AllOnesPredicate 正确生成 φ = 1 向量")


def test_predicate_spatial_box_on_training_data():
    load_builtin_components()
    X, y = _make_synthetic_data(50, 10, 42)
    rng = np.random.default_rng(42)
    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    features = pd.DataFrame(X, columns=feature_cols)
    features["X"] = rng.uniform(140, 155, size=50)
    features["Y"] = rng.uniform(-40, -25, size=50)
    data = TrainingData(
        features=features,
        labels=pd.Series(y, index=features.index),
        sample_weight=pd.Series(np.ones(50), index=features.index),
    )

    predicate = PREDICATE_REGISTRY.create(
        "spatial_box",
        {"auto_center": True, "auto_side": True, "inner_value": 1.0, "outer_value": 0.0},
    )
    result = predicate.apply(data, {})

    assert "phi_vector" in result.constraints
    phi = result.constraints["phi_vector"]
    assert len(phi) == 50
    unique_vals = np.unique(phi)
    assert set(unique_vals).issubset({0.0, 1.0})
    n_in = int(np.sum(phi == 1.0))
    assert n_in > 0
    spatial_meta = result.metadata.get("spatial_box", {})
    assert spatial_meta["n_in_box"] == n_in
    print(f"  ✓ SpatialBoxPredicate 正确 (n_in={n_in}/50)")


def test_predicate_spatial_box_manual_params():
    load_builtin_components()
    X, y = _make_synthetic_data(50, 10, 42)
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    features["X"] = np.linspace(140, 155, 50)
    features["Y"] = np.linspace(-40, -25, 50)
    data = TrainingData(
        features=features,
        labels=pd.Series(y, index=features.index),
        sample_weight=pd.Series(np.ones(50), index=features.index),
    )

    predicate = PREDICATE_REGISTRY.create(
        "spatial_box",
        {"auto_center": False, "auto_side": False, "center_x": 147.5,
         "center_y": -32.5, "side_length": 5.0, "inner_value": 1.0, "outer_value": 0.0},
    )
    result = predicate.apply(data, {})
    phi = result.constraints["phi_vector"]
    x_coords = features["X"].to_numpy()
    y_coords = features["Y"].to_numpy()
    expected_in = (x_coords >= 145.0) & (x_coords <= 150.0) & \
                  (y_coords >= -35.0) & (y_coords <= -30.0)
    assert np.all(phi == np.where(expected_in, 1.0, 0.0))
    print("  ✓ SpatialBoxPredicate 手动参数正确")


def test_predicate_spatial_distance():
    load_builtin_components()
    X, y = _make_synthetic_data(50, 10, 42)
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    features["X"] = np.linspace(140, 155, 50)
    features["Y"] = np.linspace(-40, -25, 50)
    data = TrainingData(
        features=features,
        labels=pd.Series(y, index=features.index),
        sample_weight=pd.Series(np.ones(50), index=features.index),
    )

    predicate = PREDICATE_REGISTRY.create("spatial_distance", {})
    result = predicate.apply(data, {})
    phi = result.constraints["phi_vector"]
    assert len(phi) == 50
    assert np.all(phi > 0) and np.all(phi <= 1.0)
    print(f"  ✓ SpatialDistancePredicate 正确 (max_phi={phi.max():.4f})")


def test_predicate_pipeline():
    load_builtin_components()
    X, y = _make_synthetic_data(50, 10, 42)
    data = _make_training_data(X, y, with_coords=True)

    pipeline = PredicatePipeline(
        [ComponentSpec("all_ones", {}), ComponentSpec("spatial_box", {})]
    )
    result = pipeline.apply(data, {})
    # 多个 constraint 谓词应合并为 phi_vectors，而非后者覆盖前者
    assert "phi_vectors" in result.constraints
    names = [v["name"] for v in result.constraints["phi_vectors"]]
    assert names == ["all_ones", "spatial_box"]
    assert "all_ones" in result.metadata.get("predicates_applied", ())
    assert "spatial_box" in result.metadata.get("predicates_applied", ())
    print("  ✓ PredicatePipeline 顺序执行正确")


def test_predicate_pipeline_rejects_row_count_change():
    load_builtin_components()

    class BadPredicate:
        name = "row_changer_test"
        def __init__(self, params): pass
        def apply(self, data, context): return data.take([0])

    PREDICATE_REGISTRY.register("row_changer_test", BadPredicate)
    X, y = _make_synthetic_data(50, 10, 42)
    data = _make_training_data(X, y)

    pipeline = PredicatePipeline([ComponentSpec("row_changer_test", {})])
    try:
        pipeline.apply(data, {})
        assert False
    except ValueError as e:
        assert "row count" in str(e)
    print("  ✓ PredicatePipeline 正确拒绝行数变化")


def test_combined_predicate():
    load_builtin_components()
    X, y = _make_synthetic_data(50, 10, 42)
    rng = np.random.default_rng(42)
    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    features = pd.DataFrame(X, columns=feature_cols)
    features["X"] = rng.uniform(140, 155, size=50)
    features["Y"] = rng.uniform(-40, -25, size=50)
    data = TrainingData(
        features=features,
        labels=pd.Series(y, index=features.index),
        sample_weight=pd.Series(np.ones(50), index=features.index),
    )

    predicate = PREDICATE_REGISTRY.create(
        "combined",
        {"predicates": [
            {"name": "all_ones", "params": {}},
            {"name": "spatial_box", "params": {}},
        ]},
    )
    result = predicate.apply(data, {})
    assert "phi_vectors" in result.constraints
    phi_vectors = result.constraints["phi_vectors"]
    assert len(phi_vectors) == 2
    assert phi_vectors[0]["name"] == "all_ones"
    assert np.all(phi_vectors[0]["vector"] == 1.0)
    assert phi_vectors[1]["name"] == "spatial_box"
    print("  ✓ CombinedPredicate 正确组合多个谓词")


# ============================================================================
# 测试 17-21: 集成、序列化、对比、校验
# ============================================================================


def test_fit_params_for_mlp():
    load_builtin_components()
    X, y = _make_synthetic_data(50, 10, 42)
    data = _make_training_data(X, y)
    data_with_constraints = data.with_constraints(
        {"phi_vector": np.ones(50, dtype=np.float32)}
    )
    adapter = MODEL_REGISTRY.create("mlp")
    fit_kwargs = fit_params_for(adapter, data_with_constraints)
    assert "sample_weight" in fit_kwargs
    assert "constraints" in fit_kwargs
    assert "phi_vector" in fit_kwargs["constraints"]
    print("  ✓ fit_params_for 正确传递谓词约束")


def test_end_to_end_predicate_mlp_pipeline():
    load_builtin_components()
    X, y = _make_synthetic_data(200, 10, 42)
    rng = np.random.default_rng(42)
    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    features = pd.DataFrame(X, columns=feature_cols)
    features["X"] = rng.uniform(140, 155, size=200)
    features["Y"] = rng.uniform(-40, -25, size=200)
    data = TrainingData(
        features=features,
        labels=pd.Series(y, index=features.index),
        sample_weight=pd.Series(np.ones(200), index=features.index),
    )

    pipeline = PredicatePipeline(
        [ComponentSpec("spatial_box", {"auto_center": True, "auto_side": True})]
    )
    transformed = pipeline.apply(data, {})
    assert "phi_vector" in transformed.constraints

    adapter = MODEL_REGISTRY.create("mlp")
    model = adapter.build(
        {"hidden_layers": [64, 32], "n_epochs": 50, "batch_size": 64, "device": "cpu"},
        seed=42,
    )
    fit_kwargs = fit_params_for(adapter, transformed)
    model.fit(transformed.features, transformed.labels, **fit_kwargs)
    accuracy = np.mean(model.predict(transformed.features) == transformed.labels)
    assert accuracy > 0.55, f"Accuracy too low: {accuracy:.3f}"
    print(f"  ✓ 端到端管线成功 (accuracy={accuracy:.3f})")


def test_mlp_serialization():
    import pickle
    load_builtin_components()
    X, y = _make_synthetic_data(100, 10, 42)

    model = MLPClassifier(
        hidden_layers=[32, 16], n_epochs=10, batch_size=32,
        learning_rate=1e-3, random_state=42, device="cpu",
    )
    model.fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(model, f)
        temp_path = f.name

    with open(temp_path, "rb") as f:
        loaded = pickle.load(f)

    assert np.array_equal(model.predict(X), loaded.predict(X))
    Path(temp_path).unlink(missing_ok=True)
    print("  ✓ MLP 模型序列化/反序列化成功")


def test_mlp_compare_with_without_predicates():
    load_builtin_components()
    X, y = _make_synthetic_data(200, 10, 42)

    common = {"hidden_layers": [64, 32], "n_epochs": 50, "batch_size": 64,
              "learning_rate": 1e-3, "random_state": 42, "device": "cpu"}

    model_no_pred = MLPClassifier(**common)
    model_no_pred.fit(X, y)
    acc_no_pred = np.mean(model_no_pred.predict(X) == y)

    model_with_pred = MLPClassifier(**common)
    model_with_pred.fit(X, y, constraints={"phi_vector": np.ones(200, dtype=np.float32)})
    acc_with_pred = np.mean(model_with_pred.predict(X) == y)

    assert acc_no_pred > 0.55
    assert acc_with_pred > 0.55
    print(f"  ✓ 带/不带谓词对比: without={acc_no_pred:.3f}, with={acc_with_pred:.3f}")


def test_constraint_phi_length_validation():
    load_builtin_components()
    X, y = _make_synthetic_data(100, 10, 42)

    model = MLPClassifier(
        hidden_layers=[32, 16], n_epochs=5, batch_size=32, random_state=42, device="cpu",
    )

    try:
        model.fit(X, y, constraints={"phi_vector": np.ones(50, dtype=np.float32)})
        assert False
    except ValueError as e:
        assert "length" in str(e)
    print("  ✓ 正确校验 φ 向量长度")

    try:
        model.fit(X, y, constraints={
            "phi_vectors": [
                {"name": "ok", "vector": np.ones(100, dtype=np.float32)},
                {"name": "bad", "vector": np.ones(50, dtype=np.float32)},
            ]
        })
        assert False
    except ValueError as e:
        assert "length" in str(e)
    print("  ✓ 正确校验多谓词 φ 向量长度")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import torch
    print("=" * 70)
    print("MLP + TSIL 风格谓词验证测试")
    print(f"PyTorch 版本: {torch.__version__}")
    print("=" * 70)

    tests = [
        ("MLP 适配器注册", test_mlp_adapter_registration),
        ("MLP 适配器 build()", test_mlp_adapter_build),
        ("MLP 配置校验", test_mlp_config_validation),
        ("MLP 标准 BCE 训练", test_mlp_train_predict_standard),
        ("加权 MSE 数学验证", test_weighted_mse_math),
        ("φ 向量 L2 归一化", test_phi_normalization),
        ("MLP + 全 1 谓词", test_mlp_with_all_ones_predicate),
        ("MLP + 空间选框谓词", test_mlp_with_spatial_box_predicate),
        ("MLP + 多谓词组合", test_mlp_with_multi_predicate),
        ("AllOnesPredicate TrainingData", test_predicate_all_ones_on_training_data),
        ("SpatialBoxPredicate TrainingData", test_predicate_spatial_box_on_training_data),
        ("SpatialBoxPredicate 手动参数", test_predicate_spatial_box_manual_params),
        ("SpatialDistancePredicate", test_predicate_spatial_distance),
        ("PredicatePipeline", test_predicate_pipeline),
        ("PredicatePipeline 行数保护", test_predicate_pipeline_rejects_row_count_change),
        ("CombinedPredicate", test_combined_predicate),
        ("fit_params_for MLP", test_fit_params_for_mlp),
        ("端到端管线", test_end_to_end_predicate_mlp_pipeline),
        ("MLP 序列化", test_mlp_serialization),
        ("带/不带谓词对比", test_mlp_compare_with_without_predicates),
        ("φ 向量长度校验", test_constraint_phi_length_validation),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    print(f"总计: {passed} 通过, {failed} 失败, {len(tests)} 总计")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)