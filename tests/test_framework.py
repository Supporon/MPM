"""二阶段可配置实验框架的可移植测试。"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bootstrap import load_builtin_components
from src.core.config import ConfigError, ExperimentConfig, apply_cli_overrides, load_config, validate_config
from src.core.contracts import TrainingData
from src.core.contracts import OptionalDependencyError
from src.core.experiment import Experiment
from src.core.registry import ComponentRegistry, RegistryError
from src.core.spec import ComponentSpec
from src.features.spatial import SpatialFeatureExtractor
from src.features.preprocess import BaselinePreprocessor
from src.knowledge.pipeline import KnowledgePipeline
from src.knowledge.registry import KNOWLEDGE_REGISTRY
from src.models.registry import MODEL_REGISTRY
from src.models.registry import fit_params_for
from src.operators.features.pipeline import FeaturePipeline
from src.operators.features.registry import FEATURE_OPERATOR_REGISTRY
from src.predicates.pipeline import PredicatePipeline
from src.predicates.registry import PREDICATE_REGISTRY
from src.tasks.target_area import TargetAreaPredictionTask, TaskCapabilityError, create_task
from src.tuning.registry import TUNER_REGISTRY
from src.validation.metrics import build_metric_scorer, evaluate_classifier
from src.validation.registry import METRIC_REGISTRY, SPLITTER_REGISTRY


LACHLAN_CONFIG = ROOT / "configs" / "experiments" / "lachlan_rf_baseline.yaml"
RAW_GIS_CONFIG = ROOT / "configs" / "experiments" / "lachlan_rf_raw_gis.yaml"

LIB_MPM_SOURCE = """\"\"\"确定性替代 ``lib_mpm``：与 FakeLegacyOperators 等价的模块级函数。\"\"\"
import numpy as np
import pandas as pd


def get_dist_line(xs, ys, line_files, distance_type):
    assert distance_type == "geodesic"
    assert len(line_files) == 14
    return pd.DataFrame({"distance": np.arange(len(xs), dtype=float)})


def get_cat_data(xs, ys, polygon_files, field):
    prefix = "met" if field == "MetFacies" else ("rock" if "RockUnits" in str(polygon_files) else "intrusion")
    return pd.DataFrame({prefix: ["unit"] * len(xs)})


def get_grid_stat_features(xs, ys, paths, buffer_shape, buffer_size):
    assert buffer_shape == "square"
    assert buffer_size == 10
    return pd.DataFrame({"stat": np.arange(len(xs), dtype=float)})


def get_grid_tex_features(xs, ys, paths, buffer_size):
    assert buffer_size == 10
    return pd.DataFrame({"texture": np.arange(len(xs), dtype=float)})


def get_grid_grad_stat_features(xs, ys, path, buffer_shape, buffer_size):
    assert buffer_shape == "square"
    assert buffer_size == 10
    return pd.DataFrame({"gradient": np.arange(len(xs), dtype=float)})
"""


class FakeLegacyOperators:
    """外部 ``lib_mpm`` GIS 算子的确定性替代实现。"""

    def get_dist_line(self, xs, ys, line_files, distance_type):
        assert distance_type == "geodesic"
        assert len(line_files) == 14
        return pd.DataFrame({"distance": np.arange(len(xs), dtype=float)})

    def get_cat_data(self, xs, ys, polygon_files, field):
        prefix = "met" if field == "MetFacies" else ("rock" if "RockUnits" in str(polygon_files) else "intrusion")
        return pd.DataFrame({prefix: ["unit"] * len(xs)})

    def get_grid_stat_features(self, xs, ys, paths, buffer_shape, buffer_size):
        assert buffer_shape == "square"
        assert buffer_size == 10
        return pd.DataFrame({"stat": np.arange(len(xs), dtype=float)})

    def get_grid_tex_features(self, xs, ys, paths, buffer_size):
        assert buffer_size == 10
        return pd.DataFrame({"texture": np.arange(len(xs), dtype=float)})

    def get_grid_grad_stat_features(self, xs, ys, path, buffer_shape, buffer_size):
        assert buffer_shape == "square"
        assert buffer_size == 10
        return pd.DataFrame({"gradient": np.arange(len(xs), dtype=float)})


def _write_synthetic_archive(root: Path) -> Path:
    archive = root / "archive"
    archive.mkdir(parents=True)
    columns = ["f1", "f2", "sample_weight", "label"]
    train = pd.DataFrame(
        [
            [0.0, 1.0, 1.0, 0],
            [1.0, 0.0, 1.0, 1],
            [0.2, 0.8, 1.0, 0],
            [0.8, 0.2, 1.0, 1],
        ],
        columns=columns,
    )
    for name in ("Xy_train.csv", "Xy_train_new.csv", "Xy_rf_train.csv", "Xy_rf_test.csv"):
        train.to_csv(archive / name, index=False)
    pd.DataFrame({"f1": [0.1, 0.9], "f2": [0.9, 0.1]}).to_csv(
        archive / "target_features.csv", index=False
    )
    pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]}).to_csv(
        archive / "target_coords_purged.csv", index=False
    )
    pd.DataFrame([[1.0, 3.0, True], [2.0, 4.0, True]]).to_csv(
        archive / "target_mask.csv", index=False, header=False
    )
    pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0], "prob": [0.1, 0.9]}).to_csv(
        archive / "target_probs.csv", index=False
    )
    (archive / "probability_map.tif").write_bytes(b"synthetic-tiff")
    (archive / "model_rf.pkl").write_bytes(b"synthetic-model")
    return archive


def _write_synthetic_raw_gis(root: Path) -> dict:
    """构建可运行的合成 raw_gis 环境（矢量 + 栅格目录 + 假 lib_mpm）。"""
    import geopandas as gpd
    from shapely.geometry import Point, box

    (root / "lib_mpm.py").write_text(LIB_MPM_SOURCE, encoding="utf-8")

    for key in ("magnetic", "gravity", "radiometric", "remote_sensing"):
        directory = root / key
        directory.mkdir(parents=True)
        (directory / "dummy.tif").write_bytes(b"synthetic-tiff")
    elevation = root / "elevation.tif"
    elevation.write_bytes(b"synthetic-tiff")

    n_positive = 20
    xs = np.linspace(1.0, 9.0, n_positive)
    ys = np.linspace(1.0, 9.0, n_positive)
    occurrence = gpd.GeoDataFrame(
        {"SIZE_CODE": ["VLG"] * n_positive},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs="EPSG:3857",
    )
    occurrence_path = root / "occurrence.shp"
    occurrence.to_file(occurrence_path)

    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3857")
    boundary_path = root / "boundary.shp"
    boundary.to_file(boundary_path)
    training_boundary_path = root / "training_boundary.shp"
    boundary.to_file(training_boundary_path)

    geology_dir = root / "geology"
    geology_dir.mkdir(parents=True)

    return {
        "root": str(root),
        "occurrence": str(occurrence_path),
        "boundary": str(boundary_path),
        "training_boundary": str(training_boundary_path),
        "magnetic": str(root / "magnetic"),
        "gravity": str(root / "gravity"),
        "radiometric": str(root / "radiometric"),
        "remote_sensing": str(root / "remote_sensing"),
        "elevation": str(elevation),
        "seismic": str(geology_dir / "seismic.shp"),
        "geology": {
            "line_files": [str(geology_dir / f"line_{i}.shp") for i in range(13)],
            "metamorphic_facies": [str(geology_dir / "metamorphic.shp")],
            "intrusions": str(geology_dir / "intrusions.shp"),
            "rock_units": str(geology_dir / "RockUnits.shp"),
        },
    }


class FrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        load_builtin_components()

    def test_config_load_baseline_replay_spec(self) -> None:
        config = load_config(LACHLAN_CONFIG)
        self.assertEqual(config.name, "lachlan_rf_baseline")
        self.assertEqual(config.spec.execution_mode, "archive_replay")
        self.assertEqual(config.spec.task.name, "target_area_prediction")
        self.assertEqual(config.spec.model.name, "rf")
        self.assertEqual(config.spec.tuner.name, "none")
        self.assertEqual(config.spec.holdout.name, "random_holdout")
        self.assertEqual(config.spec.cross_validation.name, "stratified_kfold")
        self.assertEqual(list(config.spec.feature_operators), [])
        self.assertIsNone(config.spec.label_refinement)

    def test_phase1_config_migration_to_phase2_spec(self) -> None:
        """P0-2: 一期配置迁移仍将 pu/search/features 转为二阶段结构。"""
        from src.core.config import _migrate_phase1_config

        phase1 = {
            "experiment": {"name": "x", "output_dir": "outputs/x"},
            "dataset": {"root": "."},
            "model": {
                "name": "rf",
                "params": {"n_jobs": -1},
                "pu": {"enabled": True},
                "search": {
                    "enabled": True,
                    "n_iter": 100,
                    "space": {
                        "bootstrap": [True, False],
                        "max_depth": [5, 20],
                        "max_features": [None, "sqrt"],
                        "min_samples_leaf": [2, 20],
                        "min_samples_split": [2, 30],
                        "n_estimators": [10, 200],
                    },
                },
            },
            "features": {"raster_statistics": True, "buffer_size": 10},
            "validation": {"split": "random", "test_size": 0.25, "cv": 10, "scoring": "f1"},
        }
        migrated = _migrate_phase1_config(phase1)
        self.assertEqual(migrated["model"]["name"], "rf")
        self.assertEqual(migrated["tuning"]["name"], "bayes")
        self.assertEqual(migrated["label_refinement"]["enabled"], True)
        self.assertEqual(migrated["features"]["operators"][0]["name"], "raster_statistics")
        self.assertEqual(migrated["validation"]["holdout"]["name"], "random_holdout")

    def test_config_validation_uses_registry_names_and_component_params(self) -> None:
        gis_config = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_raw_gis.yaml")

        invalid = copy.deepcopy(gis_config.values)
        invalid["features"]["operators"][0]["params"]["buffer_size"] = 0
        with self.assertRaisesRegex(ConfigError, "buffer_size"):
            validate_config(invalid)

        invalid = copy.deepcopy(gis_config.values)
        invalid["task"]["name"] = "not_a_task"
        with self.assertRaisesRegex(ConfigError, "Unknown task"):
            validate_config(invalid)

        invalid = copy.deepcopy(gis_config.values)
        invalid["validation"]["metrics"].remove(invalid["validation"]["primary_metric"])
        with self.assertRaisesRegex(ConfigError, "primary_metric"):
            validate_config(invalid)

        invalid = copy.deepcopy(gis_config.values)
        invalid["model"]["params"]["not_an_rf_parameter"] = True
        with self.assertRaises(ConfigError):
            validate_config(invalid)

        invalid = copy.deepcopy(gis_config.values)
        invalid["knowledge"] = {
            "enabled": True,
            "items": [{"name": "empty", "params": {}}, {"name": "empty", "params": {}}],
        }
        with self.assertRaisesRegex(ConfigError, "must be unique"):
            validate_config(invalid)

        class DummyHoldout:
            kind = "holdout"

            @staticmethod
            def validate_config(params):
                if params.get("token") != "accepted":
                    raise ValueError("dummy token rejected")

        SPLITTER_REGISTRY.register("dummy_holdout", DummyHoldout)
        extended = copy.deepcopy(gis_config.values)
        extended["validation"]["holdout"] = {
            "name": "dummy_holdout",
            "params": {"token": "accepted"},
        }
        validate_config(extended)

        # raw_gis 不需要 archive_dir
        raw_gis = copy.deepcopy(gis_config.values)
        raw_gis["dataset"].pop("archive_dir")
        validate_config(raw_gis)

        # archive_replay 需要 archive_dir（且禁止算子，需先清空）
        archive = copy.deepcopy(gis_config.values)
        archive["experiment"]["execution_mode"] = "archive_replay"
        archive["features"]["operators"] = []
        archive["dataset"].pop("archive_dir")
        with self.assertRaisesRegex(ConfigError, "dataset.archive_dir"):
            validate_config(archive)

    def test_generic_registry_extension_contract(self) -> None:
        registry = ComponentRegistry("demo")
        registry.register("one", lambda value: value + 1)
        self.assertEqual(registry.create("one", 4), 5)
        with self.assertRaises(RegistryError):
            registry.register("one", lambda value: value)
        with self.assertRaisesRegex(RegistryError, "Available: one"):
            registry.create("missing")

    def test_feature_operator_pipeline_preserves_phase1_order(self) -> None:
        config = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_raw_gis.yaml").values
        extractor = SpatialFeatureExtractor(config["dataset"], config["features"])
        extractor._legacy = FakeLegacyOperators()
        extractor.pipeline.context._raster_files_cache = ["synthetic.tif"]
        units = pd.DataFrame(
            {"X": [147.1, 147.2], "Y": [-35.1, -35.2], "label": [1, 0], "sample_weight": [0.1, 0.5]}
        )
        features = extractor.extract(units)
        self.assertEqual(len(features), 2)
        self.assertTrue(
            {"stat", "texture", "gradient", "distance", "met", "intrusion", "rock"}.issubset(features.columns)
        )
        self.assertEqual(features.columns[:4].tolist(), ["X", "Y", "label", "sample_weight"])

    def test_feature_operator_pipeline_rejects_duplicate_columns(self) -> None:
        class DuplicateOperator:
            name = "duplicate_test"

            def __init__(self, context, params):
                pass

            def extract(self, units):
                return pd.DataFrame([[1.0, 2.0]], columns=["dup", "dup"])

        FEATURE_OPERATOR_REGISTRY.register("duplicate_test", DuplicateOperator)
        pipeline = FeaturePipeline({}, [type("Spec", (), {"name": "duplicate_test", "params": {}})()])
        with self.assertRaisesRegex(ValueError, "duplicate columns"):
            pipeline.extract(pd.DataFrame({"X": [1.0], "Y": [2.0]}))

    def test_preprocessor_accepts_numeric_only_features_and_preserves_unit_metadata(self) -> None:
        deposits = pd.DataFrame(
            {
                "X": [1.0, 2.0, 3.0, 4.0],
                "Y": [5.0, 6.0, 7.0, 8.0],
                "area_id": ["a", "a", "b", "b"],
                "numeric_feature": [0.1, 0.2, 0.3, 0.4],
                "label": [1, 1, 1, 1],
                "sample_weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        unlabeled = pd.DataFrame(
            {
                "X": [9.0, 10.0, 11.0, 12.0],
                "Y": [13.0, 14.0, 15.0, 16.0],
                "area_id": ["c", "c", "d", "d"],
                "numeric_feature": [0.5, 0.6, 0.7, 0.8],
                "label": [0, 0, 0, 0],
                "sample_weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        prepared = BaselinePreprocessor(
            {
                "correlation_threshold": 0.7,
                "categorical_encoding": "onehot_ignore_unknown",
                "scaling": "standard",
            },
            42,
        ).prepare_training(deposits, unlabeled, unit_columns=["X", "Y", "area_id"])
        self.assertEqual(prepared.feature_columns, ["numeric_feature"])
        self.assertEqual(prepared.xy_train_units.columns.tolist(), ["X", "Y", "area_id"])

    def test_preprocessor_accepts_categorical_only_features(self) -> None:
        deposits = pd.DataFrame(
            {
                "X": [1.0, 2.0, 3.0, 4.0],
                "Y": [5.0, 6.0, 7.0, 8.0],
                "custom_unit": ["a", "b", "a", "b"],
                "label": [1, 1, 1, 1],
                "sample_weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        unlabeled = pd.DataFrame(
            {
                "X": [9.0, 10.0, 11.0, 12.0],
                "Y": [13.0, 14.0, 15.0, 16.0],
                "custom_unit": ["a", "b", "a", "b"],
                "label": [0, 0, 0, 0],
                "sample_weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        preprocessor = BaselinePreprocessor(
            {
                "correlation_threshold": 0.7,
                "categorical_encoding": "onehot_ignore_unknown",
                "scaling": "standard",
            },
            42,
        )
        prepared = preprocessor.prepare_training(deposits, unlabeled)
        self.assertEqual(prepared.feature_columns, ["custom_unit_a", "custom_unit_b"])
        target = preprocessor.transform_target_legacy(
            pd.DataFrame({"X": [20.0], "Y": [21.0], "custom_unit": ["a"]})
        )
        self.assertEqual(target.columns.tolist(), prepared.feature_columns)

    def test_preprocessor_fit_is_isolated_from_test_split(self) -> None:
        """P0-3: 改变测试集分布不改变训练 scaler 均值/方差（split-first）。"""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "X": np.arange(20, dtype=float),
                "Y": np.arange(20, dtype=float),
                "num": rng.normal(size=20),
                "label": [0, 1] * 10,
                "sample_weight": np.ones(20),
            }
        )
        train = frame.iloc[:10]
        test = frame.iloc[10:]
        preprocessor = BaselinePreprocessor(
            {
                "correlation_threshold": 0.7,
                "categorical_encoding": "onehot_ignore_unknown",
                "scaling": "standard",
            },
            42,
        )
        preprocessor.fit(train, unit_columns=["X", "Y"])
        mean_before = preprocessor.scaler.mean_.copy()
        var_before = preprocessor.scaler.var_.copy()

        # 大幅改变测试集数值分布后再 transform，训练 scaler 统计量不得改变
        perturbed = test.copy()
        perturbed["num"] = perturbed["num"] + 1e6
        preprocessor.transform(perturbed)

        self.assertTrue(np.allclose(preprocessor.scaler.mean_, mean_before))
        self.assertTrue(np.allclose(preprocessor.scaler.var_, var_before))

    def test_preprocessor_encoder_does_not_learn_test_categories(self) -> None:
        """P0-3: 测试集独有类别不出现在训练时拟合的编码器类别中。"""
        train = pd.DataFrame(
            {
                "X": [1.0, 2.0],
                "Y": [1.0, 2.0],
                "cat": ["a", "b"],
                "label": [1, 0],
                "sample_weight": [1.0, 1.0],
            }
        )
        test = pd.DataFrame(
            {
                "X": [3.0],
                "Y": [3.0],
                "cat": ["unseen"],
                "label": [0],
                "sample_weight": [1.0],
            }
        )
        preprocessor = BaselinePreprocessor(
            {
                "correlation_threshold": 0.7,
                "categorical_encoding": "onehot_ignore_unknown",
                "scaling": "standard",
            },
            42,
        )
        preprocessor.fit(train, unit_columns=["X", "Y"])
        self.assertNotIn("unseen", preprocessor.encoder.categories_[0])
        # 未知类别经 handle_unknown="ignore" 转为全零，不应报错
        transformed = preprocessor.transform(test)
        self.assertEqual(len(transformed), 1)

    def test_model_tuner_metric_and_predicate_contracts(self) -> None:
        rng = np.random.default_rng(42)
        data = TrainingData(
            pd.DataFrame(rng.normal(size=(24, 3)), columns=["a", "b", "c"]),
            pd.Series([0, 1] * 12),
            pd.Series(np.ones(24)),
        )
        knowledge = KnowledgePipeline([]).build(data, {})
        transformed = PredicatePipeline([]).apply(data, {"knowledge": knowledge})
        adapter = MODEL_REGISTRY.create("rf")
        tuner = TUNER_REGISTRY.create("none")
        model = tuner.fit(
            adapter,
            transformed,
            {"n_estimators": 12, "n_jobs": 1},
            {},
            {"name": "stratified_kfold", "params": {"n_splits": 2, "shuffle": False}},
            "f1",
            42,
        )
        result = evaluate_classifier(model, data.features, data.labels, data.sample_weight, ["f1", "roc_auc"])
        self.assertEqual(result["row_count"], 24)
        self.assertIn("f1", result)
        self.assertIn("roc_auc", result)

    def test_predicate_consumes_namespaced_knowledge_context(self) -> None:
        class SyntheticKnowledge:
            name = "synthetic_knowledge_test"

            def __init__(self, params):
                pass

            def build(self, data, context):
                return {"allowed": [True, False]}

        class KnowledgeConstraintPredicate:
            name = "knowledge_constraint_test"

            def __init__(self, params):
                pass

            def apply(self, data, context):
                allowed = context["knowledge"]["synthetic_knowledge_test"]["allowed"]
                return data.with_constraints({"allowed": allowed})

        KNOWLEDGE_REGISTRY.register("synthetic_knowledge_test", SyntheticKnowledge)
        PREDICATE_REGISTRY.register("knowledge_constraint_test", KnowledgeConstraintPredicate)
        data = TrainingData(
            pd.DataFrame({"x": [0.0, 1.0]}),
            pd.Series([0, 1]),
            pd.Series([1.0, 1.0]),
        )
        knowledge = KnowledgePipeline([ComponentSpec("synthetic_knowledge_test")]).build(data, {})
        transformed = PredicatePipeline([ComponentSpec("knowledge_constraint_test")]).apply(
            data, {"knowledge": knowledge}
        )
        self.assertEqual(transformed.constraints["allowed"], [True, False])

    def test_spatial_extent_knowledge_provider_builds_provenanced_artifacts(self) -> None:
        """P1-2: spatial_extent 知识提供器产出带 provenance 的空间范围工件。"""
        units = pd.DataFrame(
            {"X": [147.0, 148.0, 149.0, 150.0], "Y": [-32.0, -33.0, -34.0, -35.0]}
        )
        data = TrainingData(
            pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]}),
            pd.Series([1, 0, 1, 0]),
            pd.Series([1.0, 1.0, 1.0, 1.0]),
            metadata={"units": units},
        )
        knowledge = KnowledgePipeline(
            [ComponentSpec("spatial_extent", {"crs": "EPSG:4283"})]
        ).build(data, {})

        self.assertIn("spatial_extent", knowledge)
        extent = knowledge["spatial_extent"]
        self.assertEqual(extent["bounds"], {"min_x": 147.0, "min_y": -35.0, "max_x": 150.0, "max_y": -32.0})
        self.assertEqual(extent["center"], {"x": 148.5, "y": -33.5})
        self.assertEqual(extent["span"], {"x": 3.0, "y": 3.0})
        self.assertEqual(extent["n_units"], 4)
        self.assertEqual(extent["source"], "research_unit_metadata")
        self.assertEqual(extent["crs"], "EPSG:4283")

    def test_spatial_extent_knowledge_returns_empty_without_coordinates(self) -> None:
        """无坐标来源时 spatial_extent 返回空工件（消费方回退）。"""
        data = TrainingData(
            pd.DataFrame({"f": [1.0, 2.0]}),
            pd.Series([1, 0]),
            pd.Series([1.0, 1.0]),
        )
        knowledge = KnowledgePipeline([ComponentSpec("spatial_extent")]).build(data, {})
        self.assertEqual(knowledge["spatial_extent"], {})

    def test_spatial_box_consumes_spatial_extent_knowledge(self) -> None:
        """P1-2: spatial_box 谓词优先消费 spatial_extent 知识而非重新估计。"""
        units = pd.DataFrame(
            {"X": [147.0, 148.0, 149.0, 150.0], "Y": [-32.0, -33.0, -34.0, -35.0]}
        )
        data = TrainingData(
            pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]}),
            pd.Series([1, 0, 1, 0]),
            pd.Series([1.0, 1.0, 1.0, 1.0]),
            metadata={"units": units},
        )
        knowledge = KnowledgePipeline([ComponentSpec("spatial_extent")]).build(data, {})
        result = PredicatePipeline([ComponentSpec("spatial_box")]).apply(
            data, {"knowledge": knowledge}
        )
        meta = result.metadata["spatial_box"]
        self.assertAlmostEqual(meta["center_x"], 148.5)
        self.assertAlmostEqual(meta["center_y"], -33.5)
        self.assertAlmostEqual(meta["side_length"], 1.5)
        # φ 向量：中心在 (148.5, -33.5)，边长 1.5 → 选框 [147.75, 149.25] × [-34.25, -32.75]
        expected = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(result.constraints["phi_vector"], expected)

    def test_weighted_confusion_matrix_uses_sample_weight(self) -> None:
        class FixedModel:
            def predict(self, features):
                return np.array([0, 0])

            def predict_proba(self, features):
                return np.array([[0.8, 0.2], [0.7, 0.3]])

        result = evaluate_classifier(
            FixedModel(),
            pd.DataFrame({"x": [0.0, 1.0]}),
            pd.Series([0, 1]),
            pd.Series([1.0, 4.0]),
            ["confusion_matrix"],
        )
        self.assertEqual(result["confusion_matrix"], [[1.0, 0.0], [4.0, 0.0]])

    def test_registry_metric_builds_weighted_tuner_scorer(self) -> None:
        metric_name = "weight_sum_test"
        METRIC_REGISTRY.register(
            metric_name,
            lambda labels, predictions, probabilities, sample_weight: float(sample_weight.sum()),
        )
        data = TrainingData(
            pd.DataFrame({"x": [0.0, 1.0, 2.0]}, index=[10, 11, 12]),
            pd.Series([0, 1, 0], index=[10, 11, 12]),
            pd.Series([1.0, 2.0, 4.0], index=[10, 11, 12]),
        )

        class FixedModel:
            def predict(self, features):
                return np.zeros(len(features), dtype=int)

            def predict_proba(self, features):
                return np.column_stack((np.ones(len(features)), np.zeros(len(features))))

        scorer = build_metric_scorer(metric_name, data)
        self.assertEqual(scorer(FixedModel(), data.features.loc[[11, 12]], data.labels.loc[[11, 12]]), 6.0)

    def test_training_data_rejects_index_misalignment_and_slices_unit_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "indexes must align"):
            TrainingData(
                pd.DataFrame({"x": [1.0, 2.0]}),
                pd.Series([0, 1], index=[1, 2]),
                pd.Series([1.0, 1.0]),
            )
        data = TrainingData(
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0]),
            pd.Series([1.0, 2.0, 3.0]),
            metadata={"units": pd.DataFrame({"X": [10.0, 20.0, 30.0]})},
        )
        selected = data.take([2, 0])
        self.assertEqual(selected.metadata["units"]["X"].tolist(), [30.0, 10.0])

    def test_constraint_fit_contract_rejects_unsupported_models(self) -> None:
        data = TrainingData(
            pd.DataFrame({"x": [0.0, 1.0]}),
            pd.Series([0, 1]),
            pd.Series([1.0, 1.0]),
            constraints={"geology": [True, False]},
        )
        with self.assertRaisesRegex(ValueError, "does not support predicate constraints"):
            fit_params_for(MODEL_REGISTRY.create("rf"), data)

    def test_bayes_tuner_reports_optional_dependency_when_skopt_is_unavailable(self) -> None:
        try:
            import skopt  # noqa: F401
        except ImportError:
            data = TrainingData(
                pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
                pd.Series([0, 1, 0, 1]),
                pd.Series([1.0, 1.0, 1.0, 1.0]),
            )
            with self.assertRaisesRegex(OptionalDependencyError, "scikit-optimize"):
                TUNER_REGISTRY.create("bayes").fit(
                    MODEL_REGISTRY.create("rf"),
                    data,
                    {"n_estimators": 2, "n_jobs": 1},
                    {"n_iter": 1, "search_space": {}},
                    {"name": "stratified_kfold", "params": {"n_splits": 2}},
                    "f1",
                    42,
                )

    def test_raw_target_mask_filtering_keeps_features_coords_and_true_cells_aligned(self) -> None:
        target_data = pd.DataFrame(
            {"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0], "f": [0.1, np.nan, 0.3]}
        )
        target_mask = pd.DataFrame(
            [[0.0, 0.0, False], [1.0, 4.0, True], [2.0, 5.0, True], [3.0, 6.0, True]]
        )
        filtered, coords, aligned_mask = TargetAreaPredictionTask(
            {}, {"score_type": "probability", "normalization": "none", "export_geotiff": False}
        ).prepare_prediction_data(target_data, target_mask)
        self.assertEqual(filtered[["X", "Y"]].values.tolist(), [[1.0, 4.0], [3.0, 6.0]])
        self.assertEqual(coords.values.tolist(), [[1.0, 4.0], [3.0, 6.0]])
        flags = aligned_mask.iloc[:, 2].astype(bool)
        self.assertEqual(int(flags.sum()), len(filtered))
        self.assertFalse(bool(aligned_mask.iloc[2, 2]))

    def test_end_to_end_synthetic_archive_replay(self) -> None:
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            output_dir = root / "outputs" / "replay"
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(output_dir)
            values["experiment"]["execution_mode"] = "archive_replay"
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            manifest = experiment.run()

            self.assertEqual(manifest["feature_count"], 2)
            self.assertEqual(manifest["components"]["model"], "rf")
            actual_dir = experiment.output_dir
            self.assertTrue((actual_dir / "config_resolved.yaml").is_file())
            self.assertTrue((actual_dir / "manifest.json").is_file())
            self.assertTrue((actual_dir / "metrics.json").is_file())
            self.assertEqual((actual_dir / "models" / "model_rf.pkl").read_bytes(), b"synthetic-model")
            self.assertEqual((actual_dir / "predictions" / "probability_map.tif").read_bytes(), b"synthetic-tiff")
            stored = json.loads((actual_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["components"]["feature_operators"], [])
            artifacts = stored["input_paths"]["archive_artifacts"]
            for filename in (
                "Xy_train.csv",
                "Xy_rf_train.csv",
                "model_rf.pkl",
                "target_probs.csv",
                "probability_map.tif",
            ):
                source = archive / filename
                self.assertEqual(
                    artifacts[filename]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
                )
            self.assertNotIn("occurrence", stored["input_paths"])

    def test_end_to_end_train_from_archive_features_uses_registered_model_and_tuner(self) -> None:
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            output_dir = root / "outputs" / "train"
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(output_dir)
            values["experiment"]["execution_mode"] = "train_from_archive_features"
            values["tuning"] = {"name": "none", "params": {}}
            values["model"] = {"name": "rf", "params": {"n_estimators": 8, "n_jobs": 1}}
            values["prediction"]["export_geotiff"] = False
            values["knowledge"] = {"enabled": True, "items": [{"name": "empty", "params": {}}]}
            values["predicates"] = {"enabled": False, "combine": "sequential", "items": []}
            validate_config(values)

            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            manifest = experiment.run()
            metrics = json.loads((experiment.output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["components"]["tuner"], "none")
            self.assertEqual(manifest["components"]["knowledge"], ["empty"])
            self.assertEqual(manifest["components"]["predicates"], [])
            self.assertEqual(manifest["components"]["label_refinement"], None)
            self.assertEqual(
                manifest["component_metadata"]["holdout_execution"],
                "precomputed_in_archive",
            )
            self.assertNotIn("cross_validation", manifest["component_metadata"])
            self.assertIn("f1", metrics)
            self.assertTrue((experiment.output_dir / "models" / "model_rf.pkl").is_file())
            self.assertTrue((experiment.output_dir / "predictions" / "target_probs.csv").is_file())
            self.assertFalse((experiment.output_dir / "predictions" / "probability_map.tif").exists())

    def test_bayes_run_records_tuning_summary_and_target_scaling(self) -> None:
        """P1-07: manifest 记录调参轨迹（best_score/best_params/cv_results）与 target scaling。"""
        try:
            import skopt  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("scikit-optimize not installed")
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(root / "outputs" / "bayes")
            values["experiment"]["execution_mode"] = "train_from_archive_features"
            values["model"] = {"name": "rf", "params": {"n_jobs": 1}}
            values["tuning"] = {
                "name": "bayes",
                "params": {
                    "n_iter": 2,
                    "n_jobs": 1,
                    "search_space": {
                        "n_estimators": {"type": "integer", "low": 5, "high": 15},
                    },
                },
            }
            values["validation"]["cross_validation"] = {
                "name": "stratified_kfold",
                "params": {"n_splits": 2, "shuffle": False},
            }
            values["prediction"] = {
                "score_type": "relative_score",
                "normalization": "minmax",
                "export_geotiff": False,
            }
            validate_config(values)
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            manifest = experiment.run()
            self.assertEqual(manifest["status"], "completed")
            summary = manifest["tuning_summary"]
            self.assertIsNotNone(summary)
            self.assertEqual(summary["tuner"], "bayes")
            self.assertIn("best_score", summary)
            self.assertIn("best_params", summary)
            self.assertGreater(len(summary["cv_results"]), 0)
            self.assertEqual(summary["cv_results"][0]["rank"], 1)
            self.assertIsNotNone(manifest["target_scaling"])

    def test_independent_cv_records_oof_predictions_and_fold_metrics(self) -> None:
        """P1-01: tuning=none 时独立评价 CV 产出 OOF 预测与逐折指标。"""
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(root / "outputs" / "cv")
            values["experiment"]["execution_mode"] = "train_from_archive_features"
            values["model"] = {"name": "rf", "params": {"n_jobs": 1}}
            values["tuning"] = {"name": "none", "params": {}}
            values["validation"]["cross_validation"] = {
                "name": "stratified_kfold",
                "params": {"n_splits": 2, "shuffle": False},
            }
            values["prediction"] = {
                "score_type": "probability",
                "normalization": "none",
                "export_geotiff": False,
            }
            validate_config(values)
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            manifest = experiment.run()
            self.assertEqual(manifest["status"], "completed")
            cv = manifest["independent_cv"]
            self.assertTrue(cv["run"])
            self.assertEqual(cv["n_splits"], 2)
            self.assertEqual(len(cv["folds"]), 2)
            self.assertIn("f1", cv["aggregate_metrics"])
            for fold in cv["folds"]:
                self.assertGreater(fold["val_rows"], 0)
                self.assertIn("metrics", fold)
            oof_path = experiment.output_dir / "intermediate" / "oof_predictions.csv"
            self.assertTrue(oof_path.is_file())
            oof = pd.read_csv(oof_path)
            self.assertEqual(len(oof), 4)
            self.assertCountEqual(oof["row_index"].tolist(), [0, 1, 2, 3])
            self.assertEqual(
                set(oof.columns),
                {"fold", "row_index", "y_true", "y_pred", "score", "sample_weight"},
            )

    def test_independent_cv_rf_constrained_uses_adapter_params(self) -> None:
        """P1-02: rf_constrained + tuning=none 关闭谓词时，折内重建不再把
        get_params() 的 base_estimator__... 嵌套键回灌给 RandomForestClassifier。"""
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(root / "outputs" / "cv_constrained")
            values["experiment"]["execution_mode"] = "train_from_archive_features"
            values["model"] = {
                "name": "rf_constrained",
                "params": {"n_estimators": 10, "n_jobs": 1, "n_iter": 2, "eta": 0.5, "tol": 1e-3},
            }
            values["tuning"] = {"name": "none", "params": {}}
            values["validation"]["cross_validation"] = {
                "name": "stratified_kfold",
                "params": {"n_splits": 2, "shuffle": False},
            }
            values["prediction"] = {
                "score_type": "probability",
                "normalization": "none",
                "export_geotiff": False,
            }
            validate_config(values)
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            manifest = experiment.run()
            self.assertEqual(manifest["status"], "completed")
            cv = manifest["independent_cv"]
            self.assertTrue(cv["run"])
            self.assertEqual(cv["n_splits"], 2)
            self.assertEqual(cv["n_valid_folds"], 2)
            self.assertEqual(cv["model_params_source"], "model.params")
            self.assertIn("f1", cv["aggregate_metrics"])

    def test_independent_cv_skips_when_too_few_samples(self) -> None:
        """P1-01: 训练样本过少（少数类 < n_splits）时独立 CV 记录跳过原因。"""
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(root / "outputs" / "cv_skip")
            values["experiment"]["execution_mode"] = "train_from_archive_features"
            values["model"] = {"name": "rf", "params": {"n_jobs": 1}}
            values["tuning"] = {"name": "none", "params": {}}
            values["validation"]["cross_validation"] = {
                "name": "stratified_kfold",
                "params": {"n_splits": 10, "shuffle": False},
            }
            values["prediction"] = {
                "score_type": "probability",
                "normalization": "none",
                "export_geotiff": False,
            }
            validate_config(values)
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            manifest = experiment.run()
            self.assertEqual(manifest["status"], "completed")
            cv = manifest["independent_cv"]
            self.assertFalse(cv["run"])
            self.assertIn("reason", cv)

    def test_per_sample_test_table_and_prediction_unit_id(self) -> None:
        """P1-10: 逐样本测试表与预测表的稳定 unit_id 闭环。"""
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(root / "outputs" / "per_sample")
            values["experiment"]["execution_mode"] = "train_from_archive_features"
            values["model"] = {"name": "rf", "params": {"n_jobs": 1}}
            values["tuning"] = {"name": "none", "params": {}}
            values["validation"]["cross_validation"] = {
                "name": "stratified_kfold",
                "params": {"n_splits": 2, "shuffle": False},
            }
            values["prediction"] = {
                "score_type": "probability",
                "normalization": "none",
                "export_geotiff": False,
            }
            validate_config(values)
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            manifest = experiment.run()
            self.assertEqual(manifest["status"], "completed")

            test_table_path = experiment.output_dir / "intermediate" / "test_predictions.csv"
            self.assertTrue(test_table_path.is_file())
            test_table = pd.read_csv(test_table_path)
            for column in ("row_index", "y_true", "y_pred", "score", "sample_weight"):
                self.assertIn(column, test_table.columns)
            self.assertEqual(len(test_table), 4)

            pred_path = experiment.output_dir / "predictions" / "target_probs.csv"
            pred = pd.read_csv(pred_path)
            self.assertIn("unit_id", pred.columns)
            self.assertEqual(list(pred["unit_id"]), list(range(len(pred))))

    def test_injection_audit_flags_unconsumed_knowledge(self) -> None:
        """P1-04: 启用了 knowledge 但无谓词消费时，manifest 记录可归因审计标志。"""
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(root / "outputs" / "audit")
            values["experiment"]["execution_mode"] = "train_from_archive_features"
            values["model"] = {"name": "rf", "params": {"n_jobs": 1}}
            values["tuning"] = {"name": "none", "params": {}}
            values["knowledge"] = {
                "enabled": True,
                "items": [{"name": "spatial_extent", "params": {}}],
            }
            values["predicates"] = {"enabled": False, "combine": "sequential", "items": []}
            values["prediction"] = {
                "score_type": "probability",
                "normalization": "none",
                "export_geotiff": False,
            }
            validate_config(values)
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            manifest = experiment.run()
            self.assertEqual(manifest["status"], "completed")
            audit = manifest["injection_audit"]
            self.assertTrue(audit["knowledge_enabled"])
            self.assertEqual(audit["knowledge_providers"], ["spatial_extent"])
            self.assertFalse(audit["predicates_enabled"])
            self.assertTrue(audit["unconsumed_knowledge"])
            self.assertFalse(audit["constraints_produced"])
            self.assertIn("constraints_consumed", audit)

    def test_torch_import_isolation_keeps_rf_path_working(self) -> None:
        """P1-05: 无 torch 环境下 RF/归档路径仍可导入构建，MLP 显式报可选依赖错误。"""
        import os
        import subprocess

        script = (
            "import importlib.abc\n"
            "import sys\n"
            "import numpy as np\n"
            "class _BlockTorch(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
            "            raise ImportError(f\"No module named '{fullname}'\")\n"
            "        return None\n"
            "sys.meta_path.insert(0, _BlockTorch())\n"
            "from src.core.bootstrap import load_builtin_components\n"
            "load_builtin_components()\n"
            "from src.models.registry import MODEL_REGISTRY\n"
            "from src.core.contracts import OptionalDependencyError\n"
            "rf = MODEL_REGISTRY.create('rf')\n"
            "assert rf.build({'n_jobs': 1}, 42) is not None\n"
            "X = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)\n"
            "y = np.array([0, 1, 0, 1])\n"
            "mlp = MODEL_REGISTRY.create('mlp').build({}, 42)\n"
            "try:\n"
            "    mlp.fit(X, y, sample_weight=np.ones(4))\n"
            "except OptionalDependencyError:\n"
            "    print('NO_TORCH_OK')\n"
            "else:\n"
            "    raise SystemExit('MLP fit without torch')\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO_TORCH_OK", result.stdout)

    def test_yaml_plugin_can_add_and_select_model_without_experiment_changes(self) -> None:
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            plugin = root / "dummy_plugin.py"
            plugin.write_text(
                "from sklearn.linear_model import LogisticRegression\n"
                "from src.models.registry import MODEL_REGISTRY\n"
                "@MODEL_REGISTRY.decorator('dummy_logistic')\n"
                "class DummyLogisticAdapter:\n"
                "    name = 'dummy_logistic'\n"
                "    artifact_filename = 'model_dummy_logistic.pkl'\n"
                "    supports_constraints = False\n"
                "    def build(self, params, seed):\n"
                "        resolved = dict(params)\n"
                "        resolved.setdefault('random_state', seed)\n"
                "        return LogisticRegression(**resolved)\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(root))
            try:
                output_dir = root / "outputs" / "plugin"
                values["plugins"] = ["dummy_plugin"]
                values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
                values["experiment"]["output_dir"] = str(output_dir)
                values["experiment"]["execution_mode"] = "train_from_archive_features"
                values["model"] = {
                    "name": "dummy_logistic",
                    "params": {"max_iter": 100, "solver": "liblinear"},
                }
                values["label_refinement"] = {"enabled": False, "name": "pub", "params": {}}
                values["tuning"] = {"name": "none", "params": {}}
                values["prediction"]["export_geotiff"] = False
                config_path = root / "plugin_experiment.yaml"
                config_path.write_text(
                    yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
                )
                plugin_config = load_config(config_path)
                experiment = Experiment(plugin_config)
                manifest = experiment.run()
            finally:
                sys.path.remove(str(root))
            self.assertEqual(manifest["components"]["model"], "dummy_logistic")
            self.assertTrue((experiment.output_dir / "models" / "model_dummy_logistic.pkl").is_file())

    def test_cli_overrides_merge_and_validate(self) -> None:
        config = load_config(LACHLAN_CONFIG)
        # 覆盖 model.name
        overridden = apply_cli_overrides(config, ["model.name=spe"])
        self.assertEqual(overridden.values["model"]["name"], "spe")
        self.assertEqual(overridden.spec.model.name, "spe")
        self.assertEqual(overridden.source_path, config.source_path)

        # 覆盖 tuning.params.n_iter（数值）
        overridden = apply_cli_overrides(config, ["tuning.params.n_iter=200"])
        self.assertEqual(overridden.values["tuning"]["params"]["n_iter"], 200)

        # 覆盖 experiment.seed（整数）
        overridden = apply_cli_overrides(config, ["experiment.seed=123"])
        self.assertEqual(overridden.values["experiment"]["seed"], 123)
        self.assertEqual(overridden.spec.seed, 123)

        # 覆盖布尔值
        overridden = apply_cli_overrides(config, ["prediction.export_geotiff=false"])
        self.assertFalse(overridden.values["prediction"]["export_geotiff"])

        # 多参数覆盖
        overridden = apply_cli_overrides(
            config,
            ["model.name=cnn", "tuning.name=none", "experiment.seed=999"],
        )
        self.assertEqual(overridden.values["model"]["name"], "cnn")
        self.assertEqual(overridden.values["tuning"]["name"], "none")
        self.assertEqual(overridden.values["experiment"]["seed"], 999)

        # 无效覆盖应抛出 ConfigError
        with self.assertRaises(ConfigError):
            apply_cli_overrides(config, ["model.name=not_a_model"])

        # 原配置不受影响（不可变语义）
        self.assertEqual(config.values["model"]["name"], "rf")

        # 有效的覆盖后配置可通过 Experiment 运行
        overridden = apply_cli_overrides(
            config,
            ["experiment.execution_mode=train_from_archive_features",
             "tuning.name=none",
             "prediction.export_geotiff=false"],
        )
        self.assertEqual(overridden.spec.execution_mode, "train_from_archive_features")

    def test_cli_set_rejects_list_values(self) -> None:
        """P0-3: --set features.operators=[...] 必须抛出清晰的 ConfigError"""
        config = load_config(LACHLAN_CONFIG)
        with self.assertRaisesRegex(ConfigError, "does not support list"):
            apply_cli_overrides(config, ["features.operators=[{name: test}]"])

    def test_cli_set_rejects_dict_values(self) -> None:
        """P0-3: --set model.params={...} 必须抛出清晰的 ConfigError"""
        config = load_config(LACHLAN_CONFIG)
        with self.assertRaisesRegex(ConfigError, "does not support list"):
            apply_cli_overrides(config, ["model.params={n_estimators: 10}"])

    def test_cli_set_rejects_list_index_path(self) -> None:
        """P0-3: --set predicates.items.0.name=all_ones 必须抛出清晰的 ConfigError"""
        config = load_config(LACHLAN_CONFIG)
        with self.assertRaisesRegex(ConfigError, "does not support list index"):
            apply_cli_overrides(config, ["predicates.items.0.name=all_ones"])

    def test_cli_override_resolves_dataset_path_relative_to_config_root(self) -> None:
        """P2: CLI 覆盖的 dataset 路径相对 config 根目录解析，而非 CWD。"""
        config = load_config(LACHLAN_CONFIG)
        overridden = apply_cli_overrides(config, ["dataset.root=tests"])
        self.assertEqual(
            Path(overridden.values["dataset"]["root"]), (ROOT / "tests").resolve()
        )

    def test_cli_override_rejects_nonexistent_dataset_path(self) -> None:
        """P2: CLI 覆盖指向不存在路径时明确失败，而不是静默忽略。"""
        config = load_config(LACHLAN_CONFIG)
        with self.assertRaisesRegex(ConfigError, "non-existent path"):
            apply_cli_overrides(config, ["dataset.root=definitely_missing_dir_xyz"])

    def test_cli_override_allows_existing_dataset_path(self) -> None:
        """P2: CLI 覆盖指向存在路径时通过。"""
        config = load_config(LACHLAN_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            overridden = apply_cli_overrides(config, [f"dataset.root={temporary}"])
            self.assertTrue(Path(overridden.values["dataset"]["root"]).is_dir())

    def test_rf_seed_propagation_same_seed_same_model(self) -> None:
        """P0-5: 相同 seed 产生相同的 RF 模型"""
        adapter = MODEL_REGISTRY.create("rf")
        model1 = adapter.build({"n_estimators": 10}, seed=42)
        model2 = adapter.build({"n_estimators": 10}, seed=42)
        self.assertEqual(model1.random_state, 42)
        self.assertEqual(model2.random_state, 42)

    def test_rf_seed_propagation_different_seed_different_behavior(self) -> None:
        """P0-5: 不同 seed 产生不同的 RF 随机行为"""
        adapter = MODEL_REGISTRY.create("rf")
        model1 = adapter.build({"n_estimators": 10}, seed=42)
        model2 = adapter.build({"n_estimators": 10}, seed=99)
        self.assertEqual(model1.random_state, 42)
        self.assertEqual(model2.random_state, 99)

    def test_rf_seed_overrides_yaml_random_state(self) -> None:
        """P0-5: build() 始终使用实验 seed，覆盖 YAML 中可能残留的 random_state"""
        adapter = MODEL_REGISTRY.create("rf")
        model = adapter.build({"n_estimators": 10, "random_state": 999}, seed=42)
        self.assertEqual(model.random_state, 42)

    def test_phase2_config_has_train_execution_mode(self) -> None:
        """P0-1: lachlan_rf_phase2.yaml 加载后 execution_mode 为 train_from_archive_features"""
        phase2 = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_phase2.yaml")
        self.assertEqual(phase2.values["experiment"]["execution_mode"], "train_from_archive_features")

    def test_replay_mode_rejects_unsupported_components(self) -> None:
        """P0-2: archive_replay 模式配置了 tuning 时验证失败（不再静默警告）。"""
        config = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(config.values)
        values["tuning"]["name"] = "bayes"
        with self.assertRaisesRegex(ConfigError, "archive_replay mode does not use tuning"):
            validate_config(values)

    def test_archive_features_mode_rejects_feature_operators(self) -> None:
        """P0-2: train_from_archive_features 模式声明特征算子时验证失败。"""
        phase2 = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_phase2.yaml")
        values = copy.deepcopy(phase2.values)
        values["features"]["operators"] = [
            {"name": "raster_statistics", "params": {"buffer_size": 10, "buffer_shape": "square"}}
        ]
        with self.assertRaisesRegex(ConfigError, "feature operators do not execute"):
            validate_config(values)

    def test_archive_features_mode_rejects_pub(self) -> None:
        """P0-2: train_from_archive_features 模式启用 PUB 时验证失败。"""
        phase2 = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_phase2.yaml")
        values = copy.deepcopy(phase2.values)
        values["label_refinement"] = {
            "enabled": True,
            "name": "pub",
            "params": {
                "n_iter": 1,
                "search_space": {
                    "bootstrap": {"type": "categorical", "values": [True]},
                    "max_depth": {"type": "integer", "low": 5, "high": 6},
                    "max_features": {"type": "categorical", "values": ["sqrt"]},
                    "min_samples_leaf": {"type": "integer", "low": 2, "high": 3},
                    "min_samples_split": {"type": "integer", "low": 2, "high": 3},
                    "n_estimators": {"type": "integer", "low": 10, "high": 11},
                },
            },
        }
        with self.assertRaisesRegex(ConfigError, "PUB label refinement does not execute"):
            validate_config(values)

    def test_strict_schema_rejects_unknown_keys(self) -> None:
        """P0-2: 严格 schema 拒绝未知/拼写错误的配置键。"""
        phase2 = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_phase2.yaml")

        values = copy.deepcopy(phase2.values)
        values["model"]["paramz"] = {}
        with self.assertRaisesRegex(ConfigError, "Unknown configuration key"):
            validate_config(values)

        values = copy.deepcopy(phase2.values)
        values["label"]["strategy"] = "does_not_exist"
        with self.assertRaisesRegex(ConfigError, "label.strategy"):
            validate_config(values)

        values = copy.deepcopy(phase2.values)
        values["research_unit"]["train_positive"] = "does_not_exist"
        with self.assertRaisesRegex(ConfigError, "train_positive"):
            validate_config(values)

    def test_validate_reproduction_rejects_zero_run_false_positive(self) -> None:
        """P0-7: 零运行验证不得报告"可正确复现"。"""
        sys.path.insert(0, str(ROOT / "scripts"))
        import validate_reproduction

        report = validate_reproduction.generate_report(
            {"total": 0, "passed": 0, "failed": 0, "details": []},
            [],
            {},
        )
        self.assertIn("未验证复现", report)
        self.assertNotIn("可正确复现", report)

    def test_run_manifest_records_reproducibility_fields(self) -> None:
        """P0-7: manifest 记录 run_id/status/environment/git_dirty/输出哈希。"""
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(root / "outputs" / "replay")
            values["experiment"]["execution_mode"] = "archive_replay"
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            experiment.run()
            stored = json.loads((experiment.output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["status"], "completed")
            self.assertIn("run_id", stored)
            self.assertIn("environment", stored)
            self.assertIn("python_version", stored["environment"])
            self.assertIn("git_dirty", stored)
            self.assertTrue(any("sha256" in item for item in stored["output_files"]))

    def test_failed_run_writes_failure_manifest(self) -> None:
        """P2: 运行失败仍写出 status=failed 的 manifest 并记录异常摘要。"""
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _write_synthetic_archive(root)
            (archive / "model_rf.pkl").unlink()  # 触发 archive_replay train 失败
            values["dataset"] = {"root": str(root), "archive_dir": str(archive)}
            values["experiment"]["output_dir"] = str(root / "outputs" / "replay")
            values["experiment"]["execution_mode"] = "archive_replay"
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            with self.assertRaises(FileNotFoundError):
                experiment.run()
            manifest_path = experiment.output_dir / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["status"], "failed")
            self.assertIn("failure", stored)
            self.assertEqual(stored["failure"]["exception_type"], "FileNotFoundError")
            self.assertEqual(stored["failure"]["phase"], "train")

    def test_raw_gis_run_persists_fitted_preprocessor(self) -> None:
        """P1-07: raw_gis 成功运行持久化已拟合预处理器（可独立重建预处理语义）。"""
        try:
            import geopandas  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("geopandas/shapely not installed")
        loaded = load_config(RAW_GIS_CONFIG)
        values = copy.deepcopy(loaded.values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values["dataset"] = _write_synthetic_raw_gis(root)
            values["experiment"]["output_dir"] = str(root / "outputs" / "rawgis")
            values["prediction"] = {
                "score_type": "probability",
                "normalization": "none",
                "export_geotiff": False,
            }
            experiment = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path))
            experiment.run()
            preprocessor_path = experiment.output_dir / "models" / "preprocessor.pkl"
            self.assertTrue(preprocessor_path.is_file())
            with preprocessor_path.open("rb") as handle:
                restored = pickle.load(handle)
            self.assertIsInstance(restored, BaselinePreprocessor)
            self.assertTrue(restored.scaler_fitted)
            self.assertTrue(restored.encoder_fitted)
            self.assertTrue(restored.feature_columns)
            manifest = json.loads(
                (experiment.output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertIn(
                "models/preprocessor.pkl",
                [item["path"] for item in manifest["output_files"]],
            )

    def test_setup_logging_closes_previous_handlers(self) -> None:
        """P0-7: 连续多个实验目录不应累积 FileHandler。"""
        import logging as _logging

        from src.utils.logging import setup_logging

        logger = _logging.getLogger("mpm")
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("a", "b", "c"):
                setup_logging(Path(temporary) / name)
        file_handlers = [h for h in logger.handlers if isinstance(h, _logging.FileHandler)]
        self.assertEqual(len(file_handlers), 1)

    def test_spatial_block_kfold_requires_coordinates(self) -> None:
        """P0-4: spatial_block_kfold 缺少坐标时抛出明确错误"""
        data = TrainingData(
            pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0, 1]),
            pd.Series([1.0, 1.0, 1.0, 1.0]),
        )
        splitter = SPLITTER_REGISTRY.create("spatial_block_kfold")
        with self.assertRaisesRegex(ValueError, "coordinate metadata"):
            splitter.build_cv({"n_splits": 2, "block_size_m": 1000}, 42, data)

    def test_spatial_block_kfold_separates_train_test_spatially(self) -> None:
        """P0-4: 空间块划分后 train 和 test 的空间坐标不重叠"""
        data = TrainingData(
            pd.DataFrame({"a": np.arange(12, dtype=float)}),
            pd.Series([0, 1] * 6),
            pd.Series(np.ones(12)),
            metadata={
                "units": pd.DataFrame({
                    "X": [0.0, 10.0, 100.0, 110.0, 200.0, 210.0,
                          0.0, 10.0, 100.0, 110.0, 200.0, 210.0],
                    "Y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                })
            },
        )
        splitter = SPLITTER_REGISTRY.create("spatial_block_kfold")
        cv = splitter.build_cv(
            {"n_splits": 2, "block_size_m": 50, "shuffle": False}, 42, data
        )
        folds = list(cv.split(data.features, data.labels))
        self.assertEqual(len(folds), 2)
        for train_idx, test_idx in folds:
            train_coords = data.metadata["units"].iloc[train_idx]
            test_coords = data.metadata["units"].iloc[test_idx]
            # 确保 train 和 test 的空间块不重叠
            train_blocks = set(
                (int(x // 50), int(y // 50))
                for x, y in zip(train_coords["X"], train_coords["Y"])
            )
            test_blocks = set(
                (int(x // 50), int(y // 50))
                for x, y in zip(test_coords["X"], test_coords["Y"])
            )
            self.assertTrue(
                train_blocks.isdisjoint(test_blocks),
                f"Train and test spatial blocks overlap: {train_blocks & test_blocks}"
            )

    def test_spatial_block_holdout_separates_spatially(self) -> None:
        """P0-4: spatial_block_holdout train/test 在空间上不重叠"""
        data = TrainingData(
            pd.DataFrame({"a": np.arange(12, dtype=float)}),
            pd.Series([0, 1] * 6),
            pd.Series(np.ones(12)),
            metadata={
                "units": pd.DataFrame({
                    "X": [0.0, 10.0, 100.0, 110.0, 200.0, 210.0,
                          0.0, 10.0, 100.0, 110.0, 200.0, 210.0],
                    "Y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                })
            },
        )
        splitter = SPLITTER_REGISTRY.create("spatial_block_holdout")
        result = splitter.split(data, {"test_size": 0.5, "block_size_m": 50}, 42)
        train_coords = result.train.metadata["units"]
        test_coords = result.test.metadata["units"]
        train_blocks = set(
            (int(x // 50), int(y // 50))
            for x, y in zip(train_coords["X"], train_coords["Y"])
        )
        test_blocks = set(
            (int(x // 50), int(y // 50))
            for x, y in zip(test_coords["X"], test_coords["Y"])
        )
        self.assertTrue(
            train_blocks.isdisjoint(test_blocks),
            f"Train and test spatial blocks overlap: {train_blocks & test_blocks}"
        )

    def test_spatial_group_kfold_injects_groups_without_explicit_arg(self) -> None:
        """P0-4: spatial_group_kfold 返回的 CV 无需调用方显式传 groups。"""
        data = TrainingData(
            pd.DataFrame({"a": np.arange(12, dtype=float)}),
            pd.Series([0, 1] * 6),
            pd.Series(np.ones(12)),
            metadata={
                "units": pd.DataFrame({
                    "X": np.arange(12, dtype=float),
                    "Y": np.arange(12, dtype=float),
                    "group_id": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                })
            },
        )
        splitter = SPLITTER_REGISTRY.create("spatial_group_kfold")
        cv = splitter.build_cv({"n_splits": 3, "group_column": "group_id"}, 42, data)
        folds = list(cv.split(data.features, data.labels))  # 不传 groups 也应成功
        self.assertEqual(len(folds), 3)
        for train_idx, test_idx in folds:
            train_groups = set(data.metadata["units"].iloc[train_idx]["group_id"])
            test_groups = set(data.metadata["units"].iloc[test_idx]["group_id"])
            self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_spatial_block_kfold_rejects_degree_coordinates(self) -> None:
        """P0-4: 经纬度坐标下 spatial_block_kfold 给出明确错误而非单块。"""
        data = TrainingData(
            pd.DataFrame({"a": np.arange(6, dtype=float)}),
            pd.Series([0, 1] * 3),
            pd.Series(np.ones(6)),
            metadata={
                "units": pd.DataFrame({
                    "X": [147.0, 147.1, 147.2, 147.3, 147.4, 147.5],
                    "Y": [-35.0, -35.1, -35.2, -35.3, -35.4, -35.5],
                })
            },
        )
        splitter = SPLITTER_REGISTRY.create("spatial_block_kfold")
        with self.assertRaisesRegex(ValueError, "decimal degrees"):
            splitter.build_cv({"n_splits": 2, "block_size_m": 50000}, 42, data)

    def test_research_units_union_crs_and_sampling_contract(self) -> None:
        """P0-6: 边界需 CRS、多要素取并集、采样达到请求数量。"""
        try:
            import geopandas as gpd
            from shapely.geometry import box
        except ImportError:  # pragma: no cover
            self.skipTest("geopandas/shapely not installed")
        from src.data.research_units import build_prediction_grid, sample_unlabeled_units

        label_config = {"unlabeled_value": 0, "sample_weight": {"unlabeled": 0.5}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            # 无 CRS 边界 → 明确失败
            no_crs = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
            no_crs_path = root / "no_crs.shp"
            no_crs.to_file(no_crs_path)
            with self.assertRaisesRegex(ValueError, "CRS"):
                sample_unlabeled_units(str(no_crs_path), 5, label_config, seed=42)

            # 多要素边界：并集（dissolve）后仍能采样/建网格
            multi = gpd.GeoDataFrame(
                geometry=[box(0, 0, 5, 5), box(5, 0, 10, 5)], crs="EPSG:3857"
            )
            multi_path = root / "multi.shp"
            multi.to_file(multi_path)

            sampled = sample_unlabeled_units(str(multi_path), 20, label_config, seed=42)
            self.assertEqual(len(sampled), 20)
            self.assertTrue(((sampled["X"] >= 0) & (sampled["X"] <= 10)).all())

            grid, _mask = build_prediction_grid(str(multi_path), 2.0)
            self.assertGreater(len(grid), 0)
            # 网格覆盖整个并集边界（含第二个 box）
            self.assertGreater(grid["X"].max(), 5)

    def test_research_variable_registries_are_populated(self) -> None:
        """P1-1: 研究变量注册表由内置实现填充，取代硬编码枚举。"""
        from src.data.registries import (
            BACKGROUND_SAMPLER_REGISTRY,
            LABEL_STRATEGY_REGISTRY,
            RESEARCH_UNIT_REGISTRY,
            WEIGHT_STRATEGY_REGISTRY,
        )

        self.assertEqual(RESEARCH_UNIT_REGISTRY.names(), ("point_local_environment",))
        self.assertEqual(BACKGROUND_SAMPLER_REGISTRY.names(), ("random_points_in_nsw_boundary",))
        self.assertEqual(LABEL_STRATEGY_REGISTRY.names(), ("positive_unlabeled_as_zero",))
        self.assertEqual(WEIGHT_STRATEGY_REGISTRY.names(), ("size_code", "uniform"))

    def test_weight_strategy_size_code_maps_attributes(self) -> None:
        """P1-1: size_code 权重策略映射 SIZE_CODE 到配置权重。"""
        from src.data.registries import WEIGHT_STRATEGY_REGISTRY

        attributes = pd.DataFrame({"SIZE_CODE": ["VLG", "MED", "SML"]})
        weights = {"VLG": 0.5, "LGE": 0.4, "MED": 0.3, "SML": 0.2, "OCC": 0.1}
        strategy = WEIGHT_STRATEGY_REGISTRY.create("size_code", {})
        result = strategy.compute(attributes, weights)
        self.assertEqual(result.tolist(), [0.5, 0.3, 0.2])

    def test_weight_strategy_size_code_rejects_missing_code(self) -> None:
        """P1-1: size_code 权重策略对未配置的 SIZE_CODE 明确失败。"""
        from src.data.registries import WEIGHT_STRATEGY_REGISTRY

        attributes = pd.DataFrame({"SIZE_CODE": ["VLG", "UNKNOWN"]})
        weights = {"VLG": 0.5}
        strategy = WEIGHT_STRATEGY_REGISTRY.create("size_code", {})
        with self.assertRaisesRegex(ValueError, "SIZE_CODE"):
            strategy.compute(attributes, weights)

    def test_label_strategy_positive_unlabeled_as_zero(self) -> None:
        """P1-1: PU 标签策略编码正样本与未标注样本。"""
        from src.data.registries import LABEL_STRATEGY_REGISTRY

        label_config = {
            "strategy": "positive_unlabeled_as_zero",
            "positive_value": 1,
            "unlabeled_value": 0,
            "sample_weight": {"VLG": 0.5, "LGE": 0.4, "MED": 0.3, "SML": 0.2, "OCC": 0.1, "unlabeled": 0.5},
        }
        strategy = LABEL_STRATEGY_REGISTRY.create("positive_unlabeled_as_zero", {})

        occurrences = pd.DataFrame({"SIZE_CODE": ["VLG", "MED"]})
        positive = strategy.apply_positive(occurrences, label_config)
        self.assertEqual(positive["label"].tolist(), [1, 1])
        self.assertEqual(positive["sample_weight"].tolist(), [0.5, 0.3])

        points = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
        unlabeled = strategy.apply_unlabeled(points, label_config)
        self.assertEqual(unlabeled["label"].tolist(), [0, 0])
        self.assertEqual(unlabeled["sample_weight"].tolist(), [0.5, 0.5])

    def test_task_builds_units_via_registry(self) -> None:
        """P1-1: Task 通过注册表构建研究单元，而非直接 import 模块函数。"""
        try:
            import geopandas as gpd
            from shapely.geometry import box
        except ImportError:  # pragma: no cover
            self.skipTest("geopandas/shapely not installed")

        label_config = {
            "strategy": "positive_unlabeled_as_zero",
            "positive_value": 1,
            "unlabeled_value": 0,
            "sample_weight": {"VLG": 0.5, "LGE": 0.4, "MED": 0.3, "SML": 0.2, "OCC": 0.1, "unlabeled": 0.5},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3857")
            boundary_path = root / "boundary.shp"
            boundary.to_file(boundary_path)

            research_unit_config = {
                "type": "point_local_environment",
                "train_unlabeled": "random_points_in_nsw_boundary",
                "prediction_grid_size": 2.0,
            }
            dataset_config = {"training_boundary": str(boundary_path), "boundary": str(boundary_path)}

            task = TargetAreaPredictionTask({}, {"score_type": "probability", "normalization": "none", "export_geotiff": False})
            unlabeled = task.build_unlabeled_units(
                dataset_config, research_unit_config, label_config, count=10, seed=42
            )
            self.assertEqual(len(unlabeled), 10)
            self.assertTrue(((unlabeled["X"] >= 0) & (unlabeled["X"] <= 10)).all())

            units, mask = task.build_prediction_units(dataset_config, research_unit_config)
            self.assertGreater(len(units), 0)

    def test_score_type_contract_rejects_minmax_probability(self) -> None:
        """P0-5: probability/raw_score + minmax 是非法组合，验证失败。"""
        with self.assertRaisesRegex(ValueError, "relative_score"):
            TargetAreaPredictionTask.validate_config(
                {"name": "target_area_prediction"},
                {"type": "point_local_environment", "prediction_grid_size": 0.1},
                {"score_type": "probability", "normalization": "minmax", "export_geotiff": False},
                {},
                "raw_gis",
            )

    def test_score_type_raw_score_requires_decision_function(self) -> None:
        """P0-5: raw_score 需要 decision_function，仅 predict_proba 的模型报错。"""

        class ProbOnlyModel:
            def predict_proba(self, features):
                return np.column_stack([np.ones(len(features)) * 0.3, np.ones(len(features)) * 0.7])

        task = TargetAreaPredictionTask(
            {}, {"score_type": "raw_score", "normalization": "none", "export_geotiff": False}
        )
        with self.assertRaisesRegex(ValueError, "decision_function"):
            task.predict(
                ProbOnlyModel(),
                pd.DataFrame({"a": [0.0]}),
                pd.DataFrame({"X": [1.0], "Y": [2.0]}),
            )

    def test_score_type_relative_score_records_range(self) -> None:
        """P0-5: relative_score + minmax 输出列名为 relative_score 并记录范围。"""

        class ProbModel:
            def predict_proba(self, features):
                return np.array([[0.8, 0.2], [0.6, 0.4]])

        task = TargetAreaPredictionTask(
            {}, {"score_type": "relative_score", "normalization": "minmax", "export_geotiff": False}
        )
        result = task.predict(
            ProbModel(),
            pd.DataFrame({"a": [0.0, 1.0]}),
            pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]}),
        )
        self.assertIn("relative_score", result.columns)
        self.assertNotIn("prob", result.columns)
        self.assertTrue((result["relative_score"] >= 0).all())
        self.assertTrue((result["relative_score"] <= 1).all())
        self.assertIsNotNone(task._normalization_range)

    def test_additional_classification_metrics(self) -> None:
        """P0-5: average_precision / balanced_accuracy / mcc 可计算。"""

        class FixedModel:
            def predict(self, features):
                return np.array([0, 1, 0, 1])

            def predict_proba(self, features):
                return np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

        result = evaluate_classifier(
            FixedModel(),
            pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0, 1]),
            pd.Series([1.0, 1.0, 1.0, 1.0]),
            ["average_precision", "balanced_accuracy", "mcc"],
        )
        self.assertIn("average_precision", result)
        self.assertIn("balanced_accuracy", result)
        self.assertIn("mcc", result)
        self.assertAlmostEqual(result["balanced_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(result["mcc"], 1.0, places=6)

    def test_mpm_area_metrics(self) -> None:
        """P0-5: prediction-rate AUC 与面积捕获率符合预期。"""
        from src.validation.mpm_metrics import (
            capture_rate_at_area_fraction,
            prediction_rate_auc,
        )

        # 完美排序：正类始终排在负类之前。
        # 正类面积占比 0.5，理想 AUC = 1 - 0.5/2 = 0.75；50% 面积即可捕获全部正类。
        perfect_scores = np.array([0.9, 0.8, 0.3, 0.2])
        perfect_labels = np.array([1, 1, 0, 0])
        area = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(prediction_rate_auc(perfect_scores, perfect_labels, area), 0.75, places=2)
        self.assertAlmostEqual(
            capture_rate_at_area_fraction(perfect_scores, perfect_labels, area, 0.5), 1.0, places=2
        )

        # 随机排序 → AUC≈0.5
        rng = np.random.default_rng(0)
        random_scores = rng.random(2000)
        random_labels = rng.integers(0, 2, 2000).astype(float)
        random_area = np.ones(2000)
        self.assertAlmostEqual(
            prediction_rate_auc(random_scores, random_labels, random_area), 0.5, delta=0.05
        )

    def test_mpm_metrics_are_configurable(self) -> None:
        """P1-03: MPM 面积指标可配置到 validation.metrics 并通过校验。"""
        loaded = load_config(LACHLAN_CONFIG)
        values = copy.deepcopy(loaded.values)
        values["validation"]["metrics"] = ["f1", "prediction_rate_auc"]
        values["validation"]["primary_metric"] = "f1"
        validate_config(values)  # 不抛异常

    def test_evaluate_classifier_computes_configured_mpm_metrics(self) -> None:
        """P1-03: evaluate_classifier 按配置计算 MPM 面积指标（unit_area 存在时）。"""

        class FixedModel:
            def predict(self, features):
                return np.array([0, 1, 0, 1])

            def predict_proba(self, features):
                return np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

        result = evaluate_classifier(
            FixedModel(),
            pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0, 1]),
            pd.Series([1.0, 1.0, 1.0, 1.0]),
            ["f1", "prediction_rate_auc"],
            unit_area=pd.Series([1.0, 1.0, 1.0, 1.0]),
        )
        self.assertIn("f1", result)
        self.assertIn("prediction_rate_auc", result)

    def test_evaluate_classifier_records_reason_when_unit_area_missing(self) -> None:
        """P1-03: 请求 MPM 面积指标但 unit_area 缺失时记录明确原因。"""

        class FixedModel:
            def predict(self, features):
                return np.array([0, 1, 0, 1])

            def predict_proba(self, features):
                return np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

        result = evaluate_classifier(
            FixedModel(),
            pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0, 1]),
            pd.Series([1.0, 1.0, 1.0, 1.0]),
            ["prediction_rate_auc"],
        )
        self.assertIsNone(result["prediction_rate_auc"])
        self.assertEqual(
            result["prediction_rate_auc_not_computed_reason"], "unit_area not provided"
        )

    def test_prediction_grid_carries_unit_area(self) -> None:
        """P1-03: point_local_environment 预测网格携带 unit_area = grid_size²。"""
        try:
            import geopandas as gpd
            from shapely.geometry import box
        except ImportError:  # pragma: no cover
            self.skipTest("geopandas/shapely not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3857")
            boundary_path = root / "boundary.shp"
            boundary.to_file(boundary_path)
            task = TargetAreaPredictionTask(
                {}, {"score_type": "probability", "normalization": "none", "export_geotiff": False}
            )
            units, _mask = task.build_prediction_units(
                {"boundary": str(boundary_path)},
                {"type": "point_local_environment", "prediction_grid_size": 2.0},
            )
            self.assertEqual(task.unit_area, 4.0)
            self.assertEqual(units.attrs["unit_area"], 4.0)

    def test_multiple_constraint_predicates_merge_phi_vectors(self) -> None:
        """P1-2: 多个 constraint 谓词合并为 phi_vectors，不互相覆盖。"""
        data = TrainingData(
            pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0, 1]),
            pd.Series([1.0, 1.0, 1.0, 1.0]),
            metadata={
                "units": pd.DataFrame({"X": [0.0, 1.0, 10.0, 11.0], "Y": [0.0, 1.0, 10.0, 11.0]})
            },
        )
        pipeline = PredicatePipeline([ComponentSpec("all_ones"), ComponentSpec("spatial_box")])
        result = pipeline.apply(data, {"knowledge": {}})
        self.assertIn("phi_vectors", result.constraints)
        self.assertNotIn("phi_vector", result.constraints)
        names = [vector["name"] for vector in result.constraints["phi_vectors"]]
        self.assertEqual(names, ["all_ones", "spatial_box"])
        for vector in result.constraints["phi_vectors"]:
            self.assertEqual(len(vector["vector"]), 4)

    def test_weighted_mse_applies_sample_weight(self) -> None:
        """P1-2: LUSI 谓词损失的 MSE 项应用 sample_weight。"""
        try:
            import torch
        except ImportError:  # pragma: no cover
            self.skipTest("torch not installed")
        from src.training.losses import weighted_mse_with_predicate

        pred = torch.tensor([0.0, 0.9, 0.0, 0.0])
        target = torch.zeros(4)
        phi = torch.ones(4)
        tau_hat = torch.tensor(0.5)
        tau = torch.tensor(0.5)

        unweighted = weighted_mse_with_predicate(pred, target, phi, tau_hat, tau)
        weighted = weighted_mse_with_predicate(
            pred, target, phi, tau_hat, tau,
            sample_weight=torch.tensor([0.0, 10.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(unweighted["mse"].item(), 0.2025, places=5)
        self.assertAlmostEqual(weighted["mse"].item(), 0.81, places=5)

    def test_deep_edge_task_is_registered_but_explicitly_unavailable(self) -> None:
        with self.assertRaises(TaskCapabilityError):
            create_task(
                "deep_edge_prediction",
                {"score_type": "probability", "normalization": "minmax", "export_geotiff": True},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
