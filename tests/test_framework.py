"""二阶段可配置实验框架的可移植测试。"""

from __future__ import annotations

import copy
import hashlib
import json
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
from src.core.config import ConfigError, ExperimentConfig, load_config, validate_config
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


class FrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        load_builtin_components()

    def test_config_load_migrates_phase1_yaml_to_phase2_spec(self) -> None:
        config = load_config(LACHLAN_CONFIG)
        self.assertEqual(config.name, "lachlan_rf_baseline")
        self.assertEqual(config.spec.task.name, "target_area_prediction")
        self.assertEqual(config.spec.model.name, "rf")
        self.assertEqual(config.spec.tuner.name, "bayes")
        self.assertEqual(config.spec.holdout.name, "random_holdout")
        self.assertEqual(config.spec.cross_validation.name, "stratified_kfold")
        self.assertEqual(
            [item.name for item in config.spec.feature_operators],
            ["raster_statistics", "texture", "elevation_gradient", "line_distance", "categorical_geology"],
        )
        phase2 = load_config(ROOT / "configs" / "experiments" / "lachlan_rf_phase2.yaml")
        for section in (
            "features",
            "model",
            "label_refinement",
            "tuning",
            "validation",
            "research_unit",
            "prediction",
        ):
            self.assertEqual(config.values[section], phase2.values[section])

    def test_config_validation_uses_registry_names_and_component_params(self) -> None:
        config = load_config(LACHLAN_CONFIG)
        invalid = copy.deepcopy(config.values)
        invalid["features"]["operators"][0]["params"]["buffer_size"] = 0
        with self.assertRaisesRegex(ConfigError, "buffer_size"):
            validate_config(invalid)

        invalid = copy.deepcopy(config.values)
        invalid["task"]["name"] = "not_a_task"
        with self.assertRaisesRegex(ConfigError, "Unknown task"):
            validate_config(invalid)

        invalid = copy.deepcopy(config.values)
        invalid["validation"]["metrics"].remove(invalid["validation"]["primary_metric"])
        with self.assertRaisesRegex(ConfigError, "primary_metric"):
            validate_config(invalid)

        invalid = copy.deepcopy(config.values)
        invalid["model"]["params"]["not_an_rf_parameter"] = True
        with self.assertRaises(ConfigError):
            validate_config(invalid)

        invalid = copy.deepcopy(config.values)
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
        extended = copy.deepcopy(config.values)
        extended["validation"]["holdout"] = {
            "name": "dummy_holdout",
            "params": {"token": "accepted"},
        }
        validate_config(extended)

        raw_gis = copy.deepcopy(config.values)
        raw_gis["experiment"]["execution_mode"] = "raw_gis"
        raw_gis["dataset"].pop("archive_dir")
        validate_config(raw_gis)

        archive = copy.deepcopy(raw_gis)
        archive["experiment"]["execution_mode"] = "archive_replay"
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
        config = load_config(LACHLAN_CONFIG).values
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
            self.assertTrue((output_dir / "config_resolved.yaml").is_file())
            self.assertTrue((output_dir / "manifest.json").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertEqual((output_dir / "models" / "model_rf.pkl").read_bytes(), b"synthetic-model")
            self.assertEqual((output_dir / "predictions" / "probability_map.tif").read_bytes(), b"synthetic-tiff")
            stored = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["components"]["feature_operators"][0], "raster_statistics")
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
            values["predicates"] = {"enabled": True, "combine": "sequential", "items": [{"name": "identity", "params": {}}]}
            validate_config(values)

            manifest = Experiment(ExperimentConfig(values=values, source_path=loaded.source_path)).run()
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["components"]["tuner"], "none")
            self.assertEqual(manifest["components"]["knowledge"], ["empty"])
            self.assertEqual(manifest["components"]["predicates"], ["identity"])
            self.assertEqual(
                manifest["component_metadata"]["label_refinement_execution"],
                "precomputed_in_archive",
            )
            self.assertEqual(
                manifest["component_metadata"]["holdout_execution"],
                "precomputed_in_archive",
            )
            self.assertNotIn("cross_validation", manifest["component_metadata"])
            self.assertIn("f1", metrics)
            self.assertTrue((output_dir / "models" / "model_rf.pkl").is_file())
            self.assertTrue((output_dir / "predictions" / "target_probs.csv").is_file())
            self.assertFalse((output_dir / "predictions" / "probability_map.tif").exists())

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
                manifest = Experiment(plugin_config).run()
            finally:
                sys.path.remove(str(root))
            self.assertEqual(manifest["components"]["model"], "dummy_logistic")
            self.assertTrue((output_dir / "models" / "model_dummy_logistic.pkl").is_file())

    def test_deep_edge_task_is_registered_but_explicitly_unavailable(self) -> None:
        with self.assertRaises(TaskCapabilityError):
            create_task(
                "deep_edge_prediction",
                {"score_type": "probability", "normalization": "minmax", "export_geotiff": True},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
