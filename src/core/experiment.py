"""由已注册组件组装而成的二阶段实验编排器。"""

from __future__ import annotations

import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger, setup_logging
from ..data.dataset import ArchiveDataset, DatasetRepository
from ..features.preprocess import BaselinePreprocessor
from ..knowledge.pipeline import KnowledgePipeline
from ..models.registry import LABEL_REFINER_REGISTRY, MODEL_REGISTRY
from ..operators.features.pipeline import FeaturePipeline
from ..predicates.pipeline import PredicatePipeline
from ..tasks.registry import create_task
from ..tuning.registry import TUNER_REGISTRY
from ..utils.files import git_commit, sha256_file, utc_timestamp, write_json, write_yaml
from ..validation.metrics import evaluate_classifier
from ..validation.splitters import create_holdout
from .bootstrap import load_builtin_components
from .config import ExperimentConfig
from .contracts import SplitData, TrainingData


class Experiment:
    """运行一个已解析实验，无需了解具体的 Model、Operator 或 Predicate 实现。"""

    def __init__(self, config: ExperimentConfig):
        load_builtin_components()
        self.config = config
        self.values = config.values
        self.spec = config.spec
        self.output_dir = config.output_dir
        # 追加时间戳后缀，防止重复运行同一配置时覆盖前次结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_dir.parent / f"{self.output_dir.name}_{timestamp}"
        self.values["experiment"]["output_dir"] = str(self.output_dir)
        self.dataset: ArchiveDataset | None = None
        self.preprocessor: BaselinePreprocessor | None = None
        self.model: Any | None = None
        self.metrics: dict[str, Any] = {}
        self.output_files: list[Path] = []
        self.component_metadata: dict[str, Any] = {}
        self.task = create_task(self.spec.task.name, self.values["task"], self.values["prediction"])
        self.model_adapter = MODEL_REGISTRY.create(self.spec.model.name)
        self.knowledge_pipeline = KnowledgePipeline(self.spec.knowledge)
        self.predicate_pipeline = PredicatePipeline(self.spec.predicates)
        self._prepare_output_layout()
        self._log = setup_logging(self.output_dir)
        self._log.info(
            "Experiment '%s' initialized (mode=%s, task=%s, model=%s, tuner=%s)",
            self.spec.name, self.mode, self.spec.task.name,
            self.spec.model.name, self.spec.tuner.name,
        )

    @property
    def mode(self) -> str:
        return self.spec.execution_mode

    def _prepare_output_layout(self) -> None:
        for directory in (
            self.output_dir,
            self.output_dir / "models",
            self.output_dir / "intermediate",
            self.output_dir / "predictions",
            self.output_dir / "figures",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / "config_resolved.yaml"
        write_yaml(config_path, self.values)
        self.output_files.append(config_path)

    def prepare_data(self) -> None:
        """加载归档数据表，或委托所选 Task 构建原始研究单元。"""
        if self.mode in {"archive_replay", "train_from_archive_features"}:
            self._log.info("Loading archive dataset from %s", self.values["dataset"]["archive_dir"])
            self.dataset = DatasetRepository(self.values["dataset"]).load_archive()
            summary = self.dataset.summary()
            self._log.info(
                "Archive loaded: %d features, train=%d rows, test=%d rows",
                summary["feature_count"],
                summary["xy_rf_train_shape"][0],
                summary["xy_rf_test_shape"][0],
            )
            return
        if self.mode != "raw_gis":
            raise ValueError(f"Unknown execution mode: {self.mode}")
        self._raw_positive_units = self.task.build_positive_units(
            self.values["dataset"], self.values["research_unit"], self.values["label"]
        )
        self._research_unit_columns = [
            column
            for column in self._raw_positive_units.columns
            if column not in {"label", "sample_weight"}
        ]

    def build_features(self) -> None:
        """快照归档特征，或运行已配置的 Feature Operator 管线。"""
        if self.mode in {"archive_replay", "train_from_archive_features"}:
            assert self.dataset is not None
            self._log.info("Snapshotting archive features to %s/intermediate", self.output_dir)
            self.output_files.extend(DatasetRepository.snapshot(self.dataset, self.output_dir))
            return

        self._log.info("Running feature operators: %s", [op.name for op in self.spec.feature_operators])
        pipeline = FeaturePipeline(self.values["dataset"], self.spec.feature_operators)
        deposits = pipeline.extract(self._raw_positive_units).dropna().reset_index(drop=True)
        unlabelled_units = self.task.build_unlabeled_units(
            self.values["dataset"],
            self.values["research_unit"],
            self.values["label"],
            len(deposits),
            seed=self.spec.seed,
        )
        unlabelled = pipeline.extract(unlabelled_units).dropna().reset_index(drop=True)
        unlabelled = unlabelled.loc[:, deposits.columns]
        target_units, target_mask = self.task.build_prediction_units(
            self.values["dataset"], self.values["research_unit"]
        )

        self._raw_deposits = deposits
        self._raw_unlabelled = unlabelled
        self._raw_target_data = pipeline.extract(target_units)
        self._raw_target_mask = target_mask.reset_index(drop=True)
        self.component_metadata["feature_operators"] = pipeline.names

    def prepare_dataset(self) -> None:
        """校验归档 schema，或应用保留的基线预处理契约。"""
        if self.mode in {"archive_replay", "train_from_archive_features"}:
            assert self.dataset is not None
            self._log.info("Validating archive schema")
            self.dataset.validate_schema()
            return

        self.preprocessor = BaselinePreprocessor(
            self.values["preprocess"], self.values["experiment"]["seed"]
        )
        # 不在此处拟合；预处理拟合推迟到外层划分之后的训练折内进行（split-first）。
        self._raw_training_frame = pd.concat(
            [self._raw_deposits, self._raw_unlabelled], ignore_index=True
        )
        target, target_coords, target_mask = self.task.prepare_prediction_data(
            self._raw_target_data, self._raw_target_mask
        )

        self._raw_target_frame = target
        self._raw_target_coords = target_coords
        self._raw_target_mask = target_mask

    @staticmethod
    def _training_from_frame(
        frame: pd.DataFrame,
        feature_columns: list[str],
        *,
        units: pd.DataFrame | None = None,
    ) -> TrainingData:
        return TrainingData(
            frame[feature_columns].reset_index(drop=True),
            frame["label"].reset_index(drop=True),
            frame["sample_weight"].reset_index(drop=True),
            metadata={"units": units.reset_index(drop=True)} if units is not None else {},
        )

    def _apply_label_refinement(self, data: TrainingData) -> TrainingData:
        if self.spec.label_refinement is None:
            return data
        refiner = LABEL_REFINER_REGISTRY.create(self.spec.label_refinement.name)
        result = refiner.refine(
            data,
            self.spec.label_refinement.params,
            self.values["validation"],
            self.spec.seed,
        )
        refiner_path = self.output_dir / "models" / getattr(
            refiner, "artifact_filename", f"label_refiner_{self.spec.label_refinement.name}.pkl"
        )
        with refiner_path.open("wb") as handle:
            pickle.dump(result.model, handle)
        self.output_files.append(refiner_path)
        self.component_metadata["label_refinement"] = self.spec.label_refinement.name
        return result.data

    def _apply_knowledge_and_predicates(self, data: TrainingData) -> TrainingData:
        base_context = {
            "experiment": self.spec.name,
            "task": self.spec.task.name,
            "dataset": self.values["dataset"],
            "research_unit": self.values["research_unit"],
        }
        knowledge = self.knowledge_pipeline.build(data, base_context)
        context = {**base_context, "knowledge": knowledge}
        result = self.predicate_pipeline.apply(data, context)
        self.component_metadata["knowledge"] = self.knowledge_pipeline.names
        self.component_metadata["predicates"] = self.predicate_pipeline.names
        if knowledge:
            self.component_metadata["knowledge_artifacts"] = sorted(knowledge)
        return result

    def train(self) -> None:
        """回放归档模型，或通过所选 Tuner 训练已注册的目标模型。"""
        if self.mode == "archive_replay":
            assert self.dataset is not None
            source = self.dataset.archive_dir / self.model_adapter.artifact_filename
            if not source.is_file():
                raise FileNotFoundError(f"Archived model not found: {source}")
            target = self.output_dir / "models" / self.model_adapter.artifact_filename
            shutil.copy2(source, target)
            self.output_files.append(target)
            self.component_metadata["model_artifact"] = "replayed_from_archive"
            self._log.info("Model replayed from archive: %s", source)
            return

        if self.mode == "train_from_archive_features":
            assert self.dataset is not None
            training = self._training_from_frame(self.dataset.train_split, self.dataset.feature_columns)
            self._evaluation_data = self._training_from_frame(
                self.dataset.test_split, self.dataset.feature_columns
            )
            self.component_metadata["training_source"] = "archive:Xy_rf_train.csv"
            self.component_metadata["evaluation_source"] = "archive:Xy_rf_test.csv"
            self.component_metadata["holdout_execution"] = "precomputed_in_archive"
            if self.spec.label_refinement is not None:
                self.component_metadata["label_refinement_execution"] = "precomputed_in_archive"
            self._log.info(
                "Training data: %d rows, class distribution: pos=%d neg=%d",
                len(training.labels),
                int((training.labels == 1).sum()),
                int((training.labels == 0).sum()),
            )
        else:
            frame = self._raw_training_frame
            unit_columns = self._research_unit_columns
            raw_feature_columns = [
                column
                for column in frame.columns
                if column not in set(unit_columns) | {"label", "sample_weight"}
            ]
            training = TrainingData(
                frame[raw_feature_columns].reset_index(drop=True),
                frame["label"].reset_index(drop=True),
                frame["sample_weight"].reset_index(drop=True),
                metadata={"units": frame[unit_columns].reset_index(drop=True)},
            )
            # 先做外层 holdout 划分，再仅对训练折拟合预处理，避免测试集泄漏
            holdout = create_holdout(
                self.spec.holdout.name, self.spec.holdout.params, self.spec.seed
            )
            split: SplitData = holdout.split(training, self.spec.holdout.params, self.spec.seed)
            train_split = split.train
            test_split = split.test

            self.preprocessor.fit(train_split.features, unit_columns=unit_columns)
            train_features = self.preprocessor.transform(train_split.features)
            test_features = self.preprocessor.transform(test_split.features)

            training = TrainingData(
                train_features.reset_index(drop=True),
                train_split.labels.reset_index(drop=True),
                train_split.sample_weight.reset_index(drop=True),
                metadata={"units": train_split.metadata["units"].reset_index(drop=True)},
            )
            self._evaluation_data = TrainingData(
                test_features.reset_index(drop=True),
                test_split.labels.reset_index(drop=True),
                test_split.sample_weight.reset_index(drop=True),
                metadata={"units": test_split.metadata["units"].reset_index(drop=True)},
            )
            training = self._apply_label_refinement(training)
            self.component_metadata["holdout"] = self.spec.holdout.name

        training = self._apply_knowledge_and_predicates(training)
        if training.constraints and not getattr(self.model_adapter, "supports_constraints", False):
            raise ValueError(
                f"Model '{self.model_adapter.name}' does not support predicate constraints: "
                f"{sorted(training.constraints)}"
            )
        tuner = TUNER_REGISTRY.create(self.spec.tuner.name)
        self._log.info(
            "Training model '%s' with tuner '%s' (primary_metric=%s)",
            self.spec.model.name, self.spec.tuner.name, self.spec.primary_metric,
        )
        t_start = time.monotonic()
        self.model = tuner.fit(
            self.model_adapter,
            training,
            self.spec.model.params,
            self.spec.tuner.params,
            {
                "name": self.spec.cross_validation.name,
                "params": self.spec.cross_validation.params,
            },
            self.spec.primary_metric,
            self.spec.seed,
        )
        elapsed = time.monotonic() - t_start
        self._log.info("Training completed in %.1fs", elapsed)
        self.component_metadata["model"] = self.spec.model.name
        self.component_metadata["tuner"] = self.spec.tuner.name
        if getattr(tuner, "uses_cross_validation", False):
            self.component_metadata["cross_validation"] = self.spec.cross_validation.name

        model_path = self.output_dir / "models" / self.model_adapter.artifact_filename
        with model_path.open("wb") as handle:
            pickle.dump(self.model, handle)
        self.output_files.append(model_path)

    def evaluate(self) -> None:
        """持久化已配置的指标，或记录归档回放元数据。"""
        if self.mode == "archive_replay":
            assert self.dataset is not None
            self._log.info("Recording archive replay metadata (metrics not recomputed)")
            reference_path = self.dataset.archive_dir / "model_comparison" / "model_comparison_results.csv"
            archived_reference = None
            if reference_path.is_file():
                reference = pd.read_csv(reference_path)
                candidate_names = list(
                    getattr(
                        self.model_adapter,
                        "archive_metric_aliases",
                        (self.spec.model.name, self.spec.model.name.upper()),
                    )
                )
                row = reference.loc[reference["Model"].isin(candidate_names)] if "Model" in reference.columns else pd.DataFrame()
                if not row.empty:
                    archived_reference = {
                        key: float(value) if isinstance(value, (float, np.floating)) else value
                        for key, value in row.iloc[0].to_dict().items()
                    }
            self.metrics = {
                "mode": "archive_replay",
                "recomputed": False,
                "reason": "Archived model and predictions are retained byte-for-byte; metrics are not recomputed during exact replay.",
                "train_rows": int(len(self.dataset.train_split)),
                "test_rows": int(len(self.dataset.test_split)),
                "train_label_distribution": self.dataset.summary()["xy_train_labels"],
                "archived_reference_metrics": archived_reference,
                "archived_reference_metrics_source": str(reference_path) if archived_reference is not None else None,
            }
        else:
            if self.model is None:
                raise RuntimeError("Model must be trained before evaluation")
            evaluation = self._evaluation_data
            self._log.info("Evaluating model on %d test samples", len(evaluation.labels))
            self.metrics = evaluate_classifier(
                self.model,
                evaluation.features,
                evaluation.labels,
                evaluation.sample_weight,
                self.spec.metrics,
            )
            self.metrics["mode"] = self.mode
            self.metrics["primary_metric"] = self.spec.primary_metric
            primary_val = self.metrics.get(self.spec.primary_metric)
            self._log.info("Evaluation complete: %s=%.4f", self.spec.primary_metric, primary_val)

        path = self.output_dir / "metrics.json"
        write_json(path, self.metrics)
        self.output_files.append(path)

    def predict(self) -> None:
        """回放归档预测，或委托所选 Task 完成评分与导出。"""
        prediction_dir = self.output_dir / "predictions"
        if self.mode == "archive_replay":
            assert self.dataset is not None
            self._log.info("Replaying archived predictions")
            for filename, source in self.task.archived_prediction_artifacts(
                self.dataset.archive_dir
            ).items():
                if not source.is_file():
                    raise FileNotFoundError(f"Archived prediction artifact not found: {source}")
                target = prediction_dir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                self.output_files.append(target)
            return

        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        if self.mode == "train_from_archive_features":
            assert self.dataset is not None
            target_features = self.dataset.target_features
            target_coords = self.dataset.target_coords
            target_mask = self.dataset.target_mask
        else:
            target_features = self.preprocessor.transform(self._raw_target_frame)
            target_coords = self._raw_target_coords
            target_mask = self._raw_target_mask

        self._log.info("Generating predictions for %d target points", len(target_features))
        predictions = self.task.predict(self.model, target_features, target_coords)
        self.output_files.extend(
            self.task.export_predictions(predictions, target_mask, prediction_dir)
        )
        self._log.info("Predictions exported to %s", prediction_dir)

    def export(self) -> dict[str, Any]:
        """所有实验阶段完成后写入可复现性清单。"""
        if self.dataset is not None:
            feature_schema = self.dataset.feature_columns
        elif self.preprocessor is not None:
            feature_schema = self.preprocessor.feature_columns
        else:
            feature_schema = []

        manifest = {
            "timestamp": utc_timestamp(),
            "experiment": self.spec.name,
            "variant": self.values["experiment"].get("variant", "baseline"),
            "config_source": str(self.config.source_path),
            "config": self.values,
            "seed": self.spec.seed,
            "task": self.spec.task.name,
            "execution_mode": self.mode,
            "components": {
                "feature_operators": [spec.name for spec in self.spec.feature_operators],
                "model": self.spec.model.name,
                "label_refinement": self.spec.label_refinement.name if self.spec.label_refinement else None,
                "knowledge": [spec.name for spec in self.spec.knowledge],
                "predicates": [spec.name for spec in self.spec.predicates],
                "holdout": self.spec.holdout.name,
                "cross_validation": self.spec.cross_validation.name,
                "metrics": list(self.spec.metrics),
                "primary_metric": self.spec.primary_metric,
                "tuner": self.spec.tuner.name,
            },
            "component_metadata": self.component_metadata,
            "input_paths": self._input_paths(),
            "feature_schema": feature_schema,
            "feature_count": len(feature_schema),
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "output_files": [
                str(path.relative_to(self.output_dir))
                for path in self.output_files
                if path.exists()
            ],
        }
        path = self.output_dir / "manifest.json"
        manifest["output_files"].append("manifest.json")
        write_json(path, manifest)
        return manifest

    def _input_paths(self) -> dict[str, Any]:
        """解析实际输入文件，包括栅格目录内容和 Shapefile 附属文件。"""

        def describe_file(path: Path) -> dict[str, Any]:
            item = {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            if path.suffix.lower() == ".shp" and path.exists():
                sidecars = []
                for candidate in sorted(path.parent.glob(path.stem + ".*")):
                    if candidate.is_file():
                        sidecars.append(
                            {
                                "path": str(candidate),
                                "size_bytes": candidate.stat().st_size,
                                "sha256": sha256_file(candidate),
                            }
                        )
                item["sidecars"] = sidecars
            return item

        if self.dataset is not None:
            archive_inputs = dict(self.dataset.source_files)
            if self.mode == "archive_replay":
                archive_inputs[self.model_adapter.artifact_filename] = (
                    self.dataset.archive_dir / self.model_adapter.artifact_filename
                )
                archive_inputs.update(
                    self.task.archived_prediction_artifacts(self.dataset.archive_dir)
                )
                reference = self.dataset.archive_dir / "model_comparison" / "model_comparison_results.csv"
                if reference.is_file():
                    archive_inputs["model_comparison_results.csv"] = reference
            return {
                "config_source": describe_file(self.config.source_path),
                "archive_dir": str(self.dataset.archive_dir),
                "archive_artifacts": {
                    name: describe_file(path) for name, path in sorted(archive_inputs.items())
                },
            }

        raster_directory_keys = {"magnetic", "gravity", "radiometric", "remote_sensing"}

        def describe(value: Any, key: str | None = None) -> Any:
            if isinstance(value, str):
                path = Path(value)
                if path.is_dir():
                    item: dict[str, Any] = {
                        "path": str(path),
                        "exists": True,
                        "kind": "directory",
                    }
                    if key in raster_directory_keys:
                        raster_files = [
                            candidate
                            for candidate in sorted(path.rglob("*"))
                            if candidate.is_file() and candidate.suffix.lower() in {".tif", ".tiff"}
                        ]
                        item["resolved_rasters"] = [describe_file(candidate) for candidate in raster_files]
                    return item
                return describe_file(path)
            if isinstance(value, list):
                return [describe(item, key) for item in value]
            if isinstance(value, dict):
                return {child_key: describe(item, child_key) for child_key, item in value.items()}
            return value

        return {
            "config_source": describe_file(self.config.source_path),
            **{key: describe(value, key) for key, value in self.values["dataset"].items()},
        }

    def run(self) -> dict[str, Any]:
        self._log.info("=" * 60)
        self._log.info("Experiment '%s' starting (mode=%s)", self.spec.name, self.mode)
        self._log.info("=" * 60)

        phases = [
            ("prepare_data", self.prepare_data),
            ("build_features", self.build_features),
            ("prepare_dataset", self.prepare_dataset),
            ("train", self.train),
            ("evaluate", self.evaluate),
            ("predict", self.predict),
        ]
        for name, method in phases:
            t0 = time.monotonic()
            method()
            self._log.info("Phase '%s' completed in %.1fs", name, time.monotonic() - t0)

        manifest = self.export()
        self._log.info("Experiment '%s' complete. Output: %s", self.spec.name, self.output_dir)
        return manifest
