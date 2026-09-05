"""由已注册组件组装而成的二阶段实验编排器。"""

from __future__ import annotations

import pickle
import shutil
import time
from dataclasses import replace
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
from ..utils.files import (
    environment_summary,
    git_commit,
    git_dirty,
    sha256_file,
    utc_timestamp,
    write_json,
    write_yaml,
)
from ..utils.seeds import derive_seeds
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
        # 追加微秒级时间戳后缀，防止重复运行同一配置时覆盖前次结果；
        # 目录原子创建见 _prepare_output_layout。
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_dir = self.output_dir.parent / f"{self.output_dir.name}_{self.run_id}"
        self.values["experiment"]["output_dir"] = str(self.output_dir)
        self.status = "created"
        self._failure: dict[str, Any] | None = None
        self._tuning_summary: dict[str, Any] | None = None
        self._target_scaling: tuple[float, float] | None = None
        self._training_data: TrainingData | None = None
        self._independent_cv: dict[str, Any] | None = None
        self._injection_audit: dict[str, Any] | None = None
        # 从实验基础种子派生各用途种子，保证采样/划分/调参互不干扰且可复现
        self.seeds = derive_seeds(self.spec.seed)
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
        # 顶层目录原子创建：已存在则失败，防止覆盖前次运行
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for directory in (
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
            seed=self.seeds["sampling_seed"],
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
        # 记录研究单元 CRS，供 manifest 溯源与导出一致性校验。
        if self.task._research_crs_id is not None:
            self.component_metadata["research_crs"] = self.task._research_crs_id

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

    def _cross_validation_needs_coordinates(self) -> bool:
        """空间 CV 划分需要逐行坐标；否则 ``build_cv`` 会因缺 metadata 报错。"""
        return self.spec.cross_validation.name.startswith("spatial_")

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
        # P1-04 注入可归因审计：记录知识/谓词启用、约束产出与消费关系，
        # 使"约束有无"的差异可归因（消融实验只切换目标机制）。
        unconsumed_knowledge = bool(self.knowledge_pipeline.names) and not bool(
            self.predicate_pipeline.names
        )
        self._injection_audit = {
            "knowledge_enabled": bool(self.knowledge_pipeline.names),
            "knowledge_providers": list(self.knowledge_pipeline.names),
            "knowledge_artifacts": sorted(knowledge) if knowledge else [],
            "predicates_enabled": bool(self.predicate_pipeline.names),
            "predicates": [
                {"name": item.name, "kind": getattr(item, "kind", "data_transform")}
                for item in self.predicate_pipeline.items
            ],
            "constraints_produced": bool(result.constraints),
            "unconsumed_knowledge": unconsumed_knowledge,
        }
        if unconsumed_knowledge:
            self._log.warning(
                "Knowledge providers %s are enabled but no predicate consumes them; "
                "their artifacts will not affect training.",
                self.knowledge_pipeline.names,
            )
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

        # 折内隔离所需的未变换外层训练折特征；仅 raw_gis + 内层 CV 时填充。
        raw_training: TrainingData | None = None

        if self.mode == "train_from_archive_features":
            assert self.dataset is not None
            grid_model = bool(getattr(self.model_adapter, "grid_model", False))
            # 空间谓词需要坐标；grid 模型需要坐标 + 网格；空间 CV 需要坐标。
            # 仅当需要时重建，避免额外开销。
            units = self.dataset.point_units() if (
                self.spec.predicates or grid_model or self._cross_validation_needs_coordinates()
            ) else (None, None)
            training = self._training_from_frame(
                self.dataset.train_split, self.dataset.feature_columns, units=units[0]
            )
            self._evaluation_data = self._training_from_frame(
                self.dataset.test_split, self.dataset.feature_columns, units=units[1]
            )
            if grid_model:
                from ..data.archive_coords import build_grid, snap_cells

                grid, inside, x_axis, y_axis = build_grid(self.values["dataset"]["archive_dir"])
                train_cells = snap_cells(units[0], x_axis, y_axis)
                self._grid_test_cells = snap_cells(units[1], x_axis, y_axis)
                training = replace(
                    training,
                    metadata={
                        **training.metadata,
                        "grid": grid,
                        "inside": inside,
                        "train_cells": train_cells,
                    },
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
                self.spec.holdout.name, self.spec.holdout.params, self.seeds["split_seed"]
            )
            split: SplitData = holdout.split(training, self.spec.holdout.params, self.seeds["split_seed"])
            train_split = split.train
            test_split = split.test

            self.preprocessor.fit(train_split.features, unit_columns=unit_columns)
            # 持久化已拟合的预处理器（相关性筛选 + OHE + scaler），使成功运行
            # 可仅凭输出目录重建预处理语义（P1-07 工件 provenance）。
            preprocessor_path = self.output_dir / "models" / "preprocessor.pkl"
            with preprocessor_path.open("wb") as handle:
                pickle.dump(self.preprocessor, handle)
            self.output_files.append(preprocessor_path)
            # 保留未变换的外层训练折特征（仅预处理器实际使用的列），供带内层
            # CV 的 tuner（如 bayes）逐 fold 重拟合 scaler，避免验证折参与缩放
            # 统计（P0-03 折内隔离）。
            raw_columns = list(self.preprocessor.numerical_columns) + list(
                self.preprocessor.categorical_columns
            )
            raw_training = TrainingData(
                train_split.features[raw_columns].reset_index(drop=True),
                train_split.labels.reset_index(drop=True),
                train_split.sample_weight.reset_index(drop=True),
                metadata={"units": train_split.metadata["units"].reset_index(drop=True)},
            )
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
        self._training_data = training
        self._split_summary = {
            "train_rows": int(len(training.labels)),
            "test_rows": int(len(self._evaluation_data.labels)),
            "train_positives": int((training.labels == 1).sum()),
            "test_positives": int((self._evaluation_data.labels == 1).sum()),
        }
        if training.constraints and not getattr(self.model_adapter, "supports_constraints", False):
            raise ValueError(
                f"Model '{self.model_adapter.name}' does not support predicate constraints: "
                f"{sorted(training.constraints)}"
            )
        if self._injection_audit is not None:
            self._injection_audit["constraints_consumed"] = bool(training.constraints)
        tuner = TUNER_REGISTRY.create(self.spec.tuner.name)
        # 折内隔离仅在「raw_gis + 内层 CV」时启用；PUB 标签细化与内层 CV 的组合
        # 已在配置层拒绝，故这里 raw_training 与 label_refinement 不会同时出现。
        fold_safe = bool(getattr(tuner, "uses_cross_validation", False)) and self.mode == "raw_gis"
        self._log.info(
            "Training model '%s' with tuner '%s' (primary_metric=%s, fold_safe=%s)",
            self.spec.model.name, self.spec.tuner.name, self.spec.primary_metric, fold_safe,
        )
        t_start = time.monotonic()
        # 折内隔离参数仅传递给支持它们的 CV tuner；``none`` 等无内层 CV 的
        # tuner 直接拟合已变换特征，无需这些参数。
        fold_safe_kwargs = (
            {"preprocessor": self.preprocessor, "raw_data": raw_training}
            if fold_safe
            else {}
        )
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
            self.seeds["tuning_seed"],
            model_seed=self.seeds["model_seed"],
            dataloader_seed=self.seeds["dataloader_seed"],
            **fold_safe_kwargs,
        )
        self._tuning_summary = getattr(self.model, "_tuning_summary", None)
        elapsed = time.monotonic() - t_start
        self._log.info("Training completed in %.1fs", elapsed)
        self.component_metadata["model"] = self.spec.model.name
        self.component_metadata["tuner"] = self.spec.tuner.name
        if getattr(tuner, "uses_cross_validation", False):
            self.component_metadata["cross_validation"] = self.spec.cross_validation.name

        self._best_params = (
            dict(self.model.get_params()) if hasattr(self.model, "get_params") else None
        )
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
            if getattr(self.model_adapter, "grid_model", False):
                self.model.set_predict_cells(self._grid_test_cells)
            self.metrics = evaluate_classifier(
                self.model,
                evaluation.features,
                evaluation.labels,
                evaluation.sample_weight,
                self.spec.metrics,
                unit_area=evaluation.metadata.get("unit_area"),
            )
            self.metrics["mode"] = self.mode
            self.metrics["primary_metric"] = self.spec.primary_metric
            self.metrics["independent_cv"] = self._run_independent_cv()
            self._independent_cv = self.metrics["independent_cv"]
            self._write_test_predictions(evaluation)
            primary_val = self.metrics.get(self.spec.primary_metric)
            self._log.info("Evaluation complete: %s=%.4f", self.spec.primary_metric, primary_val)

        path = self.output_dir / "metrics.json"
        write_json(path, self.metrics)
        self.output_files.append(path)

    def _write_test_predictions(self, evaluation: TrainingData) -> None:
        """保存测试集逐样本预测表（y_true/y_pred/score/权重，含逐行坐标）。

        与 aggregate 指标互补：跨运行汇总、逐样本错误分析与逐空间点关联需要
        该表。归档重放模式不调用（预测直接复制归档，不重新计算）。逐样本表为
        尽力而为——任何模型评分异常只告警，不阻断评价或运行。
        """
        try:
            y_pred = np.asarray(self.model.predict(evaluation.features)).ravel()
            score = np.asarray(
                self.model.predict_proba(evaluation.features)[:, 1], dtype=float
            ).ravel()
        except Exception as error:  # noqa: BLE001 - 逐样本表是尽力而为
            self._log.warning("Skipping per-sample test table: %s", error)
            return
        table = pd.DataFrame(
            {
                "row_index": np.arange(len(evaluation.labels)),
                "y_true": evaluation.labels.to_numpy(),
                "y_pred": y_pred,
                "score": score,
                "sample_weight": evaluation.sample_weight.to_numpy(),
            }
        )
        units = evaluation.metadata.get("units")
        if isinstance(units, pd.DataFrame):
            for col in ("X", "Y"):
                if col in units.columns:
                    table[col] = units[col].to_numpy()
        path = self.output_dir / "intermediate" / "test_predictions.csv"
        table.to_csv(path, index=False)
        self.output_files.append(path)
        self._log.info("Per-sample test table written to %s", path)

    def _run_independent_cv(self) -> dict[str, Any]:
        """在训练集上运行独立评价 CV，产出 OOF 预测与逐折指标。

        与调参 CV 分离：即使 ``tuning=none``，也按
        ``validation.cross_validation`` 配置运行一次评价，避免该配置被静默
        忽略（P1-01）。每折构建全新模型，评估最终模型配置在未见过折上的
        泛化，并保存每折 train/val 规模与类别计数、逐折指标与 OOF 表。

        无法安全切片时（grid 模型、约束谓词、外层全量拟合的 PUB 标签细化）
        返回带 ``reason`` 的跳过说明，而非产生有偏评价。
        """
        from ..models.registry import fit_params_for
        from ..validation.splitters import build_cv

        if self.mode == "archive_replay":
            return {"run": False, "reason": "archive_replay replays archived model; no refit."}
        if self.spec.cross_validation.name == "none":
            return {"run": False, "reason": "validation.cross_validation.name is 'none'."}
        training = self._training_data
        if training is None:
            return {"run": False, "reason": "no training data recorded for independent CV."}
        if getattr(self.model_adapter, "grid_model", False):
            return {"run": False, "reason": "grid models require fold-specific cells; not supported."}
        if training.constraints:
            return {"run": False, "reason": "predicate constraints cannot be sliced per fold."}
        if self.spec.label_refinement is not None:
            return {
                "run": False,
                "reason": "label refinement (PUB) labels are fit on the full outer train set; "
                "per-fold refit would leak validation labels.",
            }

        cv = build_cv(
            self.spec.cross_validation.name,
            self.spec.cross_validation.params,
            self.seeds["split_seed"],
            training,
        )
        model_params = (
            dict(self._best_params) if self._best_params is not None else dict(self.spec.model.params)
        )
        try:
            fold_splits = list(cv.split(training.features, training.labels))
        except ValueError as error:
            # 训练样本过少（如 StratifiedKFold 少数类样本数 < n_splits）时，
            # 独立 CV 无法切片，记录原因并跳过，而非让整次运行失败。
            if "n_splits" in str(error) or "populated" in str(error):
                return {
                    "run": False,
                    "reason": f"cross_validation could not split the training data: {error}",
                }
            raise

        folds: list[dict[str, Any]] = []
        oof_rows: list[dict[str, Any]] = []
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            fold_train = training.take([int(i) for i in train_idx])
            fold_val = training.take([int(i) for i in val_idx])
            model = self.model_adapter.build(model_params, self.seeds["model_seed"])
            if self.seeds.get("dataloader_seed") is not None and hasattr(model, "dataloader_seed"):
                model.dataloader_seed = self.seeds["dataloader_seed"]
            model.fit(
                fold_train.features,
                fold_train.labels,
                **fit_params_for(self.model_adapter, fold_train),
            )
            fold_metrics = evaluate_classifier(
                model,
                fold_val.features,
                fold_val.labels,
                fold_val.sample_weight,
                self.spec.metrics,
                unit_area=fold_val.metadata.get("unit_area"),
            )
            proba = model.predict_proba(fold_val.features)[:, 1]
            preds = model.predict(fold_val.features)
            for row_idx, (y_true, y_pred, prob, wgt) in enumerate(
                zip(
                    fold_val.labels.to_numpy(),
                    preds,
                    proba,
                    fold_val.sample_weight.to_numpy(),
                )
            ):
                oof_rows.append(
                    {
                        "fold": fold,
                        "row_index": int(val_idx[row_idx]),
                        "y_true": int(y_true),
                        "y_pred": int(y_pred),
                        "score": float(prob),
                        "sample_weight": float(wgt),
                    }
                )
            folds.append(
                {
                    "fold": fold,
                    "train_rows": int(len(fold_train.labels)),
                    "val_rows": int(len(fold_val.labels)),
                    "train_positives": int((fold_train.labels == 1).sum()),
                    "val_positives": int((fold_val.labels == 1).sum()),
                    "metrics": fold_metrics,
                }
            )

        aggregate: dict[str, Any] = {}
        for name in self.spec.metrics:
            values = [f["metrics"].get(name) for f in folds]
            scalars = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if scalars:
                aggregate[name] = {
                    "mean": float(np.mean(scalars)),
                    "std": float(np.std(scalars)),
                }
            else:
                aggregate[name] = None

        oof_frame = pd.DataFrame(oof_rows)
        oof_path = self.output_dir / "intermediate" / "oof_predictions.csv"
        oof_frame.to_csv(oof_path, index=False)
        self.output_files.append(oof_path)

        self._independent_cv = {
            "run": True,
            "splitter": self.spec.cross_validation.name,
            "n_splits": len(folds),
            "model_params_source": (
                "fitted_model.get_params()" if self._best_params is not None else "model.params"
            ),
            "folds": folds,
            "aggregate_metrics": aggregate,
            "oof_predictions": str(oof_path.relative_to(self.output_dir)),
        }
        return dict(self._independent_cv)

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
        if getattr(self.model_adapter, "grid_model", False):
            self.model.set_predict_cells(None)  # 预测阶段：在整幅有效单元上滑动
        predictions = self.task.predict(self.model, target_features, target_coords)
        self._target_scaling = self.task.normalization_range
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

        git_root = Path(__file__).resolve().parents[2]

        def describe_output(path: Path) -> dict[str, Any]:
            return {
                "path": str(path.relative_to(self.output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "timestamp": utc_timestamp(),
            "status": self.status,
            "experiment": self.spec.name,
            "variant": self.values["experiment"].get("variant", "baseline"),
            "config_source": str(self.config.source_path),
            "config": self.values,
            "seed": self.spec.seed,
            "derived_seeds": self.seeds,
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
            "split_summary": getattr(self, "_split_summary", None),
            "best_params": getattr(self, "_best_params", None),
            "tuning_summary": self._tuning_summary,
            "independent_cv": self._independent_cv,
            "injection_audit": self._injection_audit,
            "target_scaling": self._target_scaling,
            "failure": self._failure,
            "git_commit": git_commit(git_root),
            "git_dirty": git_dirty(git_root),
            "environment": environment_summary(),
            "output_files": [
                describe_output(path)
                for path in self.output_files
                if path.exists()
            ],
        }
        path = self.output_dir / "manifest.json"
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
        self.status = "running"
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
        t_start = time.monotonic()
        current_phase: str | None = None
        try:
            for name, method in phases:
                current_phase = name
                t0 = time.monotonic()
                method()
                self._log.info("Phase '%s' completed in %.1fs", name, time.monotonic() - t0)
            self.status = "completed"
        except Exception as error:
            self.status = "failed"
            self._failure = {
                "exception_type": type(error).__name__,
                "message": str(error),
                "phase": current_phase,
                "elapsed_seconds": round(time.monotonic() - t_start, 3),
            }
            # 失败也原子写出 status=failed 的 manifest，保留异常摘要、耗时与已消费
            # seed 账目，且不覆盖已有输出（manifest.json 在成功路径才写入）。
            try:
                self.export()
            except Exception:
                self._log.exception("Failed to persist failure manifest")
            raise

        manifest = self.export()
        self._log.info("Experiment '%s' complete. Output: %s", self.spec.name, self.output_dir)
        return manifest
