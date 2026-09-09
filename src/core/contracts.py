"""Task、Predicate、Splitter、Tuner 与 Model 之间共享的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd


class OptionalDependencyError(RuntimeError):
    """所选组件依赖尚未安装的可选依赖时抛出的异常。"""


def validate_training_data(data: "TrainingData") -> None:
    """主训练/调参前的类别与数值契约检查（P1-05）。

    单类训练集、非有限特征、非法权重都会让后续 ``predict_proba[:, 1]`` 或
    指标计算崩溃/产生无意义结果；在进入训练前显式拒绝，而不是在评估阶段抛
    ``IndexError``。空间留区或稀少矿点留下单类训练集时应报告不可评价。
    """
    labels = data.labels.to_numpy()
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError(
            f"training set is single-class (classes={classes.tolist()}); cannot fit a "
            "binary classifier or produce a two-column predict_proba. Check the "
            "holdout/split configuration or provide more labeled positives."
        )
    numeric = data.features.select_dtypes(include=[np.number])
    if numeric.shape[1] and not np.all(np.isfinite(numeric.to_numpy())):
        raise ValueError("training features contain non-finite values (NaN/inf).")
    weights = data.sample_weight.to_numpy(dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample weights must be finite and non-negative.")
    if weights.sum() <= 0:
        raise ValueError("sample weights sum to zero; no effective training signal.")


@dataclass(frozen=True)
class TrainingData:
    """在实验组件之间传递的特征、标签与权重数据包。"""

    features: pd.DataFrame
    labels: pd.Series
    sample_weight: pd.Series
    constraints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        row_count = len(self.features)
        if len(self.labels) != row_count or len(self.sample_weight) != row_count:
            raise ValueError("TrainingData features, labels, and sample_weight must have equal row counts")
        if not self.features.index.equals(self.labels.index) or not self.features.index.equals(
            self.sample_weight.index
        ):
            raise ValueError("TrainingData features, labels, and sample_weight indexes must align")

    def with_labels(self, labels: pd.Series, **metadata: Any) -> "TrainingData":
        merged = dict(self.metadata)
        merged.update(metadata)
        aligned_labels = labels.copy()
        aligned_labels.index = self.features.index
        return replace(self, labels=aligned_labels, metadata=merged)

    def with_constraints(self, constraints: Mapping[str, Any], **metadata: Any) -> "TrainingData":
        """合并约束与元数据，返回新的 ``TrainingData``。

        约束按原键合并：单个约束谓词写入 ``phi_vector``，多个谓词由
        ``PredicatePipeline`` 归一化为 ``phi_vectors``；本方法不做载荷改写。
        """
        merged_constraints = dict(self.constraints)
        merged_constraints.update(constraints)
        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata)
        return replace(self, constraints=merged_constraints, metadata=merged_metadata)

    def take(self, indices: list[int]) -> "TrainingData":
        """选择指定行，并同步保持逐行元数据对齐。"""
        selected_metadata: dict[str, Any] = {}
        for key, value in self.metadata.items():
            if isinstance(value, (pd.DataFrame, pd.Series)) and len(value) == len(self.features):
                selected_metadata[key] = value.iloc[indices].reset_index(drop=True)
            else:
                selected_metadata[key] = value
        return TrainingData(
            self.features.iloc[indices].reset_index(drop=True),
            self.labels.iloc[indices].reset_index(drop=True),
            self.sample_weight.iloc[indices].reset_index(drop=True),
            constraints=dict(self.constraints),
            metadata=selected_metadata,
        )


@dataclass(frozen=True)
class SplitData:
    """由留出法 Splitter 生成的训练集与评估集分区。"""

    train: TrainingData
    test: TrainingData


class TaskProtocol(Protocol):
    def build_positive_units(self, dataset_config, research_unit_config, label_config): ...
    def build_unlabeled_units(self, dataset_config, research_unit_config, label_config, count: int, seed: int | None = None): ...
    def build_prediction_units(self, dataset_config, research_unit_config): ...
    def prepare_prediction_data(self, target_data: pd.DataFrame, target_mask: pd.DataFrame): ...
    def predict(self, model: Any, target_features: pd.DataFrame, target_coords: pd.DataFrame) -> pd.DataFrame: ...
    def export_predictions(self, predictions: pd.DataFrame, target_mask: pd.DataFrame, output_dir): ...
    def archived_prediction_artifacts(self, archive_dir): ...


class FeatureOperatorProtocol(Protocol):
    name: str
    def extract(self, units: pd.DataFrame) -> pd.DataFrame: ...


class ModelAdapterProtocol(Protocol):
    name: str
    artifact_filename: str
    supports_constraints: bool
    def build(self, params: Mapping[str, Any], seed: int) -> Any: ...
    def fit_params(self, data: TrainingData) -> Mapping[str, Any]: ...


class PredicateProtocol(Protocol):
    name: str
    kind: str  # "data_transform" | "constraint"
    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData: ...


class KnowledgeProviderProtocol(Protocol):
    name: str
    def build(self, data: TrainingData, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ResearchUnitProtocol(Protocol):
    """研究单元构建器：将原始 GIS 数据转换为带 X/Y 坐标的点单元。"""
    name: str
    def build_positive_units(self, dataset_config, research_unit_config, label_config) -> pd.DataFrame: ...
    def build_prediction_units(self, dataset_config, research_unit_config) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class BackgroundSamplerProtocol(Protocol):
    """未标注/背景采样器：在边界内生成未标注点。"""
    name: str
    def sample(self, boundary_path: str, count: int, seed: int | None = None) -> pd.DataFrame: ...


class LabelStrategyProtocol(Protocol):
    """标签策略：将原始观测编码为模型训练目标与样本权重。"""
    name: str
    def apply_positive(self, occurrences: pd.DataFrame, label_config: Mapping[str, Any]) -> pd.DataFrame: ...
    def apply_unlabeled(self, points: pd.DataFrame, label_config: Mapping[str, Any]) -> pd.DataFrame: ...


class WeightStrategyProtocol(Protocol):
    """样本权重策略：将属性（如 SIZE_CODE）映射为样本权重。"""
    name: str
    def compute(self, attributes: pd.DataFrame, weight_config: Mapping[str, Any]) -> pd.Series: ...
