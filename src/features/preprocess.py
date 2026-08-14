"""与基线兼容的特征预处理。

本实现有意保留当前实验行为；警告会明确指出科学方法或验证方面的
技术债务，而不会在框架迁移过程中悄然改变这些行为。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_PREFIXES = ("Intrusions_", "MetamorphicFacies", "RockUnits")


def split_feature_columns(
    frame: pd.DataFrame,
    unit_columns: tuple[str, ...] | list[str] = ("X", "Y"),
) -> tuple[list[str], list[str]]:
    """拆分特征列，同时排除由 Task 管理的研究单元元数据。"""
    excluded = {*unit_columns, "label", "sample_weight"}
    categorical = [
        column
        for column in frame.columns
        if column not in excluded
        and (
            column.startswith(CATEGORICAL_PREFIXES)
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
        )
    ]
    numerical = [
        column for column in frame.columns if column not in categorical and column not in excluded
    ]
    return numerical, categorical


@dataclass
class PreparedTrainingData:
    """保存与基线兼容的训练集、正例留出集和特征 schema。"""

    xy_train: pd.DataFrame
    xy_pos_test: pd.DataFrame
    xy_train_original: pd.DataFrame
    feature_columns: list[str]
    xy_train_units: pd.DataFrame
    xy_pos_test_units: pd.DataFrame


class BaselinePreprocessor:
    """与 Notebook 兼容的相关性筛选、OHE 和标准化。"""

    def __init__(self, config: Mapping[str, Any], seed: int):
        """初始化配置的转换器和已拟合 schema 状态。"""
        self.config = config
        self.seed = seed
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.numerical_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.correlation: pd.DataFrame | None = None
        self.encoder_fitted = False
        self.scaler_fitted = False

    def prepare_training(
        self,
        deposits: pd.DataFrame,
        unlabeled: pd.DataFrame,
        unit_columns: tuple[str, ...] | list[str] = ("X", "Y"),
    ) -> PreparedTrainingData:
        """按照 Notebook 语义筛选、编码、缩放并切分原始单元。"""
        warnings.warn(
            "Baseline compatibility: correlation filtering is performed before the split.",
            UserWarning,
            stacklevel=2,
        )
        training = pd.concat([deposits, unlabeled], ignore_index=True)
        numerical, categorical = split_feature_columns(training, unit_columns)
        correlation = training[numerical].corr(method="spearman").abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        dropped = [column for column in upper.columns if any(upper[column] > self.config["correlation_threshold"])]
        self.numerical_columns = [column for column in numerical if column not in dropped]
        self.categorical_columns = categorical
        self.correlation = correlation

        if categorical:
            encoded = self.encoder.fit_transform(training[categorical]).toarray()
            self.encoder_fitted = True
            try:
                encoded_columns = self.encoder.get_feature_names(categorical).tolist()
            except AttributeError:
                encoded_columns = self.encoder.get_feature_names_out(categorical).tolist()
            encoded_frame = pd.DataFrame(encoded, columns=encoded_columns)
        else:
            encoded_columns = []
            encoded_frame = pd.DataFrame(index=training.index)
        combined = pd.concat(
            [training[self.numerical_columns].reset_index(drop=True), encoded_frame,
             training[["sample_weight", "label"]].reset_index(drop=True)],
            axis=1,
        )
        positives = combined.loc[combined["label"] == 1].reset_index(drop=True)
        unlabelled = combined.loc[combined["label"] == 0].reset_index(drop=True)
        available_unit_columns = [column for column in unit_columns if column in training.columns]
        units = training[available_unit_columns].reset_index(drop=True)
        positive_units = units.loc[combined["label"] == 1].reset_index(drop=True)
        unlabelled_units = units.loc[combined["label"] == 0].reset_index(drop=True)
        features = [column for column in combined.columns if column != "label"]
        pos_train, pos_test = train_test_split(
            positives[features], train_size=0.75, random_state=self.seed
        )
        xy_train_original = pd.concat(
            [pos_train.reset_index(drop=True), unlabelled[features].reset_index(drop=True)],
            ignore_index=True,
        )
        xy_train_original["label"] = pd.concat(
            [positives.loc[pos_train.index, "label"].reset_index(drop=True), unlabelled["label"]],
            ignore_index=True,
        )
        xy_train_units = pd.concat(
            [
                positive_units.loc[pos_train.index].reset_index(drop=True),
                unlabelled_units.reset_index(drop=True),
            ],
            ignore_index=True,
        )
        xy_pos_test_units = positive_units.loc[pos_test.index].reset_index(drop=True)
        # 与当前 Notebook 一致，此处仅拟合训练数据。
        if self.numerical_columns:
            scaled_train = self.scaler.fit_transform(xy_train_original[self.numerical_columns])
            scaled_test = self.scaler.transform(pos_test[self.numerical_columns])
            self.scaler_fitted = True
        else:
            scaled_train = np.empty((len(xy_train_original), 0))
            scaled_test = np.empty((len(pos_test), 0))
        xy_train = pd.concat(
            [pd.DataFrame(scaled_train, columns=self.numerical_columns),
             xy_train_original[encoded_columns].reset_index(drop=True),
             xy_train_original[["sample_weight", "label"]].reset_index(drop=True)],
            axis=1,
        )
        xy_pos_test = pd.concat(
            [pd.DataFrame(scaled_test, columns=self.numerical_columns),
             pos_test[encoded_columns].reset_index(drop=True),
             pos_test[["sample_weight"]].reset_index(drop=True),
             positives.loc[pos_test.index, ["label"]].reset_index(drop=True)],
            axis=1,
        )
        self.feature_columns = [column for column in xy_train.columns if column not in {"sample_weight", "label"}]
        return PreparedTrainingData(
            xy_train,
            xy_pos_test,
            xy_train_original,
            self.feature_columns,
            xy_train_units,
            xy_pos_test_units,
        )

    def transform_target_legacy(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """复刻 Notebook 在目标侧执行 ``scaler.fit`` 的行为。

        TODO(science)：只有在经过有意版本化的科学实验后，才应将其改为
        ``scaler.transform``，不应在迁移过程中修改。
        """
        warnings.warn(
            "Baseline compatibility: refitting StandardScaler on target features before transform.",
            UserWarning,
            stacklevel=2,
        )
        if self.numerical_columns:
            target_numeric = self.scaler.fit_transform(target_data[self.numerical_columns])
        else:
            target_numeric = np.empty((len(target_data), 0))
        if self.categorical_columns:
            target_categorical = self.encoder.transform(target_data[self.categorical_columns]).toarray()
        else:
            target_categorical = np.empty((len(target_data), 0))
        target = np.hstack((target_numeric, target_categorical))
        return pd.DataFrame(target, columns=self.feature_columns)
