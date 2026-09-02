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
        """按照 Notebook 语义筛选、编码、缩放并切分原始单元。

        与旧版基线实现的关键区别：相关性筛选和 OneHotEncoder 拟合
        现在在训练/测试划分之后执行，仅在训练集上拟合，避免数据泄漏。
        """
        training = pd.concat([deposits, unlabeled], ignore_index=True)
        numerical, categorical = split_feature_columns(training, unit_columns)

        # 先划分正样本为训练/测试集
        pos_mask = training["label"] == 1
        positives = training.loc[pos_mask].reset_index(drop=True)
        unlabelled_all = training.loc[~pos_mask].reset_index(drop=True)
        pos_train, pos_test = train_test_split(
            positives, train_size=0.75, random_state=self.seed
        )

        # 训练集 = pos_train + 全部未标注样本
        train_set = pd.concat([pos_train, unlabelled_all], ignore_index=True)

        # 仅在训练集上计算相关性
        correlation = train_set[numerical].corr(method="spearman").abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        dropped = [
            column for column in upper.columns
            if any(upper[column] > self.config["correlation_threshold"])
        ]
        self.numerical_columns = [column for column in numerical if column not in dropped]
        self.categorical_columns = categorical
        self.correlation = correlation

        # 仅在训练集上拟合 OneHotEncoder
        if categorical:
            encoded_train = self.encoder.fit_transform(train_set[categorical]).toarray()
            encoded_test = self.encoder.transform(pos_test[categorical]).toarray()
            self.encoder_fitted = True
            try:
                encoded_columns = self.encoder.get_feature_names(categorical).tolist()
            except AttributeError:
                encoded_columns = self.encoder.get_feature_names_out(categorical).tolist()
        else:
            encoded_train = np.empty((len(train_set), 0))
            encoded_test = np.empty((len(pos_test), 0))
            encoded_columns = []

        # 构建训练集 DataFrame
        train_combined = pd.concat(
            [
                train_set[self.numerical_columns].reset_index(drop=True),
                pd.DataFrame(encoded_train, columns=encoded_columns),
                train_set[["sample_weight", "label"]].reset_index(drop=True),
            ],
            axis=1,
        )
        train_pos = train_combined.loc[train_combined["label"] == 1].reset_index(drop=True)
        train_unlabelled = train_combined.loc[train_combined["label"] == 0].reset_index(drop=True)

        # 构建测试集 DataFrame
        test_combined = pd.concat(
            [
                pos_test[self.numerical_columns].reset_index(drop=True),
                pd.DataFrame(encoded_test, columns=encoded_columns),
                pos_test[["sample_weight", "label"]].reset_index(drop=True),
            ],
            axis=1,
        )

        # 单元元数据
        available_unit_columns = [column for column in unit_columns if column in training.columns]
        units = training[available_unit_columns].reset_index(drop=True)
        positive_units = units.loc[pos_mask].reset_index(drop=True)
        train_pos_units = positive_units.loc[pos_train.index].reset_index(drop=True)
        test_pos_units = positive_units.loc[pos_test.index].reset_index(drop=True)
        unlabelled_units = units.loc[~pos_mask].reset_index(drop=True)

        features = [column for column in train_combined.columns if column != "label"]

        xy_train_original = pd.concat(
            [
                train_pos[features].reset_index(drop=True),
                train_unlabelled[features].reset_index(drop=True),
            ],
            ignore_index=True,
        )
        xy_train_original["label"] = pd.concat(
            [
                train_pos["label"].reset_index(drop=True),
                train_unlabelled["label"].reset_index(drop=True),
            ],
            ignore_index=True,
        )
        xy_train_units = pd.concat(
            [
                train_pos_units.reset_index(drop=True),
                unlabelled_units.reset_index(drop=True),
            ],
            ignore_index=True,
        )
        xy_pos_test_units = test_pos_units.reset_index(drop=True)

        # 仅在训练集上拟合 StandardScaler
        if self.numerical_columns:
            scaled_train = self.scaler.fit_transform(xy_train_original[self.numerical_columns])
            scaled_test = self.scaler.transform(test_combined[self.numerical_columns])
            self.scaler_fitted = True
        else:
            scaled_train = np.empty((len(xy_train_original), 0))
            scaled_test = np.empty((len(test_combined), 0))

        xy_train = pd.concat(
            [
                pd.DataFrame(scaled_train, columns=self.numerical_columns),
                xy_train_original[encoded_columns].reset_index(drop=True),
                xy_train_original[["sample_weight", "label"]].reset_index(drop=True),
            ],
            axis=1,
        )
        xy_pos_test = pd.concat(
            [
                pd.DataFrame(scaled_test, columns=self.numerical_columns),
                test_combined[encoded_columns].reset_index(drop=True),
                test_combined[["sample_weight"]].reset_index(drop=True),
                test_combined[["label"]].reset_index(drop=True),
            ],
            axis=1,
        )
        self.feature_columns = [
            column for column in xy_train.columns if column not in {"sample_weight", "label"}
        ]
        return PreparedTrainingData(
            xy_train,
            xy_pos_test,
            xy_train_original,
            self.feature_columns,
            xy_train_units,
            xy_pos_test_units,
        )

    def transform_target_legacy(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """对目标数据应用与训练集相同的预处理变换。

        使用已拟合的 Scaler 和 OneHotEncoder 进行 transform（不再重新 fit）。

        TODO(science)：旧版基线实现在目标侧执行 ``scaler.fit_transform``，
        现已修复为 ``scaler.transform``。通过实验种子确保可复现性。
        """
        if self.scaler_fitted:
            if self.numerical_columns:
                target_numeric = self.scaler.transform(target_data[self.numerical_columns])
            else:
                target_numeric = np.empty((len(target_data), 0))
        else:
            warnings.warn(
                "Scaler not fitted during training; falling back to fit_transform on target data.",
                UserWarning,
                stacklevel=2,
            )
            if self.numerical_columns:
                target_numeric = self.scaler.fit_transform(target_data[self.numerical_columns])
            else:
                target_numeric = np.empty((len(target_data), 0))
        if self.encoder_fitted:
            if self.categorical_columns:
                target_categorical = self.encoder.transform(target_data[self.categorical_columns]).toarray()
            else:
                target_categorical = np.empty((len(target_data), 0))
        else:
            warnings.warn(
                "Encoder not fitted during training; falling back to fit_transform on target data.",
                UserWarning,
                stacklevel=2,
            )
            if self.categorical_columns:
                target_categorical = self.encoder.fit_transform(target_data[self.categorical_columns]).toarray()
            else:
                target_categorical = np.empty((len(target_data), 0))
        target = np.hstack((target_numeric, target_categorical))
        return pd.DataFrame(target, columns=self.feature_columns)
