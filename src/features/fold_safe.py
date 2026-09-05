"""折内隔离的 sklearn 兼容预处理 Transformer。

用于 ``raw_gis`` 执行模式下带内层 CV 的调参（如 ``bayes``）。特征 schema
（相关性筛选列选择 + OHE 类别）在完整外层训练集确定一次，每个内层 CV
fold 只重拟合 StandardScaler，从而避免验证折样本参与缩放统计。OHE 类别
固定并配 ``handle_unknown="ignore"``，折内未见类别在验证时映射为全零。

与直接把 ``BaselinePreprocessor`` 放进 ``Pipeline`` 不同，本步骤**只**重
拟合 scaler；列选择与 OHE 类别来自构造时传入的 ``schema``，保证每个折的
特征维度一致，避免相关性筛选在折间产生不同的列集合。
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocess import BaselinePreprocessor


class FoldSafePreprocessor(BaseEstimator, TransformerMixin):
    """在交叉验证折内重拟合 scaler 的预处理步骤。"""

    def __init__(self, schema: BaselinePreprocessor):
        self.schema = schema

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        # 防御性回退：numpy 数组按 schema 列顺序还原为 DataFrame。
        columns = list(self.schema.numerical_columns) + list(self.schema.categorical_columns)
        return pd.DataFrame(X, columns=columns)

    def fit(self, X, y=None):
        frame = self._frame(X)
        pp = self.schema.clone()
        # 复用完整外层训练集确定的 schema，只重拟合 scaler。
        pp.numerical_columns = list(self.schema.numerical_columns)
        pp.categorical_columns = list(self.schema.categorical_columns)
        pp._encoded_columns = list(self.schema._encoded_columns)
        pp.feature_columns = list(self.schema.feature_columns)
        pp.correlation = self.schema.correlation
        pp.encoder = self.schema.encoder
        pp.encoder_fitted = self.schema.encoder_fitted
        pp.refit_scaler(frame)
        self.pp_ = pp
        return self

    def transform(self, X):
        return self.pp_.transform(self._frame(X))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
