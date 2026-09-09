"""折内隔离的 sklearn 兼容预处理 Transformer。

用于 ``raw_gis`` 执行模式下带内层 CV 的调参（如 ``bayes``）。每个内层 CV
fold 都拿到**筛选前完整特征**，独立重跑相关性筛选、OneHotEncoder 类别学习
与 StandardScaler 拟合，从而避免验证折样本参与任何学习型预处理统计（P0-01）。
不同折特征维度不同是正常现象，各折模型独立即可；最终外层模型在完整外层训练
集上重拟合。

早期实现只重拟合 scaler，相关性筛选与 OHE 类别复用完整外层训练集的 schema，
验证折专属类别会被编码为 1（而非未知类别的全零），即验证特征已经改变训练
管线。折内隔离要求所有学习型预处理都只在训练折子集上拟合。
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocess import BaselinePreprocessor


class FoldSafePreprocessor(BaseEstimator, TransformerMixin):
    """在交叉验证折内完整重拟合预处理的步骤。"""

    def __init__(self, schema: BaselinePreprocessor):
        self.schema = schema

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        # 防御性回退：numpy 数组按筛选前完整列顺序还原为 DataFrame。
        columns = (
            list(getattr(self.schema, "_all_numerical_columns", []) or [])
            + list(getattr(self.schema, "_all_categorical_columns", []) or [])
        )
        if not columns:
            columns = list(self.schema.numerical_columns) + list(self.schema.categorical_columns)
        return pd.DataFrame(X, columns=columns)

    def fit(self, X, y=None):
        frame = self._frame(X)
        pp = self.schema.clone()
        # 在折内训练子集上完整重拟合相关性筛选 + OHE + scaler（P0-01）。
        unit_columns = getattr(self.schema, "unit_columns", ("X", "Y"))
        pp.fit(frame, unit_columns=unit_columns)
        self.pp_ = pp
        return self

    def transform(self, X):
        return self.pp_.transform(self._frame(X))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
