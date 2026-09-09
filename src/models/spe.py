#!/usr/bin/env python3
"""Self-Paced Ensemble (SPE) 模型适配器。

Self-Paced Ensemble 是一种专为高度不平衡分类设计的集成方法，参考论文：
"Self-paced Ensemble for Highly Imbalanced Massive Data Classification" (ICDE 2020, Zhining Liu et al.)

核心思想：通过"自步学习"（self-paced learning）机制，逐步从简单到困难地选择
多数类样本，训练多个基分类器，最终通过软投票进行集成预测。

该实现是自包含的，不依赖外部 self-paced-ensemble 包。

.. warning::
    本实现是**自定义变体**，不等于 ICDE 2020 原论文 Algorithm 1 的数值复现
    （P1-01）：这里按硬度降序**等人数分箱**并使用 ``w_j = 1/(1 + alpha*j)``；
    原论文使用各分箱的**平均硬度** ``h_j`` 与采样权重 ``1/(h_j + alpha)``，且
    alpha 的调度边界不同。alpha 很大时，本实现的归一化权重趋向最难箱，而论文
    形式趋向各非空箱等额。若需作为标准 SPE 基线，应改为对齐作者实现并固定版本，
    否则请在论文中把它作为「自定义硬度分箱 SPE 变体」与标准 SPE 分列报告。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..core.contracts import TrainingData
from .registry import MODEL_REGISTRY


# ============================================================================
# SelfPacedEnsemble 分类器 — 自步集成核心算法
# ============================================================================


class SelfPacedEnsemble(BaseEstimator, ClassifierMixin):
    """自步集成（Self-Paced Ensemble）分类器。

    通过迭代训练基分类器，每轮从多数类中按"硬度"（hardness）采样平衡子集，
    并结合自步因子 α 逐步从简单样本过渡到困难样本，最终以软投票集成所有基分类器。

    Parameters
    ----------
    base_estimator : estimator, default=DecisionTreeClassifier()
        基分类器。每轮迭代将克隆该分类器并在平衡子集上训练。
    n_estimators : int, default=10
        基分类器数量（集成轮数）。
    k_bins : int, default=5
        硬度分箱数。多数类样本按硬度排序后等分为 k_bins 个箱。
    replacement : bool, default=False
        从多数类采样时是否使用有放回采样。
    random_state : int or None, default=None
        随机种子，用于控制采样和基分类器的可复现性。

    Attributes
    ----------
    estimators_ : list of fitted estimators
        训练完成的基分类器列表。
    classes_ : ndarray of shape (n_classes,)
        训练数据中的类别标签。

    Notes
    -----
    算法流程：

    1. 分离少数类（正样本，label=1）和多数类（负样本，label=0）。
    2. 对于每轮迭代 i = 0..n_estimators-1：
       a. 计算自步因子 α = tan(π * (i+1) / (2 * n_estimators))。
          α 从接近 0 逐渐增大，控制"课程"的难度进度。
       b. 如果不是第一轮：
          - 用当前集成计算每个多数类样本的"硬度"。
            硬度 = 被预测为正类（少数类）的概率。
            对于多数类样本，概率越高说明模型越"困惑"——即越难正确分类。
          - 按硬度降序排列多数类样本，等分为 k_bins 个箱。
          - 箱 0 为最硬样本（高概率被误判为正类），箱 k_bins-1 为最简单样本。
       c. 如果是第一轮（尚无集成模型）：
          - 随机欠采样多数类至与少数类等量。
       d. 计算各箱的采样权重：w_j = 1 / (1 + α * j)，j=0..k_bins-1。
          当 α 小时，权重分布均匀 → 简单样本为主；
          当 α 大时，硬样本（低 j）权重增加 → 逐步引入困难样本。
       e. 按权重从各箱中采样，使多数类子集与少数类等量。
       f. 合并少数类全量 + 采样的多数类子集，训练基分类器（通过 clone 克隆）。
       g. 将训练好的基分类器加入集成。
    3. 预测：predict() 使用多数表决，predict_proba() 使用软投票（所有基分类器概率均值）。
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        k_bins: int = 5,
        replacement: bool = False,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.k_bins = k_bins
        self.replacement = replacement
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """训练自步集成模型。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练特征。
        y : array-like of shape (n_samples,)
            二分类标签，1 为少数类（正样本），0 为多数类（负样本）。
        sample_weight : array-like of shape (n_samples,), optional
            样本权重，将透传给每轮基分类器的 fit() 方法。

        Returns
        -------
        self : SelfPacedEnsemble
        """
        # 校验输入数据
        X, y = check_X_y(X, y, accept_sparse=False, dtype=None, ensure_min_samples=2)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"SelfPacedEnsemble 仅支持二分类任务，当前数据包含 {len(self.classes_)} 个类别: {self.classes_}"
            )

        # 确定少数类与多数类。平衡数据时 min/max(dict) 可能返回同一类，
        # 因此以稳定的 classes_ 顺序打破平局，并显式保存两个不同的类别。
        counts = np.asarray([np.sum(y == label) for label in self.classes_], dtype=int)
        minority_pos = int(np.argmin(counts))
        majority_pos = int(np.argmax(counts))
        if minority_pos == majority_pos:
            majority_pos = 1 - minority_pos
        minority_class = self.classes_[minority_pos]
        majority_class = self.classes_[majority_pos]
        self.minority_class_ = minority_class
        self.majority_class_ = majority_class

        # 分离少数类和多数类样本索引
        min_idx = np.where(y == minority_class)[0]
        maj_idx = np.where(y == majority_class)[0]

        # 将数据转为 numpy 数组便于索引
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)
        else:
            sw_arr = None

        # 少数类全量数据
        X_min = X_arr[min_idx]
        y_min = y_arr[min_idx]
        sw_min = sw_arr[min_idx] if sw_arr is not None else None

        n_min = len(min_idx)  # 少数类样本数，用于确定多数类子集大小

        # 初始化基分类器：若未指定则使用默认决策树
        if self.base_estimator is None:
            base = DecisionTreeClassifier(random_state=self.random_state)
        else:
            base = clone(self.base_estimator)

        # 随机数生成器
        rng = np.random.RandomState(self.random_state)

        # 集成容器
        self.estimators_ = []

        for i in range(self.n_estimators):
            # --- 步骤 2a: 计算自步因子 α ---
            # α = tan(π * (i+1) / (2 * n_estimators))
            # 当 i=0 时 α ≈ π/(2n) 很小；随 i 增大 α 逐渐增大
            # 使用 (n_estimators + 1) 作为分母以避免 i=n_estimators-1 时 α 趋近无穷
            alpha = np.tan(np.pi * 0.5 * (i + 1) / (self.n_estimators + 1))

            if i == 0 or len(self.estimators_) == 0:
                # --- 步骤 2c: 第一轮迭代，随机欠采样多数类 ---
                sampled_maj_idx = rng.choice(
                    maj_idx, size=n_min, replace=self.replacement
                )
            else:
                # --- 步骤 2b: 计算硬度并按硬度分箱采样 ---
                # 硬度 = 集成对多数类样本预测为正类（少数类）的概率
                # 概率越高 → 模型越困惑 → 样本越"硬"
                hardness = self._compute_hardness(X_arr[maj_idx])

                # 按硬度降序排列（最硬的在前）
                sort_order = np.argsort(-hardness)
                sorted_maj_idx = maj_idx[sort_order]

                # 使用 array_split，确保样本数小于 k_bins 时不会产生空箱配额丢失。
                # 每个样本恰好属于一个箱，且箱数最多为 k_bins。
                n_maj = len(sorted_maj_idx)
                bins = [
                    chunk
                    for chunk in np.array_split(sorted_maj_idx, min(self.k_bins, n_maj))
                    if len(chunk)
                ]
                n_bins = len(bins)

                # --- 步骤 2d: 计算各箱的采样权重 ---
                # w_j = 1 / (1 + α * j)，j=0 为最硬箱
                # α 小时权重均匀，α 大时硬箱（小 j）权重更高
                bin_weights = np.array(
                    [1.0 / (1.0 + alpha * j) for j in range(n_bins)]
                )
                bin_probs = bin_weights / bin_weights.sum()

                # --- 步骤 2e: 按权重从各箱中采样 ---
                samples_per_bin = rng.multinomial(n_min, bin_probs)
                sampled_indices = []
                for bin_j, bin_indices in enumerate(bins):
                    n_sample = int(samples_per_bin[bin_j])
                    if not self.replacement:
                        n_sample = min(n_sample, len(bin_indices))
                    if n_sample > 0:
                        chosen = rng.choice(
                            bin_indices, size=n_sample, replace=self.replacement
                        )
                        sampled_indices.extend(chosen)

                # 无放回时小箱可能无法满足原 multinomial 配额；从尚未
                # 选中的多数类补齐，保证每个基学习器仍是平衡子集。
                remaining = n_min - len(sampled_indices)
                if remaining > 0:
                    selected = {int(value) for value in sampled_indices}
                    available = np.asarray(
                        [
                            value
                            for value in maj_idx
                            if self.replacement or int(value) not in selected
                        ],
                        dtype=int,
                    )
                    if len(available) < remaining and not self.replacement:
                        raise ValueError(
                            "SPE could not allocate a balanced majority subset; "
                            f"need {remaining} additional samples, found {len(available)}"
                        )
                    sampled_indices.extend(
                        rng.choice(available, size=remaining, replace=self.replacement)
                    )

                sampled_maj_idx = np.asarray(sampled_indices, dtype=int)

            # --- 步骤 2f: 合并少数类 + 采样多数类，训练基分类器 ---
            train_idx = np.concatenate([min_idx, sampled_maj_idx])
            X_train = X_arr[train_idx]
            y_train = y_arr[train_idx]

            # 克隆基分类器并在当前子集上训练
            estimator = clone(base)
            if sw_arr is not None:
                sw_train = sw_arr[train_idx]
                estimator.fit(X_train, y_train, sample_weight=sw_train)
            else:
                estimator.fit(X_train, y_train)

            # --- 步骤 2g: 将训练好的基分类器加入集成 ---
            self.estimators_.append(estimator)

        return self

    def _compute_hardness(self, X_maj):
        """计算多数类样本的硬度。

        硬度 = 当前集成预测多数类样本为正类（少数类）的平均概率。
        概率越高说明模型越倾向于将其误判为正类，即该样本越"困难"。

        Parameters
        ----------
        X_maj : ndarray
            多数类样本特征矩阵。

        Returns
        -------
        hardness : ndarray of shape (n_maj,)
            每个多数类样本的硬度值，范围 [0, 1]。
        """
        n_maj = X_maj.shape[0]
        probas = np.zeros(n_maj)
        for est in self.estimators_:
            if hasattr(est, "predict_proba"):
                proba = np.asarray(est.predict_proba(X_maj), dtype=float)
                est_classes = np.asarray(getattr(est, "classes_", self.classes_))
                matching = np.flatnonzero(est_classes == self.minority_class_)
                if len(matching) == 0:
                    # 单类基学习器没有 minority 概率；若它只知道 majority，
                    # 其 minority 概率确定为 0，绝不将单列数组广播到两类。
                    contribution = np.zeros(n_maj, dtype=float)
                else:
                    contribution = proba[:, int(matching[0])]
                probas += contribution
            else:
                # 降级：使用 predict + 明确的 minority 类别比较
                pred = est.predict(X_maj)
                probas += (pred == self.minority_class_).astype(float)
        return probas / len(self.estimators_)

    def predict_proba(self, X):
        """软投票：返回所有基分类器预测概率的均值。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            待预测样本。

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            每个类别的平均预测概率。
        """
        check_is_fitted(self, "estimators_")
        X = check_array(X, accept_sparse=False)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))

        for est in self.estimators_:
            est_proba = np.asarray(est.predict_proba(X), dtype=float)
            est_classes = np.asarray(getattr(est, "classes_", self.classes_))
            aligned = np.zeros((n_samples, n_classes), dtype=float)
            for source_col, label in enumerate(est_classes):
                target_col = np.flatnonzero(self.classes_ == label)
                if len(target_col) == 0:
                    raise ValueError(
                        f"Base estimator returned unknown class {label!r}; "
                        f"ensemble classes are {self.classes_.tolist()}"
                    )
                aligned[:, int(target_col[0])] = est_proba[:, source_col]
            probas += aligned

        return probas / len(self.estimators_)

    def predict(self, X):
        """多数表决：返回预测类别标签。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            待预测样本。

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            预测类别标签。
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_params(self, deep=True):
        """获取模型参数，支持嵌套的 base_estimator__ 前缀参数。

        该方法使 BayesSearchCV 能够通过 set_params 调优基分类器参数。
        """
        params = {
            "n_estimators": self.n_estimators,
            "k_bins": self.k_bins,
            "replacement": self.replacement,
            "random_state": self.random_state,
        }
        if deep and self.base_estimator is not None:
            for key, val in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{key}"] = val
        else:
            params["base_estimator"] = self.base_estimator
        return params

    def set_params(self, **params):
        """设置模型参数，自动将 base_estimator__ 前缀参数路由到基分类器。

        该方法使 BayesSearchCV 能够通过 set_params 调优基分类器参数。
        """
        own_params = {}
        base_params = {}
        for key, val in params.items():
            if key.startswith("base_estimator__"):
                # 嵌套参数：base_estimator__max_depth → max_depth
                base_params[key[len("base_estimator__"):]] = val
            elif key == "base_estimator":
                self.base_estimator = val
            else:
                own_params[key] = val

        for key, val in own_params.items():
            setattr(self, key, val)

        if base_params and self.base_estimator is not None:
            self.base_estimator.set_params(**base_params)

        return self


# ============================================================================
# SPEAdapter — MPM 框架适配器
# ============================================================================


@MODEL_REGISTRY.decorator("spe")
class SPEAdapter:
    """Self-Paced Ensemble 模型适配器，将 SPE 集成到 MPM 二阶段实验框架。

    通过 ComponentRegistry 机制注册为 "spe" 模型，与框架的 Experiment、
    Tuner、Splitter 等组件无缝协作。

    Attributes
    ----------
    name : str
        注册名称 "spe"。
    artifact_filename : str
        序列化模型文件名。
    supports_constraints : bool
        SPE 不支持谓词约束。
    """

    name = "spe"
    artifact_filename = "model_spe.pkl"
    supports_constraints = False

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        """校验 SPE 配置参数的合法性。

        校验内容包括：
        - n_estimators 为正整数
        - k_bins 为正整数
        - 基分类器参数可通过 DecisionTreeClassifier 构造校验
        """
        n_estimators = int(params.get("n_estimators", 10))
        if n_estimators < 1:
            raise ValueError("spe n_estimators must be positive")
        k_bins = int(params.get("k_bins", 5))
        if k_bins < 1:
            raise ValueError("spe k_bins must be positive")

        # 提取基分类器参数并校验
        base_params = {}
        for key, val in params.items():
            if key.startswith("base_estimator__"):
                base_params[key[len("base_estimator__"):]] = val

        # 基分类器默认从 params 中提取 "base_estimator" 类型
        # 当前仅支持 decision_tree（默认），通过校验 DecisionTreeClassifier 参数来验证
        base_type = params.get("base_estimator", "decision_tree")
        if base_type == "decision_tree":
            DecisionTreeClassifier(**base_params)

    def build(self, params: Mapping[str, Any], seed: int) -> SelfPacedEnsemble:
        """根据配置参数构建 SelfPacedEnsemble 实例。

        参数解析逻辑：
        1. 从 params 中提取 SPE 级参数（n_estimators, k_bins, replacement）。
        2. 从 params 中提取 base_estimator__ 前缀的基分类器参数。
        3. 构建默认基分类器 DecisionTreeClassifier 并应用参数。
        4. 构建并返回 SelfPacedEnsemble。

        Parameters
        ----------
        params : Mapping[str, Any]
            模型配置参数（来自 YAML 配置的 model.params 节）。
        seed : int
            实验级随机种子，用于确保可复现性。

        Returns
        -------
        SelfPacedEnsemble
            配置好的 SPE 分类器实例。
        """
        resolved = dict(params)

        # 提取 SPE 级参数
        n_estimators = int(resolved.get("n_estimators", 10))
        k_bins = int(resolved.get("k_bins", 5))
        replacement = bool(resolved.get("replacement", False))

        # 提取基分类器参数（base_estimator__ 前缀）
        base_params = {}
        for key, val in resolved.items():
            if key.startswith("base_estimator__"):
                base_params[key[len("base_estimator__"):]] = val

        # 构建基分类器
        base_type = resolved.get("base_estimator", "decision_tree")
        if base_type == "decision_tree":
            base_params.setdefault("random_state", seed)
            base = DecisionTreeClassifier(**base_params)
        else:
            raise ValueError(f"SPE 不支持的基分类器类型: {base_type}")

        # 构建 SPE
        return SelfPacedEnsemble(
            base_estimator=base,
            n_estimators=n_estimators,
            k_bins=k_bins,
            replacement=replacement,
            random_state=seed,
        )

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        """返回传递给模型 fit() 方法的额外参数。

        SPE 内部通过重采样处理类别不平衡，但 sample_weight 仍会透传给
        基分类器（如 DecisionTreeClassifier），以反映各地质点的置信度权重。

        Parameters
        ----------
        data : TrainingData
            训练数据包，包含 features, labels, sample_weight 等。

        Returns
        -------
        Mapping[str, Any]
            包含 sample_weight 的字典。
        """
        return {"sample_weight": data.sample_weight}