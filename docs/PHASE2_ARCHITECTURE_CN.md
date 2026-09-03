# MPM 二阶段工程化架构

## 1. 二阶段目标

二阶段不重写原始地学算法，重点解决“新增一种研究变量时是否需要修改核心实验编排器”的问题。

核心规则：`Experiment` 只依赖稳定契约和注册表，不直接依赖 RF、具体 GIS 算子、具体谓词、具体交叉验证或调参实现。

当前组件图：

```text
ExperimentSpec
    |
    +-- Task Registry
    +-- Feature Operator Registry -> FeaturePipeline
    +-- Label Refiner Registry (legacy PUB compatibility)
    +-- Knowledge Registry -> KnowledgePipeline
    +-- Predicate Registry -> PredicatePipeline
    +-- Splitter Registry
    +-- Model Registry
    +-- Tuner Registry
    +-- Metric Registry
```

## 2. 主要目录

```text
src/
├── core/
│   ├── bootstrap.py       # 加载内置插件
│   ├── config.py          # YAML、phase-1 兼容迁移、注册表校验
│   ├── contracts.py       # TrainingData / SplitData / Protocol
│   ├── experiment.py      # 纯编排
│   ├── registry.py        # 通用注册表
│   └── spec.py            # ExperimentSpec / ComponentSpec
├── tasks/                 # 找矿任务语义
├── operators/features/    # 可组合特征算子
├── models/                # 模型适配器与兼容性标签重构器
├── knowledge/             # 知识提供者，和谓词分离
├── predicates/            # 谓词/约束入口
├── validation/            # holdout / CV / metrics
├── tuning/                # none / bayes 等调参策略
├── data/                  # 数据集、研究变量（研究单元/采样器/标签/权重，注册化）
└── features/              # 保留的预处理与 phase-1 兼容入口
```

## 3. 已完成的解耦

### Task

`target_area_prediction` 已注册；研究单元、未标注样本、预测网格和预测输出由 Task 通过注册表调用。研究变量（`RESEARCH_UNIT_REGISTRY`、`BACKGROUND_SAMPLER_REGISTRY`、`LABEL_STRATEGY_REGISTRY`、`WEIGHT_STRATEGY_REGISTRY`）已取代硬编码枚举，Task 不再直接 import 模块级研究单元/标签函数。

`deep_edge_prediction` 已注册为能力占位，但会明确报错。原因是当前仓库没有 3D/voxel 研究单元、钻孔/深度标签、3D 特征算子和输出契约，不能用二维逻辑伪装支持。

### Model

RF 通过 `MODEL_REGISTRY` 选择。`Experiment` 不再调用 `train_rf()`。

增加模型应新增 adapter 并注册，不修改 `Experiment`。

### Tuner

模型构造与调参分离：

- `none`：直接训练；
- `bayes`：可选 `scikit-optimize`。

搜索空间改成显式类型：`categorical / integer / real`。

### Feature Operator

原 `SpatialFeatureExtractor` 已降为兼容层。新主路径使用 `FeaturePipeline`，内置：

- `raster_statistics`
- `texture`
- `elevation_gradient`
- `line_distance`
- `categorical_geology`

算子按 YAML 顺序执行，并检查行数和重复列。

### Predicate 与 Knowledge

二者单独建模：

- Knowledge Provider：产生知识工件；
- Predicate：消费训练数据和知识上下文，产生约束或训练数据变换。

当前内置 `empty`（接口占位）和 `spatial_extent`（从研究单元坐标提取研究区域空间范围）两个 knowledge provider，以及 `all_ones`/`spatial_box`/`spatial_distance`/`combined` 四个 constraint 谓词。`spatial_box` 会在 `auto_center`/`auto_side` 时优先消费 `spatial_extent` 知识，回退到从数据重新估计。

### Validation / Metric

holdout、交叉验证和指标由注册表选择。

当前：

- `random_holdout`
- `stratified_kfold`
- accuracy / precision / recall / f1 / roc_auc / confusion_matrix

空间 CV 尚未实现，应新增 splitter，而不是修改 `Experiment`。

## 4. 配置契约

二阶段配置核心：

```yaml
features:
  operators:
    - name: raster_statistics
      params:
        buffer_size: 10

model:
  name: rf
  params: {}

tuning:
  name: none
  params: {}

validation:
  holdout:
    name: random_holdout
    params:
      test_size: 0.25
  cross_validation:
    name: stratified_kfold
    params:
      n_splits: 5
      shuffle: false
  metrics: [accuracy, precision, recall, f1, roc_auc]
  primary_metric: f1

knowledge:
  enabled: false
  items: []

predicates:
  enabled: false
  combine: sequential
  items: []
```

旧 phase-1 YAML 仍可加载，`core/config.py` 会将布尔 feature 配置、`model.search`、`model.pu`、旧 validation 字段迁移成二阶段结构。

## 5. 三种执行模式

### archive_replay

只读取已保存特征/模型/预测结果，适合验证迁移前后 artifact 不变。

### train_from_archive_features

使用已保存特征重新训练模型。适合快速验证模型、tuner、指标、predicate/knowledge 接口，不需要重新跑 GIS。

### raw_gis

从原始 GIS 构造研究单元和特征。此路径保留原 Notebook 的若干科学兼容行为，相关 warning 不应被当作最终研究方案。

二阶段同时修复了 phase-1 中 `target_mask[:, 2]` 对 DataFrame 进行 NumPy 索引导致的 raw GIS `KeyError`。

## 6. 扩展规则

新增组件时遵守一条硬规则：**如果新增实现需要修改 `core/experiment.py` 的 if/elif 才能被选择，说明扩展边界设计失败。**

典型扩展：

- 新模型：新增 model adapter + `MODEL_REGISTRY.register`；
- 新特征：新增 operator + `FEATURE_OPERATOR_REGISTRY.register`；
- 新谓词：新增 predicate + `PREDICATE_REGISTRY.register`；
- 新知识：新增 provider + `KNOWLEDGE_REGISTRY.register`；
- 新空间 CV：新增 splitter + `SPLITTER_REGISTRY.register`；
- 新指标：新增函数 + `METRIC_REGISTRY.register`；
- 新调参方法：新增 tuner + `TUNER_REGISTRY.register`；
- 新任务：新增 Task；只有当共享契约确实不足时才修改核心 contract。

## 7. 当前仍未解决的问题

二阶段是框架重构，不代表研究能力已经全部实现：

1. `deep_edge_prediction` 仍缺 3D 数据契约；
2. 当前主模型只有 RF；
3. 当前没有真实地质 Knowledge Provider 和 Predicate；
4. 当前没有 spatial block / leave-area-out 等空间验证；
5. 基线预处理仍保留“split 前相关性筛选、目标侧重新 fit scaler”等兼容逻辑，应作为后续科学实验单独版本化；
6. PUB/Bayes 调参依赖 `pulearn` / `scikit-optimize`，采用延迟导入；
7. 自然语言不直接控制运行器。正确路径应为“自然语言 -> Experiment YAML -> schema/registry validation -> shell”。
