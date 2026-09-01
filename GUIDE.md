# MPM_codex Phase 2 — 找矿预测模型综合实验框架 使用说明书

## 目录

1. [概述](#1-概述)
2. [快速上手](#2-快速上手)
3. [代码结构与目录说明](#3-代码结构与目录说明)
4. [核心概念](#4-核心概念)
5. [基本使用方式](#5-基本使用方式)
6. [归档 CSV 数据格式](#6-归档-csv-数据格式)
7. [模型适配器原理](#7-模型适配器原理)
8. [如何扩展组件](#8-如何扩展组件)
9. [实验对比指南](#9-实验对比指南)
10. [YAML 配置参考](#10-yaml-配置参考)
11. [输出目录结构](#11-输出目录结构)
12. [常见问题排查](#12-常见问题排查)

---

## 1. 概述

MPM_codex 是一个面向**找矿预测（Mineral Prospectivity Mapping）**的工程化实验框架，核心设计理念：

- **可注册**：模型、特征算子、调优器、指标、谓词等全部通过统一注册表管理，新增组件无需修改框架核心代码
- **可配置**：所有实验参数通过 YAML 配置文件驱动，配合 `--set` CLI 参数覆盖，无需硬编码
- **可验证**：内置完整的实验管线（数据→特征→训练→评估→预测→导出），自动生成可复现性清单（manifest.json）

### 数据流总览

```
YAML 配置 → load_config() → DEFAULTS 合并 → Phase1 迁移 → 路径解析
    → validate_config() → ExperimentConfig → Experiment.run()
        ├── ① prepare_data     （加载归档数据或构建研究单元）
        ├── ② build_features   （快照特征或运行 GIS 算子）
        ├── ③ prepare_dataset  （校验 schema 或预处理）
        ├── ④ train            （回放归档模型 或 调优器训练）
        ├── ⑤ evaluate         （计算指标）
        ├── ⑥ predict          （生成预测概率，导出 CSV/GeoTIFF）
        └── export             （写入 manifest.json 可复现性清单）
```

---

## 2. 快速上手

```bash
# 进入项目目录
cd MPM_codex_phase2

# 查看所有已注册组件
python scripts/list_components.py

# 验证配置是否正确（不访问数据文件，快速检查）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml --validate-only

# 运行 RF 基线回放实验（archive_replay 模式，直接复制归档结果）
python run.py --config configs/experiments/lachlan_rf_baseline.yaml

# 运行 RF 正式训练实验（train_from_archive_features 模式）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml

# 运行 SPE 模型训练（带调优）
python run.py --config configs/experiments/lachlan_spe.yaml

# 运行 CNN 模型训练
python run.py --config configs/experiments/lachlan_cnn.yaml

# 运行测试
python -m pytest tests/ -q
```

---

## 3. 代码结构与目录说明

```
MPM_codex_phase2/
├── run.py                                    # ★ 统一入口文件
├── baseline_tools.py                         # 基线工件安全检查工具（无科学计算依赖）
│
├── configs/                                  # 配置文件目录
│   └── experiments/                          #   实验 YAML 配置
│       ├── lachlan_rf_baseline.yaml          #     RF 基线回放（Lachlan，Phase1 格式）
│       ├── lachlan_rf_phase2.yaml            #     RF + PUB + 调优（Lachlan，Phase2 格式）
│       ├── lachlan_cnn.yaml                  #     CNN 模型训练 + 调优
│       ├── lachlan_spe.yaml                  #     SPE 模型训练 + 调优
│       ├── lachlan_spe_notune.yaml           #     SPE 模型训练（无调优，固定参数）
│       └── nsw_rf_baseline.yaml              #     RF 基线回放（NSW 地区，Phase1 格式）
│
├── src/                                      # 框架源码
│   ├── core/                                 # ★ 框架核心（骨架层，扩展时不需要改）
│   │   ├── config.py                         #   配置加载/合并/校验/Phase1迁移/CLI覆盖
│   │   ├── experiment.py                     #   实验编排器（6阶段管线）
│   │   ├── registry.py                       #   ★ 通用组件注册表 ComponentRegistry
│   │   ├── bootstrap.py                      #   启动时自动导入所有内置组件触发注册
│   │   ├── contracts.py                      #   数据契约（TrainingData/SplitData）与接口协议
│   │   └── spec.py                           #   配置的类型化视图 ExperimentSpec
│   │
│   ├── models/                               # ★ 模型适配器（新增模型在此写）
│   │   ├── registry.py                       #   MODEL_REGISTRY + LABEL_REFINER_REGISTRY
│   │   ├── rf.py                             #   随机森林适配器
│   │   ├── spe.py                            #   Self-Paced Ensemble 适配器
│   │   ├── cnn.py                            #   1D CNN（PyTorch）适配器
│   │   └── pu.py                             #   PUB 标签细化器
│   │
│   ├── data/                                 # 数据加载与研究单元构建
│   │   ├── dataset.py                        #   归档数据集加载/校验/快照
│   │   ├── research_units.py                 #   构建正样本/无标签/预测网格单元
│   │   └── labels.py                         #   SIZE_CODE → 标签 + 样本权重映射
│   │
│   ├── features/                             # 特征预处理
│   │   └── preprocess.py                     #   基线预处理管线（相关性滤波→OneHot→标准化）
│   │
│   ├── operators/features/                   # 特征算子（GIS 特征提取）
│   │   ├── pipeline.py                       #   算子管线编排器
│   │   ├── builtins.py                       #   5个内置算子实现
│   │   ├── context.py                        #   GIS 算子访问与栅格文件扫描
│   │   └── registry.py                       #   算子注册表
│   │
│   ├── tasks/                                # 找矿任务定义
│   │   ├── registry.py                       #   TASK_REGISTRY
│   │   ├── target_area.py                    #   靶区预测任务（已完整实现）
│   │   └── deep_edge.py                      #   深边部预测任务（占位）
│   │
│   ├── tuning/                               # 超参数调优
│   │   ├── registry.py                       #   TUNER_REGISTRY
│   │   ├── bayes.py                          #   贝叶斯优化（基于 scikit-optimize）
│   │   └── none.py                           #   无调优，直接训练
│   │
│   ├── validation/                           # 验证与评估
│   │   ├── registry.py                       #   SPLITTER_REGISTRY + METRIC_REGISTRY
│   │   ├── splitters.py                      #   留出法 + 分层 K 折交叉验证
│   │   └── metrics.py                        #   6个评估指标
│   │
│   ├── knowledge/                            # 知识注入
│   │   ├── pipeline.py                       #   知识管线
│   │   ├── providers.py                      #   知识提供者（仅 empty 占位实现）
│   │   └── registry.py                       #   KNOWLEDGE_REGISTRY
│   │
│   ├── predicates/                           # 谓词约束
│   │   ├── pipeline.py                       #   谓词管线（顺序执行）
│   │   ├── builtins.py                       #   谓词实现（仅 identity 占位）
│   │   └── registry.py                       #   PREDICATE_REGISTRY
│   │
│   └── utils/                                # 工具函数
│       ├── logging.py                        #   日志配置
│       └── files.py                          #   SHA256/Git信息/YAML-JSON读写
│
├── outputs/                                  # 实验输出目录（.gitignore）
│   └── <实验名称>_<时间戳>/
│       ├── config_resolved.yaml              #   合并后的完整配置（含DEFAULTS+CLI覆盖）
│       ├── experiment.log                    #   结构化日志
│       ├── manifest.json                     #   可复现性清单（含Git commit/文件哈希）
│       ├── metrics.json                      #   评估指标
│       ├── models/                           #   序列化模型（.pkl）
│       ├── predictions/                      #   预测结果（CSV + GeoTIFF）
│       ├── figures/                          #   评估图表
│       └── intermediate/                     #   中间数据快照
│
├── scripts/                                  # 辅助脚本
│   ├── list_components.py                    #   列出所有已注册组件
│   ├── create_baseline_manifest.py           #   创建基线工件清单
│   ├── render_run_sh.py                      #   从配置生成 shell 命令
│   └── validate_reproduction.py              #   可复现性验证
│
├── tests/                                    # 测试
│   ├── test_framework.py                     #   框架核心测试
│   ├── test_spe_model.py                     #   SPE 模型单元测试
│   └── baseline/                             #   基线回归测试
│
└── docs/                                     # 文档
    ├── PHASE2_ARCHITECTURE_CN.md             #   架构说明
    └── PHASE2_USAGE_CN.md                    #   使用说明
```

### 分层架构

```
┌──────────────────────────────────────────────────────────┐
│  run.py                     ← 入口层（argparse + 调用）  │
├──────────────────────────────────────────────────────────┤
│  src/core/config.py         ← 配置层（加载/合并/校验）   │
│  src/core/experiment.py     ← 编排层（6阶段管线）        │
│  src/core/spec.py           ← 类型化视图                 │
├──────────────────────────────────────────────────────────┤
│  src/models/                ← 组件层（全部可插拔替换）    │
│  src/tuning/                                           │
│  src/validation/                                       │
│  src/tasks/                                            │
│  src/operators/features/                               │
│  src/predicates/                                       │
│  src/knowledge/                                        │
│  src/data/                                             │
│  src/features/                                         │
├──────────────────────────────────────────────────────────┤
│  src/core/registry.py       ← 基础设施层（注册表）       │
│  src/core/contracts.py      ← 数据契约                  │
│  src/utils/                 ← 工具层                    │
└──────────────────────────────────────────────────────────┘
```

**核心原则：扩展只需改组件层，入口层/编排层/配置层/基础设施层全部保持不动。**

---

## 4. 核心概念

### 4.1 统一入口

所有实验都通过 `run.py` 执行，通过 `--config` 参数指定配置文件：

```bash
python run.py --config configs/experiments/lachlan_rf_phase2.yaml
```

### 4.2 实验管线（6 个阶段）

`Experiment.run()` 按顺序执行以下阶段：

| 阶段 | 方法 | 说明 | 关键操作 |
|:----:|------|------|---------|
| ① | `prepare_data` | 加载数据 | `archive_replay`/`train_from_archive_features` 模式加载归档CSV；`raw_gis` 模式构建研究单元 |
| ② | `build_features` | 构建特征 | 快照归档特征，或运行配置的 GIS 特征算子管线 |
| ③ | `prepare_dataset` | 准备数据集 | 校验 schema，或执行相关性滤波→OneHot编码→标准化 |
| ④ | `train` | 训练模型 | 回放归档模型，或通过 Tuner 训练（可含超参数搜索） |
| ⑤ | `evaluate` | 评估指标 | 计算 accuracy/precision/recall/f1/roc_auc/confusion_matrix |
| ⑥ | `predict` | 预测导出 | 生成预测概率，导出 CSV + GeoTIFF |

### 4.3 三种执行模式

| 模式 | 适用场景 | 数据来源 | 模型来源 | 需要 GIS 依赖 |
|------|---------|---------|---------|:--:|
| `archive_replay` | 回归验证，确认框架能复现历史结果 | 归档 CSV | 归档 .pkl 文件直接复制 | 否 |
| `train_from_archive_features` | 快速模型迭代，换模型/换参数 | 归档 CSV（预计算特征） | 重新训练 | 否 |
| `raw_gis` | 完整管线，从原始 GIS 数据开始 | 原始 GIS 文件（.shp/.tif） | 重新训练 | 是 |

### 4.4 组件注册表机制

框架内所有可扩展组件均通过 `ComponentRegistry` 管理，按名称注册和引用。这是框架的核心扩展机制。

**注册表工作原理：**

```python
# 1. 定义注册表（框架已做）
METRIC_REGISTRY = ComponentRegistry("metric")

# 2. 注册组件（用装饰器）
@METRIC_REGISTRY.decorator("mcc")
def metric_mcc(labels, predictions, probabilities, sample_weight):
    return float(matthews_corrcoef(labels, predictions, sample_weight=sample_weight))

# 3. 在 YAML 中按名称引用
# validation:
#   metrics: [accuracy, precision, recall, f1, mcc]  ← "mcc" 自动匹配上面的注册名

# 4. 框架内部通过注册表查找和调用
metric_fn = METRIC_REGISTRY.get("mcc")
value = metric_fn(labels, predictions, probabilities, sample_weight)
```

**所有注册表一览：**

| 注册表变量 | 领域 | 已注册项 | 对应 YAML 路径 |
|-----------|------|---------|---------------|
| `MODEL_REGISTRY` | 模型 | `rf`, `spe`, `cnn` | `model.name` |
| `FEATURE_OPERATOR_REGISTRY` | 特征算子 | `raster_statistics`, `texture`, `elevation_gradient`, `line_distance`, `categorical_geology` | `features.operators[].name` |
| `TASK_REGISTRY` | 找矿任务 | `target_area_prediction`, `deep_edge_prediction`（占位） | `task.name` |
| `TUNER_REGISTRY` | 调优器 | `none`, `bayes` | `tuning.name` |
| `LABEL_REFINER_REGISTRY` | 标签细化 | `pub` | `label_refinement.name` |
| `SPLITTER_REGISTRY` | 数据分拆 | `random_holdout`, `stratified_kfold` | `validation.holdout.name`, `validation.cross_validation.name` |
| `METRIC_REGISTRY` | 评估指标 | `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `confusion_matrix` | `validation.metrics[]`, `validation.primary_metric` |
| `PREDICATE_REGISTRY` | 谓词约束 | `identity` | `predicates.items[].name` |
| `KNOWLEDGE_REGISTRY` | 知识注入 | `empty` | `knowledge.items[].name` |

### 4.5 配置加载流程

```
YAML 文件
    ↓ yaml.safe_load()
原始字典
    ↓ _migrate_phase1_config()     ← 如果是一期格式，自动迁移到二期格式
迁移后字典
    ↓ _deep_merge(DEFAULTS, migrated)  ← 合并默认值，用户配置覆盖默认值
完整字典
    ↓ _resolve_dataset_paths()     ← 相对路径 → 绝对路径
    ↓ validate_config()            ← 校验所有组件名在注册表中存在
    ↓ ExperimentConfig(values=resolved, source_path=...)
```

**`--set` CLI 覆盖的插入位置：**

```
load_config() 返回 ExperimentConfig
    ↓
apply_cli_overrides(config, ["model.name=spe", "tuning.params.n_iter=200"])
    ↓ ① 解析 "model.name=spe" → {"model.name": "spe"}
    ↓ ② _coerce_value: "200" → 200 (int)
    ↓ ③ _dot_to_nested: {"model.name": "spe"} → {"model": {"name": "spe"}}
    ↓ ④ _deep_merge(config.values, overrides_nested)
    ↓ ⑤ validate_config(merged)  ← 重新校验
    ↓ ⑥ 返回新的 ExperimentConfig
```

---

## 5. 基本使用方式

### 5.1 运行实验

```bash
# 完整运行一个实验
python run.py --config configs/experiments/lachlan_rf_phase2.yaml

# 运行 SPE 实验（无调优，固定参数）
python run.py --config configs/experiments/lachlan_spe_notune.yaml
```

### 5.2 验证配置（不访问数据文件）

```bash
# 快速检查配置是否正确
python run.py --config configs/experiments/lachlan_rf_phase2.yaml --validate-only

# 输出 JSON 格式的组件概览
# {
#   "experiment": "lachlan_rf_phase2_test",
#   "execution_mode": "archive_replay",
#   "task": "target_area_prediction",
#   "feature_operators": ["raster_statistics", "texture", ...],
#   "model": "rf",
#   "tuner": "bayes",
#   ...
# }
```

### 5.3 CLI 参数覆盖（`--set`）

**不修改 YAML 文件，直接在命令行覆盖任意配置项**。这是进行快速实验对比的核心功能。

```bash
# 单参数覆盖：换模型
python run.py --config configs/experiments/lachlan_rf_phase2.yaml --set model.name=spe

# 多参数覆盖：换模型 + 换调优器 + 调参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set model.name=spe tuning.name=none tuning.params.n_iter=200

# 覆盖调优参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.params.n_iter=500 tuning.params.search_space.max_depth.low=3

# 覆盖实验级参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set experiment.seed=999 experiment.name=my_custom_exp

# 覆盖验证参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set validation.holdout.params.test_size=0.3 validation.primary_metric=roc_auc

# 开关型参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set label_refinement.enabled=false prediction.export_geotiff=false

# 配合 validate-only 预览覆盖效果
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --validate-only --set model.name=spe tuning.name=none
```

**类型推断规则：**
- `"true"` / `"false"` → `True` / `False`
- `"123"` → `123`（int）
- `"0.5"` → `0.5`（float）
- 其他 → 保持字符串

**重要行为：**
- 原 YAML 文件不会被修改，覆盖只在内存中生效
- 覆盖后会重新走 `validate_config` 校验，无效覆盖在运行前被拦截
- `config_resolved.yaml` 会记录最终合并后的参数（包含 CLI 覆盖）

### 5.4 输出目录时间戳

**每次运行实验会自动在输出目录名后追加时间戳**，防止重复运行同一配置时覆盖前次结果。

```
# 配置中写的是: output_dir: outputs/lachlan_rf_phase2_test
# 实际创建的目录: outputs/lachlan_rf_phase2_test_20260901_143052/
#                                                      ↑
#                                               YYYYMMDD_HHMMSS
```

这意味着你可以连续多次运行同一个 YAML 配置（每次用不同 `--set` 参数），每次结果都会保存在独立的目录中。

### 5.5 查看所有已注册组件

```bash
python scripts/list_components.py
```

### 5.6 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -q

# 运行特定测试
python -m pytest tests/test_framework.py -q

# 运行特定测试方法
python -m pytest tests/test_framework.py::test_cli_overrides_merge_and_validate -v
```

---

## 6. 归档 CSV 数据格式

`train_from_archive_features` 和 `archive_replay` 模式直接读取归档 CSV，无需 GIS 提取。

### 6.1 核心 CSV 文件

| 文件 | 行数 | 用途 |
|------|:----:|------|
| `Xy_train.csv` | 479 | 全部训练数据（138 特征 + `sample_weight` + `label`） |
| `Xy_rf_train.csv` | 359 | 训练集（75%），用于模型训练 |
| `Xy_rf_test.csv` | 120 | 测试集（25%），用于模型评估 |
| `target_features.csv` | 4071 | 预测网格点的特征（无标签） |
| `target_coords_purged.csv` | 4071 | 预测网格点的 X/Y 坐标 |
| `target_mask.csv` | 4200 | 完整研究区网格掩膜（含研究区内/外标记） |
| `Xy_train_new.csv` | 479 | 训练数据副本（PUB 兼容用） |

### 6.2 训练数据列结构

每一行是一个**采样点**（研究单元）。`Xy_train.csv` 的列结构：

```
138 个特征列（磁法统计、重力统计、纹理、高程梯度、线距离、地质分类）
    +
sample_weight（样本权重：VLG 大矿 0.8，OCC 小矿 0.1，unlabeled 背景点 0.5）
    +
label（标签：1=已知矿点，0=背景点）
```

### 6.3 预测文件关联方式

`target_features.csv`、`target_coords_purged.csv`、`target_mask.csv` 三个文件通过**行号顺序**关联：

```
target_features.csv（4071行）    target_coords_purged.csv（4071行）
    第 1 行  ← 行号一一对应 →  X=147.05, Y=-35.45
    第 2 行  ← 行号一一对应 →  X=147.10, Y=-35.45
    ...

target_mask.csv（4200行，完整网格，含研究区内/外）
    0=研究区外, 1=研究区内（共4071个有效点）

重建 2D 网格时：
  ① 创建 4200 个空位，全填 NaN
  ② 找到所有标记为 1 的位置（研究区内，共 4071 个）
  ③ 把预测概率按行号顺序填进去
  ④ 按 X/Y 唯一值 reshape 成 2D 矩阵 → 画图/导出 GeoTIFF
```

---

## 7. 模型适配器原理

### 7.1 两个层次

框架通过 **适配器（Adapter）** 和 **模型实例（Model）** 两层分离来统一操作所有模型：

```
适配器（Adapter）— 2 个方法（工厂，负责"造模型"）
    ├── build(params, seed)       → 返回一个模型实例
    └── fit_params(data)          → 返回训练参数（如 sample_weight）

模型实例（Model）— 3 个方法（由 build() 返回的对象，负责实际训练和预测）
    ├── fit(X, y, **kwargs)       → 训练
    ├── predict(X)                → 预测类别
    └── predict_proba(X)          → 预测概率
```

**适配器是"工厂"，模型是"产品"。** 框架内部调用流程：

```python
model = model_adapter.build(params, seed)          # ① 适配器造模型
model.fit(X, y, **model_adapter.fit_params(data))  # ② 模型训练
model.predict(X)                                   # ③ 预测类别
model.predict_proba(X)                             # ④ 预测概率
```

### 7.2 三种适配策略

| 模型情况 | 策略 | 代码量 | 例子 |
|---------|------|:------:|------|
| 本身就是 sklearn 模型 | 适配器直接返回，`build()` 只做参数传递 | ~30行 | RF 适配器 |
| 自实现算法，无 sklearn 接口 | 写包装类继承 `BaseEstimator` + `ClassifierMixin`，手动实现 `fit/predict/predict_proba` | ~100行 | SPE 适配器 |
| 底层库完全不同（如 PyTorch） | 写包装类，`fit()` 里写训练循环，`predict_proba()` 里写前向推理 | ~200行 | CNN 适配器 |

### 7.3 四种特殊场景适配

| 变更程度 | 场景 | 需要改什么 | 不需要改 |
|---------|------|-----------|---------|
| **最小** | 换另一个 sklearn 模型 | 写 `ModelAdapter` | 框架、数据管线、Task |
| **中等** | 模型无 sklearn 接口 | 写 `ModelAdapter` + 包装类 | 框架、数据管线 |
| **较大** | 输入格式不同（如图像） | 加 `FeatureOperator` + `ModelAdapter` | 框架、Experiment |
| **最大** | 完全不同的找矿方法 | 加 `Task` + 全套 | 框架 core |

---

## 8. 如何扩展组件

### 8.1 新增模型

完整步骤参见下方。以 XGBoost 为例：

**第一步：编写适配器** — `src/models/xgb.py`

```python
"""XGBoost 模型适配器"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.contracts import OptionalDependencyError, TrainingData
from .registry import MODEL_REGISTRY

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None
    _XGB_AVAILABLE = False


@MODEL_REGISTRY.decorator("xgb")  # ← 注册名，YAML 中通过 model.name 引用
class XGBoostAdapter:
    name = "xgb"
    artifact_filename = "model_xgb.pkl"
    supports_constraints = False

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if not _XGB_AVAILABLE:
            raise OptionalDependencyError(
                "XGBoost model requires xgboost. Install: pip install xgboost"
            )
        xgb.XGBClassifier(**dict(params))

    def build(self, params: Mapping[str, Any], seed: int) -> Any:
        resolved = dict(params)
        resolved.setdefault("random_state", seed)
        resolved.setdefault("n_jobs", -1)
        resolved.setdefault("verbosity", 0)
        return xgb.XGBClassifier(**resolved)

    def fit_params(self, data: TrainingData) -> Mapping[str, Any]:
        return {"sample_weight": data.sample_weight}
```

**第二步：注册到引导加载** — 编辑 `src/core/bootstrap.py`

```python
def load_builtin_components() -> None:
    # ... 在已有的 import 后面加一行
    from ..models import xgb as _xgb    # ← 新增
```

**第三步：编写 YAML 配置** — `configs/experiments/lachlan_xgb.yaml`

```yaml
experiment:
  name: lachlan_xgb
  seed: 2026
  output_dir: outputs/lachlan_xgb
  execution_mode: train_from_archive_features
  variant: baseline

model:
  name: xgb                        # ← 对应 @MODEL_REGISTRY.decorator("xgb")
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8

tuning:
  name: bayes
  params:
    n_iter: 100
    search_space:
      n_estimators:     { type: integer, low: 50, high: 500 }
      max_depth:        { type: integer, low: 3, high: 15 }
      learning_rate:    { type: real, low: 0.01, high: 0.3, prior: log-uniform }
# ... 其余配置节（dataset/research_unit/label/features/validation 等）复用已有配置
```

**第四步：运行**

```bash
# 验证配置
python run.py --config configs/experiments/lachlan_xgb.yaml --validate-only

# 运行实验
python run.py --config configs/experiments/lachlan_xgb.yaml
```

### 8.2 新增验证指标

**第一步：在 `src/validation/metrics.py` 中注册新指标**

```python
from sklearn.metrics import matthews_corrcoef

@_register("mcc")
def metric_mcc(labels, predictions, probabilities, sample_weight):
    return float(matthews_corrcoef(labels, predictions, sample_weight=sample_weight))
```

**第二步：在 YAML 配置中引用**

```yaml
validation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc
    - confusion_matrix
    - mcc                    # ← 新增
  primary_metric: f1
```

**不需要改任何其他文件。** 指标函数签名固定为 `(labels, predictions, probabilities, sample_weight)`，框架自动调用。

**其他常用指标示例：**

```python
# Balanced Accuracy（平衡准确率，适合不平衡数据）
from sklearn.metrics import balanced_accuracy_score

@_register("balanced_accuracy")
def metric_balanced_accuracy(labels, predictions, probabilities, sample_weight):
    return float(balanced_accuracy_score(labels, predictions, sample_weight=sample_weight))

# Average Precision（平均精度，适合不平衡数据）
from sklearn.metrics import average_precision_score

@_register("average_precision")
def metric_average_precision(labels, predictions, probabilities, sample_weight):
    return float(average_precision_score(labels, probabilities, sample_weight=sample_weight))

# Cohen's Kappa
from sklearn.metrics import cohen_kappa_score

@_register("kappa")
def metric_kappa(labels, predictions, probabilities, sample_weight):
    return float(cohen_kappa_score(labels, predictions, sample_weight=sample_weight))
```

### 8.3 新增交叉验证方法

**第一步：在 `src/validation/splitters.py` 中注册新分拆器**

```python
from sklearn.model_selection import RepeatedStratifiedKFold

@SPLITTER_REGISTRY.decorator("repeated_stratified_kfold")
class RepeatedStratifiedKFoldSplitter:
    kind = "cross_validation"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        if int(params.get("n_splits", 5)) < 2:
            raise ValueError("n_splits must be at least 2")
        if int(params.get("n_repeats", 10)) < 1:
            raise ValueError("n_repeats must be at least 1")

    def build_cv(self, params: Mapping[str, Any], seed: int, data=None):
        return RepeatedStratifiedKFold(
            n_splits=int(params.get("n_splits", 5)),
            n_repeats=int(params.get("n_repeats", 10)),
            random_state=seed,
        )
```

**第二步：在 YAML 中引用**

```yaml
validation:
  cross_validation:
    name: repeated_stratified_kfold    # ← 新分拆器
    params:
      n_splits: 5
      n_repeats: 10
```

**新增留出法分拆器示例（空间分拆）：**

```python
@SPLITTER_REGISTRY.decorator("spatial_holdout")
class SpatialHoldoutSplitter:
    kind = "holdout"

    @staticmethod
    def validate_config(params: Mapping[str, Any]) -> None:
        test_size = float(params.get("test_size", 0.25))
        if not 0 < test_size < 1:
            raise ValueError("test_size must be in (0, 1)")

    def split(self, data: TrainingData, params: Mapping[str, Any], seed: int) -> SplitData:
        # 基于空间坐标的分拆逻辑（需要 units 在 metadata 中）
        units = data.metadata.get("units")
        if units is None:
            raise ValueError("spatial_holdout requires units in metadata")
        # ... 实现空间分拆逻辑 ...
        return SplitData(train_data, test_data)
```

### 8.4 新增研究单元构建方式

研究单元的构建由 `Task` 负责。当前 `target_area_prediction` Task 在 `src/tasks/target_area.py` 中实现了三种单元构建方法：

- `build_positive_units()` — 从矿点 shapefile 构建正样本
- `build_unlabeled_units()` — 从边界内随机采样构建无标签样本
- `build_prediction_units()` — 构建预测网格

**方式一：修改正样本构建逻辑**（在 `src/data/research_units.py` 中修改 `build_positive_units()`）

```python
def build_positive_units(occurrence_path: str, label_config: Mapping[str, Any]) -> pd.DataFrame:
    occurrences = load_vector(occurrence_path)
    # 例：只用 SIZE_CODE 为大矿的矿点
    if label_config.get("filter_large_only", False):
        occurrences = occurrences[occurrences["SIZE_CODE"].isin(["VLG", "LGE"])]
    labelled = positive_labels_from_size_code(occurrences, label_config)
    return pd.DataFrame({
        "X": labelled.geometry.x,
        "Y": labelled.geometry.y,
        "label": labelled["label"],
        "sample_weight": labelled["sample_weight"],
    }).reset_index(drop=True)
```

**方式二：修改无标签样本采样策略**（在 `src/data/research_units.py` 中修改 `sample_unlabeled_units()`）

```python
def sample_unlabeled_units(boundary_path: str, count: int,
                           label_config: Mapping[str, Any]) -> pd.DataFrame:
    # 例：根据 label_config 调整正负样本比例
    ratio = label_config.get("unlabeled_ratio", 1.0)
    actual_count = int(count * ratio)
    # ... 采样逻辑 ...
```

**方式三：在 YAML 中通过 `research_unit` 配置控制行为**

```yaml
research_unit:
  type: point_local_environment
  train_positive: occurrence_points
  train_unlabeled: random_points_in_nsw_boundary
  prediction_grid_size: 0.05      # 调整预测网格精度
  # 可扩展的自定义字段（需要 Task 代码支持读取）
  unlabeled_ratio: 1.5
  buffer_distance: 500            # 矿点缓冲区距离（米）
```

### 8.5 新增标签构造方式

标签策略的核心在 `src/data/labels.py`，当前实现为 `SIZE_CODE → sample_weight` 映射。

**修改权重映射：** 直接在 YAML 中调整 `label.sample_weight` 即可，无需改代码：

```yaml
label:
  strategy: positive_unlabeled_as_zero
  positive_value: 1
  unlabeled_value: 0
  sample_weight:
    VLG: 0.8       # 调整大矿权重
    LGE: 0.4
    MED: 0.3
    SML: 0.2
    OCC: 0.1
    unlabeled: 0.5
```

**新增标签策略：** 在 `src/data/labels.py` 中添加新的标签函数：

```python
def binary_labels_from_mineral_type(
    occurrences: pd.DataFrame,
    label_config: Mapping[str, Any],
) -> pd.DataFrame:
    """根据矿种类型（COMMODITY 字段）进行二分类标签。"""
    if "COMMODITY" not in occurrences.columns:
        raise ValueError("Occurrence data must include COMMODITY")
    result = occurrences.copy()
    target_commodity = label_config.get("target_commodity", "Cu")
    result["label"] = (result["COMMODITY"] == target_commodity).astype(int)
    result["sample_weight"] = 0.5  # 统一权重
    return result
```

然后在 `src/tasks/target_area.py` 的 `build_positive_units()` 中根据 `label_config["strategy"]` 选择不同的标签函数。

### 8.6 新增谓词（Predicate）

谓词是在训练前对 `TrainingData` 进行变换的组件，要求**不改变行数**。

**第一步：在 `src/predicates/builtins.py` 中注册**

```python
@PREDICATE_REGISTRY.decorator("positive_constraint")
class PositiveConstraintPredicate:
    """强制要求某个地质分类特征必须为正（即该特征=1），
    不符合约束的样本被标记为负类。"""

    name = "positive_constraint"

    def __init__(self, params: Mapping[str, Any]):
        self.feature = str(params["feature"])
        self.params = dict(params)

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        import numpy as np
        if self.feature in data.features.columns:
            mask = data.features[self.feature] == 1
            new_labels = data.labels.copy()
            new_labels[~mask] = 0  # 不满足约束的样本 → 标记为负类
            result = data.with_labels(
                new_labels,
                predicate_applied=self.name,
                constrained_feature=self.feature,
            )
            return TrainingData(
                result.features, result.labels, result.sample_weight,
                constraints={**data.constraints, self.feature: "positive_required"},
                metadata=result.metadata,
            )
        return data
```

**第二步：在 YAML 中启用**

```yaml
predicates:
  enabled: true
  combine: sequential
  items:
    - name: positive_constraint
      params:
        feature: Intrusions_Tabberabberan   # 要求侵入岩特征必须为正
```

**另一个谓词示例（样本加权调整）：**

```python
@PREDICATE_REGISTRY.decorator("weight_adjustment")
class WeightAdjustmentPredicate:
    """根据知识上下文调整样本权重。"""

    name = "weight_adjustment"

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        knowledge = context.get("knowledge", {})
        multiplier = float(self.params.get("multiplier", 2.0))
        favorable_ids = knowledge.get("favorable_indices", [])

        new_weights = data.sample_weight.copy()
        for idx in favorable_ids:
            if idx < len(new_weights):
                new_weights.iloc[idx] *= multiplier

        return TrainingData(
            data.features, data.labels, new_weights,
            constraints=dict(data.constraints),
            metadata={**data.metadata, "weight_adjusted": True},
        )
```

### 8.7 新增知识注入（Knowledge Provider）

知识注入在谓词之前执行，用于**从外部知识源构建知识工件**（如地质图、已知矿化带、专家标注区域），供谓词或模型使用。

**第一步：在 `src/knowledge/providers.py` 中注册**

```python
@KNOWLEDGE_REGISTRY.decorator("favorable_zone")
class FavorableZoneKnowledgeProvider:
    """从 shapefile 加载已知有利成矿区域，识别出位于有利区内的训练样本索引。"""

    name = "favorable_zone"

    def __init__(self, params: Mapping[str, Any]):
        self.zone_path = str(params["zone_path"])
        self.params = dict(params)

    def build(self, data: TrainingData, context: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            import geopandas as gpd
        except ImportError:
            return {"favorable_indices": [], "zone_path": self.zone_path}

        units = data.metadata.get("units")
        if units is None:
            return {"favorable_indices": [], "zone_path": self.zone_path}

        zones = gpd.read_file(self.zone_path)
        from shapely.geometry import Point
        favorable = []
        for i, (_, row) in enumerate(units.iterrows()):
            point = Point(row["X"], row["Y"])
            if any(zone.contains(point) for zone in zones.geometry):
                favorable.append(i)

        return {
            "favorable_indices": favorable,
            "zone_path": self.zone_path,
            "count": len(favorable),
        }
```

**第二步：在 YAML 中启用**

```yaml
knowledge:
  enabled: true
  items:
    - name: favorable_zone
      params:
        zone_path: ../path/to/favorable_zones.shp
```

### 8.8 知识注入 vs 谓词约束：区别是什么

这是一个容易混淆的概念，这里重点说明：

| 维度 | 知识注入（Knowledge） | 谓词约束（Predicate） |
|------|---------------------|---------------------|
| **执行顺序** | 先执行（在谓词之前） | 后执行（在知识之后） |
| **目的** | **读取外部知识，构建知识工件**（如索引列表、距离场、掩膜） | **使用知识工件，对训练数据做变换**（如筛选、加权、约束） |
| **输入/输出** | 输入 `TrainingData` + `context`，输出 `dict`（知识工件） | 输入 `TrainingData` + `context`（含知识工件），输出 `TrainingData` |
| **是否改变数据** | **不改变** `TrainingData` | **改变** `TrainingData`（标签、权重、约束等） |
| **典型场景** | 加载地质图、计算矿化带距离、查询已知矿区 | 筛选有利区、调整权重、添加约束条件 |
| **数据流** | `Knowledge.build()` → `context["knowledge"]` | `Predicate.apply(data, context["knowledge"])` |

**数据流示意：**

```
TrainingData
    │
    ├── KnowledgePipeline.build(data, context)
    │   ├── FavorableZoneKnowledgeProvider.build()  → {"favorable_indices": [0,3,5]}
    │   └── 结果存入 context["knowledge"]
    │
    └── PredicatePipeline.apply(data, context)
        └── WeightAdjustmentPredicate.apply(data, context)
            └── 读取 context["knowledge"]["favorable_indices"] → 调整权重
```

**设计原则：知识提供者负责"知道什么"，谓词负责"怎么用"。** 同一个知识工件可以被多个谓词以不同方式使用。

### 8.9 新增其他组件总结

| 扩展类型 | 在哪个文件写代码 | 注册装饰器 | 是否需要改 bootstrap.py | 是否需要改其他文件 |
|---------|----------------|-----------|:--:|:--:|
| 新模型 | `src/models/<name>.py` | `@MODEL_REGISTRY.decorator("name")` | 是 | 否 |
| 新指标 | `src/validation/metrics.py` | `@_register("name")` | 否 | 否 |
| 新分拆器 | `src/validation/splitters.py` | `@SPLITTER_REGISTRY.decorator("name")` | 否 | 否 |
| 新特征算子 | `src/operators/features/builtins.py` | `@FEATURE_OPERATOR_REGISTRY.decorator("name")` | 否 | 否 |
| 新任务 | `src/tasks/<name>.py` | `@TASK_REGISTRY.decorator("name")` | 是 | 否 |
| 新调优器 | `src/tuning/<name>.py` | `@TUNER_REGISTRY.decorator("name")` | 是 | 否 |
| 新谓词 | `src/predicates/builtins.py` | `@PREDICATE_REGISTRY.decorator("name")` | 否 | 否 |
| 新知识提供者 | `src/knowledge/providers.py` | `@KNOWLEDGE_REGISTRY.decorator("name")` | 否 | 否 |
| 新标签细化器 | `src/models/<name>.py` | `@LABEL_REFINER_REGISTRY.decorator("name")` | 是 | 否 |

---

## 9. 实验对比指南

以下是完整的实验对比类型及对应的 shell 命令示例。假设基准配置为 `lachlan_rf_phase2.yaml`。

### 9.1 不同模型对比（固定参数，无调优）

**目的：** 比较不同模型在相同数据、相同参数下的性能差异。

```bash
# RF 固定参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none experiment.name=compare_rf_fixed

# SPE 固定参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set model.name=spe tuning.name=none experiment.name=compare_spe_fixed

# CNN 固定参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set model.name=cnn tuning.name=none experiment.name=compare_cnn_fixed
```

### 9.2 不同模型对比（带调优）

**目的：** 比较不同模型在各自最优参数下的性能上限。

```bash
# RF + 贝叶斯调优
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=bayes tuning.params.n_iter=200 experiment.name=compare_rf_tuned

# SPE + 贝叶斯调优
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set model.name=spe tuning.name=bayes tuning.params.n_iter=200 \
    experiment.name=compare_spe_tuned

# CNN + 贝叶斯调优
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set model.name=cnn tuning.name=bayes tuning.params.n_iter=100 \
    experiment.name=compare_cnn_tuned
```

### 9.3 相同模型多种子点实验

**目的：** 评估模型的随机稳定性，报告均值±标准差。

```bash
# 用不同随机种子重复运行
for seed in 42 123 456 789 2026; do
    python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
        --set tuning.name=none experiment.seed=$seed \
        experiment.name="rf_seed_${seed}"
done
```

### 9.4 相同模型不同研究单元实验

**目的：** 比较不同研究单元构建方式对预测结果的影响。

```bash
# 默认网格精度 0.05
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none research_unit.prediction_grid_size=0.05 \
    experiment.name=ru_grid_005

# 粗网格 0.1
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none research_unit.prediction_grid_size=0.1 \
    experiment.name=ru_grid_010

# 细网格 0.025
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none research_unit.prediction_grid_size=0.025 \
    experiment.name=ru_grid_0025
```

### 9.5 相同模型不同标签策略实验

**目的：** 比较不同标签权重设置对模型的影响。

```bash
# 默认权重（VLG=0.5, LGE=0.4, MED=0.3, SML=0.2, OCC=0.1, unlabeled=0.5）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none experiment.name=label_default

# 高权重（大矿权重更高，强调大矿）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none \
    label.sample_weight.VLG=0.8 \
    label.sample_weight.LGE=0.6 \
    label.sample_weight.MED=0.4 \
    experiment.name=label_high_weight

# 均匀权重（不区分矿点大小）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none \
    label.sample_weight.VLG=0.5 \
    label.sample_weight.LGE=0.5 \
    label.sample_weight.MED=0.5 \
    label.sample_weight.SML=0.5 \
    label.sample_weight.OCC=0.5 \
    experiment.name=label_uniform
```

### 9.6 相同模型不同谓词对比

**目的：** 比较不同谓词约束对预测结果的影响。

```bash
# 无谓词（基线）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none predicates.enabled=false experiment.name=pred_none

# 使用 identity 谓词（验证链路）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none predicates.enabled=true \
    predicates.items.0.name=identity \
    experiment.name=pred_identity

# 使用自定义谓词（需要先注册）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none predicates.enabled=true \
    predicates.items.0.name=positive_constraint \
    predicates.items.0.params.feature=Intrusions_Tabberabberan \
    experiment.name=pred_intrusion_constraint
```

### 9.7 不同知识注入对比

**目的：** 比较不同知识来源对模型的增强效果。

```bash
# 无知识注入（基线）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none knowledge.enabled=false experiment.name=knowledge_none

# 使用 empty 知识提供器（验证链路）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none knowledge.enabled=true \
    knowledge.items.0.name=empty \
    experiment.name=knowledge_empty

# 知识 + 谓词联合使用（知识识别有利区，谓词调整权重）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none \
    knowledge.enabled=true \
    knowledge.items.0.name=favorable_zone \
    predicates.enabled=true \
    predicates.items.0.name=weight_adjustment \
    experiment.name=knowledge_pred_combined
```

### 9.8 不同知识使用方式对比

**目的：** 比较同一知识的不同使用方式。

```bash
# 方式一：知识→谓词（筛选样本）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none \
    knowledge.enabled=true knowledge.items.0.name=favorable_zone \
    predicates.enabled=true predicates.items.0.name=positive_constraint \
    experiment.name=knowledge_use_filter

# 方式二：知识→谓词（加权调整）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none \
    knowledge.enabled=true knowledge.items.0.name=favorable_zone \
    predicates.enabled=true predicates.items.0.name=weight_adjustment \
    experiment.name=knowledge_use_weight

# 方式三：知识→模型约束（通过 constraints）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none \
    knowledge.enabled=true knowledge.items.0.name=favorable_zone \
    predicates.enabled=true predicates.items.0.name=positive_constraint \
    experiment.name=knowledge_use_constraint
```

### 9.9 相同模型不同特征算子实验

**目的：** 比较不同特征集对预测性能的影响。

```bash
# 全部特征（基线）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none experiment.name=feat_all

# 仅地质+线距离特征
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none \
    'features.operators=[{"name":"line_distance","params":{"distance_type":"geodesic"}},{"name":"categorical_geology","params":{}}]' \
    experiment.name=feat_geology_only

# 仅地球物理特征
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none \
    'features.operators=[{"name":"raster_statistics","params":{"buffer_size":10,"buffer_shape":"square"}},{"name":"texture","params":{"buffer_size":10}},{"name":"elevation_gradient","params":{"buffer_size":10,"buffer_shape":"square"}}]' \
    experiment.name=feat_geophysics_only
```

### 9.10 相同模型不同预处理参数实验

**目的：** 比较不同预处理参数的影响。

```bash
# 默认相关性阈值 0.7
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none preprocess.correlation_threshold=0.7 \
    experiment.name=prep_corr_07

# 更严格的相关性阈值 0.5
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none preprocess.correlation_threshold=0.5 \
    experiment.name=prep_corr_05

# 更宽松的相关性阈值 0.9
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none preprocess.correlation_threshold=0.9 \
    experiment.name=prep_corr_09
```

### 9.11 相同模型不同验证策略实验

**目的：** 比较不同验证方式对评估结果的影响。

```bash
# 默认：random_holdout test_size=0.25 + stratified_kfold n_splits=10
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none experiment.name=val_default

# 更小的测试集
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none validation.holdout.params.test_size=0.2 \
    experiment.name=val_testsize_02

# 更大的测试集
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none validation.holdout.params.test_size=0.33 \
    experiment.name=val_testsize_033

# 不同的 K 折
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none validation.cross_validation.params.n_splits=5 \
    experiment.name=val_kfold_5

# 不同的主要指标
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none validation.primary_metric=roc_auc \
    experiment.name=val_metric_auc
```

### 9.12 相同模型不同标签细化（PUB）实验

**目的：** 比较是否使用 PUB 标签细化对模型性能的影响。

```bash
# 启用 PUB（标签细化）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none label_refinement.enabled=true \
    experiment.name=pub_enabled

# 禁用 PUB（基线）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none label_refinement.enabled=false \
    experiment.name=pub_disabled
```

### 9.13 不同地区对比实验

**目的：** 比较同一模型在不同研究区的表现。

```bash
# Lachlan 地区
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none experiment.name=region_lachlan

# NSW 地区（需要对应的 NSW 配置文件）
python run.py --config configs/experiments/nsw_rf_baseline.yaml \
    --set tuning.name=none experiment.name=region_nsw
```

### 9.14 消融实验（Ablation Study）

**目的：** 逐一移除组件，确定每个组件对最终性能的贡献。

```bash
# 完整配置（基线）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set experiment.name=ablation_full

# 移除 PUB 标签细化
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set label_refinement.enabled=false experiment.name=ablation_no_pub

# 移除调优（固定参数）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none experiment.name=ablation_no_tune

# 移除 PUB + 移除调优
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set label_refinement.enabled=false tuning.name=none \
    experiment.name=ablation_no_pub_no_tune
```

### 9.15 参数敏感性分析

**目的：** 分析单个参数对模型性能的影响。

```bash
# 分析 n_estimators 的影响
for n in 10 50 100 200 500; do
    python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
        --set tuning.name=none model.params.n_estimators=$n \
        experiment.name="sensitivity_n_est_${n}"
done

# 分析 max_depth 的影响
for d in 3 5 10 15 20 30; do
    python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
        --set tuning.name=none model.params.max_depth=$d \
        experiment.name="sensitivity_depth_${d}"
done
```

### 9.16 完整批处理实验脚本模板

```bash
#!/bin/bash
# batch_experiments.sh — 批量运行实验对比

BASE_CONFIG="configs/experiments/lachlan_rf_phase2.yaml"
BASE_SET="tuning.name=none label_refinement.enabled=false"

echo "=== 1. 不同模型对比 ==="
for model in rf spe cnn; do
    echo "Running model=$model..."
    python run.py --config "$BASE_CONFIG" \
        --set $BASE_SET model.name=$model \
        experiment.name="batch_model_${model}"
done

echo "=== 2. 不同随机种子 ==="
for seed in 42 123 456 789 2026; do
    echo "Running seed=$seed..."
    python run.py --config "$BASE_CONFIG" \
        --set $BASE_SET model.name=rf experiment.seed=$seed \
        experiment.name="batch_seed_${seed}"
done

echo "=== 3. 不同特征组合 ==="
echo "Running all features..."
python run.py --config "$BASE_CONFIG" \
    --set $BASE_SET model.name=rf experiment.name="batch_feat_all"

echo "Running geology only..."
python run.py --config "$BASE_CONFIG" \
    --set $BASE_SET model.name=rf \
    'features.operators=[{"name":"line_distance","params":{"distance_type":"geodesic"}},{"name":"categorical_geology","params":{}}]' \
    experiment.name="batch_feat_geology"

echo "=== 4. 不同标签权重 ==="
echo "Running default weights..."
python run.py --config "$BASE_CONFIG" \
    --set $BASE_SET model.name=rf experiment.name="batch_label_default"

echo "Running uniform weights..."
python run.py --config "$BASE_CONFIG" \
    --set $BASE_SET model.name=rf \
    label.sample_weight.VLG=0.5 label.sample_weight.LGE=0.5 \
    label.sample_weight.MED=0.5 label.sample_weight.SML=0.5 \
    label.sample_weight.OCC=0.5 \
    experiment.name="batch_label_uniform"

echo "=== All experiments complete ==="
echo "Results are in outputs/ directory"
```

### 9.17 实验对比类型总览

| 编号 | 对比类型 | 变动的 YAML 配置节 | 示例命令节 |
|:----:|---------|-------------------|:--:|
| 1 | 不同模型对比（固定参数） | `model.name` | 9.1 |
| 2 | 不同模型对比（带调优） | `model.name` + `tuning.*` | 9.2 |
| 3 | 相同模型多种子 | `experiment.seed` | 9.3 |
| 4 | 不同研究单元 | `research_unit.*` | 9.4 |
| 5 | 不同标签策略 | `label.sample_weight.*` | 9.5 |
| 6 | 不同谓词约束 | `predicates.*` | 9.6 |
| 7 | 不同知识注入 | `knowledge.*` | 9.7 |
| 8 | 知识+谓词不同组合方式 | `knowledge.*` + `predicates.*` | 9.8 |
| 9 | 不同特征集 | `features.operators[]` | 9.9 |
| 10 | 不同预处理参数 | `preprocess.*` | 9.10 |
| 11 | 不同验证策略 | `validation.*` | 9.11 |
| 12 | 不同标签细化 | `label_refinement.*` | 9.12 |
| 13 | 不同地区 | `dataset.*` + `task.region` | 9.13 |
| 14 | 消融实验 | 多配置节组合 | 9.14 |
| 15 | 参数敏感性 | `model.params.*` | 9.15 |

---

## 10. YAML 配置参考

完整 YAML 结构说明（以 `lachlan_rf_phase2.yaml` 为例）：

| 配置节 | 功能 | 关键字段 | 是否必须 |
|--------|------|---------|:--:|
| `experiment` | 实验元信息 | `name`, `seed`, `output_dir`, `execution_mode`, `variant` | 是 |
| `task` | 找矿任务 | `name`（`target_area_prediction` 或 `deep_edge_prediction`）, `region` | 是 |
| `dataset` | 数据路径 | `root`, `archive_dir`, `occurrence`, `boundary`, `training_boundary`, `geology`, `magnetic`, `gravity`, `radiometric`, `remote_sensing`, `elevation`, `seismic` | 是 |
| `research_unit` | 研究单元参数 | `type`, `train_positive`, `train_unlabeled`, `prediction_grid_size` | 是 |
| `label` | 标签策略 | `strategy`, `positive_value`, `unlabeled_value`, `sample_weight`（SIZE_CODE 映射） | 是 |
| `features.operators` | 特征算子列表 | 每个算子有 `name` + `params` | 是 |
| `preprocess` | 预处理参数 | `correlation_threshold`, `categorical_encoding`, `scaling` | 是 |
| `model` | 模型配置 | `name`（注册名）, `params`（模型参数） | 是 |
| `tuning` | 超参数调优 | `name`（`none`/`bayes`）, `params.n_iter`, `params.search_space` | 是 |
| `validation` | 验证方式 | `holdout`, `cross_validation`, `metrics[]`, `primary_metric` | 是 |
| `prediction` | 预测输出 | `score_type`, `normalization`, `export_geotiff` | 是 |
| `label_refinement` | PUB 标签细化 | `enabled`, `name`, `params` | 否 |
| `knowledge` | 知识注入 | `enabled`, `items[]` | 否 |
| `predicates` | 谓词约束 | `enabled`, `combine`, `items[]` | 否 |
| `plugins` | 外部插件模块 | 模块名列表 | 否 |

### 搜索空间配置语法

贝叶斯调优和 PUB 标签细化都使用搜索空间配置：

```yaml
# 分类参数（从列表中选择）
bootstrap:
  type: categorical
  values: [true, false]

# 整数参数（均匀分布）
max_depth:
  type: integer
  low: 5
  high: 20

# 实数参数（对数均匀分布，适合学习率等）
learning_rate:
  type: real
  low: 0.0001
  high: 0.01
  prior: log-uniform
```

---

## 11. 输出目录结构

每次实验运行后，`outputs/<实验名称>_<时间戳>/` 目录包含：

```
lachlan_rf_phase2_test_20260901_143052/
├── config_resolved.yaml    # ★ 合并后的完整配置（含 DEFAULTS + Phase1迁移 + CLI覆盖）
├── experiment.log          # 结构化日志（包含每个阶段的耗时）
├── manifest.json           # 可复现性清单
│   ├── timestamp           #   运行时间
│   ├── git_commit          #   Git commit hash
│   ├── config              #   完整配置
│   ├── components          #   使用的组件列表
│   ├── input_paths         #   输入文件路径 + SHA256 哈希
│   ├── feature_schema      #   特征列名列表
│   └── output_files        #   输出文件列表 + SHA256 哈希
├── metrics.json            # 评估指标
├── models/
│   ├── model_rf.pkl        # 序列化模型（或 model_spe.pkl / model_cnn.pkl 等）
│   └── model_pub.pkl       # PUB 标签细化模型（如果启用）
├── predictions/
│   ├── target_probs.csv    # 预测概率表（X, Y, prob）
│   └── probability_map.tif # 概率 GeoTIFF（如果 export_geotiff=true）
├── figures/                # 评估图表
└── intermediate/           # 中间数据快照
    ├── Xy_train.csv
    ├── Xy_rf_train.csv
    ├── Xy_rf_test.csv
    ├── target_features.csv
    ├── target_coords_purged.csv
    └── target_mask.csv
```

### 如何对比两次实验结果

```bash
# 比较不同实验的指标
cat outputs/experiment_a_*/metrics.json | python -m json.tool
cat outputs/experiment_b_*/metrics.json | python -m json.tool

# 比较 config_resolved.yaml 确认参数差异
diff outputs/experiment_a_*/config_resolved.yaml outputs/experiment_b_*/config_resolved.yaml

# 比较 manifest.json 确认组件差异
diff <(cat outputs/experiment_a_*/manifest.json | python -c "import sys,json;c=json.load(sys.stdin);print(json.dumps(c['components'],indent=2))") \
     <(cat outputs/experiment_b_*/manifest.json | python -c "import sys,json;c=json.load(sys.stdin);print(json.dumps(c['components'],indent=2))")
```

---

## 12. 常见问题排查

### 配置加载失败

```bash
# 错误：Unknown model component 'xxx'
# 原因：YAML 中 model.name 写错了，或者组件未注册
# 解决：查看可用组件
python scripts/list_components.py
```

### 归档数据找不到

```bash
# 错误：Configuration file not found 或 archive_dir 不存在
# 原因：dataset 路径配置错误
# 解决：检查 YAML 中的 dataset.archive_dir 路径是否正确
# 路径相对于 configs/experiments/ 的父目录（即项目根目录）解析
```

### CLI 覆盖不生效

```bash
# 先用 --validate-only 检查覆盖后的配置
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --validate-only --set model.name=spe

# 确认输出的 JSON 中 model 字段是否已变为 "spe"
# 如果覆盖失败，检查 key 路径是否正确（使用点号分隔：model.name 不是 model:name）
```

### 模型训练失败

```bash
# 错误：OptionalDependencyError
# 原因：缺少模型所需的可选依赖
# 解决：
pip install xgboost          # XGBoost
pip install scikit-optimize  # 贝叶斯调优
pip install torch            # CNN (PyTorch)
pip install pulearn          # PUB 标签细化
```

### 输出目录被覆盖

```bash
# 框架已自动为每次运行添加时间戳后缀，不会覆盖
# 输出目录格式：outputs/<实验名>_YYYYMMDD_HHMMSS/
# 可以通过 experiment.name 区分不同实验
python run.py --config ... --set experiment.name=my_unique_name
```

### 如何确认 config_resolved.yaml 是最终参数

`config_resolved.yaml` 中包含的是 `_deep_merge(DEFAULTS, migrated_yaml)` 再经过 CLI `--set` 覆盖后的**最终完整配置**。它记录了：

1. 框架默认值（DEFAULTS）
2. YAML 文件中的配置（覆盖默认值）
3. Phase1 格式迁移结果（如果使用了一期格式）
4. CLI `--set` 覆盖（覆盖 YAML 配置）
5. 解析后的绝对路径（dataset 路径）

所以 `config_resolved.yaml` 是实验实际使用的完整参数，可直接用于复现。