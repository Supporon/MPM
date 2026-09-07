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
13. [能力状态与已知限制](#13-能力状态与已知限制)

---

## 1. 概述

MPM_codex 是一个面向**找矿预测（Mineral Prospectivity Mapping）**的工程化实验框架，核心设计理念：

- **可注册**：模型、特征算子、调优器、指标、谓词、知识提供者，以及研究单元、采样、标签、权重等研究变量全部通过统一注册表管理，新增组件无需修改框架核心代码。
- **可配置**：所有实验参数通过 YAML 配置文件驱动，配合 `--set` CLI 参数覆盖（仅标量），无需硬编码。
- **可验证**：内置完整的实验管线（数据→特征→训练→评估→预测→导出），自动生成可复现性清单（manifest.json）。

### 能力状态速览

| 能力 | 状态 | 说明 |
|------|------|------|
| 二维靶区预测 `target_area_prediction` | `implemented` | 二维任务与归档特征流程 |
| 三种执行模式（replay / archive-features / raw_gis） | `implemented`（当前仅 NSW archive-features 有内置配置） | 配置校验会**拒绝不生效的字段** |
| 8 个模型适配器 | 已实现 / 部分实验性 | `rf`、`spe`、`cnn`、`cnn2d`、`label_spreading`、`mlp`、`rf_constrained`、`spe_constrained`；算法级验证程度不同 |
| 谓词约束链路 | 已实现 / 实验性 | CNN/CNN2D/MLP 使用约束损失，约束 RF/SPE 使用样本权重迭代 |
| 概率校准（`calibrated_probability`） | **未实现** | `probability` 为未校准正类概率 |
| 深边部预测 `deep_edge_prediction` | `placeholder` | 无三维能力，实例化即报错 |

> 本框架当前可用于**归档回放、固定归档特征上的模型/调参/指标对比，以及合成数据上的 raw_gis 端到端开发与测试**；在完成空间验证、PU 语义、概率校准和原始 GIS 端到端复现之前，不宜直接用于可发表的二维研究结论。详见 [§13 能力状态与已知限制](#13-能力状态与已知限制)。

### 数据流总览

```
YAML 配置 → load_config() → DEFAULTS 合并 → Phase1 迁移 → 路径解析
    → validate_config()（严格 schema + 模式能力边界） → ExperimentConfig → Experiment.run()
        ├── ① prepare_data     （加载归档数据 或 构建研究单元）
        ├── ② build_features   （快照归档特征 或 运行 GIS 算子管线）
        ├── ③ prepare_dataset  （校验归档 schema 或 准备预处理）
        ├── ④ train            （回放归档模型 或 调优器训练；raw_gis 先外层划分再折内拟合）
        ├── ⑤ evaluate         （计算指标）
        ├── ⑥ predict          （生成预测分数，导出 CSV/GeoTIFF）
        └── export             （写入 manifest.json 可复现性清单）
```

---

## 2. 快速上手

```bash
# 进入项目目录
cd MPM_codex_phase2

# 查看所有已注册组件
python scripts/list_components.py

# 验证配置（不访问数据文件）
python run.py --config configs/experiments/nsw_rf_train.yaml --validate-only

# 从归档特征重新训练 RF（无调参）
python run.py --config configs/experiments/nsw_rf_train.yaml

# Label Spreading：使用归档目标网格进行转导式训练
python run.py --config configs/experiments/nsw_label_spreading.yaml

# 2D CNN 网格模型
python run.py --config configs/experiments/nsw_cnn2d.yaml

# MLP + all_ones 谓词约束
python run.py --config configs/experiments/nsw_mlp_pred_allones.yaml

# 约束重加权 RF / SPE
python run.py --config configs/experiments/nsw_rf_constrained_allones.yaml
python run.py --config configs/experiments/nsw_spe_constrained_allones.yaml

# 运行测试（当前有历史测试引用已删除配置，详见 §13）
python -m pytest -q
```

> 仓库预置配置中，22 个 `nsw_*.yaml` 均为 `train_from_archive_features`；另有 3 个 `lachlan_*.yaml` 分别对应 `archive_replay`（`lachlan_rf_baseline.yaml`）、`train_from_archive_features` + bayes（`lachlan_rf_phase2.yaml`）与 `raw_gis`（`lachlan_rf_raw_gis.yaml`，需 GIS 数据与 geopandas/rasterio）。各 NSW 配置的执行模式、模型、调优方式见 [§2.1](#21-内置实验配置一览)。

### 2.1 内置实验配置一览

当前磁盘中有 **22 个** `nsw_*.yaml`，全部使用 `train_from_archive_features`、`relative_score + minmax`，默认读取 NSW 归档特征：

| 配置文件 | 模型 | 调优 | 谓词 | 说明 |
|---------|:--:|:--:|:--:|------|
| `nsw_rf_train.yaml` | `rf` | none | — | RF 基线 |
| `nsw_rf_constrained_allones.yaml` | `rf_constrained` | none | all_ones | RF 约束重加权 |
| `nsw_rf_constrained_spatial_box.yaml` | `rf_constrained` | none | spatial_box | RF 空间谓词约束 |
| `nsw_spe_train.yaml` | `spe` | none | — | SPE 基线 |
| `nsw_spe_constrained_allones.yaml` | `spe_constrained` | none | all_ones | SPE 约束重加权 |
| `nsw_spe_constrained_spatial_box.yaml` | `spe_constrained` | none | spatial_box | SPE 约束重加权 |
| `nsw_cnn_train.yaml` | `cnn` | none | — | 1D CNN |
| `nsw_cnn_pred_allones.yaml` | `cnn` | none | all_ones | CNN 约束实验 |
| `nsw_cnn2d.yaml` | `cnn2d` | none | — | 2D CNN 网格模型 |
| `nsw_cnn2d_pred_allones.yaml` | `cnn2d` | none | all_ones | 2D CNN 约束实验 |
| `nsw_cnn2d_pred_spatial_box.yaml` | `cnn2d` | none | spatial_box | 2D CNN 空间谓词 |
| `nsw_cnn2d_pred_spatial_distance.yaml` | `cnn2d` | none | spatial_distance | 2D CNN 空间谓词 |
| `nsw_label_spreading.yaml` | `label_spreading` | none | — | 转导式标签传播 |
| `nsw_mlp_train.yaml` | `mlp` | none | — | MLP 基线 |
| `nsw_mlp_bayes.yaml` | `mlp` | bayes | — | MLP 贝叶斯调优 |
| `nsw_mlp_pred_allones.yaml` | `mlp` | none | all_ones | MLP 约束 |
| `nsw_mlp_pred_spatial_box.yaml` | `mlp` | none | spatial_box | MLP 空间谓词 |
| `nsw_mlp_pred_spatial_box_tuned.yaml` | `mlp` | none | spatial_box | MLP 空间谓词变体 |
| `nsw_mlp_pred_spatial_distance.yaml` | `mlp` | none | spatial_distance | MLP 空间谓词 |
| `nsw_mlp_tuned_allones.yaml` | `mlp` | none | all_ones | 固定调优参数的约束变体 |
| `nsw_mlp_tuned_spatial_box.yaml` | `mlp` | none | spatial_box | 固定调优参数的约束变体 |
| `nsw_mlp_tuned_spatial_distance.yaml` | `mlp` | none | spatial_distance | 固定调优参数的约束变体 |

> 配置文件清单应以 `find configs/experiments -name 'nsw_*.yaml'` 为准。`archive_replay` 和 `raw_gis` 的示例会在 §9 标为“用户自建模板”，不会伪装成内置配置。

---

## 3. 代码结构与目录说明

```
MPM_codex_phase2/
├── run.py                                    # ★ 统一入口文件
├── baseline_tools.py                         # 基线工件安全检查工具（无科学计算依赖）
│
├── configs/                                  # 配置文件目录
│   └── experiments/                          #   实验 YAML 配置（见 §2.1 一览）
│
├── src/                                      # 框架源码
│   ├── core/                                 # ★ 框架核心（骨架层，扩展时不需要改）
│   │   ├── config.py                         #   配置加载/合并/校验/Phase1迁移/CLI覆盖/模式能力边界
│   │   ├── experiment.py                     #   实验编排器（6阶段管线 + manifest 导出）
│   │   ├── registry.py                       #   ★ 通用组件注册表 ComponentRegistry
│   │   ├── bootstrap.py                      #   启动时自动导入所有内置组件触发注册
│   │   ├── contracts.py                      #   数据契约（TrainingData/SplitData）与接口协议
│   │   └── spec.py                           #   配置的类型化视图 ExperimentSpec
│   │
│   ├── models/                               # ★ 模型适配器（新增模型在此写）
│   │   ├── registry.py                       #   MODEL_REGISTRY + LABEL_REFINER_REGISTRY + fit_params_for
│   │   ├── rf.py                             #   随机森林适配器
│   │   ├── spe.py                            #   Self-Paced Ensemble 适配器（自包含实现）
│   │   ├── cnn.py                            #   1D CNN（PyTorch）适配器
│   │   ├── cnn2d.py                          #   patch-based 2D CNN 网格适配器
│   │   ├── label_spreading.py                #   转导式 Label Spreading 网格适配器
│   │   ├── mlp.py                            #   MLP（PyTorch）+ LUSI 谓词约束加权损失
│   │   ├── rf_constrained.py                 #   RF + ConstrainedReweighting
│   │   ├── spe_constrained.py                #   SPE + ConstrainedReweighting
│   │   └── pu.py                             #   PUB 标签细化器（legacy 兼容）
│   │
│   ├── training/                             # 共享训练与损失
│   │   ├── losses.py                         #   LUSI 谓词约束加权损失
│   │   ├── constrained.py                    #   sklearn 模型的约束重加权包装器
│   │   └── trainer.py                        #   TorchTrainingLoop / TorchTrainingConfig
│   │
│   ├── data/                                 # 数据加载与研究变量（注册化）
│   │   ├── dataset.py                        #   归档数据集加载/校验/快照
│   │   ├── registries.py                     #   研究单元/采样器/标签/权重注册表（4 个）
│   │   ├── research_unit_builtins.py         #   研究单元构建器（point_local_environment）
│   │   ├── samplers.py                       #   背景采样器（random_points_in_nsw_boundary）
│   │   ├── label_strategies.py               #   标签策略（positive_unlabeled_as_zero）
│   │   ├── weight_strategies.py              #   样本权重策略（size_code、uniform）
│   │   ├── research_units.py                 #   （遗留）点研究单元函数集合，注册化后仅兼容
│   │   └── labels.py                         #   （遗留）标签/权重函数集合
│   │
│   ├── features/                             # 特征预处理
│   │   ├── preprocess.py                     #   ★ BaselinePreprocessor（fit/transform/fit_transform，split-first）
│   │   └── spatial.py                        #   SpatialFeatureExtractor（旧 API 兼容层）
│   │
│   ├── operators/features/                   # 特征算子（GIS 特征提取）
│   │   ├── pipeline.py                       #   算子管线编排器（行数/schema 契约）
│   │   ├── builtins.py                       #   5 个内置算子实现（封装 lib_mpm）
│   │   ├── context.py                        #   LegacyFeatureContext（延迟加载 lib_mpm）
│   │   └── registry.py                       #   FEATURE_OPERATOR_REGISTRY
│   │
│   ├── plotting/                             # 结果绘图
│   │   └── plot.py                            #   训练点与全区栅格概览图
│   │
│   ├── tasks/
│   │   ├── registry.py                       #   TASK_REGISTRY
│   │   ├── target_area.py                    #   靶区预测任务（已完整实现）
│   │   └── deep_edge.py                      #   深边部预测任务（占位，实例化即报错）
│   │
│   ├── tuning/                               # 超参数调优
│   │   ├── registry.py                       #   TUNER_REGISTRY
│   │   ├── bayes.py                          #   贝叶斯优化（scikit-optimize）
│   │   └── none.py                           #   无调优，直接训练
│   │
│   ├── validation/                           # 验证与评估
│   │   ├── registry.py                       #   SPLITTER_REGISTRY + METRIC_REGISTRY
│   │   ├── splitters.py                      #   留出法 + 分层/空间 K 折交叉验证
│   │   ├── metrics.py                        #   9 个通用分类指标
│   │   └── mpm_metrics.py                    #   MPM 面积捕获指标（prediction-rate curve 等）
│   │
│   ├── knowledge/                            # 知识注入
│   │   ├── pipeline.py                       #   知识管线（命名空间 knowledge[provider]）
│   │   ├── providers.py                      #   知识提供者（empty、spatial_extent）
│   │   └── registry.py                       #   KNOWLEDGE_REGISTRY
│   │
│   ├── predicates/                           # 谓词约束
│   │   ├── pipeline.py                       #   谓词管线（data_transform 先于 constraint）
│   │   ├── builtins.py                       #   谓词实现（all_ones, spatial_box, spatial_distance, combined）
│   │   └── registry.py                       #   PREDICATE_REGISTRY
│   │
│   └── utils/                                # 工具函数
│       ├── logging.py                        #   日志配置（handler 清理，防跨目录累积）
│       ├── files.py                          #   SHA256/Git信息/环境版本/YAML-JSON读写
│       └── seeds.py                          #   SeedContext（派生 sampling/split/tuning/model/dataloader 种子）
│
├── outputs/                                  # 实验输出目录（.gitignore）
│   └── <实验名称>_<run_id>/                   #   每次运行自动追加微秒级时间戳
│       ├── config_resolved.yaml              #   合并后的完整配置（含DEFAULTS+CLI覆盖）
│       ├── experiment.log                    #   结构化日志
│       ├── manifest.json                     #   可复现性清单（含Git commit/派生种子/split摘要/哈希）
│       ├── metrics.json                      #   评估指标
│       ├── models/                           #   序列化模型（.pkl）
│       ├── predictions/                      #   预测结果（CSV + 可选 GeoTIFF）
│       ├── figures/                          #   评估图表
│       └── intermediate/                     #   中间数据快照
│
├── scripts/                                  # 辅助脚本
│   ├── list_components.py                    #   列出主注册表组件（不含研究变量注册表）
│   ├── create_baseline_manifest.py           #   创建基线工件清单
│   ├── render_run_sh.py                      #   从配置生成 shell 命令
│   ├── validate_reproduction.py              #   复现验证（防零执行假阳性）
│   └── baseline/                             #   基线 shell（nsw_rf.sh；另有历史 Lachlan 脚本）
│
├── tests/                                    # 测试
│   ├── test_framework.py                     #   框架核心测试
│   ├── test_spe_model.py                     #   SPE 模型单元测试
│   ├── test_mlp_predicates.py                #   MLP + 谓词/LUSI 测试
│   ├── test_mpm_metrics.py                   #   MPM 面积指标测试
│   ├── test_seeds.py                         #   SeedContext 派生测试
│   ├── fixtures/gis/rasters/                 #   最小 GIS fixture（raw_gis 冒烟用）
│   └── baseline/                             #   基线回归测试
│
└── docs/                                     # 文档
    ├── PHASE2_ARCHITECTURE_CN.md             #   架构说明
    ├── PHASE2_USAGE_CN.md                    #   使用说明（简洁版）
    ├── PHASE2_USAGE_VERIFIED.md              #   历史验证版使用说明
    ├── PHASE1_MIGRATION.md                   #   Phase1 迁移说明
    ├── BASELINE.md                           #   重构前基线契约
    └── baseline_manifest.json                #   基线工件清单
```

### 分层架构

```
┌──────────────────────────────────────────────────────────┐
│  run.py                     ← 入口层（argparse + 调用）  │
├──────────────────────────────────────────────────────────┤
│  src/core/config.py         ← 配置层（加载/合并/严格校验）│
│  src/core/experiment.py     ← 编排层（6阶段管线）        │
│  src/core/spec.py           ← 类型化视图                 │
├──────────────────────────────────────────────────────────┤
│  src/models/  src/tuning/  src/validation/  src/tasks/  │
│  src/operators/features/  src/predicates/  src/knowledge/│
│  src/data/（研究单元/采样/标签/权重）  src/features/     │
│                                ← 组件层（全部可插拔替换） │
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

所有实验都通过 `run.py` 执行，通过 `--config` 参数指定配置文件。当前仓库内置示例均为 NSW 归档特征实验：

```bash
python run.py --config configs/experiments/nsw_rf_train.yaml
```

`archive_replay` 与 `raw_gis` 的能力已在编排器和配置校验中实现，但当前仓库没有预置这两种模式的 YAML；需要按 §4.3 的边界自行创建配置。

### 4.2 实验管线（6 个阶段）

`Experiment.run()` 按顺序执行以下阶段：

| 阶段 | 方法 | 说明 | 关键操作 |
|:----:|------|------|---------|
| ① | `prepare_data` | 加载数据 | `archive_replay`/`train_from_archive_features` 模式加载归档CSV；`raw_gis` 模式构建研究单元 |
| ② | `build_features` | 构建特征 | 快照归档特征，或运行配置的 GIS 特征算子管线 |
| ③ | `prepare_dataset` | 准备数据集 | 校验归档 schema，或准备 raw_gis 预处理（**不在此处拟合**） |
| ④ | `train` | 训练模型 | 回放归档模型，或（raw_gis 先外层划分、仅训练折拟合预处理）通过 Tuner 训练 |
| ⑤ | `evaluate` | 评估指标 | 计算配置的指标；回放模式仅记录元数据不重算 |
| ⑥ | `predict` | 预测导出 | 生成预测分数，导出 CSV + 可选 GeoTIFF |

### 4.3 三种执行模式与能力边界

| 模式 | 适用场景 | 数据来源 | 模型来源 | 需要 GIS 依赖 |
|------|---------|---------|---------|:--:|
| `archive_replay` | 回归验证，确认归档工件可被当前框架读取 | 归档 CSV 和 `.pkl`/预测工件 | 归档 `.pkl` 文件直接复制 | 否 |
| `train_from_archive_features` | 快速模型迭代，换模型/换参数/换指标 | 归档 CSV（预计算特征） | 重新训练 | 否 |
| `raw_gis` | 完整管线，从原始 GIS 数据开始 | 原始 GIS 文件（.shp/.tif） | 重新训练 | 是 |

**每种模式都拒绝不生效的配置（配置必须产生行为）。** 在 `validate_config` 阶段即硬失败，而不是静默忽略：

| 配置能力 | `archive_replay` | `train_from_archive_features` | `raw_gis` |
|---|:---:|:---:|:---:|
| 复制并核验归档工件 | 是 | 否 | 否 |
| 训练新模型 | 否 | 是 | 是 |
| 更换模型/调参 | **拒绝**（须 `tuning=none`） | 是 | 是 |
| 修改外层 train/test（holdout） | 拒绝 | **拒绝**（须 `random_holdout`，实际用归档固定划分） | 是 |
| 特征算子 | **拒绝**（须 `features.operators=[]`） | **拒绝**（须 `[]`，归档特征已固定） | 是 |
| 预处理 | 拒绝 | 已在归档中固定 | 是（fold-safe） |
| PUB/标签细化 | **拒绝** | **拒绝**（归档标签已固定） | 是（仅训练折内） |
| Knowledge/Predicate | **拒绝** | 仅能从归档字段计算者可用（空间谓词缺坐标会报错） | 是 |
| 区域预测 | 复制归档结果 | 使用归档目标特征 | 从 GIS 构建目标特征 |

> 本仓库没有预置 `archive_replay` 或 `raw_gis` 配置；上表描述的是代码支持的模式边界，而不是可直接复制运行的内置实验。

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
| `MODEL_REGISTRY` | 模型 | `rf`, `spe`, `cnn`, `cnn2d`, `label_spreading`, `mlp`, `rf_constrained`, `spe_constrained` | `model.name` |
| `FEATURE_OPERATOR_REGISTRY` | 特征算子 | `raster_statistics`, `texture`, `elevation_gradient`, `line_distance`, `categorical_geology` | `features.operators[].name` |
| `TASK_REGISTRY` | 找矿任务 | `target_area_prediction`, `deep_edge_prediction`（占位） | `task.name` |
| `TUNER_REGISTRY` | 调优器 | `none`, `bayes` | `tuning.name` |
| `LABEL_REFINER_REGISTRY` | 标签细化 | `pub` | `label_refinement.name` |
| `SPLITTER_REGISTRY` | 数据分拆 | `random_holdout`, `stratified_kfold`, `spatial_block_kfold`, `spatial_group_kfold`, `spatial_block_holdout` | `validation.holdout.name`, `validation.cross_validation.name` |
| `METRIC_REGISTRY` | 评估指标 | `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `confusion_matrix`, `average_precision`, `balanced_accuracy`, `mcc` | `validation.metrics[]`, `validation.primary_metric` |
| `PREDICATE_REGISTRY` | 谓词约束 | `all_ones`, `spatial_box`, `spatial_distance`, `combined` | `predicates.items[].name` |
| `KNOWLEDGE_REGISTRY` | 知识注入 | `empty`, `spatial_extent` | `knowledge.items[].name` |
| `RESEARCH_UNIT_REGISTRY` | 研究单元 | `point_local_environment` | `research_unit.type` |
| `BACKGROUND_SAMPLER_REGISTRY` | 背景采样 | `random_points_in_nsw_boundary` | `research_unit.train_unlabeled` |
| `LABEL_STRATEGY_REGISTRY` | 标签策略 | `positive_unlabeled_as_zero` | `label.strategy` |
| `WEIGHT_STRATEGY_REGISTRY` | 样本权重 | `size_code`, `uniform` | `label.sample_weight`（由标签策略消费） |

> 注意：`scripts/list_components.py` 目前只打印前 9 个主注册表；研究单元/背景采样/标签策略/样本权重这 4 个研究变量注册表已注册化（见 `src/data/registries.py`），但未列入该脚本输出。

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
    ↓ validate_config()            ← 严格 schema + 组件名校验 + 模式能力边界
    ↓ ExperimentConfig(values=resolved, source_path=...)
```

严格 schema：顶层、`model`、`research_unit`、`label` 节拒绝未知键；组件名必须存在于对应注册表；`train_positive` 仅支持 `occurrence_points`；`categorical_encoding`/`scaling` 目前仅实现单一取值。

**`--set` CLI 覆盖的插入位置：**

```
load_config() 返回 ExperimentConfig
    ↓
apply_cli_overrides(config, ["model.name=spe", "tuning.params.n_iter=200"])
    ↓ ① 解析 "model.name=spe" → {"model.name": "spe"}
    ↓ ② _coerce_value: "200" → 200 (int)
    ↓ ③ _dot_to_nested: {"model.name": "spe"} → {"model": {"name": "spe"}}
    ↓ ④ _deep_merge(config.values, overrides_nested)
    ↓ ⑤ validate_config(merged)  ← 重新校验（含模式能力边界）
    ↓ ⑥ 返回新的 ExperimentConfig
```

### 4.6 评分语义契约（score_type）

预测输出的列名与语义由 `prediction.score_type` 与 `prediction.normalization` 共同决定：

| score_type | 语义 | 来源 | 输出列名 | 允许的 normalization |
|-----------|------|------|---------|---------------------|
| `probability` | 未校准正类概率 | `predict_proba` | `prob` | 仅 `none` |
| `raw_score` | 模型原始决策分数 | `decision_function`（需模型支持） | `raw_score` | 仅 `none` |
| `relative_score` | 相对分数（可 MinMax） | `predict_proba`（+可选 MinMax） | `relative_score` | `none` 或 `minmax` |

- `probability` / `raw_score` 与 `normalization=minmax` 互斥（MinMax 结果不是校准概率），配置校验会拒绝。
- `raw_score` 需要模型暴露 `decision_function`（如 SVM/逻辑回归）；当前 8 个内置模型均以 `predict_proba` 为主要接口，应使用 `probability` 或 `relative_score`。
- 当前 **没有实现 `calibrated_probability`**：`probability` 是未校准的正类概率，不能直接当作成矿概率阈值使用。

### 4.7 随机种子派生（SeedContext）

框架从实验基础种子 `experiment.seed` 派生出互不干扰的用途种子（`src/utils/seeds.py`），保证改某个阶段的随机性不影响其他阶段：

```text
experiment.seed
    ├── sampling_seed    # 未标注点采样
    ├── split_seed       # 外层划分
    ├── tuning_seed      # 调优（BayesSearchCV）
    ├── model_seed       # 模型 random_state（各适配器接收派生种子）
    └── dataloader_seed  # PyTorch DataLoader
```

因此 `--set experiment.seed=N` 现在会真正改变模型的随机性（RF 的 `build()` 用 `resolved["random_state"] = seed` 直接覆盖，而非 `setdefault`）。派生种子表会写入 manifest 的 `derived_seeds`。

---

## 5. 基本使用方式

### 5.1 运行实验

```bash
# 完整运行一个实验
python run.py --config configs/experiments/nsw_rf_train.yaml

# 运行 SPE 实验（无调优，固定参数）
python run.py --config configs/experiments/nsw_spe_train.yaml
```

### 5.2 验证配置（不访问数据文件）

```bash
# 快速检查配置是否正确
python run.py --config configs/experiments/nsw_mlp_bayes.yaml --validate-only

# 输出 JSON 格式的组件概览
# {
#   "experiment": "nsw_mlp_bayes_test",
#   "execution_mode": "train_from_archive_features",
#   "task": "target_area_prediction",
#   "feature_operators": [],
#   "model": "rf",
#   "label_refinement": null,
#   "knowledge": [],
#   "predicates": [],
#   "holdout": "random_holdout",
#   "cross_validation": "stratified_kfold",
#   "metrics": ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"],
#   "primary_metric": "f1",
#   "tuner": "bayes"
# }
```

`--validate-only` 只加载配置、校验 schema 与组件兼容性、打印组件图，**不实例化 `Experiment`、不访问数据**，也不证明实验可复现。

### 5.3 CLI 参数覆盖（`--set`）

**不修改 YAML 文件，直接在命令行覆盖任意标量配置项**。这是快速实验对比的核心功能。

```bash
# 单参数覆盖：换模型
python run.py --config configs/experiments/nsw_rf_train.yaml --set model.name=spe

# 多参数覆盖：换调优器 + 调参数
python run.py --config configs/experiments/nsw_mlp_bayes.yaml \
    --set tuning.name=none tuning.params.n_iter=200

# 覆盖调优参数
python run.py --config configs/experiments/nsw_mlp_bayes.yaml \
    --set tuning.params.n_iter=500

# 覆盖实验级参数
python run.py --config configs/experiments/nsw_rf_train.yaml \
    --set experiment.seed=999 experiment.name=my_custom_exp

# 覆盖验证参数
python run.py --config configs/experiments/nsw_rf_train.yaml \
    --set validation.holdout.params.test_size=0.3 validation.primary_metric=roc_auc

# 开关型参数
python run.py --config configs/experiments/nsw_rf_train.yaml \
    --set prediction.export_geotiff=false

# 配合 validate-only 预览覆盖效果
python run.py --config configs/experiments/nsw_mlp_bayes.yaml \
    --validate-only --set model.name=spe tuning.name=none
```

**类型推断规则：**
- `"true"` / `"false"` → `True` / `False`
- `"123"` → `123`（int）
- `"0.5"` → `0.5`（float）
- 其他 → 保持字符串

**重要限制：**
- `--set` **不支持列表值或 JSON 值**（如 `'features.operators=[{...}]'`），请使用专用 YAML 配置文件。
- `--set` **不支持列表索引路径**（如 `predicates.items.0.name=xxx`），请使用专用 YAML 配置文件。
- 遇到列表/JSON/列表索引路径时，框架会直接报错并提示改用 YAML。

**重要行为：**
- 原 YAML 文件不会被修改，覆盖只在内存中生效。
- 覆盖后会重新走 `validate_config` 校验（含模式能力边界），无效覆盖在运行前被拦截。
- `config_resolved.yaml` 会记录最终合并后的参数（包含 CLI 覆盖）。

### 5.4 输出目录时间戳

**每次运行实验会自动在输出目录名后追加微秒级 `run_id`**，防止重复运行同一配置时覆盖前次结果。目录以原子方式创建，已存在则失败。

```
# 配置中写的是: output_dir: outputs/nsw_rf_train
# 实际创建的目录: outputs/nsw_rf_train_20260901_143052_123456/
#                                                        ↑
#                                        YYYYMMDD_HHMMSS_微秒
```

这意味着你可以连续多次运行同一个 YAML 配置（每次用不同 `--set` 参数），每次结果都会保存在独立的目录中。

### 5.5 查看所有已注册组件

```bash
python scripts/list_components.py
```

如需同时加载 YAML `plugins` 列表中的模块：

```bash
python scripts/list_components.py --config configs/experiments/my_experiment.yaml
```

### 5.6 运行测试

```bash
# 运行全部测试
python -m pytest -q

# 运行特定测试
python -m pytest tests/test_framework.py -q

# 运行特定测试方法
python -m pytest tests/test_framework.py::FrameworkTests::test_config_load -v
```

---

## 6. 归档 CSV 数据格式

`train_from_archive_features` 和 `archive_replay` 模式直接读取归档 CSV，无需 GIS 提取。

### 6.1 核心 CSV 文件

| 文件 | 行数 | 用途 |
|------|:----:|------|
| `Xy_train.csv` | 479 | 全部训练数据（138 特征 + `sample_weight` + `label`） |
| `Xy_train_new.csv` | 479 | 训练数据副本（PUB 兼容用，schema 与 Xy_train 一致） |
| `Xy_rf_train.csv` | 359 | 训练集（75%），用于模型训练 |
| `Xy_rf_test.csv` | 120 | 测试集（25%），用于模型评估 |
| `target_features.csv` | 4071 | 预测网格点的特征（无标签） |
| `target_coords_purged.csv` | 4071 | 预测网格点的 X/Y 坐标 |
| `target_mask.csv` | 4200 | 完整研究区网格掩膜（含研究区内/外标记） |

`DatasetRepository.REQUIRED_FILES` 强制要求上述 7 个文件齐全，且 schema 一致；缺失即报错。

### 6.2 训练数据列结构

每一行是一个**采样点**（研究单元）。`Xy_train.csv` 的列结构：

```
138 个特征列（磁法统计、重力统计、纹理、高程梯度、线距离、地质分类）
    +
sample_weight（样本权重：VLG 大矿 0.8，OCC 小矿 0.1，unlabeled 背景点 0.5）
    +
label（标签：1=已知矿点，0=背景点）
```

> 注意：未标注背景点被临时编码为 `0` 是当前训练策略，不代表它们是已验证的无矿负样本（见 [§13 能力状态与已知限制](#13-能力状态与已知限制)）。

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
  ③ 把预测分数按行号顺序填进去
  ④ 按 X/Y 唯一值 reshape 成 2D 矩阵 → 画图/导出 GeoTIFF
```

---

## 7. 模型适配器原理

### 7.1 两个层次

框架通过 **适配器（Adapter）** 和 **模型实例（Model）** 两层分离来统一操作所有模型：

```
适配器（Adapter）— 3 个职责（工厂，负责"造模型"）
    ├── build(params, seed)       → 返回一个模型实例
    ├── fit_params(data)          → 返回训练参数（sample_weight、constraints）
    └── validate_config(params)   → 校验模型参数合法性
    （可选属性：supports_constraints、artifact_filename、archive_metric_aliases）

模型实例（Model）— 3 个方法（由 build() 返回的对象，负责实际训练和预测）
    ├── fit(X, y, **kwargs)       → 训练
    ├── predict(X)                → 预测类别
    └── predict_proba(X)          → 预测概率
```

**适配器是"工厂"，模型是"产品"。** 框架内部调用流程：

```python
model = model_adapter.build(params, seed)           # ① 适配器造模型
model.fit(X, y, **model_adapter.fit_params(data))   # ② 模型训练
model.predict(X)                                    # ③ 预测类别
model.predict_proba(X)                              # ④ 预测概率
```

### 7.2 八种内置模型

| 模型 | 实现 | `supports_constraints` | `grid_model` | 依赖 | 说明 |
|------|------|:--:|:--:|------|------|
| `rf` | sklearn `RandomForestClassifier` | 否 | 否 | 核心 | 归档特征基线；`random_state=seed` 直接覆盖 |
| `spe` | 自包含 `SelfPacedEnsemble` | 否 | 否 | 核心 | 面向高度不平衡；基分类器默认决策树 |
| `cnn` | PyTorch 1D CNN | **是** | 否 | `torch` | 将特征列作为 1D 信号；实验性 |
| `cnn2d` | PyTorch patch-based 2D CNN | **是** | **是** | `torch` | 在预测网格上提取局部 patch；实验性 |
| `label_spreading` | sklearn `LabelSpreading` 包装器 | 否 | **是** | sklearn | 转导式使用有效网格单元；默认带内部 OOF Platt scaling |
| `mlp` | PyTorch MLP | **是** | 否 | `torch` | 消费 LUSI 谓词约束；实验性 |
| `rf_constrained` | RF + `ConstrainedReweighting` | **是** | 否 | 核心 | 无梯度模型以样本权重迭代近似约束 |
| `spe_constrained` | SPE + `ConstrainedReweighting` | **是** | 否 | 核心 | 无梯度模型以样本权重迭代近似约束；实验性 |

适配器的 `supports_constraints=True` 表示该模型可以接收谓词产生的约束参数，并不表示算法级数值一致性已完成验证。`rf_constrained` / `spe_constrained` 将约束转换为 `ConstrainedReweighting` 的样本权重迭代；它们不是把 LUSI 项直接加入树模型梯度。

**网格模型协议：** `cnn2d` 与 `label_spreading` 的 `grid_model=True`。适配器的 `fit_params(data)` 注入 `grid`（网格特征）、`inside`（有效单元掩膜）和 `train_cells`（训练点对应的网格行列）；训练后框架通过模型的 `set_predict_cells(cells)` 指定点对应的网格单元。此类模型的 `predict_proba` 主要在网格单元上产生结果，绘图时再按最近网格单元关联训练点。

### 7.3 LUSI 谓词约束加权损失（`cnn` / `cnn2d` / `mlp`）

`cnn`、`cnn2d` 和 `mlp` 适配器可以在损失中引入谓词统计不变量（Vapnik & Izmailov 的 LUSI，实现参考 TSIL）：

```
loss = τ̂ · MSE + τ · (1/N) · ‖φ̃ᵀ e‖²
```

- 当存在谓词约束时，`cnn`、`cnn2d` 和 `mlp` 使用约束损失（在 sigmoid 概率上计算）；无约束时使用各自的标准分类损失。
- φ 向量由 `constraint` 谓词生成（`all_ones`/`spatial_box`/`spatial_distance`/`combined`），经 L2 归一化。
- τ 通过可学习参数 α 经 sigmoid 得到（`tau_init`/`learn_tau` 可配）。
- **限制**：LUSI 数值对齐尚未验证（`experimental`）；贝叶斯搜索 + 谓词约束当前显式报错（`BayesSearchCV` 无法按折切分 φ），须用 `tuning.name=none`。

### 7.4 三种适配策略

| 模型情况 | 策略 | 代码量 | 例子 |
|---------|------|:------:|------|
| 本身就是 sklearn 模型 | 适配器直接返回，`build()` 只做参数传递 | ~30行 | RF 适配器 |
| 自实现算法，无 sklearn 接口 | 写包装类继承 `BaseEstimator` + `ClassifierMixin`，手动实现 `fit/predict/predict_proba` | ~100行 | SPE 适配器 |
| 底层库完全不同（如 PyTorch） | 写包装类，`fit()` 里写训练循环，`predict_proba()` 里写前向推理 | ~200行 | CNN / MLP 适配器 |

### 7.5 四种特殊场景适配

| 变更程度 | 场景 | 需要改什么 | 不需要改 |
|---------|------|-----------|---------|
| **最小** | 换另一个 sklearn 模型 | 写 `ModelAdapter` | 框架、数据管线、Task |
| **中等** | 模型无 sklearn 接口 | 写 `ModelAdapter` + 包装类 | 框架、数据管线 |
| **较大** | 输入格式不同（如图像） | 加 `FeatureOperator` + `ModelAdapter` | 框架、Experiment |
| **最大** | 完全不同的找矿方法 | 加 `Task` + 全套 | 框架 core |

---

## 8. 如何扩展组件

### 8.1 新增模型

以 XGBoost 为例（**教程示例**：`xgb` 组件与 `xgb_experiment.yaml` 需你自行创建，当前仓库未预置）。如果新增或修改任何代码文件，请按仓库 `AGENTS.md` 的要求在 `change_log.md` 追加记录。

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

（或用 `plugins` 机制从外部模块加载，见 [§8.10](#810-通过-plugins-扩展组件)。）

**第三步：编写 YAML 配置** — `configs/experiments/xgb_experiment.yaml`

```yaml
experiment:
  name: xgb_experiment
  seed: 2026
  output_dir: outputs/xgb_experiment
  execution_mode: train_from_archive_features

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
      n_estimators:  { type: integer, low: 50, high: 500 }
      max_depth:     { type: integer, low: 3, high: 15 }
      learning_rate: { type: real, low: 0.01, high: 0.3, prior: log-uniform }
# ... 其余配置节（dataset/research_unit/label/features/validation 等）复用已有配置
```

**第四步：运行（以下文件名只是示例，需先自行创建并放在指定路径）**

```bash
python run.py --config configs/experiments/xgb_experiment.yaml --validate-only
python run.py --config configs/experiments/xgb_experiment.yaml
```

**支持约束的模型**：如需消费谓词约束，设置 `supports_constraints = True`，并在 `fit_params(data)` 中返回约束参数（参考 `src/models/mlp.py`、`src/models/cnn2d.py`）。声明支持约束却未实现 `fit_params(data)` 会显式失败。

### 8.2 新增验证指标

**第一步：在 `src/validation/metrics.py` 中注册新指标**

```python
from sklearn.metrics import matthews_corrcoef

@METRIC_REGISTRY.decorator("mcc")
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
    - mcc                    # ← 新增
  primary_metric: f1
```

**不需要改任何其他文件。** 指标函数签名固定为 `(labels, predictions, probabilities, sample_weight)`，框架自动调用。主指标必须返回标量（用于调优评分）；结构化指标（如 `confusion_matrix`）可用于报告但不能作为 `primary_metric`。

> `average_precision`、`balanced_accuracy`、`mcc` 已内置注册，无需再添加。

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

**空间分拆器**（`spatial_block_kfold` / `spatial_group_kfold` / `spatial_block_holdout`）需要 `TrainingData.metadata["units"]` 中的 X/Y 坐标，且要求**投影坐标（米）**——经纬度坐标会触发明确错误。它们只能在 `raw_gis` 模式使用（`train_from_archive_features` 使用归档固定 train/test，且其研究单元不含坐标元数据）。空间分拆器的 `build_cv`/`split` 需要 `data` 参数，`build_cv` 会自动检测并传入。

### 8.4 新增研究单元 / 采样器 / 标签策略 / 权重策略

研究变量已注册化，由四个注册表驱动（`src/data/registries.py`）：

| 注册表 | 内置实现 | 配置键 |
|--------|---------|--------|
| `RESEARCH_UNIT_REGISTRY` | `point_local_environment` | `research_unit.type` |
| `BACKGROUND_SAMPLER_REGISTRY` | `random_points_in_nsw_boundary` | `research_unit.train_unlabeled` |
| `LABEL_STRATEGY_REGISTRY` | `positive_unlabeled_as_zero` | `label.strategy` |
| `WEIGHT_STRATEGY_REGISTRY` | `size_code`, `uniform` | `label.sample_weight`（由标签策略消费） |

`Task` 只声明所需维度、空间支撑和输出类型，通过注册表调用具体实现，不再直接 import 模块级函数。配置中的 `research_unit.type`、`research_unit.train_unlabeled`、`label.strategy` 会被注册表校验并消费。

**新增研究单元类型**（在 `src/data/research_unit_builtins.py` 中注册）：

```python
@RESEARCH_UNIT_REGISTRY.decorator("regular_grid")
class RegularGridUnit:
    name = "regular_grid"

    def __init__(self, params):
        self.params = dict(params)

    def build_positive_units(self, occurrence_path, label_config):
        ...

    def build_prediction_units(self, boundary_path, grid_size):
        ...
```

**新增背景采样器**（在 `src/data/samplers.py` 中注册）：

```python
@BACKGROUND_SAMPLER_REGISTRY.decorator("archive_fixed")
class ArchiveFixedSampler:
    name = "archive_fixed"

    def sample(self, boundary_path, count, seed=None):
        ...
```

**新增标签策略**（在 `src/data/label_strategies.py` 中注册）：

```python
@LABEL_STRATEGY_REGISTRY.decorator("occurrence_binary")
class OccurrenceBinaryStrategy:
    name = "occurrence_binary"

    def apply_positive(self, occurrences, label_config):
        ...

    def apply_unlabeled(self, points, label_config):
        ...
```

**新增样本权重策略**（在 `src/data/weight_strategies.py` 中注册）：

```python
@WEIGHT_STRATEGY_REGISTRY.decorator("my_weight")
class MyWeightStrategy:
    name = "my_weight"

    def compute(self, attributes, weight_config):
        ...
```

> 注意：`research_units.py` 与 `labels.py` 中的旧函数仍保留（一期兼容），新代码应使用注册化组件。

### 8.5 调整标签与权重

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

`label.strategy` 的值由 `LABEL_STRATEGY_REGISTRY` 校验；`label.sample_weight` 中的 SIZE_CODE 映射由 `size_code` 权重策略消费。

> 提醒：`label.sample_weight` 只在 `raw_gis` 模式生效（那里才从 occurrence 的 SIZE_CODE 构造标签与权重）。`train_from_archive_features` 使用归档中已固定的标签与权重，改 `label.sample_weight` 不生效。

### 8.6 新增谓词（Predicate）

谓词按 `kind` 分为两类（`src/predicates/builtins.py`）：

- `kind="constraint"`：生成 φ 向量写入 `TrainingData.constraints`，供声明支持约束的模型（`cnn`、`cnn2d`、`mlp` 以及 `rf_constrained` / `spe_constrained` 的约束重加权）消费。内置 `all_ones`、`spatial_box`、`spatial_distance`、`combined`。
- `kind="data_transform"`：修改标签/权重/特征。**当前无内置 data_transform 谓词**，需自行注册。

**所有谓词必须保持行数不变**；改变行数会在管线中报错（需显式重采样契约）。

**第一步：注册一个 constraint 谓词**（生成 φ 向量）

```python
@PREDICATE_REGISTRY.decorator("positive_constraint")
class PositiveConstraintPredicate:
    """根据某特征是否为 1 生成 φ 向量（示例，需自行注册）。"""

    name = "positive_constraint"
    kind = "constraint"

    def __init__(self, params: Mapping[str, Any]):
        self.feature = str(params["feature"])
        self.params = dict(params)

    def apply(self, data: TrainingData, context: Mapping[str, Any]) -> TrainingData:
        import numpy as np
        phi = np.where(data.features[self.feature] == 1, 1.0, 0.0).astype(np.float32)
        return data.with_constraints({"phi_vector": phi})
```

**第二步：在 YAML 中启用（须 `model.name=mlp` 且 `tuning.name=none`）**

```yaml
model:
  name: mlp
tuning:
  name: none
predicates:
  enabled: true
  combine: sequential
  items:
    - name: positive_constraint
      params:
        feature: Intrusions_Tabberabberan
```

> 注意：`positive_constraint`、`weight_adjustment` 等是文档示例组件，**当前未内置注册**，需自行注册后方可使用。`cnn`、`cnn2d`、`mlp` 可消费约束损失；`rf_constrained` / `spe_constrained` 使用约束重加权；`rf`、`spe` 和普通无约束配置会因不支持约束而报错。

### 8.7 新增知识注入（Knowledge Provider）

知识注入在谓词之前执行，用于**从外部知识源构建知识工件**（如地质图、已知矿化带、专家标注区域），供谓词或模型使用。命名空间约定为 `context["knowledge"][provider_name]`。

**第一步：在 `src/knowledge/providers.py` 中注册**

```python
@KNOWLEDGE_REGISTRY.decorator("favorable_zone")
class FavorableZoneKnowledgeProvider:
    """从 shapefile 加载已知有利成矿区域，识别位于有利区内的训练样本索引。"""

    name = "favorable_zone"

    def __init__(self, params: Mapping[str, Any]):
        self.zone_path = str(params["zone_path"])
        self.params = dict(params)

    def build(self, data: TrainingData, context: Mapping[str, Any]) -> Mapping[str, Any]:
        # 返回 dict 知识工件，例如 {"favorable_indices": [...]}
        return {"favorable_indices": [...]}
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

内置 `spatial_extent` 提供者会从研究单元坐标提取研究区域 bounds/center/span（供 `spatial_box`/`spatial_distance` 消费）；`empty` 用于验证链路。

### 8.8 知识注入 vs 谓词约束：区别是什么

| 维度 | 知识注入（Knowledge） | 谓词约束（Predicate） |
|------|---------------------|---------------------|
| **执行顺序** | 先执行（在谓词之前） | 后执行（在知识之后） |
| **目的** | **读取外部知识，构建知识工件**（如索引列表、范围、掩膜） | **生成 φ 向量或变换数据**（constraint / data_transform） |
| **输入/输出** | 输入 `TrainingData` + `context`，输出 `dict`（知识工件） | 输入 `TrainingData` + `context`（含 `context["knowledge"]`），输出 `TrainingData` |
| **是否改变数据** | **不改变** `TrainingData` | **改变** `TrainingData`（constraints 或标签/权重） |
| **典型场景** | 加载地质图、计算矿化带范围、查询已知矿区 | 生成 LUSI φ 向量（`all_ones`/`spatial_box`/`spatial_distance`/`combined`） |

**数据流示意：**

```
TrainingData
    │
    ├── KnowledgePipeline.build(data, context)
    │   └── 结果存入 context["knowledge"]
    │
    └── PredicatePipeline.apply(data, context)
        ├── data_transform 谓词（修改标签/权重）—— 当前无内置
        └── constraint 谓词（生成 φ 向量）→ data.constraints → MLP 消费
```

**设计原则：知识提供者负责"知道什么"，谓词负责"计算什么描述量/约束"。** 完整的三层分离（Knowledge → Predicate 描述量 → Injection 使用方式）是后续演进方向，当前实现中谓词直接生成 constraint（LUSI φ 向量），尚未独立出 Injection 层。

### 8.9 新增其他组件总结

| 扩展类型 | 在哪个文件写代码 | 注册装饰器 | 是否需要改 bootstrap.py | 是否需要改其他文件 |
|---------|----------------|-----------|:--:|:--:|
| 新模型 | `src/models/<name>.py` | `@MODEL_REGISTRY.decorator("name")` | 是（或 plugins） | 否 |
| 新指标 | `src/validation/metrics.py` | `@METRIC_REGISTRY.decorator("name")` | 否 | 否 |
| 新分拆器 | `src/validation/splitters.py` | `@SPLITTER_REGISTRY.decorator("name")` | 否 | 否 |
| 新特征算子 | `src/operators/features/builtins.py` | `@FEATURE_OPERATOR_REGISTRY.decorator("name")` | 否 | 否 |
| 新任务 | `src/tasks/<name>.py` | `@TASK_REGISTRY.decorator("name")` | 是 | 否 |
| 新调优器 | `src/tuning/<name>.py` | `@TUNER_REGISTRY.decorator("name")` | 是 | 否 |
| 新谓词 | `src/predicates/builtins.py` | `@PREDICATE_REGISTRY.decorator("name")` | 否 | 否 |
| 新知识提供者 | `src/knowledge/providers.py` | `@KNOWLEDGE_REGISTRY.decorator("name")` | 否 | 否 |
| 新研究单元 | `src/data/research_unit_builtins.py` | `@RESEARCH_UNIT_REGISTRY.decorator("name")` | 否 | 否 |
| 新采样器 | `src/data/samplers.py` | `@BACKGROUND_SAMPLER_REGISTRY.decorator("name")` | 否 | 否 |
| 新标签策略 | `src/data/label_strategies.py` | `@LABEL_STRATEGY_REGISTRY.decorator("name")` | 否 | 否 |
| 新权重策略 | `src/data/weight_strategies.py` | `@WEIGHT_STRATEGY_REGISTRY.decorator("name")` | 否 | 否 |
| 新标签细化器 | `src/models/<name>.py` | `@LABEL_REFINER_REGISTRY.decorator("name")` | 是 | 否 |

### 8.10 通过 plugins 扩展组件

也可以把组件放在外部模块，用 `plugins` 列表加载（不修改 bootstrap）：

```yaml
plugins:
  - plugins.my_models

model:
  name: my_model
```

插件模块会在 `validate_config` 时被 import，触发其注册装饰器。

---

## 9. 实验对比指南

> **先确认配置是否真的存在。** 当前仓库实际预置 22 个 `nsw_*.yaml`，且全部是 `train_from_archive_features`。本节中使用 `nsw_*.yaml` 的命令可作为现有归档数据实验的起点；标注为“用户自建模板”的 `raw_gis` / `archive_replay` 示例不是仓库内置命令。涉及 `--set` 的覆盖均为标量；列表、JSON 和列表索引必须写入 YAML。

### 9.1 不同模型对比（固定配置）

使用各模型自己的配置，避免把一个模型的参数误套到另一个模型。以下命令均引用已存在的文件：

```bash
python run.py --config configs/experiments/nsw_rf_train.yaml
python run.py --config configs/experiments/nsw_spe_train.yaml
python run.py --config configs/experiments/nsw_cnn_train.yaml
python run.py --config configs/experiments/nsw_cnn2d.yaml
python run.py --config configs/experiments/nsw_label_spreading.yaml
python run.py --config configs/experiments/nsw_mlp_train.yaml
```

约束变体另见 `nsw_rf_constrained_allones.yaml`、`nsw_spe_constrained_allones.yaml`、`nsw_cnn_pred_allones.yaml`、`nsw_cnn2d_pred_allones.yaml` 和 `nsw_mlp_pred_allones.yaml`。这些是不同训练机制，不应与无约束基线混为同一模型。

### 9.2 可用调优配置对比

当前预置文件中只有 `nsw_mlp_bayes.yaml` 使用 `tuning: bayes`；其余列出的 NSW 配置使用 `tuning: none`。因此，不能把旧文档中的“RF/SPE/CNN 贝叶斯配置”当作现成实验。

```bash
# MLP 的已预置贝叶斯调优配置
python run.py --config configs/experiments/nsw_mlp_bayes.yaml \
    --set experiment.name=nsw_mlp_bayes_compare

# 固定参数基线
python run.py --config configs/experiments/nsw_mlp_train.yaml \
    --set experiment.name=nsw_mlp_fixed_compare
```

若为其他模型增加搜索空间，须先在 YAML 中配置与该模型匹配的 `tuning.params.search_space`，并用 `--validate-only` 检查；不要仅凭旧配置名推断该文件存在。

### 9.3 相同模型多种子实验

`experiment.seed` 会派生 sampling、split、tuning、model 和 dataloader 用途种子。多种子应保持 `experiment.name`（及 `experiment.variant`）不变，只改变 `experiment.seed`：这样 `scripts/aggregate_runs.py` 才能把这些运行归入同一组，按 seed 汇总 mean±std。不要把 seed 拼进 `experiment.name`，否则汇总脚本会按名称拆成每组一个运行，无法得到跨种子统计：

```bash
for seed in 42 123 456 789 2026; do
    python run.py --config configs/experiments/nsw_rf_train.yaml \
        --set experiment.seed=$seed
done

# 运行结束后汇总（各 run 的 manifest.json 已分别记录 seed）
python scripts/aggregate_runs.py --runs-dir outputs/ --output outputs/_summary
```

`experiment.name` 相同不会相互覆盖：每个运行输出目录都会追加微秒级时间戳 run_id。如需有意区分的实验变体，用 `experiment.variant` 而非名称。

多种子只能衡量随机稳定性，不能消除随机点级划分的空间泄漏或 PU 标签偏差。

### 9.4 不同预测网格尺度（`raw_gis`，用户自建模板）

`research_unit.prediction_grid_size` 只在 `raw_gis` 中构建预测网格时生效。仓库没有内置 `raw_gis` YAML，以下仅示意配置方式，不能直接执行：

```yaml
# 用户新建的 raw_gis 配置中
experiment:
  execution_mode: raw_gis
research_unit:
  prediction_grid_size: 0.1
```

复制并改写完整的 raw GIS 配置后，分别使用 0.05、0.1 等值，并核对 CRS、ROI、栅格分辨率和输出面积是否一致。不要用归档特征配置测试这一变量。

### 9.5 不同样本权重（`raw_gis`，用户自建模板）

`label.sample_weight` 只在 `raw_gis` 从 occurrence 的 SIZE_CODE 构建样本时消费。`train_from_archive_features` 使用归档中已固定的 `sample_weight`，修改该节不会重建标签或权重。

```yaml
# 用户自建 raw_gis 配置中的示例
label:
  sample_weight:
    VLG: 0.8
    LGE: 0.6
    MED: 0.4
    SML: 0.2
    OCC: 0.1
    unlabeled: 0.5
```

### 9.6 空间验证与随机验证（`raw_gis`，用户自建模板）

随机留出可作为偏差诊断，但在空间自相关数据上可能过于乐观。空间分拆器要求 `TrainingData.metadata["units"]` 中有投影坐标（米）；经纬度会明确报错。归档特征模式采用归档固定 train/test，不能通过 `--set` 切换为空间划分。

```yaml
# 用户自建 raw_gis 配置中的空间块示例
validation:
  holdout:
    name: spatial_block_holdout
    params:
      block_size_m: 50000
  cross_validation:
    name: spatial_block_kfold
    params:
      block_size_m: 50000
```

### 9.7 谓词约束对比（支持约束的模型，`tuning=none`）

谓词由 `all_ones`、`spatial_box`、`spatial_distance` 或 `combined` 生成约束。当前已有 all_ones 的 NSW 配置，可直接运行；空间谓词配置虽然已随仓库提供，但其归档特征数据缺少 raw GIS 的完整坐标上下文时可能在运行阶段报错，应先验证数据契约。

```bash
# MLP 无约束基线与 all_ones 约束
python run.py --config configs/experiments/nsw_mlp_train.yaml
python run.py --config configs/experiments/nsw_mlp_pred_allones.yaml

# 其他已预置的 all_ones 约束模型
python run.py --config configs/experiments/nsw_cnn2d_pred_allones.yaml
python run.py --config configs/experiments/nsw_rf_constrained_allones.yaml
```

`cnn`、`cnn2d`、`mlp` 使用约束损失；`rf_constrained`、`spe_constrained` 使用 `ConstrainedReweighting` 的样本权重迭代。贝叶斯调优与谓词约束当前显式不兼容。`predicates.items` 是列表，不能用 `--set predicates.items.0.name=...` 修改。

### 9.8 知识注入对比（用户自建模板）

当前 22 个 NSW 配置没有启用 `knowledge`。内置知识提供者是 `empty` 和 `spatial_extent`；若要比较知识注入，需为 `raw_gis`（或满足坐标契约的自定义数据）编写配置：

```yaml
experiment:
  execution_mode: raw_gis
knowledge:
  enabled: true
  items:
    - name: spatial_extent
      params: {}
```

知识提供者先构建知识工件，谓词管线再消费它；启用知识不等于模型自动获得约束。

### 9.9 不同特征算子（`raw_gis`，用户自建模板）

特征算子只在 `raw_gis` 读取 GIS 并执行；归档特征模式要求 `features.operators: []`，因此当前没有可直接运行的归档特征算子对比。用户需复制完整 raw GIS 配置并在 YAML 中改列表：

```yaml
features:
  operators:
    - name: line_distance
      params: {distance_type: geodesic}
    - name: categorical_geology
      params: {}
```

可用算子名以 `python scripts/list_components.py` 的 `feature_operator` 列表为准，并先运行 `--validate-only`。

### 9.10 不同预处理参数（`raw_gis`，用户自建模板）

raw GIS 模式的预处理应在外层划分之后、每个训练折内拟合，以避免泄漏。当前归档特征已经预计算，不能用它比较 `correlation_threshold` 等拟合参数。复制 raw GIS 配置后再比较：

```yaml
preprocess:
  correlation_threshold: 0.7
```

### 9.11 PUB 标签细化（`raw_gis`，用户自建模板）

PUB 依赖可选包并且只在 `raw_gis` 训练折内拟合。当前所有内置 NSW 配置都没有启用 PUB；在 `train_from_archive_features` 中设置 `label_refinement.enabled: true` 会被配置校验拒绝，因为归档标签已经固定。

```yaml
# 用户自建 raw_gis 配置
label_refinement:
  enabled: true
  name: pub
```

### 9.12 不同地区对比

当前仓库实际配置和归档数据均指向 NSW；没有可直接运行的 Lachlan 或其他地区配置。要做地区比较，必须准备另一地区的归档/GIS 数据、坐标参考系和完整 YAML，并保持标签、网格、指标和验证策略可比。不能通过 `--set task.region` 把 NSW 归档变成另一个地区。

### 9.13 参数敏感性分析（归档特征）

RF 的模型参数是标量，可以用现有配置做单因素实验；每次使用不同实验名：

```bash
for n in 10 50 100 200 500; do
    python run.py --config configs/experiments/nsw_rf_train.yaml \
        --set model.params.n_estimators=$n experiment.name=nsw_rf_n_est_${n}
done

for d in 3 5 10 15 20 30; do
    python run.py --config configs/experiments/nsw_rf_train.yaml \
        --set model.params.max_depth=$d experiment.name=nsw_rf_depth_${d}
done
```

固定数据、划分、种子和评价协议；不要把一次运行的最优参数当作无偏测试结论。

### 9.14 实验对比类型总览

| 编号 | 对比类型 | 当前可运行性/模式 | 主要配置节 |
|:----:|---------|------------------|-----------|
| 1 | 不同模型（固定配置） | 可运行；`train_from_archive_features` | `model`（各模型独立 YAML） |
| 2 | 调优配置 | 仅 MLP 有预置 bayes；归档特征 | `tuning.*` |
| 3 | 相同模型多种子 | 可运行；训练模式 | `experiment.seed` |
| 4 | 预测网格尺度 | 用户自建 `raw_gis` | `research_unit.prediction_grid_size` |
| 5 | 样本权重 | 用户自建 `raw_gis` | `label.sample_weight` |
| 6 | 空间 vs 随机验证 | 用户自建 `raw_gis` | `validation.holdout/cross_validation` |
| 7 | 谓词约束 | 可运行的部分配置；支持约束模型 | `predicates.items[]` |
| 8 | 知识注入 | 用户自建配置 | `knowledge.items[]` |
| 9 | 特征集 | 用户自建 `raw_gis` | `features.operators[]` |
| 10 | 预处理参数 | 用户自建 `raw_gis` | `preprocess.*` |
| 11 | PUB 标签细化 | 用户自建 `raw_gis` | `label_refinement.*` |
| 12 | 地区 | 需另备数据与配置 | `dataset.*` + `task.region` |
| 13 | 模型参数敏感性 | 可运行；归档特征 | `model.params.*` |

---

## 10. YAML 配置参考

完整 YAML 结构说明（以已预置的 `nsw_rf_train.yaml` 为例）：

| 配置节 | 功能 | 关键字段 | 是否必须 |
|--------|------|---------|:--:|
| `experiment` | 实验元信息 | `name`, `seed`, `output_dir`, `execution_mode`, `variant` | 是 |
| `task` | 找矿任务 | `name`（`target_area_prediction` 或 `deep_edge_prediction`）, `region` | 是 |
| `dataset` | 数据路径 | `root`, `archive_dir`, `occurrence`, `boundary`, `training_boundary`, `geology`, `magnetic`, `gravity`, `radiometric`, `remote_sensing`, `elevation`, `seismic` | 是 |
| `research_unit` | 研究单元参数 | `type`, `train_positive`, `train_unlabeled`, `prediction_grid_size` | 是 |
| `label` | 标签策略 | `strategy`, `positive_value`, `unlabeled_value`, `sample_weight`（SIZE_CODE 映射） | 是 |
| `features.operators` | 特征算子列表 | 每个算子有 `name` + `params`；archive 模式须为 `[]` | 是 |
| `preprocess` | 预处理参数 | `correlation_threshold`, `categorical_encoding`, `scaling` | 是 |
| `model` | 模型配置 | `name`（注册名）, `params`（模型参数） | 是 |
| `tuning` | 超参数调优 | `name`（`none`/`bayes`）, `params.n_iter`, `params.search_space` | 是 |
| `validation` | 验证方式 | `holdout`, `cross_validation`, `metrics[]`, `primary_metric` | 是 |
| `prediction` | 预测输出 | `score_type`, `normalization`, `export_geotiff`, `target_crs` | 是 |
| `label_refinement` | PUB 标签细化 | `enabled`, `name`, `params` | 否 |
| `knowledge` | 知识注入 | `enabled`, `items[]` | 否 |
| `predicates` | 谓词约束 | `enabled`, `combine`, `items[]` | 否 |
| `plugins` | 外部插件模块 | 模块名列表 | 否 |

### `prediction` 节字段

| 字段 | 取值 | 说明 |
|------|------|------|
| `score_type` | `probability` / `raw_score` / `relative_score` | 输出列语义（见 §4.6） |
| `normalization` | `none` / `minmax` | 仅 `relative_score` 可用 `minmax` |
| `export_geotiff` | `true` / `false` | 是否导出 GeoTIFF |
| `target_crs` | EPSG 整数 或 WKT/proj4 字符串 | **`export_geotiff=true` 时必填**（Lachlan/NSW 归档用 GDA94/EPSG:4283） |

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

每次实验运行后，`outputs/<实验名称>_<run_id>/` 目录包含：

```
nsw_rf_train_20260901_143052_123456/
├── config_resolved.yaml    # ★ 合并后的完整配置（含 DEFAULTS + Phase1迁移 + CLI覆盖）
├── experiment.log          # 结构化日志（包含每个阶段的耗时）
├── manifest.json           # 可复现性清单
│   ├── run_id              #   运行 ID（微秒级时间戳）
│   ├── timestamp / status  #   运行时间与状态（created/running/completed/failed）
│   ├── seed / derived_seeds#   基础种子与派生种子表
│   ├── git_commit / git_dirty  # Git 信息
│   ├── environment         #   Python/平台/核心依赖版本
│   ├── config              #   完整配置
│   ├── components          #   声明的组件列表
│   ├── component_metadata  #   实际执行/标记为预计算的组件
│   ├── input_paths         #   输入文件路径 + SHA256（含 shapefile sidecar/栅格目录）
│   ├── feature_schema / feature_count  #   特征列名/数量
│   ├── split_summary       #   train/test 行数与正类数
│   ├── best_params         #   调优得到的最优参数
│   └── output_files        #   输出文件列表 + SHA256
├── metrics.json            # 评估指标
├── models/
│   ├── model_rf.pkl        # 序列化模型（按适配器可能为 model_spe/cnn/cnn2d/label_spreading/mlp/rf_constrained/spe_constrained.pkl）
│   └── model_pub.pkl       # PUB 标签细化模型（如果启用）
├── predictions/
│   ├── target_probs.csv    # 预测分数表（X, Y, prob/raw_score/relative_score）
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

### 模式能力边界报错

```bash
# 错误：archive_replay mode does not use tuning ...
# 或：train_from_archive_features mode uses archived features; feature operators do not execute ...
# 原因：在某个执行模式下配置了不生效的字段，框架在 validate 阶段硬失败
# 解决：按 §4.3 能力边界表调整配置，或切换到支持该字段的执行模式
```

### CLI 覆盖不生效

```bash
# 先用 --validate-only 检查覆盖后的配置
python run.py --config configs/experiments/nsw_rf_train.yaml \
    --validate-only --set model.name=spe

# 确认输出的 JSON 中 model 字段是否已变为 "spe"
# 如果覆盖失败，检查 key 路径是否正确（使用点号分隔：model.name 不是 model:name）
# 列表/JSON/列表索引路径的 --set 不支持，会直接报错
```

### 模型训练失败（可选依赖）

```bash
# 错误：OptionalDependencyError
# 原因：缺少模型所需的可选依赖
# 解决：
pip install scikit-optimize  # 贝叶斯调优（tuning.name=bayes）
pip install pulearn          # PUB 标签细化
pip install torch            # CNN / MLP (PyTorch)
pip install geopandas shapely rasterio scikit-image  # raw_gis 特征提取
# export_geotiff=true 需要 GDAL Python 绑定（osgeo）
```

### GeoTIFF 导出失败

```bash
# 错误：prediction.target_crs must be set when export_geotiff=true
# 原因：导出 GeoTIFF 需要显式目标 CRS
# 解决：在 prediction 节添加 target_crs（如 NSW 归档用 EPSG:4283），或设 export_geotiff=false
```

### 空间分拆器报错

```bash
# 错误：spatial block splitting requires projected coordinates (meters) ...
# 原因：坐标是经纬度，不能按米为单位分块
# 解决：先把研究单元重投影到 UTM 等投影 CRS

# 错误：Spatial splitter requires coordinate metadata ...
# 原因：该模式没有 units 坐标元数据（如 archive-features 模式）
# 解决：改用 raw_gis 模式
```

### 贝叶斯 + 谓词约束报错

```bash
# 错误：Bayesian tuning with predicate constraints is not yet supported ...
# 原因：BayesSearchCV 无法按折切分 φ 向量
# 解决：使用 tuning.name=none + 支持约束的模型（cnn/cnn2d/mlp 或约束重加权适配器）
```

### 输出目录被覆盖

```bash
# 框架已自动为每次运行添加微秒级时间戳后缀，不会覆盖
# 输出目录格式：outputs/<实验名>_YYYYMMDD_HHMMSS_微秒/
# 可以通过 experiment.name 区分不同实验
python run.py --config ... --set experiment.name=my_unique_name
```

### 空间泄漏风险

使用 `random_holdout` 分拆器时，训练集和测试集的样本可能在空间上相邻或重叠，导致**空间泄漏（spatial leakage）**，使评估指标过于乐观。框架已内置空间分拆器（仅 `raw_gis` 模式）：

```bash
# 用户自建 raw_gis 配置后使用空间块留出法
python run.py --config configs/experiments/my_raw_gis.yaml \
    --set validation.holdout.name=spatial_block_holdout \
    validation.holdout.params.block_size_m=50000

# 用户自建 raw_gis 配置后使用空间块 K-Fold
python run.py --config configs/experiments/my_raw_gis.yaml \
    --set validation.cross_validation.name=spatial_block_kfold \
    validation.cross_validation.params.block_size_m=50000
```

### 如何确认 config_resolved.yaml 是最终参数

`config_resolved.yaml` 中包含的是 `_deep_merge(DEFAULTS, migrated_yaml)` 再经过 CLI `--set` 覆盖后的**最终完整配置**。它记录了：

1. 框架默认值（DEFAULTS）
2. YAML 文件中的配置（覆盖默认值）
3. Phase1 格式迁移结果（如果使用了一期格式）
4. CLI `--set` 覆盖（覆盖 YAML 配置）
5. 解析后的绝对路径（dataset 路径）

所以 `config_resolved.yaml` 是实验实际使用的完整参数，可直接用于复现。

---

## 13. 能力状态与已知限制

### 13.1 能力状态

| 组件/能力 | 状态 | 说明 |
|-----------|------|------|
| `target_area_prediction` | `implemented` | 二维靶区预测任务 |
| `deep_edge_prediction` | `placeholder` | 实例化即报错，无三维能力 |
| `rf` | `implemented` | 归档特征 RF 基线 |
| `spe` | `experimental` | 可训练，但未与参考 SPE 实现完成固定种子一致性验证 |
| `cnn` | `experimental` | 将特征列视为 1D 邻域，尚未完成列置换敏感性评估 |
| `cnn2d` | `experimental` | patch-based 网格模型，网格映射和训练行为仍需科学验证 |
| `label_spreading` | `experimental` | 转导式网格模型；默认用 5 折 OOF 分数拟合内部 Platt 校准器 |
| `mlp` | `experimental` | 支持 LUSI 谓词约束，但损失数值对齐尚未验证 |
| `rf_constrained` | `experimental` | `ConstrainedReweighting` 以样本权重迭代近似约束 |
| `spe_constrained` | `experimental` | `ConstrainedReweighting` 以样本权重迭代近似约束，尚未完成算法验证 |
| 概率校准输出 | **未实现** | 框架没有 `calibrated_probability` score type；Label Spreading 的内部 Platt 校准不改变该契约 |
| MPM 面积指标 | `implemented`（条件性） | 已接入评估，但数据源必须提供 `unit_area` 才会计算 |
| 空间验证 | `implemented`（受模式限制） | 需要投影坐标与 `units` 元数据；归档特征模式不能切换空间划分 |
| 贝叶斯 + 谓词约束 | **未支持** | `BayesSearchCV` 无法按折切分 φ，配置会显式报错 |
| GeoTIFF 模板继承 | 部分 | 已做半像元定位，但尚未完全继承源栅格 transform、分辨率和 nodata |
| Knowledge → Predicate → Injection 三层分离 | 部分 | Knowledge 与 Predicate 已分离，独立 Injection 层尚未实现 |

> 预置的 22 个 NSW 配置全部是 `train_from_archive_features`，没有内置 `archive_replay` 或 `raw_gis` YAML。能力存在不等于当前仓库有可立即运行的示例。

### 13.2 科学使用注意事项

1. **未标注样本 ≠ 已验证负样本。** 当前 `positive_unlabeled_as_zero` 把随机背景点临时编码为 0 类；随机未标注点没有已知矿点排除缓冲、勘探覆盖修正或 verified barren 证据，不能作为真实负样本解释评估指标。
2. **`probability` 不是校准概率。** 未经独立校准集或 OOF 校准，不能按概率阈值做勘查决策；`relative_score`（MinMax）更不能跨区比较。
3. **随机 CV 只用于偏差诊断。** 空间自相关会高估随机点划分下的泛化能力；正式研究结论应使用空间分块/分组验证并分开报告。
4. **`archive_replay` 不是计算复现。** 它复制归档模型与预测、不重算指标，只能证明工件一致性，不能证明当前代码能从原始 GIS 复现相同结果，回放结果不应进入新模型性能对比表。
5. **不同模型应使用各自独立配置与搜索空间**，不应只改 `model.name`（RF/SPE/CNN/MLP 的参数空间不同）。
6. **调参只使用验证数据**，最终测试集不参与模型/谓词/阈值/超参数选择。

### 13.3 当前版本定位

本版本是**局部工程修复版 / experimental development build**：框架组件化、配置能力边界、评分语义和复现工件已有实现，但尚未达到“二维研究就绪（research_ready）”。在完成空间验证、PU 标签语义、概率校准和原始 GIS 端到端复现前，不应把当前归档特征随机留出结果直接作为可发表的区域泛化结论。遗留问题详见 `POST_REFACTOR_REVIEW.md` 与 `CODEING_UPDATE_MD/` 中的评审和验收材料。

- **当前测试状态**：在当前工作树运行 `python -m pytest -q` 得到 `16 failed, 75 passed, 3 warnings, 26 subtests passed`。16 个失败均来自 `tests/test_framework.py` 仍引用已删除的 `configs/experiments/lachlan_rf_phase2.yaml` 或 `lachlan_rf_baseline.yaml`，不是模型指标失败；本次文档更新不修改测试文件。

---

## 参考依据

- Time-Series-Library（统一入口与脚本化实验组织）：https://github.com/thuml/Time-Series-Library
- TSIL（单次训练/搜索/多种子/谓词不变量）：https://github.com/Supporon/TSIL
- scikit-learn 数据泄漏与 Pipeline：https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn 嵌套交叉验证：https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- scikit-learn StratifiedGroupKFold：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- scikit-learn 概率校准：https://scikit-learn.org/stable/modules/calibration.html
- MPM 空间 CV、PU 数据特点及 prediction/success-rate 指标：https://link.springer.com/article/10.1007/s10618-018-00607-x
- imbalanced-ensemble（SPE 参考实现）：https://imbalanced-ensemble.readthedocs.io/en/stable/
