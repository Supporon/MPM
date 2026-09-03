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

| 能力 | 状态 |
|------|------|
| 二维靶区预测 `target_area_prediction` | `implemented` |
| 三种执行模式（replay / archive-features / raw_gis） | `implemented`，且配置校验会**拒绝不生效的字段** |
| RF 基线 | `implemented` |
| SPE / CNN / MLP | `implemented`，但算法一致性尚未验证 → 科研上标 `experimental` |
| MLP + LUSI 谓词约束损失 | `implemented`，数值对齐尚未验证 → `experimental` |
| 概率校准（`calibrated_probability`） | **未实现**（`probability` 为未校准正类概率） |
| 深边部预测 `deep_edge_prediction` | `placeholder`（无三维能力，实例化即报错） |

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

# 验证配置是否正确（不访问数据文件，快速检查）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml --validate-only

# 归档回放（archive_replay：复制归档模型与预测，不训练）
python run.py --config configs/experiments/lachlan_rf_baseline.yaml

# 从归档特征重新训练 RF（无调参，最快）
python run.py --config configs/experiments/lachlan_rf_train.yaml

# 从归档特征重新训练 RF（贝叶斯调参）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml

# 从归档特征重新训练 SPE（无调参 / 带调优）
python run.py --config configs/experiments/lachlan_spe_notune.yaml
python run.py --config configs/experiments/lachlan_spe.yaml

# 从归档特征重新训练 CNN（带调优） / MLP（无调参）
python run.py --config configs/experiments/lachlan_cnn.yaml
python run.py --config configs/experiments/lachlan_mlp.yaml

# 原始 GIS 端到端（冒烟版：完整栅格 / fixture 版：极小栅格子集，更快）
python run.py --config configs/experiments/lachlan_rf_raw_gis_smoke.yaml
python run.py --config configs/experiments/lachlan_rf_raw_gis_fixture.yaml

# 运行测试
python -m pytest -q
```

> 各配置文件的执行模式、模型、调优方式见 [§2.1](#21-内置实验配置一览)。

### 2.1 内置实验配置一览

| 配置文件 | 执行模式 | 模型 | 调优 | 说明 |
|---------|:--:|:--:|:--:|------|
| `lachlan_rf_baseline.yaml` | `archive_replay` | rf | — | Lachlan 归档回放 |
| `nsw_rf_baseline.yaml` | `archive_replay` | rf | — | NSW 归档回放 |
| `lachlan_rf_train.yaml` | `train_from_archive_features` | rf | none | 无调参重训 |
| `lachlan_rf_simple_train.yaml` | `train_from_archive_features` | rf | none | 无调参重训（权重 VLG=0.8） |
| `lachlan_rf_phase2.yaml` | `train_from_archive_features` | rf | bayes（n_iter=1000） | 完整 RF 调参 + GeoTIFF 导出 |
| `lachlan_rf_pub_train.yaml` | `train_from_archive_features` | rf | bayes（n_iter=50） | 调参重训（`label_refinement` 已禁用） |
| `lachlan_spe.yaml` | `train_from_archive_features` | spe | bayes（n_iter=100） | SPE 调参 |
| `lachlan_spe_notune.yaml` | `train_from_archive_features` | spe | none | SPE 固定参数 |
| `lachlan_cnn.yaml` | `train_from_archive_features` | cnn | bayes（n_iter=50） | CNN 调参 |
| `lachlan_mlp.yaml` | `train_from_archive_features` | mlp | none（默认） | MLP 基线（LUSI 需 `tuning=none`） |
| `lachlan_rf_raw_gis.yaml` | `raw_gis` | rf | none | 完整 GIS 数据，预测网格 0.05 |
| `lachlan_rf_raw_gis_smoke.yaml` | `raw_gis` | rf | none | 完整 GIS 数据，预测网格 0.2（更快） |
| `lachlan_rf_raw_gis_fixture.yaml` | `raw_gis` | rf | none | 栅格指向 `tests/fixtures/gis/rasters`（CI 可用） |

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
│   │   ├── mlp.py                            #   MLP（PyTorch）+ LUSI 谓词约束加权损失
│   │   └── pu.py                             #   PUB 标签细化器（legacy 兼容）
│   │
│   ├── training/                             # 共享 PyTorch 训练与损失（CNN/MLP 复用）
│   │   ├── losses.py                         #   LUSI 谓词约束加权损失（weighted_mse_with_predicate）
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
│   ├── tasks/                                # 找矿任务定义
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
│   └── baseline/                             #   基线 shell（lachlan_rf.sh / nsw_rf.sh）
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

所有实验都通过 `run.py` 执行，通过 `--config` 参数指定配置文件：

```bash
python run.py --config configs/experiments/lachlan_rf_train.yaml
```

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
| `archive_replay` | 回归验证，确认框架能复现历史结果 | 归档 CSV | 归档 .pkl 文件直接复制 | 否 |
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
| `MODEL_REGISTRY` | 模型 | `rf`, `spe`, `cnn`, `mlp` | `model.name` |
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
- `raw_score` 需要模型暴露 `decision_function`（如 SVM/逻辑回归）；RF/SPE/CNN/MLP 只暴露 `predict_proba`，应使用 `probability` 或 `relative_score`。
- 当前 **没有实现 `calibrated_probability`**：`probability` 是未校准的正类概率，不能直接当作成矿概率阈值使用。

### 4.7 随机种子派生（SeedContext）

框架从实验基础种子 `experiment.seed` 派生出互不干扰的用途种子（`src/utils/seeds.py`），保证改某个阶段的随机性不影响其他阶段：

```text
experiment.seed
    ├── sampling_seed    # 未标注点采样
    ├── split_seed       # 外层划分
    ├── tuning_seed      # 调优（BayesSearchCV）
    ├── model_seed       # 模型 random_state（RF/SPE/CNN/MLP 均直接覆盖）
    └── dataloader_seed  # PyTorch DataLoader
```

因此 `--set experiment.seed=N` 现在会真正改变模型的随机性（RF 的 `build()` 用 `resolved["random_state"] = seed` 直接覆盖，而非 `setdefault`）。派生种子表会写入 manifest 的 `derived_seeds`。

---

## 5. 基本使用方式

### 5.1 运行实验

```bash
# 完整运行一个实验
python run.py --config configs/experiments/lachlan_rf_train.yaml

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
python run.py --config configs/experiments/lachlan_rf_train.yaml --set model.name=spe

# 多参数覆盖：换调优器 + 调参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.name=none tuning.params.n_iter=200

# 覆盖调优参数
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.params.n_iter=500

# 覆盖实验级参数
python run.py --config configs/experiments/lachlan_rf_train.yaml \
    --set experiment.seed=999 experiment.name=my_custom_exp

# 覆盖验证参数
python run.py --config configs/experiments/lachlan_rf_train.yaml \
    --set validation.holdout.params.test_size=0.3 validation.primary_metric=roc_auc

# 开关型参数
python run.py --config configs/experiments/lachlan_rf_train.yaml \
    --set prediction.export_geotiff=false

# 配合 validate-only 预览覆盖效果
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
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
# 配置中写的是: output_dir: outputs/lachlan_rf_train
# 实际创建的目录: outputs/lachlan_rf_train_20260901_143052_123456/
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

### 7.2 四种内置模型

| 模型 | 实现 | `supports_constraints` | 依赖 | 说明 |
|------|------|:--:|------|------|
| `rf` | sklearn `RandomForestClassifier` | 否 | 核心 | 主要基线；`random_state=seed` 直接覆盖 |
| `spe` | 自包含 `SelfPacedEnsemble`（ICDE 2020） | 否 | 核心 | 面向高度不平衡；基分类器默认决策树 |
| `cnn` | PyTorch 1D CNN | 否 | `torch` | 将 138 维特征视为 1D 信号；`experimental` |
| `mlp` | PyTorch MLP | **是** | `torch` | 唯一支持谓词约束/LUSI 加权损失；`experimental` |

### 7.3 LUSI 谓词约束加权损失（MLP）

`mlp` 适配器支持在损失中引入谓词统计不变量（Vapnik & Izmailov 的 LUSI，实现参考 TSIL）：

```
loss = τ̂ · MSE + τ · (1/N) · ‖φ̃ᵀ e‖²
```

- 当存在谓词约束时使用 LUSI 加权 MSE（在 sigmoid 概率上计算）；无约束时使用标准 `BCEWithLogitsLoss`。
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

以 XGBoost 为例（**教程示例**：`xgb` 组件与 `lachlan_xgb.yaml` 需你自行创建，当前仓库未预置）：

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

**第三步：编写 YAML 配置** — `configs/experiments/lachlan_xgb.yaml`

```yaml
experiment:
  name: lachlan_xgb
  seed: 2026
  output_dir: outputs/lachlan_xgb
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

**第四步：运行**

```bash
python run.py --config configs/experiments/lachlan_xgb.yaml --validate-only
python run.py --config configs/experiments/lachlan_xgb.yaml
```

**支持约束的模型**：如需消费谓词约束，设置 `supports_constraints = True`，并在 `fit_params(data)` 中返回约束参数（参考 `src/models/mlp.py`）。声明支持约束却未实现 `fit_params(data)` 会显式失败。

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

- `kind="constraint"`：生成 φ 向量写入 `TrainingData.constraints`，供支持 LUSI 约束的模型（`mlp`）消费。内置 `all_ones`、`spatial_box`、`spatial_distance`、`combined`。
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

> 注意：`positive_constraint`、`weight_adjustment` 等是文档示例组件，**当前未内置注册**，需自行注册后方可使用。RF/SPE/CNN 不支持约束（`supports_constraints=False`），配置了约束谓词会报错。

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

> **重要前提**：不同实验变体只有在其生效的执行模式下才有效。下表列出各对比类型与所需模式/配置，命令示例均为可直接运行的形式；涉及 `--set` 的覆盖都是标量。

### 9.1 不同模型对比（各自固定配置）

**目的：** 比较不同模型在各自推荐参数下的性能差异。不同模型的参数空间不同，应使用各自独立的 base config，而不是只改 `model.name`。

```bash
# RF 固定参数（无调参）
python run.py --config configs/experiments/lachlan_rf_train.yaml \
    --set experiment.name=compare_rf_fixed

# SPE 固定参数（无调参）
python run.py --config configs/experiments/lachlan_spe_notune.yaml \
    --set experiment.name=compare_spe_fixed

# MLP 固定参数（无调参）
python run.py --config configs/experiments/lachlan_mlp.yaml \
    --set experiment.name=compare_mlp_fixed
```

### 9.2 不同模型对比（带调优）

**目的：** 比较不同模型在各自搜索空间、相同调参预算下的性能上限。

```bash
# RF + 贝叶斯调优
python run.py --config configs/experiments/lachlan_rf_phase2.yaml \
    --set tuning.params.n_iter=200 experiment.name=compare_rf_tuned

# SPE + 贝叶斯调优
python run.py --config configs/experiments/lachlan_spe.yaml \
    --set tuning.params.n_iter=200 experiment.name=compare_spe_tuned

# CNN + 贝叶斯调优
python run.py --config configs/experiments/lachlan_cnn.yaml \
    --set tuning.params.n_iter=100 experiment.name=compare_cnn_tuned
```

### 9.3 相同模型多种子点实验

**目的：** 评估模型的随机稳定性，报告均值±标准差。现在修改 `experiment.seed` 会真正改变 RF/SPE/CNN/MLP 的随机性。

```bash
for seed in 42 123 456 789 2026; do
    python run.py --config configs/experiments/lachlan_rf_train.yaml \
        --set experiment.seed=$seed experiment.name="rf_seed_${seed}"
done
```

### 9.4 不同预测网格尺度实验（`raw_gis`）

**目的：** 比较不同预测网格精度。`research_unit.prediction_grid_size` 只在 `raw_gis` 模式生效。

```bash
# 默认网格精度 0.05
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set experiment.name=ru_grid_005

# 粗网格 0.1
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set research_unit.prediction_grid_size=0.1 experiment.name=ru_grid_010

# 细网格 0.025
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set research_unit.prediction_grid_size=0.025 experiment.name=ru_grid_0025
```

### 9.5 不同样本权重策略实验（`raw_gis`）

**目的：** 比较不同样本权重设置。`label.sample_weight` 只在 `raw_gis` 模式生效（从 SIZE_CODE 构造）。

```bash
# 默认权重
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set experiment.name=label_default

# 高权重（强调大矿）
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set label.sample_weight.VLG=0.8 label.sample_weight.LGE=0.6 label.sample_weight.MED=0.4 \
    experiment.name=label_high_weight

# 均匀权重
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set label.sample_weight.VLG=0.5 label.sample_weight.LGE=0.5 label.sample_weight.MED=0.5 \
    label.sample_weight.SML=0.5 label.sample_weight.OCC=0.5 experiment.name=label_uniform
```

### 9.6 空间验证 vs 随机验证（`raw_gis`）

**目的：** 比较空间分块与随机点划分对评估结果的影响。空间分拆器只在 `raw_gis` 模式可用，且需投影坐标（米）。

```bash
# 随机留出（基线，存在空间泄漏风险）
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set experiment.name=val_random

# 空间块留出法
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set validation.holdout.name=spatial_block_holdout \
    validation.holdout.params.block_size_m=50000 \
    experiment.name=val_spatial_holdout

# 空间块 K-Fold 交叉验证
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set validation.cross_validation.name=spatial_block_kfold \
    validation.cross_validation.params.block_size_m=50000 \
    experiment.name=val_spatial_kfold
```

> 空间分拆器要求 `TrainingData.metadata["units"]` 中的 X/Y 为投影坐标；经纬度会触发明确错误，需先重投影到 UTM 等度量 CRS。

### 9.7 谓词约束对比（MLP + LUSI，`tuning=none`）

**目的：** 比较不同谓词约束（LUSI 统计不变量）对 MLP 的影响。约束谓词只能由 `mlp` 消费，且需 `tuning=none`（贝叶斯 + 约束未支持）。

```bash
# 无谓词（基线）
python run.py --config configs/experiments/lachlan_mlp.yaml \
    --set experiment.name=pred_none

# all_ones 控制谓词（验证链路）
python run.py --config configs/experiments/lachlan_mlp.yaml \
    --set predicates.enabled=true experiment.name=pred_all_ones
```

> 注意：谓词 `items` 是列表，`--set` 不支持列表值或列表索引。要配置 `all_ones`/`spatial_box` 等具体谓词，需复制 `lachlan_mlp.yaml` 为专用配置，在 YAML 中写入 `predicates.items`。`spatial_box`/`spatial_distance` 需要坐标，只能在 `raw_gis` 模式（或带坐标元数据的场景）使用。

### 9.8 知识注入对比

**目的：** 比较知识提供者对模型的影响。知识 `items` 是列表，需专用 YAML。

```bash
# 无知识注入（基线）
python run.py --config configs/experiments/lachlan_mlp.yaml \
    --set experiment.name=knowledge_none

# 使用 spatial_extent 知识（需 raw_gis 模式才有坐标元数据）
# 需复制 lachlan_rf_raw_gis.yaml 为专用配置，写入 knowledge.items: [{name: spatial_extent, params: {}}]
```

### 9.9 不同特征算子实验（`raw_gis`，专用 YAML）

**目的：** 比较不同特征集。特征算子只在 `raw_gis` 模式执行；`--set` 不支持列表值。需以 `lachlan_rf_raw_gis.yaml` 为模板，自行创建专用 YAML（如 `lachlan_rf_feat_geology.yaml`），在其中修改 `features.operators` 列表。

```yaml
# 例如：仅地质 + 线距离特征
features:
  operators:
    - name: line_distance
      params: {distance_type: geodesic}
    - name: categorical_geology
      params: {}
```

```bash
python run.py --config configs/experiments/lachlan_rf_feat_geology.yaml \
    --set experiment.name=feat_geology_only
```

### 9.10 不同预处理参数实验（`raw_gis`）

**目的：** 比较相关性筛选阈值。预处理只在 `raw_gis` 模式拟合（fold-safe）。

```bash
# 默认相关性阈值 0.7
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set preprocess.correlation_threshold=0.7 experiment.name=prep_corr_07

# 更严格 0.5
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set preprocess.correlation_threshold=0.5 experiment.name=prep_corr_05
```

### 9.11 不同标签细化（PUB）实验（`raw_gis`）

**目的：** 比较是否使用 PUB 标签细化。PUB 只在 `raw_gis` 模式执行，且仅在训练折内拟合。

```bash
# 启用 PUB（label_refinement）
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set label_refinement.enabled=true experiment.name=pub_enabled

# 禁用 PUB（基线）
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set label_refinement.enabled=false experiment.name=pub_disabled
```

> `train_from_archive_features` 模式下 `label_refinement.enabled=true` 会在配置校验阶段被拒绝（归档标签已固定）。

### 9.12 不同地区对比实验

```bash
# Lachlan 地区
python run.py --config configs/experiments/lachlan_rf_train.yaml \
    --set experiment.name=region_lachlan

# NSW 地区（归档回放）
python run.py --config configs/experiments/nsw_rf_baseline.yaml \
    --set experiment.name=region_nsw
```

### 9.13 参数敏感性分析

```bash
# 分析 RF n_estimators 的影响（无调参 + 覆盖模型参数）
for n in 10 50 100 200 500; do
    python run.py --config configs/experiments/lachlan_rf_train.yaml \
        --set model.params.n_estimators=$n experiment.name="sensitivity_n_est_${n}"
done

# 分析 RF max_depth 的影响
for d in 3 5 10 15 20 30; do
    python run.py --config configs/experiments/lachlan_rf_train.yaml \
        --set model.params.max_depth=$d experiment.name="sensitivity_depth_${d}"
done
```

### 9.14 实验对比类型总览

| 编号 | 对比类型 | 生效模式 | 变动的 YAML 配置节 |
|:----:|---------|:--:|-------------------|
| 1 | 不同模型（固定配置） | archive-features | `model.name`（用各模型独立 YAML） |
| 2 | 不同模型（带调优） | archive-features | `model.name` + `tuning.*` |
| 3 | 相同模型多种子 | 任意训练模式 | `experiment.seed` |
| 4 | 不同预测网格尺度 | raw_gis | `research_unit.prediction_grid_size` |
| 5 | 不同样本权重 | raw_gis | `label.sample_weight.*` |
| 6 | 空间 vs 随机验证 | raw_gis | `validation.holdout/cross_validation.name` |
| 7 | 谓词约束（LUSI） | mlp + tuning=none | `predicates.items[]`（专用 YAML） |
| 8 | 知识注入 | raw_gis | `knowledge.items[]`（专用 YAML） |
| 9 | 不同特征集 | raw_gis | `features.operators[]`（专用 YAML） |
| 10 | 不同预处理参数 | raw_gis | `preprocess.*` |
| 11 | 不同标签细化（PUB） | raw_gis | `label_refinement.enabled` |
| 12 | 不同地区 | 各自配置 | `dataset.*` + `task.region` |
| 13 | 参数敏感性 | archive-features | `model.params.*` |

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
lachlan_rf_train_20260901_143052_123456/
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
│   ├── model_rf.pkl        # 序列化模型（或 model_spe.pkl / model_cnn.pkl / model_mlp.pkl）
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
python run.py --config configs/experiments/lachlan_rf_train.yaml \
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
# 解决：在 prediction 节添加 target_crs（如 Lachlan 用 4283），或设 export_geotiff=false
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
# 解决：使用 tuning.name=none + 支持约束的模型（mlp）
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
# 使用空间块留出法
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
    --set validation.holdout.name=spatial_block_holdout \
    validation.holdout.params.block_size_m=50000

# 使用空间块 K-Fold
python run.py --config configs/experiments/lachlan_rf_raw_gis.yaml \
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
| `target_area_prediction` | `implemented` | 二维靶区预测 |
| `deep_edge_prediction` | `placeholder` | 实例化即报错，无三维能力 |
| RF | `implemented` | 主要基线 |
| SPE | `experimental` | 未与 `imbalanced-ensemble.SelfPacedEnsembleClassifier` 做固定种子一致性验证 |
| CNN | `experimental` | 将任意列顺序视为 1D 邻域，未做列置换敏感性实验 |
| MLP + LUSI | `experimental` | LUSI 损失数值对齐未验证（`(1/N)·‖φ̃ᵀe‖²` vs TSIL `‖(1/B)·Φᵀe‖²`） |
| 概率校准 | **未实现** | 无 `calibrated_probability`、calibration curve、Brier/log loss |
| MPM 面积指标 | `implemented`（未接入数据） | `mpm_metrics.py` 已接入 `evaluate_classifier`，但需数据源提供 `unit_area` 才会计算 |
| 空间验证 | `implemented`（仅 raw_gis） | 需要投影坐标与 units 元数据 |
| 贝叶斯 + 谓词约束 | **未支持** | 显式报错，需 constraint-aware CV splitter |
| GeoTIFF 模板继承 | 部分 | 半像元定位已完成；尚未完全继承源栅格 transform/分辨率/nodata |
| Knowledge → Predicate → Injection 三层分离 | 部分 | Knowledge 与 Predicate 已分离；Injection 层未独立 |

### 13.2 科学使用注意事项

1. **未标注样本 ≠ 已验证负样本。** 当前 `positive_unlabeled_as_zero` 把随机背景点临时编码为 0 类；随机未标注点没有已知矿点排除缓冲、勘探覆盖修正或 verified barren 证据，不能作为真实负样本解释评估指标。
2. **`probability` 不是校准概率。** 未经独立校准集或 OOF 校准，不能按概率阈值做勘查决策；`relative_score`（MinMax）更不能跨区比较。
3. **随机 CV 只用于偏差诊断。** 空间自相关会高估随机点划分下的泛化能力；正式研究结论应使用空间分块/分组验证并分开报告。
4. **`archive_replay` 不是计算复现。** 它复制归档模型与预测、不重算指标，只能证明工件一致性，不能证明当前代码能从原始 GIS 复现相同结果，回放结果不应进入新模型性能对比表。
5. **不同模型应使用各自独立配置与搜索空间**，不应只改 `model.name`（RF/SPE/CNN/MLP 的参数空间不同）。
6. **调参只使用验证数据**，最终测试集不参与模型/谓词/阈值/超参数选择。

### 13.3 当前版本定位

按验收规格（`CODEING_UPDATE_MD/MPM_CODE_OPTIMIZATION_AND_ACCEPTANCE_SPEC.md`）与复现评审（`POST_REFACTOR_REVIEW.md`），本版本为**局部工程修复版 / experimental development build**：已完成 P0（无泄漏预处理、模式能力边界、评分语义、验证器防假阳性、空间验证、CRS/ROI/GeoTIFF 硬约束）与 P1（复现工件、MLP/Predicate/LUSI 链路）的主要修复，但尚未达到“二维研究就绪（research_ready）”。遗留待办详见 `POST_REFACTOR_REVIEW.md` 的 backlog。

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
