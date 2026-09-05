# MPM_codex — Phase 2

MPM_codex 是面向**矿产远景预测（Mineral Prospectivity Mapping, MPM）**的工程化实验框架。Phase 2 在 Phase-1 归档数据迁移的基础上，将数据、特征、模型、验证、知识和谓词约束拆分为可注册、可配置、可验证的组件。

> **当前版本定位**：这是局部工程修复版 / experimental development build。它适合归档特征上的模型对比、框架开发和合成 `raw_gis` 测试；当前仓库没有预置 `archive_replay` 或 `raw_gis` 实验配置，且现有随机留出与 PU 标签语义不足以直接支持可发表的科学结论。

## 快速开始

在本目录（`MPM_codex_phase2`）下执行：

```bash
# 查看已注册的 9 个主注册表
python scripts/list_components.py

# 只加载并校验配置（不访问归档数据）
python run.py --config configs/experiments/nsw_rf_train.yaml --validate-only

# 从 NSW 归档特征重新训练 RF（无调参）
python run.py --config configs/experiments/nsw_rf_train.yaml

# 运行转导式 Label Spreading（需要网格注入）
python run.py --config configs/experiments/nsw_label_spreading.yaml

# 运行 2D CNN 网格模型
python run.py --config configs/experiments/nsw_cnn2d.yaml

# 运行带 all_ones 谓词约束的 MLP（约束实验使用 tuning=none）
python run.py --config configs/experiments/nsw_mlp_pred_allones.yaml

# 运行约束重加权 RF
python run.py --config configs/experiments/nsw_rf_constrained_allones.yaml

# 只覆盖标量配置，不修改 YAML
python run.py --config configs/experiments/nsw_rf_train.yaml \
  --set experiment.seed=123 experiment.name=nsw_rf_seed_123

# 运行测试（当前有历史测试引用已删除配置，详见下文）
python -m pytest -q
```

所有命令都应从 `MPM_codex_phase2/` 目录执行。归档配置默认读取相邻目录中的 `../EarthByte-MPM_Lachlan_Porphyry/Datasets/Outputs_Cu_NSW_v1.6`；若数据目录不在该位置，请在 YAML 中修改 `dataset.archive_dir`。

## 当前能力

- **三种执行模式**：
  - `archive_replay`：复制并核验归档模型与预测，不训练、不重算指标；
  - `train_from_archive_features`：在归档特征和归档固定 train/test 划分上重新训练，可更换模型、指标和调参；
  - `raw_gis`：从原始 GIS 数据构建研究单元和特征并训练，但当前仓库没有预置该模式的实验 YAML，需用户自行编写并准备数据。
- **8 个模型适配器**：`rf`、`spe`、`cnn`、`cnn2d`、`label_spreading`、`mlp`、`rf_constrained`、`spe_constrained`。
- **注册表驱动**：模型、任务、特征算子、知识提供者、谓词、分拆器、指标、调优器、标签细化器，以及研究单元/背景采样/标签/权重策略均按名称装配。
- **空间验证**：提供 `spatial_block_kfold`、`spatial_group_kfold`、`spatial_block_holdout`；要求投影坐标（米），且主要适用于 `raw_gis` 的坐标元数据。
- **评分契约**：支持 `probability`（未校准正类概率）、`raw_score`（`decision_function`）和 `relative_score`（可 MinMax）；没有实现 `calibrated_probability` 输出类型。
- **可复现工件**：每次运行使用微秒级 `run_id` 创建独立输出目录；manifest 记录派生种子、配置、划分摘要、组件、Git/环境信息及输入输出哈希。

## 模型能力矩阵

| 模型 | 实现 | `supports_constraints` | `grid_model` | 主要说明 |
|---|---|:---:|:---:|---|
| `rf` | sklearn `RandomForestClassifier` | 否 | 否 | 归档特征基线 |
| `spe` | 自包含 `SelfPacedEnsemble` | 否 | 否 | 面向类别不平衡的集成 |
| `cnn` | PyTorch 1D CNN | 是 | 否 | 将特征列作为 1D 序列；实验性 |
| `cnn2d` | PyTorch patch-based 2D CNN | 是 | 是 | 从预测网格提取局部 patch；实验性 |
| `label_spreading` | sklearn `LabelSpreading` 包装器 | 否 | 是 | 转导式使用有效预测网格；内部可用 OOF Platt scaling |
| `mlp` | PyTorch MLP | 是 | 否 | 可消费 LUSI 谓词约束；实验性 |
| `rf_constrained` | RF + `ConstrainedReweighting` | 是 | 否 | 用样本权重迭代近似约束，而非梯度损失 |
| `spe_constrained` | SPE + `ConstrainedReweighting` | 是 | 否 | 用样本权重迭代近似约束；实验性 |

`grid_model=True` 的模型通过适配器的 `fit_params()` 接收 `grid`、`inside` 和 `train_cells`，并用 `set_predict_cells()` 将网格单元映射到点预测。`supports_constraints=True` 只表示框架允许传入谓词约束，不等于算法和论文实现已完成一致性验证。

## 配置清单

当前仓库实际预置 **22 个** NSW 配置，全部为 `train_from_archive_features`，全部使用 `relative_score + minmax`：

- CNN2D：`nsw_cnn2d.yaml`、`nsw_cnn2d_pred_allones.yaml`、`nsw_cnn2d_pred_spatial_box.yaml`、`nsw_cnn2d_pred_spatial_distance.yaml`
- CNN：`nsw_cnn_train.yaml`、`nsw_cnn_pred_allones.yaml`
- Label Spreading：`nsw_label_spreading.yaml`
- MLP：`nsw_mlp_train.yaml`、`nsw_mlp_bayes.yaml`、`nsw_mlp_pred_allones.yaml`、`nsw_mlp_pred_spatial_box.yaml`、`nsw_mlp_pred_spatial_box_tuned.yaml`、`nsw_mlp_pred_spatial_distance.yaml`、`nsw_mlp_tuned_allones.yaml`、`nsw_mlp_tuned_spatial_box.yaml`、`nsw_mlp_tuned_spatial_distance.yaml`
- RF：`nsw_rf_train.yaml`、`nsw_rf_constrained_allones.yaml`、`nsw_rf_constrained_spatial_box.yaml`
- SPE：`nsw_spe_train.yaml`、`nsw_spe_constrained_allones.yaml`、`nsw_spe_constrained_spatial_box.yaml`

> 以上清单共 22 个文件；可用 `find configs/experiments -maxdepth 1 -name 'nsw_*.yaml' | sort` 复核。仓库没有 `lachlan_*.yaml`、`archive_replay` 或 `raw_gis` 配置；旧文档中的这些命令不能直接运行。

## 能力状态与已知限制

| 能力 | 状态 | 说明 |
|---|---|---|
| `target_area_prediction` | `implemented` | 二维任务与归档特征流程已实现 |
| RF / SPE / CNN / MLP | 已实现 / 部分实验性 | 可训练；SPE、CNN 及约束损失仍需算法级验证 |
| CNN2D / Label Spreading | 已实现 / 实验性 | 网格模型协议已实现；Label Spreading 的 Platt 校准仅用于模型内部网格分数 |
| `rf_constrained` / `spe_constrained` | 已实现 / 实验性 | 通过 `ConstrainedReweighting` 迭代样本权重，不是梯度约束 |
| 空间分块验证 | 已实现但受模式限制 | 需要 `raw_gis` 的投影坐标和 `units` 元数据 |
| `calibrated_probability` | 未实现 | `probability` 仍是未校准概率；`relative_score` 不是概率 |
| `deep_edge_prediction` | `placeholder` | 无三维/深部预测能力，实例化会报错 |
| 完整原始 GIS 示例 | 未预置 | 代码路径存在，但需用户自行提供配置、GIS 数据和可选依赖 |

科学使用时还需注意：

1. 当前 `positive_unlabeled_as_zero` 将未标注样本编码为 0，但它们不是经过验证的 barren/无矿负样本。
2. 归档特征配置使用 `random_holdout`，点级随机划分可能产生空间泄漏；现有指标不能直接解释为区域外推能力。
3. `archive_replay` 是归档工件回放，不是从原始 GIS 重新计算的复现。
4. `relative_score` 经过 MinMax 后只表示本次预测范围内的相对排序，不应跨区域或跨运行当作概率比较。
5. 当前测试套件并非全绿：`python -m pytest -q` 的现状为 **16 failed, 75 passed, 3 warnings, 26 subtests passed**。失败均源于历史 `tests/test_framework.py` 仍引用仓库中已不存在的 `lachlan_rf_phase2.yaml` / `lachlan_rf_baseline.yaml`；本次文档更新不修改测试文件。

## 输出目录

每次运行都会在配置的 `output_dir` 后追加微秒级 `run_id`，例如：

```text
outputs/nsw_rf_train_<YYYYMMDD_HHMMSS_微秒>/
├── config_resolved.yaml
├── manifest.json
├── metrics.json
├── experiment.log
├── models/
├── predictions/
├── intermediate/
└── figures/
```

详见完整手册 [`GUIDE.md`](GUIDE.md)。架构说明 [`docs/PHASE2_ARCHITECTURE_CN.md`](docs/PHASE2_ARCHITECTURE_CN.md) 仍包含部分历史描述，使用时以源码、当前配置和本 README/GUIDE 为准。

## 相关文档

- [`GUIDE.md`](GUIDE.md)：完整使用、扩展和实验对比手册
- [`POST_REFACTOR_REVIEW.md`](POST_REFACTOR_REVIEW.md)：复现评审与遗留问题
- [`CODEING_UPDATE_MD/MPM_CODE_OPTIMIZATION_AND_ACCEPTANCE_SPEC.md`](../CODEING_UPDATE_MD/MPM_CODE_OPTIMIZATION_AND_ACCEPTANCE_SPEC.md)：验收规格
