# MPM_codex — Phase 2

该目录是在 phase-1 基线迁移之上的二阶段工程化重构，也是后续“可复现性与科学可信度”修复（P0/P1）之后的最新版本。核心目标是把找矿预测实验拆成**可注册、可配置、可验证**的组件，而不是继续在 `Experiment` 中增加算法分支。

## 快速开始

```bash
# 查看已注册组件
python scripts/list_components.py

# 只验证配置（不访问数据文件）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml --validate-only

# 归档回放（复制归档模型与预测，不训练）
python run.py --config configs/experiments/lachlan_rf_baseline.yaml

# 从归档特征重新训练 RF（无调参，最快）
python run.py --config configs/experiments/lachlan_rf_train.yaml

# 从归档特征重新训练 RF（贝叶斯调参）
python run.py --config configs/experiments/lachlan_rf_phase2.yaml

# 原始 GIS 端到端（需要 geopandas/rasterio 等可选依赖）
python run.py --config configs/experiments/lachlan_rf_raw_gis_fixture.yaml

# 由配置生成 shell 命令
python scripts/render_run_sh.py --config configs/experiments/lachlan_rf_train.yaml

# 单元/集成测试
python -m pytest -q
```

## 当前能力

- **三种执行模式**，且各自有严格的能力边界（不支持的能力在配置校验阶段直接报错，不再静默忽略）：
  - `archive_replay` — 复制并核验归档模型与预测，不训练、不重算指标；
  - `train_from_archive_features` — 在固定归档特征与固定 train/test 划分上重新训练（换模型/换调参/换指标）；
  - `raw_gis` — 从原始 GIS 数据端到端构建特征并训练（研究单元/采样/标签/权重/算子/空间验证/知识/谓词均在此生效）。
- **四种模型**：`rf`（sklearn 随机森林）、`spe`（自实现 Self-Paced Ensemble）、`cnn`（PyTorch 1D CNN）、`mlp`（PyTorch MLP，唯一支持谓词约束/LUSI 加权损失）。
- **无泄漏预处理**：`BaselinePreprocessor` 提供 `fit/transform/fit_transform`，外层划分先于预处理拟合，目标折只调用训练折 fitted 的 `transform`。
- **空间交叉验证**：`spatial_block_kfold` / `spatial_group_kfold` / `spatial_block_holdout`（需投影坐标，经纬度会触发明确错误）。
- **评分语义契约**：`probability`（未校准正类概率）/ `raw_score`（`decision_function`）/ `relative_score`（MinMax 相对分数），互斥校验；GeoTIFF 导出需显式 `prediction.target_crs` 并采用半像元定位。
- **更完整的指标**：通用分类指标 `accuracy/precision/recall/f1/roc_auc/confusion_matrix/average_precision/balanced_accuracy/mcc`，以及 MPM 面积捕获指标模块（`src/validation/mpm_metrics.py`，需数据源提供 `unit_area` 才会计算）。
- **研究变量注册化**：研究单元（`point_local_environment`）、背景采样（`random_points_in_nsw_boundary`）、标签策略（`positive_unlabeled_as_zero`）、样本权重（`size_code` / `uniform`）均通过注册表驱动。
- **知识/谓词分离**：知识提供者（`empty` / `spatial_extent`）与谓词（`all_ones` / `spatial_box` / `spatial_distance` / `combined`）分离；谓词约束通过 LUSI 加权损失由 `mlp` 消费。
- **可复现工件**：微秒级 `run_id` 原子创建输出目录；manifest 记录派生种子、split 摘要、best params、git 信息、环境版本与输入/输出 SHA-256。

## 能力状态（诚实标注）

| 能力 | 状态 |
|------|------|
| 二维靶区预测（`target_area_prediction`） | `implemented` |
| 归档回放 / 归档特征重训 / 原始 GIS | `implemented` |
| RF 基线 | `implemented` |
| SPE / CNN / MLP | `implemented`（SPE/CNN 与 MLP 的 LUSI 链路的算法一致性尚未验证，标为 `experimental`） |
| 概率校准（`calibrated_probability`） | 未实现（当前 `probability` 为未校准正类概率） |
| 深边部预测（`deep_edge_prediction`） | `placeholder`（无三维能力，显式报错） |

## 文档

- 使用说明（完整手册）：[`GUIDE.md`](GUIDE.md)
- 架构说明：[`docs/PHASE2_ARCHITECTURE_CN.md`](docs/PHASE2_ARCHITECTURE_CN.md)
- 使用说明（历史验证版）：[`docs/PHASE2_USAGE_VERIFIED.md`](docs/PHASE2_USAGE_VERIFIED.md)
- Phase-1 迁移背景：[`docs/PHASE1_MIGRATION.md`](docs/PHASE1_MIGRATION.md)
- 重构前基线契约：[`docs/BASELINE.md`](docs/BASELINE.md)
- 本轮 P0/P1 修复与遗留待办：[`POST_REFACTOR_REVIEW.md`](POST_REFACTOR_REVIEW.md)

## 输出目录

每次运行都会在配置的 `output_dir` 名后追加微秒级 `run_id`，避免覆盖前次结果：

```text
outputs/<实验名>_<YYYYMMDD_HHMMSS_微秒>/
├── config_resolved.yaml    # 合并后的完整配置（含 DEFAULTS + 迁移 + CLI 覆盖）
├── manifest.json           # 可复现性清单（派生种子/split 摘要/best params/git/环境/哈希）
├── metrics.json            # 评估指标
├── experiment.log          # 结构化日志
├── models/                 # 序列化模型
├── predictions/            # 预测 CSV + 可选 GeoTIFF
├── intermediate/           # 归档特征快照
└── figures/                # 评估图表
```

> 提示：`scripts/list_components.py` 目前只打印 9 个主注册表（task/model/label_refiner/feature_operator/knowledge/predicate/splitter/metric/tuner）；研究单元/背景采样/标签策略/样本权重这 4 个研究变量注册表已注册化，但未列入该脚本输出，详见 `GUIDE.md` 第 4 节。
