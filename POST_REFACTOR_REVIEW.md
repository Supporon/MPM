# POST_REFACTOR_REVIEW — MPM_codex_phase2

> 本文件记录本轮“可复现性与科学可信度”修复的完成情况、测试证据与遗留待办。
> 对应分支：`readiness/p0-p1-fixes`（基于 `main` @ `4121bcf`）。

## 1. 变更摘要

按验收规格（`CODEING_UPDATE_MD/MPM_CODE_OPTIMIZATION_AND_ACCEPTANCE_SPEC.md`）逐项修复：

| 任务 | 内容 | 状态 |
|---|---|---|
| P0-A | split-first 无泄漏预处理（`BaselinePreprocessor.fit/transform/fit_transform`，外层划分先于预处理拟合） | 完成 |
| P0-B | 执行模式能力边界 + 严格 schema（replay 拒绝调参/PUB/知识/谓词/算子；archive-feature 拒绝 PUB/算子；未知键/枚举校验） | 完成 |
| P0-C | 评分/概率语义契约（raw_score/relative_score/probability 互斥）+ 指标（average_precision/balanced_accuracy/mcc）+ MPM 面积指标模块 | 完成 |
| P0-D | 验证器防零运行假阳性 + 实验清单修正 | 完成 |
| P0-E | 空间验证修正（group 注入、经纬度检测、PUB 传 data） | 完成 |
| P0-F | CRS/ROI/GeoTIFF 硬约束（边界并集、CRS 校验、采样不足失败、显式 target_crs） | 完成 |
| P1-A | 复现工件与日志（微秒 run_id 原子创建、run 状态、manifest 增强、日志 handler 清理） | 完成 |
| P1-B | MLP/Predicate/TSIL 链路（φ 合并、sample_weight、bayes+constraints 显式报错、torch 依赖、MLP YAML） | 完成 |
| P1-C | GUIDE/README 一致性（移除无效命令与不存在配置引用、修正 stray fence、能力状态说明） | 部分完成* |

*P1-C 仅做针对性修正；GUIDE 第 9 节大量消融命令仍需按新能力边界全面重写（见 §4）。

## 2. 测试证据

```bash
$ python -m pytest -q
76 passed, 3 warnings, 26 subtests passed
```

- 新增无泄漏测试：改变测试分布不改变训练 scaler 统计量、测试独有类别不入编码器。
- 新增模式能力/严格 schema/空间 group/经纬度/score_type/指标/φ 合并/sample_weight/日志累积 等回归测试。
- `python scripts/validate_reproduction.py --validate-only` 现在输出“配置验证完成，但未运行任何实验（未验证复现）”，不再声称“可正确复现”。

## 3. 兼容性变化

- `DEFAULTS.features.operators` 改为 `[]`；算子改由各 raw_gis 配置显式声明。
- `DEFAULTS.prediction.normalization` 改为 `none`；`score_type=probability` 不再允许 `minmax`（MinMax 结果必须显式 `score_type=relative_score`）。
- `archive_replay` / `train_from_archive_features` 现对不支持字段**硬失败**（原为警告）。
- `BaselinePreprocessor` 新增 `fit/transform/fit_transform`；`prepare_training`/`transform_target_legacy` 保留为遗留接口。
- 输出目录追加微秒级 `run_id`，`outputs/<name>/<name>_<run_id>/`，已存在则失败。

## 4. 遗留待办（backlog）

按优先级排序：

1. **GUIDE 第 9 节全面重写**：以 `CODEING_UPDATE_MD/GUIDE.optimized-draft.md` 为底稿，结合新能力边界，用 CI smoke test 验证每条非耗时命令。
2. **TSIL 损失数值对齐**：`losses.py` 的 P 项缩放 `(1/N)·‖φ̃ᵀe‖²` 需与 TSIL 参考实现的 `‖(1/B)·Φᵀe‖²` 做固定输入数值对齐后再定性。
3. **SPE / CNN 科学有效性验证**：SPE 与 `imbalanced-ensemble.SelfPacedEnsembleClassifier` 固定种子一致性；CNN 列置换敏感性实验，未证明前保持 `experimental`。
4. **SeedContext 派生种子**：当前各组件共用 `experiment.seed`，需派生 sampling/split/tuning/model/dataloader 种子并贯通（manifest 记录）。
5. **MPM 指标接入评估链**：`mpm_metrics.py` 已实现纯函数，但需 `unit_area` 扩展评估上下文后方可接入 `evaluate_classifier`。
6. **模式能力完整性**：`train_from_archive_features` 还应拒绝 holdout / research_unit 变更（当前仅拒绝 PUB 与算子）。
7. **GeoTIFF 模板继承**：半像元定位、继承源栅格 transform/分辨率/nodata（当前已移除硬编码 EPSG，但未继承模板）。
8. **贝叶斯搜索 + 约束模型**：当前显式报错；完整方案需 constraint-aware CV splitter。
9. **三维契约**：`deep_edge_prediction` 仍为占位，需 voxel 单元/钻孔标签/3D CRS/3D 特征/3D 验证等契约完成后才能实现。

## 5. 未完成能力（能力状态）

- `deep_edge_prediction`：placeholder（无三维能力）。
- SPE / CNN / TSIL 链路：experimental（未做算法一致性验证）。
- 概率校准（calibrated_probability）：未实现（需独立校准集或 OOF）。
