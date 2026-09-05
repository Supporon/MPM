# CHANGE_LOG

## 2026-09-04T09:47:00Z
- 建立变更日志机制：
  - 新增 `change_log.md`（本文件），格式对齐实验代码 `lachlan_dataset_audit/WORKLOG.md`：以 ISO 8601 UTC 时间戳为节标题，逐条记录变更。
  - 新增 `AGENTS.md`，固化「每次修改代码后必须在 `change_log.md` 追加一条记录」的执行规则。
- 后续每次代码修改（新增/修改/删除文件、修复、重构、功能变更、配置变更）均按上述格式追加一节，记录追加到文件末尾，时间递增。

## 2026-09-05T04:04:37Z
- 依据 `MPM-main_static_audit_2026-09-05.md` 复核结论，修复确定性正确性问题（P0 批次）：
  - `src/models/spe.py`：平衡类别时 minority/majority 不再选中同一类（用 classes_ 顺序打破平局）；k_bins 采样改用 `np.array_split` + 补齐，修复多数类样本数小于 k_bins 时的配额丢失；hardness 与 predict_proba 按各基学习器 `classes_` 显式对齐到集成类别，避免单列概率广播伪造另一类。
  - `src/training/trainer.py`：`TorchTrainingConfig` 新增 `dataloader_seed`；移除全局 `np.random.seed`；DataLoader 用独立 `torch.Generator` 消费 dataloader seed；新增 `has_batch_norm` + `batch_size<2` 守卫；统计 `optimizer_steps`，零优化步骤时抛出明确 RuntimeError。
  - `src/models/mlp.py` / `src/models/cnn.py` / `src/models/cnn2d.py`：`torch.manual_seed(random_state)` 提前到构建 `nn.Module` 之前；透传 `dataloader_seed`；`validate_config` 要求 `use_batch_norm` 时 `batch_size>=2`；CNN2D 的 dropout 参数化、张量迁移到 device、`@torch.no_grad()` 改为方法体内 context manager（隔离导入期 torch 访问）。
  - `src/core/experiment.py` / `src/tuning/none.py` / `src/tuning/bayes.py`：调参路径按用途区分并透传 `model_seed` / `tuning_seed` / `dataloader_seed`。
  - `src/core/contracts.py` / `src/predicates/pipeline.py` / `src/predicates/builtins.py` / `src/training/losses.py`：单一 constraint 谓词写 `phi_vector`、多谓词管线与 `combined` 归一化为 `phi_vectors`；每个 constraint 谓词基于去除历史 phi 载荷的干净副本独立计算，避免混用导致 combined 约束丢失；管线累积 `predicates_applied` 元数据；losses 增加两键共存守卫。
  - `configs/experiments/`：恢复测试引用的 `lachlan_rf_baseline.yaml`、`lachlan_rf_phase2.yaml`、`lachlan_rf_raw_gis.yaml`（自提交 9a985a8 还原）。
- 验证：`python -m pytest -q` → 91 passed（含 26 个 unittest subtests），3 warnings；`tests/test_mlp_predicates.py`、`tests/test_spe_model.py`、`tests/test_seeds.py` 全绿。

## 2026-09-05T04:24:40Z
- 依据 `MPM-main_static_audit_2026-09-05.md` 复核结论，继续修复确定性正确性问题（P0-06 CRS 贯通、P0-03 折内隔离）：
  - `src/utils/crs.py`（新增）：`crs_identifier` / `resolve_crs_identifier` / `require_consistent_crs`，用 pyproj/GDAL 稳健解析 EPSG/WKT/PROJ，跨组件 CRS 一致性校验（不一致时明确报错，不做隐式重投影）。
  - `src/data/samplers.py` / `src/data/research_unit_builtins.py`：`UniformBoundarySampler`、`build_positive_units`、`build_prediction_units` 将源边界/矿点 CRS 写入结果 `DataFrame.attrs["crs"]`。
  - `src/tasks/target_area.py`：`_record_crs` 跨 occurrence/boundary/training_boundary 校验并记录研究单元 CRS；`export_geotiff` 校验 `prediction.target_crs` 与已记录研究单元 CRS 一致，否则明确拒绝（防止只改标签不改坐标）。
  - `src/core/experiment.py`：`build_features` 记录 `component_metadata["research_crs"]` 供 manifest 溯源。
  - `src/features/preprocess.py`：`BaselinePreprocessor` 拆出 `fit_schema`（列选择 + OHE 类别）与 `refit_scaler`（折内重拟合），`fit` 保持为两者组合（向后兼容）。
  - `src/features/fold_safe.py`（新增）：`FoldSafePreprocessor` sklearn 兼容 Transformer，折内只重拟合 scaler、复用完整外层训练集确定的 schema，避免验证折参与缩放统计。
  - `src/tuning/bayes.py`：`raw_gis` + 内层 CV 时用 `Pipeline(FoldSafePreprocessor, model)` 逐折重拟合 scaler，搜索空间/拟合参数加 `model__` 前缀，结束后解包返回裸模型，供后续 evaluate/predict 复用外层 preprocessor。
  - `src/core/experiment.py`：`raw_gis` 分支保留未变换外层训练折特征（仅预处理器实际使用的列），仅对内层 CV tuner 透传 `preprocessor` / `raw_data`；`none` 等无 CV tuner 不受影响。
  - `src/core/config.py`：新增 raw_gis 能力守卫——`label_refinement(PUB)` 与内层 CV tuner 组合在配置层拒绝（PUB 在完整外层训练集拟合后泄漏到内层验证折，且无法在 BayesSearchCV 折切片内安全重拟合）。
  - `tests/test_crs.py`（新增）：8 项 CRS 解析/一致性/导出校验测试。`tests/test_fold_safe.py`（新增）：5 项折内隔离单元 + bayes 集成 + 配置守卫测试。
- 验证：`python -m pytest -q` → 103 passed, 1 skipped（GDAL 未装时导出测试跳过）；`tests/test_crs.py` 7 passed, 1 skipped；`tests/test_fold_safe.py` 5 passed。

## 2026-09-05T04:32:27Z
- 依据复核结论修复归档坐标匹配（P1-08 部分）：
  - `src/data/archive_coords.py::match_rows`：新增 `max_error`（默认 1e-3，实测重建/归档特征切比雪夫误差 ~1e-15）与 `require_unique` 歧义检测；超阈值或歧义匹配明确报错并附行号/误差，避免静默错配。
  - `src/data/archive_coords.py::snap_cells`：新增越界检测——坐标超出格网范围（超过半个格网间距）时发出警告（训练点常覆盖比靶区更大的区域，吸附到边缘单元是既有行为，故仅警告不报错），提示核对坐标 CRS 与格网范围。
  - `tests/test_archive_coords.py`（新增）：5 项容差/歧义/越界测试。
- 验证：`python -m pytest -q` → 108 passed, 1 skipped；真实归档 `match_rows` 仍成功（train=359, test=120）。

## 2026-09-05T04:55:21Z
- 依据复核结论修复研究单元/标签/权重配置透传（P1-02）：
  - `src/core/config.py`：`research_unit` 新增 `params`、`label` 新增 `params` 与 `weight_strategy`（name + params），严格 schema 同步放行并校验（params 须为 mapping，weight_strategy.name 须已注册）。
  - `src/data/label_strategies.py`：`apply_positive` 的权重策略由硬编码 `size_code` 改为可配置（默认仍 size_code，亦可 uniform）。
  - `src/tasks/target_area.py` / `src/data/research_unit_builtins.py`：研究单元、背景采样器、标签策略创建不再固定传 `{}`，改为透传 `research_unit.params` / `label.params`。
  - `tests/test_config_passthrough.py`（新增）：4 项权重策略可配置/校验/透传测试。
- 验证：`python -m pytest -q` → 112 passed, 1 skipped。

## 2026-09-05T05:00:20Z
- 依据复核结论在配置层补齐 grid 模型能力守卫（P1-06/P1-09）：
  - `src/core/config.py::_validate_mode_capabilities`：新增模型 `grid_model` 能力判断——grid 模型（cnn2d/label_spreading）与内层 CV tuner 组合拒绝（grid/inside/train_cells 无法按折切片）；`raw_gis` 模式拒绝 grid 模型（网格数据仅 train_from_archive_features 构建）。
  - `tests/test_mode_guards.py`（新增）：3 项 grid 模型 × 模式/调参守卫测试。
- 验证：`python -m pytest -q` → 115 passed, 1 skipped；全部 25 个实验配置 `load_config` 校验通过。

## 2026-09-05T05:29:02Z
- 依据复核结论补齐失败 manifest 与预处理工件 provenance（P2 失败记录 + P1-07 工件重建）：
  - `src/core/experiment.py`：`run()` 异常分支设置 `self._failure`（exception_type/message/phase/elapsed_seconds）并尝试写出 `status=failed` 的 manifest（不覆盖成功输出）；`export()` 新增 `failure` 字段。`train()` 的 `raw_gis` 分支在拟合预处理器后立即持久化 `models/preprocessor.pkl`（相关性筛选 + OHE + scaler 状态），使成功运行可仅凭输出目录重建预处理语义。
  - `src/data/research_unit_builtins.py`：修复 `_require_crs` 未导入即调用的 `NameError`（此前无端到端测试覆盖 `build_positive_units`，该缺陷未被捕获）。
  - `tests/test_framework.py`：新增 2 项端到端测试——`test_failed_run_writes_failure_manifest`（archive 缺失模型文件触发 train 失败，断言 manifest `status=failed`、`failure.exception_type=FileNotFoundError`、`failure.phase=train`）与 `test_raw_gis_run_persists_fitted_preprocessor`（合成矢量/栅格 + 假 `lib_mpm` 的完整 raw_gis 运行，断言 `models/preprocessor.pkl` 可反序列化、`scaler_fitted`/`encoder_fitted`/`feature_columns` 就绪且计入 manifest output_files）。
- 验证：`python -m pytest -q` → 117 passed, 1 skipped。

## 2026-09-05T05:34:40Z
- 依据复核结论修复 CLI 路径覆盖的解析语义与存在性校验（P2）：
  - `src/core/config.py`：抽出 `_config_root`（与 `load_config` 共享）；`apply_cli_overrides` 在合并后对 dataset 路径重新执行 `_resolve_dataset_paths`（相对 config 根目录解析，与 YAML 配置语义一致，而非相对 CWD）；新增 `_validate_overridden_paths`，对被 `--set dataset.<key>=...` 覆盖的标量路径键做存在性校验，指向不存在路径时抛出明确 `ConfigError`（`experiment.output_dir` 等运行期创建的目标路径不做要求）。
  - `tests/test_framework.py`：新增 3 项 CLI 覆盖测试——相对路径解析、不存在路径拒绝、存在路径通过。
- 验证：`python -m pytest -q` → 120 passed, 1 skipped；`tests/test_framework.py -k cli` → 7 passed。

## 2026-09-05T05:40:16Z
- 依据复核结论补齐调参轨迹与 target scaling 的工件 provenance（P1-07 部分）：
  - `src/tuning/bayes.py`：`fit` 结束后将 `tuning_summary`（tuner/n_iter/n_splits/scoring/best_score/best_params/逐候选 `cv_results` 轨迹）附着到返回的模型 `_tuning_summary`；新增 `_to_json_value` 将 numpy 标量/映射/列表递归转为 JSON 可序列化值。
  - `src/core/experiment.py`：`train` 阶段读取 `self.model._tuning_summary`；`predict` 阶段读取 `self.task.normalization_range` 存为 `_target_scaling`；`export()` 新增 `tuning_summary` 与 `target_scaling` 字段（`none` tuner 无轨迹时为 null）。
  - `src/tasks/target_area.py`：新增公开属性 `normalization_range`，暴露 `relative_score + minmax` 的原始分数范围。
  - `tests/test_framework.py`：新增 `test_bayes_run_records_tuning_summary_and_target_scaling`（合成归档 + bayes n_iter=2，断言 manifest 记录 best_score/best_params/rank=1 的 cv_results 轨迹与 target_scaling）。
- 验证：`python -m pytest -q` → 121 passed, 1 skipped。

## 2026-09-05T05:42:51Z
- 依据复核结论修复入口脚本与文档中引用不存在配置的问题（P1-11 部分）：
  - `scripts/baseline/nsw_rf.sh`：默认配置由不存在的 `nsw_rf_baseline.yaml` 改为 `nsw_rf_train.yaml`。
  - `scripts/validate_reproduction.py`：删除 `archive_replay` 组中引用不存在 `nsw_rf_baseline.yaml` 的条目，改为在 `train_from_archive_features` 组新增 `nsw_rf_train` 条目（指向存在的 `nsw_rf_train.yaml`）。校验所有 manifest 引用的配置均存在。
  - `README.md` / `GUIDE.md`：更新已过时的说明——当前预置 22 个 `nsw_*.yaml` + 3 个 `lachlan_*.yaml`（archive_replay / train_from_archive_features / raw_gis）；测试套件现状改为 `121 passed, 1 skipped`（不再声称无 lachlan 配置或 16 failed）。
- 验证：`python -m pytest -q` → 121 passed, 1 skipped；`run.py --validate-only` 代表配置通过；`validate_reproduction.py` 语法与引用配置存在性校验通过。

## 2026-09-05T05:48:25Z
- 依据复核结论补齐 MPM 面积捕获指标的配置/调用/unit_area 生产闭环（P1-03）：
  - `src/validation/mpm_metrics.py`：新增 `MPM_METRIC_NAMES` frozenset，列出 `evaluate_mpm_metrics` 产出的全部指标键（capture_rate_at_area_*、area_fraction_at_capture_*、prediction_rate_auc）。
  - `src/validation/metrics.py`：将 `MPM_METRIC_NAMES` 注册到 `METRIC_REGISTRY`（占位函数在直接调用时抛错，提示须经 `evaluate_classifier` 传入 unit_area），使 `validation.metrics` 配置校验可识别；`evaluate_classifier` 按显式配置拆分 `mpm_requested` / `standard_names`——仅在显式请求 MPM 指标且提供 `unit_area` 时计算，缺 `unit_area` 时写 `None` + `<name>_not_computed_reason="unit_area not provided"`，不再隐式全量计算。
  - `src/data/research_unit_builtins.py`：`build_prediction_units` 在规则预测网格上写入 `attrs["unit_area"] = grid_size²`。
  - `src/tasks/target_area.py`：新增 `unit_area` 属性，从 `build_prediction_units` 产生的网格读取并随任务上下文暴露，供评价/溯源使用。
  - `tests/test_framework.py`：新增 4 项测试——MPM 指标可配置校验、`evaluate_classifier` 按配置计算、缺 `unit_area` 记录不可计算原因、预测网格携带 `unit_area=grid_size²`；`tests/test_mpm_metrics.py` 更新为「显式配置才计算」语义。
- 验证：`python -m pytest -q` → 125 passed, 1 skipped。

## 2026-09-05T05:59:20Z
- 依据复核结论补齐独立评价 CV 与 OOF 预测（P1-01）：
  - `src/core/experiment.py`：新增 `_run_independent_cv`——在训练集上按 `validation.cross_validation` 配置运行独立评价 CV，与调参 CV 分离；即使 `tuning=none`，`cross_validation` 也不再被静默忽略。每折构建全新模型（用最终模型 `get_params()` 或 `model.params`），评估未见过折，产出 OOF 预测表（`intermediate/oof_predictions.csv`：fold/row_index/y_true/y_pred/score/sample_weight）、逐折指标与类别计数、以及标量指标的 mean±std 汇总。
  - 无法安全切片时（grid 模型、约束谓词、外层全量拟合的 PUB 标签细化、训练样本过少导致 StratifiedKFold 无法切片）返回带 `reason` 的跳过说明，而非产生有偏评价或使运行失败。
  - `train_from_archive_features` 在配置空间 CV 时显式加载逐行坐标（`_cross_validation_needs_coordinates`），不再依赖不存在的 `point_units` 偶然分支。
  - `evaluate()` 将 `independent_cv` 写入 `metrics.json` 与 `manifest.json`（新增 `independent_cv` 字段）；OOF 表计入 output_files。
  - `tests/test_framework.py`：新增 2 项测试——独立 CV 记录 OOF 覆盖全部训练样本与逐折指标；样本过少时记录跳过原因。
- 验证：`python -m pytest -q` → 127 passed, 1 skipped。

## 2026-09-05T06:03:40Z
- 依据复核结论补齐 Knowledge/Predicate 注入可归因审计（P1-04 部分）：
  - `src/core/experiment.py`：`_apply_knowledge_and_predicates` 产出 `injection_audit`（knowledge 是否启用/提供器与工件、predicate 是否启用/名称与 kind、是否产出约束、knowledge 是否无人消费），并在「启用 knowledge 但无谓词消费」时发出 warning；`train` 阶段记录 `constraints_consumed`（约束是否真正被支持约束的模型消费）。manifest 新增 `injection_audit` 字段，使「约束有无」的差异可归因，消融实验可只切换目标机制。
  - `tests/test_framework.py`：新增 `test_injection_audit_flags_unconsumed_knowledge`（启用 spatial_extent 但无谓词时，manifest 记录 knowledge_enabled/unconsumed_knowledge/constraints_consumed 等审计标志）。
- 验证：`python -m pytest -q` → 128 passed, 1 skipped。

## 2026-09-05T06:22:10Z
- 依据复核结论验证可选 PyTorch 依赖的导入隔离（P1-05）：确认 `src/models/mlp.py` / `cnn.py` / `cnn2d.py`、`src/training/trainer.py` / `losses.py` 均在 `try/except ImportError` 内导入 torch 并降级（`_TORCH_AVAILABLE=False`），`_require_torch()` 在构建/拟合时才报 `OptionalDependencyError`；无 torch 环境下 RF/归档路径可正常 `load_builtin_components` 与 `build`。
  - `tests/test_framework.py`：新增 `test_torch_import_isolation_keeps_rf_path_working`——子进程内以 `MetaPathFinder` 拦截 `torch`/`torch.*` 导入，断言 `load_builtin_components` 成功、RF `build` 成功、MLP `fit` 抛 `OptionalDependencyError`。
- 验证：`python -m pytest -q` → 129 passed, 1 skipped。

## 2026-09-05T06:32:23Z
- 依据复核结论清理绘图硬编码（P2 绘图部分，步骤 7）：
  - `src/plotting/plot.py`：新增 `point_kind_labels`，从归档 `training_data_deposit.csv`/`training_data_unlab.csv` 行数推导点 kind，移除硬编码的 277 矿点 + 272 未标注点；`build_point_table` 改用它。
  - 分数语义贯通：`plot_run` 依据预测列名（prob/raw_score/relative_score）动态决定 colorbar 标签（预测概率/原始决策分数/相对分数）与归一化范围——只有 probability 固定 [0,1]，其余取实际 min/max，替换原先硬编码的「预测概率」标签与 `vmin=0, vmax=1`。
  - 轴标签由硬编码「经度 (°E) / 纬度 (°N)」改为中性「X / Y」，标题/颜色条标签改为分数语义，支持非 NSW 数据。
  - 新增 `_resolve_archive_from_manifest`：`plot_run` 的 `archive_dir` 缺省时优先从 `run_dir/manifest.json` 的 `input_paths.archive_dir` 读取，回退 NSW 归档默认路径，使绘图不绑定单一数据集。
  - `tests/test_plotting.py`（新增）：3 项轻量测试——`point_kind_labels` 由文件行数推导（非 277/272）、manifest 归档路径解析、manifest 缺失时回退 None。
- 验证：`python -m pytest -q` → 132 passed, 1 skipped；`tests/test_plotting.py` → 3 passed。

## 2026-09-05T06:42:08Z
- 依据复核结论推进 raw_gis 字段可配置化与按算子收紧输入校验（P1-09 部分）：
  - `src/operators/features/builtins.py`：`_BaseOperator` 新增 `REQUIRED_DATASET_KEYS` / `REQUIRED_GEOLOGY_KEYS` 类属性，声明各算子实际消费的数据集顶层键与 geology 子键（line_distance→seismic+line_files；categorical_geology→geology；raster_statistics/texture→magnetic/gravity/radiometric/remote_sensing；elevation_gradient→elevation）。`CategoricalGeologyOperator` 的类别字段名（原硬编码 MetFacies/Dominant_L）改为可配置 params（`metamorphic_field`/`intrusions_field`/`rock_units_field`，缺省沿用 NSW 字段）。
  - `src/data/weight_strategies.py`：`SizeCodeWeightStrategy` 的属性列名由硬编码 `SIZE_CODE` 改为可配置 params `column`（缺省 SIZE_CODE），缺失列时报错并提示实际列名。
  - `src/tasks/target_area.py`：`validate_config` 新增可选 `feature_operators` 参数——raw_gis 下按所选算子仅要求其消费的输入字段，不再无条件要求整套 NSW 字段；未提供算子列表时保守回退到全量要求（向后兼容直接调用）。
  - `src/core/config.py`：`TASK_REGISTRY.validate` 透传 `config["features"]["operators"]`，使配置校验与实际算子选择闭环。
  - `tests/test_raw_gis_generalization.py`（新增）：5 项测试——算子依赖声明、按算子收紧校验、缺算子输入拒绝、无算子回退全量要求、权重列名可配置。
- 验证：`python -m pytest -q` → 137 passed, 1 skipped；`tests/test_raw_gis_generalization.py` → 5 passed。

## 2026-09-05T06:50:22Z
- 依据复核结论补齐逐样本结果与多种子汇总闭环（P1-10 部分）：
  - `src/tasks/target_area.py`：`predict` 输出的靶区表新增稳定 `unit_id` 列（有效单元在规范网格序中的 0 基索引，同一网格定义下跨运行稳定），供逐样本关联与跨运行汇总。
  - `src/core/experiment.py`：`evaluate` 阶段新增 `_write_test_predictions`，写出 `intermediate/test_predictions.csv`（row_index/y_true/y_pred/score/sample_weight，含逐行 X/Y 坐标若可用），与 aggregate 指标互补；逐样本表为尽力而为，评分异常只告警不阻断。归档重放模式不重新计算（预测直接复制归档），故不产出该表。
  - `scripts/aggregate_runs.py`（新增）：跨运行汇总脚本，扫描多个 run 目录的 `manifest.json`+`metrics.json`，产出 `runs.csv`（逐 run 索引）、`summary.csv`（按 experiment/variant/mode/model/tuner 分组，对标量指标计算 mean±std）、`failures.csv`（失败运行）；统计排除 failed 与 archive_replay。
  - `tests/test_framework.py`：新增 `test_per_sample_test_table_and_prediction_unit_id`（断言逐样本表列与预测表 unit_id）。`tests/test_aggregate_runs.py`（新增）：汇总脚本端到端测试（4 run 含 failed/replay，断言 runs/summary/failures 及 mean±std）。
- 验证：`python -m pytest -q` → 139 passed, 1 skipped；`tests/test_aggregate_runs.py` → 1 passed。

## 2026-09-05T06:57:06Z
- 收口验证（最终门禁）：
  - `python -m pytest -q` → 139 passed, 1 skipped（GDAL 未装时导出测试跳过），26 个 unittest subtests 通过。
  - 全部 25 个实验配置 `python run.py --config <cfg> --validate-only` 通过（含新增 feature_operators 透传的 raw_gis 校验路径）。
  - 真实归档端到端最小 run（`nsw_rf_train.yaml`，输出到临时目录）：train 359 行（pos=151/neg=208）→ 独立 CV `run=True` → evaluate f1=0.9333 且写出逐样本 `test_predictions.csv` → predict 7583 目标单元（`target_probs.csv` 含 `unit_id`）→ 自动绘图成功（score semantics + manifest 归档路径）。`target_scaling=[0.0, 0.86]` 记录在 manifest。
- 剩余已知限制（未在本轮完成，属于计划中标注的「大」项，需结合实际数据/管线再定）：
  - P1-09 全量：石龙头当前数据为 MapGIS 逐几何特征矩阵（`numeric_feature_matrix.csv`/`all_features.csv` + GeoJSON 图层），与现有 NSW 归档（`Xy_rf_train`/`target_features`）和 `raw_gis`(lib_mpm) 均不同构，需新增显式输入适配与完整可运行 YAML（本轮已先完成字段可配置化与按算子收紧校验）。
  - P1-03 全量：在预测网格上以「含矿单元」标签做 prediction-rate/面积捕获率端到端评价（需明确 occurrence→网格单元的 deposit-in-cell 标签生成规则）；本轮已打通 unit_area 生产、指标注册、配置闭环与缺面积不可计算原因。
