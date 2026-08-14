# Codex 提示词：验证 MPM 二阶段重构、代码分析与使用说明

你是一名负责科研软件架构与机器学习实验可复现性的高级工程师。请对当前仓库进行**代码级验证**，不要根据 README、架构文档或文件名直接下结论；文档只能作为“设计意图”，实际代码、测试和运行结果才是判断依据。

## 一、背景和目标

该仓库是找矿预测 MPM 工程的二阶段重构。最终希望形成可长期扩展的找矿预测研究框架，后续要支持：

- 不同任务：靶区预测、深边部预测等；
- 不同机器学习/深度学习模型；
- 不同 holdout / cross-validation，尤其后续空间交叉验证；
- 可配置评价指标；
- 可增加/选择的地质知识与 Predicate，且二者概念分离；
- 不同数据类型和特征提取 Operator；
- 独立的调参策略；
- baseline / +knowledge / +predicate / +knowledge+predicate 消融对比；
- YAML 参数化实验；
- 后续自然语言先编译成 YAML，验证后再生成 sh，而不是让 LLM 直接控制实验内部算法。

本次二阶段重构的核心验收原则是：

> 新增一个模型、特征算子、Predicate、Knowledge Provider、Splitter、Metric 或 Tuner 时，正常情况下不应修改 `src/core/experiment.py`。

## 二、必须检查的代码，不得只读文档

至少逐文件检查：

```text
run.py
src/core/config.py
src/core/spec.py
src/core/contracts.py
src/core/registry.py
src/core/bootstrap.py
src/core/experiment.py

src/tasks/
src/models/
src/operators/features/
src/knowledge/
src/predicates/
src/validation/
src/tuning/
src/data/
src/features/

tests/test_framework.py
tests/baseline/test_baseline_artifacts.py

configs/experiments/lachlan_rf_baseline.yaml
configs/experiments/lachlan_rf_phase2.yaml
```

同时搜索全仓库，检查是否还有关键算法选择被硬编码在核心层，例如：

```text
RandomForest
train_rf
BayesSearchCV
random_split
line_distance
raster_statistics
identity predicate
具体 task 名称
```

如果这些具体实现出现在 `Experiment` 中，要判断是否造成核心编排器与实现耦合。

## 三、逐项验证

### 1. Registry 是否是真扩展点

验证：

- Task Registry
- Model Registry
- Feature Operator Registry
- Knowledge Registry
- Predicate Registry
- Splitter Registry
- Metric Registry
- Tuner Registry
- legacy PUB label refiner registry

重点判断：

1. registry 不是只做“名字映射”，但最终仍由 `Experiment` if/elif 选择；
2. 新组件是否可以通过“新增实现 + 注册 + YAML”接入；
3. registry 未知名称是否能给出明确错误；
4. 注册重复名称是否有保护；
5. 是否存在循环 import 或注册顺序依赖。

### 2. ExperimentSpec / 配置系统

验证：

- phase-1 YAML 是否能迁移为 phase-2 结构；
- phase-2 YAML 是否直接生成 `ExperimentSpec`；
- config validation 是否基于 registry，而不是继续维护硬编码允许值列表；
- operator 参数、split 参数、metric 名称等错误是否尽早失败；
- 路径解析是否正确；
- `archive_replay / train_from_archive_features / raw_gis` 三种 mode 是否边界清楚。

特别检查旧配置迁移是否会：

- 丢失 `model.search`；
- 丢失 `model.pu.enabled`；
- 改变 feature operator 顺序；
- 改变 random split / CV 的基线语义。

### 3. Task 抽象

验证 `target_area_prediction` 是否至少负责：

- positive research units；
- unlabeled research units；
- prediction units/grid；
- prediction score；
- task-specific output/export。

判断 `deep_edge_prediction` 当前只是显式 placeholder 是否合理。不要把“未实现 3D 数据契约”判成普通代码 bug；重点确认它不会伪装成已经支持。

### 4. Model 与 Tuner 解耦

验证：

- RF adapter 是否只负责构造模型；
- BayesSearchCV 是否已经移出 RF 主实现；
- `tuning=none` 是否可以直接训练；
- `tuning=bayes` 是否延迟导入 `skopt`；
- 搜索空间是否由配置描述，而不是写死在 Experiment；
- sample_weight 是否正确传入。

重点检查是否仍存在“配置里 model.name 可变，但实际永远训练 RF”的假抽象。

### 5. Feature Operator Pipeline

验证 phase-1 特征顺序是否保持：

```text
raster statistics
-> texture
-> elevation gradient
-> line distance
-> categorical geology
```

验证：

- operator 输出行数错误是否会失败；
- duplicate columns 是否会失败；
- 新 operator 是否无需修改 Pipeline；
- `SpatialFeatureExtractor` 是否只是向后兼容层，而非新的核心扩展点。

### 6. Knowledge 与 Predicate 分离

验证：

- Knowledge Provider 和 Predicate 有独立 registry / pipeline；
- predicate 可以通过 context 获取 knowledge；
- 当前 `empty` / `identity` 只是接口验证组件，文档没有把它们描述成真实地质知识方法；
- predicate 如果生成模型不支持的 constraints，是否能够明确失败，而不是被静默忽略。

分析这个接口是否足以继续实现真实 geological/geochemical/geophysical predicate；如果不足，指出**契约级原因**，不要只给风格建议。

### 7. Splitter / Metric

验证：

- random holdout；
- stratified k-fold；
- metric registry；
- primary metric 与 tuner scoring 的连接；
- weighted metrics 是否正确使用 sample_weight；
- confusion matrix 是否刻意保持原行为。

明确指出：当前没有 spatial CV 是“未实现能力”还是“架构阻塞”。如果新增 spatial block CV 仍需修改 Experiment，则判为架构问题。

### 8. raw_gis 路径

重点复核二阶段修复：

原 phase-1 有类似：

```python
target_mask[:, 2]
```

但 `target_mask` 已经是 DataFrame。

确认当前实现是否使用 `.iloc` / `to_numpy` 等正确方式，并检查：

- mask true-cell count 与 target row 对齐；
- dropna 后 invalid target cell 是否正确回写；
- target coords / target features 是否仍然一一对应。

不要擅自修改以下 baseline scientific compatibility 行为，除非单独建立版本化实验：

- split 前 correlation filtering；
- target 侧重新 fit scaler；
- random point-level validation；
- random unlabeled sampling 语义。

这些应记录为 research/science debt，而不是在“工程重构验证”里偷偷修正。

### 9. 可复现性和 Manifest

验证 manifest 是否记录：

- resolved config；
- task/model/operator/knowledge/predicate/splitter/metric/tuner；
- seed；
- feature schema；
- git commit；
- output files；
- 具体输入文件 hash。

重点检查 raster directory 是否最终解析到实际参与的 `.tif` 并记录 hash；shapefile 是否考虑 sidecar 文件。

同时判断当前 input provenance 是否存在明显性能问题，例如无必要地递归 hash 整个 repository root。

### 10. Optional dependency

确认：

- `skopt`
- `pulearn`
- GDAL
- geopandas / shapely
- legacy `lib_mpm`

是否只在需要对应能力时才要求，而不是 import 仓库就直接失败。

## 四、必须实际执行的验证

先执行：

```bash
python -m compileall -q src run.py
pytest -q
python run.py --config configs/experiments/lachlan_rf_phase2.yaml --validate-only
python run.py --config configs/experiments/lachlan_rf_baseline.yaml --validate-only
python scripts/list_components.py
python scripts/render_run_sh.py --config configs/experiments/lachlan_rf_phase2.yaml
```

记录每个命令的实际结果。

如果本地有 EarthByte baseline 数据，再执行：

```bash
MPM_BASELINE_SOURCE_ROOT=/实际/EarthByte-MPM_Lachlan_Porphyry pytest -q
```

如果没有，不要伪造 baseline parity 结果；明确写“external baseline artifacts unavailable”，并确认相关测试是 skip 而不是 pass。

## 五、增加两个架构验证实验

在不依赖真实 GIS 数据的前提下，用 temporary synthetic data 验证：

### A. train_from_archive_features

至少走通：

```text
DatasetRepository
-> Model Registry
-> Tuner Registry (none)
-> Knowledge Pipeline
-> Predicate Pipeline
-> Metric Registry
-> Task predict
-> manifest
```

### B. 插件扩展测试

临时增加一个最简单的测试组件，例如：

- dummy metric；或
- dummy feature operator；或
- dummy model adapter。

证明接入它不需要改 `Experiment`。

测试代码可以放 tests，不要污染生产 registry 名称。

## 六、发现问题后的处理原则

按以下优先级：

### P0/P1 工程缺陷

例如：

- 代码无法 import；
- 配置声称支持但实际无法切换；
- dataframe schema/label/sample_weight 契约错误；
- Experiment 仍写死具体算法；
- manifest 明显错误；
- phase-1 配置迁移改变实验含义。

可以直接做最小修复，并重新运行全部验证。

### Research/science debt

例如：

- random point split 有空间泄漏；
- target scaler 重新 fit；
- unlabeled 样本构造不严谨。

不要在本次验证中直接改算法行为。记录问题、影响和后续实验建议。

### Future capability

例如：

- 还没有 deep-edge；
- 还没有 spatial CV；
- 还没有真实 predicate；
- 还没有 XGBoost/LightGBM/GNN。

判断“现有架构是否允许新增”，不要因为能力尚未实现就否定整个重构。

## 七、最终必须输出的文件

### 1. `docs/PHASE2_CODE_REVIEW.md`

必须包含：

1. 总体结论：可作为后续主线 / 有条件可用 / 不建议继续；
2. 验证命令与实际结果；
3. 按 Task / Model / Operator / Knowledge / Predicate / Validation / Tuning / Config / Reproducibility 分项分析；
4. P0 / P1 / P2 问题清单；
5. research/science debt 单独列出；
6. 是否满足“新增组件不修改 Experiment”原则；
7. 下一阶段优先级。

不要写空泛的“模块化很好”。每个判断给出代码路径和具体依据。

### 2. `docs/PHASE2_USAGE_VERIFIED.md`

按实际代码写使用说明，至少覆盖：

- 查看 registry；
- validate-only；
- archive replay；
- train from archive features；
- raw GIS；
- 换模型；
- 加 operator；
- 加 predicate；
- 加 knowledge provider；
- 加 splitter / spatial CV；
- 加 metric；
- 换 tuner；
- YAML -> shell；
- baseline / knowledge / predicate 消融配置建议；
- 输出目录和 manifest 怎么读；
- 可选依赖。

### 3. 最终终端摘要

最后只输出：

```text
Review result: PASS / PASS WITH ISSUES / FAIL
Tests: ...
Critical issues fixed: ...
Remaining blockers: ...
Generated docs:
- docs/PHASE2_CODE_REVIEW.md
- docs/PHASE2_USAGE_VERIFIED.md
```

## 八、禁止事项

- 不要只读 README/MD 就判断完成度；
- 不要为“代码更简洁”改变 baseline 数学/科学行为；
- 不要把尚未实现的 deep-edge、spatial CV、真实 predicate 写成已支持；
- 不要通过在 `Experiment` 增加更多 if/elif 来修复插件问题；
- 不要因为缺少外部 baseline 数据就把 parity 测试写成通过；
- 不要删除 phase-1 compatibility wrapper，除非确认没有调用者且有替代迁移方案。
