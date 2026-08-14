# MPM 二阶段使用说明

## 1. 查看当前可用组件

```bash
python scripts/list_components.py
```

## 2. 只验证配置，不访问数据

```bash
python run.py --config configs/experiments/lachlan_rf_phase2.yaml --validate-only
```

该命令用于检查 YAML 是否可解析、组件名称是否已经注册、参数基本约束是否成立，并输出最终组件图。

## 3. 运行实验

```bash
python run.py --config configs/experiments/lachlan_rf_phase2.yaml
```

如果只需要复用旧 YAML，原 `lachlan_rf_baseline.yaml` / `nsw_rf_baseline.yaml` 仍可被自动迁移。

## 4. 由配置生成 Shell

```bash
python scripts/render_run_sh.py \
  --config configs/experiments/lachlan_rf_phase2.yaml \
  --output scripts/generated/lachlan_rf_phase2.sh
```

自然语言入口不应绕过 YAML。推荐后续 Agent 工作流：

```text
用户自然语言
  -> 生成/修改 YAML
  -> python run.py --validate-only
  -> scripts/render_run_sh.py
  -> 执行 shell
```

这样 LLM 只负责配置编译，不负责决定运行时算法分支。

## 5. 常见研究修改路径

### 换模型

1. 在 `src/models/` 新增 adapter；
2. 注册到 `MODEL_REGISTRY`；
3. YAML 修改 `model.name`；
4. 不修改 `Experiment`。

### 增加特征算子

1. 在 `src/operators/features/` 新增 operator；
2. 注册到 `FEATURE_OPERATOR_REGISTRY`；
3. 在 `features.operators` 中增加配置项。

### 增加谓词

1. 在 `src/predicates/` 新增 predicate；
2. 注册；
3. 在 `predicates.items` 中启用；
4. 如果谓词需要外部地质知识，通过 `context["knowledge"]` 使用，不把知识源写死进 predicate。

### 增加空间交叉验证

1. 在 `src/validation/splitters.py` 或独立模块新增 splitter；
2. 注册；
3. 修改 `validation.holdout` 或 `validation.cross_validation`。

### 换调参方法

增加 Tuner 并修改：

```yaml
tuning:
  name: <new_tuner>
```

Model adapter 不应包含 GridSearch/Bayes/Optuna 等搜索逻辑。

## 6. Baseline / Knowledge / Predicate 对比

建议每个实验使用独立 YAML，并显式设置：

```yaml
experiment:
  variant: baseline
```

或：

```yaml
experiment:
  variant: knowledge
knowledge:
  enabled: true
```

或：

```yaml
experiment:
  variant: predicate
predicates:
  enabled: true
```

`manifest.json` 会记录最终组件选择，避免只靠文件名判断实验条件。

## 7. 测试

```bash
pytest -q
```

压缩包本身不含 EarthByte 原始数据，因此真实 baseline artifact 测试会 skip。提供原数据后执行：

```bash
MPM_BASELINE_SOURCE_ROOT=/path/to/EarthByte-MPM_Lachlan_Porphyry pytest -q
```

当前 portable tests 同时覆盖：配置迁移、注册表、Feature Pipeline、RF + Tuner、Metric、Knowledge/Predicate 接口、archive replay 和 archive-feature 重新训练。
