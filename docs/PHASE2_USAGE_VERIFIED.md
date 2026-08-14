# 二阶段框架使用说明（已验证）

本说明基于 2026-08-13 实际执行过的代码编写。以下命令均假定当前目录为仓库根目录 `MPM_codex_phase2`。

## 环境安装

安装核心运行依赖和测试依赖：

```bash
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-test.txt
```

仅在需要对应能力时安装可选依赖：

```bash
python -m pip install -r requirements-optional.txt
```

`requirements-optional.txt` 中定义的可选能力对应关系如下：

- `scikit-optimize`：用于 `tuning.name: bayes` 和旧版 PUB 搜索。
- `pulearn`：用于 `label_refinement.name: pub`。
- `geopandas`、`shapely`、`rasterio`、`scikit-image`：用于 `raw_gis` 研究单元构建和旧版特征提取。
- GDAL Python 绑定（`osgeo`）：当 `prediction.export_geotiff: true` 时用于导出 GeoTIFF。

在已验证环境中，GeoPandas、Shapely 和 GDAL 可用；`rasterio`、`scikit-image`、`scikit-optimize` 和 `pulearn` 不可用。归档回放和 `tuning=none` 仅依赖核心依赖即可运行。

如果用户安装的 `pytest` 脚本的 shebang 指向了其他 Python，请使用 `python -m pytest -q`。仓库中的 `pytest.ini` 还禁用了一个与本项目无关且已损坏的 LangSmith 自动插件。

## 查看已注册组件

查看内置组件：

```bash
python scripts/list_components.py
```

如需同时加载 YAML 的 `plugins` 列表中指定的模块：

```bash
python scripts/list_components.py --config configs/experiments/my_experiment.yaml
```

组件注册由 `ComponentRegistry`（`src/core/registry.py:13`）管理。除非代码显式请求替换，否则重复名称会立即报错；使用未知名称时，错误信息会同时列出当前可用组件。

## 只校验 YAML，不读取数据

```bash
python run.py \
  --config configs/experiments/lachlan_rf_phase2.yaml \
  --validate-only
```

该命令会加载内置组件和 YAML 插件，应用一期配置迁移与默认值，根据 Registry 校验组件名称及组件自有参数，解析路径，构建 `ExperimentSpec`，最后打印选定的组件图。它不会实例化 `Experiment`，也不会访问数据集文件（`run.py:28`）。

仓库提供的一期 YAML 也可直接使用：

```bash
python run.py \
  --config configs/experiments/lachlan_rf_baseline.yaml \
  --validate-only
```

迁移后的 Operator、RF 参数、PUB 配置、贝叶斯搜索空间、随机留出、10 折分层交叉验证以及 F1 主指标，与仓库提供的二阶段 YAML 保持一致。

## 运行归档回放

当实验要求原样复用归档特征、模型和预测结果时，使用 `archive_replay`：

```yaml
experiment:
  name: lachlan_rf_replay
  output_dir: outputs/lachlan_rf_replay
  execution_mode: archive_replay

dataset:
  root: ../EarthByte-MPM_Lachlan_Porphyry
  archive_dir: ../EarthByte-MPM_Lachlan_Porphyry/Datasets/Outputs_Cu_Lachlan_v1.6
```

运行：

```bash
python run.py --config configs/experiments/lachlan_rf_phase2.yaml
```

已验证的执行路径如下：

```text
Config -> DatasetRepository -> 归档快照 -> 归档模型/预测 -> 指标元数据 -> manifest
```

归档回放不会重新训练，也不会重新计算指标。`component_metadata.model_artifact` 会记录 `replayed_from_archive`（`src/core/experiment.py:181`）。输入哈希覆盖实际使用的归档 CSV、模型和预测文件。所选 Task 负责声明需要回放哪些归档预测工件。

## 从归档特征重新训练

当需要在不重新执行 GIS 特征提取的情况下，基于持久化的一期特征与划分契约比较 Model、Tuner、Knowledge、Predicate 或 Metric 时，使用 `train_from_archive_features`：

```yaml
experiment:
  name: lachlan_logistic_archive_features
  output_dir: outputs/lachlan_logistic_archive_features
  execution_mode: train_from_archive_features

dataset:
  root: ../EarthByte-MPM_Lachlan_Porphyry
  archive_dir: ../EarthByte-MPM_Lachlan_Porphyry/Datasets/Outputs_Cu_Lachlan_v1.6

model:
  name: my_model
  params: {}

label_refinement:
  enabled: false

tuning:
  name: none
  params: {}

knowledge:
  enabled: true
  items:
    - name: empty
      params: {}

predicates:
  enabled: true
  combine: sequential
  items:
    - name: identity
      params: {}

prediction:
  score_type: probability
  normalization: minmax
  export_geotiff: false
```

实际执行路径如下：

```text
DatasetRepository -> Model Registry -> Knowledge Pipeline -> Predicate Pipeline
-> Tuner -> Metric Registry -> Task prediction -> manifest
```

该模式读取 `Xy_rf_train.csv` 和 `Xy_rf_test.csv`。这些文件已经包含归档中的 PUB 与留出划分结果，因此运行时不会再次执行 `label_refinement` 或留出划分；manifest 会将二者标记为 `precomputed_in_archive`（`src/core/experiment.py:192`）。使用 `tuning=none` 时，交叉验证会出现在声明组件 `components` 中，但不会在 `component_metadata` 中被标记为已执行。

## 运行原始 GIS 流程

配置示例：

```yaml
experiment:
  execution_mode: raw_gis

task:
  name: target_area_prediction

dataset:
  root: /path/to/EarthByte-MPM_Lachlan_Porphyry
  occurrence: /path/to/porphyry.shp
  boundary: /path/to/Lachlan_Boundary_edited.shp
  training_boundary: /path/to/NSW_boundary.shp
  geology:
    line_files: [/path/to/line1.shp]
    metamorphic_facies: [/path/to/facies.shp]
    intrusions: /path/to/intrusions.shp
    rock_units: /path/to/rock_units.shp
  magnetic: /path/to/Magnetic
  gravity: /path/to/Gravity
  radiometric: /path/to/Radiometric
  remote_sensing: /path/to/Remote Sensing
  elevation: /path/to/dem.tif
  seismic: /path/to/seismic.shp
```

`raw_gis` 模式不要求配置 `dataset.archive_dir`。Task 会创建正例、未标注样本和预测单元；之后依次执行已配置的 Operator 特征提取、预处理、可选 PUB、留出划分、Knowledge/Predicate、调优、评估和预测。

靶区预测 Task 显式使用 `.iloc[:, 2]`。删除无效特征行之后，其对应的真值掩膜单元会在预测和 GeoTIFF 重建前被清除（`src/tasks/target_area.py:85`）。有效目标数据准备、坐标语义以及输出文件名和格式均由 `Experiment` 委托给所选 Task。

当前环境无法完整运行该路径，因为 EarthByte 的 `lib_mpm.py` 会导入当前不可用的 `rasterio` 和 `scikit-image`。运行前需先安装相应可选依赖。

## 新增模型

创建一个可导入模块，例如 `plugins/my_models.py`：

```python
from sklearn.linear_model import LogisticRegression

from src.models.registry import MODEL_REGISTRY


@MODEL_REGISTRY.decorator("logistic")
class LogisticAdapter:
    name = "logistic"
    artifact_filename = "model_logistic.pkl"
    supports_constraints = False

    def build(self, params, seed):
        resolved = dict(params)
        resolved.setdefault("random_state", seed)
        return LogisticRegression(**resolved)

    def fit_params(self, data):
        return {"sample_weight": data.sample_weight}
```

在 YAML 中引用该模块和模型：

```yaml
plugins:
  - plugins.my_models

model:
  name: logistic
  params:
    max_iter: 1000

tuning:
  name: none
  params: {}
```

无需修改 `Experiment`。`tests/test_framework.py:525` 包含一个通过真实 YAML 加载 Logistic Regression 插件的测试。

对于支持约束的模型，应设置 `supports_constraints = True`，并通过 `fit_params(data)` 返回估计器所需的约束拟合参数。模型声明支持约束却未提供该方法时会显式失败（`src/models/registry.py:13`）。

## 新增 Feature Operator

```python
import pandas as pd

from src.operators.features.registry import FEATURE_OPERATOR_REGISTRY


@FEATURE_OPERATOR_REGISTRY.decorator("geochemical_ratio")
class GeochemicalRatioOperator:
    name = "geochemical_ratio"

    def __init__(self, context, params):
        self.context = context
        self.params = dict(params)

    @staticmethod
    def validate_config(params):
        if "numerator" not in params:
            raise ValueError("numerator is required")

    def extract(self, units: pd.DataFrame) -> pd.DataFrame:
        values = ...
        return pd.DataFrame({"geochemical_ratio": values})
```

YAML 中的排列顺序就是执行和输出顺序：

```yaml
features:
  operators:
    - name: raster_statistics
      params: {buffer_size: 10, buffer_shape: square}
    - name: geochemical_ratio
      params: {numerator: Cu, denominator: Mo}
```

Operator 输出必须与研究单元一一对应，行数完全相同，且列名必须唯一。输出列不得与 Task 单元列或前序 Operator 的列重复（`src/operators/features/pipeline.py:30`）。当前每个 Operator 接收的是原始研究单元 DataFrame，而不是前序 Operator 输出累积后的 DataFrame。

`SpatialFeatureExtractor` 仅用于兼容旧 API；新代码应使用 Registry Operator 和 `FeaturePipeline`（`src/features/spatial.py:32`）。

## 新增 Knowledge Provider

```python
from src.knowledge.registry import KNOWLEDGE_REGISTRY


@KNOWLEDGE_REGISTRY.decorator("geological_domains")
class GeologicalDomains:
    name = "geological_domains"

    def __init__(self, params):
        self.params = dict(params)

    def build(self, data, context):
        return {"domain_id": ...}
```

```yaml
knowledge:
  enabled: true
  items:
    - name: geological_domains
      params:
        source: /path/to/domains.gpkg
```

生成的工件以 `context["knowledge"]["geological_domains"]` 作为命名空间（`src/knowledge/pipeline.py:23`）。同一实验中的 Provider 名称必须唯一。

## 新增 Predicate

```python
from src.predicates.registry import PREDICATE_REGISTRY


@PREDICATE_REGISTRY.decorator("domain_constraint")
class DomainConstraint:
    name = "domain_constraint"

    def __init__(self, params):
        self.params = dict(params)

    def apply(self, data, context):
        domains = context["knowledge"]["geological_domains"]["domain_id"]
        return data.with_constraints({"domain_id": domains})
```

```yaml
predicates:
  enabled: true
  combine: sequential
  items:
    - name: domain_constraint
      params: {}
```

Predicate 可以修改标签、权重、元数据或约束，但必须保持行数不变。在引入明确的重采样契约之前，改变行数的 Predicate 会被拒绝（`src/predicates/pipeline.py:23`）。约束不得被静默忽略。

## 新增空间交叉验证或留区交叉验证

注册一个交叉验证 Splitter：

```python
from src.validation.registry import SPLITTER_REGISTRY


@SPLITTER_REGISTRY.decorator("spatial_block_kfold")
class SpatialBlockKFold:
    kind = "cross_validation"

    @staticmethod
    def validate_config(params):
        if int(params.get("n_splits", 0)) < 2:
            raise ValueError("n_splits must be at least 2")

    def build_cv(self, params, seed, data):
        units = data.metadata["units"]
        return ...  # 与 sklearn 兼容的 CV 对象，或由索引对组成的可迭代对象
```

在 YAML 中选择该组件：

```yaml
validation:
  cross_validation:
    name: spatial_block_kfold
    params:
      n_splits: 5
      block_size: 0.5
```

Task 的研究单元列保存在 `TrainingData.metadata["units"]` 中，并在留出划分切片时保持对齐（`src/core/contracts.py:46`）。留区 Splitter 可使用研究单元中的 `area_id` 列。对于 `train_from_archive_features`，当前旧版归档的 `Xy_rf_train/test` 不包含研究单元元数据；应使用 `raw_gis`，或先扩展并版本化归档契约，再使用空间交叉验证。

## 新增 Metric

```python
from src.validation.registry import METRIC_REGISTRY


@METRIC_REGISTRY.decorator("balanced_cost")
def balanced_cost(labels, predictions, probabilities, sample_weight):
    return float(...)
```

```yaml
validation:
  metrics: [accuracy, balanced_cost]
  primary_metric: balanced_cost
```

评估阶段直接调用 Registry 中的 Metric。贝叶斯调优也会基于同一个函数构建评分器，并传入与各折对齐的样本权重（`src/validation/metrics.py:52`）。主指标必须返回标量；混淆矩阵等结构化指标可以用于报告，但不能作为 Tuner 的优化目标。

## 新增 Tuner

```python
from src.models.registry import fit_params_for
from src.tuning.registry import TUNER_REGISTRY


@TUNER_REGISTRY.decorator("grid")
class GridTuner:
    name = "grid"
    uses_cross_validation = True

    def fit(self, model_adapter, data, model_params, tuning_params, cv_config, scoring, seed):
        model = model_adapter.build(model_params, seed)
        # 通过 src.validation.splitters.build_cv 构建 CV，
        # 通过 src.validation.metrics.build_metric_scorer 构建评分器。
        ...
        return fitted_model
```

```yaml
tuning:
  name: grid
  params:
    search_space: {...}
```

应使用 `fit_params_for(model_adapter, data)`，以统一处理样本权重和约束。请准确设置 `uses_cross_validation`，以便 manifest 如实记录组件是否实际执行。

## 根据 YAML 生成 Shell 脚本

直接打印脚本：

```bash
python scripts/render_run_sh.py \
  --config configs/experiments/lachlan_rf_phase2.yaml
```

写入可执行文件：

```bash
python scripts/render_run_sh.py \
  --config configs/experiments/lachlan_rf_phase2.yaml \
  --output scripts/generated/lachlan_rf_phase2.sh
```

生成器会在写入前校验 YAML（`scripts/render_run_sh.py:22`）。

## 组织基线与消融实验

除被检验的维度外，其余配置应完全一致。每个实验使用独立的实验名称和输出目录：

```text
configs/experiments/lachlan_rf_baseline.yaml
configs/experiments/lachlan_rf_knowledge.yaml
configs/experiments/lachlan_rf_predicate.yaml
configs/experiments/lachlan_rf_knowledge_predicate.yaml
```

建议按以下方式组织实验变体：

```yaml
# 基线
experiment: {variant: baseline}
knowledge: {enabled: false, items: []}
predicates: {enabled: false, items: []}
```

```yaml
# +knowledge（仅生成知识工件）
experiment: {variant: knowledge}
knowledge:
  enabled: true
  items: [{name: geological_domains, params: {...}}]
predicates: {enabled: false, items: []}
```

```yaml
# +predicate（仅当 Predicate 不依赖缺失的 Knowledge 时有效）
experiment: {variant: predicate}
knowledge: {enabled: false, items: []}
predicates:
  enabled: true
  combine: sequential
  items: [{name: prior_weight, params: {...}}]
```

```yaml
# +knowledge +predicate
experiment: {variant: knowledge_predicate}
knowledge:
  enabled: true
  items: [{name: geological_domains, params: {...}}]
predicates:
  enabled: true
  combine: sequential
  items: [{name: domain_constraint, params: {...}}]
```

如果 Model、数据划分、随机种子、特征归档或预处理来源也在无意中发生变化，则不应直接比较这些变体。

## 读取输出与 manifest

每次运行会创建：

```text
outputs/<experiment>/
  config_resolved.yaml
  manifest.json
  metrics.json
  intermediate/
  models/
  predictions/
  figures/
```

manifest 关键字段：

- `config`：完全解析后的配置。
- `components`：YAML/`ExperimentSpec` 声明的组件图。
- `component_metadata`：实际执行或标记为预计算的组件与阶段。
- `feature_schema`、`feature_count`：模型输入契约。
- `input_paths`：实际文件、文件大小和 SHA-256；原始 Shapefile 包含附属文件，栅格目录包含解析后的 TIFF 文件。
- `git_commit`：代码检出目录的提交哈希；源目录不是 Git 工作树时为 `null`。
- `output_files`：本次运行写入的、相对于运行目录的文件列表。

快速查看：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/lachlan_rf_phase2/manifest.json")
manifest = json.loads(path.read_text())
print(manifest["components"])
print(manifest["component_metadata"])
print(manifest["feature_count"])
print(manifest["output_files"])
PY
```

对于归档回放，复制后的模型与预测文件应与 `input_paths.archive_artifacts` 中相应条目的 SHA-256 一致。对于新训练的运行，应结合查看 `metrics.json`、模型工件、`predictions/target_probs.csv` 以及声明组件和实际组件元数据。
