# 重构前基线

## 目的与范围

本文记录 `../EarthByte-MPM_Lachlan_Porphyry` 在重构开始时的可执行链路和
已存在产物。它是行为兼容性基线，不是对科学假设的认可，也不改变任何算法、
标签、数据划分或模型。本阶段没有运行特征提取或训练。

机器可读的完整产物清单在 [baseline_manifest.json](baseline_manifest.json)。它
记录了 19 个源代码/环境文件的 SHA-256、71 个输入 GeoTIFF、22 个输入
Shapefile，以及四个 `Outputs*` 目录内的 307 个文件。CSV 记录完整列顺序、
行数、标签分布、特征数和 schema SHA-256；GeoTIFF 记录 shape、band、像素
类型和可读取的 TIFF 元数据；pickle 在不执行反序列化的前提下记录模型类型。

### 基线边界

- **contract artifacts**：`Outputs_Cu_Lachlan_v1.6` 与
  `Outputs_Cu_NSW_v1.6`。后续仅在明确声明实验变更时，才允许其 schema、
  标签、shape 或输出语义变化。
- **non-contract snapshots**：`Outputs` 是不完整早期中间结果；
  `Outputs_test` 是带有已知缓存/路径不一致的 Notebook 快照。二者保留用于
  审计，不作为需要复现的规范结果。
- 当前 Python 环境缺少 `pulearn`、`skopt`、`rasterio`、`xgboost` 和
  `lightgbm`，且归档模型来自 sklearn 1.1.2，当前环境为 sklearn 1.6.1。
  因此模型类型用 pickle 字节安全识别，未加载模型作推断，更未触发重训。

## A. 当前执行链路

```text
原始 GIS 输入
  -> MPM_Porphyry_{Lachlan,NSW}.ipynb
  -> lib_mpm.py 的点-线距离 / point-in-polygon / raster 窗口特征
  -> training_data_{deposit,unlab}_*.csv
  -> Xy_train_original.csv / corr.csv / encoder.pkl / st_scaler.pkl
  -> Xy_train.csv / Xy_pos_test.csv
  -> PUB (BaggingPuClassifier, RF base) -> Xy_train_new.csv
  -> RF -> Xy_rf_{train,test}.csv / model_rf.pkl / feature importance
  -> target grid -> target_*.csv / target_features.csv
  -> RF predict_proba -> 全预测区 MinMaxScaler -> target_probs.csv
  -> probability_map.tif

已有特征 CSV（NSW 归档）
  -> model_comparison/run_all.py + model_comparison/common.py
  -> RF / XGB / LGBM / GBDT / LR / SVM / PU-direct / iForest / Random
  -> 指标、P-A 图、概率 CSV、模型 pickle、GeoTIFF 和 PNG
```

### Notebook 主链

`MPM_Porphyry_Lachlan.ipynb` 的主分支读取 Lachlan 边界为预测区，网格间距
为 `0.05` 度；训练随机样本仍从 NSW 边界中取。其主分支多数缓存读写到
`Datasets/Outputs_test`，而后续“important features”重复分支改写为
`Datasets/Outputs`。

`MPM_Porphyry_NSW.ipynb` 使用 NSW 边界作为预测区，网格间距为 `0.1` 度，
读写路径为 `Datasets/Outputs`。这两个 Notebook 都不直接读
`Outputs_Cu_*_v1.6` 归档目录。

`lib_mpm.py` 是底层空间实现，未在本阶段修改：

- `get_dist_line`：14 个地质/地震线层的最近距离，Notebook 使用
  `distance_type='geodesic'`。
- `get_cat_data`：3 个 metamorphic facies、intrusion、rock unit 的
  `within` 空间连接类别。
- `get_grid_stat_features`：70 个磁性/重力/放射性/遥感层的
  mean/std/min/max/median。
- `get_grid_tex_features`：同一 raster 窗口的 GLCM 纹理。
- `get_grid_grad_stat_features`：DEM 的 x/y/both 梯度统计。

三类 raster 特征均以 `buffer_size=10` 调用；这是像元半边长，完整正方形为
`21 x 21` 像元，并非 10 km buffer。

### model_comparison 分支

`model_comparison/common.py` 固定读取
`Datasets/Outputs_Cu_NSW_v1.6`，并在其 `model_comparison/` 子目录写入结果。
`run_all.py` 固定 `N_ITER=100`、`CV=10`，在相同保存切分上运行：RF、XGB、
LGBM、GBDT、LR、SVM、PU-direct、Isolation Forest 与随机分数基线。它不从
原始 GIS 重建数据。

归档 `model_comparison_results.csv` 有 18 行（pre-PUB/post-PUB 各一行）；指标列
为 `Accuracy, Precision, Recall, F1-Score, ROC_AUC`。这些仍属于把随机
unlabelled 当作零类的评价，不构成空间独立或 verified-barren 验证。

## 当前数据、研究单元与标签

### 输入数据

完整文件级清单见 manifest。当前实际输入集合为：

| 类别 | 当前内容 | 数量/空间表达 |
|---|---|---:|
| 矿化 occurrence | `Mineral Occurrences/MinView/GSNSWDataset/porphyry.shp` | 277 个 point，EPSG:4283 |
| 研究边界 | NSW state polygon、Lachlan boundary | 各 1 个 polygon |
| 线要素 | intrusion/metamorphic/rock-unit boundary、isograd、VS cluster | 14 个距离特征 |
| 分类 polygon | 3 metamorphic facies + intrusion + rock unit | 5 原始分类列 |
| 磁性 | Magnetic | 22 raster |
| 重力 | Gravity | 21 raster |
| 放射性 | Radiometric | 12 raster |
| 遥感 | Remote Sensing | 15 raster |
| 高程 | Elevation DEM | 1 raster |
| 环境 | `env.yml`, `env_loose.yml`, `env_minimal.yml` | Python 3.10.14 被固定于 env.yml |

输入 raster 共 71 个；源数据、Notebook、`lib_mpm.py`、`model_comparison/*.py`
与 3 个环境文件均在 manifest 中有当前 SHA-256。现有 GIS 层的代码使用
EPSG:4283，未发现训练链中 reproject、resample 或 warp。

### 研究单元和标签

- 训练一行表示一个矿化 occurrence 或 NSW 内随机点的局部二维环境，不是单个
  raster pixel、geological object 或 3D voxel。
- 正类：`porphyry.shp` 的所有 point，hard label 为 1。`SIZE_CODE` 样本权重为
  `VLG=.5, LGE=.4, MED=.3, SML=.2, OCC=.1`。
- 未标注：NSW polygon 内随机点，hard label 为 0、权重 `.5`。该变量在代码中叫
  `unlab`，但 RF/PUB 的评估按 negative 使用。
- 预测一行是目标 polygon 内规则网格点；Lachlan 为 `.05` 度，NSW 为 `.1` 度。
- `dropna()` 后的 saved train input 是 277 positive 与 272 unlabelled。
  预处理先从 277 个 positive 中随机留出 70 个 `Xy_pos_test`，得到 `Xy_train`
  中 207 positive + 272 zero = 479 行。

## 特征、预处理、模型和评价

### 特征与 preprocessing

1. 合并坐标、785 个 raster 派生列、14 条距离列、5 个分类列，完整原始表为
   806 列（含 `X,Y,label,sample_weight`）。
2. 在合并后的全训练数据上，对 799 个数值候选作 absolute Spearman correlation；
   `> .7` 的列被删除。
3. 对分类列作 `OneHotEncoder(handle_unknown='ignore')`。
4. 对保留数值特征作 `StandardScaler`；归档结果的可训练特征为 138 列，外加
   `sample_weight,label`，所以 `Xy_*.csv` 是 140 列。
5. 重要特征重复分支以 RF importance `>= .01` 留 26 个特征；其 `Xy_important*`
   共有 28 列（26 + weight + label）。

### 数据划分和 CV

- 正例保留集：`train_test_split(..., train_size=.75, random_state=42)`，只切正例；
  所有 unlabelled 都进入初始训练表。
- PUB 和 RF：再对 `Xy_train` / `Xy_train_new` 使用
  `train_test_split(..., train_size=.75, random_state=42)`，归档为 359 train / 120 test。
- BayesSearchCV：`cv=10`、`n_iter=100`、`scoring='f1'`、`random_state=42`、
  `n_jobs=4`（model_comparison 通常 `n_jobs=-1`）。
- 无 spatial block、leave-region-out 或 spatial CV。

### 模型

| 分支 | 当前模型 | 已保存类型 |
|---|---|---|
| 主 Notebook | `BaggingPuClassifier(RandomForestClassifier)` | `model_pub.pkl` |
| 主 Notebook | `RandomForestClassifier` | `model_rf.pkl` |
| 重要特征重复分支 | 上述 PUB/RF | `*_important.pkl` |
| model_comparison | RF/XGB/LGBM/GBDT/LR/SVM/PU-direct/iForest/random | NSW `model_comparison/` 归档 |

RF 搜索空间：`bootstrap`、`max_depth=5..20`、`max_features in {None,sqrt,log2}`、
`min_samples_leaf=2..20`、`min_samples_split=2..30`、`n_estimators=10..200`。PUB
使用 RF base 和对应参数，并另调 `max_samples=.4..9`（主分支；重要特征分支用
训练集未标注数的约 `.5..9` 范围）。所有受监督 `fit` 都传入样本权重。

评价当前输出 confusion matrix、accuracy、precision、recall、F1、ROC/AUC；
model_comparison 还输出概率分布、P-A 图和汇总面板。Notebook 的 `predict_proba`
正类列还会在**整个目标区**使用 `MinMaxScaler.fit_transform` 后才写出。

## B. Baseline artifacts

### 主要 archive contracts

| Artifact | Lachlan v1.6 | NSW v1.6 |
|---|---:|---:|
| `training_data_deposit.csv` | 277 x 806，label `{1:277}` | 277 x 806，label `{1:277}` |
| `training_data_unlab.csv` | 272 x 806，label `{0:272}` | 272 x 806，label `{0:272}` |
| `Xy_train.csv` | 479 x 140，label `{0:272,1:207}` | 同左 |
| `Xy_train_new.csv` | 479 x 140，label `{0:272,1:207}` | 同左 |
| `Xy_rf_train.csv` | 359 x 140，`{0:208,1:151}` | 同左 |
| `Xy_rf_test.csv` | 120 x 140，`{0:64,1:56}` | 同左 |
| `Xy_pos_test.csv` | 70 x 140，`{1:70}` | 同左 |
| `target_features.csv` | 4,071 x 138 | 7,583 x 138 |
| `target_probs.csv` | 4,071 x 4 | 7,583 x 4 |
| `target_mask.csv` | 4,200 x 3 | 11,938 x 3 |
| `probability_map.tif` | 70 rows x 60 cols, one float32 band | 94 rows x 127 cols, one float32 band |

在这两份归档中，`target_features.csv` 的 138 列与 `Xy_train.csv` 删除
`sample_weight,label` 后的训练 feature 列顺序完全一致。`Xy_train_new.csv` 与
`Xy_train.csv` 的标签分布完全相同，表明该存档的 PUB 步骤没有留下新增 pseudo-positive。

两份归档还分别包含 26-feature `*_important` 数据、两个 main/important GeoTIFF、
encoder/scaler、PUB/RF pickle 和 Bayesian-optimization PNG。NSW archive 另有
166 个 `model_comparison/` 产物（CSV、PNG、pickle、GeoTIFF）。详单、列名和每个
文件的大小都在 manifest。

### 非 contract 快照

`Outputs` 仅包含 4 个早期 deposit 中间 CSV。`Outputs_test` 有 37 个文件：其
`target_data.csv`、`target_lines.csv`、`target_categorical.csv` 与
`target_coords.csv` 有 4,661 行，`target_grids.csv`、`target_features.csv`、
`target_probs.csv` 只有 4,071 行，`target_mask.csv` 有
4,800 行，最终 GeoTIFF 是 80 x 60。这是既有快照状态，不应被误读为一致的单次运行。

## 已知问题：仅标记，不在本阶段修复

1. **Outputs 路径不一致**：Lachlan 主分支使用 `Outputs_test`，但重要特征分支
   又使用 `Outputs`；NSW 使用 `Outputs`，归档使用 `Outputs_Cu_*_v1.6`，而
   `model_comparison` 固定使用 NSW archive。
2. **陈旧缓存复用**：大量 `if os.path.isfile(...)` 直接读取旧 CSV、pickle 或
   GeoTIFF；上游参数、输入或代码改变时不会使缓存失效。
3. **target scaler 重 fit**：目标 feature 分支执行
   `st_scaler.fit(target_data_num)` 再 `transform`，并非训练 scaler 的 transform。
4. **random train_test_split**：存在随机点级切分（尽管固定 `random_state=42`），
   不代表空间独立切分。
5. **普通 10-fold CV 空间泄漏**：BayesSearchCV 的 `cv=10` 没有空间块约束。
6. **correlation filtering 在 split 前**：Spearman 筛列观察了完整合并训练数据。
7. **unlabelled 作为 negative 评价**：随机 0 没有 verified barren 证据，却用于
   negative、混淆矩阵和各项监督指标。
8. **probability 后再 MinMaxScaler**：全目标区的 MinMax 改变数值标度；输出是
   相对归一化远景评分，不能作为校准概率或跨区直接比较。
9. **deep/deep-edge prediction 不存在**：仅有二维 GIS 和 VS boundary 线；没有
   3D geology、voxel、钻孔 interval、深度标签或对应实现。
10. **环境/模型兼容性风险**：归档部分模型不能在当前 sklearn 版本反序列化，PUB
    还依赖未安装的 `pulearn`。这不是模型差异证据，也不在本阶段通过重训掩盖。

## D. 后续重构必须保持不变的行为

除非新实验明确版本化并写出与本基线的差异，结构重构必须保持：

- 同一输入层、输入列名和 feature 列顺序；训练 feature 为 138，重要 feature 为 26。
- 同一训练研究单元、正/未标注标签、`SIZE_CODE` 权重和样本行数/标签分布。
- 同一 `buffer_size=10` 像元窗口、14 距离特征、5 个原始分类列、相关筛选阈值和
  OneHot/StandardScaler 行为。
- 同一正例 `.75/.25` 保留逻辑、PUB/RF `.75/.25` 随机 split、10-fold CV、RF/PUB
  搜索空间、F1 搜索目标及 sample weights。
- 同一 Lachlan/NSW target 网格分辨率、有效 target feature schema/行数、
  `predict_proba -> target-area MinMax` 的当前评分语义，以及 GeoTIFF shape。
- 在没有明确科学变更的情况下，继续将输出称为二维相对 prospectivity score，而非
  calibrated probability、verified negative classification 或 deep prediction。

这些约束由 `tests/baseline/test_baseline_artifacts.py` 对两份 archive contract
执行：CSV schema、特征数、标签分布、target-vs-train schema、PUB/RF 保存标签关系
和 GeoTIFF shape。`Outputs_test` 的不一致也有显式测试，但不作为 parity gate。

## E. 新增文件

- `MPM_codex/README.md`
- `MPM_codex/baseline_tools.py`
- `MPM_codex/scripts/create_baseline_manifest.py`
- `MPM_codex/docs/BASELINE.md`
- `MPM_codex/docs/baseline_manifest.json`
- `MPM_codex/tests/baseline/test_baseline_artifacts.py`

运行检查：

```bash
python -m unittest discover -s MPM_codex/tests/baseline -v
```

为将来迁移后的 artifact tree 检查同一 contract：

```bash
MPM_BASELINE_SOURCE_ROOT=/path/to/migrated-artifacts \
  python -m unittest discover -s MPM_codex/tests/baseline -v
```

## F. 下一阶段建议

先做不改变科学逻辑的工程迁移：建立单一 run/output layout、显式 input/parameter
manifest、可复用的 orchestrator，并逐步从 Notebook 调用现有 `lib_mpm.py`。每一步
只迁移一个阶段，并运行本基线测试，确保 schema、label、target schema 和 map shape
不变。等结构等价通过后，才以独立、版本化实验讨论空间 CV、scaler 修复、标签假设、
概率校准或 3D/deep prediction；这些都不应混入结构重构提交。
