# Phase 2 Code Review and Runtime Verification

Review date: 2026-08-13 UTC

## Final result

**PASS WITH ISSUES**

After the fixes made during this review, the repository implements a real Registry-driven experiment framework for Task, Model, Feature Operator, Knowledge Provider, Predicate, Splitter, Metric, Tuner, and the legacy PUB label refiner. A model plugin was loaded from YAML, registered at import time, selected by `model.name`, trained, evaluated, and used for prediction without changing `src/core/experiment.py`; see `tests/test_framework.py:525`.

The remaining issues are operational or capability gaps rather than a fake Registry architecture:

- `raw_gis` was not run end to end because this environment cannot import the original `lib_mpm.py`: `rasterio` and `scikit-image` are unavailable. Their requirements are now explicit in `requirements-optional.txt`.
- `tuning=bayes` and `label_refinement=pub` cannot run in this environment because `scikit-optimize` and `pulearn` are unavailable. The optional-dependency error path is tested at `tests/test_framework.py:417`.
- This copied source tree and the EarthByte data tree have no usable Git metadata, so the verified manifest records `git_commit: null`; the code-root lookup is in `Experiment.export()` (`src/core/experiment.py:334`).
- `deep_edge_prediction` remains an explicit placeholder, as allowed by the acceptance criteria (`src/tasks/deep_edge.py:11`).

No P0 or P1 blocker remains after the fixes below. Remaining items are P2 or research/science debt.

## Commands and actual results

| Command | Result |
|---|---|
| `python -m compileall -q src run.py scripts` | PASS, exit 0, no output |
| `pytest -q` | Environment launcher failure before collection: `/home/dmz-ai/.local/bin/pytest` uses `/usr/bin/python3`, where `pytest` is absent |
| `python -m pytest -q` after repository `pytest.ini` isolation | PASS, 24 passed, 5 expected baseline warnings |
| `MPM_BASELINE_SOURCE_ROOT=/home/dmz-ai/zhengzengrong/MPM/EarthByte-MPM_Lachlan_Porphyry python -m pytest -q` | PASS, 24 passed, 5 warnings; external artifacts were present and no baseline test was skipped |
| `python run.py --config configs/experiments/lachlan_rf_phase2.yaml --validate-only` | PASS; resolved Task `target_area_prediction`, Model `rf`, PUB, five ordered operators, random holdout, stratified CV, six metrics, Bayes tuner |
| `python run.py --config configs/experiments/lachlan_rf_baseline.yaml --validate-only` | PASS; resolved component graph equals phase-2 config for model, operators, PUB, tuning, validation, research unit, and prediction |
| `python scripts/list_components.py` | PASS; lists 2 tasks, 1 model, 1 refiner, 5 operators, 1 knowledge provider, 1 predicate, 2 splitters, 6 metrics, 2 tuners |
| `python scripts/render_run_sh.py --config configs/experiments/lachlan_rf_phase2.yaml` | PASS; emitted strict Bash wrapper and `python run.py --config ...` |
| `python run.py --config configs/experiments/lachlan_rf_phase2.yaml` | PASS; real archive replay completed under `outputs/lachlan_rf_phase2` |
| `python -m pytest -q tests/test_framework.py -k 'end_to_end or yaml_plugin'` | PASS; archive replay, archive-feature training, and YAML dummy-model plugin all passed |

The external artifacts were available, so the required wording `external baseline artifacts unavailable` does not apply. The parity checks were real passes, not skips (`tests/baseline/test_baseline_artifacts.py:48`).

## Architecture assessment

### Registry and bootstrap

`ComponentRegistry` rejects empty names, rejects duplicate registration unless `replace=True`, reports unknown names with the available set, constructs factories, and invokes component-owned configuration validators (`src/core/registry.py:20`, `src/core/registry.py:37`, `src/core/registry.py:55`). Separate registries exist for every required domain. Built-ins are imported once by `load_builtin_components()` (`src/core/bootstrap.py:8`); built-in modules depend on registry modules rather than importing `Experiment`, so no circular Experiment/implementation dependency was found.

External modules are loaded through top-level YAML `plugins`, before Registry validation (`src/core/config.py:265`). `scripts/list_components.py --config <yaml>` can include those plugin registrations. This closes the previous bootstrap gap where a new out-of-tree implementation could not be selected from a standalone YAML process.

### Config and ExperimentSpec

`ExperimentSpec` contains component choices rather than implementation objects (`src/core/spec.py:20`). Phase-1 migration preserves boolean feature order, `model.search`, `model.pu`, random holdout, CV folds, and primary scoring (`src/core/config.py:118`). Direct comparison of the supplied phase-1 and phase-2 YAMLs passed in `tests/test_framework.py:109`.

Validation resolves component names from registries and delegates Task, Splitter, and Operator parameter rules to component hooks (`src/core/config.py:253`). Neutral defaults are now `model.params={}`, `label_refinement.enabled=false`, and `tuning=none` (`src/core/config.py:71`), preventing RF-only parameters/search space from leaking into a newly selected model. Archive modes require `dataset.archive_dir`; `raw_gis` does not (`src/core/config.py:330`).

Mode boundaries are explicit in `Experiment.prepare_data/build_features/prepare_dataset/train/predict` (`src/core/experiment.py:67`, `src/core/experiment.py:83`, `src/core/experiment.py:110`, `src/core/experiment.py:184`, `src/core/experiment.py:301`):

- `archive_replay`: load and snapshot archive tables, copy archived model and predictions, do not recompute metrics.
- `train_from_archive_features`: train on archived `Xy_rf_train.csv`, evaluate on archived `Xy_rf_test.csv`; PUB and holdout are correctly marked `precomputed_in_archive` in runtime metadata (`src/core/experiment.py:192`).
- `raw_gis`: construct research units with the Task, execute operators, preprocess, refine labels, split, train, evaluate, and predict.

### Task

`TargetAreaPredictionTask` owns positive units, unlabeled units, prediction grid, valid-target filtering, coordinate semantics, score normalization, archive prediction artifacts, tabular output, mask reconstruction, and GeoTIFF export (`src/tasks/target_area.py:22`, `src/tasks/target_area.py:85`, `src/tasks/target_area.py:161`). Task-owned unit columns are retained as row-aligned metadata for future spatial/leave-area splitters (`src/core/experiment.py:78`, `src/features/preprocess.py:71`, `src/core/contracts.py:46`).

`deep_edge_prediction` is registered but raises a capability error naming the missing 3D/voxel, depth-label, operator, and output contracts (`src/tasks/deep_edge.py:12`). Raw 3D prediction/outputs can be implemented as a new Task plus suitable operators/data providers without a deep-edge branch in `Experiment`, because target preparation and export belong to the Task contract. Reusing archive modes will additionally require a versioned 3D archive/DatasetRepository contract.

### Model and tuner

The RF adapter builds `RandomForestClassifier` and exposes fit parameters; Bayes search is in `src/tuning/bayes.py`, not the primary RF adapter (`src/models/rf.py:17`, `src/tuning/bayes.py:44`). The phase-1 `train_rf` function remains only as a compatibility wrapper (`src/models/rf.py:38`) and is not called by `Experiment`.

`tuning=none` performs direct weighted fit (`src/tuning/none.py:13`). `tuning=bayes` imports `skopt` lazily, converts the YAML search space, builds CV through the Splitter Registry, and builds its scorer through the Metric Registry (`src/tuning/bayes.py:14`, `src/tuning/bayes.py:59`, `src/tuning/bayes.py:67`). Both tuners use `fit_params_for`, so sample weight is propagated and a constraint-capable adapter has a defined `fit_params(data)` channel (`src/models/registry.py:13`). Unsupported constraints fail explicitly.

The dummy Logistic Regression plugin proves that `model.name` changes the actual estimator and artifact, not just manifest text (`tests/test_framework.py:525`).

### Feature operators and preprocessing

The preserved order is `raster_statistics -> texture -> elevation_gradient -> line_distance -> categorical_geology`; both supplied YAMLs resolve to that exact list (`tests/test_framework.py:109`). `FeaturePipeline` instantiates registry entries in config order and enforces equal row count plus duplicate-column checks, including duplicates internal to one output frame (`src/operators/features/pipeline.py:15`). `SpatialFeatureExtractor` is only a phase-1 wrapper around that pipeline (`src/features/spatial.py:32`).

The preprocessor now accepts numeric-only new operators and detects object/category/string feature columns rather than requiring legacy name prefixes; Task-owned unit metadata is excluded explicitly (`src/features/preprocess.py:22`). Baseline scientific behavior was not changed: correlation filtering still happens before splitting and target scaling still refits (`src/features/preprocess.py:76`, `src/features/preprocess.py:151`).

### Knowledge and predicates

Knowledge providers build namespaced artifacts (`src/knowledge/pipeline.py:23`). Predicates consume a context containing those artifacts and return `TrainingData` (`src/predicates/pipeline.py:23`). The synthetic test proves an artifact is read from `context["knowledge"]` and converted to a constraint (`tests/test_framework.py:319`). `empty` and `identity` remain wiring components only.

The contract is sufficient for geological/geochemical/geophysical artifacts and row-preserving constraints. A model that cannot consume constraints fails; a model claiming support must implement `fit_params(data)` (`src/models/registry.py:13`). Row-changing predicates still require a future explicit resampling contract, as enforced at `src/predicates/pipeline.py:27`.

### Validation and metrics

`random_holdout` and `stratified_kfold` are registered implementations (`src/validation/splitters.py:15`, `src/validation/splitters.py:41`). Splitters can receive complete `TrainingData`, including row-aligned research-unit metadata, so spatial block and leave-area-out implementations can be added without changing `Experiment` (`src/validation/splitters.py:75`).

All evaluation metrics come from `METRIC_REGISTRY`; accuracy, precision, recall, F1, ROC AUC, and confusion matrix use sample weights, including the corrected confusion matrix (`src/validation/metrics.py:20`, `src/validation/metrics.py:47`). `primary_metric` is converted from a registry function into the Bayes scorer with fold-aligned sample weights (`src/validation/metrics.py:52`). Non-scalar primary metrics fail clearly.

### raw_gis data alignment

The invalid DataFrame access `target_mask[:, 2]` is gone. The target-area Task uses `.iloc[:, 2]`, converts the flags explicitly, maps invalid feature rows back to the true mask-cell indices, clears those cells, and keeps target features and `X/Y` coordinates aligned (`src/tasks/target_area.py:85`). Archive schemas also require target feature/coordinate equality and true-mask count equality (`src/data/dataset.py:42`). The focused regression is at `tests/test_framework.py:437`.

The real raw GIS path was not executed because importing EarthByte `lib_mpm.py` fails at `import rasterio`; `scikit-image` is also absent. This is a remaining operational blocker for raw extraction in the current environment, not a demonstrated data-alignment failure.

### Reproducibility

`manifest.json` records the resolved config, seed, Task, all component selections, feature schema, output files, runtime component metadata, and code-root Git commit (`src/core/experiment.py:334`). In archive modes it hashes the actual consumed CSV/model/prediction artifacts and the config source rather than recursively hashing raw GIS inputs that were not used (`src/core/experiment.py:380`). Real replay produced 10 hashed archive artifacts and byte-identical copied model, CSV prediction, and GeoTIFF.

For `raw_gis`, raster directories resolve to actual `.tif` and `.tiff` files recursively; shapefile families include all same-stem sidecars (`src/core/experiment.py:390`, `src/core/experiment.py:442`). Ordinary directories are described but not recursively hashed, so the repository root is not traversed.

## Issues and fixes

### P0

None found.

### P1 fixed during review

- External components could not be imported from YAML. Added `plugins` module loading and component listing with config.
- RF defaults leaked into arbitrary models. Made model/tuning/refiner defaults neutral while preserving both supplied baseline configs.
- Archive manifests did not hash actual consumed artifacts and looked up Git in the data root. Fixed mode-specific hashes and code-root Git lookup.
- Metric Registry did not drive tuner scoring; weighted confusion matrix ignored weights. Added registry scorer and weighted confusion matrix.
- Constraints could be silently bypassed by a third-party tuner. Added the common model `fit_params(data)` contract and Experiment boundary check.
- Spatial CV had no research-unit metadata channel. Added row-aligned metadata and splitter data access.
- Core config validation still named built-in Task/Splitters. Moved rules to component-owned validation hooks.
- `raw_gis` incorrectly required `archive_dir`. Corrected mode-specific requirements.
- Feature preprocessing assumed legacy categorical names and at least one categorical column. Added dtype detection and numeric-only handling.
- Archive/run metadata conflated declared and executed PUB/holdout/CV. Added explicit precomputed/executed metadata.

### P2 remaining

- Built-in bootstrap imports are an explicit list (`src/core/bootstrap.py:8`); new in-tree built-ins must be imported there or exposed through a YAML plugin module. `Experiment` does not change, but automatic package discovery is not implemented.
- `train_from_archive_features` intentionally uses legacy-named persisted files `Xy_rf_train/test`; `DatasetRepository.train_split/test_split` hides that naming from `Experiment`, but a future generic archive format should make split roles explicit in dataset configuration (`src/data/dataset.py:32`).
- `TargetAreaPredictionTask.export_geotiff` hardcodes EPSG:4283 (`src/tasks/target_area.py:143`). This should become task/dataset configuration before supporting regions in other CRSs.
- `FeaturePipeline` operators receive original research units, not accumulated preceding operator outputs (`src/operators/features/pipeline.py:30`). This is correct for the current independent GIS operators but should be documented or extended if derived operators need upstream features.
- There is no static type-check or formal runtime Protocol conformance test; failures are currently duck-typed at construction/use time.
- The user-level `pytest` executable has a wrong interpreter. Repository `pytest.ini` isolates the broken LangSmith plugin for `python -m pytest`, but cannot repair `/home/dmz-ai/.local/bin/pytest`.

## Research/science debt (not changed)

- Correlation filtering before split can leak information (`src/features/preprocess.py:76`).
- Target `StandardScaler` is refit on the prediction area (`src/features/preprocess.py:151`).
- Validation remains random point-level holdout and can leak spatial autocorrelation (`src/validation/splitters.py:24`).
- Unlabeled examples are random boundary samples, use the global NumPy RNG, and do not use exclusion buffers or verified barren masks (`src/data/research_units.py:36`).
- Target-area min-max normalization produces relative scores, not calibrated probabilities (`src/tasks/target_area.py:113`).
- PUB itself is a legacy RF-based relabeling method with a Bayes search; it is intentionally separated from the selected primary model but remains a scientific design choice (`src/models/pu.py:21`).

## Extension principle verdict

**Satisfied after the review fixes.** Under the normal path, a new component is added by implementing the appropriate contract, registering it, importing its module through built-in bootstrap or YAML `plugins`, and selecting it in YAML. No `src/core/experiment.py` modification is required for new models, feature operators, knowledge providers, row-preserving predicates, splitters, metrics, or tuners. The command-level YAML plugin test is the decisive evidence (`tests/test_framework.py:525`).

Exceptions that would require a new core contract rather than an implementation are genuinely new orchestration semantics: row-changing predicates/resampling, non-classifier prediction interfaces without `predict_proba`, or 3D outputs not expressible through the current Task protocol.

## Next priorities

1. Create a reproducible environment and run the full `raw_gis` path with `rasterio`, `scikit-image`, GDAL, GeoPandas, and the real EarthByte data; compare regenerated intermediate schemas and map geometry.
2. Install and run `scikit-optimize` and `pulearn` with a reduced synthetic search, then run a controlled real-data Bayes/PUB experiment.
3. Implement and test `spatial_block_kfold` and `leave_area_out` using `TrainingData.metadata["units"]`.
4. Version the archive dataset contract so split roles and preprocessing provenance are generic rather than RF-named.
5. Add CRS to Task/dataset configuration before expanding beyond EPSG:4283.
6. Treat correlation leakage, target scaler refit, random negatives, and score calibration as separate preregistered research comparisons, not silent engineering cleanups.
