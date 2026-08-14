# MPM_codex Code Review and Docstrings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document every Python class, function, and method without changing behavior, verify the project, and deliver a source-grounded Chinese assessment of its suitability for mineral prospectivity mapping research.

**Architecture:** Keep the current configured phase-1 pipeline intact. Add docstrings only at AST definition boundaries, then describe the existing configuration → data/research unit → feature → preprocessing → PU/RF → validation → target-map flow in one review document. Treat archived replay capability and scientific research readiness as separate conclusions.

**Tech Stack:** Python 3, AST, unittest, pandas, NumPy, scikit-learn, GeoPandas/Shapely, YAML, GDAL, legacy `lib_mpm` adapter.

---

### Task 1: Establish the verification baseline

**Files:**
- Inspect: all `MPM_codex/**/*.py`
- Inspect: `MPM_codex/tests/test_framework.py`
- Inspect: `MPM_codex/tests/baseline/test_baseline_artifacts.py`

- [ ] **Step 1: Record the AST documentation baseline**

Run an AST scan over all Python files and record totals for modules, classes, functions/methods, and missing docstrings.

Expected baseline: 28 modules, 15 classes with 7 undocumented, and 99 functions/methods with 73 undocumented.

- [ ] **Step 2: Run the framework tests before editing**

Run: `python -m unittest tests.test_framework -v`

Expected: all seven framework tests pass, or any missing external GIS/runtime dependency is recorded separately from code failures.

- [ ] **Step 3: Run the artifact-contract tests before editing**

Run: `python -m unittest discover -s tests/baseline -v`

Expected: all five persisted-artifact contract tests pass, or missing adjacent baseline data is recorded explicitly.

### Task 2: Document core orchestration and data contracts

**Files:**
- Modify: `MPM_codex/src/core/config.py`
- Modify: `MPM_codex/src/core/experiment.py`
- Modify: `MPM_codex/src/data/dataset.py`
- Modify: `MPM_codex/src/data/research_units.py`

- [ ] **Step 1: Add missing configuration docstrings**

Document `_deep_merge`, `_require`, `_resolve_path`, `_resolve_dataset_paths`, `ExperimentConfig.output_dir`, `ExperimentConfig.name`, `ExperimentConfig.section`, `validate_config`, and `load_config`. The text must state merge/path-resolution behavior, required-key validation, and the returned resolved configuration without changing code.

- [ ] **Step 2: Add missing experiment-orchestrator docstrings**

Document `Experiment.__init__`, `mode`, `_prepare_output_layout`, `_input_paths`, its nested `describe`, and `run`. The descriptions must identify output-directory creation, recursive input provenance, stage order, and returned manifest.

- [ ] **Step 3: Add missing archived-dataset docstrings**

Document `ArchiveDataset`, `feature_columns`, `validate_schema`, `summary`, `DatasetRepository.__init__`, and `load_archive`. State that the repository reads persisted CSV artifacts and enforces feature/row alignment.

- [ ] **Step 4: Add missing research-unit docstrings**

Document `load_vector` and `build_positive_units`, including GeoPandas loading and conversion of occurrence geometry to `X`, `Y`, `label`, and `sample_weight` columns.

- [ ] **Step 5: Re-run the AST scan for these files**

Expected: zero undocumented classes/functions in the four modified modules.

### Task 3: Document features, models, tasks, validation, and utilities

**Files:**
- Modify: `MPM_codex/src/features/preprocess.py`
- Modify: `MPM_codex/src/features/spatial.py`
- Modify: `MPM_codex/src/models/rf.py`
- Modify: `MPM_codex/src/tasks/target_area.py`
- Modify: `MPM_codex/src/utils/files.py`
- Modify: `MPM_codex/src/validation/metrics.py`
- Modify: `MPM_codex/src/validation/splitters.py`

- [ ] **Step 1: Document preprocessing data and transformations**

Document `split_feature_columns`, `PreparedTrainingData`, `BaselinePreprocessor.__init__`, and `prepare_training`. Explicitly describe prefix-based categorical detection, pre-split correlation filtering, one-hot encoding, scaling, and positive holdout behavior.

- [ ] **Step 2: Document spatial adapters**

Document `SpatialFeatureExtractor.__init__`, `_operators`, `line_distances`, `categorical`, `raster_features`, and `_raster_files`. Preserve the fact that `lib_mpm` performs the actual GIS operations.

- [ ] **Step 3: Document RF construction**

Document `build_rf`, including default `n_jobs` and seed-derived `random_state` behavior.

- [ ] **Step 4: Document target-area task behavior**

Document `TargetAreaPredictionTask`, `__init__`, `predict`, `reconstruct_grid`, `export_geotiff`, and `create_task`. State that MinMax output is a relative score and that the GeoTIFF uses EPSG:4283.

- [ ] **Step 5: Document file, metric, and split containers**

Document `utc_timestamp`, `sha256_file`, `write_json`, and `write_yaml` in
`src/utils/files.py`, plus `evaluate_classifier` and `SplitData`.

- [ ] **Step 6: Re-run the AST scan for these files**

Expected: zero undocumented classes/functions in the seven modified modules.

### Task 4: Document command-line, baseline-inspection, and test code

**Files:**
- Modify: `MPM_codex/baseline_tools.py`
- Modify: `MPM_codex/run.py`
- Modify: `MPM_codex/scripts/create_baseline_manifest.py`
- Modify: `MPM_codex/tests/test_framework.py`
- Modify: `MPM_codex/tests/baseline/test_baseline_artifacts.py`

- [ ] **Step 1: Document baseline metadata helpers**

Document `sha256_file`, `schema_digest`, `_prepend`, `_read_at`, `_unpack_values`, nested `scalar`, and `artifact_metadata`, including the no-unpickle safety boundary.

- [ ] **Step 2: Document both command-line entry points**

Document `run.main` and `relative`, `source_files`, `raster_inventory`,
`vector_inventory`, `output_artifacts`, `count_by_parent`, and `main` in
`scripts/create_baseline_manifest.py`, stating arguments, inventories created,
output written, and exit status where applicable.

- [ ] **Step 3: Document test doubles and test cases**

Document `FakeLegacyOperators`, `FrameworkTests`, `get_dist_line`,
`get_cat_data`, `get_grid_stat_features`, `get_grid_tex_features`,
`get_grid_grad_stat_features`, `test_config_load`, `test_config_validation`,
`test_research_unit_smoke`, `test_feature_extraction_smoke`,
`test_rf_model_smoke`,
`test_end_to_end_small_archive_replay_and_baseline_schema`, and
`test_deep_edge_task_is_explicitly_unavailable`. Also document
`BaselineArtifactTests`, `setUpClass`, `expected`, `actual_path`, and
`assert_csv_contract`. Test docstrings must state the behavior or contract
being asserted.

- [ ] **Step 4: Run the full AST documentation audit**

Expected: 28/28 module docstrings, 15/15 class docstrings, and 99/99 function/method docstrings.

### Task 5: Produce the Chinese code and research-readiness report

**Files:**
- Create: `MPM_codex/docs/CODE_STRUCTURE_AND_RESEARCH_READINESS_CN.md`

- [ ] **Step 1: Describe the directory and execution architecture**

Include the complete Python file tree and the main flow from YAML loading through manifest export. Distinguish `archive_replay`, `train_from_archive_features`, and `raw_gis` modes.

- [ ] **Step 2: Describe every Python file**

For all 28 files, list path, responsibility, and contained classes/functions (or identify package marker files with no executable definitions).

- [ ] **Step 3: Assess mineral-prospectivity research suitability**

Support each conclusion with implementation/config/test evidence. Separate current strengths from limitations in spatial validation, PU labels, preprocessing leakage, score calibration, reproducibility, dependency compatibility, model breadth, uncertainty, explainability, multi-region generalization, and 3D/deep prediction.

- [ ] **Step 4: Give a prioritized research roadmap**

Recommend staged changes with P0/P1/P2 priority while preserving the archived baseline as a comparison contract.

### Task 6: Verify and review the final change set

**Files:**
- Verify: all `MPM_codex/**/*.py`
- Verify: `MPM_codex/docs/CODE_STRUCTURE_AND_RESEARCH_READINESS_CN.md`

- [ ] **Step 1: Compile every Python file**

Run: `python -m compileall -q .`

Expected: exit code 0.

- [ ] **Step 2: Run all existing tests**

Run: `python -m unittest discover -s tests -v`

Expected: all discovered tests pass; warnings that intentionally expose scientific debt may remain.

- [ ] **Step 3: Re-run the final AST audit**

Expected: no missing module, class, function, method, private-function, or nested-function docstrings.

- [ ] **Step 4: Review for behavioral changes**

Inspect every modified Python file and confirm only string-literal docstrings were inserted. Record the lack of Git metadata and use content/AST/test evidence instead of a Git diff.

- [ ] **Step 5: Check report completeness**

Verify that all 28 Python paths occur in the Chinese report and that its readiness conclusion is consistent with the implementation evidence.
