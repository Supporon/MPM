# Phase 1 Migration Map

## Scope

Phase 1 converts the baseline workflow into a minimal configured experiment
entry point. It does not change scientific assumptions or make a fresh full
baseline training run. The standard configurations use `archive_replay`: the
exact saved RF, feature tables, probability CSV, and GeoTIFF are copied to a
new isolated `outputs/<experiment_name>/` run directory. This is the only
mode guaranteed to run in the current environment because the legacy
`pulearn` and `scikit-optimize` dependencies are not installed.

`train_from_archive_features` and `raw_gis` are implemented execution modes.
They call the same new interfaces, but a legacy-compatible Bayes/PUB run needs
the environment declared in `../EarthByte-MPM_Lachlan_Porphyry/env.yml`.

## Notebook Mapping

| Original Notebook responsibility | Phase 1 destination |
|---|---|
| Imports, fixed paths, scattered constants | `configs/experiments/*.yaml`, `src/core/config.py` |
| Cell 4: occurrence, NSW/Lachlan boundaries | `src/data/research_units.py`, `dataset.{occurrence,boundary,training_boundary}` |
| Cells 6-30: input layer lists | YAML `dataset.geology`, raster directories, elevation, seismic |
| Cell 32: SIZE_CODE hard label/weight | `src/data/labels.py` |
| Cell 34: distance to lines | `src/features/spatial.py:line_distances()` -> `lib_mpm.get_dist_line()` |
| Cell 36: polygon categories | `src/features/spatial.py:categorical()` -> `lib_mpm.get_cat_data()` |
| Cell 38: raster statistics/texture/gradient | `src/features/spatial.py:raster_features()` -> three `lib_mpm` operators |
| Cells 40-45: merge, Spearman, OHE, scaler, positive holdout | `src/features/preprocess.py`, `src/data/dataset.py` |
| Cells 48-51: PUB/relabel | `src/models/pu.py` |
| Cells 53-61: RF, random split, metrics | `src/models/rf.py`, `src/validation/{splitters,metrics}.py` |
| Cells 69-72: target grid and target feature preparation | `src/data/research_units.py`, `src/features/preprocess.py`, `src/tasks/target_area.py` |
| Cells 75, 78-79: MinMax score, reconstruction, GeoTIFF | `src/tasks/target_area.py` |
| Cells 83-106: important-feature duplicate branch | Not made a task in phase 1; 26-feature archive remains in baseline manifest |

The NSW Notebook maps to the same modules. Its only phase-1 config differences
are the target boundary, `prediction_grid_size: 0.1`, archive directory, and
experiment/output names.

## Reused Legacy Code

`src/features/spatial.py` deliberately wraps, rather than rewrites:

- `lib_mpm.get_dist_line`
- `lib_mpm.get_cat_data`
- `lib_mpm.get_grid_stat_features`
- `lib_mpm.get_grid_tex_features`
- `lib_mpm.get_grid_grad_stat_features`

`model_comparison/common.py` was evaluated but is not imported directly:

- it hard-codes `Outputs_Cu_NSW_v1.6` and calls `os.makedirs()` there on import;
- it imports `skopt`, which is unavailable in the current environment;
- `predict_full_area()` and `export_geotiff()` hard-code a 0.1-degree export
  path and therefore cannot support Lachlan correctly without changing behavior.

The contained logic was extracted into phase-appropriate modules instead:
`train_model` -> `src/models/rf.py`; `evaluate_model` ->
`src/validation/metrics.py`; `predict_full_area`/`export_geotiff` ->
`src/tasks/target_area.py`. The extracted code preserves current random split,
weighted metrics, target-area MinMax normalization, and EPSG:4283 output
semantics, with explicit TODO warnings where the baseline is scientifically
unsafe.

## Configuration Contract

Every experiment config has these top-level sections:

```text
experiment, task, dataset, research_unit, label, features,
preprocess, model, validation, prediction
```

`src/core/config.py` supplies defaults and validates task/model names, required
dataset keys, positive grid/buffer size, correlation threshold, current
encoding/scaling, random split, F1 scoring, and MinMax probability output.
Relative paths resolve from the `MPM_codex` root, not the caller's working
directory.

## Current Run Contract

Each run creates only its own directory:

```text
outputs/<experiment_name>/
  config_resolved.yaml
  manifest.json
  metrics.json
  models/
  intermediate/
  predictions/
  figures/
```

The manifest records timestamp, fully resolved config, seed, nested input paths
and file hashes, feature schema/count, model config, source git commit when it
exists, and output files. No phase-1 code writes to `Outputs` or `Outputs_test`.

## Intentionally Unchanged

- Random point-level `train_test_split` and ordinary 10-fold CV.
- Correlation filtering before split.
- Random unlabelled points encoded/evaluated as zero.
- Target-side `StandardScaler.fit()` behavior in raw GIS mode.
- Target-area `MinMaxScaler.fit_transform()` of RF probabilities.
- No deep/deep-edge task: `deep_edge_prediction` raises a clear capability error.
