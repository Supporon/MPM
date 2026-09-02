"""二维靶区预测 Task。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..data.research_units import build_positive_units, build_prediction_grid, sample_unlabeled_units
from .registry import TASK_REGISTRY


class TaskCapabilityError(NotImplementedError):
    """Task 缺少某项操作所需的数据或实现时抛出的异常。"""


@TASK_REGISTRY.decorator("target_area_prediction")
class TargetAreaPredictionTask:
    """负责二维靶区找矿预测的研究单元与预测语义。"""

    def __init__(self, task_config: Mapping[str, Any], prediction_config: Mapping[str, Any]):
        self.task_config = dict(task_config)
        self.prediction_config = dict(prediction_config)
        self._normalization_range: tuple[float, float] | None = None

    @staticmethod
    def validate_config(
        task_config,
        research_unit_config,
        prediction_config,
        dataset_config,
        execution_mode,
    ) -> None:
        if research_unit_config["type"] != "point_local_environment":
            raise ValueError(
                "target_area_prediction currently requires research_unit.type=point_local_environment"
            )
        if prediction_config["score_type"] not in {"probability", "raw_score", "relative_score"}:
            raise ValueError(
                "target_area_prediction score_type must be probability, raw_score, or relative_score"
            )
        if prediction_config["normalization"] not in {"minmax", "none", None}:
            raise ValueError("target_area_prediction normalization must be minmax or none")
        if (
            prediction_config["score_type"] in {"probability", "raw_score"}
            and prediction_config["normalization"] == "minmax"
        ):
            raise ValueError(
                "target_area_prediction normalization=minmax is only valid with "
                "score_type=relative_score; probability/raw_score must use normalization=none "
                "(MinMax-normalized values are not calibrated probabilities)."
            )
        if float(research_unit_config["prediction_grid_size"]) <= 0:
            raise ValueError("target_area_prediction prediction_grid_size must be positive")
        if execution_mode == "raw_gis":
            required = {
                "occurrence",
                "boundary",
                "training_boundary",
                "geology",
                "magnetic",
                "gravity",
                "radiometric",
                "remote_sensing",
                "elevation",
                "seismic",
            }
            missing = sorted(required.difference(dataset_config))
            if missing:
                raise ValueError(f"target_area_prediction raw_gis dataset is missing: {missing}")
            geology_required = {"line_files", "metamorphic_facies", "intrusions", "rock_units"}
            geology_missing = sorted(geology_required.difference(dataset_config["geology"]))
            if geology_missing:
                raise ValueError(
                    f"target_area_prediction raw_gis geology is missing: {geology_missing}"
                )

    def build_positive_units(self, dataset_config, research_unit_config, label_config) -> pd.DataFrame:
        return build_positive_units(dataset_config["occurrence"], label_config)

    def build_unlabeled_units(
        self, dataset_config, research_unit_config, label_config, count: int, seed: int | None = None
    ) -> pd.DataFrame:
        return sample_unlabeled_units(
            dataset_config["training_boundary"], count, label_config, seed=seed
        )

    def build_prediction_units(self, dataset_config, research_unit_config):
        units, mask = build_prediction_grid(
            dataset_config["boundary"], float(research_unit_config["prediction_grid_size"])
        )
        return units, pd.DataFrame(mask)

    @staticmethod
    def prepare_prediction_data(
        target_data: pd.DataFrame, target_mask: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """移除无效目标特征，并清除其对应的二维掩膜单元。"""
        valid_target_rows = ~target_data.isna().any(axis=1)
        target = target_data.loc[valid_target_rows].reset_index(drop=True)
        aligned_mask = target_mask.copy()
        inside_flags = (
            aligned_mask.iloc[:, 2]
            .astype(str)
            .str.lower()
            .isin({"1", "true", "1.0"})
            .to_numpy()
        )
        inside_indices = np.flatnonzero(inside_flags)
        if len(inside_indices) != len(valid_target_rows):
            raise ValueError("Raw target grid and target mask no longer align")
        invalid_inside = inside_indices[~valid_target_rows.to_numpy()]
        if len(invalid_inside):
            aligned_mask.iloc[invalid_inside, 2] = False
        return target, target[["X", "Y"]].reset_index(drop=True), aligned_mask

    def predict(self, model: Any, target_features: pd.DataFrame, target_coords: pd.DataFrame) -> pd.DataFrame:
        """返回有效目标单元的坐标与正类得分。

        根据 score_type 配置决定输出列名与语义：
        - ``probability``: ``predict_proba`` 的正类概率（未校准），列名 ``prob``（向后兼容）
        - ``raw_score``: 模型 ``decision_function`` 原始决策分数，列名 ``raw_score``
        - ``relative_score``: 相对分数；``normalization=minmax`` 时做 MinMax 并记录范围，
          列名 ``relative_score``
        """
        score_type = self.prediction_config.get("score_type", "probability")
        if score_type not in {"probability", "raw_score", "relative_score"}:
            raise ValueError(f"Unsupported score_type: {score_type!r}")

        if score_type == "raw_score":
            if not hasattr(model, "decision_function"):
                raise ValueError(
                    "score_type=raw_score requires a model exposing decision_function "
                    "(e.g. SVM / logistic regression). RandomForest only exposes "
                    "predict_proba; use score_type=probability or relative_score."
                )
            scores = np.asarray(model.decision_function(target_features), dtype=float).ravel()
            score_col = "raw_score"
        else:
            scores = np.asarray(model.predict_proba(target_features)[:, 1], dtype=float).ravel()
            score_col = "relative_score" if score_type == "relative_score" else "prob"

        normalization = self.prediction_config.get("normalization", "none")
        if normalization == "minmax":
            scaler = MinMaxScaler()
            scores = scaler.fit_transform(scores.reshape(-1, 1)).ravel()
            self._normalization_range = (
                float(scaler.data_min_[0]),
                float(scaler.data_max_[0]),
            )
            score_col = "relative_score"
        elif normalization not in {None, "none"}:
            raise ValueError(f"Unsupported target score normalization: {normalization!r}")

        return pd.DataFrame(
            {"X": target_coords["X"].to_numpy(), "Y": target_coords["Y"].to_numpy(), score_col: scores}
        )

    @staticmethod
    def reconstruct_grid(probabilities: pd.DataFrame, target_mask: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flags = target_mask.iloc[:, 2].astype(str).str.lower().isin({"1", "true", "1.0"}).to_numpy()
        if int(flags.sum()) != len(probabilities):
            raise ValueError(
                "target_mask true-cell count does not match target probability rows; cannot reconstruct GeoTIFF"
            )
        # 动态检测分数列名（prob / raw_score / relative_score）
        score_col = next(
            (c for c in probabilities.columns if c in {"prob", "raw_score", "relative_score"}),
            "prob",
        )
        values = np.full(len(target_mask), np.nan, dtype=np.float32)
        values[flags] = probabilities[score_col].to_numpy(dtype=np.float32)
        x_values = np.unique(target_mask.iloc[:, 0].to_numpy(dtype=float))
        y_values = np.unique(target_mask.iloc[:, 1].to_numpy(dtype=float))
        return values.reshape((len(y_values), len(x_values))), x_values, y_values

    def export_geotiff(self, probabilities: pd.DataFrame, target_mask: pd.DataFrame, path: Path) -> Path | None:
        if not self.prediction_config["export_geotiff"]:
            return None
        try:
            from osgeo import gdal, osr
        except ImportError as error:  # pragma: no cover
            raise TaskCapabilityError("GeoTIFF export requires GDAL Python bindings") from error
        grid, x_values, y_values = self.reconstruct_grid(probabilities, target_mask)
        path.parent.mkdir(parents=True, exist_ok=True)
        step_x = float(np.diff(x_values).min())
        step_y = float(np.diff(y_values).min())
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(path), len(x_values), len(y_values), 1, gdal.GDT_Float32
        )
        dataset.SetGeoTransform((float(x_values.min()), step_x, 0, float(y_values.max()), 0, -step_y))
        srs = osr.SpatialReference()
        target_crs = self.prediction_config.get("target_crs")
        if target_crs is None:
            raise ValueError(
                "prediction.target_crs must be set when export_geotiff=true "
                "(an EPSG integer or a WKT/proj4 string). The Lachlan/NSW archive "
                "uses GDA94/EPSG:4283."
            )
        if isinstance(target_crs, int):
            srs.ImportFromEPSG(target_crs)
        else:
            srs.ImportFromWkt(str(target_crs))
        dataset.SetProjection(srs.ExportToWkt())
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(-9999.0)
        band.WriteArray(np.flipud(grid))
        dataset.FlushCache()
        dataset = None
        return path

    def export_predictions(
        self, predictions: pd.DataFrame, target_mask: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """持久化靶区表格评分，以及可选的概率 GeoTIFF。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "target_probs.csv"
        predictions.to_csv(csv_path, index=False)
        paths = [csv_path]
        geotiff_path = self.export_geotiff(
            predictions, target_mask, output_dir / "probability_map.tif"
        )
        if geotiff_path is not None:
            paths.append(geotiff_path)
        return paths

    @staticmethod
    def archived_prediction_artifacts(archive_dir: Path) -> dict[str, Path]:
        return {
            "target_probs.csv": archive_dir / "target_probs.csv",
            "probability_map.tif": archive_dir / "probability_map.tif",
        }


# 供一期调用方使用的向后兼容导入。
def create_task(name: str, config: Mapping[str, Any]) -> TargetAreaPredictionTask:
    from ..core.bootstrap import load_builtin_components
    from .registry import create_task as create_registered_task

    load_builtin_components()
    return create_registered_task(name, {}, config)
