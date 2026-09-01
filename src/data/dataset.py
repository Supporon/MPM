"""以只读方式访问归档的基线特征数据集。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..utils.logging import get_logger


@dataclass
class ArchiveDataset:
    """从单个基线归档加载的内存数据表和源路径。"""

    archive_dir: Path
    xy_train: pd.DataFrame
    xy_train_new: pd.DataFrame
    xy_rf_train: pd.DataFrame
    xy_rf_test: pd.DataFrame
    target_features: pd.DataFrame
    target_coords: pd.DataFrame
    target_mask: pd.DataFrame
    source_files: dict[str, Path] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        """返回排除权重和标签字段后的训练列。"""
        return [column for column in self.xy_train.columns if column not in {"sample_weight", "label"}]

    @property
    def train_split(self) -> pd.DataFrame:
        """返回旧版归档训练划分，不向运行器暴露 RF 策略。"""
        return self.xy_rf_train

    @property
    def test_split(self) -> pd.DataFrame:
        """返回旧版归档测试划分，不向运行器暴露 RF 策略。"""
        return self.xy_rf_test

    def validate_schema(self) -> None:
        """要求特征 schema 一致，且目标特征与坐标的行数对齐。"""
        required_training = {"label", "sample_weight"}
        for name, frame in {
            "Xy_train": self.xy_train,
            "Xy_train_new": self.xy_train_new,
            "Xy_rf_train": self.xy_rf_train,
            "Xy_rf_test": self.xy_rf_test,
        }.items():
            missing = sorted(required_training.difference(frame.columns))
            if missing:
                raise ValueError(f"{name} is missing required columns: {missing}")
        if self.target_features.columns.tolist() != self.feature_columns:
            raise ValueError("target_features schema does not match Xy_train feature schema")
        for name, frame in {"Xy_train_new": self.xy_train_new, "Xy_rf_train": self.xy_rf_train, "Xy_rf_test": self.xy_rf_test}.items():
            if frame.columns.tolist() != self.xy_train.columns.tolist():
                raise ValueError(f"{name} schema does not match Xy_train")
        if len(self.target_features) != len(self.target_coords):
            raise ValueError("target_features and target_coords_purged row counts differ")
        if not {"X", "Y"}.issubset(self.target_coords.columns):
            raise ValueError("target_coords_purged must contain X and Y columns")
        if self.target_mask.shape[1] < 3:
            raise ValueError("target_mask must contain X, Y, and inside-mask columns")
        inside_flags = self.target_mask.iloc[:, 2].astype(str).str.lower().isin({"1", "true", "1.0"})
        if int(inside_flags.sum()) != len(self.target_features):
            raise ValueError("target_mask true-cell count does not match target_features rows")

    def summary(self) -> dict[str, Any]:
        """返回可序列化的归档形状、标签和特征元数据。"""
        return {
            "archive_dir": str(self.archive_dir),
            "feature_count": len(self.feature_columns),
            "feature_schema": self.feature_columns,
            "xy_train_shape": list(self.xy_train.shape),
            "xy_train_labels": {str(key): int(value) for key, value in self.xy_train["label"].value_counts().sort_index().items()},
            "xy_rf_train_shape": list(self.xy_rf_train.shape),
            "xy_rf_test_shape": list(self.xy_rf_test.shape),
            "target_features_shape": list(self.target_features.shape),
            "target_mask_shape": list(self.target_mask.shape),
        }


class DatasetRepository:
    """加载持久化输入，不使用旧的 Outputs/Outputs_test 路径。"""

    REQUIRED_FILES = (
        "Xy_train.csv",
        "Xy_train_new.csv",
        "Xy_rf_train.csv",
        "Xy_rf_test.csv",
        "target_features.csv",
        "target_coords_purged.csv",
        "target_mask.csv",
    )

    def __init__(self, config: Mapping[str, Any]):
        """将仓库绑定到配置的只读归档目录。"""
        self.config = config
        self.archive_dir = Path(config["archive_dir"])

    def load_archive(self) -> ArchiveDataset:
        """加载必需的归档 CSV 文件并校验其共享 schema。"""
        log = get_logger("dataset")
        if not self.archive_dir.is_dir():
            raise FileNotFoundError(f"Configured archive directory does not exist: {self.archive_dir}")
        paths = {name: self.archive_dir / name for name in self.REQUIRED_FILES}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Archive is missing required artifacts: {missing}")
        log.info("Loading %d archive files from %s", len(paths), self.archive_dir)
        dataset = ArchiveDataset(
            archive_dir=self.archive_dir,
            xy_train=pd.read_csv(paths["Xy_train.csv"]),
            xy_train_new=pd.read_csv(paths["Xy_train_new.csv"]),
            xy_rf_train=pd.read_csv(paths["Xy_rf_train.csv"]),
            xy_rf_test=pd.read_csv(paths["Xy_rf_test.csv"]),
            target_features=pd.read_csv(paths["target_features.csv"]),
            target_coords=pd.read_csv(paths["target_coords_purged.csv"]),
            target_mask=pd.read_csv(paths["target_mask.csv"], header=None),
            source_files=paths,
        )
        dataset.validate_schema()
        log.info("Archive loaded: %d features, train=%d, test=%d",
                 len(dataset.feature_columns), len(dataset.train_split), len(dataset.test_split))
        return dataset

    @staticmethod
    def snapshot(dataset: ArchiveDataset, output_dir: Path) -> list[Path]:
        """将本次运行实际使用的特征表原样复制到专属输出目录。"""
        destination = output_dir / "intermediate"
        destination.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for name, source in dataset.source_files.items():
            target = destination / name
            shutil.copy2(source, target)
            copied.append(target)
        return copied
