"""现有 MPM 工件的回归契约。

这些检查有意仅校验已持久化的工件，不导入 Notebook、不提取 GIS 特征、
不训练模型，也不重新生成地图。

可通过 MPM_BASELINE_SOURCE_ROOT 将检查指向未来迁移后的工件目录树；
该目录树需保留原有的 ``Datasets/Outputs_*`` 结构。
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

CODEX_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = CODEX_ROOT.parent
sys.path.insert(0, str(CODEX_ROOT))

from baseline_tools import csv_metadata, tiff_metadata  # noqa: E402


MANIFEST_PATH = CODEX_ROOT / "docs" / "baseline_manifest.json"
SOURCE_ROOT = Path(
    os.environ.get(
        "MPM_BASELINE_SOURCE_ROOT",
        str(WORKSPACE_ROOT / "EarthByte-MPM_Lachlan_Porphyry"),
    )
).resolve()

CONTRACT_FILES = (
    "training_data_deposit.csv",
    "training_data_unlab.csv",
    "Xy_train.csv",
    "Xy_train_new.csv",
    "Xy_rf_train.csv",
    "Xy_rf_test.csv",
    "Xy_pos_test.csv",
    "target_features.csv",
    "target_probs.csv",
    "target_mask.csv",
    "probability_map.tif",
)


class BaselineArtifactTests(unittest.TestCase):
    """将持久化基线工件与已记录的清单契约进行比较。"""

    @classmethod
    def setUpClass(cls) -> None:
        """加载清单，并要求调用方提供可用的基线数据集根目录。"""
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.datasets_root = SOURCE_ROOT / "Datasets"
        if not cls.datasets_root.is_dir():
            raise unittest.SkipTest(
                f"External baseline artifacts are not bundled: {cls.datasets_root}. "
                "Set MPM_BASELINE_SOURCE_ROOT to run byte/schema parity gates."
            )

    def expected(self, output_set: str, filename: str) -> dict[str, object]:
        """返回某个输出集中一个工件的已记录元数据。"""
        path = f"{output_set}/{filename}"
        return self.manifest["output_sets"][output_set]["artifacts"][path]

    def actual_path(self, output_set: str, filename: str) -> Path:
        """返回已存在的工件路径，否则报告其准确位置并失败。"""
        path = self.datasets_root / output_set / filename
        self.assertTrue(path.is_file(), f"Missing baseline artifact: {path}")
        return path

    def assert_csv_contract(self, output_set: str, filename: str) -> dict[str, object]:
        """依据清单断言 CSV 的 schema、行数、标签与特征元数据。"""
        expected = self.expected(output_set, filename)
        actual = csv_metadata(self.actual_path(output_set, filename))
        for field in ("column_count", "row_count", "schema_sha256"):
            self.assertEqual(
                actual[field], expected[field], f"{output_set}/{filename}: {field} changed"
            )
        if "label_distribution" in expected:
            self.assertEqual(
                actual.get("label_distribution"),
                expected["label_distribution"],
                f"{output_set}/{filename}: labels changed",
            )
            self.assertEqual(
                actual.get("feature_count"),
                expected["feature_count"],
                f"{output_set}/{filename}: feature count changed",
            )
        return actual

    def test_csv_columns_feature_counts_and_labels_are_preserved(self) -> None:
        """确认 CSV 列、特征、行数和标签保持不变。"""
        for output_set in self.manifest["contract_output_sets"]:
            for filename in CONTRACT_FILES:
                if filename.endswith(".csv"):
                    with self.subTest(output_set=output_set, filename=filename):
                        self.assert_csv_contract(output_set, filename)

    def test_target_feature_schema_equals_training_feature_schema(self) -> None:
        """确认目标转换输出与训练特征 schema 完全一致。"""
        for output_set in self.manifest["contract_output_sets"]:
            with self.subTest(output_set=output_set):
                train = self.assert_csv_contract(output_set, "Xy_train.csv")
                target = self.assert_csv_contract(output_set, "target_features.csv")
                expected_features = [
                    column
                    for column in train["columns"]
                    if column not in {"sample_weight", "label"}
                ]
                self.assertEqual(target["columns"], expected_features)
                self.assertEqual(target["column_count"], train["feature_count"])

    def test_train_test_and_pub_label_contracts_are_preserved(self) -> None:
        """无需重新训练即可测试 PUB/RF 持久化标签分布。"""
        for output_set in self.manifest["contract_output_sets"]:
            with self.subTest(output_set=output_set):
                train = self.assert_csv_contract(output_set, "Xy_train.csv")
                train_new = self.assert_csv_contract(output_set, "Xy_train_new.csv")
                rf_train = self.assert_csv_contract(output_set, "Xy_rf_train.csv")
                rf_test = self.assert_csv_contract(output_set, "Xy_rf_test.csv")
                self.assertEqual(train_new["label_distribution"], train["label_distribution"])
                train_rows = sum(int(value) for value in rf_train["label_distribution"].values())
                test_rows = sum(int(value) for value in rf_test["label_distribution"].values())
                self.assertEqual(train_rows + test_rows, train["row_count"])

    def test_geotiff_output_shapes_are_preserved(self) -> None:
        """确认持久化 GeoTIFF 的网格形状与数值存储保持稳定。"""
        for output_set in self.manifest["contract_output_sets"]:
            with self.subTest(output_set=output_set):
                expected = self.expected(output_set, "probability_map.tif")
                actual = tiff_metadata(self.actual_path(output_set, "probability_map.tif"))
                for field in ("shape", "band_count", "bits_per_sample", "sample_format", "dtype"):
                    self.assertEqual(
                        actual[field], expected[field], f"{output_set}/probability_map.tif: {field} changed"
                    )

    def test_outputs_test_inconsistency_is_visible_not_contractual(self) -> None:
        """保留已知缓存不一致的可见性，但不将其纳入一致性门禁。"""
        output_set = "Outputs_test"
        target_data = self.assert_csv_contract(output_set, "target_data.csv")
        target_lines = self.assert_csv_contract(output_set, "target_lines.csv")
        target_grids = self.assert_csv_contract(output_set, "target_grids.csv")
        target_features = self.assert_csv_contract(output_set, "target_features.csv")
        self.assertNotEqual(target_data["row_count"], target_features["row_count"])
        self.assertNotEqual(target_lines["row_count"], target_features["row_count"])
        self.assertEqual(target_grids["row_count"], target_features["row_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
