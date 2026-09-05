"""找矿潜力评价（MPM）专用的排序与面积捕获指标。

这些指标与一般分类指标不同：它们衡量"按得分从高到低圈定靶区时，
用多少面积能捕获多少已知矿床"。需要每个预测单元的面积 ``unit_area``。

参考文献：
- 空间留一法 / prediction-rate curve 与 MPM 评价（springer 10.1007/s10618-018-00607-x）
- 面积需使用投影坐标下的真实面积，经纬度不得直接按角度计算。

当前为纯函数实现，尚未接入 ``evaluate_classifier`` 的指标注册表，
因为后者签名 ``(labels, predictions, probabilities, sample_weight)``
不含 ``unit_area``。接入时需扩展评估上下文。
"""

from __future__ import annotations

import numpy as np


# evaluate_mpm_metrics 默认产出的指标键。注册到 METRIC_REGISTRY 后可在
# validation.metrics 中配置；这些指标需要 unit_area，缺失时不可计算。
MPM_METRIC_NAMES = frozenset(
    {
        "capture_rate_at_area_0_01",
        "capture_rate_at_area_0_05",
        "capture_rate_at_area_0_10",
        "area_fraction_at_capture_0_50",
        "area_fraction_at_capture_0_80",
        "area_fraction_at_capture_0_90",
        "prediction_rate_auc",
    }
)


def _validate(scores, labels, unit_area) -> None:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    unit_area = np.asarray(unit_area, dtype=float)
    if not (len(scores) == len(labels) == len(unit_area)):
        raise ValueError("scores, labels, unit_area must have equal length")
    if len(scores) == 0:
        raise ValueError("empty inputs")
    if np.any(~np.isfinite(unit_area)) or np.any(unit_area < 0) or unit_area.sum() <= 0:
        raise ValueError("unit_area must be finite, non-negative and have positive sum")


def prediction_rate_curve(
    scores, labels, unit_area, n_points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """返回 prediction-rate curve 的 (area_fraction, capture_rate)。

    按得分降序排序预测单元，横轴为累计面积占比，纵轴为累计已捕获
    正类占比。落在 held-out / out-of-fold 预测上即为 prediction-rate
    curve（若落在训练拟合上，只能称 success-rate curve）。
    """
    _validate(scores, labels, unit_area)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    unit_area = np.asarray(unit_area, dtype=float)

    order = np.argsort(-scores, kind="mergesort")
    cum_area = np.cumsum(unit_area[order]) / unit_area.sum()
    cum_pos = np.cumsum(labels[order]) / max(float(labels.sum()), 1e-12)

    # 曲线起点为原点 (0 面积, 0 捕获)，终点为 (1, 1)
    cum_area = np.concatenate(([0.0], cum_area))
    cum_pos = np.concatenate(([0.0], cum_pos))

    area_grid = np.linspace(0.0, 1.0, n_points)
    capture_grid = np.interp(area_grid, cum_area, cum_pos)
    return area_grid, capture_grid


def capture_rate_at_area_fraction(scores, labels, unit_area, area_fraction: float) -> float:
    """在给定的面积占比下，能捕获的正类占比。"""
    if not 0.0 <= area_fraction <= 1.0:
        raise ValueError("area_fraction must be in [0, 1]")
    area_grid, capture_grid = prediction_rate_curve(scores, labels, unit_area)
    return float(np.interp(area_fraction, area_grid, capture_grid))


def area_fraction_at_capture_rate(scores, labels, unit_area, capture_rate: float) -> float:
    """捕获指定比例正类所需的最小面积占比。"""
    if not 0.0 <= capture_rate <= 1.0:
        raise ValueError("capture_rate must be in [0, 1]")
    area_grid, capture_grid = prediction_rate_curve(scores, labels, unit_area)
    index = int(np.searchsorted(capture_grid, capture_rate))
    if index >= len(area_grid):
        return 1.0
    return float(area_grid[index])


def prediction_rate_auc(scores, labels, unit_area) -> float:
    """prediction-rate curve 的 AUC。

    0.5 对应随机排序；排序越优 AUC 越高。上界取决于正类面积占比
    （正类越稀疏，理想 AUC 越接近 1）。
    """
    area_grid, capture_grid = prediction_rate_curve(scores, labels, unit_area)
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapezoid(capture_grid, area_grid))


def _metric_key(prefix: str, value: float) -> str:
    """把小数占比转为稳定的指标键名，如 0.05 -> 0_05、0.10 -> 0_10。"""
    return f"{prefix}_{value:.2f}".replace(".", "_")


def evaluate_mpm_metrics(
    scores,
    labels,
    unit_area,
    *,
    area_fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
    capture_rates: tuple[float, ...] = (0.50, 0.80, 0.90),
) -> dict[str, float]:
    """计算一组标准 MPM 面积捕获指标，返回可直接写入 metrics 的 dict。

    包含：
    - ``capture_rate_at_area_<frac>``：给定面积占比下的正类捕获率；
    - ``area_fraction_at_capture_<rate>``：捕获给定比例正类所需面积占比；
    - ``prediction_rate_auc``：prediction-rate curve 的 AUC。
    """
    result: dict[str, float] = {}
    for frac in area_fractions:
        result[_metric_key("capture_rate_at_area", frac)] = capture_rate_at_area_fraction(
            scores, labels, unit_area, frac
        )
    for rate in capture_rates:
        result[_metric_key("area_fraction_at_capture", rate)] = area_fraction_at_capture_rate(
            scores, labels, unit_area, rate
        )
    result["prediction_rate_auc"] = prediction_rate_auc(scores, labels, unit_area)
    return result
