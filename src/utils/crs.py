"""CRS（坐标参考系）解析与一致性校验。

研究单元、空间划分与 GeoTIFF 导出共享同一套 CRS 约定：

- 输入矢量（矿点 / 边界 / 训练边界）的 CRS 不一致时严格拒绝，而不是静默混用；
- 导出 GeoTIFF 时校验 ``prediction.target_crs`` 与实际研究单元 CRS 一致；
- 框架不做隐式重投影，避免把 ``target_crs`` 当作自动重投影参数。

这些函数对可选 GIS 依赖保持惰性：仅在调用方传入实际的 geopandas /
pyproj / osgeo CRS 对象时才触及相应属性，缺依赖环境仍可导入本模块。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def crs_identifier(crs: Any) -> str | None:
    """将 CRS 对象归一化为可比较的字符串标识。

    优先返回 ``EPSG:xxxx``；无法提取 EPSG 时返回 WKT；传入 None 返回 None。
    支持 geopandas/pyproj 的 ``pyproj.CRS`` 与 ``osgeo.osr.SpatialReference``。
    """
    if crs is None:
        return None

    # pyproj.CRS（geopandas 默认返回）
    if hasattr(crs, "to_epsg") and hasattr(crs, "to_wkt"):
        try:
            epsg = crs.to_epsg()
            if epsg is not None:
                return f"EPSG:{epsg}"
            wkt = crs.to_wkt()
            if wkt:
                return wkt
        except Exception:  # pragma: no cover - 防御性兜底
            pass

    # osgeo.osr.SpatialReference
    if hasattr(crs, "GetAuthorityCode") and hasattr(crs, "GetAuthorityName"):
        try:
            name = crs.GetAuthorityName(None)
            code = crs.GetAuthorityCode(None)
            if name and code:
                return f"{name}:{code}"
        except Exception:  # pragma: no cover - 防御性兜底
            pass
        try:
            if hasattr(crs, "ExportToWkt"):
                return crs.ExportToWkt()
        except Exception:  # pragma: no cover - 防御性兜底
            pass

    return str(crs)


def resolve_crs_identifier(value: Any) -> str:
    """把配置中的 ``target_crs``（EPSG 整数或 WKT/proj4 字符串）解析为标识。

    使用 GDAL/OSR 的 ``SetFromUserInput`` 做稳健解析，可接受 EPSG 码、
    WKT 与 PROJ 字符串。解析失败时原样返回字符串以便比较，并交由调用方决定。
    """
    if value is None:
        raise ValueError("target_crs must not be None when CRS validation is requested")

    if isinstance(value, bool):  # bool 是 int 子类，先排除
        raise ValueError(f"target_crs must be an EPSG integer or WKT/proj4 string, got {value!r}")

    if isinstance(value, int):
        try:
            from osgeo import osr
        except ImportError:  # pragma: no cover - 依赖 GDAL 绑定
            return f"EPSG:{value}"
        srs = osr.SpatialReference()
        if srs.ImportFromEPSG(value) == 0:
            return f"EPSG:{value}"
        return f"EPSG:{value}"  # 无效码仍返回，交给一致性比较暴露问题

    text = str(value)
    try:
        from osgeo import osr
    except ImportError:  # pragma: no cover - 依赖 GDAL 绑定
        return text
    srs = osr.SpatialReference()
    if srs.SetFromUserInput(text) == 0:
        authority = srs.GetAuthorityName(None)
        code = srs.GetAuthorityCode(None)
        if authority and code:
            return f"{authority}:{code}"
        return srs.ExportToWkt()
    return text


def require_consistent_crs(
    pairs: Iterable[tuple[str, Any]], context: str
) -> str | None:
    """校验多个 ``(标签, crs)`` 对的一致性，返回统一标识（无已知 CRS 时为 None）。

    仅比较已知 CRS（非 None）；一旦发现两个不同的标识即抛出 ValueError。
    """
    identifiers: dict[str, str] = {}
    for label, crs in pairs:
        ident = crs_identifier(crs)
        if ident is not None:
            identifiers[label] = ident

    unique = set(identifiers.values())
    if len(unique) > 1:
        detail = "; ".join(f"{label}={ident}" for label, ident in identifiers.items())
        raise ValueError(
            f"CRS mismatch in {context}: {detail}. "
            "Reproject inputs to a common CRS before running; the framework "
            "does not silently reproject coordinates."
        )
    return next(iter(unique), None) if unique else None
