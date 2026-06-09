import math

import ezdxf
import numpy as np
from ezdxf.path import make_path
from shapely.geometry import (
    LineString,
    MultiLineString,
    Point,
    MultiPolygon,
    Polygon,
)
from shapely.ops import polygonize, snap, unary_union
from shapely.affinity import scale as shapely_scale


def _sample_arc(center, radius, start_angle, end_angle, num_samples):
    angles = np.linspace(start_angle, end_angle, num_samples, endpoint=True)
    return [
        (
            center[0] + radius * math.cos(a),
            center[1] + radius * math.sin(a),
        )
        for a in angles
    ]


def _path_to_lines(path, distance, min_segments):
    lines = []
    for sub_path in path.sub_paths():
        pts = [(v.x, v.y) for v in sub_path.flattening(distance, min_segments)]
        if len(pts) >= 2:
            lines.append(LineString(pts))
    return lines


def load_geometry_from_dxf(
    file_path,
    arc_samples=4,
    angular_step_deg=3.0,
    simplify_tolerance=0.1,
    snap_tolerance=0.0,
    max_flatten_distance=0.2,
):
    """从DXF文件加载几何轮廓，返回 Polygon/MultiPolygon（不区分内外轮廓）"""
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    lines = []
    for entity in msp:
        etype = entity.dxftype()
        try:
            segments = max(1, int(math.ceil(90.0 / angular_step_deg)))
            path = make_path(entity, segments=segments)
        except TypeError:
            path = None

        if path is not None:
            if max_flatten_distance is None:
                bbox = path.bbox()
                size = bbox.size
                scale = max(size.x, size.y, 1.0)
                distance = max(scale * 1e-4, 1e-6)
            else:
                distance = max_flatten_distance
            lines.extend(_path_to_lines(path, distance, segments))
            continue

        if etype == "LINE":
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            lines.append(LineString([start, end]))
        elif etype == "CIRCLE":
            center = (entity.dxf.center.x, entity.dxf.center.y)
            pts = _sample_arc(
                center, entity.dxf.radius, 0.0, 2 * math.pi, arc_samples
            )
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            lines.append(LineString(pts))
        elif etype == "ARC":
            center = (entity.dxf.center.x, entity.dxf.center.y)
            start = math.radians(entity.dxf.start_angle)
            end = math.radians(entity.dxf.end_angle)
            if end < start:
                end += 2 * math.pi
            pts = _sample_arc(
                center, entity.dxf.radius, start, end, arc_samples
            )
            if len(pts) >= 2:
                lines.append(LineString(pts))

    if not lines:
        return Polygon()

    if simplify_tolerance > 0.0:
        lines = [
            line.simplify(simplify_tolerance, preserve_topology=True)
            for line in lines
        ]
    if snap_tolerance > 0.0:
        reference = MultiLineString(lines)
        lines = [snap(line, reference, snap_tolerance) for line in lines]

    merged = unary_union(lines)
    polys = list(polygonize(merged))
    if not polys:
        return Polygon()
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def load_gear_profile_from_dxf(
    file_path,
    arc_samples=16,
    angular_step_deg=2.0,
    simplify_tolerance=0.01,
    snap_tolerance=0.02,
    max_flatten_distance=0.02,
    return_inner_circle=False,
):
    """从DXF文件加载几何轮廓，返回 Polygon/MultiPolygon 或标注后的轮廓"""
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    lines = []
    circle_polys = []
    for entity in msp:
        etype = entity.dxftype()
        if etype == "CIRCLE":
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = float(entity.dxf.radius)
            circle_polys.append(
                Point(center[0], center[1]).buffer(
                    radius, resolution=arc_samples
                )
            )
            continue
        try:
            segments = max(1, int(math.ceil(90.0 / angular_step_deg)))
            path = make_path(entity, segments=segments)
        except TypeError:
            path = None

        if path is not None:
            if max_flatten_distance is None:
                bbox = path.bbox()
                size = bbox.size
                scale = max(size.x, size.y, 1.0)
                distance = max(scale * 1e-4, 1e-6)
            else:
                distance = max_flatten_distance
            lines.extend(_path_to_lines(path, distance, segments))
            continue

        if etype == "LINE":
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            lines.append(LineString([start, end]))
        elif etype == "ARC":
            center = (entity.dxf.center.x, entity.dxf.center.y)
            start = math.radians(entity.dxf.start_angle)
            end = math.radians(entity.dxf.end_angle)
            if end < start:
                end += 2 * math.pi
            pts = _sample_arc(
                center, entity.dxf.radius, start, end, arc_samples
            )
            if len(pts) >= 2:
                lines.append(LineString(pts))

    if not lines:
        outer_poly = Polygon()
    else:
        if simplify_tolerance > 0.0:
            lines = [
                line.simplify(simplify_tolerance, preserve_topology=True)
                for line in lines
            ]
        if snap_tolerance > 0.0:
            reference = MultiLineString(lines)
            lines = [snap(line, reference, snap_tolerance) for line in lines]

        merged = unary_union(lines)
        polys = list(polygonize(merged))
        if not polys:
            outer_poly = Polygon()
        elif len(polys) == 1:
            outer_poly = polys[0]
        else:
            outer_poly = max(polys, key=lambda p: p.area)

    if circle_polys:
        inner_circle = min(circle_polys, key=lambda p: p.area)
    else:
        inner_circle = None

    if return_inner_circle:
        outer_poly = shapely_scale(
            outer_poly, xfact=0.001, yfact=0.001, origin=(0, 0)
        )
        inner_circle = shapely_scale(
            inner_circle, xfact=0.001, yfact=0.001, origin=(0, 0)
        )
        return {"outer": outer_poly, "inner_circle": inner_circle}

    if inner_circle is not None and isinstance(outer_poly, Polygon):
        outer_poly = outer_poly.difference(inner_circle)
        outer_poly = shapely_scale(
            outer_poly, xfact=0.001, yfact=0.001, origin=(0, 0)
        )
    return outer_poly


def load_profile_from_dxf(file_path):
    """从DXF文件加载几何轮廓，返回 Polygon/MultiPolygon（不区分内外轮廓）"""
    # TODO: 支持更多的轮廓实体类型（如 ARC, CIRCLE 等）
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    lines = []
    for entity in msp:
        etype = entity.dxftype()
        if etype == "LINE":
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            lines.append(LineString([start, end]))

    if not lines:
        return Polygon()

    merged = unary_union(lines)
    polys = list(polygonize(merged))
    polys = [
        shapely_scale(poly, xfact=0.001, yfact=0.001, origin=(0, 0))
        for poly in polys
    ]
    if not polys:
        return Polygon()
    if len(polys) == 1:
        return polys[0]
    else:
        return MultiPolygon(polys)


def boolean_operation(profile1: Polygon, profile2: Polygon, operation="union"):
    """几何轮廓布尔运算"""
    if not (
        isinstance(profile1, (Polygon, MultiPolygon))
        and isinstance(profile2, (Polygon, MultiPolygon))
    ):
        raise TypeError("输入轮廓类型必须是 Polygon 或 MultiPolygon")
    if operation == "union":
        return profile1.union(profile2)
    elif operation == "difference":
        return profile1.difference(profile2)
    elif operation == "intersection":
        return profile1.intersection(profile2)
    else:
        raise ValueError("不支持的布尔运算类型")
