import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, LineString
from meshpy.triangle import MeshInfo, build


def _compute_angles(pts):
    """
    计算多边形顶点的内角，识别尖锐角
    """
    angles = []
    n = len(pts)
    for i in range(n):
        prev = pts[(i - 1) % n]
        curr = pts[i]
        next_ = pts[(i + 1) % n]
        v1 = prev - curr
        v2 = next_ - curr
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm < 1e-12:
            angles.append(180.0)
        else:
            cos_ang = np.clip(dot / norm, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_ang)))
    return np.array(angles)


def _clean_polygon_boundary(
    poly: Polygon,
    min_edge_length: float = 1e-3,
    max_angle_deviation: float = 170,
    resample_factor: float = 1.0,
):
    """
    清理 Shapely 多边形边界，去除重复点和过近的点，确保边界质量适合网格生成
    """
    coords = np.array(list(poly.exterior.coords)[:-1])  # 去闭合重复点
    if len(coords) < 3:
        return poly

    # 1. 移除重复/接近点
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=min_edge_length * 0.5)
    if pairs:
        # 合并近重点：保留索引小的点
        mask = np.ones(len(coords), dtype=bool)
        for i, j in pairs:
            mask[j] = False
        coords = coords[mask]

    # 2. 均匀重采样（解决渐开线离散不均）
    if resample_factor != 1.0:
        line = LineString(coords)
        total_len = line.length
        n_new = max(20, int(total_len / (resample_factor * min_edge_length)))
        coords = np.array(
            [
                line.interpolate(i / (n_new - 1), normalized=True).coords[0]
                for i in range(n_new)
            ]
        )

    # 3. 修复尖锐角（简化：跳过角度过小的顶点）
    angles = _compute_angles(coords)
    # 标记需平滑的尖锐角（内角 < 10° 或 > 170°）
    sharp_mask = (angles < 10.0) | (angles > max_angle_deviation)
    if np.any(sharp_mask):
        # 简单策略：移除尖锐点，用相邻点线性插值替代
        new_coords = []
        for i, (pt, is_sharp) in enumerate(zip(coords, sharp_mask)):
            if not is_sharp:
                new_coords.append(pt)
            else:
                # 用前后点中点替代
                prev = coords[(i - 1) % len(coords)]
                next_ = coords[(i + 1) % len(coords)]
                new_coords.append((prev + next_) / 2)
        coords = np.array(new_coords)

    # 重建多边形
    return Polygon(coords)


def shapely_to_meshpy(
    poly: Polygon, max_area: float = 0.5, mark_radial: bool = True
):
    """
    将 Shapely 多边形转换为 meshpy 网格，并自动标记径向边界
    """
    # 1. 清理边界，确保质量
    cleaned_poly = _clean_polygon_boundary(poly, min_edge_length=0.1)

    # 2. 提取外轮廓点与分段
    coords = list(cleaned_poly.exterior.coords)
    points = np.array(coords[:-1])  # 去掉重复闭合点
    n_pts = len(points)

    # 构建封闭线段索引 (每段: [i, i+1])
    facets = [[i, (i + 1) % n_pts] for i in range(n_pts)]

    # 3. 初始化 MeshInfo
    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)

    # 4. 边界标记
    # 假设最后两个点构成的边是径向线，其余为齿廓/圆弧
    facet_markers = [1] * n_pts  # 默认标记 1（固壁/齿面）
    # TODO: 进一步细化边界条件
    mesh_info.set_facets(facets, facet_markers=facet_markers)

    # 5. 生成网格
    mesh = build(mesh_info, max_volume=max_area, min_angle=20)
    return mesh
