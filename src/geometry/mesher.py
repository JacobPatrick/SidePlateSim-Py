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


def _clean_single_loop(coords, min_edge_length=1e-5, resample_factor=1.0):
    """清理单个闭合环（外轮廓或内孔）"""
    coords = np.array(coords[:-1])  # 去除首尾重复点
    if len(coords) < 3:
        return coords

    # 1. 移除重复/近重点
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=min_edge_length * 0.5)
    if pairs:
        mask = np.ones(len(coords), dtype=bool)
        for _, j in pairs:  # 保留较小索引，合并较大索引
            mask[j] = False
        coords = coords[mask]

    # 2. 均匀重采样
    if resample_factor != 1.0 and len(coords) > 2:
        line = LineString(coords)
        if line.length < 1e-12:
            return coords
        n_new = max(
            12,
            int(line.length / (resample_factor * min_edge_length)),
        )
        coords = np.array(
            [
                line.interpolate(i / (n_new - 1), normalized=True).coords[0]
                for i in range(n_new)
            ]
        )

    # 3. 修复尖锐角
    angles = _compute_angles(coords)
    sharp_mask = (angles < 10.0) | (angles > 170.0)
    if np.any(sharp_mask):
        new_coords = []
        n = len(coords)
        for i in range(n):
            if not sharp_mask[i]:
                new_coords.append(coords[i])
            else:
                prev = coords[(i - 1) % n]
                next_ = coords[(i + 1) % n]
                new_coords.append((prev + next_) / 2)
        coords = np.array(new_coords)

    return coords


def shapely_to_meshpy(
    poly: Polygon,
    max_area: float = 0.5,
    min_edge_length: float = 1e-5,
    markers: list = None,
):
    """
    将含内孔的 Shapely 多边形转换为 meshpy 网格
    Args:
        poly: 可能含 interior 的 Shapely Polygon
        max_area: 全局最大单元面积
        min_edge_length: 边界清理阈值
        markers: 边界标记列表 [(marker_coordinate, pressure_value), ...]
    """
    if not poly.is_valid:
        poly = poly.buffer(0)  # 修复自相交等拓扑错误

    all_points = []
    all_facets = []
    facet_markers = []
    holes = []

    # 1. 处理外轮廓
    ext_clean = _clean_single_loop(list(poly.exterior.coords), min_edge_length)
    ext_arr = np.array(ext_clean)
    n_ext = len(ext_clean)

    # 默认外边界标记为 1
    ext_marker_ids = np.ones(n_ext, dtype=int)

    markers = markers[1:]  # 排除内边界
    if markers and n_ext > 0:
        splits = []  # [(vertex_index, marker_id), ...]
        for m_id, (coord, _) in enumerate(markers):
            coord_arr = np.asarray(coord)
            # 向量化计算到所有边界顶点的距离
            dists = np.linalg.norm(ext_arr - coord_arr, axis=1)
            nearest_idx = int(np.argmin(dists))
            splits.append((nearest_idx, int(m_id) + 1))

            if dists[nearest_idx] > min_edge_length * 3:
                print(f"警告: 标记点{m_id}：{coord}距离边界顶点过远")

        # 按边界索引排序分割点，插入标记
        splits.sort(key=lambda x: x[0])

        # 去重： 如果多个标记落在同一边界顶点，保留第一个
        unique_splits = []
        seen_indices = set()
        for idx, m_id in splits:
            if idx not in seen_indices:
                unique_splits.append((idx, m_id))
                seen_indices.add(idx)
        splits = unique_splits

        # 分配区间标记
        num_splits = len(splits)
        for i in range(num_splits):
            s_idx, marker = splits[i]
            e_idx = splits[(i + 1) % num_splits][
                0
            ]  # 总是选择下一个点，实现闭环

            # 仅有一个分段点时，起点终点重合，无需标记区间
            if s_idx == e_idx:
                break

            # 标记边区间 [s_idx, e_idx)
            if s_idx < e_idx:
                ext_marker_ids[s_idx:e_idx] = marker

            # 跨越边界起点/终点 (wrap-around)
            else:
                ext_marker_ids[s_idx:] = marker
                ext_marker_ids[:e_idx] = marker

    # 生成外边界 Facets 及对应标记
    for i in range(n_ext):
        all_facets.append([i, (i + 1) % n_ext])
        facet_markers.append(int(ext_marker_ids[i]))

    all_points.extend(ext_clean)

    # 2. 处理内轮廓（孔）
    for interior in poly.interiors:
        int_clean = _clean_single_loop(list(interior.coords), min_edge_length)
        if len(int_clean) < 3:
            continue  # 跳过退化孔

        start_idx = len(all_points)
        all_points.extend(int_clean)
        n_int = len(int_clean)
        for i in range(n_int):
            all_facets.append([start_idx + i, start_idx + (i + 1) % n_int])
            facet_markers.append(0)  # 内孔边界标记为 0

        # 孔定位点： Triangle 依赖此点识别空洞区域
        holes.append([interior.centroid.x, interior.centroid.y])

    points_arr = np.array(all_points, dtype=float)

    # 3. 配置 MeshInfo
    mesh_info = MeshInfo()
    mesh_info.set_points(points_arr)
    mesh_info.set_facets(all_facets, facet_markers=facet_markers)
    if holes:
        mesh_info.set_holes(holes)

    # 4. 生成网格
    mesh = build(
        mesh_info,
        max_volume=max_area,
        min_angle=20.0,
        quality_meshing=True,
        verbose=False,
    )
    return mesh
