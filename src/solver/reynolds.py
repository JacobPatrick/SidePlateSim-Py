import numpy as np
from interface.types import FilmParam, FluidProp, Pressure
from meshpy.triangle import MeshInfo
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


class ReynoldsSolver:
    """
    Reynolds 方程求解器，基于单元中心有限体积法 (FVM)
    默认求解油膜压力分布与总压力，可选求解流速场与泄漏流量
    """

    def __init__(self, mesh: MeshInfo, fluid_prop: FluidProp):
        self.mesh = mesh
        self.mu = fluid_prop.mu
        self.equ = ()
        self._prepare_mesh_geometry()

    def _prepare_mesh_geometry(self):
        """预计算在同一网格上的所有 Reynolds 求解共享的几何量。"""
        self.points = np.asarray(self.mesh.points)
        self.elements = np.asarray(self.mesh.elements, dtype=int)
        self.facets = np.asarray(self.mesh.facets, dtype=int)
        self.facet_markers = np.asarray(
            self.mesh.facet_markers, dtype=int
        )
        self.n_cells = len(self.elements)

        triangle_points = self.points[self.elements]
        self.centroids = np.mean(triangle_points, axis=1)
        edge_1 = triangle_points[:, 1] - triangle_points[:, 0]
        edge_2 = triangle_points[:, 2] - triangle_points[:, 0]
        self.areas = 0.5 * np.abs(
            edge_1[:, 0] * edge_2[:, 1]
            - edge_1[:, 1] * edge_2[:, 0]
        )

        edge_to_cell = {}
        owners = []
        neighbors = []
        face_nodes = []
        markers = []
        for cell_idx, element in enumerate(self.elements):
            for local_idx in range(3):
                node_1 = int(element[local_idx])
                node_2 = int(element[(local_idx + 1) % 3])
                key = (min(node_1, node_2), max(node_1, node_2))
                if key in edge_to_cell:
                    owners.append(edge_to_cell.pop(key))
                    neighbors.append(cell_idx)
                    face_nodes.append(key)
                    markers.append(0)
                else:
                    edge_to_cell[key] = cell_idx

        for facet, marker in zip(self.facets, self.facet_markers):
            node_1, node_2 = int(facet[0]), int(facet[1])
            key = (min(node_1, node_2), max(node_1, node_2))
            if key in edge_to_cell:
                owners.append(edge_to_cell.pop(key))
                neighbors.append(-1)
                # 保留边界边原有方向，供泄漏流量计算使用。
                face_nodes.append((node_1, node_2))
                markers.append(int(marker))

        if edge_to_cell:
            raise ValueError(
                "网格未闭合或 facets 不匹配，"
                f"剩余 {len(edge_to_cell)} 条边"
            )

        self.face_owners = np.asarray(owners, dtype=int)
        self.face_neighbors = np.asarray(neighbors, dtype=int)
        self.face_nodes = np.asarray(face_nodes, dtype=int)
        self.face_markers = np.asarray(markers, dtype=int)
        self.internal_face_indices = np.flatnonzero(
            self.face_neighbors >= 0
        )
        self.boundary_face_indices = np.flatnonzero(
            self.face_neighbors < 0
        )

        face_vectors = (
            self.points[self.face_nodes[:, 1]]
            - self.points[self.face_nodes[:, 0]]
        )
        face_lengths = np.linalg.norm(face_vectors, axis=1)
        self.face_geometry = np.empty(len(self.face_owners))

        internal = self.internal_face_indices
        owner_internal = self.face_owners[internal]
        neighbor_internal = self.face_neighbors[internal]
        center_distance = np.linalg.norm(
            self.centroids[neighbor_internal]
            - self.centroids[owner_internal],
            axis=1,
        )
        self.face_geometry[internal] = (
            face_lengths[internal] / center_distance
        )

        boundary = self.boundary_face_indices
        owner_boundary = self.face_owners[boundary]
        boundary_midpoints = 0.5 * (
            self.points[self.face_nodes[boundary, 0]]
            + self.points[self.face_nodes[boundary, 1]]
        )
        boundary_distance = np.linalg.norm(
            self.centroids[owner_boundary] - boundary_midpoints,
            axis=1,
        )
        self.face_geometry[boundary] = (
            face_lengths[boundary] / boundary_distance
        )

        # 稀疏矩阵非零位置只依赖网格拓扑。
        self.internal_owners = owner_internal
        self.internal_neighbors = neighbor_internal
        self.boundary_owners = owner_boundary
        self.matrix_rows = np.concatenate(
            [
                owner_internal,
                owner_internal,
                neighbor_internal,
                neighbor_internal,
                owner_boundary,
            ]
        )
        self.matrix_cols = np.concatenate(
            [
                owner_internal,
                neighbor_internal,
                owner_internal,
                neighbor_internal,
                owner_boundary,
            ]
        )

    def _assemble_reynolds_fvm(self, film_param: FilmParam):
        """
        组装 2D Reynolds 方程的稀疏矩阵与右端项
        """
        h_cells = film_param.h_cells
        U_cells = film_param.U_cells
        ht_cells = film_param.ht_cells
        h_grad = film_param.h_grad
        bc_lst = film_param.bc_lst
        # 扩散系数 D = h^3 / (12μ)
        D_cells = h_cells**3 / (12.0 * self.mu)
        if not bc_lst:
            raise ValueError("Dirichlet 边界必需，但 bc_lst 为空")
        default_p = bc_lst[0]
        bc_map = {idx + 1: p_val for idx, p_val in enumerate(bc_lst)}

        internal = self.internal_face_indices
        boundary = self.boundary_face_indices
        internal_transmissibility = (
            0.5
            * (
                D_cells[self.internal_owners]
                + D_cells[self.internal_neighbors]
            )
            * self.face_geometry[internal]
        )
        boundary_transmissibility = (
            D_cells[self.boundary_owners]
            * self.face_geometry[boundary]
        )
        matrix_data = np.concatenate(
            [
                -internal_transmissibility,
                internal_transmissibility,
                internal_transmissibility,
                -internal_transmissibility,
                -boundary_transmissibility,
            ]
        )
        A = coo_matrix(
            (matrix_data, (self.matrix_rows, self.matrix_cols)),
            shape=(self.n_cells, self.n_cells),
        ).tocsr()

        conv = 0.5 * (
            U_cells[:, 0] * h_grad[0]
            + U_cells[:, 1] * h_grad[1]
        )
        b = (conv + ht_cells) * self.areas
        boundary_pressures = np.fromiter(
            (
                bc_map.get(marker, default_p)
                for marker in self.face_markers[boundary]
            ),
            dtype=float,
            count=len(boundary),
        )
        np.add.at(
            b,
            self.boundary_owners,
            -boundary_transmissibility * boundary_pressures,
        )

        self.equ = (A, b)

    def solve(self, film_param: FilmParam) -> Pressure:
        """
        1. 求解线性系统 A * x = b，返回压力分布 p（若有碰撞，进行额外处理）
        2. 计算油膜压力 F 和作用点坐标 (i, j)
        """
        self._assemble_reynolds_fvm(film_param)
        A, b = self.equ

        # 检查是否存在碰撞
        h_cells = film_param.h_cells
        if np.any(h_cells <= 0):
            area = np.flatnonzero(h_cells <= 0)
            p_contact = 0  # 碰撞区域压力固定为标准大气压
            A_film, b_film = _process_contact_area(A, b, area, p_contact)

            p_film = spsolve(A_film, b_film)
            film_mask = np.ones(len(h_cells), dtype=bool)
            film_mask[area] = False
            p = np.full(len(h_cells), p_contact, dtype=float)
            p[film_mask] = p_film
            p = np.clip(p, 0.0, None)  # 负压截断
            F, center = self._calc_force(p)
        else:
            p = spsolve(A, b)
            p = np.array(np.clip(p, 0.0, None))  # 负压截断
            F, center = self._calc_force(p)

        return Pressure(p=p, F=F, center=center)

    def _calc_force(self, p):
        """
        根据压力分布求油膜压力
        """
        weighted_pressure = p * self.areas
        F = np.sum(weighted_pressure)
        if abs(F) < 1e-8:
            center = np.array([0.0, 0.0])
            return F, center
        i = np.dot(weighted_pressure, self.centroids[:, 0]) / F
        j = np.dot(weighted_pressure, self.centroids[:, 1]) / F
        center = np.array([i, j])

        return F, center

    def _calc_flow(self, p, film_param: FilmParam):
        """
        根据压力场求流速场
        """
        # 1. 计算单元中心的压力梯度
        px, py = self._calc_pressure_grediant(p)

        # 2. 根据压力梯度求流速
        h_cells = film_param.h_cells
        U_cells = film_param.U_cells
        h2 = h_cells * h_cells
        average_u = -(h2 * px) / (12 * self.mu) + 0.5 * U_cells[:, 0]
        average_v = -(h2 * py) / (12 * self.mu) + 0.5 * U_cells[:, 1]

        return average_u, average_v

    def _calc_pressure_grediant(self, p):
        """
        计算单元中心的压力梯度
        """
        # 1. 对单元 i，找到其相邻单元 j，获得相邻单元的压力 p_j 和单元中心坐标
        points = np.array(self.mesh.points)
        elements = np.array(self.mesh.elements)
        centroids = np.mean(points[elements], axis=1)
        n_cells = len(elements)

        if hasattr(self.mesh, "neighbors"):
            neighbors_raw = np.array(
                self.mesh.neighbors
            )  # shape: (n_cells, 3), -1 表示边界
        else:
            raise AttributeError("meshpy 对象未提供 neighbors 属性")

        # 构建单元邻接列表
        cell_neighbors = [None] * n_cells
        for i in range(n_cells):
            cell_neighbors[i] = neighbors_raw[i].tolist()

        # 筛选边界单元
        internal_mask = np.all(neighbors_raw >= 0, axis=1)
        external_indices = np.where(~internal_mask)[0]

        # 镜像得到虚拟相邻单元
        for i in external_indices:
            for k in range(3):
                if neighbors_raw[i, k] == -1:
                    n1, n2 = int(elements[i, k]), int(elements[i, (k + 1) % 3])
                    p1, p2 = points[n1], points[n2]
                    edge_vec = p2 - p1
                    length = np.linalg.norm(edge_vec)
                    normal = np.array([edge_vec[1], -edge_vec[0]]) / length

                    mid_pt = (p1 + p2) / 2.0
                    if np.dot(normal, mid_pt - centroids[i]) < 0:
                        normal = -normal

                    # 镜像单元中心坐标
                    vec_to_mid = mid_pt - centroids[i]
                    mirrored_centroid = centroids[i] + 2 * vec_to_mid

                    # 添加虚拟单元 j
                    j = len(cell_neighbors)
                    cell_neighbors[i][k] = j
                    centroids = np.vstack([centroids, mirrored_centroid])
                    p = np.append(p, p[i])  # 虚拟单元压力与原单元相同

        # 2. 构建矩阵 A 、向量 b 和权重矩阵 W
        A = lil_matrix((3 * n_cells, 2 * n_cells))
        b = np.zeros(3 * n_cells)
        W = lil_matrix((3 * n_cells, 3 * n_cells))

        for i in range(n_cells):
            count = 0
            for j in cell_neighbors[i]:
                dx = centroids[j, 0] - centroids[i, 0]
                dy = centroids[j, 1] - centroids[i, 1]
                dp = p[j] - p[i]

                A[3 * i + count, 2 * i] = dx
                A[3 * i + count, 2 * i + 1] = dy
                b[3 * i + count] = dp

                W[3 * i + count, 3 * i + count] = 1 / (
                    (centroids[j, 0] - centroids[i, 0]) ** 2
                    + (centroids[j, 1] - centroids[i, 1]) ** 2
                )

                count += 1

        # 3. LS 求解超定方程组 A * (px, py) = b
        p = spsolve(A.T @ W @ A, A.T @ W @ b)
        px = np.array([p[2 * i] for i in range(n_cells)])
        py = np.array([p[2 * i + 1] for i in range(n_cells)])

        return px, py

    def calc_leakage(self, p, film_param: FilmParam):
        """
        求齿槽向端面的泄漏流量
        """
        h_cells = film_param.h_cells

        points = np.array(self.mesh.points)
        elements = np.array(self.mesh.elements)
        facets = np.array(self.mesh.facets)
        facet_markers = np.array(self.mesh.facet_markers)

        edge_to_cell = {}
        faces = []

        for i, e in enumerate(elements):
            for k in range(3):
                n1, n2 = int(e[k]), int(e[(k + 1) % 3])
                key = (min(n1, n2), max(n1, n2))
                if key in edge_to_cell:
                    j = edge_to_cell.pop(key)
                    faces.append(
                        {
                            "cells": (j, i),
                            "nodes": key,
                            "marker": 0,
                        }
                    )
                else:
                    edge_to_cell[key] = i

        facet_markers = np.array(self.mesh.facet_markers, dtype=int)
        for idx in range(len(facets)):
            n1, n2 = int(facets[idx][0]), int(facets[idx][1])
            key = (min(n1, n2), max(n1, n2))
            if key in edge_to_cell:
                i = edge_to_cell.pop(key)
                faces.append(
                    {
                        "cells": (i, None),
                        "nodes": (n1, n2),
                        "marker": int(facet_markers[idx]),
                    }
                )

        local_u, local_v = self._calc_flow(p)

        leak_rate = []

        for face in faces:
            if face["cells"][1] is not None or face["marker"] == 0:
                continue

            i = face["cells"][0]
            n1, n2 = face["nodes"]
            p1, p2 = points[n1], points[n2]
            edge_vec = p2 - p1
            length = np.linalg.norm(edge_vec)
            normal = np.array([edge_vec[1], -edge_vec[0]]) / length

            u_vec = np.array([local_u[i], local_v[i]])
            flow_rate = -np.dot(u_vec, normal) * length * h_cells[i]
            leak_rate.append(flow_rate)

        return leak_rate


def _process_contact_area(A: np.ndarray, b: np.ndarray, S: list, x0: float):
    """
    处理侧板和齿轮端面接触区域的油膜，将压力固定为常数，并作为其余部分的边界条件继续求解
    """
    n = A.shape[0]
    S = np.array(S)

    # 获取补集索引 (S^c)
    S_comp = np.setdiff1d(np.arange(n), S)

    # 删除 S 对应的行，得到欠定方程组 A'x = b'
    A_prime = A[S_comp, :]
    b_prime = b[S_comp]

    # 计算移项后的常数项 b_tilde
    Ap = np.asarray(x0 * np.sum(A_prime[:, S], axis=1))
    b_tilde = b_prime - Ap.squeeze()

    # 提取 A_tilde，即 A' 中删除 S 对应的列
    A_tilde = A_prime[:, S_comp]

    return A_tilde.tocsr(), b_tilde
