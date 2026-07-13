import numpy as np
from interface.type import FilmParam, FluidProp, Pressure
from meshpy.triangle import MeshInfo
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class ReynoldsSolver:
    """
    Reynolds 方程求解器，基于单元中心有限体积法 (FVM)
    默认求解油膜压力分布与总压力，可选求解流速场与泄漏流量
    """

    def __init__(
        self,
        mesh: MeshInfo,
        fluid_prop: FluidProp,
        film_param: FilmParam,
    ):
        """
        Args:
            mesh: meshpy 生成的网格对象 (mesh.points, mesh.elements, mesh.facets, mesh.facet_markers)
            h_cells: 网格单元处的油膜厚度 (N,) [m]
            mu: 动力粘度 [Pa·s]
            U_cells: 壁面相对速度向量场 (N, 2) [m/s]
            ht_cells: 挤压速度场 (N,) [m/s]
            bc_lst: 边界条件列表 [(facet_marker, pressure_value [Pa]), ...]，默认空列表表示无 Dirichlet 边界
        """
        self.mesh = mesh
        self.mu = fluid_prop.mu

        self.equ = ()

    def _assemble_reynolds_fvm(self, film_param: FilmParam):
        """
        组装 2D Reynolds 方程的稀疏矩阵与右端项
        """
        h_cells = film_param.h_cells
        U_cells = film_param.U_cells
        ht_cells = film_param.ht_cells
        h_grad = film_param.h_grad
        bc_lst = film_param.bc_lst
        points = np.array(self.mesh.points)
        elements = np.array(self.mesh.elements)
        facets = np.array(self.mesh.facets)
        facet_markers = np.array(self.mesh.facet_markers)

        n_cells = len(elements)

        # 1. 单元几何属性计算
        centroids = np.mean(points[elements], axis=1)  # (nC, 2)
        areas = np.zeros(n_cells)
        for i, e in enumerate(elements):
            p0, p1, p2 = points[e]
            areas[i] = 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))

        # 扩散系数 D = h^3 / (12μ)
        D_cells = h_cells**3 / (12.0 * self.mu)

        # 2. 构建面列表 (内部面 + 边界面)
        edge_to_cell = {}
        faces = []
        # face 结构: {'cells': (owner, neighbor_or_None), 'nodes': (n1, n2), 'marker': int}

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

        facets = np.array(self.mesh.facets, dtype=int)
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

        # 安全断言：所有边界边应已被 facets 匹配
        assert (
            len(edge_to_cell) == 0
        ), f"网格未闭合或 facets 不匹配，剩余 {len(edge_to_cell)} 条边"

        # 3. 稀疏矩阵组装
        if not bc_lst:
            raise ValueError("Dirichlet 边界必需，但 bc_lst 为空")
        default_p = bc_lst[0]
        bc_map = {idx: p_val for idx, p_val in enumerate(bc_lst)}

        A = lil_matrix((n_cells, n_cells))
        b = np.zeros(n_cells)

        for face in faces:
            n1, n2 = face["nodes"]
            p1, p2 = points[n1], points[n2]
            edge_vec = p2 - p1
            length = np.linalg.norm(edge_vec)
            # 初始局部法向 (基于边向量逆时针旋转90°)
            normal = np.array([edge_vec[1], -edge_vec[0]]) / length

            i = face["cells"][0]

            if face["cells"][1] is not None:  # 内部面
                j = face["cells"][1]
                # 校准法向：确保从 owner(i) 指向 neighbor(j)
                vec_ij = centroids[j] - centroids[i]
                if np.dot(normal, vec_ij) < 0:
                    normal = -normal

                # 扩散项通量系数
                dist = np.linalg.norm(vec_ij)
                avg_D = 0.5 * (D_cells[i] + D_cells[j])
                T = avg_D * length / dist
                A[i, i] -= T
                A[i, j] += T
                A[j, i] += T
                A[j, j] -= T

            else:  # 边界面
                # 校准法向：确保指向单元外部
                mid_pt = (p1 + p2) / 2.0
                if np.dot(normal, mid_pt - centroids[i]) < 0:
                    normal = -normal

                marker = face["marker"]
                p_bc = bc_map.get(marker, default_p)
                dist = np.linalg.norm(centroids[i] - mid_pt)
                T = D_cells[i] * length / dist
                A[i, i] -= T
                b[i] -= T * p_bc

        # RHS
        for i in range(n_cells):
            conv = 0.5 * (U_cells[i, 0] * h_grad[0] + U_cells[i, 1] * h_grad[1])
            ht = ht_cells[i]
            b[i] += (conv + ht) * areas[i]

        self.equ = (A.tocsr(), b)

    def solve(self, film_param: FilmParam):
        """
        1. 求解线性系统 A * x = b，返回压力分布 p
        2. 计算油膜压力 F 和作用点坐标 (i, j)
        """
        self._assemble_reynolds_fvm(film_param)
        p = spsolve(self.equ[0], self.equ[1])
        F, center = self._calc_force(p)

        return Pressure(p=p, F=F, center=center)

    def _calc_force(self, p):
        """
        根据压力分布求油膜压力
        """
        points = np.array(self.mesh.points)
        elements = np.array(self.mesh.elements)

        centroids = np.mean(points[elements], axis=1)
        areas = np.zeros(len(elements))
        for i, e in enumerate(elements):
            p0, p1, p2 = points[e]
            areas[i] = 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))

        F = np.sum(p * areas)
        i = np.sum(p * centroids[:, 0] * areas) / F
        j = np.sum(p * centroids[:, 1] * areas) / F

        return F, (i, j)

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
