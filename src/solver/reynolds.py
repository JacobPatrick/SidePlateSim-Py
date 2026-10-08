import numpy as np
from interface.types import FilmParam, FluidProp, Pressure
from meshpy.triangle import MeshInfo
from scipy.sparse import coo_matrix
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
        self._gradient_weights = None
        self._prepare_mesh_geometry()

    def _prepare_mesh_geometry(self):
        """预计算在同一网格上的所有 Reynolds 求解共享的几何量。"""
        self.points = np.asarray(self.mesh.points)
        self.elements = np.asarray(self.mesh.elements, dtype=int)
        self.facets = np.asarray(self.mesh.facets, dtype=int)
        self.facet_markers = np.asarray(self.mesh.facet_markers, dtype=int)
        self.n_cells = len(self.elements)

        triangle_points = self.points[self.elements]
        self.centroids = np.mean(triangle_points, axis=1)
        edge_1 = triangle_points[:, 1] - triangle_points[:, 0]
        edge_2 = triangle_points[:, 2] - triangle_points[:, 0]
        self.areas = 0.5 * np.abs(
            edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
        )

        edge_to_cell = {}
        owners = []
        neighbors = []
        face_nodes = []
        markers = []
        boundary_facets = []
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

        for facet_idx, (facet, marker) in enumerate(
            zip(self.facets, self.facet_markers)
        ):
            node_1, node_2 = int(facet[0]), int(facet[1])
            key = (min(node_1, node_2), max(node_1, node_2))
            if key in edge_to_cell:
                owners.append(edge_to_cell.pop(key))
                neighbors.append(-1)
                face_nodes.append((node_1, node_2))
                markers.append(int(marker))
                boundary_facets.append(facet_idx)

        if edge_to_cell:
            raise ValueError(
                "网格未闭合或 facets 不匹配，" f"剩余 {len(edge_to_cell)} 条边"
            )

        self.face_owners = np.asarray(owners, dtype=int)
        self.face_neighbors = np.asarray(neighbors, dtype=int)
        self.face_nodes = np.asarray(face_nodes, dtype=int)
        self.face_markers = np.asarray(markers, dtype=int)
        self.internal_face_indices = np.flatnonzero(self.face_neighbors >= 0)
        self.boundary_face_indices = np.flatnonzero(self.face_neighbors < 0)
        self.boundary_facet_indices = np.asarray(boundary_facets, dtype=int)

        face_vectors = (
            self.points[self.face_nodes[:, 1]]
            - self.points[self.face_nodes[:, 0]]
        )
        face_lengths = np.linalg.norm(face_vectors, axis=1)
        self.face_lengths = face_lengths
        self.face_midpoints = 0.5 * (
            self.points[self.face_nodes[:, 0]]
            + self.points[self.face_nodes[:, 1]]
        )
        self.face_geometry = np.empty(len(self.face_owners))

        internal = self.internal_face_indices
        owner_internal = self.face_owners[internal]
        neighbor_internal = self.face_neighbors[internal]
        center_distance = np.linalg.norm(
            self.centroids[neighbor_internal] - self.centroids[owner_internal],
            axis=1,
        )
        self.face_geometry[internal] = face_lengths[internal] / center_distance

        boundary = self.boundary_face_indices
        owner_boundary = self.face_owners[boundary]
        boundary_midpoints = self.face_midpoints[boundary]
        self.boundary_offsets = (
            boundary_midpoints - self.centroids[owner_boundary]
        )
        boundary_vectors = face_vectors[boundary]
        normals = (
            np.column_stack((boundary_vectors[:, 1], -boundary_vectors[:, 0]))
            / face_lengths[boundary, None]
        )
        inward = np.einsum("ij,ij->i", normals, self.boundary_offsets) < 0
        normals[inward] *= -1.0
        self.boundary_normals = normals
        boundary_distance = np.linalg.norm(
            self.boundary_offsets,
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

    def _boundary_pressures(self, film_param: FilmParam):
        """与压力组装共享 marker 到 Dirichlet 压力的映射。"""
        values = np.asarray(film_param.bc_lst, dtype=float)
        if values.size == 0:
            raise ValueError("Dirichlet 边界必需，但 bc_lst 为空")
        markers = self.face_markers[self.boundary_face_indices]
        indices = np.where(
            (markers > 0) & (markers <= values.size), markers - 1, 0
        )
        return values[indices]

    def _boundary_transmissibility(self, h_cells):
        # 保持与现有 Reynolds 压力离散完全一致。
        return (
            h_cells[self.boundary_owners] ** 3
            / (12.0 * self.mu)
            * self.face_geometry[self.boundary_face_indices]
        )

    def _assemble_reynolds_fvm(self, film_param: FilmParam):
        """
        组装 2D Reynolds 方程的稀疏矩阵与右端项
        """
        h_cells = film_param.h_cells
        U_cells = film_param.U_cells
        ht_cells = film_param.ht_cells
        h_grad = film_param.h_grad
        # 扩散系数 D = h^3 / (12μ)
        D_cells = h_cells**3 / (12.0 * self.mu)
        boundary_pressures = self._boundary_pressures(film_param)

        internal = self.internal_face_indices
        internal_transmissibility = (
            0.5
            * (D_cells[self.internal_owners] + D_cells[self.internal_neighbors])
            * self.face_geometry[internal]
        )
        boundary_transmissibility = self._boundary_transmissibility(h_cells)
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

        conv = 0.5 * (U_cells[:, 0] * h_grad[0] + U_cells[:, 1] * h_grad[1])
        b = (conv + ht_cells) * self.areas
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
        px, py = self._calc_pressure_gradient(p, film_param)

        # 2. 根据压力梯度求流速
        h_cells = np.maximum(film_param.h_cells, 0.0)
        U_cells = film_param.U_cells
        h2 = h_cells * h_cells
        average_u = -(h2 * px) / (12 * self.mu) + 0.5 * U_cells[:, 0]
        average_v = -(h2 * py) / (12 * self.mu) + 0.5 * U_cells[:, 1]
        closed = film_param.h_cells <= 0.0
        average_u[closed] = 0.0
        average_v[closed] = 0.0

        return average_u, average_v

    def _prepare_gradient_reconstruction(self):
        """缓存每个单元的局部加权最小二乘算子，不创建虚拟单元。"""
        boundary = self.boundary_face_indices
        owners = self.internal_owners
        neighbors = self.internal_neighbors
        cells = np.concatenate((owners, neighbors, self.boundary_owners))
        samples = np.concatenate(
            (neighbors, owners, self.n_cells + np.arange(len(boundary)))
        )
        displacement = self.centroids[neighbors] - self.centroids[owners]
        deltas = np.concatenate(
            (displacement, -displacement, self.boundary_offsets)
        )
        # 每个三角形恰有三个面，按所属单元排列三个采样点。
        order = np.argsort(cells, kind="stable")
        self._gradient_samples = samples[order].reshape(self.n_cells, 3)
        deltas = deltas[order].reshape(self.n_cells, 3, 2)
        weighted = deltas / np.sum(deltas * deltas, axis=2)[:, :, None]
        gram = np.einsum("nki,nkj->nij", weighted, deltas)
        self._gradient_weights = np.linalg.solve(
            gram, weighted.transpose(0, 2, 1)
        )

    def _calc_pressure_gradient(self, p, film_param: FilmParam):
        """用相邻单元及边界面 Dirichlet 压力重构单元梯度。"""
        p = np.asarray(p, dtype=float)
        if self._gradient_weights is None:
            self._prepare_gradient_reconstruction()
        samples = np.concatenate((p, self._boundary_pressures(film_param)))
        differences = samples[self._gradient_samples] - p[:, None]
        gradient = np.einsum("nij,nj->ni", self._gradient_weights, differences)
        return gradient[:, 0], gradient[:, 1]

    def _calc_pressure_grediant(self, p, film_param: FilmParam):
        """保留旧拼写；边界梯度计算现在必须提供油膜边界条件。"""
        return self._calc_pressure_gradient(p, film_param)

    def calc_leakage(
        self,
        p,
        film_param: FilmParam,
        *,
        include_couette=True,
        per_unit_length=False,
    ):
        """返回与 mesh.facets 一一对应的边界流量列表，流入油膜为正。

        默认包含压力驱动与 Couette 拖曳流，单位 m³/s。
        include_couette=False 仅计算泊肃叶压差流；per_unit_length=True
        返回单位边长流量 (m²/s)，用于比较沿边界的流量分布。
        marker=0 的外边界沿用压力组装的默认压力，内部 facet 返回零。

        压差通量直接复用压力方程的传导系数，不重构全场梯度。
        若压力经过负压截断，不能据此保证空化区域的质量守恒。
        """
        p = np.asarray(p, dtype=float)
        h_cells = np.asarray(film_param.h_cells)
        boundary = self.boundary_face_indices
        owners = self.boundary_owners
        lengths = self.face_lengths[boundary]
        flow = self._boundary_transmissibility(h_cells) * (
            self._boundary_pressures(film_param) - p[owners]
        )

        if include_couette:
            # 膜厚为线性场；速度梯度可描述刚体旋转等仿射速度场。
            h_face = h_cells[owners] + self.boundary_offsets @ film_param.h_grad
            velocity = film_param.U_cells[owners]
            edge_vectors = (
                self.points[self.face_nodes[boundary, 1]]
                - self.points[self.face_nodes[boundary, 0]]
            )
            # 开放油膜内，两点 Gauss 精确积分线性膜厚与仿射速度的乘积。
            h_delta = (edge_vectors @ film_param.h_grad) / np.sqrt(12.0)
            velocity_delta = np.zeros(len(boundary))
            if film_param.U_grad is not None:
                U_grad = np.asarray(film_param.U_grad)
                velocity = velocity + self.boundary_offsets @ U_grad.T
                velocity_delta = np.einsum(
                    "ij,ij->i",
                    edge_vectors @ U_grad.T,
                    self.boundary_normals,
                ) / np.sqrt(12.0)
            normal_velocity = np.einsum(
                "ij,ij->i", velocity, self.boundary_normals
            )
            normal_flux = 0.5 * (
                np.maximum(h_face - h_delta, 0.0)
                * (normal_velocity - velocity_delta)
                + np.maximum(h_face + h_delta, 0.0)
                * (normal_velocity + velocity_delta)
            )
            flow -= 0.5 * lengths * normal_flux

        # 接触单元没有开放流道，不允许负膜厚产生反向传导率。
        flow[h_cells[owners] <= 0.0] = 0.0
        if per_unit_length:
            flow /= lengths
        result = np.zeros(len(self.facets))
        result[self.boundary_facet_indices] = flow
        return result.tolist()


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
