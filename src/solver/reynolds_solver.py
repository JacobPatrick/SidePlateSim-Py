import numpy as np
from meshpy.triangle import MeshInfo
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class ReynoldsSolver:
    """
    Reynolds 方程求解器，基于单元中心有限体积法 (FVM)
    """

    def __init__(
        self,
        mesh: MeshInfo,
        h_cells: np.ndarray,
        mu: float,
        U_cells: np.ndarray,
        ht_cells: np.ndarray,
        h_grad: tuple,
        bc_lst: list = [],
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
        self.h_cells = h_cells
        self.mu = mu
        self.U_cells = U_cells
        self.ht_cells = ht_cells
        self.h_grad = h_grad
        self.bc_lst = bc_lst

        self.equ = ()
        self.assemble_reynolds_fvm()

    def assemble_reynolds_fvm(self):
        """
        组装 2D Reynolds 方程的稀疏矩阵与右端项
        """
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
        D_cells = self.h_cells**3 / (12.0 * self.mu)

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
                    faces.append({'cells': (j, i), 'nodes': key, 'marker': 0})
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
                        'cells': (i, None),
                        'nodes': (n1, n2),
                        'marker': int(facet_markers[idx]),
                    }
                )

        # 安全断言：所有边界边应已被 facets 匹配
        assert (
            len(edge_to_cell) == 0
        ), f"网格未闭合或 facets 不匹配，剩余 {len(edge_to_cell)} 条边"

        # 3. 稀疏矩阵组装
        if not self.bc_lst:
            raise ValueError("Dirichlet 边界必需，但 bc_lst 为空")
        default_p = self.bc_lst[0]
        bc_map = {idx: p_val for idx, p_val in enumerate(self.bc_lst)}

        A = lil_matrix((n_cells, n_cells))
        b = np.zeros(n_cells)

        for face in faces:
            n1, n2 = face['nodes']
            p1, p2 = points[n1], points[n2]
            edge_vec = p2 - p1
            length = np.linalg.norm(edge_vec)
            # 初始局部法向 (基于边向量逆时针旋转90°)
            normal = np.array([edge_vec[1], -edge_vec[0]]) / length

            i = face['cells'][0]

            if face['cells'][1] is not None:  # 内部面
                j = face['cells'][1]
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

                marker = face['marker']
                p_bc = bc_map.get(marker, default_p)
                dist = np.linalg.norm(centroids[i] - mid_pt)
                T = D_cells[i] * length / dist
                A[i, i] -= T
                b[i] -= T * p_bc

        # RHS
        for i in range(n_cells):
            conv = 0.5 * (
                self.U_cells[i, 0] * self.h_grad[0]
                + self.U_cells[i, 1] * self.h_grad[1]
            )
            ht = self.ht_cells[i]
            b[i] -= (conv + ht) * areas[i]

        self.equ = (A.tocsr(), b)

    def solve(self):
        """
        求解线性系统 Ax=b，返回压力分布 p
        """

        p = spsolve(self.equ[0], self.equ[1])
        return p

    def calc_force(self, p):
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
