import numpy as np
from interface.types import FilmParam, Pressure
from meshpy.triangle import MeshInfo


class ContactSolver:
    """
    侧板与齿轮端面接触力求解器，使用线性弹簧-阻尼模型，复用油膜网格
    """

    def __init__(
        self,
        mesh: MeshInfo,
        k: float,
        c: float,
        verbose: bool = False,
    ):
        """
        Args:
            mesh: 齿轮端面网格，复用自油膜网格
            k: 接触刚度，单位 N/m^3
            c: 接触阻尼，单位 N·s/m^3
        """
        self.mesh = mesh
        self.k = k
        self.c = c
        self.verbose = verbose

        points = np.asarray(mesh.points)
        self.elements = np.asarray(mesh.elements, dtype=int)
        triangle_points = points[self.elements]
        self.centroids = np.mean(triangle_points, axis=1)
        edge_1 = triangle_points[:, 1] - triangle_points[:, 0]
        edge_2 = triangle_points[:, 2] - triangle_points[:, 0]
        self.areas = 0.5 * np.abs(
            edge_1[:, 0] * edge_2[:, 1]
            - edge_1[:, 1] * edge_2[:, 0]
        )

    def solve(self, film_param: FilmParam) -> Pressure:
        """
        求解接触力与接触力等效作用点
        """
        h_cells = film_param.h_cells
        ht_cells = film_param.ht_cells
        contact_cells = np.where(h_cells <= 0)[0]

        p = np.zeros(len(self.elements))
        contact_pressure = (
            -self.k * h_cells[contact_cells]
            - self.c * ht_cells[contact_cells]
        )
        # 单边接触只能产生压力，不能产生拉力。
        p[contact_cells] = np.maximum(contact_pressure, 0.0)

        weighted_pressure = p * self.areas
        F = np.sum(weighted_pressure)

        if self.verbose:
            contact_area = np.sum(self.areas[contact_cells])
            print(
                f"接触面积: {contact_area * 1e6:.3g}mm^2, "
                f"最大穿透深度: "
                f"{-np.min(h_cells[contact_cells]) * 1e6:.3g}mu m, "
                f"接触力: {F:.3g}N"
            )

        # 即使接触区域正在分离，也保持返回类型稳定。
        if F <= 1e-8:
            return Pressure(
                p=p,
                F=0.0,
                center=np.array([0.0, 0.0]),
            )

        i = np.dot(weighted_pressure, self.centroids[:, 0]) / F
        j = np.dot(weighted_pressure, self.centroids[:, 1]) / F
        center = np.array([i, j])

        return Pressure(p=p, F=F, center=center)
