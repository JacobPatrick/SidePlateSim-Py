import numpy as np
from interface.types import FilmParam
from meshpy.triangle import MeshInfo


class ContactSolver:
    """
    侧板与齿轮端面接触力求解器，使用线性弹簧-阻尼模型，复用油膜网格
    """

    def __init__(self, mesh: MeshInfo, k: float, c: float):
        """
        Args:
            mesh: 齿轮端面网格，复用自油膜网格
            k: 接触刚度，单位 N/m^3
            c: 接触阻尼，单位 N·s/m^3
        """
        self.mesh = mesh
        self.k = k
        self.c = c

    def solve(self, film_param: FilmParam):
        """
        求解接触力与接触力等效作用点
        """
        h_cells = film_param.h_cells
        ht_cells = film_param.ht_cells
        contact_cells = np.where(h_cells <= 0)[0]

        points = np.array(self.mesh.points)
        elements = np.array(self.mesh.elements)
        n_cells = len(elements)
        centroids = np.mean(points[elements], axis=1)
        areas = np.zeros(len(elements))
        for i, e in enumerate(elements):
            p0, p1, p2 = points[e]
            areas[i] = 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))

        p = np.zeros(n_cells)
        for cell_idx in contact_cells:
            h = h_cells[cell_idx]
            ht = ht_cells[cell_idx]
            force_magnitude = -self.k * h - self.c * ht
            p[cell_idx] = force_magnitude

        F = np.sum(p * areas)

        contact_area = np.sum(areas[contact_cells])
        print(
            f"接触面积: {contact_area * 1e6:.3g}mm^2, 最大穿透深度: {-np.min(h_cells[contact_cells]) * 1e6:.3g}mu m, 接触力: {F:.3g}N"
        )

        # 若阻尼力大于弹力，截断，避免出现负接触力
        if F <= 1e-8:
            return 0.0, (0.0, 0.0)

        i = np.sum(p * centroids[:, 0] * areas) / F
        j = np.sum(p * centroids[:, 1] * areas) / F
        center = np.array([i, j])

        return F, center
