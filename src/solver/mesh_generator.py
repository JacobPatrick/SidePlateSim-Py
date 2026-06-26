import numpy as np
from interface.type import (
    GearProfileDir,
    SidePlateState,
    FilmParam,
)
from utils.math_tools import quaternion_to_euler
from utils.load_geometry import (
    load_geometry_from_dxf,
    load_gear_profile_from_dxf,
)
from utils.geo_trans import (
    boolean_operation,
    transform_operation,
)
from src.geometry.mesher import shapely_to_meshpy


class MeshGenerator:
    def __init__(
        self,
        gear_profile_dir: GearProfileDir,
        omega: float,
    ):
        self.gear_profile_dir = gear_profile_dir
        self.omega = omega

    def solve(self, t, p_lst, status: SidePlateState):
        # 1. 导入齿轮轮廓
        gear_poly = load_gear_profile_from_dxf(
            self.gear_profile_dir.gear_poly_dir
        )

        # 油膜区域随齿轮旋转而变化
        deg = (t * self.omega * 180 / np.pi) % 30  # 12 齿齿轮
        roll, pitch, _ = quaternion_to_euler(*status.q)
        rotated = transform_operation(
            gear_poly,
            transform="rotate",
            rotate_param=(np.radians(deg), (0, 0)),
        )
        relief_poly = load_geometry_from_dxf(
            self.gear_profile_dir.relief_poly_dir
        )
        film_poly = boolean_operation(
            rotated, relief_poly, operation="difference"
        )
        # 2. 划分网格
        mesh = shapely_to_meshpy(film_poly, max_area=1e-7, markers=p_lst)

        # 3. 求仿真油膜参数表
        points = np.array(mesh.points)
        elements = np.array(mesh.elements)
        centroids = np.mean(points[elements], axis=1)

        # 3.1 计算节点处的油膜厚度
        h_cells = (
            -np.sin(pitch) * np.array([p[0] for p in centroids])
            + np.sin(roll) * np.array([p[1] for p in centroids])
            + status.p[2] * np.ones(len(centroids))
        )
        # 非负检查
        assert np.any(
            h_cells > 0
        ), "警告: 油膜厚度存在非正值，请检查齿轮位姿参数设置！"
        # 油膜厚度梯度 (∂h/∂x, ∂h/∂y)
        h_grad = (-np.sin(pitch), np.sin(roll))

        # 3.2 确定边界条件
        bc_lst = [p_lst[0]]
        for _, p_val in p_lst[1:]:
            bc_lst.append(p_val)

        # 3.3 计算三角网格中心处的相对运动速度
        U_cells = np.array(
            [[-self.omega * p[1], self.omega * p[0]] for p in centroids]
        )

        # 3.4 计算三角网格中心处的挤压速度（两表面相互远离为正）
        ht_cells = (
            status.v[2] * np.ones(len(centroids))
            + np.cos(roll) * status.w[0] * np.array([p[1] for p in centroids])
            - np.cos(pitch) * status.w[1] * np.array([p[0] for p in centroids])
        )

        return mesh, FilmParam(
            h_cells=h_cells,
            h_grad=h_grad,
            U_cells=U_cells,
            ht_cells=ht_cells,
            bc_lst=bc_lst,
        )
