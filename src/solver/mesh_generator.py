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

DRIVE_GEAR_CENTER = (0.0305, 0)
SLAVE_GEAR_CENTER = (-0.0305, 0)


class MeshGenerator:
    def __init__(
        self,
        gear_profile_dir: GearProfileDir,
        omega: float,
        gear_type: str = "drive" | "slave",
    ):
        self.gear_profile_dir = gear_profile_dir
        self.omega = omega
        assert gear_type in [
            "drive",
            "slave",
        ], "警告: 齿轮类型必须是 'drive' 或 'slave'"
        self.gear_type = gear_type

    def solve(self, t, p_lst, status: SidePlateState):
        # 1. 导入齿轮轮廓
        gear_poly = load_gear_profile_from_dxf(
            self.gear_profile_dir.gear_poly_dir
        )

        # 油膜区域随齿轮旋转而变化
        deg = (t * self.omega * 180 / np.pi) % 30  # 12 齿齿轮
        roll, pitch, _ = quaternion_to_euler(*status.q)
        if self.gear_type == "drive":
            # 主动轮逆时针旋转，齿轮轴心在原点，偏移到 DRIVE_GEAR_CENTER
            translated = transform_operation(
                gear_poly,
                transform="translate",
                translate_param=(DRIVE_GEAR_CENTER[0], DRIVE_GEAR_CENTER[1]),
            )
            rotated = transform_operation(
                translated,
                transform="rotate",
                rotate_param=(np.radians(-deg), DRIVE_GEAR_CENTER),
            )
        else:
            # 从动轮顺时针旋转，齿轮轴心在 (-0.061, 0)，偏移到 SLAVE_GEAR_CENTER
            translated = transform_operation(
                gear_poly,
                transform="translate",
                translate_param=(
                    SLAVE_GEAR_CENTER[0] + 0.061,
                    SLAVE_GEAR_CENTER[1],
                ),
            )
            rotated = transform_operation(
                translated,
                transform="rotate",
                rotate_param=(np.radians(deg), SLAVE_GEAR_CENTER),
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
            -np.sin(pitch) * np.array([point[0] for point in centroids])
            + np.sin(roll) * np.array([point[1] for point in centroids])
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
        if self.gear_type == "drive":
            U_cells = np.array(
                [
                    [
                        -self.omega * point[1],
                        self.omega * (point[0] - DRIVE_GEAR_CENTER[0]),
                    ]
                    for point in centroids
                ]
            )
        else:
            U_cells = np.array(
                [
                    [
                        self.omega * point[1],
                        -self.omega * (point[0] - SLAVE_GEAR_CENTER[0]),
                    ]
                    for point in centroids
                ]
            )

        # 3.4 计算三角网格中心处的挤压速度（两表面相互远离为正）
        ht_cells = (
            status.v[2] * np.ones(len(centroids))
            + np.cos(roll)
            * status.w[0]
            * np.array([point[1] for point in centroids])
            - np.cos(pitch)
            * status.w[1]
            * np.array([point[0] for point in centroids])
        )

        return mesh, FilmParam(
            h_cells=h_cells,
            h_grad=h_grad,
            U_cells=U_cells,
            ht_cells=ht_cells,
            bc_lst=bc_lst,
        )
