import numpy as np
from interface.types import (
    GearProfilePath,
    SidePlateState,
)
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
        gear_profile_path: GearProfilePath,
        omega: float,
        gear_type: str = "drive",
    ):
        self.gear_profile_path = gear_profile_path
        self.omega = omega
        assert gear_type in [
            "drive",
            "slave",
        ], "警告: 齿轮类型必须是 'drive' 或 'slave'"
        self.gear_type = gear_type

    def solve(self, t, p_lst, state: SidePlateState):
        # 1. 导入齿轮轮廓
        gear_poly = load_gear_profile_from_dxf(self.gear_profile_path.gear_poly_path)

        # 油膜区域随齿轮旋转而变化
        deg = (t * self.omega * 180 / np.pi) % 30  # 12 齿齿轮
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
                rotate_param=(np.radians(deg - 4), SLAVE_GEAR_CENTER),
            )
        # TODO: 暂时不考虑油槽区域
        # relief_poly = load_geometry_from_dxf(
        #     self.gear_profile_path.relief_poly_path
        # )
        # film_poly = boolean_operation(
        #     rotated, relief_poly, operation="difference"
        # )
        film_poly = rotated
        # 2. 划分网格
        mesh = shapely_to_meshpy(film_poly, max_area=1e-7, markers=p_lst)

        return mesh
