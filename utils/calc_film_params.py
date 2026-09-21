import numpy as np
from interface.types import SidePlateState, FilmParam
from utils.math_tools import quaternion_to_euler

DRIVE_GEAR_CENTER = (0.0305, 0)
SLAVE_GEAR_CENTER = (-0.0305, 0)


class FilmParameterCalculator:
    """缓存固定网格量，并为不同侧板预测状态计算油膜参数。"""

    def __init__(self, mesh, p_lst, omega, gear_type):
        if gear_type not in {"drive", "slave"}:
            raise ValueError("gear_type 必须是 'drive' 或 'slave'")

        points = np.asarray(mesh.points)
        elements = np.asarray(mesh.elements, dtype=int)
        centroids = np.mean(points[elements], axis=1)
        self.x = centroids[:, 0]
        self.y = centroids[:, 1]
        self.ones = np.ones(len(centroids))

        if gear_type == "drive":
            self.U_cells = np.column_stack(
                (-omega * self.y, omega * (self.x - DRIVE_GEAR_CENTER[0]))
            )
        else:
            self.U_cells = np.column_stack(
                (omega * self.y, -omega * (self.x - SLAVE_GEAR_CENTER[0]))
            )

        self.bc_lst = [p_lst[0], *(p_val for _, p_val in p_lst[1:])]

    def solve(self, state: SidePlateState) -> FilmParam:
        roll, pitch, _ = quaternion_to_euler(*state.q)

        # 1. 单元中心膜厚与膜厚梯度。
        sin_pitch = np.sin(pitch)
        sin_roll = np.sin(roll)
        h_cells = (
            -sin_pitch * self.x
            + sin_roll * self.y
            + state.p[2] * self.ones
        )
        h_grad = (-sin_pitch, sin_roll)

        # 2. 两表面相互远离时挤压速度为正。
        ht_cells = (
            state.v[2] * self.ones
            + np.cos(roll) * state.w[0] * self.y
            - np.cos(pitch) * state.w[1] * self.x
        )

        return FilmParam(
            h_cells=h_cells,
            h_grad=h_grad,
            U_cells=self.U_cells,
            ht_cells=ht_cells,
            bc_lst=self.bc_lst,
        )


def calc_film_params(mesh, state: SidePlateState, p_lst, omega, gear_type):
    """一次性计算油膜参数；FSI 子迭代应复用 FilmParameterCalculator。"""

    return FilmParameterCalculator(mesh, p_lst, omega, gear_type).solve(state)
