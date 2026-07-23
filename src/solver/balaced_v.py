import numpy as np
from meshpy.triangle import MeshInfo
from interface.types import (
    SidePlateState,
    SidePlateMassProp,
)
from utils.calc_film_params import calc_film_params
from src.solver.reynolds import ReynoldsSolver
from scipy.optimize import root
from utils.math_tools import quaternion_to_euler

DRIVE_GEAR_CENTER = (0.0305, 0)
SLAVE_GEAR_CENTER = (-0.0305, 0)
GEAR_RADIUS = 0.035
P_AIR = 1e5 * 0.0024687143080106173


def solve_va_vb_vc(vz, wx, wy, roll, pitch):
    A = np.cos(roll) * GEAR_RADIUS
    B = np.cos(pitch) * SLAVE_GEAR_CENTER[0]
    C = np.cos(pitch) * (DRIVE_GEAR_CENTER[0] + GEAR_RADIUS)

    va = vz + A * wx - B * wy
    vb = vz - A * wx - B * wy
    vc = vz - C * wy

    return va, vb, vc


def solve_vz_wx_wy(va, vb, vc, roll, pitch):
    A = np.cos(roll) * GEAR_RADIUS
    B = np.cos(pitch) * SLAVE_GEAR_CENTER[0]
    C = np.cos(pitch) * (DRIVE_GEAR_CENTER[0] + GEAR_RADIUS)

    wx = (va - vb) / (2 * A)
    vz = vc + C * (va - vb) / (2 * A)
    wy = ((C - A) * va - (C + A) * vb + 2 * A * vc) / (2 * A * B)

    return vz, wx, wy


class BalancedVSolver:
    def __init__(
        self,
        drive_mesh: MeshInfo,
        slave_mesh: MeshInfo,
        drive_p_lst: list,
        slave_p_lst: list,
        omega: float,
        side_plate_mass_prop: SidePlateMassProp,
        drive_reynolds_solver: ReynoldsSolver,
        slave_reynolds_solver: ReynoldsSolver,
        state: SidePlateState,
        F_else: np.ndarray,
        M_else: np.ndarray,
    ):
        self.drive_mesh = drive_mesh
        self.slave_mesh = slave_mesh
        self.drive_p_lst = drive_p_lst
        self.slave_p_lst = slave_p_lst
        self.omega = omega
        self.side_plate_mass_prop = side_plate_mass_prop
        self.drive_reynolds_solver = drive_reynolds_solver
        self.slave_reynolds_solver = slave_reynolds_solver
        self.state = state
        self.F_else = F_else
        self.M_else = M_else

    def solve(self) -> SidePlateState:
        roll, pitch, _ = quaternion_to_euler(*self.state.q)
        va, vb, vc = solve_va_vb_vc(
            self.state.v[2], self.state.w[0], self.state.w[1], roll, pitch
        )

        result = root(
            self._res_func,
            x0=np.array([va, vb, vc]),
            method="hybr",
            options={
                "xtol": 1e-6,
                "maxfev": 1000,
                "factor": 0.1,
            },
        )
        if result.success:
            va, vb, vc = result.x
            va = np.clip(va, -10.0, 10.0)
            vb = np.clip(vb, -10.0, 10.0)
            vc = np.clip(vc, -10.0, 10.0)
            vz, wx, wy = solve_vz_wx_wy(va, vb, vc, roll, pitch)
            new_state = SidePlateState(
                p=self.state.p,
                v=np.array([0.0, 0.0, vz]),
                q=self.state.q,
                w=np.array([wx, wy, 0.0]),
            )
            return new_state
        else:
            return None

    def _res_func(self, squeeze_v: np.ndarray):
        va, vb, vc = squeeze_v
        roll, pitch, _ = quaternion_to_euler(*self.state.q)
        vz, wx, wy = solve_vz_wx_wy(va, vb, vc, roll, pitch)
        state = SidePlateState(
            p=self.state.p.copy(),
            v=np.array([0.0, 0.0, vz]),
            q=self.state.q.copy(),
            w=np.array([wx, wy, 0.0]),
        )
        # 1. 计算油膜参数
        drive_film_param = calc_film_params(
            self.drive_mesh, state, self.drive_p_lst, self.omega, "drive"
        )
        slave_film_param = calc_film_params(
            self.slave_mesh, state, self.slave_p_lst, self.omega, "slave"
        )

        # 2. 求解油膜压力
        drive_pressure = self.drive_reynolds_solver.solve(drive_film_param)
        slave_pressure = self.slave_reynolds_solver.solve(slave_film_param)
        F_drive = drive_pressure.F
        F_slave = slave_pressure.F

        # 3. 计算合力和力矩
        center_drive = drive_pressure.center
        center_slave = slave_pressure.center

        F = np.array([0, 0, F_drive + F_slave - 2 * P_AIR])
        M_drive = np.cross(
            [*center_drive, 0] - self.side_plate_mass_prop.barycenter,
            [0, 0, F_drive],
        )
        M_slave = np.cross(
            [*center_slave, 0] - self.side_plate_mass_prop.barycenter,
            [0, 0, F_slave],
        )
        M = M_drive + M_slave

        # 4. 求受力残差
        res_Fz = F[2] + self.F_else[2]
        res_Mx = M[0] + self.M_else[0]
        res_My = M[1] + self.M_else[1]

        return np.array([res_Fz, res_Mx, res_My])
