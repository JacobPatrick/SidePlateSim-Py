import numpy as np
from interface.type import (
    SidePlateState,
    FilmParam,
    ForceTorque,
    SidePlateMassProp,
)
from meshpy.triangle import MeshInfo
from src.solver.reynolds import ReynoldsSolver
from src.solver.forward_dynamics import ForwardDynamicsSolver
from utils.math_tools import quat_slerp
from utils.calc_film_params import calc_film_params

P_AIR = 1e5 * 0.0024687143080106173


# TODO: 重写残差计算方法，特别注意数量级问题
def _calc_residual(s1, s2):
    t1 = np.concatenate([s1.p, s1.v, s1.q, s1.w])
    t2 = np.concatenate([s2.p, s2.v, s2.q, s2.w])
    residual = np.linalg.norm(t1 - t2)
    return residual


def _relax(self, s_old, s_new, alpha):
    """线性插值松弛（保持物理量纲一致）"""
    return SidePlateState(
        p=(1 - alpha) * s_old.p + alpha * s_new.p,
        v=(1 - alpha) * s_old.v + alpha * s_new.v,
        q=quat_slerp(s_old.q, s_new.q, alpha),  # 四元数球面插值
        w=(1 - alpha) * s_old.w + alpha * s_new.w,
    )


class SingleStepFSISolver:
    """
    FSI 单步内收敛循环求解器
    """

    def __init__(
        self,
        drive_mesh: MeshInfo,
        slave_mesh: MeshInfo,
        drive_p_lst: list,
        slave_p_lst: list,
        omega: float,
        drive_reynolds_solver: ReynoldsSolver,
        slave_reynolds_solver: ReynoldsSolver,
        dynamics_solver: ForwardDynamicsSolver,
        side_plate_mass_prop: SidePlateMassProp,
        max_sub_iter: int = 10,
        tol: float = 1e-4,
    ):
        self.drive_mesh = drive_mesh
        self.slave_mesh = slave_mesh
        self.drive_p_lst = drive_p_lst
        self.slave_p_lst = slave_p_lst
        self.omega = omega
        self.drive_reynolds_solver = drive_reynolds_solver
        self.slave_reynolds_solver = slave_reynolds_solver
        self.dynamics_solver = dynamics_solver
        self.side_plate_mass_prop = side_plate_mass_prop
        self.max_sub_iter = max_sub_iter
        self.tol = tol

        # Aitken 参数历史
        self.aitken_alpha = 0.5  # 初始松弛因子
        self.prev_residual = None

    def solve(
        self,
        dt: float,
        state_prev: SidePlateState,
        force_torque: ForceTorque,
    ):
        self.prev_residual = None

        # 1. 状态预测
        state_pred = self.dynamics_solver.solve(dt, state_prev, force_torque)

        # 2. 内收敛循环
        num_iter = 0
        state_calc = state_prev.copy()
        while True:
            #  2.1. 油膜求解
            drive_film_param = calc_film_params(
                self.drive_mesh,
                state_calc,
                self.drive_p_lst,
                self.omega,
                "drive",
            )
            slave_film_param = calc_film_params(
                self.slave_mesh,
                state_calc,
                self.slave_p_lst,
                self.omega,
                "slave",
            )
            calc_drive_pressure = self.drive_reynolds_solver.solve(
                drive_film_param
            )
            calc_slave_pressure = self.slave_reynolds_solver.solve(
                slave_film_param
            )
            F_drive = calc_drive_pressure.F
            F_slave = calc_slave_pressure.F
            center_drive = calc_drive_pressure.center
            center_slave = calc_slave_pressure.center

            # 2.2. 动力学求解
            F = np.array(
                [0, 0, F_drive + F_slave - 2 * P_AIR]
            )  # TODO: 加入齿腔油压和背压
            M_drive = np.cross(
                [0, 0, F_drive],
                self.side_plate_mass_prop.barycenter - [*center_drive, 0],
            )
            M_slave = np.cross(
                [0, 0, F_slave],
                self.side_plate_mass_prop.barycenter - [*center_slave, 0],
            )
            M = M_drive + M_slave  # TODO: 加入齿腔油压产生的力矩
            force_torque = ForceTorque(F=F, M=M)
            state_calc = self.dynamics_solver.solve(
                dt, state_prev, force_torque
            )

            # 2.3. 收敛判定
            residual = _calc_residual(state_calc, state_pred)
            if residual < self.tol:
                state_pred = state_calc
                break

            #  2.4. Aitken 松弛
            if self.prev_residual is not None:
                dres = residual - self.prev_residual
                dres = np.dot(dres, dres)
                if dres > 1e-12:
                    self.aitken_alpha *= (
                        -np.dot(self.prev_residual, dres) / dres
                    )
                    # 极薄油膜刚度极大，建议将 alpha 上界压得更低（如 0.5），防止超调发生碰撞
                    self.aitken_alpha = np.clip(self.aitken_alpha, 0.05, 0.5)

            self.prev_residual = residual.copy()

            state_pred = _relax(state_pred, state_calc, self.aitken_alpha)

            num_iter += 1
            if num_iter >= self.max_sub_iter:
                print("警告: FSI 单步求解器在最大迭代次数内未收敛")
                break

        return state_pred
