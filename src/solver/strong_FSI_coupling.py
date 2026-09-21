import numpy as np
from dataclasses import dataclass
from interface.types import (
    SidePlateState,
    ForceTorque,
    SidePlateMassProp,
)
from meshpy.triangle import MeshInfo
from src.solver.reynolds import ReynoldsSolver
from src.solver.contact import ContactSolver
from src.solver.forward_dynamics import ForwardDynamicsSolver
from utils.math_tools import quaternion_multiply, quat_slerp
from utils.calc_film_params import calc_film_params


@dataclass(frozen=True)
class FSIConvergenceTolerances:
    """FSI 固定点残差的分量容差。"""

    position: float = 1e-8
    velocity: float = 1e-5
    angle: float = 1e-6
    angular_velocity: float = 1e-3
    relative: float = 1e-3


def _scaled_difference(calc, pred, absolute_tol, relative_tol):
    scale = absolute_tol + relative_tol * np.maximum(
        np.abs(calc), np.abs(pred)
    )
    return (calc - pred) / scale


def _calc_res_vec(
    s_calc,
    s_pred,
    tolerances: FSIConvergenceTolerances | None = None,
):
    """
    计算 FSI 子迭代的向量残差

    Params:
        s_calc: 动力学求解后计算出的状态 (t_{n+1} 的预测)
        s_pred: 当前子迭代步的预测状态
        tolerances: 各状态分量的绝对/相对容差
    """
    if tolerances is None:
        tolerances = FSIConvergenceTolerances()

    # 每类物理量使用独立尺度，避免某一分量仅因单位选择而支配残差。
    res_p = _scaled_difference(
        s_calc.p,
        s_pred.p,
        tolerances.position,
        tolerances.relative,
    )
    res_v = _scaled_difference(
        s_calc.v,
        s_pred.v,
        tolerances.velocity,
        tolerances.relative,
    )
    res_w = _scaled_difference(
        s_calc.w,
        s_pred.w,
        tolerances.angular_velocity,
        tolerances.relative,
    )

    # 3. 姿态四元数 q 的残差
    # 四元数不能直接相减，需计算误差四元数 q_err = q_calc * q_pred_inv
    q_pred_conj = np.array(
        [s_pred.q[0], -s_pred.q[1], -s_pred.q[2], -s_pred.q[3]]
    )
    q_err = quaternion_multiply(s_calc.q, q_pred_conj)

    # 保证误差四元数走最短路径（实部为正）
    if q_err[0] < 0:
        q_err = -q_err

    # 小角度下 2*q_err[1:] 是旋转向量。
    res_q = 2.0 * q_err[1:] / tolerances.angle

    # 拼接成完整的一维向量
    return np.concatenate([res_p, res_v, res_q, res_w])


def _relax(s_old: SidePlateState, s_new: SidePlateState, alpha: float):
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
        drive_contact_solver: ContactSolver,
        slave_contact_solver: ContactSolver,
        dynamics_solver: ForwardDynamicsSolver,
        side_plate_mass_prop: SidePlateMassProp,
        max_sub_iter: int = 10,
        tol: float = 1.0,
        convergence_tolerances: FSIConvergenceTolerances | None = None,
        initial_relaxation: float = 0.5,
        min_relaxation: float = 0.01,
        max_relaxation: float = 0.9,
    ):
        self.drive_mesh = drive_mesh
        self.slave_mesh = slave_mesh
        self.drive_p_lst = drive_p_lst
        self.slave_p_lst = slave_p_lst
        self.omega = omega
        self.drive_reynolds_solver = drive_reynolds_solver
        self.slave_reynolds_solver = slave_reynolds_solver
        self.drive_contact_solver = drive_contact_solver
        self.slave_contact_solver = slave_contact_solver
        self.dynamics_solver = dynamics_solver
        self.side_plate_mass_prop = side_plate_mass_prop
        self.max_sub_iter = max_sub_iter
        self.tol = tol
        self.convergence_tolerances = (
            convergence_tolerances or FSIConvergenceTolerances()
        )
        self.initial_relaxation = initial_relaxation
        self.min_relaxation = min_relaxation
        self.max_relaxation = max_relaxation

        # Aitken 参数历史
        self.aitken_alpha = initial_relaxation
        self.prev_res_vec = None

    def solve(
        self,
        dt: float,
        state_prev: SidePlateState,
        non_film_force_torque: ForceTorque,
    ):
        self.prev_res_vec = None
        self.aitken_alpha = self.initial_relaxation

        # 1. 状态预测
        state_pred = state_prev

        # 2. 内收敛循环
        base_tol = self.tol
        state_calc = state_prev
        for num_iter in range(1, self.max_sub_iter + 1):
            contact_flag = False
            #  2.1 油膜求解
            drive_film_param = calc_film_params(
                self.drive_mesh,
                state_pred,
                self.drive_p_lst,
                self.omega,
                "drive",
            )
            slave_film_param = calc_film_params(
                self.slave_mesh,
                state_pred,
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

            # 2.2 接触力求解
            if np.any(drive_film_param.h_cells <= 0):
                contact_flag = True
                calc_drive_contact = self.drive_contact_solver.solve(
                    drive_film_param
                )
                C_drive = calc_drive_contact.F
                center_contact = calc_drive_contact.center
                center_drive = (
                    center_drive * F_drive + center_contact * C_drive
                ) / (F_drive + C_drive)
                F_drive += C_drive

            if np.any(slave_film_param.h_cells <= 0):
                contact_flag = True
                calc_slave_contact = self.slave_contact_solver.solve(
                    slave_film_param
                )
                C_slave = calc_slave_contact.F
                center_contact = calc_slave_contact.center
                center_slave = (
                    center_slave * F_slave + center_contact * C_slave
                ) / (F_slave + C_slave)
                F_slave += C_slave

            # 2.3 动力学求解
            F = np.array([0, 0, F_drive + F_slave + non_film_force_torque.F[2]])
            M_drive = np.cross(
                [*center_drive, 0] - self.side_plate_mass_prop.barycenter,
                [0, 0, F_drive],
            )
            M_slave = np.cross(
                [*center_slave, 0] - self.side_plate_mass_prop.barycenter,
                [0, 0, F_slave],
            )
            M = M_drive + M_slave + non_film_force_torque.M
            force_torque = ForceTorque(F=F, M=M)
            state_calc = self.dynamics_solver.solve(
                dt,
                state_prev,
                force_torque,
                state_eval=state_pred,
            )

            # 2.4 收敛判定
            res_vec = _calc_res_vec(
                state_calc,
                state_pred,
                self.convergence_tolerances,
            )
            res_norm = np.linalg.norm(res_vec, ord=np.inf)
            self.tol = base_tol
            if res_norm < self.tol:
                state_pred = state_calc
                print(f"单步 FSI 求解完成，迭代次数: {num_iter}")
                solve_info = {
                    'success': True,
                    'num_iter': num_iter,
                    'res_norm': res_norm,
                    'F_drive': F_drive,
                    'F_slave': F_slave,
                    'M': M,
                    'F_side_plate': F[2],
                }
                return state_pred, solve_info

            #  2.5 Aitken 松弛
            if self.prev_res_vec is not None:
                dres = res_vec - self.prev_res_vec
                dres_dot = np.dot(dres, dres)
                if dres_dot > 1e-12:
                    self.aitken_alpha *= (
                        -np.dot(self.prev_res_vec, dres) / dres_dot
                    )
                    self.aitken_alpha = np.clip(
                        self.aitken_alpha,
                        self.min_relaxation,
                        self.max_relaxation,
                    )

            self.prev_res_vec = res_vec.copy()

            state_pred = _relax(state_pred, state_calc, self.aitken_alpha)

        else:
            print(
                f"警告: FSI 单步求解器在最大迭代次数内未收敛，步长: {dt * 1000:.3f}ms, 残差: {res_norm:.3e}"
            )
            solve_info = {
                'success': False,
                'num_iter': self.max_sub_iter,
                'res_norm': res_norm,
                'F_drive': F_drive,
                'F_slave': F_slave,
                'M': M,
                'F_side_plate': F[2],
            }

            return state_pred, solve_info
