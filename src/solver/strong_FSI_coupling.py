import numpy as np
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


def _calc_res_vec(s_calc, s_pred, L_ref=3e-2, H_ref=1e-4, W_ref=1.0):
    """
    计算 FSI 子迭代的向量残差

    Params:
        s_calc: 动力学求解后计算出的状态 (t_{n+1} 的预测)
        s_pred: 当前子迭代步的预测状态
        L_ref: 端面的几何特征尺度 (例如齿轮半径，如 3e-2 m)
        H_ref: 位置的特征尺度 (例如侧板间隙量级，如 1e-4 m)
        W_ref: 角速度的特征尺度 (例如转子角速度量级，如 1.0 rad/s)
    """
    # 1. 平动和转动角速度的绝对差值，并除以特征尺度进行无量纲化
    res_p = (s_calc.p - s_pred.p) / H_ref
    res_w = (s_calc.w - s_pred.w) / W_ref

    # 2. 线速度 v 的残差
    V_ref = L_ref * W_ref
    res_v = (s_calc.v - s_pred.v) / V_ref

    # 3. 姿态四元数 q 的残差
    # 四元数不能直接相减，需计算误差四元数 q_err = q_calc * q_pred_inv
    q_pred_conj = np.array(
        [s_pred.q[0], -s_pred.q[1], -s_pred.q[2], -s_pred.q[3]]
    )
    q_err = quaternion_multiply(s_calc.q, q_pred_conj)

    # 保证误差四元数走最短路径（实部为正）
    if q_err[0] < 0:
        q_err = -q_err

    # 对于小角度误差，四元数的虚部近似等于旋转矢量 (Rodrigues 参数)
    res_q = q_err[1:]

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
        tol: float = 1e-4,
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

        # Aitken 参数历史
        self.aitken_alpha = 0.5  # 初始松弛因子
        self.prev_res_vec = None

    def solve(
        self,
        dt: float,
        state_prev: SidePlateState,
        non_film_force_torque: ForceTorque,
    ):
        self.prev_res_vec = None

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
            F = np.array([0, 0, F_drive + F_slave])
            M_drive = np.cross(
                [*center_drive, 0] - self.side_plate_mass_prop.barycenter,
                [0, 0, F_drive],
            )
            M_slave = np.cross(
                [*center_slave, 0] - self.side_plate_mass_prop.barycenter,
                [0, 0, F_slave],
            )
            M = M_drive + M_slave
            force_torque = ForceTorque(
                F=F + non_film_force_torque.F, M=M + non_film_force_torque.M
            )
            state_calc = self.dynamics_solver.solve(
                dt, state_prev, force_torque
            )

            # 2.4 收敛判定
            res_vec = _calc_res_vec(
                state_calc, state_pred, L_ref=1e-4, W_ref=1.0
            )
            res_norm = np.linalg.norm(res_vec)
            if contact_flag:
                # 若侧板和齿轮端面存在接触，放宽收敛判据
                self.tol = base_tol * 1000
            else:
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
                    'F_side_plate': F[2] + non_film_force_torque.F[2],
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
                    self.aitken_alpha = np.clip(self.aitken_alpha, 0.1, 0.9)

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
                'F_side_plate': F[2] + non_film_force_torque.F[2],
            }

            return state_pred, solve_info
