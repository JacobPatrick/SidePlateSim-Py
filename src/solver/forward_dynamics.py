import numpy as np
from interface.types import (
    SidePlateState,
    SidePlateMassProp,
    ForceTorque,
)
from utils.math_tools import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)

DRIVE_GEAR_CENTER = (0.0305, 0, 0)
SLAVE_GEAR_CENTER = (-0.0305, 0, 0)
GEAR_RADIUS = 0.035


class ForwardDynamicsSolver:
    def __init__(self, physics_param: SidePlateMassProp):
        """
        Params:
            m: 质量
            Ic: 3x3 惯性张量 (体坐标系下)
            g_vec: 重力加速度矢量 (惯性系下)
        """
        self.m = physics_param.m
        self.Ic = physics_param.Ic
        self.g_vec = physics_param.g_vec
        self.Ic_inv = np.linalg.inv(self.Ic)

        # 约束自由度: 平动仅沿 z 方向，转动仅绕 x、y 轴
        self.trans_mask = np.array([0, 0, 1])
        self.rot_mask = np.array([1, 1, 0])

    def solve(
        self,
        dt: float,
        state_prev: SidePlateState,
        force_torque: ForceTorque,
        state_eval: SidePlateState | None = None,
    ) -> SidePlateState:
        """
        单步正向动力学求解。

        状态更新始终从上一物理时间层 ``state_prev`` 开始；加速度、
        陀螺项和姿态导数则在 ``state_eval`` 上评价。这样 FSI 固定点
        迭代可以在预测的 n+1 状态上评价导数，又不会在子迭代中重复
        累加时间步。

        Params:
            dt: 时间步长
            state_prev: 上一物理时间层的已收敛状态
            force_torque: 外力和力矩（体坐标系下）
            state_eval: 导数评价状态；省略时使用 state_prev
        """
        m = self.m
        Ic = self.Ic
        if state_eval is None:
            state_eval = state_prev

        # 时间积分基准和导数评价状态必须相互独立。
        p_prev = state_prev.p.copy() * self.trans_mask
        v_prev = state_prev.v.copy() * self.trans_mask
        q_prev = state_prev.q.copy()
        w_prev = state_prev.w.copy() * self.rot_mask

        q_eval = state_eval.q.copy()
        q_eval /= np.linalg.norm(q_eval)
        w_eval = state_eval.w.copy() * self.rot_mask

        # 1. 计算旋转矩阵 R (从体坐标系到惯性坐标系)
        R = quaternion_to_rotation_matrix(q_eval)

        # 2. 所受合力
        F_inertial = R @ force_torque.F  # 转换到惯性系
        F_total = F_inertial  # 不考虑重力影响

        # 3. 求解加速度 (牛顿-欧拉方程)
        a_c = (F_total / m) * self.trans_mask

        # 欧拉方程: M = I * w_dot + w x (I * w)  =>  w_dot = I^-1 * (M - w x (I * w))
        Ic_w = Ic @ w_eval
        gyroscopic_term = np.cross(w_eval, Ic_w)
        w_dot = (
            self.Ic_inv @ (force_torque.M - gyroscopic_term)
        ) * self.rot_mask

        # 4. 从上一物理时间层构造候选状态，避免 FSI 子迭代累积时间。
        v = v_prev + a_c * dt
        p = p_prev + v * dt
        w = w_prev + w_dot * dt

        # 5. 姿态导数在预测状态上评价。
        # q_dot = 0.5 * q * [0, wx, wy, wz]
        omega_quat_eval = np.array(
            [0.0, w_eval[0], w_eval[1], w_eval[2]]
        )
        q_dot = 0.5 * quaternion_multiply(q_eval, omega_quat_eval)

        q = q_prev + q_dot * dt
        q /= np.linalg.norm(q)

        return SidePlateState(p=p, v=v, q=q, w=w)
