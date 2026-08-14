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
        self, dt: float, state: SidePlateState, force_torque: ForceTorque
    ) -> SidePlateState:
        """
        单步正向动力学求解
        Params:
            dt: 时间步长
            state: 侧板状态，包含位置p、速度v、姿态q和角速度w
            force_torque: 外力和力矩（体坐标系下）
        """
        m = self.m
        Ic = self.Ic

        # 使用 copy 避免污染原始状态
        p = state.p.copy()
        v = state.v.copy()
        q = state.q.copy()
        w = state.w.copy()

        # 1. 受约束自由度位移和速度强制归零
        p = p * self.trans_mask
        v = v * self.trans_mask
        w = w * self.rot_mask

        # 1. 计算旋转矩阵 R (从体坐标系到惯性坐标系)
        R = quaternion_to_rotation_matrix(q)

        # 2. 所受合力
        F_inertial = R @ force_torque.F  # 转换到惯性系
        F_total = F_inertial  # 不考虑重力影响

        # 3. 求解加速度 (牛顿-欧拉方程)
        a_c = (F_total / m) * self.trans_mask

        # 欧拉方程: M = I * w_dot + w x (I * w)  =>  w_dot = I^-1 * (M - w x (I * w))
        Ic_w = Ic @ w
        gyroscopic_term = np.cross(w, Ic_w)
        w_dot = (
            self.Ic_inv @ (force_torque.M - gyroscopic_term)
        ) * self.rot_mask

        # 4. 更新平动状态
        v += a_c * dt
        p += v * dt

        # 4. 姿态导数 (四元数微分方程)
        # q_dot = 0.5 * q * [0, wx, wy, wz]
        w += w_dot * dt
        omega_quat_new = np.array([0.0, w[0], w[1], w[2]])
        q_dot = 0.5 * quaternion_multiply(q, omega_quat_new)

        q += q_dot * dt
        q /= np.linalg.norm(q)

        return SidePlateState(p=p, v=v, q=q, w=w)
