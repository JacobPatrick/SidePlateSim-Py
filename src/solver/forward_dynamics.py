import numpy as np
from interface.type import (
    SidePlateState,
    SidePlateMassProp,
    ForceTorque,
)
from utils.math_tools import quaternion_multiply


class ForwardDynamicsSolver:
    def __init__(self, dt, physics_param: SidePlateMassProp):
        self.dt = dt
        self.m = physics_param.m
        self.Ic = physics_param.Ic
        self.g_vec = physics_param.g_vec

    def solve(
        self,
        state: SidePlateState,
        force_torque: ForceTorque,
    ) -> SidePlateState:
        """
        单步正向动力学求解
        Params:
            state: {'p': [px, py, pz], 'v': [vx, vy, vz], 'q': [qw, qx, qy, qz], 'w': [wx, wy, wz]}
            dt: 时间步长
            m: 质量
            Ic: 3x3 惯性张量 (体坐标系下)
            F: 外力 (惯性系下)
            M: 外力矩 (体坐标系下)
            g_vec: 重力加速度矢量 (惯性系下)
        """
        dt = self.dt
        m = self.m
        Ic = self.Ic
        g_vec = self.g_vec

        p = state.p
        v = state.v
        q = state.q
        w = state.w

        # 1. 计算旋转矩阵 R (从体坐标系到惯性坐标系)
        qw, qx, qy, qz = q
        R = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ]
        )

        # 2. 所受合力
        G = np.array([m * i for i in g_vec])
        F_inertial = R @ force_torque.F  # 转换到惯性系
        F_total = G + F_inertial

        # 3. 求解加速度 (牛顿-欧拉方程)
        a_c = F_total / m

        # 欧拉方程: M = I * w_dot + w x (I * w)  =>  w_dot = I^-1 * (M - w x (I * w))
        Ic_w = Ic @ w
        gyroscopic_term = np.cross(w, Ic_w)
        w_dot = np.linalg.inv(Ic) @ (force_torque.M - gyroscopic_term)

        # 4. 姿态导数 (四元数微分方程)
        # q_dot = 0.5 * q * [0, wx, wy, wz]
        omega_quat = np.array([0.0, w[0], w[1], w[2]])
        q_dot = 0.5 * quaternion_multiply(q, omega_quat)

        # 5. 欧拉积分更新状态
        p_new = p + v * dt
        v_new = v + a_c * dt

        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)

        w_new = w + w_dot * dt

        state_new = SidePlateState(p=p_new, v=v_new, q=q_new, w=w_new)

        return state_new
