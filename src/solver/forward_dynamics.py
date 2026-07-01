import numpy as np
from interface.type import (
    SidePlateState,
    SidePlateMassProp,
    ForceTorque,
)
from utils.math_tools import (
    quaternion_multiply,
    quaternion_to_euler,
    euler_to_quaternion,
)


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
        g_vec = self.g_vec

        p, v, q, w = state.p, state.v, state.q, state.w

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

        # 6. 处理被限制的自由度
        p_new = np.array([0.0, 0.0, p_new[2]])
        v_new = np.array([0.0, 0.0, v_new[2]])
        roll, pitch, _ = quaternion_to_euler(*q_new)
        euler_new = np.array([roll, pitch, 0.0])
        q_new = euler_to_quaternion(*euler_new)
        w_new = np.array([w_new[0], w_new[1], 0.0])

        return SidePlateState(p=p_new, v=v_new, q=q_new, w=w_new)


class ImplicitDynamicsSolver:
    def __init__(self, physics_param: SidePlateMassProp, beta=0.25, gamma=0.5):
        self.m = physics_param.m
        self.Ic = physics_param.Ic
        self.Ic_inv = np.linalg.inv(self.Ic)
        self.g_vec = physics_param.g_vec
        # Newmark 参数: beta=0.25, gamma=0.5 → 无条件稳定（常平均加速度法）
        self.beta = beta
        self.gamma = gamma

    def solve(
        self, dt: float, state: SidePlateState, force_torque: ForceTorque
    ) -> SidePlateState:
        """隐式单步推进：接受 CFD 给定的固定 dt，内部解线性系统"""
        m = self.m
        Ic = self.Ic
        Ic_inv = self.Ic_inv
        g_vec = self.g_vec

        p, v, q, w = state.p, state.v, state.q, state.w

        # 1. 体坐标系相对于惯性坐标系的旋转矩阵 R
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

        # 2. 所受合力与合力矩
        G = np.array([m * gi for gi in g_vec])
        F_inertial = R @ force_torque.F
        M_body = force_torque.M
        F_total = G + F_inertial

        # 3. Newmark-beta 隐式更新（平动）
        # v_new = v + dt * ((1-gamma)*a_old + gamma*a_new)
        # p_new = p + dt*v + dt^2*((0.5-beta)*a_old + beta*a_new)
        a_old = F_total / m
        a_new = F_total / m
        v_new = v + dt * ((1 - self.gamma) * a_old + self.gamma * a_new)
        p_new = (
            p + dt * v + dt**2 * ((0.5 - self.beta) * a_old + self.beta * a_new)
        )

        # 4. 转动隐式更新（需解 3x3 线性系统，此处用牛顿-欧拉隐式离散）
        Ic_w = Ic @ w
        gyro = np.cross(w, Ic_w)
        w_dot_old = Ic_inv @ (M_body - gyro)
        w_dot_new = w_dot_old
        w_new = w + dt * ((1 - self.gamma) * w_dot_old + self.gamma * w_dot_new)

        # 5. 四元数更新（一阶近似 + 归一化）
        omega_quat = np.array([0.0, w[0], w[1], w[2]])
        q_new = q + dt * 0.5 * quaternion_multiply(q, omega_quat)
        q_new /= np.linalg.norm(q_new)

        # 6. 处理被限制的自由度
        p_new = np.array([0.0, 0.0, p_new[2]])
        v_new = np.array([0.0, 0.0, v_new[2]])
        roll, pitch, _ = quaternion_to_euler(*q_new)
        euler_new = np.array([roll, pitch, 0.0])
        q_new = euler_to_quaternion(*euler_new)
        w_new = np.array([w_new[0], w_new[1], 0.0])

        return SidePlateState(p=p_new, v=v_new, q=q_new, w=w_new)
