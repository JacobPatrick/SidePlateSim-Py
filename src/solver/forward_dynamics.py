import numpy as np
from interface.types import (
    SidePlateState,
    SidePlateMassProp,
    ForceTorque,
)
from utils.math_tools import (
    quaternion_multiply,
    quaternion_to_euler,
    rotate_vector_by_quaternion,
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
        g_vec = self.g_vec

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
        G = np.array([m * i for i in g_vec])
        F_inertial = R @ force_torque.F  # 转换到惯性系
        F_total = G + F_inertial

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

        # 5. 碰撞检测与处理
        roll, pitch, _ = quaternion_to_euler(*q)

        A = -GEAR_RADIUS * np.sin(pitch)
        B = GEAR_RADIUS * np.sin(roll)
        max_h = np.hypot(A, B)  # 倾斜带来的最大高度差

        offset_1 = DRIVE_GEAR_CENTER[0] * np.sin(pitch)
        offset_2 = SLAVE_GEAR_CENTER[0] * np.sin(pitch)
        z_crit_1 = offset_1 + max_h
        z_crit_2 = offset_2 + max_h
        z_safe = max(z_crit_1, z_crit_2)

        z = p[2]
        if z < z_safe:
            # 1. 还原位姿为上一时间步的位姿
            p = state.p.copy()
            q = state.q.copy()

            # 2. 确定接触点坐标
            n_sideplate = rotate_vector_by_quaternion(
                q, np.array([0.0, 0.0, 1.0])
            )  # 侧板法向量
            n_projected = np.array(
                [n_sideplate[0], n_sideplate[1], 0.0]
            )  # 投影到 xy 平面
            product = np.dot(
                n_projected,
                (np.array(DRIVE_GEAR_CENTER) - np.array(SLAVE_GEAR_CENTER)),
            )
            if product > 0:
                # 接触点在驱动齿轮侧
                P = np.array(
                    DRIVE_GEAR_CENTER
                ) + GEAR_RADIUS * n_projected / np.linalg.norm(n_projected)
            else:
                # 接触点在从动齿轮侧
                P = np.array(
                    SLAVE_GEAR_CENTER
                ) + GEAR_RADIUS * n_projected / np.linalg.norm(n_projected)

            r = P - p  # 侧板质心到接触点的向量
            n = np.array([0.0, 0.0, 1.0])  # 齿轮端面法向量
            e = 0.2  # 恢复系数

            # 3. 求接触点在惯性系下的速度
            v_p = v + np.cross(w, r)

            # 4. 求法向侧板有效质量
            m_eff = 1 / (
                1 / m + np.cross(r, n) @ (self.Ic_inv @ np.cross(r, n))
            )

            # 5. 求冲量的法向分量（标量）
            J_n = -(1 + e) * m_eff * np.dot(v_p, n)

            # 6. 更新速度和角速度
            v += (J_n / m) * n
            w += self.Ic_inv @ np.cross(r, J_n * n)

        return SidePlateState(p=p, v=v, q=q, w=w)
