"""
数学工具函数
"""

import numpy as np


def rot_matrix(angle):
    """生成 2D 旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotation(X, angle, center=None):
    """Numpy 数组旋转变换"""
    if center is None:
        return np.dot(X, rot_matrix(angle))
    else:
        return np.dot(X - center, rot_matrix(angle)) + center


def deg2rad(x):
    """角度转弧度"""
    return (np.pi / 180.0) * x


def quaternion_multiply(q1, q2):
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quaternion_to_euler(w, x, y, z):
    """
    将四元数转换为 Z-Y-X 顺序的欧拉角 (Roll, Pitch, Yaw)
    """
    # 1. 归一化
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("四元数不能为零向量")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # 2. 计算 Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.atan2(sinr_cosp, cosr_cosp)

    # 3. 计算 Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.asin(sinp)

    # 4. 计算 Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    """
    将 Z-Y-X 顺序的欧拉角 (Roll, Pitch, Yaw) 转换为四元数
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    """
    w, x, y, z = q
    R = np.array(
        [
            [
                1 - 2 * (y**2 + z**2),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x**2 + z**2),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x**2 + y**2),
            ],
        ]
    )
    return R


def quat_slerp(q1, q2, t):
    """四元数简单归一化线性插值，严格可用 SLERP"""
    q = (1 - t) * q1 + t * q2
    return q / np.linalg.norm(q)
