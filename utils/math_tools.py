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
