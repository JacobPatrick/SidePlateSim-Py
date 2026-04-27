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
