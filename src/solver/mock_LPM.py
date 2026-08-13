"""
模拟集中参数模型的输入输出
"""

import numpy as np

DRIVE_GEAR_CENTER = (0.0305, 0, 0)
SLAVE_GEAR_CENTER = (-0.0305, 0, 0)
GEAR_RADIUS = 0.035


class MockLPM:
    def __init__(self):
        pass

    def solve(self, t):
        """
        模拟LPM求解齿腔压力的过程，返回两个列表，分别表示主动轮和从动轮的压力分布
        """
        # return [0.0, ((0.0, -0.01), 0.0)], [0.0, ((0.0, -0.01), 0.0)]
        return [
            0.0,
            ((DRIVE_GEAR_CENTER[0] + GEAR_RADIUS, DRIVE_GEAR_CENTER[1]), 1.9e6),
            ((DRIVE_GEAR_CENTER[0], DRIVE_GEAR_CENTER[1] - GEAR_RADIUS), 4.9e6),
            ((DRIVE_GEAR_CENTER[0] - GEAR_RADIUS, DRIVE_GEAR_CENTER[1]), 0.0),
        ], [
            0.0,
            ((SLAVE_GEAR_CENTER[0] + GEAR_RADIUS, SLAVE_GEAR_CENTER[1]), 4.9e6),
            ((SLAVE_GEAR_CENTER[0], SLAVE_GEAR_CENTER[1] - GEAR_RADIUS), 1.9e6),
            ((SLAVE_GEAR_CENTER[0] - GEAR_RADIUS, SLAVE_GEAR_CENTER[1]), 0.0),
        ]
