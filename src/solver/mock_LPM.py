"""
模拟集中参数模型的输入输出
"""

import numpy as np


class MockLPM:
    def __init__(self):
        pass

    def solve(self, t):
        """
        模拟LPM求解齿腔压力的过程，返回两个列表，分别表示主动轮和从动轮的压力分布
        """
        return [1e5, ((0.0, -0.01), 1e5)], [1e5, ((0.0, -0.01), 1e5)]
