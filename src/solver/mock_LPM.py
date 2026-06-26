"""
模拟集中参数模型的输入输出
"""

import numpy as np


class MockLPM:
    def __init__(self):
        pass

    def solve(self, t):
        return [1e5, ((0.0, -0.01), 1e5)]
