"""
齿轮生成模块
"""
import madcad as cad
from util.math_tools import *


class InvoluteGear:
    """生成渐开线齿轮"""
    def __init__(self, module, teeth_num, thickness, pressure_angle=20):
        self.module = module
        self.teeth_num = teeth_num
        self.thickness = thickness
        self.pressure_angle = pressure_angle

        self.pitch_diameter = self.module * self.teeth_num
        self.base_diameter = self.pitch_diameter * np.cos(np.radians(self.pressure_angle))
        self.addendum = self.module
        self.dedendum = 1.25 * self.module
        self.outside_diameter = self.pitch_diameter + 2 * self.addendum
        self.root_diameter = self.pitch_diameter - 2 * self.dedendum


    def generate_profile(self):
        """生成渐开线齿廓"""
        pass


    def get_closed_loop(self):
        """获取齿廓闭合曲线"""
        pass