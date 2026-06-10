"""
齿轮生成模块
"""

import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.affinity import rotate, scale

from utils.math_tools import rotation


class InvoluteGear:
    """生成渐开线齿轮"""

    def __init__(
        self, module, teeth_num, inner_radius, thickness=5e-2, pressure_angle=20
    ):
        self.module = module * 1e-3
        self.teeth_num = teeth_num
        self.inner_radius = inner_radius
        self.thickness = thickness
        self.pressure_angle = pressure_angle

        self.pitch_diameter = self.module * self.teeth_num
        self.base_diameter = self.pitch_diameter * np.cos(
            np.radians(self.pressure_angle)
        )
        self.addendum = self.module
        self.dedendum = 1.25 * self.module
        self.outside_diameter = self.pitch_diameter + 2 * self.addendum
        self.root_diameter = self.pitch_diameter - 2 * self.dedendum

        self.r_p = self.pitch_diameter / 2.0
        self.r_b = self.base_diameter / 2.0
        self.r_a = self.outside_diameter / 2.0
        self.r_f = self.root_diameter / 2.0

        # 检查内径合法性（内径不能大于等于齿根圆）
        if self.inner_radius >= self.r_f:
            raise ValueError("Inner radius is invalid.")

    def generate_gear_profile(self, backlash=0.0, frame_count=32):
        """生成标准渐开线直齿轮齿廓"""
        tooth_width = self.module * np.pi / 2.0 - backlash

        # 基础梯形刀具轮廓（4个顶点）
        profile = np.array(
            [
                [
                    -(
                        0.5 * tooth_width
                        + self.addendum
                        * np.tan(np.radians(self.pressure_angle))
                    ),
                    self.addendum,
                ],
                [
                    -(
                        0.5 * tooth_width
                        - self.dedendum
                        * np.tan(np.radians(self.pressure_angle))
                    ),
                    -self.dedendum,
                ],
                [
                    (
                        0.5 * tooth_width
                        - self.dedendum
                        * np.tan(np.radians(self.pressure_angle))
                    ),
                    -self.dedendum,
                ],
                [
                    (
                        0.5 * tooth_width
                        + self.addendum
                        * np.tan(np.radians(self.pressure_angle))
                    ),
                    self.addendum,
                ],
            ]
        )

        poly_list = []
        prev_X = None

        # 纯滚动展开角
        _l = 2.0 * tooth_width / self.r_p

        for theta in np.linspace(0, _l, frame_count):
            # 刀具平移 + 工件旋转（模拟纯滚动切削）
            X = rotation(
                profile + np.array([-theta * self.r_p, self.r_p]), theta
            )
            if prev_X is not None:
                # 用相邻两帧的顶点构造凸包，填补离散间隙
                pts = np.vstack([X, prev_X])
                poly_list.append(MultiPoint(pts).convex_hull)
            prev_X = X

        # 合并扫掠片生成单侧齿面
        tooth_poly = unary_union(poly_list)
        # 沿 Y 轴镜像得到完整单齿
        tooth_poly = tooth_poly.union(
            scale(tooth_poly, xfact=-1, yfact=1, origin=(0, 0))
        )

        # 迭代切削生成完整齿轮
        gear_poly = Point(0.0, 0.0).buffer(self.r_a)
        angle_step = (2.0 * np.pi) / self.teeth_num
        for _ in range(self.teeth_num):
            gear_poly = rotate(
                gear_poly.difference(tooth_poly),
                angle_step,
                origin=(0.0, 0.0),
                use_radians=True,
            )

        # 绘制内径
        if self.inner_radius > 0:
            inner_circle = Point(0.0, 0.0).buffer(self.inner_radius)
            gear_poly = gear_poly.difference(inner_circle)

        # 绘制单齿分割扇区
        # 1. 计算单齿分配角度
        angle_step = 2 * np.pi / self.teeth_num
        start_angle = np.pi / 2
        end_angle = start_angle + angle_step

        # 2. 构造扇区多边形（含圆心）
        n_arc = 120
        angles = np.linspace(start_angle, end_angle, n_arc)
        sector_pts = [
            (self.r_a * np.cos(a), self.r_a * np.sin(a)) for a in angles
        ]
        sector_pts.append((0.0, 0.0))  # 闭合至圆心，自然形成两条径向直边
        sector = Polygon(sector_pts)

        # 3. 布尔交集：裁剪出单齿
        tooth_poly = gear_poly.intersection(sector)

        # 4. 鲁棒性处理：若返回 MultiPolygon，保留面积最大的主体
        if tooth_poly.geom_type == 'MultiPolygon':
            tooth_poly = max(tooth_poly.geoms, key=lambda p: p.area)
        elif tooth_poly.is_empty:
            raise ValueError(
                "扇区与齿轮无交集，请检查 angle_offset 或 teeth_count"
            )

        return tooth_poly, gear_poly
