import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from src.geometry.gear_profile import InvoluteGear
from src.postproc.visualize import plot_shapely_poly


def main():
    # 齿轮参数
    module = 2.0
    teeth_num = 17
    pressure_angle = 20.0

    print(
        f"生成参数: 齿数={teeth_num}, 模数={module}, 压力角={pressure_angle}°"
    )
    gear = InvoluteGear(module=module, teeth_num=teeth_num, pressure_angle=pressure_angle)
    tooth_poly, gear_poly = gear.generate_single_tooth_profile()
    # plot_shapely_poly(gear_poly, fig_name=f"gear_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    plot_shapely_poly(tooth_poly, fig_name=f"tooth_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

if __name__ == '__main__':
    main()
