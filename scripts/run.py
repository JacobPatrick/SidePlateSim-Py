import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.gear_profile import InvoluteGear


def main():
    # 齿轮参数
    module = 2.0
    teeth_num = 17
    inner_radius = 7.5
    pressure_angle = 20.0

    print(
        f"生成参数: 齿数={teeth_num}, 模数={module}, 内径={inner_radius}, 压力角={pressure_angle}°"
    )
    gear = InvoluteGear(
        module=module,
        teeth_num=teeth_num,
        inner_radius=inner_radius,
        pressure_angle=pressure_angle,
    )


if __name__ == '__main__':
    main()
