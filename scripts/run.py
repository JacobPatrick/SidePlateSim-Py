import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.geometry.gear_profile import InvoluteGear
from src.config.config import load_config


def main():
    params = load_config('SimParams_1')

    # 齿轮参数
    module = np.float64(params.gear.module)
    teeth_num = int(params.gear.num_teeth)
    inner_radius = np.float64(params.gear.inner_radius)
    pressure_angle = np.float64(params.gear.pressure_angle)

    # 油液物性
    # oil_rho = np.float64(params.fluid.density)
    oil_mu = np.float64(params.fluid.viscosity)

    # 油膜参数
    h_base = np.float64(params.film.h_base)
    h_tilt = eval(params.film.h_tilt)
    p0 = np.float64(params.film.p_0)
    U_vec = eval(params.film.U_vec)
    ht = np.float64(params.film.ht)

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
