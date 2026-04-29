import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.geometry.gear_profile import InvoluteGear
from src.postproc.visualize import (
    plot_shapely_poly,
    plot_mesh,
    plot_pressure_distribution,
)
from src.geometry.mesher import shapely_to_meshpy
from src.solver.reynolds_solver import ReynoldsSolver
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
    tooth_poly, gear_poly = gear.generate_single_tooth_profile(frame_count=4)
    # plot_shapely_poly(gear_poly, fig_name=f"gear_profile")
    # plot_shapely_poly(
    #     tooth_poly,
    #     fig_name=f"tooth_profile",
    # )
    mesh = shapely_to_meshpy(tooth_poly, max_area=1e-8)
    h_nodes = (
        h_base * np.ones(len(mesh.points))
        + h_tilt[0] * np.array([p[0] for p in mesh.points])
        + h_tilt[1] * np.array([p[1] for p in mesh.points])
    )
    bc_dict = dict(enumerate([p0] * len(mesh.facet_markers)))

    case = ReynoldsSolver(
        mesh, h_nodes, mu=oil_mu, U_vec=U_vec, ht=ht, bc_dict=bc_dict
    )
    p = case.solve()
    F, (i, j) = case.calc_force(p)

    plot_pressure_distribution(
        mesh, p, fig_name="pressure_distribution", mode='show'
    )
    print(f"油膜压力: {F:.3f}, 作用点坐标: ({i:.5f}, {j:.5f})")


if __name__ == '__main__':
    main()
