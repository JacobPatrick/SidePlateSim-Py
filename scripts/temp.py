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
    tooth_poly, gear_poly = gear.generate_single_tooth_profile(frame_count=4)
    # plot_shapely_poly(gear_poly, fig_name=f"gear_profile")
    # plot_shapely_poly(
    #     tooth_poly,
    #     fig_name=f"tooth_profile",
    # )
    mesh = shapely_to_meshpy(tooth_poly, max_area=0.01)
    h_nodes = 1e-5 * np.ones(len(mesh.points)) - 3e-7 * np.array(
        [p[1] for p in mesh.points]
    )
    bc_dict = dict(enumerate(mesh.facet_markers))

    case = ReynoldsSolver(
        mesh, h_nodes, mu=0.1, U_vec=(5.0, 0.0), ht=1e-3, bc_dict=bc_dict
    )
    p = case.solve()

    plot_pressure_distribution(
        mesh, p, fig_name="pressure_distribution", mode='show'
    )


if __name__ == '__main__':
    main()
