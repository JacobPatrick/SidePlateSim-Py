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
    rotation_speed = np.float64(params.gear.rotation_speed)
    omega = rotation_speed * 2 * np.pi / 60.0  # 转速转换为角速度 [rad/s]
    status_vec = eval(params.gear.status_vec)

    # 油液物性
    # oil_rho = np.float64(params.fluid.density)
    oil_mu = np.float64(params.fluid.viscosity)

    # 油膜参数
    p0 = np.float64(params.film.p_0)

    # 1. 生成齿轮轮廓（单齿轮廓）
    print(
        f"生成参数: 齿数={teeth_num}, 模数={module}, 内径={inner_radius}, 压力角={pressure_angle}°"
    )
    gear = InvoluteGear(
        module=module,
        teeth_num=teeth_num,
        inner_radius=inner_radius,
        pressure_angle=pressure_angle,
    )
    tooth_poly, gear_poly = gear.generate_single_tooth_profile(frame_count=8)

    # 2. 划分网格
    mesh = shapely_to_meshpy(gear_poly, max_area=1e-7)

    # 3. 求仿真油膜参数表
    points = np.array(mesh.points)
    elements = np.array(mesh.elements)
    centroids = np.mean(points[elements], axis=1)

    # 3.1 计算节点处的油膜厚度
    h_cells = (
        np.sin(status_vec[2]) * np.array([p[1] for p in centroids])
        - np.sin(status_vec[4]) * np.array([p[0] for p in centroids])
        + status_vec[0] * np.ones(len(centroids))
    )
    # 非负检查
    assert np.any(
        h_cells > 0
    ), "警告: 油膜厚度存在非正值，请检查齿轮位姿参数设置！"

    # 3.2 确定边界条件
    bc_dict = dict(enumerate([p0] * len(mesh.facet_markers)))

    # 3.3 计算三角网格中心处的相对运动速度
    U_cells = np.array([[omega * p[1], omega * p[0]] for p in centroids])

    # 3.4 计算三角网格中心处的挤压速度
    ht_cells = (
        np.cos(status_vec[2])
        * status_vec[3]
        * np.array([p[1] for p in centroids])
        - np.cos(status_vec[4])
        * status_vec[5]
        * np.array([p[0] for p in centroids])
        + status_vec[1] * np.ones(len(centroids))
    )

    case = ReynoldsSolver(
        mesh,
        h_cells,
        mu=oil_mu,
        U_cells=U_cells,
        ht_cells=ht_cells,
        bc_dict=bc_dict,
    )
    p = case.solve()
    F, (i, j) = case.calc_force(p)

    plot_pressure_distribution(
        mesh, p, fig_name="pressure_distribution", mode='save'
    )
    print(f"油膜压力: {F:.3f}N, 作用点坐标: ({i:.5f}, {j:.5f})m")


if __name__ == '__main__':
    main()
