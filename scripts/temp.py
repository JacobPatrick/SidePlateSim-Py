import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.postproc.visualize import plot_pressure_distribution
from src.config.config import load_config
from interface.type import (
    GearProfileDir,
    SidePlateState,
    FilmParam,
    FluidProp,
    Pressure,
    SidePlateMassProp,
    ForceTorque,
)
from src.solver.mock_LPM import MockLPM
from src.solver.mesh_generator import MeshGenerator
from src.solver.reynolds import ReynoldsSolver
from src.solver.forward_dynamics import ForwardDynamicsSolver


def one_step_pipeline(
    t,
    dt,
    omega,
    oil_mu,
    side_plate_mass_prop: SidePlateMassProp,
    prev_status: SidePlateState,
    mock_lpm: MockLPM,
    mesh_generator: MeshGenerator,
    forward_dynamics_solver: ForwardDynamicsSolver,
):
    # 1. 集中参数法求齿腔压力
    p_lst = mock_lpm.solve(t)

    # 2. 网格划分与油膜参数求解
    mesh, film_param = mesh_generator.solve(t, p_lst, prev_status)

    # 3. 求解油膜压力
    fluid_prop = FluidProp(mu=oil_mu)
    reynolds_solver = ReynoldsSolver(mesh, fluid_prop, film_param)
    pressure = reynolds_solver.solve()
    p = pressure.p
    F = pressure.F
    center = pressure.center

    # 4. 求解正向动力学
    F = np.array([0.0, 0.0, F])
    M = np.cross(F, side_plate_mass_prop.barycenter - [*center, 0])
    force_torque = ForceTorque(F=F, M=M)
    new_status = forward_dynamics_solver.solve(prev_status, force_torque)

    return p, mesh, new_status


def main():
    # 1. 读取配置文件，计算固定的参数
    params = load_config('SimParams_2')

    # 齿轮参数
    rotation_speed = np.float64(params.gear.rotation_speed)
    omega = rotation_speed * 2 * np.pi / 60.0  # 转速转换为角速度 [rad/s]

    # 油液物性
    oil_mu = np.float64(params.fluid.viscosity)

    # 时间步长与总时间
    dt = np.float64(params.iteration.step_size)
    total_time = np.float64(params.iteration.total_time)

    # 侧板质量属性
    m = np.float64(params.side_plate.m)
    barycenter = np.array(eval(params.side_plate.barycenter))
    Ic = np.array(eval(params.side_plate.Ic))
    g_vec = np.array(eval(params.side_plate.g_vec))
    side_plate_mass_prop = SidePlateMassProp(
        m=m, barycenter=barycenter, Ic=Ic, g_vec=g_vec
    )

    # 2. 初始化迭代参数
    prev_status = SidePlateState(
        p=np.array([0.0, 0.0, 1e-5]),
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        w=np.zeros(3),
    )

    # 3. 迭代求解
    p = None
    mesh = None
    new_status = None

    mock_lpm = MockLPM()
    profile_path = GearProfileDir(
        gear_poly_dir="assets/gear_profile.DXF",
        relief_poly_dir="assets/relief.DXF",
    )
    mesh_generator = MeshGenerator(profile_path, omega)
    forward_dynamics_solver = ForwardDynamicsSolver(dt, side_plate_mass_prop)

    total_steps = int(round(total_time / dt))
    for step in range(total_steps + 1):
        p, mesh, new_status = one_step_pipeline(
            t=step * dt,
            dt=dt,
            omega=omega,
            oil_mu=oil_mu,
            side_plate_mass_prop=side_plate_mass_prop,
            prev_status=prev_status,
            mock_lpm=mock_lpm,
            mesh_generator=mesh_generator,
            forward_dynamics_solver=forward_dynamics_solver,
        )
        prev_status = new_status

    # 4. 可视化最终状态下油膜压力分布
    plot_pressure_distribution(
        mesh, p, fig_name="pressure_distribution", mode='save'
    )
    print(
        f"齿轮状态：p={new_status.p}, v={new_status.v}, q={new_status.q}, w={new_status.w}"
    )
    # print(f"油膜压力: {F:.3f}N, 作用点坐标: ({i:.5f}, {j:.5f})m")


if __name__ == '__main__':
    main()
