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
    oil_mu,
    side_plate_mass_prop: SidePlateMassProp,
    prev_status: SidePlateState,
    mock_lpm: MockLPM,
    drive_mesh_generator: MeshGenerator,
    slave_mesh_generator: MeshGenerator,
    forward_dynamics_solver: ForwardDynamicsSolver,
):
    # 1. 集中参数法求齿腔压力
    drive_p_lst, slave_p_lst = mock_lpm.solve(t)

    # 2. 网格划分与油膜参数求解
    drive_mesh, drive_film_param = drive_mesh_generator.solve(t, drive_p_lst, prev_status)
    slave_mesh, slave_film_param = slave_mesh_generator.solve(t, slave_p_lst, prev_status)

    # 3. 求解油膜压力
    fluid_prop = FluidProp(mu=oil_mu)
    drive_reynolds_solver = ReynoldsSolver(drive_mesh, fluid_prop, drive_film_param)
    slave_reynolds_solver = ReynoldsSolver(slave_mesh, fluid_prop, slave_film_param)

    drive_pressure = drive_reynolds_solver.solve()
    slave_pressure = slave_reynolds_solver.solve()

    p_drive = drive_pressure.p
    p_slave = slave_pressure.p
    F_drive = drive_pressure.F
    F_slave = slave_pressure.F
    center_drive = drive_film_param.center
    center_slave = slave_film_param.center

    # 4. 求解正向动力学
    F = np.array([0, 0, F_drive + F_slave]) # TODO: 加入齿腔油压和背压
    M_drive = np.cross(F_drive, side_plate_mass_prop.barycenter - [*center_drive, 0])
    M_slave = np.cross(F_slave, side_plate_mass_prop.barycenter - [*center_slave, 0])
    M = M_drive + M_slave   # TODO: 加入齿腔油压产生的力矩
    force_torque = ForceTorque(F=F, M=M)
    new_status = forward_dynamics_solver.solve(prev_status, force_torque)

    return p_drive, p_slave , drive_mesh, slave_mesh, new_status


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
    p_drive = None
    p_slave = None
    drive_mesh = None
    slave_mesh = None
    new_status = None

    mock_lpm = MockLPM()
    profile_path = GearProfileDir(
        gear_poly_dir="assets/gear_profile.DXF",
        relief_poly_dir="assets/relief.DXF",
    )
    drive_mesh_generator = MeshGenerator(profile_path, omega, "drive")
    slave_mesh_generator = MeshGenerator(profile_path, omega, "slave")
    forward_dynamics_solver = ForwardDynamicsSolver(dt, side_plate_mass_prop)

    total_steps = int(round(total_time / dt))
    for step in range(total_steps + 1):
        p_drive, p_slave, drive_mesh, slave_mesh, new_status = one_step_pipeline(
            t=step * dt,
            oil_mu=oil_mu,
            side_plate_mass_prop=side_plate_mass_prop,
            prev_status=prev_status,
            mock_lpm=mock_lpm,
            drive_mesh_generator=drive_mesh_generator,
            slave_mesh_generator=slave_mesh_generator,
            forward_dynamics_solver=forward_dynamics_solver,
        )
        prev_status = new_status

    # 4. 可视化最终状态下油膜压力分布
    plot_pressure_distribution(drive_mesh, p_drive, fig_name="drive_p_dist", mode='save')
    plot_pressure_distribution(slave_mesh, p_slave, fig_name="slave_p_dist", mode='save')
    # TODO: 先分开画，之后再写合一起画
    print(f"齿轮状态：p={new_status.p}, v={new_status.v}, q={new_status.q}, w={new_status.w}")


if __name__ == '__main__':
    main()
