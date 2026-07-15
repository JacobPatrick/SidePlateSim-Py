import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.postproc.visualize import plot_pressure_distribution
from src.config.config import load_config
from interface.types import (
    GearProfilePath,
    SidePlateState,
    SidePlateMassProp,
    FilmParam,
    FluidProp,
    Pressure,
    ForceTorque,
)
from src.solver.mock_LPM import MockLPM
from src.solver.mesh_generator import MeshGenerator
from solver.reynolds import ReynoldsSolver
from utils.calc_film_params import calc_film_params
from utils.math_tools import quaternion_multiply


def vec_converge(state1, state2, tol=1e-6):
    """
    侧板平衡判定准则
    """
    return np.allclose(state1.v, state2.v, atol=tol) and np.allclose(
        state1.w, state2.w, atol=tol
    )


def main():
    params = load_config("SimParams_2")

    # 齿轮参数
    rotation_speed = np.float64(params.gear.rotation_speed)
    omega = rotation_speed * 2 * np.pi / 60.0  # 转速转换为角速度 [rad/s]
    
    # 油液物性
    oil_mu = np.float64(params.fluid.viscosity)

    # 时间步长与总时间
    dt = float(params.iteration.step_size)
    total_time = np.float64(params.iteration.total_time)

    # 侧板质量属性
    m = np.float64(params.side_plate.m)
    barycenter = np.array(eval(params.side_plate.barycenter))
    Ic = np.array(eval(params.side_plate.Ic))
    g_vec = np.array(eval(params.side_plate.g_vec))
    side_plate_mass_prop = SidePlateMassProp(
        m=m, barycenter=barycenter, Ic=Ic, g_vec=g_vec
    )

    mock_lpm = MockLPM()
    drive_gear_profile_path = GearProfilePath(
        gear_poly_path="assets/drive_gear.DXF",
        relief_poly_path="assets/relief.DXF",
    )
    slave_gear_profile_path = GearProfilePath(
        gear_poly_path="assets/slave_gear.DXF",
        relief_poly_path="assets/relief.DXF",
    )
    drive_mesh_generator = MeshGenerator(drive_gear_profile_path, omega, "drive")
    slave_mesh_generator = MeshGenerator(slave_gear_profile_path, omega, "slave")

    # 侧板迭代平衡
    t = 0
    state = SidePlateState(
        p=np.array([0.0, 0.0, 1e-4]),
        v=np.array([0.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        w=np.zeros(3),
    )
    while t <= total_time:
        # 1. 集中参数法求齿腔压力
        drive_p_lst, slave_p_lst = mock_lpm.solve(t)

        # 2. 网格划分与油膜参数求解
        drive_mesh = drive_mesh_generator.solve(t=t, p_lst=drive_p_lst, state=state)
        slave_mesh = slave_mesh_generator.solve(t=t, p_lst=slave_p_lst, state=state)
        drive_film_param = calc_film_params(
            drive_mesh, state, drive_p_lst, omega, "drive"
        )
        slave_film_param = calc_film_params(
            slave_mesh, state, slave_p_lst, omega, "slave"
        )

        # 3. 求解油膜压力
        fluid_prop = FluidProp(mu=oil_mu)
        drive_reynolds_solver = ReynoldsSolver(drive_mesh, fluid_prop)
        slave_reynolds_solver = ReynoldsSolver(slave_mesh, fluid_prop)

        drive_pressure = drive_reynolds_solver.solve(drive_film_param)
        slave_pressure = slave_reynolds_solver.solve(slave_film_param)

        p_drive = drive_pressure.p
        p_slave = slave_pressure.p
        F_drive = drive_pressure.F
        F_slave = slave_pressure.F
        center_drive = drive_pressure.center
        center_slave = slave_pressure.center

        # TEST: 只是一个简单的测试
        P_air = 1e5 * 0.0024687143080106173
        F = np.array([0, 0, F_drive + F_slave - 2 * P_air])
        f = F[2] - m * 9.81
        M_drive = np.cross(
            side_plate_mass_prop.barycenter - [*center_drive, 0],
            [0, 0, F_drive],
        )
        M_slave = np.cross(
            side_plate_mass_prop.barycenter - [*center_slave, 0],
            [0, 0, F_slave],
        )
        M = M_drive + M_slave

        # TODO: 内循环: 寻找合适的速度使侧板受力平衡
        v_z = 0
        w_x = 0
        w_y = 0

        # 更新侧板速度（当前时刻）
        state.v = [0.0, 0.0, v_z]
        state.w = [w_x, w_y, 0.0]

        # 更新侧板位置和姿态（下一时刻）
        state.p += state.v * dt
        omega_quat_new = np.array([0.0, *state.w])
        q_dot = 0.5 * quaternion_multiply(state.q, omega_quat_new)
        state.q += q_dot * dt
        state.q /= np.linalg.norm(state.q)

        t += dt

    plot_pressure_distribution(
        drive_mesh, p_drive, fig_name="p_drive_dist", mode="save"
    )
    plot_pressure_distribution(
        slave_mesh, p_slave, fig_name="p_slave_dist", mode="save"
    )


if __name__ == "__main__":
    main()
