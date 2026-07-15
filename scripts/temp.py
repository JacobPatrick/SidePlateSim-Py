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
    FluidProp,
)
from src.solver.mock_LPM import MockLPM
from src.solver.mesh_generator import MeshGenerator
from src.solver.reynolds import ReynoldsSolver
from src.solver.balaced_v import BalancedVSolver
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
        
        fluid_prop = FluidProp(mu=oil_mu)
        drive_reynolds_solver = ReynoldsSolver(drive_mesh, fluid_prop)
        slave_reynolds_solver = ReynoldsSolver(slave_mesh, fluid_prop)

        balance_v_solver = BalancedVSolver(
            drive_mesh=drive_mesh,
            slave_mesh=slave_mesh,
            drive_p_lst=drive_p_lst,
            slave_p_lst=slave_p_lst,
            omega=omega,
            side_plate_mass_prop=side_plate_mass_prop,
            drive_reynolds_solver=drive_reynolds_solver,
            slave_reynolds_solver=slave_reynolds_solver,
            state=state,
            F_else=np.array([0.0, 0.0, -9.81 * m]),
            M_else=np.zeros(3),
        )

        current_state = balance_v_solver.solve()

        log1 = f"时间: {t * 1000:.3f}ms\n"
        log2 = f"侧板位置: z = {current_state.p[2]}\n"
        log3 = f"侧板速度: v_z = {current_state.v[2]}\n"
        log4 = f"侧板姿态: q = {current_state.q}\n"
        log5 = f"侧板角速度: w = {current_state.w}\n\n"
        with open("results/log/20260715_1.txt", "a") as f:
            f.write(
                log1 + log2 + log3 + log4 + log5
            )

        # 更新侧板位置和姿态（下一时刻）
        state.p += current_state.v * dt
        omega_quat_new = np.array([0.0, *current_state.w])
        q_dot = 0.5 * quaternion_multiply(state.q, omega_quat_new)
        state.q += q_dot * dt
        state.q /= np.linalg.norm(state.q)

        # 传递侧板速度，加速下一轮平衡求解收敛
        state.v = current_state.v
        state.w = current_state.w

        t += dt

    # plot_pressure_distribution(
    #     drive_mesh, p_drive, fig_name="p_drive_dist", mode="save"
    # )
    # plot_pressure_distribution(
    #     slave_mesh, p_slave, fig_name="p_slave_dist", mode="save"
    # )


if __name__ == "__main__":
    main()
