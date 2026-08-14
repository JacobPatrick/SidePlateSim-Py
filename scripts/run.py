import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.postproc.visualize import plot_gear_pressure_distribution
from src.config.config import load_config
from interface.types import (
    GearProfilePath,
    SidePlateState,
    FluidProp,
    SidePlateMassProp,
    ForceTorque,
)
from src.controller.adaptive_time_step import AdaptiveTimeStepController
from src.solver.mock_LPM import MockLPM
from src.solver.mesh_generator import MeshGenerator
from src.solver.reynolds import ReynoldsSolver
from src.solver.contact import ContactSolver
from src.solver.forward_dynamics import ForwardDynamicsSolver
from src.solver.strong_FSI_coupling import SingleStepFSISolver
from utils.calc_film_params import calc_film_params
from utils.math_tools import euler_to_quaternion


def smooth_loading(t, load):
    ratio = np.clip(t / 1e-2, 0, 1)
    return ratio * load


def main():
    # 1. 读取配置文件，计算固定的参数
    params = load_config('SimParams_2')

    # 齿轮参数
    rotation_speed = np.float64(params.gear.rotation_speed)
    omega = rotation_speed * 2 * np.pi / 60.0  # 转速转换为角速度 [rad/s]

    # 油液物性
    oil_mu = np.float64(params.fluid.viscosity)

    # 时间步长与总时间
    base_dt = float(params.iteration.base_step_size)
    max_dt = float(params.iteration.max_step_size)
    min_dt = float(params.iteration.min_step_size)
    total_time = np.float64(params.iteration.total_time)

    # 侧板质量属性
    m = np.float64(params.side_plate.m)
    barycenter = np.array(eval(params.side_plate.barycenter))
    Ic = np.array(eval(params.side_plate.Ic))
    g_vec = np.array(eval(params.side_plate.g_vec))
    side_plate_mass_prop = SidePlateMassProp(
        m=m, barycenter=barycenter, Ic=Ic, g_vec=g_vec
    )

    # 2. 初始化迭代控制器、迭代参数与求解器
    controller = AdaptiveTimeStepController(
        dt_init=base_dt, dt_max=max_dt, dt_min=min_dt
    )
    dt_state = {"value": base_dt}

    t = 0.0
    z, roll, pitch = 3e-05, -2.575e-5, 0.0
    q = euler_to_quaternion(roll, pitch, 0.0)
    state = SidePlateState(
        p=np.array([0.0, 0.0, z]),
        v=np.array([0.0, 0.0, 0.0]),
        q=np.array(q),
        w=np.array([0.0, 0.0, 0.0]),
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
    drive_mesh_generator = MeshGenerator(
        drive_gear_profile_path, omega, "drive"
    )
    slave_mesh_generator = MeshGenerator(
        slave_gear_profile_path, omega, "slave"
    )
    forward_dynamics_solver = ForwardDynamicsSolver(side_plate_mass_prop)

    # 3. 迭代求解
    p_drive = None
    p_slave = None
    drive_mesh = None
    slave_mesh = None
    new_state = None
    dt_state["value"] = base_dt

    F_balance = 10600.0

    while t < total_time:
        # 1. 集中参数法求齿腔压力
        drive_p_lst, slave_p_lst = mock_lpm.solve(t)

        # 2. 网格划分与油膜参数求解
        drive_mesh = drive_mesh_generator.solve(t=t, p_lst=drive_p_lst)
        slave_mesh = slave_mesh_generator.solve(t=t, p_lst=slave_p_lst)
        drive_film_param = calc_film_params(
            drive_mesh, state, drive_p_lst, omega, "drive"
        )

        # 3.1 初始化 Reynolds 求解器、接触求解器和 FSI 求解器
        fluid_prop = FluidProp(mu=oil_mu)
        drive_reynolds_solver = ReynoldsSolver(drive_mesh, fluid_prop)
        slave_reynolds_solver = ReynoldsSolver(slave_mesh, fluid_prop)

        drive_contact_solver = ContactSolver(drive_mesh, k=1e16, c=1e9)
        slave_contact_solver = ContactSolver(slave_mesh, k=1e16, c=1e9)

        F = np.array([0, 0, -F_balance])
        M = np.array([0, 0, 0])

        single_step_fsi_solver = SingleStepFSISolver(
            drive_mesh=drive_mesh,
            slave_mesh=slave_mesh,
            drive_p_lst=drive_p_lst,
            slave_p_lst=slave_p_lst,
            omega=omega,
            drive_reynolds_solver=drive_reynolds_solver,
            slave_reynolds_solver=slave_reynolds_solver,
            drive_contact_solver=drive_contact_solver,
            slave_contact_solver=slave_contact_solver,
            dynamics_solver=forward_dynamics_solver,
            side_plate_mass_prop=side_plate_mass_prop,
            max_sub_iter=10,
            tol=1e-2,
        )

        # 3.2 单步 FSI 求解
        new_state, solve_info = single_step_fsi_solver.solve(
            dt=dt_state["value"],
            state_prev=state,
            non_film_force_torque=ForceTorque(F=F, M=M),
        )

        if solve_info["success"]:
            # 单步 FSI 求解成功，推进时间
            side_plate_vec_z = new_state.v[2]
            side_plate_acc_z = (new_state.v[2] - state.v[2]) / dt_state["value"]
            controller.compute_next_dt(
                h_cells=drive_film_param.h_cells,
                ht_cells=drive_film_param.ht_cells,
                structural_vec=side_plate_vec_z,
                structural_acc=side_plate_acc_z,
            )

            state = new_state
            t += dt_state["value"]
            dt_state["value"] = controller.get_dt()
            with open("results/log/20260814_1.txt", "a") as f:
                f.write(f"时间: {t*1000:.4f}ms\n")
                f.write(
                    f"迭代次数: {solve_info['num_iter']}, 残差: {solve_info['res_norm']:.3e}\n"
                )
                f.write(
                    f"侧板受力: 主动轮 F={solve_info['F_drive']:.2f}N, 从动轮 F={solve_info['F_slave']:.2f}N\n"
                )
                f.write(f"侧板受合力矩: M={solve_info['M']}N·m\n")
                f.write(f"侧板受力: F={solve_info['F_side_plate']:.2f}N\n")
                f.write(
                    f"侧板状态: p={state.p}, v={state.v}, q={state.q}, w={state.w}\n\n"
                )

        elif not solve_info["success"] and dt_state["value"] >= 2 * min_dt:
            # 单步 FSI 求解失败，尝试减小 dt 并重做
            dt_state["value"] *= 0.5

        else:
            # 单步 FSI 求解失败，且 dt 已经小于最小值，终止仿真
            raise RuntimeError(
                f"FSI 单步求解器未收敛，且时间步长已经小于最小值，终止仿真。"
            )

    # 4. 可视化最终状态下油膜压力分布
    drive_p_lst, slave_p_lst = mock_lpm.solve(t)

    drive_mesh = drive_mesh_generator.solve(t=t, p_lst=drive_p_lst)
    slave_mesh = slave_mesh_generator.solve(t=t, p_lst=slave_p_lst)
    drive_film_param = calc_film_params(
        drive_mesh, state, drive_p_lst, omega, "drive"
    )
    slave_film_param = calc_film_params(
        slave_mesh, state, slave_p_lst, omega, "slave"
    )

    drive_reynolds_solver = ReynoldsSolver(drive_mesh, fluid_prop)
    slave_reynolds_solver = ReynoldsSolver(slave_mesh, fluid_prop)
    drive_pressure = drive_reynolds_solver.solve(drive_film_param)
    slave_pressure = slave_reynolds_solver.solve(slave_film_param)
    p_drive = drive_pressure.p
    p_slave = slave_pressure.p

    plot_gear_pressure_distribution(
        drive_mesh, p_drive, slave_mesh, p_slave, fig_name="p_dist", mode='save'
    )


if __name__ == '__main__':
    main()
