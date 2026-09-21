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
from src.solver.static_equilibrium import StaticEquilibriumSolver
from utils.calc_film_params import calc_film_params
from utils.math_tools import euler_to_quaternion
from utils.simulation_logger import SimulationLogger


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
    fluid_prop = FluidProp(mu=oil_mu)

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

    t = 0.0
    z, roll, pitch = 2.5e-06, 5e-5, 0.0
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
    F_balance = 6.5e3
    M_balance = 90.0
    non_film_force_torque = ForceTorque(
        F=np.array([0.0, 0.0, -F_balance]),
        M=np.array([M_balance, 0.0, 0.0]),
    )

    def build_fsi_solver(
        drive_mesh,
        slave_mesh,
        drive_p_lst,
        slave_p_lst,
    ):
        return SingleStepFSISolver(
            drive_mesh=drive_mesh,
            slave_mesh=slave_mesh,
            drive_p_lst=drive_p_lst,
            slave_p_lst=slave_p_lst,
            omega=omega,
            drive_reynolds_solver=ReynoldsSolver(drive_mesh, fluid_prop),
            slave_reynolds_solver=ReynoldsSolver(slave_mesh, fluid_prop),
            drive_contact_solver=ContactSolver(
                drive_mesh,
                k=1e17,
                c=1e10,
            ),
            slave_contact_solver=ContactSolver(
                slave_mesh,
                k=1e17,
                c=1e10,
            ),
            dynamics_solver=forward_dynamics_solver,
            side_plate_mass_prop=side_plate_mass_prop,
            max_sub_iter=10,
            tol=1.0,
        )

    if params.iteration.equilibrate_initial_state:
        drive_p_lst, slave_p_lst = mock_lpm.solve(t)
        drive_mesh = drive_mesh_generator.solve(t=t, p_lst=drive_p_lst)
        slave_mesh = slave_mesh_generator.solve(t=t, p_lst=slave_p_lst)
        equilibrium_solver = StaticEquilibriumSolver(
            build_fsi_solver(
                drive_mesh,
                slave_mesh,
                drive_p_lst,
                slave_p_lst,
            ),
            non_film_force_torque,
            max_evaluations=params.iteration.equilibrium_max_evaluations,
        )
        state, equilibrium_info = equilibrium_solver.solve(state)
        if not equilibrium_info.success:
            raise RuntimeError(
                "初始静力平衡求解失败: "
                f"{equilibrium_info.message}; "
                f"残差={equilibrium_info.residual}"
            )
        print(
            "初始静力平衡求解完成: "
            f"评估次数={equilibrium_info.num_evaluations}, "
            f"残差={equilibrium_info.residual}, "
            f"状态={state}"
        )

    simulation_logger = SimulationLogger(
        params.iteration.log_path,
        every_steps=params.iteration.log_every_steps,
    )
    step_index = 0
    while t < total_time:
        # 1. 集中参数法求齿腔压力
        drive_p_lst, slave_p_lst = mock_lpm.solve(t)

        # 2. 网格划分与油膜参数求解
        drive_mesh = drive_mesh_generator.solve(t=t, p_lst=drive_p_lst)
        slave_mesh = slave_mesh_generator.solve(t=t, p_lst=slave_p_lst)
        # 3.1 初始化当前相位的 Reynolds、接触和 FSI 求解器
        single_step_fsi_solver = build_fsi_solver(
            drive_mesh,
            slave_mesh,
            drive_p_lst,
            slave_p_lst,
        )

        # 3.2 单步 FSI 求解。失败重试仍处于同一物理时刻，因此复用
        # 当前网格和求解器，只减小时间步。
        step_dt = controller.get_dt()
        retry_count = 0
        while True:
            new_state, solve_info = single_step_fsi_solver.solve(
                dt=step_dt,
                state_prev=state,
                non_film_force_torque=non_film_force_torque,
            )
            if solve_info["success"]:
                break
            if step_dt <= min_dt:
                raise RuntimeError(
                    "FSI 单步求解器未收敛，且时间步长已经达到最小值，"
                    "终止仿真。"
                )
            step_dt = controller.reject_step(step_dt)
            retry_count += 1

        # 单步 FSI 求解成功，使用接受状态下两个油膜的局部闭合速度
        # 计算下一步步长。
        drive_film_param, slave_film_param = (
            single_step_fsi_solver.calc_film_params(new_state)
        )
        side_plate_vec_z = new_state.v[2]
        side_plate_acc_z = (new_state.v[2] - state.v[2]) / step_dt
        controller.compute_next_dt(
            h_cells=np.concatenate(
                [drive_film_param.h_cells, slave_film_param.h_cells]
            ),
            ht_cells=np.concatenate(
                [drive_film_param.ht_cells, slave_film_param.ht_cells]
            ),
            structural_vec=side_plate_vec_z,
            structural_acc=side_plate_acc_z,
            accepted_dt=step_dt,
        )

        state = new_state
        t += step_dt
        step_index += 1
        simulation_logger.write_step(
            step_index=step_index,
            time=t,
            dt=step_dt,
            retry_count=retry_count,
            solve_info=solve_info,
            state=state,
        )

    simulation_logger.close()

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
