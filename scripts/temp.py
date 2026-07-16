import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.postproc.visualize import plot_pressure_distribution
from src.config.config import load_config
from interface.types import (
    GearProfilePath,
    SidePlateState,
    FilmParam,
    FluidProp,
    Pressure,
    SidePlateMassProp,
    ForceTorque,
)
from src.controller.adaptive_time_step import AdaptiveTimeStepController
from src.solver.mock_LPM import MockLPM
from src.solver.mesh_generator import MeshGenerator
from src.solver.reynolds import ReynoldsSolver
from src.solver.forward_dynamics import ForwardDynamicsSolver
from src.solver.strong_FSI_coupling import SingleStepFSISolver
from utils.calc_film_params import calc_film_params


def main():
    # 1. 读取配置文件，计算固定的参数
    params = load_config('SimParams_2')

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

    # 2. 初始化迭代控制器、迭代参数与求解器
    controller = AdaptiveTimeStepController()
    dt_state = {"value": dt}

    def on_dt_update(new_dt):
        dt_state["value"] = new_dt

    t = 0.0
    state = SidePlateState(
        p=np.array([0.0, 0.0, 1e-4]),
        v=np.array([0.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        w=np.zeros(3),
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

    while t < total_time:
        with open("results/log/20260714_6.txt", "a") as f:
            f.write(f"时间: {t*1000:.3f}ms\n")
            
        dt_state["value"] = controller.get_dt()
        # 1. 集中参数法求齿腔压力
        drive_p_lst, slave_p_lst = mock_lpm.solve(t)

        # 2. 网格划分与油膜参数求解
        drive_mesh = drive_mesh_generator.solve(t=t, p_lst=drive_p_lst, state=state)
        slave_mesh = slave_mesh_generator.solve(t=t, p_lst=slave_p_lst, state=state)
        drive_film_param = calc_film_params(drive_mesh, state, drive_p_lst, omega, "drive")
        slave_film_param = calc_film_params(slave_mesh, state, slave_p_lst, omega, "slave")

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

        # 4. 求解正向动力学
        P_air = 1e5 * 0.0024687143080106173
        F = np.array([0, 0, F_drive + F_slave - 2 * P_air])  # TODO: 加入齿腔油压和背压
        f = F[2] - m * 9.81
        M_drive = np.cross(
            side_plate_mass_prop.barycenter - [*center_drive, 0],
            [0, 0, F_drive],
        )
        M_slave = np.cross(
            side_plate_mass_prop.barycenter - [*center_slave, 0],
            [0, 0, F_slave],
        )
        M = M_drive + M_slave  # TODO: 加入齿腔油压产生的力矩

        single_step_fsi_solver = SingleStepFSISolver(
            drive_mesh=drive_mesh,
            slave_mesh=slave_mesh,
            drive_p_lst=drive_p_lst,
            slave_p_lst=slave_p_lst,
            omega=omega,
            drive_reynolds_solver=drive_reynolds_solver,
            slave_reynolds_solver=slave_reynolds_solver,
            dynamics_solver=forward_dynamics_solver,
            side_plate_mass_prop=side_plate_mass_prop,
            max_sub_iter=15,
            tol=1e-4,
        )

        new_state = single_step_fsi_solver.solve(
            dt=dt,
            state_prev=state,
            force_torque=ForceTorque(F=F, M=M),
            on_retry_callback=on_dt_update
        )
        dt = dt_state["value"]
        
        side_plate_vec_z = new_state.v[2]
        side_plate_acc_z = (new_state.v[2] - state.v[2]) / dt
        controller.compute_next_dt(
            h_cells=drive_film_param.h_cells,
            ht_cells=drive_film_param.ht_cells,
            structural_vec=side_plate_vec_z,
            structural_acc=side_plate_acc_z,
        )
        
        state = new_state
        t += dt

    # 4. 可视化最终状态下油膜压力分布
    plot_pressure_distribution(
        drive_mesh, p_drive, fig_name="drive_p_dist", mode='save'
    )
    plot_pressure_distribution(
        slave_mesh, p_slave, fig_name="slave_p_dist", mode='save'
    )
    # TODO: 先分开画，之后再写合一起画


if __name__ == '__main__':
    main()
