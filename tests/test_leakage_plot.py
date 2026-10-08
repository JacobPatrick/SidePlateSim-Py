"""给定侧板状态下的端面边界泄漏可视化，直接运行，不在导入时求解。

从仓库根目录运行，例如：
    python tests/test_leakage_plot.py
    python tests/test_leakage_plot.py --z 5e-6 --tilt 2e-5 0 \
        --vz=-1e-3 --wx 0.1 --wy -0.2 --show
    python tests/test_leakage_plot.py --quaternion 1 0 0 0 --quantity flow

z: m；倾角: rad；vz: m/s；wx、wy: rad/s；四元数顺序: w x y z。
使用 MockLPM 的边界压力，绘制 calc_leakage 的总流量（含 Couette）。
默认颜色表示单位边长流量 m²/s；--quantity flow 表示每条边流量 m³/s。
流入油膜为正。位姿/速度含义遵循 FilmParameterCalculator；当前模型只
使用 z、roll、pitch、vz、wx、wy，未引入额外的侧板平面平移运动。
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interface.types import FluidProp, GearProfilePath, SidePlateState
from src.config.config import SimulationConfig
from src.solver.mesh_generator import MeshGenerator
from src.solver.mock_LPM import MockLPM
from src.solver.reynolds import ReynoldsSolver
from utils.calc_film_params import calc_film_params
from utils.math_tools import euler_to_quaternion


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--time", type=float, default=0.0, help="齿轮相位时间 [s]"
    )
    parser.add_argument(
        "--z", type=float, default=2.5e-6, help="侧板法向位置 [m]"
    )
    attitude = parser.add_mutually_exclusive_group()
    attitude.add_argument(
        "--tilt",
        type=float,
        nargs=2,
        metavar=("ROLL", "PITCH"),
        help="侧板倾角 [rad]，默认 5e-5 0，yaw=0",
    )
    attitude.add_argument(
        "--quaternion",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="侧板姿态四元数，自动归一化",
    )
    parser.add_argument(
        "--vz", type=float, default=0.0, help="法向速度 [m/s]，正值为远离"
    )
    parser.add_argument(
        "--wx", type=float, default=0.0, help="绕 x 轴角速度 [rad/s]"
    )
    parser.add_argument(
        "--wy", type=float, default=0.0, help="绕 y 轴角速度 [rad/s]"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/SimParams_2.yml",
        help="读取齿轮转速和油液黏度的配置文件",
    )
    parser.add_argument("--rpm", type=float, help="覆盖配置中的齿轮转速 [rpm]")
    parser.add_argument("--mu", type=float, help="覆盖配置中的动力黏度 [Pa·s]")
    parser.add_argument(
        "--quantity",
        choices=("density", "flow"),
        default="density",
        help="density: 单位边长流量；flow: 每条边积分流量",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/figures",
        help="图片输出目录，每次运行生成带时间戳的 PNG",
    )
    parser.add_argument(
        "--show", action="store_true", help="保存图片后打开交互窗口"
    )
    args = parser.parse_args(argv)
    values = [args.time, args.z, args.vz, args.wx, args.wy]
    values += args.quaternion or args.tilt or [5e-5, 0.0]
    values += [value for value in (args.rpm, args.mu) if value is not None]
    if not np.all(np.isfinite(values)):
        parser.error("位姿、速度、时间及物性参数必须是有限数值")
    if args.quaternion is not None and np.linalg.norm(args.quaternion) < 1e-12:
        parser.error("四元数不能为零向量")
    return args


def solve_gear(gear, state, t, omega, mu, boundary_pressures):
    profile = GearProfilePath(
        gear_poly_path=str(PROJECT_ROOT / f"assets/{gear}_gear.DXF"),
        relief_poly_path=str(PROJECT_ROOT / "assets/relief.DXF"),
    )
    mesh = MeshGenerator(profile, omega, gear).solve(t, boundary_pressures)
    film = calc_film_params(mesh, state, boundary_pressures, omega, gear)
    if np.all(film.h_cells <= 0.0):
        raise ValueError(f"{gear}: 所有油膜单元膜厚均非正，请调整侧板位姿")
    solver = ReynoldsSolver(mesh, FluidProp(mu=mu))
    pressure = solver.solve(film)
    flow = np.asarray(solver.calc_leakage(pressure.p, film))
    if not np.all(np.isfinite(flow)):
        raise ValueError(f"{gear}: 泄漏求解产生了非有限值，请检查输入状态")
    return {
        "gear": gear,
        "mesh": mesh,
        "film": film,
        "pressure": pressure,
        "flow": flow,
    }


def plot_results(results, args):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    from src.postproc.visualize import plot_leak_rate

    for result in results:
        figure_name = (
            f"leakage_{result['gear']}_{args.quantity}_{datetime.now():%f}"
        )
        # 传入 m³/s 积分流量，边长归一化由已有绘图函数处理。
        plot_leak_rate(
            result["mesh"],
            result["flow"],
            figure_name,
            mode="save",
            per_unit_length=args.quantity == "density",
            output_dir=args.output_dir,
        )
        if args.show:
            plot_leak_rate(
                result["mesh"],
                result["flow"],
                figure_name,
                mode="show",
                per_unit_length=args.quantity == "density",
            )
    print(f"\nSaved figures: {args.output_dir.resolve()}")
    return args.output_dir


def main(argv=None):
    args = parse_args(argv)
    config = SimulationConfig.load_from_yaml(str(args.config))
    rpm = float(config.gear.rotation_speed if args.rpm is None else args.rpm)
    mu = float(config.fluid.viscosity if args.mu is None else args.mu)
    if not np.isfinite(rpm) or not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("转速必须有限，动力黏度必须为有限正值")
    quaternion = (
        np.asarray(args.quaternion, dtype=float)
        if args.quaternion is not None
        else euler_to_quaternion(*(args.tilt or [5e-5, 0.0]), 0.0)
    )
    quaternion = quaternion / np.linalg.norm(quaternion)
    state = SidePlateState(
        p=np.array([0.0, 0.0, args.z]),
        v=np.array([0.0, 0.0, args.vz]),
        q=quaternion,
        w=np.array([args.wx, args.wy, 0.0]),
    )
    omega = rpm * 2.0 * np.pi / 60.0
    print(f"MockLPM boundary pressures; rpm={rpm:g}, mu={mu:g} Pa s")
    print(
        "Positive flow enters the film; totals are net flux over fixed spatial boundaries."
    )
    print(
        "The current solver clips negative pressure; these totals are not a cavitation mass-balance check."
    )
    results = [
        solve_gear(gear, state, args.time, omega, mu, pressures)
        for gear, pressures in zip(
            ("drive", "slave"), MockLPM().solve(args.time)
        )
    ]
    return plot_results(results, args)


if __name__ == "__main__":
    main()
