import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import src.solver.mesh_generator as mesh_generator_module

from interface.types import (
    FilmParam,
    FluidProp,
    ForceTorque,
    GearProfilePath,
    SidePlateMassProp,
    SidePlateState,
)
from src.controller.adaptive_time_step import AdaptiveTimeStepController
from src.solver.contact import ContactSolver
from src.solver.forward_dynamics import ForwardDynamicsSolver
from src.solver.mesh_generator import MeshGenerator
from src.solver.mock_LPM import MockLPM
from src.solver.reynolds import ReynoldsSolver
from src.solver.strong_FSI_coupling import (
    FSIConvergenceTolerances,
    SingleStepFSISolver,
    _calc_res_vec,
)
from src.solver.static_equilibrium import StaticEquilibriumSolver
from utils.math_tools import euler_to_quaternion, quaternion_to_euler
from utils.calc_film_params import calc_film_params
from utils.simulation_logger import SimulationLogger


class ForwardDynamicsTests(unittest.TestCase):
    def setUp(self):
        mass = SidePlateMassProp(
            m=1.0,
            barycenter=np.zeros(3),
            Ic=np.eye(3),
        )
        self.solver = ForwardDynamicsSolver(mass)
        self.state_prev = SidePlateState()

    def test_force_is_evaluated_using_predicted_orientation(self):
        state_eval = SidePlateState(
            q=euler_to_quaternion(np.pi / 3.0, 0.0, 0.0)
        )
        result = self.solver.solve(
            1.0,
            self.state_prev,
            ForceTorque(F=np.array([0.0, 0.0, 1.0])),
            state_eval=state_eval,
        )

        self.assertAlmostEqual(result.v[2], 0.5)
        self.assertAlmostEqual(result.p[2], 0.5)

    def test_candidate_is_integrated_from_previous_state(self):
        state_eval = SidePlateState(
            p=np.array([0.0, 0.0, 100.0]),
            v=np.array([0.0, 0.0, 3.0]),
            w=np.array([2.0, 0.0, 0.0]),
        )
        result = self.solver.solve(
            0.1,
            self.state_prev,
            ForceTorque(),
            state_eval=state_eval,
        )

        self.assertAlmostEqual(result.p[2], 0.0)
        self.assertAlmostEqual(result.v[2], 0.0)
        self.assertAlmostEqual(result.w[0], 0.0)
        self.assertGreater(result.q[1], 0.0)


class ResidualTests(unittest.TestCase):
    def test_quaternion_sign_does_not_change_residual(self):
        state = SidePlateState(q=euler_to_quaternion(1e-5, 0.0, 0.0))
        negated = SidePlateState(q=-state.q)

        residual = _calc_res_vec(state, negated)

        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_each_state_group_uses_its_own_tolerance(self):
        tolerances = FSIConvergenceTolerances(
            position=1e-8,
            velocity=1e-5,
            angle=1e-6,
            angular_velocity=1e-3,
            relative=0.0,
        )
        calc = SidePlateState(
            p=np.array([0.0, 0.0, 1e-8]),
            v=np.array([0.0, 0.0, 1e-5]),
            q=euler_to_quaternion(1e-6, 0.0, 0.0),
            w=np.array([1e-3, 0.0, 0.0]),
        )

        residual = _calc_res_vec(calc, SidePlateState(), tolerances)

        np.testing.assert_allclose(
            [residual[2], residual[5], residual[6], residual[9]],
            1.0,
            rtol=1e-6,
        )


class ContactTests(unittest.TestCase):
    def test_separating_contact_never_returns_tension(self):
        mesh = SimpleNamespace(
            points=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            elements=[(0, 1, 2)],
        )
        solver = ContactSolver(mesh, k=1.0, c=10.0)
        film = FilmParam(
            h_cells=np.array([-1e-6]),
            h_grad=(0.0, 0.0),
            U_cells=np.zeros((1, 2)),
            ht_cells=np.array([1.0]),
            bc_lst=[0.0],
        )

        result = solver.solve(film)

        self.assertEqual(result.F, 0.0)
        np.testing.assert_array_equal(result.p, np.zeros(1))
        np.testing.assert_array_equal(result.center, np.zeros(2))


class AdaptiveTimeStepTests(unittest.TestCase):
    def test_rejected_step_updates_controller_state(self):
        controller = AdaptiveTimeStepController(
            dt_init=5e-7,
            dt_min=1e-7,
            dt_max=1e-5,
        )

        retry_dt = controller.reject_step(5e-7)

        self.assertEqual(retry_dt, 2.5e-7)
        self.assertEqual(controller.get_dt(), retry_dt)

    def test_receding_film_does_not_limit_time_step(self):
        controller = AdaptiveTimeStepController(
            dt_init=1e-6,
            dt_min=1e-8,
            dt_max=1e-4,
            smoothing_alpha=0.0,
        )

        next_dt = controller.compute_next_dt(
            h_cells=np.array([1e-6]),
            ht_cells=np.array([1.0]),
            structural_vec=1.0,
            structural_acc=1.0,
            accepted_dt=1e-6,
        )

        self.assertEqual(next_dt, 2e-6)

    def test_closing_film_applies_immediate_safety_limit(self):
        controller = AdaptiveTimeStepController(
            dt_init=1e-4,
            dt_min=1e-8,
            dt_max=1e-3,
            eta=0.5,
            smoothing_alpha=0.9,
        )

        next_dt = controller.compute_next_dt(
            h_cells=np.array([1e-6]),
            ht_cells=np.array([-1.0]),
            structural_vec=-1.0,
            structural_acc=0.0,
            accepted_dt=1e-4,
        )

        self.assertAlmostEqual(next_dt, 5e-7)


class MeshGeneratorTests(unittest.TestCase):
    def test_dxf_profile_is_loaded_only_during_initialization(self):
        path = GearProfilePath(
            gear_poly_path="assets/drive_gear.DXF",
            relief_poly_path="assets/relief.DXF",
        )
        pressures, _ = MockLPM().solve(0.0)

        with patch(
            "src.solver.mesh_generator.load_gear_profile_from_dxf",
            wraps=mesh_generator_module.load_gear_profile_from_dxf,
        ) as loader:
            generator = MeshGenerator(path, omega=1.0, gear_type="drive")
            generator.solve(t=0.0, p_lst=pressures)
            generator.solve(t=1e-3, p_lst=pressures)

        self.assertEqual(loader.call_count, 1)


class ReynoldsSolverTests(unittest.TestCase):
    def test_vectorized_assembly_preserves_reference_solution(self):
        omega = 2000.0 * 2.0 * np.pi / 60.0
        path = GearProfilePath(
            gear_poly_path="assets/drive_gear.DXF",
            relief_poly_path="assets/relief.DXF",
        )
        pressures, _ = MockLPM().solve(0.0)
        mesh = MeshGenerator(path, omega, "drive").solve(0.0, pressures)
        state = SidePlateState(
            p=np.array([0.0, 0.0, 2.5e-6]),
            q=euler_to_quaternion(5e-5, 0.0, 0.0),
        )
        film = calc_film_params(
            mesh,
            state,
            pressures,
            omega,
            "drive",
        )

        result = ReynoldsSolver(mesh, FluidProp(mu=5e-4)).solve(film)

        self.assertAlmostEqual(result.F, 3011.352098270156, places=8)
        np.testing.assert_allclose(
            result.center,
            np.array([0.01923931, -0.01479037]),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            result.p[[0, 1, 2, 10, 100, 500, 1000, 1500]],
            np.array(
                [
                    0.0,
                    5.01793392e6,
                    5.37063518e6,
                    5.29866892e6,
                    4.34712397e5,
                    3.20223774e5,
                    0.0,
                    3.59520427e3,
                ]
            ),
            rtol=1e-8,
            atol=1e-6,
        )


class StaticEquilibriumTests(unittest.TestCase):
    def test_initial_guess_converges_to_force_and_moment_balance(self):
        omega = 2000.0 * 2.0 * np.pi / 60.0
        drive_pressures, slave_pressures = MockLPM().solve(0.0)
        drive_mesh = MeshGenerator(
            GearProfilePath(
                "assets/drive_gear.DXF",
                "assets/relief.DXF",
            ),
            omega,
            "drive",
        ).solve(0.0, drive_pressures)
        slave_mesh = MeshGenerator(
            GearProfilePath(
                "assets/slave_gear.DXF",
                "assets/relief.DXF",
            ),
            omega,
            "slave",
        ).solve(0.0, slave_pressures)
        mass = SidePlateMassProp(
            m=1.695,
            barycenter=np.array([0.0, 0.0, 0.025]),
            Ic=np.diag([1.657e-3, 5.443e-3, 4.493e-3]),
        )
        fluid = FluidProp(mu=5e-4)
        fsi_solver = SingleStepFSISolver(
            drive_mesh=drive_mesh,
            slave_mesh=slave_mesh,
            drive_p_lst=drive_pressures,
            slave_p_lst=slave_pressures,
            omega=omega,
            drive_reynolds_solver=ReynoldsSolver(drive_mesh, fluid),
            slave_reynolds_solver=ReynoldsSolver(slave_mesh, fluid),
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
            dynamics_solver=ForwardDynamicsSolver(mass),
            side_plate_mass_prop=mass,
        )
        non_film_load = ForceTorque(
            F=np.array([0.0, 0.0, -6500.0]),
            M=np.array([90.0, 0.0, 0.0]),
        )
        initial_state = SidePlateState(
            p=np.array([0.0, 0.0, 2.5e-6]),
            q=euler_to_quaternion(5e-5, 0.0, 0.0),
        )

        state, info = StaticEquilibriumSolver(
            fsi_solver,
            non_film_load,
        ).solve(initial_state)

        self.assertTrue(info.success, info.message)
        self.assertLess(np.linalg.norm(info.residual, ord=np.inf), 1e-8)
        self.assertAlmostEqual(state.p[2], 2.40589847e-6, places=13)
        roll, pitch, _ = quaternion_to_euler(*state.q)
        self.assertAlmostEqual(roll, 5.34763050e-5, places=12)
        self.assertAlmostEqual(pitch, -2.26373677e-6, places=12)


class SimulationLoggerTests(unittest.TestCase):
    def test_logger_records_first_and_configured_steps(self):
        state = SidePlateState()
        solve_info = {
            "num_iter": 3,
            "res_norm": 0.1,
            "F_drive": 1.0,
            "F_slave": 2.0,
            "M": np.zeros(3),
            "F_side_plate": 3.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/simulation.txt"
            logger = SimulationLogger(path, every_steps=100)
            for step in (1, 2, 99, 100):
                logger.write_step(
                    step,
                    time=step * 1e-6,
                    dt=1e-6,
                    retry_count=0,
                    solve_info=solve_info,
                    state=state,
                )
            logger.close()
            with open(path, encoding="utf-8") as log_file:
                contents = log_file.read()

        self.assertEqual(contents.count("时间:"), 2)
        self.assertIn("0.001000ms", contents)
        self.assertIn("0.100000ms", contents)


if __name__ == "__main__":
    unittest.main()
