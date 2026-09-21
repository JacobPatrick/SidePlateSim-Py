import unittest
from types import SimpleNamespace

import numpy as np

from interface.types import (
    FilmParam,
    ForceTorque,
    SidePlateMassProp,
    SidePlateState,
)
from src.controller.adaptive_time_step import AdaptiveTimeStepController
from src.solver.contact import ContactSolver
from src.solver.forward_dynamics import ForwardDynamicsSolver
from src.solver.strong_FSI_coupling import (
    FSIConvergenceTolerances,
    _calc_res_vec,
)
from utils.math_tools import euler_to_quaternion


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


if __name__ == "__main__":
    unittest.main()
