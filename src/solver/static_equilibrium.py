from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from interface.types import ForceTorque, SidePlateState
from src.solver.strong_FSI_coupling import SingleStepFSISolver
from utils.math_tools import euler_to_quaternion, quaternion_to_euler


@dataclass(frozen=True)
class EquilibriumSolveInfo:
    success: bool
    num_evaluations: int
    residual: np.ndarray
    message: str


class StaticEquilibriumSolver:
    """求解给定齿轮相位下侧板的三自由度静力平衡。"""

    def __init__(
        self,
        load_solver: SingleStepFSISolver,
        non_film_force_torque: ForceTorque,
        z_bounds=(1e-7, 1e-4),
        angle_bounds=(-1e-3, 1e-3),
        position_scale=1e-6,
        angle_scale=1e-5,
        residual_tolerance=1e-6,
        max_evaluations=50,
    ):
        self.load_solver = load_solver
        self.non_film_force_torque = non_film_force_torque
        self.position_scale = position_scale
        self.angle_scale = angle_scale
        self.residual_tolerance = residual_tolerance
        self.max_evaluations = max_evaluations
        self.lower_bounds = np.array(
            [
                z_bounds[0] / position_scale,
                angle_bounds[0] / angle_scale,
                angle_bounds[0] / angle_scale,
            ]
        )
        self.upper_bounds = np.array(
            [
                z_bounds[1] / position_scale,
                angle_bounds[1] / angle_scale,
                angle_bounds[1] / angle_scale,
            ]
        )
        self.force_scale = max(
            abs(non_film_force_torque.F[2]),
            1.0,
        )
        self.moment_scale = max(
            np.linalg.norm(non_film_force_torque.M[:2]),
            1.0,
        )

    def _state_from_scaled_coordinates(self, coordinates):
        z = coordinates[0] * self.position_scale
        roll = coordinates[1] * self.angle_scale
        pitch = coordinates[2] * self.angle_scale
        return SidePlateState(
            p=np.array([0.0, 0.0, z]),
            v=np.zeros(3),
            q=euler_to_quaternion(roll, pitch, 0.0),
            w=np.zeros(3),
        )

    def _residual(self, coordinates):
        state = self._state_from_scaled_coordinates(coordinates)
        force_torque, _, _ = self.load_solver.evaluate_loads(
            state,
            self.non_film_force_torque,
        )
        return np.array(
            [
                force_torque.F[2] / self.force_scale,
                force_torque.M[0] / self.moment_scale,
                force_torque.M[1] / self.moment_scale,
            ]
        )

    def solve(self, initial_state: SidePlateState):
        roll, pitch, _ = quaternion_to_euler(*initial_state.q)
        initial_coordinates = np.array(
            [
                initial_state.p[2] / self.position_scale,
                roll / self.angle_scale,
                pitch / self.angle_scale,
            ]
        )
        initial_coordinates = np.clip(
            initial_coordinates,
            self.lower_bounds,
            self.upper_bounds,
        )
        result = least_squares(
            self._residual,
            x0=initial_coordinates,
            bounds=(self.lower_bounds, self.upper_bounds),
            x_scale="jac",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=self.max_evaluations,
        )
        state = self._state_from_scaled_coordinates(result.x)
        residual = self._residual(result.x)
        success = bool(
            result.success
            and np.linalg.norm(residual, ord=np.inf)
            <= self.residual_tolerance
        )
        info = EquilibriumSolveInfo(
            success=success,
            num_evaluations=result.nfev,
            residual=residual,
            message=result.message,
        )
        return state, info
