"""边界泄漏的方向、解析解、离散守恒及梯度重构回归测试。"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from interface.types import FilmParam, FluidProp, SidePlateState
from src.solver.reynolds import ReynoldsSolver
from utils.calc_film_params import FilmParameterCalculator


def equilateral_mesh():
    height = np.sqrt(3.0) / 2.0
    # 不提供 neighbors；重构应只依赖已缓存的真实面拓扑。
    return SimpleNamespace(
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, height], [1.5, height]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]]),
        facets=np.array([[0, 1], [1, 3], [3, 2], [2, 0]]),
        facet_markers=np.arange(1, 5),
    )


def skew_mesh():
    return SimpleNamespace(
        points=np.array(
            [[0.0, 0.0], [2.0, 0.0], [1.8, 1.1], [-0.2, 0.8], [0.6, 0.35]]
        ),
        elements=np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
        facets=np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
        facet_markers=np.arange(1, 5),
    )


def film_for(solver, pressures=None, h=1e-4):
    return FilmParam(
        h_cells=np.full(solver.n_cells, h),
        h_grad=(0.0, 0.0),
        U_cells=np.zeros((solver.n_cells, 2)),
        ht_cells=np.zeros(solver.n_cells),
        bc_lst=[1e6] * 4 if pressures is None else list(pressures),
    )


class ReynoldsLeakageTests(unittest.TestCase):
    def setUp(self):
        self.mesh = equilateral_mesh()
        self.solver = ReynoldsSolver(self.mesh, FluidProp(mu=0.01))

    def test_poiseuille_linear_pressure_solution_and_boundary_flux(self):
        s = self.solver
        gradient = np.array([2e5, -3e5])
        boundary_pressure = (
            1e6 + s.face_midpoints[s.boundary_face_indices] @ gradient
        )
        film = film_for(s, boundary_pressure)
        pressure = s.solve(film).p
        np.testing.assert_allclose(pressure, 1e6 + s.centroids @ gradient)
        flow = s.calc_leakage(pressure, film, include_couette=False)
        expected = (
            film.h_cells[0] ** 3
            / (12 * s.mu)
            * (s.boundary_normals @ gradient)
            * s.face_lengths[s.boundary_face_indices]
        )
        np.testing.assert_allclose(flow, expected, rtol=1e-12, atol=1e-20)
        self.assertAlmostEqual(sum(flow), 0.0, delta=1e-20)

    def test_arbitrary_facet_directions_do_not_change_flux(self):
        film = film_for(self.solver, [1e6, 1.1e6, 1.2e6, 1.3e6])
        film.h_grad = (1e-5, -2e-5)
        film.h_cells += self.solver.centroids @ film.h_grad
        film.U_grad = np.array([[0.0, -0.7], [0.7, 0.0]])
        film.U_cells = self.solver.centroids @ film.U_grad.T + [2.0, -0.7]
        pressure = self.solver.solve(film).p
        facets = self.mesh.facets.copy()
        facets[[0, 2, 3]] = facets[[0, 2, 3], ::-1]
        reversed_solver = ReynoldsSolver(
            SimpleNamespace(**{**vars(self.mesh), "facets": facets}),
            FluidProp(mu=0.01),
        )
        np.testing.assert_allclose(
            reversed_solver.calc_leakage(pressure, film),
            self.solver.calc_leakage(pressure, film),
            atol=1e-20,
        )

    def test_hole_normals_point_from_film_into_hole(self):
        mesh = SimpleNamespace(
            points=np.array(
                [
                    [-1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                    [-0.5, -0.5],
                    [0.5, -0.5],
                    [0.5, 0.5],
                    [-0.5, 0.5],
                ]
            ),
            elements=np.array(
                [
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ]
            ),
            # 内外环都按逆时针给出，法向不能只依赖环的方向。
            facets=np.array(
                [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]]
            ),
            facet_markers=np.ones(8, dtype=int),
        )
        solver = ReynoldsSolver(mesh, FluidProp(mu=0.01))
        expected = np.array(
            [
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 0.0],
            ]
        )
        np.testing.assert_allclose(solver.boundary_normals, expected)
        film = film_for(solver)
        film.U_cells[:] = [2.0, -0.5]
        lengths = solver.face_lengths[solver.boundary_face_indices]
        np.testing.assert_allclose(
            solver.calc_leakage(np.full(solver.n_cells, 1e6), film),
            -0.5 * 1e-4 * lengths * (expected @ film.U_cells[0]),
        )

    def test_facet_order_marker_zero_and_internal_facet_mapping(self):
        film = film_for(self.solver, [1e6, 1.1e6, 1.2e6, 1.3e6])
        pressure = self.solver.solve(film).p
        expected = self.solver.calc_leakage(pressure, film)
        order = np.array([2, 0, 3, 1])
        mesh = SimpleNamespace(
            **{
                **vars(self.mesh),
                "facets": np.vstack((self.mesh.facets[order], [1, 2])),
                "facet_markers": np.r_[self.mesh.facet_markers[order], 99],
            }
        )
        # marker=0 外边界使用 bc_lst[0]，与原来的 marker=1 等价。
        mesh.facet_markers[1] = 0
        solver = ReynoldsSolver(mesh, FluidProp(mu=0.01))
        actual = solver.calc_leakage(pressure, film)
        self.assertEqual(len(actual), len(mesh.facets))
        np.testing.assert_allclose(actual[:-1], np.asarray(expected)[order])
        self.assertEqual(actual[-1], 0.0)
        np.testing.assert_allclose(solver.solve(film).p, pressure)

    def test_pressure_flux_conserves_squeeze_source(self):
        s = self.solver
        film = film_for(s)
        film.ht_cells[:] = [1e-8, 2e-8]
        pressure = s.solve(film).p
        self.assertTrue(np.all(pressure > 0.0))
        flow = s.calc_leakage(pressure, film, include_couette=False)
        self.assertAlmostEqual(
            sum(flow), np.dot(film.ht_cells, s.areas), delta=1e-20
        )

    def test_total_flux_conserves_inclined_rotating_film_source(self):
        s = ReynoldsSolver(skew_mesh(), FluidProp(mu=0.01))
        film = film_for(s)
        gradient = np.array([1e-5, -2e-5])
        film.h_cells += s.centroids @ gradient
        film.h_grad = tuple(gradient)
        film.U_grad = np.array([[0.0, -3.0], [3.0, 0.0]])
        film.U_cells = s.centroids @ film.U_grad.T + [1.0, -0.5]
        film.ht_cells[:] = 2e-8
        pressure = s.solve(film).p
        self.assertTrue(np.all(pressure > 0.0))
        flow = s.calc_leakage(pressure, film)
        self.assertAlmostEqual(
            sum(flow), np.dot(film.ht_cells, s.areas), delta=1e-19
        )

    def test_constant_couette_flow_sign_and_pressure_only_option(self):
        s = self.solver
        film = film_for(s)
        film.U_cells[:] = [2.0, -0.5]
        pressure = np.full(s.n_cells, 1e6)
        expected = (
            -0.5
            * film.h_cells[0]
            * (s.boundary_normals @ film.U_cells[0])
            * s.face_lengths[s.boundary_face_indices]
        )
        np.testing.assert_allclose(s.calc_leakage(pressure, film), expected)
        np.testing.assert_array_equal(
            s.calc_leakage(pressure, film, include_couette=False), 0.0
        )

    def test_density_integrates_to_face_flow(self):
        s = ReynoldsSolver(skew_mesh(), FluidProp(mu=0.01))
        film = film_for(s, [1e6, 2e6, 3e6, 4e6])
        pressure = s.solve(film).p
        density = s.calc_leakage(pressure, film, per_unit_length=True)
        lengths = np.linalg.norm(
            s.points[s.facets[:, 1]] - s.points[s.facets[:, 0]], axis=1
        )
        np.testing.assert_allclose(
            density * lengths, s.calc_leakage(pressure, film)
        )

    def test_closed_boundary_cells_have_no_leakage_or_velocity(self):
        s = self.solver
        film = film_for(s)
        film.h_cells[:] = [0.0, -1e-6]
        film.U_cells[:] = [2.0, 1.0]
        pressure = np.full(s.n_cells, 5e5)
        np.testing.assert_array_equal(s.calc_leakage(pressure, film), 0.0)
        np.testing.assert_array_equal(s._calc_flow(pressure, film), 0.0)

    def test_leakage_does_not_compute_full_field_gradient(self):
        s = self.solver
        with (
            patch.object(s, "_calc_flow", side_effect=AssertionError),
            patch.object(
                s, "_calc_pressure_gradient", side_effect=AssertionError
            ),
        ):
            s.calc_leakage(np.full(s.n_cells, 1e6), film_for(s))
        self.assertIsNone(s._gradient_weights)

    def test_empty_boundary_conditions_fail_for_solve_and_leakage(self):
        film = film_for(self.solver, [])
        with self.assertRaisesRegex(ValueError, "bc_lst"):
            self.solver.solve(film)
        with self.assertRaisesRegex(ValueError, "bc_lst"):
            self.solver.calc_leakage(np.zeros(2), film)


class PressureGradientTests(unittest.TestCase):
    def test_linear_field_is_exact_at_skew_boundary_cells(self):
        s = ReynoldsSolver(skew_mesh(), FluidProp(mu=0.01))
        gradient = np.array([2e5, -3e5])
        midpoint = s.face_midpoints[s.boundary_face_indices]
        film = film_for(s, 1e6 + midpoint @ gradient)
        pressure = 1e6 + s.centroids @ gradient
        px, py = s._calc_pressure_gradient(pressure, film)
        np.testing.assert_allclose(
            np.column_stack((px, py)),
            np.tile(gradient, (s.n_cells, 1)),
            rtol=1e-13,
        )
        u, v = s._calc_flow(pressure, film)
        np.testing.assert_allclose(
            u, -film.h_cells**2 * gradient[0] / (12 * s.mu)
        )
        np.testing.assert_allclose(
            v, -film.h_cells**2 * gradient[1] / (12 * s.mu)
        )

    def test_cached_geometry_uses_current_boundary_pressures(self):
        s = ReynoldsSolver(skew_mesh(), FluidProp(mu=0.01))
        film = film_for(s)
        pressure = np.full(s.n_cells, 1e6)
        px, py = s._calc_pressure_gradient(pressure, film)
        np.testing.assert_array_equal([px, py], 0.0)
        weights = s._gradient_weights
        changed = replace(film, bc_lst=[2e6, 1e6, 1e6, 1e6])
        px, py = s._calc_pressure_gradient(pressure, changed)
        self.assertIs(s._gradient_weights, weights)
        self.assertGreater(np.linalg.norm([px, py]), 0.0)
        alias_px, alias_py = s._calc_pressure_grediant(pressure, changed)
        np.testing.assert_array_equal([px, py], [alias_px, alias_py])

    def test_film_calculator_supplies_exact_rigid_velocity_gradient(self):
        mesh = skew_mesh()
        for gear in ("drive", "slave"):
            with self.subTest(gear=gear):
                film = FilmParameterCalculator(
                    mesh, [0.0, ((0.0, 0.0), 1e6)], 200.0, gear
                ).solve(SidePlateState(p=np.array([0.0, 0.0, 1e-4])))
                s = ReynoldsSolver(mesh, FluidProp(mu=0.01))
                face_velocity = film.U_cells[s.boundary_owners] + (
                    s.boundary_offsets @ film.U_grad.T
                )
                midpoint = s.face_midpoints[s.boundary_face_indices]
                center = 0.0305 if gear == "drive" else -0.0305
                omega = 200.0 if gear == "drive" else -200.0
                expected = np.column_stack(
                    (-omega * midpoint[:, 1], omega * (midpoint[:, 0] - center))
                )
                np.testing.assert_allclose(face_velocity, expected, atol=1e-13)


class LeakagePlotTests(unittest.TestCase):
    def test_uniform_density_has_uniform_color_on_unequal_edges(self):
        import matplotlib

        matplotlib.use("Agg")
        from src.postproc import visualize

        mesh = skew_mesh()
        lengths = np.linalg.norm(
            mesh.points[mesh.facets[:, 1]] - mesh.points[mesh.facets[:, 0]],
            axis=1,
        )
        with (
            patch.object(visualize.plt, "savefig"),
            patch.object(visualize.plt, "close"),
        ):
            visualize.plot_leak_rate(mesh, lengths * 2e-8, "unused")
            figure = visualize.plt.gcf()
            colors = [line.get_color() for line in figure.axes[0].lines]
            np.testing.assert_allclose(colors, np.tile(colors[0], (4, 1)))
            self.assertEqual(
                figure.axes[-1].get_ylabel(), "Leak Rate per Length [m^2/s]"
            )
        visualize.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
