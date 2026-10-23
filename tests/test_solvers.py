"""Независимые проверки схем, устойчивости, сходимости и краевых условий."""

import unittest

import numpy as np

from solvers import (
    HEAT_METHODS, calculate_max_error, classify_pde, exact_heat_solution,
    exact_wave_solution, heat_convergence, solve_heat, solve_heat_explicit,
    solve_laplace, solve_tridiagonal, solve_wave,
)


class ClassificationTests(unittest.TestCase):
    def test_types_and_scaling(self):
        for scale in (1e-100, 1.0, 1e100, -10):
            for coefficients, kind in [((1, 0, 1), "ЭЛЛИПТИЧЕСКИЙ"),
                                       ((1, 2, 1), "ПАРАБОЛИЧЕСКИЙ"),
                                       ((1, 0, -1), "ГИПЕРБОЛИЧЕСКИЙ")]:
                with self.subTest(scale=scale, coefficients=coefficients):
                    self.assertEqual(classify_pde(*(c * scale for c in coefficients))[1], kind)
        self.assertEqual(classify_pde(1, 0, -1)[0], 4)

    def test_invalid_coefficients(self):
        for coefficients in ((0, 0, 0), (np.nan, 0, 1), (1, np.inf, 1), (1e308, 0, 1e308)):
            with self.assertRaises(ValueError):
                classify_pde(*coefficients)


class HeatTests(unittest.TestCase):
    def test_reference_example(self):
        x, t, u, r = solve_heat_explicit(0.1, 0.2, 0.25, 0.1)
        np.testing.assert_allclose(x, [0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_allclose(t, [0, 0.1, 0.2])
        np.testing.assert_allclose(u[1], [0, 0.68, 0, -0.68, 0], atol=1e-15)
        np.testing.assert_allclose(u[2], [0, 0.4624, 0, -0.4624, 0], atol=1e-15)
        self.assertAlmostEqual(r, 0.16)
        self.assertAlmostEqual(calculate_max_error(u[-1], exact_heat_solution(x, 0.2, 0.1)),
                               0.4624 - np.exp(-4 * np.pi ** 2 * 0.1 * 0.2))

    def test_schemes_against_discrete_eigenmode(self):
        # Синус — собственный вектор дискретного оператора Лапласа.
        # Это проверяет все слои независимо от циклов решателя.
        for method, theta in zip(HEAT_METHODS, (0, 1, 0.5)):
            x, t, u, r = solve_heat(0.1, 0.2, 0.05, 0.01, method)
            eigenvalue = 4 * r * np.sin(np.pi * 0.05) ** 2
            growth = (1 - (1 - theta) * eigenvalue) / (1 + theta * eigenvalue)
            expected = growth ** np.arange(len(t))[:, None] * np.sin(2 * np.pi * x)
            expected[:, 0] = expected[:, -1] = 0
            np.testing.assert_allclose(u, expected, atol=2e-14)

    def test_convergence(self):
        for method in HEAT_METHODS:
            rows = heat_convergence(0.1, 0.2, 0.1, 0.02, method)
            self.assertLess(rows[1][3], rows[0][3] * 0.4)
            self.assertLess(rows[2][3], rows[1][3] * 0.4)
            self.assertAlmostEqual(rows[0][2], rows[-1][2])

    def test_implicit_stability_for_large_time_step(self):
        for method in HEAT_METHODS[1:]:
            x, t, u, r = solve_heat(1, 1, 0.05, 0.5, method)
            self.assertGreater(r, 0.5)
            self.assertTrue(np.all(np.isfinite(u)))
            self.assertLessEqual(np.max(np.abs(u)), 1 + 1e-12)

    def test_invalid_parameters(self):
        cases = [(1, 0.2, 0.25, 0.1), (0, 1, 0.1, 0.01),
                 (np.nan, 1, 0.1, 0.01), (0.1, 0, 0.1, 0.01),
                 (0.1, 0.2, 0.3, 0.01), (0.1, 0.2, 0.1, 0.03),
                 (0.1, 0.2, 0.5, 0.01), (0.1, 1e-15, 0.1, 1e-14),
                 (0.1, 1, 1e-300, 0.01), (0.1, 1, 0.1, 1e-300),
                 (0.1, 1, 0.0001, 1e-5)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                solve_heat_explicit(*args)
        with self.assertRaises(ValueError):
            solve_heat(0.1, 1, 0.1, 0.01, "unknown")

    def test_small_valid_time_domain(self):
        x, t, u, r = solve_heat_explicit(0.1, 1e-15, 0.1, 1e-16)
        self.assertEqual(len(t), 11)
        self.assertEqual(u.shape, (11, 11))


class TridiagonalTests(unittest.TestCase):
    def test_against_dense_solver(self):
        rng = np.random.default_rng(42)
        for n in (1, 2, 20):
            lower, upper = rng.random(n - 1), rng.random(n - 1)
            diagonal, rhs = 3 + rng.random(n), rng.random(n)
            matrix = np.diag(diagonal) + np.diag(lower, -1) + np.diag(upper, 1)
            original = diagonal.copy()
            np.testing.assert_allclose(solve_tridiagonal(lower, diagonal, upper, rhs),
                                       np.linalg.solve(matrix, rhs), atol=1e-14)
            np.testing.assert_array_equal(diagonal, original)

    def test_invalid_system(self):
        for args in (([], [], [], []), ([], [0], [], [1]), ([1], [1], [], [1])):
            with self.assertRaises(ValueError):
                solve_tridiagonal(*args)


class WaveTests(unittest.TestCase):
    def test_courant_one_is_exact_at_nodes(self):
        x, t, u, courant = solve_wave(1, 1, 0.05, 0.05)
        self.assertAlmostEqual(courant, 1)
        for n, time in enumerate(t):
            np.testing.assert_allclose(u[n], exact_wave_solution(x, time, 1), atol=5e-15)

    def test_convergence_and_boundaries(self):
        errors = []
        for h in (0.1, 0.05):
            x, t, u, _ = solve_wave(2, 0.3, h, h / 4, length=2, amplitude=3)
            np.testing.assert_array_equal(u[:, [0, -1]], 0)
            errors.append(calculate_max_error(u[-1], exact_wave_solution(x, t[-1], 2, 2, 3)))
        self.assertLess(errors[1], errors[0] * 0.3)

    def test_invalid_courant(self):
        with self.assertRaises(ValueError):
            solve_wave(1, 1, 0.1, 0.2)


class LaplaceTests(unittest.TestCase):
    def test_constant_solution(self):
        x, y, u, iterations, residual = solve_laplace(left=42, right=42, bottom=42, top=42)
        np.testing.assert_array_equal(u, 42)
        self.assertEqual(iterations, 1)
        self.assertEqual(residual, 0)

    def test_symmetry_bounds_residual_and_superposition(self):
        x, y, u, _, residual = solve_laplace(h=0.1, epsilon=1e-7)
        np.testing.assert_allclose(u, u[:, ::-1], atol=1e-8)
        self.assertGreaterEqual(u.min(), 0)
        self.assertLessEqual(u.max(), 100)
        self.assertLessEqual(residual, 1e-7)
        np.testing.assert_array_equal(u[0, 1:-1], 0)
        np.testing.assert_array_equal(u[-1, 1:-1], 100)
        # Сумма четырёх повёрнутых решений даёт постоянную температуру 100.
        np.testing.assert_allclose(u + np.rot90(u) + np.rot90(u, 2) + np.rot90(u, 3), 100, atol=1e-7)

    def test_iteration_limit_and_invalid_input(self):
        for kwargs in ({"max_iterations": 1}, {"h": 0.3}, {"epsilon": 0}, {"top": np.inf},
                       {"max_iterations": 1.5}, {"h": 0.001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solve_laplace(**kwargs)


class ErrorTests(unittest.TestCase):
    def test_max_error(self):
        self.assertEqual(calculate_max_error([1, 2], [1.5, 1]), 1)
        for a, b in (([], []), ([1], [1, 2]), ([np.nan], [0]), ([np.inf], [0])):
            with self.assertRaises(ValueError):
                calculate_max_error(a, b)
        with self.assertRaises(ValueError):
            exact_heat_solution([0], -1, 0.1)


if __name__ == "__main__":
    unittest.main()
