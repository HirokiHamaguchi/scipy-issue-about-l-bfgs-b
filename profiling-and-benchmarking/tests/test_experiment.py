"""Independent mathematical checks that do not require compiled SciPy builds."""

import unittest

import numpy as np

import experiment as ex


class TestQuadratics(unittest.TestCase):
    def test_gradients_against_finite_differences(self):
        rng = np.random.default_rng(261128)
        for n in (1, 8, 31):
            for family in (*ex.FAMILIES, "mixed_bounds"):
                with self.subTest(n=n, family=family):
                    problem = ex.Problem(family, n)
                    x = rng.uniform(-0.3, 0.3, n)
                    numerical = np.empty(n)
                    step = 1e-6
                    for i in range(n):
                        plus, minus = x.copy(), x.copy()
                        plus[i] += step
                        minus[i] -= step
                        numerical[i] = (problem.fun(plus) - problem.fun(minus)) / (2*step)
                    np.testing.assert_allclose(problem.jac(x), numerical, rtol=1e-6, atol=1e-7)

    def test_analytic_minima_and_projected_gradients(self):
        for n in (1, 8, 31):
            for family in (*ex.FAMILIES, "mixed_bounds"):
                with self.subTest(n=n, family=family):
                    problem = ex.Problem(family, n, check_bounds=True)
                    x = problem.solution
                    expected = -0.5 if family == "zero_chain" else 0.0
                    self.assertAlmostEqual(problem.fun(x), expected)
                    gradient = problem.jac(x)
                    projected = x - np.clip(x - gradient, problem.lower, problem.upper)
                    np.testing.assert_array_equal(projected, np.zeros(n))

    def test_all_bound_types_are_present(self):
        problem = ex.Problem("mixed_bounds", 8)
        kinds = {(bool(np.isfinite(lo)), bool(np.isfinite(hi)))
                 for lo, hi in zip(problem.lower, problem.upper, strict=True)}
        self.assertEqual(kinds, {(False, False), (True, False), (False, True), (True, True)})
