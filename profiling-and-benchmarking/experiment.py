"""Shared configuration and quadratic problems for the unconstrained proposal."""

import hashlib
import importlib.util
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCIPY_SOURCE = ROOT.parent / "scipy"
ENVIRONMENT = Path("/home/hirok/miniforge3/envs/scipy-dev")
SCIPY_PYTHON = Path("/home/hirok/scipy/build-install/usr/lib/python3.13/site-packages")
WORK = ROOT / "tmp/codex-unconstrained-proposal-20261126-a2c7"
RESULTS = HERE / "results/unconstrained-two-loop"
PATCH = HERE / "unconstrained_two_loop.patch"
CPU = 2
VARIANTS = ("baseline", "proposal")
FAMILIES = ("zero_chain", "diagonal", "wide_bounds", "late_bound", "mixed_active")
BASELINE_SHA256 = "f2d11e19a61b4f6119657a326dde67f6f8e804bb8e83935a9644a2823e129a10"

# Reuse the installed Python driver; only the C extension is rebuilt locally.
sys.path.insert(0, str(SCIPY_PYTHON))

import numpy as np
import scipy
from scipy.optimize import Bounds, minimize
import scipy.optimize._lbfgsb_py as wrapper
from threadpoolctl import threadpool_info, threadpool_limits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata():
    return {
        "python": sys.version, "python_executable": sys.executable,
        "numpy": np.__version__, "scipy_python_driver": scipy.__version__,
        "scipy_python_driver_path": scipy.__file__, "platform": platform.platform(),
        "scipy_checkout_commit": subprocess.check_output(
            ["git", "-C", str(SCIPY_SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        "patch_sha256": digest(PATCH), "threadpools": threadpool_info(),
        "limitations": ["Current C core with an existing older SciPy Python driver",
                        "Synthetic problems; CUTEst and full SciPy testing remain pending"],
    }


def load_module(variant):
    path = WORK / "build" / variant / ("_lbfgsb" + sysconfig.get_config_var("EXT_SUFFIX"))
    spec = importlib.util.spec_from_file_location(f"experiment_{variant}._lbfgsb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Problem:
    def __init__(self, family, n, check_bounds=False):
        self.family, self.n = family, n
        self.check_bounds = check_bounds
        self.checked_evaluations = 0
        self.lower, self.upper = np.full(n, -np.inf), np.full(n, np.inf)
        self.x0 = np.zeros(n) if family == "zero_chain" else np.ones(n)
        self.weights = np.arange(1, n + 1, dtype=float)
        self.target = np.zeros(n)
        if family == "wide_bounds":
            self.lower[:], self.upper[:] = -1e6, 1e6
        elif family == "late_bound":
            self.weights = np.geomspace(1.0, 1000.0, n)
            self.lower[0] = 0.5
        elif family == "mixed_active":
            self.lower[:n // 4] = 0.0
            self.x0[:n // 4] = 0.0
            self.target[:n // 4] = -1.0
        elif family == "mixed_bounds":
            # Exercise all four nbd codes, including upper-only bounds.
            q = n // 4
            self.lower[:q], self.target[:q], self.x0[:q] = 0.0, -1.0, 0.0
            self.upper[q:2*q], self.target[q:2*q], self.x0[q:2*q] = 0.0, 1.0, 0.0
            self.lower[2*q:3*q], self.upper[2*q:3*q] = -0.5, 0.5
            self.target[2*q:3*q], self.x0[2*q:3*q] = 2.0, 0.0
        self.solution = (np.ones(n) if family == "zero_chain"
                         else np.clip(self.target, self.lower, self.upper))
        self.bounds = (None if family in ("zero_chain", "diagonal")
                       else Bounds(self.lower, self.upper))

    def fun(self, x):
        if self.check_bounds:
            assert np.all(x >= self.lower) and np.all(x <= self.upper), (
                self.family, self.n, float(np.max(self.lower - x)), float(np.max(x - self.upper)))
            self.checked_evaluations += 1
        if self.family == "zero_chain":
            delta = x[:-1] - x[1:]
            return float(0.5*x[0]**2 - x[0] + 0.5*np.sum(delta*delta))
        delta = x - self.solution
        value = 0.5*np.sum(self.weights*delta*delta)
        if self.family in ("mixed_active", "mixed_bounds", "late_bound"):
            # Shift the constrained minimum to zero without subtracting constants.
            value += np.sum(self.weights*(self.solution - self.target)*delta)
        return float(value)

    def jac(self, x):
        if self.family == "zero_chain":
            delta = x[:-1] - x[1:]
            gradient = np.zeros_like(x)
            gradient[0] = x[0] - 1.0
            gradient[:-1] += delta
            gradient[1:] -= delta
            return gradient
        return self.weights*(x - self.target)


def solve(module, problem, memory=10, iterations=100, gtol=0.0, maxls=20):
    previous = wrapper._lbfgsb
    wrapper._lbfgsb = module
    try:
        # Option semantics: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        return minimize(problem.fun, problem.x0.copy(), jac=problem.jac,
                        method="L-BFGS-B", bounds=problem.bounds,
                        options={"maxcor": memory, "maxiter": iterations,
                                 "maxfun": 50*iterations + 100, "maxls": maxls,
                                 "ftol": 0.0, "gtol": gtol})
    finally:
        wrapper._lbfgsb = previous


def audit(problem, result):
    gradient = problem.jac(result.x)
    projected = result.x - np.clip(result.x - gradient, problem.lower, problem.upper)
    violation = max(0.0, float(np.max(problem.lower - result.x)),
                    float(np.max(result.x - problem.upper)))
    assert np.isfinite(result.fun) and np.all(np.isfinite(result.x))
    assert np.all(np.isfinite(result.jac)) and violation == 0.0
    return {"iterations": int(result.nit), "nfev": int(result.nfev),
            "njev": int(result.njev), "status": int(result.status),
            "message": str(result.message), "objective": float(result.fun),
            "objective_hex": float(result.fun).hex(),
            "projected_gradient_inf_norm": float(np.max(np.abs(projected))),
            "bound_violation": violation,
            "x_sha256": hashlib.sha256(result.x.tobytes()).hexdigest(),
            "jac_sha256": hashlib.sha256(result.jac.tobytes()).hexdigest()}
