"""Shared configuration and quadratic problems for the unconstrained proposal."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())
BUILD = HERE / "build"
RESULTS = HERE / "results/full-scipy"
PATCH = HERE / MANIFEST["patch"]["file"]
VARIANTS = ("baseline", "proposal")
FAMILIES = tuple(MANIFEST["benchmark"]["families"])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_dir(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return BUILD / f"scipy-{variant}"


def installed_site(variant):
    prefix = source_dir(variant) / "build-install"
    matches = list(prefix.glob("**/site-packages/scipy/__init__.py"))
    if len(matches) != 1:
        raise RuntimeError(f"Build {variant} first with build_proposal.py: {prefix}")
    return matches[0].parent.parent


def pin_cpu():
    # Affinity is part of the Linux/WSL protocol, using an available CPU by default.
    available = os.sched_getaffinity(0)
    requested = MANIFEST["runtime"]["cpu"]
    cpu = min(available) if requested is None else requested
    if cpu not in available:
        raise RuntimeError(f"Requested CPU {cpu} is unavailable: {sorted(available)}")
    os.sched_setaffinity(0, {cpu})


def metadata(variant):
    import scipy
    from scipy.optimize import _lbfgsb
    from scipy.version import git_revision

    site = installed_site(variant).resolve()
    # A separately installed SciPy must never be substituted for the tested build.
    for path in (Path(scipy.__file__), Path(_lbfgsb.__file__)):
        if not path.resolve().is_relative_to(site):
            raise RuntimeError(f"SciPy was loaded outside the requested build: {path}")
    if git_revision != MANIFEST["scipy"]["commit"]:
        raise RuntimeError(f"The installed SciPy reports a different commit: {git_revision}")
    return {
        "python": sys.version, "python_executable": sys.executable,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scipy_path": scipy.__file__, "platform": platform.platform(),
        "variant": variant, "manifest": MANIFEST,
        "extension_sha256": digest(Path(_lbfgsb.__file__)),
        "patch_sha256": digest(PATCH), "threadpools": threadpool_info(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }


class VariantProcess:
    """Keep each complete SciPy build in its own process and Python import path."""

    def __init__(self, variant):
        self.variant = variant
        site = installed_site(variant)
        record = json.loads((RESULTS / "build.json").read_text())
        if (record["manifest"]["scipy"] != MANIFEST["scipy"]
                or record["patch_sha256"] != digest(PATCH)
                or record["python_version"] != list(sys.version_info[:2])):
            raise RuntimeError("Build provenance differs from the current recipe; rebuild both variants")
        extensions = list((site / "scipy/optimize").glob("_lbfgsb*.so"))
        if len(extensions) != 1 or digest(extensions[0]) != record["extension_sha256"][variant]:
            raise RuntimeError(f"The installed {variant} extension differs from build.json")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(site)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = str(MANIFEST["runtime"]["blas_threads"])
        self.log = (BUILD / f"worker-{variant}.log").open("w")
        self.process = subprocess.Popen(
            [sys.executable, "-B", str(HERE / "experiment_worker.py")],
            cwd=HERE, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1)

    def request(self, operation, **payload):
        request = {"operation": operation, "variant": self.variant, **payload}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Worker exited; see build/worker-{self.variant}.log")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()
        self.log.close()


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
                       else list(zip(self.lower, self.upper)))

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


def solve(problem, memory=10, iterations=100, gtol=0.0, maxls=20):
    from scipy.optimize import minimize

    # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    return minimize(problem.fun, problem.x0.copy(), jac=problem.jac,
                    method="L-BFGS-B", bounds=problem.bounds,
                    options={"maxcor": memory, "maxiter": iterations,
                             "maxfun": 50*iterations + 100, "maxls": maxls,
                             "ftol": 0.0, "gtol": gtol})


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
