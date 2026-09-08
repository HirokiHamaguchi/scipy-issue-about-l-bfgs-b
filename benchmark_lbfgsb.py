"""Benchmark SciPy L-BFGS-B with the active BLAS backend."""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_info, threadpool_limits


HERE = Path(__file__).resolve().parent

DIMENSIONS = (
    500,
    1_000,
    3_000,
    5_000,
    7_500,
    9_000,
    10_000,
    11_000,
    12_000,
    13_000,
    14_000,
    15_000,
    20_000,
    30_000,
    50_000,
    75_000,
    100_000,
    300_000,
    1_000_000,
)

MEMORY = 10
REPEATS = 5
THREAD_COUNTS = (None, 1)


class ZeroChainQuadratic:
    iterations = 300

    def __init__(self, dimension: int):
        self.x0 = np.zeros(dimension)

    def fun(self, x: np.ndarray) -> float:
        differences = x[:-1] - x[1:]
        squared_norm = np.sum(differences * differences)
        return float(
            0.5 * x[0] ** 2
            - x[0]
            + 0.5 * squared_norm
        )

    def jac(self, x: np.ndarray) -> np.ndarray:
        differences = x[:-1] - x[1:]
        gradient = np.zeros_like(x)
        gradient[0] = x[0] - 1.0
        gradient[:-1] += differences
        gradient[1:] -= differences
        return gradient


class DiagonalQuadratic:
    iterations = 100

    def __init__(self, dimension: int):
        self.x0 = np.ones(dimension)
        self.weights = np.arange(1, dimension + 1, dtype=float)

    def fun(self, x: np.ndarray) -> float:
        # Avoid BLAS in the objective.
        return float(0.5 * np.sum(self.weights * x * x))

    def jac(self, x: np.ndarray) -> np.ndarray:
        return self.weights * x


PROBLEMS = {
    "zero_chain": ZeroChainQuadratic,
    "diagonal_quadratic": DiagonalQuadratic,
}


def active_backend(pools: list[dict]) -> str:
    backends = {
        pool["internal_api"]
        for pool in pools
        if pool.get("user_api") == "blas"
    }
    if len(backends) != 1:
        raise RuntimeError(f"Expected one BLAS backend, found {backends}")
    return backends.pop()


def solve(problem) -> object:
    return minimize(
        problem.fun,
        problem.x0.copy(),
        jac=problem.jac,
        method="L-BFGS-B",
        options={
            "maxcor": MEMORY,
            "maxiter": problem.iterations,
            "maxfun": 50 * problem.iterations + 100,
            "ftol": 0.0,
            "gtol": 0.0,
        },
    )


def benchmark_problem(
    name: str,
    problem_class: type,
    dimension: int,
    threads: int | None,
) -> dict:
    context = (
        nullcontext()
        if threads is None
        else threadpool_limits(threads, user_api="blas")
    )

    elapsed = []
    result = None

    with context:
        for _ in range(REPEATS):
            problem = problem_class(dimension)

            start = time.perf_counter()
            result = solve(problem)
            elapsed.append(time.perf_counter() - start)

    assert result is not None

    thread_label = "default" if threads is None else threads
    print(
        f"{name}, threads={thread_label}, n={dimension}: "
        f"{statistics.median(elapsed):.6f} s",
        flush=True,
    )

    return {
        "problem": name,
        "blas_threads": thread_label,
        "dimension": dimension,
        "seconds": elapsed,
        "median_seconds": statistics.median(elapsed),
        "requested_iterations": problem_class.iterations,
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "final_objective": float(result.fun),
        "gradient_inf_norm": float(
            np.linalg.norm(result.jac, ord=np.inf)
        ),
        "status": int(result.status),
        "message": str(result.message),
    }


def main() -> None:
    pools = threadpool_info()
    backend = active_backend(pools)

    rows = []

    for threads in THREAD_COUNTS:
        for name, problem_class in PROBLEMS.items():
            # Warm up imports and the compiled extension.
            context = (
                nullcontext()
                if threads is None
                else threadpool_limits(threads, user_api="blas")
            )
            with context:
                solve(problem_class(600))

            for dimension in DIMENSIONS:
                rows.append(
                    benchmark_problem(
                        name,
                        problem_class,
                        dimension,
                        threads,
                    )
                )

    results_dir = Path(os.environ.get("LBFGSB_RESULTS_DIR", HERE / "results"))
    output = results_dir / f"{backend}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "backend": backend,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scipy_git_revision": scipy.version.git_revision,
        "scipy_file": scipy.__file__,
        "threadpools": pools,
        "dimensions": DIMENSIONS,
        "memory": MEMORY,
        "repeats": REPEATS,
        "rows": rows,
    }

    output.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
