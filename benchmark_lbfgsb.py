"""Small, dependency-free L-BFGS-B timing benchmark."""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
import numpy as np
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_info, threadpool_limits

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
ITERATIONS = 300
MEMORY = 10
REPEATS = 3


def zero_chain_value(x: np.ndarray) -> float:
    diff = x[:-1] - x[1:]
    # Avoid BLAS here so backend differences come from L-BFGS-B itself.
    squared_norm = np.sum(diff * diff)
    return float(0.5 * x[0] ** 2 - x[0] + 0.5 * squared_norm)


def zero_chain_grad(x: np.ndarray) -> np.ndarray:
    diff = x[:-1] - x[1:]
    grad = np.zeros_like(x)
    grad[0] = x[0] - 1.0
    grad[:-1] += diff
    grad[1:] -= diff
    return grad


def solve(n: int, iterations: int, memory: int):
    return minimize(
        zero_chain_value,
        np.zeros(n),
        jac=zero_chain_grad,
        method="L-BFGS-B",
        options={
            "maxcor": memory,
            "maxiter": iterations,
            "maxfun": 50 * iterations + 100,
            "ftol": 0.0,
            "gtol": 0.0,
        },
    )


def main() -> None:
    with threadpool_limits(1, user_api="blas"):
        solve(600, 5, MEMORY)  # Warm imports and the extension.
        pools = threadpool_info()
        rows = []
        for n in DIMENSIONS:
            elapsed = []
            result = None
            for _ in range(REPEATS):
                start = time.perf_counter()
                result = solve(n, ITERATIONS, MEMORY)
                elapsed.append(time.perf_counter() - start)
            assert result is not None
            if result.nit != ITERATIONS:
                raise RuntimeError(
                    f"n={n}: expected {ITERATIONS} iterations, got {result.nit}: "
                    f"{result.message}"
                )
            rows.append({
                "dimension": n,
                "seconds": elapsed,
                "median_seconds": statistics.median(elapsed),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "final_objective": float(result.fun),
                "gradient_inf_norm": float(np.linalg.norm(result.jac, ord=np.inf)),
                "status": int(result.status),
                "message": str(result.message),
            })

    print(json.dumps({
        "requested_blas_threads": 1,
        "objective_reduction": "numpy",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scipy_git_revision": scipy.version.git_revision,
        "scipy_file": scipy.__file__,
        "threadpools": pools,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
