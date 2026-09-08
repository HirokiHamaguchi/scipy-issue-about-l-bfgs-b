"""Long-running L-BFGS-B workload for Linux perf."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy
from scipy.optimize import Bounds, minimize
from threadpoolctl import threadpool_info, threadpool_limits


def zero_chain_value(x: np.ndarray) -> float:
    # Avoid np.dot: BLAS samples should come from the solver, not the objective.
    diff = x[:-1] - x[1:]
    value = 0.5 * x[0] ** 2 - x[0] + 0.5 * np.sum(diff * diff)
    return float(value)


def zero_chain_grad(x: np.ndarray) -> np.ndarray:
    diff = x[:-1] - x[1:]
    grad = np.zeros_like(x)
    grad[0] = x[0] - 1.0
    grad[:-1] += diff
    grad[1:] -= diff
    return grad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=300_000)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--memory", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bounds", choices=("none", "box"), default="none")
    args = parser.parse_args()
    if args.dimension <= args.iterations:
        parser.error("--dimension must exceed --iterations")

    bounds = None
    if args.bounds == "box":
        bounds = Bounds(np.zeros(args.dimension), np.full(args.dimension, 2.0))

    start = time.perf_counter()
    with threadpool_limits(args.threads, user_api="blas"):
        print(json.dumps({
            "scipy_file": scipy.__file__,
            "scipy_git_revision": scipy.version.git_revision,
            "blas_threads": args.threads,
            "threadpools": threadpool_info(),
            "bounds": args.bounds,
        }, indent=2), flush=True)
        for repeat in range(args.repeats):
            result = minimize(
                zero_chain_value,
                np.zeros(args.dimension),
                jac=zero_chain_grad,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxcor": args.memory,
                    "maxiter": args.iterations,
                    "maxfun": 50 * args.iterations + 100,
                    "ftol": 0.0,
                    "gtol": 0.0,
                },
            )
            if result.nit != args.iterations:
                raise RuntimeError(f"unexpected termination: {result.message}")
            print(f"repeat={repeat + 1}/{args.repeats}, nit={result.nit}, nfev={result.nfev}")

    print(f"total_elapsed_seconds={time.perf_counter() - start:.6f}")


if __name__ == "__main__":
    main()
