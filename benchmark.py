"""Benchmark SciPy L-BFGS-B with the active BLAS implementation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg  # noqa: F401 - load SciPy's BLAS/LAPACK libraries
from scipy.optimize import minimize
from threadpoolctl import threadpool_info, threadpool_limits


DIMENSIONS = np.array(
    [
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
    ],
    dtype=np.int64,
)


class ZeroChainQuadratic:
    def __init__(self, dimension: int):
        self.x0 = np.zeros(dimension)
        self.function_calls = 0
        self.gradient_calls = 0

    def fun(self, x: np.ndarray) -> np.float64:
        self.function_calls += 1
        differences = x[:-1] - x[1:]
        return np.float64(0.5 * x[0] ** 2 - x[0] + 0.5 * differences @ differences)

    def jac(self, x: np.ndarray) -> np.ndarray:
        self.gradient_calls += 1
        differences = x[:-1] - x[1:]
        gradient = np.zeros_like(x)
        gradient[0] = x[0] - 1.0
        gradient[:-1] += differences
        gradient[1:] -= differences
        return gradient


class DiagonalQuadratic:
    def __init__(self, dimension: int):
        self.x0 = np.ones(dimension)
        self.weights = np.arange(1, dimension + 1, dtype=float)
        self.function_calls = 0
        self.gradient_calls = 0

    def fun(self, x: np.ndarray) -> np.float64:
        self.function_calls += 1
        return np.float64(0.5 * np.dot(self.weights * x, x))

    def jac(self, x: np.ndarray) -> np.ndarray:
        self.gradient_calls += 1
        return self.weights * x


PROBLEMS = {
    "zero_chain": ZeroChainQuadratic,
    "diagonal": DiagonalQuadratic,
}


@dataclass(frozen=True)
class Experiment:
    name: str
    problem: str
    iterations: int
    memory: int
    blas_threads: int | None


@dataclass
class TimingResult:
    experiment: str
    problem: str
    dimension: int
    requested_iterations: int
    memory: int
    blas_threads: int
    repeat: int
    elapsed_seconds: float
    iterations: int
    function_calls: int
    gradient_calls: int
    final_objective: float
    exit_status: str


EXPERIMENTS = (
    Experiment("zero_chain_m10_default", "zero_chain", 300, 10, None),
    Experiment("diagonal_m10_default", "diagonal", 100, 10, None),
    Experiment("diagonal_m3_default", "diagonal", 100, 3, None),
    Experiment("diagonal_m10_single", "diagonal", 100, 10, 1),
)


def check_blas(expected: str) -> list[dict[str, object]]:
    pools = threadpool_info()
    implementations = {
        str(pool.get("internal_api"))
        for pool in pools
        if pool.get("user_api") == "blas"
    }
    if implementations != {expected}:
        raise RuntimeError(
            f"Expected only BLAS implementation {expected!r}, found "
            f"{sorted(implementations)!r}. Full threadpool information: {pools!r}"
        )
    return pools


def run_once(
    experiment: Experiment, dimension: int, repeat: int
) -> TimingResult:
    problem = PROBLEMS[experiment.problem](dimension)
    evaluation_limit = 50 * experiment.iterations + 100
    start = time.perf_counter()
    result = minimize(
        problem.fun,
        problem.x0.copy(),
        jac=problem.jac,
        method="L-BFGS-B",
        bounds=None,
        callback=None,
        options={
            "maxcor": experiment.memory,
            "maxiter": experiment.iterations,
            "maxfun": evaluation_limit,
            "ftol": 0.0,
            "gtol": 0.0,
            "maxls": 40,
        },
    )
    elapsed_seconds = time.perf_counter() - start
    return TimingResult(
        experiment=experiment.name,
        problem=experiment.problem,
        dimension=dimension,
        requested_iterations=experiment.iterations,
        memory=experiment.memory,
        blas_threads=-1 if experiment.blas_threads is None else experiment.blas_threads,
        repeat=repeat,
        elapsed_seconds=elapsed_seconds,
        iterations=int(result.nit),
        function_calls=problem.function_calls,
        gradient_calls=problem.gradient_calls,
        final_objective=float(result.fun),
        exit_status=str(result.message),
    )


def benchmark(
    experiments: tuple[Experiment, ...], dimensions: np.ndarray, repeats: int
) -> list[TimingResult]:
    rows: list[TimingResult] = []
    for experiment in experiments:
        context = (
            threadpool_limits(limits=experiment.blas_threads, user_api="blas")
            if experiment.blas_threads is not None
            else nullcontext()
        )
        with context:
            run_once(experiment, 600, -1)
            for repeat in range(repeats):
                for dimension in dimensions:
                    row = run_once(experiment, int(dimension), repeat)
                    rows.append(row)
                    print(
                        f"{experiment.name}: repeat={repeat + 1}/{repeats}, "
                        f"n={dimension}, elapsed={row.elapsed_seconds:.6f} s",
                        flush=True,
                    )
    return rows


def save_results(
    destination: Path,
    rows: list[TimingResult],
    backend: str,
    repeats: int,
    dimensions: np.ndarray,
    pools: list[dict[str, object]],
    experiments: tuple[Experiment, ...],
) -> None:
    columns = {
        field: np.asarray([asdict(row)[field] for row in rows])
        for field in TimingResult.__dataclass_fields__
    }
    metadata = {
        "backend": backend,
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scipy_git_revision": getattr(scipy.version, "git_revision", "unknown"),
        "platform": platform.platform(),
        "processor": platform.processor() or "not reported",
        "cpu_count": os.cpu_count(),
        "threadpools": pools,
        "dimensions": dimensions.tolist(),
        "repeats": repeats,
        "experiments": [asdict(experiment) for experiment in experiments],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **columns,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("openblas", "mkl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use two dimensions, five iterations, and one repeat for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    dimensions = DIMENSIONS
    repeats = args.repeats
    experiments = EXPERIMENTS
    if args.quick:
        dimensions = np.array([500, 3_000], dtype=np.int64)
        repeats = 1
        experiments = tuple(replace(experiment, iterations=5) for experiment in EXPERIMENTS)

    pools = check_blas(args.backend)
    rows = benchmark(experiments, dimensions, repeats)
    destination = args.output or Path("results") / f"{args.backend}.npz"
    save_results(
        destination, rows, args.backend, repeats, dimensions, pools, experiments
    )


if __name__ == "__main__":
    main()
