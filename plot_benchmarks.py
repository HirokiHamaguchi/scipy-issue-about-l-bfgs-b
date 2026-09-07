"""Plot OpenBLAS and MKL results produced by benchmark.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKEND_LABELS = {"openblas": "OpenBLAS", "mkl": "MKL"}
BACKEND_COLORS = {"openblas": "tab:blue", "mkl": "tab:orange"}


def load_result(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"]))
    return arrays, metadata


def median_curve(
    arrays: dict[str, np.ndarray], experiment: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = arrays["experiment"] == experiment
    dimensions = np.unique(arrays["dimension"][mask])
    medians = np.empty(dimensions.size)
    minima = np.empty(dimensions.size)
    maxima = np.empty(dimensions.size)
    for index, dimension in enumerate(dimensions):
        values = arrays["elapsed_seconds"][mask & (arrays["dimension"] == dimension)]
        medians[index] = np.median(values)
        minima[index] = np.min(values)
        maxima[index] = np.max(values)
    return dimensions, medians, minima, maxima


def draw_curve(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    backend: str,
    experiment: str,
) -> None:
    dimensions, medians, minima, maxima = median_curve(arrays, experiment)
    color = BACKEND_COLORS[backend]
    ax.loglog(
        dimensions,
        medians,
        "o-",
        color=color,
        label=BACKEND_LABELS[backend],
    )
    ax.fill_between(dimensions, minima, maxima, color=color, alpha=0.15)


def finish_figure(fig: plt.Figure, destination: Path) -> None:
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {destination}")


def plot_single(
    results: dict[str, dict[str, np.ndarray]],
    experiment: str,
    title: str,
    destination: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for backend, arrays in results.items():
        draw_curve(ax, arrays, backend, experiment)
    ax.set(xlabel="Dimension n", ylabel="Median elapsed time [s]", title=title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    finish_figure(fig, destination)


def plot_facets(
    results: dict[str, dict[str, np.ndarray]],
    experiments: tuple[str, str],
    panel_titles: tuple[str, str],
    title: str,
    destination: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, experiment, panel_title in zip(axes, experiments, panel_titles):
        for backend, arrays in results.items():
            draw_curve(ax, arrays, backend, experiment)
        ax.set(xlabel="Dimension n", title=panel_title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Median elapsed time [s]")
    fig.suptitle(title)
    finish_figure(fig, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("results/openblas.npz"), Path("results/mkl.npz")],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, dict[str, np.ndarray]] = {}
    metadata_by_backend: dict[str, dict[str, object]] = {}
    for path in args.inputs:
        arrays, metadata = load_result(path)
        backend = str(metadata["backend"])
        if backend not in BACKEND_LABELS:
            raise ValueError(f"Unsupported backend {backend!r} in {path}")
        if backend in results:
            raise ValueError(f"Duplicate backend {backend!r}")
        results[backend] = arrays
        metadata_by_backend[backend] = metadata
    if set(results) != set(BACKEND_LABELS):
        raise ValueError("Exactly one OpenBLAS result and one MKL result are required")
    configurations = {
        json.dumps(metadata["experiments"], sort_keys=True)
        for metadata in metadata_by_backend.values()
    }
    if len(configurations) != 1:
        raise ValueError("The input files use different experiment configurations")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_single(
        results,
        "zero_chain_m10_default",
        "Zero-chain quadratic: SciPy L-BFGS-B (300 iterations, m=10)",
        args.output_dir / "zero_chain_backend_comparison.png",
    )
    plot_single(
        results,
        "diagonal_m10_default",
        "Diagonal quadratic: SciPy L-BFGS-B (100 iterations, m=10)",
        args.output_dir / "diagonal_backend_comparison.png",
    )
    plot_facets(
        results,
        ("diagonal_m3_default", "diagonal_m10_default"),
        ("Memory m=3", "Memory m=10"),
        "Diagonal quadratic: effect of L-BFGS memory",
        args.output_dir / "memory_backend_comparison.png",
    )
    plot_facets(
        results,
        ("diagonal_m10_default", "diagonal_m10_single"),
        ("Default BLAS threads", "Single BLAS thread"),
        "Diagonal quadratic: effect of BLAS thread count (m=10)",
        args.output_dir / "threads_backend_comparison.png",
    )


if __name__ == "__main__":
    main()
