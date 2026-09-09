"""Plot all L-BFGS-B benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

BACKEND_LABELS = {
    "openblas": "OpenBLAS",
    "mkl": "MKL",
}

BACKEND_COLORS = {
    "openblas": "tab:blue",
    "mkl": "tab:orange",
}

THREAD_PANELS = (
    ("default", "Default BLAS threads"),
    (1, "1 BLAS thread"),
)


def load_results() -> list[dict]:
    rows = []

    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        backend = data["backend"]

        for row in data["rows"]:
            row["backend"] = backend
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No JSON benchmark results found in {RESULTS_DIR}"
        )

    return rows


def draw_curve(
    ax: plt.Axes,
    rows: list[dict],
    backend: str,
) -> None:
    rows.sort(key=lambda row: row["dimension"])

    dimensions = [row["dimension"] for row in rows]
    medians = [row["median_seconds"] for row in rows]
    minima = [min(row["seconds"]) for row in rows]
    maxima = [max(row["seconds"]) for row in rows]

    color = BACKEND_COLORS.get(backend)

    ax.loglog(
        dimensions,
        medians,
        "o-",
        color=color,
        label=BACKEND_LABELS.get(backend, backend),
    )
    ax.fill_between(
        dimensions,
        minima,
        maxima,
        color=color,
        alpha=0.15,
    )


def plot_problem(rows: list[dict], problem: str) -> None:
    problem_rows = [
        row for row in rows if row["problem"] == problem
    ]
    backends = sorted({
        row["backend"] for row in problem_rows
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharey=True,
    )

    for ax, (threads, panel_title) in zip(
        axes,
        THREAD_PANELS,
    ):
        for backend in backends:
            selected = [
                row
                for row in problem_rows
                if row["backend"] == backend
                and row["blas_threads"] == threads
            ]
            if selected:
                draw_curve(ax, selected, backend)

        ax.set(
            xlabel="Dimension n",
            title=panel_title,
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Median elapsed time [s]")

    title = problem.replace("_", " ").title()
    fig.suptitle(f"SciPy L-BFGS-B: {title}")

    output = FIGURES_DIR / f"{problem}.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output}")


def main() -> None:
    rows = load_results()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    problems = sorted({
        row["problem"] for row in rows
    })

    for problem in problems:
        plot_problem(rows, problem)


if __name__ == "__main__":
    main()