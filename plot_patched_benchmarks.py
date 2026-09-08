"""Plot baseline and patched L-BFGS-B benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "patched-comparison"
FIGURES = HERE / "figures"

VARIANTS = {
    "Baseline": RESULTS / "baseline" / "openblas.json",
    "Two-loop": RESULTS / "patched-ver1" / "openblas.json",
}

THREAD_PANELS = (
    ("default", "Default BLAS threads"),
    (1, "1 BLAS thread"),
)


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def draw_curve(ax: plt.Axes, rows: list[dict], label: str) -> None:
    rows.sort(key=lambda row: row["dimension"])
    dimensions = [row["dimension"] for row in rows]
    medians = [row["median_seconds"] for row in rows]
    minima = [min(row["seconds"]) for row in rows]
    maxima = [max(row["seconds"]) for row in rows]

    (line,) = ax.loglog(dimensions, medians, "o-", label=label)
    ax.fill_between(
        dimensions,
        minima,
        maxima,
        color=line.get_color(),
        alpha=0.15,
    )


def plot_problem(results: dict[str, list[dict]], problem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, (threads, title) in zip(axes, THREAD_PANELS):
        for variant, rows in results.items():
            selected = [
                row
                for row in rows
                if row["problem"] == problem
                and row["blas_threads"] == threads
            ]
            draw_curve(ax, selected, variant)

        ax.set(xlabel="Dimension n", title=title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Median elapsed time [s]")
    title = problem.replace("_", " ").title()
    fig.suptitle(f"SciPy L-BFGS-B: {title}")

    output = FIGURES / f"patched_{problem}.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    results = {
        variant: load_rows(path)
        for variant, path in VARIANTS.items()
    }
    FIGURES.mkdir(parents=True, exist_ok=True)

    for problem in ("zero_chain", "diagonal_quadratic"):
        plot_problem(results, problem)


if __name__ == "__main__":
    main()
