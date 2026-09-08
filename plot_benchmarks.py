"""Plot every JSON benchmark result in the results directory."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["rows"]


def draw(ax: plt.Axes, rows: list[dict], label: str) -> None:
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


def finish(fig: plt.Figure, ax: plt.Axes, output: Path) -> None:
    ax.set(
        xlabel="Dimension n",
        ylabel="Median elapsed time [s]",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    paths = sorted(RESULTS_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files found in {RESULTS_DIR}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = [(path.stem, load_rows(path)) for path in paths]

    # Save one graph for each JSON file.
    for label, rows in results:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        draw(ax, rows, label)
        ax.set_title(f"SciPy L-BFGS-B: {label}")
        finish(fig, ax, FIGURES_DIR / f"{label}.png")

    # Save a graph comparing all JSON results.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, rows in results:
        draw(ax, rows, label)

    ax.set_title("SciPy L-BFGS-B benchmark comparison")
    finish(fig, ax, FIGURES_DIR / "all_results.png")


if __name__ == "__main__":
    main()