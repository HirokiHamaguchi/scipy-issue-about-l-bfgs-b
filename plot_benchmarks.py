"""Plot the fixed OpenBLAS and MKL benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = {
    "OpenBLAS": HERE / "results" / "openblas.json",
    "MKL": HERE / "results" / "mkl.json",
}
COLORS = {
    "OpenBLAS": "tab:blue",
    "MKL": "tab:orange",
}
OUTPUT = HERE / "figures" / "backend_comparison.png"


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)["rows"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for backend, path in RESULTS.items():
        rows = load_rows(path)
        dimensions = [row["dimension"] for row in rows]
        medians = [row["median_seconds"] for row in rows]
        minima = [min(row["seconds"]) for row in rows]
        maxima = [max(row["seconds"]) for row in rows]

        ax.loglog(
            dimensions,
            medians,
            "o-",
            color=COLORS[backend],
            label=backend,
        )
        ax.fill_between(
            dimensions,
            minima,
            maxima,
            color=COLORS[backend],
            alpha=0.15,
        )

    ax.set(
        xlabel="Dimension n",
        ylabel="Median elapsed time [s]",
        title="SciPy L-BFGS-B: OpenBLAS versus MKL",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()