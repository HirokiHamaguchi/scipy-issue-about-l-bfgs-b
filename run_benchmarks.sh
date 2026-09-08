#!/usr/bin/env bash
set -euo pipefail

# Run this script from the Conda base environment.
if [[ ${CONDA_DEFAULT_ENV:-} != "base" ]]; then
    echo "Error: run from the Conda base environment." >&2
    exit 1
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
scipy=${SCIPY_DIR:-"$here/../scipy"}
mkdir -p "$here/results"
cd "$scipy"

run_one() {
    local backend=$1 env=$2 build=$3

    # The two SciPy builds must already exist.
    if [[ ! -d "$build" ]]; then
        echo "Build directory not found: $scipy/$build" >&2
        exit 1
    fi

    # Verify that the expected SciPy and BLAS backend are loaded.
    conda run --no-capture-output -n "$env" \
        spin python --build-dir="$build" --no-build -- \
        "$here/check_blas_backends.py" "$backend"

    # Python performs every experiment and writes the JSON directly.
    conda run --no-capture-output -n "$env" \
        spin python --build-dir="$build" --no-build -- \
        "$here/benchmark_lbfgsb.py"
}

run_one openblas scipy-dev-openblas build-openblas
run_one mkl scipy-dev-mkl build-mkl

# Save plots using this repository's uv environment.
uv run --project "$here" python "$here/plot_benchmarks.py"

echo "Results written to $here/results"
