#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCIPY_DIR=${SCIPY_DIR:-/home/hirok/Hobby/scipy}
CONDA_EXE=${CONDA_EXE:-/home/hirok/anaconda3/bin/conda}
CONDA_ENV=${CONDA_ENV:-scipy-dev}
BUILD_JOBS=${BUILD_JOBS:-1}

if [[ ! -x "$CONDA_EXE" ]]; then
    echo "Conda executable not found: $CONDA_EXE" >&2
    exit 1
fi
if [[ ! -d "$SCIPY_DIR/.git" ]]; then
    echo "SciPy source tree not found: $SCIPY_DIR" >&2
    exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate "$CONDA_ENV"
mkdir -p "$REPO_ROOT/results/threadpoolctl" "$REPO_ROOT/figures"

install_backend() {
    local backend=$1
    conda install -y \
        -c conda-forge \
        --override-channels \
        --solver=libmamba \
        "libblas=*=*${backend}"
}

run_in_scipy_build() {
    local command=$1
    (
        cd "$SCIPY_DIR"
        spin run --no-build "$command"
    )
}

check_backend() {
    local backend=$1
    local report="$REPO_ROOT/results/threadpoolctl/${backend}.json"
    local command
    printf -v command 'cd %q && python -m threadpoolctl -i scipy.linalg' "$REPO_ROOT"
    echo "Checking the active BLAS implementation for $backend"
    run_in_scipy_build "$command" | tee "$report"
    python - "$report" "$backend" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    pools = json.load(stream)
implementations = {
    pool.get("internal_api")
    for pool in pools
    if pool.get("user_api") == "blas"
}
expected = sys.argv[2]
if implementations != {expected}:
    raise SystemExit(
        f"Expected only BLAS implementation {expected!r}, found "
        f"{sorted(implementations)!r}"
    )
PY
}

# Configure SciPy against conda-forge's generic BLAS/LAPACK shim once. The
# implementation behind the shim can then be changed without rebuilding SciPy.
install_backend openblas
(
    cd "$SCIPY_DIR"
    spin build --clean -j "$BUILD_JOBS" \
        -S-Dblas=blas \
        -S-Dlapack=lapack \
        -S-Duse-g77-abi=true
)

for backend in openblas mkl; do
    install_backend "$backend"
    check_backend "$backend"

    printf -v benchmark_command \
        'cd %q && python %q --backend %q --output %q' \
        "$REPO_ROOT" \
        "$REPO_ROOT/benchmark.py" \
        "$backend" \
        "$REPO_ROOT/results/${backend}.npz"
    for argument in "$@"; do
        printf -v argument_quoted '%q' "$argument"
        benchmark_command+=" $argument_quoted"
    done
    run_in_scipy_build "$benchmark_command"
done

python "$REPO_ROOT/plot_benchmarks.py" \
    "$REPO_ROOT/results/openblas.npz" \
    "$REPO_ROOT/results/mkl.npz" \
    --output-dir "$REPO_ROOT/figures"
