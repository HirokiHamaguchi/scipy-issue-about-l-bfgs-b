#!/usr/bin/env bash
set -euo pipefail

if [[ ${CONDA_DEFAULT_ENV:-} != "base" ]]; then
    echo "Error: run from the Conda base environment." >&2
    exit 1
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
scipy="$here/../../scipy-reset-trace"
build=build-reset-trace
output="$here/results/lbfgsb-resets.txt"

if [[ ! -d "$scipy/$build" ]]; then
    echo "Build directory not found: $scipy/$build" >&2
    exit 1
fi

mkdir -p "$here/results"
cd "$scipy"

conda run --no-capture-output -n scipy-dev-reset-trace \
    spin python --build-dir="$build" --no-build -- \
    "$here/check_blas_backends.py" openblas

conda run --no-capture-output -n scipy-dev-reset-trace \
    spin python --build-dir="$build" --no-build -- \
    "$here/profile_lbfgsb.py" \
    2>"$output"

count=$(grep -c '^LBFGSB_RESET ' "$output" || true)
printf 'LBFGSB_RESET_COUNT=%s\n' "$count" >>"$output"
echo "L-BFGS-B resets: $count"
echo "Trace written to $output"
