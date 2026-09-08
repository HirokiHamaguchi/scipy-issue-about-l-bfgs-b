#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_perf.sh LABEL CONDA_ENV BUILD_DIR [profile arguments ...]
label=$1
env=$2
build=$3
shift 3

here=$(cd -- "$(dirname -- "$0")" && pwd)
scipy=${SCIPY_DIR:-"$here/../scipy"}
mkdir -p "$here/results"
cd "$scipy"

# Record user-space stacks from the solver and any BLAS worker threads.
perf record -o "$here/results/perf-$label.data" -e cycles:u -F 999 -g \
    --call-graph dwarf,16384 -- \
    conda run --no-capture-output -n "$env" \
    spin python --build-dir="$build" --no-build -- \
    "$here/profile_lbfgsb.py" "$@"

# Keep a text report that can be attached to the GitHub issue.
perf report -i "$here/results/perf-$label.data" --stdio --children \
    --percent-limit 0.5 >"$here/results/perf-$label.txt"

echo "Profile written to $here/results/perf-$label.{data,txt}"
