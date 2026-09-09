#!/usr/bin/env bash
set -euo pipefail

# Run this script from the Conda base environment.
if [[ ${CONDA_DEFAULT_ENV:-} != "base" ]]; then
    echo "Error: run from the Conda base environment." >&2
    exit 1
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
scipy=${SCIPY_DIR:-"$here/../../scipy"}
flamegraph="$here/tools/FlameGraph"

mkdir -p "$here/results" "$here/figures"
cd "$scipy"

if [[ ! -x "$flamegraph/stackcollapse-perf.pl" ]] ||
   [[ ! -x "$flamegraph/flamegraph.pl" ]]; then
    echo "FlameGraph submodule is missing." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

run_one() {
    local backend=$1
    local env=$2
    local build=$3

    local data="$here/results/perf-$backend.data"
    local flat="$here/results/perf-$backend-flat.txt"
    local callgraph="$here/results/perf-$backend-callgraph.txt"
    local svg="$here/figures/perf-$backend.svg"
    local png="$here/figures/perf-$backend.png"

    if [[ ! -d "$build" ]]; then
        echo "Build directory not found: $scipy/$build" >&2
        exit 1
    fi

    # Confirm that the intended SciPy build and BLAS backend are loaded.
    conda run --no-capture-output -n "$env" \
        spin python --build-dir="$build" --no-build -- \
        "$here/check_blas_backends.py" "$backend"

    # Record user-space CPU samples and DWARF call stacks.
    perf record \
        -o "$data" \
        -e cycles:u \
        -F 199 \
        -m 1024 \
        -g \
        --call-graph dwarf,8192 \
        -- \
        conda run --no-capture-output -n "$env" \
        spin python --build-dir="$build" --no-build -- \
        "$here/profile_lbfgsb.py"

    # Flat profile: functions where CPU samples directly landed.
    perf report \
        -i "$data" \
        --stdio \
        --no-children \
        --sort dso,symbol \
        --percent-limit 0.5 \
        >"$flat"

    # Inclusive profile: caller/callee relationships.
    perf report \
        -i "$data" \
        --stdio \
        --children \
        --sort dso,symbol \
        --call-graph graph,0.5,caller \
        --percent-limit 0.5 \
        >"$callgraph"

    # Interactive SVG flame graph.
    perf script -i "$data" \
        | "$flamegraph/stackcollapse-perf.pl" \
        | "$flamegraph/flamegraph.pl" \
            --title "SciPy L-BFGS-B: $backend" \
        >"$svg"

    # High-resolution PNG for GitHub Discussions.
    rsvg-convert \
        --width 2400 \
        --keep-aspect-ratio \
        --output "$png" \
        "$svg"

    echo "Profile written for $backend:"
    echo "  $data"
    echo "  $flat"
    echo "  $callgraph"
    echo "  $svg"
    echo "  $png"
}

run_one openblas scipy-dev-openblas build-openblas
run_one mkl scipy-dev-mkl build-mkl
