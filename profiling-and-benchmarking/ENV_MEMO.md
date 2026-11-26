# Existing environment and reproduction

The scripts use the existing WSL Ubuntu `scipy-dev` environment. No downloads,
installation, environment creation, or Git writes are part of the workflow.

## Paths

| Setting | Existing location |
| --- | --- |
| Python and compiler | `/home/hirok/miniforge3/envs/scipy-dev/` |
| SciPy checkout | `/mnt/c/Users/hirok/Documents/Hobby/scipy/` |
| Driver | `/home/hirok/scipy/build-install/usr/lib/python3.13/site-packages/` |
| Repo | `/mnt/c/Users/hirok/Documents/Hobby/scipy-issue-about-l-bfgs-b/` |

These paths are explicit constants in `experiment.py`. The existing Python
driver is SciPy `1.17.0.dev0+git20250826.dd36f95`, with Python 3.13.5 and
NumPy 2.3.2. Both tested C extensions are built from the same current checkout
using the same compiler, options, and OpenBLAS library. The proposal's kernel
test additionally builds an exported wrapper around its exact recursion body.

This provides local C-core checks, rather than the coverage of a complete
current SciPy build. A future full build and CUTEst run on the measurement
machine are separate preparation steps.

## Run

From WSL:

```bash
cd /mnt/c/Users/hirok/Documents/Hobby/scipy-issue-about-l-bfgs-b/profiling-and-benchmarking
/home/hirok/miniforge3/envs/scipy-dev/bin/python -B build_proposal.py
/home/hirok/miniforge3/envs/scipy-dev/bin/python -B validate_proposal.py
/home/hirok/miniforge3/envs/scipy-dev/bin/python -B benchmark_proposal.py
```

Each script has a `main()` and can also run from an editor configured to use
that interpreter. There is no custom command-line interface. For another
machine, prepare the user's normal SciPy development environment first, then
update the path/compiler configuration and the available CPU constant.

## Outputs

Generated source copies, patched sources, compiler logs, and shared libraries
are written under the repository's
`tmp/codex-unconstrained-proposal-20261126-a2c7/` directory. Repeated runs reuse
that directory. The SciPy checkout remains unchanged.

Stable records are written under `results/unconstrained-two-loop/`:

- `build.json`: hashes, checkout commit, compiler commands, and environment;
- `validation.json`: numerical cases, dispatch probes, and upstream tests;
- `benchmark.json`: raw timings and numerical outcomes;
- `summary.json` and `summary.csv`: summaries derived from the raw timings.

The build uses GCC `-O2` without fast-math; see the
[GCC optimization documentation][gcc]. Solver options follow the
[SciPy L-BFGS-B documentation][options].

[gcc]: https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/Optimize-Options.html
[options]: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
