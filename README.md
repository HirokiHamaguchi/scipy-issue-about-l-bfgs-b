# Unconstrained L-BFGS-B two-loop recursion

By Hiroki Hamaguchi.

This repository evaluates a small change to SciPy's L-BFGS-B implementation:
use the standard L-BFGS two-loop recursion when there are no bounds and at
least one correction pair is available.

The [proposal patch][patch] adds 37 lines and removes 2 lines from
`scipy/optimize/src/lbfgsb.c`. It reuses the existing workspace and line
search. Problems with finite bounds retain the existing subspace calculation.

- [Implementation, validation, and benchmark results][experiments]
- [Environment and reproduction instructions][environment]
- [Raw build, validation, and timing records][results]

The reproduction recipe compares complete baseline and proposal SciPy builds
from one pinned upstream commit. The source pin, patch hash, dependencies,
experiment code, and results are version-controlled. Native build products
are regenerated under `profiling-and-benchmarking/build/`.

Recorded C-core checks are preserved with their original environment metadata.
The complete-build recipe is implemented; execution currently requires the
environment and source preparation described in the reproduction instructions.
CUTEst benchmarks on the measurement machine remain pending.

Background: SciPy issues [#26038][issue26038] and [#26137][issue26137].

[patch]: profiling-and-benchmarking/unconstrained_two_loop.patch
[experiments]: profiling-and-benchmarking/README.md
[environment]: profiling-and-benchmarking/ENV_MEMO.md
[results]: profiling-and-benchmarking/results/README.md
[issue26038]: https://github.com/scipy/scipy/issues/26038
[issue26137]: https://github.com/scipy/scipy/issues/26137
