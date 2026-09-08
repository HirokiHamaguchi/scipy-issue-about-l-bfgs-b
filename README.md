# Profiling and benchmarking SciPy L-BFGS-B

## About

The purpose of this issue is to profile and benchmark the L-BFGS-B performance
problem discussed in [#26038](https://github.com/scipy/scipy/issues/26038), and
to provide information that may help improve the current L-BFGS-B
implementation.

## Benchmarking with a Conda environment linked against MKL

> The OpenBLAS issue is still in the back of our minds. But to eliminate this issue, I would suggest that you do the benchmarks with Conda environment linked to MKL library to get the true situation.

(From [this comment](https://github.com/scipy/scipy/issues/26038#issuecomment-5438749423))

## Environment setup

The development environments were prepared according to the following SciPy
documentation:

- [Contributor guide](https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html)
- [Building from source](https://scipy.github.io/devdocs/building/index.html#building-from-source)
- [Debugging linear algebra issues](https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html)

Specifically, Conda environments were created from `environment.yml`. The
original code from the issue was then run with SciPy linked against MKL
(`conda install "libblas=*=*mkl"`), rather than OpenBLAS
(`conda install "libblas=*=*openblas"`). For the comparison below, separate
OpenBLAS and MKL environments and separate SciPy build directories were used.

## Benchmark setup

The benchmark contains two unconstrained quadratic problems:

- a zero-chain quadratic, run for 300 L-BFGS-B iterations;
- a diagonal quadratic, run for 100 L-BFGS-B iterations.

Both use `maxcor=10`. Each measurement is repeated five times over dimensions
from 500 to 1,000,000. The shaded regions in the figures show the minimum and
maximum elapsed times, and the lines show the medians. The objective functions
use NumPy elementwise operations and reductions rather than a BLAS dot product,
so the comparison is not dominated by BLAS work in the objective itself.

Each problem was measured both with the BLAS backend's default thread setting
and with all BLAS thread pools limited to one thread.

## Benchmark results

### Zero-chain quadratic

![Zero-chain quadratic benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/zero_chain.png)

### Diagonal quadratic

![Diagonal quadratic benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/diagonal_quadratic.png)

With one BLAS thread, OpenBLAS and MKL are nearly indistinguishable at large
dimensions. At `n=1,000,000`, the zero-chain medians are 32.44 s for OpenBLAS
and 32.68 s for MKL; the diagonal-quadratic medians are 10.46 s and 10.44 s,
respectively.

The backend's default threading is slower than one thread for these workloads.
At `n=1,000,000`, the default-thread zero-chain times are 50.70 s for OpenBLAS
and 39.81 s for MKL, while the diagonal-quadratic times are 15.81 s and 12.52 s.
The slowdown becomes visible near the threading crossover around
`n=10,000–11,000`. MKL has lower default-thread overhead, but that difference
mostly disappears when both libraries are restricted to one thread.

## CPU profiling

Linux `perf` was used with DWARF call stacks on a long-running unconstrained
zero-chain problem. The following flame graphs show the sampled user-space CPU
stacks.

### OpenBLAS

![OpenBLAS flame graph PNG](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-openblas.png)

![OpenBLAS flame graph SVG](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-openblas.svg)

### MKL

![MKL flame graph PNG](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-mkl.png)

![MKL flame graph SVG](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-mkl.svg)

The profiles are strikingly similar:

| Function or region | OpenBLAS | MKL |
| --- | ---: | ---: |
| `setulb` shared object region | 58.18% | 59.83% |
| `subsm` | 30.54% | 30.88% |
| `formk` | 23.21% | 24.08% |
| BLAS `ddot` kernel | 10.18% | 10.03% |
| BLAS `dcopy` kernel | 4.26% | 4.42% |

The dominant sampled work is therefore in the L-BFGS-B subspace machinery,
especially `subsm` and `formk`, rather than in a backend-specific BLAS kernel.
The near-identical shares for OpenBLAS and MKL also argue against the main
bottleneck being a peculiarity of one BLAS implementation. BLAS kernels are
not negligible, but their individual shares are substantially smaller than
the two SciPy C routines.

`perf record` measures on-CPU samples. These profiles alone cannot rule out
blocked or sleeping BLAS threads; elapsed time, task-clock, and context-switch
measurements are needed for that question.

## What are `formk` and `subsm`?

The file is currently named
[`scipy/optimize/src/lbfgsb.c`](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c),
rather than `l-bfgs-b.c`.

`mainlb` calls `formk` at
[lines 852–853](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L852-L853).
The implementation is at
[lines 1882–2240](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L1882-L2240).
It forms and factorizes the small compact matrix used for subspace
minimization. It updates inner products involving the stored `S` and `Y`
correction vectors and the current free/active variable sets, then performs
Cholesky and triangular factorizations.

`mainlb` calls `subsm` at
[lines 875–876](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L875-L876).
The implementation begins at
[line 2808](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L2808).
It computes an approximate solution of the box-constrained subspace problem.
In particular, it applies the compact limited-memory representation, solves
the small triangular systems, constructs the Newton-like direction over the
free variables, and safeguards that direction against the bounds.

For an unconstrained problem, `mainlb` skips the generalized Cauchy-point
calculation after the first correction has been stored, but it still enters
the subspace-minimization path and calls `formk`, `cmprlb`, and `subsm`; see
[lines 829–876](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L829-L876).
This explains why two routines designed around the bound-constrained subspace
formulation dominate an unconstrained profile.

## Interpretation and possible next step

The current evidence suggests two separate effects:

1. Default BLAS threading introduces a substantial size-dependent overhead,
   especially for OpenBLAS, while OpenBLAS and MKL have almost identical timing
   when both are limited to one thread.
2. Independently of the BLAS backend, most sampled solver CPU time is spent in
   `formk` and `subsm` on this unconstrained workload.

This does not show that the L-BFGS correction history is reset every iteration.
In the source, `col = 0` resets occur on initialization or recovery from
factorization, triangular-solve, or line-search failures. What happens every
accepted update in the unconstrained path is the formation and solution of the
compact subspace system.

A focused follow-up would be to test an unconstrained fast path using the
standard L-BFGS two-loop recursion, bypassing `formk`, `cmprlb`, and `subsm`
only when no bounds are present. Such a change should be evaluated in a
separate build against the existing implementation, checking the final point,
objective value, gradient norm, iteration count, function evaluations, and the
existing SciPy test suite in addition to wall-clock time.
