# Profiling and benchmarking SciPy L-BFGS-B

## About

The purpose of this issue is to profile and benchmark the L-BFGS-B performance problem discussed in [#26038](https://github.com/scipy/scipy/issues/26038), and to provide information that may help improve the current L-BFGS-B implementation.

## Experiment overview

To follow up on the discussion, I compare two independently built SciPy
environments, one linked against OpenBLAS and the other against MKL. For each
backend, I benchmark the same problems using both the backend's default thread
setting and a one-thread limit. I then profile the compiled L-BFGS-B code to
separate backend-specific threading behavior from costs in the solver's common
C implementation.

The motivation for this experiment is the following comment by @ilayn in the
original issue:

> The OpenBLAS issue is still in the back of our minds. But to eliminate this issue, I would suggest that you do the benchmarks with Conda environment linked to MKL library to get the true situation.

([@ilayn's comment](https://github.com/scipy/scipy/issues/26038#issuecomment-5438749423))

## Environment setup

The development environments were prepared according to the following SciPy documentation:

- [Contributor guide](https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html)
- [Building from source](https://scipy.github.io/devdocs/building/index.html#building-from-source)
- [Debugging linear algebra issues](https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html)

Specifically, Conda environments were created from `environment.yml`.
The benchmark was run in environments linked against MKL (`conda install "libblas=*=*mkl"`) and OpenBLAS (`conda install "libblas=*=*openblas"`). Separate environments and SciPy build directories were used.
(If necessary, see also [ENV_MEMO.md](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/ENV_MEMO.md) for the detailed setup steps.)

The measurements reported here were obtained in the following environment:

| Component | Value |
| --- | --- |
| Platform | WSL2, Linux 6.6.114.1-microsoft-standard-WSL2, x86-64 |
| Host CPU | Intel Core i7-8565U, 4 cores / 8 logical processors |
| Python | 3.14.7, conda-forge build |
| NumPy | 2.5.3 |
| SciPy | 2.0.0 development build |
| OpenBLAS | 0.3.34, pthreads, Haswell, default 8 threads |
| MKL | 2026.1, Intel threading layer, default 4 BLAS threads |

Threading crossovers can depend strongly on the CPU, BLAS build, scheduler,
and runtime environment. The numerical locations and sizes of the effects
below should therefore not be assumed to transfer unchanged to other machines.

## Benchmark setup

The
[`benchmark_lbfgsb.py`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/benchmark_lbfgsb.py)
script deliberately uses L-BFGS-B without bounds because the issue concerns
bound-oriented solver work that remains on the unconstrained path. Setting
`ftol=0` and `gtol=0` with fixed iteration limits makes this a comparison of
per-iteration cost rather than time to convergence.

The benchmark contains two unconstrained quadratic problems from the original issue:

- a zero-chain quadratic, run for 300 L-BFGS-B iterations;
- a diagonal quadratic, run for 100 L-BFGS-B iterations.

Both use `maxcor=10`.
The shaded regions in the figures show the minimum and maximum elapsed times, and the lines show the medians.
The objective functions use NumPy elementwise operations and reductions rather than a BLAS dot product, so the comparison is not dominated by BLAS work in the objective itself.

Each problem was measured both with the BLAS backend's default thread setting and with all BLAS thread pools limited to one thread.

## Benchmark results

Raw results are available for
[OpenBLAS](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/results/openblas.json)
and [MKL](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/results/mkl.json).
The recorded default BLAS thread counts are 8 for OpenBLAS and 4 for MKL.

### Zero-chain quadratic

![Zero-chain quadratic benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/zero_chain.png)

### Diagonal quadratic

![Diagonal quadratic benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/diagonal_quadratic.png)

With the default OpenBLAS thread setting, both figures show a pronounced
increase in cost around `n=10,000–11,000`. The corresponding MKL curves do not
show the same sharp crossover. Since MKL uses a different implementation, this
suggests that the abrupt overhead near `10^4` is specific to OpenBLAS rather
than an unavoidable property of L-BFGS-B.

The sharp OpenBLAS crossover also disappears when BLAS is limited to one
thread. This strongly points to OpenBLAS threading behavior, rather than the
arithmetic performed by the objective function, as the source of that feature.

With one BLAS thread, OpenBLAS and MKL are nearly indistinguishable at large dimensions.
At `n=1,000,000`, the zero-chain medians are 32.44 s for OpenBLAS and 32.68 s for MKL.
The diagonal-quadratic medians are 10.46 s and 10.44 s, respectively.

Although one thread is faster for the workloads on this particular machine, I
would personally avoid fixing the BLAS thread count to one as a general SciPy
solution. The crossover is likely hardware- and implementation-dependent, and
a global limit could penalize other machines or workloads. It seems preferable
to identify the relevant OpenBLAS threading behavior, or avoid only the
problematic calls if that can be done without overriding the user's thread
configuration.

## CPU profiling

Linux `perf` was used with DWARF call stacks on a long-running unconstrained
zero-chain problem. The following flame graphs show the sampled user-space CPU
stacks.

BLAS was limited to one thread here to avoid the backend-specific threading
effect and isolate costs in the solver's common C implementation.

The workload is
[`profile_lbfgsb.py`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/profile_lbfgsb.py):
an unconstrained zero-chain problem with 300,000 variables, `maxcor=10`, 300
iterations, three repetitions, and one BLAS thread. The corresponding benchmark
medians are 8.44 s per solve with OpenBLAS and 8.36 s with MKL, so the three
solver runs take approximately 25 s per backend, excluding process startup and
`perf` post-processing. The profiling script also prints its directly
measured total elapsed time for future runs.

| OpenBLAS | MKL |
| --- | --- |
| ![OpenBLAS flame graph](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-openblas.png) | ![MKL flame graph](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-mkl.png) |

The profiles are strikingly similar. In both, most samples attributed within
`setulb` fall in the inlined `subsm` and `formk` routines, while the visible
BLAS kernels are smaller. This solver-side cost is therefore common to both
builds and is separate from the OpenBLAS wall-time crossover near `10^4`.

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

## reset-trace instrumentation

The
[`trace_lbfgsb_resets.patch`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/trace_lbfgsb_resets.patch)
instrumented build produced zero `LBFGSB_RESET` events for the
profiling workload. Thus, frequent history resets do not explain this
reproducer's runtime. The recurring `formk` and `subsm` work remains the more
direct hypothesis.

## Experimental unconstrained two-loop path

Using OpenBLAS, I tested a small
[`unconstrained_two_loop.patch`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/unconstrained_two_loop.patch)
prototype that replaces the unconstrained
`formk`/`cmprlb`/`subsm` path with the standard L-BFGS two-loop recursion. The
box-constrained path is unchanged.

Raw results are available for the
[baseline](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/results/patched-comparison/baseline/openblas.json)
and [prototype](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/results/patched-comparison/patched-ver1/openblas.json).

### Zero-chain quadratic

![Baseline and two-loop zero-chain benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/patched_zero_chain.png)

### Diagonal quadratic

![Baseline and two-loop diagonal benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/patched_diagonal_quadratic.png)

The prototype substantially reduces the per-iteration wall time. Representative
median speedups are:

| Problem | BLAS threads | `n=10,000` | `n=100,000` | `n=1,000,000` |
| --- | --- | ---: | ---: | ---: |
| Zero-chain | default | 2.26x | 2.31x | 1.78x |
| Zero-chain | 1 | 1.90x | 1.83x | 1.55x |
| Diagonal quadratic | default | 1.98x | 2.00x | 1.84x |
| Diagonal quadratic | 1 | 2.09x | 1.62x | 1.54x |

All runs reached the same requested iteration limits. The diagonal problem also
used the same number of function evaluations in the baseline and prototype.
The zero-chain prototype used 309 evaluations instead of 308. Final objectives
and gradient norms are close, but not bitwise identical; differences are most
visible for the largest diagonal problem, where neither run has converged and
both stop at the 100-iteration limit. These measurements therefore establish a
large reduction in per-iteration cost, but do not by themselves establish full
numerical equivalence.

A focused follow-up would be to test an unconstrained fast path using the
standard L-BFGS two-loop recursion, bypassing `formk`, `cmprlb`, and `subsm`
only when no bounds are present. Before proposing the prototype for SciPy, the
next steps should be:

1. run the existing SciPy L-BFGS-B tests against the patched build;
2. add converged unconstrained problems and compare solutions, objectives,
   gradients, statuses, and evaluation counts within explicit tolerances;
3. profile the patched build to verify that `formk` and `subsm` disappear from
   the unconstrained hot path and identify the new bottleneck;
4. confirm that the box-constrained path is unchanged.
