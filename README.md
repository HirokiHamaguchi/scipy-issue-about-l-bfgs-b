# Profiling and benchmarking SciPy L-BFGS-B

## About

The purpose of this issue is to profile and benchmark the L-BFGS-B performance problem discussed in [#26038](https://github.com/scipy/scipy/issues/26038), and to provide information that may help improve the current L-BFGS-B implementation.

## Benchmarking with a Conda environment linked against MKL

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
We run codes in envs linked against MKL (`conda install "libblas=*=*mkl"`) and OpenBLAS (`conda install "libblas=*=*openblas"`) to see the effects of the BLAS backend. For the comparison below, separate OpenBLAS and MKL environments and separate SciPy build directories were used.
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

The benchmark contains two unconstrained quadratic problems from the original issue:

- a zero-chain quadratic, run for 300 L-BFGS-B iterations;
- a diagonal quadratic, run for 100 L-BFGS-B iterations.

Both use `maxcor=10`.
The shaded regions in the figures show the minimum and maximum elapsed times, and the lines show the medians.
The objective functions use NumPy elementwise operations and reductions rather than a BLAS dot product, so the comparison is not dominated by BLAS work in the objective itself.

Each problem was measured both with the BLAS backend's default thread setting and with all BLAS thread pools limited to one thread.

## Benchmark results

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

The workload is
[`profile_lbfgsb.py`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/profile_lbfgsb.py):
an unconstrained zero-chain problem with 300,000 variables, `maxcor=10`, 300
iterations, three repetitions, and one BLAS thread. The corresponding benchmark
medians are 8.44 s per solve with OpenBLAS and 8.36 s with MKL, so the three
solver runs take approximately 25 s per backend, excluding process startup and
`perf` post-processing. The profiling script now also prints its directly
measured total elapsed time for future runs.

| OpenBLAS | MKL |
| --- | --- |
| ![OpenBLAS flame graph SVG](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-openblas.svg) | ![MKL flame graph SVG](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-mkl.svg) |

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
The near-identical shares for OpenBLAS and MKL indicate that this solver-side
CPU cost is common to both builds. This does not conflict with the benchmark
result above: the sharp wall-time crossover near `10^4` appears
OpenBLAS-specific, while `formk` and `subsm` are separate algorithmic costs in
the common SciPy C implementation. BLAS kernels are not negligible, but their
individual shares are substantially smaller than the two SciPy C routines.

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

1. The pronounced default-thread crossover near `n=10^4` appears with
   OpenBLAS, but not with MKL, and disappears when OpenBLAS is restricted to one
   thread. This points to an OpenBLAS-specific threading effect on this machine.
2. Independently of the BLAS backend, most sampled solver CPU time is spent in
   `formk` and `subsm` on this unconstrained workload.

Because BLAS threading behavior can differ substantially across machines, I
do not think these results alone justify forcing L-BFGS-B to use one BLAS
thread. A targeted change should preserve user control over the backend and
thread count.

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
