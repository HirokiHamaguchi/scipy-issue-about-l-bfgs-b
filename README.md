# Profiling and benchmarking SciPy L-BFGS-B

## About

This issue follows up on the L-BFGS-B performance problem discussed in
[#26038](https://github.com/scipy/scipy/issues/26038). It compares OpenBLAS and
MKL, profiles the solver, and tests a possible improvement to the unconstrained
path.

## Experiment overview

I built SciPy in two separate Conda environments, linked against OpenBLAS and
MKL respectively. I ran the same benchmarks with each backend's default thread
count and with BLAS limited to one thread, then profiled the common C solver
code separately.

This follows a suggestion from @ilayn in the original issue:

> The OpenBLAS issue is still in the back of our minds. But to eliminate this issue, I would suggest that you do the benchmarks with Conda environment linked to MKL library to get the true situation.

([source](https://github.com/scipy/scipy/issues/26038#issuecomment-5438749423))

## Environment setup

I followed the SciPy documentation for:

- [Contributor guide](https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html)
- [Building from source](https://scipy.github.io/devdocs/building/index.html#building-from-source)
- [Debugging linear algebra issues](https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html)

I created both Conda environments from `environment.yml`, selected MKL with
`conda install "libblas=*=*mkl"` and OpenBLAS with
`conda install "libblas=*=*openblas"`, and used a separate SciPy build directory
for each. Full commands are in
[ENV_MEMO.md](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/ENV_MEMO.md).

Measurements were taken on:

| Component | Value |
| --- | --- |
| Platform | WSL2, Linux 6.6.114.1-microsoft-standard-WSL2, x86-64 |
| Host CPU | Intel Core i7-8565U, 4 cores / 8 logical processors |
| Python | 3.14.7, conda-forge build |
| NumPy | 2.5.3 |
| SciPy | 2.0.0 development build |
| OpenBLAS | 0.3.34, pthreads, Haswell, default 8 threads |
| MKL | 2026.1, Intel threading layer, default 4 BLAS threads |

Threading crossovers depend on the CPU, BLAS build, scheduler, and runtime.
Their location and magnitude may differ on other machines.

## Benchmark setup

[`benchmark_lbfgsb.py`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/benchmark_lbfgsb.py)
uses L-BFGS-B without bounds to expose bound-oriented work that remains on the
unconstrained path. With `ftol=0`, `gtol=0`, and fixed iteration limits, the
benchmark measures per-iteration cost rather than time to convergence.

The benchmark contains two unconstrained quadratic problems from the original issue:

- a zero-chain quadratic, run for 300 L-BFGS-B iterations;
- a diagonal quadratic, run for 100 L-BFGS-B iterations.

Both use `maxcor=10`. Each was run with the backend's default thread count and
with all BLAS thread pools limited to one thread. Lines show median elapsed
times; shading spans the minimum and maximum. The objectives use NumPy
elementwise operations and reductions, not BLAS dot products, so BLAS work in
the objective does not dominate the comparison.

## Benchmark results

Raw results:
[OpenBLAS](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/results/openblas.json)
and [MKL](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/results/mkl.json).
The recorded defaults are 8 BLAS threads for OpenBLAS and 4 for MKL.

### Zero-chain quadratic

![Zero-chain quadratic benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/zero_chain.png)

### Diagonal quadratic

![Diagonal quadratic benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/diagonal_quadratic.png)

With OpenBLAS's default thread count, both problems become markedly more
expensive around `n=10,000–11,000`. MKL shows no comparable crossover. The
OpenBLAS crossover also disappears at one thread, pointing to OpenBLAS
threading rather than the objective or an inherent L-BFGS-B cost.

At one thread, the backends are nearly indistinguishable at large dimensions.
For `n=1,000,000`, the OpenBLAS and MKL medians are 32.44 s and 32.68 s for the
zero-chain problem, and 10.46 s and 10.44 s for the diagonal problem.

One thread is fastest for these workloads on this machine, but I would not make
that a general SciPy default. The crossover is hardware- and
implementation-dependent, and a global limit may hurt other workloads. A
better fix would identify the OpenBLAS threading behavior or avoid only the
problematic calls without overriding the user's thread configuration.

## CPU profiling

I used Linux `perf` with DWARF call stacks to sample user-space CPU stacks for
[`profile_lbfgsb.py`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/profile_lbfgsb.py).
The workload is an unconstrained zero-chain problem with 300,000 variables,
`maxcor=10`, 300 iterations, and three repetitions. BLAS is limited to one
thread to exclude the backend-specific threading effect and isolate the common
C solver code.

The benchmark medians are 8.44 s per solve with OpenBLAS and 8.36 s with MKL,
or about 25 s for the three solves, excluding startup and `perf`
post-processing. The script also reports its measured total elapsed time.

| OpenBLAS | MKL |
| --- | --- |
| ![OpenBLAS flame graph](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-openblas.png) | ![MKL flame graph](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/perf-mkl.png) |

The profiles are very similar. In both, most samples within `setulb` fall in
the inlined `subsm` and `formk` routines; the visible BLAS kernels are smaller.
This common solver cost is separate from the OpenBLAS wall-time crossover near
`10^4`.

Because `perf record` samples on-CPU work, these profiles cannot rule out
blocked or sleeping BLAS threads. That requires elapsed-time, task-clock, and
context-switch measurements.

## What are `formk` and `subsm`?

The implementation is in
[`scipy/optimize/src/lbfgsb.c`](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c)
(not `l-bfgs-b.c`).

`mainlb` calls `formk` at
[lines 852–853](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L852-L853).
The implementation is at
[lines 1882–2240](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L1882-L2240).
`formk` builds and factorizes the compact matrix used for subspace
minimization. It updates inner products involving the stored `S` and `Y`
corrections and the free/active variable sets, then performs Cholesky and
triangular factorizations.

`mainlb` calls `subsm` at
[lines 875–876](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L875-L876).
The implementation begins at
[line 2808](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L2808).
`subsm` approximately solves the box-constrained subspace problem. It applies
the compact limited-memory representation, solves the small triangular
systems, constructs a Newton-like direction over the free variables, and
safeguards the direction against the bounds.

After storing the first correction for an unconstrained problem, `mainlb`
skips the generalized Cauchy point but still calls `formk`, `cmprlb`, and
`subsm`; see
[lines 829–876](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L829-L876).
This is why bound-constrained subspace routines dominate the unconstrained
profile.

## Reset tracing

An instrumented build using
[`trace_lbfgsb_resets.patch`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/trace_lbfgsb_resets.patch)
reported no `LBFGSB_RESET` events for the profiling workload. Frequent history
resets therefore do not explain this result; the repeated `formk` and `subsm`
work is the more direct lead.

## Experimental unconstrained two-loop path

With OpenBLAS, I tested
[`unconstrained_two_loop.patch`](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/unconstrained_two_loop.patch),
a small prototype that replaces the unconstrained
`formk`/`cmprlb`/`subsm` path with the standard L-BFGS two-loop recursion. It
does not change the box-constrained path.

Raw results:
[baseline](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/results/patched-comparison/baseline/openblas.json)
and [prototype](https://github.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/blob/master/results/patched-comparison/patched-ver1/openblas.json).

### Zero-chain quadratic

![Baseline and two-loop zero-chain benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/patched_zero_chain.png)

### Diagonal quadratic

![Baseline and two-loop diagonal benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-blas-in-l-bfgs-b/master/figures/patched_diagonal_quadratic.png)

The prototype substantially reduces per-iteration wall time. Representative
median speedups:

| Problem | BLAS threads | `n=10,000` | `n=100,000` | `n=1,000,000` |
| --- | --- | ---: | ---: | ---: |
| Zero-chain | default | 2.26x | 2.31x | 1.78x |
| Zero-chain | 1 | 1.90x | 1.83x | 1.55x |
| Diagonal quadratic | default | 1.98x | 2.00x | 1.84x |
| Diagonal quadratic | 1 | 2.09x | 1.62x | 1.54x |

All runs reached the requested iteration limit. Baseline and prototype used the
same number of function evaluations for the diagonal problem; the zero-chain
prototype used 309 instead of 308. Final objectives and gradient norms are
close but not bitwise identical. The largest differences occur for the largest
diagonal problem, where both runs stop at 100 iterations without converging.
The prototype clearly lowers per-iteration cost, but these results do not
establish numerical equivalence.

The candidate fast path would use the two-loop recursion only when there are no
bounds, bypassing `formk`, `cmprlb`, and `subsm`. Before proposing it for SciPy:

1. run the existing SciPy L-BFGS-B tests against the patched build;
2. add converged unconstrained problems and compare solutions, objectives,
   gradients, statuses, and evaluation counts within explicit tolerances;
3. profile the patched build to verify that `formk` and `subsm` disappear from
   the unconstrained hot path and identify the new bottleneck;
4. confirm that the box-constrained path is unchanged.
