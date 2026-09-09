# Profiling and benchmarking SciPy L-BFGS-B

## About

This issue follows up on the L-BFGS-B performance problem discussed in [#26038](https://github.com/scipy/scipy/issues/26038).
It has been pointed out that the performance of L-BFGS-B can be improved and may be affected by the BLAS backend used.
To clarify and improve the L-BFGS-B performance, in this issue, I would like to share profiles of L-BFGS-B with OpenBLAS and MKL, and test results of a possible improvement to the unconstrained optimization problems.

## Experiment overview and setup

First, I describe the experiment overview and setup.

This follows a suggestion from @ilayn in the original issue:

> The OpenBLAS issue is still in the back of our minds. But to eliminate this issue, I would suggest that you do the benchmarks with Conda environment linked to MKL library to get the true situation.

([source](https://github.com/scipy/scipy/issues/26038#issuecomment-5438749423))

I built SciPy in two separate Conda environments, linked against OpenBLAS and MKL respectively.
I ran the same benchmarks with each backend's default thread count and with BLAS limited to one thread, then profiled the common C solver code separately.

I followed the SciPy documentation for:

- [Contributor guide](https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html)
- [Building from source](https://scipy.github.io/devdocs/building/index.html#building-from-source)
- [Debugging linear algebra issues](https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html)

I created both Conda environments from `environment.yml`, selected MKL with `conda install "libblas=*=*mkl"` and OpenBLAS with `conda install "libblas=*=*openblas"`, and used a separate SciPy build directory for each.
Full commands are in [ENV_MEMO.md](https://github.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/blob/master/profiling-and-benchmarking/ENV_MEMO.md).

Measurements were taken on:

| Component | Value |
| :---: | :---: |
| Platform | WSL2, Linux 6.6.114.1-microsoft-standard-WSL2, x86-64 |
| Host CPU | Intel Core i7-8565U, 4 cores / 8 logical processors |
| Python | 3.14.7, conda-forge build |
| NumPy | 2.5.3 |
| SciPy | 2.0.0 development build |
| OpenBLAS | 0.3.34, pthreads, Haswell, default 8 threads |
| MKL | 2026.1, Intel threading layer, default 4 BLAS threads |

Threading crossovers depend on the CPU, BLAS build, scheduler, and runtime.
Their location and magnitude may differ on other machines.

## Experiment 1: Benchmark

Firstly, I ran a benchmark to compare the elapsed time of L-BFGS-B with OpenBLAS and MKL.

### 1-1: Setup

The benchmark contains two unconstrained quadratic problems from the original issue [#26038](https://github.com/scipy/scipy/issues/26038):

- a zero-chain quadratic, run for 300 L-BFGS-B iterations;
- a diagonal quadratic, run for 100 L-BFGS-B iterations.

Each was run with the backend's default thread count and with all BLAS thread pools limited to one thread.
[`benchmark_lbfgsb.py`](https://github.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/blob/master/profiling-and-benchmarking/benchmark_lbfgsb.py) uses L-BFGS-B without bounds to expose bound-oriented work that remains on the
unconstrained path.
The objectives use NumPy elementwise operations and reductions, not BLAS dot products, so BLAS work in the objective does not dominate the comparison.

### 1-2: Results

Raw results are available for [OpenBLAS](https://github.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/blob/master/profiling-and-benchmarking/results/openblas.json) and [MKL](https://github.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/blob/master/profiling-and-benchmarking/results/mkl.json).

The following figures show the median elapsed time for each problem and backend, with shading spanning the minimum and maximum. The x-axis is the number of variables `n`, and the y-axis is the elapsed time in seconds.

![Zero-chain quadratic benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/master/profiling-and-benchmarking/figures/zero_chain.png)

![Diagonal quadratic benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/master/profiling-and-benchmarking/figures/diagonal_quadratic.png)

With OpenBLAS's default thread count, both problems become markedly more expensive around `n=10,000–11,000`. No comparable crossover was observed with the MKL configuration tested here.
The crossover disappears when OpenBLAS is limited to one thread, indicating that it is associated with OpenBLAS multithreading in this setup.

At one thread, the backends are nearly indistinguishable at large dimensions.
For `n=1,000,000`, the OpenBLAS and MKL medians are 32.44 s and 32.68 s for the zero-chain problem, and 10.46 s and 10.44 s for the diagonal problem.

One thread is fastest for these workloads on this machine, but I would like to emphasize that the crossover can be hardware-dependent, and a global limit may hurt other workloads.

## Experiment 2: CPU profiling

Secondly, I used Linux `perf` to sample user-space CPU stacks for
[`profile_lbfgsb.py`](https://github.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/blob/master/profiling-and-benchmarking/profile_lbfgsb.py).

### 2-1: Setup

The workload is an unconstrained zero-chain problem with 300,000 variables, `maxcor=10`, 300 iterations, and three repetitions.
BLAS is limited to one thread to exclude the backend-specific threading effect and isolate the common C solver code.

### 2-2: Results

The benchmark medians are 8.44 s per solve with OpenBLAS and 8.36 s with MKL, excluding startup and `perf` post-processing.

| OpenBLAS | MKL |
| --- | --- |
| ![OpenBLAS flame graph](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/master/profiling-and-benchmarking/figures/perf-openblas.png) | ![MKL flame graph](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/master/profiling-and-benchmarking/figures/perf-mkl.png) |

The profiles are similar.
With BLAS limited to one thread, most samples within `setulb` fall in the inlined `subsm` and `formk` routines.
This identifies a common source of per-iteration cost in both tested backends, separately from the OpenBLAS threading crossover.

### 2-3: Implementation Details

Let us investigate the implementation of L-BFGS-B to understand why `formk` and `subsm` dominate the profile.
The implementation of L-BFGS-B is in [`scipy/optimize/src/lbfgsb.c`](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c).

Here, a correction pair consists of the step `s = x_new - x_old` and gradient difference `y = g_new - g_old`; `S` and `Y` store the most recent `s` and `y` vectors, `maxcor` is the memory limit, and `col` is the number currently stored. Free variables can move within the current subspace, whereas active variables are fixed at a bound. The generalized Cauchy point is found along the projected-gradient path and determines this free/active split. The compact matrix is a small matrix, sized in terms of `maxcor`, that represents the limited-memory Hessian information.

The key point is that, once at least one correction pair has been stored, the unconstrained path skips the generalized Cauchy-point computation but continues to use the subspace-minimization machinery.

`mainlb` conditionally calls `formk` at [lines 852–853](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L852-L853).
The implementation begins at [lines 1882](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L1882).
`formk` builds and factorizes the compact matrix used for subspace minimization.
It updates inner products involving the stored `S` and `Y` corrections and the free/active variable sets, then performs Cholesky and triangular factorizations.

When `col > 0`, `mainlb` calls `subsm` at [lines 875–876](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L875-L876).
The implementation begins at [line 2808](https://github.com/scipy/scipy/blob/41eeb590207dc4d8517abd90fc824a3a240832b5/scipy/optimize/src/lbfgsb.c#L2808).
`subsm` approximately solves the box-constrained subspace problem.
It applies the compact limited-memory representation, solves the small triangular systems, constructs a Newton-like direction over the free variables, and safeguards the direction against the bounds.

For an unconstrained problem with `col > 0`, `mainlb` calls `cmprlb` and `subsm`, and calls `formk` when the compact factorization must be updated.
This is why bound-constrained subspace routines dominate the unconstrained profile.

## Experiment 3: Prototype two-loop recursion

Finally, I tested a small prototype that replaces the unconstrained `formk`/`cmprlb`/`subsm` path with the standard L-BFGS two-loop recursion.

### 3-1: Setup

With OpenBLAS, I tested [`unconstrained_two_loop.patch`](https://github.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/blob/master/profiling-and-benchmarking/unconstrained_two_loop.patch), a small prototype that replaces the unconstrained part with the standard L-BFGS two-loop recursion when there are no bounds.

This is only a preliminary experiment, not a patch that I consider ready to propose in a pull request. Before pursuing such a change, I would need to understand the current solver's control flow, memory-reset behavior, and numerical behavior more deeply, and then evaluate the change with appropriate tests.

### 3-2: Results

Raw results are available for [baseline](https://github.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/blob/master/profiling-and-benchmarking/results/patched-comparison/baseline/openblas.json) and [prototype](https://github.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/blob/master/profiling-and-benchmarking/results/patched-comparison/patched-ver1/openblas.json).

The following figures compare the baseline and prototype under the two OpenBLAS thread settings, with shading spanning the minimum and maximum. The x-axis is the number of variables `n`, and the y-axis is the elapsed time in seconds.

![Baseline and two-loop zero-chain benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/master/profiling-and-benchmarking/figures/patched_zero_chain.png)

![Baseline and two-loop diagonal benchmark](https://raw.githubusercontent.com/HirokiHamaguchi/scipy-issue-about-l-bfgs-b/master/profiling-and-benchmarking/figures/patched_diagonal_quadratic.png)

The prototype substantially reduces per-iteration wall time.
Representative median speedups:

| Problem | BLAS threads | `n=10,000` | `n=100,000` | `n=1,000,000` |
| --- | --- | ---: | ---: | ---: |
| Zero-chain | default | 2.26x | 2.31x | 1.78x |
| Zero-chain | 1 | 1.90x | 1.83x | 1.55x |
| Diagonal quadratic | default | 1.98x | 2.00x | 1.84x |
| Diagonal quadratic | 1 | 2.09x | 1.62x | 1.54x |

The recorded result for every configuration reached the requested iteration limit.
Baseline and prototype used the same number of function evaluations for the diagonal problem.
The zero-chain prototype used 309 instead of 308.
Final objectives and gradient norms are close but not bitwise identical.
The largest differences occur for the largest diagonal problem, where both runs stop at 100 iterations without converging.
The two-loop recursion is algebraically equivalent for the same correction pairs, but different floating-point operation ordering can alter the line search. These timing results therefore do not establish numerical equivalence.

## Discussion

What we have observed is mainly the following three points:

- On this machine, the tested OpenBLAS configuration exhibits a threading crossover around $n=10^4$; no comparable crossover was observed with the tested MKL configuration.
- Independently of that crossover, single-threaded profiles show that the box-constrained subspace routines `subsm` and `formk` dominate the common solver-side CPU cost.
- For unconstrained problems, we can confirm that using the standard L-BFGS two-loop recursion can reduce the elapsed time.

We have confirmed that the elapsed time can be reduced for unconstrained problems, but I would like to hear the SciPy community's opinions.
If you would like to see further profiling results or have any opinions on the direction of the fix, please let me know.
