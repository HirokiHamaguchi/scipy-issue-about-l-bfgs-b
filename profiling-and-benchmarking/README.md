# Unconstrained two-loop proposal

The [patch](unconstrained_two_loop.patch) replaces the unconstrained
`formk`/`cmprlb`/`subsm` calculation with the standard L-BFGS two-loop
recursion. Its scope is the existing branch with no bounds and `col > 0`.
It adds 37 lines and removes 2 lines.

## Mathematical and implementation scope

For the stored correction pairs, the recursion applies the inverse BFGS
approximation $H$ with initial scaling $H_0 = I/\theta$. It computes the trial
point $z = x - Hg$. The existing common code then constructs the displacement
and runs the line search.

The first loop visits corrections from newest to oldest, and the second
visits them from oldest to newest. `WS` and `WY` use circular storage indexed
by `(head + i) % m`; the diagonal entries of `SY` use logical history order.
The saved coefficients use the existing `WA` scratch space, which contains
`8*m` doubles. No allocation is introduced, including when `m > n`.

The first iteration and iterations with empty history retain the current
generalized Cauchy-point path. Finite bounds retain the current subspace
calculation, even when every variable is free. History updates, `formt`,
memory-reset conditions, convergence tests, and the line search are retained.
The unconstrained branch sets `iword = 0` to preserve the diagnostic for a
subspace solution within the box.

The recursion and compact solve are algebraically equivalent for the same
history, but their floating-point operations differ. Full optimization runs
can consequently differ in iterates, line-search decisions, and evaluation
counts. Bitwise equality is checked for the bounded path and first iteration;
unconstrained correctness also uses independent mathematical references.

## Files and reproduction

| File | Purpose |
| --- | --- |
| `unconstrained_two_loop.patch` | The proposed SciPy source change |
| `experiment.py` | Paths, problem definitions, solver loading, result audits |
| `build_proposal.py` | Isolated baseline/proposal builds and source records |
| `validate_proposal.py` | Numerical, workspace, dispatch, and upstream checks |
| `benchmark_proposal.py` | Paired timings and identical-binary control |

Follow [ENV_MEMO.md](ENV_MEMO.md) to run the three scripts in the existing
environment. They neither edit the SciPy checkout nor perform Git writes.
Builds and generated C sources are under the configured task directory in
`tmp/`. Stable records are under
[`results/unconstrained-two-loop/`](results/unconstrained-two-loop/).

The baseline C source is checked by its normalized SHA-256 before patching.
The build record contains the patch/source/binary hashes, checkout commit,
compiler, commands, and Python-driver version. A different source hash requires
reviewing the patch and updating the explicit baseline configuration.

## Numerical validation

[`validation.json`](results/unconstrained-two-loop/validation.json) records:

- 148 positive-curvature histories compared with dense inverse-BFGS updates,
  including all circular heads, partially filled histories, and `m > n`;
- signed-zero, subnormal, and large finite inputs;
- guards around output/workspace arrays and checks for unchanged inputs;
- 72 paired solver cases across dimensions and history lengths, including
  48 bounded cases checked for bitwise-identical results and diagnostics;
- 108 solves checked against known analytic optima;
- first-iteration, explicitly infinite-bound, and fixed-coordinate probes;
- three unmodified upstream Hessian tests on each of the two builds.

The paired solver cases use `gtol=1e-10` and at most 100 iterations. The known
optimum checks use `gtol=1e-8` and at most 4000 iterations. Every objective
evaluation in these solver checks is checked against the bounds. Option
semantics follow the [SciPy L-BFGS-B documentation][options].

The recorded checks passed. The maximum absolute error against dense BFGS was
approximately `6.7e-15`; the maximum coordinate error in the analytic-optimum
checks was approximately `6.5e-7`.

The driver is an existing older SciPy installation, rather than a full build
of the current checkout. The current checkout's `test_lbfgsb_setulb.py`
requires a newer `HAS_ILP64` utility symbol unavailable in that driver. That
test, the full SciPy suite, and a forced memory-reset test remain pending.

## Timing protocol

The benchmark compares `baseline` and `proposal`. `baseline_control` calls
the same loaded module as `baseline`, measuring timing variation without any
implementation difference.

- Dimensions: 1000, 10,000, and 100,000; `maxcor=10`.
- Problems: zero-chain and diagonal quadratics, wide finite bounds, a bound
  activated after the start, and a persistent mix of active/free variables.
- Twelve rounds, balanced execution positions and predecessors, with
  shuffled problem order in each round.
- One BLAS thread and affinity to logical CPU 2.
- Zero-chain: 300 iterations; other problems: 100 iterations.
- `ftol=gtol=0` for these fixed-iteration timing workloads.
- Wall time, thread CPU time, context switches, and numerical results retained
  for each solve; audits run outside the timed interval.

[`benchmark.json`](results/unconstrained-two-loop/benchmark.json) contains the
raw observations and metadata. [`summary.csv`][summary] and
[`summary.json`](results/unconstrained-two-loop/summary.json) report paired
speedups as baseline time divided by candidate time. Values above one favor
the candidate. Bootstrap intervals describe these sampled rounds; they do
not establish a worst-case or cross-machine runtime guarantee. The
identical-binary control should be considered alongside the proposal.

## Recorded local results

The completed run contains 540 timed solves: 15 problem/dimension combinations,
12 rounds, and three labels. Every solve reached its requested iteration
limit. The proposal's median paired speedups were:

| Problem | `n=1000` | `n=10,000` | `n=100,000` |
| --- | ---: | ---: | ---: |
| Zero-chain | 1.91 | 1.84 | 1.19 |
| Diagonal | 2.01 | 1.80 | 1.24 |
| Wide finite bounds | 0.96 | 0.96 | 0.98 |
| Later bound activation | 1.01 | 1.08 | 1.00 |
| Mixed active/free | 0.98 | 0.99 | 0.99 |

Across these cases, the identical-binary control's median speedups ranged
from approximately 0.92 to 1.07. Bounded results, evaluation counts, and
termination states matched the baseline bitwise in all timing rounds.
Small timing changes on that path need to be interpreted with the control
and the intervals in the raw summary. These observations do not prove a
zero-overhead or worst-case runtime guarantee.

CUTEst measurements and additional BLAS/backend configurations remain pending.

[options]: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
[summary]: results/unconstrained-two-loop/summary.csv
