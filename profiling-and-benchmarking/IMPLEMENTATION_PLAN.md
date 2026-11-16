# Full-free-set fast path: implementation and validation plan

## Objective

Investigate a localized L-BFGS-B optimization that applies when the free set
after the generalized Cauchy-point calculation contains every variable
(`nfree == n`). The change should also retain the existing unconstrained fast
path and must not alter SciPy's BLAS thread policy.

The intended structural improvement is to avoid `formk`, `cmprlb`, and `subsm`
when the subspace is the full space. In that case, compute the ordinary L-BFGS
direction using the stored correction pairs and the two-loop recursion.

## Safety boundary

Do not use merely "the current point is strictly inside the bounds" as the
condition. A variable can encounter a bound along the projected-gradient path
used to construct the generalized Cauchy point.

For a constrained problem, retain `cauchy` and `freev`, then select the fast
path only when they report `nfree == n`. If any variable is active, execute the
current `formk`/`cmprlb`/`subsm` path without modification.

For a problem with no bounds, preserve the existing behavior that skips the
generalized Cauchy-point calculation, but use the same full-space direction
helper.

## Proposed localized implementation

Work in `scipy/optimize/src/lbfgsb.c`.

1. Add one static helper that applies the inverse L-BFGS approximation with
   the standard two-loop recursion. Reuse an existing work-array segment of at
   least `col` elements for the saved alpha coefficients; do not allocate.
2. After the existing GCP and `freev` logic, branch on `nfree == n && col > 0`.
3. In that branch, compute the full-space trial point from the current iterate
   and gradient. Apply the same lower/upper-bound safeguard and `iword`
   semantics currently used at the end of `subsm`.
4. Continue into the existing line-search code. Do not change memory updates,
   reset behavior, convergence tests, or line-search logic.
5. Keep the existing path byte-for-byte unchanged for `nfree < n`.

The two-loop correction order must follow the logical history represented by
`head` and `col`, including the circular-buffer case. Use the diagonal entries
of `sy` for the stored step/gradient inner products and the current
`1 / theta` initial inverse-Hessian scaling.

## Correctness checks

Build separate baseline and patched SciPy trees from the same commit. Before
performance measurements, compare them on:

- unconstrained quadratics;
- bounded quadratics whose solutions and all tested iterates remain well
  inside the bounds;
- problems that begin inside the bounds but later activate a bound;
- problems beginning exactly on lower and upper bounds;
- mixed bound types (`nbd` values 0, 1, 2, and 3);
- correction-history wraparound (`iterations > maxcor`);
- cases that trigger an L-BFGS-memory reset;
- the existing SciPy L-BFGS-B test suite.

Record at least the final point, objective, projected-gradient norm, iteration
count, function/gradient evaluation counts, termination status, and whether a
bound was violated at any evaluation.

Bitwise equality is not expected because the compact solve and two-loop
recursion use different floating-point operation orders. First compare the
single direction produced from identical correction history. Then compare full
optimization runs with scale-aware tolerances. Treat changes in line-search
decisions or evaluation counts as results requiring investigation rather than
automatically accepting them.

## Instrumentation

Temporarily count, per solve:

- iterations taking the full-free-set fast path;
- iterations using the active-set path;
- transitions between those paths;
- memory resets and their source;
- fast-path results that touch a bound during the safeguard.

Keep this instrumentation out of the final patch unless maintainers request
diagnostics.

## Benchmarks

Extend `benchmark_lbfgsb.py` or add a neighboring script with at least these
families:

1. The current unconstrained zero-chain and diagonal quadratics.
2. The same problems with wide finite bounds that never become active.
3. Bounds chosen so that they become active only in later iterations.
4. Problems with a persistent mixture of free and active variables, which
   should measure the fast-path miss overhead and show no regression.

Vary `n` and `maxcor`. Run baseline and patched builds under both OpenBLAS and
MKL, with their default thread settings and with one BLAS thread. The purpose
of the thread variants is diagnosis and robustness measurement only; do not
turn them into a proposed global thread limit.

Report wall time, iteration and evaluation counts, fast-path hit rate, and
numerical-result differences. Profile representative full-free and mixed
active-set cases to confirm that time removed from `formk`/`subsm` is not merely
moved elsewhere.

## Acceptance criteria

- No bound violations.
- Existing L-BFGS-B tests pass.
- No meaningful regression when active variables force the original path.
- A measurable improvement for bounded problems with a high full-free-set hit
  rate, in addition to the unconstrained improvement.
- Numerical differences are consistent with floating-point reordering and do
  not degrade termination quality.
- No SciPy-side change to BLAS thread counts or global BLAS configuration.
