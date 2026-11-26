# Unconstrained two-loop proposal

The [patch](unconstrained_two_loop.patch) replaces the unconstrained
`formk`/`cmprlb`/`subsm` calculation with the standard L-BFGS two-loop
recursion. Its scope is the existing branch with no bounds and `col > 0`.
It adds 37 lines and removes 2 lines.

## Mathematical and implementation scope

For the stored correction pairs, the recursion applies the inverse BFGS
approximation $H$ with initial scaling $H_0 = I/\theta$. It computes the trial
point $z = x - Hg$. The common code constructs the displacement and runs the
existing line search.

The first loop visits corrections from newest to oldest, and the second
visits them from oldest to newest. `WS` and `WY` use circular storage indexed
by `(head + i) % m`; the diagonal of `SY` uses logical history order. The
saved coefficients use the existing `WA` scratch space of `8*m` doubles.
No allocation is introduced, including when `m > n`.

The first iteration and iterations with empty history retain the current
generalized Cauchy-point path. Finite bounds retain the current subspace
calculation, even when every variable is free. History updates, `formt`,
memory-reset conditions, convergence tests, and the line search are retained.
The branch sets `iword = 0` to preserve the unconstrained diagnostic.

The two-loop recursion and compact solve are algebraically equivalent for
the same history. Their floating-point operations differ, so unconstrained
iterates, line-search decisions, and evaluation counts can differ.

## Reproducible inputs and builds

[`manifest.json`](manifest.json) pins the upstream SciPy commit, baseline and
proposal source hashes, patch hash, build options, and timing protocol.
[`environment.yml`](environment.yml) defines the Linux/WSL dependencies.
The [reproduction instructions](ENV_MEMO.md) describe user-managed setup and
the commands to run the experiment.

| File | Responsibility |
| --- | --- |
| `experiment.py` | Common problems, build paths, worker lifecycle, audits |
| `build_proposal.py` | Prerequisites, source checks, standard SciPy builds |
| `validate_proposal.py` | Dense BFGS, solver comparisons, upstream tests |
| `benchmark_proposal.py` | Balanced timings and statistical summaries |
| `experiment_worker.py` | Internal requests in one complete SciPy build |
| `tests/test_experiment.py` | Independent checks of the objective functions |

Prepare two SciPy checkouts at the pinned commit under `build/`. The build
script verifies source state, applies the patch to the proposal checkout with
GNU patch, and uses upstream `python -m spin build` for each complete SciPy.
It preserves unexpected manual edits by refusing to overwrite them.
Subproject downloads are disabled. The scripts perform no installation,
environment creation, cloning, or Git writes.

Workers use the active Python interpreter and the corresponding complete
build's installed package. They verify package location, reported SciPy
commit, and extension hash. Separate workers isolate the two builds without
replacing a module inside an unrelated SciPy installation.

Generated checkouts, build products, compiler logs, and test caches are under
`build/`, which is excluded from Git. Source pins, recipes, tests, and stable
results are tracked. No experiment uses the agent's scratch directories.

## Validation protocol

`validate_proposal.py` performs:

- 148 dense inverse-BFGS reference comparisons covering circular heads,
  partial histories, and `m > n`;
- signed-zero, subnormal, and large finite input checks;
- input-preservation and output/workspace guard checks;
- 72 paired solver cases, including 48 bounded cases requiring identical
  results and diagnostics;
- 108 solves checked against known analytic optima;
- first-iteration, explicitly infinite-bound, and fixed-coordinate probes;
- the pinned SciPy's unmodified Hessian and `setulb` test modules through
  pytest for each complete build.

Paired solver checks use `gtol=1e-10` and at most 100 iterations. Analytic
optimum checks use `gtol=1e-8` and at most 4000 iterations. Every objective
evaluation in these checks is checked against the bounds. See the
[SciPy option definitions][options].

The dependency-light unit tests independently check objective gradients by
finite differences, analytic minima, and coverage of all four bound types.
They require NumPy and run before compiling SciPy.

## Timing protocol

The manifest specifies dimensions 1000, 10,000, and 100,000; `maxcor=10`;
and 12 rounds over five problem families. These are zero-chain and diagonal
quadratics, wide finite bounds, later bound activation, and a persistent mix
of active/free variables.

The two SciPy workers are started before timing. The parent sends requests
sequentially, with balanced label positions and predecessors and shuffled
problem order. The `baseline_control` label uses the same running baseline
worker and loaded binary. Only the solve is timed inside the worker; process
startup, request transport, problem construction, and audits are excluded.

Both workers use one BLAS thread and the first available logical CPU. Set
`runtime.cpu` in the manifest to select a different available CPU explicitly.
The actual affinity and thread pools are recorded. Zero-chain uses 300
iterations; other problems use 100, with `ftol=gtol=0` for fixed workloads.

Wall time, thread CPU time, context switches, and numerical results are
retained for every solve. Summaries use paired baseline/candidate time ratios
and bootstrap intervals. Values above one favor the candidate. The intervals
describe sampled rounds and do not establish a worst-case runtime guarantee.

## Results and current status

See the [result index](results/README.md). Complete-build outputs go to
`results/full-scipy/`. They have not yet been produced: the existing local
environment needs newer Cython/Pythran and the two pinned source checkouts.

The recorded C-core experiment remains under `results/c-core-20261126/`, with
its original files unchanged. It used an older installed SciPy Python driver
and freshly compiled C extensions, and is identified accordingly. Its numeric
and timing results are not attributed to the complete-build recipe.

A forced memory-reset test, the full SciPy suite, additional BLAS/backend
configurations, and CUTEst remain separate pending checks.

[options]: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
