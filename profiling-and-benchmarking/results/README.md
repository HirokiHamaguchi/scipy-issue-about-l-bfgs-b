# Experiment records

## Complete SciPy builds

[`full-scipy/`](full-scipy/) is the output location of the current reproduction
recipe. Build and experiment results are pending user-managed environment
and source preparation. No C-core result has been relabeled as a complete
SciPy-build measurement.

## Recorded C-core experiment

[`c-core-20261126/`](c-core-20261126/) preserves five original records, unchanged
from their measured contents: `build.json`, `validation.json`, `benchmark.json`,
`summary.json`, and `summary.csv`.

This experiment used Python 3.13.5, NumPy 2.3.2, an older installed SciPy
Python driver, and freshly compiled baseline/proposal C extensions. The
metadata specifies exact versions, source/patch hashes, compiler commands,
thread configuration, and the limitations of that combination.

The records contain 148 dense BFGS histories, 72 paired solver cases,
108 analytic-optimum solves, dispatch probes, six upstream Hessian-test
executions, and 540 timed solves. The validation checks passed in that
environment. These are records of that execution; the current complete-build
recipe has not reproduced them yet.

Recorded median paired timing ratios, baseline time divided by proposal time:

| Problem | `n=1000` | `n=10,000` | `n=100,000` |
| --- | ---: | ---: | ---: |
| Zero-chain | 1.91 | 1.84 | 1.19 |
| Diagonal | 2.01 | 1.80 | 1.24 |
| Wide finite bounds | 0.96 | 0.96 | 0.98 |
| Later bound activation | 1.01 | 1.08 | 1.00 |
| Mixed active/free | 0.98 | 0.99 | 0.99 |

The identical-binary control's median ratios ranged approximately from 0.92
to 1.07. Interpret small changes with the raw intervals and control;
these observations do not establish a worst-case runtime guarantee.
