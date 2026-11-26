# Reproduction on Linux or WSL

Run the commands below from this repository's `profiling-and-benchmarking/`
directory. The recipe uses the active interpreter and relative repository
paths. It does not require a particular username or installation directory.

## Environment preparation by the user

For a new machine, create the environment yourself:

```bash
conda env create --file environment.yml
conda activate scipy-lbfgsb
```

To update the existing local development environment instead:

```bash
conda env update --name scipy-dev --file environment.yml
conda activate scipy-dev
```

The existing local environment was checked: Cython is 3.1.3 and Pythran is
0.18.0, below the pinned SciPy requirements of 3.2.0 and 0.18.1. Its Python,
NumPy, compiler, Meson, Ninja, spin, and remaining checked dependencies are
already available. No environment update has been performed by the agent.

The environment definition pins Python, NumPy, and OpenBLAS versions, and
lists source-build requirements. It is a dependency recipe rather than an
exact lock for every platform-specific Conda package. Each completed build
records the resolved Python packages; runtime records identify the actual
BLAS implementation and version. See the [SciPy contributor build guide][build].

## Source preparation by the user

The two directories below must be ordinary SciPy checkouts at the commit in
`manifest.json`, with initialized submodules. These Git write commands are
for the user to run; the experiment scripts never execute them.

```bash
mkdir -p build
git clone https://github.com/scipy/scipy.git build/scipy-baseline
git -C build/scipy-baseline checkout --detach \
  23c62b952e710e15cedffa4221aaafc49deca0e9
git -C build/scipy-baseline submodule update --init --recursive

git clone --no-hardlinks build/scipy-baseline build/scipy-proposal
git -C build/scipy-proposal checkout --detach \
  23c62b952e710e15cedffa4221aaafc49deca0e9
git -C build/scipy-proposal submodule update --init --recursive
```

The build script checks the commit, submodule status, and tracked changes.
It accepts a clean baseline and either an unpatched proposal checkout or one
already carrying exactly the recorded proposal. Unexpected edits are refused.
GNU patch applies the proposal as a source-file edit; no Git index is changed.

## Execute

```bash
python -B -m unittest discover -s tests -v
python -B build_proposal.py
python -B validate_proposal.py
python -B benchmark_proposal.py
```

Run validation to completion before starting timing. Each entry script has
a `main()` and can use the same interpreter from an editor. Experiment
parameters are ordinary entries in the tracked manifest; there is no custom
command-line interface or required experiment-specific environment variable.

`build_proposal.py` invokes upstream spin separately in the two checkouts.
Each uses its normal `build/` and `build-install/` directories. The recipe
uses OpenBLAS, two build jobs, and `--wrap-mode=nodownload`. Compiler selection
and activation flags come from the active development environment.

The generated recursion test bridge, compiler logs, and worker logs are in
the top-level `build/` directory. Pytest caches and compiler scratch files
are also configured below that directory. The worker's Python import path is
set to the requested build's installation and its provenance is verified.

## Records and a clean reproduction

Complete-build records are written to `results/full-scipy/`:

- `build.json`: source state, packages, compiler/build options, binary hashes;
- `validation.json`: independent references, paired outcomes, upstream tests;
- `benchmark.json`: raw solve timings, numerical outcomes, and runtime data;
- `summary.json` and `summary.csv`: summaries derived from those timings.

Starting with no generated files, run the environment/source preparation and
the four execution commands above. No existing native library or historical
result is required as input. Existing generated build directories may be
reused for incremental builds; unexpected source edits still stop the run.

Historical measurements under `results/c-core-20261126/` are immutable
observations of their recorded environment. Paths inside those records are
historical metadata, never executable configuration for the current recipe.

[build]: https://scipy.github.io/devdocs/dev/building/contributor.html
