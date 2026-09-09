# Memo

I referenced the following links:

https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html

https://scipy.github.io/devdocs/building/index.html#building-from-source

https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html

https://scipy.github.io/devdocs/building/blas_lapack.html

## Conda environments

We assume that `scipy` is cloned in `~/Hobby/scipy/` and this repository is cloned in `~/Hobby/scipy-issue-blas-in-l-bfgs-b/`.

```bash
conda env list
# only base

cd ~/Hobby/scipy/

conda env create -f environment.yml -n scipy-dev-openblas

conda activate scipy-dev-openblas

conda list | grep -E '^(libblas|libcblas|liblapack|libopenblas|openblas|mkl|mkl-devel|blas-devel)[[:space:]]'
# check only openblas is installed

# https://scipy.github.io/devdocs/building/index.html#building-from-source
spin build --build-dir=build-openblas -j2

spin python --build-dir=build-openblas --no-build -- ../scipy-issue-blas-in-l-bfgs-b/check_blas_backends.py openblas

conda create --clone scipy-dev-openblas  -n scipy-dev-mkl

conda activate scipy-dev-mkl

# https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html
conda install -c conda-forge --override-channels --solver=libmamba "libblas=*=*mkl" "blas-devel=*=*mkl"

conda list | grep -E '^(libblas|libcblas|liblapack|libopenblas|openblas|mkl|mkl-devel|blas-devel)[[:space:]]'
# check only mkl is installed

# https://github.com/scipy/scipy/blob/main/pixi.toml?utm_source=chatgpt.com
spin build --clean --build-dir=build-mkl --setup-args=-Dblas=mkl-dynamic-lp64-seq --setup-args=-Duse-g77-abi=true -j2

spin python --build-dir=build-mkl --no-build -- ../scipy-issue-blas-in-l-bfgs-b/check_blas_backends.py mkl
```

## Isolated patched environment (ver1)

`unconstrained_two_loop.patch` is the performance candidate. Use a separate
Conda environment, source clone, and build directory:

```bash
cd ~/Hobby/

git clone --no-hardlinks scipy scipy-patched-ver1

conda create --clone scipy-dev-openblas -n scipy-dev-patched-ver1
conda activate scipy-dev-patched-ver1

cd ~/Hobby/scipy-patched-ver1/

git remote set-url origin https://github.com/scipy/scipy.git
git submodule update --init

git apply --check ../scipy-issue-blas-in-l-bfgs-b/unconstrained_two_loop.patch
git apply ../scipy-issue-blas-in-l-bfgs-b/unconstrained_two_loop.patch

spin build --build-dir=build-patched-ver1 -j2

spin python --build-dir=build-patched-ver1 --no-build -- ../scipy-issue-blas-in-l-bfgs-b/check_blas_backends.py openblas
```

After setup, run the baseline and patched benchmark from Conda `base`:

```bash
cd ~/Hobby/scipy-issue-blas-in-l-bfgs-b/

conda activate base

bash run_patched_benchmarks.sh
```

## Isolated reset-trace environment

```bash
cd ~/Hobby/

git clone --no-hardlinks scipy scipy-reset-trace

conda create --clone scipy-dev-openblas -n scipy-dev-reset-trace
conda activate scipy-dev-reset-trace

cd ~/Hobby/scipy-reset-trace/

git remote set-url origin https://github.com/scipy/scipy.git
git submodule update --init

git apply --check ../scipy-issue-blas-in-l-bfgs-b/trace_lbfgsb_resets.patch
git apply ../scipy-issue-blas-in-l-bfgs-b/trace_lbfgsb_resets.patch

spin build --build-dir=build-reset-trace -j2

spin python --build-dir=build-reset-trace --no-build -- ../scipy-issue-blas-in-l-bfgs-b/check_blas_backends.py openblas
```

After setup, run the reset trace from Conda `base`:

```bash
cd ~/Hobby/scipy-issue-blas-in-l-bfgs-b/

conda activate base

bash run_reset_trace.sh
```
