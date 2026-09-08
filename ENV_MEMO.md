# Memo

I referenced the following links:

https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html

https://scipy.github.io/devdocs/building/index.html#building-from-source

https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html

https://scipy.github.io/devdocs/building/blas_lapack.html

## How to Set Up

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
