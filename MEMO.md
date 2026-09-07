# Memo

https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html

https://scipy.github.io/devdocs/building/index.html#building-from-source

https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html

https://scipy.github.io/devdocs/building/blas_lapack.html

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



```bash
(base) hirok@DESKTOP-RJ3MAHN:~$ conda activate scipy-dev
(scipy-dev) hirok@DESKTOP-RJ3MAHN:~$ cd Hobby/scipy/
(scipy-dev) hirok@DESKTOP-RJ3MAHN:~/Hobby/scipy$ spin build -j 1
```

(scipy-dev) hirok@DESKTOP-RJ3MAHN:~/Hobby/scipy$ conda install "libblas=*=*netlib"
2 channel Terms of Service accepted
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: failed
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: failed

PackagesNotFoundInChannelsError: The following packages are not available from current channels:

  - libblas[build=*netlib]

Current channels:

  - defaults

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.

-----

https://docs.conda.io/projects/conda/en/stable/user-guide/concepts/conda-performance.html

```bash
cd ~/Hobby/scipy/
conda activate scipy-dev

# conda install "libblas=*=*netlib" # This causes an error
conda install \
    -c conda-forge \
    --override-channels \
    --solver=libmamba \
    "libblas=*=*netlib"

spin build --clean -j 1 \
    -S-Dblas=blas \
    -S-Dlapack=lapack \
    -S-Duse-g77-abi=true

# The following command is described in https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html
# spin test -s linalg
# We can also check the BLAS library used by SciPy with the following command:
# conda list libblas
# conda list libcblas
# conda list liblapack

conda install  \
    -c conda-forge \
    --override-channels \
    --solver=libmamba \
    "libblas=*=*mkl"

spin test -s linalg

# conda list libblas
# conda list libcblas
# conda list liblapack

conda install  \
    -c conda-forge \
    --override-channels \
    --solver=libmamba \
    "libblas=*=*openblas"
```


