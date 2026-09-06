# Memo

https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html

https://scipy.github.io/devdocs/building/index.html#building-from-source

https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html

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
# conda install "libblas=*=*netlib"

conda install \
    -c conda-forge \
    --override-channels \
    --solver=libmamba \
    "libblas=*=*netlib"

spin build -j 1 \
    -S-Dblas=blas \
    -S-Dlapack=lapack \
    -S-Duse-g77-abi=true

spin test -s linalg

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


