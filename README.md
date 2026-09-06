# SciPy

## About

このIssueの目的は、[#26038](https://github.com/scipy/scipy/issues/26038)で議論されていた、L-BFGS-Bの性能問題に対するprofilingとベンチマークを行い、現在のL-BFGS-Bの実装を改善するための情報を提供することである。

## benchmark with Conda environment liked to MKL library

> The OpenBLAS issue is still in the back of our minds. But to eliminate this issue, I would suggest that you do the benchmarks with Conda environment linked to MKL library to get the true situation.

(From https://github.com/scipy/scipy/issues/26038#issuecomment-5438749423)

## 環境構築

以下に従って環境構築をしている。

https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html

https://scipy.github.io/devdocs/building/index.html#building-from-source

https://scipy.github.io/devdocs/dev/contributor/debugging_linalg_issues.html


具体的には、`environment.yml`に基づいたConda環境を作成し、BLAS libraryとして、OpenBLASを使用した場合(`conda install "libblas=*=*openblas"`)ではなく、MKLを使用した場合(`conda install "libblas=*=*mkl"`)に、先述のissueでも扱ったコードを実行した。
