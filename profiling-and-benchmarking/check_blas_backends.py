import argparse
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
from scipy.linalg.blas import dgemm
from threadpoolctl import threadpool_info


def check_blas_backend(backend: str):
    print("scipy.__version__:", scipy.__version__)
    print("scipy.__file__   :", scipy.__file__)
    print("scipy show_config:")
    print(scipy.show_config())

    A = np.ones((100, 100))

    _ = A @ A
    _ = dgemm(alpha=1.0, a=A, b=A)

    info = threadpool_info()
    print(info)

    backends = {
        x.get("internal_api")
        for x in info
        if x.get("user_api") == "blas"
    }
    print("\nBLAS backends:", backends)

    maps = Path("/proc/self/maps").read_text().lower()
    print("libmkl loaded     :", "libmkl" in maps)
    print("libopenblas loaded:", "libopenblas" in maps)

    assert backends == {backend}, backends
    assert f"lib{backend}" in maps

    other_backend = "mkl" if backend == "openblas" else "openblas"
    assert f"lib{other_backend}" not in maps

    print(f"\nOK: {backend.upper()}-only runtime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend",
        choices=["openblas", "mkl"],
        help="Expected BLAS backend",
    )
    args = parser.parse_args()

    check_blas_backend(args.backend)