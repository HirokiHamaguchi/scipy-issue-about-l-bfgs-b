"""Validate the proposal against dense BFGS and the unmodified SciPy core."""

import ctypes
import json
import os
import subprocess
import sys
from contextlib import ExitStack

import experiment as ex

np = ex.np
SEED = ex.MANIFEST["validation"]["seed"]


def kernel_checks():
    library = ctypes.CDLL(str(ex.BUILD / "direction.so"))
    kernel = library.proposal_direction
    pointer = ctypes.POINTER(ctypes.c_double)
    kernel.argtypes = [ctypes.c_int, ctypes.c_int, pointer, pointer, pointer,
                       pointer, pointer, pointer, ctypes.c_double,
                       ctypes.c_int, ctypes.c_int, pointer]
    kernel.restype = None
    rng = np.random.default_rng(SEED)
    rows = []

    def check(x, g, ws, wy, sy, theta, col, head, reference):
        n, m = ws.shape
        inputs = (x, g, ws, wy, sy)
        snapshots = [a.tobytes(order="A") for a in inputs]
        guarded = np.full(n + 2, 12345.0)
        alpha = np.full(8*m + 2, 67890.0)
        z = guarded[1:-1]
        arrays = (x, g, z, ws, wy, sy)
        kernel(n, m, *(a.ctypes.data_as(pointer) for a in arrays), theta, col,
               head, alpha[1:-1].ctypes.data_as(pointer))
        np.testing.assert_allclose(z, reference, rtol=2e-12, atol=2e-12)
        assert guarded[0] == guarded[-1] == 12345.0
        assert alpha[0] == 67890.0 and np.all(alpha[col + 1:] == 67890.0)
        assert snapshots == [a.tobytes(order="A") for a in inputs]
        return float(np.max(np.abs(z - reference)))

    for n in (1, 4, 17, 64):
        for m in (1, 3, 10):
            for col in sorted({1, max(1, m // 2), m}):
                for head in range(m):
                    weights = rng.uniform(0.5, 2.0, n)
                    theta = float(rng.uniform(0.5, 2.0))
                    ws = np.full((n, m), np.nan, order="F")
                    wy = np.full((n, m), np.nan, order="F")
                    sy = np.full((m, m), np.nan, order="F")
                    identity = np.eye(n)
                    inverse = identity / theta
                    for i in range(col):
                        s = rng.normal(size=n)
                        y = weights*s
                        ws[:, (head + i) % m], wy[:, (head + i) % m] = s, y
                        sy[i, i] = s @ y
                        rho = 1.0 / sy[i, i]
                        transform = identity - rho*np.outer(s, y)
                        inverse = transform @ inverse @ transform.T + rho*np.outer(s, s)
                    x, g = rng.normal(size=(2, n))
                    error = check(x, g, ws, wy, sy, theta, col, head, x - inverse @ g)
                    rows.append({"n": n, "m": m, "col": col, "head": head,
                                 "maximum_absolute_error": error})
    tiny = np.nextafter(0.0, 1.0)
    for x, g in (([0.0, -0.0, tiny, -tiny], [0.0, -0.0, tiny, -tiny]),
                 ([1e100, -1e100, 0.0, -0.0], [1.0, 0.0, -0.0, tiny])):
        x, g = np.array(x), np.array(g)
        ws = wy = np.array([[1.0], [0.0], [0.0], [0.0]], order="F")
        check(x, g, ws, wy, np.ones((1, 1)), 1.0, 1, 0, x - g)
    return {"histories": rows, "edge_cases": 2, "input_and_guard_checks": True}


class DiagnosticProbe:
    def __init__(self, module):
        self.module, self.records = module, []

    def setulb(self, *args):
        self.module.setulb(*args)
        task, isave = args[11], args[13]
        if task[0] == 1:
            self.records.append({"updates": int(isave[30]), "iword": int(isave[36])})


def solver_checks(variant):
    import scipy.optimize._lbfgsb_py as wrapper
    from scipy.optimize import _lbfgsb

    comparisons, accuracy = [], []
    families = (*ex.FAMILIES, "mixed_bounds")
    for n in (8, 100, 1000, 10_000):
        for memory in (1, 3, 10):
            for family in families:
                problem = ex.Problem(family, n, check_bounds=True)
                probe = DiagnosticProbe(_lbfgsb)
                previous = wrapper._lbfgsb
                wrapper._lbfgsb = probe
                try:
                    result = ex.solve(problem, memory=memory, gtol=1e-10)
                finally:
                    wrapper._lbfgsb = previous
                if variant == "proposal" and problem.bounds is None:
                    assert all(r["iword"] == 0 for r in probe.records if r["updates"] > 0)
                record = {**ex.audit(problem, result),
                          "checked_evaluations": problem.checked_evaluations,
                          "diagnostics": probe.records}
                comparisons.append({"n": n, "memory": memory, "family": family,
                                    "result": record})
    # Known analytic optima are independent of either solver's arithmetic path.
    for n in (1, 8, 31):
        for memory in (1, 3, 10):
            for family in families:
                problem = ex.Problem(family, n, check_bounds=True)
                result = ex.solve(problem, memory=memory, iterations=4000, gtol=1e-8)
                record = ex.audit(problem, result)
                error = float(np.max(np.abs(result.x - problem.solution)))
                assert error < 2e-4 and record["projected_gradient_inf_norm"] < 2e-7
                accuracy.append({"n": n, "memory": memory, "family": family,
                                 "solution_maximum_error": error, **record})
    return {"comparison_gtol": 1e-10, "comparison_maxiter": 100,
            "accuracy_gtol": 1e-8, "accuracy_maxiter": 4000,
            "comparisons": comparisons, "analytic_optima": accuracy}


def dispatch_checks():
    rows = []
    for case in ("first_iteration", "explicit_infinite_bounds", "fixed_coordinates"):
        problem = ex.Problem("diagonal", 1000, check_bounds=True)
        if case == "fixed_coordinates":
            problem.lower[:10] = problem.upper[:10] = 1.0
        if case != "first_iteration":
            problem.bounds = list(zip(problem.lower, problem.upper))
        result = ex.solve(problem, iterations=1 if case == "first_iteration" else 20)
        rows.append({"case": case, "result": ex.audit(problem, result)})
    return rows


def upstream_checks(variant):
    log = ex.BUILD / f"upstream-{variant}.log"
    report = ex.BUILD / f"upstream-{variant}.xml"
    command = [sys.executable, "-B", "-m", "pytest", "--pyargs",
               "scipy.optimize.tests.test_lbfgsb_hessinv",
               "scipy.optimize.tests.test_lbfgsb_setulb", "-q",
               f"--basetemp={ex.BUILD / f'pytest-{variant}'}",
               "-o", f"cache_dir={ex.BUILD / f'pytest-cache-{variant}'}",
               f"--junitxml={report}"]
    scratch = ex.BUILD / f"pytest-runtime-{variant}"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # SciPy's conftest places Hypothesis data under tempfile.gettempdir().
    environment["TMPDIR"] = str(scratch)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ex.HERE, stdin=subprocess.DEVNULL,
                                env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Upstream tests failed; see {log}")
    return {"passed": True, "command": command, "junit": str(report)}


def validate_variant(variant):
    return {"solver": solver_checks(variant), "dispatch": dispatch_checks(),
            "upstream": upstream_checks(variant)}


def main():
    with ExitStack() as stack:
        workers = {name: stack.enter_context(ex.VariantProcess(name)) for name in ex.VARIANTS}
        output = {"metadata": {name: worker.request("metadata") for name, worker in workers.items()},
                  "seed": SEED, "build": json.loads((ex.RESULTS / "build.json").read_text())}
        output["kernel"] = workers["proposal"].request("kernel")
        output["variants"] = {name: worker.request("validation") for name, worker in workers.items()}
    baseline, proposal = (output["variants"][name] for name in ex.VARIANTS)
    for before, after in zip(baseline["solver"]["comparisons"], proposal["solver"]["comparisons"], strict=True):
        assert (before["n"], before["memory"], before["family"]) == (after["n"], after["memory"], after["family"])
        if before["family"] not in ("zero_chain", "diagonal"):
            assert before["result"] == after["result"]
    for before, after in zip(baseline["dispatch"], proposal["dispatch"], strict=True):
        assert before["case"] == after["case"]
        if before["case"] != "explicit_infinite_bounds":
            assert before["result"] == after["result"]
    output["limitations"] = ["No forced memory-reset test", "Full SciPy suite and CUTEst not run"]
    (ex.RESULTS / "validation.json").write_text(json.dumps(output, indent=2))
    print("Solver comparisons, analytic optima, dispatch, and upstream tests passed", flush=True)


if __name__ == "__main__":
    main()
