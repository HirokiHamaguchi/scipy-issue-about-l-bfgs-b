"""Validate the proposal against dense BFGS and the unmodified SciPy core."""

import ctypes
import json
import os
import runpy

import experiment as ex


np = ex.np
SEED = 261126


def kernel_checks():
    library = ctypes.CDLL(str(ex.WORK / "build/proposal/direction.so"))
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


def solver_checks(modules):
    comparisons, accuracy = [], []
    families = (*ex.FAMILIES, "mixed_bounds")
    for n in (8, 100, 1000, 10_000):
        for memory in (1, 3, 10):
            for family in families:
                records = {}
                for name, module in modules.items():
                    problem = ex.Problem(family, n, check_bounds=True)
                    probe = DiagnosticProbe(module)
                    try:
                        result = ex.solve(probe, problem, memory=memory, gtol=1e-10)
                    except AssertionError as error:
                        error.add_note(f"variant={name}, memory={memory}, family={family}, n={n}")
                        raise
                    if name == "proposal" and problem.bounds is None:
                        assert all(r["iword"] == 0 for r in probe.records if r["updates"] > 0)
                    records[name] = {**ex.audit(problem, result),
                                     "checked_evaluations": problem.checked_evaluations,
                                     "diagnostics": probe.records}
                if family not in ("zero_chain", "diagonal"):
                    assert records["baseline"] == records["proposal"]
                comparisons.append({"n": n, "memory": memory, "family": family,
                                    "variants": records})
    # Known analytic optima are independent of either solver's arithmetic path.
    for n in (1, 8, 31):
        for memory in (1, 3, 10):
            for family in families:
                for name, module in modules.items():
                    problem = ex.Problem(family, n, check_bounds=True)
                    result = ex.solve(module, problem, memory=memory, iterations=4000, gtol=1e-8)
                    record = ex.audit(problem, result)
                    error = float(np.max(np.abs(result.x - problem.solution)))
                    assert error < 2e-4 and record["projected_gradient_inf_norm"] < 2e-7
                    accuracy.append({"n": n, "memory": memory, "family": family,
                                     "variant": name, "solution_maximum_error": error, **record})
    return {"comparison_gtol": 1e-10, "comparison_maxiter": 100,
            "accuracy_gtol": 1e-8, "accuracy_maxiter": 4000,
            "comparisons": comparisons, "analytic_optima": accuracy}


def dispatch_checks(modules):
    rows = []
    for case in ("first_iteration", "explicit_infinite_bounds", "fixed_coordinates"):
        records = {}
        for name, module in modules.items():
            problem = ex.Problem("diagonal", 1000, check_bounds=True)
            if case == "fixed_coordinates":
                problem.lower[:10] = problem.upper[:10] = 1.0
            if case != "first_iteration":
                problem.bounds = ex.Bounds(problem.lower, problem.upper)
            result = ex.solve(module, problem, iterations=1 if case == "first_iteration" else 20)
            records[name] = ex.audit(problem, result)
        if case != "explicit_infinite_bounds":
            assert records["baseline"] == records["proposal"]
        rows.append({"case": case, "variants": records})
    return rows


def upstream_checks(modules):
    # Use the checkout's unmodified tests. The older Python driver cannot import
    # test_lbfgsb_setulb.py, which requires the newer HAS_ILP64 utility symbol.
    tests = runpy.run_path(str(ex.SCIPY_SOURCE / "scipy/optimize/tests/test_lbfgsb_hessinv.py"))
    previous = ex.wrapper._lbfgsb
    rows = []
    try:
        for name, module in modules.items():
            ex.wrapper._lbfgsb = module
            for test in ("test_1", "test_2", "test_3"):
                tests[test]()
                rows.append({"variant": name, "test": test, "passed": True})
    finally:
        ex.wrapper._lbfgsb = previous
    return rows


def main():
    os.sched_setaffinity(0, {ex.CPU})
    modules = {name: ex.load_module(name) for name in ex.VARIANTS}
    with ex.threadpool_limits(1, user_api="blas"):
        output = {"metadata": ex.metadata(), "seed": SEED,
                  "build": json.loads((ex.RESULTS / "build.json").read_text())}
        output["kernel"] = kernel_checks()
        print("Dense BFGS and workspace checks passed", flush=True)
        output["solver"] = solver_checks(modules)
        output["dispatch"] = dispatch_checks(modules)
        output["upstream"] = upstream_checks(modules)
    output["limitations"] = ["No forced memory-reset test", "Full SciPy suite and CUTEst not run",
                              "Upstream setulb test requires a newer Python driver"]
    (ex.RESULTS / "validation.json").write_text(json.dumps(output, indent=2))
    print("Solver comparisons, analytic optima, dispatch, and upstream tests passed", flush=True)


if __name__ == "__main__":
    main()
