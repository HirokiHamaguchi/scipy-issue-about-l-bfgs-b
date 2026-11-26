"""Internal worker for one complete SciPy build; requests arrive from Python."""

import json
import resource
import sys
import time
import traceback

from threadpoolctl import threadpool_limits

import experiment as ex


def measure(problem, memory, iterations):
    before = resource.getrusage(resource.RUSAGE_THREAD)
    cpu_start = time.thread_time_ns()
    start = time.perf_counter_ns()
    result = ex.solve(problem, memory=memory, iterations=iterations)
    seconds = (time.perf_counter_ns() - start) / 1e9
    cpu_seconds = (time.thread_time_ns() - cpu_start) / 1e9
    after = resource.getrusage(resource.RUSAGE_THREAD)
    return {"seconds": seconds, "thread_seconds": cpu_seconds,
            "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
            "involuntary_context_switches": after.ru_nivcsw - before.ru_nivcsw,
            **ex.audit(problem, result)}


def main():
    ex.pin_cpu()
    variant, problems = None, {}
    with threadpool_limits(ex.MANIFEST["runtime"]["blas_threads"], user_api="blas"):
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if variant is None:
                    variant = request["variant"]
                    runtime = ex.metadata(variant)
                if request["variant"] != variant:
                    raise ValueError("A worker cannot switch SciPy builds")
                operation = request["operation"]
                if operation == "metadata":
                    result = runtime
                elif operation in ("warmup", "measure"):
                    key = request["family"], request["n"]
                    if key not in problems:
                        problems[key] = ex.Problem(*key)
                    problem = problems[key]
                    memory, iterations = request["memory"], request["iterations"]
                    if operation == "warmup":
                        ex.solve(problem, memory=memory, iterations=iterations)
                        result = None
                    else:
                        result = measure(problem, memory, iterations)
                elif operation == "validation":
                    import validate_proposal
                    result = validate_proposal.validate_variant(variant)
                elif operation == "kernel":
                    import validate_proposal
                    result = validate_proposal.kernel_checks()
                else:
                    raise ValueError(f"Unknown worker operation: {operation}")
                print(json.dumps({"result": result}), flush=True)
            except Exception:
                # Preserve the full failure across the process boundary and stop;
                # the parent raises instead of accepting a partial result.
                print(json.dumps({"error": traceback.format_exc()}), flush=True)
                return


if __name__ == "__main__":
    main()
