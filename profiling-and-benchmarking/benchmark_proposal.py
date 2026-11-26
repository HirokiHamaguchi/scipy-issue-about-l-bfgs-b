"""Paired timing of the SciPy baseline, proposal, and identical-binary control."""

import csv
import itertools
import json
import os
import random
import resource
import statistics
import time

import experiment as ex


DIMENSIONS = (1000, 10_000, 100_000)
REPEATS = 12
MEMORY = 10
SEED = 261127
LABELS = ("baseline", "proposal", "baseline_control")


def interval(values, rng):
    values = ex.np.asarray(values)
    samples = rng.choice(values, size=(2000, len(values)), replace=True)
    low, high = ex.np.quantile(ex.np.median(samples, axis=1), (0.025, 0.975))
    return float(low), float(high)


def summarize(rows):
    summaries = []
    rng = ex.np.random.default_rng(SEED)
    for n in DIMENSIONS:
        for family in ex.FAMILIES:
            group = [r for r in rows if r["n"] == n and r["family"] == family]
            index = {(r["round"], r["variant"]): r for r in group}
            for label in LABELS:
                samples = [index[i, label] for i in range(REPEATS)]
                speedups = [index[i, "baseline"]["seconds"] / index[i, label]["seconds"]
                            for i in range(REPEATS)]
                low, high = interval(speedups, rng)
                summaries.append({
                    "n": n, "family": family, "variant": label,
                    "median_seconds": statistics.median(r["seconds"] for r in samples),
                    "median_paired_speedup": statistics.median(speedups),
                    "bootstrap_95_low": low, "bootstrap_95_high": high,
                    "minimum_paired_speedup": min(speedups), "maximum_paired_speedup": max(speedups),
                    "iterations_min": min(r["iterations"] for r in samples),
                    "iterations_max": max(r["iterations"] for r in samples),
                    "nfev_min": min(r["nfev"] for r in samples),
                    "nfev_max": max(r["nfev"] for r in samples),
                    "objective": samples[0]["objective"],
                    "bound_violation": max(r["bound_violation"] for r in samples),
                })
    return summaries


def main():
    os.sched_setaffinity(0, {ex.CPU})
    modules = {name: ex.load_module(name) for name in ex.VARIANTS}
    modules["baseline_control"] = modules["baseline"]
    rng = random.Random(SEED)
    cases = [(family, n) for n in DIMENSIONS for family in ex.FAMILIES]
    problems = {(family, n): ex.Problem(family, n) for family, n in cases}
    orders = {}
    for case in cases:
        orders[case] = []
        for _ in range(REPEATS // 6):
            block = list(itertools.permutations(LABELS))
            rng.shuffle(block)
            orders[case].extend(block)
    rows, reference_records = [], {}
    with ex.threadpool_limits(1, user_api="blas"):
        info = {**ex.metadata(), "affinity": sorted(os.sched_getaffinity(0)),
                "seed": SEED, "repeats": REPEATS, "memory": MEMORY,
                "dimensions": DIMENSIONS, "families": ex.FAMILIES,
                "iterations": {f: 300 if f == "zero_chain" else 100 for f in ex.FAMILIES},
                "ftol": 0.0, "gtol": 0.0,
                "build": json.loads((ex.RESULTS / "build.json").read_text()),
                "interpretation": "Fixed iteration workloads; bootstrap intervals describe these rounds only"}
        for problem in problems.values():
            for name in ex.VARIANTS:
                ex.solve(modules[name], problem, iterations=5)
        for repeat in range(REPEATS):
            shuffled = cases.copy()
            rng.shuffle(shuffled)
            for family, n in shuffled:
                problem = problems[family, n]
                for position, label in enumerate(orders[family, n][repeat]):
                    before = resource.getrusage(resource.RUSAGE_THREAD)
                    cpu_start = time.thread_time_ns()
                    start = time.perf_counter_ns()
                    result = ex.solve(modules[label], problem, memory=MEMORY,
                                      iterations=300 if family == "zero_chain" else 100)
                    elapsed = (time.perf_counter_ns() - start) / 1e9
                    cpu_elapsed = (time.thread_time_ns() - cpu_start) / 1e9
                    after = resource.getrusage(resource.RUSAGE_THREAD)
                    record = ex.audit(problem, result)
                    # Deterministic output is checked outside the timed interval.
                    key = (family, n, "baseline" if label == "baseline_control" else label)
                    if key in reference_records:
                        assert record == reference_records[key]
                    reference_records[key] = record
                    rows.append({"family": family, "n": n, "round": repeat,
                                 "position": position, "variant": label, "seconds": elapsed,
                                 "thread_seconds": cpu_elapsed,
                                 "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
                                 "involuntary_context_switches": after.ru_nivcsw - before.ru_nivcsw,
                                 **record})
            (ex.RESULTS / "benchmark.json").write_text(json.dumps({"metadata": info, "rows": rows}, indent=2))
            print(f"Completed timing round {repeat + 1}/{REPEATS}", flush=True)
    summary = summarize(rows)
    (ex.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    with (ex.RESULTS / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
