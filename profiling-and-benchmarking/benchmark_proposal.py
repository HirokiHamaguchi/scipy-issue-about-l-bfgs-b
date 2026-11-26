"""Paired timing of the SciPy baseline, proposal, and identical-binary control."""

import csv
import itertools
import json
import random
import statistics
from contextlib import ExitStack

import experiment as ex

SETTINGS = ex.MANIFEST["benchmark"]
DIMENSIONS = tuple(SETTINGS["dimensions"])
REPEATS = SETTINGS["repeats"]
MEMORY = SETTINGS["memory"]
SEED = SETTINGS["seed"]
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
    if REPEATS <= 0 or REPEATS % 6:
        raise ValueError("Balanced timings require a positive multiple of six rounds")
    rng = random.Random(SEED)
    cases = [(family, n) for n in DIMENSIONS for family in ex.FAMILIES]
    orders = {}
    for case in cases:
        orders[case] = []
        for _ in range(REPEATS // 6):
            block = list(itertools.permutations(LABELS))
            rng.shuffle(block)
            orders[case].extend(block)
    rows, reference_records = [], {}
    with ExitStack() as stack:
        workers = {name: stack.enter_context(ex.VariantProcess(name)) for name in ex.VARIANTS}
        workers["baseline_control"] = workers["baseline"]
        info = {"runtime": {name: workers[name].request("metadata") for name in ex.VARIANTS},
                "seed": SEED, "repeats": REPEATS, "memory": MEMORY,
                "dimensions": DIMENSIONS, "families": ex.FAMILIES,
                "iterations": {f: SETTINGS["zero_chain_iterations"] if f == "zero_chain"
                               else SETTINGS["other_iterations"] for f in ex.FAMILIES},
                "ftol": 0.0, "gtol": 0.0,
                "build": json.loads((ex.RESULTS / "build.json").read_text()),
                "interpretation": "Fixed iteration workloads; bootstrap intervals describe these rounds only"}
        for family, n in cases:
            for name in ex.VARIANTS:
                workers[name].request("warmup", family=family, n=n, memory=MEMORY, iterations=5)
        for repeat in range(REPEATS):
            shuffled = cases.copy()
            rng.shuffle(shuffled)
            for family, n in shuffled:
                for position, label in enumerate(orders[family, n][repeat]):
                    measured = workers[label].request(
                        "measure", family=family, n=n, memory=MEMORY,
                        iterations=info["iterations"][family])
                    timing_keys = {"seconds", "thread_seconds", "voluntary_context_switches",
                                   "involuntary_context_switches"}
                    record = {key: value for key, value in measured.items() if key not in timing_keys}
                    # Deterministic output is checked outside the timed interval.
                    key = (family, n, "baseline" if label == "baseline_control" else label)
                    if key in reference_records:
                        assert record == reference_records[key]
                    reference_records[key] = record
                    rows.append({"family": family, "n": n, "round": repeat,
                                 "position": position, "variant": label, **measured})
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
