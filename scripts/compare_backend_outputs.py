"""Compare deterministic synthetic artifacts from forced CPU versus CUDA JAX.

Comparison is numerical with explicit tolerances, not byte-for-byte across
hardware backends. Image PNGs and packaging timestamps are intentionally
excluded. A mismatch is reported and causes a nonzero exit status.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def compare_value(a, b, where, *, atol, rtol):
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            raise AssertionError(f"{where}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # The existing Bayesian comparator deliberately records NaN for a
        # latent-inference field that is not defined by that model. Allow this
        # same missing-value marker at the same CSV location on both devices,
        # but do not silently accept new NaNs elsewhere or infinities.
        if math.isnan(a) or math.isnan(b):
            if (math.isnan(a) and math.isnan(b)
                    and "training_predictions.csv" in where
                    and where.endswith(".latent_before_feedback")):
                return
            raise AssertionError(f"{where}: unexpected NaN or unequal missingness")
        if not math.isfinite(a) or not math.isfinite(b):
            raise AssertionError(f"{where}: infinite value")
        if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
            raise AssertionError(f"{where}: CPU {a!r}, CUDA {b!r}")
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{where}: JSON keys differ")
        for k in a:
            compare_value(a[k], b[k], f"{where}.{k}", atol=atol, rtol=rtol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(f"{where}: list length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            compare_value(x, y, f"{where}[{i}]", atol=atol, rtol=rtol)
    elif a != b:
        raise AssertionError(f"{where}: {a!r} != {b!r}")


def compare_csv(cpu, cuda, *, atol, rtol):
    with cpu.open(newline="", encoding="utf-8") as file:
        cpu_rows = list(csv.DictReader(file))
    with cuda.open(newline="", encoding="utf-8") as file:
        cuda_rows = list(csv.DictReader(file))
    if len(cpu_rows) != len(cuda_rows):
        raise AssertionError(f"{cpu.name}: row counts differ")
    for i, (a, b) in enumerate(zip(cpu_rows, cuda_rows)):
        if a.keys() != b.keys():
            raise AssertionError(f"{cpu.name} row {i}: column names differ")
        for k in a:
            x, y = a[k], b[k]
            try:
                nx, ny = float(x), float(y)
            except ValueError:
                compare_value(x, y, f"{cpu.name}[{i}].{k}", atol=atol, rtol=rtol)
            else:
                compare_value(nx, ny, f"{cpu.name}[{i}].{k}", atol=atol, rtol=rtol)
    return len(cpu_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/backend_comparison"))
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-5)
    args = p.parse_args()
    if args.atol < 0 or args.rtol < 0:
        p.error("tolerances must be nonnegative")
    checks = (
        ("truncated_inference", "summary.json"),
        ("truncated_inference", "counterfactual_grid.csv"),
        ("auditory_learning", "summary.json"),
        ("auditory_learning", "training_predictions.csv"),
        ("auditory_learning", "no_feedback_probes.csv"),
    )
    for folder, filename in checks:
        cpu = args.root / "cpu" / folder / filename
        cuda = args.root / "cuda" / folder / filename
        if filename.endswith(".json"):
            compare_value(
                json.loads(cpu.read_text(encoding="utf-8")),
                json.loads(cuda.read_text(encoding="utf-8")),
                f"{folder}/{filename}", atol=args.atol, rtol=args.rtol,
            )
            print(f"MATCH {folder}/{filename}")
        else:
            n = compare_csv(cpu, cuda, atol=args.atol, rtol=args.rtol)
            print(f"MATCH {folder}/{filename} ({n} rows)")
    print(f"CPU and CUDA outputs agree within atol={args.atol}, rtol={args.rtol}")


if __name__ == "__main__":
    main()
