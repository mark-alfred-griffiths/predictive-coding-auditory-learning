"""Deterministic K x L counterfactual experiment using original research equations.

No random draw, participant data, external files, or GPU is required. Every
condition starts from identical model parameters and observation; the label
becomes available only for the L-step post-feedback inference and update.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ubel_jpc.models.analytic_gaussian import equilibrium
from ubel_jpc.models.coupled_pc import pre_feedback_state
from ubel_jpc.experiments.phase2_diagnostics import counterfactual_grid, plot_update_grid
from ubel_jpc.schemas import Config, initial_state, causal_mean


def run(outdir: Path, *, x: float = 0.9, sd: float = 0.9) -> dict:
    """Write a fixed-condition counterfactual and independently known Gaussian reference.

    `x` is a synthetic scalar acoustic observation, `sd` is its strictly
    positive Gaussian standard deviation. K and L are solver iterations, not
    human time. Returns a compact result manifest in dimensionless units.
    """
    if not np.isfinite([x, sd]).all() or sd <= 0:
        raise ValueError("x must be finite and sd finite/positive")
    cfg = Config()
    ks, ls = (0, 1, 4, 32), (0, 1, 8, 32)
    rows = counterfactual_grid(cfg, x=x, sd=sd, pre_budgets=ks, post_budgets=ls)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "counterfactual_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_update_grid(rows, outdir / "parameter_updates.png")
    state = initial_state(cfg)
    prior = float(causal_mean(state, cfg))
    z_star = float(equilibrium(x, prior, sd, cfg.prior_sd))
    z_32 = float(pre_feedback_state(state, x, sd, 32, cfg))
    result = {
        "model": "source scalar Gaussian PC with label-aware joint feedback energy",
        "seed": None,
        "x": x, "sensory_sd": sd, "K_values": ks, "L_values": ls,
        "gaussian_equilibrium": z_star,
        "gaussian_finite_K32": z_32,
        "absolute_K32_error": abs(z_star - z_32),
        "rows": len(rows),
        "claim_scope": "synthetic fixed-parameter one-trial demonstration, not human data",
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/truncated_inference"))
    parser.add_argument("--x", type=float, default=0.9)
    parser.add_argument("--sd", type=float, default=0.9)
    args = parser.parse_args()
    print(json.dumps(run(args.outdir, x=args.x, sd=args.sd), indent=2))


if __name__ == "__main__":
    main()
