"""Small seeded auditory-category learning study with later no-feedback probes.

Category labels and changing category prevalences are synthetic. Current-trial
labels are revealed only after the pre-feedback prediction. All models receive
identical generated trials and probe sounds, without any human participant data.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ubel_jpc.schemas import Config
from ubel_jpc.task import generate_trials
from ubel_jpc.experiments.learning_and_probes import run_model

MODELS = ("pc_truncated", "pc_coupled", "pc_reference", "accumulator_matched",
          "adaptive_accumulator", "bayes_mapping")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write observed synthetic rows without rewriting or rounding source values."""
    if not rows:
        raise ValueError("Cannot write an empty study table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(outdir: Path, *, seed: int = 2027, trials: int = 80) -> dict:
    """Run shared stimuli for all observational models and export probe histories.

    Inputs: integer seed and positive number of training trials. Each block
    spans 20 trials and shifts the true category prevalence by construction.
    Output: deterministic numeric CSVs and a machine-readable summary. No model
    is allowed to update its parameters on probe-only observations.
    """
    if trials < 1:
        raise ValueError("trials must be positive")
    cfg = Config(seed=seed, trials=trials, block_size=20, probe_every=20,
                 post_steps=8, feedback_probability=1.0).validate()
    data = generate_trials(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    training, probes, analyses = [], [], {}
    for name in MODELS:
        simulation = run_model(name, cfg, data)
        training.extend(simulation.trial_rows)
        probes.extend(simulation.probe_rows)
        last = [r for r in simulation.probe_rows if r["t"] == trials]
        analyses[name] = {
            "n_training": len(simulation.trial_rows),
            "n_probe_predictions": len(simulation.probe_rows),
            "final_probe_probabilities": [r["p"] for r in last],
            "final_w": float(simulation.final_state.w),
            "final_b": float(simulation.final_state.b),
        }
    write_csv(outdir / "training_predictions.csv", training)
    write_csv(outdir / "no_feedback_probes.csv", probes)
    result = {
        "seed": seed,
        "config": asdict(cfg),
        "models": list(MODELS),
        "results": analyses,
        "scientific_limit": "synthetic choices, observational model equivalence, no recovery or human data",
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/auditory_learning"))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--trials", type=int, default=80)
    args = parser.parse_args()
    result = run(args.outdir, seed=args.seed, trials=args.trials)
    print(json.dumps({"seed": result["seed"], "trials": args.trials,
                      "models": result["models"],
                      "output": str(args.outdir)}, indent=2))


if __name__ == "__main__":
    main()
