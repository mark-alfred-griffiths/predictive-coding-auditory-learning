"""Independent test gates for the two new public examples."""
import csv
import json
from pathlib import Path

import numpy as np

from ubel_jpc.schemas import Config, initial_state
from ubel_jpc.models.coupled_pc import pre_feedback_state, learn_from_feedback
from ubel_jpc.models.feedback_learning import update_readout
from examples.truncated_inference import run as run_inference
from examples.auditory_learning import run as run_auditory


def test_demo_k_l_output_and_closed_form(tmp_path):
    result = run_inference(tmp_path)
    assert result["rows"] == 32
    assert result["absolute_K32_error"] < 1e-5
    with (tmp_path / "counterfactual_grid.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {(int(r["pre_steps_K"]), int(r["post_steps_L"])) for r in rows} == {
        (k, l) for k in (0, 1, 4, 32) for l in (0, 1, 8, 32)
    }
    assert all(np.isfinite(float(r["delta_w"])) for r in rows)
    assert (tmp_path / "parameter_updates.png").stat().st_size > 1000


def test_auditory_demo_repeatability_and_equivalence(tmp_path):
    a = run_auditory(tmp_path / "a", trials=24)
    b = run_auditory(tmp_path / "b", trials=24)
    assert a == b
    assert (tmp_path / "a" / "summary.json").read_bytes() == (
        tmp_path / "b" / "summary.json").read_bytes()
    pc = a["results"]["pc_truncated"]["final_probe_probabilities"]
    acc = a["results"]["accumulator_matched"]["final_probe_probabilities"]
    assert np.allclose(pc, acc, rtol=1e-12, atol=1e-12)
    assert all(len(x["final_probe_probabilities"]) == 15 for x in a["results"].values())


def test_fixed_activity_and_label_gating():
    cfg = Config(post_steps=0)
    state = initial_state(cfg)
    z = pre_feedback_state(state, .85, .8, 4, cfg)
    update, diag = learn_from_feedback(state, .85, .8, 1., z, cfg)
    expected = update_readout(state, z, 1., cfg)
    assert np.isclose(update.w, expected.w)
    assert np.isclose(update.b, expected.b)
    assert np.isclose(diag.z_pre, diag.z_post)
    assert np.isclose(state.w, 1.)  # immutable input
