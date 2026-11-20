"""Sequential synthetic auditory learning and STANDARDISED later probes.

Every trial obeys this causal order:
    (1) compute prior from PREVIOUS feedback;
    (2) infer perceptual state WITHOUT knowing current label;
    (3) record probability for current decision;
    (4) optionally observe category feedback and update parameters/prior;
    (5) perform no-feedback probes (if scheduled) without touching the agent.

The environment's true category prevalence is never passed to `predict`.

One model's sampled choices do not control the category labels or learning;
labels come from the environment. This is an explicitly supervised task.
"""
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import asdict
import json
import math
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from ..schemas import Config, AgentState, initial_state, causal_probability, update_prior
from ..task import Dataset
from ..models.feedback_learning import (
    adjusted, perceptual_state, readout_probability, update_readout)
from ..models.bayesian import bayesian_probability, update_bayesian_means

MODEL_NAMES = (
    'pc_truncated', 'pc_reference', 'leaky_accumulator',
    'adaptive_accumulator', 'fixed_delta', 'bayes_mapping',
    'pc_coupled', 'accumulator_matched',
)


@dataclass(frozen=True)
class Simulation:
    model: str
    trial_rows: list[dict]
    probe_rows: list[dict]
    final_state: AgentState


def predict(state: AgentState, cfg: Config, model: str, x, sd, k):
    """Strictly pre-feedback; note there is NO label argument.

    Returns (observed-choice probability, latent state). The Bayesian model
    returns `None` for the latent as it has a different internal state family.
    """
    if model == 'bayes_mapping':
        p = bayesian_probability(state, jnp.asarray(x), jnp.asarray(sd), int(k), cfg)
        return adjusted(p, cfg.lapse), None
    if model == 'pc_coupled':
        from ..models.coupled_pc import pre_feedback_state
        z = pre_feedback_state(state, jnp.asarray(x), jnp.asarray(sd), int(k), cfg)
    else:
        z = perceptual_state(state, jnp.asarray(x), jnp.asarray(sd), int(k), cfg, model)
    p = readout_probability(state.w, state.b, z)
    return adjusted(p, cfg.lapse), z


def learn(state: AgentState, cfg: Config, model: str, x, sd, k, y, z=None):
    """Supervised update: label enters only HERE, after prediction.

    For a Bayesian learner, category-conditional acoustic means are updated
    from feedback. For delta learners, the readout is updated at the fixed
    already-inferred state z. Everyone updates the causal category prior from
    the same feedback (this avoids using the environment's true frequency).
    """
    if model == 'bayes_mapping':
        result = update_bayesian_means(state, jnp.asarray(x), jnp.asarray(sd),
                                       jnp.asarray(y), cfg)
    elif model == 'pc_coupled':
        if z is None:
            raise ValueError('Coupled learning requires a saved pre-feedback state')
        from ..models.coupled_pc import learn_from_feedback
        result, _ = learn_from_feedback(state, x, sd, y, z, cfg)
    else:
        if z is None:
            raise ValueError('A fixed pre-feedback state is required for readout learning')
        if model == 'adaptive_accumulator':
            p_internal = readout_probability(state.w, state.b, z)
            mult = 1.0 + cfg.adaptive_strength*jnp.abs(y-p_internal)
        else:
            mult = 1.0
        result = update_readout(state, z, y, cfg, multiplier=mult)
    return update_prior(result, jnp.asarray(y), cfg)


def probe(state: AgentState, cfg: Config, model: str, probe_values):
    """Evaluate choices at common K and SD, WITHOUT a state update.

    The Python-level function is explicitly read-only and consumes no RNG.
    It is safe to call multiple times or change the number of probe points.
    """
    return np.asarray([
        float(predict(state, cfg, model, float(value), cfg.probe_sd,
                      cfg.probe_k)[0]) for value in probe_values
    ])


def run_model(model: str, cfg: Config, data: Dataset, *, probe_every: int | None = None,
              initial: AgentState | None = None) -> Simulation:
    if model not in MODEL_NAMES:
        raise ValueError(f'Unknown model {model}; choices: {MODEL_NAMES}')
    cfg.validate()
    state = initial_state(cfg) if initial is None else initial
    n = len(data.x)
    every = cfg.probe_every if probe_every is None else probe_every
    trial_rows, probe_rows = [], []
    for t in range(n):
        # Fetch current trial inputs and metadata. Note `q_true` is read only
        # for RESULTS, and not used to construct the agent's prior or state.
        x, sd, k, label, has_feedback = (float(data.x[t]), float(data.sd[t]),
                                           int(data.k[t]), float(data.y[t]),
                                           bool(data.feedback[t]))
        before = float(causal_probability(state))
        w_before, b_before = float(state.w), float(state.b)

        # Stage A: label-blind inference. The true label is not an argument.
        prob, z = predict(state, cfg, model, x, sd, k)
        prob = float(prob)

        # Stage B: current correct category revealed only if feedback exists.
        # No feedback means NO updates to readout, Bayesian mapping, or prior.
        if has_feedback:
            state = learn(state, cfg, model, x, sd, k, label, z)
        trial_rows.append({
            't': t, 'model': model, 'x': x, 'sd': sd, 'k': k,
            'y': label, 'feedback': int(has_feedback),
            'ambiguous': int(bool(data.ambiguous[t])),
            'q_true_analysis_only': float(data.q_true[t]),
            'prior_before': before, 'p_before_feedback': prob,
            'latent_before_feedback': float(z) if z is not None else float('nan'),
            'w_before': w_before, 'w_after': float(state.w),
            'b_before': b_before, 'b_after': float(state.b),
            'prior_after': float(causal_probability(state)),
        })
        if every and ((t+1) % every == 0 or t == n-1):
            probs = probe(state, cfg, model, data.probe_x)
            for xi, pi in zip(np.asarray(data.probe_x), probs):
                probe_rows.append({'t': t+1, 'model': model,
                                   'probe_x': float(xi), 'p': float(pi),
                                   'common_k': cfg.probe_k,
                                   'probe_sd': cfg.probe_sd})
    return Simulation(model=model, trial_rows=trial_rows,
                      probe_rows=probe_rows, final_state=state)


def sample_choices(simulation: Simulation, seed: int):
    """Sample synthetic participant choices using an INDEPENDENT RNG stream.

    The generated labels and their causal feedback history remain unchanged.
    These simulated choices are for likelihood / recovery experiments only.
    """
    key = jax.random.PRNGKey(seed)
    probabilities = jnp.asarray([r['p_before_feedback'] for r in simulation.trial_rows])
    return np.asarray(jax.random.bernoulli(key, probabilities)).astype('int64')


def loglikelihood(rows: list[dict], choices, start=0, stop=None):
    """Sum of proper Bernoulli log likelihoods on a chronological time slice."""
    if stop is None:
        stop = len(rows)
    p = np.clip(np.asarray([r['p_before_feedback'] for r in rows[start:stop]]),
                1e-9, 1.0-1e-9)
    y = np.asarray(choices[start:stop])
    if len(y) != len(p):
        raise ValueError('The choice sequence and score window have different lengths')
    return float(np.sum(y*np.log(p)+(1-y)*np.log1p(-p)))
