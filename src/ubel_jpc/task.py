"""Synthetic auditory-category sequences with explicit information boundaries.

The environment knows hidden class probabilities `q_true` and category labels.
The AGENT never receives `q_true`. A label is revealed to the agent only after
its current pre-feedback prediction has been recorded.

Random streams are split by purpose so inserting more probe trials cannot
consume randomness that would otherwise affect the training schedule.
"""
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from .schemas import Config


@dataclass(frozen=True)
class Dataset:
    x: jnp.ndarray
    y: jnp.ndarray
    sd: jnp.ndarray
    k: jnp.ndarray
    q_true: jnp.ndarray  # log/analysis only; NEVER pass to predictive models
    ambiguous: jnp.ndarray
    feedback: jnp.ndarray
    probe_x: jnp.ndarray


def generate_trials(cfg: Config, seed: int | None = None) -> Dataset:
    cfg.validate()
    key = jax.random.PRNGKey(cfg.seed if seed is None else seed)
    # Independent PRNG keys ensure stability if we change one component.
    k_label, k_catnoise, k_obs, k_ambiguity, k_budget, k_feedback = jax.random.split(key, 6)
    n = cfg.trials
    block_index = (jnp.arange(n) // cfg.block_size) % 4
    probabilities = jnp.asarray([.2, .8, .2, .8])
    q_true = probabilities[block_index]
    y = jax.random.bernoulli(k_label, q_true).astype(jnp.float64)
    ambiguous = jax.random.bernoulli(k_ambiguity, 0.5, (n,))
    sd = jnp.where(ambiguous, cfg.sd_ambiguous, cfg.sd_clear)
    # Two-stage sensory process: underlying acoustic feature and fixed-duration
    # noisy observation. Extra inference iterations NEVER supply extra samples.
    acoustic = cfg.category_mean * (2 * y - 1) + cfg.category_sd * jax.random.normal(k_catnoise, (n,))
    x = acoustic + sd * jax.random.normal(k_obs, (n,))
    values = jnp.asarray(cfg.k_values)
    k = jax.random.choice(k_budget, values, shape=(n,))
    feedback = jax.random.bernoulli(k_feedback, cfg.feedback_probability, (n,))
    # These probe features are deliberately independent of the training stream.
    return Dataset(x=x, y=y, sd=sd, k=k, q_true=q_true,
                   ambiguous=ambiguous, feedback=feedback,
                   probe_x=jnp.linspace(-2.5, 2.5, 15))
