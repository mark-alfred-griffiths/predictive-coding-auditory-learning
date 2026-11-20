"""Immutable research settings and agent state.

Why immutable? During counterfactual experiments we repeatedly clone the same
starting point and vary *only* the inference budget. Frozen records reduce the
risk of hidden mutation between experimental conditions.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal
import jax.numpy as jnp

ModelName = Literal[
    'pc_truncated', 'pc_reference', 'leaky_accumulator',
    'adaptive_accumulator', 'fixed_delta', 'bayes_mapping',
    'pc_coupled', 'accumulator_matched',
]


@dataclass(frozen=True)
class Config:
    seed: int = 2027
    trials: int = 240
    block_size: int = 60
    probe_every: int = 12
    category_mean: float = 0.9
    category_sd: float = 0.20
    sd_clear: float = 0.50
    sd_ambiguous: float = 1.15
    prior_sd: float = 1.0
    inference_rate: float = 0.35  # fraction of remaining state error removed per step
    prior_strength: float = 0.8
    lr_w: float = 0.075
    lr_b: float = 0.045
    adaptive_strength: float = 0.9
    prior_decay: float = 0.96   # causal exponential category-count forgetting
    feedback_probability: float = 1.0
    lapse: float = 0.02
    probe_sd: float = 0.80
    probe_k: int = 32
    k_values: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    # Feedback-aware extension: L is deliberately independent of K.
    # L=0 collapses to the original fixed-state readout update.
    feedback_weight: float = 1.0
    post_steps: int = 8
    post_rate: float = 0.5
    jpc_backend: bool = False  # explicit opt-in, NEVER silently substitute JAX

    def validate(self):
        if self.trials < 1 or self.block_size < 1 or self.probe_every < 0:
            raise ValueError('trials/block_size must be positive; probe_every >= 0')
        if not self.k_values or any(k < 0 for k in self.k_values):
            raise ValueError('k_values must be nonempty, nonnegative integers')
        for value in (self.sd_clear, self.sd_ambiguous, self.prior_sd,
                      self.category_sd, self.probe_sd):
            if value <= 0:
                raise ValueError('all standard deviations must be positive')
        if not (0 < self.inference_rate < 1):
            raise ValueError('inference_rate must lie strictly between 0 and 1')
        if not (0 <= self.feedback_probability <= 1 and 0 <= self.lapse < 1):
            raise ValueError('feedback_probability and lapse must be valid probabilities')
        if not (0 < self.prior_decay <= 1):
            raise ValueError('prior_decay must lie in (0,1]')
        if self.feedback_weight < 0 or self.post_steps < 0 or not (0 < self.post_rate <= 1):
            raise ValueError('feedback parameters out of range')
        if min(self.lr_w, self.lr_b) < 0:
            raise ValueError('learning rates must be nonnegative')
        return self


@dataclass(frozen=True)
class AgentState:
    # Mapping from the perceptual representation to category probability.
    w: jnp.ndarray
    b: jnp.ndarray
    # Exponentially discounted *feedback counts*, NOT the environment's hidden q.
    alpha: jnp.ndarray
    beta: jnp.ndarray
    # Bayesian competitor's learned class-conditional acoustic means and precisions.
    class_mu: jnp.ndarray
    class_precision: jnp.ndarray


def initial_state(cfg: Config) -> AgentState:
    # Both category counts start at 2, expressing a neutral, weak prior.
    return AgentState(
        w=jnp.asarray(1.0), b=jnp.asarray(0.0),
        alpha=jnp.asarray(2.0), beta=jnp.asarray(2.0),
        class_mu=jnp.asarray([-cfg.category_mean, cfg.category_mean]),
        class_precision=jnp.asarray([1.0, 1.0]),
    )


def causal_probability(state: AgentState):
    """Prior for NEXT decision; reads only previously processed feedback."""
    return state.alpha / (state.alpha + state.beta)


def causal_mean(state: AgentState, cfg: Config):
    """Map a learned category prior into the mean of the acoustic latent state.

    This is explicitly a modelling assumption, not a result of JPC itself.
    """
    return cfg.prior_strength * cfg.category_mean * (2 * causal_probability(state) - 1)


def update_prior(state: AgentState, label: float, cfg: Config):
    """Apply forgetting and assimilate the label, *after* the choice is made."""
    return replace(state,
                   alpha=2.0 + cfg.prior_decay * (state.alpha - 2.0) + label,
                   beta=2.0 + cfg.prior_decay * (state.beta - 2.0) + (1.0-label))
