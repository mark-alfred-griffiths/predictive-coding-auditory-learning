"""Feedback-aware, *two-phase* predictive-coding learning.

Scientific motivation
---------------------
A perceptual latent state z is inferred while the correct category is unknown.
Later, category feedback supplies a SECOND energy term, which can change the
same hidden state before parameters are updated. Importantly, changing the
post-feedback inference budget L is not the same intervention as changing
pre-feedback inference budget K.

The energy after feedback is

   E_post(z) = 1/2 ((z-m)/prior_sd)**2 + 1/2 ((x-z)/sd)**2
               + lambda * [softplus(w*z+b) - y*(w*z+b)]

The first two terms define a generative Gaussian sensory model. The third
term is the Bernoulli negative log likelihood of the correct category under a
logistic readout. The label is never passed to pre_feedback_state().

The feedback-stage update of w and b is a gradient at *fixed* z_post, i.e.
stop_gradient(z_post). This is a fixed-activity energy parameter update, NOT
backpropagation through the entire settling trajectory. For a logistic output
it reduces to ordinary cross entropy SGD at the settled state. All sensory
parameters are frozen in this small model; learning a complete generative
network is an explicitly separate future extension.

The sensory term can optionally call the REAL JPC pc_energy_fn. We cannot
claim that JPC's package is tested until the adapter smoke test runs in an
actual compatible JPC installation. With backend='jax', the same sensory
term uses the transparent analytical Gaussian equation.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import jax
import jax.numpy as jnp

from ..schemas import AgentState, Config, causal_mean
from .analytic_gaussian import energy as gaussian_energy, gradient as gaussian_gradient, infer
from .feedback_learning import update_readout, readout_probability


@dataclass(frozen=True)
class Settling:
    """Before/after states and numerical diagnostics for one feedback event.

    `z_pre` is computed WITHOUT the label and is the state used to make the
    current choice. `z_post` may use the label, but it is never used to
    rewrite the already recorded prediction for this trial.
    """
    z_pre: jax.Array
    z_post: jax.Array
    energy_before: jax.Array
    energy_after: jax.Array
    residual_after: jax.Array


def sensory_energy(z, x, prior, sd, cfg: Config):
    """One sensory objective, selected with an explicit backend flag.

    This is the only point where the coupled experiment needs the JPC adapter.
    The JAX implementation is exact for the Gaussian model. Neither form has
    access to the category label, which makes counterfactual leakage tests
    straightforward.
    """
    if cfg.jpc_backend:
        from .jpc_adapter import jpc_energy
        return jpc_energy(z, x, prior, sd, cfg.prior_sd)
    return gaussian_energy(z, x, prior, sd, cfg.prior_sd)


def pre_feedback_state(state: AgentState, x, sd, k, cfg: Config):
    """Compute z_K using only the observation, past information and K.

    NOTE: There is deliberately NO `y` argument. The simulator may know the
    correct category, but the agent is not permitted to consult it here.
    """
    prior = causal_mean(state, cfg)
    if cfg.jpc_backend:
        from .jpc_adapter import infer_jpc
        return infer_jpc(x, prior, sd, cfg.prior_sd, k, cfg.inference_rate)
    return infer(x, prior, sd, cfg.prior_sd, k, cfg.inference_rate)


def joint_energy(z, x, prior, sd, w, b, y, cfg: Config):
    """Negative log joint objective (constants independent of z omitted).

    `jax.nn.softplus` and the linear correction implement a stable Bernoulli
    loss, avoiding direct log(sigmoid) at extreme parameter values. Weight
    `feedback_weight` determines the relative influence of category evidence
    during *post*-feedback settling; it is not a learning-rate multiplier.
    """
    logit = w*z + b
    label_loss = jax.nn.softplus(logit) - y * logit
    return sensory_energy(z, x, prior, sd, cfg) + cfg.feedback_weight * label_loss


def joint_gradient_analytic(z, x, prior, sd, w, b, y, cfg: Config):
    """Closed-form activity gradient for the explicit JAX Gaussian energy.

    Used independently of autodiff to catch sign/scaling errors. With the
    real JPC backend, we instead differentiate `joint_energy` so the activity
    dynamics are driven by JPC's actual sensory energy implementation.
    """
    return (gaussian_gradient(z, x, prior, sd, cfg.prior_sd)
            + cfg.feedback_weight * (jax.nn.sigmoid(w*z+b)-y) * w)


def feedback_settle(state: AgentState, x, sd, y, z_pre, cfg: Config,
                    steps: int | None = None) -> Settling:
    """Settle the *same* activity after observing labelled feedback.

    A global bound on the second derivative of the logistic loss is w**2/4.
    Thus the total curvature is at most

        H_max = 1/prior_sd**2 + 1/sd**2 + feedback_weight*w**2/4.

    An Euler step `post_rate/H_max` with 0 < post_rate <= 1 is conservative
    for this strictly convex, one-dimensional energy. It avoids spuriously
    enlarging steps under large sensory precision or readout weights.

    For the default JAX backend we can use the exact analytical gradient.
    On the optional JPC backend, autodiff differentiates the actual JPC
    sensory energy plus the same feedback term. No silent JPC substitution.
    """
    count = cfg.post_steps if steps is None else int(steps)
    if count < 0:
        raise ValueError('post-feedback inference steps must be nonnegative')
    prior = causal_mean(state, cfg)
    x, sd, y, z_pre = map(jnp.asarray, (x, sd, y, z_pre))
    h_max = (cfg.prior_sd**-2 + sd**-2
             + cfg.feedback_weight * state.w**2 / 4.0)
    step_size = cfg.post_rate / h_max

    if cfg.jpc_backend:
        activity_gradient = jax.grad(lambda z: joint_energy(
            z, x, prior, sd, state.w, state.b, y, cfg))
    else:
        activity_gradient = lambda z: joint_gradient_analytic(
            z, x, prior, sd, state.w, state.b, y, cfg)

    # A Python loop deliberately mirrors the scientific definition of L.
    # K and L are small explicit experimental manipulations; a compiled
    # lax.fori_loop can replace this after profiling, not before validation.
    z = z_pre
    for _ in range(count):
        z = z - step_size * activity_gradient(z)

    return Settling(z_pre=z_pre, z_post=z,
                    energy_before=joint_energy(z_pre,x,prior,sd,state.w,state.b,y,cfg),
                    energy_after=joint_energy(z,x,prior,sd,state.w,state.b,y,cfg),
                    residual_after=jnp.abs(activity_gradient(z)))


def learn_from_feedback(state: AgentState, x, sd, y, z_pre, cfg: Config,
                        *, steps: int | None = None):
    """Perform label-aware settling, then fixed-activity learning.

    The caller must reveal `y` only after `pre_feedback_state` and the
    pre-feedback choice probability have been computed. This function
    returns the new readout and diagnostics, but leaves category-prior
    updates to the existing causal `update_prior` in the trial controller.

    L=0 reproduces the earlier 'pre-state logistic update' exactly.
    This compatibility relation is checked by automated tests.
    """
    settling = feedback_settle(state,x,sd,y,z_pre,cfg,steps=steps)
    updated = update_readout(state,jax.lax.stop_gradient(settling.z_post),y,cfg)
    return updated, settling
