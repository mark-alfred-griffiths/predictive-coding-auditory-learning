"""Separated, feedback-dependent category readout.

The perceptual state is computed BEFORE a category label is available. The
current label enters only the negative-log-likelihood and gradient update.
We use explicit JAX SGD with separate learning rates to make the update easy
to audit against its closed-form equation. Optax can be used as an independent
validation path (``optax_sgd_update``) after installing the JPC environment.

This is *not* a claim that a logistic readout is local PC synaptic plasticity.
"""
from __future__ import annotations
from dataclasses import replace
import jax
import jax.numpy as jnp
from .analytic_gaussian import infer, equilibrium, accumulator, accumulator_equivalent
from ..schemas import Config, AgentState, causal_mean


def adjusted(p, lapse):
    """Lapse-contaminated observational probability (never alters learning)."""
    return jnp.clip((1.0-lapse)*p + 0.5*lapse, 1e-8, 1-1e-8)


def readout_probability(w, b, z):
    return jax.nn.sigmoid(w*z+b)


def cross_entropy(params, z, y):
    """Binary logistic negative log likelihood, numerically stable."""
    w, b = params
    logit = w*jax.lax.stop_gradient(z)+b
    return jnp.logaddexp(0.0, logit) - y*logit


def update_readout(state: AgentState, z, y, cfg: Config, multiplier=1.0):
    """Exact gradient step at a FIXED perceptual state.

    dCE/dw = (sigmoid(w*z+b)-y) * z
    dCE/db = sigmoid(w*z+b)-y

    Clipping is intentionally not applied: artificial clipping could hide
    real differences in update size or direction. We instead detect NaNs and
    constrain experimental learning rates in the config.
    """
    z = jax.lax.stop_gradient(jnp.asarray(z))
    grad_w, grad_b = jax.grad(cross_entropy)((state.w, state.b), z, jnp.asarray(y))
    return replace(state, w=state.w-cfg.lr_w*multiplier*grad_w,
                   b=state.b-cfg.lr_b*multiplier*grad_b)


def optax_sgd_update(state: AgentState, z, y, cfg: Config):
    """Independent Optax check against the explicit SGD update above.

    Requires optional `optax` installation. We use two learning-rate groups
    for exact comparability. There is no momentum, so state reinitialisation
    does not change the mathematical update. The main engine does not require
    optax in order to run isolated JAX mathematical tests.
    """
    try:
        import optax
    except ImportError as e:
        raise RuntimeError('Install the optional JPC environment to use Optax') from e
    params = {'w': state.w, 'b': state.b}
    optim = optax.multi_transform(
        {'w': optax.sgd(cfg.lr_w), 'b': optax.sgd(cfg.lr_b)},
        {'w': 'w', 'b': 'b'}
    )
    opt_state = optim.init(params)
    grads = dict(zip(('w', 'b'), jax.grad(cross_entropy)((state.w, state.b),
                                                      jax.lax.stop_gradient(z), jnp.asarray(y))))
    delta, _ = optim.update(grads, opt_state, params)
    new = optax.apply_updates(params, delta)
    return replace(state, w=new['w'], b=new['b'])


def perceptual_state(state: AgentState, x, sd, k, cfg: Config,
                     model: str='pc_truncated'):
    """Pre-feedback inference: its interface has NO `y`/`label` argument.

    In the reference model we use the exact posterior, not a fixed large K.
    Both PC conditions therefore share precisely the same objective and
    learning rule; only inference completeness differs.
    """
    prior = causal_mean(state, cfg)
    if model == 'pc_reference':
        return equilibrium(x, prior, sd, cfg.prior_sd)
    if model == 'pc_truncated':
        if cfg.jpc_backend:
            from .jpc_adapter import infer_jpc
            return infer_jpc(x, prior, sd, cfg.prior_sd, k, cfg.inference_rate)
        return infer(x, prior, sd, cfg.prior_sd, k, cfg.inference_rate)
    if model == 'accumulator_matched':
        return accumulator_equivalent(x, prior, sd, cfg.prior_sd, k, cfg.inference_rate)
    if model in ('leaky_accumulator','adaptive_accumulator'):
        return accumulator(x, prior, sd, cfg.prior_sd, k, cfg.inference_rate)
    if model == 'fixed_delta':
        return jnp.asarray(x)
    raise ValueError(f'No perceptual feature for model {model}')
