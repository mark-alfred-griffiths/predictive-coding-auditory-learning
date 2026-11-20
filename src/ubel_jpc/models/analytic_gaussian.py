"""Gaussian sensory inference: exact reference and finite-step JAX inference.

For a single latent acoustic feature z, the energy is

    E(z;x,m) = (z-m)^2/(2*sigma_prior^2) + (x-z)^2/(2*sd^2).

It has a known posterior mode z_star (also the posterior mean, in this special
Gaussian model). A stable, precision-normalised Euler step is

    z <- z - rate/H * dE/dz, where H = 1/sigma_prior^2 + 1/sd^2.

Therefore each iteration removes `rate` of the remaining state error. This
is an intentional normalisation: different sensory noise values do NOT lead
to numerically unstable step sizes. In the JPC adapter we compare the same
objective against JPC's public predictive-coding energy (see jpc_adapter.py).

A lower energy is NOT equivalent to calibrated epistemic uncertainty, and K
cannot be interpreted as milliseconds of human perception.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def energy(z, x, m, sd, prior_sd):
    return .5 * ((z - m) / prior_sd) ** 2 + .5 * ((x - z) / sd) ** 2


def gradient(z, x, m, sd, prior_sd):
    return (z - m) / prior_sd**2 + (z - x) / sd**2


def equilibrium(x, m, sd, prior_sd):
    """Posterior mean/mode; independent mathematical convergence reference."""
    a, b = prior_sd**-2, sd**-2
    return (a*m + b*x) / (a+b)


@jax.jit
def _infer(x, m, sd, prior_sd, k, rate):
    """Fixed-K scan; the K argument may vary without recompilation."""
    curvature = prior_sd**-2 + sd**-2
    step_size = rate / curvature

    def step(_, z):
        return z - step_size * gradient(z, x, m, sd, prior_sd)

    return jax.lax.fori_loop(0, k, step, m)


def infer(x, m, sd, prior_sd, k, rate=0.35):
    return _infer(jnp.asarray(x), jnp.asarray(m), jnp.asarray(sd),
                  jnp.asarray(prior_sd), jnp.asarray(k, dtype=jnp.int32),
                  jnp.asarray(rate))


def trajectory(x, m, sd, prior_sd, max_k=40, rate=0.35):
    """Returns states including K=0; useful to audit inference dynamics."""
    z = jnp.asarray(m)
    states = [z]
    for _ in range(max_k):
        z = z - rate * gradient(z, x, m, sd, prior_sd) / (prior_sd**-2 + sd**-2)
        states.append(z)
    zs = jnp.stack(states)
    en = jax.vmap(lambda zz: energy(zz, x, m, sd, prior_sd))(zs)
    residuals = jax.vmap(lambda zz: jnp.abs(gradient(zz, x, m, sd, prior_sd)))(zs)
    return zs, en, residuals


def accumulator(x, m, sd, prior_sd, k, rate):
    """Leaky evidence accumulator: ignores contextual *state* prior.

    For m=0, it is EXACTLY mathematically equivalent to Gaussian inference.
    Keeping this equivalent case is mandatory for honest model comparison.
    """
    sensory_weight = prior_sd**2 / (prior_sd**2 + sd**2)
    return (1.0 - (1.0-rate)**k) * sensory_weight * x


def accumulator_equivalent(x, m, sd, prior_sd, k, rate):
    """Explicit equivalent accumulator *including* a prior baseline.

    Its update z <- z + rate*(z_star-z) is a first-order leaky accumulator,
    and equals the PC Gaussian gradient step identically. Any model-selection
    routine MUST acknowledge this exact equivalence.
    """
    star = equilibrium(x, m, sd, prior_sd)
    return m + (1.0 - (1.0-rate)**k) * (star-m)
