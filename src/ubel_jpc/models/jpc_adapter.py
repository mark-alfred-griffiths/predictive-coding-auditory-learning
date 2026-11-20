"""A strict, OPTIONAL bridge to the real JPC library.

This module does not bundle, imitate, or silently replace JPC. In particular,
`jpc_energy()` calls the public `jpc.pc_energy_fn` from the installed package.
The analytic JAX engine lives in `analytic_gaussian.py` and is deliberately
labelled separately. If `jpc` is unavailable, a JPC request fails loudly.

Embedding Gaussian precision in JPC's standard squared-error energy:

    u = z / sigma_prior                 (JPC activity)
    x_jpc = x / sigma_obs               (JPC observed output)
    r_jpc = m / sigma_prior             (JPC input/prior)
    f1(r_jpc) = r_jpc
    f2(u) = (sigma_prior/sigma_obs) * u

At JPC commit 64eeab0334fe7e1a4365da5b7f02afaefecd0261, pc_energy_fn
already includes 0.5 per squared residual and divides by batch size:
    0.5 * (u-r_jpc)^2 + 0.5 * (x_jpc-f2(u))^2 = Gaussian_energy(z).
Our scalar case uses batch size 1; no further energy scaling is needed.
JPC expects one activity slot per model layer, including an output slot.
Its sensory residual uses activities[-2], with y clamping the observation.

This contract is validated against the pinned upstream package, not a copy
or emulation. See docs/native_jpc_gate.md for versions and scope.
"""
from __future__ import annotations
import importlib
import jax
import jax.numpy as jnp


def require_jpc():
    try:
        return importlib.import_module('jpc')
    except ImportError as e:
        raise RuntimeError(
            'JPC backend requested, but the real `jpc` package is not installed. '
            'Follow docs/jpc_install.md and run `python -m ubel_jpc jpc-smoke`. '
            'The JAX backend is available as an independently validated baseline.'
        ) from e


def jpc_energy(z, x, m, sd, prior_sd):
    """Call JPC's public PC energy using a two-layer, fixed-weight model.

    The pinned JPC source vmaps ordinary callable layers. Its final activity
    slot is unused by pc_energy_fn: the output residual uses the penultimate
    activity and the separate y observation. This slot is acoustic data, never
    category feedback. Unsupported APIs fail rather than switching backends.
    """
    jpc = require_jpc()
    # Shapes are batch x feature = 1x1, as documented for JPC's PC energy.
    prior_scaled = jnp.reshape(jnp.asarray(m / prior_sd), (1, 1))
    observed_scaled = jnp.reshape(jnp.asarray(x / sd), (1, 1))
    activity_scaled = jnp.reshape(jnp.asarray(z / prior_sd), (1, 1))
    ratio = prior_sd / sd

    # Note: no ground-truth category occurs anywhere in the JPC inference graph.
    def prior_layer(r):
        return r

    def acoustic_layer(u):
        return ratio * u

    model = (prior_layer, acoustic_layer)
    total = jpc.pc_energy_fn(
        params=(model, None), activities=(activity_scaled, observed_scaled),
        y=observed_scaled, x=prior_scaled, loss='mse', param_type='sp'
    )
    return total


def infer_jpc(x, m, sd, prior_sd, k, rate=0.35):
    """JPC ENERGY-driven discrete gradient flow (not JPC's built-in solver).

    We explicitly control exactly K inference steps by differentiating JPC's
    native energy with JAX. This avoids accidental supervised-target clamping
    in convenience training functions and keeps the comparison with analytic
    Euler steps interpretable. The JPC solver is a distinct future extension.
    """
    require_jpc()  # Explicit dependency gate even when K=0.
    x, m, sd, prior_sd = map(jnp.asarray, (x, m, sd, prior_sd))
    k = int(k)
    curvature = prior_sd**-2 + sd**-2
    step_size = rate / curvature

    def e(z):
        return jpc_energy(z, x, m, sd, prior_sd)

    dz = jax.grad(e)
    z = m
    for _ in range(k):
        z = z - step_size * dz(z)
    return z


def smoke_test():
    """Check native JPC energy, autodiff and inference against known solution."""
    import math
    from .analytic_gaussian import energy, equilibrium, infer
    require_jpc()
    cases = [(0.3, -0.5, 0.9, 1.0), (1.2, 0.4, 0.5, 1.5),
             (-0.8, 0.1, 1.2, 0.7)]
    results = []
    for x, m, sd, prior_sd in cases:
        ref_energy = energy(m, x, m, sd, prior_sd)
        native = jpc_energy(m, x, m, sd, prior_sd)
        if not math.isclose(float(native), float(ref_energy), rel_tol=2e-5, abs_tol=1e-5):
            raise AssertionError(f'JPC energy mismatch: native={native}, reference={ref_energy}')
        z1 = infer_jpc(x, m, sd, prior_sd, 12, 0.35)
        ref = infer(x, m, sd, prior_sd, 12, 0.35)
        if not math.isclose(float(z1), float(ref), rel_tol=2e-5, abs_tol=1e-5):
            raise AssertionError(f'JPC trajectory mismatch: {z1} vs {ref}')
        zstar = equilibrium(x, m, sd, prior_sd)
        results.append(dict(x=x, m=m, k=12, jpc=float(z1), jax=float(ref),
                            equilibrium=float(zstar)))
    return results


def smoke_test_joint():
    """REAL-JPC end-to-end smoke test for the new feedback-aware objective.

    The earlier smoke test checks the two Gaussian sensory error terms. This
    additional test checks whether a pre-feedback state, post-feedback
    activity-gradient steps, and the resulting parameter update agree with
    the independent JAX analytical implementation. We never monkeypatch or
    emulate `jpc`: this routine requires the actual installed distribution.
    """
    from dataclasses import replace
    from ..schemas import Config, initial_state
    from .coupled_pc import (
        joint_energy, feedback_settle, pre_feedback_state,
        learn_from_feedback)
    # Explicitly fail when the native JPC library is unavailable.
    require_jpc()
    jax_cfg = Config(post_steps=9, feedback_weight=1.4)
    native_cfg = replace(jax_cfg, jpc_backend=True)
    initial = initial_state(jax_cfg)
    comparisons=[]
    for x, sd, k, y in ((0.8,.7,4,1),(-1.1,1.25,2,0),(.2,.9,12,1)):
        z_jax = pre_feedback_state(initial,x,sd,k,jax_cfg)
        z_native = pre_feedback_state(initial,x,sd,k,native_cfg)
        jax_update,jax_diag = learn_from_feedback(initial,x,sd,y,z_jax,jax_cfg)
        native_update,native_diag = learn_from_feedback(initial,x,sd,y,z_native,native_cfg)
        values = ((z_jax,z_native),
                  (jax_diag.z_post,native_diag.z_post),
                  (jax_diag.energy_after,native_diag.energy_after),
                  (jax_update.w,native_update.w),
                  (jax_update.b,native_update.b))
        for reference,obtained in values:
            if not jnp.allclose(reference,obtained,atol=1e-5,rtol=2e-5):
                raise AssertionError(
                    f'Native JPC joint model differs from JAX: '
                    f'{float(reference)} != {float(obtained)}')
        comparisons.append({'x':x,'sd':sd,'k':k,'y':y,
                            'pre_jax':float(z_jax),
                            'pre_jpc':float(z_native),
                            'post_jax':float(jax_diag.z_post),
                            'post_jpc':float(native_diag.z_post)})
    return comparisons
