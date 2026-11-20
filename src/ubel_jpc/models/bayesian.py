"""Strong(er) conjugate Bayesian competitor with category-mapping learning.

The former PyTorch Bayesian baseline tracked only category frequency. This
model ALSO learns the Gaussian acoustic mean for each label. The two mean
posteriors are conjugate Normal distributions with a fixed observational
variance, and a discounted Beta-Bernoulli distribution tracks category rate.

Known limitations: fixed class variance, no full volatility posterior, a
heuristic temporal evidence gain, and no model selection fitting yet. It is a
serious improvement, not claimed to be an optimal Bayesian observer.
"""
from __future__ import annotations
from dataclasses import replace
import jax
import jax.numpy as jnp
from ..schemas import Config, AgentState, causal_probability
from .feedback_learning import adjusted


def bayesian_probability(state: AgentState, x, sd, k, cfg: Config):
    pi = jnp.clip(causal_probability(state), 1e-6, 1-1e-6)
    # Posterior predictive variance includes class-feature variability,
    # current sensory noise, and uncertainty in the learned class means.
    variance = cfg.category_sd**2 + sd**2 + 1.0/state.class_precision
    logprob = -.5*(jnp.log(2*jnp.pi*variance) + (x-state.class_mu)**2/variance)
    # The elapsed-inference gain is a behavioural approximation, explicitly
    # not a derivation of an exact finite-time Bayesian inference algorithm.
    evidence_gain = 1.0-(1.0-cfg.inference_rate)**k
    logit = jnp.log(pi)-jnp.log1p(-pi)+evidence_gain*(logprob[1]-logprob[0])
    return jax.nn.sigmoid(logit)


def update_bayesian_means(state: AgentState, x, sd, y, cfg: Config):
    """Update ONLY the class observed in current feedback.

    With prior precision P, likelihood precision 1/V and observation x,
        P_new = P + 1/V,
        mu_new = (P*mu + x/V)/P_new.
    This update is independent of K, by design. Learning acoustic statistics
    from fully informative correct labels need not be perceptually gated.
    """
    var = cfg.category_sd**2 + sd**2
    idx = jnp.asarray(y, dtype=jnp.int32)
    oldp = state.class_precision[idx]
    newp = oldp + 1.0/var
    newmu = (oldp*state.class_mu[idx] + x/var)/newp
    return replace(state,
                   class_mu=state.class_mu.at[idx].set(newmu),
                   class_precision=state.class_precision.at[idx].set(newp))
