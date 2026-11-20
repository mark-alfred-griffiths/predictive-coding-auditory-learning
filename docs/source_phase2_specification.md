# Phase 2 mathematical model and scientific interpretation

## What is new

The original JAX prototype coupled Gaussian perceptual inference to a logistic readout, but category feedback did not alter the latent perceptual state. The new `models/coupled_pc.py` includes **two independently controllable inference stages** and a *joint* feedback energy. The real JPC sensory-energy adapter remains an optional, explicitly gated dependency.

### Phase A: perception before feedback

The generative Gaussian sensory energy is

\[
 E_s(z;x,m)=\frac{(z-m)^2}{2\sigma_p^2}+\frac{(x-z)^2}{2\sigma_x^2}.
\]

The contextual mean `m` is inferred causally from previous feedback only. The observed sound `x`, known assumed precisions and model parameters are available. The category label `y` is **not** available. An Euler solver produces a partially inferred state after exactly `K` steps:

\[
 z_{k+1}=z_k-\frac{\rho}{1/\sigma_p^2+1/\sigma_x^2}\nabla_zE_s(z_k),\qquad z_0=m.
\]

The pre-feedback category prediction is `sigmoid(w * z_K + b)`. The sampled current choice, where one is drawn, is independent of how the environment generates feedback labels.

### Phase B: after corrective feedback

Once `y` is revealed, the same latent state can undergo another `L` steps of inference, driven by a **combined energy**:

\[
E_{post}(z)=E_s(z;x,m)+\lambda\bigl[\log(1+e^{wz+b})-y(wz+b)\bigr].
\]

The second term is a Bernoulli negative log likelihood. It allows feedback to influence the hidden activity, not just the final readout. The first two terms are the same sensory energy used before feedback. This joint energy is strictly convex in the single latent dimension under the current model assumptions.

A conservative step size uses an upper curvature bound:

\[
H_{max}=1/\sigma_p^2+1/\sigma_x^2+\lambda w^2/4,\quad
 z_{l+1}=z_l-(\rho_{post}/H_{max})\nabla E_{post}(z_l).
\]

Thus `K` and `L` can be manipulated separately. For `L=0`, feedback learning uses the saved `z_K` exactly as in the original model. Following `L>0` steps, it uses the post-feedback state `z_post`.

### Phase C: update at fixed activity

At `z_post` held fixed (`stop_gradient`), the readout parameters update via

\[
\Delta w=\eta_w[y-\sigma(wz_{post}+b)]z_{post},\qquad
\Delta b=\eta_b[y-\sigma(wz_{post}+b)].
\]

We deliberately do **not** differentiate through the entire inference trajectory. Generative sensory weights and uncertainty parameters remain frozen. The model is therefore **a scalar JPC-compatible sensory predictive-coding system with a unified feedback energy and fixed-activity readout learning**. It is not yet a trained multi-layer JPC network with local plasticity at every layer.

### Key falsifying prediction

Because the post-feedback energy has a unique optimum, sufficiently complete post-feedback settling can remove the dependence of subsequent learning on the initial pre-feedback state `z_K`. Thus the effect of truncating pre-feedback inference may disappear when post-feedback integration is extensive. This is a structural limitation of the hypothesis and a reason not to conflate perceptual response time with the time at which plasticity occurs.

### Caveats

- Neither `K` nor `L` is calibrated to milliseconds of human processing.
- The inferred latent is a **posterior mode approximation**, not a full uncertainty posterior.
- The model uses known sensory precisions; changes in `K` do not change acoustic exposure.
- Correct labels may be available only on feedback trials; no-feedback probes must remain read-only.
- Strong accumulator competitors may have exactly the same behavioural likelihood as the Gaussian PC model.
- A small exploratory recovery matrix cannot establish uniqueness of a neural algorithm.

## Relevant implementation modules

- `models/coupled_pc.py` — two-phase inference, energy, analytical gradient and fixed-state learning.
- `experiments/learning_and_probes.py` — causal trial controller and read-only probes.
- `experiments/phase2_diagnostics.py` — counterfactual K×L comparison.
- `analysis/fast_likelihood.py` — independent NumPy trial replay, verified against JAX.
- `analysis/recovery_v2.py` — bounded fitted likelihood and synthetic recovery.
- `models/jpc_adapter.py` — the only direct dependency on actual JPC API.
