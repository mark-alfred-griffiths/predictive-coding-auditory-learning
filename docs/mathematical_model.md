# Mathematical model and causal sequencing

This is a faithful guide to the retained source implementation, not a new algorithm. Consult [`source_phase2_specification.md`](source_phase2_specification.md) for the original mathematical explanation, and inspect the exact functions under `src/ubel_jpc/`.

## Sensory inference, before feedback

For observation \(x\), contextual mean \(m\), latent acoustic activity \(z\), and fixed positive standard deviations \(\sigma_p,\sigma_x\), the scalar Gaussian sensory energy is

\[
E_s(z;x,m)=\frac{(z-m)^2}{2\sigma_p^2}+\frac{(x-z)^2}{2\sigma_x^2}.
\]

Its fixed-context gradient and unique optimum are

\[
\frac{\partial E_s}{\partial z}=\frac{z-m}{\sigma_p^2}+\frac{z-x}{\sigma_x^2},\qquad
z_*=\frac{m/\sigma_p^2+x/\sigma_x^2}{1/\sigma_p^2+1/\sigma_x^2}.
\]

The pre-feedback state starts at \(z_0=m\). With \(H_s=\sigma_p^{-2}+\sigma_x^{-2}\) and \(0<\rho<1\), exactly \(K\) explicit steps are

\[
z_{k+1}=z_k-\frac{\rho}{H_s}\nabla_z E_s(z_k).
\]

The resulting error obeys \(z_K-z_*=(1-\rho)^K(m-z_*)\). The *current label is unavailable* during this phase. Before corrective feedback arrives, the model predicts category probability \(p_K=\sigma(w z_K+b)\), optionally mixed with a small lapse probability for observed choices.

The contextual mean \(m\) is based on a causal, discounted feedback-derived category prior. Its mapping from category frequency to latent acoustic mean is an explicit modelling assumption, not a consequence of JPC.

## Post-feedback activity inference

After feedback \(y\in\{0,1\}\) is revealed, the activity starts from the saved pre-feedback state and follows the *joint* objective

\[
E_{post}(z)=E_s(z;x,m)+\lambda\{\operatorname{softplus}(wz+b)-y(wz+b)\}.
\]

Write \(H_{max}=\sigma_p^{-2}+\sigma_x^{-2}+\lambda w^2/4\), then perform exactly \(L\) further steps with rate \(\rho_{post}/H_{max}\). The logistic curvature bound makes the prescribed solver conservative for the strictly convex scalar energy. When \(L=0\), the learning state remains \(z_K\). With extensive post-feedback settling, this model's unique joint optimum may remove pre-feedback K dependence.

## Fixed-activity parameter update

At the resulting \(z_{post}\), hold latent activity fixed with `jax.lax.stop_gradient` and update only the logistic readout:

\[
\Delta w=\eta_w[y-\sigma(wz_{post}+b)]z_{post},\qquad
\Delta b=\eta_b[y-\sigma(wz_{post}+b)].
\]

This is **not** differentiating through the inference trajectory, not updating the fixed acoustic generative weights, and not a biological synaptic-plasticity demonstration. Feedback-derived category counts update after the trial prediction; without feedback, no update is allowed. K and L describe **computational iterations** and are not calibrated to neural dynamics or milliseconds of processing.

## Exact matched-accumulator equivalence

The linear Gaussian finite-step rule has an exact rearrangement as an accumulator that starts at \(m\), integrates toward the precision-weighted observation, and uses the same per-step contraction. The `accumulator_equivalent` implementation therefore yields the **same activities**, choices and matching learning histories as its PC counterpart (to numerical tolerance). This is a deliberately retained non-identifiability control: differences in observed behaviour cannot distinguish algorithms with identical observation maps.

## Simulated auditory task

The simulator generates a category label from piecewise category probabilities, a Gaussian class-conditioned acoustic feature, and sensory noise under clear/ambiguous conditions. K is assigned independently of the label and category sequence. The agent cannot read the true environment prevalence (the `q_true_analysis_only` diagnostic field). Probes evaluate model responses at a shared sensory noise and K, leave the state unchanged, and receive no labels. The categorical frequency, model likelihood and noise parameters are synthetic design assumptions, not empirically estimated quantities.

## Numerical assumptions and scope

- All variances are strictly positive; inference rates and feedback settings are validated.
- The Gaussian posterior mean equals its mode in this special model; this does *not* generalise to arbitrary non-Gaussian posteriors.
- The joint energy is strictly convex in this single activity dimension for the given fixed readout; finite L should not be called converged merely because a fixed step count has elapsed.
- Analytical/automatic-differentiation agreement validates the implemented equations; it does not validate the proposed human or neural mechanism.
