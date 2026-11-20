# Results, negative controls and limitations

## What the codebase shows

- The scalar Gaussian energy has an independent analytical optimum, and finite K has a closed-form geometric convergence relation.
- Feedback-aware settling controls L separately from K. K can change the fixed-activity learning update when L is short; sufficiently extensive settling can attenuate K dependence because the joint scalar energy has one optimum.
- Synthetic training histories with feedback can alter later standardised probe probabilities; no-feedback probes must be read-only.
- An exactly matched accumulator is observationally equivalent to the scalar Gaussian PC inference rule, so that observational comparison alone cannot discriminate the mechanisms.

## Original private-source model-recovery limitation

The completed independent recovery study numerically validated causal training/probe-choice likelihoods, but its **scientific discrimination gate did not pass**. The recorded untouched 16-subject synthetic evaluation selected the generating observational family for **5 subjects**, with **2 unresolved ties**, and **311 of 544 fitted parameter entries** were within 1% of a fitting bound. These are source-reported figures, not rerun in this small curated project. The original code and frozen recovery study have **not** been silently transferred or simplified into a claim of reliable recovery.

## Path-integral/low-rank forecast finding

The separate private forecasting pilot contains 256 synthetic cases and 3,072 method evaluations. The multimodal low-rank forecast improved gradient error over geometric extrapolation in **55/256** comparisons, but its incremental cost was substantially higher, and Newton–CG remained competitive. This was reported from original user-supplied experimental summaries; the forecasting pilot raw local data and full benchmark were not independently ingested by the curator. See [exploratory forecasting](exploratory_forecasting.md).

## Do not infer

No human behavioural finding, neural mechanism identification, human power calculation, biological credit assignment, improved training, GPU cost superiority, physiological processing latency or general mechanism recovery follows. A correct numerical solver is not equivalent to a predictive-coding account of the brain. A model that makes the same predictions as a competitor cannot be distinguished by its fitted choices alone.

## Unverified parts of this staging project

The native external JPC energy adapter is retained for optional future comparison but is **not exercised** in the standard CPU suite. The owner reported a successful private GitHub Actions CPU run at `58fb9855f54ad2c07feb8e8363575dce933a732a` (20 November 2026); no CUDA/GPU result has yet been certified for this curated project. Earlier main-branch and forecasting-branch claims are documented source evidence, not substitute tests for the curated snapshot.
