# Curated synthetic experimental design

This is a **demonstration** of the existing modelling structure, not a preregistered human experiment and not a recovery or confirmatory design.

## Demo 1: controlled K×L counterfactual

`python examples/truncated_inference.py --outdir results/truncated_inference`

- One synthetic scalar observation `x=0.9`, sensory SD `0.9`, and fixed initial agent state.
- Exact same observation and parameters across every cell: K = {0, 1, 4, 32}, L = {0, 1, 8, 32}, feedback y = {0,1}.
- 32 cells; each records the pre-feedback response, post-feedback activity, activity-gradient residual, and fixed-activity readout update.
- Closed-form Gaussian posterior is the independent reference for the pre-feedback phase.
- A labelled-feedback condition is used *only* after computing the pre-feedback prediction.

The figure shows parameter updates for differing K and L. Do not interpret the figure as a fitted participant effect or biological response time.

## Demo 2: successive auditory learning and common probes

`python examples/auditory_learning.py --seed 2027 --trials 80 --outdir results/auditory_learning`

- 80 default synthetic training trials (four 20-trial blocks); category prevalence follows the source task's repeating 0.2/0.8 block schedule.
- Clear and ambiguous acoustic observations are assigned by a separate RNG stream, and K is varied independently.
- Correct category feedback is available to update the model *only after the present response probability is recorded*.
- Standardised 15-point no-feedback probe grid after every 20 trials, with fixed probe SD and fixed probe K, distinct from training-trial K.
- All six models receive the exact same generated trial records. Identically matched PC and accumulator methods should be observationally equivalent.
- A coupled-PC model adds L=8 post-feedback settling; some comparator models have other learning assumptions and cannot be interpreted as equally fitted empirical models.

`summary.json` stores the complete config including seed and parameter values. CSV output contains training predictions, subsequent probe predictions and diagnostic fields. No participant choices are present; the "training predictions" are probability forecasts, not measured choices.

## Reproducibility and causal validity

The seed controls independent JAX random streams for categories, category-feature variation, sensory noise, ambiguity, K assignment and availability of feedback. Probing must not consume learning randomness or alter state. The original tests verify causal prior updates, no label leakage before feedback, no probe-induced state changes and fixed-seed deterministic generation. Results depend on numeric precision/software; compare within recorded tolerances and package versions.

## Human research remains proposed

The UBEL proposal anticipates speech-like stimuli and response-signal manipulations with matched later no-feedback probes, plus appropriate controls and model recovery before human testing. **Those data, stimulus recordings, participant measures and ethics procedures are not part of this codebase.**
