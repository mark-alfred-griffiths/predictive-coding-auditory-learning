# Exploratory trajectory and path-integral gradient forecasting

**Separate historical computational investigation, not part of the curated core demo.**

The completed private branch `feature/path-integral-forecast` extends the scalar setup to a multidimensional fixed-activity gradient of a full joint energy with parameters `(A,w,b)`. It uses partial post-feedback trajectories to forecast an unseen endpoint and compares direct endpoint gradient evaluation, mixed-Jacobian-vector product corrections, and predicted-tail trapezoidal integration. The low-rank forecast estimates contracting increment dynamics with regularisation and explicit rejection/fallback safeguards; a numerical acceptance does not imply greater forecasting accuracy.

The original exploration was motivated by independent decoupled predictive-coding work, including mathematical ideas for forecasting equilibrium gradients using incomplete inference trajectories, discussed with Professor Karl Friston. The public-facing proposal is about **subsequent auditory learning**, so transferring the benchmark infrastructure would distract from the research question.

**Source-reported pilot outcomes (not independently rerun here):** 256 completed synthetic cases, 3,072 method evaluations, no execution failures. The multimodal estimate beat geometric extrapolation in 55/256 paired comparisons and was typically several times slower; Newton–CG baselines generally remained preferable. The study showed no general accuracy–cost advantage, no successful trained-network learning, and no behavioural or biological mechanism.

For provenance, the private source's `docs/multimodal_forecast_final_report_20261120.md` and `docs/multimodal_forecast_dataset_card.md` should be consulted by authorised collaborators. Its complete local run output, source-code dependencies and freeze records are intentionally **not** part of this curated distributable. No accuracy/cost benchmark should be described as independently replicated from this repository.
