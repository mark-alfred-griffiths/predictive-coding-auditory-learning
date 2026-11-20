# File-by-file curated manifest

## Unmodified copies from original main (source or regression tests)

| Staged path | Treatment / reason |
| --- | --- |
| `src/ubel_jpc/__init__.py` | Retain: original package 64-bit JAX configuration |
| `src/ubel_jpc/schemas.py` | Retain: immutable state, prior and configuration constraints |
| `src/ubel_jpc/task.py` | Retain: seeded synthetic auditory task with independent RNG streams |
| `src/ubel_jpc/models/__init__.py` | Retain: namespace and package compatibility |
| `src/ubel_jpc/models/analytic_gaussian.py` | Retain: Gaussian analytical reference and finite-step dynamics |
| `src/ubel_jpc/models/coupled_pc.py` | Retain: distinct pre-/post-feedback inference and joint energy |
| `src/ubel_jpc/models/feedback_learning.py` | Retain: readout, fixed-activity SGD, matched accumulation |
| `src/ubel_jpc/models/bayesian.py` | Retain: category mapping competitor |
| `src/ubel_jpc/models/jpc_adapter.py` | Retain: explicit, optional real-JPC energy gate (not tested in curated CPU workflow) |
| `src/ubel_jpc/experiments/__init__.py` | Retain: package namespace |
| `src/ubel_jpc/experiments/learning_and_probes.py` | Retain: causal training and read-only probes |
| `src/ubel_jpc/experiments/phase2_diagnostics.py` | Retain: same-state K×L counterfactual controls |
| `tests/test_analytic_posterior.py` | Retain: exact posterior, gradient, finite-K convergence |
| `tests/test_coupled_pc.py` | Retain: feedback gradients, K/L controls and causal checks |
| `tests/test_equivalence.py` | Retain: crucial PC/matched-accumulator negative control |
| `tests/test_feedback_learning.py` | Retain: fixed-activity update and label gating |
| `tests/test_probe_and_causality.py` | Retain: probes must not alter state; reproducibility |
| `docs/source_phase2_specification.md` | Retain original mathematical source documentation, new filename for clarity |

See machine-readable [`source_hashes.json`](source_hashes.json) for SHA256 hashes and original paths. All source-code and original test copies were left byte-identical.

## Curated additions/replacements

| Staged path | Treatment |
| --- | --- |
| `README.md`, `pyproject.toml`, `.gitignore`, `CITATION.cff`, `LICENSE` | New, self-contained project metadata and instructions, including the selected MIT licence; source ownership and public-release approval still require review |
| `constraints-cpu.txt`, `.github/workflows/tests.yml` | Tested CPU package versions and GitHub Actions CPU workflow (run #2 passed at `58fb985`, 20 November 2026) |
| `examples/__init__.py`, `examples/truncated_inference.py`, `examples/auditory_learning.py` | Small example wrappers; original algorithms unchanged |
| `tests/test_curated_examples.py` | Independent checks of both examples and their outputs |
| `scripts/validate_local.sh`, `scripts/sync_local.sh` | Local CPU test/build and safe Git sync commands |
| `scripts/validate_gpu.sh`, `scripts/check_gpu_backend.py`, `docs/gpu_validation.md` | Optional CUDA-only diagnostic and local GPU validation instructions; no scientific model changes |
| `docs/mathematical_model.md`, `docs/experimental_design.md`, `docs/results_and_limitations.md`, `docs/exploratory_forecasting.md` | Concise science, negative controls, and historical forecast outcome |
| `docs/provenance.md`, `docs/file_manifest.md`, `docs/attribution_and_licensing.md`, `docs/local_pycharm_and_git.md`, `docs/validation.md` | Audit, local workflow, reproducibility and publication gate |

## Explicit exclusions

The original README, AGENTS.md, original `__main__.py` CLI, source model recovery/fitting modules, heavyweight sequential-design/recovery pipelines, plotting/analysis infrastructure not imported by this demo, original configs and freezes, large `docs/experiments/` archives, old Git history, local machine paths, proposals/CV/correspondence and full path-integral forecasting benchmarks remain in the **private original** only. The new package exercises a subset and therefore **cannot claim to rerun the original full test matrix**.

`docs/jpc_install.md` is a curated, safety-conscious note clarifying the optional native adapter and its separately archived validation; it is not a re-run of the original upstream integration.
