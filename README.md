# Predictive Coding, Truncated Inference and Auditory Learning

**A small, independently runnable research demonstration — synthetic data only.**

## Scientific question

**How does incomplete perceptual inference affect subsequent feedback-based learning, and can its behavioural consequences distinguish predictive coding from alternative learning mechanisms?**

This computational question motivates a proposed 2027 UBEL DTP PhD project in auditory categorisation at UCL, under the proposed supervision of Professor Peter Howell. The experiments proposed for human participants have **not** been conducted here.

## Why inference truncation matters

A perceptual learner may have to act before its inference algorithm has settled. If learning is computed from a partial activity estimate, shorter inference can change not just a present decision but the learning update and *later* categorisation. The source model explicitly separates **K**, pre-feedback inference steps, from **L**, post-feedback, label-aware settling steps. Neither corresponds automatically to milliseconds of human perception.

## What is implemented

The retained scalar JAX research engine has: (1) Gaussian sensory inference with an exact posterior reference; (2) finite-step inference and a strictly convex post-feedback joint objective; (3) fixed-activity readout learning; (4) seeded synthetic auditory trials with clear/ambiguous stimuli and changing category frequencies; (5) read-only no-feedback probes; and (6) PC, exact matched accumulation, and other comparison models.

Two small examples export inspectable CSV and JSON and one plot without external data:

| Example | Scientific purpose | Outputs |
| --- | --- | --- |
| `truncated_inference.py` | Counterfactual K×L inference/feedback update grid | `counterfactual_grid.csv`, `parameter_updates.png`, `summary.json` |
| `auditory_learning.py` | Synthetic sequential auditory categorisation and later standardised probes | `training_predictions.csv`, `no_feedback_probes.csv`, `summary.json` |

## Research contribution and provenance

This repository curates the original predictive-coding computational investigation and numerical validation carried out by Mark A. Griffiths. It retains the tested underlying model code unchanged and adds separately testable example entry points and documentation. Prior independent predictive-coding work was undertaken with guidance from Professors Christopher Buckley and Karl Friston; the UBEL doctoral proposal concerns proposed work with Professor Peter Howell. See [provenance](docs/provenance.md), the [MIT License](LICENSE), and the [attribution/licensing review](docs/attribution_and_licensing.md). The owner has confirmed independent development outside paid appointments and commissioned work; the curated code carries his selected MIT License. This owner statement does not replace a review of third-party fragments or the final Git history. **Release status:** This curated repository is public; the original `ubel-jpc` source repository remains private. The linked release-review notes document the scope and limits of checks performed before publication.

## Install and run (CPU — PyCharm compatible)

Use **Python 3.13** to reproduce the environment tested for this curated version. From the repository root:

```bash
conda create -n pc-auditory python=3.13 pip setuptools wheel -y
conda activate pc-auditory
python -m pip install -e '.[test]' -c constraints-cpu.txt
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MPLBACKEND=Agg
python examples/truncated_inference.py --outdir results/truncated_inference
python examples/auditory_learning.py --seed 2027 --trials 80 --outdir results/auditory_learning
```

The confirmed workstation environment used Python 3.13.18 and JAX/jaxlib 0.9.0.1. A compatible but unverified Python/JAX environment may work; do not silently treat it as covered by this validation. `JAX_PLATFORMS=cpu` prevents accidental selection of a GPU. For Windows/PyCharm activation paths, see [local setup](docs/local_pycharm_and_git.md). The scripts do not require network access or external datasets at runtime.

## Test and build

```bash
python -m compileall -q src tests examples
python -m pytest -q
python -m pip wheel --no-deps -w dist .
```

Run `bash scripts/validate_local.sh` to execute the CPU checks and examples. The owner confirmed **28/28 CPU tests**, a separate CUDA-13 execution on the RTX 3060 Ti with **28/28 tests**, **18/18 original-source hashes**, and matching CPU/GPU results for five JSON/CSV outputs (`atol=rtol=1e-5`). **Curated CPU validation #3 passed on private GitHub Actions at commit `f7644b235d1a7f9b33acf7278e2c9b956c5e6e52`** (owner-supplied log/screenshots, 20 November 2026). The cited CI validation applies to the recorded historical commit; consult the repository's Actions tab for the status of later commits.

## What these experiments demonstrate

The code illustrates mathematically controlled differences between finite and more extensive inference, the role of post-feedback settling, causal feedback updates, and changes in subsequent synthetic probe predictions. The closed-form Gaussian posterior and exact matched-accumulator equivalence supply strong correctness and negative controls.

## What these experiments do not demonstrate

No human auditory behavioural observations, biological synaptic credit assignment, unique predictive-coding mechanism, general identification of model families, statistical power for human recruitment, or superior computational efficiency are established. In the private source's completed probe-recovery evaluation, generating observational family recovery was not reliable. The optional real-JPC adapter requires a separate upstream installation; it is **not used by these standard CPU demonstrations** and was **not retested** in this curated workflow.

The exploratory path-integral/low-rank forecasting extension is [summarised separately](docs/exploratory_forecasting.md), with its negative accuracy/cost result. It is not part of the standard example imports.

## Relation to proposed UBEL work

The proposed doctoral work would test whether experimentally constrained processing intervals change **later** auditory categorisation using standardised no-feedback probes, and would compare predictive coding with equally capable accumulation and adaptive-learning alternatives. The present repo is a *reproducible synthetic preparation and validation demonstration*, not a result from that planned human experiment.

## Read more

- [Mathematical model, inference and learning](docs/mathematical_model.md)
- [Synthetic experimental design](docs/experimental_design.md)
- [Validation procedure](docs/validation.md) and [results/limitations](docs/results_and_limitations.md)
- [Source provenance](docs/provenance.md) and [file-by-file manifest](docs/file_manifest.md)
- [PyCharm/Git synchronisation and local validation](docs/local_pycharm_and_git.md) and [optional GPU validation](docs/gpu_validation.md)
- [Licensing, attribution and historical release checklist](docs/attribution_and_licensing.md)

**Release status: PUBLIC.** This repository provides reproducible synthetic demonstrations, not human experimental results or a claim of unique model identification. Public access does not imply that every historical release-checklist item was independently verified. The original `ubel-jpc` remains private.
