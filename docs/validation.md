# Independent validation protocol

## Required reproducibility environment

The first curated Linux CPU validation used Python 3.13.5, JAX/jaxlib 0.9.0.1, NumPy 2.3.5, Matplotlib 3.10.8 and pytest 9.0.2. The reference execution set `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, and `MPLBACKEND=Agg`. `constraints-cpu.txt` records the key tested versions but is not a complete transitive lockfile. Use a separate Conda or virtual environment to avoid mixed system packages. The owner independently validated a fresh GitHub clone in `pc-auditory-verify` with Python 3.13.18, JAX/jaxlib 0.9.0.1, NumPy 2.3.5, Matplotlib 3.10.8 and pytest 9.0.2.

## Local commands, from repo root

```bash
python -m pip install -e '.[test]' -c constraints-cpu.txt
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MPLBACKEND=Agg
python scripts/check_source_hashes.py
python -m compileall -q src tests examples
python -m pytest -q
python examples/truncated_inference.py --outdir results/truncated_inference
python examples/auditory_learning.py --seed 2027 --trials 80 --outdir results/auditory_learning
python -m pip wheel --no-deps -w dist .
git rev-parse HEAD
```

Or `bash scripts/validate_local.sh`, which additionally prints the environment and HEAD. Report test passes, failures, skips, every skipped reason, whether the examples ran, and the exact commit. Do not treat previous source-repo tests or a green workflow icon as a replacement for checking this clone.

## Scientific correctness checks

Original source tests retained here cover closed-form Gaussian optima, finite-K convergence, automatic differentiation/finite differences, label-blind prediction, fixed-state learning, separate K/L settling, exact PC/matched-accumulator equivalence, synthetic-trial reproducibility, and no probe-induced parameter changes. New tests exercise both generated example artifacts, numerical finiteness/dimensions, repeated seeded outputs, model equivalence, and zero-L consistency.

Check finite-L residuals and training/probe differences *without* selecting only favourable conditions. `summary.json` records the full synthetic configuration; CSVs are raw model predictions and diagnostics, not human choices. The reproducibility outputs are intentionally ignored by Git.

## Local clone verification

After creating the initial fresh Git history, use `git clone` into a new directory, create a **new** environment (or record if an isolated site-packages environment was not feasible), and run the exact commands above. Run a second clone after pushing to GitHub, on the user's own Pop!_OS machine, to establish end-to-end independent Git/pip reproducibility.

## CI and optional JPC

`.github/workflows/tests.yml` is configured to install the package in a clean Linux Python 3.13 CPU environment and run the same tests and examples. **Curated CPU validation #3 passed on private GitHub Actions at commit `f7644b235d1a7f9b33acf7278e2c9b956c5e6e52` on 20 November 2026**, as confirmed by the owner-provided screenshot. The earlier run at `affe7b8` failed only at wheel building due to missing setuptools under `--no-build-isolation`; the passing workflow removed that option at `58fb985`. Any later documentation-only release-review commit needs a new CI run before the final commit can be declared checked.

The main demonstrations use JAX's analytic Gaussian backend. The strictly optional `jpc_adapter.py` imports real upstream JPC at runtime and raises a `RuntimeError` if requested but missing; it must never silently substitute analytical JAX. This curated version has **not** independently retested the original pinned JPC native CPU energy integration, upstream ODE solver, GPU benchmarking, or exploratory model-recovery study.

## Optional NVIDIA GPU validation

The standard reference remains CPU-only. For the independently configured RTX 3060 Ti/CUDA 13 smoke test, see [GPU validation](gpu_validation.md). The owner-supplied log for `f7644b2` confirms `cuda:0` device placement, 28 passing tests and 5/5 matching CPU/CUDA output comparisons at `atol=rtol=1e-5`; GPU execution does not supersede the CPU scientific reference. No CUDA test was physically run on the document author's computer.
