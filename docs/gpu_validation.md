# Optional NVIDIA GPU execution — separate from the validated CPU reference

**Observed workstation result (20 November 2026):** The owner ran `scripts/validate_gpu.sh` at `f7644b235d1a7f9b33acf7278e2c9b956c5e6e52` in a separate Python 3.13.18 / JAX 0.9.0.1 CUDA-13 environment. The RTX 3060 Ti was reported as `cuda:0` under NVIDIA driver 595.104.02; original and compiled inference kernels were placed on the GPU; 18 source hashes matched, 28 tests passed, and all five CPU/CUDA comparison files matched within `atol=rtol=1e-5`. These are owner-supplied workstation logs, not a second independent hardware run. Keep the original `pc-auditory` and `pc-auditory-verify` environments unchanged. Neither the optional upstream JPC adapter nor the exploratory path-integral benchmark is tested here.

## Prerequisites and installation (Pop!_OS, RTX 3060 Ti)

The owner's reported NVIDIA driver is 595.104.02, above the current JAX CUDA 13 minimum of 580. The RTX 3060 Ti is an Ampere GPU with CUDA 13-compatible compute capability. Confirm your *current* hardware and driver with `nvidia-smi` first. JAX's official Linux GPU guide recommends pip-provided CUDA libraries using the `jax[cuda13]` extra; this is a separate installation from a CPU-only `jax` environment.

From the **private Git-cloned** repository root (after pulling the latest commit):

```bash
nvidia-smi
conda create -n pc-auditory-gpu python=3.13 pip setuptools wheel -y
conda activate pc-auditory-gpu
which python
python -m pip install -e '.[test]' 'jax[cuda13]==0.9.0.1' -c constraints-cpu.txt
python -m pip check
```

The GPU packages may be large. Do **not** install them into the existing tested CPU environments. If JAX fails to find the intended pip-provided CUDA libraries, inspect `LD_LIBRARY_PATH`: a system CUDA toolkit may override the pip packages. Do not circumvent JAX's CUDA constraints checks.

## Real GPU execution and CPU/GPU comparison

```bash
conda activate pc-auditory-gpu
bash scripts/validate_gpu.sh |& tee gpu-validation.log
```

The command explicitly requests `JAX_PLATFORMS=cuda`; it fails rather than silently falling back to CPU. It reports the CUDA device, executes the original Gaussian inference kernel on GPU, runs the curated 28-test suite and both synthetic examples with forced CUDA, reruns both examples on CPU **within the same GPU-capable environment**, and compares the five JSON/CSV outputs with `atol=rtol=1e-5`. It permits the pre-existing Bayesian model's explicit `nan` placeholder for an inapplicable latent variable only when both devices agree on its location; other nonfinite values fail. GPU output is saved under ignored `results/backend_comparison/`; this is numerical parity, not a speed benchmark or biological test. The plot PNG is not compared byte-for-byte.

Tests run in this project are intentionally tiny; an RTX 3060 Ti need not run them faster than CPU. Float64 is requested via `JAX_ENABLE_X64=1`. If a GPU numerical tolerance fails, report the exact mismatch and investigate; do not change algorithms or tolerances merely to claim passing tests.

When reporting this optional validation, include the exact commit (`git rev-parse HEAD`), Python/JAX/CUDA plugin versions, `nvidia-smi` GPU and driver, test pass/fail/skip counts, every discrepancy, and full saved log. This does **not** establish the omitted multidimensional path-integral benchmark, upstream native-JPC integration or human/biological credit assignment.

Restore the normal development environment with `conda activate pc-auditory`; no separate CUDA packages should have been installed there.
