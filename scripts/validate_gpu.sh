#!/usr/bin/env bash
# Optional CUDA test. Never run as the standard CPU release gate.
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

printf 'Repository HEAD: '
git rev-parse HEAD
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python scripts/check_gpu_backend.py  # exits nonzero on CPU fallback
python scripts/check_source_hashes.py
python -m compileall -q src tests examples scripts
python -m pytest -q

# Compare two forced backends using the SAME pinned GPU-capable environment.
root=results/backend_comparison
for backend in cpu cuda; do
    printf '\nRunning synthetic examples with JAX_PLATFORMS=%s\n' "$backend"
    JAX_PLATFORMS="$backend" python examples/truncated_inference.py \
        --outdir "$root/$backend/truncated_inference"
    JAX_PLATFORMS="$backend" python examples/auditory_learning.py \
        --seed 2027 --trials 80 --outdir "$root/$backend/auditory_learning"
done
python scripts/compare_backend_outputs.py --root "$root"
printf '\nCUDA device, GPU tests, examples, and CPU/CUDA output comparison completed.\n'
