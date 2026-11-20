#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MPLBACKEND=Agg
printf 'Repository HEAD: '
git rev-parse HEAD
python --version
python - <<'PY'
import jax,jaxlib,numpy,matplotlib,pytest
for lib in (jax,jaxlib,numpy,matplotlib,pytest):
    print(f"{lib.__name__}=={lib.__version__}")
PY
python scripts/check_source_hashes.py
python -m compileall -q src tests examples
python -m pytest -q
python examples/truncated_inference.py --outdir results/truncated_inference
python examples/auditory_learning.py --seed 2027 --trials 80 --outdir results/auditory_learning
python -m pip wheel --no-deps --no-build-isolation -w dist .
printf '\nValidation command sequence completed. Review outputs and exact SHA above.\n'
