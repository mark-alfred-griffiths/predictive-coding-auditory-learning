"""Fail rather than silently running a requested NVIDIA JAX test on CPU.

The native-JPC adapter and the exploratory path-integral benchmark are not
exercised by this diagnostic. This checks actual placement of the curated
Gaussian predictive-coding inference kernel on an NVIDIA GPU.
"""
from __future__ import annotations

import json
import jax
import jax.numpy as jnp

from ubel_jpc.models.analytic_gaussian import infer, equilibrium


def main():
    devices = jax.devices()
    gpu = [device for device in devices if device.platform == "gpu"]
    if not gpu:
        raise RuntimeError(
            "No active JAX GPU backend. Ensure the CUDA plugin is installed and "
            "JAX_PLATFORMS=cuda; CPU fallback is not accepted."
        )

    # Real original research inference computation, not just device detection.
    z = infer(x=0.9, m=0.0, sd=0.9, prior_sd=1.0, k=32, rate=0.35)
    z.block_until_ready()
    device = z.device
    if device.platform != "gpu":
        raise AssertionError(f"Inference result placed on {device}, not on GPU")

    # Independently ensure GPU executes a compiled JAX kernel.
    x = jnp.arange(2048, dtype=jnp.float64)
    v = jax.jit(lambda a: jnp.sum(jnp.sin(a) ** 2))(x)
    v.block_until_ready()
    if v.device.platform != "gpu":
        raise AssertionError("Compiled GPU kernel did not execute on GPU")

    reference = float(equilibrium(0.9, 0.0, 0.9, 1.0))
    err = abs(float(z) - reference)
    if err > 1e-5:
        raise AssertionError(f"Unexpected finite-inference deviation: {err}")

    print(json.dumps({
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in devices],
        "original_inference_device": str(device),
        "gpu_inference_value": float(z),
        "reference_equilibrium": reference,
        "abs_inference_error": err,
        "compiled_kernel_device": str(v.device),
        "scope": "JAX GPU execution of curated inference kernel; no GPU speed or biological claim",
    }, indent=2))


if __name__ == "__main__":
    main()
