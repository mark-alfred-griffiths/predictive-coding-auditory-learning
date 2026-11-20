"""UBEL predictive-coding research package.

64-bit JAX floats are enabled deliberately for reproducible gradient and
Gaussian-reference checks. Import this package before creating JAX arrays in
an interactive session if those precision settings are desired.
"""
import jax
jax.config.update('jax_enable_x64', True)
__version__ = '0.1.0'
