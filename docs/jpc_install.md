# Optional native JPC adapter — outside the standard validated workflow

The standard curated CPU examples deliberately use the independently tested **JAX Gaussian reference**; they **do not require or install JPC**. The retained `src/ubel_jpc/models/jpc_adapter.py` invokes *real* upstream JPC only if `Config(jpc_backend=True)` is selected. Without that dependency, it raises an explicit error instead of silently switching to JAX.

The original private source documents a separately validated CPU scalar-energy connection against upstream [thebuckleylab/jpc](https://github.com/thebuckleylab/jpc) revision `64eeab0334fe7e1a4365da5b7f02afaefecd0261`, under Python 3.11.18, JAX/jaxlib 0.4.38, and its compatible native dependencies. That historical integration exercised the JPC *energy*, differentiated by JAX, not JPC's own ODE inference or parameter-learning solvers.

If reproducing this optional historical check, create a **second, independent Python 3.11 environment** and use the original private `ubel-jpc/docs/jpc_install.md` and exact pinned dependency versions, verifying the current upstream terms and packages first. Do not install arbitrary recent JPC packages into the curated Python 3.13 environment and assume parity. This optional branch has **not** been tested in the new curated package and is explicitly not required for the release gate's standard CPU demonstrations.
