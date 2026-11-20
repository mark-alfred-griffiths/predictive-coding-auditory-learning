# Ownership, attribution, security and publication gate

## MIT licence selected; ownership and public-release review pending

The source archive did not contain an applicable root `LICENSE` file. This curated staging repository now contains an [MIT License](../LICENSE) with copyright notice for Mark A. Griffiths, and the package metadata identifies MIT. This licence selection **does not establish that all original code can lawfully be licensed by this copyright holder**. Confirm the necessary rights and permissions before any public release; do not treat the presence of `LICENSE` as public-release approval.

Confirm: (1) ownership of the original research code and whether employment/institutional or collaborator rights affect redistribution; (2) whether code modules contain externally supplied fragments; (3) attribution for independent guidance from Christopher Buckley and Karl Friston, the UBEL proposal's association with Peter Howell, and contributors, without suggesting the professors authored the software without evidence; (4) the licensing terms of JAX, NumPy, Matplotlib, Pytest and optional external JPC. JPC is **not bundled**.

## Security and privacy checks

Before any change to public visibility, inspect all files and complete Git history, not only the latest checkout. Search for credentials/tokens, private research files, local absolute paths, user credentials, identifiable participant data and large generated artifacts. Confirm the staged Git history contains only curated files. Repeat a repository-wide scan on the real GitHub repository before release; local pre-push review does not replace that check.

`results/`, `.venv/`, IDE settings and caches are ignored. The supplied model uses synthetic data only, does not depend on external datasets, and the standard demonstration code performs no network accesses. One namespace is retained from the original project; no external private code is imported for CPU mode.

## Mandatory release gate

1. Verify a clean installation and deterministic local tests on a *fresh clone*, including the user's own PyCharm workstation.
2. Inspect private GitHub Actions result, test count, software versions and exact source SHA.
3. Independently review all demonstration output, claims and negative controls; confirm source ownership and permission to distribute under the selected MIT licence.
4. Audit the final complete Git history, credentials, private documents and attribution.
5. Request **explicit user permission** for `private` → `public`. No implied permission from pushing to a private repository.
6. After release, verify anonymous access and installation from the public URL.

The original private `ubel-jpc` must never have its visibility changed as part of this curation.
