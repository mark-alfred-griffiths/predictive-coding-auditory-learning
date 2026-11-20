# Ownership, attribution, security and publication record

## MIT licence and owner-confirmed provenance (20 November 2026)

The original privately supplied `ubel-jpc` source archive did not contain a root `LICENSE`. The owner, **Mark A. Griffiths**, has selected the [MIT License](../LICENSE) for this separate curated distribution. Its copyright notice is `Copyright (c) 2026 Mark A. Griffiths`, and the package metadata identifies MIT.

**Owner's statement, recorded 20 November 2026:** Mark A. Griffiths confirmed that the original `ubel-jpc` implementation was developed independently on his own computers during 2026, outside any paid UCL/KCL appointment or commissioned research agreement. The owner describes developing the code with ChatGPT assistance. No other author of the retained original implementation has been identified in the inspected archive. This is a **statement by the owner**, not an independently reviewed employment contract or a formal legal opinion. The MIT licence applies to the curated project's distribution, not by itself to the entirety of the separate original private repository.

OpenAI's [Terms of Use](https://openai.com/policies/terms-of-use/) state that, as between the user and OpenAI and to the extent permitted by law, the user owns the output. This does not establish exclusive copyright protection for every AI-assisted passage, convey independently owned third-party rights, or eliminate the need to review any pasted or imported third-party material.

Eighteen retained source, test and original specification files were verified byte-identical against the owner-supplied private source snapshot. The optional adapter calls installed JPC using its public API; **the upstream JPC implementation is not bundled**. JPC is [MIT-licensed](https://github.com/thebuckleylab/jpc/blob/main/LICENSE). JAX, NumPy, Matplotlib and Pytest are installed third-party dependencies rather than copied implementations. The automated provenance review found no third-party ownership notice in retained code; this is not an exhaustive code-similarity or contractual-rights analysis.

Scientific guidance from Professors Christopher Buckley and Karl Friston, and the proposed UBEL project association with Professor Peter Howell, are acknowledgements of intellectual and research context; they do not assert authorship of or copyright title to this software. The first three curated Git commits have staging author metadata (`Research Staging <staging@localhost.invalid>`), which is not evidence of a software contributor and does not require rewriting the tested Git history. Use the correct owner Git identity for subsequent commits.

## Security, privacy and provenance disclosures

The historical pre-publication checklist called for scanning **all reachable objects and refs in a new mirror clone of the actual private GitHub remote**, not just the checked-out files. This remains a relevant follow-up check if it was not completed. Review tokens, private keys, local absolute paths, personal data, participant material, large artefacts, binary attachments and every historical filename. Check GitHub's available security alerts/settings as a separate signal; a pattern scan is not a guarantee. Avoid printing or uploading any discovered credential values.

The 20 November 2026 **archived staging history** and a reconstructed snapshot of the owner's later validated documentation/GPU changes were scanned with no recognised secret patterns or suspicious committed binary paths found. The reconstructed snapshot is **not** a clone of the live remote, so final private remote-history verification is still required. The owner supplied successful CPU/GPU results and GitHub Actions run #3 evidence at `f7644b235d1a7f9b33acf7278e2c9b956c5e6e52`.

This project contains only synthetic model data; `results/`, `.venv/`, IDE settings and caches are ignored. No real participant data or recorded speech files were intentionally included. The provenance documents **do reveal the original private repository's name, internal branch names, source commit identifiers and historical report filenames**. These identifiers do not confer access to the original repository, but the owner should review and approve these metadata disclosures, or redact them deliberately while preserving a private traceability record, even though this curated repository is already public.

## Historical pre-publication checklist (completion not fully documented)

1. Maintain the owner-confirmed copyright record, verify no undisclosed copied third-party source, and retain the selected MIT licence.
2. Confirm clean installations and numerical reproducibility, including the CPU/GPU comparison and an independent fresh clone.
3. Confirm GitHub Actions succeeds on the **exact final release candidate commit**; source/test hashes and scientific limitations must remain unchanged.
4. Scan the **actual live private repository's complete fetched Git history** (all remote refs and historical blobs) and manually review every flag, including the original-private-repo metadata disclosures.
5. Obtain **explicit owner permission** before changing a repository's visibility. The curated repository is now public; this historical record does not independently document when the visibility change was approved.
6. Following release, verify anonymous repository access, external installation and licence/attribution display.

**Current visibility: PUBLIC (curated repository).** The original `ubel-jpc` remains private and unchanged. This document preserves the historical release checklist and does not certify that the full original-remote history audit or CI validation of the exact final public commit has been completed.
