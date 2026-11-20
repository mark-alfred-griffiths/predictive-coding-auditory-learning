# Pop!_OS, PyCharm and safe Git synchronisation

This is the intended **regular development workflow**, not only a one-off release test. Keep the private original `ubel-jpc` and the curated repository in **different directories and Git remotes**. Do not copy `.git` from the original project.

## Clone the existing private GitHub remote

The private GitHub repository already exists. The **preferred** setup for another machine is an ordinary clean clone:

```bash
mkdir -p ~/PycharmProjects
cd ~/PycharmProjects
git clone git@github.com:mark-alfred-griffiths/predictive-coding-auditory-learning.git
cd predictive-coding-auditory-learning
git remote -v
git status -sb
git rev-parse HEAD
```

This SSH form requires configured SSH keys. With HTTPS authentication instead, use:

```bash
git clone https://github.com/mark-alfred-griffiths/predictive-coding-auditory-learning.git
```

**Existing working copies:** the owner already established an upstream `origin/main` and an independent clean clone. Use `git pull --ff-only origin main` in each checkout, not `git init` and not another repository creation. Check `git remote -v` first; never add the old `ubel-jpc` as `origin`. Keep the GitHub repository private.

## PyCharm environment

1. Open this directory with **File → Open**; verify its Git root points to the new curated repository only.
2. Choose **Settings → Project → Python Interpreter → Add Interpreter → Existing Conda** and select `pc-auditory` (or `pc-auditory-verify` for the separate verification clone). The owner used Python 3.13.18. A virtual environment remains an unverified alternative.
3. In the integrated terminal, run `conda activate pc-auditory` (or `pc-auditory-verify`) and `python -m pip install -e '.[test]' -c constraints-cpu.txt`. Verify the interpreter with `which python` and dependencies with `python -m pip check`.
4. Set Run Configuration **Script path** to `examples/truncated_inference.py` or `examples/auditory_learning.py`. Working directory must be the repository root; environment variables: `JAX_PLATFORMS=cpu;JAX_ENABLE_X64=1;MPLBACKEND=Agg` (enter as separate pairs in the UI).
5. Set arguments to `--outdir results/truncated_inference` or `--seed 2027 --trials 80 --outdir results/auditory_learning`.
6. Run tests via `python -m pytest -q` in the terminal or PyCharm's pytest runner. For reproducibility, save the terminal output and `git rev-parse HEAD` together.

In the terminal you can execute `bash scripts/validate_local.sh`. The standard workflow is **CPU-only**, even on the NVIDIA RTX 3060 Ti machine. The owner independently tested the development and fresh-clone Conda environments (28/28 tests), and private GitHub Actions run #2 passed at `58fb985`. For a separate CUDA-only check, follow [GPU validation](gpu_validation.md); do not modify the shared CPU environment or treat a CUDA test as new biological evidence.

## Everyday pull/test/commit/push cycle

Start by checking your branch, current HEAD and clean working state:

```bash
git status -sb
git remote -v
bash scripts/sync_local.sh status
bash scripts/sync_local.sh pull
bash scripts/validate_local.sh
```

`sync_local.sh pull` requires a clean checkout and uses `git fetch --prune` followed by `git pull --ff-only` (it will **not** silently merge diverged histories). It refuses missing remotes/uncommitted changes. Never force-push `main`; instead create a topic branch for a scientific change:

```bash
git switch -c docs/my-review-note
# edit, test, inspect changes
git diff --check
git status -sb
git add docs/my-review-note.md
git commit -m "Document local review findings"
bash scripts/sync_local.sh push
```

Create a private pull request, let GitHub Actions run and review the diff before merging. Other machines should `git pull --ff-only` again before their next work session. If PyCharm and a terminal are open together, avoid simultaneous Git write operations in different windows.

## Keep experiments distinct from Git source

Generated `results/` files, `.venv/`, `.idea/`, caches and wheel builds are ignored. Version *code/configuration/expected small reference metadata* intentionally, not raw simulation output by default. Record for every meaningful execution: date, exact commit `git rev-parse HEAD`, environment (`python --version`, `python -m pip freeze`), command, seed, configuration JSON and relevant outputs. Do not copy private proposals or personal data into a future public commit.

A SHA identifies a particular version. **A local pull changes the version**, so rerun tests and compare outputs before asserting that the pulled version matches the one you previously validated.

## Verify the remote and release state

Use `git remote -v`, `git log -3 --oneline`, `git status -sb` and private GitHub Actions. Verify the target GitHub repository is still **private** in its settings. Only an explicit later release authorisation permits visibility change. Never edit the original private repository or its history to create the public project.
