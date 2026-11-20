#!/usr/bin/env bash
# Safe human-controlled Git synchronisation; never merge or force-push silently.
set -euo pipefail
cd "$(dirname "$0")/.."
command="${1:-status}"
branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  echo 'Detached HEAD: select an explicit local branch first.' >&2
  exit 2
fi
if [[ "$command" == status ]]; then
  git status -sb
  printf 'HEAD: '; git rev-parse HEAD
  git remote -v
  exit 0
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  echo 'No origin remote; create/clone the new PRIVATE repository first.' >&2
  exit 2
fi
case "$command" in
  pull)
    if [[ -n "$(git status --porcelain)" ]]; then
      echo 'Refusing pull: commit, stash, or discard uncommitted changes first.' >&2
      exit 2
    fi
    git fetch --prune origin
    git pull --ff-only origin "$branch"
    printf 'Pulled HEAD: '; git rev-parse HEAD
    echo 'Rerun tests: bash scripts/validate_local.sh'
    ;;
  push)
    # Only committed changes are pushed; no automatic commit or force option.
    git push -u origin HEAD
    ;;
  *)
    echo 'Usage: bash scripts/sync_local.sh [status|pull|push]' >&2
    exit 2
    ;;
esac
