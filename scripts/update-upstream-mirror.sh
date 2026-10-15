#!/bin/sh
# Keep the local `upstream` branch a pristine mirror of public upstream/master.
# Never touches the working tree or the checked-out branch: it fetches, then
# fast-forwards the `upstream` ref only if it is strictly behind, using a
# non-checkout update. If `upstream` is somehow checked out or has diverged
# (should never happen — nothing is implemented on it), it refuses and logs.
set -eu

REPO=/local/mnt2/workspace2/llama.cpp.qualcomm
LOG="$REPO/.git/upstream-mirror.log"
BRANCH=upstream
REMOTE=upstream
REMOTE_BRANCH=master

cd "$REPO"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Never update the branch while it is the one checked out (would touch work tree).
cur=$(git symbolic-ref --quiet --short HEAD || echo DETACHED)
if [ "$cur" = "$BRANCH" ]; then
  echo "$(ts) SKIP: '$BRANCH' is checked out; not updating" >> "$LOG"
  exit 0
fi

git fetch --quiet "$REMOTE" "$REMOTE_BRANCH" || { echo "$(ts) FETCH FAILED" >> "$LOG"; exit 0; }

local_sha=$(git rev-parse "$BRANCH")
remote_sha=$(git rev-parse "$REMOTE/$REMOTE_BRANCH")

if [ "$local_sha" = "$remote_sha" ]; then
  exit 0   # already current, stay quiet
fi

# Only fast-forward: local must be an ancestor of remote (i.e. strictly behind,
# no local commits). This guarantees we never clobber accidental local work.
if git merge-base --is-ancestor "$local_sha" "$remote_sha"; then
  git branch -f "$BRANCH" "$remote_sha"
  echo "$(ts) UPDATED $BRANCH ${local_sha%${local_sha#???????}}..${remote_sha%${remote_sha#???????}}" >> "$LOG"
else
  echo "$(ts) REFUSED: '$BRANCH' has diverged from $REMOTE/$REMOTE_BRANCH (local=$local_sha); manual check needed" >> "$LOG"
fi
