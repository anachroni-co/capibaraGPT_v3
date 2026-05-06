#!/usr/bin/env bash
#
# Small fix: remove ".github/" from .gitignore so CI workflows, issue
# templates, dependabot config, etc. are not silently dropped by git.
#
# Pattern is the same worktree approach as commit_all_backlog_002_to_005.sh.
# Run from repo root, after:
#   sed -i 's/\r$//' commit_gitignore_fix.sh

set -euo pipefail

test -d .git || { echo "Run from repo root (no .git directory here)"; exit 1; }

REPO_ROOT="$(pwd)"
BRANCH="chore/gitignore-unignore-github-dir"

echo ">>> repo root: $REPO_ROOT"
echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
echo ">>> target branch: $BRANCH"

git fetch origin

# Snapshot .gitignore into a tmp with the patch applied.
SNAP_DIR="$(mktemp -d)"
cp .gitignore "$SNAP_DIR/.gitignore.orig"

python3 - "$SNAP_DIR" <<'PY'
import os, sys
snap = sys.argv[1]
orig = open(os.path.join(snap, ".gitignore.orig"), "rb").read()
crlf = orig.count(b"\r\n") > orig.count(b"\n") // 2
text = orig.decode("utf-8").replace("\r\n", "\n")

needle = "\n.github/\n"
if needle not in text:
    raise SystemExit(".github/ line not found in .gitignore")

# Replace with an explanatory comment so the diff is self-documenting.
replacement = (
    "\n# .github/ was previously ignored which silently dropped workflows,\n"
    "# dependabot config and issue templates. Tracked again.\n"
)
text = text.replace(needle, replacement, 1)

out = text.replace("\n", "\r\n") if crlf else text
open(os.path.join(snap, ".gitignore.new"), "wb").write(out.encode("utf-8"))
print("patched .gitignore (CRLF=" + str(crlf) + ")")
PY

MSG='chore(gitignore): unignore .github/ so workflows & templates are tracked

Line 59 of .gitignore had a blanket ".github/" pattern. That silently
prevented any NEW file under .github/ (issue templates, dependabot
config, additional workflows) from being added by git. Existing
workflows (docs.yml, python-app.yml) kept working only because they
were already tracked before the ignore was added; they were an
accident waiting for a "git add -f" bandaid.

Removing the pattern + short explanatory comment so future contributors
do not re-add it.'

wt_dir="$(mktemp -d)"
echo
echo "========================================================="
echo " BRANCH: $BRANCH"
echo " worktree: $wt_dir"
echo "========================================================="

git worktree add -B "$BRANCH" "$wt_dir" origin/main >/dev/null

cp "$SNAP_DIR/.gitignore.new" "$wt_dir/.gitignore"

(
    cd "$wt_dir"

    echo "--- diff ---"
    git --no-pager diff -- .gitignore

    git add .gitignore
    git status --short

    echo "--- press Enter to commit $BRANCH, Ctrl+C to abort ---"
    read -r _

    git commit -m "$MSG"
    git push -u origin "$BRANCH"
)

git worktree remove "$wt_dir"
rm -rf "$SNAP_DIR"

echo
echo "Done. Open the PR at:"
echo "  https://github.com/anachroni-co/capibaraGPT_v3/pull/new/$BRANCH"
echo
echo "Your original branch is unchanged."
