#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/mas}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/submission}"
PDF="${PDF:-$REPO_DIR/report/main.pdf}"

cd "$REPO_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/results"

cp README.md "$OUT_DIR/"
cp report/main.tex report/refs.bib "$OUT_DIR/"
if [ -f "$PDF" ]; then
  cp "$PDF" "$OUT_DIR/main.pdf"
else
  echo "warning: $PDF not found; add the compiled PDF before submitting"
fi

cp results/kaggle-3b/*.json "$OUT_DIR/results/"
git archive --format=zip HEAD -o "$OUT_DIR/mas-code.zip"

cd "$OUT_DIR"
zip -r mas-submission.zip . >/dev/null
echo "wrote $OUT_DIR/mas-submission.zip"
ls -lh "$OUT_DIR/mas-submission.zip"
