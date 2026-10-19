#!/usr/bin/env bash
# One-command demo: generate a synthetic dataset with known ground truth and analyse it.
#
#   bash examples/run_demo.sh [OUTPUT_ROOT]
#
# Produces, for every input image, the displacement and strain overlays, the machine-readable
# fields, the summary report and the extra visualizations.  Everything is written under
# OUTPUT_ROOT (default: /tmp/dic_demo).
set -euo pipefail

OUT_ROOT="${1:-/tmp/dic_demo}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PYTHON:-python}"

echo "== 1/3  generating demo datasets into ${OUT_ROOT}/data"
"$PY" "${HERE}/generate_demo.py" --output "${OUT_ROOT}/data" --preset all --frames 12

for preset in rotation masked illumination noisy nonuniform; do
  echo
  echo "== 2/3  analysing preset '${preset}'"
  "$PY" -m dic "${OUT_ROOT}/data/${preset}" "${OUT_ROOT}/results/${preset}" \
      --scale 0.05 --unit mm --dt 0.5
  echo "    -> ${OUT_ROOT}/results/${preset}/summary_report.md"
done

echo
echo "== 3/3  running the synthetic validation suite"
"$PY" -m dic --validate "${OUT_ROOT}/validation" --validate-quick || true
echo "    -> ${OUT_ROOT}/validation/validation_report.md"

echo
echo "Done.  Open, for example:"
echo "  ${OUT_ROOT}/results/rotation/figures/displacement/disp_0012.png"
echo "  ${OUT_ROOT}/results/rotation/summary_report.md"
