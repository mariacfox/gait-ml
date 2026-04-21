#!/usr/bin/env bash
# Train all six model variants (3 classification + 3 regression) with 10-fold CV.
# Uses 25% subsample of cycles per subject/condition due to memory constraints.
# Checkpoints saved to checkpoints/, results to results/.

set -euo pipefail

echo "=== Classification (walking / slow run / fast run) ==="
uv run python scripts/run_cv.py --task classification --n-classes 3 --n-folds 10 --model grf_only    --subsample-frac 0.25
uv run python scripts/run_cv.py --task classification --n-classes 3 --n-folds 10 --model marker_only --subsample-frac 0.25
uv run python scripts/run_cv.py --task classification --n-classes 3 --n-folds 10 --model two_tower   --subsample-frac 0.25

echo "=== Regression (speed in m/s) ==="
uv run python scripts/run_cv.py --task regression --n-folds 10 --subject-data data/raw/d_subjectData.csv --model grf_only    --subsample-frac 0.25
uv run python scripts/run_cv.py --task regression --n-folds 10 --subject-data data/raw/d_subjectData.csv --model marker_only --subsample-frac 0.25
uv run python scripts/run_cv.py --task regression --n-folds 10 --subject-data data/raw/d_subjectData.csv --model two_tower   --subsample-frac 0.25

echo "=== All done ==="
