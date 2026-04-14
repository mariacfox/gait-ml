# gait-ml

## Project overview

Python pipeline for processing 3D motion capture and force plate data from multi-speed gait trials, with downstream ML classification of gait patterns across speed conditions.

**Dataset:** 70 anonymized subjects (IRB-approved, already anonymized). Each subject has:
- Multiple walking speeds: slower, fast, faster (subject-defined and prescribed)
- Multiple running speeds: slower, faster (prescribed paces) + self-selected comfortable pace
- Force plate data (ground reaction forces) from same trials
- Standing center-of-pressure (CoP) trials
- Standing pose trials
- Demographic data

**Data format:** CSV output from QualityAssist (motion capture QA software) with labeled 3D marker positions. GRF data is in separate force plate CSVs.

**Goal:** Build a modular, well-tested Python biomechanics pipeline (Phase 1), then a ML classification system to predict gait speed condition from kinematics and GRF features (Phase 2), then a PyTorch sequence model (Phase 3).

---

## Repo structure

```
gait-ml/
??? CLAUDE.md                   # this file
??? README.md
??? pyproject.toml
??? data/
?   ??? raw/                    # original CSVs Ñ gitignored, never modify
?   ??? processed/              # pipeline output Ñ gitignored
??? notebooks/
?   ??? 01_data_exploration.ipynb
?   ??? 02_preprocessing_validation.ipynb
?   ??? 03_feature_engineering.ipynb
??? gait_ml/
?   ??? __init__.py
?   ??? io.py                   # data loading and file parsing
?   ??? preprocessing.py        # filtering, normalization
?   ??? kinematics.py           # joint angle computation
?   ??? grf.py                  # GRF feature extraction
?   ??? segmentation.py         # gait event detection
?   ??? features.py             # feature matrix assembly for ML
??? tests/
?   ??? test_kinematics.py
?   ??? test_preprocessing.py
??? mlflow/                     # experiment tracking (Phase 2+)
```

---

## Current phase: Phase 1 Ñ Preprocessing pipeline

**Status:** Scaffolding complete. Need to:
1. Inspect actual CSV header structure from QualityAssist output to finalize `io.py`
2. Confirm marker set (determines which joint angles are computable)
3. Confirm whether GRF and kinematic data are in the same file or separate files per trial
4. Validate Python filter output against existing MATLAB scripts

**Do not start on ML (Phase 2) until the preprocessing pipeline has been validated against MATLAB output.**

---

## Architecture decisions

- **pyomeca** for biomechanics signal processing (filters, normalization, file I/O). Prefer pyomeca utilities over rolling custom implementations where possible.
- **scipy.signal** for Butterworth filtering (zero-phase via `filtfilt`, matching MATLAB behavior)
- **pandas + numpy** as core data structures throughout pipeline
- **MLflow** for experiment tracking in Phase 2+
- **PyTorch** for sequence models in Phase 3 (1D CNN or LSTM on time-normalized waveforms)
- **scikit-learn** for baseline classifiers in Phase 2
- All processing functions should be **pure functions** (no side effects, no global state)
- All functions should have **type annotations** and **numpy-style docstrings**

---

## Biomechanics context

- **Gait cycle:** heel strike ? heel strike (ipsilateral). Each trial contains multiple cycles.
- **Time normalization:** 101 points (0Ð100% of gait cycle) is the field standard. Use this everywhere.
- **Filtering:** 6 Hz low-pass Butterworth for kinematics is standard. Confirm cutoff against original MATLAB scripts before hardcoding.
- **GRF event detection:** threshold crossing on vertical GRF (typically 20N). Subject body weight needed for normalization Ñ pull from demographics CSV.
- **Coordinate system:** confirm axis convention from QualityAssist output (likely X=mediolateral, Y=vertical, Z=anteroposterior, but verify).
- **Joint angles of interest (priority order):**
  1. Knee flexion/extension (sagittal)
  2. Hip flexion/extension (sagittal)
  3. Ankle dorsiflexion/plantarflexion (sagittal)
  4. Pelvic tilt, obliquity, rotation (if full-body marker set available)

---

## Speed condition labels

Each subject has these conditions Ñ use these exact string labels throughout:
- `walk_slow`
- `walk_fast`
- `walk_faster`
- `run_slow`
- `run_fast`
- `run_selfselected`

Confirm label naming against actual file naming convention in `data/raw/` before hardcoding.

---

## Key validation requirement

Before Phase 2: run a subject through both the MATLAB pipeline and the Python pipeline and plot the outputs overlaid. They should be nearly identical. Document this in `notebooks/02_preprocessing_validation.ipynb`. This is the single most important correctness check.

---

## Python environment

- **Package manager:** uv
- **Python:** 3.11+
- Install: `uv sync`
- Run tests: `uv run pytest`
- Run notebooks: `uv run jupyter lab`

---

## Data privacy

- Data is already anonymized per original IRB approval
- `data/` directory is fully gitignored Ñ **never commit raw or processed data**
- Subject IDs in filenames are numeric codes only, no PII
- Do not log or print any subject-level data to stdout in production code; use aggregate summaries only

---

## Code style

- **Formatter:** ruff (via uv)
- **Type hints:** required on all public functions
- **Docstrings:** numpy style
- **Tests:** pytest; aim for at least one test per public function in `gait_ml/`
- **No notebooks in CI** Ñ notebooks are for exploration only; finalized logic moves to `gait_ml/`

---

## Phases

### Phase 1 (current): Python biomechanics pipeline
- `io.py`: load QualityAssist CSVs, parse marker trajectories and GRF
- `preprocessing.py`: Butterworth filter, time normalization, gap detection
- `kinematics.py`: joint angle computation from marker triads
- `grf.py`: GRF feature extraction (peak, loading rate, impulse, contact time)
- `segmentation.py`: heel strike / toe-off event detection from GRF threshold
- `features.py`: assemble feature matrix across subjects and conditions

### Phase 2: ML classification (speed condition)
- Baseline: sklearn classifiers (RF, SVM, logistic regression) on tabular features
- Evaluation: leave-one-subject-out cross-validation (within-subject repeated measures design)
- MLflow tracking for all experiments
- Interpretability: feature importance, SHAP values

### Phase 3: PyTorch sequence model
- Input: time-normalized waveforms (101 points * n_channels)
- Architecture: 1D CNN baseline, then LSTM/GRU
- Training: same LOSO cross-validation scheme as Phase 2
- Compare against Phase 2 tabular baseline
