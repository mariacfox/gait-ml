# gait-ml

## Project overview

Python pipeline for processing 3D motion capture and force plate data from multi-speed gait trials, with downstream ML classification of gait patterns across speed conditions.

**Dataset:** 70 anonymized subjects (IRB-approved, already anonymized). Each subject has:
- 3 walking conditions × 3 trials each: preferred, predetermined, and Froude-matched speeds
- 3 running conditions × 3 trials each: predetermined, Froude A, and Froude B speeds
- Split-belt treadmill GRF data (left and right belts) from the same trials
- Standing CoP trials (QuietStance, 5 trials) on a separate AMTI force plate
- Static T-pose calibration trial (Tcap)
- Subject demographics and speed data in `d_subjectData.csv` and `d_surveyData.csv`

**Data format:** TSV output from Qualisys (motion capture QA software). GRF data is in separate TSV files per force plate, named with a suffix on the base trial name.

**Goal:** Build a modular, well-tested Python biomechanics pipeline (Phase 1), then a ML classification system to predict gait speed condition from kinematics and GRF features (Phase 2), then a PyTorch sequence model (Phase 3).

---

## Repo structure

```
gait-ml/
├── CLAUDE.md                   # this file
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/                    # original TSVs — gitignored, never modify
│   └── processed/              # pipeline output — gitignored
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   └── 03_feature_engineering.ipynb
├── gait_ml/
│   ├── __init__.py
│   ├── io.py                   # data loading and file parsing
│   ├── preprocessing.py        # filtering, normalization
│   ├── kinematics.py           # joint angle computation
│   ├── grf.py                  # GRF feature extraction
│   ├── segmentation.py         # gait event detection
│   └── features.py             # feature matrix assembly for ML
├── tests/
│   ├── test_kinematics.py
│   └── test_preprocessing.py
└── mlflow/                     # experiment tracking (Phase 2+)
```

---

## Current phase: Phase 1 — Preprocessing pipeline

**Status:** Data structure and MATLAB processing parameters fully confirmed. Ready to implement.

All filter parameters, TSV parsing offsets, gap-fill strategy, and event detection logic have been confirmed directly from the MATLAB source scripts.

**Do not start on ML (Phase 2) until the preprocessing pipeline has been validated against MATLAB output.**

---

## Data format

### File naming convention

Files are named `{SubjectID}_{ConditionName}{TrialNumber}{Suffix}.tsv`.

- **Subject IDs** follow a two-letter + number code (e.g., `FS6`, `MS1`, `FT1`). First letter is likely sex (F/M), second is height group (S=short, T=tall) — but treat as opaque codes; ground truth is in `d_subjectData.csv`.
- **Trial numbers** are 1–3 (three trials per condition).
- **Suffixes** indicate file type:
  - *(no suffix)*: kinematics — 3D marker trajectories, 160 Hz
  - `_f_4`: GRF — Bertec treadmill **left belt**, 1120 Hz
  - `_f_5`: GRF — Bertec treadmill **right belt**, 1120 Hz
  - `_f_3`: GRF — AMTI static force plate (QuietStance and Tcap trials only), 1120 Hz
  - `_a`: analog channels — raw force + harness load cell (see note below)

**Analog `_a` files:** contain L_Fx/Fy/Fz/Mx/My/Mz, R_Fx/Fy/Fz/Mx/My/Mz, and Load_Cell (the running harness we built). These are raw unprocessed analog outputs. For GRF analysis use `_f_4`/`_f_5` instead (already processed into Force/Moment/COP). The Load_Cell channel can be ignored.

### Condition names in filenames

| Filename condition   | Description                                      | Label (use in code)        |
|----------------------|--------------------------------------------------|----------------------------|
| `WalkingPreferred`   | Subject's self-selected preferred walking speed  | `walk_preferred`           |
| `WalkingPreDetermined` | Prescribed fixed walking speed (1.3 m/s)       | `walk_predetermined`       |
| `WalkingFroude`      | Froude number-matched walking speed              | `walk_froude`              |
| `RunningPreDetermined` | Prescribed fixed running speed (3.3 m/s)       | `run_predetermined`        |
| `RunningFroudeA`     | Froude A running speed                           | `run_froude_a`             |
| `RunningFroudeB`     | Froude B running speed                           | `run_froude_b`             |
| `QuietStance`        | Standing still (CoP/balance trials)              | `quiet_stance`             |
| `Tcap`               | Static T-pose calibration                        | `tcap`                     |
| `empty`              | Empty treadmill — baseline noise for the session | `empty`                    |

Actual speeds per subject are in `d_subjectData.csv` (columns: `WalkingPreferred`, `WalkingPreDetermined`, `WalkingFroude`, `RunningPreDetermined`, `RunningFroudeA`, `RunningFroudeBcalc`, `RunningFroudeBrec`).

### Kinematics TSV format

Header lines (key–value, tab-separated) before the data:

```
NO_OF_FRAMES    <n>
NO_OF_CAMERAS   6
NO_OF_MARKERS   48          # 54 for Tcap and QuietStance (adds medial calibration markers)
FREQUENCY       160
NO_OF_ANALOG    0
ANALOG_FREQUENCY 0
DESCRIPTION     --
TIME_STAMP      <date>  <timestamp>
DATA_INCLUDED   3D
MARKER_NAMES    CLAV  C7  LASIS  ...   (space-separated on one line)
Frame   Time    CLAV X  CLAV Y  CLAV Z  C7 X  ...   (column header row)
<data rows>
```

Data columns: `Frame`, `Time` (seconds), then `{MARKER} X`, `{MARKER} Y`, `{MARKER} Z` for all markers.
Missing/occluded markers are represented as `0.000 0.000 0.000`.

**Parsing notes (from MATLAB `importMarkerData.m` / `findMarkerNames.m`):**
- Skip 11 header lines; data begins at line 12.
- Marker column names are derived from the column header row (line 11: `Frame   Time   CLAV X   CLAV Y   ...`) by stripping all non-word characters — so `CLAV X` → `CLAVX`, `C7 X` → `C7X`, etc. This is the naming convention used throughout: `{MARKER}{AXIS}` with no separator (e.g., `RANKX`, `LANKY`, `CLAVZ`).
- The first two columns are `Frame` (int) and `Time` (float, seconds). Marker data starts at column 3.

### Force plate TSV format (`_f_4` / `_f_5`)

Header lines, then:
```
SAMPLE   TIME   Force_X   Force_Y   Force_Z   Moment_X   Moment_Y   Moment_Z   COP_X   COP_Y   COP_Z
```

Units: forces in N, moments in N·mm, COP in mm. Frequency: 1120 Hz (7× kinematics rate).

- `_f_4` = `FORCE_PLATE_NAME: Bertec Treadmill L Belt`
- `_f_5` = `FORCE_PLATE_NAME: Bertec Treadmill R Belt`

The static AMTI plate (`_f_3`) has the same column structure but is only present for QuietStance and Tcap.

**Parsing notes (from MATLAB `importForceData.m`):**
- Skip 24 header lines; data begins at line 25.
- Read 10 columns: `SAMPLE`, `TIME`, `Force_X`, `Force_Y`, `Force_Z`, `Moment_X`, `Moment_Y`, `Moment_Z`, `COP_X`, `COP_Y`. The 11th column (`COP_Z`) is all zeros and is dropped.
- `Force_Z` is the vertical ground reaction force (used for gait event detection and normalization).

### Demographic CSVs

- `d_subjectData.csv`: one row per subject — SubID, Date, Sex, Age, Weight (kg), HeightShoes, HeightNoShoes (cm), ArmLength_R, LegLength_R, LegLength_L (cm), actual speeds for each condition (m/s), Froude numbers, BMI.
- `d_surveyData.csv`: one row per subject — detailed demographics, shoe size, injury history, sport background.

Subject body weight for GRF normalization: use mean `Force_Z` from `QuietStance3_f_3.tsv` (the AMTI plate), not the CSV weight column. This is what the MATLAB pipeline does (`importFiles.m`). The `Weight` column in `d_subjectData.csv` is in kg and can be used as a fallback.

---

## Architecture decisions

- **scipy.signal** for Butterworth filtering (zero-phase via `filtfilt`, matching MATLAB behavior)
- **pandas + numpy** as core data structures throughout pipeline
- **MLflow** for experiment tracking in Phase 2+
- **PyTorch** for sequence models in Phase 3 (1D CNN or LSTM on time-normalized waveforms)
- **scikit-learn** for baseline classifiers in Phase 2
- All processing functions should be **pure functions** (no side effects, no global state)
- All functions should have **type annotations** and **numpy-style docstrings**

---

## Biomechanics context

- **Gait cycle:** heel strike → heel strike (ipsilateral). Each trial contains multiple cycles.
- **Time normalization:** 101 points (0–100% of gait cycle), pchip interpolation. Confirmed from `gaitCycleNormalization.m`.
- **Filtering — kinematics (confirmed from `filterMarkerData.m`):**
  - 4th-order zero-phase Butterworth low-pass, **fc = 8 Hz**, fs = 160 Hz
  - Applied via `filtfilt` (matches `scipy.signal.filtfilt`)
- **Filtering — GRF forces (confirmed from `filtForceCOP.m`):**
  - 4th-order zero-phase Butterworth low-pass, **fc = 8 Hz**, fs = 1120 Hz
  - Applied to Force_X/Y/Z only — **moments are NOT filtered**, passed through as-is
- **Filtering — COP (confirmed from `filtForceCOP.m`):**
  - 4th-order zero-phase Butterworth low-pass, **fc = 15 Hz**, fs = 1120 Hz
  - Applied to COP_X and COP_Y; COP_Z is zeros, ignored
- **Missing marker handling (confirmed from `filterMarkerData.m`):** replace zeros with NaN, then pchip interpolation (`fillmissing(..., 'pchip')`), then filter. Fields 1–2 (Frame, Time) are skipped; only marker coordinate fields are processed.
- **GRF event detection (confirmed from `gaitEventDetection.m`):**
  - Threshold: **15 N** on vertical GRF (Force_Z)
  - Stances with peak force < 150 N are discarded (foot crossover / artifact filter)
  - Returns: heel strike index, toe-off index, next heel strike index per cycle
- **Running belt selection (confirmed from `importFiles.m`):** for running trials, only the belt on which the subject is running is used. Determined by `max(Force_Z) > body_weight_N`. Walking uses both belts (left and right independently).
- **Coordinate system (confirmed from data):** Y = vertical (up), X = anteroposterior (forward along treadmill), Z = mediolateral. Confirmed by inspection: clavicle Y ≈ 1210 mm (plausible standing height), left/right ASIS markers differ primarily in Z (mediolateral separation ≈ 227 mm).
- **Force plate timing:** GRF is sampled at 7× the kinematics rate (1120 Hz vs 160 Hz). Downsample or upsample to a common rate before synchronizing.
- **Joint angles of interest (priority order):**
  1. Knee flexion/extension (sagittal)
  2. Hip flexion/extension (sagittal)
  3. Ankle dorsiflexion/plantarflexion (sagittal)
  4. Pelvic tilt, obliquity, rotation

### Marker set — dynamic trials (48 markers)

| Segment      | Markers                                                |
|--------------|--------------------------------------------------------|
| Trunk        | CLAV, C7                                               |
| Pelvis       | LASIS, RASIS, LPSIS, RPSIS                             |
| Left arm     | LACR, LBICEPU, LBICEP, LBICEPL, LELBOW, LFARM, LRAD, LULNA |
| Right arm    | RACR, RBICEPU, RBICEP, RBICEPL, RELBOW, RFARM, RRAD, RULNA |
| Left leg     | LTRO, LTHIGHU, LTHIGH, LTHIGHL, LKNEE, LSHANKU, LSHANK, LSHANKL, LANK, LHEEL, LMTI, LMTV, LTOE |
| Right leg    | RTRO, RTHIGHU, RTHIGH, RTHIGHL, RKNEE, RSHANKU, RSHANK, RSHANKL, RANK, RHEEL, RMTI, RMTV, RTOE |

Static/calibration trials (Tcap, QuietStance) have 54 markers — adds medial calibration markers: LELBOWM, RELBOWM, LKNEEM, RKNEEM, LANKM, RANKM. These are used for joint center estimation and are not present in dynamic trials.

---

## Key validation requirement

Before Phase 2: run a subject through both the MATLAB pipeline and the Python pipeline and plot the outputs overlaid. They should be nearly identical. Document this in `notebooks/02_preprocessing_validation.ipynb`. This is the single most important correctness check.

---

## Python environment

- **Package manager:** uv
- **Python:** 3.14
- Install: `uv sync`
- Run tests: `uv run pytest`
- Run notebooks: `uv run jupyter lab`

---

## Data privacy

- Data is already anonymized per original IRB approval
- `data/` directory is fully gitignored — **never commit raw or processed data**
- Subject IDs in filenames are alphanumeric codes only, no PII
- Do not log or print any subject-level data to stdout in production code; use aggregate summaries only

---

## Code style

- **Formatter:** ruff (via uv)
- **Type hints:** required on all public functions
- **Docstrings:** numpy style
- **Tests:** pytest; aim for at least one test per public function in `gait_ml/`
- **No notebooks in CI** — notebooks are for exploration only; finalized logic moves to `gait_ml/`

---

## Phases

### Phase 1 (current): Python biomechanics pipeline
- `io.py`: load Qualisys TSVs, parse marker trajectories and GRF
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
