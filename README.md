# gait-ml

A Python pipeline for processing 3D motion capture and force plate data from multi-speed gait trials. Computes joint kinematics and ground reaction force features from raw marker trajectory CSVs, with downstream ML classification of gait patterns across speed conditions.

## Dataset

70 anonymized subjects (IRB-approved). Each subject completed:
- Walking at slower, fast, and faster speeds
- Running at slower and faster prescribed paces, plus self-selected comfortable pace
- Standing balance trials (center of pressure)
- Standing pose captures

Data includes 3D marker trajectories (Qualisys CSV output), synchronized force plate GRF data, and subject demographics.

## Project structure

```
gait_ml/          Python biomechanics pipeline package
notebooks/        Exploratory analysis and validation
tests/            pytest test suite
data/raw/         Raw CSVs — gitignored
data/processed/   Pipeline output — gitignored
```

## Setup

```bash
uv sync
uv run pytest
uv run jupyter lab
```

## Phases

- **Phase 1 (current):** Python biomechanics pipeline — filtering, joint angle computation, GRF feature extraction
- **Phase 2:** ML classification of speed condition (sklearn baseline, leave-one-subject-out CV, MLflow tracking)
- **Phase 3:** PyTorch sequence model (1D CNN / LSTM on time-normalized waveforms)

## Key validation

Before Phase 2: outputs are validated against existing MATLAB reference implementations on matched trials. See `notebooks/02_preprocessing_validation.ipynb`.
