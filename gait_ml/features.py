"""
features.py Ñ Feature matrix assembly across subjects and conditions.

Assembles the tabular feature matrix used in Phase 2 ML classification.
Each row = one gait cycle. Columns = extracted features + metadata.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def build_feature_row(
    subject_id: str,
    condition: str,
    cycle_idx: int,
    grf_features: dict[str, float],
    kinematic_features: dict[str, float] | None = None,
) -> dict:
    """Build a single feature row for the ML feature matrix.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    condition : str
        Speed condition label (e.g., 'walk_slow').
    cycle_idx : int
        Gait cycle index within the trial.
    grf_features : dict
        Output from grf.extract_grf_features().
    kinematic_features : dict, optional
        Kinematic summary features (e.g., peak joint angles, ROM).

    Returns
    -------
    dict
        Single row dict suitable for pd.DataFrame construction.
    """
    row = {
        "subject_id": subject_id,
        "condition": condition,
        "cycle_idx": cycle_idx,
    }
    row.update(grf_features)
    if kinematic_features:
        row.update(kinematic_features)
    return row


def build_feature_matrix(rows: list[dict]) -> pd.DataFrame:
    """Assemble feature matrix from list of row dicts.

    Parameters
    ----------
    rows : list[dict]
        Output of repeated calls to build_feature_row().

    Returns
    -------
    pd.DataFrame
        Feature matrix with subject_id, condition, and all feature columns.
    """
    return pd.DataFrame(rows)


def kinematic_summary_features(
    normalized_angle: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Compute summary statistics from a time-normalized joint angle waveform.

    Parameters
    ----------
    normalized_angle : np.ndarray
        Time-normalized angle in degrees, shape (101,).
    prefix : str
        Feature name prefix (e.g., 'knee', 'hip', 'ankle').

    Returns
    -------
    dict[str, float]
        Summary features: peak, min, range, mean, and values at key events.
    """
    return {
        f"{prefix}_peak_deg": float(np.max(normalized_angle)),
        f"{prefix}_min_deg": float(np.min(normalized_angle)),
        f"{prefix}_rom_deg": float(np.max(normalized_angle) - np.min(normalized_angle)),
        f"{prefix}_mean_deg": float(np.mean(normalized_angle)),
        f"{prefix}_at_hs_deg": float(normalized_angle[0]),    # at heel strike
        f"{prefix}_at_midstance_deg": float(normalized_angle[50]),  # ~50% of cycle
    }
