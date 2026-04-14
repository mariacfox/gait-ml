"""
io.py Ñ Data loading and file parsing for QualityAssist CSV output.

TODO: Inspect actual CSV header structure before finalizing these functions.
Key unknowns:
  - Number of header rows
  - Column naming convention (e.g., LASI_X or LASI X or L.ASI.X)
  - Whether GRF and kinematics are in the same file or separate files
  - Sampling rate (typically 100Hz kinematics, 1000Hz force plate)
"""

from pathlib import Path
import pandas as pd
import numpy as np


# Update these once you've inspected actual files
KINEMATIC_SAMPLE_RATE: float = 100.0   # Hz Ñ confirm from QualityAssist metadata
GRF_SAMPLE_RATE: float = 1000.0        # Hz Ñ confirm from force plate output

# Speed condition labels Ñ confirm against actual file naming convention
SPEED_CONDITIONS: list[str] = [
    "walk_slow",
    "walk_fast",
    "walk_faster",
    "run_slow",
    "run_fast",
    "run_selfselected",
]


def load_marker_csv(filepath: Path) -> pd.DataFrame:
    """Load a QualityAssist marker trajectory CSV.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: frame, time, and one column per marker-axis
        combination (e.g., LASI_X, LASI_Y, LASI_Z).

    Notes
    -----
    TODO: Adjust header row count and column parsing once actual file
    structure is confirmed. Run notebooks/01_data_exploration.ipynb first.
    """
    # Placeholder Ñ adjust skiprows and header based on actual file
    df = pd.read_csv(filepath, skiprows=0)
    return df


def load_grf_csv(filepath: Path) -> pd.DataFrame:
    """Load force plate GRF data.

    Parameters
    ----------
    filepath : Path
        Path to the GRF CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: frame, time, Fx, Fy, Fz (and moments if present).

    Notes
    -----
    TODO: Confirm whether GRF is in same file as kinematics or separate.
    Confirm axis convention: typically Fy is vertical in lab coordinate systems
    but verify against QualityAssist output.
    """
    df = pd.read_csv(filepath, skiprows=0)
    return df


def load_demographics(filepath: Path) -> pd.DataFrame:
    """Load subject demographic data.

    Parameters
    ----------
    filepath : Path
        Path to demographics CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by subject ID with demographic columns.
        Expected columns: subject_id, age, sex, height_m, mass_kg.
    """
    df = pd.read_csv(filepath)
    return df


def load_subject(
    subject_dir: Path,
    demographics: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Load all trials for a single subject.

    Parameters
    ----------
    subject_dir : Path
        Directory containing all trial files for one subject.
    demographics : pd.DataFrame, optional
        Demographics DataFrame; if provided, attaches subject metadata.

    Returns
    -------
    dict
        Keys are speed condition labels (e.g., 'walk_slow').
        Values are dicts with keys 'markers' and 'grf' (DataFrames).

    Notes
    -----
    TODO: Update file discovery logic once actual file naming convention
    is confirmed from data/raw/.
    """
    trials: dict[str, dict] = {}
    for condition in SPEED_CONDITIONS:
        marker_files = list(subject_dir.glob(f"*{condition}*markers*.csv"))
        grf_files = list(subject_dir.glob(f"*{condition}*grf*.csv"))
        if marker_files:
            trials[condition] = {
                "markers": load_marker_csv(marker_files[0]),
                "grf": load_grf_csv(grf_files[0]) if grf_files else None,
            }
    return trials


def get_body_weight_newtons(subject_id: str, demographics: pd.DataFrame) -> float:
    """Return subject body weight in Newtons for GRF normalization.

    Parameters
    ----------
    subject_id : str
        Subject identifier matching demographics index.
    demographics : pd.DataFrame
        Demographics DataFrame with mass_kg column.

    Returns
    -------
    float
        Body weight in Newtons (mass_kg * 9.81).
    """
    mass_kg = demographics.loc[subject_id, "mass_kg"]
    return float(mass_kg) * 9.81
