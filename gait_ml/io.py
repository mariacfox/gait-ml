"""
io.py — Data loading and file parsing for Qualisys TSV output.

TSV format details and processing logic confirmed directly from MATLAB scripts:
  importMarkerData.m, importForceData.m, importFiles.m, findMarkerNames.m
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from gait_ml.config import DEFAULT_LAB_CONFIG as _CFG


def load_marker_tsv(filepath: Path) -> pd.DataFrame:
    """Load a Qualisys marker trajectory TSV file.

    Parameters
    ----------
    filepath : Path
        Path to the kinematics TSV file (no suffix variant).

    Returns
    -------
    pd.DataFrame
        Columns: Frame (int), Time (float, seconds), then one column per
        marker-axis combination using ``{MARKER}{AXIS}`` naming —
        e.g. ``CLAVX``, ``CLAVY``, ``CLAVZ``, ``RANKX``. Occluded frames
        are NaN (Qualisys zeros replaced on load).

    Notes
    -----
    TSV structure (confirmed from ``importMarkerData.m``,
    ``findMarkerNames.m``):

    - Lines 1–10: key-value metadata
    - Line 11: column header (``Frame  Time  CLAV X  CLAV Y  ...``)
    - Line 12+: numeric data

    Column names are derived by stripping all non-word characters from the
    header row, matching MATLAB ``regexprep(name, '\\W', '')``. This
    converts ``CLAV X`` → ``CLAVX``, ``C7 X`` → ``C7X``, etc.
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=_CFG.qualisys.marker_header_row,
        na_values=list(_CFG.qualisys.na_values),
        low_memory=False,
    )

    # Strip non-word chars from column names — matches MATLAB regexprep(name,'\W','')
    df.columns = pd.Index([re.sub(r"\W", "", col) for col in df.columns])

    # Drop spurious trailing columns from trailing tabs in Qualisys TSVs.
    # pandas names extra columns "Unnamed: N"; after \W-stripping that becomes
    # "UnnamedN" — so filter on both empty names and the Unnamed pattern.
    df = df.loc[:, ~df.columns.str.match(r"^(|Unnamed\d+)$")]

    # Replace zeros with NaN in marker columns (Frame=col 0, Time=col 1 are kept as-is)
    marker_cols = df.columns[2:]
    df[marker_cols] = df[marker_cols].replace(0.0, np.nan)

    return df


def load_force_tsv(filepath: Path) -> pd.DataFrame:
    """Load a Qualisys force plate TSV file.

    Parameters
    ----------
    filepath : Path
        Path to the force TSV file (``_f_3``, ``_f_4``, or ``_f_5`` suffix).

    Returns
    -------
    pd.DataFrame
        Columns: SAMPLE (int), TIME (float, seconds), Force_X, Force_Y,
        Force_Z (N), Moment_X, Moment_Y, Moment_Z (N·mm), COP_X, COP_Y (mm).
        ``COP_Z`` is dropped — it is always zero in the Qualisys output.

    Notes
    -----
    TSV structure (confirmed from ``importForceData.m``):

    - Lines 1–23: key-value metadata (plate type, corners, dimensions, etc.)
    - Line 24: column header
    - Line 25+: numeric data

    ``Force_Z`` is the vertical ground reaction force (positive upward).
    Belt identities:

    - ``_f_4``: Bertec Treadmill L Belt
    - ``_f_5``: Bertec Treadmill R Belt
    - ``_f_3``: AMTI static plate (QuietStance / Tcap only)
    """
    # index_col=False prevents pandas from auto-promoting SAMPLE to the row
    # index when trailing tabs give data rows one more field than the header.
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=_CFG.qualisys.force_header_row,
        index_col=False,
        na_values=list(_CFG.qualisys.na_values),
        low_memory=False,
    )
    # Drop COP_Z — always zero in Qualisys output
    df = df.drop(columns=["COP_Z"], errors="ignore")
    # Drop rows where all force/COP columns are NaN — AMTI last-row artifact
    # (Qualisys writes 1.#QNAN0 on the final row for AMTI plates)
    force_cop_cols = [c for c in ("Force_X", "Force_Y", "Force_Z", "COP_X", "COP_Y") if c in df.columns]
    if force_cop_cols:
        df = df.dropna(subset=force_cop_cols, how="all")
    return df


def load_demographics(filepath: Path) -> pd.DataFrame:
    """Load subject demographic and speed data from d_subjectData.csv.

    Parameters
    ----------
    filepath : Path
        Path to ``d_subjectData.csv``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``SubID``. Key columns: Age, Sex, Weight (kg),
        HeightNoShoes (cm), LegLength_R/L (cm), WalkingPreferred,
        WalkingPreDetermined, WalkingFroude, RunningPreDetermined,
        RunningFroudeA, RunningFroudeBcalc (all in m/s).
    """
    return pd.read_csv(filepath, index_col="SubID")


def load_subject_weight_newtons(subject_id: str, raw_dir: Path) -> float:
    """Compute subject body weight in Newtons from a quiet stance trial.

    Uses the mean vertical GRF (Force_Z) from QuietStance trial 3,
    matching the MATLAB pipeline (``importFiles.m``).

    Parameters
    ----------
    subject_id : str
        Subject ID as it appears in filenames (e.g. ``FS6``).
    raw_dir : Path
        Directory containing the raw TSV files.

    Returns
    -------
    float
        Body weight in Newtons.

    Raises
    ------
    FileNotFoundError
        If ``{subject_id}_QuietStance3_f_3.tsv`` does not exist.
    """
    path = raw_dir / f"{subject_id}_QuietStance3_f_3.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Weight trial not found: {path}")
    df = load_force_tsv(path)
    return float(df["Force_Z"].mean())


def detect_running_belt(
    force_left: pd.DataFrame,
    force_right: pd.DataFrame,
    body_weight_n: float,
) -> str:
    """Determine which treadmill belt the subject ran on.

    Parameters
    ----------
    force_left : pd.DataFrame
        Force data from the left belt (``_f_4``).
    force_right : pd.DataFrame
        Force data from the right belt (``_f_5``).
    body_weight_n : float
        Subject body weight in Newtons.

    Returns
    -------
    str
        ``'L'`` if subject ran on the left belt, ``'R'`` if right.

    Notes
    -----
    Matches MATLAB logic in ``importFiles.m``: the active belt is the one
    whose peak ``Force_Z`` exceeds the subject's body weight. If both or
    neither exceed the threshold, the belt with the higher peak is returned.
    """
    left_active = force_left["Force_Z"].max() > body_weight_n
    right_active = force_right["Force_Z"].max() > body_weight_n
    if left_active and not right_active:
        return "L"
    if right_active and not left_active:
        return "R"
    # Fallback: return whichever belt has the higher peak vertical force
    return "L" if force_left["Force_Z"].max() >= force_right["Force_Z"].max() else "R"


def load_trial(
    subject_id: str,
    condition: str,
    trial_num: int,
    raw_dir: Path,
    belt: str | None = None,
) -> "Trial":
    """Build a Trial object for a single recorded trial.

    The returned ``Trial`` is lazy — marker and force DataFrames are not read
    from disk until first accessed via ``trial.markers``, ``trial.force_left``,
    etc.

    Parameters
    ----------
    subject_id : str
        Subject ID as it appears in filenames (e.g. ``FS6``).
    condition : str
        Condition filename stem (e.g. ``WalkingPreferred``).
    trial_num : int
        Trial number (1–3).
    raw_dir : Path
        Directory containing the raw TSV files.
    belt : str or None
        ``'L'`` or ``'R'`` for running trials; ``None`` otherwise.

    Returns
    -------
    Trial
    """
    from gait_ml.subject import Trial

    return Trial(
        subject_id=subject_id,
        condition=condition,
        trial_num=trial_num,
        raw_dir=raw_dir,
        belt=belt,
    )


def load_subject(
    subject_id: str,
    raw_dir: Path,
    demographics_path: Path,
    body_weight_n: float | None = None,
) -> "Subject":
    """Load all walking and running trials for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject ID as it appears in filenames (e.g. ``FS6``).
    raw_dir : Path
        Directory containing the raw TSV files.
    demographics_path : Path
        Path to ``d_subjectData.csv``. Used to build the validated
        ``SubjectMeta`` attached to the returned ``Subject``.
    body_weight_n : float, optional
        Subject body weight in Newtons. If not provided, computed from
        ``QuietStance3_f_3.tsv``.

    Returns
    -------
    Subject
    """
    from gait_ml.subject import Subject, SubjectMeta, Trial

    if body_weight_n is None:
        body_weight_n = load_subject_weight_newtons(subject_id, raw_dir)

    demo_df = load_demographics(demographics_path)
    meta = SubjectMeta.from_csv_row(demo_df.loc[subject_id], body_weight_n)

    # Discover which trials exist and determine running belt assignments
    trials: dict[str, list[Trial]] = {}

    all_conditions = list(_CFG.protocol.walk_conditions) + list(_CFG.protocol.run_conditions)
    for condition in all_conditions:
        condition_trials: list[Trial] = []
        for n in range(1, _CFG.protocol.n_trials + 1):
            marker_path = raw_dir / f"{subject_id}_{condition}{n}.tsv"
            if not marker_path.exists():
                continue

            belt: str | None = None
            if condition in _CFG.protocol.run_conditions:
                left_path = raw_dir / f"{subject_id}_{condition}{n}_f_4.tsv"
                right_path = raw_dir / f"{subject_id}_{condition}{n}_f_5.tsv"
                if left_path.exists() and right_path.exists():
                    belt = detect_running_belt(
                        load_force_tsv(left_path),
                        load_force_tsv(right_path),
                        body_weight_n,
                    )

            condition_trials.append(
                Trial(
                    subject_id=subject_id,
                    condition=condition,
                    trial_num=n,
                    raw_dir=raw_dir,
                    belt=belt,
                )
            )

        if condition_trials:
            trials[condition] = condition_trials

    return Subject(meta=meta, trials=trials)
