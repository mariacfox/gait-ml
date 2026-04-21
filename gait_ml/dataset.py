"""
dataset.py — PyTorch Dataset for time-normalized gait cycles.

Loads preprocessed cycles from the cache written by scripts/build_dataset.py.
Each sample is one gait cycle represented as:
    grf:     float32 tensor (3, 101)   — Fz_L, Fz_R, Fz_total (BW-normalized)
    markers: float32 tensor (90, 101)  — 30 lower-body markers × 3 axes (mm,
             mean-centred per channel per cycle)
    label:   int64 tensor ()           — condition index 0–5  (target='label')
          OR float32 tensor ()         — speed in m/s          (target='speed_ms')

Constants
---------
CONDITION_LABELS : dict[str, int]
    Maps internal condition label string → class index (alphabetical order
    within walk/run groups for reproducibility).
CONDITION_SPEED_COLS : dict[str, str]
    Maps internal condition label → column name in d_subjectData.csv.
LOWER_BODY_MARKERS : list[str]
    30 marker names used to build the 90-channel marker tensor.
GRF_CHANNELS, MARKER_CHANNELS, N_TIME_POINTS, N_CLASSES : int
    Tensor shape constants shared with gait_ml/models.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITION_LABELS: dict[str, int] = {
    "walk_preferred": 0,
    "walk_predetermined": 1,
    "walk_froude": 2,
    "run_predetermined": 3,
    "run_froude_a": 4,
    "run_froude_b": 5,
}
LABEL_CONDITIONS: dict[int, str] = {v: k for k, v in CONDITION_LABELS.items()}
N_CLASSES: int = len(CONDITION_LABELS)

# 3-class grouping: walking / slow run (predetermined + Froude A) / fast run (Froude B)
CONDITION_LABELS_3: dict[str, int] = {
    "walk_preferred":     0,
    "walk_predetermined": 0,
    "walk_froude":        0,
    "run_predetermined":  1,
    "run_froude_a":       1,
    "run_froude_b":       2,
}
LABEL_CONDITIONS_3: dict[int, str] = {
    0: "walking",
    1: "slow_run",
    2: "fast_run",
}
N_CLASSES_3: int = 3

# Maps 6-class label index → 3-class label index
LABEL_MAP_6_TO_3: dict[int, int] = {
    old: CONDITION_LABELS_3[cond] for cond, old in CONDITION_LABELS.items()
}

# Maps condition label → speed column name in d_subjectData.csv
CONDITION_SPEED_COLS: dict[str, str] = {
    "walk_preferred":    "WalkingPreferred",
    "walk_predetermined": "WalkingPreDetermined",
    "walk_froude":       "WalkingFroude",
    "run_predetermined": "RunningPreDetermined",
    "run_froude_a":      "RunningFroudeA",
    "run_froude_b":      "RunningFroudeBcalc",
}

GRF_CHANNELS: int = 3       # Fz_L, Fz_R, Fz_total — BW-normalized
MARKER_CHANNELS: int = 90   # 30 lower-body markers × 3 axes
N_TIME_POINTS: int = 101    # 0–100 % gait cycle

# Lower-body marker set used for the 90-channel marker tensor.
# Order is fixed — changing it invalidates existing caches.
LOWER_BODY_MARKERS: list[str] = [
    # Pelvis (4 markers)
    "LASIS", "RASIS", "LPSIS", "RPSIS",
    # Left thigh (4)
    "LTRO", "LTHIGHU", "LTHIGH", "LTHIGHL",
    # Left shank (4)
    "LKNEE", "LSHANKU", "LSHANK", "LSHANKL",
    # Left foot (5)
    "LANK", "LHEEL", "LMTI", "LMTV", "LTOE",
    # Right thigh (4)
    "RTRO", "RTHIGHU", "RTHIGH", "RTHIGHL",
    # Right shank (4)
    "RKNEE", "RSHANKU", "RSHANK", "RSHANKL",
    # Right foot (5)
    "RANK", "RHEEL", "RMTI", "RMTV", "RTOE",
]  # 30 markers × 3 axes = 90 channels — matches MARKER_CHANNELS


# ---------------------------------------------------------------------------
# Speed helpers
# ---------------------------------------------------------------------------


def add_speed_column(manifest: pd.DataFrame, subject_data_csv: Path) -> pd.DataFrame:
    """Join actual trial speeds (m/s) from d_subjectData.csv onto the manifest.

    Adds a ``speed_ms`` column to the manifest. Rows whose condition is not in
    ``CONDITION_SPEED_COLS`` (e.g. quiet_stance) receive NaN.

    Parameters
    ----------
    manifest : pd.DataFrame
        Cycle manifest from ``load_manifest()``.
    subject_data_csv : Path
        Path to ``d_subjectData.csv``. Must contain a ``SubID`` column and
        one speed column per condition (see ``CONDITION_SPEED_COLS``).

    Returns
    -------
    pd.DataFrame
        Copy of manifest with a new ``speed_ms`` float column.
    """
    subj_df = pd.read_csv(subject_data_csv)
    # Normalise subject ID column (may be 'SubID' or 'SubjectID')
    id_col = next(
        (c for c in subj_df.columns if c.lower() in ("subid", "subjectid", "subject_id")),
        subj_df.columns[0],
    )
    subj_df = subj_df.rename(columns={id_col: "subject_id"})
    subj_df["subject_id"] = subj_df["subject_id"].astype(str).str.strip()

    manifest = manifest.copy()
    manifest["speed_ms"] = float("nan")

    for condition, speed_col in CONDITION_SPEED_COLS.items():
        if speed_col not in subj_df.columns:
            continue
        speed_lookup = subj_df.set_index("subject_id")[speed_col]
        mask = manifest["condition"] == condition
        manifest.loc[mask, "speed_ms"] = manifest.loc[mask, "subject_id"].map(speed_lookup)

    return manifest


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def load_manifest(processed_dir: Path) -> pd.DataFrame:
    """Load the cycle manifest written by scripts/build_dataset.py.

    Parameters
    ----------
    processed_dir : Path
        Root of the processed data directory (contains ``manifest.csv``).

    Returns
    -------
    pd.DataFrame
        One row per gait cycle. Required columns:
        subject_id, condition, trial_num, cycle_idx, label,
        grf_path, marker_path, belt.
    """
    path = processed_dir / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"manifest.csv not found at {path}. "
            "Run scripts/build_dataset.py first."
        )
    df = pd.read_csv(path)
    # Ensure label column is present; derive from condition if missing
    if "label" not in df.columns:
        df["label"] = df["condition"].map(CONDITION_LABELS)
    return df


def kfold_splits(
    manifest: pd.DataFrame,
    n_folds: int,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """Split manifest into k folds by subject.

    Subjects are randomly shuffled then divided into ``n_folds`` groups of
    roughly equal size. All cycles for a subject stay in the same fold so
    there is no cycle-level leakage between train and test.

    Parameters
    ----------
    manifest : pd.DataFrame
        Full cycle manifest from ``load_manifest()``.
    n_folds : int
        Number of folds.
    seed : int
        Random seed for subject shuffling (default 42 for reproducibility).

    Returns
    -------
    List of ``(train_df, test_df, test_subjects)`` tuples, one per fold.
    """
    subjects = np.array(sorted(manifest["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects)
    subject_groups = np.array_split(shuffled, n_folds)

    folds = []
    for group in subject_groups:
        test_mask = manifest["subject_id"].isin(group)
        folds.append((
            manifest[~test_mask].reset_index(drop=True),
            manifest[test_mask].reset_index(drop=True),
            list(group),
        ))
    return folds



# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GaitCycleDataset(Dataset):
    """PyTorch Dataset for time-normalized gait cycles.

    Loads GRF and marker tensors from the per-trial .pt files written by
    ``scripts/build_dataset.py``.  Trial files are loaded lazily and cached
    in memory on first access so repeated epoch passes are fast.

    Parameters
    ----------
    manifest : pd.DataFrame
        Subset of the full manifest (e.g. from ``loso_splits``).
    processed_dir : Path
        Root processed data directory (parent of ``cycles/``).
    modality : str
        Which tensors to load and return:
        ``'grf'`` — GRF only, returns (grf, dummy_zeros, label).
        ``'markers'`` — markers only, returns (dummy_zeros, markers, label).
        ``'both'`` — both modalities (default).
    center_markers : bool
        If True (default), subtract per-channel mean from each marker cycle
        so the network learns motion rather than absolute position.

    Parameters
    ----------
    target : str
        What to return as the third element of each sample:
        ``'label'``    — int64 condition index 0–5 (classification).
        ``'speed_ms'`` — float32 speed in m/s (regression). Requires the
                         manifest to have a ``speed_ms`` column; use
                         ``add_speed_column()`` to add it before constructing
                         the dataset.
    return_group_id : bool
        If True (and ``target='speed_ms'``), return a 4th element: an int64
        group ID that is unique per ``(subject_id, condition, trial_num)``
        tuple. Used by the aggregate-prediction training loop to collapse
        within-trial cycles before computing the loss. Group IDs are computed
        locally on the provided manifest slice (train or test split) so they
        are not globally comparable across splits.

    Returns (from __getitem__)
    --------------------------
    grf : Tensor, shape (3, 101), float32
    markers : Tensor, shape (90, 101), float32
    target : Tensor, shape ()
        int64 class index when ``target='label'``;
        float32 speed in m/s when ``target='speed_ms'``.
    group_id : Tensor, shape (), int64  — only when ``return_group_id=True``
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        processed_dir: Path,
        modality: str = "both",
        center_markers: bool = True,
        target: str = "label",
        return_group_id: bool = False,
        label_map: dict[int, int] | None = None,
    ) -> None:
        if modality not in ("grf", "markers", "both"):
            raise ValueError(f"modality must be 'grf', 'markers', or 'both'; got {modality!r}")
        if target not in ("label", "speed_ms"):
            raise ValueError(f"target must be 'label' or 'speed_ms'; got {target!r}")
        if target == "speed_ms" and "speed_ms" not in manifest.columns:
            raise ValueError(
                "target='speed_ms' requires a 'speed_ms' column in the manifest. "
                "Call add_speed_column(manifest, subject_data_csv) first."
            )
        if return_group_id and target != "speed_ms":
            raise ValueError("return_group_id=True is only supported with target='speed_ms'.")

        manifest = manifest.reset_index(drop=True)
        if return_group_id:
            # Assign a unique integer per (subject_id, condition, trial_num) within this split.
            manifest = manifest.copy()
            manifest["_group_id"] = manifest.groupby(
                ["subject_id", "condition", "trial_num"], sort=False
            ).ngroup()

        self.manifest = manifest
        self.processed_dir = Path(processed_dir)
        self.modality = modality
        self.center_markers = center_markers
        self.target = target
        self.return_group_id = return_group_id
        self.label_map = label_map

        # Cache: (grf_path, marker_path) → (grf_tensor, marker_tensor)
        # Keyed by the trial-level file path strings to avoid redundant disk reads.
        self._grf_cache: dict[str, Tensor] = {}
        self._marker_cache: dict[str, Tensor] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        row = self.manifest.iloc[idx]

        if self.target == "speed_ms":
            target = torch.tensor(float(row["speed_ms"]), dtype=torch.float32)
        else:
            raw_label = int(row["label"])
            if self.label_map is not None:
                raw_label = self.label_map[raw_label]
            target = torch.tensor(raw_label, dtype=torch.int64)

        grf = self._load_grf(str(row["grf_path"]), int(row["cycle_idx"]))
        markers = self._load_markers(str(row["marker_path"]), int(row["cycle_idx"]))

        if self.return_group_id:
            group_id = torch.tensor(int(row["_group_id"]), dtype=torch.int64)
            return grf, markers, target, group_id
        return grf, markers, target

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_grf(self, path: str, cycle_idx: int) -> Tensor:
        if path not in self._grf_cache:
            full = self.processed_dir / path
            self._grf_cache[path] = torch.load(full, weights_only=True)
        trial_tensor = self._grf_cache[path]  # (n_cycles, 3, 101)

        if self.modality == "markers":
            return torch.zeros(GRF_CHANNELS, N_TIME_POINTS, dtype=torch.float32)
        return trial_tensor[cycle_idx]  # (3, 101)

    def _load_markers(self, path: str, cycle_idx: int) -> Tensor:
        if path not in self._marker_cache:
            full = self.processed_dir / path
            self._marker_cache[path] = torch.load(full, weights_only=True)
        trial_tensor = self._marker_cache[path]  # (n_cycles, 90, 101)

        if self.modality == "grf":
            return torch.zeros(MARKER_CHANNELS, N_TIME_POINTS, dtype=torch.float32)

        cycle = trial_tensor[cycle_idx]  # (90, 101)
        if self.center_markers:
            # Subtract per-channel mean so network sees motion, not absolute position
            cycle = cycle - cycle.mean(dim=1, keepdim=True)
        return cycle

    def clear_cache(self) -> None:
        """Release cached trial tensors to free memory."""
        self._grf_cache.clear()
        self._marker_cache.clear()
