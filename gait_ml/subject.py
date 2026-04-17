"""
subject.py — Subject and Trial data structures.

SubjectMeta: pydantic model for validated demographics (constructed from CSV).
Trial: lazy-loading dataclass for per-trial kinematic and force data.
Subject: composes SubjectMeta + trials dict; primary access point for a subject.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from gait_ml.config import DEFAULT_LAB_CONFIG as _CFG
from gait_ml.io import load_force_tsv, load_marker_tsv

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# SubjectMeta — validated demographics from d_subjectData.csv
# ---------------------------------------------------------------------------


class SubjectMeta(BaseModel):
    """Demographic and anthropometric data for one subject.

    Constructed from a row of ``d_subjectData.csv`` plus the body weight
    derived from the QuietStance force plate trial.  Pydantic validates all
    fields at construction time so downstream code can assume they are sane.

    Parameters
    ----------
    subject_id : str
        Subject identifier as it appears in filenames (e.g. ``FS6``).
    age : float
        Age in years.
    sex : 'M' or 'F'
    body_weight_n : float
        Body weight in Newtons, from mean Force_Z of QuietStance3_f_3.tsv.
    height_cm : float
        Standing height without shoes, in cm.
    leg_length_r_cm : float
        Right leg length in cm.
    leg_length_l_cm : float
        Left leg length in cm.
    speeds : dict[str, float]
        Actual trial speeds in m/s keyed by condition filename stem
        (e.g. ``'WalkingPreferred'``).
    """

    model_config = ConfigDict(frozen=True)

    subject_id: str
    age: float = Field(gt=0, lt=120)
    sex: Literal["M", "F"]
    body_weight_n: float = Field(gt=0, description="Newtons, from QuietStance trial")
    height_cm: float = Field(gt=0, lt=250)
    leg_length_r_cm: float = Field(gt=0, lt=150)
    leg_length_l_cm: float = Field(gt=0, lt=150)
    speeds: dict[str, float] = Field(
        description="condition stem → actual speed in m/s"
    )

    @classmethod
    def from_csv_row(
        cls,
        row: pd.Series,
        body_weight_n: float,
    ) -> "SubjectMeta":
        """Construct from a row of ``d_subjectData.csv``.

        Parameters
        ----------
        row : pd.Series
            Row indexed by column name; index (name) must be the subject ID.
        body_weight_n : float
            Body weight in Newtons from the QuietStance force plate trial.
        """
        speed_cols = [
            "WalkingPreferred",
            "WalkingPreDetermined",
            "WalkingFroude",
            "RunningPreDetermined",
            "RunningFroudeA",
            "RunningFroudeBcalc",
        ]
        speeds = {c: float(row[c]) for c in speed_cols if c in row.index}

        raw_sex = str(row["Sex"]).strip()
        sex: Literal["M", "F"] = (
            "F" if raw_sex.upper().startswith("F") else "M"
        )

        return cls(
            subject_id=str(row.name),
            age=float(row["Age"]),
            sex=sex,
            body_weight_n=body_weight_n,
            height_cm=float(row["HeightNoShoes"]),
            leg_length_r_cm=float(row["LegLength_R"]),
            leg_length_l_cm=float(row["LegLength_L"]),
            speeds=speeds,
        )

    @property
    def body_weight_kg(self) -> float:
        """Body weight in kilograms."""
        return self.body_weight_n / 9.81

    @property
    def mean_leg_length_cm(self) -> float:
        """Average of left and right leg length in cm."""
        return (self.leg_length_r_cm + self.leg_length_l_cm) / 2.0


# ---------------------------------------------------------------------------
# Trial — lazy-loading dataclass
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """A single recorded trial for one subject and condition.

    Data (markers, forces) is loaded from disk on first access and cached.
    Constructing a ``Trial`` is cheap — it only stores paths and metadata.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    condition : str
        Condition filename stem (e.g. ``'WalkingPreferred'``).
    trial_num : int
        Trial number (1–3).
    raw_dir : Path
        Directory containing the raw TSV files.
    belt : str or None
        ``'L'`` or ``'R'`` for running trials (active treadmill belt).
        ``None`` for walking and static trials.
    """

    subject_id: str
    condition: str
    trial_num: int
    raw_dir: Path
    belt: str | None = None

    # Private cache fields — not shown in repr
    _markers: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    _force_left: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    _force_right: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Lazy-loaded properties
    # ------------------------------------------------------------------

    @property
    def markers(self) -> pd.DataFrame:
        """Marker trajectory DataFrame, loaded on first access."""
        if self._markers is None:
            path = self.raw_dir / f"{self.subject_id}_{self.condition}{self.trial_num}.tsv"
            self._markers = load_marker_tsv(path)
        return self._markers

    @property
    def force_left(self) -> pd.DataFrame | None:
        """Left belt (_f_4) force DataFrame, or None if file absent."""
        if self._force_left is None:
            path = self.raw_dir / f"{self.subject_id}_{self.condition}{self.trial_num}_f_4.tsv"
            if path.exists():
                self._force_left = load_force_tsv(path)
        return self._force_left

    @property
    def force_right(self) -> pd.DataFrame | None:
        """Right belt (_f_5) force DataFrame, or None if file absent."""
        if self._force_right is None:
            path = self.raw_dir / f"{self.subject_id}_{self.condition}{self.trial_num}_f_5.tsv"
            if path.exists():
                self._force_right = load_force_tsv(path)
        return self._force_right

    @property
    def force(self) -> pd.DataFrame | None:
        """Active belt force DataFrame for running trials.

        For running, returns the belt identified by ``self.belt``.
        For walking, returns ``None`` (use ``force_left`` / ``force_right``).
        """
        if self.belt == "L":
            return self.force_left
        if self.belt == "R":
            return self.force_right
        return None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_walking(self) -> bool:
        return self.condition in _CFG.protocol.walk_conditions

    @property
    def is_running(self) -> bool:
        return self.condition in _CFG.protocol.run_conditions

    def unload(self) -> None:
        """Release cached DataFrames to free memory."""
        self._markers = None
        self._force_left = None
        self._force_right = None


# ---------------------------------------------------------------------------
# Subject — composes SubjectMeta + trials
# ---------------------------------------------------------------------------


@dataclass
class Subject:
    """All data for a single subject.

    Parameters
    ----------
    meta : SubjectMeta
        Validated demographic and anthropometric data.
    trials : dict[str, list[Trial]]
        Condition filename stem → list of Trial objects (length ≤ N_TRIALS).

    Notes
    -----
    Demographic fields are accessible directly on ``Subject`` (e.g.
    ``subject.body_weight_n``) as well as via ``subject.meta``.
    """

    meta: SubjectMeta
    trials: dict[str, list[Trial]]

    # ------------------------------------------------------------------
    # Passthrough properties for the most-used demographic fields
    # ------------------------------------------------------------------

    @property
    def subject_id(self) -> str:
        return self.meta.subject_id

    @property
    def body_weight_n(self) -> float:
        return self.meta.body_weight_n

    @property
    def body_weight_kg(self) -> float:
        return self.meta.body_weight_kg

    @property
    def mean_leg_length_cm(self) -> float:
        return self.meta.mean_leg_length_cm

    # ------------------------------------------------------------------
    # Trial access
    # ------------------------------------------------------------------

    def get_trial(self, condition: str, trial_num: int) -> Trial:
        """Return a specific trial.

        Parameters
        ----------
        condition : str
            Condition filename stem (e.g. ``'WalkingPreferred'``).
        trial_num : int
            Trial number (1–3).

        Raises
        ------
        KeyError
            If the condition was not loaded for this subject.
        IndexError
            If ``trial_num`` is out of range for this condition.
        """
        trials = self.trials[condition]
        matches = [t for t in trials if t.trial_num == trial_num]
        if not matches:
            raise IndexError(
                f"{self.subject_id}: no trial {trial_num} for condition '{condition}'"
            )
        return matches[0]

    def walk_trials(self) -> list[Trial]:
        """All walking trials across all walking conditions."""
        out = []
        for cond in _CFG.protocol.walk_conditions:
            out.extend(self.trials.get(cond, []))
        return out

    def run_trials(self) -> list[Trial]:
        """All running trials across all running conditions."""
        out = []
        for cond in _CFG.protocol.run_conditions:
            out.extend(self.trials.get(cond, []))
        return out

    def all_trials(self) -> list[Trial]:
        """All trials in condition order."""
        return self.walk_trials() + self.run_trials()

    def unload(self) -> None:
        """Release all cached DataFrames across all trials."""
        for trial_list in self.trials.values():
            for t in trial_list:
                t.unload()
