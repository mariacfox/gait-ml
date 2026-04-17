"""
config.py — Lab equipment and acquisition configuration.

All values confirmed from MATLAB reference scripts and direct inspection of
Qualisys TSV output. Construct LabConfig() with no arguments to get the
default configuration for this lab setup.

Nested sub-models:
  AcquisitionConfig   — sample rates, camera/marker counts
  FilterConfig        — Butterworth cutoff frequencies and order
  ForcePlateConfig    — treadmill belt and static plate assignments
  GaitEventConfig     — GRF threshold and stance validity criteria
  NormalizationConfig — gait cycle time normalization settings

Usage
-----
    from gait_ml.config import DEFAULT_LAB_CONFIG

    fs = DEFAULT_LAB_CONFIG.acquisition.grf_sample_rate_hz   # 1120.0
    fc = DEFAULT_LAB_CONFIG.filter.kinematic_lowpass_hz      # 8.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class AcquisitionConfig(BaseModel):
    """Motion capture and force plate acquisition settings.

    Confirmed from Qualisys TSV metadata headers and ``importFiles.m``.
    """

    model_config = ConfigDict(frozen=True)

    kinematic_sample_rate_hz: float = Field(
        default=160.0,
        gt=0,
        description="Marker trajectory sampling rate in Hz.",
    )
    grf_sample_rate_hz: float = Field(
        default=1120.0,
        gt=0,
        description="Force plate sampling rate in Hz (7× kinematics).",
    )
    n_cameras: int = Field(
        default=6,
        gt=0,
        description="Number of motion capture cameras.",
    )
    n_markers_dynamic: int = Field(
        default=48,
        gt=0,
        description="Marker count for dynamic (walking/running) trials.",
    )
    n_markers_static: int = Field(
        default=54,
        gt=0,
        description="Marker count for static trials (Tcap, QuietStance). "
                    "Adds 6 medial calibration markers used for joint centre estimation.",
    )

    @property
    def grf_kinematic_ratio(self) -> float:
        """Integer ratio of GRF to kinematic sample rate (nominally 7)."""
        return self.grf_sample_rate_hz / self.kinematic_sample_rate_hz


class FilterConfig(BaseModel):
    """Zero-phase Butterworth low-pass filter parameters.

    Confirmed from ``filterMarkerData.m`` and ``filtForceCOP.m``.
    Moments (Moment_X/Y/Z) are NOT filtered — passed through as-is.
    """

    model_config = ConfigDict(frozen=True)

    order: int = Field(
        default=4,
        gt=0,
        description="Butterworth filter order. Applied as zero-phase via filtfilt "
                    "(effective order is doubled).",
    )
    kinematic_lowpass_hz: float = Field(
        default=8.0,
        gt=0,
        description="Low-pass cutoff for marker trajectories (Hz).",
    )
    grf_lowpass_hz: float = Field(
        default=8.0,
        gt=0,
        description="Low-pass cutoff for Force_X/Y/Z (Hz).",
    )
    cop_lowpass_hz: float = Field(
        default=15.0,
        gt=0,
        description="Low-pass cutoff for COP_X/Y (Hz). Higher than GRF "
                    "to preserve centre-of-pressure dynamics.",
    )


class ForcePlateConfig(BaseModel):
    """Force plate layout and file suffix assignments.

    Confirmed from ``importFiles.m`` and ``importForceData.m``.

    The lab uses a Bertec split-belt instrumented treadmill (two independent
    belts) plus an AMTI force plate for static trials.
    """

    model_config = ConfigDict(frozen=True)

    left_belt_suffix: str = Field(
        default="_f_4",
        description="TSV filename suffix for the Bertec left belt.",
    )
    right_belt_suffix: str = Field(
        default="_f_5",
        description="TSV filename suffix for the Bertec right belt.",
    )
    static_plate_suffix: str = Field(
        default="_f_3",
        description="TSV filename suffix for the AMTI static force plate "
                    "(QuietStance and Tcap trials only).",
    )
    left_belt_name: str = Field(
        default="Bertec Treadmill L Belt",
        description="Plate name as it appears in the TSV header.",
    )
    right_belt_name: str = Field(
        default="Bertec Treadmill R Belt",
        description="Plate name as it appears in the TSV header.",
    )
    static_plate_name: str = Field(
        default="AMTI",
        description="Static plate manufacturer / identifier.",
    )
    weight_trial_suffix: str = Field(
        default="_f_3",
        description="Suffix of the file used to compute body weight "
                    "(mean Force_Z from QuietStance3).",
    )
    weight_trial_condition: str = Field(
        default="QuietStance",
        description="Condition used for body weight computation.",
    )
    weight_trial_num: int = Field(
        default=3,
        description="Trial number used for body weight computation.",
    )


class GaitEventConfig(BaseModel):
    """Gait event detection thresholds.

    Confirmed from ``gaitEventDetection.m``.
    """

    model_config = ConfigDict(frozen=True)

    threshold_n: float = Field(
        default=15.0,
        gt=0,
        description="Vertical GRF threshold for heel strike / toe-off detection (N). "
                    "Confirmed 15 N from gaitEventDetection.m.",
    )
    min_stance_peak_n: float = Field(
        default=150.0,
        gt=0,
        description="Minimum peak Force_Z for a valid stance phase (N). "
                    "Stances below this are rejected as foot crossover artefacts.",
    )


class NormalizationConfig(BaseModel):
    """Gait cycle time normalization settings.

    Confirmed from ``gaitCycleNormalization.m``.
    """

    model_config = ConfigDict(frozen=True)

    n_points: int = Field(
        default=101,
        gt=1,
        description="Number of output time points (0–100% inclusive). "
                    "Interpolation method: pchip.",
    )


class CoordinateSystemConfig(BaseModel):
    """Lab coordinate system convention.

    Confirmed by inspection of marker data:
    - CLAV Y ≈ 1210 mm  (plausible standing height → Y is vertical)
    - LASIS/RASIS differ primarily in Z (mediolateral separation ≈ 227 mm)
    """

    model_config = ConfigDict(frozen=True)

    vertical: str = Field(
        default="Y",
        description="Axis pointing vertically upward.",
    )
    anteroposterior: str = Field(
        default="X",
        description="Axis pointing forward along the treadmill.",
    )
    mediolateral: str = Field(
        default="Z",
        description="Axis pointing laterally (right positive convention "
                    "not yet confirmed — verify before computing joint angles).",
    )
    units: str = Field(
        default="mm",
        description="Spatial units for marker trajectories.",
    )
    force_units: str = Field(
        default="N",
        description="Units for force plate forces.",
    )
    moment_units: str = Field(
        default="N·mm",
        description="Units for force plate moments.",
    )
    cop_units: str = Field(
        default="mm",
        description="Units for centre of pressure.",
    )


class QualisysConfig(BaseModel):
    """Qualisys motion capture software TSV output format details.

    These values are specific to the Qualisys export format and are used
    by ``io.py`` to parse the raw TSV files correctly.

    Confirmed from ``importMarkerData.m``, ``importForceData.m``,
    ``findMarkerNames.m``.
    """

    model_config = ConfigDict(frozen=True)

    marker_header_row: int = Field(
        default=10,
        description="0-indexed row number passed to pandas header= for "
                    "kinematics TSVs (line 11 in 1-indexed terms).",
    )
    force_header_row: int = Field(
        default=23,
        description="0-indexed row number passed to pandas header= for "
                    "force plate TSVs (line 24 in 1-indexed terms).",
    )
    na_values: tuple[str, ...] = Field(
        default=("1.#QNAN0", "1.#QNAN", "-1.#QNAN0", "-1.#QNAN"),
        description="Windows NaN representations written by Qualisys that "
                    "pandas does not recognise by default.",
    )
    force_columns: tuple[str, ...] = Field(
        default=("Force_X", "Force_Y", "Force_Z"),
        description="Force component column names in force plate TSVs.",
    )
    cop_columns: tuple[str, ...] = Field(
        default=("COP_X", "COP_Y"),
        description="Centre-of-pressure column names. COP_Z is always zero "
                    "and is dropped on load.",
    )
    moment_columns: tuple[str, ...] = Field(
        default=("Moment_X", "Moment_Y", "Moment_Z"),
        description="Moment column names. Moments are NOT filtered — "
                    "passed through as-is (confirmed from filtForceCOP.m).",
    )


class ProtocolConfig(BaseModel):
    """Trial protocol — conditions, trial counts, and label mappings.

    Captures the study design: which conditions were collected, how many
    trials per condition, and how filename stems map to internal labels.
    """

    model_config = ConfigDict(frozen=True)

    n_trials: int = Field(
        default=3,
        gt=0,
        description="Number of trials recorded per condition.",
    )
    walk_conditions: tuple[str, ...] = Field(
        default=("WalkingPreferred", "WalkingPreDetermined", "WalkingFroude"),
        description="Condition filename stems for walking trials.",
    )
    run_conditions: tuple[str, ...] = Field(
        default=("RunningPreDetermined", "RunningFroudeA", "RunningFroudeB"),
        description="Condition filename stems for running trials.",
    )
    condition_labels: dict[str, str] = Field(
        default={
            "WalkingPreferred": "walk_preferred",
            "WalkingPreDetermined": "walk_predetermined",
            "WalkingFroude": "walk_froude",
            "RunningPreDetermined": "run_predetermined",
            "RunningFroudeA": "run_froude_a",
            "RunningFroudeB": "run_froude_b",
            "QuietStance": "quiet_stance",
            "Tcap": "tcap",
            "empty": "empty",
        },
        description="Condition filename stem → internal short label used "
                    "in DataFrames and feature matrices.",
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class LabConfig(BaseModel):
    """Full lab equipment and acquisition configuration.

    All sub-models default to the confirmed values for this lab.
    Construct with ``LabConfig()`` to get the default configuration.

    Parameters
    ----------
    acquisition : AcquisitionConfig
    filter : FilterConfig
    force_plates : ForcePlateConfig
    gait_events : GaitEventConfig
    normalization : NormalizationConfig
    coordinate_system : CoordinateSystemConfig
    qualisys : QualisysConfig
    protocol : ProtocolConfig

    Examples
    --------
    >>> from gait_ml.config import DEFAULT_LAB_CONFIG as cfg
    >>> cfg.acquisition.kinematic_sample_rate_hz
    160.0
    >>> cfg.filter.kinematic_lowpass_hz
    8.0
    >>> cfg.protocol.walk_conditions
    ('WalkingPreferred', 'WalkingPreDetermined', 'WalkingFroude')
    """

    model_config = ConfigDict(frozen=True)

    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    force_plates: ForcePlateConfig = Field(default_factory=ForcePlateConfig)
    gait_events: GaitEventConfig = Field(default_factory=GaitEventConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    coordinate_system: CoordinateSystemConfig = Field(
        default_factory=CoordinateSystemConfig
    )
    qualisys: QualisysConfig = Field(default_factory=QualisysConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)


# ---------------------------------------------------------------------------
# Default singleton — import this throughout the codebase
# ---------------------------------------------------------------------------

DEFAULT_LAB_CONFIG = LabConfig()
