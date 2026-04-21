"""
build_dataset.py — Preprocess all trials and cache time-normalized gait cycles.

Run from the repo root with the full data directory:
    uv run python scripts/build_dataset.py \\
        --data-root '/Users/mariafox/Library/Mobile Documents/com~apple~CloudDocs/academic_research/UIUC_dissertation/Qualisys/Data' \\
        --out-dir data/processed

For a single subject (smoke test):
    uv run python scripts/build_dataset.py --data-root /path/to/Data --subjects FS6

For a flat directory of TSV files (legacy / local dev):
    uv run python scripts/build_dataset.py --data-root data/raw --flat

Data root layout expected by default:
    Data/
      subFS6/_tsv/    ← FS6's TSV files (FS6_WalkingPreferred1.tsv, etc.)
      subMS1/_tsv/
      subMT1/_tsv/
      ...other dirs ignored...

Only directories matching sub(FS|FT|MS|MT)\\d+ are processed.

Output structure
----------------
data/processed/
    manifest.csv                                     # one row per gait cycle
    cycles/
        {subject_id}/
            {condition}/
                {trial_num}_grf.pt      (n_cycles, 3, 101)  float32
                {trial_num}_markers.pt  (n_cycles, 90, 101) float32

GRF tensor channels (index 0–2):
    0  Fz_L   — left belt vertical GRF, BW-normalized
    1  Fz_R   — right belt vertical GRF, BW-normalized
    2  Fz_tot — Fz_L + Fz_R

Marker tensor channels (index 0–89):
    LOWER_BODY_MARKERS × [X, Y, Z] — raw filtered positions in mm.
    Mean-centering (per channel, per cycle) is applied by GaitCycleDataset,
    NOT here, so the cache stores absolute positions.

Cycle validity criteria (matching PLAN.md):
    - GRF stance must pass detect_gait_events_grf artifact filter (≥150 N peak)
    - Cycle duration: 0.5 s ≤ duration ≤ 2.5 s (at kinematic rate)
    - Marker data: at least half of lower-body channels must be non-NaN
      for the full cycle window
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure the package is importable when run as a script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from gait_ml.config import DEFAULT_LAB_CONFIG as _CFG
from gait_ml.dataset import CONDITION_LABELS, LOWER_BODY_MARKERS
from gait_ml.io import load_force_tsv, load_marker_tsv, load_subject_weight_newtons
from gait_ml.preprocessing import (
    butterworth_lowpass,
    fill_marker_gaps,
    normalize_gait_cycle,
)
from gait_ml.segmentation import detect_gait_events_grf

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS_GRF = _CFG.acquisition.grf_sample_rate_hz          # 1120 Hz
FS_KIN = _CFG.acquisition.kinematic_sample_rate_hz    # 160 Hz
GRF_KIN_RATIO = int(FS_GRF / FS_KIN)                  # 7
N_POINTS = _CFG.normalization.n_points                 # 101

# Cycle duration bounds in kinematic frames.
# Walking cycles: ~0.8–1.3 s; running cycles at 3.3 m/s: ~0.33–0.40 s.
# Min is set conservatively to cover fast running; max covers slow walking.
MIN_CYCLE_FRAMES = int(0.2 * FS_KIN)   # 32 frames  (~0.2 s)
MAX_CYCLE_FRAMES = int(2.5 * FS_KIN)   # 400 frames (~2.5 s)

# Minimum fraction of lower-body marker channels that must be valid
MIN_MARKER_VALID_FRACTION = 0.5

# Internal condition label → condition filename stems
_LABEL_TO_STEMS: dict[str, str] = {v: k for k, v in _CFG.protocol.condition_labels.items()}

# ---------------------------------------------------------------------------
# Marker column helpers
# ---------------------------------------------------------------------------

# All column names for lower-body markers in the order stored in the tensor.
# Each marker contributes 3 columns: {MARKER}X, {MARKER}Y, {MARKER}Z.
LOWER_BODY_COLS: list[str] = [
    f"{m}{ax}" for m in LOWER_BODY_MARKERS for ax in ("X", "Y", "Z")
]  # 90 columns


def _extract_marker_array(df: pd.DataFrame) -> np.ndarray:
    """Extract lower-body marker columns as a float32 array.

    Missing columns (marker not present in this trial's marker set) are
    filled with NaN so the shape is always (n_frames, 90).

    Returns
    -------
    np.ndarray
        Shape (n_frames, 90).
    """
    n = len(df)
    out = np.full((n, len(LOWER_BODY_COLS)), np.nan, dtype=np.float32)
    for i, col in enumerate(LOWER_BODY_COLS):
        if col in df.columns:
            out[:, i] = df[col].to_numpy(dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

# Subdirectory names that identify subject folders in the data root.
# Pattern: sub + two-letter group code + number (e.g. subFS6, subMT12).
_SUBJECT_DIR_RE = re.compile(r"^sub((?:FS|FT|MS|MT)\d+)$", re.IGNORECASE)


def _discover_subjects(data_root: Path, flat: bool = False) -> list[tuple[str, Path]]:
    """Return sorted list of (subject_id, tsv_dir) pairs.

    Parameters
    ----------
    data_root : Path
        Root directory. In nested mode (flat=False), expected to contain
        subdirectories named ``sub{ID}/_tsv/``. In flat mode (flat=True),
        all TSV files are directly in ``data_root``.
    flat : bool
        If True, treat data_root as a single flat directory of TSV files
        (legacy / local dev layout).

    Returns
    -------
    list of (subject_id, tsv_dir)
        ``tsv_dir`` is the directory that contains the TSV files for that subject.
    """
    if flat:
        # Flat mode: discover subjects by filename prefix in data_root
        file_re = re.compile(r"^([A-Z]{2}\d+)_")
        ids: set[str] = set()
        for f in data_root.iterdir():
            m = file_re.match(f.name)
            if m:
                ids.add(m.group(1))
        return sorted((sid, data_root) for sid in ids)

    # Nested mode: sub{ID}/_tsv/ layout
    results: list[tuple[str, Path]] = []
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        m = _SUBJECT_DIR_RE.match(entry.name)
        if not m:
            continue
        subject_id = m.group(1).upper()
        tsv_dir = entry / "_tsv"
        if not tsv_dir.exists():
            log.debug("  No _tsv/ subdirectory in %s — skipping", entry.name)
            continue
        results.append((subject_id, tsv_dir))
    return results


# ---------------------------------------------------------------------------
# Per-trial processing
# ---------------------------------------------------------------------------


def _process_trial(
    subject_id: str,
    condition_stem: str,
    trial_num: int,
    condition_label: str,
    raw_dir: Path,
    body_weight_n: float,
    is_running: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]] | None:
    """Process a single trial and return cycle arrays.

    Returns None if required files are missing or no valid cycles found.

    Returns
    -------
    grf_cycles : list of np.ndarray, each shape (3, 101)
    marker_cycles : list of np.ndarray, each shape (90, 101)
    meta_rows : list of dict (manifest metadata per cycle)
    """
    base = raw_dir / f"{subject_id}_{condition_stem}{trial_num}"
    kin_path = Path(str(base) + ".tsv")
    left_path = Path(str(base) + "_f_4.tsv")
    right_path = Path(str(base) + "_f_5.tsv")

    if not kin_path.exists():
        log.debug("  Missing kinematic file: %s", kin_path.name)
        return None
    if not left_path.exists() or not right_path.exists():
        log.debug("  Missing GRF file(s) for %s", base.name)
        return None

    # ------------------------------------------------------------------
    # Load and filter GRF
    # ------------------------------------------------------------------
    try:
        df_left = load_force_tsv(left_path)
        df_right = load_force_tsv(right_path)
    except Exception as exc:
        log.warning("  Failed to load GRF for %s: %s", base.name, exc)
        return None

    fz_l = butterworth_lowpass(
        df_left["Force_Z"].to_numpy(dtype=np.float64),
        cutoff_hz=_CFG.filter.grf_lowpass_hz,
        sample_rate_hz=FS_GRF,
    )
    fz_r = butterworth_lowpass(
        df_right["Force_Z"].to_numpy(dtype=np.float64),
        cutoff_hz=_CFG.filter.grf_lowpass_hz,
        sample_rate_hz=FS_GRF,
    )

    # Align lengths (belt signals can differ by 1–2 frames due to rounding)
    n_grf = min(len(fz_l), len(fz_r))
    fz_l = fz_l[:n_grf]
    fz_r = fz_r[:n_grf]
    fz_tot = fz_l + fz_r

    # ------------------------------------------------------------------
    # Detect gait events
    # ------------------------------------------------------------------
    # For running, only the active belt drives event detection.
    # For walking, both belts are processed independently.
    if is_running:
        active_fz = fz_l if fz_l.max() >= fz_r.max() else fz_r
        events_l = detect_gait_events_grf(active_fz, sample_rate_hz=FS_GRF)
        belt_pairs = [("active", events_l)]
    else:
        events_l = detect_gait_events_grf(fz_l, sample_rate_hz=FS_GRF)
        events_r = detect_gait_events_grf(fz_r, sample_rate_hz=FS_GRF)
        belt_pairs = [("L", events_l), ("R", events_r)]

    # ------------------------------------------------------------------
    # Load and filter markers
    # ------------------------------------------------------------------
    try:
        df_kin = load_marker_tsv(kin_path)
    except Exception as exc:
        log.warning("  Failed to load markers for %s: %s", base.name, exc)
        return None

    markers_raw = _extract_marker_array(df_kin)                # (n_kin, 90)
    markers_raw = fill_marker_gaps(markers_raw.astype(np.float64)).astype(np.float32)
    markers_filt = butterworth_lowpass(
        markers_raw.astype(np.float64),
        cutoff_hz=_CFG.filter.kinematic_lowpass_hz,
        sample_rate_hz=FS_KIN,
    ).astype(np.float32)
    n_kin = len(markers_filt)

    # ------------------------------------------------------------------
    # Extract and validate cycles
    # ------------------------------------------------------------------
    grf_cycles: list[np.ndarray] = []
    marker_cycles: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for belt_id, events in belt_pairs:
        hs_grf = events["heel_strike"]

        # For running, the active belt captures alternating left and right foot
        # contacts (both feet run on one belt). Consecutive HS pairs are therefore
        # step-level (~0.34 s), not stride-level. Taking every-other pair gives
        # true gait cycles (same foot → same foot, ~0.68 s).
        # For walking, each belt sees only one foot, so stride=1 is correct.
        hs_stride = 2 if is_running else 1
        if len(hs_grf) < hs_stride + 1:
            continue

        for i in range(0, len(hs_grf) - hs_stride, hs_stride):
            grf_start = hs_grf[i]
            grf_end = hs_grf[i + hs_stride]

            # Convert GRF indices to kinematic frame indices
            kin_start = int(grf_start // GRF_KIN_RATIO)
            kin_end = int(grf_end // GRF_KIN_RATIO)

            # Cycle duration check (kinematic frames)
            cycle_len_kin = kin_end - kin_start
            if cycle_len_kin < MIN_CYCLE_FRAMES or cycle_len_kin > MAX_CYCLE_FRAMES:
                continue

            # Bounds check
            if grf_end > n_grf or kin_end > n_kin:
                continue

            # Marker validity check
            marker_cycle_raw = markers_filt[kin_start:kin_end].copy()  # (n_frames, 90)
            valid_frac = np.isfinite(marker_cycle_raw).mean()
            if valid_frac < MIN_MARKER_VALID_FRACTION:
                continue

            # Fill any residual NaN (pchip leaves NaN at trial edges it can't
            # extrapolate to; a cycle clipping those edges inherits them).
            # Nearest-neighbor fill per channel; fully-missing channels → 0.
            for ch in range(marker_cycle_raw.shape[1]):
                col = marker_cycle_raw[:, ch]
                bad = ~np.isfinite(col)
                if not bad.any():
                    continue
                if bad.all():
                    marker_cycle_raw[:, ch] = 0.0
                else:
                    good_idx = np.where(~bad)[0]
                    marker_cycle_raw[bad, ch] = np.interp(
                        np.where(bad)[0], good_idx, col[~bad]
                    )

            # Time-normalize GRF: (3, 101)
            grf_window = np.stack([
                fz_l[grf_start:grf_end] / body_weight_n,
                fz_r[grf_start:grf_end] / body_weight_n,
                fz_tot[grf_start:grf_end] / body_weight_n,
            ], axis=1)  # (n_grf_frames, 3)
            grf_norm = normalize_gait_cycle(grf_window).T.astype(np.float32)  # (3, 101)

            # Time-normalize markers: (90, 101)
            marker_norm = normalize_gait_cycle(marker_cycle_raw).T.astype(np.float32)  # (90, 101)

            grf_cycles.append(grf_norm)
            marker_cycles.append(marker_norm)
            meta_rows.append({
                "subject_id": subject_id,
                "condition": condition_label,
                "trial_num": trial_num,
                "cycle_idx": len(meta_rows),
                "belt": belt_id,
                "label": CONDITION_LABELS[condition_label],
                "grf_path": "",      # filled in by caller after saving
                "marker_path": "",   # filled in by caller after saving
            })

    if not grf_cycles:
        return None

    return grf_cycles, marker_cycles, meta_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dataset(
    data_root: Path,
    out_dir: Path,
    subjects: list[str] | None = None,
    flat: bool = False,
) -> None:
    """Process all subjects and write the cycle cache.

    Parameters
    ----------
    data_root : Path
        Root data directory. In nested mode (flat=False, default), must contain
        subdirectories named ``sub{ID}/_tsv/`` (e.g. ``subFS6/_tsv/``).
        In flat mode (flat=True), must contain TSV files directly.
    out_dir : Path
        Output root (``manifest.csv`` and ``cycles/`` written here).
    subjects : list[str] or None
        If provided, process only these subject IDs (e.g. ['FS6', 'MS1']).
        Default: discover all valid subjects in data_root.
    flat : bool
        If True, treat data_root as a flat TSV directory (legacy / dev layout).
    """
    cycles_dir = out_dir / "cycles"
    cycles_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_subjects(data_root, flat=flat)
    if subjects is not None:
        subjects_upper = {s.upper() for s in subjects}
        discovered = [(sid, d) for sid, d in discovered if sid.upper() in subjects_upper]

    if not discovered:
        log.error("No subjects found in %s (flat=%s)", data_root, flat)
        return

    log.info("Processing %d subjects", len(discovered))

    all_conditions = list(_CFG.protocol.condition_labels.items())
    # Only process gait conditions (skip Tcap, QuietStance, empty)
    gait_stems = set(_CFG.protocol.walk_conditions) | set(_CFG.protocol.run_conditions)

    all_meta: list[dict] = []

    for subject_id, raw_dir in discovered:
        log.info("Subject %s  (tsv_dir: %s)", subject_id, raw_dir)

        try:
            bw_n = load_subject_weight_newtons(subject_id, raw_dir)
        except FileNotFoundError:
            log.warning("  QuietStance weight file missing — skipping %s", subject_id)
            continue

        log.info("  Body weight: %.1f N", bw_n)

        for condition_stem, condition_label in all_conditions:
            if condition_stem not in gait_stems:
                continue
            if condition_label not in CONDITION_LABELS:
                continue

            is_running = condition_stem in _CFG.protocol.run_conditions

            for trial_num in range(1, _CFG.protocol.n_trials + 1):
                result = _process_trial(
                    subject_id=subject_id,
                    condition_stem=condition_stem,
                    trial_num=trial_num,
                    condition_label=condition_label,
                    raw_dir=raw_dir,
                    body_weight_n=bw_n,
                    is_running=is_running,
                )
                if result is None:
                    continue

                grf_cycles, marker_cycles, meta_rows = result

                # Save trial tensors
                trial_dir = cycles_dir / subject_id / condition_stem
                trial_dir.mkdir(parents=True, exist_ok=True)

                grf_tensor = torch.from_numpy(np.stack(grf_cycles, axis=0))      # (n, 3, 101)
                marker_tensor = torch.from_numpy(np.stack(marker_cycles, axis=0)) # (n, 90, 101)

                grf_rel = Path("cycles") / subject_id / condition_stem / f"{trial_num}_grf.pt"
                marker_rel = Path("cycles") / subject_id / condition_stem / f"{trial_num}_markers.pt"

                torch.save(grf_tensor, out_dir / grf_rel)
                torch.save(marker_tensor, out_dir / marker_rel)

                for row in meta_rows:
                    row["grf_path"] = str(grf_rel)
                    row["marker_path"] = str(marker_rel)
                all_meta.extend(meta_rows)

                log.info(
                    "  %s trial %d → %d cycles",
                    condition_label, trial_num, len(grf_cycles),
                )

        subject_cycle_count = sum(
            1 for r in all_meta if r["subject_id"] == subject_id
        )
        log.info("  Total cycles for %s: %d", subject_id, subject_cycle_count)

    # Write manifest
    manifest = pd.DataFrame(all_meta)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    log.info(
        "Done. %d total cycles from %d subjects written to %s",
        len(manifest), manifest["subject_id"].nunique(), out_dir,
    )
    # Quick summary table
    if not manifest.empty:
        summary = (
            manifest.groupby("condition")["cycle_idx"]
            .count()
            .rename("n_cycles")
            .reset_index()
        )
        log.info("\nCycles per condition:\n%s", summary.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-normalized gait cycle cache from raw TSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset (nested sub{ID}/_tsv/ layout)
  uv run python scripts/build_dataset.py \\
      --data-root '/path/to/Qualisys/Data' \\
      --out-dir data/processed

  # Single subject smoke test
  uv run python scripts/build_dataset.py \\
      --data-root '/path/to/Qualisys/Data' \\
      --subjects FS6 --verbose

  # Flat directory of TSV files (local dev with data/raw/)
  uv run python scripts/build_dataset.py \\
      --data-root data/raw --flat --subjects FS6
        """,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Root data directory. Expected layout: sub{ID}/_tsv/ subdirectories "
            "(e.g. subFS6/_tsv/). Use --flat for a flat directory of TSV files."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for cycle cache and manifest (default: data/processed).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="SUBJECT_ID",
        help="Process only these subject IDs (e.g. --subjects FS6 MS1). "
             "Default: all sub(FS|FT|MS|MT)* directories in --data-root.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Treat --data-root as a flat directory of TSV files (legacy / dev layout).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    build_dataset(
        data_root=args.data_root.resolve(),
        out_dir=args.out_dir.resolve(),
        subjects=args.subjects,
        flat=args.flat,
    )
