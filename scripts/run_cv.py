"""
run_cv.py — k-fold cross-validation (subject-level) for gait classifiers.

Trains one of the three model variants and writes per-fold results to
results/ and checkpoints to checkpoints/.

Usage
-----
    # 10-fold CV — classification
    uv run python scripts/run_cv.py --task classification --n-classes 3 --model two_tower

    # 10-fold CV — regression
    uv run python scripts/run_cv.py --task regression --subject-data data/raw/d_subjectData.csv --model two_tower

    # Low-memory run (25% of cycles)
    uv run python scripts/run_cv.py --task classification --n-classes 3 --model two_tower --subsample-frac 0.25

    # Quick smoke test: only run 2 folds
    uv run python scripts/run_cv.py --max-folds 2

    # Override training hyperparameters
    uv run python scripts/run_cv.py --lr 5e-4 --batch-size 32 --max-epochs 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from gait_ml.dataset import (
    GaitCycleDataset,
    CONDITION_LABELS,
    LABEL_CONDITIONS,
    LABEL_CONDITIONS_3,
    LABEL_MAP_6_TO_3,
    N_CLASSES,
    N_CLASSES_3,
    add_speed_column,
    kfold_splits,
    load_manifest,
)
from gait_ml.models import (
    GRFOnlyClassifier,
    GRFOnlyRegressor,
    MarkerOnlyClassifier,
    MarkerOnlyRegressor,
    TwoTowerClassifier,
    TwoTowerRegressor,
)
from gait_ml.train import evaluate, evaluate_regression, get_device, run_cv_fold, run_cv_fold_regression

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

CLASSIFIER_REGISTRY = {
    "grf_only": GRFOnlyClassifier,
    "marker_only": MarkerOnlyClassifier,
    "two_tower": TwoTowerClassifier,
}
REGRESSOR_REGISTRY = {
    "grf_only": GRFOnlyRegressor,
    "marker_only": MarkerOnlyRegressor,
    "two_tower": TwoTowerRegressor,
}

MODALITY_FOR_MODEL = {
    "grf_only": "grf",
    "marker_only": "markers",
    "two_tower": "both",
}


def _make_model(model_name: str, task: str) -> torch.nn.Module:
    registry = REGRESSOR_REGISTRY if task == "regression" else CLASSIFIER_REGISTRY
    return registry[model_name]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _subsample_manifest(manifest: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    """Return a stratified subsample of the manifest.

    Samples ``frac`` of cycles within each (subject_id, condition) group so
    class and subject balance is preserved.
    """
    return (
        manifest.groupby(["subject_id", "condition"], group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )


def run_cv(
    processed_dir: Path,
    model_name: str,
    task: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    checkpoint_dir: Path,
    results_dir: Path,
    max_folds: int | None,
    subsample_frac: float | None = None,
    subject_data_csv: Path | None = None,
    n_classes: int = 6,
    n_folds: int = 10,
) -> None:
    device = get_device()
    log.info("Device: %s  |  Task: %s  |  n_classes: %s", device, task, n_classes)

    # Label scheme for classification
    if task == "classification":
        if n_classes == 3:
            label_map = LABEL_MAP_6_TO_3
            label_conditions = LABEL_CONDITIONS_3
            n_cls = N_CLASSES_3
        else:
            label_map = None
            label_conditions = LABEL_CONDITIONS
            n_cls = N_CLASSES
    else:
        label_map = None
        label_conditions = LABEL_CONDITIONS
        n_cls = N_CLASSES

    manifest = load_manifest(processed_dir)

    if task == "regression":
        if subject_data_csv is None:
            raise ValueError(
                "--task regression requires --subject-data pointing to d_subjectData.csv"
            )
        manifest = add_speed_column(manifest, subject_data_csv)
        n_missing = manifest["speed_ms"].isna().sum()
        if n_missing > 0:
            log.warning(
                "%d cycles have no speed_ms (unknown condition or missing subject row) "
                "— they will be dropped", n_missing,
            )
            manifest = manifest.dropna(subset=["speed_ms"]).reset_index(drop=True)
        log.info(
            "Speed range: %.2f – %.2f m/s",
            manifest["speed_ms"].min(), manifest["speed_ms"].max(),
        )

    subjects = sorted(manifest["subject_id"].unique())

    if subsample_frac is not None:
        if not 0.0 < subsample_frac <= 1.0:
            raise ValueError(f"--subsample-frac must be in (0, 1]; got {subsample_frac}")
        manifest = _subsample_manifest(manifest, subsample_frac)
        log.info(
            "Subsampled to %.0f%% of cycles  →  %d cycles remaining",
            subsample_frac * 100, len(manifest),
        )

    log.info("Subjects: %d  |  Total cycles: %d", len(subjects), len(manifest))
    log.info(
        "Condition distribution:\n%s",
        manifest.groupby("condition")["cycle_idx"].count().to_string(),
    )

    # Build fold iterator: (fold_id, train_df, test_df, test_subjects)
    # fold_id is used for checkpoint paths and MLflow run names.
    raw_folds = kfold_splits(manifest, n_folds)
    fold_iter = [
        (f"fold_{i}", train_df, test_df, test_subjs)
        for i, (train_df, test_df, test_subjs) in enumerate(raw_folds)
    ]
    cv_label = f"{n_folds}-fold CV"

    if max_folds is not None:
        fold_iter = fold_iter[:max_folds]
        log.info("Running %d folds only (--max-folds %d)", max_folds, max_folds)

    log.info("CV strategy: %s  |  %d folds total", cv_label, len(fold_iter))

    modality = MODALITY_FOR_MODEL[model_name]
    ds_target = "speed_ms" if task == "regression" else "label"

    results_dir.mkdir(parents=True, exist_ok=True)
    all_fold_results: list[dict] = []

    for fold_idx, (fold_id, train_df, test_df, test_subjects) in enumerate(fold_iter):
        log.info(
            "\n--- Fold %d/%d: %s (%d test subjects) ---",
            fold_idx + 1, len(fold_iter), fold_id, len(test_subjects),
        )
        log.info(
            "  Train: %d cycles (%d subjects)  |  Test: %d cycles",
            len(train_df), train_df["subject_id"].nunique(), len(test_df),
        )

        aggregate = task == "regression"
        train_ds = GaitCycleDataset(
            train_df, processed_dir, modality=modality, target=ds_target,
            return_group_id=aggregate, label_map=label_map,
        )
        test_ds = GaitCycleDataset(
            test_df, processed_dir, modality=modality, target=ds_target,
            return_group_id=aggregate, label_map=label_map,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )

        model = _make_model(model_name, task) if task == "regression" else CLASSIFIER_REGISTRY[model_name](n_classes=n_cls)
        ckpt_task = f"classification_{n_cls}class" if task == "classification" else task
        ckpt_path = checkpoint_dir / ckpt_task / model_name / fold_id / "best.pt"

        if task == "regression":
            fold_result = run_cv_fold_regression(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                patience=patience,
                checkpoint_path=ckpt_path,
                device=device,
                log_every=max(1, max_epochs // 10),
            )
            final = fold_result["final_test_metrics"]
            row = {
                "fold": fold_id,
                "test_subjects": ",".join(str(s) for s in test_subjects),
                "model": model_name,
                "task": task,
                "best_epoch": fold_result["best_epoch"],
                "best_val_loss": fold_result["best_val_loss"],
                "rmse": final["rmse"],
                "mae": final["mae"],
                "r2": final["r2"],
            }
            log.info(
                "  Result: rmse=%.4f m/s  mae=%.4f m/s  r2=%.3f  best_epoch=%d",
                final["rmse"], final["mae"], final["r2"], fold_result["best_epoch"],
            )
        else:
            fold_result = run_cv_fold(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                patience=patience,
                checkpoint_path=ckpt_path,
                device=device,
                log_every=max(1, max_epochs // 10),
                n_classes=n_cls,
            )
            final = fold_result["final_test_metrics"]
            per_class = {
                f"acc_{label_conditions[c]}": final["per_class_acc"][c]
                for c in range(n_cls)
            }
            row = {
                "fold": fold_id,
                "test_subjects": ",".join(str(s) for s in test_subjects),
                "model": model_name,
                "task": task,
                "best_epoch": fold_result["best_epoch"],
                "best_val_loss": fold_result["best_val_loss"],
                "overall_acc": final["accuracy"],
                **per_class,
            }
            log.info(
                "  Result: overall_acc=%.3f  best_epoch=%d",
                final["accuracy"], fold_result["best_epoch"],
            )

        all_fold_results.append(row)

    # Aggregate and save
    results_df = pd.DataFrame(all_fold_results)
    cv_prefix = f"{n_folds}fold"
    if task == "classification" and n_classes != 6:
        out_path = results_dir / f"{cv_prefix}_classification_{n_classes}class_{model_name}.csv"
    else:
        out_path = results_dir / f"{cv_prefix}_{task}_{model_name}.csv"
    results_df.to_csv(out_path, index=False)
    log.info("\n=== %s complete ===", cv_label)

    if task == "regression":
        log.info(
            "Mean RMSE: %.4f ± %.4f m/s  |  Mean R²: %.3f ± %.3f",
            results_df["rmse"].mean(), results_df["rmse"].std(),
            results_df["r2"].mean(), results_df["r2"].std(),
        )
    else:
        log.info(
            "Mean accuracy: %.3f ± %.3f",
            results_df["overall_acc"].mean(), results_df["overall_acc"].std(),
        )
        acc_cols = [c for c in results_df.columns if c.startswith("acc_")]
        summary = results_df[acc_cols].mean().rename("mean_acc")
        log.info("\nPer-condition mean accuracy:\n%s", summary.to_string())

    log.info("Results saved to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="k-fold cross-validation (subject-level) for gait condition classifiers."
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="classification: predict condition label (default). regression: predict speed in m/s.",
    )
    parser.add_argument(
        "--model",
        choices=list(CLASSIFIER_REGISTRY.keys()),
        default="two_tower",
        help="Which model variant to train (default: two_tower).",
    )
    parser.add_argument(
        "--subject-data",
        type=Path,
        default=None,
        metavar="CSV",
        help="Path to d_subjectData.csv. Required for --task regression.",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--max-folds", type=int, default=None,
        help="Only run this many folds (useful for smoke-testing).",
    )
    parser.add_argument(
        "--subsample-frac", type=float, default=None, metavar="FRAC",
        help=(
            "Keep only this fraction (0, 1] of cycles per subject/condition "
            "(stratified). Useful for low-memory runs. E.g. --subsample-frac 0.2"
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=10,
        metavar="K",
        help="Number of cross-validation folds (default: 10).",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        choices=[3, 6],
        default=6,
        help=(
            "Number of output classes for classification (ignored for regression). "
            "6 (default): one per condition. "
            "3: walking / slow run (predetermined + Froude A) / fast run (Froude B)."
        ),
    )
    parser.add_argument(
        "--dev", action="store_true",
        help=(
            "Dev/smoke-test mode: 3 folds, 20%% of cycles, batch-size 16, "
            "max-epochs 10, patience 3, no MLflow. "
            "Individual flags override these defaults."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # --dev sets safe small defaults; individual flags still override them
    if args.dev:
        if args.max_folds is None:
            args.max_folds = 3
        if args.subsample_frac is None:
            args.subsample_frac = 0.2
        if args.batch_size == 64:       # only override if still at default
            args.batch_size = 16
        if args.max_epochs == 150:
            args.max_epochs = 10
        if args.patience == 15:
            args.patience = 3
        log.info("--dev mode: 3 folds, 20%% cycles, batch=16, epochs=10, patience=3")

    run_cv(
        processed_dir=args.processed_dir.resolve(),
        model_name=args.model,
        task=args.task,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        results_dir=args.results_dir.resolve(),
        max_folds=args.max_folds,
        subsample_frac=args.subsample_frac,
        subject_data_csv=args.subject_data.resolve() if args.subject_data else None,
        n_classes=args.n_classes,
        n_folds=args.n_folds,
    )
