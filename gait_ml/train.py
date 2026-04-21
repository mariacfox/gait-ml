"""
train.py — Training loop and evaluation for gait condition classifiers.

Provides three reusable functions:

    train_one_epoch          — one pass over a DataLoader, returns mean loss
    evaluate                 — loss + accuracy on a DataLoader, no gradient
    run_cv_fold              — train one CV fold (classification) with early stopping
    run_cv_fold_regression   — train one CV fold (regression) with early stopping

All models follow the two-argument forward signature:
    logits = model(grf, markers)

This works for GRFOnlyClassifier (ignores markers), MarkerOnlyClassifier
(ignores grf), and TwoTowerClassifier (uses both). You don't need separate
training code for each model variant.

Mixed precision (torch.amp) is used automatically when a CUDA GPU is available,
which roughly halves memory usage and speeds up training by 1.5–2×.

Why early stopping?
-------------------
With only 69 subjects per training fold and ~540 cycles per subject, overfitting
is a real risk if you train for a fixed number of epochs. Early stopping monitors
validation loss and stops training when it hasn't improved for `patience` epochs,
then restores the best checkpoint. This prevents the model from memorising the
training set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one full pass over the training DataLoader.

    Parameters
    ----------
    model : nn.Module
        The classifier. Must accept (grf, markers) and return logits.
    loader : DataLoader
        Training DataLoader.
    optimizer : Optimizer
        e.g. Adam.
    criterion : nn.Module
        Loss function, e.g. CrossEntropyLoss.
    device : torch.device
        Where to run computation.
    scaler : GradScaler or None
        For mixed-precision training on CUDA. Pass None to use full precision.

    Returns
    -------
    float
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for grf, markers, labels in loader:
        grf = grf.to(device)
        markers = markers.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(grf, markers)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(grf, markers)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int = 6,
) -> dict[str, float]:
    """Evaluate the model on a DataLoader without gradient computation.

    Parameters
    ----------
    model : nn.Module
        The classifier in eval mode.
    loader : DataLoader
        Validation or test DataLoader.
    criterion : nn.Module
        Loss function.
    device : torch.device
    n_classes : int
        Number of output classes for per-class accuracy computation.

    Returns
    -------
    dict with keys:
        loss          — mean cross-entropy loss
        accuracy      — overall fraction correct
        per_class_acc — list of per-class accuracy (index = class label)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    class_correct = np.zeros(n_classes, dtype=int)
    class_total = np.zeros(n_classes, dtype=int)

    with torch.no_grad():
        for grf, markers, labels in loader:
            grf = grf.to(device)
            markers = markers.to(device)
            labels = labels.to(device)

            logits = model(grf, markers)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    per_class_acc = [
        float(class_correct[c] / class_total[c]) if class_total[c] > 0 else float("nan")
        for c in range(n_classes)
    ]
    overall_acc = float(class_correct.sum() / max(class_total.sum(), 1))

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": overall_acc,
        "per_class_acc": per_class_acc,
    }


# ---------------------------------------------------------------------------
# Single CV fold (classification)
# ---------------------------------------------------------------------------


def run_cv_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 150,
    patience: int = 15,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
    log_every: int = 10,
    n_classes: int = 6,
) -> dict:
    """Train one CV fold to convergence with early stopping.

    Parameters
    ----------
    model : nn.Module
        Untrained (or freshly re-initialised) classifier.
    train_loader : DataLoader
        Training cycles.
    val_loader : DataLoader
        Test cycles for the held-out fold.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    max_epochs : int
        Hard cap on training epochs (safety net if patience never triggers).
    patience : int
        Early stopping: stop if val loss doesn't improve for this many epochs.
    checkpoint_path : Path or None
        If provided, save the best model weights here (.pt file).
    device : torch.device or None
        Defaults to get_device().
    log_every : int
        Log metrics every this many epochs.

    Returns
    -------
    dict
        best_val_loss, best_val_acc, best_epoch, final_test_metrics (dict).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    # Adam adapts the learning rate per parameter, which works well for CNNs where
    # early layers have much smaller gradients than the classification head.
    # weight_decay adds L2 regularization — important with only ~60 training subjects.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealingLR smoothly decays the LR from `lr` down to ~0 over max_epochs.
    # No milestones to tune, and the gentle decay avoids an abrupt LR drop that could
    # stop convergence prematurely when used alongside early stopping.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    # CrossEntropyLoss expects raw logits (not softmax'd) — it applies log-softmax
    # internally for numerical stability.
    criterion = nn.CrossEntropyLoss()

    # Mixed-precision scaler — only useful on CUDA
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict | None = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device, n_classes=n_classes)
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            # Clone the state dict in memory so we can restore best weights at the end
            # without an extra disk round-trip. Overwritten each time a better epoch is found.
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epoch % log_every == 0 or epoch == 1:
            log.info(
                "  Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | patience=%d/%d",
                epoch, train_loss, val_loss, val_acc, epochs_without_improvement, patience,
            )

        if epochs_without_improvement >= patience:
            log.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on val/test set with best weights
    final_metrics = evaluate(model, val_loader, criterion, device)
    log.info(
        "  Best epoch %d | val_loss=%.4f | val_acc=%.3f",
        best_epoch, best_val_loss, best_val_acc,
    )

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_test_metrics": final_metrics,
    }


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------


def _aggregate_by_group(
    preds: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average predictions and targets within each group (differentiable).

    All cycles in the same ``(subject, condition, trial)`` share one true speed,
    so averaging targets is equivalent to taking the single true value.
    Averaging predictions reduces within-trial variance before the loss is
    computed, which prevents individual noisy cycles from dominating.

    Parameters
    ----------
    preds : Tensor, shape (N,)
    targets : Tensor, shape (N,)
    group_ids : Tensor, shape (N,) int64

    Returns
    -------
    group_preds : Tensor, shape (G,)  — mean prediction per group
    group_targets : Tensor, shape (G,)  — mean target per group (= true speed)
    """
    unique_groups, inverse = group_ids.unique(return_inverse=True)
    n_groups = unique_groups.shape[0]

    group_preds = torch.zeros(n_groups, dtype=preds.dtype, device=preds.device)
    group_targets = torch.zeros(n_groups, dtype=targets.dtype, device=targets.device)
    counts = torch.zeros(n_groups, dtype=preds.dtype, device=preds.device)

    # scatter_add_ is differentiable — gradients from the averaged group prediction
    # flow back through the sum and divide to every cycle in that group.
    group_preds.scatter_add_(0, inverse, preds)
    group_targets.scatter_add_(0, inverse, targets)
    counts.scatter_add_(0, inverse, torch.ones_like(preds))

    return group_preds / counts, group_targets / counts


# ---------------------------------------------------------------------------
# Regression training
# ---------------------------------------------------------------------------


def train_regression_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """One training epoch for regression models.

    Parameters
    ----------
    model : nn.Module
        Regressor with forward(grf, markers) → (batch,) float predictions.
    loader : DataLoader
        Yields (grf, markers, speed_ms, group_id) 4-tuples. ``group_id`` is a
        per-``(subject, condition, trial)`` integer used to aggregate cycle
        predictions before computing the loss, reducing within-trial
        correlation. Use ``GaitCycleDataset(..., return_group_id=True)``.
    criterion : nn.Module
        e.g. HuberLoss.

    Returns
    -------
    float
        Mean training loss (computed on trial-averaged predictions).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for grf, markers, targets, group_ids in loader:
        grf = grf.to(device)
        markers = markers.to(device)
        targets = targets.to(device, dtype=torch.float32)
        group_ids = group_ids.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                preds = model(grf, markers)
                group_preds, group_targets = _aggregate_by_group(preds, targets, group_ids)
                loss = criterion(group_preds, group_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(grf, markers)
            group_preds, group_targets = _aggregate_by_group(preds, targets, group_ids)
            loss = criterion(group_preds, group_targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a regression model without gradient computation.

    Expects the loader to yield 4-tuples ``(grf, markers, speed, group_id)``
    (i.e. ``GaitCycleDataset(..., return_group_id=True)``). Metrics are
    computed on **trial-averaged** predictions — one value per
    ``(subject, condition, trial)`` group — which matches how the training
    loss is computed and reflects the true effective sample size.

    Returns
    -------
    dict with keys:
        loss  — mean criterion loss on trial-averaged predictions
        rmse  — root mean squared error (m/s), trial-level
        mae   — mean absolute error (m/s), trial-level
        r2    — coefficient of determination, trial-level
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_group_ids: list[np.ndarray] = []

    with torch.no_grad():
        for grf, markers, targets, group_ids in loader:
            grf = grf.to(device)
            markers = markers.to(device)
            targets = targets.to(device, dtype=torch.float32)

            preds = model(grf, markers)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_group_ids.append(group_ids.numpy())

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    gids_arr = np.concatenate(all_group_ids)

    # Aggregate globally across all batches — gives true trial-level metrics
    unique_groups = np.unique(gids_arr)
    group_preds = np.array([preds_arr[gids_arr == g].mean() for g in unique_groups])
    group_targets = np.array([targets_arr[gids_arr == g].mean() for g in unique_groups])

    # Compute loss on aggregated tensors (mirrors training)
    gp_t = torch.from_numpy(group_preds)
    gt_t = torch.from_numpy(group_targets)
    loss_val = float(criterion(gp_t, gt_t).item())

    residuals = group_preds - group_targets
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((group_targets - group_targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "loss": loss_val,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def run_cv_fold_regression(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 150,
    patience: int = 15,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
    log_every: int = 10,
) -> dict:
    """Train one CV fold for speed regression with early stopping.

    Same interface as ``run_loso_fold`` but uses HuberLoss and reports
    RMSE / MAE / R² instead of accuracy.

    Returns
    -------
    dict
        best_val_loss, best_val_rmse, best_epoch, final_test_metrics (dict).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    # HuberLoss(delta=0.5): quadratic for errors < 0.5 m/s, linear beyond that.
    # Individual gait cycles can produce noisy speed predictions (off-pace strides,
    # stumbles). Huber limits the influence of those outliers vs pure MSE, which
    # squares the error and lets a single bad cycle dominate the gradient.
    criterion = nn.HuberLoss(delta=0.5)

    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict | None = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_regression_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = evaluate_regression(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_rmse = val_metrics["rmse"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epoch % log_every == 0 or epoch == 1:
            log.info(
                "  Epoch %3d | train_loss=%.4f | val_loss=%.4f | "
                "val_rmse=%.4f m/s | val_mae=%.4f m/s | val_r2=%.3f | patience=%d/%d",
                epoch, train_loss, val_loss,
                val_metrics["rmse"], val_metrics["mae"], val_metrics["r2"],
                epochs_without_improvement, patience,
            )

        if epochs_without_improvement >= patience:
            log.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_regression(model, val_loader, criterion, device)
    log.info(
        "  Best epoch %d | val_loss=%.4f | val_rmse=%.4f m/s | val_r2=%.3f",
        best_epoch, best_val_loss, best_val_rmse, final_metrics["r2"],
    )

    return {
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "final_test_metrics": final_metrics,
    }
