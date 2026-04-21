"""
models.py — PyTorch model architectures for gait condition classification.

Three classifiers share the same encoder building blocks:

    GRFOnlyClassifier     — baseline: GRF waveforms only
    MarkerOnlyClassifier  — ablation: lower-body marker trajectories only
    TwoTowerClassifier    — late fusion: GRF + marker embeddings concatenated

Shared constants (must match build_dataset.py):
    GRF_CHANNELS     = 3   (Fz_L, Fz_R, Fz_total — normalized to BW)
    MARKER_CHANNELS  = 90  (30 lower-body markers × 3 axes)
    N_TIME_POINTS    = 101 (0–100 % gait cycle, pchip-normalized)
    N_CLASSES        = 6   (one per speed condition)

Architecture summary
--------------------
Each encoder:
    stem conv → three ResBlock stages with stride-2 downsampling between
    stages → global average pool → linear projection to embedding_dim.

Fusion:
    [grf_emb | marker_emb] → FC(256→128) → ReLU → dropout → FC(128→6)

Input shapes (batch first):
    GRF:     (B, 3,  101)
    Markers: (B, 90, 101)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants — must match gait_ml/dataset.py and scripts/build_dataset.py
# ---------------------------------------------------------------------------

GRF_CHANNELS = 3       # Fz_L, Fz_R, Fz_total (BW-normalized)
MARKER_CHANNELS = 90   # 30 lower-body markers × 3 axes
N_TIME_POINTS = 101    # 0–100 % gait cycle
N_CLASSES = 6          # walk_preferred … run_froude_b


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # "same" padding: keeps time dimension unchanged for stride=1
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding,
                bias=False,  # BatchNorm already learns a per-channel offset, so Conv bias is redundant
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),  # inplace saves memory; safe because we don't reuse the pre-ReLU value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """Two Conv1d layers with a residual skip connection.

    Spatial dimensions are preserved (stride=1, same padding throughout).
    Channel count is fixed at ``channels`` for both layers.
    """

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        # kernel_size=5 sees ~30 ms of signal at 160 Hz after one 2× downsample —
        # wide enough to capture local shape (loading peak, push-off), small enough
        # to stay local and limit parameter count.
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Adding the residual before the final ReLU lets gradients flow directly
        # to earlier layers during backprop, avoiding vanishing gradients in deeper stacks.
        return self.relu(out + residual)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class GRFEncoder(nn.Module):
    """1D CNN encoder for GRF waveforms.

    Parameters
    ----------
    in_channels : int
        Number of GRF input channels. Default 3 (Fz_L, Fz_R, Fz_total).
    embedding_dim : int
        Size of the output embedding vector. Default 128.

    Input : (batch, in_channels, 101)
    Output: (batch, embedding_dim)
    """

    def __init__(self, in_channels: int = GRF_CHANNELS, embedding_dim: int = 128) -> None:
        super().__init__()
        # kernel_size=7 on the stem gives a wider initial receptive field (~44 ms at 160 Hz),
        # letting the first layer see full loading/push-off transitions before downsampling.
        self.stem = ConvBlock(in_channels, 16, kernel_size=7)   # (B, 16, 101)
        self.stage1 = nn.Sequential(
            ResBlock(16, kernel_size=5),
            # stride=2 halves the time dimension while doubling channels —
            # compresses the sequence as features become more abstract.
            ConvBlock(16, 32, kernel_size=3, stride=2),         # (B, 32, 51)
        )
        self.stage2 = nn.Sequential(
            ResBlock(32, kernel_size=5),
            ConvBlock(32, 64, kernel_size=3, stride=2),         # (B, 64, 26)
        )
        self.stage3 = nn.Sequential(
            ResBlock(64, kernel_size=5),
            ConvBlock(64, 128, kernel_size=3, stride=2),        # (B, 128, 13)
        )
        # AdaptiveAvgPool collapses the 13 remaining timesteps to a single vector.
        # Averaging (vs flattening) is position-invariant — the embedding captures
        # *what* features are present, not *where* in the cycle they peak.
        self.pool = nn.AdaptiveAvgPool1d(1)                     # (B, 128, 1)
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class MarkerEncoder(nn.Module):
    """1D CNN encoder for lower-body marker trajectories.

    Handles the higher channel count (90) via a wider stem convolution
    that compresses channels before the ResBlock stages.

    Parameters
    ----------
    in_channels : int
        Number of marker input channels. Default 90 (30 markers × 3 axes).
    embedding_dim : int
        Size of the output embedding vector. Default 128.

    Input : (batch, in_channels, 101)
    Output: (batch, embedding_dim)
    """

    def __init__(self, in_channels: int = MARKER_CHANNELS, embedding_dim: int = 128) -> None:
        super().__init__()
        # 90 input channels (30 markers × XYZ) is too many to compress aggressively in one step.
        # A wider stem (90→64) blends the per-marker signals before the ResBlocks,
        # rather than bottlenecking to 16 and discarding cross-marker correlations.
        self.stem = ConvBlock(in_channels, 64, kernel_size=7)   # (B, 64, 101)
        self.stage1 = nn.Sequential(
            ResBlock(64, kernel_size=5),
            # Stage 1 holds at 64 channels (no doubling) — we're already at a higher
            # channel count than GRFEncoder, so we defer the expansion to stage 2.
            ConvBlock(64, 64, kernel_size=3, stride=2),         # (B, 64, 51)
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, kernel_size=5),
            ConvBlock(64, 128, kernel_size=3, stride=2),        # (B, 128, 26)
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, kernel_size=5),
            ConvBlock(128, 128, kernel_size=3, stride=2),       # (B, 128, 13)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)                     # (B, 128, 1)
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


class GRFOnlyClassifier(nn.Module):
    """GRF-only baseline classifier.

    GRFEncoder → dropout → linear head.

    Parameters
    ----------
    embedding_dim : int
        Encoder output size. Default 128.
    n_classes : int
        Number of output classes. Default 6.
    dropout : float
        Dropout probability before the classification head. Default 0.3.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = GRFEncoder(embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            # Dropout after pooling, before the linear head — regularizes the
            # 128-d bottleneck representation where overfitting is most likely.
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, n_classes),
        )

    def forward(self, grf: torch.Tensor, _markers: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        grf : torch.Tensor
            Shape (batch, 3, 101).
        _markers : ignored
            Accepted for API compatibility with TwoTowerClassifier.
        """
        return self.head(self.encoder(grf))

    def encode_grf(self, grf: torch.Tensor) -> torch.Tensor:
        """Return GRF embedding (useful for UMAP visualization)."""
        return self.encoder(grf)


class MarkerOnlyClassifier(nn.Module):
    """Marker-only ablation classifier.

    MarkerEncoder → dropout → linear head.

    Parameters
    ----------
    embedding_dim : int
        Encoder output size. Default 128.
    n_classes : int
        Number of output classes. Default 6.
    dropout : float
        Dropout probability before the classification head. Default 0.3.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = MarkerEncoder(embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            # Same dropout placement as GRFOnlyClassifier for consistency.
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, n_classes),
        )

    def forward(self, _grf: torch.Tensor | None, markers: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        _grf : ignored
            Accepted for API compatibility with TwoTowerClassifier.
        markers : torch.Tensor
            Shape (batch, 90, 101).
        """
        return self.head(self.encoder(markers))

    def encode_markers(self, markers: torch.Tensor) -> torch.Tensor:
        """Return marker embedding (useful for UMAP visualization)."""
        return self.encoder(markers)


class TwoTowerClassifier(nn.Module):
    """Late-fusion classifier combining GRF and marker modalities.

    GRFEncoder + MarkerEncoder → concat [256] → FC(256→128) → ReLU
    → dropout → FC(128→n_classes).

    The two encoders are independent and can be trained end-to-end or
    initialized from pre-trained single-modality models.

    Parameters
    ----------
    embedding_dim : int
        Per-tower embedding size. Concatenated dim = 2 × embedding_dim.
    n_classes : int
        Number of output classes. Default 6.
    dropout : float
        Dropout probability in the fusion head. Default 0.3.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.grf_encoder = GRFEncoder(embedding_dim=embedding_dim)
        self.marker_encoder = MarkerEncoder(embedding_dim=embedding_dim)
        fused_dim = embedding_dim * 2  # 256
        self.head = nn.Sequential(
            # A two-layer head gives the network a chance to learn non-linear
            # interactions between the GRF and marker embeddings before classifying.
            # A single linear layer over the concatenation would miss cross-modal patterns.
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, grf: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        grf : torch.Tensor
            Shape (batch, 3, 101).
        markers : torch.Tensor
            Shape (batch, 90, 101).

        Returns
        -------
        torch.Tensor
            Logits, shape (batch, n_classes).
        """
        grf_emb = self.grf_encoder(grf)
        marker_emb = self.marker_encoder(markers)
        fused = torch.cat([grf_emb, marker_emb], dim=1)
        return self.head(fused)

    def encode(self, grf: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
        """Return fused embedding before the classification head."""
        grf_emb = self.grf_encoder(grf)
        marker_emb = self.marker_encoder(markers)
        return torch.cat([grf_emb, marker_emb], dim=1)

    def encode_grf(self, grf: torch.Tensor) -> torch.Tensor:
        return self.grf_encoder(grf)

    def encode_markers(self, markers: torch.Tensor) -> torch.Tensor:
        return self.marker_encoder(markers)


# ---------------------------------------------------------------------------
# Regression models  (same encoders, scalar head)
# ---------------------------------------------------------------------------


class GRFOnlyRegressor(nn.Module):
    """GRF-only speed regressor.

    GRFEncoder → dropout → Linear(embedding_dim, 1) → scalar output.

    Parameters
    ----------
    embedding_dim : int
        Encoder output size. Default 128.
    dropout : float
        Dropout probability before the regression head. Default 0.3.

    Input / Output
    --------------
    forward(grf, _markers=None) → Tensor shape (batch,), float32 speed in m/s.
    """

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = GRFEncoder(embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, grf: torch.Tensor, _markers: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encoder(grf)).squeeze(-1)  # (B,)


class MarkerOnlyRegressor(nn.Module):
    """Marker-only speed regressor.

    MarkerEncoder → dropout → Linear(embedding_dim, 1) → scalar output.

    Parameters
    ----------
    embedding_dim : int
        Encoder output size. Default 128.
    dropout : float
        Dropout probability before the regression head. Default 0.3.

    Input / Output
    --------------
    forward(_grf, markers) → Tensor shape (batch,), float32 speed in m/s.
    """

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = MarkerEncoder(embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, _grf: torch.Tensor | None, markers: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(markers)).squeeze(-1)  # (B,)


class TwoTowerRegressor(nn.Module):
    """Late-fusion speed regressor combining GRF and marker modalities.

    GRFEncoder + MarkerEncoder → concat [256] → FC(256→128) → ReLU
    → dropout → FC(128→1) → scalar output.

    Parameters
    ----------
    embedding_dim : int
        Per-tower embedding size. Default 128.
    dropout : float
        Dropout probability in the fusion head. Default 0.3.

    Input / Output
    --------------
    forward(grf, markers) → Tensor shape (batch,), float32 speed in m/s.
    """

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.grf_encoder = GRFEncoder(embedding_dim=embedding_dim)
        self.marker_encoder = MarkerEncoder(embedding_dim=embedding_dim)
        fused_dim = embedding_dim * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, grf: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
        grf_emb = self.grf_encoder(grf)
        marker_emb = self.marker_encoder(markers)
        fused = torch.cat([grf_emb, marker_emb], dim=1)
        return self.head(fused).squeeze(-1)  # (B,)
