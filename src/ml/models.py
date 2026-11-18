"""Neural network model architectures and loss functions for piece classification.

Implements multi-head EfficientNet-B0 / ResNet-18 architectures, masked loss,
and 13-class probability synthesis according to:
- STAGE_3.md, Section 4.3
- 06_DATA_AND_MLOPS.md, Section 5
- Explanatory Note, Section 2.4.2
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.datatypes import PIECE_CLASSES

logger = logging.getLogger(__name__)

# Check if PyTorch is available in current environment
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


def synthesize_13_class_probabilities(
    p_occ: np.ndarray,
    p_color: np.ndarray,
    p_type: np.ndarray,
) -> np.ndarray:
    """Combines multi-head probability outputs into canonical 13-class probability distribution.

    Args:
        p_occ: Occupancy probabilities (B, 2), where [:, 0] is empty, [:, 1] is occupied.
        p_color: Color probabilities (B, 2), where [:, 0] is white, [:, 1] is black.
        p_type: Piece type probabilities (B, 6), corresponding to [P, N, B, R, Q, K].

    Returns:
        Array of shape (B, 13) containing probabilities for ['empty', 'wP'..'wK', 'bP'..'bK'].
    """
    p_occ = np.asarray(p_occ, dtype=np.float32)
    p_color = np.asarray(p_color, dtype=np.float32)
    p_type = np.asarray(p_type, dtype=np.float32)

    batch_size = p_occ.shape[0]
    probs = np.zeros((batch_size, 13), dtype=np.float32)

    # Class 0: empty
    probs[:, 0] = p_occ[:, 0]

    # Classes 1..6: White pieces (wP, wN, wB, wR, wQ, wK)
    p_white = p_occ[:, 1] * p_color[:, 0]
    for t_idx in range(6):
        probs[:, 1 + t_idx] = p_white * p_type[:, t_idx]

    # Classes 7..12: Black pieces (bP, bN, bB, bR, bQ, bK)
    p_black = p_occ[:, 1] * p_color[:, 1]
    for t_idx in range(6):
        probs[:, 7 + t_idx] = p_black * p_type[:, t_idx]

    # Numerical normalization guarantee
    row_sums = probs.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums


if TORCH_AVAILABLE:
    class MultiHeadPieceClassifier(nn.Module):
        """Multi-head convolutional neural network for piece classification.

        Features:
        - Shared feature extractor backbone;
        - Occupancy head: 2 logits (empty vs occupied);
        - Color head: 2 logits (white vs black);
        - Type head: 6 logits (P, N, B, R, Q, K).
        """

        def __init__(
            self,
            architecture: str = "efficientnet_b0",
            pretrained: bool = True,
            dropout_rate: float = 0.20,
        ) -> None:
            super().__init__()
            self.architecture = architecture
            feat_dim = 1280

            # Backbone selection
            if architecture == "efficientnet_b0":
                try:
                    import torchvision.models as models
                    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                    base = models.efficientnet_b0(weights=weights)
                    self.features = base.features
                    self.avgpool = base.avgpool
                    feat_dim = base.classifier[1].in_features
                except Exception as exc:
                    logger.warning("torchvision EfficientNet-B0 not available (%s), using lightweight conv backbone.", exc)
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.SiLU(),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.SiLU(),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.SiLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                    )
                    self.avgpool = nn.Identity()
                    feat_dim = 128
            else:  # resnet18 or fallback
                try:
                    import torchvision.models as models
                    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                    base = models.resnet18(weights=weights)
                    self.features = nn.Sequential(*list(base.children())[:-2])
                    self.avgpool = base.avgpool
                    feat_dim = base.fc.in_features
                except Exception as exc:
                    logger.warning("torchvision ResNet-18 not available (%s), using lightweight conv backbone.", exc)
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                    )
                    self.avgpool = nn.Identity()
                    feat_dim = 64

            self.dropout = nn.Dropout(p=dropout_rate)

            # Three task heads
            self.occupancy_head = nn.Linear(feat_dim, 2)
            self.color_head = nn.Linear(feat_dim, 2)
            self.type_head = nn.Linear(feat_dim, 6)

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            return x

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            feats = self.extract_features(x)
            return {
                "occ": self.occupancy_head(feats),
                "color": self.color_head(feats),
                "type": self.type_head(feats),
            }

        def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
            """Evaluates forward pass and returns synthesized (B, 13) probabilities."""
            outputs = self.forward(x)
            p_occ = F.softmax(outputs["occ"], dim=-1)
            p_color = F.softmax(outputs["color"], dim=-1)
            p_type = F.softmax(outputs["type"], dim=-1)

            p_occ_np = p_occ.detach().cpu().numpy()
            p_color_np = p_color.detach().cpu().numpy()
            p_type_np = p_type.detach().cpu().numpy()

            probs_13 = synthesize_13_class_probabilities(p_occ_np, p_color_np, p_type_np)
            return torch.from_numpy(probs_13).to(x.device)


    class SingleHeadPieceClassifier(nn.Module):
        """Standard baseline model predicting 13 classes via single softmax."""

        def __init__(self, architecture: str = "resnet18", pretrained: bool = False) -> None:
            super().__init__()
            try:
                import torchvision.models as models
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                base = models.resnet18(weights=weights)
                in_feats = base.fc.in_features
                base.fc = nn.Linear(in_feats, 13)
                self.model = base
            except Exception:
                self.model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, 13),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


    class MaskedMultiHeadLoss(nn.Module):
        """Masked multi-task loss function for MultiHeadPieceClassifier.

        Masks color and type loss penalties for empty cells (y_occ == 0).
        """

        def __init__(
            self,
            weight_occ: float = 1.0,
            weight_color: float = 1.0,
            weight_type: float = 1.0,
        ) -> None:
            super().__init__()
            self.w_occ = weight_occ
            self.w_color = weight_color
            self.w_type = weight_type
            self.ce_loss = nn.CrossEntropyLoss()

        def forward(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            occ_pred = predictions["occ"]
            occ_target = targets["occ_label"]
            loss_occ = self.ce_loss(occ_pred, occ_target)

            # Mask for occupied cells
            occupied_mask = occ_target == 1

            if occupied_mask.any():
                color_pred = predictions["color"][occupied_mask]
                color_target = targets["color_label"][occupied_mask]
                loss_color = self.ce_loss(color_pred, color_target)

                type_pred = predictions["type"][occupied_mask]
                type_target = targets["type_label"][occupied_mask]
                loss_type = self.ce_loss(type_pred, type_target)
            else:
                loss_color = torch.tensor(0.0, device=occ_pred.device)
                loss_type = torch.tensor(0.0, device=occ_pred.device)

            total_loss = (
                self.w_occ * loss_occ
                + self.w_color * loss_color
                + self.w_type * loss_type
            )

            metrics = {
                "loss_total": float(total_loss.item()),
                "loss_occ": float(loss_occ.item()),
                "loss_color": float(loss_color.item()),
                "loss_type": float(loss_type.item()),
            }

            return total_loss, metrics

else:
    # Fallback placeholders when PyTorch is not installed in runtime environment
    class MultiHeadPieceClassifier:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for MultiHeadPieceClassifier.")

    class SingleHeadPieceClassifier:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for SingleHeadPieceClassifier.")

    class MaskedMultiHeadLoss:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for MaskedMultiHeadLoss.")

