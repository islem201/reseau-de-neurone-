"""Anchor-based, class-agnostic extension of the original shape detector.

One grid cell can be responsible for several objects. Each anchor is a prior
width/height in *pixels* and gets five outputs: objectness, dx, dy, tw, th.
"""

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.detector import Detector, STRIDE, nms


# Chosen to cover dataset_generator_3's 10–21 pixel objects at size 64.
# These are initial dimensions, not fixed sizes of the final predicted boxes.
def default_anchors(size: int = 64) -> tuple[tuple[float, float], ...]:
    return tuple((side * size / 64, side * size / 64) for side in (10, 14, 18, 22))


def _check_anchors(anchors: Sequence[Sequence[float]]) -> torch.Tensor:
    result = torch.as_tensor(anchors, dtype=torch.float32)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] == 0:
        raise ValueError("anchors must contain at least one (width, height) pair")
    if not torch.isfinite(result).all() or (result <= 0).any():
        raise ValueError("anchor widths and heights must be finite and positive")
    return result


def encode_targets(
    boxes: torch.Tensor,
    anchors: Sequence[Sequence[float]] | torch.Tensor,
    size: int = 64,
) -> tuple[torch.Tensor, int]:
    """Encode pixel ``[cx, cy, w, h]`` boxes into ``(A, G, G, 5)``.

    The center selects a cell. Within that cell, process objects in annotation
    order and assign each to the free anchor with greatest width/height IoU.
    The final two values are log(w / anchor_w), log(h / anchor_h). If more
    than A centers fall in a cell, unmatched objects are counted, not silently
    overwritten. This greedy assignment is deterministic but order-dependent.
    """
    if size < STRIDE or size % STRIDE:
        raise ValueError("size must be a positive multiple of the stride (8)")
    anchor_tensor = _check_anchors(anchors).cpu()
    boxes = torch.as_tensor(boxes, dtype=torch.float32).cpu()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (number_of_objects, 4)")
    if not torch.isfinite(boxes).all():
        raise ValueError("box coordinates must be finite")
    grid = size // STRIDE
    target = torch.zeros((len(anchor_tensor), grid, grid, 5), dtype=torch.float32)
    dropped = 0

    for cx, cy, width, height in boxes.tolist():
        if not (0 <= cx < size and 0 <= cy < size and width > 0 and height > 0):
            raise ValueError("box centers must lie in the image; sizes must be positive")
        gx, gy = cx / STRIDE, cy / STRIDE
        col, row = int(gx), int(gy)
        # IoU between the object and each anchor after aligning their centers.
        overlaps = []
        for aw, ah in anchor_tensor.tolist():
            intersection = min(width, aw) * min(height, ah)
            overlaps.append(intersection / (width * height + aw * ah - intersection))
        best_first = sorted(range(len(overlaps)), key=lambda a: (-overlaps[a], a))
        free = next((a for a in best_first if target[a, row, col, 0] == 0), None)
        if free is None:
            dropped += 1
            continue
        aw, ah = anchor_tensor[free].tolist()
        target[free, row, col] = torch.tensor(
            [1.0, gx - col, gy - row, math.log(width / aw), math.log(height / ah)]
        )
    return target, dropped


class AnchorDetector(Detector):
    """Reuse the original backbone; replace its 5-channel head with 5*A channels."""

    def __init__(self, anchors: Sequence[Sequence[float]], size: int = 64):
        super().__init__()
        if size < STRIDE or size % STRIDE:
            raise ValueError("size must be a positive multiple of 8")
        self.size = size
        self.register_buffer("anchors", _check_anchors(anchors))
        self.head = nn.Conv2d(32, 5 * len(self.anchors), kernel_size=1)
        # Start with a low objectness prior; most anchor/cell slots are empty.
        with torch.no_grad():
            self.head.bias.view(len(self.anchors), 5)[:, 0].fill_(-2.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, self.size, self.size):
            raise ValueError(f"expected input (B, 3, {self.size}, {self.size})")
        raw = super().forward(images)  # (B, A*5, G, G)
        batch, _, rows, cols = raw.shape
        return raw.reshape(batch, len(self.anchors), 5, rows, cols).permute(0, 1, 3, 4, 2)


def anchor_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weighted objectness BCE + positive-only center and log-size losses."""
    if logits.shape != target.shape or logits.ndim != 5 or logits.shape[-1] != 5:
        raise ValueError("logits and targets must both have shape (B, A, G, G, 5)")
    # Upweight scarce positives, with the same weight for A=1 and A=4.
    # This is a hyperparameter, not an automatic class-balancing rule.
    positive_weight = logits.new_tensor(10.0)
    objectness = F.binary_cross_entropy_with_logits(
        logits[..., 0], target[..., 0], pos_weight=positive_weight
    )
    positive = target[..., 0] > 0.5
    if not positive.any():
        # The generators normally have positives; keep empty batches finite too.
        return objectness + logits[..., 1:].sum() * 0.0
    centers = F.mse_loss(torch.sigmoid(logits[..., 1:3][positive]), target[..., 1:3][positive])
    sizes = F.smooth_l1_loss(logits[..., 3:5][positive], target[..., 3:5][positive])
    return objectness + 10.0 * centers + 2.0 * sizes


def decode_anchors(
    logits: torch.Tensor, anchors: torch.Tensor, size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode ``(A, G, G, 5)`` into pixel xyxy boxes and objectness scores."""
    if logits.ndim != 4 or logits.shape[-1] != 5:
        raise ValueError("logits must have shape (A, G, G, 5)")
    anchors = _check_anchors(anchors).to(device=logits.device, dtype=logits.dtype)
    grid = size // STRIDE
    if size % STRIDE or logits.shape[:3] != (len(anchors), grid, grid):
        raise ValueError("anchor count/grid does not match logits and image size")
    rows, cols = torch.meshgrid(
        torch.arange(grid, device=logits.device, dtype=logits.dtype),
        torch.arange(grid, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    offsets = torch.sigmoid(logits[..., 1:3])
    cx = (cols + offsets[..., 0]) * STRIDE
    cy = (rows + offsets[..., 1]) * STRIDE
    # Clamp prevents overflow for extreme, untrained log-scale predictions.
    wh = anchors[:, None, None, :] * torch.exp(logits[..., 3:5].clamp(-4, 4))
    width, height = wh[..., 0], wh[..., 1]
    boxes = torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), dim=-1
    ).clamp(0, size)
    scores = torch.sigmoid(logits[..., 0])
    return boxes.reshape(-1, 4), scores.reshape(-1)


def predict_anchors(
    logits: torch.Tensor,
    anchors: torch.Tensor,
    size: int = 64,
    score_threshold: float = 0.01,
    nms_iou: float = 0.60,
    pre_nms_topk: int = 100,
    max_detections: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter scores, apply NMS, and return up to max_detections per image."""
    if not (0 <= score_threshold <= 1 and 0 < nms_iou <= 1):
        raise ValueError("invalid score or NMS threshold")
    if pre_nms_topk < 1 or max_detections < 1:
        raise ValueError("detection limits must be positive")
    boxes, scores = decode_anchors(logits, anchors, size)
    selected = scores >= score_threshold
    boxes, scores = boxes[selected], scores[selected]
    if scores.numel() == 0:
        return boxes, scores
    order = scores.argsort(descending=True)[:pre_nms_topk]
    boxes, scores = boxes[order], scores[order]
    keep = nms(boxes, scores, iou_threshold=nms_iou)[:max_detections]
    return boxes[keep], scores[keep]
