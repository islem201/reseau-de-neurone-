"""Multi-object labels and datasets shared by the comparison experiments."""

from collections import Counter
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.anchors import default_anchors, encode_targets
from src.dataset_generator_3 import make_example
from src.detector import STRIDE, code


def parse_labels(text: str) -> torch.Tensor:
    """Read one pixel ``cx cy w h`` box per nonempty line."""
    boxes = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"line {line_number}: expected exactly four box values")
        try:
            boxes.append([float(value) for value in parts])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid numeric box value") from exc
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert pixel centers and sizes to pixel corner coordinates."""
    centers, half_sizes = boxes[..., :2], boxes[..., 2:] / 2
    return torch.cat((centers - half_sizes, centers + half_sizes), dim=-1)


class MultiShapeDataset(Dataset):
    """Images, all ground truths, and targets for three comparable detectors.

    The unmodified original model can train on only one object per image. We
    give it the *last drawn* object, which cannot be occluded by a later shape.
    The one-anchor and four-anchor controls attempt to encode every object.
    """

    def __init__(self, samples: Iterable[tuple[Image.Image, str]], size: int = 64):
        self.size = size
        self.single_anchors = ((size / 4, size / 4),)
        self.multi_anchors = default_anchors(size)
        self.items = []
        self.stats = {
            "images": 0,
            "objects": 0,
            "images_with_shared_cell": 0,
            "one_anchor_dropped": 0,
            "four_anchor_dropped": 0,
        }
        for image, text in samples:
            if image.size != (size, size):
                raise ValueError(f"expected {size}x{size} images, got {image.size}")
            boxes = parse_labels(text)
            if len(boxes) == 0:
                raise ValueError("each comparison image needs at least one labeled object")
            # For the single-slot control, prefer the most recently drawn
            # (least occluded) object when multiple centers share a cell.
            one_target, one_dropped = encode_targets(boxes.flip(0), self.single_anchors, size)
            multi_target, multi_dropped = encode_targets(boxes, self.multi_anchors, size)
            baseline_target = code(*boxes[-1].tolist(), size=size)
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
            image_tensor = torch.from_numpy(pixels).permute(2, 0, 1) / 255.0
            self.items.append(
                (image_tensor, xywh_to_xyxy(boxes), baseline_target, one_target, multi_target)
            )
            cells = Counter((int(box[0] // STRIDE), int(box[1] // STRIDE)) for box in boxes)
            self.stats["images"] += 1
            self.stats["objects"] += len(boxes)
            self.stats["images_with_shared_cell"] += int(any(n > 1 for n in cells.values()))
            self.stats["one_anchor_dropped"] += one_dropped
            self.stats["four_anchor_dropped"] += multi_dropped
        if not self.items:
            raise ValueError("dataset is empty")

    @classmethod
    def from_generator(
        cls, count: int, seed: int, size: int = 64, shape: str = "circle",
        min_objects: int = 2, max_objects: int = 4,
    ):
        if count < 1:
            raise ValueError("count must be positive")
        rng = random.Random(seed)
        samples = (
            make_example(size, shape, rng, min_objects=min_objects, max_objects=max_objects)
            for _ in range(count)
        )
        return cls(samples, size=size)

    @classmethod
    def from_folder(cls, folder: Path, size: int = 64):
        folder = Path(folder)
        paths = sorted(folder.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"no PNG images found in {folder}")

        def samples():
            for path in paths:
                label_path = path.with_suffix(".txt")
                text = label_path.read_text(encoding="utf-8")
                with Image.open(path) as image:
                    yield image.convert("RGB"), text

        return cls(samples(), size=size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_multi(batch):
    """Stack images/targets but keep variable-length ground-truth lists."""
    images, boxes, original, one_anchor, four_anchors = zip(*batch)
    return {
        "images": torch.stack(images),
        "boxes": list(boxes),
        "original": torch.stack(original),
        "one_anchor": torch.stack(one_anchor),
        "four_anchors": torch.stack(four_anchors),
    }
