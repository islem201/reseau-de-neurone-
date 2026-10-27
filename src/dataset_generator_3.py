"""Generate images containing several overlapping, class-agnostic shapes.

Each image has a matching text file containing one ``cx cy w h`` line per
object. Coordinates and dimensions are in pixels, not normalized. Objects
are drawn in annotation order, so later objects may occlude earlier ones.
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")
SHAPE_CHOICES = (*SHAPES, "mixed")
COLOR = (230, 75, 60)
OUTLINE = (25, 25, 25)


def draw_shape(draw, shape, box):
    """Draw one shape; ``box`` is ``(x1, y1, x2, y2)`` in pixels."""
    x1, y1, x2, y2 = box
    if shape == "circle":
        draw.ellipse(box, fill=COLOR, outline=OUTLINE)
    elif shape == "square":
        draw.rectangle(box, fill=COLOR, outline=OUTLINE)
    elif shape == "triangle":
        draw.polygon(
            ((x1 + (x2 - x1) // 2, y1), (x2, y2), (x1, y2)),
            fill=COLOR,
            outline=OUTLINE,
        )
    else:
        raise ValueError(f"Unknown shape {shape!r}; choose one of {SHAPES}.")


def make_example(size, shape, rng, min_objects=2, max_objects=2):
    """Return a PIL image and a multi-line label, using the supplied RNG.

    Each later object is placed near the previous one. This intentionally
    creates overlaps and frequent cases where two centers occupy one grid cell.
    The whole-object boxes are retained even when an object is partly hidden.
    With shape="mixed", each figure randomly uses a circle, square or triangle;
    the labels remain class-agnostic and contain no class identifier.
    """
    if size < 24 or size % 8:
        raise ValueError("size must be at least 24 and a multiple of 8")
    if shape not in SHAPE_CHOICES:
        raise ValueError(f"Unknown shape {shape!r}; choose one of {SHAPE_CHOICES}.")
    if min_objects < 1 or max_objects < min_objects:
        raise ValueError("Require 1 <= min_objects <= max_objects")

    background = rng.randint(225, 250)
    image = Image.new("RGB", (size, size), (background,) * 3)
    draw = ImageDraw.Draw(image)
    n_objects = rng.randint(min_objects, max_objects)
    boxes = []

    for i in range(n_objects):
        side = rng.randint(size // 6, size // 3)
        if i == 0:
            x1 = rng.randint(0, size - side - 1)
            y1 = rng.randint(0, size - side - 1)
        else:
            px1, py1, px2, py2 = boxes[-1]
            prev_cx, prev_cy = (px1 + px2) / 2, (py1 + py2) / 2
            jitter_x = rng.randint(-max(1, side // 3), max(1, side // 3))
            jitter_y = rng.randint(-max(1, side // 3), max(1, side // 3))
            x1 = round(prev_cx - side / 2 + jitter_x)
            y1 = round(prev_cy - side / 2 + jitter_y)
            x1 = max(0, min(x1, size - side - 1))
            y1 = max(0, min(y1, size - side - 1))

        box = (x1, y1, x1 + side, y1 + side)
        boxes.append(box)
        # Mixed mode varies geometry per object but keeps class-agnostic labels.
        object_shape = rng.choice(SHAPES) if shape == "mixed" else shape
        draw_shape(draw, object_shape, box)

    labels = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        labels.append(f"{cx} {cy} {x2 - x1} {y2 - y1}")
    return image, "\n".join(labels)


def generate(
    output: Path,
    size: int = 64,
    train: int = 400,
    val: int = 100,
    shape: str = "circle",
    min_objects: int = 2,
    max_objects: int = 4,
    seed: int = 7,
    test: int = 0,
):
    """Write train/val/(optional) test folders of PNGs and matching labels.

    Use a fresh output directory: existing numbered samples are overwritten,
    while files from an older, larger run are not removed automatically.
    """
    output = Path(output)
    if size < 24 or size % 8:
        raise ValueError("size must be at least 24 and a multiple of 8")
    if shape not in SHAPE_CHOICES:
        raise ValueError(f"Unknown shape {shape!r}; choose one of {SHAPE_CHOICES}.")
    if min_objects < 2 or max_objects < min_objects:
        raise ValueError("Require 2 <= min_objects <= max_objects")
    if any(count < 0 for count in (train, val, test)):
        raise ValueError("Split sizes cannot be negative")

    rng = random.Random(seed)
    for split, total in (("train", train), ("val", val), ("test", test)):
        if total == 0:
            continue
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(total):
            # Pass keywords: the previous version accidentally passed the RNG
            # and object counts in the wrong positional order.
            image, label = make_example(
                size, shape, rng, min_objects=min_objects, max_objects=max_objects
            )
            image.save(folder / f"{i:05d}.png")
            (folder / f"{i:05d}.txt").write_text(label + "\n", encoding="utf-8")

    print(f"Generated {train + val + test} images in {output.resolve()}")
    print(f"Shape: {shape}; objects per image: {min_objects}–{max_objects}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--shape", choices=SHAPE_CHOICES, default="circle")
    parser.add_argument("--min-objects", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate(**vars(args))


if __name__ == "__main__":
    main()
