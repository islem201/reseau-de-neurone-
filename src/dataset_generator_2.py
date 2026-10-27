"""Générateur de données : une figure par image, de classe et couleur aléatoires.

Classes : circle=0, square=1, triangle=2.
Étiquette : "cx cy w h class_id" (coordonnées en pixels).
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")
COLORS = (
    (230, 75, 60),    # rouge
    (50, 130, 220),   # bleu
    (50, 175, 100),   # vert
    (235, 180, 45),   # jaune
    (160, 80, 200),   # violet
)
OUTLINE = (25, 25, 25)


def draw_shape(draw, shape, box, color):
    x1, y1, x2, y2 = box

    if shape == "circle":
        draw.ellipse(box, fill=color, outline=OUTLINE)
    elif shape == "square":
        draw.rectangle(box, fill=color, outline=OUTLINE)
    else:
        draw.polygon(
            ((x1 + (x2 - x1) // 2, y1),
             (x2, y2),
             (x1, y2)),
            fill=color,
            outline=OUTLINE,
        )


def make_example(size, rng):
    background = rng.randint(225, 250)
    image = Image.new("RGB", (size, size), (background,) * 3)

    # Une seule figure par image
    class_id = rng.randrange(len(SHAPES))
    shape = SHAPES[class_id]
    color = rng.choice(COLORS)

    side = rng.randint(size // 6, size // 3)
    x1 = rng.randint(0, size - side - 1)
    y1 = rng.randint(0, size - side - 1)
    x2, y2 = x1 + side, y1 + side

    draw_shape(
        ImageDraw.Draw(image),
        shape,
        (x1, y1, x2, y2),
        color,
    )

    cx = x1 + side / 2
    cy = y1 + side / 2

    label = f"{cx} {cy} {side} {side} {class_id}"
    return image, label


def generate(
    output: Path,
    size: int = 64,
    train: int = 400,
    val: int = 100,
    seed: int = 7,
):
    if size % 8:
        raise ValueError("size doit être un multiple de 8.")
    if size < 24:
        raise ValueError("size doit être supérieur ou égal à 24.")

    rng = random.Random(seed)

    for split, total in (("train", train), ("val", val)):
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)

        for i in range(total):
            image, label = make_example(size, rng)
            image.save(folder / f"{i:05d}.png")
            (folder / f"{i:05d}.txt").write_text(
                label + "\n", encoding="utf-8"
            )

    print(
        f"{train + val} images générées dans {output.resolve()}"
    )
    print("Classes : circle=0, square=1, triangle=2")

