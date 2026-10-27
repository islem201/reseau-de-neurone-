"""Simulateur : une seule figure (toujours la même) par image, position/taille aléatoires.

Étiquette (un fichier .txt par image) : "cx cy w h" normalisés dans [0, 1].
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")
COLOR = (230, 75, 60)
OUTLINE = (25, 25, 25)


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    if shape == "circle":
        draw.ellipse(box, fill=COLOR, outline=OUTLINE)
    elif shape == "square":
        draw.rectangle(box, fill=COLOR, outline=OUTLINE)
    else:
        draw.polygon(((x1 + (x2 - x1) // 2, y1), (x2, y2), (x1, y2)), fill=COLOR, outline=OUTLINE)


def make_example(size: int, shape: str, rng: random.Random) -> tuple[Image.Image, str]:
    background = rng.randint(225, 250)
    image = Image.new("RGB", (size, size), (background,) * 3)
    side = rng.randint(size // 6, size // 3)
    x1 = rng.randint(0, size - side - 1)
    y1 = rng.randint(0, size - side - 1)
    draw_shape(ImageDraw.Draw(image), shape, (x1, y1, x1 + side, y1 + side))
    label = (f"{(x1 + side / 2)} {(y1 + side / 2)} "
             f"{side} {side}")
    return image, label


def generate(output: Path, size: int = 64, train: int = 400, val: int = 100,
             shape: str = "circle", seed: int = 7) -> None:
    if size % 8:
        raise ValueError("size doit être un multiple de 8.")
    rng = random.Random(seed)
    for split, total in (("train", train), ("val", val)):
        (output / split).mkdir(parents=True, exist_ok=True)
        for i in range(total):
            image, label = make_example(size, shape, rng)
            image.save(output / split / f"{i:05d}.png")
            (output / split / f"{i:05d}.txt").write_text(label + "\n", encoding="utf-8")
    print(f"{train + val} images ({shape}) générées dans {output.resolve()}")


