"""Run a saved comparison checkpoint on a 64x64 PNG image.

Example: python -m src.infer_anchors --checkpoint results/anchor_comparison/four_anchors.pt \\
    --image path/to/image.png --output prediction.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.anchors import AnchorDetector, predict_anchors
from src.detector import Detector, predict


def load_checkpoint(path: Path):
    """Load our tensor-only checkpoint on CPU; only use trusted checkpoint files."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    name, size = checkpoint["model_kind"], checkpoint["image_size"]
    if name == "original":
        model = Detector()
    elif name in ("one_anchor", "four_anchors"):
        model = AnchorDetector(checkpoint["anchors"], size=size)
    else:
        raise ValueError(f"unknown checkpoint kind: {name!r}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def detect_image(model, checkpoint, image, confidence=None):
    size = checkpoint["image_size"]
    if image.size != (size, size):
        raise ValueError(f"expected an image of {size}x{size} pixels, got {image.size}")
    if confidence is None:
        confidence = checkpoint["confidence_threshold"]
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    logits = model(image_tensor)[0]
    if checkpoint["model_kind"] == "original":
        box, score = predict(logits, size=size)
        boxes, scores = box.unsqueeze(0), torch.tensor([score])
        keep = scores >= confidence
        return boxes[keep], scores[keep]
    return predict_anchors(
        logits, model.anchors, size=size, score_threshold=confidence,
        nms_iou=checkpoint["nms_iou"], pre_nms_topk=checkpoint["pre_nms_topk"],
        max_detections=checkpoint["max_detections"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="annotated PNG output")
    parser.add_argument("--confidence", type=float, help="override validation-selected threshold")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error("output must be a .png file")
    if args.output.resolve() == args.image.resolve():
        parser.error("output must be different from the input image")
    torch.set_num_threads(1)
    model, checkpoint = load_checkpoint(args.checkpoint)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    boxes, scores = detect_image(model, checkpoint, image, args.confidence)

    scale = 6  # Upscale for a readable result; reported coordinates stay in pixels.
    visual = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(visual)
    for box, score in zip(boxes.tolist(), scores.tolist()):
        x1, y1, x2, y2 = [round(value * scale) for value in box]
        draw.rectangle((x1, y1, x2, y2), outline=(40, 255, 70), width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{score:.2f}", fill=(0, 0, 0),
                  stroke_width=1, stroke_fill=(40, 255, 70))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    visual.save(args.output)
    report = {"model_kind": checkpoint["model_kind"], "image_size": checkpoint["image_size"],
              "confidence_threshold": args.confidence if args.confidence is not None else checkpoint["confidence_threshold"],
              "boxes_xyxy": boxes.tolist(), "scores": scores.tolist()}
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Annotated image: {args.output}; predictions: {json_path}")


if __name__ == "__main__":
    main()
