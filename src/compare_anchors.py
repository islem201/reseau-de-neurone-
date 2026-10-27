"""Train and compare the original, one-anchor, and four-anchor detectors.

Example: python -m src.compare_anchors --epochs 20 --output results/anchors
All default data are generated in memory by dataset_generator_3; no dataset
needs to be checked into the repository.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures on a headless machine, too.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anchors import AnchorDetector, anchor_loss, predict_anchors
from src.dataset_generator_3 import SHAPE_CHOICES
from src.detection_metrics import choose_threshold, evaluate_detections
from src.detector import Detector, compute_loss, predict
from src.multi_dataset import MultiShapeDataset, collate_multi

MODEL_NAMES = ("original", "one_anchor", "four_anchors")
DISPLAY_NAMES = {
    "original": "Original: 1 box / image",
    "one_anchor": "Control: 1 anchor / cell",
    "four_anchors": "Proposed: 4 anchors / cell",
}


def verify_disjoint_images(datasets):
    """Reject an identical input image appearing in more than one split."""
    seen = {}
    for split, dataset in datasets.items():
        for image, *_ in dataset:
            digest = hashlib.sha256(image.numpy().tobytes()).digest()
            previous = seen.get(digest)
            if previous is not None and previous != split:
                raise ValueError(f"identical image found in both {previous} and {split} splits")
            seen[digest] = split


def build_model(name: str, dataset: MultiShapeDataset):
    if name == "original":
        return Detector()
    if name == "one_anchor":
        return AnchorDetector(dataset.single_anchors, dataset.size)
    if name == "four_anchors":
        return AnchorDetector(dataset.multi_anchors, dataset.size)
    raise ValueError(f"unknown model: {name}")


def train_one_model(name, train_set, args, device):
    """Train on identical images, shuffle order, optimizer settings and epochs."""
    # The inherited backbones start from the same seed and therefore the same
    # initial backbone weights. A fresh sampler gives all models the same order.
    torch.manual_seed(args.seed)
    model = build_model(name, train_set).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    shuffle_rng = torch.Generator().manual_seed(args.seed + 1000)
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        generator=shuffle_rng, collate_fn=collate_multi,
    )
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            images = batch["images"].to(device)
            targets = batch[name].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = compute_loss(logits, targets) if name == "original" else anchor_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {name} loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
        mean_loss = total_loss / len(train_set)
        losses.append(mean_loss)
        print(f"{name:12s} epoch {epoch:3d}/{args.epochs}: train loss {mean_loss:.4f}", flush=True)
    return model, losses


@torch.no_grad()
def collect_predictions(model, name, dataset, args, device):
    """Collect one result per image; all metric inputs are kept on the CPU."""
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multi)
    all_predictions, ground_truths = [], []
    for batch in loader:
        logits = model(batch["images"].to(device)).detach().cpu()
        for logit, truth in zip(logits, batch["boxes"]):
            if name == "original":
                # This is the original repo's top-one prediction behavior.
                box, score = predict(logit, size=dataset.size)
                boxes, scores = box.unsqueeze(0), torch.tensor([score])
            else:
                boxes, scores = predict_anchors(
                    logit, model.anchors.detach().cpu(), size=dataset.size,
                    score_threshold=args.candidate_threshold, nms_iou=args.nms_iou,
                    pre_nms_topk=args.pre_nms_topk, max_detections=args.max_detections,
                )
            all_predictions.append((boxes, scores))
            ground_truths.append(truth)
    return all_predictions, ground_truths


def save_detection_figure(dataset, results, output, n_examples):
    """First N test images, not hand-picked by performance."""
    n_examples = min(n_examples, len(dataset))
    fig, axes = plt.subplots(n_examples, 4, figsize=(13, 2.85 * n_examples), squeeze=False)
    for index in range(n_examples):
        image, gt, *_ = dataset[index]
        image = image.permute(1, 2, 0).numpy()
        for col, key in enumerate((None, *MODEL_NAMES)):
            ax = axes[index, col]
            ax.imshow(image)
            ax.set_xlim(0, dataset.size)
            ax.set_ylim(dataset.size, 0)
            ax.axis("off")
            for x1, y1, x2, y2 in gt.tolist():
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, color="#09c8ff", lw=1.5, ls="--"))
            if key is None:
                ax.set_title(f"Test #{index}: {len(gt)} ground truths", fontsize=9)
                continue
            boxes, scores = results[key]["test_predictions"][index]
            threshold = results[key]["threshold"]
            keep = scores >= threshold
            for box, score in zip(boxes[keep].tolist(), scores[keep].tolist()):
                x1, y1, x2, y2 = box
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, color="#35ff54", lw=1.4))
                ax.text(x1, max(2, y1), f"{score:.2f}", color="black", fontsize=5.5,
                        bbox={"facecolor": "#35ff54", "edgecolor": "none", "alpha": 0.8, "pad": 0.5})
            sample_metrics = evaluate_detections([(boxes, scores)], [gt], confidence=threshold)
            ax.set_title(f"{DISPLAY_NAMES[key]}\nTP {sample_metrics['tp']}/{len(gt)} (score ≥ {threshold:.3f})", fontsize=8)
    fig.suptitle("Multi-object detections on the first held-out test images", y=0.998, fontsize=13)
    fig.legend(handles=[Patch(facecolor="none", edgecolor="#09c8ff", linestyle="--", label="Ground truth"),
                        Patch(facecolor="none", edgecolor="#35ff54", label="Prediction")],
               loc="upper center", bbox_to_anchor=(0.5, 0.977), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_metrics_figure(results, output):
    metrics = ("precision", "recall", "f1", "ap50")
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.7))
    colors = ("#8b91a1", "#4285c5", "#38a169")
    for i, name in enumerate(MODEL_NAMES):
        values = [results[name]["test_metrics"][key] for key in metrics]
        bars = ax.bar(x + (i - 1) * width, values, width, label=DISPLAY_NAMES[name], color=colors[i])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["Precision", "Recall", "F1", "AP@0.5"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Same held-out multi-object test set")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/anchor_comparison"))
    parser.add_argument("--data-dir", type=Path, help="optional directory with train/val/test PNG + TXT folders")
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--shape", choices=SHAPE_CHOICES, default="circle")
    parser.add_argument("--min-objects", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--candidate-threshold", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument("--pre-nms-topk", type=int, default=100)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.examples < 1:
        parser.error("epochs, batch-size, threads and examples must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if not (0 <= args.candidate_threshold <= 0.05):
        parser.error("candidate-threshold must be between 0 and 0.05 for the validation sweep")
    if not (0 < args.nms_iou <= 1) or args.pre_nms_topk < 1 or args.max_detections < 1:
        parser.error("invalid NMS threshold or detection limits")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        datasets = {key: MultiShapeDataset.from_folder(args.data_dir / folder, size=args.size)
                    for key, folder in (("train", "train"), ("validation", "val"), ("test", "test"))}
        source = f"PNG/TXT folders in {args.data_dir}"
        split_seeds = None
    else:
        split_seeds = {"train": args.seed, "validation": args.seed + 1, "test": args.seed + 2}
        datasets = {
            key: MultiShapeDataset.from_generator(
                count, split_seeds[key], size=args.size, shape=args.shape,
                min_objects=args.min_objects, max_objects=args.max_objects,
            )
            for key, count in (("train", args.train), ("validation", args.val), ("test", args.test))
        }
        source = "dataset_generator_3.make_example (separate seeded streams, in memory)"
    verify_disjoint_images(datasets)
    for split, dataset in datasets.items():
        print(f"{split}: {dataset.stats}", flush=True)

    results = {}
    for name in MODEL_NAMES:
        print(f"\n--- {DISPLAY_NAMES[name]} ---", flush=True)
        model, losses = train_one_model(name, datasets["train"], args, device)
        val_preds, val_gt = collect_predictions(model, name, datasets["validation"], args, device)
        threshold, val_metrics = choose_threshold(val_preds, val_gt)
        test_preds, test_gt = collect_predictions(model, name, datasets["test"], args, device)
        test_metrics = evaluate_detections(test_preds, test_gt, confidence=threshold)
        parameters = sum(p.numel() for p in model.parameters())
        print(f"validation threshold={threshold:.3f}; test P={test_metrics['precision']:.3f} "
              f"R={test_metrics['recall']:.3f} F1={test_metrics['f1']:.3f} "
              f"AP50={test_metrics['ap50']:.3f}", flush=True)
        checkpoint = {
            "model_kind": name,
            "image_size": args.size,
            "anchors": model.anchors.detach().cpu().tolist() if isinstance(model, AnchorDetector) else [],
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "confidence_threshold": threshold,
            "nms_iou": args.nms_iou,
            "candidate_threshold": args.candidate_threshold,
            "pre_nms_topk": args.pre_nms_topk,
            "max_detections": args.max_detections,
        }
        torch.save(checkpoint, args.output / f"{name}.pt")
        results[name] = {"parameters": parameters, "train_losses": losses,
                         "threshold": threshold, "validation_metrics": val_metrics,
                         "test_metrics": test_metrics, "test_predictions": test_preds}

    save_detection_figure(datasets["test"], results, args.output / "detections.png", args.examples)
    save_metrics_figure(results, args.output / "metrics.png")
    examples = []
    for i in range(min(args.examples, len(datasets["test"]))):
        entry = {"test_index": i, "ground_truth_xyxy": datasets["test"][i][1].tolist(), "models": {}}
        for name in MODEL_NAMES:
            boxes, scores = results[name]["test_predictions"][i]
            keep = scores >= results[name]["threshold"]
            entry["models"][name] = {"boxes_xyxy": boxes[keep].tolist(), "scores": scores[keep].tolist()}
        examples.append(entry)
    (args.output / "examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {
        "config": config,
        "data_source": source,
        "split_seeds": split_seeds,
        "exact_input_images_disjoint": True,
        "dataset_stats": {key: value.stats for key, value in datasets.items()},
        "anchors_pixels": {"one_anchor": datasets["train"].single_anchors,
                           "four_anchors": datasets["train"].multi_anchors},
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "models": {name: {key: value for key, value in result.items() if key != "test_predictions"}
                   for name, result in results.items()},
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nArtifacts written to {args.output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    main()
