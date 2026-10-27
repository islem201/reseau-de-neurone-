"""Class-agnostic detection metrics with one-to-one, per-image IoU matching."""

import torch

from src.detector import iou


def _rank_and_match(predictions, ground_truths, iou_threshold):
    """Sort all predictions by score, then greedily match within each image."""
    if len(predictions) != len(ground_truths):
        raise ValueError("need one prediction and ground-truth set per image")
    if not 0 < iou_threshold <= 1:
        raise ValueError("invalid matching IoU threshold")
    truths, records = [], []
    for image_index, ((boxes, scores), gt) in enumerate(zip(predictions, ground_truths)):
        boxes = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu().reshape(-1, 4)
        scores = torch.as_tensor(scores, dtype=torch.float32).detach().cpu().reshape(-1)
        gt = torch.as_tensor(gt, dtype=torch.float32).detach().cpu().reshape(-1, 4)
        if len(boxes) != len(scores):
            raise ValueError("one score is required per predicted box")
        if not (torch.isfinite(boxes).all() and torch.isfinite(scores).all() and torch.isfinite(gt).all()):
            raise ValueError("boxes and scores must be finite")
        truths.append(gt)
        for index, score in enumerate(scores.tolist()):
            records.append((score, image_index, index, boxes[index]))
    records.sort(key=lambda item: (-item[0], item[1], item[2]))

    used = [torch.zeros(len(gt), dtype=torch.bool) for gt in truths]
    true_positives, matched_ious, sorted_scores = [], [], []
    for score, image_index, _, box in records:
        gt = truths[image_index]
        match_iou = 0.0
        if len(gt):
            overlaps = iou(box, gt)
            overlaps[used[image_index]] = -1
            best_iou, best_index = overlaps.max(dim=0)
            if float(best_iou) >= iou_threshold:
                used[image_index][int(best_index)] = True
                match_iou = float(best_iou)
        true_positives.append(int(match_iou > 0))
        matched_ious.append(match_iou)
        sorted_scores.append(score)
    return sorted_scores, true_positives, matched_ious, sum(len(gt) for gt in truths)


def _average_precision(true_positives, total_gt):
    """101-point interpolated precision over the full supplied ranked list."""
    if not true_positives or not total_gt:
        return 0.0
    tps = torch.tensor(true_positives, dtype=torch.float64).cumsum(0)
    fps = torch.arange(1, len(tps) + 1, dtype=torch.float64) - tps
    recall_curve = tps / total_gt
    precision_curve = tps / (tps + fps)
    interpolated = [
        float(precision_curve[recall_curve >= r].max()) if (recall_curve >= r).any() else 0.0
        for r in torch.linspace(0, 1, 101, dtype=torch.float64)
    ]
    return sum(interpolated) / 101


def _summary(ranked, confidence, iou_threshold, ap):
    if not 0 <= confidence <= 1:
        raise ValueError("invalid confidence threshold")
    sorted_scores, true_positives, matched_ious, total_gt = ranked
    selected = [i for i, score in enumerate(sorted_scores) if score >= confidence]
    tp = sum(true_positives[i] for i in selected)
    fp = len(selected) - tp
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_mean_iou = sum(matched_ious[i] for i in selected) / tp if tp else 0.0
    return {
        "ground_truths": total_gt,
        "predictions": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_of_true_positives": matched_mean_iou,
        "ap50": ap,
        "confidence_threshold": confidence,
        "matching_iou": iou_threshold,
        "candidate_predictions": len(sorted_scores),
    }


def evaluate_detections(predictions, ground_truths, confidence: float = 0.5, iou_threshold: float = 0.5):
    """Return precision, recall, F1, matched IoU and 101-point AP at an IoU.

    Each element of predictions is (xyxy boxes, scores) for one image. Greedy
    matching processes predictions globally in descending score order and
    matches a prediction to the highest-IoU *unmatched* ground truth in that
    image, if IoU >= iou_threshold. AP uses every supplied candidate; the
    confidence threshold affects the operating-point metrics, not AP.
    """
    ranked = _rank_and_match(predictions, ground_truths, iou_threshold)
    return _summary(ranked, confidence, iou_threshold, _average_precision(ranked[1], ranked[3]))


def choose_threshold(predictions, ground_truths, thresholds=None, iou_threshold: float = 0.5):
    """Choose the highest-F1 operating threshold on validation data only."""
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)] + [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
    if not thresholds:
        raise ValueError("need at least one threshold")
    ranked = _rank_and_match(predictions, ground_truths, iou_threshold)
    ap = _average_precision(ranked[1], ranked[3])
    metrics = [_summary(ranked, t, iou_threshold, ap) for t in thresholds]
    # A higher threshold breaks F1 ties, avoiding gratuitous predictions.
    best = max(metrics, key=lambda m: (m["f1"], m["confidence_threshold"]))
    return best["confidence_threshold"], best
