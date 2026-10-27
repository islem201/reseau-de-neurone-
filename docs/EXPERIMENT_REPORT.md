# Experiment report: original detector versus anchors

## Question and setup

Can an anchor-based extension represent and detect multiple overlapping objects, including objects whose centers share a feature-map cell? We compare three configurations on exactly the same synthetic images:

1. **Original:** the original backbone, head, loss and top-one `predict`. Since it accepts one target per image, it is trained on the last-drawn object (the one not obscured by a later shape), and evaluated against every object.
2. **One-anchor control:** the new anchor parameterization and multi-box inference with one slot per cell. When centers collide, training retains the most recently drawn object in that cell.
3. **Four anchors:** the same new pipeline with four slots per cell and sizes 10, 14, 18 and 22 pixels.

All shapes are circles in the main run. A separate mixed-geometry run is reported below. Both are **class-agnostic detection**, not circle/square/triangle classification. All models use the same backbone architecture, initial backbone seed, training images and shuffle order, Adam learning rate 0.003, batch size 64, and 20 epochs. The original has 6,309 trainable parameters, the one-anchor control 6,309, and the four-anchor model 6,804. The original and anchor models use different box parameterizations, targets, losses, and inference limits. Consequently, this is a comparison of complete configurations, not a perfectly controlled proof that anchor count alone caused every difference.

## Data and evaluation

| Split | Images | Annotated objects | Images with ≥2 centers in one cell | One-anchor assignments dropped | Four-anchor assignments dropped |
|---|---:|---:|---:|---:|---:|
| Train | 1,200 | 3,617 | 918 | 1,293 | 0 |
| Validation | 200 | 596 | 149 | 206 | 0 |
| Test | 200 | 601 | 148 | 217 | 0 |

Generator 3's `make_example` creates 2–4 overlapping shapes in each 64×64 image. Train, validation, and test use separate seeded streams (42, 43 and 44). The experiment checks for exact duplicate input images across splits. The dropped-assignment counts describe the **target encoders' capacity**, not the models' test false negatives. For example, 217 test objects would not fit into a single slot in their center cell, whereas four slots fit all 601. Approximate boxes predicted from neighboring cells can still match ground truths at IoU ≥ 0.5, so this count is not a strict bound on measured recall.

The positive objectness weight is 10. The anchor controls use a 0.01 candidate score cutoff, pre-NMS top 100, NMS IoU 0.60, and at most 20 final candidates per image. NMS settings and training hyperparameters were explored on separate development seeds; each model's operating confidence threshold was then chosen by best validation F1 (original 0.25; one anchor 0.70; four anchors 0.65). Those thresholds were frozen before scoring the final test split. The original `predict` returns at most one candidate per image.

At matching IoU ≥ 0.5, predictions are processed in descending confidence order and greedily matched to the highest-IoU unmatched ground truth **in the same image**. Precision, recall and F1 use each model's validation-selected confidence. AP@0.5 uses the full supplied ranked candidate list and 101-point interpolated precision; it is single-class AP at a single IoU, **not** COCO mAP. It depends on the candidate cutoff, NMS and output caps.

## Held-out test results

| Model | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original, one box/image | 191 | 4 | 410 | 97.9% | 31.8% | 48.0% | 32.4% |
| One anchor/cell | 348 | 103 | 253 | 77.2% | 57.9% | 66.2% | 64.4% |
| Four anchors/cell | 411 | 223 | 190 | 64.8% | 68.4% | 66.6% | 73.0% |

Mean IoU **among true-positive matches only** was 0.801 for the original, 0.716 for one anchor, and 0.710 for four anchors. This conditional mean must be read alongside recall: it does not include missed objects.

### Interpretation

- The original model is precise when it predicts, but can return at most 200 boxes for 601 test objects. Even perfect top-one predictions could reach at most `200/601 = 33.3%` recall. Its measured recall is 31.8%.
- Four anchors achieved 68.4% recall and 73.0% AP@0.5 on this test set, compared with 57.9% recall and 64.4% AP@0.5 for the one-anchor control. All four same-cell slots were sufficient for this dataset's 2–4 objects.
- Four anchors produced more false positives (223 versus 103 for the one-anchor control), so precision was lower. Its F1 improvement over one anchor is small (66.6% versus 66.2%). Do not present the result as an across-the-board win or as statistically significant: this is one test split and one training seed, with no confidence intervals.
- AP evaluates ranking over many score cutoffs; F1 is measured at one validation-selected operating point. They can favor models differently.
- The original-versus-four-anchor gain also includes multi-object targets and multi-box inference. The one-anchor control helps separate those effects, but does not eliminate all differences in priors, assignment or optimization.

The annotations include full boxes for partially or completely occluded shapes. Some hidden-object configurations may be impossible to infer uniquely from pixels. The NMS threshold can suppress a genuine overlapping object, while a looser threshold can retain duplicates. Greedy matching is not a globally optimal box assignment. These limitations explain why more slots cannot guarantee perfect detection.

## Additional experiment: mixed shape geometries

To demonstrate circles, squares and triangles within the same image, generator 3 also supports `--shape mixed`. We trained a **separate** set of models using 1,200/200/200 images and seeds 142/143/144, with the same training and evaluation settings. The model still predicts only boxes, not the shape type. Its 200 test images contain 587 objects; 149 images have at least two centers in one cell. The one-anchor encoder drops 216 test assignments and the four-anchor encoder drops none.

| Mixed-shape model | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original, one box/image | 187 | 13 | 400 | 93.5% | 31.9% | 47.5% | 30.6% |
| One anchor/cell | 350 | 128 | 237 | 73.2% | 59.6% | 65.7% | 62.5% |
| Four anchors/cell | 327 | 185 | 260 | 63.9% | 55.7% | 59.5% | 66.7% |

The validation-selected thresholds were 0.20, 0.70 and 0.80 respectively. Here four anchors have higher AP@0.5 but **lower test precision, recall and F1** than the one-anchor control at their selected operating thresholds. This is an important counterexample to the idea that adding anchors automatically improves every metric. The circle and mixed runs use different datasets and seeds, so their differences cannot be attributed to shape diversity alone. [See the mixed-shape examples](../results/mixed_shape_comparison/detections.png) and [full mixed metrics](../results/mixed_shape_comparison/metrics.json).

## Artifacts

- [Detection examples](../results/anchor_comparison/detections.png): the **first six** test images, not cherry-picked successes. Dashed cyan boxes are ground truth; green boxes are predictions. The title shows matched objects for each model.
- [Metric chart](../results/anchor_comparison/metrics.png).
- [Complete configuration, split statistics and metrics](../results/anchor_comparison/metrics.json).
- [Exact numeric predictions for the displayed examples](../results/anchor_comparison/examples.json).
- `results/anchor_comparison/original.pt`, `one_anchor.pt`, and `four_anchors.pt`: trained circle checkpoints, including anchors and inference settings.
- `results/mixed_shape_comparison/`: the separate mixed-shape figures, metrics, numeric examples and checkpoints.

Reproduce the supplied run from the repository root with:

```bash
python -m src.compare_anchors --output results/anchor_comparison
python -m src.compare_anchors --shape mixed --seed 142 --output results/mixed_shape_comparison
```

The runs used PyTorch 2.2.2+cu121 **on CPU**, NumPy 1.26.4, Pillow 12.3.0 and Matplotlib 3.11.2. Exact reproducibility is not guaranteed across hardware or library versions. Use another output directory to preserve the supplied artifacts.

## What this does not establish

These are small, class-agnostic, synthetic benchmarks. They do not classify mixed shape types or evaluate blank images, varied box aspect ratios, real photographs, localization at stricter IoU thresholds, or robustness across several random seeds. Better evaluation would include these, repeated runs with uncertainty, and a carefully parameter-matched ablation. For a detailed mechanism and exam questions, see [EXAM_GUIDE.md](EXAM_GUIDE.md).
