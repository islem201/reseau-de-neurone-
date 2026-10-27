# Multiple-shape detection with anchors

A small, educational PyTorch object detector, extended to predict multiple overlapping shapes using anchor boxes. The original model is preserved as a baseline; the comparison also includes a one-anchor-per-cell control.

## Start here

- **Detailed code and exam preparation:** [docs/EXAM_GUIDE.md](docs/EXAM_GUIDE.md)
- **Experiment, metrics and limitations:** [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)
- **Anchor assignment diagram:** [docs/anchor_mechanism.png](docs/anchor_mechanism.png)
- **Circle detection examples:** [results/anchor_comparison/detections.png](results/anchor_comparison/detections.png)
- **Mixed-shape detection examples:** [results/mixed_shape_comparison/detections.png](results/mixed_shape_comparison/detections.png)
- **Comparison chart:** [results/anchor_comparison/metrics.png](results/anchor_comparison/metrics.png)
- **Interactive walkthrough:** [notebooks/anchors_demo.ipynb](notebooks/anchors_demo.ipynb)

The experiment is **class-agnostic**. Generator 3 creates two to four overlapping figures (circles by default; `--shape mixed` can mix circles, squares and triangles). The model predicts boxes, not shape names. Circle and mixed-shape checkpoints are trained separately.

## Install

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The existing `demo.ipynb` uses TensorBoard, so it is included in the requirements. To open notebooks, install Jupyter separately (`python -m pip install jupyter`) if it is not already available. The comparison script itself does not require Jupyter.

## Reproduce the comparison

From the repository root:

```bash
python -m src.compare_anchors --output results/anchor_comparison
```

This creates the default 1,200/200/200 train/validation/test images **in memory**, trains three models for 20 epochs on CPU, and saves figures, metrics, examples and small checkpoints. No generated dataset is committed. On a small CPU it can take several minutes. For a quick functionality check—not a meaningful benchmark—use:

```bash
python -m src.compare_anchors --train 32 --val 8 --test 8 --epochs 2 --output results/quick_check
```

Use different output directories for experiments to avoid overwriting the provided results. Run `python -m src.compare_anchors --help` for all settings. To reproduce the separately trained mixed-shape experiment, use:

```bash
python -m src.compare_anchors --shape mixed --seed 142 --output results/mixed_shape_comparison
```

The mixed labels and predictions remain class-agnostic; use the checkpoints in that directory for mixed shapes.

### Optional: generate PNG/TXT data on disk

Use a **fresh** directory; rerunning into an older, larger directory does not remove leftover files.

```bash
python -m src.dataset_generator_3 --output data/multi --train 1200 --val 200 --test 200 --seed 42
python -m src.compare_anchors --data-dir data/multi --output results/from_disk
```

Every `00000.png` has a `00000.txt` containing one `cx cy w h` line per object, in pixels. Folder-based data need `train`, `val` and `test` directories. The script rejects identical images appearing in different splits. Disk generation uses one RNG sequentially across splits, while the default in-memory comparison uses three separate seeds, so the two workflows are deterministic but do not produce identical datasets.

### Run a saved model on one image

The input must be the same square size used for training (64×64 pixels for the provided checkpoints).

```bash
python -m src.infer_anchors \
  --checkpoint results/anchor_comparison/four_anchors.pt \
  --image data/multi/test/00000.png \
  --output results/my_prediction.png
```

The PNG visualizes predictions; a JSON file beside it contains pixel coordinates and scores. Override the validation-selected threshold with `--confidence 0.5` if desired. Only load checkpoints you trust.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover same-cell assignments and overflow, target/decode round trips, loss gradients, NMS, multi-line data, one-to-one matching, AP, and a small end-to-end run.

## What changed from the initial model?

- `dataset_generator_3.generate` now passes arguments correctly and can generate all three splits.
- The new head produces `5 × number_of_anchors` outputs per cell instead of five; centers are cell-relative and sizes are log ratios to prior anchor dimensions.
- Multiple annotations are assigned to different free slots in the same cell; the new inference path can return multiple boxes.
- The comparison evaluates **all** objects with one-to-one IoU matching, not just the single best prediction. The original model is trained on the last-drawn box and still returns only one prediction, exactly as its inference was designed to do.
- The original notebook now uses the original single-object generator; stale notebook outputs were removed. The original training function's displayed loss average and device handling were corrected without changing its model architecture.

This is an educational experiment, not a production detector. Heavy occlusion, NMS suppressing genuine overlaps, imperfect score calibration, and lack of blank/real-world images all limit its conclusions. See the [report](docs/EXPERIMENT_REPORT.md) and [exam guide](docs/EXAM_GUIDE.md) for the methodology and caveats.
