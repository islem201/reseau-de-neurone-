import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from src import dataset_generator_3 as generator
from src.dataset_generator_3 import generate, make_example
from src.detection_metrics import choose_threshold, evaluate_detections
from src.multi_dataset import MultiShapeDataset, collate_multi, parse_labels


class DatasetTests(unittest.TestCase):
    def test_generator_is_reproducible_and_multiline(self):
        image1, label1 = make_example(64, 'circle', random.Random(7), 2, 4)
        image2, label2 = make_example(64, 'circle', random.Random(7), 2, 4)
        self.assertEqual(image1.tobytes(), image2.tobytes())
        self.assertEqual(label1, label2)
        self.assertTrue(2 <= len(parse_labels(label1)) <= 4)

    def test_mixed_shapes_are_still_class_agnostic(self):
        with patch.object(generator, 'draw_shape', wraps=generator.draw_shape) as draw:
            _, labels = make_example(64, 'mixed', random.Random(0), 4, 4)
        self.assertGreater(len({call.args[1] for call in draw.call_args_list}), 1)
        self.assertEqual(parse_labels(labels).shape, (4, 4))

    def test_write_and_load_all_three_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate(root, train=3, val=2, test=1, seed=3)
            for split, expected in [('train', 3), ('val', 2), ('test', 1)]:
                dataset = MultiShapeDataset.from_folder(root / split)
                self.assertEqual(len(dataset), expected)
                self.assertEqual(len(list((root / split).glob('*.txt'))), expected)
                for _, gt, _, _, targets in dataset:
                    self.assertTrue(2 <= len(gt) <= 4)
                    self.assertEqual(int(targets[..., 0].sum()), len(gt))
            batch = collate_multi([MultiShapeDataset.from_folder(root / 'train')[i] for i in (0, 1)])
            self.assertEqual(batch['images'].shape, (2, 3, 64, 64))
            self.assertEqual(batch['four_anchors'].shape, (2, 4, 8, 8, 5))
            self.assertEqual(len(batch['boxes']), 2)

    def test_bad_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_labels('1 2 3 4 5')
        with self.assertRaises(ValueError):
            parse_labels('1 two 3 4')


class MetricTests(unittest.TestCase):
    def test_duplicate_cannot_match_a_ground_truth_twice(self):
        gt = [torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])]
        preds = [(torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.], [20., 20., 30., 30.]]),
                  torch.tensor([0.9, 0.8, 0.7]))]
        m = evaluate_detections(preds, gt, confidence=0.5)
        self.assertEqual((m['tp'], m['fp'], m['fn']), (2, 1, 0))
        self.assertAlmostEqual(m['precision'], 2 / 3)
        self.assertAlmostEqual(m['recall'], 1.0)
        self.assertAlmostEqual(m['f1'], 0.8)
        self.assertAlmostEqual(m['mean_iou_of_true_positives'], 1.0)
        self.assertAlmostEqual(m['ap50'], (51 + 50 * (2 / 3)) / 101, places=6)
        high_threshold = evaluate_detections(preds, gt, confidence=0.75)
        self.assertEqual((high_threshold['tp'], high_threshold['fp'], high_threshold['fn']), (1, 1, 1))
        self.assertAlmostEqual(high_threshold['ap50'], m['ap50'])  # AP ignores operating threshold.
        threshold, val = choose_threshold(preds, gt, thresholds=[0.5, 0.75, 0.95])
        self.assertEqual(threshold, 0.5)
        self.assertAlmostEqual(val['f1'], 0.8)

    def test_empty_predictions_are_finite(self):
        gt = [torch.tensor([[0., 0., 10., 10.]])]
        preds = [(torch.empty(0, 4), torch.empty(0))]
        m = evaluate_detections(preds, gt)
        self.assertEqual((m['tp'], m['fp'], m['fn']), (0, 0, 1))
        self.assertEqual((m['precision'], m['recall'], m['f1'], m['ap50']), (0.0, 0.0, 0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
