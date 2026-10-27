import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from src.compare_anchors import main, verify_disjoint_images
from src.dataset_generator_3 import make_example
from src.infer_anchors import detect_image, load_checkpoint
from src.multi_dataset import MultiShapeDataset


class EndToEndTests(unittest.TestCase):
    def test_split_leakage_is_rejected(self):
        dataset = MultiShapeDataset.from_generator(2, seed=17)
        with self.assertRaisesRegex(ValueError, 'identical image'):
            verify_disjoint_images({'train': dataset, 'test': dataset})

    def test_small_comparison_and_saved_inference(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'results'
            report = main(['--train', '8', '--val', '4', '--test', '4', '--epochs', '1',
                           '--batch-size', '4', '--seed', '777', '--examples', '2', '--output', str(out)])
            self.assertTrue(report['exact_input_images_disjoint'])
            self.assertEqual(report['dataset_stats']['test']['images'], 4)
            self.assertEqual(set(report['models']), {'original', 'one_anchor', 'four_anchors'})
            for name in ('original', 'one_anchor', 'four_anchors'):
                self.assertTrue((out / f'{name}.pt').is_file())
                metrics = report['models'][name]['test_metrics']
                self.assertEqual(metrics['tp'] + metrics['fn'], metrics['ground_truths'])
                self.assertEqual(metrics['tp'] + metrics['fp'], metrics['predictions'])
            for artifact in ('metrics.json', 'examples.json', 'detections.png', 'metrics.png'):
                self.assertTrue((out / artifact).is_file())
            stored = json.loads((out / 'metrics.json').read_text())
            self.assertEqual(stored['config']['seed'], 777)
            model, checkpoint = load_checkpoint(out / 'four_anchors.pt')
            image, _ = make_example(64, 'circle', random.Random(999), 2, 4)
            boxes, scores = detect_image(model, checkpoint, image, confidence=0.0)
            self.assertEqual(boxes.shape[1], 4)
            self.assertEqual(len(boxes), len(scores))
            self.assertTrue(torch.isfinite(boxes).all())
            self.assertTrue(torch.isfinite(scores).all())


if __name__ == '__main__':
    unittest.main()
