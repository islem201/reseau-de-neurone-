import math
import unittest

import torch

from src.anchors import AnchorDetector, anchor_loss, decode_anchors, encode_targets, predict_anchors


class AnchorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.anchors = ((10, 10), (14, 14), (18, 18), (22, 22))

    def test_same_cell_objects_get_distinct_slots(self):
        boxes = torch.tensor([[17., 18., 10., 10.], [18., 19., 11., 11.], [19., 20., 18., 18.]])
        target, dropped = encode_targets(boxes, self.anchors)
        self.assertEqual(target.shape, (4, 8, 8, 5))
        self.assertEqual(dropped, 0)
        self.assertEqual(int(target[:, 2, 2, 0].sum()), 3)
        self.assertEqual(int(target[..., 0].sum()), 3)
        # Perfect 10x10 anchor is preferred for the first object.
        self.assertEqual(float(target[0, 2, 2, 0]), 1.0)
        self.assertAlmostEqual(float(target[0, 2, 2, 1]), 1 / 8)
        self.assertAlmostEqual(float(target[0, 2, 2, 2]), 2 / 8)
        self.assertAlmostEqual(float(target[0, 2, 2, 3]), 0.0)

    def test_overflow_is_counted_not_overwritten(self):
        boxes = torch.tensor([[17., 18., 10., 10.]] * 5)
        target, dropped = encode_targets(boxes, self.anchors)
        self.assertEqual(dropped, 1)
        self.assertEqual(int(target[..., 0].sum()), 4)

    def test_decode_round_trip(self):
        boxes = torch.tensor([[17., 18., 10., 10.], [18., 19., 11., 11.]])
        target, _ = encode_targets(boxes, self.anchors)
        logits = torch.full_like(target, -20.0)
        for anchor, row, col in torch.nonzero(target[..., 0] > 0.5).tolist():
            slot = target[anchor, row, col]
            logits[anchor, row, col, 0] = 10
            logits[anchor, row, col, 1:3] = torch.logit(slot[1:3].clamp(1e-6, 1 - 1e-6))
            logits[anchor, row, col, 3:5] = slot[3:5]
        decoded, scores = decode_anchors(logits, torch.tensor(self.anchors), 64)
        selected = decoded[scores > 0.9]
        expected = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)
        # Match by geometry, regardless of which anchor was assigned.
        for box in expected:
            self.assertLess(float((selected - box).abs().sum(1).min()), 0.001)

    def test_forward_and_empty_or_positive_losses_backpropagate(self):
        model = AnchorDetector(self.anchors)
        logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 4, 8, 8, 5))
        target, _ = encode_targets(torch.tensor([[17., 18., 10., 10.]]), self.anchors)
        loss = anchor_loss(logits, torch.stack([target, target]))
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)
        empty = anchor_loss(model(torch.randn(2, 3, 64, 64)), torch.zeros_like(logits))
        self.assertTrue(math.isfinite(empty.item()))
        self.assertIn('anchors', model.state_dict())

    def test_positive_weight_is_ten_for_all_anchor_counts(self):
        for count in (1, 4):
            logits = torch.zeros((1, count, 1, 1, 5))
            target = torch.zeros_like(logits)
            target[0, 0, 0, 0] = torch.tensor([1., 0.5, 0.5, 0., 0.])
            expected = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[..., 0], target[..., 0], pos_weight=torch.tensor(10.)
            )
            self.assertAlmostEqual(anchor_loss(logits, target).item(), expected.item(), places=6)

    def test_duplicate_predictions_are_suppressed(self):
        anchors = torch.tensor([[10., 10.], [10., 10.]])
        logits = torch.full((2, 8, 8, 5), -100.)
        for a in range(2):
            logits[a, 2, 2] = torch.tensor([10., 0., 0., 0., 0.])
        boxes, scores = predict_anchors(logits, anchors, score_threshold=0.9, nms_iou=0.5)
        self.assertEqual(len(boxes), 1)
        self.assertGreater(float(scores[0]), 0.9)

    def test_validation_of_invalid_inputs(self):
        with self.assertRaises(ValueError):
            encode_targets(torch.tensor([[100., 1., 10., 10.]]), self.anchors)
        with self.assertRaises(ValueError):
            AnchorDetector(((0, 10),))
        with self.assertRaises(ValueError):
            AnchorDetector(self.anchors)(torch.zeros(1, 3, 63, 63))


if __name__ == '__main__':
    unittest.main()
