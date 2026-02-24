#!/usr/bin/env python3
"""Unit tests for stage2 utility logic."""

from __future__ import annotations

import unittest

from stage2_utils import greedy_match, is_horizontal_bbox, map_attribute_to_label, pad_bbox


class TestLabelMapping(unittest.TestCase):
    def test_red(self) -> None:
        label, reason = map_attribute_to_label(
            {"red": "on", "green": "off", "yellow": "off", "left_arrow": "off"}
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(label, "red")

    def test_green_with_arrow(self) -> None:
        label, reason = map_attribute_to_label(
            {"red": "off", "green": "on", "yellow": "off", "left_arrow": "on"}
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(label, "green")

    def test_off_variants(self) -> None:
        label, _ = map_attribute_to_label(
            {
                "red": "off",
                "green": "off",
                "yellow": "off",
                "x_light": "on",
                "others_arrow": "off",
            }
        )
        self.assertEqual(label, "off")

    def test_drop_multi_color(self) -> None:
        label, reason = map_attribute_to_label(
            {"red": "on", "yellow": "on", "green": "off", "left_arrow": "off"}
        )
        self.assertIsNone(label)
        self.assertEqual(reason, "drop_multi_color")


class TestBBoxPadding(unittest.TestCase):
    def test_padding_and_clipping(self) -> None:
        x1, y1, x2, y2 = pad_bbox((5, 5, 15, 15), padding_ratio=0.5, img_w=20, img_h=20)
        self.assertEqual((x1, y1, x2, y2), (0, 0, 20, 20))

    def test_small_box(self) -> None:
        x1, y1, x2, y2 = pad_bbox((10, 10, 12, 13), padding_ratio=0.1, img_w=100, img_h=100)
        self.assertTrue(x2 > x1)
        self.assertTrue(y2 > y1)


class TestBBoxShapeFilter(unittest.TestCase):
    def test_horizontal_pass(self) -> None:
        self.assertTrue(is_horizontal_bbox((0, 0, 20, 10), min_aspect_ratio=1.2))

    def test_square_fail(self) -> None:
        self.assertFalse(is_horizontal_bbox((0, 0, 10, 10), min_aspect_ratio=1.2))

    def test_vertical_fail(self) -> None:
        self.assertFalse(is_horizontal_bbox((0, 0, 8, 20), min_aspect_ratio=1.2))


class TestGreedyMatch(unittest.TestCase):
    def test_iou_matching(self) -> None:
        preds = [(0, 0, 10, 10), (100, 100, 120, 120)]
        gts = [(1, 1, 9, 9), (101, 101, 118, 118)]
        matches = greedy_match(preds, gts, iou_thr=0.5)
        self.assertEqual(len(matches), 2)
        self.assertEqual({(m[0], m[1]) for m in matches}, {(0, 0), (1, 1)})

    def test_threshold_filter(self) -> None:
        preds = [(0, 0, 10, 10)]
        gts = [(50, 50, 60, 60)]
        matches = greedy_match(preds, gts, iou_thr=0.5)
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
