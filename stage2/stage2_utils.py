#!/usr/bin/env python3
"""Shared helpers for stage2 traffic-light color classification pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


CLASS_NAMES: List[str] = ["red", "yellow", "green", "off"]
LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL: Dict[int, str] = {idx: name for name, idx in LABEL_TO_ID.items()}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLOR_KEYS = {"red", "yellow", "green"}


def resolve_workspace_root(script_file: Path) -> Path:
    """Resolve workspace root for both layouts.

    Supported layouts:
    1) <root>/stage1, <root>/stage2, <root>/weights, <root>/runs
    2) <repo>/<subdir>/stage1, <repo>/<subdir>/stage2, ...
    """
    script_path = Path(script_file).resolve()
    candidates = [
        script_path.parent,  # e.g. <root>/stage2
        script_path.parent.parent,  # e.g. <root>
        script_path.parent.parent.parent,  # fallback one level up
    ]
    for base in candidates:
        if (base / "stage1").is_dir() and (base / "stage2").is_dir():
            return base
    return script_path.parent.parent


def resolve_data_root(workspace_root: Path) -> Path:
    """Resolve data root near workspace root."""
    workspace_root = Path(workspace_root)
    for candidate in (workspace_root / "data", workspace_root.parent / "data"):
        if candidate.exists():
            return candidate
    return workspace_root / "data"


def list_images(source: Path) -> List[Path]:
    """List image files from a file or directory."""
    source = Path(source)
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    if not source.is_dir():
        return []
    return sorted(
        p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json_index(raw_json_root: Path) -> Dict[str, Path]:
    """Build stem->json path index. Raises if duplicate stems exist."""
    raw_json_root = Path(raw_json_root)
    index: Dict[str, Path] = {}
    duplicates: List[str] = []

    for path in raw_json_root.rglob("*.json"):
        stem = path.stem
        if stem in index:
            duplicates.append(stem)
            continue
        index[stem] = path

    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate json stems detected: {preview}")

    return index


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_xyxy(box: Sequence[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = clip(x1, 0.0, float(max(img_w - 1, 0)))
    x2 = clip(x2, 0.0, float(max(img_w - 1, 0)))
    y1 = clip(y1, 0.0, float(max(img_h - 1, 0)))
    y2 = clip(y2, 0.0, float(max(img_h - 1, 0)))
    return x1, y1, x2, y2


def xyxy_to_yolo_norm(box: Sequence[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = sanitize_xyxy(box, img_w, img_h)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def yolo_norm_to_xyxy(
    cx: float, cy: float, bw: float, bh: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    cx *= img_w
    cy *= img_h
    bw *= img_w
    bh *= img_h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return sanitize_xyxy((x1, y1, x2, y2), img_w, img_h)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def greedy_match(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_thr: float,
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching by IoU descending."""
    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pbox in enumerate(pred_boxes):
        for gt_idx, gbox in enumerate(gt_boxes):
            iou = iou_xyxy(pbox, gbox)
            if iou >= iou_thr:
                candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(reverse=True, key=lambda x: x[0])
    used_preds = set()
    used_gts = set()
    matches: List[Tuple[int, int, float]] = []

    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in used_preds or gt_idx in used_gts:
            continue
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        matches.append((pred_idx, gt_idx, iou))

    # Stable output for downstream naming and reproducibility.
    matches.sort(key=lambda x: x[0])
    return matches


def pad_bbox(
    box: Sequence[float], padding_ratio: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    px1 = clip(x1 - pad_x, 0.0, float(img_w))
    py1 = clip(y1 - pad_y, 0.0, float(img_h))
    px2 = clip(x2 + pad_x, 0.0, float(img_w))
    py2 = clip(y2 + pad_y, 0.0, float(img_h))

    return int(math.floor(px1)), int(math.floor(py1)), int(math.ceil(px2)), int(math.ceil(py2))


def bbox_wh(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_aspect_ratio(box: Sequence[float]) -> float:
    """Return width/height ratio. Returns 0 for invalid/flat boxes."""
    bw, bh = bbox_wh(box)
    if bh <= 0.0:
        return 0.0
    return bw / bh


def is_horizontal_bbox(box: Sequence[float], min_aspect_ratio: float) -> bool:
    """True when bbox is horizontally elongated enough."""
    if min_aspect_ratio <= 0.0:
        return True
    return bbox_aspect_ratio(box) >= min_aspect_ratio


def _normalize_attribute(attribute_obj: object) -> Dict[str, str]:
    if isinstance(attribute_obj, dict):
        return {str(k): str(v).lower() for k, v in attribute_obj.items()}
    if isinstance(attribute_obj, list) and attribute_obj and isinstance(attribute_obj[0], dict):
        return {str(k): str(v).lower() for k, v in attribute_obj[0].items()}
    return {}


def map_attribute_to_label(attribute_obj: object) -> Tuple[Optional[str], str]:
    """Map raw traffic-light attribute to 4-class label.

    Returns:
        (label, reason)
        - label in {"red","yellow","green","off"} on success
        - (None, "drop_multi_color") for ambiguous multi-color states.
    """
    attr = _normalize_attribute(attribute_obj)
    on_keys = {k for k, v in attr.items() if v == "on"}
    color_on = sorted(COLOR_KEYS.intersection(on_keys))
    if len(color_on) >= 2:
        return None, "drop_multi_color"
    if len(color_on) == 1:
        return color_on[0], "ok"
    return "off", "ok"


def extract_gt_traffic_lights(
    json_path: Path, target_type: str, img_w: int, img_h: int
) -> List[Dict[str, object]]:
    """Extract car/pedestrian traffic-light GT boxes from raw json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    outputs: List[Dict[str, object]] = []
    for ann_idx, ann in enumerate(data.get("annotation", [])):
        if ann.get("class") != "traffic_light":
            continue
        if target_type and ann.get("type") != target_type:
            continue
        box = ann.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = sanitize_xyxy(box, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        outputs.append(
            {
                "ann_idx": ann_idx,
                "box": (x1, y1, x2, y2),
                "attribute": ann.get("attribute", {}),
            }
        )
    return outputs


def parse_pred_label_file(
    label_path: Path,
    img_w: int,
    img_h: int,
    expected_class_id: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Parse YOLO txt prediction file (cls cx cy w h conf)."""
    preds: List[Dict[str, object]] = []
    if not label_path.exists():
        return preds

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
        except ValueError:
            continue

        if expected_class_id is not None and cls_id != expected_class_id:
            continue

        x1, y1, x2, y2 = yolo_norm_to_xyxy(cx, cy, bw, bh, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        preds.append(
            {
                "line_idx": line_idx,
                "class_id": cls_id,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
            }
        )

    return preds
