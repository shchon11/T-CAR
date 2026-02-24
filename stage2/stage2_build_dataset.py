#!/usr/bin/env python3
"""Build stage2 crop dataset from raw json GT boxes or stage1 predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2

from stage2_utils import (
    CLASS_NAMES,
    LABEL_TO_ID,
    ensure_dir,
    extract_gt_traffic_lights,
    greedy_match,
    is_horizontal_bbox,
    load_json_index,
    map_attribute_to_label,
    pad_bbox,
    parse_pred_label_file,
    resolve_data_root,
    resolve_workspace_root,
)


CSV_COLUMNS = [
    "crop_path",
    "image_path",
    "json_path",
    "split",
    "label",
    "label_id",
    "pred_conf",
    "pred_x1",
    "pred_y1",
    "pred_x2",
    "pred_y2",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "iou",
]


def default_workspace_root() -> Path:
    return resolve_workspace_root(Path(__file__))


def parse_args() -> argparse.Namespace:
    workspace_root = default_workspace_root()
    data_root = resolve_data_root(workspace_root)
    parser = argparse.ArgumentParser(
        description="Build stage2 classification dataset (GT-direct or stage1-pred matched)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--box-source",
        type=str,
        default="gt",
        choices=["gt", "pred"],
        help="Use GT boxes directly (gt) or stage1 predicted boxes matched to GT (pred)",
    )
    parser.add_argument(
        "--pred-label-dir",
        type=Path,
        default=None,
        help="Predicted label txt directory (pred mode only, auto-derived from split when omitted)",
    )
    parser.add_argument(
        "--yolo-image-dir",
        type=Path,
        default=None,
        help="YOLO image split directory (auto: data/yolo/images/<split>)",
    )
    parser.add_argument(
        "--raw-json-root",
        type=Path,
        default=data_root / "raw",
        help="Raw json root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--out-crops-root",
        type=Path,
        default=data_root / "stage2/crops",
        help="Output crop root",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="Output metadata csv path (auto: data/stage2/meta/<split>.csv)",
    )
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary json output path")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for pred->gt match")
    parser.add_argument("--padding-ratio", type=float, default=0.1, help="BBox padding ratio before crop")
    parser.add_argument("--target-type", type=str, default="car", help="traffic_light type filter")
    parser.add_argument("--pred-class-id", type=int, default=1, help="stage1 traffic_light class id")
    parser.add_argument(
        "--min-pred-conf",
        type=float,
        default=0.4,
        help="Drop predicted boxes below this confidence (pred mode only)",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=0.0,
        help="Keep only bbox with width/height >= this ratio (0 disables filter)",
    )
    parser.add_argument("--min-crop-size", type=int, default=10, help="Minimum crop width/height")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N images (0 to disable)",
    )
    args = parser.parse_args()

    if args.yolo_image_dir is None:
        args.yolo_image_dir = data_root / "yolo/images" / args.split
    if args.out_meta is None:
        args.out_meta = data_root / "stage2/meta" / f"{args.split}.csv"
    if args.box_source == "pred" and args.pred_label_dir is None:
        args.pred_label_dir = data_root / "stage2/preds" / args.split / "labels"
    return args


def save_crop_and_row(
    *,
    writer: csv.DictWriter,
    summary: Dict[str, object],
    image,
    image_path: Path,
    json_path: Path,
    split: str,
    label: str,
    crop_box,
    pred_box,
    gt_box,
    pred_conf: float,
    iou: float,
    out_crops_root: Path,
    file_stem: str,
    min_crop_size: int,
    padding_ratio: float,
) -> None:
    img_h, img_w = image.shape[:2]
    cx1, cy1, cx2, cy2 = pad_bbox(
        box=crop_box,
        padding_ratio=padding_ratio,
        img_w=img_w,
        img_h=img_h,
    )

    cw = cx2 - cx1
    ch = cy2 - cy1
    if min(cw, ch) < min_crop_size:
        summary["drop_reasons"]["small_crop"] += 1
        return

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        summary["drop_reasons"]["empty_crop"] += 1
        return

    crop_rel_path = Path(split) / label / f"{file_stem}.jpg"
    crop_abs_path = out_crops_root / crop_rel_path
    ensure_dir(crop_abs_path.parent)

    ok = cv2.imwrite(str(crop_abs_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        summary["drop_reasons"]["imwrite_fail"] += 1
        return

    summary["saved_crops"] += 1
    summary["label_distribution"][label] += 1

    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box
    writer.writerow(
        {
            "crop_path": str(crop_abs_path.resolve()),
            "image_path": str(image_path.resolve()),
            "json_path": str(json_path.resolve()),
            "split": split,
            "label": label,
            "label_id": LABEL_TO_ID[label],
            "pred_conf": f"{float(pred_conf):.6f}",
            "pred_x1": f"{float(px1):.6f}",
            "pred_y1": f"{float(py1):.6f}",
            "pred_x2": f"{float(px2):.6f}",
            "pred_y2": f"{float(py2):.6f}",
            "gt_x1": f"{float(gx1):.6f}",
            "gt_y1": f"{float(gy1):.6f}",
            "gt_x2": f"{float(gx2):.6f}",
            "gt_y2": f"{float(gy2):.6f}",
            "iou": f"{float(iou):.6f}",
        }
    )


def process_pred_mode(
    *,
    args: argparse.Namespace,
    summary: Dict[str, object],
    writer: csv.DictWriter,
    image_path: Path,
    image,
    img_w: int,
    img_h: int,
    json_path: Path,
    gts: List[Dict[str, object]],
) -> None:
    pred_label_path = args.pred_label_dir / f"{image_path.stem}.txt"
    raw_preds = parse_pred_label_file(
        label_path=pred_label_path,
        img_w=img_w,
        img_h=img_h,
        expected_class_id=args.pred_class_id,
    )
    summary["total_candidates"] += len(raw_preds)

    preds = []
    for pred in raw_preds:
        if float(pred["conf"]) < args.min_pred_conf:
            summary["drop_reasons"]["pred_low_conf"] += 1
            continue
        if not is_horizontal_bbox(pred["bbox"], min_aspect_ratio=args.min_aspect_ratio):
            summary["drop_reasons"]["pred_non_horizontal"] += 1
            continue
        preds.append(pred)

    summary["candidates_after_filter"] += len(preds)
    if preds:
        summary["images_with_candidates"] += 1

    if not preds:
        return

    if not gts:
        summary["drop_reasons"]["no_target_gt"] += len(preds)
        return

    pred_boxes = [p["bbox"] for p in preds]
    gt_boxes = [g["box"] for g in gts]
    matches = greedy_match(pred_boxes, gt_boxes, iou_thr=args.iou_thr)
    summary["matched_predictions"] += len(matches)
    unmatched = len(preds) - len(matches)
    if unmatched > 0:
        summary["drop_reasons"]["unmatched_iou"] += unmatched

    for pred_idx, gt_idx, iou in matches:
        pred = preds[pred_idx]
        gt = gts[gt_idx]

        label, reason = map_attribute_to_label(gt["attribute"])
        if label is None:
            summary["drop_reasons"][reason] += 1
            continue

        save_crop_and_row(
            writer=writer,
            summary=summary,
            image=image,
            image_path=image_path,
            json_path=json_path,
            split=args.split,
            label=label,
            crop_box=pred["bbox"],
            pred_box=pred["bbox"],
            gt_box=gt["box"],
            pred_conf=float(pred["conf"]),
            iou=float(iou),
            out_crops_root=args.out_crops_root,
            file_stem=f"{image_path.stem}_p{int(pred['line_idx'])}",
            min_crop_size=args.min_crop_size,
            padding_ratio=args.padding_ratio,
        )


def process_gt_mode(
    *,
    args: argparse.Namespace,
    summary: Dict[str, object],
    writer: csv.DictWriter,
    image_path: Path,
    image,
    gts: List[Dict[str, object]],
    json_path: Path,
) -> None:
    summary["total_candidates"] += len(gts)

    gt_candidates = []
    for gt in gts:
        if not is_horizontal_bbox(gt["box"], min_aspect_ratio=args.min_aspect_ratio):
            summary["drop_reasons"]["gt_non_horizontal"] += 1
            continue
        gt_candidates.append(gt)

    summary["candidates_after_filter"] += len(gt_candidates)
    if gt_candidates:
        summary["images_with_candidates"] += 1

    for gt in gt_candidates:
        label, reason = map_attribute_to_label(gt["attribute"])
        if label is None:
            summary["drop_reasons"][reason] += 1
            continue

        gt_box = gt["box"]
        save_crop_and_row(
            writer=writer,
            summary=summary,
            image=image,
            image_path=image_path,
            json_path=json_path,
            split=args.split,
            label=label,
            crop_box=gt_box,
            pred_box=gt_box,
            gt_box=gt_box,
            pred_conf=1.0,
            iou=1.0,
            out_crops_root=args.out_crops_root,
            file_stem=f"{image_path.stem}_g{int(gt['ann_idx'])}",
            min_crop_size=args.min_crop_size,
            padding_ratio=args.padding_ratio,
        )


def build_one_split(args: argparse.Namespace) -> Dict[str, object]:
    json_index = load_json_index(args.raw_json_root)

    ensure_dir(args.out_crops_root)
    ensure_dir(args.out_meta.parent)
    for label in CLASS_NAMES:
        ensure_dir(args.out_crops_root / args.split / label)

    summary = {
        "split": args.split,
        "box_source": args.box_source,
        "num_images": 0,
        "images_with_candidates": 0,
        "total_candidates": 0,
        "candidates_after_filter": 0,
        "matched_predictions": 0,
        "saved_crops": 0,
        "drop_reasons": Counter(),
        "label_distribution": Counter(),
    }

    image_paths: List[Path] = sorted(args.yolo_image_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No jpg images found: {args.yolo_image_dir}")

    with open(args.out_meta, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for idx, image_path in enumerate(image_paths, start=1):
            summary["num_images"] += 1
            stem = image_path.stem

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                summary["drop_reasons"]["imread_fail"] += 1
                continue
            img_h, img_w = image.shape[:2]

            json_path = json_index.get(stem)
            if json_path is None:
                summary["drop_reasons"]["missing_json"] += 1
                continue

            gts = extract_gt_traffic_lights(
                json_path=json_path,
                target_type=args.target_type,
                img_w=img_w,
                img_h=img_h,
            )

            if args.box_source == "gt":
                if not gts:
                    summary["drop_reasons"]["no_target_gt"] += 1
                    continue
                process_gt_mode(
                    args=args,
                    summary=summary,
                    writer=writer,
                    image_path=image_path,
                    image=image,
                    gts=gts,
                    json_path=json_path,
                )
            else:
                process_pred_mode(
                    args=args,
                    summary=summary,
                    writer=writer,
                    image_path=image_path,
                    image=image,
                    img_w=img_w,
                    img_h=img_h,
                    json_path=json_path,
                    gts=gts,
                )

            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(
                    f"[INFO] {args.split}: processed {idx}/{len(image_paths)} images, "
                    f"candidates={summary['total_candidates']}, crops={summary['saved_crops']}"
                )

    # Backward-compatible summary aliases for older logs/scripts.
    summary["images_with_predictions"] = summary["images_with_candidates"]
    summary["total_predictions"] = summary["total_candidates"]
    summary["predictions_after_filter"] = summary["candidates_after_filter"]

    summary["drop_reasons"] = dict(summary["drop_reasons"])
    summary["label_distribution"] = dict(summary["label_distribution"])

    summary_json = args.summary_json
    if summary_json is None:
        summary_json = args.out_meta.with_name(f"{args.out_meta.stem}_summary.json")
    ensure_dir(summary_json.parent)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] CSV saved: {args.out_meta}")
    print(f"[INFO] Summary saved: {summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    build_one_split(args)


if __name__ == "__main__":
    main()
