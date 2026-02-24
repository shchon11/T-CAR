#!/usr/bin/env python3
"""Run stage1 YOLO inference for stage2 dataset generation.

Output layout (default):
- data/stage2/preds/train/labels/*.txt
- data/stage2/preds/val/labels/*.txt

Multi-GPU mode:
- Pass comma-separated devices, e.g. --device 0,1,2,3
- Images are sharded across GPUs, inferred in parallel, then label files are merged.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import cv2

from stage2_utils import (
    ensure_dir,
    is_horizontal_bbox,
    parse_pred_label_file,
    xyxy_to_yolo_norm,
)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Generate stage1 predictions for stage2 pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage1-weights",
        type=Path,
        default=project_root / "tools/runs/traffic_stage1/weights/best.pt",
        help="Path to stage1 YOLO weight (.pt)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root / "data/yolo/images",
        help="Root containing train/val image dirs (*.jpg only)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "data/stage2/preds",
        help="Prediction output root (split dirs will be created under this path)",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="Detection confidence threshold")
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=1.2,
        help="Keep only bbox with width/height >= this ratio",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device string. Single GPU: '0'. Multi-GPU parallel shard mode: '0,1,2,3'.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Data splits under source-dir",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Remove existing split output directories before inference (default)",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Keep existing outputs and append/overwrite label files only",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep temporary shard outputs when using multi-GPU mode",
    )

    # Hidden internal worker args (used by parent process for multi-GPU sharding).
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-list", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-device", type=str, default=None, help=argparse.SUPPRESS)

    parser.set_defaults(overwrite=True)
    return parser.parse_args()


def parse_device_list(device_arg: str) -> List[str]:
    text = str(device_arg).strip().lower()
    if text in {"", "cpu", "mps"}:
        return [text]

    # normalize variants like "cuda:0,1" -> "0,1"
    text = text.replace("cuda:", "").replace(" ", "")
    if "," in text:
        return [token for token in text.split(",") if token != ""]
    return [text]


def collect_split_images(source_split_dir: Path) -> List[Path]:
    images: List[Path] = sorted(source_split_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No jpg images found: {source_split_dir}")
    return images


def write_image_list_file(images: Sequence[Path], out_dir: Path, stem: str) -> Path:
    ensure_dir(out_dir)
    list_file = out_dir / f"{stem}.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for image_path in images:
            f.write(str(image_path.resolve()))
            f.write("\n")
    return list_file


def run_predict_once(
    args: argparse.Namespace,
    source_list_file: Path,
    device: str,
    out_run_dir: Path,
    verbose: bool = True,
) -> None:
    # Import here so script can still be parsed in environments without ultralytics.
    from ultralytics import YOLO

    model = YOLO(str(args.stage1_weights))
    # stream=True avoids accumulating all Results in RAM for very large datasets.
    result_stream = model.predict(
        source=str(source_list_file),
        imgsz=args.imgsz,
        conf=args.conf,
        classes=[1],
        device=str(device),
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(out_run_dir.parent),
        name=out_run_dir.name,
        exist_ok=True,
        verbose=verbose,
        stream=True,
    )
    for _ in result_stream:
        pass


def run_worker_mode(args: argparse.Namespace) -> None:
    if args.worker_source_list is None or args.worker_out_dir is None or args.worker_device is None:
        raise ValueError("worker mode requires --worker-source-list, --worker-out-dir, --worker-device")
    run_predict_once(
        args=args,
        source_list_file=args.worker_source_list,
        device=args.worker_device,
        out_run_dir=args.worker_out_dir,
        verbose=True,
    )


def merge_shard_labels(shard_out_dirs: Sequence[Path], final_labels_dir: Path) -> None:
    ensure_dir(final_labels_dir)
    merged = 0

    for shard_out in shard_out_dirs:
        shard_labels = shard_out / "labels"
        if not shard_labels.exists():
            continue

        for src_file in shard_labels.glob("*.txt"):
            dst_file = final_labels_dir / src_file.name
            if dst_file.exists():
                dst_file.unlink()
            shutil.copy2(src_file, dst_file)
            merged += 1

    print(f"[INFO] merged label files: {merged} -> {final_labels_dir}")


def post_filter_split_labels(
    labels_dir: Path,
    image_dir: Path,
    conf_thr: float,
    min_aspect_ratio: float,
) -> None:
    stats = Counter()
    image_paths = sorted(image_dir.glob("*.jpg"))

    for image_path in image_paths:
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            stats["missing_or_invalid_image"] += 1
            continue
        img_h, img_w = image.shape[:2]

        preds = parse_pred_label_file(
            label_path=label_path,
            img_w=img_w,
            img_h=img_h,
            expected_class_id=1,
        )
        stats["pred_total"] += len(preds)

        kept_lines: List[str] = []
        for pred in preds:
            conf = float(pred["conf"])
            if conf < conf_thr:
                stats["drop_low_conf"] += 1
                continue
            if not is_horizontal_bbox(pred["bbox"], min_aspect_ratio=min_aspect_ratio):
                stats["drop_non_horizontal"] += 1
                continue

            x1, y1, x2, y2 = pred["bbox"]
            cx, cy, bw, bh = xyxy_to_yolo_norm((x1, y1, x2, y2), img_w, img_h)
            cls_id = int(pred["class_id"])
            kept_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}")
            stats["kept"] += 1

        if kept_lines:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(kept_lines) + "\n")
        else:
            label_path.unlink(missing_ok=True)
            stats["empty_after_filter"] += 1

    print(
        f"[INFO] post-filter done ({image_dir.name}): "
        f"total={stats['pred_total']} kept={stats['kept']} "
        f"drop_conf={stats['drop_low_conf']} drop_shape={stats['drop_non_horizontal']} "
        f"empty_files={stats['empty_after_filter']}"
    )


def run_split(args: argparse.Namespace, split: str) -> None:
    source_split_dir = args.source_dir / split
    if not source_split_dir.exists():
        raise RuntimeError(f"Missing source split directory: {source_split_dir}")

    split_out_dir = args.out_dir / split
    if args.overwrite and split_out_dir.exists():
        shutil.rmtree(split_out_dir)

    images = collect_split_images(source_split_dir)
    devices = parse_device_list(args.device)

    if len(devices) == 1:
        image_list_file = write_image_list_file(images, args.out_dir, f"{split}_images")
        run_predict_once(
            args=args,
            source_list_file=image_list_file,
            device=devices[0],
            out_run_dir=split_out_dir,
            verbose=True,
        )
    else:
        shard_root = args.out_dir / f"{split}_shards"
        if args.overwrite and shard_root.exists():
            shutil.rmtree(shard_root)
        ensure_dir(shard_root)

        shards: List[List[Path]] = [[] for _ in devices]
        for idx, image_path in enumerate(images):
            shards[idx % len(devices)].append(image_path)

        script_path = Path(__file__).resolve()
        procs = []
        shard_out_dirs: List[Path] = []

        for shard_idx, (device, shard_images) in enumerate(zip(devices, shards)):
            if not shard_images:
                continue

            shard_name = f"shard{shard_idx}"
            shard_list_file = write_image_list_file(
                shard_images, shard_root, f"{split}_{shard_name}_images"
            )
            shard_out_dir = shard_root / shard_name
            shard_out_dirs.append(shard_out_dir)

            cmd = [
                sys.executable,
                str(script_path),
                "--worker-mode",
                "--stage1-weights",
                str(args.stage1_weights),
                "--conf",
                str(args.conf),
                "--imgsz",
                str(args.imgsz),
                "--worker-source-list",
                str(shard_list_file),
                "--worker-out-dir",
                str(shard_out_dir),
                "--worker-device",
                str(device),
            ]
            print(
                f"[INFO] launch {split}:{shard_name} device={device} "
                f"images={len(shard_images)}"
            )
            procs.append((shard_name, subprocess.Popen(cmd)))

        failed = []
        for shard_name, proc in procs:
            rc = proc.wait()
            if rc != 0:
                failed.append(f"{shard_name}(exit={rc})")

        if failed:
            raise RuntimeError(f"One or more shard workers failed: {', '.join(failed)}")

        merge_shard_labels(shard_out_dirs, split_out_dir / "labels")
        if not args.keep_shards:
            shutil.rmtree(shard_root, ignore_errors=True)

    labels_dir = split_out_dir / "labels"
    ensure_dir(labels_dir)

    post_filter_split_labels(
        labels_dir=labels_dir,
        image_dir=source_split_dir,
        conf_thr=args.conf,
        min_aspect_ratio=args.min_aspect_ratio,
    )

    print(f"[INFO] {split}: labels saved to {labels_dir}")


def main() -> None:
    args = parse_args()

    if args.worker_mode:
        run_worker_mode(args)
        return

    ensure_dir(args.out_dir)

    if not args.stage1_weights.exists():
        raise FileNotFoundError(f"Missing stage1 weights: {args.stage1_weights}")

    for split in args.splits:
        run_split(args, split)


if __name__ == "__main__":
    main()
