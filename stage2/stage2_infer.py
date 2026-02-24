#!/usr/bin/env python3
"""Two-stage inference: YOLO stage1 detect + MobileNet stage2 color classify."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

from stage2_utils import CLASS_NAMES, ensure_dir, is_horizontal_bbox, list_images, pad_bbox


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Two-stage traffic-light color inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage1-weights",
        type=Path,
        default=project_root / "tools/weights/stage1_best.pt",
        help="Stage1 YOLO detector checkpoint",
    )
    parser.add_argument(
        "--stage2-weights",
        type=Path,
        default=project_root / "tools/weights/stage2_best.pth",
        help="Stage2 classifier checkpoint",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "data/yolo/images/val",
        help="Single image or directory",
    )
    parser.add_argument(
        "--out-json-dir",
        type=Path,
        default=project_root / "tools/runs/traffic_stage2/infer/json",
        help="Output directory for per-image JSON results",
    )
    parser.add_argument(
        "--out-vis-dir",
        type=Path,
        default=project_root / "tools/runs/traffic_stage2/infer/vis",
        help="Output directory for visualization images",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="Stage1 detection confidence")
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=1.2,
        help="Keep only stage1 bbox with width/height >= this ratio",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.1, help="BBox padding ratio before crop")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", type=str, default="0", help="YOLO device string")
    parser.add_argument("--cls-device", type=str, default="auto", help="Classifier torch device")
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        choices=["auto", "mobilenet_v3_large", "mobilenet_v3_small"],
        help="Stage2 backbone. 'auto' uses checkpoint metadata when available.",
    )
    return parser.parse_args()


def resolve_cls_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested for classifier but unavailable. Fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_classifier(num_classes: int, backbone: str) -> torch.nn.Module:
    if backbone == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=None)
    elif backbone == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def to_serializable_bbox(box: np.ndarray) -> List[float]:
    return [float(round(v, 4)) for v in box.tolist()]


def draw_detection(image: np.ndarray, bbox: List[float], text: str, color_name: str) -> None:
    color_map: Dict[str, tuple] = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "off": (180, 180, 180),
    }
    bgr = color_map.get(color_name, (255, 255, 255))

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 1)
    ty = max(12, y1 - 6)
    cv2.putText(
        image,
        text,
        (x1, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        bgr,
        1,
        lineType=cv2.LINE_AA,
    )


def run_inference(args: argparse.Namespace) -> None:
    ensure_dir(args.out_json_dir)
    ensure_dir(args.out_vis_dir)

    images = list_images(args.source)
    if not images:
        raise RuntimeError(f"No images found from source: {args.source}")

    # Lazy import so script can still be parsed without ultralytics in dry environments.
    from ultralytics import YOLO

    yolo = YOLO(str(args.stage1_weights))

    cls_device = resolve_cls_device(args.cls_device)
    ckpt = torch.load(args.stage2_weights, map_location=cls_device)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    num_classes = int(ckpt.get("num_classes", len(class_names)))
    ckpt_backbone = ckpt.get("backbone", "mobilenet_v3_small")
    backbone = ckpt_backbone if args.backbone == "auto" else args.backbone

    classifier = build_classifier(num_classes=num_classes, backbone=backbone)
    classifier.load_state_dict(ckpt["model_state_dict"], strict=True)
    classifier.to(cls_device)
    classifier.eval()
    print(f"[INFO] Stage2 backbone: {backbone}")

    preprocess = build_preprocess()

    for idx, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        vis = image.copy()
        img_h, img_w = image.shape[:2]

        results = yolo.predict(
            source=str(image_path),
            conf=args.conf,
            classes=[1],
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=False,
        )

        dets: List[Dict[str, object]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes_xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
            confs = results[0].boxes.conf.detach().cpu().numpy()

            for box, det_conf in zip(boxes_xyxy, confs):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                if not is_horizontal_bbox((x1, y1, x2, y2), min_aspect_ratio=args.min_aspect_ratio):
                    continue
                cx1, cy1, cx2, cy2 = pad_bbox(
                    box=(x1, y1, x2, y2),
                    padding_ratio=args.padding_ratio,
                    img_w=img_w,
                    img_h=img_h,
                )

                crop = image[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inp = preprocess(pil).unsqueeze(0).to(cls_device)
                with torch.no_grad():
                    logits = classifier(inp)
                    probs = F.softmax(logits, dim=1)[0]
                    pred_idx = int(torch.argmax(probs).item())
                    pred_conf = float(probs[pred_idx].item())

                color = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                det_item = {
                    "bbox": to_serializable_bbox(np.array([x1, y1, x2, y2], dtype=np.float32)),
                    "det_conf": float(round(float(det_conf), 6)),
                    "color": color,
                    "color_conf": float(round(pred_conf, 6)),
                }
                dets.append(det_item)

                draw_detection(
                    vis,
                    bbox=det_item["bbox"],
                    text=f"{color}({det_item['det_conf']:.2f}/{det_item['color_conf']:.2f})",
                    color_name=color,
                )

        payload = {
            "image": str(image_path.resolve()),
            "detections": dets,
        }

        json_path = args.out_json_dir / f"{image_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        vis_path = args.out_vis_dir / f"{image_path.stem}.jpg"
        cv2.imwrite(str(vis_path), vis)

        if idx % 50 == 0 or idx == len(images):
            print(f"[INFO] processed {idx}/{len(images)} images")

    print(f"[INFO] JSON results: {args.out_json_dir}")
    print(f"[INFO] Visualization images: {args.out_vis_dir}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
