#!/usr/bin/env python3
"""Train stage2 traffic-light color classifier with MobileNetV3."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

from stage2_utils import CLASS_NAMES, LABEL_TO_ID, ensure_dir

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm may be absent in minimal envs
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage2 MobileNetV3 classifier")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_large",
        choices=["mobilenet_v3_large", "mobilenet_v3_small"],
        help="Classifier backbone architecture",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class CropCsvDataset(Dataset):
    def __init__(self, csv_path: Path, transform: transforms.Compose):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                crop_path = Path(row["crop_path"])
                if not crop_path.exists():
                    continue

                if row.get("label_id", "") != "":
                    label_id = int(row["label_id"])
                else:
                    label_id = LABEL_TO_ID[row["label"]]

                self.samples.append((crop_path, label_id))

        if not self.samples:
            raise RuntimeError(f"No valid samples loaded from {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, int(label)


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    confusion_matrix: np.ndarray
    class_report: Dict[str, Dict[str, float]]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> Tuple[torch.device, List[int]]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0"), [0]
        return torch.device("cpu"), []

    if device_arg.lower() == "cpu":
        return torch.device("cpu"), []

    if not torch.cuda.is_available():
        if "cuda" in device_arg or "," in device_arg:
            print("[WARN] CUDA requested but unavailable. Fallback to CPU.")
        return torch.device("cpu"), []

    cuda_count = torch.cuda.device_count()

    if device_arg.startswith("cuda:"):
        spec = device_arg.split(":", 1)[1].strip()
        if "," in spec:
            device_ids = [int(x.strip()) for x in spec.split(",") if x.strip() != ""]
        elif spec == "":
            device_ids = [0]
        else:
            device_ids = [int(spec)]
    elif "," in device_arg:
        device_ids = [int(x.strip()) for x in device_arg.split(",") if x.strip() != ""]
    elif device_arg.isdigit():
        device_ids = [int(device_arg)]
    elif device_arg == "cuda":
        device_ids = [0]
    else:
        dev = torch.device(device_arg)
        if dev.type == "cuda":
            device_ids = [dev.index if dev.index is not None else 0]
        else:
            return dev, []

    if not device_ids:
        device_ids = [0]

    invalid = [idx for idx in device_ids if idx < 0 or idx >= cuda_count]
    if invalid:
        raise ValueError(
            f"Invalid CUDA device id(s): {invalid}. Available GPUs: 0..{max(cuda_count - 1, 0)}"
        )

    return torch.device(f"cuda:{device_ids[0]}"), device_ids


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_tf, val_tf


def build_model(num_classes: int, backbone: str) -> nn.Module:
    if backbone == "mobilenet_v3_large":
        try:
            from torchvision.models import MobileNet_V3_Large_Weights

            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        except Exception:
            # Fallback for older torchvision versions.
            model = mobilenet_v3_large(pretrained=True)
    elif backbone == "mobilenet_v3_small":
        try:
            from torchvision.models import MobileNet_V3_Small_Weights

            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception:
            # Fallback for older torchvision versions.
            model = mobilenet_v3_small(pretrained=True)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_sampling_weights(labels: Sequence[int], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    labels_np = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(labels_np, minlength=num_classes)

    class_weights = np.zeros(num_classes, dtype=np.float64)
    total = float(len(labels_np))
    for cls_idx in range(num_classes):
        count = class_counts[cls_idx]
        class_weights[cls_idx] = (total / (num_classes * count)) if count > 0 else 0.0

    sample_weights = class_weights[labels_np]
    return class_weights, sample_weights


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    amp_enabled: bool,
    show_progress: bool = False,
    epoch: int = 0,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(
            loader,
            total=len(loader),
            desc=f"val e{epoch:03d}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

    with torch.no_grad():
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with get_autocast(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            y_true = labels.detach().cpu().numpy()
            y_pred = preds.detach().cpu().numpy()
            for t, p in zip(y_true, y_pred):
                conf_mat[int(t), int(p)] += 1

            if show_progress and tqdm is not None:
                iterator.set_postfix({"loss": f"{(total_loss / max(total_samples, 1)):.4f}"})

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = float(np.trace(conf_mat)) / max(int(conf_mat.sum()), 1)

    class_report: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    for cls_idx in range(num_classes):
        tp = float(conf_mat[cls_idx, cls_idx])
        fp = float(conf_mat[:, cls_idx].sum() - tp)
        fn = float(conf_mat[cls_idx, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(conf_mat[cls_idx, :].sum())

        class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
        class_report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        precision_values.append(precision)
        recall_values.append(recall)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    macro_precision = float(np.mean(precision_values)) if precision_values else 0.0
    macro_recall = float(np.mean(recall_values)) if recall_values else 0.0

    return EvalResult(
        loss=avg_loss,
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        confusion_matrix=conf_mat,
        class_report=class_report,
    )


def save_confusion_matrix(conf_mat: np.ndarray, out_dir: Path) -> None:
    ensure_dir(out_dir)

    csv_path = out_dir / "confusion_matrix.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["true/pred"] + CLASS_NAMES[: conf_mat.shape[0]]
        writer.writerow(header)
        for idx, row in enumerate(conf_mat.tolist()):
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            writer.writerow([name] + row)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(conf_mat, interpolation="nearest", cmap="Blues")
        fig.colorbar(im)
        ticks = list(range(conf_mat.shape[0]))
        labels = CLASS_NAMES[: conf_mat.shape[0]]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                ax.text(j, i, str(conf_mat[i, j]), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Could not save confusion_matrix.png: {exc}")


def save_results_plot(results_rows: List[Dict[str, float]], out_dir: Path) -> None:
    if not results_rows:
        return

    columns = [
        "train/cls_loss",
        "val/cls_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    try:
        import matplotlib.pyplot as plt

        epochs = np.arange(1, len(results_rows) + 1, dtype=np.int32)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        axes_flat = axes.reshape(-1)

        for idx, col in enumerate(columns):
            ax = axes_flat[idx]
            vals = [float(row[col]) for row in results_rows]
            ax.plot(epochs, vals, marker="o", linewidth=1.5, markersize=3)
            ax.set_title(col, fontsize=9)
            ax.grid(alpha=0.3)

        for idx in range(len(columns), len(axes_flat)):
            axes_flat[idx].axis("off")

        for ax in axes_flat:
            ax.set_xlabel("epoch")

        fig.tight_layout()
        fig.savefig(out_dir / "results_per_epoch.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Could not save results_per_epoch.png: {exc}")


def get_grad_scaler(device: torch.device, amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=amp_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def get_autocast(device: torch.device, amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=amp_enabled)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=amp_enabled)
    return nullcontext()


def unwrap_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.out_dir)
    weights_dir = args.out_dir / "weights"
    ensure_dir(weights_dir)

    train_tf, val_tf = build_transforms()
    train_ds = CropCsvDataset(args.train_csv, train_tf)
    val_ds = CropCsvDataset(args.val_csv, val_tf)

    if args.num_classes <= 0:
        raise ValueError("--num-classes must be > 0")

    train_labels = [label for _, label in train_ds.samples]
    class_weights_np, sample_weights_np = build_sampling_weights(train_labels, args.num_classes)
    class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights_np, dtype=torch.double),
        num_samples=len(sample_weights_np),
        replacement=True,
    )

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    device, gpu_ids = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    model = build_model(args.num_classes, args.backbone).to(device)
    if device.type == "cuda" and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = get_grad_scaler(device, amp_enabled)

    print("[INFO] Train samples:", len(train_ds))
    print("[INFO] Val samples:", len(val_ds))
    print("[INFO] Class weights:", class_weights_np.tolist())
    print("[INFO] Backbone:", args.backbone)
    print("[INFO] Device:", device)
    if gpu_ids:
        print("[INFO] CUDA device ids:", gpu_ids)
    print("[INFO] AMP enabled:", amp_enabled)
    print("[INFO] Progress bar:", bool(args.progress and tqdm is not None))
    if args.progress and tqdm is None:
        print("[WARN] tqdm not installed. Progress bars disabled.")

    best_epoch = -1
    best_macro_f1 = -math.inf
    best_eval: EvalResult | None = None
    patience_counter = 0

    history_rows: List[Dict[str, float]] = []
    yolo_style_rows: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        train_iter = train_loader
        if args.progress and tqdm is not None:
            train_iter = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"train e{epoch:03d}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

        for images, labels in train_iter:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_samples += bs

            if args.progress and tqdm is not None:
                train_iter.set_postfix({"loss": f"{(train_loss_sum / max(train_samples, 1)):.4f}"})

        train_loss = train_loss_sum / max(train_samples, 1)
        eval_result = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
            amp_enabled=amp_enabled,
            show_progress=args.progress,
            epoch=epoch,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": eval_result.loss,
            "val_accuracy": eval_result.accuracy,
            "val_macro_f1": eval_result.macro_f1,
        }
        history_rows.append(row)

        yolo_style_rows.append(
            {
                "train/box_loss": 0.0,
                "train/cls_loss": train_loss,
                "train/dfl_loss": 0.0,
                "metrics/precision(B)": eval_result.macro_precision,
                "metrics/recall(B)": eval_result.macro_recall,
                "metrics/mAP50(B)": eval_result.macro_f1,
                "metrics/mAP50-95(B)": eval_result.accuracy,
                "val/box_loss": 0.0,
                "val/cls_loss": eval_result.loss,
                "val/dfl_loss": 0.0,
            }
        )

        print(
            f"[INFO] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={eval_result.loss:.6f} "
            f"val_acc={eval_result.accuracy:.6f} "
            f"val_macro_f1={eval_result.macro_f1:.6f}"
        )

        improved = eval_result.macro_f1 > best_macro_f1
        if improved:
            best_macro_f1 = eval_result.macro_f1
            best_epoch = epoch
            best_eval = eval_result
            patience_counter = 0

            ckpt = {
                "epoch": epoch,
                "model_state_dict": unwrap_state_dict(model),
                "num_classes": args.num_classes,
                "class_names": CLASS_NAMES[: args.num_classes],
                "input_size": 224,
                "backbone": args.backbone,
            }
            torch.save(ckpt, weights_dir / "best.pth")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    # Save last checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_state_dict(model),
            "num_classes": args.num_classes,
            "class_names": CLASS_NAMES[: args.num_classes],
            "input_size": 224,
            "backbone": args.backbone,
        },
        weights_dir / "last.pth",
    )

    # Save history csv
    history_csv = args.out_dir / "history.csv"
    with open(history_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"],
        )
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)

    # Save YOLO-style results.csv
    yolo_results_csv = args.out_dir / "results.csv"
    yolo_fields = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    with open(yolo_results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=yolo_fields)
        writer.writeheader()
        for row in yolo_style_rows:
            writer.writerow(row)

    save_results_plot(yolo_style_rows, args.out_dir)

    if best_eval is None:
        raise RuntimeError("No best model was selected during training")

    save_confusion_matrix(best_eval.confusion_matrix, args.out_dir)

    best_metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_eval.accuracy,
        "best_val_macro_f1": best_eval.macro_f1,
        "best_val_loss": best_eval.loss,
        "class_report": best_eval.class_report,
        "class_names": CLASS_NAMES[: args.num_classes],
    }
    with open(args.out_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Best checkpoint: {weights_dir / 'best.pth'}")
    print(f"[INFO] Best metrics json: {args.out_dir / 'best_metrics.json'}")
    print(f"[INFO] YOLO-style results csv: {yolo_results_csv}")
    print(f"[INFO] Epoch plot: {args.out_dir / 'results_per_epoch.png'}")


if __name__ == "__main__":
    main()
