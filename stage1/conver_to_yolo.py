#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import tarfile
import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
ROOT_055 = Path("/data4/dongmin/t-car/data/raw/055.신호등-도로표지판_인지_영상(수도권)/01.데이터/1.Training")

IMG_TAR_DIR = ROOT_055 / "원천데이터_0610" / "1280_720" / "daylight"
LBL_TAR_DIR = ROOT_055 / "라벨링데이터_1026" / "1280_720" / "daylight"

# 작업용 임시 extract (필요 최소만 쌓이게 tar별로 비우면서 진행)
TMP_IMG_DIR = Path("/data4/dongmin/t-car/data/extract/tmp_images")
TMP_LBL_DIR = Path("/data4/dongmin/t-car/data/extract/tmp_labels")

YOLO_ROOT = Path("/data4/dongmin/t-car/data/yolo")
OUT_IMG = {"train": YOLO_ROOT / "images/train", "val": YOLO_ROOT / "images/val"}
OUT_LBL = {"train": YOLO_ROOT / "labels/train", "val": YOLO_ROOT / "labels/val"}

# 클래스 매핑 (yaml과 동일)
CLASS_MAP = {"traffic_sign": 0, "traffic_light": 1}

TRAIN_RATIO = 0.9
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# =========================
# UTILS
# =========================
def ensure_dirs():
    for p in [TMP_IMG_DIR, TMP_LBL_DIR] + list(OUT_IMG.values()) + list(OUT_LBL.values()):
        p.mkdir(parents=True, exist_ok=True)

def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink (fast, no extra space)
    except Exception:
        shutil.copy2(src, dst)

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def xyxy_to_yolo(box, W, H):
    x1, y1, x2, y2 = box
    # 정리: 혹시 뒤집힌 값 방어
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    # 이미지 경계 클리핑
    x1 = clip(x1, 0, W - 1)
    x2 = clip(x2, 0, W - 1)
    y1 = clip(y1, 0, H - 1)
    y2 = clip(y2, 0, H - 1)

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    # 정규화
    return (cx / W, cy / H, bw / W, bh / H)

def extract_tar(tar_path: Path, out_dir: Path):
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(out_dir)

def build_img_index(img_dir: Path):
    """
    tar 풀면 ./s01373322.jpg 이런 식으로 나와서 out_dir 아래에 바로 파일이 있음.
    filename -> Path 매핑
    """
    idx = {}
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx[p.name] = p
    return idx

def iter_json_files(lbl_dir: Path):
    # 라벨 tar는 중간 폴더(c_train_...)가 끼어있음
    for p in lbl_dir.rglob("*.json"):
        yield p

def parse_one_json(jpath: Path):
    d = json.load(open(jpath, "r"))
    W, H = d["image"]["imsize"]
    img_name = d["image"]["filename"]

    anns = d.get("annotation", [])
    lines = []

    for a in anns:
        cls_name = a.get("class", None)
        if cls_name not in CLASS_MAP:
            continue
        box = a.get("box", None)
        if not box or len(box) != 4:
            continue

        x, y, w, h = xyxy_to_yolo(box, W, H)
        # 너무 작은 박스 제거(선택)
        if w <= 0 or h <= 0:
            continue
        cls_id = CLASS_MAP[cls_name]
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return img_name, lines

def main():
    random.seed(SEED)
    ensure_dirs()

    img_tars = sorted(IMG_TAR_DIR.glob("*.tar"))
    if not img_tars:
        raise SystemExit(f"[ERR] No image tars found: {IMG_TAR_DIR}")

    total_imgs = 0
    total_lbls = 0
    skipped_no_img = 0
    skipped_empty = 0

    for img_tar in img_tars:
        # 라벨 tar 이름이 동일하다고 가정 (c_train_..._4.tar)
        lbl_tar = LBL_TAR_DIR / img_tar.name
        if not lbl_tar.exists():
            print(f"[WARN] missing label tar for {img_tar.name} -> skip")
            continue

        # tar별로 임시폴더 비우고 진행(디스크 폭발 방지)
        clean_dir(TMP_IMG_DIR)
        clean_dir(TMP_LBL_DIR)

        print(f"\n[INFO] Extracting IMG: {img_tar.name}")
        extract_tar(img_tar, TMP_IMG_DIR)

        print(f"[INFO] Extracting LBL: {lbl_tar.name}")
        extract_tar(lbl_tar, TMP_LBL_DIR)

        img_idx = build_img_index(TMP_IMG_DIR)

        json_files = list(iter_json_files(TMP_LBL_DIR))
        if not json_files:
            print("[WARN] no jsons in label tar -> skip")
            continue

        # tar 단위로 split을 섞는 것보다, json 파일 단위로 섞어서 split하는 게 더 랜덤함
        random.shuffle(json_files)

        n_train = int(len(json_files) * TRAIN_RATIO)
        split = {"train": json_files[:n_train], "val": json_files[n_train:]}

        for split_name, jlist in split.items():
            for jpath in jlist:
                try:
                    img_name, yolo_lines = parse_one_json(jpath)
                except Exception as e:
                    print(f"[WARN] json parse fail: {jpath} ({e})")
                    continue

                # 라벨이 비면(관심 클래스 없음) 스킵할지 말지:
                if len(yolo_lines) == 0:
                    skipped_empty += 1
                    continue

                img_path = img_idx.get(img_name, None)
                if img_path is None:
                    skipped_no_img += 1
                    continue

                stem = Path(img_name).stem
                out_img = OUT_IMG[split_name] / (stem + img_path.suffix.lower())
                out_lbl = OUT_LBL[split_name] / (stem + ".txt")

                safe_link_or_copy(img_path, out_img)
                out_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

                total_imgs += 1
                total_lbls += 1

        print(f"[INFO] done tar {img_tar.name}: wrote so far imgs={total_imgs}, lbls={total_lbls}, noimg={skipped_no_img}, empty={skipped_empty}")

    print("\n===== SUMMARY =====")
    print("written images:", total_imgs)
    print("written labels:", total_lbls)
    print("skipped (no matching image):", skipped_no_img)
    print("skipped (no target classes):", skipped_empty)
    print("===================\n")

if __name__ == "__main__":
    main()
