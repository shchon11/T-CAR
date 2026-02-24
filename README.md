# T-CAR Tools (Portable Usage)

This guide is path-agnostic. You can clone the project anywhere and run with dynamic paths.

## 1) Set Project Root

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"
```

Or set it manually:

```bash
export PROJECT_ROOT=/path/to/t-car
cd "$PROJECT_ROOT"
```

## 2) Create Venv

```bash
bash "$PROJECT_ROOT/tools/setup_stage2_venv.sh"
source "$PROJECT_ROOT/.venv/bin/activate"
```

## 3) Path Variables

```bash
TOOLS_DIR="$PROJECT_ROOT/tools"
STAGE2_DIR="$TOOLS_DIR/stage2"
DATA_DIR="$PROJECT_ROOT/data"
RUNS_DIR="$TOOLS_DIR/runs"
```

## 4) Stage1 Detect (Optional)

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 yolo detect train \
  data="$DATA_DIR/yaml/traffic_stage1.yaml" \
  model="$TOOLS_DIR/yolo11s.pt" \
  imgsz=640 \
  epochs=200 \
  batch=256 \
  device=0,1,2,3 \
  workers=20 \
  cache=disk \
  amp=True \
  name=traffic_stage1
```

Validation inference:

```bash
yolo detect predict \
  model="$RUNS_DIR/detect/traffic_stage14/weights/best.pt" \
  source="$DATA_DIR/yolo/images/val" \
  imgsz=640 \
  conf=0.25 \
  device=0 \
  name=traffic_stage1_val
```

## 5) Stage2 Dataset (GT Box Direct)

Train split:

```bash
python3 "$STAGE2_DIR/stage2_build_dataset.py" \
  --box-source gt \
  --yolo-image-dir "$DATA_DIR/yolo/images/train" \
  --raw-json-root "$DATA_DIR/raw" \
  --split train \
  --out-crops-root "$DATA_DIR/stage2/crops" \
  --out-meta "$DATA_DIR/stage2/meta/train.csv" \
  --summary-json "$DATA_DIR/stage2/meta/train_summary.json" \
  --padding-ratio 0.1 \
  --min-aspect-ratio 2.0
```

Val split:

```bash
python3 "$STAGE2_DIR/stage2_build_dataset.py" \
  --box-source gt \
  --yolo-image-dir "$DATA_DIR/yolo/images/val" \
  --raw-json-root "$DATA_DIR/raw" \
  --split val \
  --out-crops-root "$DATA_DIR/stage2/crops" \
  --out-meta "$DATA_DIR/stage2/meta/val.csv" \
  --summary-json "$DATA_DIR/stage2/meta/val_summary.json" \
  --padding-ratio 0.1 \
  --min-aspect-ratio 2.0
```

## 6) Stage2 Train (MobileNetV3-Large)

```bash
python3 "$STAGE2_DIR/stage2_train_mobilenet.py" \
  --train-csv "$DATA_DIR/stage2/meta/train.csv" \
  --val-csv "$DATA_DIR/stage2/meta/val.csv" \
  --num-classes 4 \
  --backbone mobilenet_v3_large \
  --epochs 200 \
  --batch-size 256 \
  --lr 3e-4 \
  --patience 200 \
  --device cuda:0,1,2,3 \
  --out-dir "$RUNS_DIR/classify/traffic_stage2_mnv3s"
```

## 7) Final 2-Stage Inference

```bash
python3 "$STAGE2_DIR/stage2_infer.py" \
  --stage1-weights "$RUNS_DIR/detect/traffic_stage14/weights/best.pt" \
  --stage2-weights "$RUNS_DIR/classify/traffic_stage2_mnv3s/weights/best.pth" \
  --source "$DATA_DIR/yolo/images/val" \
  --out-json-dir "$RUNS_DIR/two_stage_infer/json" \
  --out-vis-dir "$RUNS_DIR/two_stage_infer/vis" \
  --conf 0.40 \
  --min-aspect-ratio 2.0 \
  --padding-ratio 0.1 \
  --imgsz 640 \
  --device 0 \
  --cls-device cuda:0
```

Outputs:
- JSON: `$RUNS_DIR/two_stage_infer/json`
- Visualization: `$RUNS_DIR/two_stage_infer/vis`
