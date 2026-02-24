# T-CAR Tools

This repository contains stage1 detection and stage2 classification commands for traffic-light recognition.

## Environment

Create and activate venv:

```bash
bash /data4/dongmin/t-car/tools/setup_stage2_venv.sh
source /data4/dongmin/t-car/.venv/bin/activate
```

## Stage1 (YOLO Detect)

Train stage1 detector:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 yolo detect train \
  data=/data4/dongmin/t-car/data/yaml/traffic_stage1.yaml \
  model=/data4/dongmin/t-car/tools/yolo11s.pt \
  imgsz=640 \
  epochs=200 \
  batch=256 \
  device=0,1,2,3 \
  workers=20 \
  cache=disk \
  amp=True \
  name=traffic_stage1
```

Run validation inference with stage1:

```bash
yolo detect predict \
  model=runs/detect/traffic_stage14/weights/best.pt \
  source=/data4/dongmin/t-car/data/yolo/images/val \
  imgsz=640 \
  conf=0.25 \
  device=0 \
  name=traffic_stage1_val
```

## Stage2 Dataset (GT Crop)

Stage2 training dataset uses GT traffic-light boxes directly (`--box-source gt`), not stage1 prediction boxes.

Build train split:

```bash
python3 tools/stage2_build_dataset.py \
  --box-source gt \
  --yolo-image-dir /data4/dongmin/t-car/data/yolo/images/train \
  --raw-json-root /data4/dongmin/t-car/data/raw \
  --split train \
  --out-crops-root /data4/dongmin/t-car/data/stage2/crops \
  --out-meta /data4/dongmin/t-car/data/stage2/meta/train.csv \
  --summary-json /data4/dongmin/t-car/data/stage2/meta/train_summary.json \
  --padding-ratio 0.1 \
  --min-aspect-ratio 2.0
```

Build val split:

```bash
python3 tools/stage2_build_dataset.py \
  --box-source gt \
  --yolo-image-dir /data4/dongmin/t-car/data/yolo/images/val \
  --raw-json-root /data4/dongmin/t-car/data/raw \
  --split val \
  --out-crops-root /data4/dongmin/t-car/data/stage2/crops \
  --out-meta /data4/dongmin/t-car/data/stage2/meta/val.csv \
  --summary-json /data4/dongmin/t-car/data/stage2/meta/val_summary.json \
  --padding-ratio 0.1 \
  --min-aspect-ratio 2.0
```

## Stage2 Train (MobileNetV3-Large)

```bash
python3 tools/stage2_train_mobilenet.py \
  --train-csv /data4/dongmin/t-car/data/stage2/meta/train.csv \
  --val-csv /data4/dongmin/t-car/data/stage2/meta/val.csv \
  --num-classes 4 --backbone mobilenet_v3_large --epochs 200 --batch-size 256 --lr 3e-4 \
  --patience 200 \
  --device cuda:0,1,2,3 \
  --out-dir /data4/dongmin/t-car/tools/runs/classify/traffic_stage2_mnv3s
```

## Final 2-Stage Inference

```bash
python3 tools/stage2_infer.py \
  --stage1-weights /data4/dongmin/t-car/tools/runs/detect/traffic_stage14/weights/best.pt \
  --stage2-weights /data4/dongmin/t-car/tools/runs/classify/traffic_stage2_mnv3s/weights/best.pth \
  --source /data4/dongmin/t-car/data/yolo/images/val \
  --out-json-dir /data4/dongmin/t-car/tools/runs/two_stage_infer/json \
  --out-vis-dir /data4/dongmin/t-car/tools/runs/two_stage_infer/vis \
  --conf 0.40 --min-aspect-ratio 2.0 --padding-ratio 0.1 --imgsz 640 \
  --device 0 --cls-device cuda:0
```

Outputs:
- JSON: `tools/runs/two_stage_infer/json`
- Visualization: `tools/runs/two_stage_infer/vis`
