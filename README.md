# T-CAR Tools

Tooling for training and inference of a 2-stage traffic-light recognition pipeline built on AIHub traffic data.

## Project Overview

This project separates detection and color recognition into two stages instead of predicting color directly in one step.
The goal is better accuracy and easier control in production.

1. Stage1 (Detection)
- Detects `traffic_light` object locations in input images using YOLO.
- Output: bounding boxes + detection confidence.

2. Stage2 (Color Classification)
- Crops each detected box region (or GT box) and classifies traffic-light color.
- Output classes: `red`, `yellow`, `green`, `off`.

## Dataset

- Source: AIHub `Traffic Light / Road Sign Recognition Video (Capital Area)`
- Raw format: images + JSON annotations
- Stage1 target: `traffic_light` object detection
- Stage2 target: `traffic_light(type=car)` color classification

Stage2 labels are generated from JSON traffic-light state attributes.
Composite states (for example, `red+yellow`) are dropped, and no-active-light cases are mapped to `off`.

## Model Architecture

### Stage1

- Framework: Ultralytics YOLO
- Base model file: `tools/yolo11s.pt`
- Default output path: `tools/runs/traffic_stage1`
- Canonical weight path: `tools/weights/stage1_best.pt`
- Main artifacts: `weights/best.pt`, `results.csv`, curve/matrix images

### Stage2

- Framework: PyTorch + torchvision
- Default backbone: `MobileNetV3-Large`
- Optional backbone: `MobileNetV3-Small`
- Input size: `224x224`
- Default output path: `tools/runs/traffic_stage2`
- Canonical weight path: `tools/weights/stage2_best.pth`
- Main artifacts: `weights/best.pth`, `results.csv`, `confusion_matrix*.csv`, plots

## Directory Structure

```text
tools/
  stage1/
    conver_to_yolo.py
  stage2/
    stage2_generate_preds.py
    stage2_build_dataset.py
    stage2_train_mobilenet.py
    stage2_infer.py
    stage2_utils.py
  runs/
    traffic_stage1/
    traffic_stage2/
  weights/
    stage1_best.pt
    stage2_best.pth
  commands
  setup_stage2_venv.sh
  requirements.txt
```

Default data locations (relative to project root):

```text
data/
  raw/                  # raw JSON annotations
  yolo/images/{train,val}
  stage2/
    preds/
    crops/
    meta/
```

## Environment Setup

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"

bash tools/setup_stage2_venv.sh
source "$PROJECT_ROOT/.venv/bin/activate"
```

## Command Entry (`tools/commands`)

`tools/commands` resolves default paths using the current structure (`stage1`, `stage2`, `runs/traffic_stage1`, `runs/traffic_stage2`).

```bash
bash tools/commands --help
bash tools/commands paths
```

Main subcommands:

- `sync-weights`: copy `best.pt`/`best.pth` from run dirs into `tools/weights`
- `stage1-train`: train Stage1 YOLO detector
- `stage1-predict`: run Stage1 prediction
- `build-train`, `build-val`, `build-all`: build Stage2 crop/CSV dataset
- `train-stage2`: train Stage2 classifier
- `infer-stage2`: run final 2-stage inference (JSON + visualization)

## Script `--help`

All key scripts expose argparse help with descriptions and defaults.

```bash
python tools/stage1/conver_to_yolo.py --help
python tools/stage2/stage2_generate_preds.py --help
python tools/stage2/stage2_build_dataset.py --help
python tools/stage2/stage2_train_mobilenet.py --help
python tools/stage2/stage2_infer.py --help
```

If path arguments are omitted, project-root-based defaults are used.

## Standard Pipeline

1. Train Stage1 (if needed)

```bash
bash tools/commands stage1-train
```

2. Build Stage2 dataset

```bash
bash tools/commands build-all
```

3. Train Stage2

```bash
bash tools/commands train-stage2
```

4. Run final 2-stage inference

```bash
bash tools/commands infer-stage2
```

## Output Paths

- Stage1 best weight (canonical): `tools/weights/stage1_best.pt`
- Stage2 best weight (canonical): `tools/weights/stage2_best.pth`
- Stage1 run checkpoint: `tools/runs/traffic_stage1/weights/best.pt`
- Stage2 run checkpoint: `tools/runs/traffic_stage2/weights/best.pth`
- Inference JSON: `tools/runs/traffic_stage2/infer/json`
- Inference visualization: `tools/runs/traffic_stage2/infer/vis`
