# T-CAR Tools

AIHub 교통 데이터 기반의 신호등 인식 2-stage 파이프라인 학습/추론 도구 모음입니다.

## 프로젝트 설명

본 프로젝트는 한 번에 색상까지 직접 예측하지 않고, 아래 2단계로 분리해 정확도와 운영 안정성을 확보합니다.

1. Stage1 (Detection)
- 입력 이미지에서 `traffic_light` 객체 위치를 YOLO로 검출합니다.
- 출력은 bounding box + detection confidence 입니다.

2. Stage2 (Color Classification)
- Stage1에서 얻은 bbox 영역(또는 GT bbox)을 crop한 뒤 분류기로 색상을 판별합니다.
- 출력 클래스는 `red`, `yellow`, `green`, `off` 4종입니다.

## 데이터셋

- 출처: AIHub `신호등/도로표지판 인지 영상(수도권)`
- 원본: 이미지 + JSON annotation
- Stage1 타깃: `traffic_light` 객체 검출
- Stage2 타깃: `traffic_light(type=car)` 색상 분류

Stage2 라벨은 JSON의 신호등 상태 attribute를 파싱하여 매핑합니다. 복합 점등(`red+yellow` 등)은 드롭하고, 단일 색만 학습 샘플로 사용하거나 상태 없음은 `off`로 처리합니다.

## 모델 아키텍처

### Stage1

- 프레임워크: Ultralytics YOLO
- 기본 모델 파일: `tools/yolo11s.pt`
- 학습 결과 기본 경로: `tools/runs/traffic_stage1`
- 핵심 산출물: `weights/best.pt`, `results.csv`, 각종 곡선/행렬 이미지

### Stage2

- 프레임워크: PyTorch + torchvision
- 백본(기본): `MobileNetV3-Large`
- 옵션 백본: `MobileNetV3-Small`
- 입력 크기: `224x224`
- 학습 결과 기본 경로: `tools/runs/traffic_stage2`
- 핵심 산출물: `weights/best.pth`, `results.csv`, `confusion_matrix*.csv`, 시각화 그래프

## 디렉토리 구조

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
  commands
  setup_stage2_venv.sh
  requirements.txt
```

데이터 기본 경로는 프로젝트 루트 기준 아래를 사용합니다.

```text
data/
  raw/                  # 원본 json
  yolo/images/{train,val}
  stage2/
    preds/
    crops/
    meta/
```

## 환경 구성

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"

bash tools/setup_stage2_venv.sh
source "$PROJECT_ROOT/.venv/bin/activate"
```

## 명령 진입점 (`tools/commands`)

`tools/commands`는 새 구조(`stage1`, `stage2`, `runs/traffic_stage1`, `runs/traffic_stage2`)를 기준으로 기본 경로를 자동 해석합니다.

```bash
bash tools/commands --help
bash tools/commands paths
```

주요 서브커맨드:

- `stage1-train`: Stage1 YOLO 학습
- `stage1-predict`: Stage1 예측
- `build-train`, `build-val`, `build-all`: Stage2 학습용 crop/csv 생성
- `train-stage2`: Stage2 분류 학습
- `infer-stage2`: 최종 2-stage 추론(JSON + 시각화)

## 각 스크립트 `--help` 지원

모든 핵심 스크립트는 argparse 기반으로 기본값과 설명을 출력합니다.

```bash
python tools/stage1/conver_to_yolo.py --help
python tools/stage2/stage2_generate_preds.py --help
python tools/stage2/stage2_build_dataset.py --help
python tools/stage2/stage2_train_mobilenet.py --help
python tools/stage2/stage2_infer.py --help
```

경로 인자를 생략하면 프로젝트 루트 기준 기본 경로를 사용하도록 구성되어 있습니다.

## 기본 파이프라인

1. Stage1 학습(필요 시)

```bash
bash tools/commands stage1-train
```

2. Stage2 데이터셋 생성

```bash
bash tools/commands build-all
```

3. Stage2 학습

```bash
bash tools/commands train-stage2
```

4. 최종 2-stage 추론

```bash
bash tools/commands infer-stage2
```

## 산출물 위치

- Stage1 best weight: `tools/runs/traffic_stage1/weights/best.pt`
- Stage2 best weight: `tools/runs/traffic_stage2/weights/best.pth`
- Inference JSON: `tools/runs/traffic_stage2/infer/json`
- Inference 시각화: `tools/runs/traffic_stage2/infer/vis`
