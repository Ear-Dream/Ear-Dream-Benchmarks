# AIHUB One-Hand Sign Candidate: Raw-to-Model Baseline

한쪽 손만 관찰됐을 때도 원래 300개 수어 단어 중 관련 후보 Top-K를 반환하는
기본 모델 프로젝트다. 상위 계획 문서
`AIHUB_ONE_HAND_SIGN_CANDIDATE_RAW_TO_MODEL_TRAINING_PLAN.md`를 구현했다.

이 프로젝트는 후속 프로젝트 `one_hand_hybrid_classification_retrieval`의 비교
기준 B0다. 현재 최종 권장 모델은 후속 hybrid 모델이며, 이 폴더의 checkpoint와
결과는 기준선 재현과 ablation을 위해 보존한다.

## 핵심 목표

- FULL 입력으로 300개 수어 단어 분류
- RIGHT_ONLY/LEFT_ONLY 입력으로 원래 단어의 Top-K 후보 검색
- MediaPipe 검출 상태인 `detected_mask`와 의도적으로 관찰할 손을 지정하는
  `view_mask`를 분리
- 기존 `spoter2_mp_xy_v1` 208D 특징을 변경하지 않고 재사용

## 데이터와 split

```text
입력 특징 [T,208]
├─ Pose        [0:50]    50D
├─ Right hand [50:92]    42D
├─ Left hand  [92:134]   42D
└─ Face       [134:208]  74D
```

- Train: REAL01~06
- Validation: REAL07~08
- Test: REAL09
- 전체 검증 샘플: 13,494개
- 데이터 schema 검사 결과: 오류 0개

현재 workspace에는 원본 MediaPipe JSON 전체가 없으므로 기본 실행 경로는 기존
H5의 208D 특징과 `part_mask[:,1:3]`을 그대로 사용한다. 원본 JSON이 연결되면
`index_onehand_raw.py`로 schema와 checksum을 감사할 수 있다.

## 모델 구조

```text
Pose/Right/Left/Face별 Conv1D Part branch
                 +
전체 208D Conv1D All branch
                 ↓
Mask-aware Part/All gate
                 ↓
Squeezeformer-style Encoder × 6
                 ↓
Masked Temporal Attention Pooling (256D)
       ├─ 300-class classifier
       └─ L2-normalized 128D retrieval embedding
```

학습 loss:

```text
FULL 300-class CE
+ FULL/partial embedding alignment
+ same-word supervised contrastive loss
```

Train split embedding으로 FULL/RIGHT/LEFT 클래스 prototype을 만들고 cosine
similarity로 후보 순위를 계산한다.

## 최종 B0 성능

REAL09 test:

| 입력 | 분류 Top-1 | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| FULL | 96.73% | 96.13% | 99.27% | 99.53% | 97.48% |
| RIGHT_ONLY | 95.67% | 95.00% | 98.60% | 98.80% | 96.50% |
| LEFT_ONLY | 73.07% | 76.53% | 88.13% | 92.33% | 81.84% |

95% precision validation calibration:

- RIGHT_ONLY: threshold `0.8326`, coverage 83%
- LEFT_ONLY: threshold `0.9734`, coverage 22%

## 실행 순서

프로젝트 폴더에서 실행한다.

```powershell
python scripts/build_onehand_data.py --config configs/pilot300_onehand_m2_mask_gated.yaml
python scripts/validate_onehand_data.py --config configs/pilot300_onehand_m2_mask_gated.yaml
python -m pytest -q

python scripts/train_onehand.py `
  --config configs/pilot300_onehand_m2_mask_gated.yaml `
  --mode overfit32

python scripts/train_onehand.py `
  --config configs/pilot300_onehand_m2_mask_gated.yaml `
  --mode train

python scripts/build_onehand_prototypes.py `
  --config configs/pilot300_onehand_m2_mask_gated.yaml `
  --checkpoint runs/pilot300_onehand_m2_mask_gated_train/best.pt

python scripts/evaluate_onehand.py `
  --config configs/pilot300_onehand_m2_mask_gated.yaml `
  --checkpoint runs/pilot300_onehand_m2_mask_gated_train/best.pt `
  --prototypes runs/pilot300_onehand_m2_mask_gated_train/prototypes.npz `
  --split test --modes full right_only left_only

python scripts/calibrate_onehand.py `
  --config configs/pilot300_onehand_m2_mask_gated.yaml `
  --checkpoint runs/pilot300_onehand_m2_mask_gated_train/best.pt `
  --prototypes runs/pilot300_onehand_m2_mask_gated_train/prototypes.npz

python scripts/export_onehand.py `
  --config configs/pilot300_onehand_m2_mask_gated.yaml `
  --checkpoint runs/pilot300_onehand_m2_mask_gated_train/best.pt `
  --prototypes runs/pilot300_onehand_m2_mask_gated_train/prototypes.npz
```

CUDA 학습에는 이 환경의 Python 3.12 CUDA PyTorch를 사용했다.

## 주요 산출물

`runs/pilot300_onehand_m2_mask_gated_train/`:

- `best.pt`: 최고 학습 checkpoint
- `prototypes.npz`: FULL/RIGHT/LEFT 300-class prototype
- `test_onehand_metrics.json`: REAL09 결과
- `calibration.json`: validation threshold/coverage curve
- `onehand_model_torchscript.pt`: 배포 모델
- `export_parity.json`: PyTorch/TorchScript 수치 비교

TorchScript 최대 출력 오차는 `3.58e-6`이고 단위 테스트는 3개 모두 통과했다.

## 한계와 후속 프로젝트

- LEFT_ONLY가 RIGHT_ONLY보다 크게 낮다.
- 한손/양손 단어 구조를 명시적으로 사용하지 않는다.
- 해부학적 R/L hand branch가 별도 파라미터다.

이 문제를 개선한 후속 모델은
`../one_hand_hybrid_classification_retrieval`에 있다.
