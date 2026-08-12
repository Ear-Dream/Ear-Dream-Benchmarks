# H1 — Body-part Conv Front-end 실험

## 실험 목적

기존 E0 SPOTER-208 baseline의 단일 `Linear(208 → 256)` 입력 투영을 신체 부위별 Conv1D front-end로 교체했을 때의 효과를 확인한다. Encoder와 class-query decoder는 E0와 동일하게 유지하여 입력 front-end만 비교했다.

## 모델 구성

```text
features [B, L, 208]
  ├─ Pose       [0:50]    → Conv1D → 64D
  ├─ Right hand [50:92]   → Conv1D → 64D
  ├─ Left hand  [92:134]  → Conv1D → 64D
  └─ Face       [134:208] → Conv1D → 64D
          ↓ concat + fusion projection
Transformer Encoder × 6
SPOTER Class-query Decoder × 6
300-class classifier
```

- 각 신체 부위는 독립적인 Conv1D 파라미터를 사용한다.
- 입력 H5의 정규화된 208차원 특징은 다시 정규화하지 않는다.
- E0 대비 변경 요소는 body-part front-end뿐이다.

## 공통 실험 조건

- 데이터: 300 classes, 총 13,494 samples
- Split: Train REAL01–06 / Validation REAL07–08 / Test REAL09
- 입력: `features [T,208]`, 최대 길이 256
- 모델 차원 256, attention head 8
- AdamW, learning rate `1e-4`, batch size 32, 최대 100 epochs
- warmup 5 epochs, cosine schedule, FP16, gradient clipping 1.0
- 주 평가 지표: Validation Macro Top-1

## 결과

| 실행 | Validation Macro Top-1 |
|---|---:|
| 32-sample overfit | 100.00% |
| 10-class smoke | 87.00% |
| 300-class seed 42 | **91.87%** |

E0 seed 42의 91.77%보다 **0.10%p 향상**됐다. 따라서 신체 부위별 front-end의 효과는 양수였지만 크기는 매우 작았다. H1은 screening 실험으로만 수행했으며 반복 seed와 별도 test 평가는 진행하지 않았다.

## 주요 파일

- 설정: `configs/pilot300_h1_seed42.yaml`
- 학습 결과: `runs/pilot300_h1_part_transformer_decoder_seed42_train/`
- 지표: `runs/pilot300_h1_part_transformer_decoder_seed42_train/metrics.json`

