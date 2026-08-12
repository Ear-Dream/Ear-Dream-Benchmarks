# H2 — Part Conv + Squeezeformer-style Encoder 실험

## 실험 목적

H1의 body-part Conv front-end와 decoder를 유지한 채 Transformer Encoder를 temporal convolution이 포함된 Squeezeformer-style Encoder로 교체하여 encoder 변경 효과를 확인한다.

## 모델 구성

```text
208D features
  → Pose / Right hand / Left hand / Face별 Conv1D
  → concat + fusion projection
  → Squeezeformer-style Encoder × 6
     (Self-Attention + FFN + Temporal Conv)
  → SPOTER Class-query Decoder × 6
  → 300-class classifier
```

H1 대비 변경 요소는 encoder뿐이다. Front-end, decoder, classifier와 학습 조건은 동일하게 유지했다.

## 공통 실험 조건

- 데이터: 300 classes, 총 13,494 samples
- Split: Train REAL01–06 / Validation REAL07–08 / Test REAL09
- 입력 차원 208, 최대 sequence length 256
- `d_model=256`, attention head 8, encoder/decoder 각 6 layers
- convolution kernel size 15
- AdamW, learning rate `1e-4`, batch size 32, 최대 100 epochs
- 주 평가 지표: Validation Macro Top-1

## 결과

| 실행 | Validation Macro Top-1 |
|---|---:|
| 32-sample overfit | 100.00% |
| 10-class smoke | 86.00% |
| seed 42 | **92.20%** |
| seed 123 | 90.80% |
| seed 2026 | 91.40% |
| 3-seed 평균 ± sample SD | **91.47 ± 0.70%** |

Seed 42 screening에서는 H1보다 **0.33%p 향상**됐지만 seed 간 변동이 컸다. 3-seed 평균은 E0의 91.68%보다 0.21%p 낮아 Squeezeformer-style Encoder 자체의 안정적인 우위는 확인되지 않았다.

## 주요 파일

- 설정: `configs/pilot300_h2_seed42.yaml`, `pilot300_h2_seed123.yaml`, `pilot300_h2_seed2026.yaml`
- 실행 결과: `runs/pilot300_h2_part_squeezeformer_decoder_seed*_train/`
- 각 실행의 `metrics.json`에 최고 validation 점수가 저장돼 있다.

