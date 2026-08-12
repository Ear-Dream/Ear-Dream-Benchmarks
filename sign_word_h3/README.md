# H3 — Part + All Branch Hybrid 실험

## 실험 목적

H2의 신체 부위별 경로에 전체 208차원 특징을 처리하는 All branch를 추가했을 때 전신 관계 정보가 성능과 seed 안정성을 개선하는지 확인한다.

## 모델 구성

```text
[Part path]
Pose / Right hand / Left hand / Face → 부위별 Conv1D → concat/project

[All path]
전체 208D features → Conv1D(208 → 256)

[Fusion]
LayerNorm(Part path + All path)
  → Squeezeformer-style Encoder × 6
  → SPOTER Class-query Decoder × 6
  → 300-class classifier
```

H2 대비 변경 요소는 전체 특징을 병렬 처리하는 All branch와 residual-add fusion이다.

## 공통 실험 조건

- 데이터: 300 classes, 총 13,494 samples
- Split: Train REAL01–06 / Validation REAL07–08 / Test REAL09
- 입력 차원 208, 최대 sequence length 256
- `d_model=256`, attention head 8, encoder/decoder 각 6 layers
- AdamW, learning rate `1e-4`, batch size 32, 최대 100 epochs
- 주 평가 지표: Validation Macro Top-1

## 결과

| 실행 | Validation Macro Top-1 |
|---|---:|
| 32-sample overfit | 100.00% |
| 10-class smoke | 80.00% |
| seed 42 | 92.23% |
| seed 123 | **92.33%** |
| seed 2026 | 92.20% |
| 3-seed 평균 ± sample SD | **92.26 ± 0.07%** |

Seed 42 기준 H2 대비 향상은 **0.03%p**로 매우 작았다. 그러나 3-seed 평균이 비교 모델 중 가장 높고 편차가 가장 작아, All branch의 주된 이점은 최고점 상승보다 **학습 안정성 개선**으로 해석할 수 있다. 이 결과에 따라 H3를 core architecture로 선정하고 A1 head ablation을 수행했다.

## 주요 파일

- 설정: `configs/pilot300_h3_seed42.yaml`, `pilot300_h3_seed123.yaml`, `pilot300_h3_seed2026.yaml`
- 실행 결과: `runs/pilot300_h3_part_all_squeezeformer_decoder_seed*_train/`
- 각 실행의 `metrics.json`에 최고 validation 점수가 저장돼 있다.

