# A1-P — H3 Attention Pooling 실험

## 실험 목적

선정된 H3 core에서 6-layer SPOTER class-query decoder를 제거하고 masked temporal attention pooling으로 교체하여 decoder의 필요성과 경량화 효과를 검증한다.

## 모델 구성

```text
H3 Part + All front-end
  → Squeezeformer-style Encoder × 6
  → Masked Temporal Attention Pooling
  → 300-class classifier
```

Padding frame은 pooling attention에서 제외한다. H3의 front-end와 encoder는 유지하고 sequence head만 변경했다.

## 공통 실험 조건

- 데이터: 300 classes, 총 13,494 samples
- Split: Train REAL01–06 / Validation REAL07–08 / Test REAL09
- 입력 차원 208, 최대 sequence length 256
- AdamW, learning rate `1e-4`, batch size 32, 최대 100 epochs
- Architecture 선택은 validation Macro Top-1만 사용하고 REAL09 test는 최종 선택 후 한 번 평가했다.

## Decoder ablation 결과

| Head | Validation Macro Top-1 | Parameters | Best checkpoint |
|---|---:|---:|---:|
| H3 Class-query Decoder × 6 | 92.23% | 12,689,196 | 152.6 MB |
| H3 Attention Pooling | **92.77%** | **6,400,813** | **77.0 MB** |

Attention pooling은 validation 정확도를 **0.53%p 높이면서** 파라미터와 checkpoint 크기를 약 **49.6% 줄였다**. 이 결과에 따라 A1-P를 300-class 최종 Hybrid로 선정했다.

## 최종 REAL09 test

| 지표 | 결과 |
|---|---:|
| Micro Top-1 | 97.80% |
| Macro Top-1 | **97.80%** |
| Top-3 | 99.53% |
| Top-5 | 99.60% |
| Macro F1 | 97.58% |

Camera별 Top-1은 D 97.00%, F 98.33%, L 97.67%, R 98.33%, U 97.67%였다.

최종 A1-P는 validation에서는 E0 baseline보다 높았지만, REAL09 test에서는 E0의 98.33%보다 **0.53%p 낮았다**. 따라서 경량화 효과는 명확하지만 unseen actor 정확도의 우위는 추가 holdout actor로 재검증해야 한다.

## 검증 및 주의사항

- Validation temperature scaling: `T=1.950361`
- Validation NLL: 0.738696 → 0.476328
- TorchScript는 길이 64/128/187/250에서 PyTorch와 logits 및 Top-k가 일치했다.
- 32-sample overfit 결과는 75%로 계획 기준 95%를 충족하지 못했다.
- A1-P는 seed 42만 수행됐으므로 최종 head의 seed 반복 검증이 필요하다.

## 주요 파일

- 전체 결과 보고서: `REPORT.md`
- 설정: `configs/pilot300_a1_pooling_seed42.yaml`
- 최종 checkpoint: `runs/pilot300_a1_h3_attention_pooling_seed42_train/best.pt`
- Test 지표: `runs/pilot300_a1_h3_attention_pooling_seed42_train/test_metrics.json`
- TorchScript: `runs/pilot300_a1_h3_attention_pooling_seed42_train/model_torchscript.pt`
