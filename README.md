# AI Hub 수어 단어 인식 모델 벤치마크

이 저장소는 AI Hub 수어 단어 landmark sequence를 이용해 여러 모델 구조를 동일한 조건에서 학습·비교한 벤치마크다. 300-class pilot에서 SPOTER baseline부터 body-part front-end, Squeezeformer-style encoder, global branch와 attention pooling까지 단계적으로 비교했으며, 별도로 3,000-class 전이학습 결과를 포함한다.

## 벤치마크 구성

```text
sign_word_300            E0: 300-class SPOTER-208 baseline
sign_word_h1             H1: Body-part Conv front-end
sign_word_h2             H2: H1 + Squeezeformer-style encoder
sign_word_h3             H3: H2 + 전체 208D All branch
sign_word_a1_pooling     A1-P: H3에서 decoder 제거 + attention pooling
sign_word_3000           3,000-class SPOTER transfer learning
```

300-class 실험은 한 번에 하나의 구성 요소를 변경해 성능 변화의 원인을 분리하도록 설계했다.

```text
E0 SPOTER baseline
  └─ H1: 입력 projection을 신체 부위별 Conv1D로 변경
       └─ H2: Transformer를 Squeezeformer-style encoder로 변경
            └─ H3: 전체 208D 특징을 처리하는 All branch 추가
                 └─ A1-P: 6-layer class-query decoder를 attention pooling으로 변경
```

## 공통 학습 조건

300-class core benchmark에는 다음 조건을 공통으로 적용했다.

| 항목 | 설정 |
|---|---|
| Task | Isolated sign-word classification |
| 입력 | MediaPipe 기반 landmark features `[T, 208]` |
| Feature version | `spoter2_mp_xy_v1` |
| 클래스 | 300 |
| 샘플 | Train 8,994 / Validation 3,000 / Test 1,500 |
| Actor split | Train REAL01–06 / Validation REAL07–08 / Test REAL09 |
| 최대 sequence length | 256 |
| 모델 차원 | 256 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Batch size | 32 |
| Training budget | 최대 100 epochs, warmup 5 epochs, cosine schedule |
| 기타 | FP16, gradient clipping 1.0, early stopping |
| 선택 지표 | Validation Macro Top-1 |

모델 선택에는 validation 결과만 사용했으며 REAL09 test는 최종 선택 후 평가했다.

## 300-class 모델 결과

### Seed 42 architecture screening

| 모델 | 핵심 구조 | Validation Macro Top-1 | 이전 단계 대비 |
|---|---|---:|---:|
| E0 | Linear front-end + Transformer + Query Decoder | 91.77% | — |
| H1 | Body-part Conv + Transformer + Query Decoder | 91.87% | +0.10%p |
| H2 | Body-part Conv + Squeezeformer + Query Decoder | 92.20% | +0.33%p |
| H3 | Part+All Conv + Squeezeformer + Query Decoder | 92.23% | +0.03%p |
| A1-P | H3 core + Attention Pooling | **92.77%** | **+0.53%p** |

Body-part front-end와 All branch의 단독 개선 폭은 작았다. Squeezeformer-style encoder는 seed 42에서 개선됐지만 반복 seed 변동이 컸다. 가장 뚜렷한 결과는 H3의 무거운 class-query decoder를 attention pooling으로 교체했을 때 나타났다.

### 반복 seed 비교

| 모델 | seed 42 | seed 123 | seed 2026 | 평균 ± sample SD |
|---|---:|---:|---:|---:|
| E0 | 91.77% | 91.33% | 91.93% | 91.68 ± 0.31% |
| H2 | 92.20% | 90.80% | 91.40% | 91.47 ± 0.70% |
| H3 | 92.23% | 92.33% | 92.20% | **92.26 ± 0.07%** |

H3가 가장 높은 평균과 가장 낮은 seed 편차를 기록해 최종 core architecture로 선택됐다. H1은 screening만 수행했으며 A1-P도 현재 seed 42 결과만 존재한다.

### Decoder ablation 및 경량화

| Head | Validation Macro Top-1 | Parameters | Best checkpoint |
|---|---:|---:|---:|
| H3 Class-query Decoder × 6 | 92.23% | 12,689,196 | 152.6 MB |
| H3 Attention Pooling | **92.77%** | **6,400,813** | **77.0 MB** |

A1-P는 H3 decoder 모델보다 validation 성능이 0.53%p 높았고 파라미터와 checkpoint 크기는 약 49.6% 감소했다. 이에 따라 A1-P를 300-class 최종 Hybrid 모델로 선정했다.

### 최종 REAL09 test

| 지표 | E0 baseline | 최종 A1-P |
|---|---:|---:|
| Macro Top-1 | **98.33%** | 97.80% |
| Micro Top-1 | **98.33%** | 97.80% |
| Top-3 | **99.87%** | 99.53% |
| Top-5 | **100.00%** | 99.60% |
| Macro F1 | **98.16%** | 97.58% |

A1-P는 validation과 모델 효율에서는 우수했지만 REAL09 test에서는 E0보다 Macro Top-1이 0.53%p 낮았다. 따라서 경량화 이점은 확인됐으나 unseen actor 정확도의 우위는 추가 holdout actor 평가가 필요하다.

## 3,000-class 전이학습

`sign_word_3000`은 300-class E0 SPOTER checkpoint에서 classifier를 제외한 가중치를 불러와 3,000-class classifier를 새로 학습한 실험이다. 최종 A1-P의 3,000-class 확장이 아니므로 300-class architecture benchmark와 직접적인 구조 비교 대상으로 사용하지 않는다.

| 항목 | 결과 |
|---|---:|
| 총 샘플 | 134,951 |
| Train / Validation / Test | 89,958 / 29,993 / 15,000 |
| Best Validation Macro Top-1 | **92.79%** |
| REAL09 Test Macro Top-1 | **97.04%** |
| Test Top-3 | 99.28% |
| Test Top-5 | 99.52% |
| Test Macro F1 | 96.68% |

이 결과는 300-class representation을 3,000 classes로 전이할 수 있음을 보여준다. 동일 예산의 from-scratch 비교와 H3/A1-P 기반 3,000-class 학습은 아직 포함하지 않는다.

## 디렉터리별 역할

| 디렉터리 | 역할 | 주요 결과 |
|---|---|---|
| [`sign_word_300`](sign_word_300/) | E0 baseline, 3-seed 학습, 최종 test/calibration/export | Val 91.68 ± 0.31%, Test 98.33% |
| [`sign_word_h1`](sign_word_h1/) | Body-part Conv front-end screening | Val 91.87% |
| [`sign_word_h2`](sign_word_h2/) | Squeezeformer-style encoder 비교, 3 seeds | Val 91.47 ± 0.70% |
| [`sign_word_h3`](sign_word_h3/) | All branch 효과 및 core 선택, 3 seeds | Val 92.26 ± 0.07% |
| [`sign_word_a1_pooling`](sign_word_a1_pooling/) | Decoder ablation, 최종 300-class Hybrid | Val 92.77%, Test 97.80% |
| [`sign_word_3000`](sign_word_3000/) | E0 기반 3,000-class transfer learning | Val 92.79%, Test 97.04% |

각 workspace는 독립적인 `configs/`, `src/`, `scripts/`, `tests/`, `data/`, `runs/`를 가진다. 실행 설정은 `configs/`, 학습 및 평가 진입점은 `scripts/`, checkpoint와 지표는 `runs/`에서 확인할 수 있다.

## 결과 파일 규칙

대표적인 학습 run에는 다음 산출물이 저장된다.

```text
runs/<experiment_name>/
├─ best.pt                  validation 기준 최고 checkpoint
├─ last.pt                  마지막 checkpoint
├─ history.csv              epoch별 학습 기록
├─ metrics.json             최고 validation 지표
├─ test_metrics.json        최종 test 지표
├─ predictions_test.csv     test 예측 결과
├─ per_class_metrics.csv    클래스별 성능
├─ calibration.json         temperature scaling 결과
└─ model_torchscript.pt     export된 TorchScript 모델
```

일부 screening run에는 최종 test, calibration 또는 export 산출물이 없을 수 있다. 상세한 실험별 구성과 해석은 각 디렉터리의 README 또는 `REPORT.md`를 참고한다.

## 해석 시 주의사항

- 300-class 모델 간 공식 비교 지표는 validation Macro Top-1이다.
- REAL09 test 결과를 보고 architecture를 다시 선택하면 test leakage가 되므로 별도 holdout 없이 모델을 재선정해서는 안 된다.
- A1-P의 32-sample overfit 결과는 75%로 계획 기준 95%를 충족하지 못했다.
- A1-P는 단일 seed이므로 최종 구조의 반복 seed 검증이 추가로 필요하다.
- 3,000-class 결과는 E0 transfer 모델이며 300-class A1-P와 동일 구조가 아니다.
