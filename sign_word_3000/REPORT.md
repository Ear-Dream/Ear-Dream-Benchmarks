# 3,000-word SPOTER 학습 결과

## 격리와 데이터

- 전용 workspace: `sign_word_3000`.
- `mediapipe_processed/staging/REAL01~09.h5`는 읽기 전용으로만 사용했다.
- Train REAL01~06 / Validation REAL07~08 / Test REAL09.
- Train 89,958 / Validation 29,993 / Test 15,000, 총 134,951개.
- 예상 135,000개 중 누락 49개는 `data/missing_samples.json`에 기록했다.
- 모든 3,000개 클래스가 Train/Validation/Test 각각에 존재한다.
- H5 134,951개 group 전수검사 통과, 오류 0개.
- 길이 min 56 / median 124 / p95 172 / p99 200 / max 268.
- max length 256을 초과하는 극소수 시퀀스만 전체 구간 uniform sampling했다.

## 모델과 학습

- SPOTER-208 Base: 208→256 projection, Encoder 6층, Class Query Decoder 6층.
- 완료된 300-word `best.pt`에서 classifier를 제외한 192개 tensor를 전이했다.
- 3,000-class classifier는 새로 초기화하고 전체 모델을 fine-tuning했다.
- AdamW 5e-5, warmup 5 + cosine, FP16, batch 64, gradient clipping 1.0.
- 100 epochs 완료. checkpoint 재개 및 완료 후 no-op 재실행을 검증했다.
- 최고 Validation Macro Top-1: **92.79% (epoch 98)**.

## 보존 Test 결과 — REAL09

- Macro Top-1: **97.04%**
- Micro Top-1: **97.04%**
- Top-3: **99.28%**
- Top-5: **99.52%**
- Macro F1: **96.68%**
- camera D/F/L/R/U: 97.50 / 97.90 / 96.53 / 96.30 / 96.97%

REAL09는 모델·epoch·threshold 선택에 사용하지 않은 actor-independent holdout이다.

## Calibration과 export

- Validation temperature scaling: T=1.65116, NLL 0.63355 → 0.47144.
- threshold sweep는 `runs/.../calibration.json`에 저장했다.
- TorchScript export 완료.
- 길이 64/128/187/250에서 PyTorch 대비 최대 logit 차이 0,
  Top-1과 Top-5가 모두 일치했다.

## 참고

이번 run은 300-word encoder/decoder transfer 경로다. 동일 예산의 3,000-class
from-scratch 비교는 별도 실험이며 현재 결과 파일과 섞이지 않도록 새 run 이름을
사용해야 한다.
