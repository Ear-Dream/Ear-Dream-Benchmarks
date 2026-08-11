# Pilot 300 결과 보고서

## 범위와 데이터 격리

- 선택 목록: `../일상_고빈도_핵심단어_300.csv`의 순서가 label 0~299를 결정한다.
- 원본: `../mediapipe_processed/staging/REAL01.h5` ~ `REAL09.h5`를 읽기 전용으로 참조한다.
- 원본 H5는 복사·수정·정규화 재적용하지 않았다.
- split: Train REAL01~06 / Validation REAL07~08 / Test REAL09.
- 샘플: Train 8,994 / Validation 3,000 / Test 1,500, 총 13,494.
- 누락 6개는 모두 Train split이며 `data/missing_samples.json`에 기록했다.
- 300개 클래스는 모든 split에 빠짐없이 존재한다.

## 데이터 검증

- 13,494개 group 전수 검사 통과, 오류 0개.
- feature dimension 208, feature version `spoter2_mp_xy_v1` 일치.
- NaN/Inf 없음, frame index 연속, part mask shape 일치.
- 길이: min 60 / median 120 / p95 162 / p99 187.07 / max 250.
- 따라서 `max_sequence_length=256`은 전체 선택 샘플을 손실 없이 포함한다.

## 검증 게이트

- Synthetic shape 및 padding invariance 통과.
- 32-sample overfit: train accuracy 100% (epoch 15).
- 10-class smoke: validation Top-1 88%, validation loss 1.81 → 0.40 (5 epochs).
- Checkpoint resume와 완료 후 no-op 재실행 검증.

## 모델 및 학습

- SPOTER-208 Base: 208→256 projection, Encoder 6층, Class Query Decoder 6층.
- AdamW, warmup 5 epochs + cosine schedule, FP16, gradient clipping 1.0.
- 최고 checkpoint: validation macro Top-1 91.77% (epoch 66).
- early stopping 완료. 이후 평가는 `best.pt`만 사용했다.

## 보존한 Test 결과 (REAL09)

- Macro Top-1: **98.33%**
- Micro Top-1: **98.33%**
- Top-3: **99.87%**
- Top-5: **100.00%**
- Macro F1: **98.16%**
- camera D/F/L/R/U: 98.0 / 98.0 / 98.33 / 99.0 / 98.33%

Test actor가 한 명뿐이므로 이 수치는 REAL09에 대한 내부 holdout 결과다. 향후
REAL10~16 또는 공식 외부 validation/test가 준비되면 일반화 성능을 다시 측정해야 한다.

## Calibration 및 export

- Validation temperature scaling: T=1.84893, NLL 0.7831 → 0.5285.
- confidence threshold sweep는 `runs/.../calibration.json`에 보존했다.
- TorchScript export 완료.
- 길이 64/128/187/250에서 PyTorch 대비 최대 logit 차이 0, Top-1/Top-5 모두 일치.

## 3000-class 확장

모델과 Dataset은 `num_classes` 및 actor 목록을 config에서 받는다. 향후 별도의
3000-class workspace/config에서 input projection, positional embedding, encoder,
decoder weight를 불러오고 classifier만 3000 출력으로 교체할 수 있다. 현재 300-word
manifest, classes, runs는 확장 실험의 산출물과 섞지 않는다.
