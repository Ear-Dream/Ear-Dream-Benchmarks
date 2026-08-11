# 한국 수어 3,000단어 분류 실험

MediaPipe로 전처리한 한국 수어 시퀀스를 SPOTER 기반 모델로 분류한 실험이다. 기존 300단어 모델의 encoder/decoder 가중치를 가져오고, 3,000개 클래스를 위한 분류기만 새로 초기화해 전체 모델을 fine-tuning했다.

## 핵심 결과

독립된 화자 `REAL09`를 테스트 세트로 사용한 결과는 다음과 같다.

| 지표 | 결과 |
| --- | ---: |
| Macro / Micro Top-1 | **97.04%** |
| Top-3 | **99.28%** |
| Top-5 | **99.52%** |
| Macro F1 | **96.68%** |
| Cross entropy | 0.1553 |

카메라별 Top-1 정확도는 D 97.50%, F 97.90%, L 96.53%, R 96.30%, U 96.97%였다. 테스트 세트는 모델, epoch, confidence threshold 선택에 사용하지 않았다.

## 데이터 구성

- 입력: `../mediapipe_processed` 아래 `REAL01~09` HDF5 파일(읽기 전용)
- 특징: `spoter2_mp_xy_v1`, 프레임당 208차원
- 클래스: 3,000단어
- 분할: 화자 독립(actor-independent) 방식
  - Train: REAL01~06, 89,958개
  - Validation: REAL07~08, 29,993개
  - Test: REAL09, 15,000개
- 전체: 134,951개

예상한 135,000개 중 누락된 49개는 `data/missing_samples.json`에 기록했다. 남은 모든 샘플은 HDF5 검사를 통과했으며, 각 분할에 3,000개 클래스가 모두 존재한다. 시퀀스 길이는 최소 56, 중앙값 124, p95 172, p99 200, 최대 268 프레임이다. 256프레임을 넘는 시퀀스는 전체 구간에서 균등 sampling한다.

## 모델과 학습 설정

- SPOTER-208: 208차원 입력을 256차원으로 projection
- Transformer encoder 6층, class-query decoder 6층
- 8 attention heads, FFN 1,024차원, dropout 0.1, GELU
- 300단어 모델의 classifier를 제외한 192개 tensor 전이
- 새 3,000-class classifier와 전이된 본체를 함께 fine-tuning
- AdamW, learning rate `5e-5`, weight decay `0.01`
- 5 epoch warm-up 후 cosine schedule
- batch size 64, FP16, gradient clipping 1.0
- seed 42, 총 100 epoch

최고 validation Macro Top-1은 **92.79%**로 epoch 98에서 기록됐다. 최종 학습 정확도는 약 99.98%였으며, 학습 재개(resume) 및 완료된 run의 no-op 재실행도 확인했다.

## 실행 방법

프로젝트 루트에서 가상환경을 만든 뒤 의존성을 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

설정 파일 `configs/full3000_transfer.yaml`의 원본 manifest, HDF5 root, 단어 목록, 300단어 checkpoint 경로가 실제 환경과 맞는지 확인한 후 아래 순서로 실행한다.

```powershell
python scripts/build_data.py --config configs/full3000_transfer.yaml
python scripts/validate_data.py --config configs/full3000_transfer.yaml

# 파이프라인 점검과 소량 과적합 검사
python scripts/train.py --config configs/full3000_transfer.yaml --mode smoke10
python scripts/train.py --config configs/full3000_transfer.yaml --mode overfit32

# 전체 학습 및 평가
python scripts/train.py --config configs/full3000_transfer.yaml --mode train
python scripts/evaluate.py --config configs/full3000_transfer.yaml
python scripts/calibrate.py --config configs/full3000_transfer.yaml
python scripts/export.py --config configs/full3000_transfer.yaml
```

`resume: true`이므로 중단된 전체 학습은 같은 명령으로 이어서 실행된다.

## Calibration 및 export 검증

Validation logits에 temperature scaling을 적용했으며 최적 temperature는 **1.65116**이었다. Validation NLL은 0.63355에서 0.47144로 감소했다. confidence threshold에 따른 coverage/accuracy 결과는 `calibration.json`에 저장된다. 예를 들어 threshold 0.90에서는 validation 샘플의 83.71%를 수용하면서 96.94% 정확도를 보였다.

최종 모델은 TorchScript로 export했다. 길이 64, 128, 187, 250인 입력에서 PyTorch checkpoint와 비교한 결과 최대 logit 차이는 모두 0이었고 Top-1/Top-5 예측도 모두 일치했다.

## 결과 파일

전체 학습 결과는 `runs/full3000_spoter_transfer_seed42_train/`에 있다.

| 파일 | 내용 |
| --- | --- |
| `best.pt`, `last.pt` | 최고 성능 및 마지막 checkpoint |
| `history.csv` | epoch별 loss, accuracy, 실행 시간 |
| `test_metrics.json` | 테스트 지표와 카메라/화자별 정확도 |
| `per_class_metrics.csv` | 클래스별 지표 |
| `top_confusions.csv` | 빈도가 높은 오분류 쌍 |
| `confusion_matrix.npy/png` | confusion matrix 원본 및 이미지 |
| `calibration.json` | temperature와 threshold sweep |
| `model_torchscript.pt` | 배포용 TorchScript 모델 |
| `export_parity.json` | checkpoint-export 출력 일치 검사 |
| `classes.json`, `splits.json` | 클래스 매핑과 데이터 분할 |

대표적인 오분류는 수량 표현(`십오억`→`이십오억`, `십만`→`10만원`)과 의미·동작이 유사한 단어 사이에서 나타났다. 상세 내역은 `top_confusions.csv`를 참고한다.

## 해석 시 주의사항

이 결과는 300단어 모델에서 encoder/decoder를 전이한 단일 seed 실험이다. 같은 계산 예산의 3,000-class from-scratch 대조군은 아직 없으므로, 전이학습 자체의 효과를 정량적으로 주장하려면 별도 비교 실험이 필요하다.
