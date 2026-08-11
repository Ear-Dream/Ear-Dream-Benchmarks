# 한국 수어 300단어 분류 실험

MediaPipe로 전처리한 수어 시퀀스를 이용해 300개 단어를 분류하는 파일럿 실험이다. 이후 진행할 3,000단어/REAL01~16 실험과 섞이지 않도록 별도 작업 공간에서 수행했다. 원본 HDF5 파일은 읽기 전용으로 참조하며 복사·수정·삭제하지 않는다.

## 실험 구성

- 입력: 프레임마다 208차원인 `spoter2_mp_xy_v1` 특징
- 클래스: `일상_고빈도_핵심단어_300.csv`에서 선정한 300단어
- 모델: SPOTER-208 (`208 → 256` projection, Transformer encoder 6층, class-query decoder 6층, 8 heads)
- 분할: 촬영자를 기준으로 Train REAL01~06 / Validation REAL07~08 / Test REAL09
- 샘플 수: Train 8,994 / Validation 3,000 / Test 1,500, 총 13,494개
- 학습 설정: seed 42, 최대 길이 256, batch size 32, AdamW, learning rate `1e-4`, weight decay `0.01`, warm-up 5 epochs와 cosine schedule, FP16, gradient clipping 1.0
- 종료 조건: 최대 100 epochs, validation macro Top-1 기준 patience 15의 early stopping

동일한 `(word_id, actor_id)` 그룹은 서로 다른 split에 섞이지 않는다. 확정된 분할은 `data/splits.json`에 저장했다. 원본에서 찾지 못한 6개 샘플은 모두 Train 구간이며 `data/missing_samples.json`에 기록했다. 모든 split에는 300개 클래스가 빠짐없이 포함되어 있다.

## 실험 진행 과정

1. **데이터 목록 생성**: 선정 단어와 원본 manifest를 결합해 샘플 목록, 클래스 정보, 고정 actor split을 생성했다.
2. **데이터 무결성 검증**: HDF5 group 13,494개에 대해 feature 차원과 버전, NaN/Inf, frame index 연속성, part mask shape를 검사했다. 오류는 0개였다. 시퀀스 길이는 60~250 frame(중앙값 120, p95 162, p99 187.07)이므로 최대 길이 256 안에 전부 포함된다.
3. **학습 전 점검**: synthetic shape/padding invariance test를 수행하고, 32개 샘플 overfit에서 정확도 100%를 확인했다. 이어 10-class smoke run에서 validation macro Top-1 88%를 확인했다.
4. **본 학습**: 전체 Train split으로 학습하고 validation macro Top-1이 가장 높은 checkpoint를 `best.pt`로 저장했다. 최고 validation macro Top-1은 91.77%였으며 early stopping으로 86 epoch에서 종료됐다.
5. **최종 평가**: 학습과 모델 선택에 사용하지 않은 REAL09를 한 번 평가했다.
6. **확률 보정 및 배포 검증**: Validation set으로 temperature scaling을 수행한 뒤 TorchScript로 내보내고, 여러 입력 길이에서 원본 PyTorch 모델과 출력이 같은지 확인했다.

## 재현 방법

프로젝트 루트에서 아래 순서대로 실행한다.

```powershell
$py = 'C:\Users\PJ15\AppData\Local\Programs\Python\Python312\python.exe'
& $py -m pip install -r requirements.txt
& $py scripts/build_data.py --config configs/pilot300_base.yaml
& $py scripts/validate_data.py --config configs/pilot300_base.yaml
& $py scripts/train.py --config configs/pilot300_base.yaml --mode overfit32
& $py scripts/train.py --config configs/pilot300_base.yaml --mode smoke10
& $py scripts/train.py --config configs/pilot300_base.yaml --mode train
& $py scripts/calibrate.py --config configs/pilot300_base.yaml
& $py scripts/evaluate.py --config configs/pilot300_base.yaml
& $py scripts/export.py --config configs/pilot300_base.yaml
```

`train.py`는 중단된 run의 `last.pt`가 있으면 이어서 학습한다. 설정 전체는 `configs/pilot300_base.yaml`, 학습 이력은 run 디렉터리의 `history.csv`에서 확인할 수 있다.

## 최종 결과

| 지표 | REAL09 Test 결과 |
|---|---:|
| Macro Top-1 | **98.33%** |
| Micro Top-1 | **98.33%** |
| Top-3 | **99.87%** |
| Top-5 | **100.00%** |
| Macro F1 | **98.16%** |

카메라별 Top-1은 D 98.0%, F 98.0%, L 98.33%, R 99.0%, U 98.33%였다. 이 수치는 REAL09 한 명에 대한 actor-independent holdout 결과이므로 일반화 성능을 확정하는 값은 아니다. 향후 REAL10~16으로 공식 validation/test split을 구성한 뒤 다시 평가해야 한다.

Temperature scaling의 최적 온도는 1.84893이었고 validation NLL은 0.7831에서 0.5285로 감소했다. TorchScript 변환 후 길이 64, 128, 187, 250에서 최대 logit 차이는 모두 0이었으며 Top-1/Top-5 결과도 일치했다.

주요 산출물은 `runs/pilot300_spoter_base_seed42_train/`에 있다.

- `best.pt`, `last.pt`: 모델 checkpoint
- `test_metrics.json`, `per_class_metrics.csv`, `top_confusions.csv`: 평가 결과
- `confusion_matrix.png`: confusion matrix
- `calibration.json`: temperature와 confidence threshold별 coverage/accuracy
- `model_torchscript.pt`, `export_parity.json`: 배포 모델과 변환 일치성 검사 결과
