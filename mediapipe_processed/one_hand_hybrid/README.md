# One-Hand Hybrid Classification and Retrieval

기본 one-hand candidate 모델 B0를 개선한 최종 hybrid 프로젝트다.
`한국수어한손양손구분.csv`를 현재 300-class 목록에 맞춰 한손 106 / 양손
194로 고정하고, 이 구분을 hard routing이 아닌 보조 supervision과 평가 그룹으로
사용한다.

현재 두 프로젝트 중 배포 권장 모델은 이 폴더의 H1b/H2 모델이다.

## B0와의 관계

```text
AIHUB_ONE_HAND_SIGN_CANDIDATE_RAW_TO_MODEL_TRAINING
└─ B0: 208D mask-aware 300 classification/retrieval
             ↓ 개선
one_hand_hybrid_classification_retrieval
└─ H1b/H2: shared hand encoder + 106/194 supervision + multi-prototype
```

기존 B0 checkpoint나 결과는 복사·덮어쓰지 않았다.

## 확정된 300-class partition

`한국수어한손양손구분.csv` 302행을 현재 선택 목록 및 원본 word ID와 결합했다.

- `목요일`: 현재 3000/300-class mapping에 없어 제외
- 중복 `팔`: 선정 순번 208, 원본 word ID 1147, 양손 항목 사용
- 선정 순번 301의 중복 `팔` 제외
- 최종 한손 106 / 양손 194 = 300개

고정 결과:

- `data/word_partition_report.json`
- `data/word_partition.csv`

## 최종 모델 H1b

```text
MediaPipe feature [B,T,208]
├─ Pose 50D ────────────────┐
├─ Right 42D ─ Shared Hand ─┤
├─ Left 42D  ─ Shared Hand ─┤─ Part fusion
└─ Face 74D ────────────────┘

Part fusion + mask-gated All 208D branch
                    ↓
Squeezeformer-style Encoder × 6
                    ↓
Masked Temporal Attention Pooling 256D
       ├─ FULL 300-class head          # 최종 분류에 사용
       ├─ one-hand 106-class head      # 보조/ablation
       ├─ one/two-hand binary head     # soft routing 신호
       └─ normalized 128D embedding    # 전체 300 retrieval
```

오른손과 왼손은 동일한 Conv1D hand encoder를 공유한다. 기존 B0 오른손 branch로
shared encoder를 초기화하고, 나머지 호환 backbone과 300 head를 B0 checkpoint에서
이식했다.

학습률:

- 기존 backbone: `1e-5`
- 신규 auxiliary head: `1e-4`

최종 loss:

```text
1.0  × FULL 300-class CE
+ 1.0  × one-hand 106 CE      # 한손 106에만 적용
+ 0.25 × full/partial alignment
+ 0.05 × same-word SupCon
+ 0.10 × one/two-hand CE
```

## H2 prototype

Train split만 사용해 다음 prototype bank를 생성한다.

- FULL 300-class prototype
- RIGHT 300-class prototype
- LEFT 300-class prototype
- motion-selected hand prototype

최종 파일은 `h2_prototypes.npz`다.

## 최종 REAL09 결과

| 지표 | B0 | H1b/H2 | 변화 |
|---|---:|---:|---:|
| 전체 FULL Top-1 | 96.73% | 98.00% | +1.27%p |
| 전체 FULL Recall@5 | 99.27% | 99.60% | +0.33%p |
| 전체 RIGHT Recall@5 | 98.60% | 98.93% | +0.33%p |
| 전체 LEFT Recall@5 | 88.13% | 90.20% | +2.07%p |
| 한손 106 FULL Top-1 | 93.96% | 94.91% | +0.95%p |
| 한손 106 RIGHT Top-1 | 93.96% | 95.85% | +1.89%p |
| 한손 106 LEFT Recall@5 | 71.89% | 78.11% | +6.22%p |
| 양손 194 FULL Top-1 | 98.25% | 99.69% | +1.44%p |
| 양손 194 LEFT Recall@5 | 97.01% | 96.80% | -0.21%p |

## 최종 추론 정책

한손/양손 label로 검색 후보를 제한하지 않는다.

### 손 하나가 입력된 경우

```text
실제 관찰된 side view
├─ 300-class logits
└─ 해당 side prototype과 cosine similarity → 전체 300 Top-K
```

### 두 손이 입력된 경우

```text
FULL view → 300-class logits + FULL prototype Top-K
```

별도 106 head의 selected-hand Top-1은 90.75%로 300 head보다 낮았으므로 최종
확정 분류에는 사용하지 않는다. 모델에는 보조 출력과 ablation을 위해 남아 있다.

hand-type head는 FULL test에서 약 99.87%지만 hard routing에는 사용하지 않는다.

## Relation reranker 결정

양손 194 FULL Top-1은 99.69%로 970개 중 오류가 3개뿐이고 FULL Recall@5는
99.79%다. 현재 test에서 relation reranker의 최대 Top-1 이득은 0.31%p이므로
복잡도와 과적합 위험을 고려해 production에 추가하지 않았다. 실제 외부 촬영
양손 데이터에서 오류가 확인되면 H3를 재개한다.

## 실행 순서

```powershell
python scripts/build_partition.py --config configs/h1b_preserve_shared_hand.yaml
python -m pytest -q

python scripts/train_h1.py `
  --config configs/h1b_preserve_shared_hand.yaml `
  --mode overfit32

python scripts/train_h1.py `
  --config configs/h1b_preserve_shared_hand.yaml `
  --mode train

python scripts/build_h2_prototypes.py `
  --config configs/h1b_preserve_shared_hand.yaml `
  --checkpoint runs/h1b_preserve_shared_hand_one106_retrieval300_train/best.pt

python scripts/evaluate_h1_h2.py `
  --config configs/h1b_preserve_shared_hand.yaml `
  --checkpoint runs/h1b_preserve_shared_hand_one106_retrieval300_train/best.pt `
  --prototypes runs/h1b_preserve_shared_hand_one106_retrieval300_train/h2_prototypes.npz

python scripts/calibrate_hybrid.py `
  --config configs/h1b_preserve_shared_hand.yaml `
  --checkpoint runs/h1b_preserve_shared_hand_one106_retrieval300_train/best.pt `
  --prototypes runs/h1b_preserve_shared_hand_one106_retrieval300_train/h2_prototypes.npz

python scripts/export_hybrid.py `
  --config configs/h1b_preserve_shared_hand.yaml `
  --checkpoint runs/h1b_preserve_shared_hand_one106_retrieval300_train/best.pt
```

## Calibration

95% precision 기준 validation 결과:

- RIGHT: threshold `0.7880`, coverage 88%
- LEFT: threshold `0.9381`, coverage 42%

## 주요 산출물

`runs/h1b_preserve_shared_hand_one106_retrieval300_train/`:

- `best.pt`: 최종 최고 checkpoint
- `h1_h2_test_metrics.json`: REAL09 그룹별 결과
- `h2_prototypes.npz`: FULL/RIGHT/LEFT/selected prototype
- `hybrid_calibration.json`: validation threshold/coverage curve
- `hybrid_model_torchscript.pt`: 최종 배포 모델
- `hybrid_export_parity.json`: PyTorch/TorchScript 비교

TorchScript 최대 출력 오차는 `2.15e-6`, 단위 테스트는 2개 모두 통과했다.

상세 실험 선택과 폐기 이유는 `REPORT.md`, 구현 순서는
`IMPLEMENTATION_PLAN.md`를 참고한다.
