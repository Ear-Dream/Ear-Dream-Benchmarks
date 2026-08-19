# Hybrid H1b/H2 결과 보고서

## 데이터

- 현재 300-class와 `한국수어한손양손구분.csv`를 원본 word ID로 결합
- `목요일` 제외, 중복 `팔`은 선정 순번 208 / word ID 1147 사용
- 한손 106, 양손 194
- Train REAL01~06 / Validation REAL07~08 / Test REAL09

## 최종 채택 모델

H1b는 기존 208D backbone을 유지하고 R/L shared hand encoder, hand-type head와
106 head를 추가했다. 기존 오른손 branch로 shared encoder를 초기화하고 backbone
LR을 `1e-5`, 신규 head LR을 `1e-4`로 분리했다. H1의 R/L 평균 초기화 결과는
성능 하락으로 폐기했다.

## REAL09 비교

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

## Head 선택

별도 106 head의 selected-hand Top-1은 90.75%로 기존 300 head보다 낮았다. 따라서
production에서는 이를 확정 분류기로 사용하지 않는다.

- 한 손이 실제 입력된 경우: 관찰 side view의 300 head + 300 retrieval 사용
- 두 손이 입력된 경우: FULL 300 head 사용
- 106 head: 보조 출력/ablation으로만 보존
- hand-type head: FULL test에서 약 99.87%, soft routing 신호로 사용

## Relation reranker 결정

양손 194 FULL Top-1이 이미 99.69%(970개 중 오류 3개)이고 FULL Recall@5가
99.79%다. relation reranker가 얻을 수 있는 최대 Top-1 이득이 0.31%p에 불과해
현재 데이터에서는 복잡도와 과적합 위험이 더 크다. H3 코드는 production에
추가하지 않고, 실제 양손 외부 촬영 데이터에서 오류가 확인될 때 재개한다.

## 결론

한손/양손 partition을 hard route로 쓰지 않고 보조 supervision과 평가 그룹으로
사용한 H1b/H2를 채택한다. 기존 모델보다 FULL과 single-hand retrieval이 모두
개선됐으며, 특히 취약했던 한손 LEFT Recall@5가 6.22%p 상승했다.
