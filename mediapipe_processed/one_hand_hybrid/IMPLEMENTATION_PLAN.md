# 한손 106 분류 보존 + 전체 300 검색 + 양손 194 관계 Hybrid 계획

## 확정 데이터 분할

`한국수어한손양손구분.csv` 302행에서 현재 학습 300-class와 대조한다.

- `목요일`: 3000-class 목록 및 현재 300-class에서 제외
- 중복 `팔`: 현재 선택된 원본 word ID 1147(선정 순번 208, 양손)만 유지
- 최종 한손 106 / 양손 194

분할은 단어 문자열만으로 런타임 추정하지 않고 `label_index`, `word_id`로 고정한
`data/word_partition_report.json`과 `data/word_partition.csv`를 사용한다.

## 모델

기존 검증된 208D 입력과 6-layer Squeezeformer backbone을 유지한다.

```text
Pose 50D ───────────────┐
Right 42D ─ Shared Hand ├─ Part fusion ─┐
Left  42D ─ Shared Hand ┤               ├─ mask gate ─ Encoder × 6 ─ pooling 256D
Face  74D ──────────────┘  All 208D ────┘
                                              ├─ FULL 300 classifier
                                              ├─ one-hand 106 classifier
                                              ├─ hand-type binary classifier
                                              └─ retrieval embedding 128D
```

한손 head는 한손 106 샘플의 motion-selected hand view에만 CE를 적용한다. 양손
194는 이 head의 CE에서 제외한다. 전체 300 retrieval은 모든 샘플을 사용한다.

## 단계

1. B0를 한손 106/양손 194 그룹으로 재평가한다.
2. H1: shared R/L hand encoder + 106 head + hand-type auxiliary head.
3. H2: 한손 selected/full, 양손 right/left/full multi-prototype.
4. H3: 양손 194의 두 손 거리·상대속도·동기성 relation reranker.
5. REAL09 test, calibration, TorchScript export 및 비교 보고서.

## H1 loss

```text
1.0 * full_300_ce
+ 1.0 * onehand_106_ce (한손 sample only)
+ 1.0 * full_partial_alignment
+ 0.1 * same_word_supcon
+ 0.2 * hand_type_ce
```

## 통과 기준

- FULL 300 Top-1 하락 0.5%p 이내
- 한손 106 전용 Top-1 상승
- 전체 Recall@10 하락 2%p 이내
- LEFT_ONLY Recall@5 상승
- 양손 194 single-hand Recall@K 별도 기록
