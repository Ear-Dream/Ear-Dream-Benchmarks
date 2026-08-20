# 시나리오 핵심 38단어 제한 모델

`시나리오_필요단어_38개_UTF8_BOM.csv`의 38단어만 분류하도록 기존 `양손한손수어단어_정리 300 Hybrid`의 후보를 제한한 프로젝트다.

> 권장 사용: 양손이 보이는 FULL 입력을 기본으로 사용하고, 손 하나만 유효할 때는 RIGHT 또는 motion-selected view를 사용한다.

## 결론

현재 AI Hub REAL09 test에서는 별도 파인튜닝 없이 후보 제한만 적용하는 방식이 가장 적절하다.

- 38단어 모두 기존 300단어 모델에 포함
- 한손 15 / 양손 23
- 단어별 train 29~30 / validation 10 / test 5
- 평가 표본: 38 × 5 = 190개
- FULL 제한 분류 Top-1: **100%**
- RIGHT 제한 분류 Top-1: **99.47%**
- FULL 제한 retrieval Recall@1: **98.42%**
- FULL/RIGHT retrieval Recall@5: **100% / 99.47%**

같은 AI Hub 데이터로 추가 파인튜닝하면 이미 100%인 평가에 과적합할 가능성이 있어 진행하지 않았다. 실제 서비스 영상에서 오류가 확인될 때 해당 영상으로 파인튜닝한다.

## 모델 관계

```text
기존 300 Hybrid backbone 및 embedding
  ├─ 300-class 원본 출력
  └─ 38개 label만 index-select
       ├─ scenario_logits [B,38]
       └─ 38개 전용 retrieval prototype
```

새 모델을 처음부터 다시 학습한 것이 아니라, 검증된 300단어 모델의 표현력을 유지하면서 허용 후보만 38개로 제한했다.

## 비교 결과

| 입력 | 방법 | 분류 Top-1 | Retrieval R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| FULL | 기존 300 후보 | 98.95% | 97.37% | - | - |
| FULL | 38 후보 제한 | **100.00%** | **98.42%** | **100.00%** | **100.00%** |
| RIGHT | 기존 300 후보 | 97.89% | 97.37% | - | - |
| RIGHT | 38 후보 제한 | **99.47%** | **99.47%** | **99.47%** | **99.47%** |
| LEFT | 기존 300 후보 | 65.26% | 70.00% | - | - |
| LEFT | 38 후보 제한 | 75.26% | 84.74% | 97.89% | 98.95% |

왼손 단독 입력은 여전히 낮으므로 실제 서비스에서는 FULL 입력 또는 motion-selected/right 입력을 우선 사용한다.

## 최종 활용 파일

`runs/` 아래 파일:

- `scenario38_model_torchscript.pt`: 38-class logits를 출력하는 최종 모델
- `scenario38_prototypes.npz`: 38개 full/right/left/selected prototype
- `restricted38_test_metrics.json`: 기존 300 후보와 38 후보 비교 결과
- `scenario38_export_parity.json`: TorchScript 출력 일치 검증

클래스 매핑:

- `data/scenario38_v1/classes.json`: 인덱스 순서의 단어 배열
- `data/scenario38_v1/scenario38_classes.csv`: 38 label과 기존 300 label 대응
- `data/scenario38_v1/scenario38_report.json`: 손 유형 및 시나리오 전체 정보

## TorchScript 입출력

입력:

```text
x         [B,T,208] float32
padding   [B,T]     bool
detected  [B,T,2]
view      [B,T,2]
```

출력 순서:

```text
scenario_logits   [B,38]
hand_type_logits  [B,2]    # 0=한손, 1=양손
embedding         [B,128]
```

`scenario_logits.argmax(1)`을 `classes.json`의 인덱스로 변환하면 최종 단어가 된다.

## 최소 추론 예제

```python
import json
from pathlib import Path

import torch

root = Path("one_hand_hybrid_scenario38")
model = torch.jit.load(str(root / "runs/scenario38_model_torchscript.pt"))
model.eval()
classes = json.loads((root / "data/scenario38_v1/classes.json").read_text(encoding="utf-8"))

# 전처리 결과를 아래 shape으로 준비한다.
x = torch.zeros(1, 16, 208, dtype=torch.float32)
padding = torch.zeros(1, 16, dtype=torch.bool)
detected = torch.ones(1, 16, 2, dtype=torch.uint8)
view = torch.ones(1, 16, 2, dtype=torch.uint8)

with torch.no_grad():
    scenario_logits, hand_type_logits, embedding = model(x, padding, detected, view)

index = int(scenario_logits.argmax(dim=1)[0])
word = classes[index]
confidence = float(scenario_logits.softmax(dim=1)[0, index])
print({"word": word, "confidence": confidence})
```

위 예제의 0 입력은 API 형태 확인용이다. 실제 추론에는 기존 MediaPipe 파이프라인이 만든 208D feature와 검출 mask를 전달해야 한다.

## 폴더 구성

```text
one_hand_hybrid_scenario38/
├─ configs/scenario38.yaml
├─ data/
│  ├─ 시나리오_필요단어_38개_UTF8_BOM.csv
│  └─ scenario38_v1/
│     ├─ classes.json
│     ├─ scenario38_classes.csv
│     └─ scenario38_report.json
├─ scripts/
│  ├─ build_scenario38.py
│  ├─ evaluate_restricted38.py
│  └─ export_scenario38.py
├─ runs/
│  ├─ scenario38_model_torchscript.pt
│  ├─ scenario38_prototypes.npz
│  ├─ restricted38_test_metrics.json
│  └─ scenario38_export_parity.json
├─ src/
└─ tests/
```

## 재현

```powershell
$py = 'C:\Users\PJ15\AppData\Local\Programs\Python\Python312\python.exe'

& $py scripts/build_scenario38.py --config configs/scenario38.yaml
& $py scripts/evaluate_restricted38.py --config configs/scenario38.yaml
& $py scripts/export_scenario38.py --config configs/scenario38.yaml
```

## 실제 데이터 파인튜닝 조건

실제 카메라 영상에서 성능이 부족할 때만 파인튜닝한다.

- 최소 5명 이상
- 단어당 10~20회 이상
- 조명·거리·배경·촬영 각도 다양화
- 학습에 없는 사람을 validation/test로 분리
- 38개 외 수어와 일반 손동작을 `unknown/other`로 별도 수집
