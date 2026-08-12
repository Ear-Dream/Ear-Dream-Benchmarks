# AIHUB Sign Word Hybrid Experiment — 300 classes

## Data contract

- Selected vocabulary: `일상_고빈도_핵심단어_300.csv`
- Samples: train 8,994 / validation 3,000 / test 1,500
- Actor split: train REAL01–06 / validation REAL07–08 / test REAL09
- Input: existing `mediapipe_processed` H5 `features [T,208]`; no source H5 was modified
- The copied `samples.csv` SHA-256 is identical in E0/H1/H2/H3/A1-P.

## Core screening (seed 42, validation macro top-1)

| Model | Result |
|---|---:|
| E0 — `sign_word_300` | 91.77% |
| H1 — Part Conv + Transformer + Decoder | 91.87% |
| H2 — Part Conv + Squeezeformer-style + Decoder | 92.20% |
| H3 — Part+All Conv + Squeezeformer-style + Decoder | 92.23% |

## Repeated seeds

| Model | seed 42 | seed 123 | seed 2026 | mean ± sample SD |
|---|---:|---:|---:|---:|
| E0 | 91.77% | 91.33% | 91.93% | 91.68 ± 0.31% |
| H2 | 92.20% | 90.80% | 91.40% | 91.47 ± 0.70% |
| H3 | 92.23% | 92.33% | 92.20% | **92.26 ± 0.07%** |

H3 was selected as the core architecture because it had the highest repeated-seed mean and the lowest seed variance.

## A1 head ablation

| Head on H3 core | seed 42 validation macro top-1 | Parameters | Best checkpoint size |
|---|---:|---:|---:|
| Class-query Decoder ×6 | 92.23% | 12,689,196 | 152.6 MB |
| Masked temporal attention pooling | **92.77%** | **6,400,813** | **77.0 MB** |

A1-P was selected: +0.53 percentage points on validation while removing about 49.6% of parameters and checkpoint bytes.

## Final REAL09 test (used after validation selection)

| Metric | Result |
|---|---:|
| Micro top-1 | 97.80% |
| Macro top-1 | 97.80% |
| Top-3 | 99.53% |
| Top-5 | 99.60% |
| Macro F1 | 97.58% |

Camera top-1: D 97.00%, F 98.33%, L 97.67%, R 98.33%, U 97.67%.

The historical E0 REAL09 result was 98.33%, so the selected Hybrid is 0.53 percentage points lower on this held-out actor despite its stronger validation result. This test result must not be used to select another architecture without creating a new untouched test protocol.

## Calibration and export

- Validation temperature: 1.950361
- Validation NLL: 0.738696 → 0.476328
- TorchScript parity at lengths 64/128/187/250: exact logits (`max_abs_difference = 0`), top-1/top-5 match
- Final checkpoint: `runs/pilot300_a1_h3_attention_pooling_seed42_train/best.pt`
- Export: `runs/pilot300_a1_h3_attention_pooling_seed42_train/model_torchscript.pt`

All models and runs remain isolated in `sign_word_h1`, `sign_word_h2`, `sign_word_h3`, and `sign_word_a1_pooling`; the existing `sign_word_300` baseline artifacts were preserved.
