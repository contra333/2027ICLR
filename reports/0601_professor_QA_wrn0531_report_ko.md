# 0601 교수님 보고 준비 문서

## 목적

이 문서는 2026-06-01 교수님 보고를 위해 5월 26일에서 31일 사이의 질문과 `WRN seed0 350eps grid-search_0531` 결과를 seed0 diagnostic evidence로 정리한 원격 공유용 요약본이다.

해석 경계:

- 현재 결과는 seed0-only diagnostic evidence이다.
- hyperparameter 선택 기준은 ID validation accuracy이다.
- OOD AUROC/FPR95/AUPR와 geometry metric은 선택 기준이 아니라 response-surface diagnostic이다.
- confirmed metric, interpretation, hypothesis를 분리한다.

## 한 페이지 요약

| 질문 | 짧은 답 |
| --- | --- |
| AdamW는 ECE가 나쁜데 왜 near-OOD Energy는 좋을 수 있는가? | `ECE`는 ID confidence correctness이고, `Energy AUROC`는 ID/OOD score ranking이다. 두 지표는 같은 질문이 아니다. |
| Mahalanobis와 kNN 차이는 global/local geometry인가? | 부분적으로 맞지만 거친 설명이다. Mahalanobis는 class mean과 shared covariance를 쓰는 parametric detector이고, kNN은 local neighbor distance를 쓰는 non-parametric detector이다. 둘 다 feature norm과 distance scale에 민감하다. |
| AdamW에서 NC 약화 원인은 feature norm, covariance, class mean 중 무엇인가? | 하나만 고르면 안 된다. AdamW는 `NC0`, `NC3`, class mean separation, covariance conditioning, feature norm scale이 함께 달라지는 partial NC regime으로 보는 것이 안전하다. |
| weight decay를 줄이면 Mahalanobis collapse가 완화되는가? | 현재 0531 seed0 grid에서는 단순히 줄이면 완화된다고 말하기 어렵다. AdamW `lr=5e-3`에서 raw Mahalanobis 평균 AUROC는 `wd=1e-4: 0.4719`, `wd=5e-4: 0.5311`, `wd=1e-3: 0.5470`이다. |
| Adam과 AdamW 차이가 decoupled weight decay 때문인가? | Neural Collapse 쪽에서는 `NC0`와 coupled/decoupled weight decay 문헌 근거가 있다. 하지만 OOD detector 변화까지는 controlled experiment로 보여야 한다. |

## 0531 실험 결론

WRN-28-10 CIFAR-10 seed0 350-epoch grid-search에서는 validation 상위 후보가 SGD 계열에 몰려 있다. Adam/AdamW는 ID accuracy가 완전히 무너지지는 않지만 raw feature detector, 특히 raw Mahalanobis와 raw kNN이 약해지는 경우가 있다. 그러나 L2-normalized Mahalanobis와 kNN은 크게 회복된다.

따라서 현재 결과는 "feature detector가 AdamW에서 실패한다"가 아니라, "optimizer가 만든 feature norm/covariance-scale regime이 raw distance/density detector의 가정과 충돌하고, detector-side normalization이 이를 상당 부분 완화한다"로 말해야 한다.

대표 숫자:

- AdamW best validation 후보 `adamw_lr5e-3_wd1e-4`
  - raw Mahalanobis mean AUROC: `0.4719`
  - Mahalanobis L2 mean AUROC: `0.9226`
  - raw kNN mean AUROC: `0.6538`
  - kNN L2 mean AUROC: `0.9418`

## Q1. AdamW는 ECE가 나쁜데 왜 near-OOD Energy는 좋을 수 있는가?

`ECE`와 `Energy AUROC`는 같은 것을 보는 지표가 아니다. `ECE`는 ID 안에서 확률값이 믿을 만한지 보고, `Energy AUROC`는 ID와 OOD score의 순위가 잘 갈리는지 본다.

대표 후보 raw calibration:

| 후보 | ID test acc | ID test NLL | ECE | T-ECE | T |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGD best val `sgd_lr1e-1_wd2e-4` | 0.9546 | 0.2364 | 0.0331 | 0.0079 | 2.1146 |
| Adam wd=0 best val `adam_lr1e-3_wd0` | 0.9436 | 0.4667 | 0.0473 | 0.0071 | 4.3919 |
| Adam wd>0 best val `adam_lr1e-3_wd1e-4` | 0.9447 | 0.2844 | 0.0390 | 0.0060 | 2.4955 |
| AdamW best val `adamw_lr5e-3_wd1e-4` | 0.9468 | 0.5234 | 0.0451 | 0.0068 | 5.1008 |

Near-OOD Energy AUROC:

| 후보 | CIFAR100 Energy AUROC | TinyImageNet Energy AUROC |
| --- | ---: | ---: |
| SGD best val | 0.8476 | 0.8431 |
| Adam wd=0 best val | 0.8973 | 0.8928 |
| Adam wd>0 best val | 0.8997 | 0.9037 |
| AdamW best val | 0.9038 | 0.8987 |

Interpretation: AdamW가 ID에서 overconfident하여 ECE가 나쁠 수 있다. 동시에 OOD sample의 Energy score가 ID보다 충분히 낮으면 Energy AUROC는 좋을 수 있다.

## Q2. Mahalanobis와 kNN 차이는 global/local geometry인가?

대표 후보 feature detector 평균 AUROC:

| 후보 | raw Mahalanobis | Mahalanobis L2 | raw kNN | kNN L2 |
| --- | ---: | ---: | ---: | ---: |
| SGD best val | 0.9092 | 0.9242 | 0.9188 | 0.9233 |
| Adam wd=0 best val | 0.6764 | 0.9351 | 0.8730 | 0.9446 |
| Adam wd>0 best val | 0.5531 | 0.8148 | 0.8692 | 0.8904 |
| AdamW best val | 0.4719 | 0.9226 | 0.6538 | 0.9418 |

AdamW best validation 후보에서는 Mahalanobis가 `0.4719 -> 0.9226`, kNN이 `0.6538 -> 0.9418`로 회복된다. 따라서 AdamW에서는 raw Mahalanobis뿐 아니라 raw kNN도 약해지고, L2 normalization control 이후 둘 다 크게 회복된다고 말하는 편이 안전하다.

## Q3. AdamW에서 NC 약화 원인은 무엇인가?

하나만의 문제가 아니다. AdamW는 `NC0`, `NC3`, class mean separation, covariance conditioning, feature norm scale이 함께 달라지는 partial Neural Collapse regime이다.

| 후보 | `nc0_width_norm` | `nc1` | `nc3_cos_alignment` | `inter_dist_l2` | condition number |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGD best val | 2.70e-10 | 0.0682 | 0.9371 | 14.3289 | 9.76e3 |
| Adam wd=0 best val | 0.6924 | 0.2279 | 0.8108 | 10.5331 | 6.67e4 |
| Adam wd>0 best val | 0.0132 | 0.1874 | 0.9069 | 13.4176 | 5.00e7 |
| AdamW best val | 9.9058 | 0.2713 | 0.6114 | 5.3713 | 6.53e11 |

## Q4. weight decay를 줄이면 Mahalanobis collapse가 완화되는가?

현재 0531 seed0 grid에서는 단순히 weight decay를 줄이면 Mahalanobis collapse가 완화된다고 말하기 어렵다.

| 후보 | WD | best val acc | raw Mahalanobis | Mahalanobis L2 | raw kNN | kNN L2 | `nc1` | `inter_dist_l2` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adamw_lr5e-3_wd1e-4` | 0.0001 | 0.9528 | 0.4719 | 0.9226 | 0.6538 | 0.9418 | 0.2713 | 5.3713 |
| `adamw_lr5e-3_wd5e-4_anchor` | 0.0005 | 0.9502 | 0.5311 | 0.9237 | 0.6703 | 0.9411 | 0.2747 | 5.5291 |
| `adamw_lr5e-3_wd1e-3` | 0.0010 | 0.9520 | 0.5470 | 0.9278 | 0.7086 | 0.9443 | 0.2570 | 5.6823 |

핵심은 단순 WD 크기보다 AdamW가 만든 feature norm/covariance-scale regime과 detector-side normalization sensitivity로 보는 것이 맞다.

## Q5. Adam과 AdamW 차이가 decoupled weight decay 때문인가?

Neural Collapse 쪽에서는 그렇게 볼 문헌 근거가 있다. 하지만 OOD detector 변화까지 바로 결론 내리면 안 된다.

Adam wd=0과 Adam wd>0 비교:

| 후보 | WD | raw Mahalanobis | Mahalanobis L2 | `nc0_width_norm` | `nc1` | `nc3_cos_alignment` | `inter_dist_l2` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam wd=0 `adam_lr1e-3_wd0` | 0.0000 | 0.6764 | 0.9351 | 0.6924 | 0.2279 | 0.8108 | 10.5331 |
| Adam wd>0 `adam_lr1e-3_wd1e-4` | 0.0001 | 0.5531 | 0.8148 | 0.0132 | 0.1874 | 0.9069 | 13.4176 |

Adam에 coupled weight decay가 들어가면 `nc0_width_norm`과 `nc3_cos_alignment`는 Adam wd=0보다 NC-like한 방향으로 움직인다. 하지만 raw Mahalanobis는 Adam wd=0보다 Adam wd>0에서 낮다. 따라서 NC0/NC3가 좋아지면 Mahalanobis도 반드시 좋아진다고 단순화하면 안 된다.

## 교수님께 말해도 되는 문장

- AdamW는 raw ECE/NLL이 나빠도 near-OOD Energy AUROC가 좋을 수 있습니다. ECE는 ID calibration이고, Energy AUROC는 ID/OOD score ranking이기 때문입니다.
- Mahalanobis와 kNN은 둘 다 feature-distance detector이지만 읽는 geometry가 다릅니다.
- 현재 AdamW 결과는 full Neural Collapse가 완전히 사라졌다는 뜻이 아니라 partial NC regime으로 보는 것이 맞습니다.
- 0531 seed0 grid에서는 AdamW raw Mahalanobis와 raw kNN이 약하지만, L2-normalized Mahalanobis와 kNN이 크게 회복됩니다.
- 따라서 핵심은 "feature OOD가 실패했다"가 아니라 "raw norm/covariance-sensitive detector가 optimizer-induced feature geometry와 충돌했다"입니다.

## 말하면 위험한 문장

- AdamW는 feature OOD가 안 된다.
- NC가 낮아서 Mahalanobis가 무너졌다.
- weight decay를 줄이면 Mahalanobis collapse가 해결된다.
- kNN은 local detector라서 feature norm 문제에서 자유롭다.
- Mahalanobis L2는 Mahalanobis++를 완전히 재현한 것이다.
- seed0 결과만으로 optimizer별 최종 결론을 냈다.

## 다음 분석

1. ID/OOD `energy_id_score`, `maxlogit`, `msp` histogram
2. ID/OOD raw Mahalanobis, Mahalanobis L2, raw kNN, kNN L2 score histogram
3. ID/OOD feature norm distribution
4. class-wise feature norm distribution
5. covariance eigenspectrum과 condition number 비교
6. L2-normalized feature에서 geometry metric 재계산
7. seed1/seed2 반복
8. total WD를 고정하고 coupled ratio만 바꾸는 `adam_coupled_decoupled` sweep

## 최종 보고 문장

이번 WRN-28-10 CIFAR-10 seed0 350-epoch grid-search는 optimizer가 단순히 accuracy만 바꾸는 것이 아니라 feature geometry와 detector behavior를 함께 바꾼다는 진단 증거를 준다. SGD 계열은 validation accuracy와 NC-like geometry, raw feature detector가 함께 안정적이다. Adam/AdamW, 특히 AdamW `lr=5e-3` regime에서는 raw Mahalanobis와 raw kNN이 약하지만 L2-normalized detector가 크게 회복된다. 따라서 현재 결론은 AdamW feature가 OOD에 쓸모없다는 것이 아니라, AdamW가 만든 feature norm/covariance-scale regime이 raw distance/density detector의 가정과 충돌한다는 것이다. 이 해석은 seed0 진단 결과이며, seed 반복과 coupled-ratio sweep으로 안정성과 원인을 더 확인해야 한다.
