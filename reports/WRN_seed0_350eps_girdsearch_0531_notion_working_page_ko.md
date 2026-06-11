# WRN seed0 350eps grid-search 작업용 Notion 정리

## 목적
이 페이지는 WRN-28-10 dropout 0.3 CIFAR-10 seed0 350-epoch grid-search 결과를 내가 직접 읽고 분석하기 좋게 정리하기 위한 작업용 대시보드다.

## 해석 경계
- 이 결과는 seed0-only diagnostic grid이다.
- hyperparameter 선택 기준은 validation accuracy이다.
- OOD AUROC/FPR95/AUPR와 geometry metric은 선택 기준이 아니라 response-surface 분석용이다.
- 모든 후보는 epoch_0350 평가를 사용하되, `sgd_lr1e-1_wd5e-4_anchor`만 package note에 따라 `final`을 350-epoch endpoint로 취급한다.

## Notion 원본 및 확인 파일
- Notion page: `https://www.notion.so/371a26cf6e72819bacacd14427eb6614`
- Notion fetch 확인일: 2026-06-01 KST
- Package manifest: `results/WRN_seed0_350eps_girdsearch_0531/WRN_seed0_350eps_girdsearch_0531/manifest.json`
- Processed manifest: `results/processed/WRN_seed0_350eps_girdsearch_0531_generation_manifest.json`
- Metric/table guide: `results/processed/WRN_seed0_350eps_girdsearch_0531_tables_and_metric_definitions.md`
- Earlier anchor bundle cross-check: `results/wrn350_seed0_eval_bundle_20260525/processed/wrn350_seed0_eval_summary_20260525.md`

## 핵심 해석 요약
### Confirmed metrics
- Validation 기준 Top-5는 모두 SGD 계열이다. 최고 validation 후보는 `sgd_lr1e-1_wd2e-4`이며 best val acc `0.9612`, ID test acc `0.9546`, ECE `0.0331`이다.
- 기존 matched anchor인 `sgd_lr1e-1_wd5e-4_anchor`는 ID test acc `0.9585`, ECE `0.0298`, `nc1=0.0489`, `nc3_cos_alignment=0.9470`으로 여전히 강한 기준점이다.
- `adamw_lr5e-3_wd1e-4`는 AdamW validation-best 후보이며 best val acc `0.9528`, ID test acc `0.9468`, NLL `0.5234`, ECE `0.0451`, fitted temperature `5.1008`이다.
- `adamw_lr5e-3_wd1e-4`의 mean AUROC는 raw Mahalanobis `0.4719`에서 Mahalanobis L2 `0.9226`, raw kNN `0.6538`에서 kNN L2 `0.9418`로 크게 회복된다.
- `adam_lr1e-3_wd0`도 raw Mahalanobis `0.6764`에서 Mahalanobis L2 `0.9351`, raw kNN `0.8730`에서 kNN L2 `0.9446`로 회복된다.

### Interpretation
- 현재 결과는 "Adam/AdamW에서 feature OOD가 실패한다"가 아니라, raw feature norm/covariance scale에 민감한 detector가 크게 흔들리고 L2-normalized control이 이를 상당히 회복한다는 쪽이 더 정확하다.
- Adam/AdamW의 raw calibration 문제와 logit OOD 성능은 같은 지표가 아니다. NLL/ECE는 ID confidence correctness를 보고, Energy/MaxLogit AUROC는 ID/OOD score ranking을 본다.
- AdamW는 `nc2_mean_cos`나 `nc4_agreement` 일부가 유지되어 보일 수 있지만, `nc0_width_norm`, `nc1`, `nc3_cos_alignment`, `inter_dist_l2`, covariance conditioning이 함께 약한 partial NC regime으로 보는 것이 안전하다.
- Adam `wd=1e-4`는 Adam `wd=0`보다 `nc0_width_norm`과 `nc3_cos_alignment`가 좋아져 coupled WD가 NC0/NC3 축을 회복시키는 방향을 보이지만, covariance condition은 여전히 불안정하다.

### Risk / missing evidence
- seed0-only라서 seed 평균, CI, 안정성 claim은 금지한다.
- OOD AUROC/FPR95/AUPR는 hyperparameter 선택 기준으로 쓰지 않는다.
- `mahalanobis_l2`는 Mahalanobis++-motivated normalization control이지 full Mahalanobis++ 재현이 아니다.
- "NC가 덜 일어나서 OOD가 무너졌다"라고 단정하지 않는다. feature norm, covariance scale, class mean separation, classifier-feature alignment를 분리해 말한다.

### Next action
- seed1/seed2 반복 후보를 정할 때 validation 기준 SGD 후보, Adam wd=0/wd>0 후보, AdamW lr=5e-3 WD 축 후보를 분리한다.
- ID/OOD score histogram, feature norm distribution, covariance eigenspectrum, L2-normalized geometry 재계산을 다음 분석 후보로 둔다.

## Notion DB로 가져갈 파일
- 메인 DB: `results/processed/WRN_seed0_350eps_girdsearch_0531_notion_run_overview_ko.csv`
- Metric 사전: `results/processed/WRN_seed0_350eps_girdsearch_0531_notion_metric_dictionary_ko.csv`
- Detector 사전: `results/processed/WRN_seed0_350eps_girdsearch_0531_notion_detector_dictionary_ko.csv`
- 분석 질문 체크리스트: `results/processed/WRN_seed0_350eps_girdsearch_0531_notion_analysis_questions_ko.csv`
- 전체 OOD long table: `results/processed/WRN_seed0_350eps_girdsearch_0531_ood_all_long.csv`

## 먼저 볼 Top-5 validation 후보
| 후보명 | optimizer | LR | WD | best val acc | ID test acc | ECE | Maha mean AUROC | Maha L2 mean AUROC | Maha L2-raw | kNN mean AUROC | kNN L2 mean AUROC | NC1 | class mean dist | 분석태그 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgd_lr1e-1_wd2e-4 | sgd | 0.1 | 0.0002 | 0.9612 | 0.9546 | 0.0331 | 0.9092 | 0.9242 | 0.0150 | 0.9188 | 0.9233 | 0.0682 | 14.3289 |  |
| sgd_lr2e-1_wd5e-4 | sgd | 0.2 | 0.0005 | 0.9610 | 0.9573 | 0.0290 | 0.8575 | 0.9258 | 0.0683 | 0.9176 | 0.9337 | 0.0726 | 15.6545 |  |
| sgd_lr5e-2_wd5e-4 | sgd | 0.05 | 0.0005 | 0.9596 | 0.9572 | 0.0309 | 0.9123 | 0.9267 | 0.0144 | 0.9128 | 0.9208 | 0.0563 | 18.3419 |  |
| sgd_lr1e-1_wd1e-3 | sgd | 0.1 | 0.001 | 0.9588 | 0.9549 | 0.0326 | 0.9242 | 0.9357 | 0.0115 | 0.9150 | 0.9192 | 0.0382 | 20.0971 |  |
| sgd_lr1e-1_wd5e-4_anchor | sgd | 0.1 | 0.0005 | 0.9586 | 0.9585 | 0.0298 | 0.9192 | 0.9300 | 0.0108 | 0.9165 | 0.9236 | 0.0489 | 16.7966 | anchor |

## Adam wd=0 vs wd=1e-4
| 후보명 | optimizer | LR | WD | best val acc | ID test acc | ECE | Maha mean AUROC | Maha L2 mean AUROC | Maha L2-raw | kNN mean AUROC | kNN L2 mean AUROC | NC1 | class mean dist | 분석태그 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adam_lr1e-3_wd0 | adam | 0.001 | 0.0 | 0.9524 | 0.9436 | 0.0473 | 0.6764 | 0.9351 | 0.2588 | 0.8730 | 0.9446 | 0.2279 | 10.5331 | Mahalanobis L2 회복 큼, NC1 약함 |
| adam_lr1e-3_wd1e-4 | adam | 0.001 | 0.0001 | 0.9494 | 0.9447 | 0.0390 | 0.5531 | 0.8148 | 0.2617 | 0.8692 | 0.8904 | 0.1874 | 13.4176 | Mahalanobis L2 회복 큼 |

## AdamW lr=5e-3 decoupled WD 축
| 후보명 | optimizer | LR | WD | best val acc | ID test acc | ECE | Maha mean AUROC | Maha L2 mean AUROC | Maha L2-raw | kNN mean AUROC | kNN L2 mean AUROC | NC1 | class mean dist | 분석태그 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adamw_lr5e-3_wd1e-4 | adamw | 0.005 | 0.0001 | 0.9528 | 0.9468 | 0.0451 | 0.4719 | 0.9226 | 0.4508 | 0.6538 | 0.9418 | 0.2713 | 5.3713 | Mahalanobis L2 회복 큼, kNN L2 회복 큼, NC1 약함, covariance ill-conditioned |
| adamw_lr5e-3_wd5e-4_anchor | adamw | 0.005 | 0.0005 | 0.9502 | 0.9437 | 0.0479 | 0.5311 | 0.9237 | 0.3926 | 0.6703 | 0.9411 | 0.2747 | 5.5291 | Mahalanobis L2 회복 큼, kNN L2 회복 큼, NC1 약함, covariance ill-conditioned, anchor |
| adamw_lr5e-3_wd1e-3 | adamw | 0.005 | 0.001 | 0.9520 | 0.9452 | 0.0466 | 0.5470 | 0.9278 | 0.3808 | 0.7086 | 0.9443 | 0.2570 | 5.6823 | Mahalanobis L2 회복 큼, kNN L2 회복 큼, NC1 약함, covariance ill-conditioned |

## 분석 메모 템플릿
### Confirmed metrics
- 값으로 확인한 사실만 적기.

### Interpretation
- 왜 그런 패턴이 나왔는지 가설로 적기.

### Risk / missing evidence
- seed0-only, OOD tuning 금지, seed1/2 반복 전 claim 금지.

### Next action
- 반복 실험, 추가 figure, 교수님 보고용 표 후보를 적기.
