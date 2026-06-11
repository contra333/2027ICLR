# WRN seed0 350eps grid-search 교수님 보고용 초안

## 한 줄 요약
WRN-28-10 seed0 diagnostic grid에서 validation 기준 top 후보는 SGD 계열이며, Adam/AdamW에서는 feature detector raw score와 L2-normalized control 사이의 차이가 커서 feature norm/covariance-scale channel을 추가로 볼 가치가 있다.

## 보고 전 주의
- 아직 seed0-only diagnostic 결과이므로 최종 claim이나 seed 평균으로 제시하지 않는다.
- hyperparameter 선택은 validation accuracy 기준으로만 말한다.
- OOD/geometry 결과는 optimizer/WD response-surface 분석으로 표현한다.

## 확인한 근거
- Notion page: `https://www.notion.so/371a26cf6e72819bacacd14427eb6614`
- Grid package: `results/WRN_seed0_350eps_girdsearch_0531/WRN_seed0_350eps_girdsearch_0531/manifest.json`
- Processed tables: `results/processed/WRN_seed0_350eps_girdsearch_0531_tables_and_metric_definitions.md`
- Earlier anchor bundle: `results/wrn350_seed0_eval_bundle_20260525/processed/wrn350_seed0_eval_summary_20260525.md`

## 보고용 결론 문장
이번 WRN-28-10 CIFAR-10 seed0 350-epoch grid-search에서는 validation 기준 상위 후보가 SGD 계열에 몰려 있으며, SGD는 raw feature detector와 NC-like geometry가 함께 안정적이다. 반면 Adam/AdamW, 특히 AdamW lr=5e-3 축에서는 ID accuracy가 완전히 무너지지는 않지만 raw NLL/ECE가 나쁘고, raw Mahalanobis/kNN/GMM tied가 크게 약해진다. 다만 L2-normalized Mahalanobis/kNN은 크게 회복되므로, 현재 결론은 "feature detector 실패"가 아니라 optimizer-induced feature norm/covariance-scale regime이 raw distance/density detector assumptions와 충돌한다는 쪽이 더 안전하다.

## 핵심 해석 포인트
- Raw calibration과 logit OOD AUROC는 다른 질문이다. AdamW는 raw ECE/NLL이 나쁠 수 있지만 Energy/MaxLogit AUROC는 dataset에 따라 강할 수 있다.
- AdamW는 full NC 부재라기보다 partial NC regime이다. Class-mean angular structure나 NC4 일부는 유지될 수 있지만 `nc0_width_norm`, `nc1`, `nc3_cos_alignment`, `inter_dist_l2`, covariance conditioning이 약하다.
- Adam `wd=0`와 Adam `wd=1e-4` 비교는 coupled weight decay가 NC0/NC3 축을 회복시키는 방향을 보여준다. 다만 covariance condition까지 SGD처럼 안정화된다고 말하기에는 부족하다.
- `mahalanobis_l2`와 `knn_l2`의 회복은 feature norm/covariance-scale channel이 detector behavior에 중요하다는 진단 근거다. 이것을 Mahalanobis++ full reproduction이라고 부르지는 않는다.

## 핵심 수치 후보: validation Top-5
| 후보명 | optimizer | LR | WD | best val acc | ID test acc | ECE | Maha mean AUROC | Maha L2 mean AUROC | Maha L2-raw | kNN mean AUROC | kNN L2 mean AUROC | NC1 | class mean dist | 분석태그 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sgd_lr1e-1_wd2e-4 | sgd | 0.1 | 0.0002 | 0.9612 | 0.9546 | 0.0331 | 0.9092 | 0.9242 | 0.0150 | 0.9188 | 0.9233 | 0.0682 | 14.3289 |  |
| sgd_lr2e-1_wd5e-4 | sgd | 0.2 | 0.0005 | 0.9610 | 0.9573 | 0.0290 | 0.8575 | 0.9258 | 0.0683 | 0.9176 | 0.9337 | 0.0726 | 15.6545 |  |
| sgd_lr5e-2_wd5e-4 | sgd | 0.05 | 0.0005 | 0.9596 | 0.9572 | 0.0309 | 0.9123 | 0.9267 | 0.0144 | 0.9128 | 0.9208 | 0.0563 | 18.3419 |  |
| sgd_lr1e-1_wd1e-3 | sgd | 0.1 | 0.001 | 0.9588 | 0.9549 | 0.0326 | 0.9242 | 0.9357 | 0.0115 | 0.9150 | 0.9192 | 0.0382 | 20.0971 |  |
| sgd_lr1e-1_wd5e-4_anchor | sgd | 0.1 | 0.0005 | 0.9586 | 0.9585 | 0.0298 | 0.9192 | 0.9300 | 0.0108 | 0.9165 | 0.9236 | 0.0489 | 16.7966 | anchor |

## 핵심 비교 후보: Adam wd=0 vs wd=1e-4
| 후보명 | optimizer | LR | WD | best val acc | ID test acc | ECE | Maha mean AUROC | Maha L2 mean AUROC | Maha L2-raw | kNN mean AUROC | kNN L2 mean AUROC | NC1 | class mean dist | 분석태그 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adam_lr1e-3_wd0 | adam | 0.001 | 0.0 | 0.9524 | 0.9436 | 0.0473 | 0.6764 | 0.9351 | 0.2588 | 0.8730 | 0.9446 | 0.2279 | 10.5331 | Mahalanobis L2 회복 큼, NC1 약함 |
| adam_lr1e-3_wd1e-4 | adam | 0.001 | 0.0001 | 0.9494 | 0.9447 | 0.0390 | 0.5531 | 0.8148 | 0.2617 | 0.8692 | 0.8904 | 0.1874 | 13.4176 | Mahalanobis L2 회복 큼 |

## 핵심 비교 후보: AdamW lr=5e-3 WD 축
| 후보명 | optimizer | LR | WD | best val acc | ID test acc | ECE | Maha mean AUROC | Maha L2 mean AUROC | Maha L2-raw | kNN mean AUROC | kNN L2 mean AUROC | NC1 | class mean dist | 분석태그 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adamw_lr5e-3_wd1e-4 | adamw | 0.005 | 0.0001 | 0.9528 | 0.9468 | 0.0451 | 0.4719 | 0.9226 | 0.4508 | 0.6538 | 0.9418 | 0.2713 | 5.3713 | Mahalanobis L2 회복 큼, kNN L2 회복 큼, NC1 약함, covariance ill-conditioned |
| adamw_lr5e-3_wd5e-4_anchor | adamw | 0.005 | 0.0005 | 0.9502 | 0.9437 | 0.0479 | 0.5311 | 0.9237 | 0.3926 | 0.6703 | 0.9411 | 0.2747 | 5.5291 | Mahalanobis L2 회복 큼, kNN L2 회복 큼, NC1 약함, covariance ill-conditioned, anchor |
| adamw_lr5e-3_wd1e-3 | adamw | 0.005 | 0.001 | 0.9520 | 0.9452 | 0.0466 | 0.5470 | 0.9278 | 0.3808 | 0.7086 | 0.9443 | 0.2570 | 5.6823 | Mahalanobis L2 회복 큼, kNN L2 회복 큼, NC1 약함, covariance ill-conditioned |

## 교수님께 물어볼 질문 후보
- 본 grid에서 validation 기준 후보를 고정한 뒤 seed1/2 반복으로 넘어갈지?
- Mahalanobis raw vs L2 회복 패턴을 main story의 norm/covariance-scale diagnostic으로 둘지?
- AdamW decoupled WD 축을 별도 subsection으로 키울지, appendix diagnostic으로 둘지?
