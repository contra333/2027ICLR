# WRN350 Seed0 Evaluation Summary

Created: 2026-05-25 KST

Scope: CIFAR-10 `standard_wrn_28_10_dropout03`, final checkpoint evaluation, seed 0 only. No averaging across seeds or repeated runs is used.

## Imported Runs

| Model | Run ID | Raw import | Original run path |
|---|---|---|---|
| SGD Nesterov | `20260519_1954_hypatia_cifar10_standard-wrn-28-10-dropout03_sgd-nesterov_seed0` | `results/raw/wrn350_seed0_eval_20260525/20260519_1954_hypatia_cifar10_standard-wrn-28-10-dropout03_sgd-nesterov_seed0` | `/home/ghjin/iclr2027_runs/transfer_sgd_seed0_final/20260519_1954_hypatia_cifar10_standard-wrn-28-10-dropout03_sgd-nesterov_seed0` |
| Adam matched | `20260520_2209_curie_cifar10_standard-wrn-28-10-dropout03_adam_matched_seed0` | `results/raw/wrn350_seed0_eval_20260525/20260520_2209_curie_cifar10_standard-wrn-28-10-dropout03_adam_matched_seed0` | `/home/ghjin/iclr2027_runs/20260520_2209_curie_cifar10_standard-wrn-28-10-dropout03_adam_matched_seed0` |
| AdamW matched | `20260521_1104_curie_cifar10_standard-wrn-28-10-dropout03_adamw_matched_seed0` | `results/raw/wrn350_seed0_eval_20260525/20260521_1104_curie_cifar10_standard-wrn-28-10-dropout03_adamw_matched_seed0` | `/home/ghjin/iclr2027_runs/20260521_1104_curie_cifar10_standard-wrn-28-10-dropout03_adamw_matched_seed0` |

## ID Classification And Calibration

| Model | Epochs | Best val acc (%) | Best val epoch | Final val acc (%) | ID test acc (%) | ID test NLL | ECE 15-bin | Temp-scaled ECE | Temp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD Nesterov | 350 | 95.86 | 342 | 95.70 | 95.85 | 0.2068 | 0.0298 | 0.0066 | 1.9264 |
| Adam matched | 350 | 86.72 | 294 | 86.28 | 87.24 | 0.3973 | 0.0263 | 0.0108 | 1.2825 |
| AdamW matched | 350 | 95.02 | 192 | 94.62 | 94.37 | 0.5418 | 0.0479 | 0.0062 | 5.2053 |

Observation: on seed0, SGD Nesterov has the highest ID test accuracy, followed by AdamW matched, then Adam matched. AdamW is much stronger than Adam under this matched setting.

## Best OOD AUROC By Detector Family

Each cell shows `detector AUROC / FPR95`. Best is selected within each detector family for that model and OOD dataset.

### Logit Detectors

| Model | cifar100 | tiny_imagenet | svhn | mnist |
|---|---:|---:|---:|---:|
| SGD Nesterov | `neg_entropy` 0.8725 / 0.5575 | `neg_entropy` 0.8649 / 0.5484 | `energy_id_score` 0.9657 / 0.1412 | `energy_id_score` 0.8914 / 0.4591 |
| Adam matched | `maxlogit` 0.8225 / 0.7038 | `maxlogit` 0.8266 / 0.6851 | `neg_entropy` 0.9289 / 0.4626 | `energy_id_score` 0.9868 / 0.0554 |
| AdamW matched | `maxlogit` 0.9011 / 0.5323 | `energy_id_score` 0.8989 / 0.5043 | `neg_entropy` 0.9291 / 0.5583 | `neg_entropy` 0.9498 / 0.3851 |

### Feature Detectors

| Model | cifar100 | tiny_imagenet | svhn | mnist |
|---|---:|---:|---:|---:|
| SGD Nesterov | `gmm_ddu_shrinkage` 0.9093 / 0.4832 | `gmm_ddu_shrinkage` 0.9024 / 0.4745 | `mahalanobis_l2` 0.9896 / 0.0610 | `mahalanobis_l2` 0.9589 / 0.2618 |
| Adam matched | `knn_l2` 0.7767 / 0.7231 | `knn_l2` 0.7741 / 0.7489 | `mahalanobis_l2` 0.9788 / 0.1075 | `mahalanobis_l2` 0.9974 / 0.0029 |
| AdamW matched | `knn_l2` 0.8904 / 0.4967 | `knn_l2` 0.8828 / 0.4984 | `knn_l2` 0.9931 / 0.0350 | `mahalanobis_l2` 0.9989 / 0.0000 |

## Geometry Scalars

| Model | within_var | inter_dist_l2 | nc1 | nc2_mean_etf | nc2_weight_etf | nc3_cos_alignment | nc4_agreement | effective_rank | anisotropy_lambda1_trace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD Nesterov | 18.0100 | 16.7966 | 0.0489 | 0.0023 | 0.0012 | 0.9470 | 0.9949 | 61.5627 | 0.0852 |
| Adam matched | 21.2002 | 7.0196 | 0.9270 | 0.0089 | 0.0032 | 0.7775 | 0.8541 | 20.9187 | 0.1349 |
| AdamW matched | 11.1569 | 5.5291 | 0.2747 | 0.0054 | 0.0033 | 0.6206 | 0.9616 | 75.7809 | 0.0774 |

## Processed Files

- `results/processed/wrn350_seed0_classification_calibration_20260525.csv`
- `results/processed/wrn350_seed0_ood_metrics_20260525.csv`
- `results/processed/wrn350_seed0_geometry_scalars_20260525.csv`
- `results/processed/wrn350_seed0_ood_best_auroc_by_dataset_20260525.csv`
- `results/manifests/wrn350_seed0_eval_20260525.json`

## Notes

- `metrics_ood_nc_hybrid.json` is present for all three runs, but the detector list is empty because `eval.detectors.nc_hybrid` was not requested in the configs.
- Raw metrics/config snapshots were copied under `results/raw/wrn350_seed0_eval_20260525/`; checkpoints were not copied.
- The SGD seed0 eval was generated on 2026-05-25 after the original training transfer; Adam and AdamW eval metrics were already present from 2026-05-21.
