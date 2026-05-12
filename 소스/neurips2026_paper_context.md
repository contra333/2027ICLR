# NeurIPS 2026 Paper Context

이 문서는 `/home/contra333/papers/V7/0_neurips_2026.tex`를 기준으로 컴파일되는 최종 NeurIPS 2026 제출본의 내용을 다른 AI 작업에서 정확히 재사용하기 위한 context 파일이다.

## 기본 메타데이터

- Main TeX: `/home/contra333/papers/V7/0_neurips_2026.tex`
- 컴파일 산출물: `/home/contra333/papers/V7/0_neurips_2026.pdf`
- 논문 제목: **When Calibration Improves but Feature-Based Uncertainty Degrades: Optimizer-Induced Geometry Shifts in Representation Space**
- 저자: Gunhak Jin, Donghyeon Kim, Wonjoon Hwang, Jinhyeok Kim, Hye-Young Jung
- 기관: Department of Mathematical Data Science, Hanyang University
- Corresponding author: Hye-Young Jung, `hyjunglove@hanyang.ac.kr`
- 제출 맥락: 사용자가 NeurIPS 2026에 제출한 최종 버전으로 지정한 원고.

## 연결된 주요 파일

Main에서 직접 `\input`되는 본문 파일:

- `V7/1_introduction_0505.tex`
- `V7/2_relatedwork.tex`
- `V7/3_preliminarlies.tex`
- `V7/4_WhyFBSCollapse_scope.tex`
- `V7/5_experiments.tex`
- `V7/6_Conclusion_toned_down_version.tex`

Main에서 직접 `\input`되는 부록 및 체크리스트:

- `appendix/Appendix_ImpactStatemnet.tex`
- `appendix/Appendix_TrainingDetails.tex`
- `appendix/Appendix_C_Metrics_v2.tex`
- `appendix/Appendix_CR_SAM_Inflation.tex`
- `appendix/Appendix_A_SN_v2.tex`
- `appendix/Appendix_Batchsize.tex`
- `appendix/Appendix_CR_Normality.tex`
- `appendix/Appendix_FeatureRank.tex`
- `appendix/Appendix_B_DirtyMnist_Cifar10Mixup_v2.tex`
- `appendix/Appendix_ASAM_GSAM.tex`
- `appendix/Appendix_CIFAR100_COV_new.tex`
- `appendix/Appendix_Extra_v2.tex`
- `checklist.tex`

부록에서 추가로 연결되는 표 파일:

- `tables/Batchsize_table.tex`
- `tables/CIFAR100_PCA_DDU_only_compact.tex`
- `tables/CIFAR100_Shrinkage_DDU_only_compact.tex`
- `tables/SN_new_geo.tex`
- `tables/SN_new.tex`
- `tables/L2_new_geo.tex`
- `tables/L2_new.tex`

주요 그림 파일:

- `Figures/Mixup_fig1.png`
- `Figures/cifar10_AUROC_crop.png`
- `Figures/norm_angle.png`
- `Figures/eigenspectrum_ASAM_GSAM.png`
- `Figures/L2_AUROC_crop.png`
- `Figures/fig_app/app_delta.png`
- `Figures/spec_full.png`

## 한 문장 요약

이 논문은 vanilla SAM이 accuracy, calibration, MSP/Energy 같은 output-level reliability는 개선하거나 유지할 수 있지만, penultimate feature geometry를 covariance inflation과 within-class dispersion 방향으로 바꾸어 Mahalanobis 및 DDU 같은 feature-based uncertainty/OOD detector를 악화시킬 수 있음을 보인다.

## 핵심 주장

1. Reliability signal은 하나가 아니다. MSP, Energy 같은 logit-based score는 confidence, margin, logit scale에 의존하고, Mahalanobis, DDU 같은 feature-based score는 penultimate representation의 class mean, covariance, density geometry에 의존한다.
2. SAM은 generalization과 calibration을 개선하는 optimizer로 알려져 있지만, 그 개선이 feature-based uncertainty까지 자동으로 보장하지 않는다.
3. Vanilla SAM에서 SAM radius `rho`가 커질수록 ID accuracy는 대체로 유지 또는 개선되고 ECE/TCE는 낮아지지만, Mahalanobis/DDU의 OOD AUROC는 특히 near-OOD에서 악화될 수 있다.
4. 악화의 주요 원인은 softmax confidence 문제가 아니라 penultimate feature geometry 변화다. 구체적으로 within-class dispersion, covariance scale, leading covariance eigenvalues가 커지고, detector-relevant class structure가 약해진다.
5. Covariance inflation은 inverse covariance 기반 penalty를 줄인다. 그래서 OOD sample이 expanded covariance directions에 크게 투영되면 Mahalanobis/DDU score에서 더 ID-like하게 보일 수 있고, ID-OOD score gap이 줄거나 ranking flip이 생길 수 있다.
6. ASAM과 GSAM은 vanilla SAM과 다른 geometry profile을 만들며 feature-based detector 결과도 다르다. 따라서 문제는 "flatness-aware optimization 전체"가 아니라 optimizer variant가 어떤 penultimate geometry를 만들고 detector가 그 geometry에 어떻게 의존하느냐이다.
7. Feature L2 normalization은 norm/covariance-scale effect가 실제로 관련되어 있음을 보여주는 diagnostic intervention이다. 일부 high-rho degradation을 회복하지만, geometry 차이를 완전히 제거하지는 않는다.

## 문제 설정

Open-world deployment에서는 ID accuracy뿐 아니라 다음 두 종류의 uncertainty가 중요하다.

- Ambiguous ID: decision boundary 근처 또는 인간 annotator 사이 label uncertainty가 높은 ID sample.
- OOD input: training distribution 밖의 sample.

논문은 DDU가 지적한 것처럼 logit-based score가 ambiguous ID와 OOD를 잘 구분하지 못할 수 있고, 이때 feature density가 유용할 수 있다는 점에서 출발한다. 그러나 optimizer가 바뀌면 feature density가 의존하는 representation geometry도 바뀌므로, optimizer reliability를 logit-level metric만으로 평가하면 위험하다는 것이 중심 문제다.

## Score 정의

Logit-based score:

- MSP:
  `S_MSP(x) = max_c p_theta(c | x)`
- Energy:
  `S_Energy(x) = T log sum_c exp(z_theta,c(x) / T)`
- 논문 convention: score가 클수록 ID-like.

Feature-based score:

- Penultimate feature: `f = phi_theta(x)`
- Class mean: `mu_c`
- Pooled/tied covariance: `Sigma`
- Class covariance: `Sigma_c`
- Mahalanobis:
  `S_Maha(x) = - min_c (f - mu_c)^T Sigma^{-1} (f - mu_c)`
- DDU:
  `S_DDU(x) = log sum_c pi_c N(f | mu_c, Sigma_c)`
- DDU Gaussian log-density의 covariance-sensitive term:
  `-1/2 (f - mu_c)^T Sigma_c^{-1} (f - mu_c)`

## SAM 설정

SAM objective:

`min_theta max_{||epsilon||_2 <= rho} L(theta + epsilon)`

First-order perturbation:

`epsilon*(theta) approx rho grad_theta L(theta) / ||grad_theta L(theta)||_2`

논문에서는 `rho`를 intervention strength/control parameter로 사용한다.

Main experiment rho sweep:

- `rho in {0 (SGD), 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}`

Training appendix의 더 넓은 sweep:

- `rho in {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7}`

## 주요 실험 설정

Main controlled study:

- ID dataset: CIFAR-10
- Main architecture: WideResNet-28-10
- Additional architectures: ResNet-18, VGG-16
- Training: 350 epochs
- Optimizers: SGD baseline, vanilla SAM, plus ASAM/GSAM variants
- Base LR: 0.1
- Momentum: 0.9
- Weight decay: `5e-4`
- Batch size: 128
- LR schedule: milestones at epochs 150 and 250, each multiplying LR by 0.1
- Spectral normalization: used unless otherwise stated
- Repeats: 3 independent runs; tables report mean +/- std
- OOD datasets:
  - CIFAR-10 ID setting: near-OOD = CIFAR-100, Tiny-ImageNet; far-OOD = MNIST, SVHN
  - CIFAR-100 ID setting: near-OOD = CIFAR-10, Tiny-ImageNet; far-OOD = MNIST, SVHN
- Metrics:
  - Accuracy
  - ECE
  - TCE
  - OOD AUROC
  - FPR@95 in appendices
  - Geometry metrics: WithinVar, InterDist, NC1, NC2, NC2^A, anisotropy, effective rank

Computing:

- Four NVIDIA RTX A6000 GPUs, each used independently
- Each SAM-trained model: about 12 wall-clock hours on one GPU

## Main results: calibration improves

WideResNet-28-10 on CIFAR-10:

| rho | Accuracy | ECE | TCE |
|---:|---:|---:|---:|
| 0 (SGD) | 0.9597 +/- 0.0023 | 0.0241 +/- 0.0029 | 0.0153 +/- 0.0079 |
| 0.01 | 0.9622 +/- 0.0009 | 0.0190 +/- 0.0009 | 0.0137 +/- 0.0043 |
| 0.02 | 0.9618 +/- 0.0020 | 0.0194 +/- 0.0013 | 0.0147 +/- 0.0042 |
| 0.05 | 0.9666 +/- 0.0002 | 0.0146 +/- 0.0003 | 0.0105 +/- 0.0028 |
| 0.1 | 0.9659 +/- 0.0005 | 0.0118 +/- 0.0016 | 0.0093 +/- 0.0030 |
| 0.2 | 0.9680 +/- 0.0018 | 0.0092 +/- 0.0016 | 0.0083 +/- 0.0031 |
| 0.5 | 0.9681 +/- 0.0012 | 0.0097 +/- 0.0010 | 0.0106 +/- 0.0030 |

Interpretation:

- Accuracy is preserved or slightly improved.
- ECE/TCE improve substantially.
- Therefore output-level reliability metrics alone suggest SAM is beneficial.

## Main results: feature-based OOD degrades

Main text Figure `Figures/cifar10_AUROC_crop.png` reports AUROC across `rho` for MSP, Energy, Mahalanobis, and DDU over CIFAR-100, Tiny-ImageNet, MNIST, and SVHN.

Core qualitative pattern:

- MSP and Energy generally improve or remain stable with larger `rho`.
- Mahalanobis and DDU degrade at larger `rho`, especially for near-OOD datasets CIFAR-100 and Tiny-ImageNet.
- Far-OOD datasets are often less sensitive because they are already more separated in feature space.

Representative main-table values for CIFAR-10 ID / CIFAR-100 OOD / WideResNet:

| Optimizer | rho | NC1 | WithinVar | InterDist | DDU AUROC | Maha AUROC |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 0 | 0.0270 +/- 0.0033 | 0.9785 +/- 0.0736 | 7.2723 +/- 0.3743 | 0.9024 +/- 0.0033 | 0.9033 +/- 0.0077 |
| SAM | 0.5 | 0.0684 +/- 0.0043 | 2.5023 +/- 0.0488 | 6.8099 +/- 0.0259 | 0.8779 +/- 0.0103 | 0.8729 +/- 0.0035 |
| ASAM | 0.5 | 0.0298 +/- 0.0042 | 1.1344 +/- 0.1197 | 7.3984 +/- 0.0164 | 0.9068 +/- 0.0050 | 0.9077 +/- 0.0076 |
| GSAM | [0.01, 0.5] | 0.0331 +/- 0.0050 | 1.1985 +/- 0.1000 | 7.1440 +/- 0.0645 | 0.9007 +/- 0.0066 | 0.9167 +/- 0.0031 |

## Geometry mechanism

Main geometry table for CIFAR-10 / WideResNet:

| rho | WithinVar | InterDist | NC1 | NC2 | NC2^A |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9785 +/- 0.0736 | 7.2723 +/- 0.3743 | 0.0270 +/- 0.0033 | 0.0864 +/- 0.0028 | -0.1110 |
| 0.01 | 1.0424 +/- 0.1261 | 7.4130 +/- 0.0279 | 0.0289 +/- 0.0044 | 0.0972 +/- 0.0025 | -0.1110 |
| 0.02 | 1.2023 +/- 0.1652 | 7.4226 +/- 0.0764 | 0.0330 +/- 0.0057 | 0.1070 +/- 0.0024 | -0.1110 |
| 0.05 | 1.3760 +/- 0.1086 | 7.1436 +/- 0.3270 | 0.0392 +/- 0.0021 | 0.1385 +/- 0.0084 | -0.1110 |
| 0.1 | 1.8585 +/- 0.0323 | 7.1617 +/- 0.0621 | 0.0528 +/- 0.0006 | 0.1993 +/- 0.0084 | -0.1109 |
| 0.2 | 2.1112 +/- 0.1081 | 6.9214 +/- 0.0416 | 0.0603 +/- 0.0050 | 0.2769 +/- 0.0012 | -0.1107 |
| 0.5 | 2.5023 +/- 0.0488 | 6.8099 +/- 0.0259 | 0.0684 +/- 0.0043 | 0.3480 +/- 0.0025 | -0.1106 |

Key interpretation:

- From `rho=0` to `rho=0.5`, WithinVar increases from 0.9785 to 2.5023, about 2.56x.
- InterDist decreases from 7.2723 to 6.8099, about 6.4%.
- NC1 increases from 0.0270 to 0.0684, about 2.53x.
- NC2^A stays near the CIFAR-10 simplex ETF value `-1/(10-1) = -0.111...`.
- Therefore the dominant geometry change is within-class expansion/covariance scale increase, not collapse of class means or loss of class-mean angular simplex structure.

Figure `Figures/norm_angle.png`:

- Vanilla SAM produces broader feature-norm distributions and larger angular deviations from class means than SGD.

Figure `Figures/eigenspectrum_ASAM_GSAM.png`:

- Vanilla SAM inflates multiple leading covariance eigenvalues.
- ASAM and GSAM stay closer to SGD.

Figure `Figures/spec_full.png`:

- Additional rho eigenspectra up to 95% explained variance show stronger leading-eigenvalue growth as rho increases.

## Formal covariance-inflation argument

Assumption:

- Detector-relevant covariance expansion:
  `Sigma^(rho) >= Sigma^(0)` and/or `Sigma_c^(rho) >= Sigma_c^(0)` in PSD order.
- This is an analysis condition, not a universal claim that SAM always induces PSD covariance expansion.

Lemma:

- If covariance expands, precision shrinks:
  `(Sigma^(rho))^{-1} <= (Sigma^(0))^{-1}`
  and similarly for class covariances.
- Consequently, for any displacement `delta`, the precision-weighted quadratic penalty decreases:
  `delta^T (Sigma^(rho))^{-1} delta <= delta^T (Sigma^(0))^{-1} delta`.

Effect on Mahalanobis:

- Mahalanobis distance term decreases.
- Since `S_Maha = - distance`, the sample becomes more ID-like under the score.

Effect on DDU:

- The quadratic part of the class Gaussian log-likelihood becomes less negative.
- The log-determinant term is input-independent for fixed covariance component, but the quadratic shift is sample-dependent.

Directional score shift:

- In whitened coordinates:
  `Sigma^(rho) = Sigma^(0)^{1/2} U diag(beta_i) U^T Sigma^(0)^{1/2}` with `beta_i >= 1`.
- Quadratic becomes:
  `sum_i delta_tilde_i^2 / beta_i`.
- Reduction:
  `sum_i (1 - 1/beta_i) delta_tilde_i^2`.
- Therefore samples aligned with expanded directions get larger score shifts. An OOD sample can move more toward ID-like score than an ID sample, producing score-gap shrinkage or ranking flips.

Local SAM perturbation view:

- Feature linearization under SAM:
  `f'(x) approx f(x) + Delta(x)`, `Delta(x) = J_theta(x) epsilon*`.
- Class covariance update:
  `Sigma_c' - Sigma_c approx A_c + A_c^T + B_c`
- `B_c` is PSD because it is an empirical covariance of perturbation deviations:
  `B_c = (1/n_c) sum_i (Delta_i - Delta_bar_c)(Delta_i - Delta_bar_c)^T`
- `A_c + A_c^T` is indefinite, so this is not a global proof of covariance inflation. It is a local mechanism explaining why SAM perturbations can add a covariance-expansion component.

## L2 normalization diagnostic

Motivation:

- Mahalanobis++ suggests feature-norm heterogeneity can cause covariance-related failures.
- The paper uses post-hoc L2 normalization as a diagnostic, not as a proposed universal detector fix.

Main geometry comparison for WideResNet:

| rho | L2 | WithinVar | InterDist | NC1 | NC2 | NC2^A |
|---:|---|---:|---:|---:|---:|---:|
| 0 | X | 0.9785 +/- 0.0736 | 7.2723 +/- 0.3743 | 0.0270 +/- 0.0033 | 0.0864 +/- 0.0028 | -0.1110 |
| 0 | O | 0.0198 +/- 0.0022 | 1.1827 +/- 0.0968 | 0.0210 +/- 0.0044 | 0.0864 +/- 0.0028 | -0.1109 |
| 0.5 | X | 2.5023 +/- 0.0488 | 6.8099 +/- 0.0259 | 0.0684 +/- 0.0043 | 0.3480 +/- 0.0025 | -0.1106 |
| 0.5 | O | 0.0642 +/- 0.0015 | 1.1320 +/- 0.0072 | 0.0589 +/- 0.0044 | 0.3480 +/- 0.0025 | -0.1108 |

Key interpretation:

- L2 normalization reduces feature scale effects and often improves high-rho feature-based AUROC.
- However, NC1 remains higher under SAM even after normalization.
- Therefore norm/covariance scale is a major contributor but not the full explanation.

Representative L2 ablation values for WideResNet:

- CIFAR-100 OOD, DDU:
  - rho 0: L2O 0.9107, L2X 0.9024
  - rho 0.5: L2O 0.9192, L2X 0.8779
- CIFAR-100 OOD, Mahalanobis:
  - rho 0: L2O 0.9122, L2X 0.9033
  - rho 0.5: L2O 0.9211, L2X 0.8729
- Tiny-ImageNet OOD, DDU:
  - rho 0: L2O 0.9072, L2X 0.9041
  - rho 0.5: L2O 0.9165, L2X 0.8725
- Tiny-ImageNet OOD, Mahalanobis:
  - rho 0: L2O 0.9071, L2X 0.9008
  - rho 0.5: L2O 0.9201, L2X 0.8775

## ASAM and GSAM

ASAM:

- Addresses scale-dependency of vanilla SAM.
- Uses weight-adaptive ellipsoidal perturbation with `T_w = diag(|w_i| + eta)`.
- Perturbation:
  `epsilon_hat_t = rho T_w_t^2 grad L_S(w_t) / ||T_w_t grad L_S(w_t)||_2`.

GSAM:

- Uses perturbed loss `f_p(w) = max_{||delta|| <= rho} f(w + delta)`.
- Defines surrogate gap `h(w) = f_p(w) - f(w)`.
- Decomposes `grad f(w)` into component orthogonal to `grad f_p(w)`.
- Uses surrogate gradient:
  `grad f^GSAM = grad f_p(w) - alpha grad f_perp(w)`.

Extended ASAM/GSAM result:

- ASAM maintains WithinVar and NC1 close to SGD across rho, and DDU AUROC remains stable.
- GSAM also remains relatively stable; Mahalanobis can improve in some settings.
- This supports the claim that detector outcomes are optimizer-variant-specific.

Representative extended table values:

| Optimizer | rho | WithinVar | InterDist | NC1 | DDU AUROC | Maha AUROC |
|---|---:|---:|---:|---:|---:|---:|
| SAM | 0.5 | 2.5023 | 6.8099 | 0.0684 | 0.8779 | 0.8729 |
| ASAM | 0.5 | 1.1344 | 7.3984 | 0.0298 | 0.9068 | 0.9077 |
| GSAM | 0.5 | 1.1985 | 7.1440 | 0.0331 | 0.9007 | 0.9167 |

## Dirty-MNIST and CIFAR-10 Mixup ambiguity stress tests

Dirty-MNIST:

- Model: ResNet-18 with spectral normalization.
- Optimizer: SGD vs SAM with `rho=0.05` and `rho=0.50`.
- Clean + ambiguous ID vs OOD:
  - SGD AUROC 0.9990, FPR@95 0.0006, overlap 0.0251
  - SAM 0.05 AUROC 0.9988, FPR@95 0.0009, overlap 0.0265
  - SAM 0.50 AUROC 0.9987, FPR@95 0.0013, overlap 0.0278
- Ambiguous ID vs OOD:
  - SGD AUROC 0.9995
  - SAM 0.05 AUROC 0.9993
  - SAM 0.50 AUROC 0.9993
- Interpretation: Dirty-MNIST is too saturated/well separated to show strong degradation.

CIFAR-10 Mixup extension:

- Training: CIFAR-10 Train with clean 50% + Mixup 50%.
- Evaluation:
  - Clean ID: CIFAR-10H low entropy / 60%
  - Ambiguous ID: CIFAR-10H high entropy / 20%
  - OOD: SVHN
- Model: ResNet-18 with spectral normalization.
- Trained 100 epochs.
- It is a stress test distinct from main CIFAR-10 OOD benchmark.

CIFAR-10 Mixup, ID (clean + ambiguous) vs OOD:

| rho | AUROC | FPR@95 | Overlap |
|---:|---:|---:|---:|
| 0 | 0.9798 +/- 0.0036 | 0.0948 +/- 0.0095 | 0.1256 +/- 0.0036 |
| 0.05 | 0.9823 +/- 0.0028 | 0.0879 +/- 0.0028 | 0.1223 +/- 0.0010 |
| 0.2 | 0.9698 +/- 0.0080 | 0.1687 +/- 0.0676 | 0.1588 +/- 0.0228 |
| 0.5 | 0.9388 +/- 0.0030 | 0.3802 +/- 0.0223 | 0.2484 +/- 0.0123 |

CIFAR-10 Mixup, ambiguous ID vs OOD:

| rho | AUROC | FPR@95 | Overlap |
|---:|---:|---:|---:|
| 0 | 0.9514 +/- 0.0092 | 0.2941 +/- 0.0823 | 0.2213 +/- 0.0113 |
| 0.05 | 0.9572 +/- 0.0070 | 0.2652 +/- 0.0609 | 0.2124 +/- 0.0069 |
| 0.2 | 0.9337 +/- 0.0206 | 0.3836 +/- 0.1186 | 0.2575 +/- 0.0500 |
| 0.5 | 0.8964 +/- 0.0056 | 0.5394 +/- 0.0395 | 0.3563 +/- 0.0194 |

Interpretation:

- Low/intermediate rho can be comparable or slightly better than SGD.
- High rho reproduces the main qualitative behavior: density-based separation becomes worse.
- This supports the claim that strong SAM intervention can destabilize feature-density separation under ambiguity-rich settings.

## Batch-size control

Purpose:

- Test whether the main SAM geometry/OOD trends are artifacts of batch size 128.
- Additional batch sizes: 64 and 256.
- Architecture: WideResNet-28-10.
- rho values: 0, 0.05, 0.50.

OOD AUROC on CIFAR-10 ID / CIFAR-100 OOD:

| Batch | Metric | rho 0 | rho 0.05 | rho 0.50 |
|---:|---|---:|---:|---:|
| 64 | DDU | 0.8736 | 0.8717 | 0.8311 |
| 64 | Maha | 0.8758 | 0.8636 | 0.7524 |
| 128 | DDU | 0.9024 | 0.9060 | 0.8779 |
| 128 | Maha | 0.9033 | 0.9073 | 0.8729 |
| 256 | DDU | 0.9049 | 0.9028 | 0.8887 |
| 256 | Maha | 0.8845 | 0.8826 | 0.8887 |

Geometry:

| Batch | Metric | rho 0 | rho 0.05 | rho 0.50 |
|---:|---|---:|---:|---:|
| 64 | WithinVar | 1.8486 | 2.2888 | 3.8302 |
| 64 | InterDist | 7.1086 | 6.9530 | 5.7722 |
| 64 | NC1 | 0.0547 | 0.0698 | 0.1336 |
| 128 | WithinVar | 0.9785 | 1.3760 | 2.5023 |
| 128 | InterDist | 7.2723 | 7.1436 | 6.8099 |
| 128 | NC1 | 0.0270 | 0.0392 | 0.0684 |
| 256 | WithinVar | 0.6056 | 1.0363 | 1.6964 |
| 256 | InterDist | 7.6525 | 7.7099 | 7.4354 |
| 256 | NC1 | 0.0166 | 0.0281 | 0.0410 |

Interpretation:

- Larger rho generally increases WithinVar and NC1 across all batch sizes.
- Feature-based AUROC often drops at rho 0.50, though detailed monotonicity varies.
- Trends are not solely a batch-size artifact.

## Normality diagnostics

Purpose:

- Mahalanobis/DDU assume Gaussian feature behavior.
- The paper tests whether degradation is simply caused by less Gaussian features.

Protocol:

- CIFAR-10 / WideResNet-28-10.
- PCA coordinates retained up to 95% cumulative explained variance.
- Compute Q-Q correlation and Wasserstein distance to standard normal after z-standardization.

Results:

| rho | Q-Q correlation | Wasserstein distance |
|---:|---:|---:|
| 0 | 0.9648 | 0.1387 |
| 0.01 | 0.9653 | 0.1400 |
| 0.02 | 0.9677 | 0.1250 |
| 0.05 | 0.9788 | 0.0961 |
| 0.1 | 0.9866 | 0.0866 |
| 0.2 | 0.9899 | 0.0701 |
| 0.5 | 0.9954 | 0.0508 |

Interpretation:

- As rho increases, marginal normality diagnostics improve rather than degrade.
- Therefore feature-based OOD degradation is unlikely to be explained by a simple Gaussianity failure.

## Feature rank diagnostics

Purpose:

- Relate the paper to prior work that SAM leads to low-rank features.
- Feature-based uncertainty uses penultimate representation, so rank is checked at intermediate and penultimate layers.

Setting:

- CIFAR-10 / ResNet-18
- With and without spectral normalization
- Feature rank = minimum PCA components explaining 99% total variance

Key values:

| rho | SN(O) intermediate | SN(O) penultimate | SN(X) intermediate | SN(X) penultimate |
|---:|---:|---:|---:|---:|
| 0.00 | 7756.3 | 19.3 | 7760.0 | 19.7 |
| 0.50 | 7232.3 | 65.7 | 7066.7 | 59.3 |
| 0.70 | 6981.3 | 79.7 | 6971.3 | 86.0 |

Interpretation:

- Intermediate-layer rank decreases as rho increases.
- Penultimate-layer rank increases substantially in large-rho regimes.
- SAM does not affect representation dimensionality uniformly across layers.
- This supports the main focus on penultimate geometry.

## Spectral normalization and residual-connection ablation

Purpose:

- Test whether geometry changes are mainly caused by spectral normalization or residual connections rather than SAM.

Setup:

- Vary spectral normalization on/off.
- Compare architectures:
  - Residual: WideResNet-28-10, ResNet-18
  - Non-residual: VGG-16
- Sweep rho.

Main interpretation:

- Absolute metric values vary with SN and architecture.
- Overall trends with rho remain largely consistent.
- The geometry changes cannot be fully attributed to SN or residual connections alone.

Representative SN ablation:

- WideResNet, WithinVar with SN on: 0.9785 at rho 0 to 2.5023 at rho 0.5.
- WideResNet, WithinVar with SN off: 1.0753 at rho 0 to 2.4563 at rho 0.5.
- WideResNet, NC1 with SN on: 0.0270 to 0.0684.
- WideResNet, NC1 with SN off: 0.0297 to 0.0685.

## CIFAR-100 high-class-count covariance-estimation control

Purpose:

- CIFAR-100 has many classes and fewer samples per class relative to feature dimension.
- Empirical class covariance can be singular or poorly conditioned.
- DDU scores in this setting reflect both representation geometry and detector-side covariance-estimation procedure.

Controls:

1. PCA-256 before covariance estimation, with covariance shrinkage disabled.
2. No-PCA shrinkage selection by validation NLL, with selected `alpha_DDU = 0.05`.

Important interpretation:

- PCA/shrinkage results are covariance-estimation controls, not primary evidence that SAM geometry does or does not degrade.
- Recovery under PCA or shrinkage means detector-side covariance estimation changed the measured DDU behavior.

PCA-256 representative DDU AUROC:

| Model | OOD | rho 0.00 | rho 0.05 | rho 0.50 |
|---|---|---:|---:|---:|
| WideResNet | CIFAR-10 | 0.7186 | 0.7197 | 0.7449 |
| WideResNet | Tiny-ImageNet | 0.7775 | 0.7851 | 0.7898 |
| WideResNet | MNIST | 0.6164 | 0.6459 | 0.7006 |
| WideResNet | SVHN | 0.8742 | 0.8797 | 0.8553 |
| ResNet-18 | CIFAR-10 | 0.6634 | 0.6854 | 0.6959 |
| ResNet-18 | Tiny-ImageNet | 0.7392 | 0.7537 | 0.7475 |
| VGG-16 | CIFAR-10 | 0.7503 | 0.7565 | 0.7549 |
| VGG-16 | Tiny-ImageNet | 0.7860 | 0.7864 | 0.7509 |

No-PCA shrinkage-selection representative DDU AUROC:

| Model | OOD | rho 0.00 | rho 0.05 | rho 0.50 |
|---|---|---:|---:|---:|
| WideResNet | CIFAR-10 | 0.7219 | 0.7248 | 0.7467 |
| WideResNet | Tiny-ImageNet | 0.7947 | 0.8043 | 0.8101 |
| WideResNet | MNIST | 0.6475 | 0.7079 | 0.7241 |
| WideResNet | SVHN | 0.8521 | 0.8586 | 0.8286 |
| ResNet-18 | CIFAR-10 | 0.6804 | 0.7002 | 0.7135 |
| ResNet-18 | Tiny-ImageNet | 0.7657 | 0.7757 | 0.7655 |
| VGG-16 | CIFAR-10 | 0.7507 | 0.7584 | 0.7530 |
| VGG-16 | Tiny-ImageNet | 0.7928 | 0.7974 | 0.7603 |

## Limitations

- Main analysis is CIFAR-scale and uses standard convolutional architectures.
- No claim of universality for ImageNet-scale, foundation models, pretrained regimes, or vision transformers.
- Optimizer comparison focuses on SAM family; it does not cover all optimizers/training procedures.
- CIFAR-100 high-class-count DDU requires covariance-estimation controls; PCA/shrinkage can confound raw optimizer geometry with detector preprocessing.
- SAM is not compute-equivalent to SGD because it requires an additional ascent step.
- Gaussian modeling in Mahalanobis/DDU is a tractable detector assumption, not a claim that penultimate features are exactly Gaussian.

## Practical guidance from the paper

Do not assume that accuracy, calibration, MSP, or Energy improvements certify feature-based uncertainty. If deployment uses Mahalanobis, DDU, or another distance/density feature-space detector, evaluate:

- within-class dispersion
- class-mean separation
- covariance spectra
- feature-norm variation
- detector-specific score distributions
- interaction between optimizer and detector

The correct reliability unit is not just the model or optimizer alone, but the tuple:

`optimizer -> learned representation geometry -> downstream uncertainty estimator`

## Useful phrasing for future AI responses

- "The paper uses vanilla SAM as a controlled case where output-level reliability and feature-based uncertainty visibly decouple."
- "The degradation is not claimed to be universal across all flatness-aware optimizers; ASAM and GSAM show different geometry and detector outcomes."
- "Covariance inflation is analyzed as a conditional mechanism: if detector-relevant covariance expands, inverse-covariance penalties shrink, and OOD samples aligned with expanded directions can become more ID-like under feature scores."
- "L2 normalization partially recovers high-rho degradation, showing norm/covariance-scale effects are important, but it does not erase residual geometry differences."
- "CIFAR-100-trained DDU results should be read as covariance-estimation controls because detector-side PCA/shrinkage changes the geometry seen by the Gaussian density model."

## Compilation status

The paper was compiled with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=V7 V7/0_neurips_2026.tex
```

Result:

- PDF generated: `V7/0_neurips_2026.pdf`
- Pages: 53
- Size: about 8.3 MB
- Final log has no unresolved citation/reference errors.
- Remaining warnings are non-fatal: float specifier changes, underfull boxes, and hyperref bookmark warnings for math tokens.
