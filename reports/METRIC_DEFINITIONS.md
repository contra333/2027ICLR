# Metric Definitions for ICLR 2027 Evaluation

Last updated: 2026-05-12 KST

## Purpose

This file is the metric contract for Codex CLI, server-side agents, web GPT, and human readers.

The project is not trying to reproduce every OpenOOD method. The goal is to measure how an optimizer-induced representation geometry regime is read by different detector families. Every server evaluator should follow this document unless a task explicitly says otherwise.

## Global Evaluation Contract

### Shared cache requirement

For one trained checkpoint, extract one shared cache and run all post-hoc detectors on that cache:

- `id_train`: logits, penultimate features, labels.
- `id_val`: logits, penultimate features, labels.
- `id_test`: logits, penultimate features, labels.
- `ood_test/<ood_name>`: logits, penultimate features, labels if available.
- `classifier`: last-layer weight `W` and bias if present.
- `metadata`: dataset, model, optimizer, seed, checkpoint type, config path, Git commit, feature layer.

Do not let detector differences be caused by different feature extraction code, stochastic transforms, batch ordering, or hidden preprocessing.

### OOD score convention

All stored detector scores must follow this convention:

- higher score = more ID-like
- ID label = 1
- OOD label = 0

If a method natively outputs an uncertainty or OOD score, store the sign-flipped ID-like version and record the native direction in `detector_params.json`.

### Output files per run

Each evaluated run should write:

- `metrics_classification.json`
- `metrics_calibration.json`
- `metrics_ood_logit.json`
- `metrics_ood_feature.json`
- `metrics_ood_nc_hybrid.json`
- `metrics_geometry.json`
- `detector_params.json`
- `feature_stats.json`

Small aggregate summaries can also be produced, but these split files are the source of truth.

## Classification Metrics

### `accuracy`

- Input: ID test logits and labels.
- Fitting split: none.
- Formula: `mean(argmax_c z_c == y)`.
- Score direction: higher is better.
- Output name: `accuracy`.
- Output value: scalar in `[0, 1]`.
- Tuning rule: none.

### `nll`

- Input: ID test logits and labels.
- Fitting split: none.
- Formula: cross-entropy / negative log-likelihood on ID test.
- Score direction: lower is better.
- Output name: `nll`.
- Output value: scalar.
- Tuning rule: none.

## Calibration Metrics

### `ece_15bin`

- Input: ID test logits and labels.
- Fitting split: none.
- Confidence: `max_c softmax(z)_c`.
- Prediction: `argmax_c z_c`.
- Formula: split confidence into 15 bins and compute `sum_b (n_b / N) * |acc_b - conf_b|`.
- Score direction: lower is better.
- Output name: `ece_15bin`.
- Output value: scalar.
- Tuning rule: none.

### `temperature_scaled_ece_15bin`

- Input: ID validation logits/labels for fitting temperature, ID test logits/labels for reporting.
- Fitting split: ID validation only for scalar temperature `T`.
- Formula: fit scalar temperature `T` on ID validation, then compute 15-bin ECE on `z / T`.
- Score direction: lower is better.
- Output name: `temperature_scaled_ece_15bin`.
- Output value: scalar.
- Tuning rule: fit temperature on ID validation only.
- Naming rule: use this name instead of ambiguous legacy names such as `TCE` or `t_ece`.

## OOD Aggregate Metrics

### `auroc`

- Input: ID-like detector scores for ID test and one OOD test set.
- Fitting split: none; this is an aggregate metric over stored scores.
- Labels: ID = 1, OOD = 0.
- Formula: ROC-AUC with ID as the positive class.
- Score direction: higher is better.
- Output name: `auroc`.
- Output value: scalar in `[0, 1]`.
- Tuning rule: none; never tune detector hyperparameters using this OOD test value.

### `fpr95`

- Input: ID-like detector scores for ID test and one OOD test set.
- Fitting split: none; this is an aggregate metric over stored scores.
- Labels: ID = 1, OOD = 0.
- Formula: choose the threshold where ID true positive rate is 95%; report the fraction of OOD samples incorrectly accepted as ID.
- Score direction: lower is better.
- Output name: `fpr95`.
- Output value: scalar in `[0, 1]`.
- Tuning rule: none; threshold is derived from ID test scores for reporting, not selected as a deployment hyperparameter.

### `aupr_in`

- Input: ID-like detector scores for ID test and one OOD test set.
- Fitting split: none; this is an aggregate metric over stored scores.
- Labels: ID = 1, OOD = 0.
- Formula: average precision with ID as the positive class.
- Score direction: higher is better.
- Output name: `aupr_in`.
- Output value: scalar in `[0, 1]`.
- Tuning rule: none; never tune detector hyperparameters using this OOD test value.

## Logit Detectors

### `msp`

- Family: logit / softmax.
- Input: logits `z`.
- Formula: `S_msp(x) = max_c softmax(z)_c`.
- Score direction: higher = ID-like.
- Fitting split: none.
- Output name: `msp`.
- Tuning rule: none.

### `maxlogit` / `mls`

- Family: logit.
- Input: logits `z`.
- Formula: `S_mls(x) = max_c z_c`.
- Score direction: higher = ID-like.
- Fitting split: none.
- Output name: `maxlogit`; alias `mls` is allowed only as a duplicate alias, not a separate metric with different semantics.
- Tuning rule: none.

### `energy_id_score`

- Family: logit.
- Input: logits `z`, temperature `T`.
- Formula: `S_energy_id(x) = T * logsumexp(z / T)`.
- Score direction: higher = ID-like.
- Fitting split: none.
- Output name: `energy_id_score`.
- Tuning rule: use fixed `T = 1` for main runs unless a task explicitly requests ID-validation temperature tuning.
- Warning: some papers define energy with the opposite sign. Store this project ID-like version.

### `neg_entropy`

- Family: logit / softmax uncertainty.
- Input: logits `z`.
- Formula: `S_neg_entropy(x) = -H(softmax(z))`.
- Score direction: higher = ID-like.
- Fitting split: none.
- Output name: `neg_entropy`.
- Tuning rule: none.
- Warning: raw entropy is an uncertainty score where higher means less ID-like. Do not store raw entropy as an ID-like detector score unless the name clearly says `raw_entropy_uncertainty`.

## Feature Detectors

Let `f = phi(x)` be the penultimate feature, `mu_c` the ID train class mean, `Sigma` a tied covariance, and `Sigma_c` a class covariance.

### `mahalanobis`

- Family: feature distance / covariance.
- Input: penultimate features.
- Fitting split: ID train features and labels.
- Formula: `S_maha(x) = - min_c (f - mu_c)^T Sigma^{-1} (f - mu_c)`.
- Score direction: higher = ID-like.
- Output name: `mahalanobis`.
- Tuning rule: no OOD tuning. If shrinkage is used, store it as a separate detector name.

### `mahalanobis_l2`

- Family: normalized feature distance; Mahalanobis++ style.
- Input: L2-normalized penultimate features.
- Fitting split: L2-normalized ID train features and labels.
- Formula: same as `mahalanobis` after `f <- f / ||f||_2`.
- Score direction: higher = ID-like.
- Output name: `mahalanobis_l2`.
- Tuning rule: none for the main version.
- Purpose: controls feature-norm / covariance-scale effects.

### `knn`

- Family: non-parametric feature distance.
- Input: penultimate features.
- Fitting split: ID train features.
- Formula: `S_knn(x) = - d_k(f, F_train)`, where `d_k` is the Euclidean distance to the kth nearest ID train feature.
- Score direction: higher = ID-like.
- Output name: `knn`.
- Main default: `k = 50`.
- Appendix grid: `k in {10, 20, 50, 100}`.
- Tuning rule: do not tune on OOD test.

### `knn_l2`

- Family: normalized non-parametric feature distance.
- Input: L2-normalized penultimate features.
- Fitting split: L2-normalized ID train features.
- Formula: same as `knn` after feature normalization.
- Score direction: higher = ID-like.
- Output name: `knn_l2`.
- Main default: `k = 50`.
- Tuning rule: do not tune on OOD test; use the same predeclared `k` rule as `knn`.

### `gmm_ddu_full`

- Family: feature density.
- Input: penultimate features.
- Fitting split: ID train features and labels.
- Formula: class-wise full-covariance Gaussian mixture; score is `logsumexp_c log pi_c N(f | mu_c, Sigma_c)`.
- Score direction: higher = ID-like.
- Output name: `gmm_ddu_full`.
- Tuning rule: none, except numerical jitter recorded in `detector_params.json`.
- Warning: can be singular or unstable for CIFAR-100, ViT, or high-dimensional pretrained features. Treat as diagnostic when covariance estimation is ill-conditioned.

### `gmm_ddu_tied`

- Family: feature density / tied covariance.
- Input: penultimate features.
- Fitting split: ID train features and labels.
- Formula: Gaussian mixture using class means and a shared covariance.
- Score direction: higher = ID-like.
- Output name: `gmm_ddu_tied`.
- Tuning rule: no OOD tuning; log numerical jitter or covariance regularization.
- Purpose: stable density baseline close to Mahalanobis assumptions.

### `gmm_ddu_diag`

- Family: feature density / diagonal covariance.
- Input: penultimate features.
- Fitting split: ID train features and labels.
- Formula: class-wise Gaussian mixture with diagonal covariance.
- Score direction: higher = ID-like.
- Output name: `gmm_ddu_diag`.
- Tuning rule: no OOD tuning; log variance floor or numerical jitter.
- Purpose: high-dimensional covariance stability control.

### `gmm_ddu_shrinkage`

- Family: feature density / covariance-estimation control.
- Input: penultimate features.
- Fitting split: ID train features and labels.
- Formula: GMM with shrinkage covariance.
- Score direction: higher = ID-like.
- Output name: `gmm_ddu_shrinkage`.
- Tuning rule: fixed grid or ID validation likelihood only. Never choose shrinkage using OOD test AUROC.

### `gmm_ddu_pca`

- Family: feature density / preprocessing control.
- Input: PCA-projected penultimate features.
- Fitting split: PCA basis from ID train features only, then GMM on projected ID train features.
- Formula: GMM score in PCA feature space.
- Score direction: higher = ID-like.
- Output name: `gmm_ddu_pca`.
- Tuning rule: fixed PCA dimension such as `128`, `256`, or `512`; if selected, use ID validation only.
- Warning: PCA changes the detector's view of geometry. Report separately from raw geometry metrics.

## NC / Prototype / Hybrid Detectors

### `ncc_distance`

- Family: nearest class center.
- Input: penultimate features and ID train class means.
- Fitting split: ID train features and labels.
- Formula: `S_ncc(x) = - min_c ||f - mu_c||_2`.
- Score direction: higher = ID-like.
- Output name: `ncc_distance`.
- Tuning rule: none.
- Purpose: simple NC4 / nearest-class-center baseline.

### `cosine_prototype`

- Family: prototype cosine.
- Input: normalized feature and normalized class prototype.
- Fitting split: ID train features and labels.
- Formula: `S_cos_proto(x) = max_c cos(normalize(f), normalize(mu_c))`.
- Score direction: higher = ID-like.
- Output name: `cosine_prototype`.
- Tuning rule: none.
- Purpose: separates angular prototype structure from feature norm.

### `neco_lite`

- Family: NC-based OOD.
- Input: ID train features, class means, centered class means, classifier weight, PCA basis when used.
- Fitting split: ID train only.
- Formula: implementation-defined combination of the required components below; if components are not combined, store component scores separately and do not report a single `neco_lite` aggregate.
- Required stored components:
  - `ncc_distance`
  - `cosine_prototype`
  - class-mean principal-subspace residual norm
  - ETF residual or simplex-structure residual when implemented
  - classifier-feature alignment component when implemented
- Score direction: store a final ID-like score if components are combined.
- Output name: `neco_lite` for the aggregate; use explicit component names such as `neco_lite_residual` for components.
- Tuning rule: fit PCA/subspace/prototypes on ID train only. Do not choose component weights using OOD test performance.
- Warning: full NECO should be implemented against the paper or official code and recorded separately as `neco_full`.

### `vim`

- Family: hybrid feature + logit.
- Input: logits, penultimate features, classifier weights, ID train feature principal subspace.
- Fitting split: ID train features/logits only.
- Formula: combine ID class-dependent logits with a virtual OOD logit generated from feature residual against the principal space.
- Score direction: store an ID-like version; if the native implementation outputs an OOD probability or OOD score, sign-flip or transform it and record the transform.
- Output name: `vim`.
- Tuning rule: fit principal subspace and scaling constants on ID train/validation only. Never tune on OOD test AUROC.
- Purpose: tests whether hybrid feature-logit detectors follow logit-only or feature-only behavior when they diverge.

## Geometry Metrics

Use ID train features unless a task explicitly asks for ID test geometry. Geometry metrics are not OOD detectors by themselves.

### Core definitions

- `K`: number of classes.
- `N`: total number of ID train samples.
- `f_i`: penultimate feature for sample `i`.
- `mu_c`: mean feature for class `c`.
- `mu_G`: global mean over class means or samples; record which convention is used.
- `M`: matrix of centered class means.
- `W`: last-layer classifier weight matrix with one row per class.
- `Sigma_W`: within-class covariance / scatter normalized by sample count.
- `Sigma_B`: between-class covariance from centered class means.

### `within_var`

- Input: ID train penultimate features and labels.
- Fitting split: ID train.
- Formula: `tr(Sigma_W)`.
- Score direction: not an OOD score; lower usually means tighter within-class features.
- Output name: `within_var`.
- Output value: scalar.
- Tuning rule: none.

### `inter_dist`

- Input: ID train class means.
- Fitting split: ID train.
- Formula: average pairwise Euclidean distance between class means.
- Score direction: not an OOD score; higher usually means greater class separation.
- Output name: `inter_dist`.
- Output value: scalar.
- Tuning rule: none.

### `nc0`

- Input: last-layer classifier weights `W`.
- Fitting split: none; read from the checkpoint.
- Formula: `1 / K * || W^T 1_K ||_2^2`.
- Score direction: not an OOD score; lower means closer to zero-row-sum last-layer weights.
- Output name: `nc0`.
- Output value: scalar.
- Tuning rule: none.
- Optional normalized variant: `nc0_normalized = (1 / K) * ||W^T 1_K||_2 / ||W||_F`.
- Source note: NC0 is a necessary but not sufficient condition for Neural Collapse in the referenced optimizer/NC paper.

### `nc1`

- Input: ID train penultimate features, labels, `Sigma_W`, and `Sigma_B`.
- Fitting split: ID train.
- Formula: `tr(Sigma_W Sigma_B^dagger) / K`, where `dagger` is the Moore-Penrose pseudo-inverse.
- Score direction: not an OOD score; lower means stronger variability collapse.
- Output name: `nc1`.
- Output value: scalar.
- Tuning rule: none; numerical pseudo-inverse tolerance must be logged if non-default.

### `nc2_mean`

- Input: centered ID train class means.
- Fitting split: ID train.
- Meaning: centered class means converge toward simplex ETF geometry.
- Formula: implementation should record the exact ETF-distance or mean-cosine version used.
- Score direction: not an OOD score; lower is better for ETF-distance, while mean-cosine variants are interpreted against `-1 / (K - 1)`.
- Output name: `nc2_mean`.
- Output value: scalar.
- Tuning rule: none.
- Naming rule: do not call this only `nc2` unless the formula is also recorded.

### `nc2_weight`

- Input: last-layer classifier weights `W`.
- Fitting split: none; read from the checkpoint.
- Meaning: rows of classifier weight `W` converge toward simplex ETF geometry.
- Formula: ETF-distance version based on normalized `W W^T`.
- Score direction: not an OOD score; lower is better for ETF-distance.
- Output name: `nc2_weight`.
- Output value: scalar.
- Tuning rule: none.

### `nc2_product`

- Input: classifier weights `W` and centered ID train class means `M`.
- Fitting split: ID train for `M`; checkpoint for `W`.
- Meaning: product/alignment geometry between classifier weights and class means.
- Formula: ETF-distance version based on `W M` or the locally chosen product metric.
- Score direction: not an OOD score; lower is better for ETF-distance.
- Output name: `nc2_product`.
- Output value: scalar.
- Tuning rule: none.

### `nc3`

- Input: classifier weights `W` and centered ID train class means `M`.
- Fitting split: ID train for `M`; checkpoint for `W`.
- Meaning: classifier-feature self-duality / alignment.
- Formula: distance between normalized classifier weights and normalized centered class-mean directions, or ETF-distance of the product as specified by implementation.
- Score direction: not an OOD score; lower means stronger self-duality for distance variants.
- Output name: `nc3`.
- Output value: scalar.
- Tuning rule: none.
- Naming rule: record the exact variant in `detector_params.json` or `metrics_geometry.json`.

### `nc4`

- Input: ID test features and labels plus ID train class means.
- Fitting split: ID train for class means; ID test for reported accuracy.
- Meaning: nearest-class-center simplification.
- Formula: accuracy of NCC classifier using ID train class means.
- Score direction: higher is better.
- Output name: `nc4`.
- Output value: scalar in `[0, 1]`.
- Tuning rule: none.

### `anisotropy`

- Input: covariance eigenvalues from ID train features.
- Fitting split: ID train.
- Formula: `lambda_max(Sigma) / tr(Sigma)` for the chosen covariance, usually class covariance then averaged across classes.
- Score direction: not an OOD score; higher means more variance concentrated in the leading direction.
- Output name: `anisotropy`.
- Output value: scalar.
- Tuning rule: none.

### `effective_rank`

- Input: covariance eigenvalues from ID train features.
- Fitting split: ID train.
- Formula: let `p_i = lambda_i / sum_j lambda_j`; `effective_rank = exp(-sum_i p_i log(p_i))`.
- Score direction: not an OOD score; higher means variance is spread over more directions.
- Output name: `effective_rank`.
- Output value: scalar.
- Tuning rule: none.

### `covariance_eigenspectrum`

- Input: covariance matrices derived from ID train features.
- Fitting split: ID train.
- Formula: sorted eigenvalues for `Sigma_W`, per-class covariance, or both.
- Score direction: not an OOD score; interpret the full spectrum, not a single direction.
- Output name: `covariance_eigenspectrum`.
- Output value: array of sorted eigenvalues.
- Tuning rule: none; log dtype and decomposition method if it affects precision.
- Required metadata: covariance type, dtype, normalization, whether features were centered, and whether L2 normalization/PCA was applied.

### `feature_norm_stats`

- Input: penultimate features.
- Fitting split: none; compute on each reported split independently.
- Formula: summary statistics over `||f||_2`.
- Score direction: not an OOD score; used as diagnostic evidence for norm variation.
- Output name: `feature_norm_stats`.
- Output value: mean, std, min, max, and quantiles by split and optionally by class.
- Tuning rule: none.
- Purpose: identifies norm variation that can affect Mahalanobis, DDU/GMM, ViM, and L2-normalized controls.

## Detector Tuning Rules

| Detector | Allowed tuning |
|---|---|
| `msp`, `maxlogit`, `energy_id_score`, `neg_entropy` | None for main runs. Temperature only from ID validation if explicitly requested. |
| `mahalanobis` | Fit mean/covariance on ID train only. |
| `mahalanobis_l2` | L2-normalize features, then fit on ID train only. |
| `knn`, `knn_l2` | Main `k=50`; appendix grid is allowed but must be predeclared. |
| `gmm_ddu_full`, `gmm_ddu_tied`, `gmm_ddu_diag` | Fit on ID train only; numerical jitter must be logged. |
| `gmm_ddu_shrinkage` | Shrinkage fixed or selected by ID validation likelihood only. |
| `gmm_ddu_pca` | PCA basis from ID train only; dimension fixed or selected by ID validation only. |
| `neco_lite`, `neco_full`, `vim` | Fit PCA/subspace/prototypes on ID train only. |
| Tier 1 activation methods | Thresholds from ID train/validation activation statistics only. |

Never use OOD test labels or OOD AUROC to choose detector hyperparameters.

## Implementation Priority

### Tier 0: first evaluator

- Logit: `msp`, `maxlogit`, `energy_id_score`, `neg_entropy`.
- Feature: `mahalanobis`, `mahalanobis_l2`, `knn`, `knn_l2`, `gmm_ddu_tied`, `gmm_ddu_diag`, `gmm_ddu_shrinkage`.
- Diagnostic feature density: `gmm_ddu_full`, `gmm_ddu_pca`.
- NC/prototype/hybrid: `ncc_distance`, `cosine_prototype`, `neco_lite`, `vim`.
- Geometry: `nc0`, `nc1`, `nc2_mean`, `nc2_weight`, `nc2_product`, `nc3`, `nc4`, `within_var`, `inter_dist`, `anisotropy`, `effective_rank`, `covariance_eigenspectrum`, `feature_norm_stats`.

### Tier 1: appendix robustness

- `react`
- `ash`
- `scale`
- `rankfeat`
- `dice`
- `she`
- `gen`

### Tier 2: not part of the first implementation

- ODIN and GradNorm as main evidence.
- OpenMax as main evidence.
- OE, VOS, NPOS, CIDER, DRL, or any method that changes the training objective or uses auxiliary outliers.
- CLIP/DINOv2 prompt-based OOD for the from-scratch optimizer-geometry main claim.

## Checkpoint Rule

- Main geometry and detector analysis uses the final checkpoint.
- Best-validation checkpoint may be reported for deployment-style appendix analysis.
- Always record checkpoint type in the run manifest and metric JSON files.

## Source Anchors

Local project files:

- `소스/neurips2026_paper_context.md`
- `소스/2027_ICLR_실험레포설계.md`
- Legacy local code checked for current conventions: `C:\Users\User\Desktop\ICML+통계학회(연구)\DDU_fork-main\evaluate_v2.py`
- Legacy local code checked for geometry formulas: `C:\Users\User\Desktop\ICML+통계학회(연구)\DDU_fork-main\utils\geometry.py`
- Legacy local code checked for OOD score signs: `C:\Users\User\Desktop\ICML+통계학회(연구)\DDU_fork-main\utils\ood_scores.py`

External references:

- OpenOOD v1.5: https://arxiv.org/abs/2306.09301
- NECO, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/04b84142b99dae8560b517401e6e5275-Abstract-Conference.html
- ViM, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html
- Mahalanobis++: https://arxiv.org/abs/2505.18032
