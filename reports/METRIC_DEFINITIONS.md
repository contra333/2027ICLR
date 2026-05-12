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

All project detector scores used for OOD aggregate metrics must follow this convention:

- higher score = more ID-like
- ID label = 1
- OOD label = 0

If a method natively outputs an uncertainty or OOD score, store the sign-flipped or transformed ID-like version as the project detector score and record the native direction in `detector_params.json`. Native OOD-direction diagnostics, such as `vim_native_ood_score`, must not be used directly for AUROC/AUPR/FPR95 unless converted to an ID-like score.

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

### Aggregate metric implementation notes

- AUROC/AUPR tie handling must follow the evaluator library default and be recorded in `detector_params.json` or the evaluator README.
- `fpr95` threshold interpolation must be recorded. The default is the smallest threshold that reaches ID TPR >= 0.95.
- OOD aggregate metrics are reporting metrics only. Never choose detector hyperparameters by OOD test AUROC, AUPR, or FPR95.

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

- Family: normalized feature distance; Mahalanobis++-motivated control.
- Input: L2-normalized penultimate features.
- Fitting split: L2-normalized ID train features and labels.
- Formula: same as `mahalanobis` after `f <- f / ||f||_2`.
- Score direction: higher = ID-like.
- Output name: `mahalanobis_l2`.
- Tuning rule: none for the main version.
- Purpose: controls feature-norm / covariance-scale effects.
- Naming rule: do not call this a full Mahalanobis++ reproduction unless the implementation exactly follows the paper; use `mahalanobis_pp_full` only for a faithful reproduction.

### `knn`

- Family: non-parametric feature distance.
- Input: penultimate features.
- Fitting split: ID train features.
- Formula: `S_knn(x) = - d_k(f, F_train)`, where `d_k` is the Euclidean distance to the kth nearest ID train feature.
- Score direction: higher = ID-like.
- Output name: `knn`.
- Main default: `k = 50`.
- Appendix grid: `k in {10, 20, 50, 100}`.
- Tuning rule: do not tune on OOD test. If scoring ID train samples themselves, exclude the sample itself from its neighbor set.

### `knn_l2`

- Family: normalized non-parametric feature distance.
- Input: L2-normalized penultimate features.
- Fitting split: L2-normalized ID train features.
- Formula: same as `knn` after feature normalization.
- Score direction: higher = ID-like.
- Output name: `knn_l2`.
- Main default: `k = 50`.
- Tuning rule: do not tune on OOD test; use the same predeclared `k` rule as `knn`. If scoring ID train samples themselves, exclude the sample itself from its neighbor set.

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
- Formula: class-wise GMM with `Sigma_c_alpha = (1 - alpha) * Sigma_c + alpha * T_c`, where default `T_c = (tr(Sigma_c) / d) * I` and `d` is feature dimension.
- Score direction: higher = ID-like.
- Output name: `gmm_ddu_shrinkage`.
- Tuning rule: use fixed `alpha` or select from `alpha_grid = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]` by ID validation likelihood only. Never choose shrinkage using OOD test AUROC.
- Required metadata: `alpha`, shrinkage target, biased/unbiased covariance convention, variance floor if any, and numerical jitter.

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
- Purpose: simple nearest-class-center detector score. Do not call this `nc4`.

### `ncc_accuracy`

- Family: nearest class center label metric.
- Input: ID test features and labels plus ID train class means.
- Fitting split: ID train for class means; ID test for reported accuracy.
- Formula: `mean(argmin_c ||f - mu_c||_2^2 == y)`.
- Score direction: higher is better.
- Output name: `ncc_accuracy`.
- Output value: scalar in `[0, 1]`.
- Tuning rule: none.
- Purpose: label accuracy of the nearest-class-center classifier. This is useful, but it is not strict NC4.

### `nc_prototype_cosine`

- Family: prototype cosine / NC diagnostic detector.
- Input: normalized feature and normalized class prototype.
- Fitting split: ID train features and labels.
- Formula: `S_cos_proto(x) = max_c cos(normalize(f), normalize(mu_c))`.
- Score direction: higher = ID-like.
- Output name: `nc_prototype_cosine`.
- Tuning rule: none.
- Purpose: separates angular prototype structure from feature norm.

### `nc_residual_subspace`

- Family: NC/PCA subspace diagnostic.
- Input: feature `f`, ID train PCA/principal subspace `P`, and feature norm.
- Fitting split: ID train only for PCA/principal subspace.
- Formula: default NECO-style component `||P f||_2 / ||f||_2`; if an implementation instead uses orthogonal residual `||P_perp f||_2`, store the sign-flipped ID-like score and record the transform.
- Score direction: higher = ID-like.
- Output name: `nc_residual_subspace`.
- Tuning rule: PCA dimension must be fixed in config or selected by ID validation only.
- Source note: NECO defines a relative norm in the ID principal/ETF subspace and optionally calibrates with MaxLogit for some architectures.

### `nc_etf_residual`

- Family: NC simplex / ETF residual diagnostic.
- Input: feature or class-mean projection plus ID train class-mean ETF reference.
- Fitting split: ID train only.
- Formula: implementation-defined residual against the simplex ETF or class-mean principal subspace; store `-residual` as the ID-like score.
- Score direction: higher = ID-like.
- Output name: `nc_etf_residual`.
- Tuning rule: none unless a PCA dimension is used; PCA dimension must be fixed in config or selected by ID validation only.
- Required metadata: exact residual formula and whether class means, classifier weights, or sample features define the reference.

### `nc_classifier_alignment`

- Family: NC classifier-feature alignment diagnostic.
- Input: feature, predicted/prototype class, centered ID train class means, and classifier weights.
- Fitting split: ID train for class means; checkpoint for classifier weights.
- Formula: implementation-defined classifier-feature alignment component, such as cosine alignment between normalized feature/class-mean direction and corresponding classifier row.
- Score direction: higher = ID-like for cosine/alignment scores; if a distance is used, store the sign-flipped value.
- Output name: `nc_classifier_alignment`.
- Tuning rule: none.
- Required metadata: exact alignment formula and class selection rule.

### `neco_lite`

- Family: NC-based OOD.
- Input: ID train features, class means, centered class means, classifier weight, PCA basis when used.
- Fitting split: ID train only.
- Formula: reserved aggregate over `nc_residual_subspace`, `nc_prototype_cosine`, `nc_etf_residual`, and/or `nc_classifier_alignment`.
- Score direction: if emitted, higher = ID-like.
- Output name: do not emit `neco_lite` by default in Tier 0 tables; emit only after the aggregate formula is fixed and recorded.
- Tuning rule: fit PCA/subspace/prototypes on ID train only. Do not choose component weights using OOD test performance.
- Warning: `neco_lite` is not official NECO. Full NECO must be implemented against the paper or official code and recorded separately as `neco_full`.

### `vim_native_ood_score`

- Family: hybrid feature + logit.
- Input: logits, penultimate features, classifier weights, ID train feature principal subspace.
- Fitting split: ID train features/logits only.
- Formula: native ViM score, usually the softmax probability of the virtual OOD logit constructed from the residual against the principal subspace.
- Score direction: higher = OOD-like. This is a native diagnostic value, not the project detector score used for AUROC/AUPR/FPR95.
- Output name: `vim_native_ood_score`.
- Tuning rule: fit principal subspace and scaling constants on ID train/validation only. Never tune on OOD test AUROC.
- Required metadata: principal dimension, centering vector or bias-origin transform, residual scaling alpha, and whether ID validation was used.
- Purpose: preserves the paper-native ViM direction for reproducibility.

### `vim_id_score`

- Family: hybrid feature + logit.
- Input: `vim_native_ood_score` or virtual OOD probability plus original logits.
- Fitting split: inherited from `vim_native_ood_score`.
- Formula: `vim_id_score = -vim_native_ood_score` for raw OOD scores, or `vim_id_score = 1 - p_virtual_ood` when the native value is a virtual-class probability.
- Score direction: higher = ID-like.
- Output name: `vim_id_score`.
- Tuning rule: fit principal subspace and scaling constants on ID train/validation only. Never tune on OOD test AUROC.
- Required metadata: exact transform from native ViM output to ID-like score.
- Purpose: tests whether hybrid feature-logit detectors follow logit-only or feature-only behavior when they diverge.

## Geometry Metrics

Use ID train features unless a task explicitly asks for ID test geometry. Geometry metrics are not OOD detectors by themselves.

### Core definitions

- `K`: number of classes.
- `N`: total number of ID train samples.
- `d`: penultimate feature dimension.
- `f_i`: penultimate feature for sample `i`.
- `mu_c`: mean feature for class `c`.
- `mu_G`: global mean over class means or samples; record which convention is used.
- `M`: matrix of centered class means with shape `d x K`, where column `c` is `mu_c - mu_G`.
- `W`: last-layer classifier weight matrix with shape `K x d`, one row per class.
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

### `inter_dist_l2`

- Input: ID train class means.
- Fitting split: ID train.
- Formula: average pairwise Euclidean distance between class means.
- Score direction: not an OOD score; higher usually means greater class separation.
- Output name: `inter_dist_l2`.
- Output value: scalar.
- Tuning rule: none.
- Compatibility note: this matches the legacy code path that uses `torch.cdist(..., p=2).mean()`.

### `inter_dist_sq`

- Input: ID train class means.
- Fitting split: ID train.
- Formula: average pairwise squared Euclidean distance between class means.
- Score direction: not an OOD score; higher usually means greater class separation.
- Output name: `inter_dist_sq`.
- Output value: scalar.
- Tuning rule: none.
- Purpose: resolves ambiguity with NeurIPS appendix wording when it refers to squared Euclidean distance.

### `nc0_width_norm`

- Input: last-layer classifier weights `W`.
- Fitting split: none; read from the checkpoint.
- Formula: `||W^T 1_K||_2^2 / d`, where `d` is feature dimension.
- Score direction: not an OOD score; lower means closer to zero-row-sum last-layer weights.
- Output name: `nc0_width_norm`.
- Output value: scalar.
- Tuning rule: none.
- Source note: this is the paper-aligned NC0 normalization for cross-architecture comparisons because it normalizes by feature dimension.

### `nc0_by_K`

- Input: last-layer classifier weights `W`.
- Fitting split: none; read from the checkpoint.
- Formula: `||W^T 1_K||_2^2 / K`.
- Score direction: not an OOD score; lower means closer to zero-row-sum last-layer weights.
- Output name: `nc0_by_K`.
- Output value: scalar.
- Tuning rule: none.
- Purpose: optional legacy/control variant only. Do not use as the main NC0 metric across architectures.

### `nc1`

- Input: ID train penultimate features, labels, `Sigma_W`, and `Sigma_B`.
- Fitting split: ID train.
- Formula: `tr(Sigma_W Sigma_B^dagger) / K`, where `dagger` is the Moore-Penrose pseudo-inverse.
- Score direction: not an OOD score; lower means stronger variability collapse.
- Output name: `nc1`.
- Output value: scalar.
- Tuning rule: none; numerical pseudo-inverse tolerance must be logged if non-default.

### `nc2_mean_etf`

- Input: centered ID train class means.
- Fitting split: ID train.
- Meaning: centered class means converge toward simplex ETF geometry.
- Formula: ETF-distance based on `M^T M / ||M^T M||_F` against the simplex ETF reference `M_star`, with the normalizing constant recorded.
- Score direction: not an OOD score; lower means stronger class-mean ETF geometry.
- Output name: `nc2_mean_etf`.
- Output value: scalar.
- Tuning rule: none.
- Required metadata: exact ETF reference, normalization constant, and matrix orientation.

### `nc2_mean_cos`

- Input: centered ID train class means.
- Fitting split: ID train.
- Meaning: average off-diagonal cosine among centered class means.
- Formula: `avg_{c != c'} cos(mu_c - mu_G, mu_c' - mu_G)`.
- Score direction: not an OOD score; values closer to `-1 / (K - 1)` indicate stronger simplex-like equiangularity.
- Output name: `nc2_mean_cos`.
- Output value: scalar.
- Tuning rule: none.
- Purpose: preserves the mean-cosine diagnostic used in the legacy NeurIPS analysis without confusing it with ETF-distance metrics.

### `nc2_weight_etf`

- Input: last-layer classifier weights `W`.
- Fitting split: none; read from the checkpoint.
- Meaning: rows of classifier weight `W` converge toward simplex ETF geometry.
- Formula: ETF-distance version based on normalized `W W^T`.
- Score direction: not an OOD score; lower is better for ETF-distance.
- Output name: `nc2_weight_etf`.
- Output value: scalar.
- Tuning rule: none.

### `nc2_product_etf`

- Input: classifier weights `W` and centered ID train class means `M`.
- Fitting split: ID train for `M`; checkpoint for `W`.
- Meaning: product/alignment geometry between classifier weights and class means.
- Formula: ETF-distance version based on normalized `W M`.
- Score direction: not an OOD score; lower is better for ETF-distance.
- Output name: `nc2_product_etf`.
- Output value: scalar.
- Tuning rule: none.
- Compatibility note: the legacy code's `NC3 (Alignment)` computes a product-ETF style quantity and should map here if it is based on `W M`, not to strict NC3 self-duality.

### `nc3_self_duality`

- Input: classifier weights `W` and centered ID train class means `M`.
- Fitting split: ID train for `M`; checkpoint for `W`.
- Meaning: classifier-feature self-duality / alignment.
- Formula: distance between `W / ||W||_F` and `M^T / ||M^T||_F`, with the paper normalization constant recorded.
- Score direction: not an OOD score; lower means stronger self-duality.
- Output name: `nc3_self_duality`.
- Output value: scalar.
- Tuning rule: none.
- Required metadata: whether classifier bias is ignored, whether class means are centered, and the normalization constant.

### `nc3_cos_alignment`

- Input: classifier weights `W` and centered ID train class means `M`.
- Fitting split: ID train for `M`; checkpoint for `W`.
- Meaning: per-class cosine alignment between classifier row and centered class-mean direction.
- Formula: `avg_c cos(w_c, mu_c - mu_G)`.
- Score direction: not an OOD score; higher means stronger per-class alignment.
- Output name: `nc3_cos_alignment`.
- Output value: scalar.
- Tuning rule: none.
- Purpose: optional diagnostic distinct from strict NC3 self-duality.

### `nc4_agreement`

- Input: ID test features/logits plus ID train class means.
- Fitting split: ID train for class means; ID test for reported agreement.
- Meaning: nearest-class-center simplification.
- Formula: `mean(argmax_c logit_c == argmin_c ||f - mu_c||_2^2)`.
- Score direction: higher is better.
- Output name: `nc4_agreement`.
- Output value: scalar in `[0, 1]`.
- Tuning rule: none.
- Purpose: strict NC4-style agreement between the learned classifier decision and nearest-class-center decision. Do not use `ncc_accuracy` as `nc4_agreement`.

### `anisotropy_lambda1_trace`

- Input: covariance eigenvalues from ID train features.
- Fitting split: ID train.
- Formula: `lambda_max(Sigma) / tr(Sigma)` for the chosen covariance, usually class covariance then averaged across classes.
- Score direction: not an OOD score; higher means more variance concentrated in the leading direction.
- Output name: `anisotropy_lambda1_trace`.
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
| `gmm_ddu_shrinkage` | Shrink toward `T_c = tr(Sigma_c) / d * I`; alpha fixed or selected from the declared grid by ID validation likelihood only. |
| `gmm_ddu_pca` | PCA basis from ID train only; dimension fixed or selected by ID validation only. |
| `nc_residual_subspace`, `nc_etf_residual`, `nc_classifier_alignment`, `neco_lite`, `neco_full` | Fit PCA/subspace/prototypes on ID train only; do not choose aggregate weights by OOD test results. |
| `vim_native_ood_score`, `vim_id_score` | Fit principal subspace and scaling constants on ID train/validation only; report the native-to-ID-like transform. |
| Tier 1 activation methods | Thresholds from ID train/validation activation statistics only. |

Never use OOD test labels or OOD AUROC to choose detector hyperparameters.

## Implementation Priority

### Tier 0: first evaluator

- Logit: `msp`, `maxlogit`, `energy_id_score`, `neg_entropy`.
- Feature: `mahalanobis`, `mahalanobis_l2`, `knn`, `knn_l2`, `gmm_ddu_tied`, `gmm_ddu_diag`, `gmm_ddu_shrinkage`.
- Diagnostic feature density: `gmm_ddu_full`, `gmm_ddu_pca`.
- NC/prototype/hybrid: `ncc_distance`, `ncc_accuracy`, `nc_prototype_cosine`, `nc_residual_subspace`, `nc_etf_residual`, `nc_classifier_alignment`, `vim_id_score`.
- Reserved or diagnostic native values: `neco_lite` only after its aggregate formula is fixed; `vim_native_ood_score` only as native-direction diagnostic.
- Geometry: `nc0_width_norm`, `nc0_by_K`, `nc1`, `nc2_mean_etf`, `nc2_mean_cos`, `nc2_weight_etf`, `nc2_product_etf`, `nc3_self_duality`, `nc3_cos_alignment`, `nc4_agreement`, `within_var`, `inter_dist_l2`, `inter_dist_sq`, `anisotropy_lambda1_trace`, `effective_rank`, `covariance_eigenspectrum`, `feature_norm_stats`.

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

## Legacy Name Mapping

Old NeurIPS tables, legacy JSON files, or `DDU_fork-main` outputs may contain ambiguous names. New server evaluators should emit the revised names below.

| Legacy name | Revised name | Note |
|---|---|---|
| `inter_dist`, `Inter-class Distance` | `inter_dist_l2` | Legacy code uses Euclidean distance from `torch.cdist`; use `inter_dist_sq` only when squared Euclidean distance is intended. |
| `nc0` | `nc0_width_norm` | Main paper-aligned version normalizes by feature dimension `d`; `nc0_by_K` is legacy/control only. |
| `nc2_mean`, `NC2 (ETF Dist)` | `nc2_mean_etf` | ETF distance over centered class means. |
| `NC2 (Mean Sim)` | `nc2_mean_cos` | Mean off-diagonal cosine; interpret against `-1 / (K - 1)`. |
| `nc2_weight` | `nc2_weight_etf` | ETF distance over classifier weight rows. |
| `nc2_product`, old `nc3`, `NC3 (Alignment)` | `nc2_product_etf` when based on `W M` ETF distance | Do not label product-ETF distance as strict NC3. |
| strict self-duality `nc3` | `nc3_self_duality` | Use only for normalized `W` versus normalized centered class means. |
| `nc4`, `NC4 (NCC Acc)` | `ncc_accuracy` | Label accuracy of NCC is not strict NC4. |
| strict NC4 | `nc4_agreement` | Agreement between learned classifier prediction and NCC prediction. |
| `anisotropy` | `anisotropy_lambda1_trace` | Make the eigenvalue ratio explicit. |
| `cosine_prototype` | `nc_prototype_cosine` | Use the NC/prototype naming in new evaluator outputs. |
| `vim` | `vim_id_score` | Use `vim_id_score` for project OOD aggregates; keep `vim_native_ood_score` only as native diagnostic. |

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
- Optimizer/NC paper page evidence for NC0/NC1/NC2/NC2W/NC2M/NC3/NC4 definitions: `소스/paper_pdf_pages.md`
- ViM page evidence for virtual logit and native OOD probability: `소스/ai_readable/arxiv-2203.10807v1/paper_pdf_pages.md`
- OpenOOD page evidence for AUROC/AUPR/FPR@95 evaluation context: `소스/ai_readable/arxiv-2306.09301v5/paper_pdf_pages.md`
- NECO page evidence for `NECO(x) = ||P h(x)|| / ||h(x)||` and MaxLogit calibration: `소스/ai_readable/arxiv-2310.06823v3/paper_pdf_pages.md`
- Mahalanobis++ page evidence for L2 feature normalization before Mahalanobis: `소스/ai_readable/arxiv-2505.18032v1/paper_pdf_pages.md`

External references:

- OpenOOD v1.5: https://arxiv.org/abs/2306.09301
- NECO, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/04b84142b99dae8560b517401e6e5275-Abstract-Conference.html
- ViM, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html
- Mahalanobis++: https://arxiv.org/abs/2505.18032
