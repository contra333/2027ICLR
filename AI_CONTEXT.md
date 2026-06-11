# AI_CONTEXT.md

최종 갱신: 2026-06-02 KST

## 이 문서의 역할

이 문서는 새 Codex/AI 세션이 이 프로젝트를 빠르게 이어받기 위한 한국어 핫 캐시다.

- 앞으로 이 파일은 기본적으로 한국어로 관리한다.
- 파일 경로, metric 이름, detector 이름, optimizer 이름, config 이름은 검색과 재현성을 위해 원래 표기를 유지한다.
- 실험 결과를 쓸 때는 항상 `confirmed`, `interpretation`, `hypothesis`를 구분한다.
- 큰 PDF, raw result, checkpoint를 읽기 전에 이 문서와 각 폴더의 index/manifest를 먼저 본다.

## 한 문장 프로젝트 맥락

이 프로젝트는 사용자의 NeurIPS 2026 논문을 ICLR 2027 제출로 확장하는 연구이며, 핵심 주제는 optimizer가 penultimate feature geometry / Neural Collapse regime을 어떻게 바꾸고, 그 변화가 Mahalanobis, DDU/GMM, kNN 같은 feature-based uncertainty/OOD detector의 성공과 실패에 어떤 영향을 주는지 밝히는 것이다.

## 현재 중심 thesis

ICLR 2027의 이야기는 단순히 "Neural Collapse가 좋다" 또는 "SAM이 나쁘다"가 아니다.

더 강한 thesis는 다음이다.

> Optimizer improvement는 하나의 reliability 의미로 환원되지 않는다. Optimizer는 서로 다른 penultimate geometry regime을 만들고, feature-based uncertainty detector는 Neural Collapse-like geometry의 어떤 요소가 보존, 왜곡, 또는 masking되는지에 따라 성공하거나 실패한다.

현재 title 방향:

> Neural Collapse Is Not One Geometry: Optimizer-Dependent Collapse Regimes and Their Consequences for OOD Detection

## 현재 repo에 있는 주요 자산

- `AGENTS.md`: Codex/AI agent의 루트 운영 규칙.
- `AI_CONTEXT.md`: 현재 파일. 프로젝트 thesis, 상태, 결과, 위험, 다음 action을 담은 빠른 재시작 문서.
- `README.md`: repo map과 현재 result context 요약.
- `소스/INDEX.md`: 문헌/source inventory와 evidence boundary.
- `소스/2027_ICLR_실험레포설계.md`: ICLR 프로젝트 설계와 실험 road map의 주요 planning 문서.
- `소스/neurips2026_paper_context.md`: 사용자의 NeurIPS 2026 제출 논문을 AI가 읽기 쉽게 정리한 문서.
- `소스/0_neurips_2026.pdf`: NeurIPS 2026 원본 PDF.
- `소스/ai_readable/arxiv-2602.16642v3/`: `Optimizer choice matters for the emergence of Neural Collapse`의 AI-readable source package.
- `ops/`: multi-server Git, server run, result sync 운영 문서.
- `code/`: standard CIFAR training/evaluation code.
- `configs/`: experiment config. 현재 핵심은 WRN350 CIFAR-10 config와 이후 seed 확장/optimizer 축 확장 config다.
- `reports/METRIC_DEFINITIONS.md`: evaluator, Codex, 사람 독자가 공유해야 하는 metric contract. `nc0_width_norm`, `nc3_self_duality`, `nc4_agreement`, `inter_dist_l2` 같은 revised metric name을 사용한다.
- `reports/DAILY_REPORT_WORKFLOW.md`: 일별 report 작성 workflow.
- `results/wrn350_seed0_eval_bundle_20260525/`: CIFAR-10 WRN-28-10 dropout0.3, seed0 final-checkpoint anchor evaluation bundle. SGD Nesterov, Adam matched, AdamW matched 포함.
- `results/WRN_seed0_350eps_girdsearch_0531/`: seed0 15-run WRN-28-10 350 epoch grid-search 원본 package.
- `results/processed/WRN_seed0_350eps_girdsearch_0531_*`: 0531 grid-search에서 파생된 processed table, metric dictionary, Notion artifact.
- `reports/WRN_seed0_350eps_girdsearch_0531_notion_working_page_ko.md`: 0531 grid-search Notion working page의 local draft.
- `reports/WRN_seed0_350eps_girdsearch_0531_professor_report_skeleton_ko.md`: 교수님 보고용 skeleton.
- `reports/0601_professor_QA_wrn0531_report_ko.md`: 0531 WRN 결과에 대한 교수님 Q&A 형식 보고서.

## 반드시 지켜야 할 evidence boundary

### NeurIPS 2026에서 확인된 claim

- vanilla SAM은 accuracy/calibration/logit-level reliability를 유지하거나 개선하면서도 Mahalanobis/DDU 같은 feature-based uncertainty detector를 악화시킬 수 있다.
- 제안된 mechanism은 penultimate feature geometry 변화다. 특히 within-class dispersion, covariance inflation, detector-relevant class structure 변화가 중요하다.

### Neural Collapse optimizer paper가 직접 뒷받침하는 claim

- optimizer choice는 Neural Collapse / representation geometry에 영향을 준다.
- coupled vs decoupled weight decay는 의미 있는 intervention axis다.
- NC metric들은 하나의 scalar로 뭉개면 안 되고, 여러 geometry component가 서로 다르게 움직일 수 있다.

### 현재 ICLR 프로젝트의 hypothesis

- optimizer-induced NC/feature geometry regime이 CNN, ViT, pretrained regime에서 feature-based uncertainty detector의 성공/실패를 설명할 수 있다.

### 주의

위 세 문장을 하나의 확정 claim으로 합치면 안 된다.

특히 `Optimizer choice matters for the emergence of Neural Collapse` 논문은 optimizer -> NC/geometry 축의 근거이지, Energy, Mahalanobis, kNN, DDU, downstream OOD detector behavior에 대한 직접 근거가 아니다. Downstream detector link는 이 repo의 controlled experiment 또는 별도 primary source가 필요하다.

## 의도한 실험 road map

1. Standard CNN main evidence
   - CIFAR-10/100, standard ResNet-18, WRN-28-10.
   - main claim에는 DDU-specific spectral normalization이나 modified activation을 넣지 않는다.
   - SGD, SGDW, Adam, AdamW, SAM, ASAM, GSAM을 비교한다.
   - accuracy, calibration, logit-based OOD, feature-based OOD, geometry metric을 함께 추적한다.

2. AdamW-to-Adam / coupled-decoupled weight decay axis
   - coupled vs decoupled weight decay를 sweep한다.
   - accuracy, NC metrics, Mahalanobis, DDU/GMM, kNN을 함께 본다.

3. DDU-style architecture diagnostics
   - old NeurIPS/DDU code의 SN/mod setting은 main evidence가 아니라 appendix 또는 diagnostic으로 다룬다.

4. ViT extension
   - CIFAR-scale ViT-Tiny/DeiT-Tiny from scratch부터 시작한다.
   - CLS, mean-pooled, pre-logit feature를 비교한다.
   - LayerNorm/bias weight-decay policy를 명시적인 experimental axis로 둔다.

5. Pretrained regime
   - frozen, linear-probe, full-finetune pretrained ViT 또는 CLIP/DINO feature를 regime analysis로 사용한다.
   - optimizer-induced geometry의 main proof로 과도하게 의존하지 않는다.

## 현재 repo 운영 상태

- 이 workspace는 research operations, source context, minimal experiment-code repo다.
- 원격 저장소는 `origin = https://github.com/contra333/2027ICLR.git`.
- `main`은 stable shared branch로 쓰고, 실험 준비는 가능하면 `exp/<short-name>` branch를 쓴다.
- 실제 학습은 local Windows가 아니라 GPU server에서 수행한다.
- local repo는 planning, analysis, report, config/code preparation, result import, manifest registry 용도다.
- server result는 `results/raw/<run_id>/`에 복사하고, manifest는 `results/manifests/*.json`에 둔다.
- Git에는 code, configs, docs, source indexes, manifests, small processed summaries만 넣는다.
- Git에 넣지 않을 것: raw server result folder, checkpoint, feature dump, large array, temporary log.
- 보고서와 교수님-facing summary는 `reports/`에 둔다.
- durable one-off task는 `tasks/`에 남긴다.

## 현재 실험 결과 상태

### 완료된 인프라 검증

초기 M1A/M1B smoke pipeline과 optimizer endpoint validation은 완료되었다. 이 내용은 이제 인프라 provenance로만 남긴다.

- smoke run은 training, cache extraction, post-hoc evaluation, run-dir check가 작동함을 확인했다.
- `adam_coupled_decoupled` endpoint check는 `coupled_ratio=0.0`을 AdamW-style, `coupled_ratio=1.0`을 Adam-style endpoint로 쓰는 plumbing을 확인했다.
- smoke metric은 pipeline validation일 뿐 ICLR evidence가 아니다.
- 이후 AI_CONTEXT에서는 smoke 세부 run 목록을 반복하지 않는다. 필요하면 `results/manifests/20260512_*.json`과 `reports/M1_SMOKE_STATUS_2026-05-13.md`를 본다.

### 1단계: seed0 matched optimizer 비교

상태: 완료 및 import됨. 이 단계는 seed0 고정 상태에서 CIFAR-10 `standard_wrn_28_10_dropout03`, 350 epochs, final checkpoint 기준으로 SGD Nesterov, Adam matched, AdamW matched를 비교한 anchor bundle이다.

확인된 source files:

- `results/wrn350_seed0_eval_bundle_20260525/README_AI_GUIDE.md`
- `results/wrn350_seed0_eval_bundle_20260525/manifests/wrn350_seed0_eval_20260525.json`
- `results/wrn350_seed0_eval_bundle_20260525/processed/wrn350_seed0_eval_summary_20260525.md`
- `results/wrn350_seed0_eval_bundle_20260525/processed/wrn350_seed0_classification_calibration_20260525.csv`
- `results/wrn350_seed0_eval_bundle_20260525/processed/wrn350_seed0_ood_metrics_20260525.csv`
- `results/wrn350_seed0_eval_bundle_20260525/processed/wrn350_seed0_geometry_scalars_20260525.csv`

scope:

- dataset: CIFAR-10 ID.
- OOD datasets: CIFAR100, TinyImageNet, SVHN, MNIST.
- model: `standard_wrn_28_10_dropout03`.
- seed: `0` only.
- runs: SGD Nesterov anchor, Adam matched, AdamW matched.
- no cross-seed average, confidence interval, or stability claim.

confirmed matched metrics:

- SGD Nesterov: ID test accuracy `0.9585`, ID test NLL `0.2068`, ECE `0.0298`, temperature-scaled ECE `0.0066`.
- Adam matched: ID test accuracy `0.8724`, ID test NLL `0.3973`, ECE `0.0263`, temperature-scaled ECE `0.0108`.
- AdamW matched: ID test accuracy `0.9437`, ID test NLL `0.5418`, ECE `0.0479`, temperature-scaled ECE `0.0062`, fitted temperature `5.2053`.

matched comparison interpretation:

- SGD Nesterov는 세 matched run 중 ID accuracy가 가장 높고 가장 NC-like geometry를 보인다.
- Adam matched는 이 matched LR/WD setting에서는 ID accuracy가 크게 낮다.
- AdamW matched는 Adam보다 ID accuracy가 강하지만 raw NLL/ECE가 나쁘고 큰 scalar temperature가 필요하다. 이는 raw logit scale/calibration mismatch를 시사한다.
- AdamW matched는 "구조가 없음"이 아니라 partial NC에 가깝다. `nc4_agreement`는 높게 남아 있지만, `nc0_width_norm`, `nc1`, `nc3_cos_alignment`, `inter_dist_l2`, covariance conditioning이 SGD보다 약하다.
- feature OOD는 "AdamW feature OOD가 실패한다"라고 쓰면 안 된다. raw Mahalanobis/kNN은 약하지만 L2-normalized controls가 강하게 회복되므로, feature norm / covariance-scale channel이 중요한 mediator일 가능성이 크다.

### 2단계: seed0 ID-val 기준 LR/WD grid-search

상태: 완료, import, processed, Notion/local report draft 요약 완료. 이 단계는 seed0 고정 상태에서 ID validation accuracy를 기준으로 SGD/Adam/AdamW의 LR/WD 후보를 평가한 diagnostic grid다.

확인된 source files:

- package manifest: `results/WRN_seed0_350eps_girdsearch_0531/WRN_seed0_350eps_girdsearch_0531/manifest.json`
- processed manifest: `results/processed/WRN_seed0_350eps_girdsearch_0531_generation_manifest.json`
- main table/metric guide: `results/processed/WRN_seed0_350eps_girdsearch_0531_tables_and_metric_definitions.md`
- local Notion working draft: `reports/WRN_seed0_350eps_girdsearch_0531_notion_working_page_ko.md`
- professor Q&A report: `reports/0601_professor_QA_wrn0531_report_ko.md`

scope:

- 15 seed0 WRN-28-10 CIFAR-10 350 epoch candidates.
- optimizer families: SGD, Adam, AdamW.
- axes: learning rate and weight decay.
- hyperparameter selection 기준: ID validation accuracy.
- OOD metrics와 geometry metrics는 response-surface diagnostics이지 selection criteria가 아니다.
- 모든 candidate는 `epoch_0350`을 사용한다. 단, `sgd_lr1e-1_wd5e-4_anchor`는 package note에 따라 350-epoch endpoint로 `final`을 사용한다.

confirmed grid-search highlights:

- validation accuracy 상위 후보는 SGD 계열에 몰려 있다.
- best validation row는 `sgd_lr1e-1_wd2e-4`: best val accuracy `0.9612`, ID test accuracy `0.9546`, ECE `0.0331`, `nc1=0.0682`.
- canonical SGD anchor `sgd_lr1e-1_wd5e-4_anchor`: ID test accuracy `0.9585`, ECE `0.0298`, `nc1=0.0489`, `nc3_cos_alignment=0.9470`.
- Adam 대표 후보 `adam_lr1e-3_wd0`: best val accuracy `0.9524`, ID test accuracy `0.9436`, raw Mahalanobis mean AUROC `0.6764`, Mahalanobis L2 mean AUROC `0.9351`, raw kNN mean AUROC `0.8730`, kNN L2 mean AUROC `0.9446`.
- Adam coupled-WD 후보 `adam_lr1e-3_wd1e-4`: raw Mahalanobis mean AUROC `0.5531`, Mahalanobis L2 mean AUROC `0.8148`, raw kNN mean AUROC `0.8692`, kNN L2 mean AUROC `0.8904`. Adam wd=0 대비 `nc0_width_norm=0.0132`, `nc3_cos_alignment=0.9069`로 NC0/NC3 축이 개선된다.
- AdamW validation-best 후보 `adamw_lr5e-3_wd1e-4`: best val accuracy `0.9528`, ID test accuracy `0.9468`, NLL `0.5234`, ECE `0.0451`, fitted temperature `5.1008`, raw Mahalanobis mean AUROC `0.4719`, Mahalanobis L2 mean AUROC `0.9226`, raw kNN mean AUROC `0.6538`, kNN L2 mean AUROC `0.9418`.
- AdamW `lr=5e-3` candidates는 partial-NC / ill-conditioned-feature regime을 보인다. class-mean distance는 약 `5.37-5.68`로 낮고, `nc0_width_norm`은 높고, `nc3_cos_alignment`는 약하며, condition number는 `1e11` 수준이다.

current interpretation:

- 현재 seed0 결과는 "SGD가 무조건 좋고 AdamW가 무조건 나쁘다"가 아니다.
- 더 정확한 해석은 optimizer가 generalization, raw calibration, logit-level OOD ranking, raw feature detector, normalized feature detector를 서로 다르게 움직인다는 것이다.
- Adam/AdamW는 ID accuracy가 완전히 무너지지 않아도 raw NLL/ECE가 나빠질 수 있고, 동시에 Energy/MaxLogit AUROC가 일부 OOD dataset에서 강할 수 있다. ECE는 ID confidence correctness이고 AUROC는 ID/OOD score ranking이기 때문이다.
- 가장 강한 feature-detector observation은 detector-side normalization sensitivity다. Adam/AdamW에서 raw norm/covariance-sensitive detectors는 크게 흔들릴 수 있지만, `mahalanobis_l2`와 `knn_l2`는 크게 회복된다.
- "NC alone caused OOD failure"라고 주장하지 않는다. 현재 evidence는 feature norm, covariance scale/conditioning, class mean separation, classifier-feature alignment가 함께 얽힌 optimizer-induced geometry regime을 지지한다.
- `mahalanobis_l2`는 Mahalanobis++ full reproduction이 아니라 Mahalanobis++-motivated normalization control이다.

### 현재 evidence level

- 초기 pipeline validation: 완료. ICLR evidence로 해석하지 않는다.
- 20260525 matched optimizer comparison: seed0 diagnostic evidence.
- 0531 ID-val grid-search: seed0 diagnostic response-surface evidence.
- 아직 없음: seed1/seed2 반복, seed-averaged mean/std, confidence interval, causal proof.

## WRN350 optimizer protocol decision

현재 결정:

- 현재 WRN350 CIFAR-10 SGD Nesterov setting을 `matched_protocol` anchor로 사용한다.
- anchor setting: WRN-28-10 dropout 0.3, 350 epochs, milestones `[150, 250]`, `lr: 0.1`, `momentum: 0.9`, `nesterov: true`, `weight_decay: 5.0e-4`, `weight_decay_policy: weights_only_no_bias_norm`.
- 이 SGD anchor는 canonical WRN-style baseline이지, SGD가 ID-validation tuned global optimum이라는 claim이 아니다.
- matched protocol에서는 dataset, architecture, augmentation, epoch budget, schedule, seed policy, total weight decay, OOD datasets, detector/geometry config, weight-decay policy를 고정하고 optimizer axis를 바꾼다.
- matched protocol은 optimizer family 간 numerical learning rate가 같아야 한다는 뜻이 아니다. SGD와 Adam-family는 stable LR scale이 다르다.
- 초기 2-epoch validation의 Adam/AdamW `lr: 1.0e-3`를 definitive WRN350 main-experiment value로 재사용하지 않는다.
- Adam/AdamW/coupled-axis `matched_protocol` run과 `ID-tuned protocol` run을 분리한다.
- Adam 또는 AdamW hyperparameter를 ID validation으로 선택한다면, 같은 protocol layer에 predeclared SGD grid도 포함해야 한다. tuned Adam/AdamW를 untuned matched SGD anchor와 같은 protocol인 것처럼 비교하지 않는다.
- ID-tuned optimizer hyperparameter는 validation accuracy, NLL, ECE 같은 ID validation metric만으로 선택한다.
- OOD AUROC, AUPR, FPR95로 optimizer나 detector hyperparameter를 선택하지 않는다.
- PyTorch `adam`, `adamw`는 named baseline optimizer로 유지한다.
- `adam_coupled_decoupled`는 controlled coupling axis로 사용한다.
- `adam_coupled_decoupled`의 `r=0.0`, `r=1.0`은 custom optimizer의 AdamW-style, Adam-style endpoint로 해석한다. PyTorch `AdamW`/`Adam`과 full-training bitwise identity를 주장하지 않는다.

May 19 server observation:

- 당시 active server는 `curie`였고, 4개의 RTX A5000 GPU가 unrelated long-running job으로 꽉 차 있었다.
- 용량이 비기 전에는 `curie`에서 WRN350 anchor를 launch하지 않는다.
- `/home/ghjin/datasets`에는 CIFAR-10과 SVHN `test_32x32.mat`가 있었지만 CIFAR-100, MNIST, TinyImageNet은 없었다.
- May 19 full config는 TinyImageNet을 OOD dataset으로 포함하지만, 현재 code는 TinyImageNet을 auto-download하지 않는다. ImageFolder data가 없으면 `build_data_bundle`이 막힌다.

## 알려진 risk와 주의점

- DDU 하나만으로 feature-based OOD behavior 전체를 대표할 수 없다. Mahalanobis, kNN, tied/diagonal/shrinkage GMM, feature normalization control을 함께 포함해야 한다.
- full covariance DDU/GMM은 CIFAR-100, ViT, high-dimensional pretrained feature에서 불안정할 수 있다.
- SN/mod DDU architecture 결과는 optimizer effect와 architecture effect가 confounded될 수 있다.
- ViT 결과는 더 noisy하거나 regime-dependent일 수 있다. CNN controlled evidence가 main claim을 지탱해야 한다.
- Pretraining은 optimizer-induced geometry shift를 mask할 수 있다. 이를 실패라기보다 regime finding으로 frame하는 것이 안전하다.
- optimizer endpoint 해석은 혼동되기 쉽다. PyTorch `AdamW`/`Adam`은 baseline anchor이고, `adam_coupled_decoupled` endpoint는 controlled interpolation endpoint다. 자세한 정책은 `reports/OPTIMIZER_ENDPOINT_SEMANTICS_2026-05-13.md` 참고.
- WRN350 SGD anchor config는 `optimizer.nesterov: true`를 사용한다. 이는 standard WRN-style SGD practice를 따르기 위한 definitive anchor 결정이다. future SGD-family WRN comparison config와 SAM-family base optimizer에서도 이 선택을 고정한다.
- 20260525와 0531 WRN result는 모두 seed0-only다. diagnostic evidence로는 유용하지만 seed-averaged paper-level conclusion이 아니다.
- validation-top grid candidate와 matched-protocol anchor는 서로 다른 질문에 답한다. tuned Adam/AdamW를 untuned SGD anchor와 같은 protocol인 것처럼 비교하지 않는다.
- AdamW calibration statement는 raw logit과 post-hoc temperature-scaled probability를 구분해야 한다.
- feature-detector statement는 raw Mahalanobis/kNN/GMM behavior와 L2-normalized control을 구분해야 한다.

## 다음 action 후보

- 20260525 matched anchor bundle과 0531 ID-val grid-search는 seed0 diagnostic evidence로 사용한다.
- 1순위 server 실험 후보: 0531에서 선택한 대표 후보를 seed `0,1,2`로 확장한다. seed0은 이미 있으므로 seed1/seed2 repetition을 우선 준비한다.
- 2순위 server 실험 후보: total weight decay를 고정하고 `adam_coupled_decoupled`의 coupled ratio를 sweep한다. 목표는 Adam-to-AdamW / coupled-to-decoupled 보간에서 geometry와 detector behavior가 어떻게 움직이는지 보는 것이다.
- 3순위 분석 후보: 왜 geometry가 OOD detector에 영향을 주는지 score-level, norm-level, covariance-level로 분해한다.
  - Energy/MaxLogit/MSP ID/OOD score histogram.
  - Mahalanobis/Mahalanobis L2 ID/OOD score histogram.
  - kNN/kNN L2 ID/OOD score histogram.
  - ID/OOD feature norm distribution과 class-wise feature norm distribution.
  - covariance eigenspectrum과 condition number 비교.
  - L2-normalized feature에서 geometry metric 재계산.
- paper framing 후보: optimizer가 generalization, calibration, logit-based OOD, feature-based OOD reliability를 하나의 방향으로 개선/악화시키는 것이 아니라, reliability component별로 서로 다른 regime을 만든다는 식으로 정리한다.
- SGD anchor에서 Adam/AdamW/coupled-decoupled sweep으로 확장하기 전에 optimizer endpoint semantics policy를 계속 보존한다.
- evaluator output을 해석하기 전에 계속 `reports/METRIC_DEFINITIONS.md`를 사용한다.
- legacy `nc0`, `nc3`, `nc4`, `inter_dist` 대신 revised metric name을 계속 사용한다.
- 각 server run마다 metrics/log/config snapshot을 복사하고 `results/manifests/*.json`을 작성한다.
- server run 준비/동기화 시 `ops/MULTI_SERVER_GIT_WORKFLOW.md`, `ops/SERVER_RUN_TEMPLATE.md`, `ops/RESULT_SYNC_GUIDE.md`, `ops/RUN_MANIFEST_RULES.md`를 사용한다.
- 큰 milestone 이후에는 `AI_CONTEXT.md`를 갱신해 새 Codex session이 싸게 재시작할 수 있게 한다.

## Windows에서 Ubuntu로 이전하기 전 정리 방침

현재 목표:

1. `C:\Users\jin\Desktop\2027ICLR`의 어지러운 Windows 작업본을 정리한다.
2. Git에 올릴 수 있는 code/config/docs/manifests/processed summaries/reports만 선별한다.
3. raw outputs, checkpoints, feature dumps, temporary logs는 Git에 올리지 않는다.
4. 정리된 변경사항을 GitHub에 push한다.
5. Ubuntu에서는 `/home/jin/code/2027ICLR`에 GitHub에서 새로 clone한다.
6. 이후 Codex 앱과 연구 coding/experiment preparation은 Ubuntu workspace를 주 작업장으로 삼는다.

정리 우선순위:

- 1순위: `AI_CONTEXT.md`, `README.md`, `reports/`, `results/manifests/`, `results/processed/`.
- 2순위: `configs/`, `code/`, `tasks/`, `ops/`.
- 3순위: `소스/INDEX.md`와 새로 ingest한 source package.
- 보존만 하고 Git 제외: `outputs/`, `results/raw/`, `results/checkpoints/`, `results/tmp/`, checkpoint, feature dump, large array, large log.
