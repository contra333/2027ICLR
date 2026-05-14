## 0513_질문1
서버에서 어제까지 작업 내용을 기록해서 2027ICLR 깃허브 레포에 push해놨다. 2027ICLR 깃허브 레포에서 아래의 내용을 읽고 현재까지 작업 상황을 파악하면된다. 1. AI_CONTEXT.md 전체 프로젝트 맥락, 현재 thesis, 완료된 M1A/M1B 상태, 다음 액션을 빠르게 복구하는 파일입니다. 2. reports/M1_SMOKE_STATUS_2026-05-13.md 실제로 어떤 smoke run을 돌렸고, 어떤 manifest가 있으며, 어떤 결과는 “확인된 metric”이고 어떤 것은 아직 “해석하면 안 되는 smoke 값”인지 정리한 파일입니다. 그 후 이 현재 프로젝트에서 다음의 문제에 관해 고민이 있다. Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances라는 survey논문이 있다. open OOD v 1.5와 비슷한데 2024년에 나와 더 최신의 논문이다. 이 논문에서 feature-based, distance-based 등 우리 논문과 연관된 OOD 탐지방법들이 많은데 이 논문을 읽고 우리 논문에 들어갈 만한 metric을 이유와 함께 말해줘라.

## 0513_질문1_GPT답변
깃허브의 AI_CONTEXT.md와 reports/M1_SMOKE_STATUS_2026-05-13.md 기준으로 보면, 현재 프로젝트의 핵심은 “SAM 하나가 나쁘다”가 아니라, optimizer가 서로 다른 penultimate feature geometry regime을 만들고, 그 geometry가 feature-based OOD detector의 성공/실패를 좌우한다는 ICLR 2027 스토리입니다. M1A/M1B는 2-epoch smoke 수준에서 CIFAR-10 standard ResNet-18, SGD/Adam/AdamW/custom Adam-coupled-decoupled endpoint pipeline이 돌아가는지 확인한 상태이고, 지금 수치는 논문 evidence가 아니라 pipeline validation으로만 해석해야 합니다.

제가 보기에는 survey에서 우리 논문에 바로 가져올 만한 것은 평가 metric과 OOD scoring method를 분리해서 넣어야 합니다.

결론부터

우리 논문에 우선 넣을 것은 다음입니다.

우선순위	넣을 것	역할
1	FPR@95, AUPR-ID/OOD를 main metric으로 승격	AUROC만으로는 “고민되는 high-recall regime”을 못 보여줌
2	kNN OOD score	Mahalanobis/DDU 악화가 Gaussian/covariance 가정 때문인지, local feature geometry 자체 때문인지 분리
3	Mahalanobis++ / L2-normalized Mahalanobis	현재 우리 L2 normalization diagnostic을 정식 baseline으로 승격
4	ViM	feature와 logit의 hybrid score라서 “Energy는 괜찮고 Mahalanobis는 망가진다” 사이의 중간 지점 확인
5	NCI 또는 Neco	2027 ICLR 제목 방향인 Neural Collapse / optimizer geometry와 가장 직접적으로 연결
6	MaxLogit	Energy/MSP와 함께 logit-only reliability 축을 깔끔하게 정리
7	ReAct 또는 ASH	feature norm / activation outlier 때문에 생긴 degradation인지 확인하는 appendix diagnostic

survey의 Table 2는 post-hoc OOD 방법을 output-based, distance-based, gradient-based, feature-based, density-based로 나누고, 각 방법이 feature/logit/gradient/probability 중 무엇을 쓰는지 정리합니다. 특히 Mahalanobis/KNN은 feature-only distance-based, ViM과 Neco는 feature+logit 기반, GEM/ConjNorm은 density-based로 분류되어 있어서 우리 논문 taxonomy와 잘 맞습니다. Survey p.11 Table 2 및 pp.11-13 기준입니다.

1. 평가 metric: AUROC만 main으로 두면 약하다

Survey는 visual OOD에서 표준 평가 metric으로 AUROC, AUPR, FPR@95를 정리합니다. 특히 FPR@95는 TPR 95%에서 OOD가 ID로 잘못 받아들여지는 비율이라, 실제 safety/rejection 상황에서 중요합니다. Survey p.20, Sec. 6.1 기준입니다.

우리 논문 main table은 이렇게 바꾸는 게 좋습니다.

Metric	방향	왜 필요한가
AUROC	↑	기존 OOD 논문과 비교 가능한 ranking metric
FPR@95	↓	“ID를 95% 살리는 조건에서 OOD를 얼마나 잘 막나”를 보여줌
AUPR-ID / AUPR-OOD	↑	ID/OOD imbalance에 민감한 보조 지표
Near-OOD avg / Far-OOD avg	↑/↓	우리 효과가 near-OOD에서 강한지 분리
Score overlap	↓	우리 논문식 mechanistic diagnostic. 표준 metric은 아니지만 분포 겹침을 직접 보여줌
cs-ID accuracy + OOD rejection	↑/↓	full-spectrum OOD 확장 시 필요

특히 현재 프로젝트는 CIFAR-10 ID, SVHN OOD smoke가 이미 파이프라인에 있고, 앞으로 CIFAR-100/TinyImageNet 같은 near-OOD가 핵심이 될 가능성이 큽니다. Survey도 near-OOD/far-OOD 구분과 full-spectrum OOD protocol을 강조합니다. p.20에서는 near-OOD가 더 어렵고, full-spectrum protocol에서는 near/far OOD를 reject하면서 ID와 covariate-shift ID는 정확히 예측해야 한다고 설명합니다.

따라서 2027 ICLR에서는 단순히 “OOD AUROC가 내려간다”보다,

optimizer가 accuracy/calibration은 유지하지만, near-OOD high-recall rejection에서는 feature-based detector를 망가뜨린다

라고 쓰는 편이 더 강합니다.

2. kNN score는 반드시 넣는 게 좋다

가장 중요한 추가 score는 kNN distance OOD score입니다.

현재 우리 논문에는 Mahalanobis와 DDU/GMM이 중심인데, 둘 다 covariance나 Gaussian-style density assumption에 민감합니다. 그래서 reviewer가 이렇게 물을 수 있습니다.

이건 optimizer가 feature geometry를 망가뜨린 게 아니라, covariance estimation이 불안정해진 것 아닌가?

kNN은 이 반박을 막아줍니다. Survey는 KNN을 non-parametric nearest-neighbor distance 기반 OOD detection으로 설명하고, Mahalanobis가 강한 분포 가정을 하는 반면 KNN은 training embedding과 test embedding 사이의 k-nearest neighbor distance를 이용한다고 정리합니다. Survey pp.11-12 기준입니다.

해석은 이렇게 됩니다.

결과 패턴	해석
Mahalanobis/DDU만 악화, kNN은 안정	covariance/Gaussian modeling 또는 inverse covariance 문제가 큼
Mahalanobis/DDU/kNN 모두 악화	local neighborhood geometry 자체가 optimizer에 의해 손상됨
kNN만 악화	class covariance보다 sample-level local density가 더 민감함
kNN은 좋아지고 Mahalanobis는 악화	global class covariance와 local manifold가 서로 다르게 변함

이건 논문 contribution을 키우는 데 매우 좋습니다. “feature-based detector가 망가졌다”가 아니라 어떤 종류의 feature geometry가 망가졌는지를 분해할 수 있습니다.

3. Mahalanobis++는 우리 L2-normalization diagnostic을 정식 baseline으로 만든다

Survey는 Mahalanobis++를 Mahalanobis distance에 L2 feature normalization을 적용해 feature extraction instability와 distribution assumption 문제를 줄이는 방법으로 설명합니다. Survey p.11 기준입니다.

이건 우리 논문에 매우 잘 맞습니다. 현재 NeurIPS 2026 버전에서도 L2 normalization이 SAM high-rho에서 Mahalanobis/DDU 회복을 어느 정도 설명하는 diagnostic 역할을 했습니다. ICLR 2027에서는 이것을 “appendix diagnostic”이 아니라 정식 detector variant로 올리는 게 좋습니다.

추천 표기는 다음처럼 하면 됩니다.

Detector	Feature preprocessing
Mahalanobis	raw penultimate feature
Mahalanobis++	L2-normalized penultimate feature
DDU/GMM	raw feature
normalized GMM	L2-normalized feature, tied/shrinkage covariance optional
kNN	raw and normalized feature both 가능

이렇게 하면 reviewer에게 다음 질문을 직접 답할 수 있습니다.

SAM/AdamW가 feature norm만 바꾸는가, 아니면 angular/class-conditional geometry까지 바꾸는가?

만약 Mahalanobis++가 성능을 크게 회복하면, norm/covariance scale channel이 주요 원인입니다. 반대로 Mahalanobis++도 회복하지 못하면, NC1, NC3, class separation, local neighborhood 자체가 손상된 것입니다.

4. ViM은 logit-based와 feature-based 사이의 bridge로 중요하다

ViM은 우리 논문에 매우 좋은 “중간 detector”입니다. Survey는 ViM이 feature, logit, probability를 통합하고, penultimate layer의 null space와 Energy/LogSumExp를 결합한다고 설명합니다. Survey pp.12-13 기준입니다.

우리 논문은 현재 구조가 약간 이분법적입니다.

MSP/Energy는 괜찮다. Mahalanobis/DDU는 망가진다.

ViM을 넣으면 이 이분법 사이를 테스트할 수 있습니다.

ViM 결과	해석
ViM이 Energy처럼 안정	logit/energy term이 feature geometry 손상을 보완함
ViM이 Mahalanobis처럼 악화	penultimate null-space geometry도 optimizer에 의해 손상됨
ViM이 중간 정도	hybrid score가 logit reliability와 feature geometry 사이의 trade-off를 반영

이건 논문 그림으로도 좋습니다. Detector를 x축에 놓고 “logit-only → hybrid → feature-density” 순서로 성능 변화를 보여주면, optimizer-induced geometry shift가 어떤 detector family에 가장 민감한지 한눈에 보입니다.

추천 순서:

MSP / MaxLogit / Energy → ViM → Mahalanobis++ / kNN / Mahalanobis / DDU

5. NCI 또는 Neco는 2027 ICLR 스토리에 가장 직접적이다

현재 repo의 target title direction은 “Neural Collapse Is Not One Geometry: Optimizer-Dependent Collapse Regimes and Their Consequences for OOD Detection” 쪽입니다. 이 제목으로 가려면 NC metric을 단순 diagnostic으로만 쓰면 약합니다. NC-based OOD score를 하나 넣어야 합니다.

Survey는 Neco와 NCI를 neural collapse 기반 OOD 방법으로 정리합니다. Neco는 ID/OOD feature의 orthogonal trend를 이용하고, NCI는 ID sample이 penultimate feature space에서 class weight vector 근처에 cluster되는 반면 OOD는 그렇지 않다는 점을 이용해 class weight vector와 feature의 proximity로 OOD를 구분한다고 설명합니다. Survey p.13 기준입니다.

둘 중에서는 저는 NCI를 우선 추천합니다.

이유는 세 가지입니다.

첫째, 현재 우리 프로젝트는 nc0_width_norm, nc1, nc2_mean_cos, nc3_self_duality, inter_dist_l2 같은 revised geometry metric naming을 이미 정리해 두었습니다. M1 smoke에서도 revised geometry metric files가 emit되는 상태입니다. NCI는 class weight vector proximity라서 nc3_self_duality, nc0_width_norm, classifier-weight geometry와 연결하기 쉽습니다.

둘째, AdamW-to-Adam / coupled-decoupled weight decay axis와도 잘 맞습니다. 현재 ICLR roadmap은 AdamW-style과 Adam-style endpoint를 controlled interpolation axis로 쓰려는 방향입니다. NCI를 넣으면 “optimizer가 NC-like geometry를 얼마나 만들었는가”와 “그 geometry가 OOD score로 실제 작동하는가”를 바로 연결할 수 있습니다.

셋째, reviewer가 “NC metric이 왜 OOD에 중요하지?”라고 물을 때, NCI/Neco가 bridge 역할을 합니다.

단, 조심할 점이 있습니다. repo의 evidence boundary가 말하듯, Neural Collapse optimizer paper는 optimizer가 NC/geometry를 바꾼다는 근거이지, downstream OOD detector degradation을 직접 증명한 근거는 아닙니다. downstream link는 우리 실험으로 만들어야 합니다.

6. MaxLogit은 작지만 넣는 게 좋다

Survey는 output-based method로 MSP, MaxLogits, Energy를 함께 정리합니다. Survey p.11 기준입니다.

우리 논문에는 이미 MSP/Energy가 있으니 MaxLogit은 비용이 거의 없습니다. 넣는 이유는 “logit scale”과 “softmax probability”를 분리하기 위해서입니다.

Score	보는 것
MSP	softmax confidence
Energy	log-sum-exp logit scale
MaxLogit	가장 큰 raw logit
ViM	logit + feature null-space hybrid

MaxLogit까지 있으면 “SAM/AdamW가 output confidence 계열을 전반적으로 보존하는가, 아니면 Energy만 특이하게 안정적인가”를 정리하기 쉽습니다.

7. ReAct/ASH는 main claim보다는 diagnostic/appendix가 좋다

Survey는 ReAct가 intermediate activation의 high activation value를 truncation해 OOD detection을 개선하고, ASH도 activation shaping/pruning 계열이라고 설명합니다. Survey pp.12-13 기준입니다.

이들은 우리 논문의 main score로는 약간 위험합니다. 이유는 ReAct/ASH가 detector-side feature를 수정하기 때문에, optimizer가 만든 원래 geometry를 그대로 측정하는 detector라기보다 test-time feature intervention에 가깝기 때문입니다.

하지만 appendix diagnostic으로는 아주 좋습니다.

결과	해석
ReAct/ASH가 SAM degradation을 회복	high activation / feature norm outlier가 원인일 가능성
ReAct/ASH도 회복 못함	class-conditional geometry, NC alignment, local density 문제가 더 근본적
ASAM/GSAM에서는 ReAct 효과가 작음	vanilla SAM만 activation tail을 크게 만든다는 근거

즉, ReAct/ASH는 “성능 비교 baseline”이라기보다 mechanism check로 넣는 것이 좋습니다.

8. Density-based는 DDU를 유지하되 covariance variants를 추가하는 쪽이 좋다

Survey는 density-based로 GEM, ConjNorm 등을 다룹니다. Survey p.13 기준입니다. 우리 논문에는 이미 DDU/GMM 계열이 핵심이므로, 완전히 새로운 density method를 많이 넣기보다 DDU/GMM의 covariance variant를 추가하는 것이 더 효과적입니다.

추천 variant는 다음입니다.

Variant	목적
full covariance GMM/DDU	기존 feature-density detector
tied covariance	class별 covariance noise 감소
diagonal covariance	high-dimensional covariance instability 완화
shrinkage covariance	CIFAR-100/ViT/pretrained feature에서 안정성 확보
PCA + GMM	feature dimension이 클 때 covariance rank 문제 완화

이건 survey의 “density-based” family를 반영하면서도 우리 논문의 mechanism에 더 직접적입니다.

넣지 않는 것이 좋은 것

Gradient-based method인 GradNorm, GradOrth, GAIA는 survey에서 중요하게 다루지만, 지금 main paper에는 우선순위가 낮습니다. 이유는 계산 비용이 크고, 우리 핵심인 penultimate feature geometry / NC / covariance / local neighborhood와 직접성이 낮기 때문입니다. Appendix에 하나 정도 넣을 수는 있지만, main claim을 선명하게 하려면 뒤로 미루는 편이 좋습니다.

Test-time adaptive OOD도 당장은 넣지 않는 편이 좋습니다. Survey는 test-time adaptive methods를 별도 category로 다루지만, 우리 논문은 “학습 optimizer가 남긴 representation geometry footprint”가 핵심입니다. Test-time adaptation을 넣으면 training-time optimizer effect와 test-time adaptation effect가 섞입니다.

최종 추천 실험 세트

현재 M1 smoke 이후 첫 200-epoch matched sweep을 설계한다면, score set은 이렇게 가는 게 좋습니다.

Main score set
Family	Scores
Output-based	MSP, MaxLogit, Energy
Hybrid	ViM
Distance-based	Mahalanobis, Mahalanobis++, kNN
Density-based	DDU/GMM, shrinkage/tied GMM
NC-based	NCI 또는 Neco
Main evaluation metrics
Group	Metrics
Classification	accuracy, NLL
Calibration	ECE, TCE
OOD ranking	AUROC, AUPR-ID, AUPR-OOD
OOD high-recall	FPR@95
Mechanism	score overlap, ID/OOD score histograms
Geometry	within_var, inter_dist_l2, nc0_width_norm, nc1, nc2_mean_cos, nc3_self_duality, feature norm std, covariance eigenspectrum, effective rank
Dataset/protocol split
Protocol	추천
ID	CIFAR-10 first, then CIFAR-100
Near-OOD	CIFAR-100, TinyImageNet, maybe NINCO later
Far-OOD	SVHN, MNIST, Texture
cs-ID/full-spectrum	CIFAR-10-C or CIFAR-100-C later
논문에서 가장 좋은 메시지

이 survey를 반영한 뒤 우리 논문의 메시지는 이렇게 바꾸는 것이 가장 좋습니다.

Optimizers do not merely change accuracy or calibration. They induce different feature geometry regimes. Output-based scores, hybrid scores, distance-based scores, density-based scores, and NC-based scores respond differently to those regimes.

한국어로 풀면:

optimizer 개선은 reliability를 하나의 방향으로 개선하지 않는다. 어떤 optimizer는 logit-level reliability는 보존하지만, Mahalanobis, kNN, GMM/DDU, NC-based score가 필요로 하는 feature geometry는 왜곡할 수 있다.

따라서 당장 구현 우선순위는 kNN → Mahalanobis++ → MaxLogit → ViM → NCI/Neco → covariance variants → ReAct/ASH appendix 순서가 가장 좋습니다.

## 0513_질문2
현재 연구에서 중요한 것은 그래도 OOD Detection 방법들이 흐름을 알고 최신 경향과 사람들이 자주 쓰는 탐지방법들을 파악해야하며 그것이 논문에 들어가야한다고 생각한다. 예를 들어 기하구조의 변하가 있어서 이러한 마지막층의 피쳐층을 보는 데서 degradetion이 일어난다고 해도 그 방법을 개선한 다른 방법을 쓰면 되지않냐라는 질문이 있을수도 있다. 현재 참고한 논문으로 OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection, Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances 두 개의 논문이 있지만 각각 2023년과 2024년에 나온 논문이다. 현재는 2026년 5월 13일이며 최근 연구동향과 사람들의 많이 쓰는 방향을 알고싶다. 검색을 통해 이렇게 OOD Detection 전반을 다루는 논문을 찾아줘라.

## 0513_질문2_GPT답변
맞습니다. 지금 ICLR 2027 방향으로는 **“Mahalanobis/DDU가 SAM에서 나빠졌다”**만으로는 부족하고, OOD detection 전체 흐름 안에서 “왜 이 detector family를 봤는지, 더 최근 detector를 쓰면 해결되는지, 어떤 계열까지 확인했는지”를 보여줘야 합니다.

검색 기준으로 보면, 현재 넣어야 할 문헌은 단순 survey 2개가 아니라 survey/benchmark/systematic analysis/recent method trend를 묶어서 구성하는 게 좋습니다.

1. 우선순위가 가장 높은 OOD 전반 문헌
A. 핵심 backbone survey: Task-Oriented Survey

이미 참고 중인 Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances는 여전히 중심 survey로 써도 됩니다. 다만 “2024 논문”이라고만 보면 조금 낡아 보일 수 있는데, arXiv는 2025년 8월 4일 v4로 갱신되어 있고 ACM Computing Surveys 2025 accepted로 표기되어 있습니다. 이 survey는 OOD 방법을 training-driven, training-agnostic, large-pretrained-model-based로 나누며, post-hoc 안에서도 output, distance, gradient, feature, density 계열을 정리합니다.

이 논문은 Related Work의 큰 뼈대로 두면 됩니다. 특히 우리 논문에서는 다음 문장을 만들 수 있습니다.

Recent OOD detection methods span training-driven approaches, post-hoc scoring rules, test-time adaptation, and large-pretrained-model-based detection. Our study focuses on the post-hoc setting because it isolates how a fixed trained representation interacts with downstream uncertainty scores.

B. benchmark/critical analysis: OOD와 OSR를 함께 보는 IJCV 2025

Dissecting Out-of-Distribution Detection and Open-Set Recognition: A Critical Analysis of Methods and Benchmarks는 2025년 IJCV 논문으로, OOD detection과 open-set recognition을 함께 비교하고, standard benchmark와 large-scale benchmark 사이의 차이를 비판적으로 분석합니다. 이 논문은 semantic shift와 covariate shift를 분리해서 봐야 한다는 점, 그리고 표준 benchmark에서 좋아 보이는 OE가 scale-up 환경에서는 약해질 수 있다는 점을 강조합니다. 또 MSP, MLS, ODIN, Energy, GradNorm, ReAct 같은 대표 scoring rule을 비교하고, feature magnitude-sensitive score가 강하게 보인다고 보고합니다.

이건 우리 논문에 매우 중요합니다. 왜냐하면 reviewer가 “다른 OOD detector 쓰면 되는 것 아니냐?”라고 물을 때, 이 논문은 detector 성능은 benchmark와 shift 정의에 따라 바뀐다는 근거가 됩니다.

C. VLM 시대 general survey: Generalized OOD Detection and Beyond in Vision Language Model Era

Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era: A Survey는 2024년에 나왔지만 2025년 6월 v2로 갱신되었고 TMLR 2025 accepted survey입니다. 이 논문은 OOD detection, anomaly detection, novelty detection, open-set recognition, outlier detection을 VLM/CLIP 시대에서 다시 정리합니다. 특히 CLIP과 LVLM이 등장하면서 기존 OOD/OSR/AD 경계가 흐려지고, benchmark와 문제 정의가 바뀌고 있다는 점을 강조합니다.

우리 논문에서는 이 문헌을 “foundation-model/VLM regime은 중요한 future direction이지만, 본 논문은 optimizer-induced penultimate geometry를 controlled CNN/CIFAR setting에서 먼저 분리한다”는 식으로 쓰면 좋습니다.

D. CLIP-like OOD survey: 2025년 최신 VLM/OOD 흐름

Recent Advances in Out-of-Distribution Detection with CLIP-Like Models: A Survey는 2025년 5월 arXiv survey입니다. 이 논문은 CLIP-like model이 OOD detection을 unimodal image detector에서 image-text multimodal detector로 바꾸고 있다고 정리하고, OOD image가 seen/unseen인지, OOD text가 known/unknown인지, train-free인지 training-required인지로 새 taxonomy를 제안합니다.

이건 main experiment에 꼭 넣을 필요는 없지만, Related Work 마지막이나 Limitation/Future Work에 넣으면 좋습니다. 특히 “pretrained ViT/CLIP에서는 optimizer effect가 다르게 나타날 수 있다”는 후속 연구 방향을 정당화해 줍니다.

E. LLM OOD/anomaly survey: 텍스트/LLM 확장용

Large Language Models for Anomaly and Out-of-Distribution Detection: A Survey는 NAACL 2025 Findings accepted paper입니다. LLM이 anomaly/OOD detection에서 전통적 paradigm을 바꾸고 있다는 survey이며, LLM이 detector로 쓰이는 경우와 LLM 대상 OOD를 다루는 경우를 분류합니다.

우리 논문이 vision/CIFAR 중심이면 main Related Work에는 짧게만 언급하면 됩니다. 다만 “OOD detection is expanding beyond discriminative image classifiers into LLM/VLM settings” 정도의 문장에 적합합니다.

2. Survey는 아니지만 현재 연구에 꼭 필요한 최신 empirical/systematic 논문
A. Representation/training paradigm shift를 직접 다루는 2025/2026 systematic analysis

A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts는 2025년 11월 제출, 2026년 2월 v2로 갱신된 systematic comparison입니다. 이 논문은 OOD detector를 representation-centric하게 비교하고, CNN from scratch와 fine-tuned ViT를 모두 보며, neural collapse metric도 분석에 포함합니다. 결과적으로 learned feature space가 OOD 효율을 크게 결정하고, 강한 shift에서는 CNN에서 NNGuide, fDBD, CTM 같은 geometry-aware score가 유리하며, ViT에서는 GradNorm과 KPCA reconstruction이 경쟁적이라고 보고합니다. PCA projection이 여러 detector를 개선할 수 있다는 점도 중요합니다.

이 논문은 우리 연구와 매우 잘 맞습니다. 우리의 메시지가 optimizer → representation geometry → detector outcome이라면, 이 논문은 representation/training paradigm이 detector ranking을 바꾼다는 최신 근거로 들어갑니다.

B. Training-induced OOD behavior: WACV 2026

One Model, Many Behaviors: Training-Induced Effects on Out-of-Distribution Detection는 2026년 1월 arXiv, WACV 2026 paper입니다. ResNet-50 architecture를 고정하고, 56개의 ImageNet-trained model과 21개의 post-hoc OOD detector를 8개 OOD test set에서 비교합니다. 핵심 결론은 ID accuracy가 올라간다고 OOD detection이 단조롭게 좋아지는 것이 아니며, training strategy, detector choice, OOD performance 사이에 강한 상호작용이 있어서 universal best detector가 없다는 것입니다.

이건 우리 논문에 거의 필수입니다. reviewer가 “그냥 더 좋은 detector 쓰면 되지 않나?”라고 물을 때, 이 문헌은 training recipe와 detector choice가 얽혀 있고, 단일 detector가 항상 최적이 아니다라는 강한 근거가 됩니다.

C. OOD tuning set 없이 detector를 어떻게 조정할 것인가: 2026

Tuning Out-of-Distribution Detectors Without Given OOD Data는 2026년 2월 arXiv입니다. 이 논문은 OOD detector가 종종 별도의 OOD validation/tuning set에 의존하지만, 실제 deployment에서는 그런 OOD set이 없거나 representative하지 않을 수 있다고 지적합니다. 또한 tuning set 선택에 따라 detector 성능과 ranking이 크게 달라질 수 있다고 보고합니다.

이 논문은 우리 실험 설계에서 중요합니다. 우리가 detector hyperparameter를 고를 때 OOD validation을 쓰면, reviewer가 “실제 OOD를 미리 본 것 아니냐?”라고 지적할 수 있습니다. 그래서 ID-only validation, fixed default, 또는 tuning sensitivity appendix를 명확히 해야 합니다.

D. Mahalanobis++: reviewer objection에 직접 대응하는 최신 방법

Mahalanobis++: Improving OOD Detection via Feature Normalization는 ICML 2025 논문입니다. 이 논문은 pre-logit feature에 Mahalanobis distance를 적용하는 post-hoc method가 ImageNet-scale에서 강하지만 모델별 성능 차이가 크고, 그 원인을 feature norm variation과 Gaussian assumption violation으로 연결합니다. 핵심 처방은 간단한 feature L2-normalization이며, 44개 모델과 다양한 architecture/pretraining setting에서 기존 Mahalanobis를 일관되게 개선한다고 보고합니다.

우리 논문에서는 이 논문이 매우 중요합니다. 현재 우리가 Mahalanobis-L2를 이미 partially 보고 있다면, 논문에서는 이를 Mahalanobis++-motivated diagnostic이라고 정확히 표현하고, 가능하면 full Maha++ protocol 또는 최소한 L2-normalized Maha/kNN/DDU를 더 확장해야 합니다.

3. 현재 사람들이 많이 쓰는 OOD detector 계열

최근 survey 기준으로, post-hoc OOD detection은 대략 다음 계열로 정리됩니다. Task-Oriented Survey의 taxonomy는 output-based에 MSP, MaxLogits, Energy, GEN 등을 두고, distance-based에 Mahalanobis, NNGuide, KNN, SSD를 두며, gradient-based에 GradNorm/GradOrth/GAIA/OPNP, feature-based에 ODIN, ReAct, ViM, Neco, ASH, SCALE, BLOOD 등을 두고, density-based에 GEM/ConjNorm을 둡니다.

우리 논문에서 최소한 커버해야 하는 계열은 다음입니다.

계열	대표 방법	우리 논문에서의 역할
Output/logit	MSP, MaxLogit, Energy, GEN/entropy	calibration/logit-level reliability와 비교
Distance	Mahalanobis, Mahalanobis++, kNN, NNGuide	penultimate geometry 민감도 확인
Density	DDU/GMM, GEM, ConjNorm	covariance/density assumption 확인
Feature shaping/hybrid	ReAct, ASH, ViM, SCALE	“feature degradation을 후처리로 고칠 수 있나?” 확인
Gradient	GradNorm, GradOrth	feature/logit 외 gradient signal이 같은 현상을 보이는지 확인
NC/prototype	Neco, NCI, prototype cosine/NCC	Neural Collapse 기반 detector와 우리 geometry thesis 연결
Test-time adaptive	AUTO, SODA, ATTA, SAL, AdaOOD 등	main보다는 related/future. test-time adaptation은 optimizer-induced geometry를 가릴 수 있음
Foundation/VLM	MCM, CLIPN, NegPrompt, LoCoOp, GL-MCM류	future/pretrained regime. main CIFAR-from-scratch와 분리
4. 지금 논문에 넣으면 좋은 최신 흐름 요약

최근 흐름은 “새 detector 하나”보다 문제 설정과 평가 protocol이 더 정교해지는 방향입니다. 특히 standard near/far OOD만 보는 것에서 벗어나 full-spectrum OOD, semantic/covariate shift 분리, OSR와 OOD의 연결, ImageNet-scale evaluation, tuning set sensitivity가 중요해졌습니다. IJCV 2025 논문은 semantic shift와 covariate shift를 분리하는 large-scale benchmark를 제안하고, OpenOOD v1.5도 full-spectrum OOD와 ImageNet-scale benchmark를 강조합니다.

둘째, post-hoc detector는 여전히 매우 많이 쓰입니다. 이유는 model retraining 없이 적용 가능하고, optimizer나 training recipe가 만든 representation을 그대로 읽을 수 있기 때문입니다. 이 점에서 우리 논문이 post-hoc MSP/Energy/Mahalanobis/DDU를 쓰는 것은 타당합니다. 다만 2025–2026 기준으로는 MaxLogit, kNN/NNGuide, Maha++, ViM, ReAct/ASH, Neco/NCI 정도를 추가해야 “대표성”이 훨씬 강해집니다.

셋째, foundation model/VLM 흐름이 커졌습니다. CLIP-like OOD survey는 OOD detection이 image-only에서 image-text multimodal detector로 이동하고 있다고 설명합니다. Generalized OOD/VLM survey도 CLIP/LVLM 시대에는 기존 OOD/AD/OSR 경계가 다시 흐려진다고 봅니다. 다만 우리 현재 논문의 main claim은 optimizer-induced geometry이므로, VLM은 main experiment가 아니라 “pretrained/foundation model regime remains future work”로 두는 것이 안전합니다.

넷째, training recipe와 detector ranking의 상호작용이 최신 핵심 이슈입니다. WACV 2026 논문은 같은 ResNet-50 architecture에서도 56개의 ImageNet-trained model과 21개 post-hoc detector를 비교했을 때, training strategy와 detector choice가 강하게 얽히며 no single method is universally optimal이라고 보고합니다. 이건 우리 논문의 중심 메시지와 거의 같은 방향입니다.

5. 우리 논문에 바로 반영할 제안

Related Work에는 다음 구조가 가장 좋습니다.

OOD detector landscape.
MSP/MaxLogit/Energy 같은 output score, Mahalanobis/kNN/NNGuide/Maha++ 같은 distance score, DDU/GMM/GEM/ConjNorm 같은 density score, ReAct/ASH/ViM 같은 feature shaping/hybrid score, Neco/NCI 같은 NC-based score를 요약합니다.
Benchmark and evaluation caveats.
OpenOOD v1.5, IJCV 2025 critical analysis, Task-Oriented Survey를 묶어서 “near/far OOD, full-spectrum OOD, semantic/covariate shift, tuning set” 문제를 언급합니다.
Recent representation/training interaction.
2025/2026 systematic analysis와 WACV 2026 training-induced effects 논문을 넣고, 우리 연구가 “detector method proposal”이 아니라 “optimizer-induced representation geometry가 downstream detector에 미치는 영향”을 다룬다고 위치시킵니다.
Why our detector set is representative but not exhaustive.
현재 MSP/Energy/Mahalanobis/DDU만 있으면 약합니다. 최소한 MaxLogit, entropy/GEN-like, kNN, kNN-L2, Mahalanobis-L2/Maha++, ViM, ReAct/ASH, NCI/Neco-lite 중 일부를 추가하면 reviewer 방어력이 커집니다.
6. “더 좋은 방법 쓰면 되지 않나?”에 대한 답변 방향

이 질문에는 이렇게 답하면 됩니다.

Our claim is not that every OOD detector fails under SAM. Rather, recent work shows that detector ranking depends strongly on representation, training recipe, and evaluation protocol. We therefore ask which detector families are robust to optimizer-induced geometry shifts, and which are not.

실험적으로는 다음 순서가 좋습니다.

우선순위	추가할 detector	이유
1	MaxLogit, entropy/GEN	logit family를 MSP/Energy보다 넓힘
2	kNN, kNN-L2, NNGuide-lite	Mahalanobis Gaussian assumption 없이 local geometry 확인
3	Mahalanobis-L2 / full Maha++ protocol	“feature normalization으로 해결되는가?”에 직접 대응
4	ViM	logit + feature hybrid라 reviewer 설득력 큼
5	ReAct, ASH	feature shaping으로 degradation 회복 가능성 확인
6	NCI/Neco-lite/prototype cosine	Neural Collapse theme와 직접 연결
7	PCA/shrinkage GMM/DDU	covariance estimation failure와 raw geometry shift 분리

이렇게 하면 결론이 훨씬 세련됩니다.

일부 detector는 normalization/shaping/PCA로 회복될 수 있다. 그러나 그 회복 자체가 optimizer-induced feature geometry shift가 detector outcome을 결정한다는 증거다. 따라서 accuracy, calibration, MSP/Energy만으로 feature-based uncertainty를 certify할 수 없다.

최종 추천 citation set

논문 Related Work에 우선 넣을 세트는 아래입니다.

OpenOOD v1.5 — benchmark/protocol 기준.
Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances — 전체 taxonomy backbone.
Dissecting OOD Detection and OSR: A Critical Analysis of Methods and Benchmarks — OOD/OSR, semantic/covariate shift, large-scale benchmark 비판.
Generalized OOD Detection and Beyond in Vision Language Model Era — VLM/CLIP 시대의 넓은 흐름.
Recent Advances in OOD Detection with CLIP-Like Models — CLIP-specific 최신 survey.
A Systematic Analysis of OOD Detection Under Representation and Training Paradigm Shifts — representation-centric detector selection.
One Model, Many Behaviors: Training-Induced Effects on OOD Detection — training recipe와 detector ranking 상호작용.
Tuning OOD Detectors Without Given OOD Data — OOD validation/tuning set 문제.
Mahalanobis++ — feature normalization으로 Mahalanobis failure를 완화하는 최신 직접 대응 문헌.

이 세트를 넣으면 현재 논문은 “SAM에서 Mahalanobis/DDU가 나빠졌다”가 아니라, 최신 OOD detection 흐름 속에서 optimizer-induced geometry가 어떤 detector family에 취약하고 어떤 detector family가 회복력을 갖는지 분석하는 논문으로 보일 수 있습니다.
