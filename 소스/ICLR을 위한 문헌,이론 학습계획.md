# ICLR을 위한 학습 계획

**개념 이해 → 수식 이해 → 실험 설계 이해 → 논문 문장으로 전환**되는 구조가 필요하다.
나는 네 계획을 아래처럼 발전시키는 게 좋다고 본다.

**핵심 보완점**

현재 4단계 앞에 0. 최소 수학/표기법을 넣는 게 좋다. 네 논문은 optimizer, feature geometry, covariance, Neural Collapse, OOD metric이 모두 연결되기 때문에 선형대수와 확률 표기법을 모르면 논문을 읽어도 계속 표면만 보게 된다.

학습 전체 구조는 이렇게 잡자.

`0. 최소 수학/표기법
1. 최적화 방법론
2. 신경망 아키텍처 구조
3. 하이퍼파라미터와 실험 프로토콜
4. Metric 정의
5. 통합: optimizer -> geometry -> detector 로 논문 주장 만들기`

**0. 최소 수학/표기법**

목표는 고급 수학을 다 배우는 게 아니라, 논문 수식을 읽을 만큼의 “작전지도”를 만드는 것이다.

반드시 익혀야 할 것:

- 벡터, 행렬, 내적, norm
- gradient가 “각 파라미터 방향으로 loss가 얼마나 변하는지”를 모은 벡터라는 점
- 평균, 분산, 공분산
- eigenvalue/eigenvector, PCA, effective rank
- Gaussian density, log-likelihood
- softmax, cross-entropy
- feature vector, class mean, covariance matrix

이 단계의 산출물은 용어장이면 된다. 예를 들어:

`covariance:
정의: feature들이 평균 주변에서 어떤 방향으로 얼마나 퍼져 있는지 나타내는 행렬.
프로젝트 연결: Mahalanobis/DDU는 covariance를 직접 쓰므로 optimizer가 covariance를 바꾸면 detector score가 바뀔 수 있다.`

**1. 최적화 방법론**

여기가 지금 가장 중요한 축이다. 순서는 이렇게 가면 좋다.

1. Gradient Descent / SGD

    전체 데이터 gradient 대신 mini-batch gradient로 업데이트한다는 개념.

2. Mini-batch와 stochasticity

    batch size가 gradient noise, update 횟수, generalization, geometry에 미치는 영향.

3. Momentum / Nesterov

    이전 gradient 방향을 누적해서 더 안정적으로 이동하는 방식.

4. Adam

    m_t: gradient 방향의 이동평균

    v_t: gradient 제곱의 이동평균

    m_hat / sqrt(v_hat): 좌표별 adaptive update

5. Weight decay와 L2 regularization

    SGD에서는 거의 같은 것처럼 볼 수 있지만 Adam에서는 다르다는 점.

6. AdamW

    loss gradient update와 weight shrink를 분리한 optimizer.

7. SAM / ASAM / GSAM

    네 NeurIPS 2026 논문과 ICLR 확장의 직접 배경. 단순 accuracy optimizer가 아니라 feature geometry를 바꿀 수 있는 intervention으로 이해해야 한다.


읽을 소스:

- Adam SOURCE_CARD.md
- AdamW SOURCE_CARD.md
- Marginal Value SOURCE_CARD.md
- SWATS SOURCE_CARD.md
- Optimizer choice / Neural Collapse SOURCE_CARD.md

이 단계의 산출물은 optimizer 비교표다.

`SGD / SGDW / Adam / AdamW / SAM / ASAM / GSAM
- 업데이트 수식
- 등장 목적
- 조절하는 hyperparameter
- 장점
- 단점
- feature geometry에 영향을 줄 수 있는 경로
- 이 repo에서 실험해야 하는 이유`

**2. 신경망 아키텍처 구조**

이 단계는 “모델 구조가 feature geometry를 어떻게 만든다”는 관점으로 공부해야 한다.

핵심 항목:

- Linear layer: feature를 class logit으로 바꾸는 마지막 층
- CNN: 이미지에서 local pattern을 추출
- ResNet: residual connection으로 깊은 네트워크 학습 안정화
- WideResNet: 폭을 넓힌 ResNet, CIFAR 실험에서 강한 baseline
- VGG: residual 없는 비교 아키텍처
- ViT: patch embedding, attention, CLS token / mean pooling
- BatchNorm / LayerNorm: feature scale과 training dynamics에 영향
- Dropout: regularization, feature 분산에 영향
- Spectral normalization: DDU 계열에서 feature density와 Lipschitz 성질에 영향

이 단계의 중요한 질문은 이것이다.

`같은 optimizer 효과를 주장하려면 architecture가 confound가 되지 않도록 어떻게 고정하거나 분리할 것인가?`

즉 논문에서는 “AdamW가 나쁘다/좋다”가 아니라, “WRN-28-10, ResNet-18, ViT 등에서 optimizer axis가 어떤 geometry regime을 만드는가”로 봐야 한다.

**3. 하이퍼파라미터와 실험 프로토콜**

하이퍼파라미터는 단순 설정값이 아니라, 논문에서 causal claim을 흐릴 수 있는 confound다.

반드시 정리할 항목:

- learning rate
- scheduler: step decay, cosine, warmup
- batch size
- epoch 수
- optimizer betas / momentum / nesterov
- weight decay 크기
- weight decay policy: bias/norm에 decay를 줄지 말지
- Adam vs AdamW vs custom coupled-decoupled ratio
- SAM rho
- augmentation
- seed 수
- checkpoint 선택 기준
- detector hyperparameter: PCA, shrinkage, k for kNN, covariance type
- feature normalization 여부

이 단계의 산출물은 실험 설정-영향 매트릭스다.

`batch size:
직접 영향: gradient noise, update 횟수
간접 영향: generalization, feature dispersion, NC metric
논문 위험: batch size 차이를 optimizer 효과로 오해할 수 있음
통제 방법: optimizer 비교에서 batch size 고정, 필요하면 64/128/256 ablation`

**4. Metric 정의**

여기는 논문 작성에서 매우 중요하다. metric을 모르면 결과 해석이 흔들린다.

분류해서 공부하자.

ID 성능:

- Accuracy
- NLL
- ECE
- TCE

Logit-based OOD:

- MSP
- Energy
- MaxLogit
- Negative entropy

Feature-based OOD:

- Mahalanobis
- DDU/GMM
- kNN
- ViM

Geometry / Neural Collapse:

- within-class variance
- inter-class distance
- covariance eigen-spectrum
- feature norm distribution
- anisotropy
- effective rank
- NC1
- NC2 / NC2 angle
- NC3 self-duality
- NC4 nearest-class-center agreement
- repo의 revised names: nc0_width_norm, nc3_self_duality, nc4_agreement, inter_dist_l2

읽을 소스:

- NeurIPS 2026 context
- Mahalanobis++ SOURCE_CARD.md
- NECO SOURCE_CARD.md
- ViM SOURCE_CARD.md
- OpenOOD SOURCE_CARD.md
- Metric definitions

**5. 통합 단계**

마지막 단계는 문헌을 “요약”하는 게 아니라, 네 논문 주장의 부품으로 재배치하는 것이다.

논문 주장은 이렇게 조립해야 한다.

`Adam 원논문:
Adam은 adaptive moment 기반 optimizer다.

AdamW 논문:
Adam에서 L2 regularization과 decoupled weight decay는 다르다.

Adaptive optimizer/generalization 문헌:
optimizer는 단순 convergence speed만 바꾸는 것이 아니라 다른 solution을 만들 수 있다.

Optimizer/NC 논문:
optimizer choice와 coupled/decoupled weight decay는 Neural Collapse / representation geometry를 바꿀 수 있다.

NeurIPS 2026 논문:
feature-based uncertainty detector는 optimizer-induced feature geometry 변화에 취약할 수 있다.

Mahalanobis++ / NECO / ViM / OpenOOD:
feature geometry와 detector score의 관계를 OOD 문헌 안에서 위치시킨다.`

여기서 지켜야 할 경계도 분명하다.

`AdamW 논문만으로 OOD detector 성능을 주장하면 안 된다.
Optimizer/NC 논문만으로 Mahalanobis/DDU 실패를 주장하면 안 된다.
NeurIPS 2026 결과와 새 ICLR 실험이 optimizer -> geometry -> detector 연결을 직접 담당해야 한다.`

**추천 학습 산출물**

매 논문마다 단순 요약 말고 아래 형식으로 정리하는 게 좋다.

`논문명:
내 논문에서의 역할:
핵심 질문:
핵심 수식:
핵심 실험:
confirmed claim:
interpretation:
내 ICLR 논문에서 쓸 수 있는 문장:
쓰면 안 되는 주장:
다음 실험으로 확인해야 할 것:`

이렇게 하면 흩어진 문헌이 하나의 논문 구조로 모인다.

**현실적인 순서**

1주차: 수학/표기법 + SGD/mini-batch/momentum

2주차: Adam, AdamW, weight decay coupling

3주차: optimizer generalization + optimizer/Neural Collapse 논문

4주차: WRN/ResNet/ViT 구조와 penultimate feature 개념

5주차: OOD detector와 metric 정의

6주차: 전체 claim map 작성

7주차 이후: 실험 결과를 claim map에 연결하고 related work / preliminaries 초안 작성

나는 특히 2주차와 3주차를 깊게 가져가는 게 좋다고 본다. 이 프로젝트의 독창성은 “AdamW가 좋냐 나쁘냐”가 아니라, **optimizer와 weight-decay coupling이 feature geometry regime을 바꾸고, 그 regime이 detector별로 다르게 작동한다**는 데 있으니까.