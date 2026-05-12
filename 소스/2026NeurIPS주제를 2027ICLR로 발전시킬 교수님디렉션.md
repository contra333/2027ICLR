맞습니다. ASAM/GSAM에서 detector degradation이 덜하거나 사라졌다는 것은 “SAM의 단점이 보완되었다”로도 읽힐 수 있습니다. 그런데 논문적으로 더 중요한 해석은 이것입니다.

optimizer가 generalization/calibration을 개선하는 방식이 다르면, penultimate feature geometry에 남기는 footprint도 다르고, 따라서 detector outcome도 달라진다.

즉, ASAM/GSAM은 vanilla SAM의 단점을 단순히 “고친” 것이 아니라, 다른 sharpness-aware intervention을 통해 다른 geometry를 유도한 것입니다. 이게 핵심입니다.

왜 ASAM/GSAM은 detector degradation이 덜할 수 있나

Vanilla SAM은 고정 반경 ρ 안에서 최악 방향 perturbation을 줍니다. 이때 weight scale이나 feature norm, class covariance 구조에 영향을 줄 수 있고, 그 결과 within-class dispersion이나 covariance scale이 커질 수 있습니다. 그러면 Mahalanobis/DDU처럼 inverse covariance와 density 구조에 의존하는 detector가 불리해집니다.

ASAM은 SAM의 sharpness 측정이 parameter scaling에 민감하다는 문제를 보완하려고 나온 방법입니다. 즉, 같은 함수라도 weight scale이 다르면 sharpness가 다르게 보이는 문제를 줄이려 합니다. 그래서 ASAM은 vanilla SAM보다 feature norm/covariance scale을 덜 과하게 밀어낼 가능성이 있습니다. 이 경우 DDU/Mahalanobis가 덜 망가지거나 오히려 좋아질 수 있습니다.

GSAM은 SAM의 update가 generalization에 유리한 방향과 그렇지 않은 방향을 섞어버리는 문제를 줄이려는 쪽입니다. 그래서 vanilla SAM보다 feature spread를 키우는 방향은 억제하고, class separation을 유지하는 쪽으로 geometry가 남을 수 있습니다. 실제로 GSAM에서 InterDist가 더 좋아진다면, detector 입장에서는 유리합니다.

그래서 ASAM/GSAM 결과는 이렇게 해석하는 것이 좋습니다.

vanilla SAM은 output-level reliability 개선과 feature geometry 왜곡이 함께 나타나는 사례이고, ASAM/GSAM은 sharpness-aware optimization 안에서도 geometry footprint가 달라질 수 있음을 보여준다.

이것은 논문 주장을 약화시키는 것이 아니라 오히려 강화합니다.

AdamW는 왜 NC 구조를 더 망가뜨릴 수 있나

AdamW는 SAM-family와는 완전히 다른 implicit bias를 가집니다. 중요한 차이는 세 가지입니다.

첫째, coordinate-wise adaptive update입니다. AdamW는 파라미터마다 다른 effective learning rate를 적용합니다. 이 때문에 SGD보다 feature space가 더 anisotropic하게 형성될 수 있습니다. SGD는 상대적으로 전체 방향을 균일하게 밀면서 class mean과 classifier가 neural collapse 구조로 정렬되는 경향이 있는데, AdamW는 좌표별 적응 update 때문에 특정 방향의 분산이나 class 구조가 더 불균형해질 수 있습니다.

둘째, decoupled weight decay입니다. AdamW는 weight decay를 gradient update와 분리합니다. 이게 optimization에는 유리하지만, feature norm과 classifier norm의 성장/균형을 SGD와 다르게 만듭니다. Neural collapse는 feature norm, classifier norm, class mean geometry의 균형과 관련이 있기 때문에, AdamW의 norm dynamics가 NC 구조를 약화시킬 수 있습니다.

셋째, margin geometry가 다르게 형성됩니다. Cross-entropy로 오래 학습하면 SGD는 암묵적으로 margin 구조와 class separation을 정리하는 경향이 있습니다. AdamW는 loss는 잘 줄이지만 feature cloud를 compact하게 만들거나 class means를 simplex-like하게 정렬하는 방향의 implicit bias가 SGD와 다를 수 있습니다. 그래서 accuracy는 좋아도 NC1, WithinVar, InterDist, NC2A는 더 나빠질 수 있습니다.

즉, AdamW에서 NC 구조가 SGD보다 망가진다면 그건 이상한 결과가 아닙니다. 오히려 이렇게 해석할 수 있습니다.

AdamW는 output optimization에는 강하지만, feature-based detector가 선호하는 class-conditional geometry를 반드시 보존하지 않는다.

이건 우리 논문의 후속 연구로 매우 좋습니다. SAM-family 안에서만이 아니라, adaptive optimizer도 detector geometry를 다르게 만든다는 방향으로 확장할 수 있습니다.

기능 차이를 한 줄로 정리하면

SGD는 비교적 단순한 update와 weight decay 구조를 통해 class separation과 collapse-like geometry를 만들 가능성이 큽니다.

Vanilla SAM은 sharpness를 줄이는 과정에서 within-class dispersion과 covariance scale을 키울 수 있습니다.

ASAM은 scale-sensitive한 sharpness 문제를 줄여 norm/covariance distortion을 완화할 수 있습니다.

GSAM은 sharpness-aware update 중 generalization에 불필요한 성분을 줄여 class separation을 더 보존할 수 있습니다.

AdamW는 adaptive coordinate-wise update와 decoupled weight decay 때문에 feature geometry를 더 anisotropic하고 less-collapse된 형태로 만들 수 있습니다.

그래서 후속 논문의 핵심 질문은 이렇게 잡으면 좋습니다.

Different optimizers improve training or generalization through different implicit biases, and these biases leave different geometric footprints in the penultimate feature space.

ViT에서는 어떻게 될까

ViT는 CNN과 상당히 다르게 나올 가능성이 큽니다. 저는 세 가지 시나리오를 예상합니다.

첫째, from-scratch ViT on CIFAR에서는 geometry가 더 불안정할 수 있습니다. ViT는 convolutional inductive bias가 약하고, 작은 데이터에서는 representation이 더 불균형하거나 class-conditional covariance가 불안정할 수 있습니다. 이 경우 Mahalanobis/DDU 성능 자체가 CNN보다 더 민감할 수 있습니다. AdamW는 ViT의 기본 optimizer처럼 쓰이므로, AdamW-induced NC degradation이 더 뚜렷하게 보일 가능성도 있습니다.

둘째, pretrained ViT 또는 vision foundation model feature에서는 optimizer effect가 약해질 수 있습니다. 이미 큰 데이터로 학습된 representation은 geometry가 어느 정도 안정화되어 있어서, fine-tuning optimizer가 penultimate geometry를 크게 바꾸지 않을 수 있습니다. 이 경우 vanilla SAM degradation이 약하게 나타나거나 사라질 수 있습니다. 하지만 이것은 논문을 약화시키는 결과가 아니라, “pretraining이 optimizer-induced geometry shift를 mask하거나 absorb한다”는 중요한 regime finding이 됩니다.

셋째, LayerNorm과 token representation 때문에 feature norm 효과가 달라질 수 있습니다. ViT는 LayerNorm이 많아서 feature norm variation이 CNN보다 다르게 나타날 수 있습니다. 그래서 L
2
	

 normalization의 효과도 CNN과 다를 수 있습니다. Mahalanobis/DDU가 CLS token feature에서 잘 작동할지, pooled token feature에서 더 안정적인지도 비교해야 합니다.

제 예상은 이렇습니다.

ViT from scratch: AdamW/SAM 모두 geometry가 불안정하고 detector 결과가 dataset-dependent하게 나올 가능성 큼.
Pretrained ViT: output metric은 좋아도 feature-based detector 변화는 작거나, detector-side covariance estimation에 더 민감할 가능성 큼.
CLIP/DINO features: optimizer보다는 pretrained representation geometry와 detector pipeline의 영향이 더 클 가능성 큼.

다음 실험은 이렇게 하면 좋습니다

가장 먼저 할 실험은 너무 크게 가지 말고, 다음 순서가 좋습니다.

CIFAR-10 ID, ResNet-18 또는 WRN에서 SGD / AdamW / SAM / ASAM / GSAM 비교
지표는 Accuracy, ECE, MSP, Energy, DDU, Mahalanobis, NC1, WithinVar, InterDist, eigenspectrum입니다.
AdamW에서 NC 구조가 무너지는지 정량화
특히 AdamW가 accuracy는 좋지만 NC1/WithinVar/NC2A가 나빠지는지 봅니다.
ViT-Tiny 또는 DeiT-Tiny from scratch
SGD는 ViT에서 잘 안 맞을 수 있으니 AdamW baseline, AdamW+SAM, AdamW+ASAM 정도로 시작합니다.
pretrained ViT feature에서 frozen-feature detector 실험
optimizer effect를 제거하고, detector pipeline 자체의 covariance sensitivity를 봅니다.
fine-tuned pretrained ViT에서 optimizer별 geometry 변화
pretrained regime에서 optimizer–geometry–detector interaction이 나타나는지, 약해지는지, 가려지는지를 봅니다.
후속 논문 주제는 이렇게 잡을 수 있습니다

가장 좋은 후속 질문은 이것입니다.

When do optimization improvements preserve, distort, or mask the geometry required by feature-based uncertainty?

이건 ICLR/ICML 후속으로 충분히 좋습니다.

그리고 방법론 논문으로 가려면:

Geometry-preserving optimization for feature-based uncertainty

즉, accuracy/calibration을 유지하면서 NC1, WithinVar, covariance spectrum, class separation을 detector-friendly하게 보존하는 optimizer 또는 regularizer를 제안할 수 있습니다.

정리하면, 지금 관찰한 AdamW와 ViT 질문은 매우 중요합니다. 특히 AdamW가 NC 구조를 망가뜨린다는 결과는 후속 논문의 좋은 축입니다. 그것은 단순히 “SAM만의 현상”을 넘어, optimizer implicit bias가 feature-based reliability를 좌우한다는 더 큰 메시지로 확장시켜 줍니다.