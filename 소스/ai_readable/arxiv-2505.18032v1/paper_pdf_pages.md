# Mahalanobis++: Improving OOD Detection via Feature Normalization - page-anchored PDF text

- Source ID: `arxiv-2505.18032v1`
- arXiv ID: `2505.18032v1`
- Original PDF: `소스/Mahalanobis++_ Improving OOD Detection via Feature Normalization (1).pdf`
- PDF pages: 34
- Extracted with: WSL poppler `pdftotext -f N -l N` on 2026-05-12T20:50:45+09:00
- Evidence note: use this file for page-anchored claims; verify equations/tables against the original PDF when exact layout matters.

## Page 1

Mahalanobis++: Improving OOD Detection via Feature Normalization

Maximilian Müller 1 Matthias Hein 1

FPR Improvement of Mahalanobis++ over Mahalanobis

Detecting out-of-distribution (OOD) examples
is an important task for deploying reliable machine learning models in safety-critial applications. While post-hoc methods based on the Mahalanobis distance applied to pre-logit features
are among the most effective for ImageNet-scale
OOD detection, their performance varies significantly across models. We connect this inconsistency to strong variations in feature norms,
indicating severe violations of the Gaussian assumption underlying the Mahalanobis distance estimation. We show that simple ℓ2 -normalization
of the features mitigates this problem effectively,
aligning better with the premise of normally distributed data with shared covariance matrix. Extensive experiments on 44 models across diverse
architectures and pretraining schemes show that
ℓ2 -normalization improves the conventional Mahalanobis distance-based approaches significantly
and consistently, and outperforms other recently
proposed OOD detection methods. Code is available at github.com/mueller-mp/maha-norm.

FPR Difference

arXiv:2505.18032v1 [cs.LG] 23 May 2025

Abstract
15
10
5
0

Models sorted by FPR difference
Figure 1. Normalizing features improves OOD detection with
the Mahalanobis distance consistently. Shown is the difference
in false-positive rate at true positive rate of 95% between unnormalized and normalized features for 44 ImageNet models, averaged
over five OOD datasets of the OpenOOD benchmark.

genuine ID samples to pass through normally. OOD detection methods are commonly divided into methods that
require modifications to the training process and so-called
post-hoc detection methods that can be applied to any pretrained network. For many downstream tasks (not only
OOD detection), the best results are achieved by models
that have been pretrained on large datasets, some of which
might not be publicly available. Since adjusting the training
scheme for these networks is usually not feasible, simple
post-hoc OOD detection is most often used in practice.

1. Introduction
Deep neural networks have demonstrated remarkable performance across a variety of real-world tasks. However, when
faced with inputs that fall outside their training distribution, they can behave unpredictably and even result in highconfidence predictions (Hendrycks & Gimpel, 2017; Hein
et al., 2019). These so-called out-of-distribution (OOD) inputs are often misclassified with high confidence as belonging to the in-distribution (ID) classes, creating significant
risks for real-world deployments. OOD detectors aim to
identify and reject such anomalous inputs — potentially
prompting human intervention, transitioning to a safe state,
or declining to provide a prediction — while still allowing

Common post-hoc OOD detection methods are based on a
scoring function that typically inputs either the logit/softmax
outputs of a model (Hendrycks & Gimpel, 2017; Hendrycks
et al., 2022; Liu et al., 2020), or the pre-logit features (Lee
et al., 2018b; Ren et al., 2021; Sun et al., 2022), or both
(Sun et al., 2021; Wang et al., 2022). VisionTransformers
have shown particular success in this area (Koner et al.,
2021). For large-scale settings where, e.g., ImageNet is
the ID dataset, they perform particularly well (Galil et al.,
2023), especially when paired with feature-based methods
(Bitterwolf et al., 2023). Among those, the Mahalanobis
distance (Lee et al., 2018b; Ren et al., 2021) stands out as a
particularly effective and simple scoring function. However,
despite leading for some models to state-of-the-art OOD
performance, it fails for others and shows high performance
variation across different models and pretraining schemes,
and brittleness when confronted with supposedly easy noise
distributions as OOD data (Bitterwolf et al., 2023).

1

University of Tübingen and Tübingen AI Center. Correspondence to: Maximilian Müller <maximilian.mueller@wsii.unituebingen.de>.
Proceedings of the 42 nd International Conference on Machine
Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).

1

## Page 2

Mahalanobis++: Improving OOD Detection via Feature Normalization

Unnormalized

Normalized

Class 10
Class 300
Class 633
global fit

Class 10
Class 300
Class 633
global fit

ID
OOD
TPR=95%

ID
OOD
TPR=
95%

FPR = 34.58%
sMaha

FPR = 18.42%

sMaha + +

Figure 2. Mahalanobis++: We illustrate how to improve Mahalanobis-based OOD detection. Left: For unnormalized features, assuming
a shared covariance matrix for all classes leads to suboptimal OOD detection (bottom) with the Mahalanobis score. Center: Normalizing
the features, i.e. projecting them onto the unit sphere mitigates this problem effectively. Right: After normalization, the fit of the shared
covariance matrix is tighter for all classes, leading to improved OOD detection as in- and out-distribution are better separated. Shown are
the Mahalanobis++ scores for a pretrained ConvNextV2-L on NINCO, which achieves a new state-of-the-art FPR of 18.4% (see Tab. 6).

2. Related Work

In this work, we observe that for models where the Mahalanobis distance does not work well as OOD detector, the
assumptions underlying the method are often not well satisfied. In particular, the feature norms vary much more than
expected when assuming a Gaussian model with a shared
covariance matrix. To mitigate this problem, we provide a
simple solution, called Mahalanobis++, which we visualize
in Figure 2: By projecting the features onto the unit sphere
before estimating the Mahalanobis distance, we significantly
reduce the class-dependent feature variability and obtain a
better fit of the covariance matrix, which ultimately leads
to consistent improvements in OOD detection, as demonstrated in Fig. 1 or Tab. 4.

Mahalanobis distance. Most closely related to our work are
the well-established OOD detection methods based on the
Mahalanobis distance. Lee et al. (2018b) proposed to estimate a class-conditional Gaussian distribution with a shared
covariance matrix ”with respect to (low- and upper-level)
features”, and to use the minimal Mahalanobis distance to
the respective mean vectors as OOD score. Since then, the
community has transitioned to using only the pre-logit features. Ren et al. (2021) proposed to additionally estimate
a class-agnostic mean and covariance matrix and use the
difference between the two resulting scores as OOD score,
called relative Mahalanobis distance. These methods have
demonstrated broad applicability, spanning domains such
as medical imaging (Anthony & Kamnitsas, 2023) and selfsupervised OOD detection (Sehwag et al., 2021). Gaussian
mixture models (GMMs) represent a more comprehensive
framework for modelling feature distributions. They have
been applied to small-scale setups but require tweaks to
the training process (e.g. spectral normalization) (Mukhoti
et al., 2023). Adapting them to ImageNet-scale setups as
post-hoc OOD detectors has so far not been successful.

In summary, our contributions are the following:
• We observe that the assumptions underlying the Mahalanobis distance as OOD detection method, in particular that the features are normally distributed with a
shared covariance matrix, are often not well satisfied
• We relate this to variations in the feature norm, which
can vary strongly across and within classes, and correlates with the Mahalanobis distance

Feature norm. The role of the feature norm for OOD detection has been investigated in several works (Yu et al., 2020).
Park et al. (2023b) underline that the norm of pre-logit features are equivalent to confidence scores and that the feature
norms of OOD samples are typically smaller than those of
ID samples. Their observations are mostly based on results
obtained with strong over-training and simple networks. We
will show that this observation does not hold generally. Gia
& Ahn (2023) investigate the role of the ℓ2 norm in contrastive learning and OOD detection. Regmi et al. (2024)

• We provide an easy solution, which we call Mahalanobis++: Normalizing the features by their ℓ2 -norm
before computing the Mahalanobis distance
• We evaluate Mahalanobis++ across a large range of
models with different pretraining schemes and architectures on ImageNet and Cifar datasets and find that
it consistently outperforms the conventional Mahalanobis distance and other baseline methods, and improves the detection of far-OOD noise distributions
2

## Page 3

Mahalanobis++: Improving OOD Detection via Feature Normalization

and Haas et al. (2024) try to leverage the feature norm to discriminate between ID and OOD samples. In particular, they
concurrently suggested training with ℓ2 -normalized features
and then using the norm of the unnormalized features as
OOD score at inference time, similar to Yu et al. (2020) and
Wei et al. (2022).

where OOD detection on NINCO with Mahalanobis score
performs significantly worse (FPR of 58.2%) than for other
similar models like the ViT-B16-In21k-augreg with 84.5%
accuracy (Steiner et al., 2022) but low FPR of 31.3% using
the Mahalanobis score.

Spherical embeddings. Spherical embeddings have been
investigated and leveraged across several fields (Liu et al.,
2018; Zhou et al., 2022; Sablayrolles et al., 2018; Yaras et al.,
2022), also within the OOD detection literature (Zheng et al.,
2022). Ming et al. (2023) proposed CIDER, a contrastive
training scheme that creates well-separated hyperspherical
embeddings via a dispersion loss and applies KNN as detection method at inference time. Sehwag et al. (2021) also
train with a contrastive loss, and apply the Mahalanobis distance as OOD detection method on the normalized features
at inference time. Haas et al. (2023) observe that normalizing features during train and inference time improves performance on the DDU benchmark (Mukhoti et al., 2023). They
hypothesize that their training scheme induces early neural
collapse, which might benefit out-of-distribution detection
capabilities of networks. Importantly, all those methods are
train-time methods, i.e. require modifications to the training
process, including feature normalization - either explicitly
in the case of Haas et al. (2023), or implicitly through the
contrastive loss in Ming et al. (2023) and Mukhoti et al.
(2023). They then apply normalization at inference time,
because they also normalized at train time. In contrast, we
highlight the benefits of feature normalization when applying the Mahalanobis distance as post-hoc OOD detection
method in this work - which is non-obvious for generic
pretraining schemes.

3.1. Mahalanobis Distance
The Mahalanobis distance is a simple, hyperparameter-free
post-hoc OOD detector that has been suggested by Lee et al.
(2018b). Given the training set (xi , yi )ni=1 with input xi and
class labels yi one estimates: i) the class-wise means µ̂c and
ii) a shared covariance matrix Σ̂:
µ̂c =

1 X
ϕ(xi )
Nc i:y =c

(1)

i

Σ̂ =

C
1 X X

N

c

(ϕ(xi ) − µ̂c )(ϕ(xi ) − µ̂c )T

(2)

i:yi =c

where ϕ(xi ) are the pre-logit features of xi , Nc the number
of train samples in class c, N the total number of train samples, and C the total number of classes. The Mahalanobis
distance of a test sample xt to a class mean µ̂c is then
dM aha (xt , µ̂c ) = (ϕ(xt ) − µ̂c )T Σ̂−1 (ϕ(xt ) − µ̂c )

(3)

and the final OOD-score sMaha (xt ) of xt is the negative
smallest distance to one of the class means:
sMaha (xt ) = − min dM aha (xt , µ̂c )
c

(4)

If sMaha (xt ) ≤ T then the sample is rejected as OOD, where
for evaluation purposes T is typically determined by fixing
a TPR of 95% on the in-distribution. The core assumption
of Lee et al. (2018a) is that “the pre-trained features of
the softmax neural classifier might also follow the classconditional Gaussian distribution”. Indeed, one implicitly
uses a probabilistic model where each class is modelled as
a Gaussian N (µ̂c , Σ̂) with a shared covariance matrix Σ̂,
which can be seen as a weighted average of thePcovariance
C
matrices of the features of each class: Σ̂ = c=1 NNc Σ̂c
P
with Σ̂c = N1c i:yi =c (ϕ(xi ) − µ̂c )(ϕ(xi ) − µ̂c )T , with the
weight Nc \N being an estimate of P(Y = c).

Cosine-based detection scores. Many previous works have
suggested using the angle, or more specifically, the cosine,
for OOD detection, but those mostly require modifications
to training or architecture (Techapanurak et al., 2020; Tack
et al., 2020), or are used for unsupervised setups (Radford
et al., 2021; Ming et al., 2022). Park et al. (2023a) and Sun
et al. (2022) use nearest neighbour search in the normalized feature-space, which amounts to a nearest neighbour
search in the cosine space. We show that Mahalanobis++
outperforms cosine-based OOD detection methods.

3. Variations in feature norm degrade the
performance of Mahalanobis-based OOD
detectors

The Mahanalobis score is a strong baseline for OOD detection as noted in Bitterwolf et al. (2023) where they report for
a particular Vision Transformer (ViT) trained with augreg (a
carefully selected combination of augmentation and regularization techniques) by Steiner et al. (2022) state-of-the-art
results on their NINCO benchmark comparing several models and OOD detection methods. On the other hand other
ViTs like DeiT or Swin that are equally strong in terms of
classification performance showed degraded OOD detection
results. Moreover, Bitterwolf et al. (2023) report that the

In this Section, we investigate the assumptions underlying
the Mahalanobis distance as OOD detection method. We
report results for NINCO (Bitterwolf et al., 2023) as OOD
dataset. For all experiments, we use a pretrained ImageNet
SwinV2-B-In21k model (Liu et al., 2022) with 87.1% ImageNet accuracy. This strong model is a prototypical example
3

## Page 4

Mahalanobis++: Improving OOD Detection via Feature Normalization

Mahalanobis-based OOD detector performs worse on their
“unit tests” of simple far-OOD test sets than other methods.

particular class are Gaussian distributed. In particular, the
feature norm would be concentrated, as the following lemma
shows.

In the remainder of this section, we will try to identify the
reasons for the varying performance of Mahalanobis-based
OOD detection. Our main hypothesis is that it is due to
violations of its core assumptions:

Lemma 3.1. Let Φ(X) ∼ N (µ, Σ). Then


2

where Var(∥Φ(X)∥2 ) :=

• Assumption II: the covariance matrix Σ̂ is the same
for all classes.

|| i ||

Observed

Avg. Norm
Min/Max
Std Dev

30
20
0

500

Sorted Classes

1000

0

500

Sorted Classes

ϵ2

(3λ2i +6µ2i λi +µ4i )−(λi +µ2i )2

i=1

This
q implies that ∥Φ(X)∥2 should be concentrated around
2
tr(Σ) + ∥µ∥2 . In the right part of Fig. 3, we show the
distribution of the norms of the training features across
classes for the SwinV2-B model, i.e. the feature norms of
those samples that were used for estimating class means and
covariance. In the left part of Fig. 3 we show the distribution
of feature norms when sampling from N (µ̂c , Σ̂) for every
class c. As expected from the derived Lemma, the sampled
norms vary little around their mean value. It is evident by the
differences of the left and right part of Fig. 3, that the fit with
class-conditional means and shared covariance matrix does
not represent the structure of the data well as the observed
feature norms of SwinV2-B show heavy tails (right) which
would not be present if the data was Gaussian (left). In
Figure 8 we show that similar heavy-tailed feature norm
distributions but with different skewness can be found even
for the same ViT-architecture where the Mahalanobis score
does not work well. This shows that Assumption I of the
Mahalanobis score is not fulfilled across models, and models
can deviate heavily from it. In contrast, for the ViT-augreg
(Steiner et al., 2022), which has been shown to have very
good OOD detection performance with the Mahalanobis
score (Bitterwolf et al., 2023), the feature norms behave
roughly as expected under the Gaussian assumption (right
plot in Figure 8).

For completeness, we mention the Relative Mahalanobis
score here, proposed by Ren et al. (2021), also suggested
as a fix to the Mahalanobis score. They argue that for the
detection of near-OOD, one should use a likelihood ratio of
two generative models compared to the likelihood used in
the Mahalanobis method. Thus they fit a global Gaussian
distribution with mean µ̂global and covariance matrix Σ̂global ,
and use the difference between the class-conditional and the
global Mahalanobis score as OOD score.

40

d
P



2
Var ∥Φ(X)∥2

and (λi )di=1 are the eigenvalues of Σ.

Below, we will show that these assumptions do not hold for
some models, as indicated in Fig. 2. One strong indicator
of this violation is the norm of the features, which turns out
to be a strong confounder, ultimately degrading the OOD
detection performance with Mahalanobis-based detectors.

50

2

P | ∥Φ(X)∥2 − (tr(Σ) + ∥µ∥2 )| ≥ ϵ ≤

• Assumption I: the class-wise features (ϕ(xi ))yi =c follow a multivariate normal distribution N (µc , Σ),

Expected

2



1000

To further evaluate the adherence to Assumption I, we center training features of the SwinV2-B by their class means:
ϕcenter (xi ) = ϕ(xi )−µc[i] . These centered features, used for
covariance estimation, should ideally follow a zero-mean
multivariate normal distribution. To quantify deviations
from normality, we use Quantile-Quantile (QQ) plots, a standard approach in statistics (see, e.g. Wilk & Gnanadesikan
(1968)) which compares sample quantiles against those of
a theoretical distribution (here, the standard normal). A
straight diagonal line indicates agreement with the theoretical distribution; deviations highlight mismatches. To
enable direct comparison between models (and later between normalized and unnormalized features), we standardize ϕcenter (xi ) by its empirical standard deviation. While
standardization technically alters the distribution (as the empirical variance is sample-dependent), we expect this to be

Figure 3. The feature norms vary strongly across and within
classes. Left: We simulate how the feature norms per class would
be distributed if they were sampled from Gaussians with the means
and covariance matrix used for the Mahalanobis distance estimation. Right: The actual feature norm distribution observed in
practice. Both the average norms across classes and the norms
within each class vary much stronger than expected.

3.2. Is the Gaussian fit in feature space justified?
As the features ϕ(xi ) ∈ Rd for input x are high-dimensional,
e.g. d = 1024 for a SwinV2-B, we expect some concentration of measure phenomena if the features ϕ(x) of a
4

,

## Page 5

Mahalanobis++: Improving OOD Detection via Feature Normalization
Table 1. Variance alignment. We measure how much the classvariances deviate from the global variance via the deviation score
(see Eq. 5). Lower values indicate better alignment. Normalization
aligns the features of SwinV2 and DeiT3, but not ViT-augreg.

SwinV2-B-In21k
DeiT3-B16-In21k
ViT-B16-In21k-augreg

unnormalized

normalized

0.26
0.24
0.05

0.12
0.15
0.05

negligible due to the large dataset size (> 106 samples). We
report QQ-plots for three directions for a SwinV2-B and a
DeiT3 (blue lines in Figure 4), and observe strong deviations
from the ideal diagonal line, indicating that the centered features have much stronger tails than expected if the features
followed a Gaussian distribution, further refuting Assumption I. We observe similar heavy tails in QQ-plots of other
models where the Mahalanobis score is not working well
for OOD detection (see Fig. 9 in App. D). Only the ViT with
augreg training has a QQ plot close to the expected one.
Figure 4. QQ-plot: ℓ2 −normalization helps transform the features to be more aligned with a normal distribution. For a
SwinV2 and DeiT3 model (where the feature norms vary strongly
across and within classes) normalization shifts the distribution
towards a Gaussian (black line).

To assess the validity of Assumption II, we measure how
strongly the individual class variances deviate from the
global variance. To this end, we compute the expected
relative deviation over all directions:
Eu [(uT Au)2 ] =

2tr(A2 ) + tr(A)2
,
d(d + 2)

norm across classes observed above. In Fig. 7, we observe
the same correlation for other models (again, the ViT-Baugreg being an exception). In Fig. 6 in the Appendix, we
substantiate this observation by artificially scaling the feature norm of OOD samples, leading to improved detection
when the feature norm is increased and worse detection
when the feature norm is decreased.

(5)

where u has a uniform distribution on the unit sphere and
1
1
A = Σ̂− 2 (Σ̂i − Σ̂)Σ̂− 2 (see App. C for a derivation). We
average over all classes i and report the results for a SwinV2B, a DeiT3-B and a ViT-augreg in Table 1. We observe that
the SwinV2 and DeiT3 show significantly larger deviations
than the ViT-augreg, indicating that the class-specific variances differ more. More models in Tab. 7 in the Appendix.

The heavy correlation between feature norm and OOD score
implies that images yielding small feature norm are not
detected as OOD (see Fig. 6 for a discussion). This also
explains why the simple OOD unit tests in Bitterwolf et al.
(2023), using synthetic images of little variation, e.g. black
or uni-colour images, often fail. These synthetic images
contain little variation in color, which often results in small
activations in the network, and thus small pre-logit features,
see Figure 10 for an analysis.

3.3. Correlation of feature norm and sMaha -score
The strong variations within and across classes we observed
in Figure 3 indicate that the feature norm might impact the
Mahalanobis estimation. To investigate this, we plot the feature norm against the Mahalanobis score sM aha assigned
by the SwinV2-B model for ID and OOD test samples (i.e.
samples that were not used for estimating means and covariance) in Figure 5. We observe a clear correlation: Samples
with large feature norms consistently receive a large OOD
score, and vice versa for samples with small feature norms
- irrespective of whether they belong to the in or out distribution. Ideally, a detector should be able to distinguish ID
from OOD samples irrespective of the norm of the OOD
samples. Since a large fraction of the OOD samples have
a comparably small feature norm, the resulting OOD detection performance is, however, poor. The reason for this
strong correlation is the strongly different average feature

4. Mahalanobis++: Normalize your features
A challenge with Mahalanobis distance-based OOD detection is its sensitivity to feature norms, which can strongly
correlate with Mahalanobis scores. We further find the
feature distribution to strongly contradict the theoretical
Gaussian assumption (with shared covariance), as empirical
feature norms vary much more in practice than expected. To
address this mismatch, we propose a simple but effective fix:
Discarding the feature norm and leveraging only directional
information in the features by ℓ2 -normalization.
5

## Page 6

Mahalanobis++: Improving OOD Detection via Feature Normalization

FPR=58.2%

FPR=31.3%

directions, normalization (green line in Figure 4) shifts the
feature quantiles closer to the diagonal line, confirming that
Mahalanobis++ better satisfies the Gaussian assumption
of Mahalanobis-based detection. We validate this for more
models in Figure 9 in the Appendix.

Feature norm

50
40
30

Variance alignment. In Table 1, we observe lower variance deviation scores for normalized features of the SwinV2
model compared to unnormalized features, indicating that
normalization aligns the class variances in Mahalanobis++.
We illustrate this effect in Figure 2, which visualizes centered training features for three selected classes along a
random direction. Without normalization, class feature variances differ substantially, and the shared covariance matrix
fails to jointly capture their distributions. After normalization, class variances become more consistent, making the
shared variance assumption more appropriate. To further
validate this, we examine which in-distribution test samples
are flagged as OOD at a 95% true-positive rate: unnormalized Mahalanobis rejects samples from 634 classes, while
Mahalanobis++ rejects samples from 728 classes. In an
ideal setting with a perfect covariance fit, one would expect
samples to be drawn uniformly from all 1,000 classes. The
increase from 634 to 728 classes suggests that normalization
reduces bias in the covariance estimation, better aligning
with the shared variance assumption. We substantiate our
observations in Figure 11 and Table 7 in the Appendix for
more models. We find that class variances are more similar
to the global variances after normalization for all models except the ViT-augreg.

20
15000

10000 5000
0
sMaha
ID
OOD

4000 3000 2000 1000 0
sMaha + +
TPR=95%

Figure 5. The feature norm correlates with the Mahalanobis
score for SwinV2-B: Left: The smaller the feature norm, the
smaller the Mahalanobis OOD score sM aha , irrespective of
whether a sample is ID or not. OOD samples with small feature
norms are systematically classified as ID. Right: After normalization, OOD samples with small feature norms can be detected, and
OOD detection is significantly improved.

Method. Instead of the original features ϕ(x), we use ℓ2 normalized ones for computing the Mahalanobis score:
ϕ̂(xi ) = ϕ(xi )/∥ϕ(xi )∥2 ,

(6)

The class-means and covariance matrix of the Mahalanobis
score are estimated using the normalized features, and also
test features are normalized when computing their score.
We denote this simple modification as Mahalanobis++.
We note that ℓ2 -normalization has been used with nonparametric post-hoc OOD detection methods like KNN
(Sun et al., 2022; Park et al., 2023a) or cosine similarity
(Techapanurak et al., 2020). With the Mahalanobis score,
however, ℓ2 -normalization has - to the best of our knowledge - only been investigated for train-time methods like
SSD+ (Sehwag et al., 2021) or CIDER (Ming et al., 2023).
Those methods normalize their features for OOD detection
because they also normalize during training. This is orthogonal to our work: The standard Mahalanobis method for
OOD detection is a post-hoc method, where adjusting the
pretraining scheme is not feasible. We show below that
Mahalanobis++ outperforms KNN and cosine similarity
in all considered cases and, in particular, improves OOD
detection consistently across tasks, architectures, training
methods and OOD datasets as post-hoc method.

Decoupling of feature norm and OOD score. In Figure 5
on the right, we plot the feature norm of ID and OOD samples against their OOD scores obtained via Mahalanobis++.
In contrast to the conventional Mahalanobis score, the correlation between OOD score and feature norm (before normalization) is much weaker. In particular, OOD samples with
small feature norm are now also detected as OOD, which
was not the case for unnormalized Mahalanobis.

5. Experiments
ImageNet. Our main goal is to investigate the effectiveness of Mahalanobis++ across a large pool of architectures,
model sizes and training schemes for ImageNet-scale OOD
detection, as this is where the conventional Mahalanobis
distance showed the most varied results in previous studies
(Bitterwolf et al., 2023; Mueller & Hein, 2024). To this end,
we use 44 publicly available model checkpoints from timm
(Wightman, 2019) and huggingface.co. Following the
OpenOOD setup (Yang et al., 2022), we report results on
Ninco (Bitterwolf et al., 2023), iNaturalist (Van Horn et al.,
2018), SSB-hard (Vaze et al., 2022), OpenImages-O (Krasin
et al., 2017) and Texture (Cimpoi et al., 2014). We report
the false positive rate at a true positive rate of 95% (FPR)

Improved normality. To evaluate how Mahalanobis++
improves the adherence to the assumption of a Gaussian
model with a shared covariance matrix, we compare the
resulting feature distributions via QQ-plots to the unnormalized features. Like for the unnormalized features, we center
ϕ̂center (xi ) = ϕ̂(xi ) − µ̂c[i] , divide by the empirical standard deviation, and plot the resulting quantiles against the
quantiles of a standard normal. We observe that across all
6

## Page 7

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 2. ImageNet. FPR (lower is better) on five OpenOOD datasets. Green indicates that normalization improves over unnormalized
features, bold indicates the best and underlined the second best method. Mahalanobis++ consistently improves over Maha and baselines.
model
ConvNeXtV2-B-In21k
SwinV2-B-In21k
DeiT3-B16-In21k
dataset
NIN SSB TxT OpO iNat Avg NIN SSB TxT OpO iNat Avg NIN SSB TxT OpO iNat Avg
MSP (Hendrycks & Gimpel, 2017) 41.4 60.1 47.4 24.6 8.7 36.5 48.2 63.8 51.7 32.5 21.1 43.4 61.0 73.2 66.0 46.5 32.9 55.9
MaxLogit (Hendrycks et al., 2022) 31.9 51.1 40.7 16.5 4.9 29.0 38.6 52.6 47.7 24.6 13.0 35.3 55.2 67.2 61.9 41.4 34.3 52.0
Energy (Liu et al., 2020)
30.1 47.8 39.5 14.6 4.2 27.2 38.3 47.8 50.9 26.3 13.9 35.5 55.9 65.2 63.2 43.3 45.6 54.7
GEN (Liu et al., 2023)
29.7 52.3 35.9 13.8 3.5 27.1 37.0 57.0 38.7 17.6 8.9 31.8 45.2 61.5 50.1 24.8 13.7 39.0
Energy+React (Sun et al., 2021)
29.5 48.0 38.6 13.9 3.7 26.7 35.1 48.8 44.8 18.5 7.2 30.9 50.9 63.8 55.2 32.1 27.3 45.9
fDBD (Liu & Qin, 2024)
37.0 60.4 37.9 15.4 3.8 30.9 50.5 74.5 41.9 19.1 6.1 38.4 53.5 70.9 50.7 24.8 11.8 42.4
ViM (Wang et al., 2022)
26.9 47.7 28.1 7.9 1.1 22.4 50.4 75.7 35.4 15.4 1.7 35.7 55.3 75.7 48.8 21.1 4.8 41.1
KNN (Sun et al., 2022)
40.9 59.0 32.9 16.5 6.5 31.2 57.2 82.3 35.4 18.5 6.8 40.0 52.6 73.7 43.7 21.1 9.7 40.2
Neco (Ammar et al., 2024)
27.7 45.9 35.0 12.5 2.8 24.8 32.6 48.7 39.8 17.6 5.8 28.9 51.5 64.8 57.4 34.1 24.2 46.4
NNguide (Park et al., 2023a)
31.7 53.5 31.6 12.7 3.3 26.6 42.7 72.5 33.0 12.3 3.5 32.8 46.4 68.4 44.3 19.3 9.3 37.5
Rel.-Mahalanobis (Ren et al., 2021) 28.1 54.6 33.2 11.7 2.0 25.9 48.2 74.0 39.7 19.3 3.5 36.9 47.4 69.8 46.2 20.1 6.0 37.9
Rel.-Mahalanobis++
24.7 51.6 32.1 11.3 2.2 24.4 34.4 62.5 36.8 15.7 3.9 30.6 38.3 61.6 42.5 17.1 4.0 32.7
Mahalanobis (Lee et al., 2018b)
30.3 53.8 30.4 9.4 1.4 25.0 58.2 81.4 41.5 23.2 3.5 41.6 52.5 72.8 47.0 21.4 5.5 39.8
Mahalanobis++
22.4 46.9 26.5 7.8 1.3 21.0 31.3 62.0 28.7 9.7 1.6 26.7 38.8 62.8 42.0 15.6 3.1 32.5

as the OOD detection metric and refer to the appendix for
other metrics, such as AUC, details on the model checkpoints, baseline methods, and extended results. In addition
to Mahalanobis++, we also report relative Mahalanobis++,
i.e the relative Mahalanobis distance with ℓ2 normalization.

Table 3. CIFAR100.
Green indicates that normalization improves the baseline, bold and underlined indicate the best and
second best method. We report FPR averaged over OpenOOD
datasets. Maha++ is the best method. The best FPR is achieved by
Maha++ for ViT-S16-21k highlighted in blue.
Model
SwinV2-S-1k
Deit3-S-21k
ConvN-T-21k
ViT-B32-21k
ViT-S16-21k
RN18
RN34
RNxt29-32
Average

We report results on the five OOD datasets in Table 2 using
three pretrained base-size models: ConvNextV2 (Woo et al.,
2023), SwinV2 (Liu et al., 2022) and DeiT3 (Touvron et al.,
2022). For all models, Mahalanobis++ outperforms the conventional Mahalanobis distance consistently across datasets,
and is the best-performing method on average, and in most
cases also per dataset. Also the relative Mahalanobis++
outperforms its counterpart across models and datasets, but
is slightly worse on average. In Table 4, we show the results
averaged over the five datasets for 44 models with different
training schemes, model sizes and network types. With the
exception of three models (two of which are trained with
augreg), Mahalanobis++ outperforms its counterparts in
all cases. relative Mahalanobis++ outperforms its counterpart in 39/44 cases. In 30/44 cases, the best performing
method is Mahalanobis++ (in 6/44 cases it is relative Mahalanobis++) and the differences to the baseline methods
are often large. Averaged over models, Mahalanobis++
is the best method, followed by relative Mahalanobis++
and outperforming the previously best method ViM by 7
FPR points. We note that Mahalanobis++ is particularly
effective for the best-performing models, as it is the best
method for 4 of the top-5 models.

MSP
47.28
48.92
60.60
48.02
52.17
80.59
76.93
82.31
62.10

Ash
92.66
94.47
92.11
93.98
80.45
78.98
78.27
72.59
85.44

ML
40.96
42.37
57.44
31.28
37.63
79.87
75.33
82.30
55.90

KNN
36.27
36.81
51.16
26.49
31.91
76.61
74.44
73.17
50.86

ViM
34.02
39.99
51.18
27.14
24.90
79.61
77.17
76.40
51.30

MD MD++
40.10 26.01
41.99 31.72
52.48 42.69
26.28 18.94
25.51 18.58
79.48 72.92
76.63 74.51
77.67 67.71
52.52 44.13

EVA and DeiT networks). For several other networks (e.g.,
ConvNexts, Mixer, ResNets, EfficientNets, Swins,...), differences are, however, larger and often in the range of 8-15%
FPR. We note that most of the OOD datasets in OpenOOD
show contamination with ID samples, as reported in Bitterwolf et al. (2023). Therefore, we report results on Ninco,
which has been cleaned from ID data, separately in Table 6,
and find even clearer improvements of Mahalanobis++.
Two of the three models for which Mahalanobis++ does
not bring an improvement are ViTs trained with augreg by
Steiner et al. (2022). Those are the models that showed stateof-the-art performance in Bitterwolf et al. (2023). We extend
our observations from the previous Sections regarding these
models in Appendix D, where we show that their feature
norms are already well-behaved; therefore, ℓ2 normalization
does not improve the normality assumptions.

We further note that NNguide (Park et al., 2023a) and
KNN (Sun et al., 2022), both of which operate in a normalized feature space, are consistently outperformed by
Mahalanobis++. The most competitive baseline method
that is not based on the Mahalanobis distance is ViM (Wang
et al., 2022), which for certain models shows similar or
slightly better performance than Mahalanobis++ (e.g. for

Bitterwolf et al. (2023) reported that Mahalanobis-based
detectors sometimes fail to detect supposedly easy-to-detect
noise distributions (called ”unit tests”). In Section 3, we
connected this to the small feature norm those samples ob7

## Page 8

Mahalanobis++: Improving OOD Detection via Feature Normalization
Table 4. FPR on OpenOOD datasets, Green indicates that a normalized method is better than its unnormalized counterpart, bold
indicates the best and underlined the second best method. Maha++ improves over Maha on average by 7.6% in FPR over all models.
Similarly, rMaha++ is, on average, 2.9% better in FPR than rMaha. In total, Maha++ improves the SOTA compared to the strongest
competitor rMaha among all OOD methods by 6.9%, which is a significant improvement. The lowest FPR is achieved by Maha++ for the
EVA02-L14-M38m-In21k highlighted in blue.
Model
Val Acc
ConvNeXt-B-In21k
86.3
ConvNeXt-B
84.4
ConvNeXtV2-T-In21k
85.1
ConvNeXtV2-B-In21k
87.6
ConvNeXtV2-L-In21k
88.2
ConvNeXtV2-T
83.5
ConvNeXtV2-B
85.5
ConvNeXtV2-L
86.1
DeiT3-S16-In21k
84.8
DeiT3-B16-In21k
86.7
DeiT3-L16-In21k
87.7
DeiT3-S16
83.4
DeiT3-B16
85.1
DeiT3-L16
85.8
EVA02-B14-In21k
88.7
EVA02-L14-M38m-In21k
90.1
EVA02-T14
80.6
EVA02-S14
85.7
EffNetV2-S
83.9
EffNetV2-L
85.7
EffNetV2-M
85.2
Mixer-B16-In21k
76.6
SwinV2-B-In21k
87.1
SwinV2-L-In21k
87.5
SwinV2-S
84.2
SwinV2-B
84.6
ResNet101
81.9
ResNet152
82.3
ResNet50
80.9
ResNet50-supcon
78.7
ViT-T16-In21k-augreg
75.5
ViT-S16-In21k-augreg
81.4
ViT-B16-In21k-augreg2
85.1
ViT-B16-In21k-augreg
84.5
ViT-B16-In21k-orig
81.8
ViT-B16-In21k-miil
84.3
ViT-L16-In21k-augreg
85.8
ViT-L16-In21k-orig
81.5
ViT-S16-augreg
78.8
ViT-B16-augreg
79.2
ViT-B16-CLIP-L2b-In12k
86.2
ViT-L14-CLIP-L2b-In12k
88.2
ViT-H14-CLIP-L2b-In12k
88.6
ViT-so400M-SigLip
89.4
Average
84.4

MSP
41.7
61.4
44.7
36.5
35.0
60.5
58.8
58.6
60.5
55.9
55.0
56.9
59.7
60.3
32.4
27.0
64.8
52.2
59.3
57.1
57.0
71.5
43.4
40.4
61.2
62.4
67.7
66.4
72.0
54.0
70.7
57.0
55.3
46.5
44.6
48.0
40.2
40.8
64.8
64.3
42.2
31.5
32.0
45.5
52.7

E
40.1
90.9
37.3
27.2
27.0
66.1
70.8
68.0
53.3
54.7
45.5
54.0
82.3
80.5
26.8
22.6
66.2
53.4
71.0
74.2
69.3
83.0
35.5
35.9
68.1
66.2
82.8
82.1
95.9
47.3
55.3
38.9
45.9
33.7
30.7
35.0
29.4
29.3
59.0
59.6
37.7
25.2
26.5
47.1
53.0

E+R
36.0
86.9
37.1
26.7
26.5
58.6
64.1
60.1
50.4
45.9
38.3
58.1
88.4
89.3
26.2
22.4
66.8
53.1
58.7
57.4
56.7
83.5
30.9
31.2
62.1
58.2
99.6
99.5
99.4
42.1
48.4
42.5
41.1
36.0
30.9
34.6
25.0
29.2
60.6
56.2
35.5
24.6
26.1
39.4
51.0

ML
37.3
70.2
38.6
29.0
28.5
58.9
59.5
58.3
54.4
52.0
46.9
52.2
64.4
64.0
28.8
24.3
63.1
49.5
61.1
58.8
57.3
75.0
35.3
34.5
60.9
60.5
70.7
70.0
75.8
48.4
58.3
41.7
47.5
34.6
33.1
38.8
30.0
31.1
60.0
60.1
37.2
26.5
27.7
41.8
49.0

ViM
29.5
52.8
27.0
22.4
28.7
49.9
46.8
48.4
47.6
41.1
36.4
43.3
44.7
46.1
22.0
18.0
49.3
34.8
52.2
48.9
54.7
71.8
35.7
39.0
51.1
49.9
50.5
49.7
53.1
72.0
51.1
33.4
53.9
26.9
29.0
37.8
23.6
30.4
68.1
63.4
35.5
21.5
22.3
30.6
41.9

AshS
88.5
99.5
96.7
95.3
95.6
99.2
99.6
99.1
99.2
99.2
98.1
85.6
99.2
78.4
87.9
91.0
98.4
99.1
99.4
99.2
99.5
95.8
77.0
85.1
99.9
99.1
80.2
80.0
80.3
40.6
94.9
76.7
98.6
94.9
62.6
96.9
94.5
49.3
96.9
90.2
99.5
97.6
97.8
93.5
90.7

KNN
37.2
58.7
41.6
31.2
30.8
72.1
53.4
48.9
49.8
40.2
35.0
69.8
66.1
54.0
29.6
25.8
60.8
44.1
45.6
48.9
51.3
77.8
40.0
38.9
58.6
55.0
53.6
52.0
67.8
47.0
76.2
55.6
47.5
54.3
38.6
45.0
50.6
34.0
71.5
65.5
35.6
29.9
31.2
28.7
48.9

tain. In Table 5 we report the number of ”failed” unit tests (a
unit test counts as failed when a detector shows FPR values
above 10%) and observe that normalization, in particular
Mahalanobis++ remedies this effectively. For results on all
models, we refer to Table 17 Appendix.

ConvNeXtV2

SwinV2

ViT-CLIP

Maha
Maha++

5/17
0/17

10/17
0/17

14/17
0/17

NEC
31.4
66.5
33.2
24.8
24.1
54.1
55.9
55.9
51.6
46.4
38.5
52.2
63.5
64.3
25.0
21.8
55.4
42.9
59.3
56.0
54.9
75.5
28.9
29.0
56.0
55.4
70.6
69.1
76.6
47.8
52.8
38.1
43.7
32.4
30.5
33.9
28.0
29.4
60.2
59.9
34.0
26.3
27.5
39.6
46.0

GMN
54.2
73.9
47.2
38.9
32.9
73.9
71.2
63.8
52.0
46.2
36.8
64.1
63.5
56.9
36.0
39.9
54.1
43.0
77.9
58.1
66.8
61.2
57.7
48.8
61.2
56.1
82.3
77.2
89.5
78.8
58.2
44.3
60.1
38.6
48.8
57.1
41.6
47.9
61.8
60.3
41.4
36.3
53.0
64.3
56.3

GEN
32.6
60.1
36.5
27.1
26.4
53.6
48.6
46.3
47.8
39.0
34.6
46.6
46.0
45.1
24.3
20.3
57.9
41.7
49.7
44.7
45.2
67.7
31.8
31.5
52.8
49.9
62.5
60.3
65.4
53.5
64.7
46.5
42.9
36.5
38.4
38.3
30.7
35.2
61.8
61.5
33.4
24.3
24.6
28.3
43.6

fDBD
37.9
60.3
42.3
30.9
31.6
61.8
53.7
48.5
54.4
42.4
38.2
54.2
54.8
52.4
28.6
23.9
66.4
48.9
54.0
49.2
52.6
71.8
38.4
37.5
60.7
56.9
71.3
69.3
74.8
48.0
58.5
44.0
51.4
36.4
35.7
44.6
30.4
33.0
64.8
63.5
38.0
27.3
28.9
29.9
47.8

Maha
33.6
54.2
32.5
25.0
27.8
55.4
46.3
41.7
50.2
39.8
34.9
52.4
51.5
45.3
25.5
19.7
51.0
36.6
47.3
41.3
46.0
63.3
41.6
41.8
52.4
47.9
45.9
44.4
49.5
95.5
55.5
36.7
54.2
25.7
30.9
47.1
21.0
30.9
49.3
49.6
43.2
28.2
27.1
28.8
42.5

Maha++
24.3
44.6
28.6
21.0
18.8
44.4
37.5
36.2
42.4
32.5
30.1
46.5
46.7
39.7
21.0
17.7
48.1
35.4
40.2
34.6
37.1
52.5
26.7
24.7
38.9
40.1
43.5
38.3
52.0
44.5
48.0
31.7
38.2
28.3
27.5
30.4
23.9
26.8
49.2
48.0
28.1
22.4
22.0
24.5
34.9

rMaha
31.7
50.0
34.6
25.9
25.8
48.9
43.0
39.0
48.9
37.9
33.4
49.1
48.3
42.7
26.2
21.1
52.6
38.1
43.6
38.0
41.1
60.0
36.9
36.2
48.7
45.2
55.6
51.8
62.5
90.2
59.2
43.0
47.0
30.8
35.4
43.6
25.2
33.6
48.4
47.6
38.0
27.1
26.8
27.3
41.8

rMaha++
29.5
45.4
33.4
24.4
23.0
44.6
39.1
38.0
43.6
32.7
29.9
44.9
45.0
38.6
23.8
20.4
50.7
36.8
40.4
34.8
36.8
52.9
30.6
28.7
39.3
39.7
66.8
64.7
70.4
63.7
57.7
40.6
39.1
31.5
33.9
36.7
25.8
32.6
48.2
46.7
32.4
25.4
25.1
25.5
38.9

ImageNet (Le & Yang, 2015), Mnist (LeCun et al., 1998),
SVHN (Netzer et al., 2011), Texture (Cimpoi et al., 2014),
Places (Zhou et al., 2017) and Cifar10 as OOD datasets for
a range of architectures and training schemes.
We report results averaged across the OOD datasets in Table 3 for the most competitive methods and standard baselines (full results in Appendix E). We observe that Mahalanobis++ consistently outperforms the conventional Mahalanobis distance, but the differences are smaller compared
to the ImageNet setup. We hypothesize that this is because
the problems of the Mahalanobis distance are less drastic at
a smaller scale, and therefore the conventional Mahalanobis
distance is already fairly effective for OOD detection. ViM
and KNN are the most competitive baseline methods, but
Mahalanobis++ remains the most consistent and effective
method across models.

Table 5. Normalization improves robustness against noise distributions. We report the number of failed unit tests (noise distributions with FPR values ≥ 10%) from Bitterwolf et al. (2023).
Normalization remedies the brittleness of Mahalanobis-based detectors. Full Table in Appendix E.
model

NNG
31.8
51.2
36.4
26.6
25.9
62.8
47.9
44.7
47.5
37.5
32.1
48.2
71.3
72.5
25.8
22.8
57.1
40.3
45.2
47.1
48.6
83.9
32.8
32.9
52.9
51.1
51.4
46.8
64.1
41.9
71.0
48.9
42.2
45.6
35.4
38.5
41.2
31.6
68.9
64.1
31.6
22.3
23.4
26.1
44.8

CIFAR We investigate Mahalanobis++ on CIFAR100
(Krizhevsky, 2009), following the OpenOOD setup with tiny
8

## Page 9

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 6. FPR on NINCO, Green indicates that normalized method is better than its unnormalized counterpart, bold indicates the best
method, and underlined indicates second best method. Maha++ improves over Maha on average by 10.9% in FPR over all models.
Similarly, rMaha++ is 6.0% better in FPR than rMaha. In total, Maha++ improves the SOTA compared to the strongest competitor rMaha
among all OOD models by 6.3% which is significant. The lowest FPR is achieved by Maha++ for the ConvNeXtV2-L-In21k highlighted
in blue.
Model
Val Acc
ConvNeXt-B-In21k
86.3
ConvNeXt-B
84.4
ConvNeXtV2-T-In21k
85.1
ConvNeXtV2-B-In21k
87.6
ConvNeXtV2-L-In21k
88.2
ConvNeXtV2-T
83.5
ConvNeXtV2-B
85.5
ConvNeXtV2-L
86.1
DeiT3-S16-In21k
84.8
DeiT3-B16-In21k
86.7
DeiT3-L16-In21k
87.7
DeiT3-S16
83.4
DeiT3-B16
85.1
DeiT3-L16
85.8
EVA02-B14-In21k
88.7
EVA02-L14-M38m-In21k
90.1
EVA02-T14
80.6
EVA02-S14
85.7
EffNetV2-S
83.9
EffNetV2-L
85.7
EffNetV2-M
85.2
Mixer-B16-In21k
76.6
SwinV2-B-In21k
87.1
SwinV2-L-In21k
87.5
SwinV2-S
84.2
SwinV2-B
84.6
ResNet101
81.9
ResNet152
82.3
ResNet50
80.9
ResNet50-supcon
78.7
ViT-T16-In21k-augreg
75.5
ViT-S16-In21k-augreg
81.4
ViT-B16-In21k-augreg2
85.1
ViT-B16-In21k-augreg
84.5
ViT-B16-In21k-orig
81.8
ViT-B16-In21k-miil
84.3
ViT-L16-In21k-augreg
85.8
ViT-L16-In21k-orig
81.5
ViT-S16-augreg
78.8
ViT-B16-augreg
79.2
ViT-B16-CLIP-L2b-In12k
86.2
ViT-L14-CLIP-L2b-In12k
88.2
ViT-H14-CLIP-L2b-In12k
88.6
ViT-so400M-SigLip
89.4
Average
84.4

MSP
46.2
64.1
51.6
41.4
38.7
66.1
62.9
63.8
68.7
61.0
59.7
64.3
66.7
67.8
35.8
29.0
72.7
61.2
67.7
63.7
63.4
77.4
48.2
45.9
67.6
69.5
73.4
71.2
76.0
60.6
79.0
67.0
62.1
56.8
52.2
57.2
47.0
46.2
72.8
72.2
49.7
35.5
36.4
50.3
58.9

E
E+R ML ViM
43.0 40.0 40.5 41.2
89.4 86.2 71.4 64.7
42.4 42.4 44.1 34.5
30.1 29.5 31.9 26.9
30.7 29.8 31.5 37.2
73.3 68.4 65.7 64.4
73.9 69.8 63.5 61.1
72.3 66.8 63.6 62.4
61.8 59.8 62.6 60.6
55.9 50.9 55.2 55.3
46.2 41.8 48.9 45.7
63.0 63.8 60.6 54.3
87.8 89.9 72.6 59.7
82.3 86.6 70.5 57.9
28.2 27.4 30.9 28.7
24.3 24.1 25.7 21.4
74.5 75.2 72.1 67.7
61.4 61.5 57.8 51.6
77.5 73.3 69.5 74.0
77.2 68.8 64.3 69.4
75.1 69.1 63.8 72.3
83.4 83.5 79.5 78.0
38.3 35.1 38.6 50.4
38.7 35.4 38.6 55.3
71.7 70.1 66.7 66.8
72.6 69.3 67.4 66.6
85.2 100.0 76.1 75.8
83.2 100.0 74.4 74.6
94.7 99.9 78.6 79.6
57.0 56.1 56.8 84.6
72.9 69.6 74.0 66.4
53.3 55.4 54.7 47.4
52.1 49.4 54.6 71.0
45.2 48.9 45.4 38.1
39.2 39.0 41.0 35.6
46.4 46.5 49.3 46.1
39.0 27.3 37.7 31.7
37.3 37.1 37.7 42.2
72.8 73.6 72.5 80.7
71.7 69.6 71.1 73.5
44.7 42.6 44.0 49.6
28.8 28.1 29.9 24.5
31.1 30.8 31.6 24.9
47.4 42.1 44.2 40.2
58.6 57.6 55.2 54.9

AshS
92.7
99.6
97.5
95.8
96.0
99.2
99.4
99.5
99.7
99.5
98.4
84.1
99.1
81.1
92.7
94.8
98.8
98.9
99.7
98.9
99.6
94.8
86.0
89.9
99.8
97.8
89.9
88.1
89.9
59.1
90.1
85.0
98.7
94.1
71.4
98.0
95.2
58.5
97.0
90.9
99.9
97.9
97.0
95.5
92.9

KNN
51.6
70.1
54.1
40.9
38.7
82.3
67.3
62.4
62.9
52.6
43.8
75.6
74.5
67.2
37.6
30.3
74.5
60.0
60.9
62.5
63.1
85.8
57.2
55.1
73.1
69.4
74.9
72.0
83.7
65.8
81.7
70.9
64.0
67.7
52.7
59.6
68.6
45.8
82.1
77.6
49.4
39.5
41.7
36.3
61.5

6. Conclusion

NNG
41.2
62.2
45.0
31.7
29.9
73.9
60.3
56.9
59.3
46.4
37.8
57.8
80.1
77.9
30.0
26.1
71.0
54.0
59.6
60.1
60.6
83.7
42.7
41.7
66.8
65.2
66.4
61.6
75.0
58.4
81.9
64.0
57.0
59.0
47.6
51.7
58.9
40.7
80.0
75.9
42.3
25.4
27.4
30.0
55.1

NEC
35.2
68.2
38.7
27.7
27.0
61.7
60.7
61.9
60.1
51.5
42.3
60.6
71.9
70.8
27.3
22.9
68.4
53.2
69.4
63.5
63.3
79.8
32.6
32.3
62.7
64.2
77.2
75.0
80.0
56.9
71.1
51.9
51.3
43.1
38.3
43.6
35.4
36.4
72.7
70.9
40.9
29.7
31.5
42.6
52.9

GMN
53.8
68.3
52.5
40.6
43.4
72.3
69.2
71.2
58.5
52.2
43.7
66.4
66.7
62.9
39.0
39.5
65.6
51.9
79.9
64.4
67.5
66.7
56.3
63.1
61.0
60.7
83.5
79.5
89.1
80.3
69.4
58.3
65.2
50.1
56.9
59.4
52.7
55.8
71.0
65.5
50.5
41.5
53.4
65.1
61.0

GEN
38.0
65.9
44.2
29.7
29.0
64.1
57.1
55.6
58.7
45.2
38.2
57.4
57.3
58.4
25.8
20.3
70.5
53.1
62.9
56.3
56.3
75.9
37.0
36.5
63.8
62.0
72.5
69.9
75.0
60.0
78.4
61.4
53.4
48.2
48.2
50.0
37.6
42.1
72.8
72.0
41.4
26.8
27.4
30.9
52.0

fDBD
48.3
68.6
51.4
37.0
36.6
71.6
65.1
59.7
65.8
53.5
46.6
65.0
67.1
64.4
32.3
26.0
73.5
60.0
67.9
62.4
64.5
80.3
50.5
50.5
73.4
70.5
84.5
82.4
85.7
63.9
72.3
57.6
64.4
49.8
44.5
58.1
40.3
40.1
75.4
73.8
48.4
31.4
33.6
36.1
58.1

Maha
48.7
64.6
40.9
30.3
34.6
66.1
58.9
53.8
62.4
52.5
42.0
61.1
63.7
57.0
31.7
22.2
65.9
49.3
67.5
58.4
61.7
73.4
58.2
57.8
68.0
63.3
66.8
64.9
69.9
98.3
61.6
44.8
69.8
31.3
35.6
56.2
24.2
39.4
63.2
62.9
57.2
35.3
33.5
36.4
53.8

Maha++
28.8
50.5
32.8
22.4
18.4
52.3
44.7
43.0
50.8
38.8
33.9
53.5
57.2
50.4
23.8
18.6
64.0
48.0
59.9
47.8
50.0
65.4
31.3
28.3
49.8
52.2
50.4
46.5
61.0
59.6
63.2
44.6
45.9
35.7
31.6
35.4
28.9
32.4
63.1
61.3
35.8
25.4
23.7
27.4
42.9

rMaha
40.2
59.5
40.0
28.1
27.7
58.6
52.4
47.1
59.8
47.4
38.1
56.3
58.9
52.0
30.3
22.1
65.9
49.1
59.2
50.8
52.2
70.3
48.2
47.6
63.7
59.1
55.8
52.7
58.2
90.7
67.6
51.1
58.5
35.2
38.3
48.6
26.5
37.6
59.2
58.4
49.0
30.3
29.5
31.3
49.2

rMaha++
32.5
49.7
36.8
24.7
21.4
49.7
44.1
42.1
50.9
38.3
32.8
50.5
53.2
46.6
25.9
20.1
64.4
47.8
52.1
44.3
45.3
63.1
34.4
32.2
48.5
50.2
53.5
52.2
56.9
61.8
68.0
50.5
44.5
37.1
36.5
40.2
28.1
36.1
58.9
57.4
38.9
27.2
26.1
26.1
43.2

more aligned with a normal distribution, less heavy-tailed,
and the class variances are more similar, leading to improved
OOD detection results across a wide range of models. Mahalanobis++ outperforms the conventional Mahalanobis distance in 41/44 cases, rendering it clearly the most effective
method across models. It outperforms the previously best
baseline ViM by 7 FPR points on average on the OpenOOD
datasets, and is the best method for 4 of the 5 top models.

We showed that the frequently occurring failure cases of
the Mahalanobis distance as an OOD detection method are
related to violations of the method’s basic assumptions. We
showed that the feature norms vary much stronger than expected under a Gaussian model, that the feature distributions
are strongly heavy-tailed and that feature norms correlate
with the Mahalanobis score - irrespective of whether a sample is ID or OOD. These insights explain why certain models
- despite impressive ID classification performance - showed
strongly degraded OOD detection results with the Mahalanobis score in previous studies (Bitterwolf et al., 2023).
We introduced Mahalanobis++, a simple remedy consisting
of ℓ2 normalization that effectively mitigates those problems. In particular, the resulting feature distributions are

Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none of which we feel must be
specifically highlighted here.

9

## Page 10

Mahalanobis++: Improving OOD Detection via Feature Normalization

Acknowledgements

Haas, J., Yolland, W., and Rabus, B. T. Linking neural collapse and l2 normalization with improved out-ofdistribution detection in deep neural networks. Transactions on Machine Learning Research, 2023. ISSN 28358856. URL https://openreview.net/forum?
id=fjkN5Ur2d6.

We thank Yannic Neuhaus for insightful discussions about
distances on unit spheres. Further, we acknowledge support from the DFG (EXC number 2064/1, Project number
390727645) and the Carl Zeiss Foundation in the project
”Certification and Foundations of Safe Machine Learning
Systems in Healthcare”. Finally, we acknowledge support
from the German Federal Ministry of Education and Research (BMBF) through the Tübingen AI Center (FKZ:
01IS18039A) and the European Laboratory for Learning
and Intelligent Systems (ELLIS).

Haas, J., Yolland, W., and Rabus, B. T. Exploring simple, high quality out-of-distribution detection with l2
normalization. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://
openreview.net/forum?id=daX2UkLMS0.
Hein, M., Andriushchenko, M., and Bitterwolf, J. Why
ReLU networks yield high-confidence predictions far
away from the training data and how to mitigate the problem. In CVPR, 2019.

References
Ammar, M. B., Belkhir, N., Popescu, S., Manzanera, A., and Franchi, G. NECO: NEural collapse
based out-of-distribution detection. In The Twelfth International Conference on Learning Representations,
2024. URL https://openreview.net/forum?
id=9ROuKblmi7.
Anthony, H. and Kamnitsas, K. On the use of mahalanobis
distance for out-of-distribution detection with neural networks for medical imaging. In Uncertainty for Safe
Utilization of Machine Learning in Medical Imaging,
pp. 136–146. Springer Nature Switzerland, 2023. doi:
10.1007/978-3-031-44336-7 14. URL https://doi.
org/10.1007%2F978-3-031-44336-7_14.
Bitterwolf, J., Mueller, M., and Hein, M. In or out? fixing imagenet out-of-distribution detection evaluation. In
ICML, 2023. URL https://proceedings.mlr.
press/v202/bitterwolf23a.html.
Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and
Vedaldi, A. Describing textures in the wild. In CVPR,
2014.
Djurisic, A., Bozanic, N., Ashok, A., and Liu, R. Extremely
simple activation shaping for out-of-distribution detection.
In The Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.
net/forum?id=ndYXTEL6cZz.

Hendrycks, D. and Gimpel, K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In ICLR, 2017. URL https://openreview.
net/forum?id=Hkg4TI9xl.
Hendrycks, D., Basart, S., Mazeika, M., Zou, A., Kwon,
J., Mostajabi, M., Steinhardt, J., and Song, D. Scaling
out-of-distribution detection for real-world settings. In
ICML, 2022.
Koner, R., Sinhamahapatra, P., Roscher, K., Günnemann, S.,
and Tresp, V. Oodformer: Out-of-distribution detection
transformer, 07 2021.
Krasin, I., Duerig, T., Alldrin, N., Ferrari, V., Abu-El-Haija,
S., Kuznetsova, A., Rom, H., Uijlings, J., Popov, S., Kamali, S., Malloci, M., Pont-Tuset, J., Veit, A., Belongie,
S., Gomes, V., Gupta, A., Sun, C., Chechik, G., Cai, D.,
Feng, Z., Narayanan, D., and Murphy, K. Openimages:
A public dataset for large-scale multi-label and multiclass image classification. Dataset available from storage.googleapis.com/openimages/web/index.html, 2017.
Krizhevsky, A. Learning multiple layers of features from
tiny images. Technical report, 2009.
Le, Y. and Yang, X. S. Tiny imagenet visual recognition challenge.
2015.
URL https://api.
semanticscholar.org/CorpusID:16664790.

Galil, I., Dabbah, M., and El-Yaniv, R. A framework for benchmarking class-out-of-distribution detection and its application to imagenet. In The Eleventh
International Conference on Learning Representations,
2023. URL https://openreview.net/forum?
id=Iuubb9W6Jtk.
Gia, T. L. and Ahn, J.
Understanding normalization in contrastive representation learning and outof-distribution detection.
ArXiv, abs/2312.15288,
2023. URL https://api.semanticscholar.
org/CorpusID:266550859.
10

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. doi:
10.1109/5.726791.
Lee, K., Lee, H., Lee, K., and Shin, J. Training confidencecalibrated classifiers for detecting out-of-distribution samples. In ICLR, 2018a.

## Page 11

Mahalanobis++: Improving OOD Detection via Feature Normalization

Lee, K., Lee, K., Lee, H., and Shin, J. A simple unified
framework for detecting out-of-distribution samples and
adversarial attacks. In NeurIPS, 2018b.

Park, J., Long Chai, J. C., Yoon, J., and Jin Teoh, A. B.
Understanding the Feature Norm for Out-of-Distribution
Detection . In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023b. URL
https://doi.ieeecomputersociety.org/
10.1109/ICCV51070.2023.00150.

Liu, L. and Qin, Y. Fast decision boundary based out-ofdistribution detector. ICML, 2024.
Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., and Song, L.
Sphereface: Deep hypersphere embedding for face recognition, 2018. URL https://arxiv.org/abs/
1704.08063.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark,
J., Krueger, G., and Sutskever, I. Learning transferable
visual models from natural language supervision. In
ICML, 2021.

Liu, W., Wang, X., Owens, J., and Li, Y. Energy-based outof-distribution detection. Advances in Neural Information
Processing Systems, 2020.

Regmi, S., Panthi, B., Dotel, S., Gyawali, P. K., Stoyanov,
D., and Bhattarai, B. T2fnorm: Train-time feature normalization for ood detection in image classification. In
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) Workshops, pp.
153–162, June 2024.

Liu, X., Lochman, Y., and Christopher, Z. Gen: Pushing the
limits of softmax-based out-of-distribution detection. In
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023.

Ren, J., Fort, S., Liu, J., Roy, A. G., Padhy, S., and Lakshminarayanan, B. A simple fix to mahalanobis distance for improving near-ood detection, 2021. URL
https://arxiv.org/abs/2106.09022.

Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning,
J., Cao, Y., Zhang, Z., Dong, L., Wei, F., and Guo, B.
Swin transformer v2: Scaling up capacity and resolution.
In CVPR, 2022. URL https://arxiv.org/abs/
2111.09883.

Sablayrolles, A., Douze, M., Schmid, C., and Jégou,
H. Spreading vectors for similarity search. arXiv:
Machine Learning, 2018.
URL https://api.
semanticscholar.org/CorpusID:62841605.

Ming, Y., Cai, Z., Gu, J., Sun, Y., Li, W., and Li, Y. Delving
into out-of-distribution detection with vision-language
representations. In NeurIPS, 2022. URL https://
openreview.net/forum?id=KnCS9390Va.
Ming, Y., Sun, Y., Dia, O., and Li, Y. How to exploit hyperspherical embeddings for out-of-distribution detection?
In The Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.
net/forum?id=aEFaE0W5pAd.

Sehwag, V., Chiang, M., and Mittal, P. Ssd: A unified
framework for self-supervised outlier detection. In International Conference on Learning Representations,
2021. URL https://openreview.net/forum?
id=v5gjXpmR8J.
Steiner, A. P., Kolesnikov, A., Zhai, X., Wightman, R.,
Uszkoreit, J., and Beyer, L. How to train your vit?
data, augmentation, and regularization in vision transformers. TMLR, 2022. URL https://openreview.
net/forum?id=4nPswr1KcP.

Mueller, M. and Hein, M. How to train your vit for ood
detection, 2024. URL https://arxiv.org/abs/
2405.17447.
Mukhoti, J., Kirsch, A., van Amersfoort, J., Torr, P. H., and
Gal, Y. Deep deterministic uncertainty: A new simple
baseline. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pp.
24384–24394, June 2023.

Sun, Y., Guo, C., and Li, Y. React: Out-of-distribution
detection with rectified activations. NeurIPS, 2021.
Sun, Y., Ming, Y., Zhu, X., and Li, Y. Out-of-distribution
detection with deep nearest neighbors. ICML, 2022.

Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B.,
and Ng, A. Y. Reading digits in natural images with
unsupervised feature learning. NIPS Workshop on Deep
Learning and Unsupervised Feature Learning, 2011.

Tack, J., Mo, S., Jeong, J., and Shin, J. Csi: Novelty detection via contrastive learning on distributionally shifted
instances. In NeurIPS, 2020.

Park, J., Jung, Y. G., and Teoh, A. B. J. Nearest neighbor
guidance for out-of-distribution detection. In Proceedings
of the IEEE/CVF International Conference on Computer
Vision, pp. 1686–1695, 2023a.

Techapanurak, E., Suganuma, M., and Okatani, T.
Hyperparameter-free out-of-distribution detection using
cosine similarity. In Proceedings of the Asian Conference
on Computer Vision, 2020.
11

## Page 12

Mahalanobis++: Improving OOD Detection via Feature Normalization

Touvron, H., Cord, M., and Jegou, H. Deit iii: Revenge of
the vit. ECCV, 2022.

Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. Places: A 10 million image database for scene
recognition. IEEE transactions on pattern analysis and
machine intelligence, 40(6):1452–1464, 2017.

Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C.,
Shepard, A., Adam, H., Perona, P., and Belongie, S. The
inaturalist species classification and detection dataset. In
CVPR, 2018.

Zhou, C., Po, L. M., and Ou, W. Angular deep supervised
vector quantization for image retrieval. IEEE Transactions on Neural Networks and Learning Systems, 33(4):
1638–1649, 2022. doi: 10.1109/TNNLS.2020.3043103.

Vaze, S., Han, K., Vedaldi, A., and Zisserman, A. Open-set
recognition: a good closed-set classifier is all you need?
In International Conference on Learning Representations,
2022.
Wang, H., Li, Z., Feng, L., and Zhang, W. Vim: Out-ofdistribution with virtual-logit matching. In CVPR, 2022.
Wei, H., Xie, R., Cheng, H., Feng, L., An, B., and Li,
Y. Mitigating neural network overconfidence with logit
normalization. 2022.
Wightman, R. Pytorch image models. https://github.
com/rwightman/pytorch-image-models,
2019.
Wilk, M. B. and Gnanadesikan, R. Probability plotting methods for the analysis for the analysis of data.
Biometrika, 55(1):1–17, 03 1968. ISSN 0006-3444. doi:
10.1093/biomet/55.1.1. URL https://doi.org/10.
1093/biomet/55.1.1.
Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon,
I. S., and Xie, S. Convnext v2: Co-designing and scaling convnets with masked autoencoders. arXiv preprint
arXiv:2301.00808, 2023.
Yang, J., Wang, P., Zou, D., Zhou, Z., Ding, K., Peng, W.,
Wang, H., Chen, G., Li, B., Sun, Y., et al. Openood:
Benchmarking generalized out-of-distribution detection.
arXiv preprint arXiv:2210.07242, 2022.
Yaras, C., Wang, P., Zhu, Z., Balzano, L., and Qu, Q. Neural
collapse with normalized features: A geometric analysis
over the riemannian manifold. In Oh, A. H., Agarwal,
A., Belgrave, D., and Cho, K. (eds.), Advances in Neural
Information Processing Systems, 2022. URL https:
//openreview.net/forum?id=Zvh6lF5b26N.
Yu, C., Zhu, X., Lei, Z., and Li, S. Z. Out-of-distribution
detection for reliable face recognition. IEEE Signal Processing Letters, 27:710–714, 2020. doi: 10.1109/LSP.
2020.2988140.
Zheng, J., Li, J., Liu, C., Wang, J., Li, J., and Liu,
H. Anomaly detection for high-dimensional space
using deep hypersphere fused with probability approach. Complex & Intelligent Systems, 8(5):4205–
4220, Oct 2022. ISSN 2198-6053. doi: 10.1007/
s40747-022-00695-9. URL https://doi.org/10.
1007/s40747-022-00695-9.
12

## Page 13

Mahalanobis++: Improving OOD Detection via Feature Normalization

A. Overview
The Appendix is structured as follows:
In Section B we provide the proof for Lemma 3.1. In Section D we provide extended analysis on feature norm and
normalization. In particular,
• we show the feature norm distribution for more models in Figure 8
• we provide QQ-plots for more models in Figure 9
• We report the feature norm distribution for ID and OOD data in Figure 10, showing that OOD features can be larger
than ID features for off-the-shelf pretrained models
• we highlight that the class variances become more similar to the global variance after normalization in Figure 11
• we plot the correlation between feature norm and OOD-score in Figure 7
In Section E we report extended results. In particular,
• we show additional ImageNet numbers (AUC for NINCO in Table 11, OpenOOD near and far in Table 8 and Table 9,
OpenOOD averaged AUC in Table 10)
• we compare cosine-based methods on ImageNet explicitly in Table 12
• we show robustness to noise distributions (unit tests) in Table 17
• we show additional CIFAR numbers (Cifar10 AUC in Table 13 and FPR in Table 14, Cifar100 AUC in Table 15 and
FPR in Table 16
• we compare Mahalanobis++ to SSD+ in Table 18 to highlight the benefits of post-hoc OOD detection methods
In Section F we report details on the model checkpoints used throughout the experiments (ImageNet models in Table 19 and
Cifar models in Table 20). In Section G we provide details on the OOD detection methods evaluated in the main paper.

B. Proof of Lemma 3.1
Proof. Let Σ = U ΛU T be the eigendecomposition of the covariance matrix with U being an orthogonal matrix containing
the eigenvectors of Σ and Λ the diagonal matrix containing the eigenvalues of σ. Let X be a random variable with distribution
N (µ, Σ) (in the main paper, we denoted the features as Φ(X), here we write them as X for notational simplicity). Then it
2
2
holds Z = U T X has distribution N (U T µ, Λ) and since U T is an orthogonal matrix: ∥X∥2 = ∥Z∥2 . We have
2

2

E[∥X∥2 ] = E[∥Z∥2 ] =

d
X

E[Zi2 ] =

i=1

d
X

Var(Zi ) + E[Zi ]2 =

i=1

d
X

2

2

λi + U T µ 2 = tr(Σ) + ∥µ∥2

i=1

We note that
2

4

2

4

2

Var(∥Z∥2 ) = E[∥Z∥2 ] − E[∥Z∥2 ]2 = E[∥Z∥2 ] − (tr(Σ) + ∥µ∥2 )2

(7)

4

and it remains to compute E[∥Z∥2 ]. We note that
4
E[∥Z∥2 ] =

d
X
i=1

E[Zi4 ] +

d
X

E[Zi2 ]E[Zj2 ] =

i̸=j

d
X
(3λ2i + 6µ2i λi + µ4i ) +

d
X

i=1

i=1

!2
(λi + µ2i )

where we have used the following calculations:
0 = E[(Zi − µi )3 ] = E[Zi3 ] − 3µi λi − µ3i
3λ2i = E[(Zi − µi )4 ] = E[Zi4 ] − 4µi E[Zi3 ] + 6µ2i E[Zi2 ] − 3µ4i
13

−

d
X
(λi + µ2i )2 ,
i=1

## Page 14

Mahalanobis++: Improving OOD Detection via Feature Normalization

and thus
E[Zi3 ] = 3µi λi + µ3i
E[Zi4 ] = 3λ2i + 6µ2i λi + µ4i
This yields
2

Var(∥Z∥2 ) =

d
X

(3λ2i + 6µ2i λi + µ4i ) −

i=1

d
X

(λi + µ2i )2

i=1

Applying Chebychev’s inequality yields the result.

C. Derivation of expected squared relative variance deviation
Here we want to derive the statement about the expected squared relative variance (denoting the covariance matrix as C
instead of Σ):
1

1

Eu [(uT C − 2 (Ci − C)C − 2 u)2 ] =

2trace(A2 ) + trace(A)2
,
d(d + 2)

(8)

1

1

where u has a uniform distribution on the unit sphere and A = C − 2 (Ci − C)C − 2 . We note that A is symmetric and thus
has an eigendecomposition A = U ΛU T . We have
d
X
X
2
Eu [(uT Au)2 ] = Eu [ (U T u)T Λ(U T u) ] = Eu [(uT Λu)2 ] =
λ2i Eu [u4i ] +
λi λj Eu [u2i u2j ]
i=1

i̸=j
2

It remains to compute these moments on the unit sphere. For this purpose we note that ∥u∥2 =
4
1 = ∥u∥2 =

d
X

!2
u2i

=

i=1

d
X

u4i +

i=1

X

Pd

2
i=1 ui = 1 and thus

u2i u2j

i̸=j

We note that u4i for i = 1, . . . , d and u2i u2j for i ̸= j are all equally distributed and thus for i ̸= j
1 = dE[u4i ] + d(d − 1)E[u2i u2j ]

(9)

Moreover,
we note

 that rotations do not change the distribution for a unifom distribution on the sphere and thus (ui , uj ) and
ui −uj ui +uj
√
, √2
have the same distribution and
2
E[u2i u2j ] = E

"

ui − uj
√
2

2 

ui + uj
√
2

2 #
=

1
1
E[u4i ] − E[u2i u2j ].
2
2

This yields E[u4i ] = 3E[u2i u2j ]. Plugging this into (9) yields
E[u4i ] =

3
,
d(d + 2)

E[u2i u2j ] =

1
d(d + 2)

Thus




d
d
d
X
X
X
X
1
1
3
2
Eu [(uT Au)2 ] =
λ2i +
λi λj  =
λ2i +
λi λj 
d(d + 2)
d(d
+
2)
i=1
i=1
i,j=1
i̸=j

Using that trace(A) =

Pd

i=1 λi finishes the derivation.

14

## Page 15

Mahalanobis++: Improving OOD Detection via Feature Normalization

D. Extended Analysis
Here, we report extended results on the experiments of Section 3 of the main paper. In particular, we show that the
observations made hold beyond the SwinV2 model. If not stated differently, all experiments are with ImageNet as ID dataset
and NINCO as OOD dataset.
Feature Norm Correlation. In the main paper, we showed that for a SwinV2 model, the feature norm of a sample
correlates strongly with the OOD score received via the Mahalanobis distance. Here, we show this phenomenon for more
models. In Figure 7, we plot the feature norm against the OOD score assigned by Mahalanobis and Mahalanobis++ for
four models. For SwinV2, ConvNext and ViT-clip, the feature norms correlate strongly with the OOD score. Normalizing
the features (bottom) mitigates this dependency, as OOD samples with small feature norms are detected as OOD, and thus
improves OOD detection strongly. A notable exception is the augreg-ViT, for which there is no correlation between feature
norms and OOD score. In Figure 6, we further investigate the dependency of the Mahalanobis score on the feature norm by
artificially scaling the feature norm of the OOD features with a prefactor α. That is, for a SwinV2 model, we use α ∗ ϕi for
each OOD feature and leave the ID validation features unchanged. We report the FPR values against α in Figure 6, and again
observe a clear correlation between the scaling factor and the FPR: Perhaps unexpectedly, upscaling the features reduces the
false-positive rate, up to a scaling factor of 2, where zero false positives are achieved. When scaling down the feature norm,
the FPR increases, and for α ≈ 0.5, i.e. at half the original feature norm, all OOD samples are identified as ID. Notably,
this does not change for smaller α values, not even for α = 0, where all OOD samples collapse to the zero vector. In other
words, everything in the vicinity of the origin is identified as in-distribution, which contradicts the intuition of tight Gaussian
clusters centered around class means. In the main paper, we hypothesized that this might explain why the Mahalanobis
distance sometimes fails to detect the unit tests since those might receive a small feature norm. In Figure 10, we plot the
feature norms of different datasets for a range of models with (top) and without (bottom) pretraining. We find that the
feature norms of natural OOD images like those from NINCO tend to be even larger than the ImageNet feature norms. This
violates basic assumptions in feature-norm-based OOD detection methods like the negative-aware-norm (Park et al., 2023b),
indicating that special training schemes might be necessary for those methods. However, noise distributions like the unit
tests from Bitterwolf et al. (2023) can lead to fairly small feature norms for most models. Since we showed that small
feature norms lead to small Mahalanobis distances for many models, this highlights why these supposedly easy-to-detect
images were not detected with the Mahalanobis distance in previous studies.
Feature norm distribution. In Figure 8, we plot the feature norms for four ViTs of exactly the same architecture (ViT-B16).
In order to make the plots comparable, we normalize by the average feature norm per model. We observe that, like for the
SwinV2 in the main paper, the norms vary strongly across and within classes - except for the augreg-ViT. This model is one
of the models that performed well with Mahalanobis ”out-of-the-box”, i.e., not requiring normalization.
QQ plots. In Figure 9 we show QQ plots for four models along three directions in feature space. We observe that the
normalized features (green) more closely resemble a normal distribution compared to the unnormalized features (blue),
which is best visible via the long tail. The only exception is the augreg-ViT, for which normalized and unnormalized features
are similarly close to a Gaussian distribution.
Variance alignment. We report extended results on the expected relative deviation scores (see Eq. 5) for more models in
Table 7. We observe that for all models - except the ViT-augreg - normalization lowers the deviations, indicating a better
alignment of the global variance with the individual class variances. In Figure 11, we illustrate this further: Instead of the
score reported in Table 7, which computes an expectation over all directions, we pick three specific directions: 1) a random
direction, 2) an eigendirection with a large eigenvalue, and 3) an eigendirection with a small eigenvalue. Ideally, along each
direction, the 1000 class variances would coincide with the globally estimated, shared variance. For each direction, we
divide the 1000 class variances by the global variance and plot the resulting distribution. Distributions peaked around 1
indicate that the global variance can capture the class variances well. We observe that the distributions of the variances after
feature normalization peak more towards one for all models, except the ViT-augreg.
Augreg ViTs. The ViTs that showed the best performance with Mahalanobis distance in previous studies were base-size
ViTs pretrained on ImageNet21k and fine-tuned on ImageNet1k by Steiner et al. (2022). The training scheme is called augreg,
a carefully tuned combination of augmentation and regularization methods. In this paper, we made several observations
regarding those models (applies for both base-size and large-size models with pretraining on ImageNet21k). In particular,
they
• show strong OOD detection performance with Mahalanobis distance without normalization, and normalization does
15

## Page 16

Mahalanobis++: Improving OOD Detection via Feature Normalization
Table 7. Deviations from global variance. We report the mean squared relative variance deviation as defined in Equation (5) for multiple
models. In all cases, except for the ViT-augreg, normalization significantly improves the fit of the global covariance matrix to the
covariance structure of the individual classes. As noted in the text for the ViT-augreg the features already follow very well the assumptions
of the Mahalanobis score and normalization leads to no improvements.
model

unnormalized

normalized

0.05
0.26
0.17
0.14
0.12
0.24
0.17
0.21
0.23
0.22
0.17
0.22

0.05
0.12
0.08
0.07
0.09
0.15
0.11
0.14
0.18
0.14
0.11
0.12

ViT-B16-In21k-augreg
SwinV2-B-In21k
ViT-B16-CLIP-L2b-In12k
ViT-B16-In21k-augreg2
ViT-B16-In21k-miil
DeiT3-B16-In21k
ConvNeXt-B-In21k
EVA02-B14-In21k
ConvNeXtV2-B-In21k
ConvNeXtV2-B
ConvNeXt-B-In21k
ConvNeXt-B

not improve Mahalanobis-based OOD detection
• show little variations in feature norm compared to all other investigated models
• show no correlation between feature norm and Mahalanobis score (in contrast to all other investigated models)
• show much weaker heavy tails than the other models
• show low values for the variance deviation metric
• loose their advantage for unnormalized Mahalanobis-based detection when the fine-tuning scheme is changed (the
augreg2 model is fine-tuned from a 21k-augreg-checkpoint, but the fine-tuning scheme differs in learning rate and
augmentations)
In short, the augreg models omit all the points that we identified as problematic for Mahalanobis-based OOD detection. This
indicates that the augreg training scheme induces a feature space that lends itself naturally towards a normal distribution,
aligning well with the assumptions of the Mahalanobis distance as OOD detection method. Understanding the exact reason
why the augreg scheme induces those features is beyond the scope of this paper. The connection of training hyperparameters
and OOD detection performance was, however, investigated by Mueller & Hein (2024). It should be stressed that for
post-hoc OOD detection, we ideally want a method that works well with all models, not only those obtained via a certain
training scheme. We provide such a method with Mahalanobis++.
100

True sample norm

FPR (%)

80
60
40
20
0
0.00

0.25

0.50

0.75

1.00

1.25

1.50

Scaling Factor of OOD samples

1.75

2.00

Figure 6. Impact of the feature norm of OOD samples on their Mahalanobis score. When scaling down the norm of the features while
leaving the feature direction unchanged, OOD samples receive a smaller Mahalanobis score and are incorrectly classified as ID samples.
When the feature norm is artificially increased, the opposite happens.

16

## Page 17

Mahalanobis++: Improving OOD Detection via Feature Normalization

ConvNext
FPR=48.7%

20

30
25
20

0

FPR=31.3%

4000 2000
sMaha

4000

2000
sMaha + +

30
25
20

0

25

90
80
70

2000
sMaha

0

1500 1000
sMaha

FPR=35.8%

45

Feature norm

Feature norm

20

30

4000

35

30

35

0

FPR=28.8%

40

100

40

20
6000

50

ViT-augreg
FPR=31.3%

500

FPR=35.7%

100

40

Feature norm

30

Feature norm

40

15000 10000 5000
sMaha

Feature norm

45

35

Feature norm

Feature norm

50

ViT-clip
FPR=57.2%
Feature norm

SwinV2
FPR=58.2%

35
30
25

90
80
70

20
4000

2000
sMaha + +
ID

0

2000
1000
sMaha + +
TPR=95%

OOD (NINCO)

0

3000

2000 1000
sMaha + +

Figure 7. Mahalanobis++ resolves feature-norm dependency of Mahalanobis score. With unnormalized features, OOD samples with
small pre-logit feature norm were systematically identified as ID, but after normalization, OOD samples with small feature norm are
rightfully detected as OOD, resulting in significantly improved OOD detection with Mahalanobis++. The only exception is an augreg
ViT, which does not show a correlation between feature norm and Mahalanobis score, even without normalization.

ViT-augreg2

ViT-miil

ViT-clip

ViT-augreg
Avg. Norm
Std Dev
Min/Max

|| i ||/|| ||avg

2.0
1.5
1.0
0.5
0

250

500

750

1000

0

250

500

750

1000

0

250

500

750

1000

0

250

500

750

1000

Figure 8. For most models, the feature norms vary strongly across and within classes. The same plot as the ”observed” part of
Figure 5 in the main paper, but normalized by the global average feature norm to make the scales of different models comparable. We
thus show how strongly the feature norms vary relative to their scale. We report results for ViT-B16 models with different pretraining
schemes. Only the augreg ViT shows little variation in feature norm and is the only model that does not benefit from normalization.
Interestingly, the augreg2 model was finetuned on ImageNet-1k from the same 21k-checkpoint as the augreg model and even achieves
higher classification accuracy, but shows a very different feature norm distribution - which reflects in the OOD detection performance
with Mahalanobis and Mahalanobis++: All models except for the augreg model benefit from normalization.

17

## Page 18

Mahalanobis++: Improving OOD Detection via Feature Normalization

Figure 9. QQ-plot: ℓ2 −normalization helps transform the features to be more aligned with a normal distribution. Normalized
features in green, unnormalized features in blue. For a SwinV2, DeiT3 and ViT-augreg2, the feature norms vary strongly across classes
(see e.g. Fig. 3 and Fig. 8) and normalization shifts the distribution towards a Gaussian. For a ViT-B-augreg the feature norms are similar
across classes (see Fig 8) and the feature norms are already fairly normal, so ℓ2 -normalization has almost no effect.

18

## Page 19

Mahalanobis++: Improving OOD Detection via Feature Normalization

ViT-B-orig

ViT-B-miil

ViT-B-clip

ViT-B-augreg

20
40
Feature norm
OOD (NINCO)

50
100
Feature norm

ConvNext

ViT-S

Density

ConvNext-B

5
10 15 20 25 10 20 30
Feature norm
Feature norm
Feature norm
ID
OOD (unit tests)

ResNet

SwinV2

Density

DeiT3

10 20 30
Feature norm

10 20 30
10 20 30 10 20 30
Feature norm
Feature norm
Feature norm
ID
OOD (unit tests)
OOD (NINCO)

30
40
Feature norm

Figure 10. Feature norm distribution. In contrast to previous work (e.g., (Park et al., 2023b)), we find that the feature norm of natural
OOD samples (NINCO in green) is often larger than that of ID samples (orange). Far-OOD data, like noise distributions, tend to have
lower feature norms. This holds for models with (top) and without (bottom) pretraining.

19

## Page 20

Mahalanobis++: Improving OOD Detection via Feature Normalization

SwinV2-B-In21k

Random direction

Large EV direction

unnormalized
normalized
global variance

1.0

1.0

0.4
0.5
0.2
0

2

var(class i) / var(global)

4

Random direction
ViT-B16-CLIP-L2b-In12k

0.8
0.6

0.5

0.0

unnormalized
normalized
global variance

1.5

0.0

0

1.00

0.5

1

2

var(class i) / var(global)

4

0.0

6

3

Small EV direction

1.0

0.25

0.5

2

0.0

4

Large EV direction

Small EV direction
2.0
1.5
1.0

0.5

0.5

2

var(class i) / var(global)

1.0

1.0

1

var(class i) / var(global)

unnormalized
normalized
global variance

1.5

3

Large EV direction

0.50

0

2

var(class i) / var(global)

1.5

0.00

1

var(class i) / var(global)

0.75

Random direction
ViT-B16-224-In21k-augreg2

2

1.0

0.0

0.5

0.0

1

2

0.0

var(class i) / var(global)
Random direction
ViT-B16-224-In21k-augreg

Small EV direction
1.5

unnormalized
normalized
global variance

1.5

0

2

var(class i) / var(global)

4

0.0

1

2

var(class i) / var(global)

Large EV direction

Small EV direction
2.0

1.5

1.5
1.0

1.0

0.5

0.5

0.0

1.0

1

2

var(class i) / var(global)

0.0

0.5
1

2

var(class i) / var(global)

0.0

0.5

1.0

1.5

2.0

var(class i) / var(global)

Figure 11. Mahalanobis++ aligns class-variances. We report the distribution of the variances of the train features for each class along
three directions: 1) a random direction, 2) a large eigendirection, 3) a small eigendirection. For each class, we compute the variance
divided by the global variance, and plot the resulting distributions. Larger deviations from one indicate larger deviations of the class
variance from the global variance. For all directions the distribution of variances is more peaked around 1 after normalization, indicating
that after normalization the shared variance assumption is more appropriate - except for the ViT-augreg.

20

## Page 21

Mahalanobis++: Improving OOD Detection via Feature Normalization

E. Extended results
Table 8. FPR on OpenOOD-near datasets, Green indicates that normalized method is better than its unnormalized counterpart, bold
indicates the best method, and underlined indicates second best method. Maha++ improves over Maha on average by 9.6% in FPR over
all models. Similarly, rMaha++ is 5.5% better in FPR than rMaha. The lowest FPR is achieved by ViM for the EVA02-L14-M38m-In21k
highlighted in blue, closely followed by Maha++ for the same model.
Model
ConvNeXt-B-In21k
ConvNeXt-B
ConvNeXtV2-T-In21k
ConvNeXtV2-B-In21k
ConvNeXtV2-L-In21k
ConvNeXtV2-T
ConvNeXtV2-B
ConvNeXtV2-L
DeiT3-S16-In21k
DeiT3-B16-In21k
DeiT3-L16-In21k
DeiT3-S16
DeiT3-B16
DeiT3-L16
EVA02-B14-In21k
EVA02-L14-M38m-In21k
EVA02-T14
EVA02-S14
EffNetV2-S
EffNetV2-L
EffNetV2-M
Mixer-B16-In21k
SwinV2-B-In21k
SwinV2-L-In21k
SwinV2-S
SwinV2-B
ResNet101
ResNet152
ResNet50
ResNet50-supcon
ViT-T16-In21k-augreg
ViT-S16-In21k-augreg
ViT-B16-In21k-augreg2
ViT-B16-In21k-augreg
ViT-B16-In21k-orig
ViT-B16-In21k-miil
ViT-L16-In21k-augreg
ViT-L16-In21k-orig
ViT-S16-augreg
ViT-B16-augreg
ViT-B16-CLIP-L2b-In12k
ViT-L14-CLIP-L2b-In12k
ViT-H14-CLIP-L2b-In12k
ViT-so400M-SigLip
Average

Val Acc
86.3
84.4
85.1
87.6
88.2
83.5
85.5
86.1
84.8
86.7
87.7
83.4
85.1
85.8
88.7
90.1
80.6
85.7
83.9
85.7
85.2
76.6
87.1
87.5
84.2
84.6
81.9
82.3
80.9
78.7
75.5
81.4
85.1
84.5
81.8
84.3
85.8
81.5
78.8
79.2
86.2
88.2
88.6
89.4
84.4

MSP
53.9
70.0
59.7
50.8
48.4
72.7
69.6
69.8
73.1
67.1
66.7
70.8
71.8
72.5
45.1
38.3
77.6
67.7
72.2
69.0
68.9
82.4
56.0
54.0
73.4
73.9
78.4
76.9
80.7
70.5
83.7
73.8
69.4
64.4
60.6
65.6
56.5
53.1
78.3
77.7
56.2
44.1
44.5
58.6
65.6

E
46.1
89.0
51.2
39.0
38.5
79.1
79.2
77.4
65.8
60.6
54.3
71.3
89.5
83.3
37.0
31.9
76.9
67.0
78.4
81.4
78.6
87.7
43.1
45.4
77.5
77.8
87.0
85.9
95.0
67.5
78.7
61.0
59.1
54.3
44.8
52.6
49.6
40.2
79.3
78.1
46.9
34.8
36.8
56.1
64.0

E+R
44.2
86.8
51.5
38.8
38.6
75.8
76.3
73.7
64.8
57.4
51.2
71.3
90.1
86.7
36.5
31.7
77.3
67.1
76.9
75.6
75.2
87.8
41.9
44.4
77.0
75.3
99.7
99.7
99.6
67.7
75.2
63.6
57.4
58.4
44.9
53.3
38.3
40.1
80.2
75.5
46.0
34.3
36.7
52.5
63.6

ML
46.7
75.1
53.3
41.5
40.6
72.9
70.5
69.9
67.4
61.2
57.1
68.5
76.8
74.4
40.3
34.6
76.5
64.8
72.9
70.1
69.3
84.6
45.6
46.4
73.0
73.1
80.4
79.2
82.5
67.7
79.8
63.1
62.4
54.8
48.0
56.8
48.9
42.2
78.6
77.4
49.1
37.2
38.7
53.4
62.0

ViM
53.8
72.1
45.4
37.3
48.8
72.6
69.0
70.6
68.7
65.5
57.3
63.8
67.3
65.9
36.5
28.1
74.6
58.1
79.7
74.7
77.7
83.3
63.1
66.6
75.4
73.6
82.1
81.4
84.1
87.4
73.5
56.6
77.8
47.9
41.4
55.3
42.6
44.4
85.5
79.0
55.4
32.0
33.6
49.9
62.7

AshS
91.3
99.0
95.7
95.5
95.6
98.6
99.0
98.4
98.8
98.9
97.2
86.7
98.7
85.0
93.8
95.2
97.8
98.2
98.7
98.6
99.0
94.9
84.5
87.2
99.7
98.4
91.5
90.7
91.5
71.5
92.1
88.7
97.4
95.3
62.5
96.2
96.3
55.9
96.6
93.2
99.3
96.9
97.0
95.3
93.0

KNN
63.1
77.5
62.7
49.9
49.5
86.7
74.9
70.3
70.8
63.2
53.9
81.1
79.6
75.3
46.9
40.2
80.6
67.7
70.3
70.0
71.2
89.4
69.8
67.9
79.8
76.0
82.6
81.0
88.3
77.3
86.9
77.2
73.4
74.9
60.1
69.2
73.2
50.7
86.8
83.1
57.9
48.6
50.5
48.6
69.5

21

NNG
52.7
72.1
55.6
42.6
41.4
81.0
70.3
66.8
67.1
57.4
49.4
67.7
83.6
80.0
40.8
35.7
77.3
62.9
68.9
68.6
69.3
87.9
57.6
57.3
75.7
73.4
75.0
72.2
81.9
69.4
86.5
72.1
67.4
68.0
54.6
61.5
65.9
46.0
85.0
81.4
50.6
33.1
35.5
43.7
63.9

NEC
42.2
72.1
48.3
36.8
35.9
69.7
68.2
68.8
65.4
58.1
52.2
68.5
76.1
74.7
36.6
31.5
73.5
60.8
72.7
69.4
68.8
84.8
40.7
41.6
70.2
70.6
81.1
79.8
83.4
67.8
77.4
60.6
59.7
52.8
44.9
52.1
46.5
40.7
78.9
77.3
45.9
37.0
38.6
51.3
59.9

GMN
53.8
72.9
56.2
46.1
54.9
74.6
71.1
77.8
64.5
59.1
52.9
70.5
71.9
67.9
45.2
45.2
67.5
56.4
79.9
70.5
70.2
69.4
60.9
72.6
65.5
66.3
87.4
84.4
91.0
85.5
73.1
62.4
68.6
55.9
61.2
62.4
54.2
59.3
75.6
69.2
52.5
46.2
55.5
63.9
65.3

GEN
46.7
73.3
54.4
41.0
39.7
72.4
65.8
64.1
65.5
53.4
48.1
66.7
65.4
65.8
36.6
30.8
76.4
61.5
69.1
63.5
64.0
82.4
47.0
47.4
72.0
69.6
78.8
76.8
80.7
71.2
83.5
69.3
63.5
57.7
57.1
60.1
48.7
49.2
78.9
78.0
49.1
36.3
36.9
42.8
60.5

fDBD
58.3
76.1
61.7
48.7
49.5
78.4
72.6
68.2
72.6
62.2
56.9
72.8
74.7
72.3
43.3
36.3
79.2
68.1
74.9
69.9
72.1
85.6
62.5
62.2
79.7
76.4
87.6
86.3
88.5
74.1
78.6
66.6
73.2
58.7
52.6
67.3
50.8
46.0
81.2
79.8
56.6
41.6
44.1
48.9
66.3

Maha
59.6
72.0
52.7
42.0
46.6
73.3
66.9
62.4
70.7
62.7
54.5
69.7
71.8
66.4
41.9
31.8
73.0
58.3
75.1
65.8
69.5
78.4
69.8
69.3
75.3
69.5
73.1
71.3
75.5
98.2
69.7
56.7
77.4
44.6
45.3
66.3
36.4
46.4
69.8
69.4
65.0
45.3
44.4
47.0
62.5

Maha++
41.7
59.1
44.9
34.7
31.2
60.5
55.0
53.6
60.4
50.8
47.3
62.4
64.8
60.7
34.8
28.7
71.0
57.0
64.6
56.4
58.1
71.8
46.7
44.0
59.2
59.6
58.7
55.5
65.4
71.3
71.6
56.0
56.8
49.0
40.6
47.5
41.9
39.7
69.9
68.3
44.7
35.9
35.8
38.6
52.9

rMaha
51.8
67.3
52.1
41.3
41.2
66.6
61.5
56.4
68.4
58.6
51.6
65.4
67.4
61.9
42.0
33.3
72.7
58.7
67.7
59.5
61.5
74.9
61.1
60.6
71.1
65.6
62.8
60.6
64.6
91.3
74.3
61.3
68.1
48.3
49.9
60.1
39.2
47.5
66.5
65.4
59.2
41.9
41.6
43.7
58.8

rMaha++
44.7
58.1
48.8
38.2
35.6
58.2
54.1
51.9
60.6
49.9
46.3
60.0
61.7
57.2
38.0
32.3
71.2
57.3
60.2
53.8
54.7
68.8
48.4
46.1
58.0
57.8
59.9
58.6
62.4
68.5
75.0
60.8
56.0
49.9
47.8
52.1
40.8
46.1
66.4
64.7
49.2
39.1
38.9
39.2
53.3

## Page 22

Mahalanobis++: Improving OOD Detection via Feature Normalization

50
90
88

Validation Accuracy (%)

FPR Mahalanobis++ (%)

45
40

86

35

84
82

30

80

25

78

20
15
20

30

40

50

FPR Mahalanobis (%)

60

Figure 12. We plot the FPR with Mahalanobis++ against the FPR with the conventional Mahalanobis score averaged over the five
OpenOOD datasets. With three minor exceptions, Mahalanobis++ improves OOD detection performance for all models. In particular, it
significantly improves all models with high accuracy.

22

## Page 23

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 9. FPR on OpenOOD-far datasets, Green indicates that normalized method is better than its unnormalized counterpart, bold
indicates the best method, and underlined indicates second best method. Maha++ improves over Maha on average by 6.1% in FPR over all
models. Similarly, rMaha++ is 1.2% better in FPR than rMaha. In total, Maha++ improves the SOTA compared to the previously strongest
method, ViM, by 5.1%, which is significant. The lowest FPR is achieved by Maha++ for the EVA02-L14-M38m-In21k highlighted in
blue.
Model
ConvNeXt-B-In21k
ConvNeXt-B
ConvNeXtV2-T-In21k
ConvNeXtV2-B-In21k
ConvNeXtV2-L-In21k
ConvNeXtV2-T
ConvNeXtV2-B
ConvNeXtV2-L
DeiT3-S16-In21k
DeiT3-B16-In21k
DeiT3-L16-In21k
DeiT3-S16
DeiT3-B16
DeiT3-L16
EVA02-B14-In21k
EVA02-L14-M38m-In21k
EVA02-T14
EVA02-S14
EffNetV2-S
EffNetV2-L
EffNetV2-M
Mixer-B16-In21k
SwinV2-B-In21k
SwinV2-L-In21k
SwinV2-S
SwinV2-B
ResNet101
ResNet152
ResNet50
ResNet50-supcon
ViT-T16-In21k-augreg
ViT-S16-In21k-augreg
ViT-B16-In21k-augreg2
ViT-B16-In21k-augreg
ViT-B16-In21k-orig
ViT-B16-In21k-miil
ViT-L16-In21k-augreg
ViT-L16-In21k-orig
ViT-S16-augreg
ViT-B16-augreg
ViT-B16-CLIP-L2b-In12k
ViT-L14-CLIP-L2b-In12k
ViT-H14-CLIP-L2b-In12k
ViT-so400M-SigLip
Average

Val Acc
86.3
84.4
85.1
87.6
88.2
83.5
85.5
86.1
84.8
86.7
87.7
83.4
85.1
85.8
88.7
90.1
80.6
85.7
83.9
85.7
85.2
76.6
87.1
87.5
84.2
84.6
81.9
82.3
80.9
78.7
75.5
81.4
85.1
84.5
81.8
84.3
85.8
81.5
78.8
79.2
86.2
88.2
88.6
89.4
84.4

MSP
33.6
55.7
34.7
26.9
26.0
52.4
51.7
51.2
52.1
48.5
47.2
47.5
51.6
52.2
23.9
19.5
56.2
41.9
50.8
49.1
49.1
64.2
35.1
31.3
53.1
54.6
60.5
59.3
66.2
43.0
62.1
45.8
46.0
34.5
33.9
36.3
29.3
32.7
55.8
55.3
32.9
23.2
23.7
36.8
44.0

E
36.1
92.2
28.0
19.4
19.3
57.4
65.2
61.7
44.9
50.7
39.7
42.5
77.4
78.6
20.0
16.5
59.0
44.4
66.1
69.4
63.2
79.8
30.4
29.7
61.8
58.5
79.9
79.5
96.5
33.9
39.6
24.2
37.1
19.9
21.3
23.3
16.0
22.1
45.4
47.2
31.6
18.8
19.6
41.1
45.7

E+R
30.6
86.9
27.5
18.7
18.5
47.1
56.0
51.0
40.8
38.2
29.6
49.3
87.2
91.0
19.3
16.2
59.7
43.8
46.6
45.3
44.4
80.7
23.5
22.4
52.2
46.9
99.5
99.4
99.2
25.0
30.6
28.5
30.2
21.0
21.5
22.2
16.1
22.0
47.5
43.3
28.5
18.1
19.0
30.7
42.6

ML
31.1
67.0
28.8
20.7
20.4
49.6
52.2
50.6
45.7
45.8
40.0
41.3
56.1
57.1
21.0
17.5
54.2
39.2
53.3
51.3
49.3
68.5
28.4
26.6
52.9
52.1
64.3
63.8
71.3
35.5
44.0
27.4
37.6
21.2
23.2
26.8
17.4
23.6
47.6
48.6
29.3
19.3
20.3
34.0
40.4

ViM
13.3
39.9
14.7
12.4
15.3
34.8
32.0
33.6
33.6
24.9
22.4
29.6
29.6
32.8
12.4
11.2
32.5
19.2
33.8
31.7
39.4
64.1
17.5
20.7
34.9
34.1
29.4
28.5
32.4
61.7
36.1
18.0
38.0
13.0
20.7
26.2
11.0
21.0
56.6
53.0
22.3
14.5
14.8
17.7
28.1

AshS
86.6
99.8
97.3
95.2
95.6
99.6
99.9
99.6
99.5
99.5
98.7
84.9
99.6
73.9
83.9
88.2
98.8
99.6
99.8
99.6
99.8
96.4
72.0
83.7
99.9
99.6
72.7
72.8
72.8
20.0
96.8
68.8
99.5
94.6
62.7
97.4
93.4
44.9
97.1
88.2
99.7
98.0
98.3
92.4
89.1

KNN
20.0
46.2
27.5
18.6
18.4
62.4
39.1
34.6
35.8
24.8
22.5
62.3
57.2
39.8
18.0
16.1
47.6
28.3
29.2
34.8
38.0
70.1
20.2
19.6
44.5
41.0
34.2
32.8
54.2
26.8
69.1
41.2
30.3
40.6
24.2
28.9
35.6
22.8
61.3
53.8
20.8
17.5
18.3
15.4
35.1

23

NNG
17.9
37.2
23.6
15.9
15.6
50.6
32.9
30.0
34.4
24.3
20.7
35.1
63.1
67.6
15.8
14.2
43.7
25.2
29.4
32.9
34.7
81.2
16.3
16.6
37.6
36.2
35.6
29.9
52.2
23.5
60.7
33.4
25.4
30.7
22.6
23.2
24.7
22.0
58.2
52.5
19.0
15.1
15.3
14.3
32.1

NEC
24.1
62.7
23.1
16.7
16.3
43.6
47.7
47.2
42.4
38.6
29.3
41.3
55.1
57.5
17.3
15.3
43.4
31.0
50.3
47.1
45.7
69.3
21.1
20.7
46.5
45.2
63.6
62.0
72.0
34.4
36.4
23.1
33.1
18.8
20.9
21.7
15.7
21.8
47.7
48.2
26.0
19.1
20.1
31.9
36.7

GMN
54.5
74.5
41.2
34.2
18.2
73.4
71.3
54.5
43.6
37.7
26.1
59.9
57.8
49.5
29.8
36.4
45.2
34.2
76.6
49.8
64.6
55.8
55.5
32.9
58.3
49.3
78.9
72.3
88.6
74.4
48.3
32.2
54.5
27.0
40.5
53.5
33.1
40.3
52.5
54.3
34.0
29.6
51.3
64.6
50.3

GEN
23.2
51.4
24.6
17.7
17.5
41.1
37.1
34.4
36.0
29.5
25.6
33.1
33.1
31.4
16.2
13.2
45.5
28.5
36.7
32.2
32.7
57.8
21.7
20.8
39.9
36.8
51.6
49.3
55.2
41.7
52.1
31.4
29.2
22.3
25.9
23.7
18.7
25.8
50.3
50.5
22.9
16.4
16.3
18.7
32.3

fDBD
24.3
49.8
29.4
19.1
19.6
50.7
41.1
35.3
42.2
29.1
25.8
41.8
41.6
39.1
18.8
15.6
57.9
36.1
40.1
35.4
39.6
62.6
22.3
21.1
48.1
43.9
60.4
58.0
65.7
30.7
45.2
29.0
36.8
21.6
24.4
29.5
16.9
24.3
53.8
52.7
25.7
17.7
18.7
17.3
35.4

Maha
16.3
42.3
19.1
13.7
15.3
43.5
32.5
27.9
36.5
24.6
21.9
40.9
37.9
31.3
14.6
11.7
36.2
22.1
28.8
25.0
30.3
53.3
22.7
23.5
37.1
33.5
27.7
26.5
32.2
93.7
46.0
23.4
38.7
13.0
21.2
34.4
10.7
20.6
35.6
36.4
28.7
16.8
15.6
16.8
29.1

Maha++
12.8
34.9
17.7
11.9
10.5
33.7
25.8
24.6
30.5
20.2
18.6
35.8
34.7
25.7
11.8
10.3
32.9
21.0
23.9
20.0
23.1
39.7
13.3
11.9
25.3
27.1
33.4
26.9
43.0
26.6
32.2
15.5
25.8
14.5
18.8
19.0
11.8
18.3
35.5
34.6
17.1
13.4
12.8
15.1
23.0

rMaha
18.2
38.5
23.0
15.7
15.5
37.2
30.6
27.4
35.9
24.1
21.3
38.3
35.6
29.9
15.6
13.0
39.3
24.4
27.6
23.7
27.5
50.0
20.8
20.0
33.8
31.6
50.8
45.9
61.2
89.5
49.1
30.9
33.0
19.1
25.8
32.6
15.8
24.3
36.3
35.8
23.9
17.2
16.9
16.3
30.5

rMaha++
19.4
37.0
23.1
15.2
14.6
35.5
29.0
28.6
32.2
21.2
19.0
34.8
33.8
26.3
14.2
12.4
37.1
23.1
27.2
22.2
24.8
42.3
18.8
17.1
26.8
27.6
71.4
68.7
75.7
60.5
46.1
27.1
27.8
19.3
24.6
26.3
15.8
23.6
36.0
34.7
21.1
16.3
15.9
16.4
29.3

## Page 24

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 10. AUC on OpenOOD, Green indicates that normalized method is better than its unnormalized counterpart, bold indicates the
best method, and underlined indicates second best method. Maha++ improves over Maha on average by 2.1% in AUC over all models.
Similarly, rMaha++ is 0.6% better in AUC than rMaha. In total, Maha++ improves the SOTA compared to the previously strongest
methods rMaha by 1.4%, which is significant. The highest AUC is achieved by Maha++ for the EVA02-L14-M38m-In21k highlighted in
blue.
Model
ConvNeXt-B-In21k
ConvNeXt-B
ConvNeXtV2-T-In21k
ConvNeXtV2-B-In21k
ConvNeXtV2-L-In21k
ConvNeXtV2-T
ConvNeXtV2-B
ConvNeXtV2-L
DeiT3-S16-In21k
DeiT3-B16-In21k
DeiT3-L16-In21k
DeiT3-S16
DeiT3-B16
DeiT3-L16
EVA02-B14-In21k
EVA02-L14-M38m-In21k
EVA02-T14
EVA02-S14
EffNetV2-S
EffNetV2-L
EffNetV2-M
Mixer-B16-In21k
SwinV2-B-In21k
SwinV2-L-In21k
SwinV2-S
SwinV2-B
ResNet101
ResNet152
ResNet50
ResNet50-supcon
ViT-T16-In21k-augreg
ViT-S16-In21k-augreg
ViT-B16-In21k-augreg2
ViT-B16-In21k-augreg
ViT-B16-In21k-orig
ViT-B16-In21k-miil
ViT-L16-In21k-augreg
ViT-L16-In21k-orig
ViT-S16-augreg
ViT-B16-augreg
ViT-B16-CLIP-L2b-In12k
ViT-L14-CLIP-L2b-In12k
ViT-H14-CLIP-L2b-In12k
ViT-so400M-SigLip
Average

Val Acc
86.3
84.4
85.1
87.6
88.2
83.5
85.5
86.1
84.8
86.7
87.7
83.4
85.1
85.8
88.7
90.1
80.6
85.7
83.9
85.7
85.2
76.6
87.1
87.5
84.2
84.6
81.9
82.3
80.9
78.7
75.5
81.4
85.1
84.5
81.8
84.3
85.8
81.5
78.8
79.2
86.2
88.2
88.6
89.4
84.4

MSP
88.8
80.9
87.4
89.9
90.6
83.5
82.8
83.0
80.3
82.4
84.3
84.2
83.5
82.9
91.0
92.4
81.3
84.4
83.2
83.9
83.7
80.4
88.0
88.7
82.3
82.4
79.9
80.4
77.5
85.2
79.9
84.4
84.4
87.5
88.0
87.7
89.6
89.5
82.3
82.7
87.9
90.5
90.6
87.6
85.0

E
86.8
61.3
88.1
90.4
91.1
78.1
73.0
73.0
78.7
74.8
81.5
83.8
69.5
76.6
90.4
91.7
79.1
79.7
74.2
73.5
74.4
78.9
87.0
86.8
74.8
75.9
68.2
67.6
52.6
87.7
85.5
90.4
84.6
91.9
93.2
90.6
93.0
93.3
84.9
85.8
85.8
89.9
89.4
79.7
81.5

E+R
88.9
71.2
88.3
91.0
91.9
82.2
78.7
79.9
81.2
82.2
86.6
83.0
63.3
66.4
91.3
92.2
78.9
80.1
83.2
83.2
83.3
78.7
90.4
90.8
81.4
82.8
23.1
21.0
27.6
88.7
86.9
89.9
87.3
91.3
93.1
91.0
94.1
93.3
84.6
86.4
87.8
90.6
90.1
85.8
81.0

ML ViM
88.4 93.3
74.7 85.4
88.3 93.3
90.4 94.8
91.1 93.8
81.7 86.7
79.0 86.7
78.9 84.5
79.5 87.1
78.4 90.6
82.7 92.0
84.7 88.2
79.0 87.6
80.5 87.0
90.9 95.0
92.3 95.9
80.8 87.2
82.1 91.6
79.3 86.7
80.6 85.6
80.6 85.3
79.8 82.5
88.3 92.1
88.1 91.9
78.9 87.0
79.4 86.1
75.8 83.7
75.8 84.2
73.3 82.0
87.6 82.3
85.1 86.5
90.0 91.6
85.4 85.9
91.7 93.6
92.7 93.5
90.1 91.8
92.9 94.6
93.0 93.4
84.8 80.4
85.7 84.1
87.1 92.3
90.4 94.6
90.2 94.2
83.8 92.6
84.4 89.1

AshS
52.1
19.8
40.3
45.2
45.4
21.6
15.8
17.3
19.5
18.4
20.5
61.9
22.4
65.3
54.0
48.0
43.1
22.7
20.8
17.8
17.8
48.6
47.5
38.4
14.8
21.2
56.4
56.0
60.8
88.6
45.7
66.5
24.6
54.8
77.1
32.1
58.1
86.2
46.7
52.9
19.5
38.2
27.6
25.7
40.4

KNN
90.1
84.4
88.6
92.0
92.1
81.2
86.3
87.2
86.0
89.7
91.1
83.0
83.1
84.6
92.2
93.6
81.7
87.2
86.8
86.9
86.3
79.3
88.8
89.8
84.3
85.7
82.8
82.7
79.7
86.4
75.9
85.3
86.8
85.6
90.4
88.4
87.8
92.0
77.9
81.2
90.4
92.2
92.0
91.8
86.6

24

NNG
92.6
86.8
91.0
93.6
93.8
85.3
88.3
88.8
87.4
90.4
91.4
86.5
79.1
80.4
93.6
94.6
85.0
89.5
88.0
87.8
87.6
79.3
91.8
92.2
86.8
87.2
85.5
85.9
81.5
88.6
80.0
88.2
89.1
89.1
91.9
91.1
90.6
92.9
81.8
84.0
92.5
94.1
93.6
93.1
88.5

NEC
91.6
76.4
91.3
93.4
94.0
83.5
80.1
78.9
81.1
82.7
88.3
84.9
79.8
80.6
93.4
94.4
85.0
86.4
82.0
82.2
82.4
79.9
91.9
91.8
81.5
82.0
76.7
77.6
73.3
87.8
86.4
90.8
87.6
92.4
93.3
91.7
93.5
93.5
84.6
85.6
89.6
90.9
90.8
85.7
86.2

GMN
88.1
81.6
89.1
91.1
92.7
82.2
83.4
85.8
87.9
89.8
90.9
85.4
85.1
86.3
92.5
92.0
86.4
90.5
82.1
86.3
84.6
84.0
87.0
88.9
85.7
86.7
74.9
77.7
65.1
78.9
84.1
88.8
86.0
90.1
87.7
86.6
89.9
87.3
83.9
84.5
91.2
92.3
88.9
88.8
86.2

GEN
91.5
83.2
90.0
92.3
92.8
86.6
86.7
87.3
85.2
87.4
89.3
87.8
87.6
87.8
93.3
94.3
85.0
87.8
87.2
87.8
88.0
82.2
91.4
91.6
86.4
87.2
83.0
83.6
81.9
86.7
83.9
89.1
88.6
91.6
91.1
91.0
93.0
91.9
85.1
85.6
90.8
92.6
92.7
91.6
88.4

fDBD
90.3
84.7
86.4
91.4
91.7
84.9
86.8
88.0
85.5
88.7
89.4
86.5
86.1
86.7
92.6
94.1
80.0
85.4
86.6
86.6
86.2
81.3
90.1
90.8
85.7
86.1
79.2
80.6
77.4
87.1
85.5
89.2
84.5
91.4
92.0
89.1
93.1
92.5
83.5
84.8
90.6
92.7
92.1
92.0
87.5

Maha
91.9
87.1
91.8
94.0
93.7
87.0
89.0
89.8
87.9
91.0
92.0
87.9
88.1
89.0
94.0
95.5
86.6
90.8
88.6
89.6
88.9
83.1
90.6
91.1
87.9
88.8
88.0
88.5
86.5
52.9
84.3
90.4
86.7
93.4
92.7
89.6
94.9
92.8
86.9
87.4
90.5
93.6
93.7
93.2
89.1

Maha++
93.7
88.7
92.7
94.9
95.3
88.9
90.5
90.6
89.5
92.3
93.0
88.9
88.9
89.9
94.9
95.9
87.2
91.2
90.0
90.8
90.5
86.4
92.9
93.7
90.0
90.2
89.5
90.0
87.5
89.2
86.5
91.5
90.3
92.5
93.5
92.6
94.0
93.8
86.8
87.6
93.3
94.8
94.7
93.8
91.2

rMaha
92.0
87.9
91.2
93.7
93.8
88.2
89.6
90.3
88.1
91.1
91.9
88.6
88.9
89.7
93.6
95.1
86.3
90.4
89.7
90.7
90.2
85.1
90.8
91.5
88.3
89.0
87.0
87.7
84.6
76.5
83.8
89.1
88.6
92.0
91.6
90.2
93.9
92.2
87.4
88.0
91.2
93.7
93.7
93.4
89.8

rMaha++
92.5
88.7
91.5
94.0
94.2
89.1
90.3
90.6
89.2
92.1
92.8
89.3
89.6
90.4
94.1
95.3
86.7
90.7
90.4
91.3
91.0
86.4
91.9
92.7
90.0
90.1
85.2
86.0
83.1
85.6
84.0
89.4
90.0
91.9
91.8
91.2
93.7
92.4
87.4
88.1
92.2
94.0
94.0
93.6
90.4

## Page 25

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 11. AUC on NINCO datasets, Green indicates that normalized method is better than its unnormalized counterpart, bold indicates
the best method, and underlined indicates second best method. Maha++ improves over Maha on average by 2.6% in AUC over all models.
Similarly, rMaha++ is 1.0% better in AUC than rMaha. In total, Maha++ improves the SOTA compared to the previously strongest
methods rMaha by 1.0%, which is significant. The highest AUC is achieved by Maha++ for the EVA02-L14-M38m-In21k highlighted in
blue.
Model
ConvNeXt-B-In21k
ConvNeXt-B
ConvNeXtV2-T-In21k
ConvNeXtV2-B-In21k
ConvNeXtV2-L-In21k
ConvNeXtV2-T
ConvNeXtV2-B
ConvNeXtV2-L
DeiT3-S16-In21k
DeiT3-B16-In21k
DeiT3-L16-In21k
DeiT3-S16
DeiT3-B16
DeiT3-L16
EVA02-B14-In21k
EVA02-L14-M38m-In21k
EVA02-T14
EVA02-S14
EffNetV2-S
EffNetV2-L
EffNetV2-M
Mixer-B16-In21k
SwinV2-B-In21k
SwinV2-L-In21k
SwinV2-S
SwinV2-B
ResNet101
ResNet152
ResNet50
ResNet50-supcon
ViT-T16-In21k-augreg
ViT-S16-In21k-augreg
ViT-B16-In21k-augreg2
ViT-B16-In21k-augreg
ViT-B16-In21k-orig
ViT-B16-In21k-miil
ViT-L16-In21k-augreg
ViT-L16-In21k-orig
ViT-S16-augreg
ViT-B16-augreg
ViT-B16-CLIP-L2b-In12k
ViT-L14-CLIP-L2b-In12k
ViT-H14-CLIP-L2b-In12k
ViT-so400M-SigLip
Average

Val Acc
86.3
84.4
85.1
87.6
88.2
83.5
85.5
86.1
84.8
86.7
87.7
83.4
85.1
85.8
88.7
90.1
80.6
85.7
83.9
85.7
85.2
76.6
87.1
87.5
84.2
84.6
81.9
82.3
80.9
78.7
75.5
81.4
85.1
84.5
81.8
84.3
85.8
81.5
78.8
79.2
86.2
88.2
88.6
89.4
84.4

MSP
88.2
81.7
86.7
89.4
90.2
82.9
82.6
82.2
77.9
81.5
83.9
82.9
82.2
81.2
90.6
92.6
79.2
81.8
81.1
82.6
82.3
79.3
87.4
88.2
80.9
80.8
78.7
79.1
76.8
84.9
78.1
83.4
83.2
86.3
87.2
86.5
89.5
89.3
80.8
81.0
86.6
90.0
89.7
87.3
84.1

E
85.7
64.8
87.2
89.1
89.3
76.2
73.2
71.7
74.8
74.5
81.9
81.4
66.6
75.5
89.7
91.1
75.3
76.1
71.1
73.3
71.8
78.1
86.0
85.9
74.8
74.9
67.3
67.2
55.0
87.2
81.0
88.6
82.7
91.1
91.8
88.1
92.7
91.7
81.8
83.3
82.9
87.9
87.2
79.5
80.2

E+R
87.8
73.2
87.4
89.9
90.7
80.3
78.2
78.6
77.2
80.5
86.3
81.2
62.4
68.8
91.0
91.8
75.0
76.5
77.6
80.3
79.5
78.1
89.2
89.9
79.7
80.4
21.1
20.0
23.7
87.1
82.1
88.4
85.2
90.3
91.8
88.6
94.9
91.8
81.5
83.8
85.5
89.0
88.2
85.1
79.3

ML
87.6
76.7
87.5
89.3
89.7
80.5
79.2
78.0
76.2
77.5
82.7
83.1
76.7
78.6
90.3
92.1
77.7
78.8
76.3
79.6
78.6
78.8
87.5
87.4
78.1
78.0
74.6
74.6
73.0
87.2
81.3
88.5
83.9
91.0
91.5
88.0
92.7
91.6
82.2
83.4
84.9
89.0
88.5
83.2
83.1

ViM
92.5
83.0
92.7
94.8
93.9
83.5
83.6
80.8
85.4
89.3
90.9
86.3
85.1
84.7
94.7
96.1
83.6
88.6
81.7
80.2
81.0
80.8
90.8
90.8
83.9
81.4
76.9
77.3
73.5
79.2
82.2
89.7
83.6
92.4
92.6
91.2
93.9
91.9
74.8
81.0
91.1
94.4
93.9
92.1
86.6

AshS
45.4
25.1
39.7
43.8
39.8
26.0
20.0
19.1
23.2
21.3
25.3
64.6
28.4
62.3
50.1
40.9
41.1
27.0
20.7
20.9
19.3
49.8
44.1
36.9
17.6
28.9
47.5
48.7
52.6
86.0
54.4
61.1
28.7
56.2
71.0
31.3
48.1
81.7
44.4
52.4
21.5
33.4
27.6
26.1
39.9

KNN
88.0
81.6
86.8
91.1
91.6
78.0
83.7
84.8
83.4
87.6
89.6
81.1
80.5
81.1
91.1
93.2
78.3
84.3
83.5
84.0
84.0
76.2
86.1
87.8
80.7
82.3
77.6
77.6
75.1
84.3
73.3
82.0
84.7
82.8
88.3
87.2
83.9
89.7
73.8
77.8
88.4
91.2
91.0
91.8
84.1

25

NNG
91.5
85.2
90.2
93.5
93.9
83.1
86.6
87.1
85.4
89.2
90.8
85.0
77.1
78.8
93.2
94.6
81.9
87.1
84.9
85.4
85.4
78.6
90.5
91.5
83.8
84.2
82.6
83.1
78.8
86.8
76.2
85.6
87.6
87.5
90.1
89.7
88.8
91.3
78.2
81.1
91.1
93.3
92.4
93.3
86.7

NEC
90.7
78.0
90.6
92.7
93.4
82.0
79.8
77.5
77.6
81.3
87.4
83.0
77.3
78.7
92.7
94.2
81.5
82.8
78.3
79.7
79.4
78.9
91.2
91.4
80.3
79.7
74.7
75.6
72.3
87.3
82.3
89.2
86.1
91.5
92.1
90.0
93.2
92.1
81.9
83.3
87.9
89.6
89.2
84.7
84.6

GMN
87.7
81.1
88.8
91.3
92.6
81.0
82.9
84.2
87.6
89.6
90.6
84.4
83.9
84.8
92.6
92.0
83.7
89.6
80.6
85.3
83.9
82.3
86.7
87.7
84.5
84.5
74.1
76.4
66.2
80.3
81.0
86.7
85.5
89.3
87.3
86.5
88.6
86.1
81.4
82.0
90.6
92.6
89.8
88.3
85.4

GEN
90.9
83.3
89.4
92.0
92.3
85.6
86.1
86.4
82.7
86.6
89.1
86.4
86.4
86.0
93.2
94.7
82.3
85.0
84.6
86.4
86.6
80.9
90.8
91.2
84.7
85.2
80.9
81.4
79.7
86.9
80.8
87.7
87.8
91.1
90.3
89.8
93.2
91.1
83.0
83.9
89.6
92.2
91.8
91.7
87.3

fDBD
89.5
83.0
85.9
91.8
92.8
82.5
85.2
86.1
84.2
87.5
88.7
85.1
84.0
84.9
93.0
94.8
78.0
82.8
83.7
84.0
84.1
79.1
88.6
89.8
82.9
83.1
73.9
75.7
71.7
85.0
83.0
88.4
82.8
90.3
91.2
88.2
92.4
91.6
81.0
82.5
90.1
93.4
92.7
92.4
85.9

Maha
91.2
85.8
91.7
94.5
94.1
85.4
87.9
88.7
87.3
90.2
91.6
87.5
87.2
88.2
94.0
96.0
84.2
89.2
85.5
87.4
87.2
80.5
89.4
90.0
86.0
86.8
85.3
85.9
82.7
48.2
84.0
90.7
86.1
94.1
93.1
89.9
95.3
92.4
85.3
85.7
89.4
93.9
94.2
93.3
88.1

Maha++
94.3
88.5
92.9
95.6
96.2
88.3
90.1
90.2
89.2
91.8
92.8
88.6
88.3
89.1
95.1
96.4
84.8
89.6
87.1
89.0
89.1
84.6
92.9
94.1
88.7
88.5
89.1
89.5
86.5
88.1
83.7
90.8
90.7
93.2
93.8
93.1
94.4
93.5
85.1
85.7
92.9
95.2
95.3
94.6
90.7

rMaha
92.3
87.2
91.4
94.5
94.7
87.5
89.0
89.9
87.6
90.6
91.9
88.4
88.3
89.2
93.9
95.9
84.3
89.3
88.1
89.5
89.5
83.8
90.4
91.2
86.6
87.3
87.8
87.9
86.1
78.7
83.0
89.5
88.6
93.2
92.7
91.1
95.1
92.9
86.6
86.9
90.6
94.2
94.3
94.2
89.7

rMaha++
93.5
88.8
91.9
95.0
95.5
89.0
90.3
90.6
89.0
91.9
92.9
89.3
89.2
90.1
94.6
96.0
84.7
89.6
89.4
90.5
90.6
85.2
92.5
93.5
88.9
88.7
88.1
88.6
86.6
87.0
82.8
89.4
90.8
93.0
92.9
92.2
94.8
93.1
86.6
86.9
92.3
94.8
94.8
94.8
90.7

## Page 26

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 12. FPR on NINCO for cosine-based methods, Green indicates that the normalized method is better than its unnormalized
counterpart, bold indicates the best method, and underlined indicates the second best method. Mahalanobis++ consistently outperforms
other cosine-based methods. In only 2 out of 44 models, another method (once NNguide and once Cosine) is better than Maha++.
Model

Accuracy

ConvNeXt-B
ConvNeXt-B-In21k
ConvNeXtV2-B
ConvNeXtV2-B-In21k
ConvNeXtV2-L
ConvNeXtV2-L-In21k
ConvNeXtV2-T
ConvNeXtV2-T-In21k
DeiT3-B16
DeiT3-B16-In21k
DeiT3-L16
DeiT3-L16-In21k
DeiT3-S16
DeiT3-S16-In21k
EVA02-B14-In21k
EVA02-L14-M38m-In21k
EVA02-S14
EVA02-T14
Mixer-B16-In21k
ResNet101
ResNet152
ResNet50
ResNet50-supcon
SwinV2-B-In21k
SwinV2-B
SwinV2-L-In21k
SwinV2-S
EffNetV2-L
EffNetV2-M
EffNetV2-S
ViT-B16-In21k-augreg2
ViT-B16-augreg
ViT-B16-In21k-augreg
ViT-B16-In21k-orig
ViT-B16-In21k-miil
ViT-B16-CLIP-L2b-In12k
ViT-H14-CLIP-L2b-In12k
ViT-L14-CLIP-L2b-In12k
ViT-L16-In21k-augreg
ViT-L16-In21k-orig
ViT-S16-augreg
ViT-S16-In21k-augreg
ViT-so400M-SigLip
ViT-T16-In21k-augreg

84.434
86.270
85.474
87.642
86.120
88.196
83.462
85.104
85.074
86.744
85.812
87.722
83.434
84.826
88.694
90.054
85.720
80.630
76.598
81.890
82.286
80.856
78.686
87.096
84.604
87.468
84.220
85.664
85.204
83.896
85.096
79.152
84.528
81.790
84.268
86.172
88.588
88.178
85.840
81.508
78.842
81.388
89.406
75.466

Maha++

Cosine

KNN (Sun et al., 2022)

NNguide (Park et al., 2023a)

SSC (Techapanurak et al., 2020)

50.5
28.8
44.7
22.4
43.0
18.4
52.3
32.8
57.2
38.8
50.4
33.9
53.5
50.8
23.8
18.6
48.0
64.0
65.4
50.4
46.5
61.0
59.6
31.3
52.2
28.3
49.8
47.8
50.0
59.9
45.9
61.3
35.7
31.6
35.4
35.8
23.7
25.4
28.9
32.4
63.1
44.6
27.4
63.2

60.6
42.2
57.1
31.1
53.8
27.0
69.4
45.8
67.1
46.9
62.2
38.9
67.6
58.4
29.1
22.7
53.6
69.8
78.2
61.5
62.1
64.0
58.9
45.9
63.4
42.8
65.2
57.0
58.0
58.8
56.4
73.1
53.4
46.0
49.7
43.5
31.5
30.9
50.0
39.2
75.8
60.3
29.0
77.2

70.1
51.6
67.3
40.9
62.4
38.7
82.3
54.1
74.5
52.6
67.2
43.8
75.6
62.9
37.6
30.3
60.0
74.5
85.8
74.9
72.0
83.7
65.8
57.2
69.4
55.1
73.1
62.5
63.1
60.9
64.0
77.6
67.7
52.7
59.6
49.4
41.7
39.5
68.6
45.8
82.1
70.9
36.3
81.7

62.2
41.2
60.3
31.7
56.9
29.9
73.9
45.0
80.1
46.4
77.9
37.8
57.8
59.3
30.0
26.1
54.0
71.0
83.7
66.4
61.6
75.0
58.4
42.7
65.2
41.7
66.8
60.1
60.6
59.6
57.0
75.9
59.0
47.6
51.7
42.3
27.4
25.4
58.9
40.7
80.0
64.0
30.0
81.9

69.3
51.3
66.6
40.4
59.7
39.2
72.8
55.1
65.7
53.2
68.7
46.3
63.6
65.2
34.4
28.8
63.8
75.5
79.5
87.2
85.8
88.2
74.1
53.7
71.6
53.6
75.2
62.4
64.4
68.7
64.8
75.7
54.0
45.2
62.1
48.8
36.5
33.3
48.2
42.0
78.1
61.5
35.6
74.2

Table 13. AUROC for CIFAR10, Green indicates that the normalized method is better than its unnormalized counterpart, bold indicates
the best method, and underlined indicates the second best method. Maha++ is clearly the best method. Only for the WRN28-10 Maha is
better (but not significantly). Maha++ improves in all cases over the previously beset methods ViM. We highlight the best AUC achieved
by Maha++ for the ViT-B16-21k-1k in blue.
Model
SwinV2-S-1k
ViT-B16-21k-1k
RN18
RN34
RNxt29-32
Average
RN50-SC
RN34-SC

Ash
69.96
82.75
87.15
78.29
78.33
79.29
—
—

Dice
92.85
99.33
89.60
84.84
71.90
87.70
—
—

Ebo
95.61
99.42
91.09
87.26
88.45
92.37
—
—

KlM
98.04
96.98
79.62
82.75
83.19
88.11
—
—

KNN
99.25
99.64
91.58
92.15
90.46
94.62
96.76
96.15

ML
95.83
99.41
90.97
87.20
88.20
92.32
—
—

MSP
96.60
98.88
89.93
88.11
87.98
92.30
—
—

O-Max
97.02
97.66
89.04
87.38
85.65
91.35
—
—

React
96.83
99.45
90.78
87.50
85.27
91.97
—
—

26

She
96.88
98.99
87.62
81.40
87.90
90.56
—
—

NNguide
67.51
87.30
63.57
55.07
29.57
60.60
—
—

T-Scal
96.61
99.06
90.32
88.07
87.97
92.41
—
—

ViM
99.53
99.67
91.12
92.50
91.36
94.84
—
—

Neco
98.86
99.56
90.67
86.39
89.62
93.02
—
—

rMD
98.83
99.03
89.92
90.34
89.84
93.59
94.46
94.72

rMD++
98.79
99.04
90.06
90.49
88.69
93.41
94.30
94.24

MD
99.50
99.60
86.87
91.53
90.70
93.64
59.00
64.21

MD++
99.57
99.71
91.69
93.61
91.56
95.23
96.80
96.77

## Page 27

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 14. FPR for CIFAR10, Green indicates that the normalized method is better than its unnormalized counterpart, bold indicates
the best method, and underlined indicates the second best method. Maha++ is the best method on average. We highlight the best FPR
achieved by Maha++ for the ViT-B16-21k-1k in blue.
Model
SwinV2-S-1k
ViT-B16-21k-1k
RN18
RN34
RNxt29-32
Average
RN50-SC
RN34-SC

Ash
93.43
60.41
47.52
46.12
97.43
68.98
—
—

Dice
19.51
2.42
41.18
44.20
63.57
34.18
—
—

Ebo
8.82
1.93
39.22
38.05
47.36
27.08
—
—

KlM
6.02
6.23
56.48
52.89
56.54
35.63
—
—

KNN
4.03
1.75
45.55
46.24
55.91
30.70
19.48
22.47

ML
8.07
1.97
40.28
39.14
50.41
27.97
—
—

MSP
6.74
3.27
56.42
51.99
53.15
34.31
—
—

O-Max
5.97
3.15
76.47
78.47
89.85
50.78
—
—

React
7.21
1.99
40.22
42.36
56.26
29.61
—
—

She
12.08
4.48
45.89
45.81
38.13
29.28
—
—

NNguide
63.18
41.88
77.65
76.79
99.97
71.89
—
—

T-Scal
6.73
2.91
52.58
48.98
53.36
32.91
—
—

ViM
2.17
1.29
51.99
48.15
36.13
27.95
—
—

Neco
3.66
1.57
41.10
41.93
51.32
27.92
—
—

rMD
3.42
2.59
52.51
52.67
58.07
33.85
33.23
30.52

rMD++
3.18
2.66
54.09
53.67
61.31
34.98
35.26
32.89

MD
2.35
1.66
69.46
54.45
41.17
33.82
81.77
78.65

MD++
2.16
1.24
46.15
38.36
34.64
24.51
18.59
17.55

Table 15. AUROC for CIFAR100, Green indicates that the normalized method is better than its unnormalized counterpart, bold indicates
the best method, and underlined indicates the second best method. Maha++ is clearly the best method. Only for the RNxt29-32 She is
slightly better. Maha++ improves in all cases over the previously best methods ViM, Maha and KNN. We highlight the best AUC achieved
by Maha++ for the ViT-B32-21k in blue.
Model
SwinV2-S-1k
Deit3-S-21k
ConvN-T-21k
ViT-B32-21k
ViT-S16-21k
RN18
RN34
RNxt29-32
Average
RN34-SC
RN50-SC

Ash
48.67
49.99
63.80
59.23
65.78
74.20
65.82
79.46
63.37
—
—

Dice
63.60
44.78
53.50
88.31
84.35
79.77
78.86
82.01
71.90
—
—

Ebo
84.72
85.69
77.76
90.28
89.85
80.31
79.88
78.58
83.38
—
—

KlM
82.52
81.59
80.48
89.13
84.23
74.11
75.63
70.79
79.81
—
—

KNN
90.06
88.06
86.60
94.87
93.97
81.22
81.51
80.89
87.15
83.76
82.41

ML
85.20
86.18
78.51
89.99
89.44
80.31
79.76
78.47
83.48
—
—

MSP
85.68
86.21
79.09
85.36
83.87
79.70
79.13
78.37
82.18
—
—

O-Max
85.82
84.52
82.60
88.00
88.09
68.22
73.14
66.11
79.56
—
—

React
87.53
88.88
80.17
88.59
88.45
80.27
80.48
78.36
84.09
—
—

She
89.66
87.47
82.94
94.10
92.32
79.18
77.15
82.59
85.68
—
—

NNguide
71.36
55.19
62.98
87.17
80.08
81.06
74.13
73.21
73.15
—
—

T-Scal
85.93
86.43
79.22
86.73
85.38
80.02
79.56
78.22
82.69
—
—

ViM
91.34
90.41
87.67
94.62
95.91
78.50
82.13
75.33
86.99
—
—

Neco
90.38
89.86
81.31
90.70
90.80
80.66
80.61
79.68
85.50
—
—

rMD
89.77
87.44
85.00
92.37
93.09
81.27
81.22
76.87
85.88
76.80
77.90

rMD++
89.30
87.85
84.89
92.82
93.32
80.91
80.94
77.06
85.89
80.03
79.67

MD
90.29
88.30
87.95
95.59
95.63
78.46
82.03
76.18
86.80
53.30
59.01

MD++
92.99
90.54
89.55
96.84
96.81
81.71
82.16
82.48
89.14
84.83
82.44

Table 16. FPR for CIFAR100, Green indicates that the normalized method is better than its unnormalized counterpart, bold indicates the
best method, and underlined indicates the second best method. Maha++ is improving in all cases over Maha and is on average the best
method. We highlight the best FPR achieved by Maha++ for the ViT-S16-21k in blue.
Model
SwinV2-S-1k
Deit3-S-21k
ConvN-T-21k
ViT-B32-21k
ViT-S16-21k
RN18
RN34
RNxt29-32
Average
RN34-SC
RN50-SC

Ash
92.66
94.47
92.11
93.98
80.45
78.98
78.27
72.59
85.44
—
—

Dice
75.98
96.34
89.10
46.59
56.38
80.53
78.31
67.03
73.78
—
—

Ebo
40.95
41.61
57.67
30.51
36.06
80.19
75.19
82.22
55.55
—
—

KlM
49.65
47.86
65.50
43.24
50.09
78.85
78.08
87.56
62.60
—
—

KNN
36.27
36.81
51.16
26.49
31.91
76.61
74.44
73.17
50.86
66.87
66.69

ML
40.96
42.37
57.44
31.28
37.63
79.87
75.33
82.30
55.90
—
—

MSP
47.28
48.92
60.60
48.02
52.17
80.59
76.93
82.31
62.10
—
—

O-Max
67.04
66.00
66.86
53.68
57.38
97.36
94.07
96.32
74.84
—
—

React
39.54
40.46
58.23
32.53
36.48
80.18
74.51
81.87
55.47
—
—

27

She
39.64
40.93
53.76
33.25
38.89
80.46
78.76
69.42
54.39
—
—

NNguide
80.29
96.35
91.25
64.86
77.85
68.16
75.07
81.89
79.47
—
—

T-Scal
45.58
47.15
60.04
40.74
46.68
80.25
76.20
82.60
59.90
—
—

ViM
34.02
39.99
51.18
27.14
24.90
79.61
77.17
76.40
51.30
—
—

Neco
33.59
37.12
53.92
28.61
33.24
79.89
74.25
80.54
52.65
—
—

rMD
41.40
41.02
62.79
33.80
34.10
76.14
75.82
86.58
56.46
90.02
83.53

rMD++
47.14
41.36
61.66
31.03
32.83
77.49
76.22
84.39
56.51
74.37
78.15

MD
40.10
41.99
52.48
26.28
25.51
79.48
76.63
77.67
52.52
93.76
82.38

MD++
26.01
31.72
42.69
18.94
18.58
72.92
74.51
67.71
44.13
63.51
67.95

## Page 28

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 17. Normalization improves robustness against noise distributions. We report the number of failed unit tests (noise distributions
with FPR values ≥ 10%) from (Bitterwolf et al., 2023). Normalization improves the brittleness of Mahalanobis-based detectors.

model

Maha

Maha++

ConvNeXt-B
ConvNeXt-B-In21k
ConvNeXtV2-B
ConvNeXtV2-B-In21k
ConvNeXtV2-L
ConvNeXtV2-L-In21k
ConvNeXtV2-T
ConvNeXtV2-T-In21k
DeiT3-B16
DeiT3-B16-In21k
DeiT3-L16
DeiT3-L16-In21k
DeiT3-S16
DeiT3-S16-In21k
EVA02-B14-In21k
EVA02-L14-M38m-In21k
EVA02-S14
EVA02-T14
Mixer-B16-In21k
ResNet101
ResNet152
ResNet50
ResNet50-supcon
SwinV2-B-In21k
SwinV2-B
SwinV2-S
EffNetV2-L
EffNetV2-M
EffNetV2-S
ViT-B16-224-In21k-augreg2
ViT-B16-224-augreg
ViT-B16-224-In21k-orig
ViT-B16-224-In21k-miil
ViT-B16-CLIP-L2b-In12k
ViT-H14-CLIP-L2b-In12k
ViT-L14-CLIP-L2b-In12k
ViT-L16-224-In21k-orig
ViT-S16-224-augreg
ViT-so400M-SigLip

16
4
14
5
13
2
17
6
14
6
8
1
15
17
3
0
8
11
17
0
0
0
17
10
12
15
13
13
11
16
11
2
17
14
4
7
5
2
8

15
0
6
0
4
0
9
0
15
3
8
0
10
11
0
0
0
0
10
1
0
1
0
0
6
4
7
4
3
7
4
0
0
0
0
0
0
1
0

28

## Page 29

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 18. Comparison to SSD+. SSD+ consists of a) training with contrastive loss (implicitly normalizing the features), b) estimating
cluster means in the normalized feature space via k-means, c) centering the train features with the closest class mean and estimating a
shared covariance matrix, and d) using the Mahalanobis distance at inference time for OOD detection. SSD+ is therefore not readily
applicable as post-hoc OOD detection method. To highlight the benefits of post-hoc methods, we report the performance of SSD+ with a
ResNet-50, which was trained for 700 epochs with supervised contrastive loss, and compare it to a ConvNext model and an EVA model
with varied pretraining schemes. The latter models outperform SSD+ clearly, underlining the importance of post-hoc methods for OOD
detection.

Model
SSD+ w. 100 clusters
SSD+ w. 500 clusters
SSD+ w. 1000 clusters
CnvNxtV2-L + Maha++
EVA02-L14 + Maha++

29

FPR (%)
66.0
65.7
67.8
18.4
18.6

## Page 30

Mahalanobis++: Improving OOD Detection via Feature Normalization

F. Models

30

## Page 31

Mahalanobis++: Improving OOD Detection via Feature Normalization

Table 19. Imagenet model checkpoints.

Modelname

Checkpoint

source

ViT-B16-In21k-augreg
ViT-L16-In21k-augreg
ViT-T16-In21k-augreg
ViT-S16-In21k-augreg
ViT-B16-augreg
ViT-S16-augreg
ViT-so400M-SigLip
ViT-H14-CLIP-L2b-In12k
ViT-L14-CLIP-L2b-In12k
ViT-B16-In21k-orig
ViT-L16-In21k-orig
ViT-B16-In21k-miil
ViT-B16-In21k-augreg2
ViT-B16-CLIP-L2b-In12k
EVA02-L14-M38m-In21k
EVA02-B14-In21k
EVA02-S14
EVA02-T14
DeiT3-B16
DeiT3-B16-In21k
DeiT3-L16-In21k
DeiT3-B16-In21k
DeiT3-L16
DeiT3-B16
DeiT3-S16-In21k
DeiT3-S16
SwinV2-S
SwinV2-B
SwinV2-L-In21k
SwinV2-B-In21k
ResNet50
ResNet101
ResNet152
ResNet50-supcon
ConvNeXt-B
ConvNeXt-B-In21k
ConvNeXtV2-L-In21k
ConvNeXtV2-B-In21k
ConvNeXtV2-T-In21k
ConvNeXtV2-T
ConvNeXtV2-B
ConvNeXtV2-L
Mixer-B16-In21k
EffNetV2-M
EffNetV2-S
EffNetV2-L

vit base patch16 224.augreg in21k ft in1k
timm / huggingface
timm / huggingface
vit large patch16 224.augreg in21k ft in1k
vit tiny patch16 224.augreg in21k ft in1k
timm / huggingface
vit small patch16 224.augreg in21k ft in1k
timm / huggingface
vit base patch16 224.augreg in1k
timm / huggingface
vit small patch16 224.augreg in1k
timm / huggingface
timm / huggingface
vit so400m patch14 siglip 378.webli ft in1k
vit huge patch14 clip 336.laion2b ft in12k in1k
timm / huggingface
timm / huggingface
vit large patch14 clip 336.laion2b ft in12k in1k
vit base patch16 224.orig in21k ft in1k
timm / huggingface
timm / huggingface
vit large patch32 384.orig in21k ft in1k
vit base patch16 224 miil.in21k ft in1k
timm / huggingface
vit base patch16 224.augreg2 in21k ft in1k
timm / huggingface
vit base patch16 clip 224.laion2b ft in12k in1k
timm / huggingface
eva02 large patch14 448.mim m38m ft in22k in1k
timm / huggingface
eva02 base patch14 448.mim in22k ft in22k in1k
timm / huggingface
timm / huggingface
eva02 small patch14 336.mim in22k ft in1k
eva02 tiny patch14 336.mim in22k ft in1k
timm / huggingface
deit3 base patch16 224
timm / huggingface
deit3 base patch16 224 in21ft1k
timm / huggingface
deit3 large patch16 384.fb in22k ft in1k
timm / huggingface
timm / huggingface
deit3 base patch16 384.fb in22k ft in1k
deit3 large patch16 384.fb in1k
timm / huggingface
deit3 base patch16 384.fb in1k
timm / huggingface
deit3 small patch16 384.fb in22k ft in1k
timm / huggingface
deit3 small patch16 384.fb in1k
timm / huggingface
swinv2 small window16 256.ms in1k
timm / huggingface
timm / huggingface
swinv2 base window16 256.ms in1k
swinv2 large window12to24 192to384.ms in22k ft in1k
timm / huggingface
swinv2 base window12to24 192to384.ms in22k ft in1k
timm / huggingface
resnet50.tv2 in1k
timm / huggingface
resnet101.tv2 in1k
timm / huggingface
resnet152.tv2 in1k
timm / huggingface
rn50supcon
github.com/roomo7time/nnguide/
timm / huggingface
convnext base.fb in1k
convnext base.fb in22k ft in1k
timm / huggingface
convnextv2 large.fcmae ft in22k in1k 384
timm / huggingface
timm / huggingface
convnextv2 base.fcmae ft in22k in1k 384
convnextv2 tiny.fcmae ft in22k in1k 384
timm / huggingface
convnextv2 tiny.fcmae ft in1k
timm / huggingface
convnextv2 base.fcmae ft in1k
timm / huggingface
convnextv2 large.fcmae ft in1k
timm / huggingface
mixer b16 224.goog in21k ft in1k
timm / huggingface
timm / huggingface
tf efficientnetv2 m.in1k
tf efficientnetv2 s.in1k
timm / huggingface
tf efficientnetv2 l.in1k
timm / huggingface

31

## Page 32

Mahalanobis++: Improving OOD Detection via Feature Normalization
Table 20. Cifar model checkpoints.

SwinV2-S-1k
Deit3-S-21k
ConvN-T-21k
ViT-B32-21k
ViT-S16-21k
RN18
RN34
RN34-SC
RN50-SC
RNxt29-32

ft from timm model
ft from timm model
ft from timm model
https://github.com/google-research/big vision
https://github.com/google-research/big vision
https://huggingface.co/edadaltocg/
https://huggingface.co/edadaltocg/
https://huggingface.co/edadaltocg/
https://huggingface.co/edadaltocg/
self trained

G. Methods
We describe OOD detection methods evaluated in our work. Let a neural network nθ (x) = g(ϕ(x)) decompose into a
feature extractor ϕ and linear layer g(ϕi ) = WT ϕi + b. For input x, ϕ(x) denotes the feature embedding, and g(ϕ(x))
produces logits o, which can be transformed to a probability vector p via the softmax function.
MSP (Hendrycks & Gimpel, 2017): Classifer confidence, i.e. max-softmax-probability
s = max(pc )
c

Max-Logit (ML or MLS) (Hendrycks et al., 2022): Max-Logit returns the largest entry of the logit-vector o, i.e.
s = max(oc )
c

Energy (E) (Liu et al., 2020): Energy or log-sum-exp of logits:
s = log

C
X

exp (oc )

c

KL-Matching (KLM) (Hendrycks et al., 2022): KL-Matching computes the average probability vector dc for each of the C
classes. For a test input, the KL-distances of all dc vectors to its probability vector p are computed, and the OOD-score is
the negative of the smallest of those distances:
s = − min KL[p||dc ]
c

In the original paper by (Hendrycks et al., 2022), the average for dc is computed over an additional validation set. Since
none of the other methods leverages extra data and we are interested in fair comparison, we deploy KL-Matching like in
(Wang et al., 2022; Yang et al., 2022), where the average is computed over the train set.
KNN (KNN) (Sun et al., 2022): Computes the k-nearest neighbour in the normalized feature-space: The feature vector
of a test input is normalized to z = ϕ/||ϕ||2 and the pairwise distances ri (z) = ||z − zi ||2 to the normalized features
Z = {z1 , ..., zN } of all samples of the training set are computed. The distances ri (z) are then sorted according to their
magnitude and the K th smallest distance, denoted rK (z) is used as negative OOD-score:
s = −rK (z)
Like suggested in (Sun et al., 2022), we use K = 1000.
ReAct (E+R) (Sun et al., 2021): The authors propose to perform a truncation of the feature vector, ϕ̄ = min(ϕ, r), where
the min operation is to be understood element-wise and r is the truncation threshold. The truncated features can then be
32

## Page 33

Mahalanobis++: Improving OOD Detection via Feature Normalization

converted to so-called rectified logits via ō = g(ϕ̄) = WT ϕ̄ + b. While the rectified logits can now be used with a variety
of existing detection methods, we follow (Sun et al., 2021) and use the rectified Energy as OOD-score:
s = log

C
X

exp (ōc )

c

As suggested in (Wang et al., 2022), we set the threshold r such that 1% of the activations from the train set would be
truncated.
Virtual Logit Matching (ViM) (Wang et al., 2022): The idea behind ViM is that meaningful features are thought to lie in a
low-dimensional manifold, called the principal space P , whereas features from OOD-samples should also lie in P ⊥ , the
space orthogonal to P . P is the D-dimensional subspace spanned by the eigenvectors with the largest D eigenvalues of
the matrix FT F, where F is the matrix of all train features offsetted by u = −(WT )+ b (+ denotes the Moore-Penrose
⊥
inverse). A sample with feature vector ϕ is then also offset to h̃ = ϕ − u and can be decomposed into h̃ = h̃P + h̃P , and
⊥
⊥
h̃P is referred to as the Residual of ϕ. ViM leverages the Residual and converts it to a virtual logit o0 = α||h̃P ||2 , where
PN
c
i=1 maxc oi
α = PN
P⊥
i=1 ||ϕi ||2
is designed to match the scale of the virtual logit to the scale of the real train logits. The virtual logit is then appended to
the original logits of the test sample, i.e. to o, and a new probability vector is computed via the softmax function. The
probability corresponding to the virtual logit is then the final OOD-score:
exp (o0 )

s = − PC

c=1 exp (oc ) + exp (o0 )

Like suggested in (Wang et al., 2022), we use D = 1000 if the dimensionality of the feature space d is d ≥ 2048, D = 512
if 2048 ≥ d ≥ 768, and D = d/2 rounded to integers otherwise.
Cosine (Cos) (Techapanurak et al., 2020; Galil et al., 2023): This method computes the maximum cosine-similarity between
the features of a test-sample and embedding vectors ũc (sometimes also called concept-vector):
s = max
c

ũTc ϕ
||ũTc ||2 ||ϕ||2

(10)

Ash (Ash) (Djurisic et al., 2023): Ash applies activation shaping at inference time by pruning acitvations below a certain
threshold, and then binarizing (Ash-b) or scaling (Ash-s) the remaining activations, which are then processed as usually in
the network. As suggested by the authors, we apply ash to the pre-logit feature activations.
Softmax-scaled-Cosine (SSC) (Tack et al., 2020): Normalize the rows of the weight matrix wi and the features, and
compute the cosine between the two:
wi · ϕ
cos θi ≡
∥wi ∥∥ϕ∥
Then scale by a scalar t and apply the softmax, to finally use the max-softmax as OOD score:
s = max (softmax(t ∗ θ)i )
i

In Tack et al. (2020) the scalr s is learned, for our post-hoc setup we set s = 1.
NeCo (Nec) (Ammar et al., 2024): Compute the covariance matrix of the feature space, and project to the d eigenvectors
with largest eigenvalues with the corresponding projection matrix P . The difference in norm of the projected features and
the original features is then scaled with the max-logit and serves as OOD score.
s
∥P ϕ(x)∥
ϕ(x)⊤ P P ⊤ ϕ(x)
∗ max oi =
∗ max oi
s=
i
i
∥ϕ(x)∥
ϕ(x)⊤ ϕ(x)
Like suggested by the authors, we standardize data and select d such that 90% of the train variance are explained.
33

## Page 34

Mahalanobis++: Improving OOD Detection via Feature Normalization

Gaussian Mixture Model (GMM or GMN) (Mukhoti et al., 2023): Estimate a Gaussian mixture model on the train features
ϕ(x), and use the log-probabilities as OOD score. We use GMN for Gaussian mixture model with normalized features, and
GMM for Gaussian mixture model with regular features.
NNguide (NNG) (Park et al., 2023a): ”Guide” the energy score by a nearest-neighbor score:
s = sEnergy ∗ sKN N
where sKN N is a KNN score in the normalized feature space, estimated on a subset of the train features. Like suggested by
the authors, we use 1% of the train features and K = 10 neighbors for ImageNet experiments. We also tried K = 1000,
as increasing K showed promising results in an ablation by the authors (Figure 4 in the paper), but found that it performs
worse on average than K = 10.
Relative Mahalanobis distance (rMaha) (Ren et al., 2021): A modification of the Mahalanobis distance method, thought
to improve near-OOD detection, is to additionally fit a global Gaussian distribution to the train set without taking classinformation into account:
1 X
1 X
µ̂global =
ϕi ,
Σ̂global =
(ϕi − µ̂global )(ϕi − µ̂global )T
N i
N i
The OOD-score is then defined as the difference between the original Mahalanobis distance and the Mahalanobis distance
w.r.t. the global Gaussian distribution:


T
s = − min (ϕ − µ̂c )Σ̂−1 (ϕ − µ̂c )T − (ϕ − µ̂global )Σ̂−1
global (ϕ − µ̂global )
c

34
