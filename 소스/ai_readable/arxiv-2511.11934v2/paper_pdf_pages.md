# A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts - page-anchored PDF text

- Source ID: `arxiv-2511.11934v2`
- arXiv ID: `2511.11934v2`
- Original PDF: `소스/A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts.pdf`
- PDF pages: 29
- Extracted with: WSL poppler `pdftotext -f N -l N -layout` on 2026-05-13T17:01:27+09:00
- Evidence note: use this file for page-anchored claims; verify equations/tables against the original PDF when exact layout matters.

## Page 1

A Systematic Analysis of Out-of-Distribution Detection Under Representation
                                                                and Training Paradigm Shifts


                                                                        Claudio César Claros-Olivares 1 Austin J. Brockmeier 1


                                                                 Abstract                                    1. Introduction
                                                                                                             In recent years, there has been a significant increase in
                                             We present the largest systematic comparison to                 approaches dealing with out-of-distribution (OOD) detec-
arXiv:2511.11934v2 [cs.LG] 1 Feb 2026




                                             date of out-of-distribution (OOD) detection meth-               tion (Ammar et al., 2023; Bibas et al., 2021; Hendrycks &
                                             ods using AURC and AUGRC as primary met-                        Gimpel, 2016; Hendrycks et al., 2019; Huang et al., 2021;
                                             rics. Our comparison explores different regimes                 Lee et al., 2018; Liu & Qin, 2023; Liu et al., 2020; 2023;
                                             of distribution shift (stratified by CLIP embed-                Ngoc-Hieu et al., 2023; Park et al., 2023; Wang et al., 2022)
                                             dings of the out-of-distribution image datasets)                given their relevance in the reliable deployment of deep
                                             with varying numbers of classes and uses a                      learning models, particularly in safety-critical applications.
                                             representation-centric view of OOD detection,                   A significant vulnerability of deep neural networks (DNNs)
                                             including neural collapse metrics, for subse-                   is their tendency to produce silent failures (Jaeger et al.,
                                             quent analysis. Together the empirical results                  2022; Traub et al., 2024), which are conceptualized as in-
                                             and representation analysis provides novel in-                  correct predictions with high confidence that are hard to
                                             sights and statistically grounded guidance for                  detect. A reliable DNN classifier should not only accu-
                                             method selection under distribution shift. Experi-              rately classify known in-distribution (ID) samples, but also
                                             ments cover two representation paradigms: CNNs                  effectively flag OOD inputs as unknown. However, many
                                             trained from scratch and a fine-tuned Vision                    OOD detection methods have inconsistent behavior across
                                             Transformer (ViT), evaluated on CIFAR-10/100,                   datasets. For instance, the widely used maximum softmax
                                             SuperCIFAR-100, and TinyImageNet. Using                         response (MSR) (Hendrycks & Gimpel, 2016), a strong
                                             a multiple-comparison–controlled, rank-based                    baseline for small-scale OOD detection, does not scale well
                                             pipeline (Friedman test with Conover–Holm post-                 to challenging conditions presented by large datasets such
                                             hoc) and Bron–Kerbosch cliques, we find that the                as ImageNet-1k (Deng et al., 2009). Hendrycks et al. (2019)
                                             learned feature space largely determines OOD                    hypothesizes that in datasets with many visually similar
                                             efficacy. For both CNNs and ViTs, probabilis-                   classes, a classifier might produce low softmax confidence
                                             tic scores (e.g., MSR, GEN) dominate misclas-                   for a legitimate ID image not due to unfamiliarity, but be-
                                             sification (ID) detection. Under stronger shifts,               cause the precise class is difficult to determine, dispersing
                                             geometry-aware scores (e.g., NNGuide, fDBD,                     the probability mass. This issue makes MSR problematic
                                             CTM) prevail on CNNs, whereas on ViTs Grad-                     for large-scale detection. In contrast, the maximum logit
                                             Norm and KPCA Reconstruction Error remain                       score (MLS) (Hendrycks et al., 2019), which uses the neg-
                                             consistently competitive. We further show a                     ative of the maximum unnormalized logit, was proposed
                                             class-count–dependent trade-off for Monte-Carlo                 as a better baseline for large-scale OOD detection. Exper-
                                             Dropout (MCD) and that a simple PCA pro-                        iments showed that MLS significantly outperforms MSR
                                             jection improves several detectors. The neural-                 in large-scale datasets, with improvements over 10% AU-
                                             collapse–based geometric analysis explains when                 ROC in some cases, whereas the difference was minor on
                                             prototype and boundary-based scores become op-                  small-scale CIFAR-10. This highlights the inconsistencies
                                             timal under strong shifts.                                      in performance as the number of training classes increases.
                                                                                                             Furthermore, some methods that performed well in small-
                                            1
                                                                                                             scale settings, such as the Mahalanobis distance (Lee et al.,
                                              Department of Electrical & Computer Engineering, Univer-
                                                                                                             2018), encountered numerical problems when scaled to
                                        sity of Delaware, Delaware, USA. Correspondence to: Claudio
                                        César Claros-Olivares <cesar@udel.edu>, Austin J. Brockmeier        1000 classes (Hendrycks et al., 2019). Mahalanobis dis-
                                        <ajb@udel.edu>.                                                      tance utilizes a mixture of class-conditional Gaussians on
                                                                                                             features to identify OOD samples, showing effectiveness
                                        Preprint. February 3, 2026.

                                                                                                         1

## Page 2

A Systematic Analysis of Out-of-Distribution Detection

on datasets like CIFAR-10. However, its scaling issues              same. Also, the authors state that despite ImageNet’s higher
underscore that methods that rely on specific feature space         complexity, ImageNet-based benchmarks sometimes yield
properties derived from training data can become unstable or        higher OOD detection performance than CIFAR-10/100. In
computationally challenging with a large number of classes.         a follow-up work, Zhang et al. (2023) extended OpenOOD’s
In addition, the observation that methods that are effective        codebase, implementing more OOD detection methods and
in natural images might differ from those for other data            extending its evaluation capabilities to larger datasets and
types, such as medical images (Gutbrod et al., 2025), where         recent foundation models. Although both studies experi-
feature-based methods might excel, also subtly suggests that        ment with multiple datasets with an increasing number of
data scale and complexity, often correlated with the number         classes, they do not explore the impact that the number of
of classes, influence the effectiveness of the method.              classes in the training sets has on the OOD detection per-
                                                                    formance. Additionally, this work does not consider some
By contrasting multiple OOD detection methods and per-
                                                                    OOD detection methods that exploit geometric properties
formance metrics obtained using CIFAR-10, CIFAR-100
                                                                    in the feature space such as NNGuide (Park et al., 2023),
(including superclasses) and TinyImagenet as ID datasets
                                                                    fDBD (Liu & Qin, 2023) or CTM (Ngoc-Hieu et al., 2023).
evaluated in near, mid and far OOD scenarios, we high-
light the varying effectiveness of methods across different         In a more theoretically grounded effort, Jaeger et al. (2022)
scales and the need for approaches specifically designed            identifies that conceptually similar problems such as mis-
or validated depending on the number of classes in the              classification detection, OOD detection, selective classifi-
training set. The four main contributions of our work are           cation, and uncertainty quantification have heterogeneous
1) a representation- and training-aware experimental de-            task definitions and incompatible benchmarks. Accordingly,
sign that factorially varies the backbone (CNN vs ViT),             this work advocates for a unified evaluation framework cen-
the training paradigm, and the confidence scoring func-             tered on failure detection using confidence scoring func-
tion (CSF); 2) a statistically rigorous comparison using            tions (CSFs), with the Area Under the Risk-Coverage Curve
AURC/AUGRC, non-parametric tests, and clique-based                  (AURC) (Geifman et al., 2018) providing a holistic met-
equivalence classes; 3) a CLIP-based stratification of OOD          ric. Additionally, Jaeger et al. (2022) provides a codebase,
datasets into near/mid/far semantic regimes; and 4) a neural-       coined FD-Shifts, which benchmarks a suite of CSFs across
collapse-based theoretical analysis that explains when and          multiple datasets and diverse shifts (covariate, semantic),
why probabilistic vs geometry-aware vs gradient-based de-           reporting that simple baselines like MSR often outperform
tectors dominate.1                                                  complex methods. Traub et al. (2024), however, argued that
                                                                    the reliance of the AURC on selective risk at fixed thresh-
2. Related Work                                                     olds does not adequately reflect holistic system performance.
                                                                    In response, they introduce a novel metric, the Area Under
The inconsistencies in OOD evaluation have spurred mul-             the Generalized Risk-Coverage Curve (AUGRC), which
tiple benchmarking efforts to standardize procedures and            aggregates misclassification risk across all thresholds and
report more comprehensive results. However, in general,             better captures the joint probability of failure and its ac-
the impact of the number of classes in the training set has         ceptance, fixing the limitations of the AURC. Both studies
been overlooked. Yang et al. (2022), for example, presents          evaluate CIFAR-10 and CIFAR-100, considering multiple
an extensive and unified benchmark framework designed               distribution shifts, but do not identify any dependence of
to evaluate and compare out-of-distribution (OOD) detec-            the OOD detection performance on the number of classes
tion methods across neighboring fields, including anomaly           in the training sets.
detection (AD), open set recognition (OSR), and model un-
                                                                    In a more applied setting, Bungert et al. (2023) explore
certainty estimation. Furthermore, the authors implement
                                                                    silent failures in the classification of medical images. The
a codebase named OpenOOD, which integrates multiple
                                                                    authors benchmark various CSFs under realistic distribution
OOD detection methods and evaluates them in a generalized
                                                                    shifts such as corruption (e.g. noise), acquisition (e.g. mul-
OOD detection framework using 9 designed benchmarks
                                                                    tiple data sources), and manifestation (e.g. unseen forms
and 4 OOD detection datasets that include MNIST, CIFAR-
                                                                    of a pathology) shifts. This study builds four biomedical
10, CIFAR-100, and ImageNet. Each benchmark features
                                                                    imaging datasets, combining in some cases multiple sources,
near-OOD (semantic shifts) and far-OOD (domain shifts)
                                                                    including dermoscopy, chest radiograph, fluorescence mi-
scenarios and reports AUROC, FPR@95, and AUPR as
                                                                    croscopy, and CT of lung nodules. This study highlights that
metrics. Specifically, they show that the best performing
                                                                    none of the advanced CSFs obtained from different selective
OOD detection methods between CIFAR-10 and CIFAR-
                                                                    classification approaches such as DeepGamblers (Liu et al.,
100, considering near- and far-OOD scenarios, are not the
                                                                    2019), ConfidNet (Corbière et al., 2019; Corbiere et al.,
  1
    Code is available at https://anonymous.4open.                   2021), or Monte Carlo Dropout (Gal & Ghahramani, 2016)
science/r/ood_systematic-C990/                                      (MCD) variants (CSFs obtained from predictions generated

                                                                2

## Page 3

A Systematic Analysis of Out-of-Distribution Detection

by MCD activated at inference time) consistently outper-                 the classifier outputs a logit vector via g(h) = W L h + bL ,
form the MSR baseline. Although MCD-MSR performs                         where W L = [w1L , . . . , wCL ] ∈ RC×D and bL =
best on average, results vary significantly depending on the             [b1L , . . . , bCL ] ∈ RC . Let w and W represent the clas-
data set and the type of shift, revealing trade-offs. In other           sifier weights, where we are intentionally omitting the layer-
words, this work emphasizes that a CSF that performs well                specific index L for simplicity in the current context. The
under one domain shift may perform poorly under another.                 predicted class is ŷ = m(x) = arg maxc [f (x)]c , and
This highlights the difficulty in developing CSFs that gener-            the softmax-based class probabilities are given by p(x) =
                                                                                             1
alize reliably in diverse biomedical conditions. Moreover,               p = P exp(f           (x)j ) [exp(f (x)1 ), . . . , exp(f (x)C )] =
                                                                                     j
the choice of AURC as the main in this study may have                    softmax(f (x)) = softmax(g(h)).
hidden some trends in the performance of the CSFs.
                                                                         We define the matrix of penultimate-layer activations
Also in the medical domain, Gutbrod et al. (2025) estab-                 for a dataset D of size N as H = [h1 , . . . , hN ] ∈
                                                                                                                               ⊤
lished OpenMIBOOD, a medical benchmark spanning 14                       R  N ×D
                                                                                 , and the corresponding matrix for class c as
datasets. They evaluated 24 post-hoc OOD detection meth-                                                            ⊤
                                                                         H c = [hi,c : xi ∈ Dc , i = 1, . . . , Nc ] ∈PRNc ×D . The
ods, including classification-based, feature-based, and hy-
                                                                         global mean of the features is µ = N1 1≤i≤N hi =
brid approaches, using standard metrics such as AUROC,                    1
                                                                             P
FPR@95, and the harmonic mean of AUPR-IN and AUPR-                       N P1≤i≤N h(xi ), and the   mean of the c-th class is µc =
                                                                          1                   1
                                                                                                  P
OUT. The results reveal that methods effective on natu-                  Nc            h
                                                                                1≤i≤Nc i,c = Nc     1≤i≤Nc h(xi ). These quantities
                                                                                xi ∈Dc                   xi ∈Dc
ral image benchmarks often fail in medical domains, with                 also permit a geometric analysis of the last-layer representa-
feature-based approaches outperforming probabilistic-based               tion via neural collapse metrics; see Appendix D.
approaches across most scenarios. The authors highlight
that covariate shift detection is more tractable than semantic           3.2. Projection Filtering
shift detection in medical imaging. Furthermore, methods
such as Residual and ViM (Wang et al., 2022), while suc-                 Leveraging the observation that image data lies on low-
cessful on natural image benchmarks, exhibit inconsistent                dimensional manifolds (Pope et al., 2021)), we use Princi-
performance in medical data sets. A key insight is that                  pal Component Analysis (PCA) to filter noise and capture
medical imaging tasks often involve lower data variance                  intrinsic dimensionality. Principal Component Analysis
and fewer classes, favoring methods that exploit deep fea-               (PCA) offers a principled way to exploit this low intrin-
ture representations over those that rely on probabilistic               sic dimensionality by filtering out directions in the data
predictions.                                                             that are unlikely to carry meaningful signal, assuming that
                                                                         the true structure of the data is concentrated along a few
                                                                         high-variance directions, whereas noise is distributed more
3. Methods                                                               evenly across all dimensions. More specifically, the set of
3.1. Definitions and Notations                                           penultimate-layer activations H are centered by subtracting
                                                                         the mean of each feature: H̃ = H − 1N µ⊤ . The empirical
We consider a supervised classification problem where                                                      ⊤
                                                                         covariance matrix Σ = N1 H̃ H̃ is then decomposed as
each input sample is denoted by x ∈ X , and its corre-
sponding label is y ∈ Y = {1, . . . , C}, representing one               Σ = V ΛV ⊤ , where V contains the eigenvectors and Λ is
of C possible classes. The training dataset is defined as                the diagonal matrix of eigenvalues λ1 ≥ λ2 ≥ · · · ≥ λD .
D = {(xi , yi )}N                       test
                                             = {(xtest test N test       To denoise the data, only the top k principal components
                i=1 , the test set as D           i , yi )}i=1 ,
and an out-of-distribution (OOD) dataset as DOOD =                       P = [v 1 , . . . , v k ] ∈ RD×k , which span a subspace that
{(xOOD  , 0)}N
               OOD                                                       captures most of the variance. The number of components k
    i        i=1 , which contains samples not belonging
to any of the known classes. For a given class c ∈ Y, we                 is decided so that certain percentage of variance is preserved
define the subset of data corresponding to that class within             at least. PCA can be applied to both the set of training acti-
a set D as Dc = {(x, y) ∈ D | y = c}, with cardinality                   vations H centered by the global mean µ, from which P
Nc = |Dc |.                                                              is obtained, and the set of training activations for a given
                                                                         class c, H c , centered by its class mean µc , from which P c
We denote the classifier as a neural network function                    is computed.
f : X → RC parameterized by weights and biases W =
{(W 1 , b1 ), . . . , (W L , bL )}. This model is decomposed             For a new sample x∗ , the corresponding penultimate-
into an encoder h : X → RD , defined by layers up to L − 1,              layer activation h∗ is projected onto this subspace: z ∗ =
and a linear classifier head g : RD → RC parametrized                    P ⊤ (h∗ − µ) = P ⊤ h̃∗ ∈ Rk , and then reconstructed as
by the weights in the L-th layer. The complete model is                  ĥ∗ = P z ∗ + µ. This reconstruction preserves the domi-
composed as f (x) = (g ◦ h)(x). For a given input x, the                 nant low-rank structure while discarding low-variance di-
penultimate-layer representation is h = h(x) ∈ RD , and                  rections. Given that we have global and class subspaces, P
                                                                         and P c respectively, we evaluate six projections: Global

                                                                     3

## Page 4

A Systematic Analysis of Out-of-Distribution Detection

projection: ĥ = P P ⊤ (h − µ) + µ; Class projection:                      Table 1. OOD dataset clustering. For each source dataset, the
                                                                           corresponding OOD datasets are categorized in near, mid and far
ĥc = P c P c ⊤ (h − µc ) + µc ; Class predicted projec-                   datasets based on the CLIP-derived distances.
tion: ĥŷ ; Class projected logit: ĝ class = [ĝ1 , . . . , ĝC ],
                                                                             Source                 Near                 Mid            Far
where ĝc = w⊤   cL ĥc + bcL ; Global projected probabili-                  CIFAR-10               CIFAR-100, TinyIm-   iSUN,          Places365,
                                                                                                    agenet               LSUN(r),       Textures
ties: p̂ = softmax(g(ĥ)); and Class projected probabilities:                                                            LSUN(c),
p̂class = softmax(ĝ class ). A summary of the combination                                                               SVHN
                                                                             SuperCIFAR-100         CIFAR-10, TinyIma-   iSUN,          Places365,
of these projections with possible OOD detection methods                                            genet                LSUN(r),       Textures
is in Appendix A. We later measure how such projections                                                                  LSUN(c),
                                                                                                                         SVHN
affect neural-collapse metrics (e.g., within-class variability               CIFAR-100              CIFAR-10, TinyIma-   iSUN,          Places365,
and prototype angularity), and relate these changes to shifts                                       genet                LSUN(r),       Textures
                                                                                                                         LSUN(c),
in detector performance in Appendix D.                                                                                   SVHN
                                                                             Tiny-Imagenet         CIFAR-10, CIFAR-      Places365,     SVHN
                                                                                                   100,        iSUN,     Textures
3.3. CLIP-based OOD Aggregation                                                                    LSUN(r), LSUN(c)

We quantify distributional proximity to an OOD image
dataset using the feature space from a CLIP model (Rad-
ford et al., 2021). Concretely, we extract L2-normalized                   c′ , ΣW = Avgi,c (hi,c − µ)(hi,c − µ)T be the within-
image embeddings for both ID and candidate OOD sets                        class covariance, ΣB = Avgc (µ̃c µ̃Tc ) be the between-class
using a fixed CLIP encoder under identical preprocessing.                  covariance, † be the Moore-Penrose pseudoinverse, and
We then compute two label-agnostic distances between the                   M = [µ̃c : c = 1, . . . , C] be the centered class-mean ma-
empirical feature distributions: Fréchet distance (FD) (Dow-              trix. Neural Collapse is classically characterized by four phe-
son & Landau, 1982; Fréchet, 1957) and Maximum Mean                       nomena in the terminal phase of training: (NC1) Variability
Discrepancy (MMD) (Gretton et al., 2006) with a poly-                      collapse, (NC2) Convergence to simplex equiangular tight
nomial kernel. Both are evaluated on CLIP embeddings                       frame (ETF), (NC3) Convergence to self-duality, and (NC4)
yielding global measures of how close the OOD distribu-                    Simplification to nearest class center (NCC). Following (Pa-
tion lies to the ID manifold; by construction, lower values                pyan et al., 2020), we operationalize proximity to NC using
indicate greater proximity. To capture class-aware prox-                   five empirical metrics: 1) Equinormness of the class means
                                                                                                             Stdc (∥µ̃c ∥2 )     Stdc (∥wc ∥2 )
imity, we complement the global measures with two class-                   and the classifier weights: Avg                   and Avg              , 2)
                                                                                                                c (∥µ̃c ∥2 )         c (∥w c ∥2 )
conditional distances. First, we represent each ID class                   Equiangularity of the class means and the classifier weights:
by an image-embedding centroid and score a sample by                       Stdc,c′ ̸=c (cosµ (c, c′ )) and Stdc,c′ ̸=c (cosw (c, c′ )), 3) Max-
its nearest-centroid angular distance in CLIP feature space.               imal Equiangularity of the class means and the classifier
Second, we form text prototypes via prompt ensembling                      weights: (cosµ (i, j) = − C−1   1
                                                                                                               ): Avgc,c′ |cosµ (c, c′ ) + C−1    1
                                                                                                                                                     |
for each ID class (e.g., a photo of a class), embed them                                                 1                            ′          1
                                                                           and (cosw (i, j) = − C−1 ): Avgc,c′ |cosw (c, c ) + C−1 |,
with the CLIP text model, and use the maximum image-text
                                                                           4) Variability Collapse: C1 Tr(ΣW Σ†B ), 5) Self-duality:
cosine similarity. For both class-conditional distances we                         Tr(W M )
compute the mean value to quantify a single metric that                    2 − 2 ∥W  ∥F ∥M ∥F . We compute these metrics for each back-
represents distance to the ID manifold. All four metrics are               bone–dataset pair under three representation regimes: (i)
oriented so lower implies closer. Finally, we cluster these                unfiltered activations, (ii) global projection, and (iii) class-
measurements using KMeans to derive near/mid/far prox-                     predicted projection. This yields a compact quantitative
imity buckets. This protocol is model-agnostic with respect                summary of how close each trained model is to an NC-like
to the downstream OOD detector and applies unchanged to                    regime, and how projection filtering changes the effective
any choice of ID label space. Table 1 shows the resulting                  geometry.
clustering and CLIP-derived distances for each dataset are
reported in Appendix B.                                                    4. Results
3.4. Neural-collapse Geometry of Last-Layer features                       4.1. Experimental Setup

We analyze the geometry of the last-layer representation                   We adopt the FD-Shifts protocol (Jaeger et al., 2022; Traub
through the lens of Neural Collapse (NC). Let µ̃c = µc − µ                 et al., 2024), which benchmarks CSFs across diverse failure
                                                  µ̃T µ̃                   sources and architectures. We consider the following CSF
be the class-centered mean, cosµ (c, c′ ) = ∥µ̃ ∥c2 ∥µ̃c′ ′ ∥2 be          methods: Maximum Softmax Response (MSR) (Hendrycks
                                                  c      c
the cosine similarity between any pair of class-centered                   & Gimpel, 2016), Generalized Entropy (GEN) (Liu et al.,
                                      wT w
means c and c′ , cosw (c, c′ ) = ∥wc ∥c2 ∥wc′ ′ ∥2 be the co-              2023) and Renyi Entropy (REN), Predictive Collision En-
                                             c
sine similarity between any pair of classifier weights c and               tropy (PCE), GradNorm (Huang et al., 2021), Guessing
                                                                           Entropy (GE), PCA Reconstruction Error (Guan et al.,

                                                                       4

## Page 5

A Systematic Analysis of Out-of-Distribution Detection

2023), Class-Typical Matching (CTM) and Class-Typical               gle winner. This clique view is faithful to the inferential
Matching with mean class features (CTMmean) (Ngoc-                  decisions and naturally handles overlaps (a method can be-
Hieu et al., 2023), Residual and Virtual Logit Match-               long to multiple near-optimal sets). Detailed procedure and
ing (ViM) (Wang et al., 2022), Maximum Logit Score                  simplified example are described in Appendix C.
(MLS) (Hendrycks et al., 2019), Nearest Neighbor Guid-
ance (NNGuide) (Park et al., 2023), Neural Collapse-based           4.3. Top Cliques
OOD detection (NeCo) (Ammar et al., 2023), Energy (Liu
et al., 2020), Kernel PCA Reconstruction Error (Fang et al.,        Figure 1 (left) depicts Conover-Holm top cliques (at
2025), fast Decision Boundary Decision (fDBD) (Liu & Qin,           α = 0.05) across evaluation regimes, with columns de-
2023), Mahalanobis Distance (Maha) (Lee et al., 2018), pre-         noting settings (e.g., cifar10→near, cifar100→ID,
dictive Normalized Maximum Likelihood (pNML) (Bibas                 tinyimagenet→far) and rows corresponding to
et al., 2021), Confidence (from ConfidNet (Corbiere et al.,         confidence-scoring functions (CSFs) when VGG-13 is the
2021), DeVries (DeVries & Taylor, 2018), and Deep Gam-              model backbone. Within each column, connected markers
blers (Liu et al., 2019) training paradigms). VGG-13 and            indicate the set of methods that are mutually indistinguish-
ViT backbone models are trained from scratch and finetuned,         able from the best; larger cliques imply broader statistical
respectively, on CIFAR-10, CIFAR-100, SuperCIFAR-100,               ties, whereas smaller cliques indicate sharper separation.
and TinyImagenet. See Appendix E for hyperparmeters.                Shaded bands highlight persistent coalitions that reappear
Monte-Carlo Dropout variants (MCD) of the scores named              as winners across multiple regimes. In the source→test (ID)
previously are also evaluated, for which we draw 50 stochas-        regime, which corresponds to misclassification detection,
tic forward passes at test time. Results are reported on ID         probability-based CSFs such as GEN and MSR dominate:
test sets and OOD data sets, which are categorized based            the softmax geometry aligns well with error, with correctly
on CLIP-derived distances. When evaluating OOD detec-               classified ID examples exhibiting low predictive entropy,
tion, we compare OOD samples against only the correctly             allowing simple confidence/energy scores to rank errors
classified in-distribution (ID) samples and discard ID mis-         effectively without class-conditional structure.
takes, so that a CSF is not rewarded for flagging ordinary          Across near shifts, E NERGY, MLS, and NNG UIDE (es-
in-distribution errors as if they were OOD. Misclassifica-          pecially the global variant) consistently appear in the top
tion detection is evaluated separately on the ID test set by        cliques. As the semantic distance from the source in-
ranking correct vs incorrect ID predictions to reflect the          creases and the number of classes grows (e.g., cifar100,
classifier’s accuracy.                                              tinyimagenet), these confidence-style scores become
                                                                    less dominant; geometry-aware CSFs such as F DBD and
4.2. Statistical Tests                                              CTM increasingly anchor the winning sets, reflecting the
                                                                    greater value of boundary proximity and class-typicality un-
We choose a Friedman test to compare CSFs in terms of
                                                                    der stronger shifts. Notably, CTM is repeatedly selected for
their paired performances across blocks defined by the OOD
                                                                    tinyimagenet (200 classes), consistent with prototype-
dataset, training paradigm, source, and metric. The Fried-
                                                                    style scoring benefiting from richer class granularity, while
man test is a nonparametric, blocked test that converts met-
                                                                    NNG UIDE (global) is among the most frequently retained
ric in each block to ranks, which is appropriate for CSF
                                                                    variants overall, indicating that aggregating neighborhood
evaluation, where distributions are skewed, heteroscedas-
                                                                    evidence globally improves OOD discrimination. Finally, as
tic, and not commensurate across blocks (the raw metric
                                                                    class count rises and shift strength increases from test→far,
scales such as AUGRC or AURC can differ markedly across
                                                                    clique sizes tend to contract, underscoring that detector per-
datasets). The null hypothesis is that the average ranks are
                                                                    formance depends sensitively on source-label cardinality
the same across methods. If the null hypothesis is rejected,
                                                                    and shift severity, with fewer methods remaining statisti-
indicating metrics have different distribution of ranks, we
                                                                    cally tied at the top in the harder regimes.
need to determine which groups of CSFs have statistically
indistinguishable performance. To achieve this, pairwise dif-       Figure 1 (right) reports Conover-Holm top cliques at α =
ferences between CSFs while controlling multiplicity over           0.05 for a ViT backbone. Consistent with the analysis
all pairs are need. Among rank-based post-hocs approaches,          for top cliques when the backbone is VGG-13, in the
Conover (with Holm correction) provides sharper pairwise            source→test (misclassification) regime, the best-performing
calls under the same nonparametric framework. From those            confidence-scoring functions (CSFs) are predominantly
adjusted p-values, we construct an Indifference Graph which         probabilistic: GEN, MSR, PCE, and REN appear in the
connects two methods if they are not significantly differ-          top cliques across class counts. In contrast to the CNN
ent. Applying the Bron–Kerbosch algorithm enumerates                case, CTM contributes less in the mid and far regimes. A
all maximal sets of CSFs mutually indistinguishable so that         plausible explanation is that CTM relies on class feature
top groups of CSFs are generated rather than a brittle sin-         prototypes whose utility can diminish after fine-tuning a


                                                                5

## Page 6

A Systematic Analysis of Out-of-Distribution Detection

                              Top cliques
               (Backbone:VGG-13, Metrics=['AUGRC', 'AURC'])                                             Top cliques
                  CTM                                          6                         (Backbone:ViT, Metrics=['AUGRC', 'AURC'])
           CTM global                                          6                         CTM                                     1
                                                         2                        CTM global                                     1
            CTMmean                                                                CTMmean                                                   4
     CTMmean global                                             5           CTMmean global                                               3
         CTMmeanOC                                          3                   CTMmeanOC                                                    4
           Confidence                                   1                             Energy                                     1
                                                                               Energy global                                         2
               Energy                                             6                        GE                                        2
        Energy global                                           5                   GE global                                        2
                   GE                                       3                            GEN                                         2
                                                                                  GEN global                                           4
             GE global                                          5                  GradNorm                                          2
                  GEN                                         4         GradNorm class pred                                                            7
           GEN global                                         4             GradNorm global                                                      5
                                                          2          KPCA RecError class pred                                                      6
            GradNorm                                                    KPCA RecError global                                                     5
     GradNorm global                                    1                                MLS                                     1
                  MLS                                           5                  MLS class                                             3
           MLS global                                             6               MLS  global                                        2
                                                                                         MSR                                             3
                  MSR                                         4                   MSR global                                                     5
           MSR global                                         4                         Maha                                                 6
             NNGuide                                            5                   NNGuide                                            3
                                                                    7        NNGuide global                                          2
      NNGuide global                                                                    NeCo                                           3
                 NeCo                                           5        PCA RecError global                                                 6
                  PCE                                     2                               PCE                                          3
           PCE global                                     2                       PCE global                                             4
                                                                                           PE                                        2
                   PE                                       3                       PE global                                                6
             PE global                                    2                              REN                                         2
                  REN                                   1                         REN global                                             4
                                                                                     Residual                                              5
           REN global                                     2                               ViM                                                  7
                 fDBD                                       3                           fDBD                                         2
          fDBD global                                           5                fDBD  global                                          3
                                                        1                               pNML                                                 6
                pNML                                                            pNML global                                      1
                          supercifar100->test


                         supercifar100->near


                          supercifar100->mid


                            supercifar100->far
                                   cifar10->test
                                 cifar100->test
                                  cifar10->near
                                cifar100->near
                                   cifar10->mid
                                 cifar100->mid
                                    cifar10->far
                                  cifar100->far
                          tinyimagenet->test


                         tinyimagenet->near


                           tinyimagenet->mid


                            tinyimagenet->far




                                                                                                 supercifar100->test


                                                                                                supercifar100->near


                                                                                                 supercifar100->mid


                                                                                                   supercifar100->far
                                                                                                          cifar10->test
                                                                                                        cifar100->test
                                                                                                         cifar10->near
                                                                                                       cifar100->near
                                                                                                          cifar10->mid
                                                                                                        cifar100->mid
                                                                                                           cifar10->far
                                                                                                         cifar100->far
                                                                                                 tinyimagenet->test


                                                                                                tinyimagenet->near


                                                                                                  tinyimagenet->mid


                                                                                                   tinyimagenet->far
Figure 1. Top-clique map for AURC/AUGRC metrics: rows are CSF; columns are evaluation regimes labeled “source→test, near, mid, far”.
Within each column, connected dots indicate the Conover–Holm top clique (α=0.05). Larger cliques imply more methods are statistically
tied. Shaded bands emphasize methods that repeatedly appear in top cliques across regimes. (Left) For VGG-13, probabilistic-derived
CSF dominate the ID regime (“test”,) and prototype/geometry-aware methods (CTM-family, NNGuide, fDBD) dominate the mid/far
regimes. (Right) For ViT, the ID regime is also dominated by probabilistic-derived CSFs, but top groups across near, mid, and far are
centered on GradNorm and KPCA RecError.


large, pretrained ViT. Fine-tuning sharpens decision bound-              number of clearly superior CSFs as shift severity increases.
aries for the source task while partially reshaping prototype            Notably, using the class-pred projection, which is the sub-
geometry, reducing CTM’s discriminative advantage under                  space selected by the predicted class, boosts KPCA R E -
stronger shifts. Two methods that remain consistently com-               C E RROR , but specially G RAD N ORM , likely because this
petitive are G RAD N ORM and KPCA R EC E RROR; in our re-                projection leverages the classifier’s own discrimination to
sults, G RAD N ORM improves as the number of classes grows               choose a feature subspace where margin curvature better
(larger label spaces yield tighter, more curved decision re-             separate atypical examples. Finally, V I M shows a strong
gions where gradient-based margin proxies are informative),              presence in near and far but not mid, consistent with its
whereas KPCA is comparatively stronger at smaller class                  reliance on virtual-logit structure that is effective when the
counts (lower intrinsic dimensionality makes reconstruction              shift is either small (logit geometry remains reliable) or very
error more stable).                                                      large (clear separation), but less decisive in intermediate,
                                                                         confusable conditions.
Across near, mid, and far shifts, the near columns retain
the largest cliques, while mid and far typically admit fewer
top-tied CSFs. This mirrors the observation that, under                  4.4. Neural-Collapse–Based Analysis
fine-tuning, mid and far shifts present similar difficulty. Pre-         We use the NC metrics from Section 3.4 to interpret the
training exposes ViT to semantically related categories, and             architecture- and training-dependent rankings of confidence
fine-tuning adjusts boundaries to the source classes while               scoring functions (CSFs). Full tables and per-dataset details
preserving partial structure from pretraining, limiting the              are given in Supplementary Section D.

                                                                     6

## Page 7

A Systematic Analysis of Out-of-Distribution Detection

CNNs (VGG-13). For TinyImageNet, the last-layer fea-                 provided in Appendix D.
tures are close to an NC regime: within-class variabil-
ity is small, class means and classifier weights are nearly          4.5. MCD vs non-MCD Performance
equinorm/equiangular, and self-duality is strong. In this
geometry, class prototypes µc and weights wc almost co-              Table 2 presents a comprehensive evaluation of various
incide, and ID features concentrate near their correspond-           model configurations and training methodologies for se-
ing class means. Cosine-based prototype scores such as               lective classification, with performance measured by the
CTM and neighbor-based scores such as NNGuide are then               Area Under the Generalized Risk-Coverage (AUGRC)
near–Bayes-optimal in feature space: ID points have co-              curve. Only the best performing scores for each source
sine ≈ 1 to the correct prototype, while OOD points lie in           and ID/OOD datasets are shown.
angular gaps and achieve a strictly smaller maximum co-              The most prominent finding is the trade-off between stan-
sine. This matches the cliques, where CTM and NNGuide                dard inference and Monte Carlo Dropout (MCD) as a func-
dominate. On CIFAR-100, where self-duality is weaker,                tion of the number of classes. For datasets with a low num-
boundary-distance methods such as fDBD also appear in the            ber of classes (e.g., 10 classes), models evaluated without
top cliques. When weights and means misalign, distance to            MCD consistently achieve lower AUGRC scores. This sug-
the learned linear decision surface becomes a more faithful          gests that for simpler tasks, methods that directly learn a
margin proxy than pure prototype alignment. Global pro-              confidence score, like those from the Deep Gamblers (DG)
jection filtering only slightly changes aggregate NC metrics         framework, are more effective at providing a reliable uncer-
but improves CTMmean global and NNGuide global on                    tainty estimate. However, as the number of classes increases,
mid-OOD, consistent with PCA acting as an ID manifold                the performance of models evaluated with MCD improves
projector that removes nuisance directions and sharpens an-          significantly and often surpasses that of non-MCD methods.
gular/neighbor structure in the subspace most aligned with           This indicates that for complex classification tasks, the epis-
between-class variability. On CIFAR-10 with CNNs, the                temic uncertainty captured by MC dropout provides a more
last-layer geometry is close to neural collapse with well-           robust signal for selective classification.
separated, roughly equinorm class means and weights, so
the correct-class logit is consistently much larger than the         The choice of training method is also critical for optimizing
others. In this regime, both Energy (log-sum-exp of logits)          performance, with different approaches yielding the best
and MLS (maximum logit) act as stable margin-like scores,            results depending on the intended inference strategy. When
giving high, well-separated values for ID samples and lower,         using MCD for inference, models trained with ConfidNet
less concentrated values for OOD inputs.                             produce the best AUGRC scores across the board. When
                                                                     MCD is not an option at inference time, the Deep Gamblers
                                                                     dg training method stands out. This end-to-end approach
Transformers (ViT). For the fine-tuned ViT, NC met-                  reframes selective classification as a gambling problem,
rics show weaker collapse, less ETF-like class means, and            learning to “abstain” from making a prediction when the
poorer alignment between means and weights across all                confidence is low. This training method, however, requires
datasets. The representation retains richer pretraining struc-       tuning of the reward hyperparameter (see Appendix A).
ture, and the linear head does not implement a clean nearest-        Our results show that this method produces the best AU-
prototype classifier. In this regime, prototype geometry is a        GRC scores for non-MCD scenarios, regardless of whether
less reliable summary of the decision rule, and the cliques          dropout layers were used during training.
instead favor GradNorm and KPCA with global and class
predicted projections. These scores aggregate information            4.5.1. L IMITATIONS
when class means are not well separated or tightly coupled           Our study focuses on image classification with a restricted
to the classifier. Class-predicted projection on ViTs reduces        set of sources and shifts (CIFAR-10/100, super-CIFAR-
within-class variability in predicted-class subspaces (im-           100, TinyImageNet, and common near/mid/far OOD bench-
proved NC1 locally) and particularly benefits GradNorm               marks), two backbone regimes (a scratch-trained CNN and
and KPCA reconstruction error by sharpening boundary                 a fine-tuned ViT), and a curated portfolio of confidence-
curvature and class-specific manifolds.                              scoring functions (CSFs). While this choice enables careful,
Overall, the NC analysis sharpens our representation-centric         paired comparisons and rigorous statistics, it limits external
conclusions: CNNs trained on many classes and approach-              validity: results may not transfer to larger or newer back-
ing NC favor prototype- and boundary-based CSFs, whereas             bones, self-supervised foundations, multi-label settings, de-
ViTs with weaker collapse favor gradient- and manifold-              tection/segmentation, or distribution drifts that are temporal,
based CSFs; projection filtering nudges models along this            causal, or task-specific. Our CLIP-based grouping uses a
continuum by refining the ID subspace geometry. Detailed             fixed encoder and k-means with k = 3; both the encoder
NC metrics computations and additional case studies are              and clustering hyperparameters determine the “semantic

                                                                 7

## Page 8

A Systematic Analysis of Out-of-Distribution Detection

Table 2. Best IID (rows labeled test) and OOD (other rows) AU-                     and the best performing CSFs might change if another met-
GRC scores and model configurations. This table summarizes the
performance of various selective classification models, with results
                                                                                   ric is used. Also, we did not evaluate computational cost
evaluated using the AUGRC metric. The findings demonstrate                         or latency, which are factors that matter in deployment and
a clear dependence on dataset complexity. For datasets with a                      may re-rank methods under real-time constraints.
low number of classes, non-MCD methods, particularly the Deep
Gamblers (DG) model, achieve superior AUGRC scores. Con-                           4.6. Conclusion
versely, as the number of classes increases, Monte Carlo Dropout
(MCD) with models trained using the confidnet method becomes                       Our study provides evidence that OOD detection perfor-
the optimal approach.
                                                                                   mance is primarily governed by the structure of the learned
                                 nonMCD     MCD        nonMCD      MCD             representation rather than the sophistication of the scoring
                                 AUGRC      AUGRC      paradigm    paradigm
                  test           3.9132     4.5384     DG          confidnet       rule. Synthesizing the empirical results from our rank-based
                  cifar100       163.4954   163.3573   confidnet   confidnet       statistical pipeline with the theoretical insights from our
                  tinyimagenet   149.7686   150.2396   confidnet   confidnet
                  isun           133.5868   133.8892   confidnet   confidnet       Neural Collapse analysis, we summarize the key findings
  cifar10




                  lsun cropped   148.6756   150.3665   DG          confidnet       regarding architecture, training, and detection mechanisms
                  lsun resize    148.4497   148.7788   confidnet   confidnet
                  places365      159.3993   161.7047   DG          confidnet       as follows:
                  textures       88.0191    89.9795    DG          DeVries
                  svhn           287.3275   287.7814   DG          DeVries
                  test           145.7328   144.6418   DG          DG               1. For Misclassification Detection (ID), simple probabilis-
                  cifar10        266.3995   267.6652   confidnet   DG                  tic scores (GEN, MSR) consistently outperform spe-
  supercifar100




                  tinyimagenet   232.1150   225.5546   DG          DG
                  isun           221.0056   215.9046   DG          DG                  cialized OOD detectors for both CNN and ViT models,
                  lsun cropped   229.9232   234.8684   DG          confidnet           confirming that softmax confidence remains the most
                  lsun resize    237.1198   230.2041   DG          DG
                  textures       158.8078   162.8347   DG          confidnet           reliable signal for identifying ordinary in-distribution
                  places365      264.2114   263.4520   DG          confidnet           prediction errors.
                  svhn           346.4916   354.0474   DG          confidnet
                  test           54.3580    52.6486    DeVries     DG
                  cifar10        209.7024   209.9227   DeVries     DG               2. On CNNs, ranking quality is primarily driven by
                  tinyimagenet   195.6496   195.5451   confidnet   confidnet           margin-based scores (Energy and MLS) which dom-
  cifar100




                  isun           181.3098   185.1370   DeVries     confidnet
                  lsun cropped   196.5520   194.7769   DG          confidnet           inate in Near-OOD acenarios where logit magnitude
                  lsun resize    196.2106   197.8931   DG          confidnet           is reliable, while geometry-aware scores (NNGuide,
                  textures       125.4978   121.1945   DeVries     confidnet
                  places365      210.3229   211.2779   DeVries     confidnet           fDBD, CTM) dominate mid/far by leveraging distinct
                  svhn           326.7164   322.6135   DG          confidnet           boundary proximity and feature-space clustering. On
                  test           97.0525    91.3660    DG          DG
                  cifar100       219.9844   215.7338   confidnet   confidnet           ViTs, efficacy shifts towards manifold and gradient-
                  cifar10        214.7156   209.2819   confidnet   confidnet           based approaches: GradNorm and KPCA Reconstruc-
  tinyimagenet




                  isun           195.8857   189.4729   DG          confidnet
                  places365      223.5343   218.2668   DG          DeVries             tion Error remain consistently competitive across Near,
                  lsun resize    211.7826   205.5667   DG          confidnet           Mid, and Far shifts, particularly when enhanced with
                  textures       150.5724   146.8387   confidnet   confidnet
                  lsun cropped   209.3178   205.0045   DG          confidnet           class-predicted projection to navigate the complex,
                  svhn           344.5073   340.2405   DG          DeVries             non-collapsed geometry of finetuned representations
                                                                                    3. Monte-Carlo Dropout yields a class-count–dependent
                                                                                       trade-off (beneficial as the label space grows, less so
distance.” In our evaluation, we emphasize threshold-free                              on simple tasks).
ranking metrics AURC/AUGRC, but do not fully explore
cost-sensitive operating points, deployment-specific utility,                       4. Projection filtering enhances separability by filtering
or calibration–utility trade-offs under label shift. Finally, the                      directions that do not support class-consistent structure,
ViT is fine-tuned from a large pretrained model; potential                             though the optimal strategy depends on the architec-
latent overlap with OOD categories was not exhaustively au-                            ture: Global Projection sharpens angular geometry for
dited and could confound prototype-style scoring in subtle                             CNNs (benefiting NNGuide and CTM) , whereas Class-
ways.                                                                                  Predicted Projection is critical for finetuned ViTs to
                                                                                       strip away pretraining residuals, enabling gradient- and
Our rank-based pipeline (Friedman → Conover with                                       manifold-based scores (GradNorm, KPCA) to perform
Holm correction→Bron–Kerbosch cliques) provides robust,                                effectively.
multiple-comparison-controlled summaries, but inherits as-
sumptions of complete blocks and exchangeability. Also,                             5. Our neural-collapse–based geometric analysis suggests
different post-hoc choices (e.g., Nemenyi, Quade, SC all-                              that the apparent superiority of certain OOD detectors
pairs) can yield slightly different borders for “indistinguish-                        is largely a consequence of how training shapes pro-
able” sets. Hyperparameter sweeps (e.g., DG rewards, ker-                              totype geometry and within-class collapse, providing
nel bandwidths for KPCA, temperature scaling, etc) were                                a blueprint for designing representations that make
optimized on the validation set to achieve the best AUGRC,                             failure detection easier.

                                                                               8

## Page 9

A Systematic Analysis of Out-of-Distribution Detection

Impact Statement                                                    Demšar, J. Statistical comparisons of classifiers over multi-
                                                                      ple data sets. Journal Of Machine Learning Research, 7
This paper presents a systematic benchmark and theoretical
                                                                      (Jan):1–30, 2006.
analysis of Out-of-Distribution (OOD) detection methods,
with the goal of advancing the reliability and safety of ma-        Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei,
chine learning systems. By identifying which detection                L. Imagenet: A large-scale hierarchical image database.
scores perform best under specific architectures and training         In 2009 IEEE Conference On Computer Vision And Pat-
regimes, this work directly supports the deployment of more           tern Recognition, pp. 248–255. Ieee, 2009.
robust deep learning models in safety-critical applications,
such as medical imaging and autonomous systems, where               DeVries, T. and Taylor, G. W. Learning confidence for
”silent failures” pose significant risks. While our primary           out-of-distribution detection in neural networks. ArXiv
contribution is methodological with theoretical insights, im-         Preprint ArXiv:1802.04865, 2018.
proved OOD detection capabilities can help mitigate au-
                                                                    Dowson, D. and Landau, B. The fréchet distance between
tomation bias and reduce the likelihood of high-confidence
                                                                      multivariate normal distributions. Journal Of Multivariate
errors in real-world deployments. We do not foresee any im-
                                                                     Analysis, 12(3):450–455, 1982.
mediate negative societal consequences or ethical concerns
arising directly from this research.                                Fang, K., Tao, Q., He, M., Lv, K., Yang, R., Hu, H.,
                                                                      Huang, X., Yang, J., and Cao, L. Kernel PCA for out-
References                                                            of-distribution detection: Non-linear kernel selections
                                                                      and approximations. ArXiv Preprint ArXiv:2505.15284,
Ammar, M. B., Belkhir, N., Popescu, S., Manzanera, A.,                2025.
 and Franchi, G. Neco: Neural collapse based out-of-
 distribution detection. ArXiv Preprint ArXiv:2310.06823,           Fréchet, M. Sur la distance de deux lois de probabilité. In
 2023.                                                                Annales De L’ISUP, volume 6, pp. 183–198, 1957.
Bibas, K., Feder, M., and Hassner, T. Single layer predictive       Gal, Y. and Ghahramani, Z. Dropout as a bayesian approxi-
  normalized maximum likelihood for out-of-distribution               mation: Representing model uncertainty in deep learning.
  detection. Advances In Neural Information Processing                In International Conference On Machine Learning, pp.
  Systems, 34:1179–1191, 2021.                                       1050–1059. PMLR, 2016.

Bron, C. and Kerbosch, J. Algorithm 457: Finding all                Geifman, Y., Uziel, G., and El-Yaniv, R. Bias-reduced
  cliques of an undirected graph. Communications Of The               uncertainty estimation for deep neural classifiers. ArXiv
  ACM, 16(9):575–577, 1973.                                           Preprint ArXiv:1805.08206, 2018.

Bungert, T. J., Kobelke, L., and Jaeger, P. F. Understanding        Granese, F., Romanelli, M., Gorla, D., Palamidessi, C., and
  silent failures in medical image classification. In Inter-          Piantanida, P. Doctor: A simple method for detecting
  national Conference On Medical Image Computing And                  misclassification errors. Advances In Neural Information
  Computer-Assisted Intervention, pp. 400–410. Springer,              Processing Systems, 34:5669–5681, 2021.
  2023.
                                                                    Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., and
Conover, W. J. Practical Nonparametric Statistics. John               Smola, A. A kernel method for the two-sample-problem.
  Wiley & Sons, 1999.                                                 Advances In Neural Information Processing Systems, 19,
                                                                      2006.
Corbière, C., Thome, N., Bar-Hen, A., Cord, M., and Pérez,
  P. Addressing failure prediction by learning model con-           Guan, X., Liu, Z., Zheng, W.-S., Zhou, Y., and Wang, R.
  fidence. Advances In Neural Information Processing                  Revisit PCA-based technique for out-of-distribution de-
  Systems, 32, 2019.                                                  tection. In Proceedings Of The IEEE/CVF International
                                                                     Conference On Computer Vision, pp. 19431–19439, 2023.
Corbiere, C., Thome, N., Saporta, A., Vu, T.-H., Cord, M.,
  and Perez, P. Confidence estimation via auxiliary models.         Gutbrod, M., Rauber, D., Nunes, D. W., and Palm, C. Open-
  IEEE Transactions On Pattern Analysis And Machine                   MIBOOD: Open medical imaging benchmarks for out-of-
  Intelligence, 44(10):6043–6055, 2021.                               distribution detection. ArXiv Preprint ArXiv:2503.16247,
                                                                      2025.
Cover, T. M. and Thomas, J. A. Elements Of Information
  Theory (Wiley Series In Telecommunications And Sig-               Hendrycks, D. and Gimpel, K. A baseline for detecting
  nal Processing). Wiley-Interscience, USA, 2006. ISBN                misclassified and out-of-distribution examples in neural
  0471241954.                                                         networks. ArXiv Preprint ArXiv:1610.02136, 2016.

                                                                9

## Page 10

A Systematic Analysis of Out-of-Distribution Detection

Hendrycks, D., Basart, S., Mazeika, M., Zou, A., Kwon, J.,               Papyan, V., Han, X., and Donoho, D. L. Prevalence of
  Mostajabi, M., Steinhardt, J., and Song, D. Scaling out-                 neural collapse during the terminal phase of deep learn-
  of-distribution detection for real-world settings. ArXiv                 ing training. Proceedings Of The National Academy Of
  Preprint ArXiv:1911.11132, 2019.                                         Sciences, 117(40):24652–24663, 2020.
Holm, S. A simple sequentially rejective multiple test pro-              Park, J., Jung, Y. G., and Teoh, A. B. J. Nearest neigh-
  cedure. Scandinavian Journal Of Statistics, pp. 65–70,                   bor guidance for out-of-distribution detection. In Pro-
 1979.                                                                     ceedings Of The IEEE/CVF International Conference On
Huang, R., Geng, A., and Li, Y. On the importance of                       Computer Vision, pp. 1686–1695, 2023.
  gradients for detecting distributional shifts in the wild.             Pope, P., Zhu, C., Abdelkader, A., Goldblum, M., and Gold-
 Advances In Neural Information Processing Systems, 34:                    stein, T. The intrinsic dimension of images and its impact
  677–689, 2021.                                                           on learning. ArXiv Preprint ArXiv:2104.08894, 2021.
Iman, R. L. and Davenport, J. M. Approximations of the                   Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
  critical region of the fbietkan statistic. Communications                Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J.,
  In Statistics-Theory And Methods, 9(6):571–595, 1980.                    et al. Learning transferable visual models from natural
Jaeger, P. F., Lüth, C. T., Klein, L., and Bungert, T. J. A call          language supervision. In International Conference On
  to reflect on evaluation practices for failure detection in              Machine Learning, pp. 8748–8763. PmLR, 2021.
  image classification. ArXiv Preprint ArXiv:2211.15259,
                                                                         Rényi, A. On measures of entropy and information. In Pro-
  2022.
                                                                            ceedings Of The Fourth Berkeley Symposium On Mathe-
Lee, K., Lee, K., Lee, H., and Shin, J. A simple unified                    matical Statistics And Probability, Volume 1: Contribu-
  framework for detecting out-of-distribution samples and                   tions To The Theory Of Statistics, volume 4, pp. 547–562.
  adversarial attacks. Advances In Neural Information Pro-                  University of California Press, 1961.
  cessing Systems, 31, 2018.
                                                                         Shannon, C. E. A mathematical theory of communication.
Liu, L. and Qin, Y. Fast decision boundary based out-of-                   The Bell System Technical Journal, 27(3):379–423, 1948.
  distribution detector. ArXiv Preprint ArXiv:2312.11536,
  2023.                                                                  Traub, J., Bungert, T. J., Lüth, C. T., Baumgartner, M.,
                                                                           Maier-Hein, K. H., Maier-Hein, L., and Jaeger, P. F. Over-
Liu, W., Wang, X., Owens, J., and Li, Y. Energy-based out-                 coming common flaws in the evaluation of selective clas-
  of-distribution detection. Advances In Neural Information                sification systems. ArXiv Preprint ArXiv:2407.01032,
  Processing Systems, 33:21464–21475, 2020.                                2024.
Liu, X., Lochman, Y., and Zach, C. Gen: Pushing the
                                                                         Wang, H., Li, Z., Feng, L., and Zhang, W. ViM: Out-of-
  limits of softmax-based out-of-distribution detection. In
                                                                          distribution with virtual-logit matching. In Proceedings
  Proceedings Of The IEEE/CVF Conference On Computer
                                                                          Of The IEEE/CVF Conference On Computer Vision And
  Vision And Pattern Recognition, pp. 23946–23955, 2023.
                                                                          Pattern Recognition, pp. 4921–4930, 2022.
Liu, Z., Wang, Z., Liang, P. P., Salakhutdinov, R. R.,
                                                                         Yang, J., Wang, P., Zou, D., Zhou, Z., Ding, K., Peng, W.,
  Morency, L.-P., and Ueda, M. Deep gamblers: Learn-
                                                                           Wang, H., Chen, G., Li, B., Sun, Y., et al. Openood:
  ing to abstain with portfolio theory. Advances In Neural
                                                                           Benchmarking generalized out-of-distribution detection.
  Information Processing Systems, 32, 2019.
                                                                           Advances In Neural Information Processing Systems, 35:
Massey, J. L. Guessing and entropy. In Proceedings Of 1994                 32598–32611, 2022.
 IEEE International Symposium On Information Theory,
 pp. 204. IEEE, 1994.                                                    Zhang, J., Yang, J., Wang, P., Wang, H., Lin, Y., Zhang,
                                                                           H., Sun, Y., Du, X., Li, Y., Liu, Z., et al. Openood v1.5:
Ngoc-Hieu, N., Hung-Quang, N., Ta, T.-A., Nguyen-Tang,                     Enhanced benchmark for out-of-distribution detection.
 T., Doan, K. D., and Thanh-Tung, H. A cosine similarity-                  ArXiv Preprint ArXiv:2306.09301, 2023.
  based method for out-of-distribution detection. ArXiv
 Preprint ArXiv:2306.14920, 2023.                                        Zhou, J., Jiang, J., and Zhu, Z. Are all layers created equal:
                                                                           A neural collapse perspective. In The Second Conference
Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D.,                    On Parsimony And Learning (Proceedings Track), 2025.
  Nowozin, S., Dillon, J., Lakshminarayanan, B., and
  Snoek, J. Can you trust your model’s uncertainty? evalu-
  ating predictive uncertainty under dataset shift. Advances
  In Neural Information Processing Systems, 32, 2019.

                                                                    10

## Page 11

A Systematic Analysis of Out-of-Distribution Detection

A. Training Paradigms, CFS Baselines and Variations
A.1. Computing Infrastructure
All experiments were executed on an internal GPU cluster. CNN runs (VGG-13 trained from scratch) were scheduled on
NVIDIA T4 GPUs, while ViT runs (fine-tuned from a large pretrained model) were scheduled on NVIDIA A100 GPUs. We
did not mix GPU types within an experiment. Every training/evaluation job for a given backbone used the same GPU class
to avoid hardware-induced variance.

A.2. Training Paradigms
A.2.1. C ONFID N ET ( REGRESSING THE TRUE - CLASS PROBABILITY ).
Let fW = Pg ◦ h be a trained classifier with weights W, logits f (x) ∈ RK and softmax pk (x) =
exp(fk (x))/ j exp(fj (x)). The standard confidence proxy is the maximum class probability, MSR(x)                       =
maxk∈{1,...,K} pk (x), but this quantity can be spuriously large for both correct and erroneous predictions, hampering
failure detection. ConfidNet replaces MSR with the true-class probability (TCP), TCP(x, y) = py (x), which is typically
high for correct predictions and low for misclassifications (Corbière et al., 2019; Corbiere et al., 2021). Because y is
unknown at test time, ConfidNet learns an auxiliary regressor sW conf : X → [0, 1] that predicts TCP(x, y) from features of
the trained classifier fW .
Denote by E = h the encoder           of fW and by gW conf : RD → [0, 1] a small MLP head. ConfidNet’s score is
sW conf (x) = gW conf ◦ E (x) ≈ TCP(x, y), trained on the labeled training set Dtrain with a mean squared error
                                                                                          2
                                   1
loss Lconf (W conf ; Dtrain ) = |Dtrain
                                          P
                                                              sW conf (xi ) − TCP(xi , yi )   . This post-hoc module transfers knowl-
                                        |  (x  ,y
                                              i i )∈D train

edge from the classifier’s encoder and yields a calibrated, label-informed confidence score at inference. It should be noted
that the weights W are kept frozen when training gW conf . In practice, ConfidNet can be viewed as a supervised alternative
to MSR that aligns the confidence target with the Bayes-relevant quantity TCP; see Corbière et al. (2019); Corbiere et al.
(2021) for more details on the implementation.

A.2.2. D E V RIES & TAYLOR : LEARNED CONFIDENCES VIA TARGET – PREDICTION INTERPOLATION .
Given a classifier fW = g ◦ h with softmax output p(x) ∈ ∆K−1 , DeVries & Taylor (2018) add a confidence
branch in parallel to the  class head. The branch shares features with the penultimate layer and outputs sW devries (x) =
softmax w⊤ h(x) + b ∈ [0, 1], interpreted as the network’s confidence that it can correctly predict the label of x. During
training, the method provides the network with “hints” by interpolating        between its own prediction and the one-hot target y
according to s: p′ (x, y) = sW devries (x) p(x) + 1 − sW conf (x) y. The task loss is then computed             on p′ rather than on p,
                                                                      
                                                               1                      ′
                                           devries
                                                                     P                               
e.g. the negative log-likelihood Ltask (W          , W) = − |Dtrain | (xi ,yi ) log [p (xi , y i )]yi . Intuitively, when the model is
uncertain (s ≈ 0), it is allowed to rely more on the target distribution (a hint); when it is confident (s ≈ 1), it must stand by its
own prediction.
To avoid the degenerate solution s =P        0 (which would always copy the target), a confidence regularizer encourages high s
values: Ldevries (W devries ) = − |D1train | xi log sW conf (xi ), Ltotal = Ltask + λ Ldevries , with λ > 0 balancing reliance on hints
versus self-prediction. The resulting balancing where copying the target when the classifier struggles vs. being penalized for
low confidence drives s(x) to be large where the model is accurate and small where it is prone to error. At test time, the
class prediction uses p(x) from the base head, while s(x) serves as a confidence score for failure/OOD detection; no hints
are used at inference. See DeVries & Taylor (2018) for more details on the implementation.

A.2.3. D EEP G AMBLERS : ABSTENTION VIA A (K+1)- ST CLASS AND A REWARD PARAMETER .
Liu et al. (2019) cast classification as a Kelly-style gambling game (Cover & Thomas, 2006). For a K-class task, the
network predicts over K+1 outcomes. This means that an additional abstain option is attached to the original K labels.
Let fW = g ◦ h with logits f (x) ∈ RK+1 , p(x) = softmax(f (x)) ∈ ∆K , and write pk (x) for class k ≤ K and
pK+1 (x) for abstain. The training objective maximizes the expected (log) wealth in a horse-race with reservation:
predicting the true class y yields P payoff o > 0 (a tunable reward), and abstaining yields payoff 1. This leads to the loss
LDG (W; Dtrain , o) := − |D1train | (xi ,yi )∈Dtrain log o pyi (xi ) + pK+1 (xi ) .
                                                                                 

When pK+1 = 0 (no abstention), LDG reduces to cross-entropy up to an additive constant (since log o adds to the true-class

                                                                  11

## Page 12

A Systematic Analysis of Out-of-Distribution Detection

logit). The head is linear, g(z) = W z + b, W ∈ R(K+1)×D , b ∈ RK+1 , so the method is a drop-in replacement for a
standard classifier.
At test time, the model accepts an output if its (reward-weighted) probability dominates abstention, and rejects otherwise.
A sufficient decision rule is accept ⇔ o · maxk≤K pk (x) > pK+1 (x), reject (abstain) otherwise. Hence, larger o
discourages abstention (the classifier must be more confident to reject), while smaller o encourages it. o is tuned on
validation to meet a desired risk-coverage trade-off. For failure/OOD detection one can use the abstention mass as a score,
sDG (x) = pK+1 (x) (higher ⇒ more likely atypical), or a margin-like variant sDG (x) = pK+1 (x) − o · maxk≤K pk (x)
(negative ⇒ accept). The former is used in this work. The (K+1)-st class thus implements a principled, calibrated
abstention mechanism consistent with the Kelly criterion, while keeping the architecture and training pipeline simple. See
Liu et al. (2019) for details on the implementation and effect of o.

    Note
    Unlike DeVries & Taylor (DeVries & Taylor, 2018) and Deep Gamblers (Liu et al., 2019), ConfidNet (Corbière
    et al., 2019; Corbiere et al., 2021) fine-tunes a copy of a trained encoder h to generate the confidence score sW conf .
    This copy does not modify f which is the output used to train and evaluate the ID/OOD detection methods. In
    fact, f in the ConfidNet training paradigm is the result of a conventional training approach, which is guided by the
    minimization of the cross-entropy loss.


A.3. ID/OOD Detection Methods
This section describes all the CSFs and their variations after applying the Projection Filtering described in section 3.2.

A.3.1. C LASS T YPICAL M ATCHING (CTM) AND C LASS T YPICAL M ATCHING WITH CLASS MEANS PROTOTYPES
       (CTM MEAN ) (N GOC -H IEU ET AL ., 2023)
Prototype matching in feature space consists of quantifying the similarity between a sample x and the last-layer trained
                                                                                                                       
weights {w1 , . . . , wK }. Therefore the similarity to the closest trained weight is CTM(x) = maxk≤C sim wk , h .
Alternatively, we can compute
                                 class means µctrain and score by similarity to the closest class mean, CTMmean(x) =
                  k
maxk≤C sim µtrain , h . Following Ngoc-Hieu et al. (2023), we use cosine similarity in this work, where sim(u, v) =
  u⊤ v
∥u∥2 ∥v∥2 . CTM scores the typicality of x by comparing its feature h against a bank of class representatives. Higher
similarity indicates greater in-distribution (ID) conformity.

A.3.2. E NERGY (L IU ET AL ., 2020)
                                                       PC                    
The energy score is defined as Energy(x) = −T log         k=1 exp g(h)k /T       , with temperature T > 0. Higher Energy score
typically indicates higher uncertainty.

A.3.3. M AXIMUM S OFTMAX R ESPONSE (MSR) (H ENDRYCKS & G IMPEL , 2016) AND M AXIMUM L OGIT S CORE
       (MLS) (H ENDRYCKS ET AL ., 2019)
A baseline confidence score given by the maximum predicted probability MSR(x) = maxk≤C pk , widely used for OOD
detection. Lower values indicate atypical inputs. Similarly, MLS is a confidence score measured in the logit space,
MLS(x) = maxk≤C g(h)k , often more stable than softmax under temperature changes.

A.3.4. P REDICTIVE E NTROPY (PE), G ENERALIZED E NTROPY (GEN), R ENYI E NTROPY (REN), G UESSING E NTROPY
       (GE), AND P REDICTIVE C OLLISION E NTROPY (PCE)
Predictive Entropy (PE) (Ovadia etPal., 2019). Uncertainty via Shannon entropy (Shannon, 1948) of the predictive
                                   C
distribution PE(x) = H p(x) = − k=1 pk log pk , with larger entropy signaling higher uncertainty.

Generalized Entropy (GEN) (Liu et al., 2023). GEN is a post-hoc OOD score that uses the softmax probabilities of
a trained classifier. Let p(1) ≥ · · · ≥ p(K) denote the probabilities sorted in descending order for a given input x. For
sensitivity and numerical stability in many-class P
                                                  settings, GEN truncates
                                                                       γ to the top-M classes and computes a generalized
                                                     M     γ
entropy with exponent γ ∈ (0, 1): GEN(x) =               p
                                                     k=1 (k)  1 − p (k) . The confidence score is the negated generalized


                                                              12

## Page 13

A Systematic Analysis of Out-of-Distribution Detection

entropy so that a larger GEN (lower entropy) indicates more ID-like predictions.


Rényi Entropy (REN). The Rényi entropy (Rényi, 1961) of order α is a smooth generalization of Shannon entropy that
                                  distribution. Similar to GEN, REN is defined by a truncation paramater M and exponent
emphasizes different parts of the P
                          1         M
α ∈ (0, 1): REN(x) = 1−α     log k=1 pα    (k) .



Guessing Entropy (GE). GE (Massey, 1994) quantifies the expected number of guesses to identify the  PCtrue class when
labels are guessed in decreasing probability pk (x): if p(1) ≥ · · · ≥ p(K) are sorted, then GE(x) = k=1 k p(k) , with
larger values denoting higher uncertainty.


Predictive Collision Entropy (PCE) (Granese et al., 2021). PCE         Pmeasures     prediction uncertainty via the collision
                                                                          C
(order-2 Rényi) entropy of the softmax distribution: PCE(x) = − log k=1 p2k . Since k p2k is the “collision probability,”
                                                                                         P
PCE grows as the distribution spreads (uncertain/atypical) and shrinks as it peaks (confident/ID-like). This uncertainty score
uses the entire predictive distribution rather than just its maximum.

A.3.5. M AHALANOBIS D ISTANCE (M AHA ) (L EE ET AL ., 2018)
Assuming Gaussian class-conditional features, score by the (negative) Mahalanobis distance to the nearest class centroid is
                                    ⊤                           ⊤
Maha(x) = maxk≤C h(x) − µktrain Σ−1 h(x) − µktrain = hk Σ−1 hk , where Σ is the empirical covariance matrix.
                                                           


A.3.6. N EAREST N EIGHBOR G UIDE (NNG UIDE ) (PARK ET AL ., 2023)
NNGuide is a post-hoc wrapper that modulates any classifier-based OOD score Sbase (h) using nearest neighbors from an
ID bank of features. This bank is formed by sampling α ∈ (0, 1) of ID training features hi (L2-normalized) and their
base scores si = Sbase (hi ). More specifically, given an input x, the corresponding feature h (L2-normalized) defines a
                                                 N
confidence-scaled similarity list si cos(h, hi ) i=1 , which is sampled by taking the top-k terms, where k = ⌊αN ⌋. The
top-k terms set the guidance G(h) = k1 i≤k si cos h, hi , and the score NNGuide(x) = Sbase (h) · G(h). In practice,
                                           P                 

Sbase is the (negative) Energy score, but NNGuide can improve other classifier-based scores. Intuitively, G(h) downscales
overconfident far-OOD points where cosine similarities are small and preserves near-ID points using high-confidence
neighbors.

A.3.7. FAST D ECISION B OUNDARY D ETECTOR ( F DBD) (L IU & Q IN , 2023)
fDBD scores a sample by how far its feature lies from the nearest class decision boundary, regularized by feature deviation
from the in-distribution mean. For each non-predicted class c ̸= m(x), the (unknown) distance in the feature space to the
                                                (wm −wk )⊤ h+(bm −bk )
c-boundary is lower bounded by Dg (h, k) =      ∥ wm −wk ∥2       , i.e., the Euclidean distance from h to the separating
hyperplane between classes m(x) and c. Averaging these distances and regularizing by the feature’s deviation from the ID
                                         1
                                           PC           Dg (h,k)
mean µtrain yields the score fDBD(x) = C−1     k=1    ∥ h−µ ∥2 ,. The regularizer compares ID/OOD at equal deviation
                                                  k̸=m(x)          train

levels, empirically sharpening separation; the distance term captures that ID features tend to reside farther from decision
boundaries than OOD features.

A.3.8. PREDICTIVE N ORMALIZED M AXIMUM L IKELIHOOD ( P NML) (B IBAS ET AL ., 2021)
pNML treats a deep network as a fixed feature extractor and for each test samples computes a regret score by simulating
in closed form the best last-layer update for every possible label. Given the matrix of normalized penultimate-layer
training activations Ĥ = [h1 /∥h1 ∥2 , . . . , hN /∥hN ∥2 ], the online–update direction g via the kernel–range projection is
                                   +                            +          +⊤                      +
                                                              Ĥ Ĥ             h
g = h⊥ /∥h⊥ ∥22 if h⊥ = (I − Ĥ Ĥ)h ̸= 0; else g =                +   +⊤
                                                                                    , where Ĥ          is the Moore–Penrose inverse of the
                                                            1+h⊤ Ĥ Ĥ    h
                                                                                PC                 pk
normalized training activations. The pNML regret is pNML(x) = log                   k=1            ⊤g
                                                                                                                and serves as an OOD/failure
                                                                                          pk +ph
                                                                                               k        1−pk
score (larger pNML ⇒ less trustworthy prediction). Intuitively, pNML is small when h lies in the high-variance ID subspace
or is far from decision boundaries (the genie’s label-specific update has little effect), and large otherwise.

                                                              13

## Page 14

A Systematic Analysis of Out-of-Distribution Detection

A.3.9. G RAD N ORM (H UANG ET AL ., 2021)
Given a trained classifier with softmax p(x), GradNorm defines the OOD score as the vector norm of the gradients obtained
by backpropagating the Kullback–Leibler divergence from a uniform target, i.e. GradNorm(x) = ∂w KL(u ∥ p(x)) p =
  1
    PC ∂LCE (g(h),k)
 C    k=1       ∂w         , typically using the L1 norm on the last-layer weights. This choice is label-agnostic and exploits
                         p
that in-distribution inputs produce more peaked predictions and thus larger gradients than OOD inputs. A simple analysis
shows GradNorm(x) factorizes into a feature-space term and an output-space term, capturing joint information that
improves separability over output-only scores.

A.3.10. PCA R ECONSTRUCTION E RROR (PCA R EC E RROR ) (G UAN ET AL ., 2023)
PCA Reconstruction Error models the in-distribution feature manifold by fitting a low-dimensional principal subspace on
penultimate features and scores a test example by the energy of its component orthogonal to that subspace, so larger residuals
indicate atypicality. This approach computes the ID mean µ and covariance Σ, then takes the top-k eigenvectors U k of      Σ,
and forms the projector M = U k U ⊤  k  . The PCA reconstruction  error   for a test point is e(x) = (I − M )  h(x)  −  µ   2
                                                                                                                              ,
i.e., the energy of the component orthogonal to the ID principal subspace. Although intuitively e(x) should be smaller on
ID than OOD, a detailed analysis shows that e(x) (i) mixes the angle between h(x) − µ and the principal subspace and
(ii) the norm ∥h(x) − µ∥2 , which is typically larger for ID than OOD; this blurs separability for vanilla PCA-OOD. To
mitigate the norm effect, a simple regularized score r(x) = ∥h−      ĥ∥2                                   
                                                                  ∥h∥2 , where ĥ = ĥ(x) = M h(x) − µ + µ, improves
discrimination, and can be fused multiplicatively with logit-based scores.

A.3.11. K ERNEL PCA R ECONSTRUCTION E RROR (KPCA R EC E RROR ) (FANG ET AL ., 2025)
KPCA Reconstruction Error models the in-distribution (ID) feature manifold in a non-linear subspace and scores a test point
by its reconstruction error in that subspace. To mitigate feature–norm imbalance and preserve useful Euclidean
                                                                                                              relations,
                                                                                                     ′
KPCA first ℓ2 –normalizes features ĥ = h/∥h∥2 and define a Gaussian kernel on the unit sphere k(x, x ) = exp − 2σ1 2 ĥ−
  ′ 2                                ′ 
                                       
ĥ 2 = exp − σ12 1 − cos(ĥ, ĥ ) , which can be viewed as a Cosine–Gaussian composition. Given ID training points,
the centered Gram matrix is defined as K c = HKH with K ij = k(xi , xj ) and H = I − n1 11⊤ . Using the centered Gram
matrix, KPCA solves the eigenproblem K c αm = n λm αm , where          P m = 1, . . . , N,P
                                                                                          and defines principal
                                                                                                             P coordinates
for a test point x via the centered kernel kc (x, xi ) = k(x, xi ) − n1 j k(x, xj ) − n1 j k(xj , xi ) + n12 jℓ k(xj , xℓ ):
                    PN
ϕm (x) = √λ1           i=1 αmi kc (x, xi ). The squared reconstruction error in feature space after projecting onto the top
                 m
                                       Pk
k components is e(x) = kc (x, x) − m=1 ϕm (x)2 . Similar to PCApReconstruction Error, the larger e(x) is, the more
atypical x becomes. A norm–regularized variant KPCA(x) = e(x)/ kc (x, x) further reduces residual norm confounds.
To avoid computing the full N × N kernel and O(N 2 ) memory, we approximate the Gaussian on the sphere with an
                                                               ′
explicit map ψ : RD → RM so that k(x, x′ ) ≈ ψ(ĥ)⊤ ψ(ĥ ) with M ≪ N . In particular, we use Nyström features with
                                                                                        −1/2
landmarks {x⋆ℓ }Mℓ=1 (e.g., low-energy ID points near the boundary), ψ(z̃) = C W             where Cℓ = k(x, x⋆ℓ ) and W is
the landmark Gram matrix. We then perform ordinary PCA on ψ(ĥ): compute mean µ and top-k eigenvectors Uk of the
                                                                                                                     2
empirical covariance, and score a test point by the Euclidean reconstruction error ẽ(x) = I − U k U ⊤
                                                                                                        
                                                                                                      k   ψ(ĥ) − µ 2 , and
           ẽ(x)
r̃(x) = ∥ψ(   ĥ)∥2
                    . Empirically, the Cosine–Gaussian kernel and the low-energy Nyström approximation improve separability
and efficiency over linear PCA and nearest-neighbor baselines.

A.3.12. R ESIDUAL P ROJECTION AND V IRTUAL M ATCHING L OGIT (V I M) (WANG ET AL ., 2022)
Residual Projection score. If the ID principal subspace P ⊂ RN from training features is defined as the span of the top-D
eigenvectors of H ⊤ H, then R ∈ RN ×(N −D) have columns spanning P ⊥ . The residual projection of x is r(x) = RR⊤ h,
and the residual projection score is Residual(x) = ∥r(x)∥2 , which increases as the feature departs from the ID principal
subspace. This score is class-agnostic and leverages feature-space geometry that is lost when projecting to logits.

ViM (Virtual-logit Matching). ViM fuses the class-agnostic residual with class-dependent logits by creating a virtual
(K +1)-st logit from the residual and matching its scale to the real logits. ViM score is defined as the softmax probability of
                                         exp{ℓ0 (x)}
this virtual class: ViM(x) = PK exp{g(x)        }+exp{ℓ (x)}
                                                              , where the virtual logit ℓ0 (x) = α ∥r(x)∥2 , and the scaling
                                   k=1         k        0



                                                              14

## Page 15

A Systematic Analysis of Out-of-Distribution Detection
                                                     
factor α = Ex∼ID maxk≤K fk (x) /Ex∼ID ∥r(x)∥2 . Equivalently, applying the transformation t(x) = − ln(1/x − 1)
                                       PK g(h)k
yields t(ViM(x)) = α ∥r(x)∥2 − log k=1 e             , i.e., a residual term minus the Energy of the logits. Thus ViM is large
when the residual is large and the (ID) logits are small.

A.3.13. N EURAL C OLLAPSE (N E C O ) (A MMAR ET AL ., 2023)
This method is motivated by the Neural Collapse phenomena (Papyan et al., 2020), which unveils geometric properties
that manifest at the end of the training process. NeCo’s new observation eatblishes ID/OOD orthogonality, which implies
that OOD features concentrate near the origin after projection onto the ID subspace. This method fits PCA on ID features
to obtain the top-d principal directions P ∈ RD×d (orthonormal columns). Then it scores an input by the normalized
                                                                            ⊤
projection of its feature onto the ID principal subspace, NeCo(x) = ∥P∥h∥h∥   2
                                                                                2
                                                                                  , so that ID points (well aligned with the
ID subspace) yield larger scores, while OOD points (near-orthogonal to that subspace) yield smaller scores. This score is
optionally calibrated by multiplying with MLS to inject class-scale information.

A.4. Methods Variations
As described in section 3.2, the global and class subspaces, P and P c respectively, allow multiple projections that can be
used to modify existing OOD detection methods. The following list shows possible modifications to the feature, logit and
probability spaces, and table 3:

   • Global projection: ĥ = P P ⊤ (h − µtrain ) + µtrain
                            c
   • Class projection: ĥ = P c P c ⊤ (h − µctrain ) + µctrain
                                       avg          PC         c
   • Class averaged projection: ĥ           = C1        c=1 ĥ

                                        pred        ŷ
   • Class predicted projection: ĥ         = ĥ , where ŷ = m(x)
                                                                                c
   • Class projected logit: ĝ class = ĝ 1 , . . . , ĝ C , where ĝ c = w⊤
                                                         
                                                                           cL ĥ + bcL

   • Global projected probabilities: p̂ = softmax(g(ĥ))

   • Class projected probabilities: p̂class = softmax(ĝ class )
                                                                    pred
   • Class-predicted probabilities: p̂pred = softmax(g(ĥ                  ))
                                                                   avg
   • Class-averaged probabilities: p̂avg = softmax(g(ĥ                  ))


B. CLIP-based OOD Aggregation
Let ϕimg and ϕtext be fixed CLIP encoders (Radford et al., 2021). For any image x, define the ℓ2 -normalized image
embedding z = ϕ̃img (x) = ϕimg (x)/∥ϕimg (x)∥2 ∈ Rd . Given an in-distribution set DID = {xi , yi }N
                                                                                                   i=1 and a candidate
OOD set DOOD = {x′j }m j=1 , we extract Ztrain = {z i }n
                                                       i=1 and ZOOD = {z ′ m
                                                                          }
                                                                         j j=1 under identical preprocessing.
Global distances. We summarize each set by its empirical Gaussian in CLIP space with means and covariances (µID , ΣID )
and (µOOD , ΣOOD ), and compute
                             the Fréchet distance (FD) (Dowson &     Landau, 1982; Fréchet, 1957): FD2 (DOOD →
                                                     1/2         1/2 1/2
DID ) = ∥µID − µOOD ∥22 + Tr ΣID + ΣOOD − 2 ΣID ΣOOD ΣID                  .

As a second global measure, we compute the (unbiased) maximum mean discrepancy (MMD) (Gretton et al., 2006)
                                                                2
with a polynomial kernel k(u, v) = (u⊤ v + c)d : MMD                   1                              1                ′     ′
                                                                            P                              P
                                                                                                            j̸=j ′ k(z j , z j ′ ) −
                                                           \ =
                                                                              i̸=i′ k(z i , z i ) + m(m−1)
                                                                                               ′
                                                                    n(n−1)
 2                  ′
    P
nm    i,j k(z i , z j ). Both quantities are evaluated on CLIP embeddings; smaller values indicate that DOOD is closer to the
ID manifold.
Class-aware distances.       P For ID    class Pc       ∈                     {1, . . . , K}, define the (normalized) image-
                        1                   1
prototype µc   =       |Dc |         z
                              i:yi =c i    |Dc |         z
                                                  i:yi =c i 2                   and the (normalized) text prototype tc      =

                                                                     15

## Page 16

Table 3. OOD Detection methods baselines and variations. This table synthesizes multiple variations using our proposed Projection Filtering approach for all the OOD detection
                                                         methods considered in this work.
                                                                     Score              Unmodified                                Global                                   Class                                     Class Pred                                           Class Avg
                                                                                                                                                                                            k                                            pred                                                   avg
                                                                     CTM              maxk sim(wk , h)                      maxk sim(wk , ĥ)                       maxk sim(wk , ĥ )                         maxk sim(wk , ĥ                  )               maxk sim(wk , ĥ                     )
                                                                                                                                                                                    k                                           pred                                                avg
                                                                  CTM(mean)           maxk sim(µk , h)                      maxk sim(µk , ĥ)                       maxk sim(µk , ĥ )                         maxk sim(µ , ĥ )
                                                                                                                                                                                                                           k                                        maxk sim(µk , ĥ )
                                                                                                                                                                                                                              pred 
                                                                                                                                                                                                                                                                                              avg
                                                                                                                       −T log k=1 eg(ĥ)k /T                                                                                                                                            g (ĥ       )k /T
                                                                                       PC                                    PC                                              PC        class                         PC    g ĥ         /T                                PC
                                                                    Energy       −T log k=1 eg(h)k /T                                                               −T log     k=1 e
                                                                                                                                                                                    ĝ k
                                                                                                                                                                                                          −T log      k=1 e
                                                                                                                                                                                                                                      k                       −T log            k=1 e
A Systematic Analysis of Out-of-Distribution Detection




                                                                     MSR                   maxk pk                             maxk p̂k                                 maxk p̂kclass                            maxk p̂kpred                                          maxk p̂kavg
                                                                                                                                                                                                                          pred
                                                                                                                                                                                                                                                                        avg 
                                                                     MLS                maxk g(h)k                            maxk g ĥ                              maxk ĝ kclass                          maxk g ĥ                                               maxk g ĥ
                                                                                                                                                 k                                                                               k                                                              k
                                                                                       PC                                    PC                               PC                                    PC       pred
                                                                                                                                                                                                                                                            PC
                                                                                                                                                                       class
                                                                                                                                                                               log p̂kclass                              log p̂kpred                                        p̂kavg log p̂kavg
                                                                                                                                                                                                                                                                                             
                                                                      PE                 k=1 pk log pk                         k=1 p̂k log p̂k                  k=1 p̂k                                   k=1 p̂k                                                   k=1
                                                                                      PM                                    PM                               PM  class γ h                γ i     PM  pred γ h                γ i                    PM                 γ            γ 
                                                                     GEN                    γ       γ
                                                                                       k=1 pk (1 − pk )
                                                                                                                                   γ        γ
                                                                                                                             k=1 p̂k (1 − p̂k )               k=1 p̂k          1 − p̂kclass             k=1 p̂k          1 − p̂kpred                          k=1      p̂kavg       1 − p̂kavg
                                                                                                PM                                    PM                               PM  class γ                              PM  pred γ                                               PM                     γ
                                                                     REN               1
                                                                                      1−γ log
                                                                                                          γ
                                                                                                     k=1 pk
                                                                                                                             1
                                                                                                                            1−γ log
                                                                                                                                                γ
                                                                                                                                          k=1 p̂k
                                                                                                                                                                1
                                                                                                                                                               1−γ log     k=1 p̂k
                                                                                                                                                                                                          1
                                                                                                                                                                                                         1−γ log     k=1 p̂k
                                                                                                                                                                                                                                                                 1
                                                                                                                                                                                                                                                                1−γ log           k=1     p̂kavg
                                                                                        PC                                     PC                                   P  C         class                          P C        pred                                        PC      avg
                                                                      GE                 k=1 kp(k)                              k=1 kp̂(k)                             k=1 kp̂(k)                                 k=1 kp̂(k)                                            k=1 kp̂(k)




                                                                                                                                                                                                                                                                                                            16
                                                                                               PC                                    PC                                   PC             2                     PC              2                                        PC                    2
                                                                      PCE              − log          2
                                                                                                 k=1 pk                      − log           2
                                                                                                                                       k=1 p̂k                    − log      k=1 p̂kclass                 − log     k=1    p̂pred
                                                                                                                                                                                                                             k                                      − log        k=1     p̂kavg
                                                                                                                                      ⊤
                                                                                                                                                                                                                ⊤                                           avg ⊤ avg
                                                                                                                                            −1                                                              pred                      pred
                                                                                                ⊤                                                                                                                        pred
                                                                                                                                                                                                                                                                               avg 
                                                                     Maha             maxk hk Σ−1 hk                        maxk ĥk Σ̂ ĥk                                                         maxk ĥ k        (Σ̂ )−1 ĥ k                          maxk ĥ k   (Σ̂ )−1 ĥ k
                                                                                                                                 
                                                                   NNGuide               E(h)G(h)                            E ĥ G ĥ                                                                          E(hpred )G(hpred )                                   E(havg )G(havg )
                                                                                                                                                                                                                                              pred                                                avg
                                                                                 1
                                                                                    PC        Dg (h,k)                 1
                                                                                                                          P         Dg (ĥ,k)                                                             1
                                                                                                                                                                                                                 P              Dg (ĥ               ,k)       1
                                                                                                                                                                                                                                                                     P         Dg (ĥ ,k)
                                                                     fDBD              k=1                                   k=1                                                                                       k=1                                              k=1      avg  avg
                                                                                                                           k̸=m(x) ∥ĥ−µ̂train ∥                                                                                                                      k̸=m(x) ∥ĥ −µ̂train ∥
                                                                                C−1          ∥h−µtrain ∥              C−1                                                                                C−1                     pred
                                                                                                                                                                                                                                 ĥ             pred
                                                                                                                                                                                                                                             −µ̂train         C−1
                                                                                     k̸=m(x)                                                                                                                         k̸=m(x)
                                                                                                                                                                                                                                      pred                                                avg
                                                                                      PC              pk                    PC              p̂k                                                          PC             p̂                                     PC            p̂
                                                                     pNML       log     k=1 p +ph     ⊤   g
                                                                                                                      log     k=1 p̂ +p̂h   ⊤   g
                                                                                                                                                                                                      log k=1 pred pred hk ⊤ g      pred                    log k=1 avg avg hk⊤ g      avg
                                                                                             k              (1−pk )                 k             (1−p̂k )                                                    p̂k +(p̂k )      (1−p̂k    )                         p̂k +(p̂k )    (1−p̂k   )
                                                                                                k                                       k
                                                                                                                                                                                                                              
                                                                                  1
                                                                                       PC ∂LCE (g(h),k)                 1
                                                                                                                            PC ∂LCE (g(ĥ),k)                                                             PC ∂LCE g ĥpred ,k
                                                                                                                                                                                                           1                                                    1
                                                                                                                                                                                                                                                                    PC          ∂LCE (g (ĥ
                                                                                                                                                                                                                                                                                            avg
                                                                                                                                                                                                                                                                                                    ),k)
                                                                   GradNorm       C     k=1    ∂w                       C    k=1    ∂w                                                                     Ck=1         ∂w                                      C      k=1            ∂w
                                                                                                                                                                                        k                                      pred
                                                                                                                                     ∥h−ĥ∥                                        h−ĥ                                 h−ĥ
                                                                   PCA Error                                                     −    ∥h∥                          maxk −  ∥h∥                                −    ∥h∥
                                                                                                                                                Φ
                                                                                                                                                                              Φ k                                  Φ pred
                                                                  KPCA Error                                                 − Φ(h) − ĥ                       maxk − Φ(h) − ĥ                            − Φ(h) − ĥ
                                                                      ViM             α R̃h + E (h)
                                                                                            ∥P̃ h∥
                                                                     NeCo                      ∥h∥
                                                                    Residual                   R̃h
                                                                   Confidence                  sW

## Page 17

A Systematic Analysis of Out-of-Distribution Detection

 1
     PL                          1
                                       PL
 L    ℓ=1 ϕtext (promptℓ (c))      L   ℓ=1 ϕtext (promptℓ (c)) 2 (using prompt ensembling for class c).   For a test
embedding z , define the nearest-centroid cosine distance dNC (z ′ ) = 1 − maxc≤K z ′⊤ µc , and
              ′
                                                                                                 Pm   the image–text
cosine distance dIT (z ′ ) = 1 − maxc≤K z ′⊤ tc . Aggregate per-dataset by averaging: dNC = m  1               ′
                                                                                                    j=1 dNC (z j ) and
       1
         P  m          ′
dIT = m     j=1 dIT (z j ). Lower values mean the OOD set is class-closer to ID.

Clustering into proximity buckets. We orient all four metrics so that lower ⇒ closer and form a feature vector v(DOOD ) =
              2
         \ , dNC , dIT ⊤ . We standardize v across candidate OOD sets (z-score per coordinate) and run k-means with
  FD2 , MMD
                           

k = 3 (fixed seed) to obtain proximity buckets labeled near/mid/far. This CLIP-based protocol is detector-agnostic and
applies unchanged to any ID label space or downstream OOD scoring rule.

C. Statistical Tests for Clique Generation
Let k be the number of CSFs and N the number of complete blocks (e.g., each block is a dataset/condition on which all k
methods are evaluated on the same metric). Within block i ∈ {1, . . . , N }, rank the methods so that rij ∈ {1, . . . , k} is the
                                                                               PN
rank of method j (use mid-ranks for ties). Define the average rank R̄j = N1 i=1 rij . The Friedman statistic tests the null
H0 : all methods are equivalent in distribution of ranks (Fréchet, 1957; Demšar, 2006):
                                                            k
                                                    12N X 2
                                            Q =               R̄ − 3N (k + 1),
                                                  k(k + 1) j=1 j

(optionally applying a standard tie correction within blocks). For finite samples, the Iman–Davenport F -approximation is
recommended (Iman & Davenport, 1980):
                                                 (N − 1) Q
                                       FF =                  ∼ Fk−1, (k−1)(N −1) .
                                               N (k − 1) − Q
If FF exceeds the critical value at level α, we reject H0 and proceed with post-hoc pairwise comparisons.

Conover post-hoc & Bron–Kerbosch cliques (top groups). For each pair (i, j) we compare average ranks using
Conover’s post-hoc test (Conover, 1999). With common standard error
                                              r
                                                 k(k + 1)                  |R̄i − R̄j |
                                      SE =                  ,     Tij =                 ,
                                                    6N                         SE
two-sided p-values are obtained from the normal (or t) reference, and multiplicity is controlled across all k2 pairs using
                                                                                                               

Holm’s step-down procedure (Holm, 1979). To summarize statistically indistinguishable winners, construct an indifference
graph G = (V, E) with nodes V = {1, . . . , k} (methods) and edges (i, j) ∈ E iff the adjusted pij ≥ α (i.e., the pair is
not significantly different). Enumerate all maximal cliques of G using the Bron–Kerbosch algorithm with pivoting (state
(R, P, X); recursively add v ∈ P \ N (u) for a high-degree pivot u; output R when P = X = ∅) (Bron & Kerbosch, 1973).
Each maximal clique is a set of methods that are mutually indistinguishable under Conover–Holm; reporting the leading
clique(s), sorted by best/mean R̄, yields layered, statistically justified “top groups,” alongside R̄j and the full adjusted
p-matrix for transparency.

C.1. Simplified example
To illustrate the evaluation methodology, we consider a simplified scenario involving k = 6 Confidence Scoring Functions
(CSFs): Confidence, GEN, MSR, CTM, fDBD, and Energy. We evaluate these models on the CIFAR-10 dataset using the
AUGRC metric, focusing solely on the misclassification scenario. The evaluation aggregates results across three training
paradigms, each executed five times, resulting in a total of N = 15 experimental blocks.
As shown in Table 5a, within each block (characterized by a unique combination of dataset, paradigm, metric, and run), the
CSFs are ranked based on their performance scores. Lower ranks indicate superior performance. These rankings are then
averaged across all N = 15 blocks, as summarized in Table 5b.
Using these average ranks, we compute the Friedman statistic, yielding FF = 46.162 with a p-value = 8.418 × 10−9 .
Since this p-value falls below the significance level α = 0.05, we reject the null hypothesis and conclude that significant
differences exist between the CSFs. Consequently, we proceed with post-hoc pairwise comparisons.

                                                               17

## Page 18

(a) CLIP-based distance metrics. Dataset: CIFAR-10                                  (b) CLIP-based distance metrics. Dataset: SuperCIFAR-100
                                                                                    Global                      Class-aware          Group                                 Global                      Class-aware          Group
                                                                          Kernel MMD          FD        Label-Text          Image                                Kernel MMD          FD        Label-Text          Image
                                                                                                        Alignment         Centroid                                                             Alignment         Centroid
                                                                                                                          Distance                                                                               Distance
                                                                  Test                                                                ID              Test                                                                   ID
                                                                         -0.0000         0.0028       0.7183           0.6349                                   0.0000          0.0748       0.7581           0.7031
                                                           CIFAR-100                                                                 Near        CIFAR-10                                                                    Near
A Systematic Analysis of Out-of-Distribution Detection




                                                                         0.0002          0.1592       0.7885           0.8085                                   0.0002          0.1705       0.7701           0.7511
                                                          TinyImagenet                                                               Near     TinyImagenet                                                                   Near
                                                                         0.0009          0.3233       0.8060           0.9256                                   0.0008          0.2307       0.7840           0.8738
                                                                 iSUN                                                                Mid             iSUN                                                                    Mid
                                                                         0.0015          0.4890       0.7943           0.8393                                   0.0012          0.3856       0.7607           0.7425
                                                          LSUN resize                                                                Mid       LSUN resize                                                                   Mid
                                                                         0.0016          0.5248       0.8045           0.8634                                   0.0013          0.4244       0.7625           0.7720
                                                         LSUN cropped                                                                Mid     LSUN cropped                                                                    Mid
                                                                         0.0015          0.5129       0.7797           0.8168                                   0.0011          0.3999       0.7696           0.7351
                                                                SVHN                                                                 Mid            SVHN                                                                     Mid
                                                                         0.0020          0.7009       0.7744           0.8607                                   0.0017          0.6208       0.7566           0.8012
                                                            Places 365                                                                Far       Places 365                                                                   Far
                                                                         0.0021          0.6379       0.8337           1.1471                                   0.0020          0.5562       0.7939           1.0964
                                                              Textures                                                                Far         Textures                                                                   Far
                                                                         0.0020          0.6698       0.8231           1.0647                                   0.0016          0.5246       0.7980           1.0003




                                                                                                                                                                                                                                    18
                                                                         (c) CLIP-based distance metrics. Dataset: CIFAR-100                                  (d) CLIP-based distance metrics. Dataset: TinyImagenet
                                                                                    Global                      Class-aware          Group                                Global                      Class-aware           Group
                                                                          Kernel MMD          FD        Label-Text          Image                               Kernel MMD          FD        Label-Text          Image
                                                                                                        Alignment         Centroid                                                            Alignment         Centroid
                                                                                                                          Distance                                                                              Distance
                                                                  Test                                                                ID              Test                                                                   ID
                                                                         -0.0000         0.0033       0.7043           0.6026                                   -0.0000        0.0036        0.7141          0.6319
                                                            CIFAR-10                                                                 Near      CIFAR-100                                                                    Near
                                                                         0.0002          0.1590       0.7494           0.7268                                   0.0008         0.2224        0.7279          0.7956
                                                          TinyImagenet                                                               Near       CIFAR-10                                                                    Near
                                                                         0.0008          0.2235       0.7512           0.8436                                   0.0009         0.3220        0.7288          0.7979
                                                                 iSUN                                                                Mid            iSUN                                                                    Near
                                                                         0.0012          0.3829       0.7484           0.7128                                   0.0012         0.3808        0.7468          0.7063
                                                          LSUN resize                                                                Mid      LSUN resize                                                                   Near
                                                                         0.0013          0.4204       0.7562           0.7388                                   0.0013         0.4039        0.7500          0.7186
                                                         LSUN cropped                                                                Mid     LSUN cropped                                                                   Near
                                                                         0.0011          0.3999       0.7364           0.7120                                   0.0016         0.4989        0.7406          0.7503
                                                                SVHN                                                                 Mid        Places 365                                                                  Mid
                                                                         0.0017          0.6222       0.7511           0.7789                                   0.0014         0.3887        0.7645          0.9846
                                                            Places 365                                                                Far         Textures                                                                  Mid
                                                                         0.0019          0.5456       0.7752           1.0568                                   0.0014         0.4697        0.7528          0.9317
                                                              Textures                                                                Far          SVHN                                                                      Far
                                                                         0.0016          0.5232       0.7613           0.9780                                   0.0025         0.7948        0.7409          0.8726

## Page 19

A Systematic Analysis of Out-of-Distribution Detection

                                        Table 5. Friedman test across entities with blocks.
                                   (a) Block example                                                       (b) Averaged rankings
      Dataset    Paradigm      Metric       Run         CSF                    Score      Rank           CSF           Avg Rank
      test       confidnet     AUGRC        1           Confidence             4.453      0.17           GEN           0.278
      test       confidnet     AUGRC        1           GEN                    4.619      0.33           Confidence    0.367
      test       confidnet     AUGRC        1           MSR                    4.737      0.50           MSR           0.511
      test       confidnet     AUGRC        1           CTM                    5.479      0.67           CTM           0.656
      test       confidnet     AUGRC        1           fDBD                   5.677      0.83           fDBD          0.767
      test       confidnet     AUGRC        1           Energy                 5.811      1.0            Energy        0.922



We apply Conover’s post-hoc test to compare the average rankings, computing two-sided p-values with controlled multiplicity
via Holm’s step-down procedure (Holm, 1979) (see Figure 2). This analysis allows us to identify “cliques” or groups
of CSFs that are statistically indistinguishable from one another (pij ≥ 0.05). In this example, we identify five cliques:
Clique 1: [’CTM’, ’MSR’], Clique 2: [’CTM’, ’fDBD’], Clique 3: [’Confidence’, ’GEN’], Clique 4:
[’Confidence’, ’MSR’], and Clique 5: [’Energy’].
While visual identification of cliques is straightforward in this small-scale example, the combinatorial complexity increases
significantly with a larger number of CSFs. To address this, we employ the Bron-Kerbosch algorithm to systematically
identify maximal cliques in larger scenarios. Once identified, the cliques are ranked based on their constituent members.
Finally, we select non-overlapping cliques using a greedy layering approach, organizing them from best to worst performance.
The resulting hierarchical layers are presented in Table 6. For Figure 1, we only report the first layer for all the possible
scenarios.


                                                                                                   1.0
                                      CTM 1.000 0.000 0.001 0.000 0.052 0.133

                                Confidence 0.000 1.000 0.000 0.228 0.052 0.000                     0.8

                                    Energy 0.001 0.000 1.000 0.000 0.000 0.037                     0.6

                                      GEN 0.000 0.228 0.000 1.000 0.002 0.000                      0.4

                                      MSR 0.052 0.052 0.000 0.002 1.000 0.001                      0.2
                                     fDBD 0.133 0.000 0.037 0.000 0.001 1.000
                                                                                                   0.0
                                                CTM




                                                                              GEN

                                                                                    MSR

                                                                                            fDBD
                                                        Confidence

                                                                     Energy




                                                 Figure 2. Conover-Holm p-values



                                        Table 6. Layered cliques ranked from best to worst

                                        Layer         Average Rank              Members
                                        1             1.933                     Confidence, GEN
                                        2             3.500                     CTM, MSR
                                        3             5.533                     Energy



                                                                         19

## Page 20

A Systematic Analysis of Out-of-Distribution Detection

D. Neural-Collapse-based Analysis
We provide a geometric interpretation of our empirical findings through the lens of Neural Collapse (NC), focusing on the
last-layer feature space and classifier geometry. Taking into consideration the Definitions and Notations in Section 3.1, we
quantify proximity to NC using the metrics described by (Papyan et al., 2020):

                                                      Stdc (∥µ̃c ∥2 )
   • Equinormness of the class means (∥µi ∥ = ∥µj ∥): Avg  (∥µ̃ ∥2 ) c       c

                                                               Stdc (∥wc ∥2 )
   •   Equinormness of the classifier weights (∥wi ∥ = ∥wj ∥): Avg
                                                                   c (∥w c ∥2 )


   • Equiangularity of the class means (cosµ (i, j) = β): Stdc,c′ ̸=c (cosµ (c, c′ ))

   • Equiangularity of the classifier weights (cosw (i, j) = ρ): Stdc,c′ ̸=c (cosw (c, c′ ))
                                                                 1
   • Maximal Equiangularity of the class means (cosµ (i, j) = − C−1 ): Avgc,c′ |cosµ (c, c′ ) + C−1
                                                                                                 1
                                                                                                    |
                                                                        1
   • Maximal Equiangularity of the classifier weights (cosw (i, j) = − C−1 ): Avgc,c′ |cosw (c, c′ ) + C−1
                                                                                                        1
                                                                                                           |

                                                                ΣW Σ†
                                                                        
   • Within-class variation collapse (h → µk ): Tr                C
                                                                    B



                                                     T                2                                     2                   2
                                               W              M                             1    T    1                 1     T
   • Self-duality (wk = αµk ):                ∥W ∥F        − ∥M ∥F           =            ∥W ∥F W − ∥M ∥F M       =   ∥W ∥F W         +
                                                                        F                                     F                   F
                   2
          1                 Tr(W M )            Tr(W M )
        ∥M ∥F M        − 2 ∥W ∥F ∥M ∥F = 2 − 2 ∥W ∥F ∥M ∥F
                   F


                                                                                 µ̃T µ̃
where µ̃c = µc − µ is the class-centered mean, cosµ (c, c′ ) = ∥µ̃ ∥c2 ∥µ̃c′ ′ ∥2 is cosine similarity between any pair of class-
                                                                                 c        c
                                          wT w
centered means c and c , cosw (c, c ) = ∥wc ∥c2 ∥wc′ ′ ∥2 is cosine similarity between any pair of classifier weights c and c′ ,
                           ′          ′
                                                   c

ΣW = Avgi,c (hi,c − µ)(hi,c − µ)T is the within-class covariance, ΣB = Avgc (µ̃c µ̃Tc ) is the between-class covariance, †
is the Moore-Penrose pseudoinverse, and M = [µ̃c : c = 1, . . . , C] is the centered class-mean matrix.
We compute these metrics for each model and dataset under three activation regimes: unfiltered, global PCA projection
filtering, and class-predicted PCA projection filtering, and interpret how shifts in these metrics align with shifts in the
relative performance of confidence scoring functions. Tables 7a and 7b show the NC metrics for all the activation regimes,
considering CNN and ViT models respectively, and Table 7c contains the NC metrics associated to the classifier weights
and.

D.1. CNNs trained from scratch (VGG-13)
D.1.1. W HY IS CTM THE DETECTION METHOD THAT PERFORMS THE BEST WHEN THE MODEL HAS BEEN TRAINED
       ON T INY I MAGE N ET USING CNN S ?

As Table 7a shows, Maximal Equiangularity, Variability Collapse, and Self Duality are the strongest NC metrics when
TinyImageNet is used to train the VGG-13 models, while Equinormness and Equiangularity are the weakest. In these
conditions prototype-alignment becomes highly reliable, which can be exploited by the CTM score. When Self-Duality
occurs the classifiers align with the class means up to a scalar : W ∝ M ⊤ =⇒ wc = αµc , then CTM(x) =
                                  w⊤ h                      αµ⊤ h
maxk sim(wk , h) = maxk ∥wk ∥k2 ∥h∥2 = maxk ∥αµ ∥k2 ∥h∥2 . If Variability Collapse is achieved ΣW → 0, meaning
                                                                k
                                                                                                      µ⊤ h        µ⊤ µ
features x collapse to their class means µk : h → µk . Therefore, CTM(x) = maxk ∥µ ∥k2 ∥h∥2 ≈ ∥µ ∥k2 ∥µk ∥2 = 1, which
                                                                                       k            k     k
creates a deterministic score for ID samples. Even if Equiangularity is not achieved completely (meaning the gap between
Class A and Class B is different from Class A and Class C), the CTM score for an ID input is approximately 1. In the other
hand, Maximal Angular Margin dictates that class means form a Simplex ETF, maximizing the separation angle θij between
                                              1
any distinct classes i, j: cos(µi , µj ) = − K−1  ∀i ̸= j. This implies that the collapsed ID feature space is maximally
sparse in terms of angular distribution. For an OOD sample xOOD lying in the subspace orthogonal to the ID manifold
(or between vertices), the maximum cosine similarity is bounded by the geometry of the simplex. Unlike dense feature
spaces where an OOD point might accidentally align with a cluster, the ETF geometry ensures wide angular gaps (negative

                                                                     20

## Page 21

A Systematic Analysis of Out-of-Distribution Detection




                                      Table 7. Neural Collapse metrics
                                    (a) NC metrics for VGG-13 models.
Dataset               Projection     EqNorm        EqAng      max-EqAng       Var Collapse   Self Duality
                      None             0.0266      0.0651           0.2447         0.0081         0.0320
CIFAR-10              Global           0.0266      0.0651           0.2447         0.0082         0.0307
                      Class pred       0.0268      0.0650           0.2447         0.0080         0.0320
                      None             0.0246      0.0678           0.1537         0.0095         0.0519
SuperCIFAR-100        Global           0.0246      0.0679           0.1537         0.0095         0.0507
                      Class pred       0.0248      0.0678           0.1537         0.0093         0.0519
                      None             0.0743      0.1049           0.0950         0.0079         0.1279
CIFAR-100             Global           0.0745      0.1052           0.0952         0.0259         0.1232
                      Class pred       0.0743      0.1049           0.0950         0.0069         0.1278
                      None             0.0544      0.0900           0.0761         0.0021         0.0322
TiinyImageNet         Global           0.0546      0.0903           0.0763         0.0044         0.0335
                      Class pred       0.0546      0.0901           0.0762         0.0016         0.0322
                                       (b) NC metrics for ViT models.
Dataset               Projection     EqNorm        EqAng      max-EqAng       Var Collapse   Self Duality
                      None             0.0869      0.2505           0.3770         0.0120         1.2692
CIFAR-10              Global           0.0870      0.2505           0.3771         0.0120         1.2716
                      Class pred       0.0870      0.2505           0.3770         0.0120         1.2693
                      None             0.1432      0.2403           0.2814         0.0152         1.0922
SuperCIFAR-100        Global           0.1432      0.2403           0.2814         0.0153         1.0937
                      Class pred       0.1435      0.2406           0.2816         0.0150         1.0929
                      None             0.1054      0.1322           0.1172         0.0124         0.8535
CIFAR-100             Global           0.1058      0.1328           0.1177         0.0162         0.8566
                      Class pred       0.1065      0.1327           0.1177         0.0107         0.8548
                      None             0.1950      0.1984           0.1647         0.0372         1.1292
TiinyImageNet         Global           0.1971      0.2008           0.1665         0.0806         1.1429
                      Class pred       0.2069      0.2066           0.1716         0.0289         1.1416
                                   (c) NC metrics for classifier weights w.
                Model        Dataset                 EqNorm        EqAng      max-EqAng
                             CIFAR-10                  0.0143      0.0707         0.2455
                             SuperCIFAR-100            0.0184      0.0529         0.1414
                VGG-13
                             CIFAR-100                 0.0730      0.0611         0.0665
                             TiinyImageNet             0.0250      0.0748         0.0666
                             CIFAR-10                  0.0141      0.0372         0.2884
                             SuperCIFAR-100            0.0217      0.0360         0.1526
                ViT
                             CIFAR-100                 0.0274      0.0450         0.0551
                             TiinyImageNet             0.0178      0.0364         0.0390




                                                     21

## Page 22

A Systematic Analysis of Out-of-Distribution Detection

correlations) between prototypes, statistically minimizing maxk cos(wk , h). OOD samples, which lie in the angular gaps,
will strictly have scores < 1 as long as Maximal Angularity holds (the vectors point in distinct directions). CTM measures
alignment with prototype: hOOD ∈ Null Space =⇒ cos(µk , hOOD ) ≈ 0. An OOD sample cannot be perfectly parallel to
any prototype, so cos(µk , hOOD ) ≪ 1. In summary, CTM strips away the noise of magnitude (Equinorm failure) and relies
on the purity of direction (Maximal Angularity) and clustering (Variability Collapse).

D.1.2. W HY E NERGY, MLS AND NNG UIDE DETECTION METHODS PERFORM THE BEST WHEN THE MODEL HAS BEEN
       TRAINED ON CIFAR-10 AND S UPER CIFAR-100 USING CNN S ?

In the context of CIFAR-10, Equinormness, Equiangularity, Variability Collapse, and Self-Duality emerge as the most
robust metrics, whereas Maximal Equiangularity demonstrates the least adherence to a collapsed regime, as Table 7a
exhibits. SuperCIFAR-100 shows a similar metrics profile to CIFAR-10 with an improved Maximal Equiangularity, but
slight deterioration in Self-Duality. In both NC regimes, logit-derived scores such as Energy and MLS benefit from stable,
class-unbiased logit scaling induced by Equinormness and the differentiated target vs. non-target angular margin implied
by Equiangular geometry. More specifically, Self-Duality implies the classifiers weights W align with the class means
M up to a scalar α: W ∝ M ⊤ =⇒ wc = αµc and Equinorm states that any pair of class means c and c′ have equal
ℓ2 norm: ∥µc ∥2 = ∥µc′ ∥2 = R. Substituting these into the logit g(h)k for a correctly classified ID sample (target logit)
h → µc : g(h)k = w⊤                   ⊤               2       2
                       k h + bk ≈ w k µk = α∥µk ∥ = αR . Thus, the minimum possible energy for ID data is uniform
                                          2
across all classes: Energy(x) ≈ −αR . This uniformity prevents class-conditional bias, where some ID classes might
otherwise have naturally higher energy (and thus higher False Positive Rates) than others due to varying feature norms.
Equiangularity dictates that any pair of class means are equally spaced cosu (i, j) = β, ∀i ̸= j, meaning that for an off-target
logit g(h)j = w⊤                ⊤               2
                  j h + bk ≈ w j µk = α∥µk ∥ β = αR β.
                                                          2


Let’s analyze the behavior of the detection scores under these conditions. The Energy score sums over the exponentiated
                       2                 2
                                                                                                          1
       P g(h)k
logits   e       = eαR + (K − 1)eαR β , which in the optimal case (Maximal Equiangularity β = − C−1          ) would be
                    2           2                  2
dominated by eαR since eαR ≫ (C − 1)eαR β or αR2 > ln(C−1)
                                                     1−β . This last expression reveals that β allows some
tolerance to make the target logit distinguishable from the off-target logits as β < 1 − ln(C−1)    αR2   when evaluating ID
inputs. This explains why the Energy score is still very effective at detecting OOD inputs in this scenario where Maximal
Equiangularity is not the strongest. However, the failure moded for this score occurs when either the OOD sample falls in
the geometric of a cluster that aligns with all C classes, provoking an artificially high score, or when the norm for an OOD
sample is anomalously large. The poor performance of the Energy score in far OOD scenarios seems to indicate that the
latter is the problem, which can be validated by the performance of MLS. The MLS score, unlike the Energy score, ignores
off-target logits entirely since MLS(x) = maxk≤C g(h)k . Thus, as long as the classes are distinct (β < 1), the target logit
is strictly greater than the non-target logits given that MLS(x) = max(αR2 , αR2 β). Assuming that the class means are not
clustered together, the failure mode of MLS lies on its dependence on feature magnitude. Given the similar performance of
the Energy score and MLS when CIFAR-10 is the training set, we argue that OOD norms can be differentiated from ID
norms in this case. Model trained SuperCIFAR-100, in the other hand, seems to be more vulnerable to atypical large norms,
specially when far OOD datasets are tested.

D.1.3. W HY NNG UIDE PERFORMS WELL WHEN THE MODEL HAS BEEN TRAINED ON CIFAR-10 USING CNN S ?
NNGuide acts as a confidence-weighted geometric filter that corrects the summation bias inherent in the Energy score when
classes are clustered (non-maximal equiangularity). In this regime, the Energy score could become unreliable because the
summation aggregates multiple medium-strength logits from clustered classes. If an OOD sample lies in the geometric
center of a cluster with cosine similarity β > 0, it activates all K classifiers moderately, resulting in a partition sum that can
rival the single high-confidence logit of an ID sample, leading to false positives. NNGuide modulates this by multiplying the
base scorewith a guidance term G(h), computed
               Pk                               from the k nearest training neighbors:NNGuide(x) = Sbase (h) · G(h) =
             1
Sbase (h) · k i=1 Sbase (h(i) ) · cos(h, h(i) ) . This term leverages the Self-Duality of Neural Collapse, where the nearest
neighbors h(i) for valid samples are high-confidence prototypes (Sbase (h(i) ) ≈ Smax ), effectively turning G(h) into a clean
geometric penalty.
NNGuide succeeds in the clustered regime (β > 0) because the guidance term enforces an angular margin that the raw
Energy score ignores. For an ID sample aligned with its class direction, the cosine similarity is near 1, yielding a guidance
factor of G(h) ≈ Smax · 1. Conversely, for an OOD sample in the angular gap between clustered vectors, the cosine
similarity to the nearest neighbors is bounded by the cluster geometry: G(hOOD ) ≈ Smax · β. Even if the base Energy scores

                                                               22

## Page 23

A Systematic Analysis of Out-of-Distribution Detection

are indistinguishable (Sbase (hOOD ) ≈ Sbase (h)) due to summation explosion, NNGuide suppresses the OOD score by the
                    OOD
                        )  Sbase (hOOD )·β
factor β: NNGuide(x
            NNGuide(x) ≈ Sbase (h)·1 = β. As long as the classes are not perfectly overlapping (β < 1), this multiplicative
correction restores the separability that the base score lost. However, NNGuide succumbs to failure modes where this
geometric penalty is either insufficient or overridden. The first mode is Tight Clustering (β → 1), where classes become so
semantically similar that the angular margin vanishes; here, the penalty factor β approaches 1, rendering the guidance term
useless (G(hOOD ) ≈ G(h)). The second and more critical failure mode is its sensitivity to feature magnitude. While the
guidance term G(h) is scale-invariant (due to cosine normalization), the base term Sbase (h) scales linearly with the feature
norm ∥h∥. If an OOD sample has an anomalously large magnitude where ∥hOOD ∥ = γ∥h∥ with γ > 1, the linear growth of
the base score overrides the constant geometric penalty: NNGuide(xOOD ) ≈ (γSbase ) · (Smax β).

D.1.4. W HY CTM MEAN GLOBAL AND NNG UIDE GLOBAL IMPROVE WITH RESPECT TO THEIR COUNTERPARTS THAT
       DO NOT USE THE GLOBAL FILTERING FOR MID -OOD DATASETS WHEN USING CNN S ?

As Table 7a shows, the NC metrics after Global Projection Filtering do not significantly change for the ID data. Therefore,
the improvement in detection cannot be attributed to the ID classes becoming tighter or more separated from each other
after filtering. Instead, the improvement comes from how the OOD data is transformed relative to the ID manifold after the
projection. Since the improvement occurs for CTMmean and NNGuide, we can attribute it to the angular information as
the main reason. Global Projection Filtering acts as a selective denoiser that benefits Mid-OOD detection by severing the
shared low-level statistics, like texture or color, that confuse angular detectors. Mid-OOD samples often reside in the same
positive feature cone as ID data. The projection operation P P ⊤ (hOOD − µ) centers the data and removes the low-variance
directions responsible for these spurious correlations. By stripping away these common-mode statistics, the projected
Mid-OOD vector effectively collapses toward the global mean, becoming angularly ambiguous relative to the distinct class
prototypes. This drastically lowers their cosine similarity scores in methods like CTMmean and NNGuide, reducing false
positives. In contrast, Far-OOD samples do not benefit from this refinement because they are already geometrically distinct.
These samples typically lie in the null space of the ID manifold or are orthogonal to the semantic class directions. Since
their angular separation from ID prototypes is already maximized (they look nothing like the training classes), the projection
step is redundant. Thus, the projection operation acts as an identity transformation regarding their detectability. However,
this projection does not solve the problem of atypically large norms from Far-OOD samples. The principal components P
often capture generic image statistics (e.g., contrast, brightness) shared by all natural images, not just semantic class features.
If a Far-OOD sample has an anomalously large norm due to domain shift (e.g., high saturation), its projection onto these
                                              OOD
generic axes preserves this magnitude: ∥ĥ      ∥ ≤ ∥P P ⊤ hOOD ∥ + ∥(I − P P ⊤ )µ∥. Consequently, magnitude-sensitive
detectors like Energy or MLS can still fail on Far-OOD data if the norm is large enough to override the angular mismatch,
necessitating the use of normalized scores like CTM (divides by ∥h∥) or fDBD (regularizes by ∥h − µ∥) which are invariant
to this preserved scale.

D.1.5. W HY F DBD PERFORMS WELL WHEN THE MODEL HAS BEEN TRAINED ON CIFAR-100 USING CNN S ?
The NC metrics for CIFAR-100 report the weakest adherence to a collapsed regime compared to the other datasets, with
Maximal Equiangularity for class means and for classifier weights being the strongest metrics. The lack of a strong
Variability Collapse, however, makes it difficult for CTM to be competitive in this case. fDBD succeeds because it exploits
the rigid geometric skeleton of the classifier weights to create a robust angular detector that is tolerant of feature noise.
The fDBD score is defined as the average distance to all non-predicted classes, regularized by the feature distance to
                                                  1
                                                      P           Dm (h,c)                          |(wm(x) −wj )⊤ h+(bm(x) −bj )|
the mean of training features: fDBD(x) = |C|−1          c̸=f (x) ∥h−µtrain ∥2 , where Dm (h, j) =          ∥wm(x) −wj ∥2           .
In this regime fDBD outperforms prototype methods because it measures the true safety margin relative to the decision
boundary, whereas prototype methods measure centrality relative to a potentially unsafe centroid. The sumation      of the fDBD
score is proportional to the projection of the mean onto the weight difference vectors: fDBD(x) ∝ j̸=k (wk − wj )⊤ h.
                                                                                                          P
When Self-Duality fails, the geometric center (µk ) and the decision center (wk ) diverge. If µk is close to the boundary
(small projection onto wk − wj ), fDBD correctly reports a low score, identifying the higher risk of misclassification that
CTM would ignore in this scenario. Even with misalignment, Maximal Angularity of W ensures P                the decision region
is a wide cone supported by the collective opposition of all other classes. Using the ETF property ( j̸=k wj = −wk ),
the fDBD summation simplifies: j̸=k (wk − wj )⊤ h = (C − 1)w⊤                                   ⊤         ⊤
                                    P                                               P
                                                                            kh−(      j̸=k w j ) h = C(w k h). As long as the
                                   ⊤
classifier is accurate (meaning wk h > 0), the ETF geometry amplifies the margin by a factor of C. fDBD captures this
                                                                                                                   OOD
collective margin, ensuring that even a misaligned ID sample is distinguished from OOD samples where w⊤        kh       ≈ 0. The


                                                                23

## Page 24

A Systematic Analysis of Out-of-Distribution Detection

regularization term improves the evaluation when atypically feature norms are atypically large.

D.2. Finetuned Transformers (ViT)
Unlike CNNs trained from scratch, ViTs that were pretrained and finetuned might not have null space for OOD inputs in
many cases. Since the ViT networks were pretrained using ImageNet1K, and finetuned using dataset with fewer number
of classes, it is likely that the OOD inputs would be projected to existing subspaces that were created during pretraining.
This can be corroborated by the NC metrics of ViT models compared to the NC metrics of VGG models. ViT’s NC metrics
are higher than their CNN’s counterparts. This phenomenon has been documented before (Zhou et al., 2025). When the
pretraining data has more classes than the finetuning data, then the finetuning process would readjust the boundaries around
the classes are the most similar, eg. if the finetuning data contains the class dog, and the pretraining data has classes dog,
fox, wolf, and coyotes, then the boundaries after finetuning would enclose all these classes. However when it comes to OOD
detection, the strategies cannot rely on the same approaches as models trained from scratch.

D.2.1. W HY THE GLOBAL PROJECTION FILTERING IMPROVES THE OOD SCORES THAT USE ENTROPY OR
       PROBABILISTIC OUTPUTS WHEN EVALUATING MISCLASSIFICATION FOR V I T MODELS THAT WERE
       PRETRAINED AND FINETUNED ?

In a finetuned ViT, the feature vector h has not fully collapsed. It contains a task-relevant component (htask ) that drives
the logits g, and a large residual component (hresidual ) inherited from ImageNet pretraining that lies in the null space of
the finetuning task: h = htask + hresidual . Since hresidual is orthogonal to the task weightsp W task , it does not affect the
final prediction ŷ for ID samples. However, it does affect the feature norm since ∥h∥2 = ∥htask ∥2 + ∥hresidual ∥2 and
the peakedness of the predictive distribution where hresidual often dominates for OOD samples. After global projection
filtering, the influence of the subspaces that are more related to the task is preserved, while the subspaces that are not that
relevant get discarded in the reconstruction P P ⊤hresidual ≈ 0. While this process does not improve the NC geometric
properties compared to their NC geometric properties that are computed when no filtering is applied, the probability outputs
will get more confident for the classes in the finetuning data, and in turn the probabilistic output will be better suited to
detect misclassified inputs.

D.2.2. W HY DOES G RADNORM GLOBAL AND G RADNORM CLASS PRED PERFORM MUCH BETTER THAN VANILLA
       G RADNORM IN THIS WHEN EVALUATING NEAR , MID , AND FAR OOD FOR V I T MODELS THAT WERE
       PRETRAINED AND FINETUNED ?

Vanilla GradNorm calculates the gradient of the KL divergence with respect to the last-layer weights W : GradNorm(x) =
                            PC
 ∂w KL(u ∥ p(x)) p = C1 k=1 ∂LCE (g(h),k)∂w        , typically using the L1 norm. This expression can be factorized into a
                                                 p
                                                                          PC
feature magnitude term U = ∥h∥1 and an output-gradient term V = j=1 |1 − C · pj |, which is the sum of absolute
differences between class probabilities and the uniform target.
As described previously, in a finetuned ViT, the raw feature h contains high-variance pretraining residuals inherited from
ImageNet, which act as stochastic perturbations in the logit space. These residuals can move an ID sample closer to
class boundaries or dilute the activation of the target class, spreading the softmax probabilities pj and artificially lowering
the output component V at the same time the residual component artificially inflates the feature magnitude term U . By
projecting features onto the principal subspace P , global projection filtering aggressive prunes these irrelevant directions
that do not support class-consistent structure. The filtered logit ĝ results in a higher dynamic range between the target class
and off-target classes for ID samples, naturally amplifying the peakedness of the predictive distribution and increasing the
discriminative score V . Conversely, for OOD samples, the projection P P ⊤ discard the majority of their characteristic
pretraining energy, forcing the feature toward the origin after centering. When projected, an OOD sample loses the spurious
confidence provided by pretraining features, leading to a flatter, more uniform logit distribution ĝ where p̂j ≈ 1/C. This
causes the output component V to collapse toward zero, significantly expanding the safety gap between ID and OOD
samples. Thus, projection filtering improves GradNorm by ensuring the output gradient is a true measure of semantic class
sensitivity relative to relevant decision boundaries, rather than a noisy response to pretraining variance.
While Global projection filtering removes the residuals, it might still retain the variance of all classes in the task. If classes
are crowded (e.g., 100 classes in CIFAR-100), the global subspace P is still quite large (high rank), allowing an OOD sample
to retain significant magnitude by aligning with the principal components of incorrect classes. Class-Predicted Projection


                                                               24

## Page 25

A Systematic Analysis of Out-of-Distribution Detection

solves this by projecting the feature h onto the specific subspace P ŷ of the predicted class ŷ: ĥŷ = P ŷ P ŷ⊤ (h − µŷ ) + µŷ .
This operation is much more aggressive. It discards not only the pretraining noise (null space) but also the variance directions
associated with all K − 1 other classes. For an ID sample correctly predicted as class ŷ, this preserves the signal perfectly
because the sample lies in that specific subspace. However, for an OOD sample (or a confused sample) that aligned weakly
with ŷ only by chance, projecting it onto this narrow, class-specific manifold destroys its magnitude almost entirely.

D.2.3. W HY DOES KPCA GLOBAL AND KPCA CLASS PRED ARE ALSO COMPETITIVE METHODS WHEN EVALUATING
       NEAR , MID , AND FAR OOD FOR V I T MODELS THAT WERE PRETRAINED AND FINETUNED ?

Kernel PCA (KPCA) excels at OOD detection for finetuned ViTs because it replaces the rigid linear assumptions of Neural
Collapse with a flexible manifold matching approach. Standard detectors like CTM or fDBD assume that In-Distribution
(ID) features collapse into simple, linearly separable clusters (Simplex ETF). However, finetuned ViTs retain a rich, complex
geometry from pretraining that violates these assumptions. KPCA addresses this by mapping input features h into a high-
dimensional Reproducing Kernel Hilbert Space via ψ(h) and identifying the principal subspace V that captures the intrinsic
non-linear structure of the ID data. The detection metric is the reconstruction error, e(x) = ∥ψ(h) − PV ψ(h)∥2 , which
quantifies how well a test sample fits this learned manifold rather than measuring its distance to a potentially misaligned
centroid.
Crucially, KPCA employs a Cosine-Gaussian kernel      to handle the specific
                                                                             geometric irregularities of finetuned representa-
                                                                    h′    2
tions. This kernel is defined as k(x, x′ ) = exp − 2σ1 2 ∥h∥h
                                                              2
                                                                −  ∥h′ ∥2 2 , neutralizes the magnitude variance common in
finetuned models, ensuring the detector focuses purely on angular alignment. Simultaneously, the Gaussian component
models local Euclidean distances on the hypersphere, allowing the subspace to wrap tightly around non-linear, sharpened
decision boundaries. This enables KPCA to enclose complex class shapes that linear methods—which effectively fit a flat
plane—would fail to capture, thereby reducing false positives from nearby OOD samples.
KPCA is distribution-agnostic regarding this global arrangement; it does not assume a specific prototype location (µc ) but
instead learns the aggregate manifold of all ID data. An OOD sample is detected not because it fails a specific angle test, but
because its feature vector contains variance components from the pretraining distribution that are orthogonal to the finetuned
task manifold. This ensures that OOD samples yield high reconstruction errors regardless of the symmetry or regularity
of the ID class clusters. Similar to GradNorm class pred, KPCA class pred shows an improved performance because the
class-predicted projection discards all variance components orthogonal to the predicted class ŷ, including generic image
statistics and features from competing classes.

E. Hyperparameter selection
All hyperparameters associated to the OOD detection methods are finetuned such that the AUGRC metric is optimized in
the validation set. Temperature scaling is tuned on the validation set and applied to all OOD detection methods that used
logits as inputs. For Deep Gamblers, the reward is also selected on the validation set.




                                                                 25

## Page 26

A Systematic Analysis of Out-of-Distribution Detection




                                          Table 8. Convolutional Neural Networks.
                                                          (a) CIFAR-10
                               confidnet                                 devries                                  dg
Variation       base   class     class class   global     base   class    class class   global   base   class   class   class   global
                                 avg pred                                 avg pred                              avg     pred
Method
CTM             0      0        0      0       0           0     0     0        0       0        0      0       0       0       0
CTMmean         1      1        0      1       0           0     0     0        0       0        0      0       0       0       0
CTMmeanOC       1      N/A      N/A    N/A     N/A         0     N/A N/A        N/A     N/A      0      N/A     N/A     N/A     N/A
Confidence      1      N/A      N/A    N/A     N/A         0     N/A N/A        N/A     N/A      0      N/A     N/A     N/A     N/A
Energy          1      1        1      1       1           1     1     0        1       1        1      1       0       1       1
GE              1      1        0      1       1           1     1     0        1       1        0      0       0       1       0
GEN             0      1        0      1       0           0     0     0        0       0        0      0       0       0       0
GradNorm        1      N/A      1      1       1           1     N/A 1          1       1        1      N/A     1       1       1
KPCA RecError   N/A    0        1      1       1           N/A 0       1        0       0        N/A    0       1       0       0
MLS             1      1        1      1       1           1     1     0        1       1        1      1       0       1       1
MSR             1      1        0      1       1           0     0     0        0       0        0      0       0       0       0
Maha            1      N/A      0      0       0           0     N/A 0          0       0        0      N/A     0       0       0
NNGuide         1      N/A      1      1       1           1     N/A 1          1       1        1      N/A     0       1       1
NeCo            1      N/A      N/A    N/A     N/A         1     N/A N/A        N/A     N/A      1      N/A     N/A     N/A     N/A
PCA RecError    N/A    0        0      0       1           N/A 0       1        0       1        N/A    0       0       0       1
PCE             1      1        0      1       1           0     0     0        0       0        0      0       0       0       0
PE              1      1        0      1       1           0     0     0        1       0        0      0       0       0       0
REN             0      0        0      0       0           0     0     0        0       0        0      0       0       0       0
Residual        1      N/A      N/A    N/A     N/A         1     N/A N/A        N/A     N/A      1      N/A     N/A     N/A     N/A
ViM             1      N/A      N/A    N/A     N/A         1     N/A N/A        N/A     N/A      0      N/A     N/A     N/A     N/A
fDBD            0      N/A      0      0       0           0     N/A 0          0       0        0      N/A     0       0       0
pNML            0      N/A      1      0       0           1     N/A 0          0       1        0      N/A     1       0       0
                                                        (b) SuperCIFAR-100
                               confidnet                                 devries                                  dg
Variation       base   class     class class   global     base   class    class class   global   base   class   class   class   global
                                 avg pred                                 avg pred                              avg     pred
Method
CTM             0      0        0      0       0          0      0       0      0       0        0      0       0       0       0
CTMmean         0      0        0      0       0          0      0       0      0       0        0      0       0       0       0
CTMmeanOC       0      N/A      N/A    N/A     N/A        0      N/A     N/A    N/A     N/A      0      N/A     N/A     N/A     N/A
Confidence      0      N/A      N/A    N/A     N/A        0      N/A     N/A    N/A     N/A      0      N/A     N/A     N/A     N/A
Energy          1      1        0      1       1          1      1       0      1       1        1      1       1       1       1
GE              0      0        0      1       0          1      1       0      1       1        1      1       1       1       1
GEN             0      0        0      0       0          0      0       0      0       0        0      0       0       1       0
GradNorm        1      N/A      1      1       1          1      N/A     1      1       1        1      N/A     1       1       1
KPCA RecError   N/A    0        1      0       0          N/A    0       1      0       0        N/A    0       1       0       0
MLS             0      0        0      0       0          1      1       0      1       1        1      1       1       1       1
MSR             0      0        0      0       0          0      0       0      1       0        1      0       0       1       1
Maha            1      N/A      0      0       0          1      N/A     0      0       0        0      N/A     0       0       0
NNGuide         0      N/A      0      0       0          0      N/A     0      0       0        1      N/A     1       1       1
NeCo            0      N/A      N/A    N/A     N/A        1      N/A     N/A    N/A     N/A      1      N/A     N/A     N/A     N/A
PCA RecError    N/A    0        0      0       1          N/A    0       1      0       1        N/A    0       1       0       1
PCE             0      0        0      0       0          0      0       0      1       0        1      0       0       1       1
PE              0      0        0      0       0          1      0       0      1       0        1      1       0       1       1
REN             0      0        0      0       0          0      0       0      0       0        0      0       0       0       0
Residual        1      N/A      N/A    N/A     N/A        1      N/A     N/A    N/A     N/A      0      N/A     N/A     N/A     N/A
ViM             0      N/A      N/A    N/A     N/A        0      N/A     N/A    N/A     N/A      0      N/A     N/A     N/A     N/A
fDBD            0      N/A      0      0       0          0      N/A     0      0       0        0      N/A     0       0       0
pNML            1      N/A      0      0       0          1      N/A     1      0       1        0      N/A     1       0       0




                                                                 26

## Page 27

A Systematic Analysis of Out-of-Distribution Detection




                                                         (c) CIFAR-100
                               confidnet                               devries                                  dg
Variation       base   class     class class   global   base   class    class class   global   base   class   class   class   global
                                 avg pred                               avg pred                              avg     pred
Method
CTM             0      0        0      0       0         0      0     0       0       0        0      0       0       0       0
CTMmean         1      1        1      1       1         0      0     0       0       0        0      0       0       0       0
CTMmeanOC       1      N/A      N/A    N/A     N/A       0      N/A N/A       N/A     N/A      0      N/A     N/A     N/A     N/A
Confidence      1      N/A      N/A    N/A     N/A       0      N/A N/A       N/A     N/A      0      N/A     N/A     N/A     N/A
Energy          1      1        1      1       1         0      0     0       0       0        0      0       0       0       0
GE              1      1        1      1       1         0      0     0       0       0        0      0       0       0       0
GEN             0      0        0      0       0         0      0     0       0       0        0      0       0       0       0
GradNorm        1      N/A      1      1       1         1      N/A 1         1       1        1      N/A     1       1       1
KPCA RecError   N/A    0        1      0       1         N/A 0        1       0       0        N/A    0       0       0       0
MLS             1      1        1      1       1         0      0     0       0       0        0      0       0       0       0
MSR             0      0        0      0       0         0      0     0       0       0        0      0       0       0       0
Maha            0      N/A      0      0       0         0      N/A 0         0       0        0      N/A     0       0       0
NNGuide         1      N/A      1      1       1         0      N/A 1         0       0        0      N/A     0       0       0
NeCo            1      N/A      N/A    N/A     N/A       0      N/A N/A       N/A     N/A      0      N/A     N/A     N/A     N/A
PCA RecError    N/A    1        0      1       1         N/A 0        1       0       1        N/A    0       0       0       1
PCE             0      0        0      0       0         0      0     0       0       0        0      0       0       0       0
PE              0      0        0      0       0         0      0     0       0       0        0      0       0       0       0
REN             0      0        0      0       0         0      0     0       0       0        0      0       0       0       0
Residual        0      N/A      N/A    N/A     N/A       0      N/A N/A       N/A     N/A      0      N/A     N/A     N/A     N/A
ViM             0      N/A      N/A    N/A     N/A       0      N/A N/A       N/A     N/A      0      N/A     N/A     N/A     N/A
fDBD            0      N/A      0      0       0         0      N/A 0         0       0        0      N/A     0       0       0
pNML            1      N/A      0      1       0         1      N/A 0         0       0        1      N/A     0       0       0
                                                        (d) TinyImageNet
                               confidnet                               devries                                  dg
Variation       base   class     class class   global   base   class    class class   global   base   class   class   class   global
                                 avg pred                               avg pred                              avg     pred
Method
CTM             0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
CTMmean         0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
CTMmeanOC       0      N/A      N/A    N/A     N/A      0      N/A     N/A    N/A     N/A      0      N/A     N/A     N/A     N/A
Confidence      0      N/A      N/A    N/A     N/A      1      N/A     N/A    N/A     N/A      0      N/A     N/A     N/A     N/A
Energy          0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
GE              0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
GEN             0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
GradNorm        0      N/A      0      0       0        0      N/A     0      0       0        0      N/A     0       0       0
KPCA RecError   N/A    0        1      0       0        N/A    0       1      0       0        N/A    0       0       0       0
MLS             0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
MSR             0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
Maha            1      N/A      0      0       0        0      N/A     0      0       0        0      N/A     0       0       0
NNGuide         0      N/A      0      0       0        0      N/A     0      0       0        0      N/A     0       0       0
NeCo            0      N/A      N/A    N/A     N/A      0      N/A     N/A    N/A     N/A      0      N/A     N/A     N/A     N/A
PCA RecError    N/A    0        1      0       0        N/A    0       1      0       0        N/A    0       0       0       0
PCE             0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
PE              0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
REN             0      0        0      0       0        0      0       0      0       0        0      0       0       0       0
Residual        1      N/A      N/A    N/A     N/A      1      N/A     N/A    N/A     N/A      1      N/A     N/A     N/A     N/A
ViM             0      N/A      N/A    N/A     N/A      1      N/A     N/A    N/A     N/A      1      N/A     N/A     N/A     N/A
fDBD            0      N/A      0      0       0        0      N/A     0      0       0        0      N/A     0       0       0
pNML            0      N/A      1      0       1        0      N/A     1      0       1        0      N/A     0       0       0




                                                               27

## Page 28

Table 9. Vision Transformers.
                                                                                                        cifar10                           supercifar100                               cifar100                    tiny-imagenet-200
                                                         variation       (none) class   class   class     global   (none) class   class    class global   (none) class   class   class global    (none) class   class class global
A Systematic Analysis of Out-of-Distribution Detection




                                                                                        avg     pred                              avg      pred                          avg     pred                           avg pred
                                                         method
                                                         CTM             1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         CTMmean         1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         CTMmeanOC       1      nan     nan     nan      nan       0      nan     nan     nan    nan      1      nan     nan     nan    nan      0      nan     nan   nan    nan
                                                         Confidence      1      nan     nan     nan      nan       0      nan     nan     nan    nan      1      nan     nan     nan    nan      0      nan     nan   nan    nan
                                                         Energy          1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         GE              1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         GEN             1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         GradNorm        1      nan     1       1        1         0      nan     0       0      0        1      nan     1       1      1        0      nan     0     0      0




                                                                                                                                                                                                                                      28
                                                         KPCA RecError   nan    1       1       1        1         nan    0       0       0      0        nan    1       1       1      1        nan    0       0     0      0
                                                         MLS             1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         MSR             1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         Maha            1      nan     1       1        1         0      nan     0       0      0        1      nan     1       1      1        0      nan     0     0      0
                                                         NNGuide         1      nan     1       1        1         0      nan     0       0      0        1      nan     1       1      1        0      nan     0     0      0
                                                         NeCo            1      nan     nan     nan      nan       0      nan     nan     nan    nan      1      nan     nan     nan    nan      0      nan     nan   nan    nan
                                                         PCA RecError    nan    1       1       1        1         nan    0       0       0      0        nan    1       0       1      1        nan    0       0     0      0
                                                         PCE             1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         PE              1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         REN             1      1       1       1        1         0      0       0       0      0        1      1       1       1      1        0      0       0     0      0
                                                         Residual        1      nan     nan     nan      nan       0      nan     nan     nan    nan      1      nan     nan     nan    nan      0      nan     nan   nan    nan
                                                         ViM             1      nan     nan     nan      nan       0      nan     nan     nan    nan      1      nan     nan     nan    nan      0      nan     nan   nan    nan
                                                         fDBD            1      nan     1       1        1         0      nan     0       0      0        1      nan     1       1      1        0      nan     1     0      0
                                                         pNML            1      nan     1       1        1         0      nan     0       0      0        1      nan     1       1      1        0      nan     0     0      0

## Page 29

Table 10. Vision Transformers.
                                                                                                        cifar10                              supercifar                               cifar100                    tiny-imagenet-200
                                                         variation       (none) class   class   class     global   (none) class   class   class global    (none) class   class   class global    (none) class   class class global
                                                                                        avg     pred                              avg     pred                           avg     pred                           avg pred
                                                         method
                                                         CTM
                                                                         2.2    2.2     10.0    2.2      3.0       10.0   3.0     10.0    3.0    10.0     15.0   10.0    15.0    15.0   15.0     15.0   15.0    15.0   15.0   15.0
                                                         CTMmean
                                                                         10.0   10.0    10.0    10.0     10.0      10.0   10.0    10.0    10.0   10.0     15.0   6.0     10.0    6.0    10.0     15.0   15.0    15.0   15.0   15.0
                                                         CTMmeanOC
                                                                         10.0   nan     nan     nan      nan       10.0   nan     nan     nan    nan      6.0    nan     nan     nan    nan      15.0   nan     nan    nan    nan
                                                         Confidence
A Systematic Analysis of Out-of-Distribution Detection




                                                                         2.2    nan     nan     nan      nan       3.0    nan     nan     nan    nan      10.0   nan     nan     nan    nan      15.0   nan     nan    nan    nan
                                                         Energy
                                                                         3.0    3.0     10.0    3.0      3.0       20.0   20.0    20.0    20.0   20.0     15.0   15.0    10.0    15.0   15.0     15.0   15.0    10.0   15.0   15.0
                                                         GE
                                                                         10.0   10.0    10.0    3.0      10.0      20.0   20.0    20.0    20.0   20.0     15.0   15.0    10.0    10.0   15.0     15.0   15.0    10.0   15.0   15.0
                                                         GEN
                                                                         10.0   10.0    10.0    10.0     10.0      10.0   10.0    10.0    20.0   10.0     15.0   15.0    10.0    10.0   15.0     15.0   15.0    10.0   15.0   15.0
                                                         GradNorm
                                                                         3.0    nan     6.0     3.0      3.0       20.0   nan     20.0    20.0   20.0     20.0   nan     20.0    20.0   20.0     15.0   nan     10.0   10.0   15.0
                                                         KPCA RecError
                                                                         nan    10.0    6.0     10.0     2.2       nan    10.0    20.0    10.0   10.0     nan    15.0    15.0    15.0   6.0      nan    15.0    15.0   15.0   15.0
                                                         MLS




                                                                                                                                                                                                                                      29
                                                                         3.0    3.0     10.0    3.0      3.0       20.0   20.0    20.0    20.0   20.0     15.0   15.0    10.0    15.0   15.0     15.0   15.0    10.0   15.0   15.0
                                                         MSR
                                                                         10.0   10.0    10.0    10.0     10.0      20.0   10.0    10.0    20.0   20.0     15.0   15.0    10.0    10.0   15.0     15.0   15.0    10.0   15.0   15.0
                                                         Maha
                                                                         2.2    nan     2.2     2.2      2.2       2.2    nan     2.2     3.0    2.2      6.0    nan     6.0     6.0    6.0      15.0   nan     20.0   15.0   15.0
                                                         NNGuide
                                                                         3.0    nan     10.0    3.0      3.0       20.0   nan     20.0    20.0   20.0     15.0   nan     10.0    15.0   15.0     15.0   nan     15.0   15.0   15.0
                                                         NeCo
                                                                         3.0    nan     nan     nan      nan       20.0   nan     nan     nan    nan      15.0   nan     nan     nan    nan      15.0   nan     nan    nan    nan
                                                         PCA RecError
                                                                         nan    10.0    6.0     10.0     10.0      nan    10.0    20.0    10.0   20.0     nan    15.0    10.0    15.0   20.0     nan    15.0    15.0   15.0   15.0
                                                         PCE
                                                                         10.0   10.0    10.0    10.0     10.0      20.0   10.0    10.0    20.0   20.0     15.0   15.0    10.0    10.0   15.0     15.0   15.0    10.0   15.0   15.0
                                                         PE
                                                                         10.0   10.0    10.0    10.0     10.0      20.0   20.0    10.0    20.0   20.0     15.0   15.0    10.0    10.0   15.0     15.0   15.0    10.0   15.0   15.0
                                                         REN
                                                                         10.0   10.0    10.0    10.0     10.0      10.0   10.0    10.0    10.0   10.0     10.0   10.0    10.0    10.0   15.0     15.0   15.0    10.0   15.0   15.0
                                                         Residual
                                                                         2.2    nan     nan     nan      nan       2.2    nan     nan     nan    nan      6.0    nan     nan     nan    nan      20.0   nan     nan    nan    nan
                                                         ViM
                                                                         2.2    nan     nan     nan      nan       2.2    nan     nan     nan    nan      15.0   nan     nan     nan    nan      20.0   nan     nan    nan    nan
                                                         fDBD
                                                                         2.2    nan     6.0     3.0      2.2       10.0   nan     10.0    10.0   10.0     15.0   nan     15.0    15.0   15.0     15.0   nan     15.0   15.0   15.0
                                                         pNML
                                                                         10.0   nan     6.0     10.0     10.0      2.2    nan     10.0    10.0   2.2      20.0   nan     10.0    6.0    15.0     15.0   nan     15.0   15.0   10.0
