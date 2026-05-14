# Recent Advances in Out-of-Distribution Detection with CLIP-Like Models: A Survey - page-anchored PDF text

- Source ID: `arxiv-2505.02448v1`
- arXiv ID: `2505.02448v1`
- Original PDF: `소스/Recent Advances in Out-of-Distribution Detection with CLIP-Like Models_ A Survey.pdf`
- PDF pages: 9
- Extracted with: WSL poppler `pdftotext -f N -l N -layout` on 2026-05-13T17:01:27+09:00
- Evidence note: use this file for page-anchored claims; verify equations/tables against the original PDF when exact layout matters.

## Page 1

Recent Advances in Out-of-Distribution Detection with CLIP-Like Models: A
                                                                             Survey
                                                     Chaohua Li1,2 , Enhao Zhang1,2 , Chuanxing Geng1,2,3 and Songcan Chen1,2∗
                                            1
                                              College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics
                                                           2
                                                             MIIT Key Laboratory of Pattern Analysis and Machine Intelligence
                                                             3
                                                               Department of Computer Science, Hong Kong Baptist University
                                                                {chaohuali, zhangeh, gengchuanxing, s.chen}@nuaa.edu.cn
arXiv:2505.02448v1 [cs.CV] 5 May 2025




                                                                        Abstract                                  Train:
                                                                                                                                            In-Distribution (ID) Data from: ImageNet-1K

                                                Out-of-distribution detection (OOD) is a pivotal
                                                task for real-world applications that trains models
                                                to identify samples distributionally different from
                                                the in-distribution (ID) data during testing. Recent
                                                advances in AI, particularly Vision-Language Mod-                                            Dog              Koala            Cat
                                                els (VLMs) like CLIP, have revolutionized OOD
                                                detection by shifting from traditional unimodal im-               Test:




                                                                                                                           Seen Classes
                                                age detectors to multimodal image-text detectors.
                                                This shift has inspired extensive research, however,
                                                existing categorization scheme (e.g., few/zero-shot
                                                types) still rely solely on the availability of ID im-
                                                ages, following a unimodal paradigm. To better                                                     Out-of-Distribution (OOD) Data
                                                align with CLIP’s cross-modal nature, we propose
                                                                                                                           Unseen Classes


                                                a new categorization scheme rooted in both im-
                                                age and text modalities. Specifically, we catego-
                                                rize existing methods guided by how the visual and
                                                textual information of OOD data is separately uti-
                                                lized in image + text modalities, further dividing
                                                                                                                      From:                 Places          iNaturalist       Texture     ……
                                                them into four groups: {OOD Images (i.e., outliers)
                                                Seen/Unseen + OOD Texts (i.e., learnable vec-
                                                tors or class names) Known/Unknown} across two                 Figure 1: Illustration of Out-of-distribution (OOD) Detection. OOD
                                                training strategies (e.g., train-free/required). More          detection is a major computer vision task that addresses semantic
                                                                                                               distribution shifts between training and testing data, reflecting real-
                                                importantly, we further discuss the open problems
                                                                                                               world scenarios. Specifically, during training, the model learns from
                                                of CLIP-like OOD detection, and highlight poten-               ID data, as shown in the upper. However, once deployed, the model
                                                tial directions for future research, including cross-          may encounter both ID (seen) and OOD (unseen) data in the testing
                                                domain integration, practical applications, and the-           phase, as depicted in the below. To ensure reliability, the trained
                                                oretical understanding.                                        model should not only classify ID samples accurately but also detect
                                                                                                               OOD instances to avoid incorrect decisions.

                                        1       Introduction
                                        Recent advances in artificial intelligence (AI) have en-               but also handle out-of-distribution (OOD) data effectively in
                                        abled modern systems to perform exceptionally well un-                 real-time, as shown in Fig. 1, [Hendrycks and Gimpel, 2016;
                                        der laboratory-level environments [Russakovsky et al., 2015;           Huang and Li, 2021]. For example, an autonomous driving
                                        He et al., 2015]. These systems are typically trained follow-          system should be able to alert the driver to unidentified obsta-
                                        ing the closed-set assumption, where both the training and             cles instead of misclassifying them as ordinary objects, which
                                        testing data are drawn from the identical label and feature            could lead to catastrophic consequences [Huang et al., 2020].
                                        spaces (i.e., in-distribution (ID)) [Scheirer et al., 2012]. How-      Similarly, in medical diagnostics [Hong et al., 2024], an AI-
                                        ever, this assumption does not always hold in real-world sce-          powered system must be adept at detecting anomalies such
                                        narios, as it is infeasible to include all possible data in training   as lesions without confusing them with normal samples, en-
                                        without omissions. As a result, a robust AI system should              suring accurate and reliable diagnoses. Moreover, detecting
                                        possess not only the ability to classify ID data accurately,           OOD data is crucial in tasks such as novel species detection
                                                                                                               in biodiversity studies [Marsland, 2003] and lifelong learning
                                            ∗
                                                Corresponding author.                                          systems [Saar and Ure, 2013] that continuously adapt to new

## Page 2

knowledge.                                                                   The Number of Recent CLIP-like OOD Detection Studies
   Out-of-distribution (OOD) detection has garnered signifi-                                  in the Top Venues     by February 2025
                                                                        18
cant attention in recent years. Traditionally, OOD detection                                                TagFog SeTAR    LSN
                                                                                  IU + TK
                                                                                                         GalLoP NegLabel 16 Tag
methods were primarily developed within a unimodal frame-               16
                                                                                  IU + TU                PEFT-MCM DPM CMA
work, as displayed in Fig. 3(a), focusing solely on image                                                     CLIP-driven ID-like
                                                                        14        IS + TU
information [Hendrycks et al., 2020; Zhang et al., 2024b].                                                EOE CATEX       LAPT
While effective to a degree, these approaches were inher-               12        IS + TK                   NegPrompt AdaNeg
ently limited due to the absence of supplementary contex-
tual information from other modalities [Ming et al., 2022a;             10
Jiang et al., 2024]. In recent developments, the advent of              8
Contrastive Language–Image Pre-trained (CLIP) [Radford et
al., 2021], a representative Vision-Language Model (VLM)                6
trained on large-scale image-text pairs, has showcased excep-                                                  4
                                                                        4                              CLIPN
tional potential in addressing computer vision (CV) tasks. In                                   2              LoCoOp
                                                                                  1            ZOC       GL-MCM
the context of OOD detection, the CLIP-like paradigm has                2     Zero-shot OE                  Textual OE
moved beyond the limitations of unimodal paradigm by in-                                      MCM
tegrating the additional information from the text modality,            0
                                                                                2021           2022          2023          2024
as depicted in Fig. 3(b). This shift has led to a surge of                                            Year
research in recent years, as illustrated in Fig. 2. One pre-
vailing categorization scheme for existing CLIP-like studies
                                                                  Figure 2: The number of recent CLIP-like OOD detection studies
is based on the availability of ID image during the train-        in top venues (up to 02/2025). The abbreviations IU + TK, IU +
ing or inference. It divides thses studies into few-shot and      TU, IS + TU and IS + TK, represent OOD Images Unseen + OOD
zero-shot categories [Wang et al., 2023; Miyai et al., 2024a;     Texts Known, OOD Images Unseen + OOD Texts Unknown, OOD
Jiang et al., 2024; Miyai et al., 2024b; Sun et al., 2024;        Images Seen + OOD Texts Unknown and OOD Images Seen + OOD
Zhang et al., 2024a], where few-shot involves limited access      Texts Known, respectively.
to ID image information, while zero-shot operates without
any. However, under this classification scheme, the term ‘un-
known’ typically implies unseen visual information of data,       (OSR) and novelty detection (ND), with its origins tracing
even though the textual information is actually known in cer-     back to early studies in statistics and machine learning. To the
tain CLIP-like methods. Thus, it overlooks an essential:          best of our knowledge, [Hendrycks and Gimpel, 2016] was
CLIP is inherently a multimodal pre-trained model, designed       the first to formally introduce the term Out-of-Distribution
to combine the strengths of both image and text modalities,       Detection and explore it in the deep learning era, establish-
representing a significant shift from the traditional OOD de-     ing the foundation for subsequent research. The following
tection paradigm.                                                 subsections will provide an overview of OOD detection from
                                                                  various perspectives.
   In this survey, we aim to bridge this gap by better align-
ing the multimodal nature of the CLIP framework and ex-           2.1    Definition of OOD Detection.
plicitly summarizing the associated methodologies. The rest
                                                                  Definition 1 (Unimodal Out-of-distribution Detection).
of the paper is organized as follows. We begin by review-
                                                                  Let Xin and Yin denote ID image space and ID label space,
ing the background of OOD detection from fundamental as-
                                                                  Xood and Yood denote OOD image space and OOD label
pects (Sec. 2). Then, we present the problem formulation for
                                                                  space, where Yin ∩ Yood = ∅. Given any text sample x, the
CLIP-like OOD detection methods (Sec. 3). In contrast to
                                                                  OOD detection is introduced to classify x into a correct ID
the few-shot and zero-shot types, we categorize the existing
                                                                  class (i.e., x ∈ Xin ) or reject it as the OOD (x ∈ Xood ) with
CLIP-like OOD detection works based on how OOD data are
                                                                  a score function S(·) that satisfies:
utilized within multimodal settings under two training strate-
gies (e.g, train-free and train-required), and divide them into
                                                                                            
                                                                                              ID,       if S(x; Yin , I) ≥ γ,
four groups: OOD Images Seen + OOD Texts Known, OOD                      Gγ (x; Yin , I) =                                    (1)
                                                                                              OOD, othersise.
Images Unseen + OOD Texts Known, OOD Images seen
+ OOD Texts Unknown, and OOD Images Unseen + OOD                     where Gγ is the OOD detector with the threshold γ, and I
Texts Unknown (i.e., notably, images seen represents the out-     is an image encoder.
liers rather than real OOD data) (Sec. 4). Furthermore, we
also discuss several open questions and limitations of CLIP-      2.2    Categorization of Unimodal OOD Detection.
like OOD detection and highlight promising directions for         The design of OOD detection algorithms involves multiple
future research at cross-domain integration, practical appli-     aspects, such as training strategy, score design, and data
cations, and theoretical understanding aspects (Sec. 5).          types. Since we will categorize the CLIP-like methods based
                                                                  on the utilization differences of OOD data, we briefly review
2   Reviewing the OOD Detection                                   unimodal approaches in this subsection through outliers-
                                                                  exposure and outliers-free [Zhang et al., 2024b]:
The concept of OOD detection has evolved from related top-             Outliers-exposure allows partial exposure of OOD data
ics such as anomaly detection (AD), open-set recognition          during the training process to attain a more discriminative

## Page 3

(a)
                       ID Input
                                                                  w0·vin                  OOD Score     w0·vood                             × OOD
                                                                  w1·vin                                w1·vood
                                                    Image
                                                   Encoder        w2·vin                         √ ID   w2·vood




                                                                                  …




                                                                                                                                …
                                                                                                                  …
                                                                   …
                      OOD Input                                   wK·vin                                wK·vood



         (b)                                                                 Cosine Similarity
                                                                   T(y1)         t1·vin    t1·vood




                                                                                                        Detection Score
                                                                   T(y2)         t2·vin    t2·vood
                                                     Text
                     “a photo of a {object }”      Encoder
                                                                   T(y3)         t3·vin    t3·vood
                                                                                                                                     Out-Of-Distribution
               dog     cat    koala   …   hockey                   T(y4)         t4·vin    t4·vood




                                                                                  …


                                                                                              …
                                                                     …
                                                                   T(yK)         tK·vin    tK·vood                        In-Distribution
                       ID Input                                                  I(xID)     I(xOOD)


                                                    Image
                                                   Encoder


                      OOD Input



Figure 3: The general pipeline of unimodal and CLIP-like OOD detection paradigms. (a) Unimodal OOD detection models typically use a
single-image encoder trained on ID data. The encoder extracts visual embeddings and often employs confidence scores, distance metrics, or
density estimation with a threshold. (b) Compared to the unimodal ones, CLIP learns a joint vision-language embedding space by aligning
images with textual descriptions. This allows CLIP to map diverse visual concepts into a more semantically structured embedding space.


classification boundary between ID and OOD classes. A key                  feature space.
principle is that the exposed OOD data used in training must
not share the same semantic distribution with the OOD data                 2.3       Benchmark Datasets & Evaluation Metrics.
used in testing (i.e., Youtliers ∩ Yoodtest = ∅). Specifically,            Benchmark Datasets. Benchmarking OOD detection gen-
these outliers may include subsets from real OOD data (i.e.,               erally involves defining one dataset as the ID set and se-
such as the background [Cho and Choo, 2022]), OOD sam-                     lecting several datasets with no category overlap to serve
ples from additional datasets [Ming et al., 2022b] and gen-                as the OOD datasets. Benchmarking OOD detection gener-
erated data [Li et al., 2024a]. In general, outliers-exposure              ally involves defining one dataset as the ID set and select-
methods exhibit superior performance because more data in-                 ing several datasets with no category overlap with the ID set
formation is accessible during training. However, they are                 to serve as the OOD datasets. A widely used benchmark
sensitive to the choice of outliers, and some generative ap-               for OOD detection is the ImageNet benchmark. When us-
proaches may introduce significant model complexity and                    ing the large-scale ImageNet-1K dataset as the ID set, iNat-
run-time consumption [Yang et al., 2024].                                  uralist, SUN, Texture, and Places are commonly employed
                                                                           as OOD datasets to evaluate algorithm performance. These
     Outliers-free methods can be broadly categorized into                 datasets undergo rigorous filtering to ensure that no samples
classification-based, density-based and distance-based ap-                 overlap with the ID classes. Recently, some studies have fur-
proaches [Yang et al., 2024]. Among classification-based                   ther divided OOD datasets into near-OOD (e.g., iNaturalist,
methods, post-hoc detection is a major branch that can be                  Species, OpenImage-O, ImageNet-O, SSB-hard and NINCO,
directly applied to any already trained models [Hendrycks                  etc) and far-OOD (Texture, MNIST and SVHN, etc) based on
and Gimpel, 2016; Liu et al., 2020]. Besides, confidence                   the semantic similarity to the ID dataset (e.g., ImageNet-1K).
enhancement is another promising branch of classification-                 The sources of these datasets are documented in [Yang et al.,
based methods, which typically requires model retraining to                2022] and [Zhang et al., 2024a].
enhance the robustness of models [Bitterwolf et al., 2020].                Evaluation Metrics. We present four evaluation metrics
Furthermore, density-based methods rely on probabilistic                   widely used in OOD detection. FPR@95 quantifies the false
models to represent the in-distribution and classify test sam-             positive rate (FPR) of OOD samples when the true positive
ples as OOD if they fall into low-density regions [Ren et al.,             rate (TPR) of ID samples is fixed at 95%. A lower FPR@95
2019; Xiao et al., 2020]. In contrast, distance-based methods              value signifies better OOD detection performance. AUROC
[Sun et al., 2022] identify OOD samples by calculating their               the Area Under the Receiver Operating Characteristic Curve
distance from the nearest ID centroids or prototypes in the                (AUROC) represents the relationship between the TPR and

## Page 4

the FPR. A higher AUROC value reflects superior model per-        One of the various score functions S(·) is then designed to-
formance. AUPR is a metric that measures the area under the       wards these outputs, with a threshold γ applied to determine
Precision-Recall (PR) curve, which illustrates the relation-      whether a test sample x belongs to the ID or OOD category:
ship between precision and recall. ACC reflects the accuracy                             
of classification for ID classes, with higher values indicating                            ID,     if S(x; Yin , I, T ) ≥ γ,
                                                                   Gγ (x; Yin , I, T ) =                                     (4)
better performance.                                                                        OOD, othersise.
                                                                     For samples identified as ID, the class prediction can be
3     Problem Formulation for CLIP-like OOD                       determined by selecting the nearest text embedding tk . What
      Detection                                                   sets Eq. (4) apart from formulation Eq. (1) most distinctly is
3.1    CLIP-like Models.                                          the integration of encoder T , which plays a crucial role in its
                                                                  overall framework.
CLIP is one of the most advanced pre-trained vision-language
models (VLMs) developed by [Radford et al., 2021] from
OpenAI. It is designed to align image-text pairs from large-      4     CLIP-like OOD Detection: Categorization
scale datasets within a shared vector space using contrastive     In this section, we introduce the methodologies for CLIP-
learning. This approach allows the model to understand and        like OOD detection. We will categorize recent studies based
process the associations between images and text within a la-     on how visual and textual information of OOD data are
tent space.                                                       utilized in multimodal settings under train-free and train-
   CLIP is composed of an image encoder I and a text en-          required strategies. Specifically, building on CLIP’s multi-
coder T . For a given test image x and its associated la-         modal nature, we analyze various design strategies for OOD
bel yk ∈ Y, CLIP-like modes extract the image embedding           data across image and text modalities and systematically clas-
v ∈ RD and text embedding t ∈ RD as follows:                      sify existing methods into four categories: Images Seen +
                                                                  Texts Known, Images Unseen + Texts Known, Images Seen +
    v = I(x),    t = T (prompt(yk )), k = 1, 2, ..., K,    (2)    Texts Unknown, and Images Unseen + Texts Unknown. These
where D denotes the embedding dimension, and prompt(·)            categories are summarized in Table. 1, with their correspond-
refers to the text prompt function, which is typically crafted    ing frameworks illustrated in Fig. 4.
manually as ‘a photo of a [LABEL]’. For instance, the                Notably, in this paper, the term OOD Images Seen often
[LABEL] token can be substituted with specific class names        refers to train-required strategy. These OOD images do not
like ’dog’ or ’cat’. The CLIP-like models then performs           correspond to the actual OOD samples used in testing.
zero-shot classification by calculating the cosine similar-
ity between the image embedding v and text embeddings             4.1    OOD Images Seen + Texts Known
t1 , t2 , ..., tK , scaled by a factor τ :                        Studies in this category typically involve incorporating out-
                                                                  liers and constructing specific prompts for fine-tuning (a.k.a
                         exp (cos (v, tk ) /τ )
          p (yk | x) = PK                          .       (3)    prompt learning). ID-like [Bai et al., 2024] generates outliers
                        j=1 exp (cos (v, tj ) /τ )                by randomly cropping a small subset of ID data and intro-
                                                                  duces a prompt learning framework tailored for these ID-like
3.2    CLIP-based Prompt Learning.                                outliers. CLIP-driven [Sun et al., 2024] generates reliable
Recent efforts have investigated diverse strategies to further    OOD data by mixing ID-related features with an unknown-
enhance the CLIP’s performance on downstream tasks [Li            aware prompt learning strategy. CATEX [Liu et al., 2024]
et al., 2024b], with a major focus on optimizing the text         synthesizes outliers independent of specific ID tasks by ap-
modality. While basic CLIP models use manually crafted            plying random perturbations, aiding in training non-trivial
prompt templates like ‘a photo of a [LABEL]’, CoOp [Zhou          spurious text descriptions. LAPT [Zhang et al., 2024a] also
et al., 2022] introduces prompt learning, where the contin-       employs the mix-up technique to synthesize outliers, explor-
uous learnable tensors are designed towards the embedding         ing the space between the ID and OOD distributions. Further-
layer of the prompts. Specifically, these tensors are initial-    more, it expands the OOD label space by NegLabel [Jiang
ized as ω k = [ω]1 [ω]2 . . . [ω]L [LABELk ], where L denotes     et al., 2024] and enables label-driven automated prompt tun-
the token length and [LABELk ] is the word embedding of the       ing via web-scale retrieval and text-to-image generation tech-
k-th class name. Then, the encoders of CLIP are frozen, and       niques. To resolve semantic misalignment caused by uni-
optimization is performed through backpropagation by mini-        form negative labels across different OOD datasets, AdaNeg
mizing classification loss on the target task with few data.      [Zhang and Zhang, 2024] adaptively generates outliers by ex-
   Given the prompt ω k as input, the text encoder T outputs      ploring actual OOD images stored in a purpose-built memory
the textual embedding as tk = T (ω k ), and the final predic-     bank, achieving better alignment with the OOD label space.
tion probability is then calculated based on Eq. (3).                Remarks: This category is defined by two main aspects:
                                                                  1) Images Seen: ID images are used to generate outlier im-
3.3    CLIP-like OOD Detection.                                   ages through techniques like image transformations or mix-
The pipeline of CLIP-like OOD detection can been seen in          up. These outlier images are then utilized for model fine-
Fig. 3(b). In particular, the text embedding t extracted by the   tuning (i.e., train-required); 2) Texts Known: OOD text (i.e.,
text encoder T in CLIP can be regarded as a cosine similarity-    prompt) is initialized initially and combined with the gener-
based classifier, outputting logits or softmax probabilities.     ated outlier images in prompt learning, enhancing the model’s

## Page 5

…                                        …
                                                                                                                   …
                            …                                        …




                                  align                                     align
                                     push                                                                                  push
                                                                                     push
                                                                            push
                                  align

             OOD Images Seen                         OOD Images Unseen                            OOD Images Seen
             OOD Texts Known                          OOD Texts Known                            OOD Texts Unknown
                      (a)                                     (b)                                           (c)
                                                                                                          Image / Text Encoder
                             …                                        …
                                                                                                            ID / OOD Inputs
                                                                                                             Outlier Images
                                                                                                         ID Prompts / Class Names
                                                                                                        OOD Prompts / Class Names
                                  align
                                                                             align                     ID / OOD Image Embeddings
                                  push
                                                                                                        ID / OOD Text Embeddings
                                                                                                            Cosine Classifier
          OOD Images Unseen                           OOD Images Unseen                                           Frozen
          OOD Texts Unknown (Train-free)              OOD Texts Unknown (Train-required)                       Learnable
                                               (d)

Figure 4: An illustration of different CLIP-like OOD detection paradigms, from the perspective of the utilization of OOD visual and textual
information. These paradigms are mainly distinguished by whether they require additional OOD image information (represented by gray
dashed circles) and whether OOD-specific textual information (represented by gray dashed squares) is incorporated into fine-tuning. In
particular, OOD Images Unseen + OOD Texts Unknown is further divided into two groups, Train-free and Train-required (illustrated in
Figure (d) with blue descriptions). The primary criterion for this division is whether prompt learning for ID classes is involved.


ability to distinguish between ID and OOD samples. Notably,               like “Give three categories that visually resemble a horse”to
AdaNeg is the only method that does not involve prompt                    align with the ID class “horse”. It also introduces a new score
learning, however, its designed memory bank is still involved             function based on these potential outliers. NegPrompt [Li et
in the training process.                                                  al., 2024b] trains a set of negative prompts exclusively on
                                                                          ID data, where each prompt represents a negative interpreta-
4.2   OOD Images Unseen + Texts Known                                     tion of a specific ID class label, helping define the boundary
This category constitutes a major branch of CLIP-based                    between ID and OOD images. Unlike conventional prompt
OOD detection methods, fully exploiting CLIP’s textual in-                learning, which learns positive prompts for all classes, LSN
terpretability. Zero-shot OE [Fort et al., 2021], which selects           [Nie et al., 2024] focuses on learning negative prompts for
the candidate labels related to OOD classes from an ID labels-            each OOD class to explicitly define what is “not”. CMA [Lee
irrelevant dataset, then applies Softmax to compute predic-               et al., 2025] employs neutral prompts as the agents to aug-
tion probabilities. ZOC [Esmaeilpour et al., 2022] expands                ment the label space, which does not apply learnable prompts
the OOD label space by employing a text generator on top                  for text enhancement, but only uses the corresponding class
of CLIP to select candidate OOD class names for each test                 names.
sample. CLIPN [Wang et al., 2023] introduces a learnable                     Remarks: This category has two main characteristics: 1)
“no” prompt and a specialized text encoder to capture neg-                Images Unseen: No additional OOD images are needed for
ative semantics within ID images. Textual OE [Park et al.,                training or testing; 2) Texts Known: The primary focus of
2023] emphasizes the text modality, highlighting the benefits             this category is on OOD text design, which can be further
of textual outliers by generating OOD class names through                 divided into: a) Learnable OOD prompts design. Methods
three strategies: word-level, description-level and caption-              in this group, such as CLIPN, LSN, NegPrompt and CMA,
level textual outliers. NegLabel [Jiang et al., 2024] derives a           construct a variety of learnable prompts with “no/not” se-
large set of negative labels from extensive corpus databases,             mantic logic, typically derived from the textual information
utilizing the distance between negative labels and ID labels              of ID data; b) OOD class names mining. This branch, in-
as a metric. Additionally, it designs a novel OOD scoring                 cluding Zero-shot OE, ZOC, Textual OE, NegLabel and EOE,
scheme collaborated with negative labels. EOE [Cao et al.,                focuses on mining OOD class name candidates from exter-
2024] constructs OOD texts to generate potential outlier class            nal text databases using various predefined metrics. Overall,
labels for OOD detection. For instance, it employs prompts                most methods in this category require training to optimize the

## Page 6

designed OOD textual information (i.e., train-required), ex-           Categories   Train Types                 Methods
cept for Zero-shot OE, NegLabel and CMA(i.e., train-free).
                                                                                                       ID-like, LAPT, AdaNeg,
                                                                         IS+TK           TR
                                                                                                        CLIP-driven, CATEX
4.3   OOD Images Seen + Texts Unknown
                                                                                                      ZOC, EOE, CLIPN, LSN,
Unlike the OOD Images Seen + OOD Texts Known, this cat-                                  TR
                                                                                                       NegPrompt, Textual OE
egory does not rely on enhancing the similarity between de-
                                                                         IU+TK                          ZerO-shot OE, CMA,
signed OOD texts and OOD images. Instead, it aims to max-                                TF
imize the dissimilarity between ID texts and OOD images                                                      NegLabel
through various OOD regularization techniques. LoCoOp                    IS+TU           TR                LoCoOp, TagFog
[Miyai et al., 2024b] first segments original ID images into
local regions, treating ID-irrelevant areas (i.e., background)                           TR                GalLop, DPM-T
as outliers. It then applies entropy maximization as an OOD              IU+TU                        MCM, GL-MCM, SeTAR,
regularization strategy to ensure that features in ID-irrelevant                         TF
                                                                                                          DPM-F, TAG
regions remain separate from any ID text embeddings. An-
other work, TagFog [Chen et al., 2024] introduces the jigsaw-      Table 1: Different Categories for CLIP-like OOD Detection. Where
based fake OOD images as outliers and incorporates rich se-        TR and TF refers to train-required and train-free, respectively. The
mantic embeddings from ChatGPT-generated description of            abbreviations in the Categories are consistent with those in Fig. 2.
ID textual information to guide the training of the image en-
coder.
   Remarks: This category has two defining characteris-            performance by further fine-tuning the ID prompts within the
tics: 1) Images Seen: Similar to OOD Images Seen + OOD             original framework.
Texts Known, it involves acquiring external images as outliers        Remarks: This category has two key characteristics: 1)
through various techniques for training (i.e., train-required);    Images Unseen: None of the methods in this category re-
2) Texts Unknown: As previously mentioned, these meth-             quire additional OOD images during training or inference,
ods apply OOD regularization to separate the obtained out-         regardless of whether they are train-free or train-required
liers from ID class text embeddings, thereby effectively re-       strategies. Furthermore, the designed OOD scores can be ap-
moving redundant information from ID text representations.         plied across various CLIP-like OOD detection approaches; 2)
                                                                   Texts Unknown: In train-free methods, prompt learning is
4.4   OOD Images Unseen + Texts Unknown                            unnecessary because OOD scores are only used during infer-
                                                                   ence. In train-required methods, only ID texts are learned,
One subcategory adopts a train-free strategy, also referred to     and there is no requirement for OOD texts.
as test-time OOD detection in CLIP-like models, correspond-
ing to post-hoc methods in unimodal OOD detection. It does         4.5    Advantages Comparisons
not require training but instead employs designed OOD score        The pipelines for the four categories are illustrated in Fig. 4.
functions at test time to recognize OOD data. A key repre-         OOD Images Seen + Texts Known integrates additional infor-
sentative work is MCM [Ming et al., 2022a], which intro-           mation from both vision and text, allowing the model to more
duces the Maximum Concept Matching score. This method              effectively identify OOD data. By incorporating prior knowl-
considers ID class names as textual concepts that align with       edge of OOD images and textual descriptions, the model can
visual embeddings during testing, supported by detailed anal-      learn richer representations that enhance detection. OOD Im-
ysis and theoretical insights. GL-MCM [Miyai et al., 2023]         ages Unseen + Texts Known takes a different approach by
improves on MCM by incorporating local features, though            avoiding the explicit introduction of additional OOD sam-
its applicability is restricted to specific scenarios. DPM-F       ples. Instead, it focuses on designing learnable OOD-specific
[Zhang et al., 2024b] argues that ID visual information plays      vectors or class labels that encourage the model to push the
a crucial role in model decision-making during testing. It         negative OOD texts away from ID visual features. OOD Im-
stores ID-specific text features as the textual patterns and in-   ages Seen + Texts Unknown offers an opportunity to explore
tegrates them with ID visual information as visual patterns        various generative strategies for creating effective OOD sam-
to aid in OOD detection. SeTAR [Li et al., 2024c] extends          ples, making it a valuable testbed for investigating the impact
MCM by introducing selective low-rank approximations to            of synthetic data in OOD detection. OOD Images Unseen +
improve performance. TAG [Liu and Zach, 2024] augments             Texts Unknown adopts a minimalistic approach by forgoing
the label space by using only ID label information to improve      both outlier generation and OOD-specific prompt learning.
OOD detection.                                                     This ensures a computationally efficient pipeline, making it
   Another subcategory adopts the train-required strategy,         particularly suitable for scenarios with limited computational
eliminating the need to design OOD scores during testing.          resources.
GalLoP [Lafon et al., 2024] also does not require informa-
tion from OOD images or OOD texts. Instead of design-              5     Limitations and Future Directions
ing OOD scores, it focuses on ID images at both the global
and local levels, separately designing and training ID prompts     5.1    Limitations
(i.e., train-required) to achieve fine-grained and precise text-   OOD Information Leakage. While this concern is widely
to-image matching. DPM-T [Zhang et al., 2024b] improves            acknowledged, it remains crucial for the advancement of

## Page 7

CLIP-like OOD detection. CLIP is trained using contrastive          aries. Therefore, we think that, by using CLIP-based simi-
learning on large-scale image-text datasets, such as LAION-         larity sampling, we can select representative historical data
400M. As a result, during pre-training, CLIP becomes ex-            for replay, and store representative samples from previous
posed to and learns certain features of samples or categories       tasks to prevent the model from forgetting the previous OOD
that may appear in the test data. The leakage of OOD in-            boundaries; 3) CLIP-like Multi-label OOD Detection. Cur-
formation can introduce two key challenges: 1) The model            rent CLIP-like OOD detection methods are largely confined
may rely on previously learned categories or features, mis-         to single-label tasks. However, real-world scenarios often in-
classifying real OOD samples as part of an ID category; 2)          volve multi-label images. In multi-label tasks, a single sam-
Some OOD Texts Known methods that incorporate “no” or               ple may simultaneously contain both ID and OOD labels. Ef-
“not” semantics, may explicitly identify OOD data during            fectively distinguishing between ID and OOD labels in multi-
testing instead of genuinely inferring them. This can result        label tasks is a crucial research direction.
in artificially inflated performance, which may lead to issues      Diverse Real Scenarios. As discussed in Sec. 5.1, CLIP ex-
when applied in real-world scenarios. Therefore, further ex-        hibits robust classification and OOD detection capabilities for
ploration of solutions, such as refining pre-trained data, intro-   natural images. However, its potential in specialized domains
ducing specialized mechanisms, and designing tailored train-        remains largely unexplored. Here, we highlight some real-
ing strategies, is necessary to mitigate this issue.                world applications: 1) Intelligent Mine Safety: Underground
Evaluation Benchmarks.           This issue stems from infor-       mining environments are confined spaces with intricate con-
mation leakage. Current CLIP-like OOD detection methods             ditions. Strict safety regulations often restrict the deployment
often rely on commonly seen object categories during test-          of electronic devices such as sensors. However, inadequate
ing, leading to inevitable semantic overlap with CLIP’s pre-        monitoring equipment may fail to detect early warning signs
trained classes. Existing benchmark datasets require further        of mining accidents. Consequently, harnessing CLIP’s robust
improvement in task difficulty and category diversity. To           generalization capabilities, we seek to identify environmen-
truly assess the real detection capability of CLIP-like OOD         tal anomalies (OOD) that deviate from normal safety condi-
detection, test datasets should be replaced with those from         tions (ID), enabling proactive risk mitigation and emergency
more specialized domains, such as Pathology Image Datasets          preparedness; 2) Medical Electrocardiogram (ECG): Current
(e.g., CAMELYON16 & CAMELYON17, DigestPath 2019)                    ECG analysis systems mainly diagnose common conditions
or Geospatial Remote Sensing Datasets (e.g., EuroSAT and            such as arrhythmias and heart disease. However, certain
BigEarthNet).                                                       anomalies in ECG signals may not be included in standard
Data Noise Issue. Given the diverse and inconsistent quality        training datasets, rendering unimodal detection methods in-
of data sources, CLIP may encounter image and label noise           effective. With the contrastive learning mechanism of CLIP,
when pre-training on large web-scale datasets, making it vul-       it becomes possible to identify even minor variations in ECG
nerable to data noise. For instance, image noise arises from        signals. By comparing these signals with abnormal descrip-
low-quality images, atypical perspectives, blurriness, or oc-       tions provided by medical professionals, the system can as-
cluded scenes. Label noise occurs when certain image-text           sess whether an anomaly is OOD and potentially detect rare
pairs are imperfectly matched, such as a direct semantic mis-       or previously unknown diseases.
match where an image of a cat is incorrectly labeled as a           More Advanced Theories. Most current research on CLIP-
dog. Such data noise issue could distort the features learned       like OOD detection primarily focuses on practical applica-
by CLIP, ultimately degrading its ability to distinguish OOD        tions, while its theoretical understanding remains relatively
samples during testing.                                             underexplored. One theoretical perspective that merits deeper
                                                                    investigation is the concept of the Modality Gap [Liang et
5.2   Future Directions                                             al., 2022]. This phenomenon describes an observation that
Combining with More Complex and Dynamic Environ-                    after being processed by CLIP’s respective encoders, im-
ments. We believe that CLIP’s exceptional capabilities ex-          age embeddings and text embeddings tend to occupy dis-
tend beyond the well-explored domain of OOD detection.              tinct and non-overlapping regions within the shared feature
Moreover, there remains substantial untapped potential in           space. While some CLIP-like OOD detection methods advo-
combining OOD detection with more complex and dynamic               cate for bridging this gap to improve performance [Zhang et
scenarios. For example: 1) CLIP-like OOD in Long-Tailed             al., 2024b], the necessity and desirability of eliminating the
Recognition. In extremely imbalanced long-tailed distribu-          modality gap remain uncertain. As a result, despite CLIP’s
tions, the models may struggle to capture sufficient features       empirical success in the OOD detection task, a more rigorous
of minority classes, erroneously treating these samples as          theoretical analysis is required to understand its underlying
OOD data. We propose addressing this issue by utilizing             mechanisms and limitations.
CLIP’s powerful textual description capabilities, which al-
low for the creation of more specific and detailed descrip-         Acknowledgments
tions for tail classes; 2) OOD Challenges in Continual Learn-       This work was supported by the National Natural Science
ing with CLIP. Catastrophic forgetting is a fundamental chal-       Foundation of China (Grant No. 62376126, 62106102), the
lenge in continual learning. As the model adapts to new             Hong Kong Scholars Program (Grant No. XJ2023035) and
tasks, it may forget the knowledge from previous tasks. How-        the Fundamental Research Funds for the Central Universities
ever, traditional OOD detection methods often rely on fea-          (Grant No. NS2024058).
tures extracted from past tasks to establish decision bound-

## Page 8

References                                                        [Huang and Li, 2021] Rui Huang and Yixuan Li. Mos: To-
[Bai et al., 2024] Yichen Bai, Zongbo Han, Bing Cao, Xi-             wards scaling out-of-distribution detection for large se-
   aoheng Jiang, Qinghua Hu, and Changqing Zhang. Id-                mantic space. In Proceedings of the IEEE/CVF Confer-
   like prompt learning for few-shot out-of-distribution de-         ence on Computer Vision and Pattern Recognition, pages
   tection. In Proceedings of the IEEE/CVF Conference on             8710–8719, 2021.
   Computer Vision and Pattern Recognition, pages 17480–          [Huang et al., 2020] Xiaowei Huang, Daniel Kroening,
   17489, 2024.                                                      Wenjie Ruan, James Sharp, Youcheng Sun, Emese Thamo,
[Bitterwolf et al., 2020] Julian     Bitterwolf,     Alexander       Min Wu, and Xinping Yi. A survey of safety and trust-
   Meinke, and Matthias Hein. Certifiably adversarially              worthiness of deep neural networks: Verification, testing,
   robust detection of out-of-distribution data. Advances in         adversarial attack and defence, and interpretability. Com-
   Neural Information Processing Systems, 33:16085–16095,            puter Science Review, 37:100270, 2020.
   2020.                                                          [Jiang et al., 2024] Xue Jiang, Feng Liu, Zhen Fang, Hong
[Cao et al., 2024] Chentao Cao, Zhun Zhong, Zhanke Zhou,             Chen, Tongliang Liu, Feng Zheng, and Bo Han. Neg-
   Yang Liu, Tongliang Liu, and Bo Han. Envisioning outlier          ative label guided ood detection with pretrained vision-
   exposure by large language models for out-of-distribution         language models. In The Twelfth International Conference
   detection. In Forty-first International Conference on Ma-         on Learning Representations, 2024.
   chine Learning, 2024.                                          [Lafon et al., 2024] Marc Lafon, Elias Ramzi, Clément
[Chen et al., 2024] Jiankang Chen, Tong Zhang, Wei-Shi               Rambour, Nicolas Audebert, and Nicolas Thome. Gallop:
   Zheng, and Ruixuan Wang. Tagfog: Textual anchor                   Learning global and local prompts for vision-language
   guidance and fake outlier generation for visual out-of-           models. In European Conference on Computer Vision,
   distribution detection. In Proceedings of the AAAI Con-           pages 264–282. Springer, 2024.
   ference on Artificial Intelligence, volume 38, pages 1100–     [Lee et al., 2025] YuXiao Lee, Xiaofeng Cao, Jingcai Guo,
   1109, 2024.                                                       Wei Ye, Qing Guo, and Yi Chang. Concept matching with
[Cho and Choo, 2022] Wonwoo Cho and Jaegul Choo. To-                 agent for out-of-distribution detection. 2025.
   wards accurate open-set recognition via background-class       [Li et al., 2024a] Chaohua Li, Enhao Zhang, Chuanxing
   regularization. In European Conference on Computer Vi-            Geng, and Songcan Chen. All beings are equal in open
   sion, pages 658–674. Springer, 2022.                              set recognition. In Proceedings of the AAAI Conference
[Esmaeilpour et al., 2022] Sepideh Esmaeilpour, Bing Liu,            on Artificial Intelligence, volume 38, pages 13446–13454,
   Eric Robertson, and Lei Shu. Zero-shot out-of-distribution        2024.
   detection based on the pre-trained model clip. In Proceed-     [Li et al., 2024b] Tianqi Li, Guansong Pang, Xiao Bai, Wen-
   ings of the AAAI conference on artificial intelligence, vol-      jun Miao, and Jin Zheng. Learning transferable negative
   ume 36, pages 6568–6576, 2022.                                    prompts for out-of-distribution detection. In Proceedings
[Fort et al., 2021] Stanislav Fort, Jie Ren, and Balaji Laksh-       of the IEEE/CVF Conference on Computer Vision and Pat-
   minarayanan. Exploring the limits of out-of-distribution          tern Recognition, pages 17584–17594, 2024.
   detection. Advances in Neural Information Processing           [Li et al., 2024c] Yixia Li, Boya Xiong, Guanhua Chen,
   Systems, 34:7068–7081, 2021.                                      and Yun Chen. Setar: Out-of-distribution detection
[He et al., 2015] Kaiming He, Xiangyu Zhang, Shaoqing                with selective low-rank approximation. arXiv preprint
   Ren, and Jian Sun. Delving deep into rectifiers: Surpass-         arXiv:2406.12629, 2024.
   ing human-level performance on imagenet classification.        [Liang et al., 2022] Victor Weixin Liang, Yuhui Zhang,
   In Proceedings of the IEEE international conference on            Yongchan Kwon, Serena Yeung, and James Y Zou. Mind
   computer vision, pages 1026–1034, 2015.                           the gap: Understanding the modality gap in multi-modal
[Hendrycks and Gimpel, 2016] Dan Hendrycks and Kevin                 contrastive representation learning. Advances in Neural
   Gimpel. A baseline for detecting misclassified and out-of-        Information Processing Systems, 35:17612–17625, 2022.
   distribution examples in neural networks. arXiv preprint       [Liu and Zach, 2024] Xixi Liu and Christopher Zach.
   arXiv:1610.02136, 2016.                                           Tag: Text prompt augmentation for zero-shot out-of-
[Hendrycks et al., 2020] Dan Hendrycks, Xiaoyuan Liu,                distribution detection.      In European Conference on
   Eric Wallace, Adam Dziedzic, Rishabh Krishnan, and                Computer Vision, pages 364–380. Springer, 2024.
   Dawn Song. Pretrained transformers improve out-of-             [Liu et al., 2020] Weitang Liu, Xiaoyun Wang, John Owens,
   distribution robustness. In Proceedings of the 58th Annual        and Yixuan Li. Energy-based out-of-distribution detec-
   Meeting of the Association for Computational Linguistics,         tion. Advances in neural information processing systems,
   pages 2744–2751, 2020.                                            33:21464–21475, 2020.
[Hong et al., 2024] Zesheng Hong, Yubiao Yue, Yubin Chen,         [Liu et al., 2024] Kai Liu, Zhihang Fu, Chao Chen, Sheng
   Lele Cong, Huanjie Lin, Yuanmei Luo, Mini Han Wang,               Jin, Ze Chen, Mingyuan Tao, Rongxin Jiang, and Jieping
   Weidong Wang, Jialong Xu, Xiaoqi Yang, et al. Out-of-             Ye. Category-extensible out-of-distribution detection via
   distribution detection in medical image analysis: A survey.       hierarchical context descriptions. Advances in Neural In-
   arXiv preprint arXiv:2404.18279, 2024.                            formation Processing Systems, 36, 2024.

## Page 9

[Marsland, 2003] Stephen Marsland. Novelty detection in          [Scheirer et al., 2012] Walter J Scheirer,           Anderson
   learning systems. Neural computing surveys, 3(2):157–            de Rezende Rocha, Archana Sapkota, and Terrance E
   195, 2003.                                                       Boult. Toward open set recognition. IEEE transac-
[Ming et al., 2022a] Yifei Ming, Ziyang Cai, Jiuxiang Gu,           tions on pattern analysis and machine intelligence,
   Yiyou Sun, Wei Li, and Yixuan Li. Delving into out-              35(7):1757–1772, 2012.
   of-distribution detection with vision-language representa-    [Sun et al., 2022] Yiyou Sun, Yifei Ming, Xiaojin Zhu, and
   tions. Advances in neural information processing systems,        Yixuan Li. Out-of-distribution detection with deep near-
   35:35087–35102, 2022.                                            est neighbors. In International Conference on Machine
[Ming et al., 2022b] Yifei Ming, Ying Fan, and Yixuan Li.           Learning, pages 20827–20840. PMLR, 2022.
   Poem: Out-of-distribution detection with posterior sam-       [Sun et al., 2024] Hao Sun, Rundong He, Zhongyi Han,
   pling. In International Conference on Machine Learning,          Zhicong Lin, Yongshun Gong, and Yilong Yin. Clip-
   pages 15650–15665. PMLR, 2022.                                   driven outliers synthesis for few-shot ood detection. arXiv
[Miyai et al., 2023] Atsuyuki Miyai, Qing Yu, Go Irie, and          preprint arXiv:2404.00323, 2024.
   Kiyoharu Aizawa. Zero-shot in-distribution detection          [Wang et al., 2023] Hualiang Wang, Yi Li, Huifeng Yao, and
   in multi-object settings using vision-language foundation        Xiaomeng Li. Clipn for zero-shot ood detection: Teaching
   models. arXiv preprint arXiv:2304.04521, 2023.                   clip to say no. In Proceedings of the IEEE/CVF Interna-
[Miyai et al., 2024a] Atsuyuki Miyai, Jingkang Yang,                tional Conference on Computer Vision, pages 1802–1812,
   Jingyang Zhang, Yifei Ming, Yueqian Lin, Qing Yu,                2023.
   Go Irie, Shafiq Joty, Yixuan Li, Hai Li, et al. Gen-          [Xiao et al., 2020] Zhisheng Xiao, Qing Yan, and Yali Amit.
   eralized out-of-distribution detection and beyond in             Likelihood regret: An out-of-distribution detection score
   vision language model era: A survey. arXiv preprint              for variational auto-encoder. Advances in neural informa-
   arXiv:2407.21794, 2024.                                          tion processing systems, 33:20685–20696, 2020.
[Miyai et al., 2024b] Atsuyuki Miyai, Qing Yu, Go Irie, and      [Yang et al., 2022] Jingkang Yang, Pengyun Wang, Dejian
   Kiyoharu Aizawa. Locoop: Few-shot out-of-distribution            Zou, Zitang Zhou, Kunyuan Ding, Wenxuan Peng, Haoqi
   detection via prompt learning. Advances in Neural Infor-         Wang, Guangyao Chen, Bo Li, Yiyou Sun, et al. Openood:
   mation Processing Systems, 36, 2024.                             Benchmarking generalized out-of-distribution detection.
[Nie et al., 2024] Jun Nie, Yonggang Zhang, Zhen Fang,              Advances in Neural Information Processing Systems,
   Tongliang Liu, Bo Han, and Xinmei Tian. Out-of-                  35:32598–32611, 2022.
   distribution detection with negative prompts. In The          [Yang et al., 2024] Jingkang Yang, Kaiyang Zhou, Yixuan
   Twelfth International Conference on Learning Represen-           Li, and Ziwei Liu. Generalized out-of-distribution detec-
   tations, 2024.                                                   tion: A survey. International Journal of Computer Vision,
[Park et al., 2023] Sangha Park, Jisoo Mok, Dahuin Jung,            132(12):5635–5662, 2024.
   Saehyung Lee, and Sungroh Yoon. On the powerful-              [Zhang and Zhang, 2024] Yabin Zhang and Lei Zhang.
   ness of textual outlier exposure for visual ood detec-
                                                                    Adaneg: Adaptive negative proxy guided ood detec-
   tion. Advances in Neural Information Processing Systems,
                                                                    tion with vision-language models.           arXiv preprint
   36:51675–51687, 2023.
                                                                    arXiv:2410.20149, 2024.
[Radford et al., 2021] Alec Radford, Jong Wook Kim, Chris
                                                                 [Zhang et al., 2024a] Yabin Zhang, Wenjie Zhu, Chenhang
   Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agar-
   wal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack          He, and Lei Zhang. Lapt: Label-driven automated prompt
   Clark, et al. Learning transferable visual models from nat-      tuning for ood detection with vision-language models. In
   ural language supervision. In International conference on        European Conference on Computer Vision, pages 271–
   machine learning, pages 8748–8763. PMLR, 2021.                   288. Springer, 2024.
[Ren et al., 2019] Jie Ren, Peter J Liu, Emily Fertig, Jasper    [Zhang et al., 2024b] Zihan Zhang, Zhuo Xu, and Xiang Xi-
   Snoek, Ryan Poplin, Mark Depristo, Joshua Dillon, and            ang. Vision-language dual-pattern matching for out-of-
   Balaji Lakshminarayanan. Likelihood ratios for out-of-           distribution detection. In European Conference on Com-
   distribution detection. Advances in neural information           puter Vision, pages 273–291. Springer, 2024.
   processing systems, 32, 2019.                                 [Zhou et al., 2022] Kaiyang Zhou,           Jingkang Yang,
[Russakovsky et al., 2015] Olga Russakovsky, Jia Deng,              Chen Change Loy, and Ziwei Liu. Learning to prompt
   Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma,              for vision-language models. International Journal of
   Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael           Computer Vision, 130(9):2337–2348, 2022.
   Bernstein, et al. Imagenet large scale visual recogni-
   tion challenge. International journal of computer vision,
   115:211–252, 2015.
[Saar and Ure, 2013] Ellu Saar and Odd Bjørn Ure. Lifelong
   learning systems: overview and extension of different ty-
   pologies. Lifelong learning in Europe, pages 46–81, 2013.
