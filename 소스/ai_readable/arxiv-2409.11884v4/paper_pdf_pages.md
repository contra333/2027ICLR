# Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances - page-anchored PDF text

- Source ID: `arxiv-2409.11884v4`
- arXiv ID: `2409.11884v4`
- Original PDF: `소스/Out-of-Distribution Detection_ A Task-Oriented Survey of Recent Advances.pdf`
- PDF pages: 35
- Extracted with: WSL poppler `pdftotext -f N -l N -layout` on 2026-05-13T17:01:27+09:00
- Evidence note: use this file for page-anchored claims; verify equations/tables against the original PDF when exact layout matters.

## Page 1

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances

                                        SHUO LU, NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences; UCAS, China
                                        YINGSHENG WANG∗ , Anhui University, China
                                        LIJUN SHENG, University of Science and Technology of China, China
                                        LINGXIAO HE, Meituan, China
                                        AIHUA ZHENG, Anhui University, China
                                        JIAN LIANG† , NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences; UCAS, China
                                        Out-of-distribution (OOD) detection aims to detect test samples outside the training category space, which is an essential component
arXiv:2409.11884v4 [cs.LG] 4 Aug 2025




                                        in building reliable machine learning systems. Existing reviews on OOD detection primarily focus on method taxonomy, surveying the
                                        field by categorizing various approaches. However, many recent works concentrate on non-traditional OOD detection scenarios, such
                                        as test-time adaptation, multi-modal data sources and other novel contexts. In this survey, we uniquely review recent advances in
                                        OOD detection from the task-oriented perspective for the first time. According to the user’s access to the model, that is, whether the
                                        OOD detection method is allowed to modify or retrain the model, we classify the methods as training-driven or training-agnostic.
                                        Besides, considering the rapid development of pre-trained models, large pre-trained model-based OOD detection is also regarded
                                        as an important category and discussed separately. Furthermore, we provide a discussion of the evaluation scenarios, a variety
                                        of applications, and several future research directions. We believe this survey with new taxonomy will benefit the proposal of
                                        new methods and the expansion of more practical scenarios. A curated list of related papers is provided in the Github repository:
                                        https://github.com/shuolucs/Awesome-Out-Of-Distribution-Detection

                                        CCS Concepts: • Trustworthy Machine Learning → Out-of-distribution Detection.

                                        Additional Key Words and Phrases: Trustworthy Machine Learning, Out-of-distribution Detection

                                        ACM Reference Format:
                                        Shuo Lu, Yingsheng Wang, Lijun Sheng, Lingxiao He, Aihua Zheng, and Jian Liang. 2024. Out-of-Distribution Detection: A Task-
                                        Oriented Survey of Recent Advances. 1, 1 (August 2024), 35 pages. https://doi.org/XXXXXXX.XXXXXXX


                                        1   INTRODUCTION
                                        Machine learning methods have made significant progress under the closed-world assumption, where test data is drawn
                                        from the same category as the training set, known as in-distribution (ID). However, in the real world, models inevitably
                                        encounter test samples that do not belong to any training set category, commonly referred to as out-of-distribution
                                        (OOD) data. OOD detection [1] aims to identify and reject OOD samples rather than make overconfident predictions
                                        ∗ The first two authors contributed equally to this research. Emails: shuolucs@gmail.com, wangys200923@gmail.com.
                                        † Corresponding author. Email: liangjian92@gmail.com.


                                        Authors’ addresses: Shuo Lu, NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences; UCAS, Beijing, China; Yingsheng Wang, Anhui
                                        University, Hefei, China; Lijun Sheng, University of Science and Technology of China, Hefei, China; Lingxiao He, Meituan, Beijing, China; Aihua Zheng,
                                        Anhui University, Hefei, China; Jian Liang, NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences; UCAS, Beijing, China.

                                        Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
                                        made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
                                        of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
                                        servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
                                        © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.
                                        Manuscript submitted to ACM


                                        Manuscript submitted to ACM                                                                                                                             1

## Page 2

2                                                                                                                           Lu et al.


arbitrarily [2] while maintaining accurate classification for ID data. Models with superior OOD detection capabilities are
more reliable and have important applications in numerous security-critical scenarios. For instance, in medical diagnosis
systems, a model that cannot detect OOD samples will misjudge unknown diseases and cause serious misdiagnosis [3].
Similarly, autonomous driving algorithms [4] should detect unknown scenarios and resort to human control to avoid
accidents caused by arbitrary judgment.

Table 1. Comparison of recent survey papers on OOD detection. “Type” indicates whether the survey is organized from a method-
ological perspective (“Method”) or from a broader task-oriented perspective (“Task”). “OOD-Only” indicates an exclusive focus on
OOD detection; “LPMs” refers to coverage of methods based on large pretrained models; “Application” refers to in-depth analysis of
real-world use cases; “Multi-Modal” indicates consideration of multiple data modalities; “Test-Time” indicates surveys that explicitly
distinguish test-time adaptive methods from post-hoc approaches and provide dedicated discussion on test-time adaptation.


               Study               Year    Type      OOD-Only      LPMs     Application    Multi-Modal     Test-Time
               Salehi et al. [5]   2021   Method         ✗           ✗           ✗              ✗              ✗
               Yang et al. [6]     2021   Method         ✗           ✗           ✓              ✗              ✗
               Cui et al. [7]      2022   Method         ✓           ✗           ✓              ✗              ✗
               Lang et al. [8]     2023   Method         ✓           ✗           ✗              ✗              ✗
               Xu et al. [9]       2024   Method         ✗           ✓           ✗              ✓              ✗
               Miyai et al. [10]   2024   Method         ✗           ✓           ✗              ✗              ✗
               Ours                2024    Task          ✓           ✓           ✓              ✓              ✓


    Notably, several previous efforts have been dedicated to surveying and summarizing OOD detection in recent
years. Salehi et al. [5] offer a review covering Anomaly Detection, Novelty Detection, Open-Set Recognition and
OOD Detection, and analyzes the relationships and distinctions among these fields. Yang et al. [6] propose a unified
framework to discuss OOD detection with several similar topics and categorize existing work into classification-based,
density-based, distance-based and reconstruction-based methods. Concurrent with our work, Miyai et al. [10] discuss
CLIP-based methods across related domains, while our survey specifically centers on OOD detection and provides a
more comprehensive coverage of recent advances in this area. Cui and Wang [7] conduct a survey on OOD detection
from a methodological perspective but with an alternative classification criterion, including supervised, semi-supervised,
and unsupervised methods. Additionally, Lang et al. [8], Xu and Ding [9] review various OOD detection methods in the
context of natural language processing. However, previous works focus too much on the discussion from the perspective
of methods and lack an in-depth exploration from the viewpoint of task scenarios. Establishing a clear taxonomy of task
scenarios can enhance a comprehensive understanding of the field and assist practitioners in selecting the appropriate
method. Moreover, given the recent introduction of new paradigms (e.g., test-time learning paradigm [11–16]) and
methods based on large pre-trained models [17–21], there is an urgent need for a comprehensive survey that incorporates
the latest technologies.
    In this survey, we for the first time review the recent advances in OOD detection with a task-oriented taxonomy,
as illustrated in Fig. 1. Based on whether the method needs to control the pre-training process, we categorize OOD
detection algorithms into training-driven and training-agnostic methods. Considering the rapid development of large
pre-trained models nowadays and the evolving definition of OOD under such models, we also regard large pre-trained
model-based OOD detection as a separate section. In detail, training-driven methods achieve high detection capability
by designing the optimization process of the training stage. They are further classified and discussed according to
whether OOD data is used in training. Training-agnostic methods distinguish OOD data from ID ones based on a
well-trained model, skipping the time-consuming and expensive pre-training process in practice. According to whether
Manuscript submitted to ACM

## Page 3

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                                                                                                3

                                                                                                            MoodCat [22], RAE [23], MOOD [24], MOODv2 [25], PRE [26], LMD [27] DiffGuard [28],
                                                                                 Reconstruction-based
                                                                                                            DenoDiff [29]

                                                                                 Probability-based          HVCM [30], DDR [31], LID [32]

                                                                                 Logits-based               LogitNorm [33], UE-NL [34], DML [35]
                                                 Approaches with only
                                                    ID Data (§ 3.1)                                         ConfiCali [36], CODEs [37], CMG [38], VOS [39], NPOS [40], SHIFT [41], ATOL [42],
                                                                                 OOD Synthesis
                                                                                                            SSOD [43], SEM [44], Forte [45], HamOS [46], POP [47]

                                                                                 Prototype-based            Step [48], SIREN [49], CIDER [50], PALM [51], ReweightOOD [52], AROS [53], PFS [54]
                    Problem: Training-driven
                       OOD detection(§ 3)                                        A special case:
                                                                                                            OLTR [55], II-Mixup [56], AREO [57], IDCP [58], Open-Sampling [59], COOD [60]
                                                                                 Long-tail ID data

                                                                                 Boundary Regularization    OE [61], ELOC [62], Why-ReLU [63], SSL-GOOD [64], EnergyOE [65], MixOE [66]

                                                 Approaches with both ID
                                                                                 Outlier Mining             BD-Resamp [67], POEM [68], DAOL [69], DOE [70], MixOE [66], DivOE [71]
                                                  and OOD Data (§ 3.2)

                                                                                 Imbalanced ID              PASCAL [72], COCL [73], BERL [74], EAT [75]

                                                                                 Output-based               MSP [1], MaxLogits [76], Energy [65], GEN [77], ZODE [78], LogicOOD [79]

                                                                                 Distance-based             Mahalanobis [80], NNGuide [81], KNN [82], SSD [83],Mahalanobis++ [84]

                                                                                 Gradient-based             Grad [85], GradNorm [86], GradOrth [87], GAIA [88], OPNP [89], PRO [90]
                                                 Post-hoc Approaches (§ 4.1)
    OOD Detection




                                                                                                            ODIN [91], ReAct [92], VRA [93], SHE [94], ViM [95], Neco [96], ASH [97], NAC [98]
                                                                                 Feature-based              Optimal-FS [99], BLOOD [100], SCALE [101], DDCS [102], LINe [103], KANs [104],
                                                                                                            CADRef [105], ITP [106], NCI [107]
                    Problem: Training-agnostic
                       OOD detection (§ 4)
                                                                                 Density-based              GEM [108], ConjNorm [109]

                                                                                 Model-optimization-based   WOODS [110], AUTO [11], SODA [12], ATTA [13], SAL [14]
                                                 Test-time Adaptive Approaches
                                                             (§ 4.2)
                                                                                 Model-optimization-free    ETLT [15], AdaOOD [111], GOODAT [16], RTL [112], OODD [113]

                                                                                                            CLIPScope [19], RONIN [18], ZOC [17], MCM [114], CLIPN [115], NegLabel [116],
                                                                                 VLM-based                  NegPrompt-C [117], LAPT [118] , AdaNeg [119], CSP [120] , SimLabel [121] , [122] ,
                                                                                                            SeTAR [123] , OT-DETECTOR [124]
                                                 Zero-shot Approaches
                                                        (§ 5.1)
                                                                                                            WK-LLM [20], ODPC [125], LLM-OOD [21], CMA [126], ReGuide [127] , COOD [128],
                                                                                 LLM-based
                                                                                                            EOE [129]

                    Problem: Large pre-trained
                                                                                                            CLIP-OS [130], Local-Prompt [131], GalLop [132], ID-like [133], DSFG [134], LoCoOp [135],
                    model-based OOD detection                                    Fine-tuning based
                                                                                                            NegPrompt [136], SUPREME [137] , SCT [138] , GaCoOp [139]
                               (§ 5)             Few-shot Approaches
                                                        (§ 5.2)
                                                                                 Meta-learning based        OOD-MAML [140], HyperMix [141]

                                                 Full-shot Approaches
                                                                                 NPOS [40], PT-OOD [142], TOE [143]
                                                         (§ 5.3)



Fig. 1. Taxonomy of OOD detection problem scenarios and solutions. The first two categories (in blue) correspond to shallow
model-based methods, while the third category (in green) highlights approaches based on foundation models (large pre-trained
models). Different colors are used to visually distinguish the traditional shallow-model-based solutions from the newly emerging
foundation model-based methods, emphasizing the conceptual differences between them.




utilizing test samples to further improve OOD detection performance, we categorize them into post-hoc and test-time
methods. Large pre-trained model-based OOD detection methods focus on models such as vision language models or
large language models, which are pre-trained on vast datasets and excel in a wide array of tasks. We discuss them in
terms of whether they have access to a few examples, including zero-shot, few-shot and full-shot scenarios.
  The structure of this survey is as follows. We discuss the related topics of OOD detection in Sec. 2. Next, we summarize
training-driven OOD detection approaches in Sec. 3, and introduce the training-agnostic OOD detection methods in
Sec. 4. Then, in Sec. 5, we introduce large pre-trained model-based OOD detection. An overview of the evaluation
metrics, experimental protocols, and applications is presented in Sec. 6. Following that, we discuss promising trends
and open challenges in Sec. 7 to shed light on underexplored and potentially critical avenues.
                                                                                                                                                                Manuscript submitted to ACM

## Page 4

4                                                                                                                 Lu et al.


2     RELATED WORK
Anomaly Detection. Anomaly detection (AD) identifies data points that deviate from normal patterns [144], with wide
applications in fraud detection [145], network security [146], and system monitoring [147]. While both AD and OOD
detection identify unusual samples, AD differs in three main aspects. First, AD is typically unsupervised, using only
normal samples for training, whereas OOD detection involves labeled categories. Second, AD focuses on perceptual
anomalies, while OOD detects semantic shifts from unseen categories. Lastly, although both use AUROC [148] for
evaluation, AD employs AUPRO [149] for pixel-level performance. Beyond these standard metrics, practical evaluation
also assesses human-centric and operational criteria, such as detection latency [150, 151], the economic cost of
errors [152, 153], model interpretability [154, 155], and computational footprint [156, 157].
    Novelty Detection. Novelty detection (ND) aims to identify previously unseen data points [158], enabling systems
to adapt to new conditions or scenarios. Unlike AD, which finds deviations within known patterns, ND discovers new
patterns, such as emerging social media trends [159], novel species [158], or new document topics [160]. In essence, ND
detects surprises within familiar contexts, while OOD detection addresses data from entirely unfamiliar contexts.
    Open Set Recognition. Open Set Recognition (OSR) extends traditional classification by identifying both known
classes and instances from unknown categories [161]. Unlike OOD detection which identifies any OOD data, OSR
specifically focuses on recognizing novel classes within the same domain. This makes it particularly relevant for
applications like robotics [162] and autonomous systems [163].
    Outlier Detection. Outlier detection (OD) identifies individual data points that significantly deviate from the
majority [164]. Unlike OOD detection which encounters outliers only during deployment, OD is transductive with
inherent access to outliers. It finds applications in fraud detection and data cleaning [165].
    Zero-shot learning. Zero-shot learning aims to recognize objects without seeing their examples during training by
transferring knowledge from seen to unseen classes [166, 167]. Unlike OOD detection which flags anomalous data [80],
zero-shot learning actively classifies new categories through semantic relationships [168].
    Selective Classification. Selective classification (or reject option classification) allows models to abstain from pre-
dictions when confidence is insufficient [169, 170]. While OOD detection identifies samples from different distributions,
selective classification focuses on managing prediction uncertainty within the training distribution [171].
    Misclassification (Failure) Detection. Misclassification detection identifies incorrect predictions on ID data [172–
175], critical for domains like autonomous driving. It differs from OOD detection by focusing on errors within the
training distribution [176], rather than external inputs.

3     PROBLEM: TRAINING-DRIVEN OOD DETECTION
In the training-driven OOD detection problem, researchers design the pre-training process to obtain models with superior
OOD detection capabilities. Based on whether OOD data is accessible during training, we further divide methods under
this scenario into two folds: training with only ID data and training with both ID and OOD data, as shown in Fig. 2.

3.1    OOD Detection Approaches with only ID Data
Overview. Given the ID data, approaches in this section train a model on them and aim to utilize the model for
detecting OOD test samples while ensuring classification performance of the ID data. They focus specifically on mining
information from ID data, without explicitly relying on other information from real OOD data. We further differentiate
these methods into the following five categories: Reconstruction-based, Probability-based, Logits-based, OOD Synthesis,
and Prototype-based. Considering real-world requirements, we also delve into a specific scenario: Long-tail ID data.
Manuscript submitted to ACM

## Page 5

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                             5




Fig. 2. Illustration of training-driven OOD detection approaches. Dashed borders indicate that they are not used in the specific phase.
OOD images are excluded in the “Approaches with only ID data” on the left but are included in the “Approaches with both ID and
OOD data” on the right. In both cases, OOD labels are not utilized.




   Reconstruction-based. Reconstruction-based OOD detection methods focus on measuring representational dif-
ferences between original and reconstructed data. These approaches leverage the model’s capability to recover input
semantics, operating on the premise that OOD samples contain distinct semantic features from ID data, as shown
in Fig. 3 (a). Several notable works have advanced this direction. MoodCat [22] employs masked image modeling
with constrained synthesis based on classification outputs. Zhou [23] enhances detection by incorporating feature
activation analysis through an auxiliary module. Building upon masked image modeling, MOOD [24] and its successor
MOODv2 [25] demonstrate improved capabilities in learning data distributions. PRE [26] further develops this approach
by combining normalized flow with typicality-based penalties to effectively distinguish between OOD and ID samples.
   Recent advances in diffusion models [177] have led to improved training stability and image quality, prompting
their application in OOD detection. Graham et al. [29] introduce DDPM for reconstructing noise-corrupted images,
using multidimensional reconstruction error for OOD identification and allowing external regulation of the information
bottleneck. LMD [27] also applies diffusion models, disrupting and reconstructing data to separate OOD samples from
the original manifold. DiffGuard [28] leverages pre-trained diffusion models to enhance semantic discrepancies between
reconstructed OOD and original images.
   Probability-based. Probability-based approaches aim to establish probability models to describe the distribution of
training data. In this domain, Li et al. [30] model each ID category during training using multiple Gaussian mixture
models [178], and during the prediction phase, it combines Mahalanobis distance metrics to assess the likelihood of
anomalous classes. Huang et al. [31] address this issue by introducing two regularization constraints. The density
consistency regularization aligns the analytical density with low-dimensional class labels, and the contrastive distribution
regularization helps separate the density between ID and OOD samples. Furthermore, LID [32] introduces a new detection
criterion for the OOD detection paradox in the context of data generation by deep generative models. It measures
whether data should be classified as ID by estimating the local intrinsic dimension (LID) of the learned manifold of the
generative model, when the data is assigned a high probability and the probability mass is non-negligible.
                                                                                                           Manuscript submitted to ACM

## Page 6

6                                                                                                                         Lu et al.




Fig. 3. Illustration of main Training-driven OOD detection approaches: (a) Reconstruction-based: Identifies OOD samples by comparing
the dissimilarity between original feature representation F(x) and its reconstruction F’(x) through networks like VAE or DDPM. (b)
Prototype-based: Learns optimal ID prototypes during training for OOD distinction, where h represents penultimate layer features.
(c) OOD Synthesis: Generates synthetic OOD samples by extrapolating from ID data when real OOD data is unavailable. (d) Outlier
Mining: Leverages available real OOD data to generate typical OOD samples during training. (e) Boundary Regularization: Optimizes
model decision boundary using real OOD data without synthesis.



    Logits-based. These methods detect OOD samples by analyzing the logits (raw outputs before softmax) from
the neural network’s final layer. Logits typically represent the model’s confidence or probability for each category.
LogitNorm [33] uses logit normalization to enforce a constant vector norm on logits during training to mitigate issues
of model overconfidence. UE-NL [34] leverages Bayesian principles to jointly learn embeddings and uncertainty scores,
normalizing logits to improve robustness in OOD detection. DML [35] further advances this area by decoupling logits
and balancing key components, thereby mitigating attribute interference and enhancing detection performance.
    OOD Synthesis. In the task of OOD detection, incorporating the features of OOD data during model training can
enhance the OOD detection performance by allowing the model to better recognize and differentiate OOD data. Due to
the challenges in acquiring distribution information for OOD samples, some methods employ ID data to estimate the
distribution of OOD data, simulating real-world scenarios where a model encounters OOD data, as shown in Fig. 3 (c).
    Lee et al. [36] and VOS [39] both synthesize OOD samples from low-density or low-likelihood regions of the
ID feature space, but VOS enables adaptive outlier generation without external data or manual tuning. In contrast,
NPOS [40] avoids distributional assumptions by using non-parametric density estimation to select boundary samples,
enhancing flexibility and generality. SSOD [43] further differs by leveraging self-supervised sampling to extract OOD
signals directly from ID backgrounds, addressing synthesis bias.
    OOD detection methods often generate synthetic outliers from in-distribution data. Techniques include recombining
features with GANs (CODEs [37]), using conditional VAEs (CMG [38]), or applying Mixup (SEM [44]). Alternatively,
Manuscript submitted to ACM

## Page 7

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                        7


other approaches estimate data typicality by analyzing the local geometric and density statistics of the feature manifold
(FORTE [45]). SHIFT [41] leverages CLIP and latent diffusion models to generate OOD images by replacing ID object
regions with contextually consistent features, though maintaining high sample quality remains a challenge. ATOL [42]
introduces an auxiliary task that constructs non-overlapping regions for ID and OOD data in latent space, using
generated samples to reinforce this separation and aligning genuine ID data to enhance reliability. Recent studies show
that HamOS [46] uses Hamiltonian Monte Carlo [179] to generate virtual OOD anomalies by sampling from a Markov
chain [180] when real OOD data is unavailable. It projects features onto a hypersphere and leverages K-nearest neighbor
distances to guide anomaly generation. POP [47] instead refines decision boundaries using virtual OOD prototypes and
penalizes misclassifications, enhancing OOD detection without explicit outlier synthesis.
  Prototype-based. During the model training process, prototype-based OOD detection methods aim to model the
ID data using prototypes to learn the common distribution characteristics of the ID data. For OOD data, there is a
significant difference between the sample features and the ID prototype. In the testing phase, the model determines the
sample’s category by measuring the difference between the sample and the ID prototype. The general procedure of
these methods is outlined in Fig. 3 (b).
  SIREN [49] first models the distribution of ID data using the von Mises-Fisher(vMF) [181] model, which enables
representing each class as a compact cluster, i.e., a class prototype. Generally, the vMF distribution modeling formula can
be represented as: 𝑝 𝐷 (𝑧; 𝑝𝑘 , 𝑘) = 𝑍 𝐷 (𝑘)𝑒𝑥𝑝 (𝑘𝑝𝑇𝑘 𝑧), where 𝑝𝑘 is the 𝑘-th prototype with unit norm, 𝜅 ≥ 0 represents the
concentration around the mean, and 𝑍 𝐷 (𝜅) is the normalization factor. In the prototype-based approach, an embedding
vector 𝑧 is assigned to class 𝑐 with the following normalized probability:
                                                                      𝑍 𝐷 (𝑘𝑐 )𝑒𝑥𝑝 (𝑘𝑐 𝜇𝑇𝑐 𝑧)
                                   𝑝 (𝑦 = 𝑐 |𝑧; (𝑃 𝑗 , 𝑘 𝑗 )𝐶𝑗=1 ) = Í𝐶                      𝑇
                                                                                                   ,                           (1)
                                                                      𝑗=1 𝑍𝑑 (𝑘 𝑗 )𝑒𝑥𝑝 (𝑘 𝑗 𝜇 𝑗 𝑧)

where 𝑐 ∈ {1, 2, ...𝐶}. Additionally, within the loss function, it enforces alignment between the embedding vectors
of ID samples and class prototypes, constraining each ID class sample. This parametric OOD score can be directly
obtained after training, without requiring separate estimation. CIDER [50] builds upon SIREN by jointly optimizing
two losses to enhance data discriminability, encouraging the maximization of angular distances between prototypes of
different classes and the internal compactness of prototypes within the same class. The optimization process during
model training pertains to the prototypes of classes. AROS [53] enhances feature separation between ID and OOD data
by applying the Lyapunov stability theorem, guiding embeddings of ID and OOD samples toward distinct stable points.
Pseudo-OOD samples are generated by sampling from low-probability regions of the ID distribution. CNC [182] controls
neural collapse (NC) at different network stages: entropy regularization in the encoder alleviates NC, while the projector
head promotes NC, resulting in more compact prototypes and improved OOD detection. ReweightOOD [52] argues
that optimizing non-class data impedes achieving clear class separability, while focusing on fewer class data makes it
challenging to achieve lower MSE scores. To address this, they propose a re-weighting optimization strategy to balance
the significance of different losses. Although the ideas behind Step [48] are different, in the context of semi-supervised
tasks, it essentially generates clusters of unlabeled ID and OOD samples through a contrastive learning process, which
is conceptually similar to prototype learning. However, PALM [51] notes that representing each class with a single
prototype often fails to capture the internal diversity of data. To address this, PALM adopts a mixed-prototype strategy,
assigning multiple prototypes per class to better model informative representations. By jointly learning class-level




                                                                                                       Manuscript submitted to ACM

## Page 8

8                                                                                                                         Lu et al.




Fig. 4. Illustration of training-agnostic OOD detection approaches. Both methods require access to a pre-trained model. Post-hoc
approaches do not involve any operations during the post-training phase, while test-time adaptive approaches necessitate adaptation
based on the samples encountered during testing. “A’/B’ ” means that the original features are deformed; “Bank” means that some
samples are stored; “Clustering” means that clustering is performed for ID images.


prototypes and contrasting them across classes, PALM’s loss function encourages intra-class compactness and inter-
class separability at prototype level. Similarly, PFS [54] introduces a clustering loss that further promotes tight feature
clustering within each ID class, enhancing the consistency of ID representations.
    A special case: Long-tail ID data. ID data in real scenarios may present a long-tailed distribution due to the difficulty
of collection and frequency of occurrence. This imbalance will strongly affect the performance of OOD detection. Many
approaches are proposed to address the challenge of ID imbalance and enhance OOD detection capabilities. OLTR [55],
POP [56], and AREO [57] each address long-tailed recognition from different perspectives. OLTR enhances tail class
robustness by leveraging visual memory to bridge head and tail embeddings. POP focuses on medium-to-tail classes
through a prototype-based mixing strategy, generating mixed samples to improve class separation. In contrast, AREO
models sample uncertainty via evidential learning and dynamically adjusts training using a multi-scheduler mechanism
to better balance majority and minority classes.
    Most existing OOD detection methods assume a uniform probability distribution between OOD and ID samples.
To address class imbalance, Jiang et al. [58] recalibrate OOD scores by incorporating class priors and KL divergence,
improving robustness under skewed distributions. Open-Sampling [59] rebalances ID class priors using noisy OOD
labels sampled from a complementary distribution. In contrast, COOD [60] ensembles multiple OOD measures via a
supervised model, effectively mitigating individual method limitations and handling data imbalance.

3.2    OOD Detection Approaches with Both ID and OOD Data
Overview. In some known deployment scenarios, real OOD data can be easily collected at a low cost. Some methods
based on this assumption focus on how to use OOD data for better detection performance. Differing from methods
involving OOD Synthesis, in these research directions, models have access to real-world OOD data during the training
phase. The primary focus in such problems is on optimizing the model’s decision boundary, rather than the OOD data
itself. Due to the introduction of real OOD information, the decision boundary between ID and OOD data is further
accurately calculated.
Manuscript submitted to ACM

## Page 9

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                  9


  Boundary Regularization. The Boundary Regularization class of methods belongs to the traditional Outlier
Exposure (OE) approaches. The central idea of Hendrycks et al. [61] and Hein et al. [63] are to fully leverage OOD
data to optimize the model’s decision boundary, thus achieving OOD detection. Proponents of this concept can utilize
auxiliary anomaly datasets to enhance the OOD detector, enabling it to generalize and detect anomalous information
not encountered during training. The central idea of this method can be grasped from Fig. 3 (e).
  Specifically, given a model 𝑓 and the original loss function, the model training process aims to minimize the objective:
                                            h                                            i
                                 E (𝑥,𝑦)∼Din L (𝑓 (𝑥), 𝑦) + 𝜆E𝑥 ′ ∼D OE [LOE (𝑈 , 𝑓 (𝑥))] ,                            (2)
                                                                         out

over the parameters of 𝑓 , where 𝑥 ′ represents auxiliary anomaly data and 𝑈 denotes the uniform distribution. L        OE
                                                                                                            OE denotes
represents the cross-entropy loss with respect to 𝑈 . Here, Din denotes the distribution of ID data, while Dout
the distribution of OOD.
  The fundamental purpose is to compel the model to optimize the OOD data distribution to a uniform distribution, a
principle that is universal in OE-type approaches. The specific design of LOE can depend on other task requirements
and the chosen OOD score. This design can utilize a maximum softmax probability baseline [1] detector to detect
anomalous data. Compared to traditional softmax scores, EnergyOE [65] builds upon OE by leveraging energy scores
for better discrimination between ID and OOD samples, and it is less prone to issues of overconfidence. Specifically, its
calculation formula:
                                                                     𝐾
                                                                    ∑︁
                                             𝐸 (𝑥; 𝑓 ) = −𝑇 · 𝑙𝑜𝑔        𝑒 𝑓𝑖 (𝑥 )/𝑇 ,                                   (3)
                                                                    𝑖
where the temperature coefficient 𝑇 is used and 𝑓 (𝑥) denotes the discriminative neural classifier 𝑓 (𝑥) : R𝐷 → R𝐾 ,
which maps an input 𝑥 ∈ R𝐷 to 𝐾 real-valued logits.
   Mohseni et al. [64] train a model using a self-supervised approach, optimizing the objective function for unlabeled
OOD samples using pseudo-labeling to generalize OOD detection capabilities. Vyas et al. [62] similarly employ self-
supervised training of the classifier. Unlike the OE approach, its aim is to find a gap between the average entropy of
OOD and ID samples. MixOE [66] takes into account the beneficial effect of subtle OOD samples on enhancing the
generalization ability of OOD detection. Its main idea is to mix ID and OOD data samples to broaden the generalization
of OOD data. Training the model with these outliers can linearly decrease the prediction confidence with inputs from
ID to OOD samples, explicitly optimizing the generalization ability of the decision maker.
  Outlier Mining. The traditional OE concept assumes the existence of ID input Din and OOD input Dout , both
independently and heterogeneously distributed, originating from different sources. However, this premise cannot be
fully guaranteed in the current training process due to potential noise in the training OOD data. Outlier Mining differs
slightly from the traditional OE approach in that, although it also utilizes real-world OOD samples to address the
issue, it focuses on identifying the optimal selection within the existing OOD data. The main process is depicted in
Fig. 3 (d). Different approaches have been proposed for selecting representative OOD samples: POEM [68] utilizes
posterior sampling to identify high boundary score anomalies, while Li and Vasconcelos [67] employs data resampling
with priority score reweighting for hard negative instances. In contrast, DAOL [69] models OOD distribution using
Wasserstein balls to select challenging OOD samples based on the disparity between real and auxiliary data.
  Beyond solely relying on raw data, another direction in addressing this issue involves synthesizing representative
outlier data by utilizing authentic OOD data through information extrapolation. DivOE [71] introduces a novel learning
objective to alleviate challenges associated with limited auxiliary OOD datasets. It achieves this by adaptively inferring
and learning information from surrogate OOD data through the maximization of differences between generated OOD
                                                                                                 Manuscript submitted to ACM

## Page 10

10                                                                                                              Lu et al.


data and original data, given the specified anomalies. This adaptive inference extends to a broader spectrum, addressing
the limitations imposed by a finite auxiliary OOD dataset. Moreover, DOE [70] introduces a Min-Max learning strategy
to identify the most challenging OOD data for a synthetic model. Through model perturbation, the data is implicitly
transformed, and the model continues learning from this perturbed data to improve its robustness.
     Long-tail ID data. Recent studies have explored OOD detection in scenarios with imbalanced ID training data.
Different approaches have been proposed to address this challenge: PASCAL [72] uses contrastive loss to separate
tail-class from OOD data, while COCL [73] introduces a learnable tail class prototype to better distinguish between
them. Choi et al. [74] focuses on balancing cross-class distribution of auxiliary OOD data through energy regularization,
and EAT [75] expands the classification space by dynamically assigning virtual labels to OOD data during training.
     Despite the success and considerable attention received by methods like Outlier Exposure in the research community,
there are voices questioning the essence of allowing access to OOD data during training. Nevertheless, concerns
are raised that the superior classification performance observed in certain datasets may not necessarily translate to
competitiveness in real-world deployment, challenging the original intention of OOD detection.

4     PROBLEM: TRAINING-AGNOSTIC OOD DETECTION
Training-agnostic OOD detection focuses on adaptation strategies at test time for models that already possess downstream
classification capabilities, as opposed to the focus on classifier performance seen in training-driven OOD detection.
Based on whether they rely on dependencies among test data, methods are categorized into two types: post-hoc
and test-time adaptive approaches, as illustrated in Fig. 4. Post-hoc methods compute results for individual samples
independently, unaffected by changes in other samples. In contrast, test-time adaptive methods leverage information
shared among test samples to enhance OOD detection.

4.1    Post-hoc OOD Detection Approaches
Overview. Given a well-trained model, this problem scenario involves utilizing only the intermediate results computed
by the trained model during testing, without modifying any parameters of the model, to accomplish the OOD detection
task. Post-hoc methods are favored for their lightweight nature, low computational costs, and the fact that they require
minimal modifications to the model and objectives. Its main objective is to construct an effective scoring function that
can accurately reflect the behavior of ID data. These characteristics make them highly desirable for convenient and
straightforward deployment in practical scenarios. Post-hoc approaches are categorized into five types: Output-based,
Distance-based, Gradient-based, Feature-based and Density-based. Recent work on this type of problem has some recent
progress. A summary of the key factors involved in such methods is given in Table 2.
     Output-based. Algorithms based on output primarily aim to explore the latent representations of the output from
the intermediate layers of neural networks, which include logits and class distributions, among others. MSP [1] is the
first to employ the maximum softmax value to validate OOD detection effectiveness. For OOD samples, their output
probability distribution tends to be closer to a uniform distribution, demonstrating the model’s inability to correctly
classify the category. Departing from the MSP method, MaxLogits [76] detects OOD samples using the maximum logit
value, while Energy [65] employs an energy-based score related to input probability density. Compared to MaxLogits,
Energy is less sensitive to model overconfidence and provides a more theoretically grounded metric.
     GEN [77] introduces the concept of generalized entropy and directly utilizes Bregman divergence to compute the
statistical distance between the model’s probability output and uniform distribution, aiming to identify OOD data. In
addition, leveraging sufficient prior knowledge might be a viable solution. ZODE [78] performs predictions on samples
Manuscript submitted to ACM

## Page 11

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                   11

                             Table 2. Comparison of key components in OOD detection methods.


                                                                              Space
                         Type              Method
                                                           feature    logit   gradient    probability
                                          MSP [1]                                             ✓
                                       Maxlogits [76]                  ✓
                     Output-Based
                                        Energy [65]                    ✓
                                         GEN [77]             ✓                                ✓
                                      Mahalanobis [80]        ✓
                                       NNGuide [81]           ✓
                    Distance-Based
                                         KNN [82]             ✓
                                          SSD [83]            ✓
                                         Grad [85]                               ✓
                                      GradNorm [86]           ✓                  ✓             ✓
                    Gradient-Based     GradOrth [87]          ✓                  ✓
                                         GAIA [88]            ✓                  ✓
                                        OPNP [89]             ✓                  ✓
                                        ODIN [91]             ✓                  ✓             ✓
                                        ReAct [92]            ✓
                                         VRA [93]             ✓
                                         Vim [95]             ✓        ✓
                                         Neco [96]            ✓        ✓
                     Feature-Based       ASH [97]             ✓
                                      Optimal-FS [99]         ✓        ✓
                                       SCALE [101]            ✓
                                      ConjNorm [109]          ✓        ✓
                     Density-Based
                                        GEM [108]             ✓        ✓                       ✓



across multiple pre-trained models simultaneously to determine whether multiple models can identify OOD samples,
using this as a basis to distinguish between data. LogicOOD [79] presents a novel approach that uses first-order logic
for knowledge representation to perform OOD detection. This reasoning system uses prior knowledge to infer whether
an input is consistent with prior knowledge about the training distribution. It is particularly user-friendly in terms of
interpretability, as it allows comparing the output of samples against a knowledge base to determine whether they
belong to the ID data.
  Distance-based. Another approach in OOD detection research focuses on measuring statistical distance metrics.
Mahalanobis [80] is typically computed by calculating the distance between the feature vector and its mean. Specifically,
for each class, we compute the mean and covariance matrix of its feature vectors. During testing, it calculates the
Mahalanobis distance between the feature vector and the mean of each class. Mahalanobis++ [183] also uses Mahalanobis
distance, but applies L2 norm normalization to the features in the early stages of calculation to eliminate the instability
in feature extraction across different models, which affects the distribution assumption they rely on. SSD [83] essentially
utilizes the Mahalanobis distance. After being trained on unlabeled ID data via self-supervised representation learning, it
employs the Mahalanobis distance as a statistical measure for classification using the pre-trained model. In comparison,
while the Mahalanobis distance makes strong distributional assumptions about the data, KNN [82] explores the
effectiveness of non-parametric nearest-neighbor distance for OOD detection. By measuring the k-nearest neighbor
distance between input embeddings and training set embeddings, a threshold is designed to determine whether the
data belongs to the ID data. NNGuide [81] takes a step further in the direction of granularity by combining the idea of
                                                                                                   Manuscript submitted to ACM

## Page 12

12                                                                                                                 Lu et al.


KNN. It assigns weights before the traditional OOD Score, depending on the nearest neighbor distance between the
sample and the embeddings in the training set.
     Gradient-based. Gradient-based methods leverage model gradients to quantify uncertainty. Grad [85] and Grad-
Norm [86] both utilize gradient magnitudes, with the latter specifically comparing KL-divergence-based gradients
between ID and OOD samples. GradOrth [87], instead of focusing on overall gradient norms, projects gradients onto
low-rank subspaces to capture OOD features. GAIA [88] introduces channel-wise and zero-deflation abnormality
scores to assess distribution shifts without prior knowledge. OPNP [89] further improves OOD detection by pruning
parameters and neurons near zero, thus enhancing generalization. In recent research, PRO [90] introduces an adversarial
score, with the core idea being to perturb the input data using gradient descent to search for the local minimum score
near the original input. This method effectively lowers the confidence of OOD samples and is compatible with existing
MSP detection methods during score calculation, thereby enhancing detection performance. S&I [184] identifies the
issue of insufficient sample gradient feature discrimination in this type of method. By introducing adversarial samples
and integrating gradients along the attribution path, it achieves a more accurate feature distinction strategy.
     Feature-based. Research in this direction explores the role of intermediate neural features for OOD detection.
ODIN [91], inspired by adversarial examples, demonstrates that small input perturbations can enhance OOD detection
performance. The perturbation formula is as follows:

                                            𝑥˜ = 𝑥 + 𝜀𝑠𝑖𝑔𝑛(▽𝑥 𝑙𝑜𝑔𝑚𝑎𝑥𝑐 𝑝𝑐 (𝑥)),                                           (4)

where the parameter 𝜀 is the perturbation magnitude. For a given input 𝑥, compute its logit output 𝑝𝑐 (𝑥). Therefore,
ReAct [92] focuses on the high activation values in the intermediate results of the model. These activation values do
not affect the model’s classification, but truncating high activation values can significantly improve OOD detection
performance. VRA [93], an extended iteration of ReAct [92], builds upon the premise of ReAct which truncates only
high activation values. However, VRA posits that this might not be the optimal solution and, therefore, employs a
variational approach to seek an optimal solution. It utilizes piecewise functions to emulate suppression or amplification
operations, aiding the model in recognizing anomalous data. KANs [104] also focus on the neuron activation states.
Due to the local neuroplasticity characteristic of Kolmogorov-Arnold Networks [185], KANs distinguish between ID
and OOD data by comparing the activation response differences between the trained and untrained KANs. SHE [94]
introduces a storage-based approach by leveraging Hopfield energy [186] to facilitate OOD detection, averaging logits
per class to form a reference pattern. Building on channel selection, DDCS [102] evaluates and corrects neural network
channels according to their discriminative ability, using inter-class similarity and variance. Similarly, LINe [103] focuses
on neuron-level feature outputs, employing Shapley value pruning to retain only the most informative neurons for OOD
detection. CADRef [105] decouples the features by using a strategy based on the sign alignment between the relative
features and model weights, dividing the sample features into positive and negative errors. Intuitively, the positive
and negative error components influence the final output logits, which helps in identifying OOD images from this
perspective. ITP [106] optimizes the model by quantifying the contribution of model parameters to ID data prediction,
specifically by evaluating the contribution of each parameter through its partial derivatives and removing those with low
contribution. Additionally, during the testing phase, ITP uses a right-tail Z-score test to assess whether any parameter
exhibits overconfidence in the classification of a given test sample, further refining the model’s decision-making.
     Feature shaping refines intermediate features during forward propagation, offering a simple and effective way to
enhance OOD detection without affecting original classification results. Building on this, ViM [95] integrates features,
logits, and probabilities by constructing virtual logits, and leverages the penultimate layer’s null space—irrelevant to
Manuscript submitted to ACM

## Page 13

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                      13


classification yet highly effective for OOD detection. Its computation of score 𝑆 can be expressed in the following form:
                                                      ⊥
                                           𝑆 = −𝛼 ∥𝑧 𝑃 ∥ 2 + LogSumExp𝑓 (𝑧),                                                  (5)

where 𝛼 is a scaling constant, computed by the model. Here 𝑧 = 𝑧 𝑃 + 𝑧   𝑃⊥   and 𝑧   𝑃⊥   is the projection of 𝑧 to 𝑃 ⊥ . And it
          ⊥
have 𝑊 𝑧 𝑃 = 0. Additionally, 𝐿𝑜𝑔𝑆𝑢𝑚𝐸𝑥𝑝 represents the computation process of the energy function [65], and 𝑓 (𝑧)
represents the logit output of the model. The first term here represents virtual logits, while the second term represents
the score of the energy function. Neco [96] subsequently reveals the prevalent phenomenon of neural collapse in
contemporary neural networks, impacting OOD detection performance. The observation of orthogonal trends between
ID data and OOD data features is leveraged to differentiate OOD data. Similarly, NCI [107] studies the phenomenon
of neural collapse and discovers that during training, ID samples tend to cluster around class weight vectors in the
penultimate feature space, while OOD samples do not exhibit this clustering behavior. Therefore, NCI leverages the
proximity between features and class weight vectors to effectively distinguish ID samples from OOD samples, while
maintaining low computational cost. ASH [97] and neuron activation pruning focus on direct modification or removal of
activations, with ASH offering a simple dynamic scheme and pruning methods building upon its foundation. In contrast,
NAC [98] introduces neuron activation coverage as a statistical indicator, aiming to capture OOD likelihood based on
rarely activated neurons. BLOOD [100] and related methods leverage the smoother intermediate representations of ID
data, using this property to design new statistical measures for OOD discrimination. Feature shaping approaches [99]
partition the feature space and estimate piecewise approximations to logits, while SCALE [101] highlights the importance
of scaling metrics and observes lower pruning rates for OOD samples. Overall, these methods differ in whether they
reshape activations directly, exploit statistical properties, or design new metrics for improved OOD detection.
  Density-based. Recent density-based OOD detection methods have achieved notable improvements by more
effectively modeling the true data distribution. For example, GEM [108] employs class-conditional Gaussian distributions
with statistical metrics for model validation. Building upon this, ConjNorm [109] extends the framework to exponential
family distributions using Bregman divergence, enabling broader applications.


4.2   Test-Time Adaptive OOD Detection Approaches
Overview. Recently, many works [187–191] have focused on test-time adaptive methods, where a model, trained on
the training set, adapts during the testing phase. In the context of OOD detection, these methods aim to leverage
unlabeled test data—whether the complete test set or mini-batches—to improve performance through model adaptation.
Test-time adaptive approaches are based on the theoretical insight [192] that detecting OOD samples using only ID
samples without any additional knowledge can be challenging and potentially limited. These methods can be divided
into two categories according to whether the model be modified during the testing time: Model-optimization-based and
Model-optimization-free. Both of them undergo a post-training phase, during which the trained model can be adapted,
regardless of whether it is updated.
  Model-optimization-based. Model optimization-based methods enhance the trained model by leveraging unlabeled
data during the post-training phase.
  A line of methods [14, 110] leverage “wild data”—a mix of unlabeled ID and OOD samples—to enhance OOD detection.
WOODS [110] focuses on cleaning wild data for reliable OOD regularization, while SAL [14] analyzes its impact through
separability and learnability. However, wild data may introduce unintended information if it differs from test OOD, and
additional training requirements can increase computational costs.
                                                                                                      Manuscript submitted to ACM

## Page 14

14                                                                                                                        Lu et al.




Fig. 5. Overview and trends in LPM-based OOD Detection Methods. (a) presents statistical data on influential research works in
recent years, which involves papers indexed in prominent journals or conferences in the field. In the bar chart, the lower and upper
parts represent the total number of papers in the first and second halves of each year, respectively. (b) shows a trend pie chart
illustrating the proportion of keyword categories related to the research works in that year.


     Another line of model-optimization-based methods draw inspiration from semi-supervised-learning techniques [193]
for efficient post-training. While basic pseudo-labeling [194] labels test data directly, AUTO [11] uniquely focuses
on pseudo-OOD data with semantic consistency. In contrast, ATTA [13] and SODA [12] utilize both pseudo-ID and
pseudo-OOD data, with SODA implementing dual-loss optimization and ATTA applying differential weighting strategies.
     Model-optimization-free. Modifying the original trained model is infeasible in certain security-sensitive scenarios.
Therefore, methods enabling test-time adaptation without requiring model updates, termed “model-optimization-
free” techniques, are increasingly garnering interest. These approaches enhance the utilization of test data by either
memorizing it or incorporating additional modules on top of the original model. Both ETLT [15] and GOODAT [16]
preserve model integrity by training add-on modules for OOD score adjustment. ETLT leverages the linear correlation
between features and OOD scores, implementing both offline and online variants, while GOODAT develops a graph-
specific masker with GIB-boosted losses. In contrast, AdaOOD [111] and OODD [113] adopt non-parametric approaches,
utilizing memory banks with k-nearest neighbours and dynamic dictionaries, respectively.
     Online v.s. Offline. Most post-hoc methods [15, 195] emphasize the offline scenario, where OOD detectors remain
static and fixed after deployment. In contrast, the majority of test-time methods [13, 111] adopt the online scenario to
obtain the decision boundary dynamically, minimizing the risk of incorrect OOD predictions at each time step.
     More challenging scenario. In the context of test-time OOD detection scenarios, some researchers have proposed
more challenging configurations that demand a higher level of capability from the models. MOL [196] introduces a more
realistic problem scenario, namely Continuous Adaptive Out-of-Distribution (CAOOD) detection, aimed at addressing
the challenge of constantly changing ID and OOD distributions in the real world. The meta-learning approach is
employed to swiftly adapt models in response to the complexities encountered in various scenarios in MOL.

5    PROBLEM: LARGE PRE-TRAINED MODEL-BASED OOD DETECTION
Large pre-trained models have showcased remarkable performance in numerous downstream ID classification tasks,
but their potential in OOD detection tasks remains a less-explored area. Recent research [197] highlights a correlation
between higher ID classification accuracy and better OOD detection performance. Consequently, large pre-trained model-
based (LPM-based) OOD detection problem comes naturally. In recent years, large pre-trained models of various types,
Manuscript submitted to ACM

## Page 15

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                        15




Fig. 6. Illustration of large pre-trained model-based OOD detection approaches. In the training phase, zero-shot approaches require
only category labels of ID classes. Few-shot approaches need a subset of images of each ID class along with the category labels
(indicated by dashed lines). Full-shot approaches utilize both the category labels and all images of each ID class. None of these
approaches use labels or images of OOD categories.


including single-modal models (ViT [198], BERT [199], Diffusion [177]), visual language models (VLMs) (CLIP [200],
multi-modal Diffusion [201], ALIGN [202], ), and large language models (LLMs) (GPT3 [203]), have been increasingly
utilized for OOD detection tasks, as shown in Fig. 5. Based on the trend chart, it can be anticipated that LPM-based OOD
Detection will continue to be a key area of focus in the future. Leveraging the powerful representational capabilities
of large pre-trained models has further relaxed the constraints of OOD detection tasks, leading to a focus on more
challenging and realistic scenarios, which has emerged as a new hotspot. Given the number of ID shots exposed to the
large pre-trained model, OOD detection in this section can be classified into Zero-shot, Few-shot, and Full-shot OOD
detection, as shown in Fig. 6. The performance evaluations of several relevant competitive methods are summarized in
Table 3 to provide an understanding of the performance level of OOD detection in this area.

5.1   Zero-shot OOD Detection Approaches
Overview. Given the large pre-trained model and ID class names, we undertake the same task as the OOD detection,
precisely detecting OOD data to abstain from prediction and accurately classifying ID data. Note that we rely solely on
textual category knowledge, without the need for access to ID images. It’s important to clarify that “ID” here refers
to the ID of the specific downstream task, not the dataset ID data during pre-training. Existing research on zero-shot
OOD detection using vision-language models (VLMs-based) can be broadly categorized into two main approaches,
depending on the underlying models employed: diffusion-based and CLIP-based methods.
   Diffusion-based. Recent generative approaches, such as diffusion models [177], effectively capture the ID data by
modeling its underlying distribution. These models typically detect OOD samples by evaluating the likelihood of a test
input belonging to the modeled ID distribution. For instance, RONIN [18] leverages a diffusion model for ID inpainting,
combined with the CLIP model to compute similarity scores for OOD detection. However, most existing works focus
more extensively on CLIP-based methods due to their superior zero-shot capabilities and flexibility.
   CLIP-based. In contrast to diffusion-based approaches, CLIP-based methods have received significant attention for
zero-shot OOD detection. These methods exploit the powerful vision-language alignment capability of CLIP to perform
OOD detection without requiring additional training on OOD data. In the following, we focus on the development and
evolution of CLIP-based zero-shot OOD detection methods, discussing both approaches that require extra training and
post-hoc methods, as well as recent advances leveraging textual information.
                                                                                                        Manuscript submitted to ACM

## Page 16

16                                                                                                                        Lu et al.


     A representative approach, ZOC [17], introduces an image description generator trained on large-scale image
captioning datasets [204], enabling the generation of candidate unseen labels for OOD detection. Notably, ZOC treats
CLIP primarily as a feature extractor and does not inherently endow it with OOD discrimination capabilities. Building
upon this, CLIPN [115] enhances CLIP’s expressiveness by incorporating a “no-prompt” encoder, thereby empowering
CLIP to explicitly reject unfamiliar inputs—a critical step toward robust OOD detection in open-world scenarios.
Beyond methods that require further training, a parallel line of research investigates post-hoc zero-shot OOD detection,
aiming to maximize CLIP’s potential without additional fine-tuning. The most straightforward baseline in this category
computes the normalized text-image similarity as the OOD score. To improve upon this, MCM [114] replaces the
similarity score with a Maximum Concept Matching (MCM) score, providing a theoretically grounded and empirically
robust alternative. The MCM score is defined as:
                                                                                     ′
                                                                           𝑒 𝑠𝑖 (𝑥 )/𝜏
                                          𝑆 𝑀𝐶𝑀 (𝑥 ′ ; 𝑌𝑖𝑛 ,𝑇 , 𝜏) = max Í𝐾                   .                                  (6)
                                                                      𝑖          𝑠 𝑗 (𝑥 ′ )/𝜏
                                                                          𝑗=1 𝑒

where 𝑥 ′ denotes the test image, 𝑌𝑖𝑛 is the set of ID labels, 𝑇 (𝑡𝑖 ) is the text embedding for prompt 𝑡𝑖 , 𝑠𝑖 (𝑥 ′ ) is the cosine
similarity between the image and text features, and 𝜏 is the softmax temperature.
     Several works have enriched the textual context in CLIP-based OOD detection by augmenting candidate labels
with curated corpora, as seen in NegLabel [205], LAPT [118], CSP [120], and CLIPScope [19]. Specifically, NegLabel
selects negative labels from a corpus that are semantically dissimilar to ID classes, thereby sharpening the model’s
ability to distinguish OOD samples. CLIPScope [19] goes a step further by integrating these negative labels into a
Bayesian scoring framework to revise the original OOD score. LAPT, in contrast, utilizes text-to-image generation
or image retrieval models to obtain visual exemplars for both ID and negative labels, followed by prompt tuning to
optimize OOD discrimination. It is worth emphasizing that methods such as ZOC [17], CLIPN [115], NegLabel [205], and
CLIPScope [19] each propose distinct OOD scoring mechanisms tailored to their respective architectures. In contrast,
the MCM score can be seamlessly integrated into diverse CLIP-based frameworks.
     Recent zero-shot OOD detection methods mainly differ in negative sampling, noise handling, and semantic grouping.
AdaNeg [119] and AdaND [206] refine negative sampling and noise separation, while SimLabel [121] focuses on
automatic semantic grouping. For outlier representation, OLE [207] expands outlier label diversity through cluster-
ing and interpolation, and prompt learning [122] introduces new scoring functions for post-hoc detection without
retraining. Matrix-based methods further diversify the field, with SeTAR [123] applying low-rank approximations and
OT-DETECTOR [124] using optimal transport to capture semantic gaps.
     LLM-based. With the rapid advancement of LLMs, new opportunities have emerged in the field of OOD detection
by leveraging the extensive world knowledge and powerful semantic reasoning capabilities of LLMs. The primary
strength of LLMs lies in their ability to provide comprehensive and context-rich descriptions of ID labels, which can
be crucial for distinguishing between ID and OOD samples. However, a significant challenge in employing LLMs for
OOD detection is their tendency to generate hallucinations—false or misleading information—which can undermine the
reliability of OOD identification.
     Recent works explore the relationship between visual and semantic spaces to improve OOD detection. Studies [20, 125]
leverage the distinct properties of visually similar categories in semantic space. Dai et al. [20] develops a consistency-
based method with object detection to enhance LLM-based class descriptions, while Huang et al. [125] proposes ODPC
to generate peer classes using both ID labels and OOD samples. These approaches differ in their scoring mechanisms,
with Dai et al. [20] using MSP and ODPC employing KNN-based scoring. Building upon these foundations, subsequent

Manuscript submitted to ACM

## Page 17

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                            17

Table 3. Performance evaluation of some competitive methods based on CLIP-ViT-B/16 using the ImageNet-1K dataset as the ID
dataset and iNaturalist, SUN, Places, and Textures as OOD datasets. The results are cited from [114, 119, 120, 136] and the best result
is emphasized in bold.

                                    iNaturalist           SUN                 Places               Textures             Average
 Scenario    Method
                               AUROC ↑ FPR95 ↓      AUROC ↑ PFR95 ↓     AUROC ↑ FPR95 ↓      AUROC ↑ FPR95 ↓      AUROC ↑ FPR95 ↓
             ZOC [17]          86.09       87.30    81.20     81.51     83.39      73.06     76.46       98.90    81.79      85.19
             MCM [114]         94.59       32.20    92.25     38.80     90.31      46.20     86.12       58.50    90.82      43.93
 Zero-shot
             CLIPN [115]       95.27       23.94    93.93     26.17     92.28      33.45     90.93       40.83    93.10      31.10
             NegLabel [205]    99.49       1.91     95.49     20.53     91.64      35.59     90.22       43.56    94.21      25.40
             LAPT [118]        99.63       1.16     96.01     19.12     92.01      33.01     91.06       40.32    94.68      23.40
             CSP [120]         99.60       1.54     96.66     13.66     92.90      29.32     93.86       25.52    95.76      17.51
             AdaNeg [119]      99.71       0.59     97.44     9.50      94.55      34.34     94.93       31.27    96.66      18.92
             CoOp [211]        93.77       29.81    93.29     40.83     90.58      40.11     89.47       45.00    91.78      51.68
 Few-shot    LoCoOp [135]      96.86       16.05    95.07     23.44     91.98      32.87     90.19       42.28    93.52      28.66
             NegPrompt [136]   98.73       6.32     95.55     22.89     93.34      27.60     91.60       35.21    94.81      23.01




studies have systematically investigated the behavior of LLMs in OOD detection. For example, Liu et al. [21] analyze
the propensity of LLMs to detect near-OOD versus far-OOD samples, the impact of different fine-tuning strategies, and
the suitability of various OOD scoring functions.
   More recently, LLM-based approaches such as CMA [126], COOD [128], and EOE [129] have extended OOD detection
into multi-label and concept-rich regimes. These methods utilize LLMs not only as knowledge generators, but also as
proxy creators and synthetic data generators [208], thereby enriching the semantic context available for OOD detection.
The synergy between VLMs for visual grounding and LLMs for semantic reasoning has been further highlighted by
works such as ReGuide [127], which propose self-guided, image-adaptive concept generation to enhance robustness.
Collectively, these advancements demonstrate the growing potential of LLM-based techniques to address the complex
challenges of OOD detection by effectively integrating visual and semantic information.
   Remark. As mentioned above, “zero-shot” here refers to no exposure to ID images and only access to ID labels.
While some works [207, 209] improve performance by using known OOD class names as candidate labels, this approach
relies on the often unrealistic assumption that OOD labels are readily available. This setting is also distinct from recent
unsupervised fine-tuning approaches [210], which address a more realistic scenario where the unlabeled data for
adaptation is possibly contaminated with OOD samples.

5.2   Few-shot OOD Detection Approaches
Overview. Given the large pre-trained model and a few ID data, we can adapt the model using the ID data and
subsequently detect OOD test data. Zero-shot OOD detection does not necessitate any training images, making it
suitable for scenarios with high-security requirements. However, it may face challenges related to domain gaps with ID
downstream data, which can limit the performance of zero-shot methods. Therefore, there are many few-shot methods
employed in OOD detection, and their effectiveness is often superior to that of zero-shot OOD detection.
   Studies. Fine-tuning large pre-trained models with limited ID samples is a common adaptation strategy. Recent
studies [134, 212] systematically evaluate fine-tuning and parameter-efficient fine-tuning (PEFT) methods for OOD
detection in VLMs, particularly CLIP. Both works highlight the superior OOD detection performance of PEFT over
traditional fine-tuning. The MCM score [114] and prompt-based approaches are also shown to be effective. FLYP [213],
which mimics CLIP-style pretraining, further outperforms PEFT in zero-shot OOD detection. Notably, few-shot OOD
detection using outlier examples [209] is not included here, as it does not align with the focus on ID samples.
                                                                                                           Manuscript submitted to ACM

## Page 18

18                                                                                                                   Lu et al.


     Few-shot OOD Detection. Few-shot OOD detection aims to leverage a limited amount of ID data to adapt LPMs,
primarily through prompt learning or parameter-efficient fine-tuning. Early approaches such as CoOp [214] and
ZegCLIP [215] focus on optimizing the contextual words in prompts while keeping the backbone fixed, thus preserving
the generalization ability of the pre-trained models. Building upon this, LoCoOp [135] and IDPL [133] further enhance
context vector learning from a near-ID perspective: LoCoOp maximizes entropy to separate ID-irrelevant local features
(e.g., backgrounds) from textual embeddings of ID classes, while IDPL synthesizes ID-like outliers near the ID boundary
and employs a diversity loss to encourage variance among sampled OOD candidates. In addition, MMFT [216] projects
image and text features into a shared hyperspherical space and introduces a cross-modal alignment loss function to
promote the alignment of image and text representations in the hyperspherical space. However, these prompt-only
methods may underutilize the rich information contained in image features. Recent advances such as GalLoP [132] and
Local-Prompt [131] address this limitation by jointly optimizing global and local prompts.
     Beyond prompt learning, recent methods address the limitations of conventional fine-tuning, such as the loss of
OOD awareness, by merging original and adapted features [134] or introducing self-calibrated tuning frameworks
that balance ID and OOD objectives [138]. Other approaches improve prompt optimization through gradient-based
techniques [139]. To mitigate overfitting under limited supervision, strategies such as plug-and-play adapters [217],
multi-modal alignment [137], and boundary regularization with synthesized OOD features [130] have been proposed.
Further refinements include optimizing negative prompts to better distinguish ID/OOD boundaries [136]. Collectively,
these methods enhance few-shot OOD detection by improving adaptation, alignment, and boundary modeling.
     Meta-learning based. Meta-learning aims to devise a learning approach that enables rapid adaptation to new
challenges [218]. OOD-MAML [140] adapt model-agnostic meta-learning (MAML) for few-shot OOD detection. It
generates OOD samples and incorporates them along with ID data for the adapted N-way K-shot task, which is divided
into N sub-tasks, each focusing on K-shot OOD detection. The decision on whether test data is OOD is based on the
outcomes of these fast and simple N sub-tasks. In contrast, HyperMix [141] advocates for employing a hypernetwork-
based method to enhance sample augmentation without the necessity for extra outliers. This is because classes not
included in a specific meta-training task can act as OOD samples.

5.3    Full-shot OOD Detection Approaches
Overview. While this setup is generally less realistic than the first two (zero-shot and few-shot), we list them separately to
ensure a comprehensive review of existing methods across the spectrum. Given the full set of ID data and corresponding
labels, VLMs can enhance OOD detection significantly by fine-tuning. Moreover, a novel task called “PT-OOD” detection
is introduced.
     Fine-tuning based. With access to the complete dataset, then more data can be used to fine-tune the large pre-
trained model or the data can be used to better simulate the ID distribution, facilitating the differentiation of OOD
data. NPOS [40] proposes a non-parametric outlier synthesis technique to distinguish ID and OOD data by fine-tuning
CLIP with complete ID data. In contrast, TOE [143], while also using CE loss to constrain the model during fine-tuning,
builds on the ideas of OE by focusing on textual outliers within the CLIP framework to control the model’s recognition
capabilities, which differs significantly from directly using OOD images.
     PT-OOD Detection. “PT-OO” samples are OOD samples with overlap in pretraining data. After investigating and
elucidating the effects of various pre-training methodologies (supervised, self-supervised) on PT-OOD detection, Miyai
et al. [142] observe the low linear separability in feature space significantly degrades the PT-OOD detection performance.
They suggest using distinctive features for each instance to distinguish between ID and OOD samples.
Manuscript submitted to ACM

## Page 19

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                          19

Table 4. Summary of Datasets. The CARLA System is a simulation platform designed for evaluating OOD Detection in the field of
autonomous driving, hence its entire row is filled with dashes. “*”indicates that this dataset is divided by category, with some data
used as ID and other data as OOD. “/” symbol in the Usage column indicates that this dataset can be used as both ID and OOD data.


    TASK                       Dataset Name                  Data Type      # Classes     # Samples       Reference      Usage
                               CIFAR-10                      Images         10            60,000          [17, 115]      ID/OOD
                               CIFAR-100                     Images         100           60,000          [17, 115]      ID/OOD
                               MNIST                         Images         10            70,000          [1, 99]        ID/OOD
                               ImageNet-1K                   Images         1,000         1,431,167       [114, 115]     ID
    Image Classification
                               iNaturalist                   Images         5,089         675,170         [114, 115]     OOD
                               SUN                           Images         397           108,754         [114, 115]     OOD
                               Places                        Images         >205          >2,500,000      [114, 115]     OOD
                               Textures                      Images         47            5,640           [114, 115]     OOD
                               Cityscapes                    Images         30            25,000          [13]           ID
    Semantic Segmentation
                               Road Anomaly Dataset          Images         >5            60              [13]           OOD
                               PASCAL VOC                    Images         20            2,913           [18, 49]       ID
                               BDD100K                       Images         10            100,000         [18, 49]       ID
    Object Detection
                               MS-COCO                       Images         80            > 930           [18, 49]       OOD
                               OpenImages                    Images         601           1761            [18, 49]       OOD
    Autonomous Driving         CARLA System                  -              -             -               [219]          -
    Medical Image Analysis     Kvasir-Capsule*               Images         14            4,741,621       [220]          ID/OOD
                               News Category*                Text           41            210,000         [221]          ID/OOD
    Text Category
                               SST-2                         Text           2             215,154         [221]          ID/OOD
                               CLINC150*                     Text           150           22,500          [222]          ID/OOD
    Intent Detection           Banking77*                    Text           77            13,083          [222]          ID/OOD
                               StackOverflow*                Text           20            20,000          [222]          ID/OOD
                               MSCW                          Audio          31            >23,400,000     [223]          ID/OOD
    Audio
                               Vocalsound                    Audio          6             21,024          [223]          ID/OOD
    Graph data                 TU/OGB*                       Graph data     10            19,766          [16]           ID/OOD



6     EVALUATION AND APPLICATION
6.1     Evaluation metrics
In the vast majority of OOD detection tasks in the visual domain, the following evaluation metrics are commonly used:
     AUROC (Area Under the Receiver Operating Characteristic curve). This metric quantifies the likelihood
that a classifier will assign higher scores to ID samples compared to OOD samples. An elevated AUROC value is
indicative of superior model performance, signifying an enhanced ability to distinguish between ID and OOD instances.
Consequently, a higher value is desirable.
     AUPR (Area under the Precision-Recall curve). This metric is pertinent when the ID class is considered the
positive class and is particularly valuable in the context of imbalanced class distributions. It assesses the balance between
precision and recall, with a higher AUPR value indicating superior model performance.
     FPR@95 (False Positive Rate at 95% True Positive Rate). This metric delineates the false positive rate (FPR)
at the juncture where the true positive rate (TPR) reaches 95%. It essentially gauges the proportion of OOD samples
erroneously identified as ID, thus providing insight into the model’s propensity for false alarms at a high sensitivity
threshold. A reduced FPR@95% TPR is indicative of a model’s enhanced specificity in correctly flagging OOD samples
while maintaining high sensitivity towards ID samples. Therefore, a lower value is desirable.
                                                                                                          Manuscript submitted to ACM

## Page 20

20                                                                                                              Lu et al.


6.2    Experimental Protocols
In the traditional experimental protocol for OOD detection, test data is exclusively classified as either ID or OOD.
However, as the field has advanced, there is now a more nuanced distinction between OOD and ID data, which has led
to variations in the evaluation process.
     Subsequently, OOD data is categorized into near-OOD and far-OOD based on the degree of covariate shift from
ID data. This categorization corresponds to dividing OOD detection tasks into near-OOD and far-OOD detection.
It is evident the near-OOD detection task is more challenging, however, numerous methods [114, 133, 209] have
demonstrated excellent performance in this area.
     Recently, Yang et al. [44], Bai et al. [224] propose that we should consider cases where covariate shift occurs in
ID data, which is not taken into account previously. This is crucial to prevent the loss of model generalization. The
samples mentioned earlier are termed as cs-ID data, an abbreviation for “covariate shift ID” data. Consequently, a new
experimental protocol has been explored, called full-spectrum OOD detection. During the testing phase, the model is
expected to identify near-OOD and far-OOD instances. Additionally, it should refuse to provide predictions for the
OOD data and accurately predict ID and cs-ID data.

6.3    Application
This section presents a comprehensive review of both academic and real-world applications of OOD detection. To
promote further research and enable standardized evaluation, commonly used benchmark datasets are systematically
summarized in Table 4.
     To comprehensively review recent advances in OOD detection, we first categorize our discussion by key technical
domains (Computer Vision, Natural Language Processing, and Beyond), as each has distinct methods and theoretical
bases shaping OOD approaches. This structure enables a systematic analysis of core innovations and challenges in each
field. We then adopt an application-oriented perspective (see Table 5), highlighting how these techniques are adapted
and implemented in real-world scenarios.

6.3.1 Computer Vision. Most of the efforts in OOD detection have been devoted to the field of computer vision, we list
extensive vision-related tasks as follows:
      • Image Classification. Image classification represents the primary testbed for OOD detection research. Common
        ID datasets include MNIST [258], CIFAR [259], and ImageNet-1K [260], while OOD evaluation typically uses
        datasets like iNaturalist [261], SUN [262], and Textures [263]. Recent research categorizes OOD datasets based
        on their detection difficulty: near-OOD (e.g., SSB-hard [197], NINCO [264]) for harder-to-distinguish categories,
        and far-OOD (e.g., iNaturalist, OpenImage-O [95]) for more distinct categories.
      • Semantic Segmentation. Recent works [13] have started delving into the dense OOD detection task, also
        known as anomaly segmentation. The datasets used for evaluation include the Cityscapes dataset [265], the Road
        Anomaly dataset [251], and the recently developed SOOD-ImageNet [266].
        Object Detection. OOD detection for object detection tasks is a relatively recent research area [49], with
        evaluations commonly conducted on datasets such as PASCAL-VOC [267] and BDD-100K [268]. Recent methods
        explore diverse strategies, including ensemble architectures [228], prototype-based similarity [229], and visual-
        contextual augmentation [230].
      • 3D Object Detection. OOD detection in LiDAR-based 3D object detection generates synthetic OOD data and
        trains MLPs to distinguish ID and OOD targets, with new evaluation protocols for realistic scenarios [269].
Manuscript submitted to ACM

## Page 21

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                                  21

Table 5. Representative applications of OOD detection across five major domains, highlighting key use cases in industrial, medical,
safety, scientific, and general-purpose settings.

                 Domain          Task                       Application Description                                      Reference
                                 Autonomous Driving         Detecting unseen obstacles or traffic scenarios to ensure    [225]
                                                            driving safety in autonomous vehicles.
                 Industrial      3D Object Detection        Identifying OOD objects in 3D industrial environments,       [226, 227]
                Applications                                such as warehouses or robotics.
                                 Object Detection           Finding anomalous or unknown items in industrial pro-        [228–230]
                                                            duction lines.
                                 Earth Observation Imagery Detecting novel land-cover types or changes in satellite      [231]
                                                            and remote sensing images.
                                 Edge Computing             Real-time detection of abnormal data on edge devices to      [232]
                                                            enhance responsiveness and security.
                                 Time Series Analysis       Monitoring equipment by identifying abnormal patterns        [233]
                                                            for predictive maintenance.
                Healthcare       Medical Image Analysis     Detecting unknown diseases or anomalies in medical           [234–236]
                   and                                      images (e.g., X-ray, MRI).
                 Biology         Biological Classification  Identifying previously unseen species and enhancing bio-     [237]
                                                            diversity monitoring.
                                 Facial Authentication      Detecting spoofed or unregistered faces for enhanced         [238]
                  Security
                                                            security.
                    and
                                 Facial Expression Recogni- Identifying unknown or abnormal facial expressions in        [239]
                Surveillance
                                 tion                       surveillance.
                                 Human Action Recognition Detecting anomalous or dangerous actions in monitored          [240–242]
                                                            environments.
                                 Audio Recognition          Recognizing abnormal sounds such as alarms or intru-         [243]
                                                            sions.
                                 Astronomical Imaging       Discovering unknown celestial objects or phenomena in        [244]
                                                            astronomical data.
                 Scientific
                                 Solar Image Analysis       Detecting novel solar activities or anomalies in solar ob-   [245]
                 Research
                                                            servation images.
                                 Graph Neural Networks      Detecting anomalies in complex scientific data such as       [246, 247]
                                                            molecular structures or social networks.
                                 Mathematical Reasoning     Identifying OOD problem types in automated reasoning         [248]
                                                            systems.
                                 Spiking Neural Networks    Recognizing abnormal neural signals in bio-inspired com-     [249, 250]
                                                            puting models.
                                 Image Classification       Detecting unknown categories in image classification         [11, 22, 92]
                                                            systems.
                                 Semantic Segmentation      Identifying unknown regions or objects in pixel-level        [13, 251]
              General-Purpose
                                                            segmentation tasks.
               Applications
                                 Text Classification        Detecting unseen topics or intents in text data.             [252]
                                 Intent Detection           Recognizing new user intents in intelligent assistants and   [222, 253]
                                                            chatbots.
                                 Question-Answering Sys- Detecting questions that are out-of-scope or unanswer-          [254]
                                 tems                       able by the system.
                                 Document Classification    Detecting anomalies in documents containing multiple         [255]
                                                            data modalities.
                                 Reinforcement Learning     Detecting abnormal behavior or novel states in agent-        [256, 257]
                                                            environment interactions.




    • Autonomous Driving. Autonomous driving is a critical application of OOD detection. Recent work uses the
       CARLA simulator [270] to evaluate OOD detection performance in driving scenarios [219]. Mao et al. [219]
       propose a language-augmented latent representation method, leveraging the image-text cosine similarity from
       the CLIP model to improve transparency and controllability in detection. Another recent work focus on real-time
       OOD perception in trajectory prediction, further enhancing vehicle safety [271]. In real-world applications,
       Tesla’s “Shadow Mode” [225] identifies cases where AI predictions differ from human drivers in real traffic—these
       OOD samples are collected and used to retrain and improve autonomous driving models.
                                                                                                                  Manuscript submitted to ACM

## Page 22

22                                                                                                              Lu et al.


     • Medical Image Analysis. OOD detection is vital in medical imaging, using datasets like CIFAR-10 and Kvasir-
       Capsul [236] depending on the image category [220]. Recent studies focus on reliable OOD detection methods
       for digital pathologyfor improving OOD detection in gastrointestinal vision [235]. In practice, IDx-DR [234], the
       first FDA-approved autonomous AI diagnostic system, implements OOD detection by rejecting images that fail
       to meet quality standards or contain anomalies, deferring these cases to human experts for assessment.
     • Human Action Recognition. OOD detection plays a crucial role in ensuring the robustness of Human Action
       Recognition (HAR) models, particularly when faced with previously unseen actions. By leveraging advanced
       techniques such as attention-based debiasing [240], generative feature synthesis [241] and energy-based skeleton
       modeling [242], recent research has significantly improved OOD detection performance.
     • Solar Image Analysis. In space weather forecasting, OOD detection serves as a crucial tool for identifying
       solar anomalies. Recent unsupervised [245]approaches leverage Solar Dynamics Observatory data to enhance
       detection accuracy.
     • Facial Expression Recognition. In facial expression analysis, the challenge of distinguishing unknown expres-
       sions from known categories has led to advanced OOD detection methods. Recent approach [239] focuses on
       leveraging the unique characteristics of facial expression features, incorporating attention map consistency and
       cycle training mechanisms to effectively identify OOD samples.
     • Facial Authentication. In facial authentication systems, OOD detection is increasingly important for ensuring
       robust and secure identity verification. The FOOD framework [238] exemplifies this trend by employing 60
       GHz FMCW radar and a convolutional encoder-decoder architecture to achieve both accurate classification and
       reliable anomaly detection.
     • Earth Observation Imagery. For earth observation imagery, OOD detection enhances the identification of rare
       or unexpected events, such as natural disasters. Diffusion models like ODEED [231] have demonstrated strong
       performance in detecting OOD samples, with evaluations conducted on benchmark datasets such as SpaceNet 8
       to validate their effectiveness.


6.3.2 Natural Language Processing. OOD detection has become a crucial research topic in NLP, especially with
the advent of LLMs and their deployment in real-world applications [8, 272]. Below, we summarize key tasks and
representative methods:

     • Intent Detection. Intent Detection is a significant application of OOD detection in NLP, particularly in dialogue
       systems, where identifying user intentions that fall outside the predefined set of intents is crucial [222, 253].
       Common datasets for evaluation include CLINC150 [273] and HWU64 [274]. These datasets are specifically
       designed to evaluate OOD detection in intent classification tasks.
     • Text Classification. In text classification tasks, OOD detection helps identify texts that do not belong to any
       of the known categories. Datasets such as News Category [275] and SST-2 [276] are frequently used for ID
       data, with 20 Newsgroups (20NG) [277] often serving as an OOD dataset. Methods like VI-OOD [252] enhance
       detection by optimizing joint distributions.
     • Question-Answering Systems. Question-answering systems require robust mechanisms to handle diverse user
       queries while maintaining response accuracy. The challenge of identifying questions that fall outside the system’s
       knowledge domain has led to sophisticated OOD detection approaches.Recent LLM-based methods [254] leverage
       likelihood ratios between pretrained and fine-tuned models for effective OOD detection.
Manuscript submitted to ACM

## Page 23

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                              23


    • Mathematical Reasoning. Mathematical problem-solving systems face challenges when encountering problem
      types not seen during training. Using MultiArith [278] as the primary ID dataset, researchers evaluate OOD
      detection against diverse benchmarks. Recent advances, such as TV-Score [248], analyze embedding trajectory
      variations to enhance OOD detection performance.
    • Document Classification. In (multimodal) document classification, where documents integrate both text and
      images, OOD detection is essential for identifying unseen document types. Datasets such as Tobacco3482 [279]
      and FinanceDocs [255] are commonly used for evaluation. The AHM [255] approach enhances OOD detection
      by improving the representation of multimodal features.

6.3.3 Beyond Computer Vision and Natural Language Processing. In addition to the two data modalities mentioned
above, OOD detection still has many important applications across various types of data.

    • Audio Recognition. OOD detection in audio recognition has evolved from traditional probabilistic scoring [280]
      to methods leveraging confidence-based classification [280], neural embeddings [281], and deep nearest neighbor
      approaches [223]. Recent advances include autoencoder models based on WavLM [282], which are particularly
      effective for synthetic speech detection. Standard benchmarks such as MSCW [243] support comprehensive
      evaluation in this domain.
    • Graph Neural Networks. In GNNs, OOD detection encompasses node- and graph-level tasks, utilizing un-
      certainty estimation [246], energy-based methods [247], and semi-supervised learning [283]. Frameworks
      like HGOE [284] and PGR-MOOD [285] introduce anomalous or pseudo-OOD samples to enhance detection.
      GOLD [286] further advances this area by generating pseudo-OOD data through latent generative modeling.
      Evaluation commonly relies on datasets from TU [287] and OGB [288].
    • Reinforcement Learning. OOD detection is vital in reinforcement learning for improving robustness to
      novel states and actions [289]. Approaches typically involve uncertainty quantification [257], environment
      modifications [290], and offline RL techniques such as ADAC [256]. Evaluation frameworks often focus on
      environmental perturbations [290] to benchmark robustness.
    • Time Series Analysis. Time series OOD detection faces unique challenges due to temporal dependencies. Recent
      research explores modality-agnostic approaches for multivariate time series [291], with practical applications in
      fraud detection systems like AWS Fraud Detector [233].
    • Biological Classification. OOD detection identifies unseen species in biological classification. DNA barcodes
      enhance image-based classification through re-ranking, improving detection accuracy [237].
    • Spiking Neural Networks (SNN). SNNs’ sparse representations challenge OOD detection. Recent methods
      combine feedforward learning with distance-based scoring [249]. The N-MNIST dataset [250] is commonly used
      for evaluation, with OOD samples generated through diverse event patterns.
    • Astronomical Imaging. OOD detection distinguishes between simulated and real galaxy images. Bayesian
      comparison and sparse variational autoencoders [244] uncover subtle differences.
    • Edge Computing. Edge computing requires efficient, privacy-preserving OOD detection. SecDOOD [232] uses
      cloud-device architecture and HyperNetwork generation to avoid device-side backpropagation. CIFAR-10 and
      Tiny ImageNet serve as evaluation datasets.

  The deployment of OOD detection in high-stakes applications introduces considerations beyond predictive accuracy,
spanning regulatory compliance, operational safety, and algorithmic fairness. For compliance, OOD detection supports
                                                                                              Manuscript submitted to ACM

## Page 24

24                                                                                                                   Lu et al.


safety standards like ISO 214481 by identifying OOD scenarios, informing release decisions and enabling in-operation
validation via shadow-mode monitoring [292]. For safety, approaches [292, 293] are advancing from runtime input
flagging [292] toward monitoring system-level properties for predictive safety assurance with formal guarantees [293].
For fairness, mitigating bias against minority subgroups involves developing fairness-aware algorithms [294] and
constructing unbiased benchmarks that control for semantic ambiguity [295].

7     EMERGING TRENDS AND OPEN CHALLENGES
Despite rapid advancements in OOD detection, numerous emerging trends and less-explored challenges remain. In this
section, we explore emerging trends and open challenges from three distinct perspectives: methodologies, scenarios,
and applications.

7.1    Better methodologies of OOD detection
Meta-Learning Adaptation. Addressing the challenge of adapting to new data distributions at test time, meta-learning
has gained attention for its ability to facilitate efficient OOD detection in novel scenarios. Recent work also highlights the
importance of developing better sampling strategies to more effectively leverage outlier information during training [68].
     Model Selection. Choosing an appropriate OOD detection method for a specific task is challenging, as performance
varies across data distributions. Automated frameworks like MetaOOD [296] use meta-learning for zero-shot selection,
while DSDE [297] and score-combining methods [298] apply proportion estimation and likelihood ratio tests to improve
ensemble and score fusion. Notably, recent studies show that partially trained models can also be effective for OOD
detection [299], highlighting new possibilities for leveraging model dynamics.
     Active Learning. Active learning prioritizes labeling informative data points for efficiency in data-scarce settings.
In OOD detection, it focuses on ambiguous or uncertain samples to improve ID and OOD discrimination. SISOM [300]
combines active learning and OOD detection using feature space distances.
     Continual and Incremental Learning. Continual and incremental learning enable OOD detection models to adapt
to streaming data while preserving previous knowledge. Recent work includes unsupervised OOD detection [301],
OpenCIL benchmark [302], and hierarchical two-sample tests [303].
     Theoretically-Driven Score Design. While many post-hoc scoring methods have been proposed for OOD detection
in single-modal regimes, the transition to multi-modal domains necessitates more principled, theoretically motivated
score designs. For example, MCM [114] extends softmax-based approaches, but further innovation is needed to better
capture the complex relationships in multi-modal data.

7.2    More practical scenarios of OOD detection
Under the current trend, there is a growing need for the emergence of more practical scenarios, driven by the limitations
of existing impractical restrictions.
     Quick Test-Time Adaptation. The introduction of test-time adaptation scenarios, such as CAOOD [196], marks a
step forward in making OOD detection more reliable and adaptable in dynamic real-world environments.
     Federated Environments. Federated learning involves training models across decentralized clients, each with its
own data distribution [304, 305]. This distributional variability presents unique OOD detection challenges. Methods like
FOOGD [306] address these issues by enhancing robustness to both covariate and semantic shifts in federated settings.

1 https://www.iso.org/standard/77490.html

Manuscript submitted to ACM

## Page 25

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                  25


    Multi-Modal Detection. With the increasing prevalence of multi-modal data, OOD detection has expanded beyond
single modalities. MultiOOD [307] established a comprehensive benchmark with algorithms like A2D and NP-Mix.
Recent works [308] have advanced cross-modal alignment techniques.
    Multi-Label OOD Detection. Real-world applications often involve multi-label data, where each instance may
belong to multiple categories. Detecting OOD samples in these scenarios is challenging due to increased dimensionality
and label correlations. Recent approaches [309] leverage uncertainty modeling and subjective logic to improve robustness
in multi-label OOD detection.
    Open-Vocabulary Scenario. Traditional approaches often assume access to all ID categories, an assumption that
does not hold in open-vocabulary contexts. Recent work [136] explores learning migratable negative prompts, enabling
OOD detection even when only a subset of ID labels is available, thus paving the way for more generalizable models.
    Noisy Label Environments. Most prior studies focus on clean label settings, but real-world data often contains
label noise. Investigations into noisy label scenarios [310] offer insights and practical recommendations for improving
OOD detection robustness under such conditions.

7.3    New applications of OOD detection
Additional Modalities. While significant advances have been made in vision and language domains, OOD detection in
modalities such as speech and physiological signals remains underexplored. These areas, particularly emotion-related
physiological signals with high inter-subject variability, present promising directions for future research [307].
    Human-in-the-Loop Systems. Human expertise integration in OOD detection is crucial for safety-critical applica-
tions. Recent frameworks [311–313] leverage human feedback to optimize detection thresholds and guide annotation.
    Web Image Scraping. OOD detection is also finding novel applications in automating web image scraping. For
example, zero-shot ID detection [314] classifies images as ID if they contain relevant objects, offering a fresh perspective
on leveraging OOD detection for large-scale, real-world data acquisition.

8     CONCLUSION
OOD detection is a critical component for trustworthy machine learning. In this paper, we provide a comprehensive
review of recent advances in OOD detection, focusing for the first time on the problem scenario perspective: training-
driven, training-agnostic, and large pre-trained model-based OOD detection. We also summarize extensively used
evaluation metrics, experimental protocols, and diverse applications. We believe that our taxonomy of existing papers
and extensive discussion of emerging trends will contribute to a better understanding of the current state of research,
assist practitioners in selecting suitable approaches, and inspire new research hotspots.

9     ACKNOWLEDGEMENTS
We thank the editor and reviewers for their constructive feedback, and Zhen Jia and Yongcan Yu for their assistance with
the manuscript. This work was supported by the National Natural Science Foundation of China (62276256, U2441251),
the Young Elite Scientists Sponsorship Program by CAST (2023QNRC001), and the Young Scientists Fund of the State
Key Laboratory of Multimodal Artificial Intelligence Systems (ES2P100117).




                                                                                                  Manuscript submitted to ACM

## Page 26

26                                                                                                                                              Lu et al.


REFERENCES
  [1] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In Proc. ICLR, 2017.
  [2] Anh Nguyen, Jason Yosinski, and Jeff Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In
      Proc. CVPR, 2015.
  [3] Thomas Schlegl, Philipp Seeböck, Sebastian M Waldstein, Ursula Schmidt-Erfurth, and Georg Langs. Unsupervised anomaly detection with
      generative adversarial networks to guide marker discovery. In Proc. IPMI, 2017.
  [4] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In Proc. CVPR, 2012.
  [5] Mohammadreza Salehi, Hossein Mirzaei, Dan Hendrycks, Yixuan Li, Mohammad Hossein Rohban, and Mohammad Sabokrou. A unified survey on
      anomaly, novelty, open-set, and out-of-distribution detection: Solutions and future challenges. 2022.
  [6] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized out-of-distribution detection: A survey. arXiv preprint arXiv:2110.11334, 2021.
  [7] Peng Cui and Jinjia Wang. Out-of-distribution (ood) detection based on deep learning: A review. Electronics, 11(21):3500, 2022.
  [8] Hao Lang, Yinhe Zheng, Yixuan Li, Jian Sun, Fei Huang, and Yongbin Li. A survey on out-of-distribution detection in nlp. Transactions on Machine
      Learning Research, 2023.
  [9] Ruiyao Xu and Kaize Ding. Large language models for anomaly and out-of-distribution detection: A survey. arXiv preprint arXiv:2409.01980, 2024.
 [10] Atsuyuki Miyai, Jingkang Yang, Jingyang Zhang, Yifei Ming, Yueqian Lin, Qing Yu, Go Irie, Shafiq Joty, Yixuan Li, Hai Li, et al. Generalized
      out-of-distribution detection and beyond in vision language model era: A survey. arXiv preprint arXiv:2407.21794, 2024.
 [11] Puning Yang, Jian Liang, Jie Cao, and Ran He. Auto: Adaptive outlier optimization for online test-time ood detection. arXiv preprint arXiv:2303.12267,
      2023.
 [12] Andrew Geng, Kangwook Lee, and Yixuan Li. Soda: Stream out-of-distribution adaptation. 2023.
 [13] Zhitong Gao, Shipeng Yan, and Xuming He. Atta: Anomaly-aware test-time adaptation for out-of-distribution detection in segmentation. In Proc.
      NeurIPS, 2023.
 [14] Xuefeng Du, Zhen Fang, Ilias Diakonikolas, and Yixuan Li. How does unlabeled data provably help out-of-distribution detection? In Proc. ICLR,
      2024.
 [15] Ke Fan, Yikai Wang, Qian Yu, Da Li, and Yanwei Fu. A simple test-time method for out-of-distribution detection. arXiv preprint arXiv:2207.08210,
      2022.
 [16] Luzhi Wang, Dongxiao He, He Zhang, Yixin Liu, Wenjie Wang, Shirui Pan, Di Jin, and Tat-Seng Chua. Goodat: Towards test-time graph
      out-of-distribution detection. arXiv preprint arXiv:2401.06176, 2024.
 [17] Sepideh Esmaeilpour, Bing Liu, Eric Robertson, and Lei Shu. Zero-shot out-of-distribution detection based on the pre-trained model clip. In Proc.
      AAAI, 2022.
 [18] Quang-Huy Nguyen, Jin Peng Zhou, Zhenzhen Liu, Khanh-Huyen Bui, Kilian Q Weinberger, and Dung D Le. Zero-shot object-level ood detection
      with context-aware inpainting. arXiv preprint arXiv:2402.03292, 2024.
 [19] Hao Fu, Naman Patel, Prashanth Krishnamurthy, and Farshad Khorrami. Clipscope: Enhancing zero-shot ood detection with bayesian scoring.
      arXiv preprint arXiv:2405.14737, 2024.
 [20] Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, and Yongbin Li. Exploring large language models for multi-modal out-of-distribution detection. In
      Proc. EMNLP, 2023.
 [21] Bo Liu, Liming Zhan, Zexin Lu, Yujie Feng, Lei Xue, and Xiao-Ming Wu. How good are large language models at out-of-distribution detection? In
      Proc. COLING, 2024.
 [22] Yijun Yang, Ruiyuan Gao, and Qiang Xu. Out-of-distribution detection with semantic mismatch under masking. In Proc. ECCV, 2022.
 [23] Yibo Zhou. Rethinking reconstruction autoencoder-based out-of-distribution detection. In Proc. CVPR, 2022.
 [24] Jingyao Li, Pengguang Chen, Zexin He, Shaozuo Yu, Shu Liu, and Jiaya Jia. Rethinking out-of-distribution (ood) detection: Masked image modeling
      is all you need. In Proc. CVPR, 2023.
 [25] Jingyao Li, Pengguang Chen, Shaozuo Yu, Shu Liu, and Jiaya Jia. Moodv2: Masked image modeling for out-of-distribution detection. arXiv preprint
      arXiv:2401.02611, 2024.
 [26] Genki Osada, Tsubasa Takahashi, Budrul Ahsan, and Takashi Nishide. Out-of-distribution detection with reconstruction error and typicality-based
      penalty. In Proc. WACV, 2023.
 [27] Zhenzhen Liu, Jin Peng Zhou, Yufan Wang, and Kilian Q Weinberger. Unsupervised out-of-distribution detection with diffusion inpainting. In Proc.
      ICML, 2023.
 [28] Ruiyuan Gao, Chenchen Zhao, Lanqing Hong, and Qiang Xu. Diffguard: Semantic mismatch-guided out-of-distribution detection using pre-trained
      diffusion models. In Proc. ICCV, 2023.
 [29] Mark S Graham, Walter HL Pinaya, Petru-Daniel Tudosiu, Parashkev Nachev, Sebastien Ourselin, and Jorge Cardoso. Denoising diffusion models
      for out-of-distribution detection. In Proc. CVPR, 2023.
 [30] Jinglun Li, Xinyu Zhou, Pinxue Guo, Yixuan Sun, Yiwen Huang, Weifeng Ge, and Wenqiang Zhang. Hierarchical visual categories modeling: A
      joint representation learning and density estimation framework for out-of-distribution detection. In Proc. ICCV, 2023.
 [31] Wenjian Huang, Hao Wang, Jiahao Xia, Chengyan Wang, and Jianguo Zhang. Density-driven regularization for out-of-distribution detection. In
      Proc. NeurIPS, 2022.

Manuscript submitted to ACM

## Page 27

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                                             27


[32] Hamidreza Kamkari, Brendan Leigh Ross, Jesse C Cresswell, Anthony L Caterini, Rahul G Krishnan, and Gabriel Loaiza-Ganem. A geometric
     explanation of the likelihood ood detection paradox. In Proc. ICML, 2024.
[33] Hongxin Wei, Renchunzi Xie, Hao Cheng, Lei Feng, Bo An, and Yixuan Li. Mitigating neural network overconfidence with logit normalization. In
     Proc. ICML, 2022.
[34] Mouxiao Huang and Yu Qiao. Uncertainty-estimation with normalized logits for out-of-distribution detection. In Proc. CAICE, 2023.
[35] Zihan Zhang and Xiang Xiang. Decoupling maxlogit for out-of-distribution detection. In Proc. CVPR, 2023.
[36] Kimin Lee, Honglak Lee, Kibok Lee, and Jinwoo Shin. Training confidence-calibrated classifiers for detecting out-of-distribution samples. In Proc.
     ICLR, 2018.
[37] Keke Tang, Dingruibo Miao, Weilong Peng, Jianpeng Wu, Yawen Shi, Zhaoquan Gu, Zhihong Tian, and Wenping Wang. Codes: Chamfer
     out-of-distribution examples against overconfidence issue. In Proc. ICCV, 2021.
[38] Mengyu Wang, Yijia Shao, Haowei Lin, Wenpeng Hu, and Bing Liu. Cmg: A class-mixed generation approach to out-of-distribution detection. In
     Proc. ECML & PKDD, 2022.
[39] Xuefeng Du, Zhaoning Wang, Mu Cai, and Yixuan Li. Vos: Learning what you don’t know by virtual outlier synthesis. In Proc. ICLR, 2022.
[40] Leitian Tao, Xuefeng Du, Xiaojin Zhu, and Yixuan Li. Non-parametric outlier synthesis. In Proc. ICLR, 2023.
[41] Gitaek Kwon, Jaeyoung Kim, Hongjun Choi, Byungmoo Yoon, Sungchul Choi, and Kyu-Hwan Jung. Improving out-of-distribution detection
     performance using synthetic outlier exposure generated by visual foundation models. In Proc. BMVC, 2023.
[42] Haotian Zheng, Qizhou Wang, Zhen Fang, Xiaobo Xia, Feng Liu, Tongliang Liu, and Bo Han. Out-of-distribution detection learning with unreliable
     out-of-distribution sources. In Proc. NeurIPS, 2023.
[43] Sen Pei. Image background serves as good proxy for out-of-distribution data. In Proc. ICLR, 2024.
[44] Jingkang Yang, Kaiyang Zhou, and Ziwei Liu. Full-spectrum out-of-distribution detection. IJCV, pages 1–16, 2023.
[45] Debargha Ganguly, Warren Morningstar, Andrew Yu, and Vipin Chaudhary. Forte: Finding outliers with representation typicality estimation. In
     Proc. ICLR, 2025.
[46] Hengzhuang Li and Teng Zhang. Outlier synthesis via hamiltonian monte carlo for out-of-distribution detection. In Proc. ICLR, 2025.
[47] Mingrong Gong, Chaoqi Chen, Qingqiang Sun, Yue Wang, and Hui Huang. Out-of-distribution detection with prototypical outlier proxy. In Proc.
     AAAI, volume 39, pages 16835–16843, 2025.
[48] Zhi Zhou, Lan-Zhe Guo, Zhanzhan Cheng, Yu-Feng Li, and Shiliang Pu. Step: Out-of-distribution detection in the presence of limited in-distribution
     labeled data. In Proc. NeurIPS, volume 34, pages 29168–29180, 2021.
[49] Xuefeng Du, Gabriel Gozum, Yifei Ming, and Yixuan Li. Siren: Shaping representations for detecting out-of-distribution objects. In Proc. NeurIPS,
     2022.
[50] Yifei Ming, Yiyou Sun, Ousmane Dia, and Yixuan Li. How to exploit hyperspherical embeddings for out-of-distribution detection? In Proc. ICLR,
     2023.
[51] Haodong Lu, Dong Gong, Shuo Wang, Jason Xue, Lina Yao, and Kristen Moore. Learning with mixture of prototypes for out-of-distribution
     detection. In Proc. ICLR, 2024.
[52] Sudarshan Regmi, Bibek Panthi, Yifei Ming, Prashnna K Gyawali, Danail Stoyanov, and Binod Bhattarai. Reweightood: Loss reweighting for
     distance-based ood detection. In Proc. CVPR, pages 131–141, 2024.
[53] Hossein Mirzaei and Mackenzie W Mathis. Adversarially robust out-of-distribution detection using lyapunov-stabilized embeddings. In Proc. ICLR,
     2025.
[54] Yingwen Wu, Ruiji Yu, Xinwen Cheng, Zhengbao He, and Xiaolin Huang. Pursuing feature separation based on neural collapse for out-of-distribution
     detection. In Proc. ICLR, 2025.
[55] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Large-scale long-tailed recognition in an open world. In
     Proc. CVPR, 2019.
[56] Deval Mehta, Yaniv Gal, Adrian Bowling, Paul Bonnington, and Zongyuan Ge. Out-of-distribution detection for long-tailed and fine-grained skin
     lesion images. In Proc. MICCAI, 2022.
[57] Hitesh Sapkota and Qi Yu. Adaptive robust evidential optimization for open set detection from imbalanced data. In Proc. ICLR, 2023.
[58] Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, and Bo Han. Detecting out-of-distribution data through in-distribution
     class prior. In Proc. ICML, 2023.
[59] Hongxin Wei, Lue Tao, Renchunzi Xie, Lei Feng, and Bo An. Open-sampling: Exploring out-of-distribution data for re-balancing long-tailed
     datasets. In Proc. ICML, 2022.
[60] Laurens E Hogeweg, Rajesh Gangireddy, Django Brunink, Vincent J Kalkman, Ludo Cornelissen, and Jacob W Kamminga. Cood: Combined
     out-of-distribution detection using multiple measures for anomaly & novel class detection in large-scale hierarchical classification. In Proc. CVPR,
     pages 3971–3980, 2024.
[61] Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich. Deep anomaly detection with outlier exposure. In Proc. ICLR, 2019.
[62] Apoorv Vyas, Nataraj Jammalamadaka, Xia Zhu, Dipankar Das, Bharat Kaul, and Theodore L Willke. Out-of-distribution detection using an
     ensemble of self supervised leave-out classifiers. In Proc. ECCV, 2018.
[63] Matthias Hein, Maksym Andriushchenko, and Julian Bitterwolf. Why relu networks yield high-confidence predictions far away from the training
     data and how to mitigate the problem. In Proc. CVPR, 2019.
                                                                                                                         Manuscript submitted to ACM

## Page 28

28                                                                                                                                          Lu et al.


 [64] Sina Mohseni, Mandar Pitale, JBS Yadawa, and Zhangyang Wang. Self-supervised learning for generalizable out-of-distribution detection. In Proc.
      AAAI, 2020.
 [65] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. In Proc. NeurIPS, 2020.
 [66] Jingyang Zhang, Nathan Inkawhich, Randolph Linderman, Yiran Chen, and Hai Li. Mixture outlier exposure: Towards out-of-distribution detection
      in fine-grained environments. In Proc. WACV, 2023.
 [67] Yi Li and Nuno Vasconcelos. Background data resampling for outlier-aware classification. In Proc. CVPR, 2020.
 [68] Yifei Ming, Ying Fan, and Yixuan Li. Poem: Out-of-distribution detection with posterior sampling. In Proc. ICML, 2022.
 [69] Qizhou Wang, Zhen Fang, Yonggang Zhang, Feng Liu, Yixuan Li, and Bo Han. Learning to augment distributions for out-of-distribution detection.
      In Proc. NeurIPS, 2023.
 [70] Qizhou Wang, Junjie Ye, Feng Liu, Quanyu Dai, Marcus Kalander, Tongliang Liu, Jianye Hao, and Bo Han. Out-of-distribution detection with
      implicit outlier transformation. In Proc. ICLR, 2023.
 [71] Jianing Zhu, Geng Yu, Jiangchao Yao, Tongliang Liu, Gang Niu, Masashi Sugiyama, and Bo Han. Diversified outlier exposure for out-of-distribution
      detection via informative extrapolation. In Proc. NeurIPS, 2023.
 [72] Haotao Wang, Aston Zhang, Yi Zhu, Shuai Zheng, Mu Li, Alex J Smola, and Zhangyang Wang. Partial and asymmetric contrastive learning for
      out-of-distribution detection in long-tailed recognition. In Proc. ICML, 2022.
 [73] Wenjun Miao, Guansong Pang, Tianqi Li, Xiao Bai, and Jin Zheng. Out-of-distribution detection in long-tailed recognition with calibrated outlier
      class learning. In Proc. AAAI, 2024.
 [74] Hyunjun Choi, Hawook Jeong, and Jin Young Choi. Balanced energy regularization loss for out-of-distribution detection. In Proc. CVPR, 2023.
 [75] Tong Wei, Bo-Lin Wang, and Min-Ling Zhang. Eat: Towards long-tailed out-of-distribution detection. In Proc. AAAI, 2024.
 [76] Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joe Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling
      out-of-distribution detection for real-world settings. In Proc. ICML, 2022.
 [77] Xixi Liu, Yaroslava Lochman, and Christopher Zach. Gen: Pushing the limits of softmax-based out-of-distribution detection. In Proc. CVPR, 2023.
 [78] Feng Xue, Zi He, Yuan Zhang, Chuanlong Xie, Zhenguo Li, and Falong Tan. Enhancing the power of ood detection via sample-aware model
      selection. In Proc. CVPR, pages 17148–17157, 2024.
 [79] Konstantin Kirchheim, Tim Gonschorek, and Frank Ortmeier. Out-of-distribution detection with logical reasoning. In Proc. WACV, pages 2122–2131,
      2024.
 [80] Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks.
      In Proc. NeurIPS, 2018.
 [81] Jaewoo Park, Yoon Gyo Jung, and Andrew Beng Jin Teoh. Nearest neighbor guidance for out-of-distribution detection. In Proc. ICCV, 2023.
 [82] Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-of-distribution detection with deep nearest neighbors. In Proc. ICML, 2022.
 [83] Vikash Sehwag, Mung Chiang, and Prateek Mittal. Ssd: A unified framework for self-supervised outlier detection. Proc. ICLR, 2021.
 [84] Maximilian Mueller and Matthias Hein. Mahalanobis++: Improving ood detection via feature normalization. In Proc. ICML, 2025.
 [85] Jinsol Lee and Ghassan AlRegib. Gradients as a measure of uncertainty in neural networks. In Proc. ICIP, 2020.
 [86] Rui Huang, Andrew Geng, and Yixuan Li. On the importance of gradients for detecting distributional shifts in the wild. In Proc. NeurIPS, 2021.
 [87] Sima Behpour, Thang Doan, Xin Li, Wenbin He, Liang Gou, and Liu Ren. Gradorth: A simple yet efficient out-of-distribution detection with
      orthogonal projection of gradients. In Proc. NeurIPS, 2023.
 [88] Jinggang Chen, Junjie Li, Xiaoyang Qu, Jianzong Wang, Jiguang Wan, and Jing Xiao. Gaia: Delving into gradient-based attribution abnormality for
      out-of-distribution detection. In Proc. NeurIPS, 2023.
 [89] Chao Chen, Zhihang Fu, Kai Liu, Ze Chen, Mingyuan Tao, and Jieping Ye. Optimal parameter and neuron pruning for out-of-distribution detection.
      In Proc. NeurIPS, volume 36, 2024.
 [90] Wenxi Chen, Raymond A Yeh, Shaoshuai Mou, and Yan Gu. Leveraging perturbation robustness to enhance out-of-distribution detection. In Proc.
      CVPR, 2025.
 [91] Shiyu Liang, Yixuan Li, and Rayadurgam Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. In Proc.
      ICLR, 2018.
 [92] Yiyou Sun, Chuan Guo, and Yixuan Li. React: Out-of-distribution detection with rectified activations. In Proc. NeurIPS, 2021.
 [93] Mingyu Xu, Zheng Lian, Bin Liu, and Jianhua Tao. Vra: Variational rectified activation for out-of-distribution detection. In Proc. NeurIPS, 2023.
 [94] Jinsong Zhang, Qiang Fu, Xu Chen, Lun Du, Zelin Li, Gang Wang, Shi Han, Dongmei Zhang, et al. Out-of-distribution detection based on
      in-distribution data patterns memorization with modern hopfield energy. In Proc. ICLR, 2022.
 [95] Haoqi Wang, Zhizhong Li, Litong Feng, and Wayne Zhang. Vim: Out-of-distribution with virtual-logit matching. In Proc. CVPR, 2022.
 [96] Mouïn Ben Ammar, Nacim Belkhir, Sebastian Popescu, Antoine Manzanera, and Gianni Franchi. Neco: Neural collapse based out-of-distribution
      detection. In Proc. ICLR, 2024.
 [97] Andrija Djurisic, Nebojsa Bozanic, Arjun Ashok, and Rosanne Liu. Extremely simple activation shaping for out-of-distribution detection. In Proc.
      ICLR, 2023.
 [98] Yibing Liu, Chris Xing Tian, Haoliang Li, Lei Ma, and Shiqi Wang. Neuron activation coverage: Rethinking out-of-distribution detection and
      generalization. In Proc. ICLR, 2024.


Manuscript submitted to ACM

## Page 29

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                                            29


 [99] Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, and Stephen Gould. Towards optimal feature-shaping methods for out-of-
      distribution detection. In Proc. ICLR, 2024.
[100] Fran Jelenić, Josip Jukić, Martin Tutek, Mate Puljiz, and Jan Šnajder. Out-of-distribution detection by leveraging between-layer transformation
      smoothness. In Proc. ICLR, 2024.
[101] Kai Xu, Rongyu Chen, Gianni Franchi, and Angela Yao. Scaling for training time and post-hoc out-of-distribution detection enhancement. In Proc.
      ICLR, 2024.
[102] Yue Yuan, Rundong He, Yicong Dong, Zhongyi Han, and Yilong Yin. Discriminability-driven channel selection for out-of-distribution detection. In
      Proc. CVPR, pages 26171–26180, 2024.
[103] Seong Tae Kim Yong Hyun Ahn, Gyeong-Moon Park. Line: Out-of-distribution detection by leveraging important neurons. In Proc. CVPR, 2023.
[104] Alessandro Canevaro, Julian Schmidt, Mohammad Sajad Marvi, Hang Yu, Georg Martius, and Julian Jordan. Advancing out-of-distribution detection
      via local neuroplasticity. In Proc. ICLR, 2025.
[105] Zhiwei Ling, Yachen Chang, Hailiang Zhao, Xinkui Zhao, Kingsum Chow, and Shuiguang Deng. Cadref: Robust out-of-distribution detection via
      class-aware decoupled relative feature leveraging. In Proc. CVPR, 2025.
[106] Haonan Xu and Yang Yang. Itp: Instance-aware test pruning for out-of-distribution detection. In Proc. AAAI, volume 39, pages 21743–21751, 2025.
[107] Litian Liu and Yao Qin. Detecting out-of-distribution through the lens of neural collapse. In Proc. CVPR, 2023.
[108] Peyman Morteza and Yixuan Li. Provable guarantees for understanding out-of-distribution detection. In Proc. AAAI, 2022.
[109] Bo Peng, Yadan Luo, Yonggang Zhang, Yixuan Li, and Zhen Fang. Conjnorm: Tractable density estimation for out-of-distribution detection. In
      Proc. ICLR, 2024.
[110] Julian Katz-Samuels, Julia B Nakhleh, Robert Nowak, and Yixuan Li. Training ood detectors in their natural habitats. In Proc. ICML, 2022.
[111] YiFan Zhang, Xue Wang, Tian Zhou, Kun Yuan, Zhang Zhang, Liang Wang, Rong Jin, and Tieniu Tan. Model-free test time adaptation for
      out-of-distribution detection. arXiv preprint arXiv:2311.16420, 2023.
[112] Ke Fan, Tong Liu, Xingyu Qiu, Yikai Wang, Lian Huai, Zeyu Shangguan, Shuang Gou, Fengjian Liu, Yuqian Fu, Yanwei Fu, and Xingqun Jiang.
      Test-time linear out-of-distribution detection. In Proc. CVPR, pages 23752–23761, June 2024.
[113] Yifeng Yang, Lin Zhu, Zewen Sun, Hengyu Liu, Qinying Gu, and Nanyang Ye. Oodd: Test-time out-of-distribution detection with dynamic
      dictionary. arXiv preprint arXiv:2503.10468, 2025.
[114] Yifei Ming, Ziyang Cai, Jiuxiang Gu, Yiyou Sun, Wei Li, and Yixuan Li. Delving into out-of-distribution detection with vision-language representa-
      tions. In Proc. NeurIPS, 2022.
[115] Hualiang Wang, Yi Li, Huifeng Yao, and Xiaomeng Li. Clipn for zero-shot ood detection: Teaching clip to say no. In Proc. ICCV, 2023.
[116] Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, and Bo Han. Negative label guided ood detection with pretrained
      vision-language models. In Proc. ICLR, 2023.
[117] Jun Nie, Yonggang Zhang, Zhen Fang, Tongliang Liu, Bo Han, and Xinmei Tian. Out-of-distribution detection with negative prompts. In Proc.
      ICLR, 2024.
[118] Yabin Zhang, Wenjie Zhu, Chenhang He, and Lei Zhang. Lapt: Label-driven automated prompt tuning for ood detection with vision-language
      models. In ECCV, 2024.
[119] Yabin Zhang and Lei Zhang. Adaneg: Adaptive negative proxy guided ood detection with vision-language models. In Proc. NeurIPS, 2024.
[120] Mengyuan Chen, Junyu Gao, and Changsheng Xu. Conjugated semantic pool improves ood detection with pre-trained vision-language models. In
      Proc. NeurIPS, 2024.
[121] Shu Zou, Xinyu Tian, Qinyu Zhao, Zhaoyuan Yang, and Jing Zhang. Simlabel: Consistency-guided ood detection with pretrained vision-language
      models. arXiv preprint arXiv:2501.11485, 2025.
[122] Myong Chol Jung, He Zhao, Joanna Dipnall, Belinda Gabbe, and Lan Du. Enhancing near ood detection in prompt learning: Maximum gains,
      minimal costs. arXiv preprint arXiv:2405.16091, 2024.
[123] Yixia Li, Boya Xiong, Guanhua Chen, and Yun Chen. Setar: Out-of-distribution detection with selective low-rank approximation. In Proc. NeurIPS,
      2024.
[124] Yu Liu, Hao Tang, Haiqi Zhang, Jing Qin, and Zechao Li. Ot-detector: Delving into optimal transport for zero-shot out-of-distribution detection.
      2025.
[125] K Huang, G Song, Hanwen Su, and Jiyan Wang. Out-of-distribution detection using peer-class generated by large language model. arXiv preprint
      arXiv:2403.13324, 2024.
[126] Yuxiao Lee, Xiaofeng Cao, Jingcai Guo, Wei Ye, Qing Guo, and Yi Chang. Concept matching with agent for out-of-distribution detection. In Proc.
      AAAI, volume 39, pages 4562–4570, 2025.
[127] Jihyo Kim, Seulbi Lee, and Sangheum Hwang. Reflexive guidance: Improving oodd in vision-language models via self-guided image-adaptive
      concept generation. Proc. ICLR, 2025.
[128] Zhendong Liu, Yi Nian, Henry Peng Zou, Li Li, Xiyang Hu, and Yue Zhao. Cood: Concept-based zero-shot ood detection. arXiv preprint
      arXiv:2411.13578, 2024.
[129] Chentao Cao, Zhun Zhong, Zhanke Zhou, Yang Liu, Tongliang Liu, and Bo Han. Envisioning outlier exposure by large language models for
      out-of-distribution detection. In Proc. ICML, 2024.


                                                                                                                        Manuscript submitted to ACM

## Page 30

30                                                                                                                                           Lu et al.


[130] Hao Sun, Rundong He, Zhongyi Han, Zhicong Lin, Yongshun Gong, and Yilong Yin. Clip-driven outliers synthesis for few-shot ood detection.
      arXiv preprint arXiv:2404.00323, 2024.
[131] Fanhu Zeng, Zhen Cheng, Fei Zhu, Hongxin Wei, and Xu-Yao Zhang. Local-prompt: Extensible local prompts for few-shot out-of-distribution
      detection. In Proc. ICLR, 2025.
[132] Marc Lafon, Elias Ramzi, Clément Rambour, Nicolas Audebert, and Nicolas Thome. Gallop: Learning global and local prompts for vision-language
      models. arXiv preprint arXiv:2407.01400, 2024.
[133] Yichen Bai, Zongbo Han, Changqing Zhang, Bing Cao, Xiaoheng Jiang, and Qinghua Hu. Id-like prompt learning for few-shot out-of-distribution
      detection. In Proc. CVPR, 2024.
[134] Jiuqing Dong, Yongbin Gao, Heng Zhou, Jun Cen, Yifan Yao, Sook Yoon, and Park Dong Sun. Towards few-shot out-of-distribution detection.
      arXiv preprint arXiv:2311.12076, 2023.
[135] Atsuyuki Miyai, Qing Yu, Go Irie, and Kiyoharu Aizawa. Locoop: Few-shot out-of-distribution detection via prompt learning. In Proc. NeurIPS, 2023.
[136] Tianqi Li, Guansong Pang, Xiao Bai, Wenjun Miao, and Jin Zheng. Learning transferable negative prompts for out-of-distribution detection. In
      Proc. CVPR, 2024.
[137] Yimu Wang, Evelien Riddell, Adrian Chow, Sean Sedwards, and Krzysztof Czarnecki. Mitigating the modality gap: Few-shot out-of-distribution
      detection with multi-modal prototypes and image bias estimation. arXiv preprint arXiv:2502.00662, 2025.
[138] Geng Yu, Jianing Zhu, Jiangchao Yao, and Bo Han. Self-calibrated tuning of vision-language models for out-of-distribution detection. In Proc.
      NeurIPS, 2024.
[139] Baoshun Tong, Kaiyu Song, and Hanjiang Lai. Enhancing few-shot out-of-distribution detection with gradient aligned context optimization. In
      Proc. ICASSP, 2025.
[140] Taewon Jeong and Heeyoung Kim. Ood-maml: Meta-learning for few-shot out-of-distribution detection and classification. In Proc. NeurIPS,
      volume 33, pages 3907–3916, 2020.
[141] Nikhil Mehta, Kevin J Liang, Jing Huang, Fu-Jen Chu, Li Yin, and Tal Hassner. Hypermix: Out-of-distribution detection and classification in
      few-shot settings. In Proc. WACV, 2024.
[142] Atsuyuki Miyai, Qing Yu, Go Irie, and Kiyoharu Aizawa. Can pre-trained networks detect familiar out-of-distribution data? arXiv preprint
      arXiv:2310.00847, 2023.
[143] Sangha Park, Jisoo Mok, Dahuin Jung, Saehyung Lee, and Sungroh Yoon. On the powerfulness of textual outlier exposure for visual ood detection.
      In Proc. NeurIPS, volume 36, pages 51675–51687, 2023.
[144] Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. ACM computing surveys (CSUR), 41(3):1–58, 2009.
[145] Tahereh Pourhabibi, Kok-Leong Ong, Booi H Kam, and Yee Ling Boo. Fraud detection: A systematic literature review of graph-based anomaly
      detection approaches. Decision Support Systems, 133:113303, 2020.
[146] Aleksandar Lazarevic, Levent Ertoz, Vipin Kumar, Aysel Ozgur, and Jaideep Srivastava. A comparative study of anomaly detection schemes in
      network intrusion detection. In Proc. SDM, pages 25–36, 2003.
[147] Min Du, Feifei Li, Guineng Zheng, and Vivek Srikumar. Deeplog: Anomaly detection and diagnosis from system logs through deep learning. In
      Proc. CCS, pages 1285–1298, 2017.
[148] Markus Wehler, Judith Kokoska, Udo Reulbach, Eckhart Georg Hahn, and Richard Strauss. Short-term prognosis in critically ill patients with
      cirrhosis assessed by prognostic scoring systems. Hepatology, 34(2):255–261, 2001.
[149] Joao PC Bertoldo, Dick Ameln, Ashwin Vaidya, and Samet Akçay. Aupimo: Redefining visual anomaly detection benchmarks with high speed and
      low tolerance. arXiv preprint arXiv:2401.01984, 2024.
[150] Alexander Lavin and Subutai Ahmad. Evaluating real-time anomaly detection algorithms–the numenta anomaly benchmark. In 2015 IEEE 14th
      International Conference on Machine Learning and Applications (ICMLA), pages 38–44. IEEE, 2015.
[151] Rohan Sinha, Amine Elhafsi, Christopher Agia, Matthew Foutter, Edward Schmerling, and Marco Pavone. Real-time anomaly detection and reactive
      planning with large language models. arXiv preprint arXiv:2407.08735, 2024.
[152] Zhixiang Xu, Matt Kusner, Kilian Weinberger, and Minmin Chen. Cost-sensitive tree of classifiers. In Proc. ICML, pages 133–141. PMLR, 2013.
[153] Ga-Yeong Kim, Su-Min Lim, and Ieck-Chae Euom. A study on performance metrics for anomaly detection based on industrial control system
      operation data. Electronics, 11(8):1213, 2022.
[154] Rui Jiang, Yijia Xue, and Dongmian Zou. Interpretability-aware industrial anomaly detection using autoencoders. IEEE Access, 11:60490–60500,
      2023.
[155] Keval Doshi and Yasin Yilmaz. Towards interpretable video anomaly detection. In Proc. WACV, pages 2655–2664, 2023.
[156] Olumuyiwa Ibidunmoye, Francisco Hernández-Rodriguez, and Erik Elmroth. Performance anomaly detection and bottleneck identification. ACM
      Computing Surveys (CSUR), 48(1):1–35, 2015.
[157] Álvaro Huertas-García, Carlos Martí-González, Rubén García Maezo, and Alejandro Echeverría Rey. A comparative study of machine learning
      algorithms for anomaly detection in industrial environments: performance and environmental impact. In International conference on trends in
      sustainable computing and machine intelligence, pages 373–389. Springer, 2023.
[158] Marco AF Pimentel, David A Clifton, Lei Clifton, and Lionel Tarassenko. A review of novelty detection. Signal processing, 99:215–249, 2014.
[159] Quanzhi Li, Armineh Nourbakhsh, Sameena Shah, and Xiaomo Liu. Real-time novel event detection from social media. In Proc. ICDE, pages
      1129–1139, 2017.
Manuscript submitted to ACM

## Page 31

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                                                  31


[160] Cem Aksoy, Fazli Can, and Seyit Kocberber. Novelty detection for topic tracking. Journal of the american society for information science and
      technology, 63(4):777–795, 2012.
[161] Abhijit Bendale and Terrance E Boult. Towards open set deep networks. In Proc. CVPR, pages 1563–1572, 2016.
[162] Stephen Marsland, Ulrich Nehmzow, and Jonathan Shapiro. On-line novelty detection for autonomous mobile robots. Robotics and Autonomous
      Systems, 51(2-3):191–206, 2005.
[163] Alexander Amini, Wilko Schwarting, Guy Rosman, Brandon Araki, Sertac Karaman, and Daniela Rus. Variational autoencoder for end-to-end
      control of autonomous driving with novelty detection and training de-biasing. In Proc. IROS, pages 568–575, 2018.
[164] Karanjit Singh and Shuchita Upadhyaya. Outlier detection: applications and techniques. International Journal of Computer Science Issues (IJCSI), 9
      (1):307, 2012.
[165] Amruta D Pawar, Prakash N Kalavadekar, and Swapnali N Tambe. A survey on outlier detection techniques for credit card fraud detection. IOSR
      Journal of Computer Engineering, 16(2):44–48, 2014.
[166] Bernardino Romera-Paredes and Philip Torr. An embarrassingly simple approach to zero-shot learning. In Proc. ICML, pages 2152–2161, 2015.
[167] Wei Wang, Vincent W Zheng, Han Yu, and Chunyan Miao. A survey of zero-shot learning: Settings, methods, and applications. ACM Transactions
      on Intelligent Systems and Technology (TIST), 10(2):1–37, 2019.
[168] Yongqin Xian, Christoph H Lampert, Bernt Schiele, and Zeynep Akata. Zero-shot learning—a comprehensive evaluation of the good, the bad and
      the ugly. IEEE transactions on pattern analysis and machine intelligence, 41(9):2251–2265, 2018.
[169] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Proc. NeurIPS, volume 30, 2017.
[170] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey
      on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology, 15(3):1–45, 2024.
[171] Ran El-Yaniv et al. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11(5), 2010.
[172] Sabeen Ahmed, Dimah Dera, Saud Ul Hassan, Nidhal Bouaynaya, and Ghulam Rasool. Failure detection in deep neural networks for medical
      imaging. Frontiers in Medical Technology, 4:919046, 2022.
[173] Hao Dong, Moru Liu, Jian Liang, Eleni Chatzi, and Olga Fink. To trust or not to trust your vision-language model’s prediction. arXiv preprint
      arXiv:2505.23745, 2025.
[174] Arda Inceoglu, Eren Erdal Aksoy, Abdullah Cihan Ak, and Sanem Sariel. Fino-net: A deep multimodal sensor fusion framework for manipulation
      failure detection. In 2021 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2021.
[175] Yijun Liu, Jiequan Cui, Zhuotao Tian, Senqiao Yang, Qingdong He, Xiaoling Wang, and Jingyong Su. Typicalness-aware learning for failure
      detection. In Proc. NeurIPS, 2024.
[176] Paul F Jaeger, Carsten T Lüth, Lukas Klein, and Till J Bungert. A call to reflect on evaluation practices for failure detection in image classification.
      In Proc. ICLR, 2023.
[177] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Proc. NeurIPS, volume 33, pages 6840–6851, 2020.
[178] Carl Rasmussen. The infinite gaussian mixture model. Proc. NeurIPS, 12, 1999.
[179] Simon Duane, Anthony D Kennedy, Brian J Pendleton, and Duncan Roweth. Hybrid monte carlo. Physics letters B, 195(2):216–222, 1987.
[180] Radford M Neal. Probabilistic inference using markov chain monte carlo methods. 1993.
[181] Kanti V Mardia and Peter E Jupp. Directional statistics. John Wiley & Sons, 2009.
[182] Md Yousuf Harun, Jhair Gallardo, and Christopher Kanan. Controlling neural collapse enhances out-of-distribution detection and transfer learning.
      In Proc. ICML, 2025.
[183] Maximilian Mueller and Matthias Hein. Mahalanobis++: Improving ood detection via feature normalization. In Proc. ICML, 2025.
[184] Jiayu Zhang, Xinyi Wang, Zhibo Jin, Zhiyu Zhu, Jiahao Huang, Xinyi Zhang, and Huaming Chen. Splitting & integrating: Out-of-distribution
      detection via adversarial gradient attribution. 2025.
[185] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y Hou, and Max Tegmark. Kan: Kolmogorov-
      arnold networks. In Proc. ICLR, 2025.
[186] Dmitry Krotov and John J Hopfield. Dense associative memory for pattern recognition. In Proc. CVPR, volume 29, 2016.
[187] Yongcan Yu, Lijun Sheng, Ran He, and Jian Liang. Stamp: Outlier-aware test-time adaptation with stable memory replay. In Proc. ECCV, 2024.
[188] Yongcan Yu, Lijun Sheng, Ran He, and Jian Liang. Benchmarking test-time adaptation against distribution shifts in image classification. arXiv
      preprint arXiv:2307.03133, 2023.
[189] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under
      distribution shifts. In Proc. ICML.
[190] Yongcan Yu, Yanbo Wang, Ran He, and Jian Liang. Test-time immunization: A universal defense framework against jailbreaks for (multimodal)
      large language models. arXiv preprint arXiv:2505.22271, 2025.
[191] Jian Liang, Ran He, and Tieniu Tan. A comprehensive survey on test-time adaptation under distribution shifts. IJCV.
[192] Zhen Fang, Yixuan Li, Jie Lu, Jiahua Dong, Bo Han, and Feng Liu. Is out-of-distribution detection learnable? In Proc. NeurIPS, 2022.
[193] Xiaojin Jerry Zhu. Semi-supervised learning literature survey. 2005.
[194] Eric Arazo, Diego Ortego, Paul Albert, Noel E O’Connor, and Kevin McGuinness. Pseudo-labeling and confirmation bias in deep semi-supervised
      learning. In Proc. IJCNN, 2020.
[195] Shiyu Liang, Yixuan Li, and R. Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. In Proc. ICLR, 2018.
                                                                                                                             Manuscript submitted to ACM

## Page 32

32                                                                                                                                            Lu et al.


[196] Xinheng Wu, Jie Lu, Zhen Fang, and Guangquan Zhang. Meta ood learning for continuously adaptive ood detection. In Proc. ICCV, 2023.
[197] Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Open-set recognition: A good closed-set classifier is all you need? In Proc. ICLR, 2022.
[198] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias
      Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In Proc. ICLR, 2021.
[199] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding.
      In Proc. NAACL, 2019.
[200] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack
      Clark, et al. Learning transferable visual models from natural language supervision. In Proc. ICML, pages 8748–8763, 2021.
[201] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu
      Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In Proc. NeurIPS, volume 35,
      pages 36479–36494, 2022.
[202] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual
      and vision-language representation learning with noisy text supervision. In Proc. ICML, pages 4904–4916, 2021.
[203] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry,
      Amanda Askell, et al. Language models are few-shot learners. In Proc. NeurIPS, volume 33, pages 1877–1901, 2020.
[204] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco:
      Common objects in context. In Proc. ECCV, pages 740–755, 2014.
[205] Anonymous. Negative label guided OOD detection with pretrained vision-language models. In Proc. ICLR, 2024.
[206] Chentao Cao, Zhun Zhong, Zhanke Zhou, Tongliang Liu, Yang Liu, Kun Zhang, and Bo Han. Noisy test-time adaptation in vision-language models.
      In Proc. ICLR, 2025.
[207] Choubo Ding and Guansong Pang. Zero-shot out-of-distribution detection with outlier label exposure. In IJCNN, 2024.
[208] Momin Abbas, Muneeza Azmat, Raya Horesh, and Mikhail Yurochkin. Out-of-distribution detection using synthetic data generation. arXiv preprint
      arXiv:2502.03323, 2025.
[209] Stanislav Fort, Jie Ren, and Balaji Lakshminarayanan. Exploring the limits of out-of-distribution detection. In Proc. NeurIPS, 2021.
[210] Jian Liang, Lijun Sheng, Zhengbo Wang, Ran He, and Tieniu Tan. Realistic unsupervised clip fine-tuning with universal entropy optimization. In
      Proc. ICML, 2024.
[211] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer
      Vision, 130(9):2337–2348, 2022.
[212] Yifei Ming and Yixuan Li. How does fine-tuning impact out-of-distribution detection for vision-language models? International Journal of Computer
      Vision, 132(2):596–609, 2024.
[213] Jeonghyeon Kim, Jihyo Kim, and Sangheum Hwang. Comparison of out-of-distribution detection performance of clip-based fine-tuning methods.
      In Proc. ICEIC, pages 1–4, 2024.
[214] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proc. CVPR, pages
      16816–16825, 2022.
[215] Ziqin Zhou, Yinjie Lei, Bowen Zhang, Lingqiao Liu, and Yifan Liu. Zegclip: Towards adapting clip for zero-shot semantic segmentation. In Proc.
      CVPR, pages 11175–11185, 2023.
[216] Jeonghyeon Kim and Sangheum Hwang. Enhanced ood detection through cross-modal alignment of multi-modal representations. In Proc. CVPR,
      2025.
[217] Xinyi Chen, Yaohui Li, and Haoxing Chen. Dual-adapter: Training-free dual adaptation for few-shot out-of-distribution detection. arXiv preprint
      arXiv:2405.16146, 2024.
[218] Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. IEEE transactions on
      pattern analysis and machine intelligence, 44(9):5149–5169, 2021.
[219] Zhenjiang Mao, Dong-You Jhong, Ao Wang, and Ivan Ruchkin. Language-enhanced latent representations for out-of-distribution detection in
      autonomous driving. In Proc. ICRA, 2024.
[220] Qiaozhi Tan, Long Bai, Guankun Wang, Mobarakol Islam, and Hongliang Ren. Endoood: Uncertainty-aware out-of-distribution detection in capsule
      endoscopy diagnosis. In Proc. ISBI, 2024.
[221] Udit Arora, William Huang, and He He. Types of out-of-distribution texts and how to detect them. In Proc. EMNLP, 2021.
[222] Li-Ming Zhan, Haowen Liang, Lu Fan, Xiao-Ming Wu, and Albert YS Lam. A closer look at few-shot out-of-distribution intent detection. In Proc.
      COLING, 2022.
[223] Zaharah Bukhsh and Aaqib Saeed. On out-of-distribution detection for audio with deep nearest neighbors. In ICASSP 2023-2023 IEEE International
      Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023.
[224] Haoyue Bai, Gregory Canal, Xuefeng Du, Jeongyeol Kwon, Robert D Nowak, and Yixuan Li. Feed two birds with one scone: Exploiting wild data
      for both out-of-distribution generalization and detection. In Proc. ICML, 2023.
[225] What is shadow mode tesla autonomy. URL https://www.youtube.com/watch?v=SAceTxSelTI.
[226] System lets robots identify an object’s properties through handling. URL https://news.mit.edu/2025/system-lets-robots-identify-objects-properties-
      through-handling-0508.
Manuscript submitted to ACM

## Page 33

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                                              33


[227] Zizhao Li, Xueyang Kang, Joseph West, and Kourosh Khoshelham. Out-of-distribution detection in 3d applications: A review. Available at SSRN
      5256108.
[228] Qutub Syed, Michael Paulitsch, Korbinian Hagn, Neslihan Kose Cihangir, Kay-Ulrich Scholl, Fabian Oboril, Gereon Hinz, and Alois Knoll. Situation
      monitor: Diversity-driven zero-shot out-of-distribution detection using budding ensemble architecture for object detection. arXiv preprint
      arXiv:2406.03188, 2024.
[229] Junkun Chen, Jilin Mei, Liang Chen, Fangzhou Zhao, Yan Xing, and Yu Hu. Proto-ood: Enhancing ood object detection with prototype feature
      similarity. arXiv preprint arXiv:2409.05466, 2024.
[230] Bin Zhang, Xiaoyang Qu, Guokuan Li, Jiguang Wan, and Jianzong Wang. Vista: Visual-contextual and text-augmented zero-shot object-level ood
      detection. arXiv preprint arXiv:2503.22291, 2025.
[231] Georges Le Bellier and Nicolas Audebert. Detecting out-of-distribution earth observation images with diffusion models. arXiv preprint
      arXiv:2404.12667, 2024.
[232] Shawn Li, Peilin Cai, Yuxiao Zhou, Zhiyu Ni, Renjie Liang, You Qin, Yi Nian, Zhengzhong Tu, Xiyang Hu, and Yue Zhao. Secure on-device video
      ood detection without backpropagation. arXiv preprint arXiv:2503.06166, 2025.
[233] How amazon fraud detector works. URL https://docs.aws.amazon.com/frauddetector/latest/ug/how-frauddetector-works.html.
[234] URL https://www.mayoclinic.org/diseases-conditions/cataracts/diagnosis-treatment/drc-20353795.
[235] Sandesh Pokhrel, Sanjay Bhandari, Eduard Vazquez, Tryphon Lambrou, Prashnna Gyawali, and Binod Bhattarai. Tta-ood: Test-time augmentation
      for improving out-of-distribution detection in gastrointestinal vision. arXiv preprint arXiv:2407.14024, 2024.
[236] Pia H Smedsrud, Vajira Thambawita, Steven A Hicks, Henrik Gjestang, Oda Olsen Nedrejord, Espen Næss, Hanna Borgli, Debesh Jha, Tor Jan Derek
      Berstad, Sigrun L Eskeland, et al. Kvasir-capsule, a video capsule endoscopy dataset. Scientific Data, 8(1):142, 2021.
[237] Suayb S Arslan, James Peng, and Turguy Goker. Talics3: Tape library cloud storage system simulator. Simulation Modelling Practice and Theory,
      134:102947, 2024.
[238] Sabri Mustafa Kahya, Boran Hamdi Sivrikaya, Muhammet Sami Yavuz, and Eckehard Steinbach. Food: Facial authentication and out-of-distribution
      detection with short-range fmcw radar. arXiv preprint arXiv:2406.04546, 2024.
[239] Yuhang Zhang, Yue Yao, Xuannan Liu, Lixiong Qin, Wenjing Wang, and Weihong Deng. Open-set facial expression recognition. arXiv preprint
      arXiv:2401.12507, 2024.
[240] Minho Sim, Young-Jun Lee, Dongkun Lee, Jongwhoa Lee, and Ho-Jin Choi. A simple debiasing framework for out-of-distribution detection in
      human action recognition. Proc. ECAI, 2023.
[241] Devraj Mandal, Sanath Narayan, Sai Kumar Dwivedi, Vikram Gupta, Shuaib Ahmed, Fahad Shahbaz KhCVPRan, and Ling Shao. Out-of-distribution
      detection for generalized zero-shot action recognition. In Proc. CVPR, pages 9985–9993, 2019.
[242] Jing Xu, Anqi Zhu, Jingyu Lin, Qiuhong Ke, and Cunjian Chen. Skeleton-ood: An end-to-end skeleton-based model for robust out-of-distribution
      human action detection. arXiv preprint arXiv:2405.20633, 2024.
[243] Mark Mazumder, Sharad Chitlangia, Colby Banbury, Yiping Kang, Juan Manuel Ciro, Keith Achorn, Daniel Galvez, Mark Sabini, Peter Mattson,
      David Kanter, et al. Multilingual spoken words corpus. In Proc. NeurIPS, 2021.
[244] Lingyi Zhou, Stefan T Radev, William H Oliver, Aura Obreja, Zehao Jin, and Tobias Buck. Evaluating sparse galaxy simulations via out-of-distribution
      detection and amortized bayesian model comparison. arXiv preprint arXiv:2410.10606, 2024.
[245] Suman Das, Michael Yuhas, Rachel Koh, and Arvind Easwaran. Interpretable latent space for meteorological out-of-distribution detection via weak
      supervision. ACM Transactions on Cyber-Physical Systems, 2024.
[246] Gleb Bazhenov, Sergei Ivanov, Maxim Panov, Alexey Zaytsev, and Evgeny Burnaev. Towards ood detection in graph classification from uncertainty
      estimation perspective. arXiv preprint arXiv:2206.10691, 2022.
[247] Qitian Wu, Yiting Chen, Chenxiao Yang, and Junchi Yan. Energy-based out-of-distribution detection for graph neural networks. In Proc. ICLR, 2023.
[248] Yiming Wang, Pei Zhang, Baosong Yang, Derek F Wong, Zhuosheng Zhang, and Rui Wang. Embedding trajectory for out-of-distribution detection
      in mathematical reasoning. arXiv preprint arXiv:2405.14039, 2024.
[249] Erik Bernardo Terres-Escudero, Javier Del Ser, Aitor Martinez-Seras, and Pablo Garcia Bringas. Forward-forward learning achieves highly selective
      latent representations for out-of-distribution detection in fully spiking neural networks. Available at SSRN 5152073.
[250] Garrick Orchard, Ajinkya Jayawant, Gregory K Cohen, and Nitish Thakor. Converting static image datasets to spiking neuromorphic datasets
      using saccades. Frontiers in neuroscience, 9:437, 2015.
[251] Krzysztof Lis, Krishna Nakka, Pascal Fua, and Mathieu Salzmann. Detecting the unexpected via image resynthesis. In Proc. ICCV, pages 2152–2161,
      2019.
[252] Li-Ming Zhan, Yuxiao Dong, and Jie Tang. Vi-ood: A unified representation learning framework for textual out-of-distribution detection. arXiv
      preprint arXiv:2404.06217, 2024.
[253] Pei Wang, Keqing He, Yejie Wang, Xiaoshuai Song, Yutao Mou, Jingang Wang, Yunsen Xian, Xunliang Cai, and Weiran Xu. Beyond the known:
      Investigating llms performance on out-of-domain intent detection. arXiv preprint arXiv:2402.17256, 2024.
[254] Andi Zhang, Tim Z Xiao, Weiyang Liu, Robert Bamler, and Damon Wischik. Your finetuned large language model is already a powerful
      out-of-distribution detector. arXiv preprint arXiv:2404.08679, 2024.
[255] Christos Constantinou, Panagiotis Papadopoulos, Alexandros Grapsas, and Christos Makropoulos. Out-of-distribution detection with attention
      head masking for multimodal document classification. arXiv preprint arXiv:2408.11237, 2024.
                                                                                                                          Manuscript submitted to ACM

## Page 34

34                                                                                                                                               Lu et al.


[256] Xuyang Chen, Keyu Yan, and Lin Zhao. Taming ood actions for offline reinforcement learning: An advantage-based approach. arXiv preprint
      arXiv:2505.05126, 2025.
[257] Andreas Sedlmeier, Thomas Gabor, Thomy Phan, Lenz Belzner, and Claudia Linnhoff-Popien. Uncertainty-based out-of-distribution detection in
      deep reinforcement learning. arXiv preprint arXiv:1901.02219, 2019.
[258] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint
      arXiv:1708.07747, 2017.
[259] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
[260] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proc. CVPR, 2009.
[261] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist
      species classification and detection dataset. In Proc. CVPR, pages 8769–8778, 2018.
[262] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo.
      In Proc. CVPR, pages 3485–3492, 2010.
[263] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, , and A. Vedaldi. Describing textures in the wild. In Proc. CVPR, 2014.
[264] Julian Bitterwolf, Maximilian Mueller, and Matthias Hein. In or out? fixing imagenet out-of-distribution detection evaluation. In Proc. ICML, 2023.
[265] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt
      Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. CVPR, 2016.
[266] Alberto Bacchin, Davide Allegro, Stefano Ghidoni, and Emanuele Menegatti. Sood-imagenet: a large-scale dataset for semantic out-of-distribution
      image classification and semantic segmentation. arXiv preprint arXiv:2409.01109, 2024.
[267] M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes challenge: A retrospective.
      International Journal of Computer Vision, 111(1):98–136, January 2015.
[268] Fisher Yu, Wenqi Xian, Yingying Chen, Fangchen Liu, Mike Liao, Vashisht Madhavan, Trevor Darrell, et al. Bdd100k: A diverse driving video
      database with scalable annotation tooling. arXiv preprint arXiv:1805.04687, 2(5):6, 2018.
[269] Michael Kösel, Marcel Schreiber, Michael Ulrich, Claudius Gläser, and Klaus Dietmayer. Revisiting out-of-distribution detection in lidar-based 3d
      object detection. arXiv preprint arXiv:2404.15879, 2024.
[270] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In Proc. CoRL,
      pages 1–16, 2017.
[271] Taposh Banerjee, Rui Liu, Lili Su, et al. Building real-time awareness of out-of-distribution in trajectory prediction for autonomous vehicles. arXiv
      preprint arXiv:2409.17277, 2024.
[272] Yijun Liu, Jinzheng Yu, Yang Xu, Zhongyang Li, and Qingfu Zhu. A survey on transformer context extension: Approaches and evaluation. arXiv
      preprint arXiv:2503.13299, 2025.
[273] Stefan Larson, Anish Mahendran, Joseph J Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K Kummerfeld, Kevin Leach, Michael A
      Laurenzano, Lingjia Tang, et al. An evaluation dataset for intent classification and out-of-scope prediction. In Proc. EMNLP-IJCNLP, 2019.
[274] Pengfei Liu, Wei Yuan, Jian Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Benchmarking natural language understanding services for
      building conversational agents. arXiv preprint arXiv:1911.03590, 2020.
[275] Rishabh Misra. News category dataset. arXiv preprint arXiv:2209.11429, 2022.
[276] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for
      semantic compositionality over a sentiment treebank. In Proc. EMNLP, pages 1631–1642, 2013.
[277] Kenneth D Lang. Newsweeder: Learning to filter netnews. pages 331–339, 1995.
[278] Subhro Roy, Tim Vieira, and Dan Roth. Solving general arithmetic word problems. pages 1746–1751, 2016.
[279] Ritesh Aggarwal, Himanshu Gothwal, Piyush Parashar, and Santanu Chaudhury. Tobacco-3482: A large-scale dataset for handwritten tobacco
      product packaging image classification. 1:1083–1088, 2017.
[280] Ian Lane, Tatsuya Kawahara, Tomoko Matsui, and Satoshi Nakamura. Out-of-domain utterance detection using classification confidences of
      multiple topics. IEEE Transactions on Audio, Speech, and Language Processing, 15(1):150–161, 2006.
[281] Seonghan Ryu, Seokhwan Kim, Junhwi Choi, Hwanjo Yu, and Gary Geunbae Lee. Neural sentence embedding using only in-domain sentences for
      out-of-domain sentence detection in dialog systems. Pattern Recognition Letters, 88:26–32, 2017.
[282] Renmingyue Du, Jixun Yao, Qiuqiang Kong, and Yin Cao. Towards out-of-distribution detection in vocoder recognition via latent feature
      reconstruction. arXiv preprint arXiv:2406.02233, 2024.
[283] Tiancheng Huang, Donglin Wang, and Yuan Fang. End-to-end open-set semi-supervised node classification with out-of-distribution detection.
      2022.
[284] He Junwei, Qianqian Xu, Yangbangyan Jiang, Zitai Wang, Yuchen Sun, and Qingming Huang. Hgoe: Hybrid external and internal graph outlier
      exposure for graph out-of-distribution detection. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 1544–1553, 2024.
[285] Xu Shen, Yili Wang, Kaixiong Zhou, Shirui Pan, and Xin Wang. Optimizing ood detection in molecular graphs: A novel approach with diffusion
      models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 2640–2650, 2024.
[286] Danny Wang, Ruihong Qiu, Guangdong Bai, and Zi Huang. Gold: Graph out-of-distribution detection via implicit adversarial latent generation. In
      Proc. ICLR, 2025.


Manuscript submitted to ACM

## Page 35

Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances                                                                           35


[287] Christopher Morris, Nils M Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. Tudataset: A collection of benchmark
      datasets for learning with graphs. In Proc. ICML Workshops, 2020.
[288] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark:
      Datasets for machine learning on graphs. In Proc. NeurIPS, 2020.
[289] Stefanos Leonardos, Daiki Matsunaga, Jongmin Lee, Jaeseok Yoon, Pieter Abeel, and Kee-Eung Kim. Addressing out-of-distribution joint actions in
      offline multi-agent rl via alternating stationary distribution correction estimation. In Proc. NeurIPS, 2023.
[290] Aaqib Parvez Mohammed and Matias Valdenegro-Toro. Benchmark for out-of-distribution detection in deep reinforcement learning. In Proc.
      NeurIPS, 2021.
[291] Onat Gungor, Amanda Sofie Rios, Nilesh Ahuja, and Tajana Rosing. Ts-ood: Evaluating time-series out-of-distribution detection and prospective
      directions for progress. arXiv preprint arXiv:2502.15901, 2025.
[292] Jens Henriksson, Stig Ursing, Murat Erdogan, Fredrik Warg, Anders Thorsén, Johan Jaxing, Ola Örsmark, and Mathias Örtenberg Toftås. Out-
      of-distribution detection as support for autonomous driving safety lifecycle. In International Working Conference on Requirements Engineering:
      Foundation for Software Quality, pages 233–242. Springer, 2023.
[293] Vivian Lin, Ramneet Kaur, Yahan Yang, Souradeep Dutta, Yiannis Kantaros, Anirban Roy, Susmit Jha, Oleg Sokolsky, and Insup Lee. Safety
      monitoring for learning-enabled cyber-physical systems in out-of-distribution scenarios. In Proceedings of the ACM/IEEE 16th International
      Conference on Cyber-Physical Systems (with CPS-IoT Week 2025), pages 1–11, 2025.
[294] Shubhranshu Shekhar, Neil Shah, and Leman Akoglu. Fairod: Fairness-aware outlier detection. In Proceedings of the 2021 AAAI/ACM Conference on
      AI, Ethics, and Society, pages 210–220, 2021.
[295] Nathan Inkawhich, Jingyang Zhang, Eric K Davis, Ryan Luley, and Yiran Chen. Improving out-of-distribution detection by learning from the
      deployment environment. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15:2070–2086, 2022.
[296] Yuehan Qin, Yichi Zhang, Yi Nian, Xueying Ding, and Yue Zhao. Metaood: Automatic selection of ood detection models. In Proc. ICLR, 2025.
[297] Jingyao Geng, Yuan Zhang, Jiaqi Huang, Feng Xue, Falong Tan, Chuanlong Xie, and Shumei Zhang. Dsde: Using proportion estimation to improve
      model selection for out-of-distribution detection. arXiv preprint arXiv:2411.01487, 2024.
[298] Edward T Reehorst and Philip Schniter. Score combining for contrastive ood detection. arXiv preprint arXiv:2501.12204, 2025.
[299] Behrooz Montazeran and Ullrich Köthe. Ood detection with immature models. arXiv preprint arXiv:2502.00820, 2025.
[300] Sebastian Schmidt, Leonard Schenk, Leo Schwinn, and Stephan Günnemann. A unified approach towards active learning and out-of-distribution
      detection. arXiv preprint arXiv:2405.11337, 2024.
[301] Lars Doorenbos, Raphael Sznitman, and Pablo Márquez-Neila. Continual unsupervised out-of-distribution detection. arXiv preprint arXiv:2406.02327,
      2024.
[302] Wenjun Miao, Guansong Pang, Trong-Tung Nguyen, Ruohang Fang, Jin Zheng, and Xiao Bai. Opencil: Benchmarking out-of-distribution detection
      in class-incremental learning. arXiv preprint arXiv:2407.06045, 2024.
[303] Yuhang Liu, Wenjie Zhao, and Yunhui Guo. H2st: Hierarchical two-sample tests for continual out-of-distribution detection. In Proc. CVPR, 2025.
[304] Kuangpu Guo, Yuhe Ding, Jian Liang, Zilei Wang, Ran He, and Tieniu Tan. Exploring vacant classes in label-skewed federated learning. In Proc.
      AAAI.
[305] Liang Kuang, Kuangpu Guo, Jian Liang, and Jianguo Zhang. An enhanced federated prototype learning method under domain shift. arXiv preprint
      arXiv:2409.18578, 2024.
[306] Xinting Liao, Weiming Liu, Pengyang Zhou, Fengyuan Yu, Jiahe Xu, Jun Wang, Wenjie Wang, Chaochao Chen, and Xiaolin Zheng. Foogd: Federated
      collaboration for both out-of-distribution generalization and detection. In Proc. NeurIPS, 2024.
[307] Hao Dong, Yue Zhao, Eleni Chatzi, and Olga Fink. Multiood: Scaling out-of-distribution detection for multiple modalities. In Proc. NeurIPS, 2024.
[308] Rena Gao, Xuetong Wu, Siwen Luo, Caren Han, and Feng Liu. ’no’matters: Out-of-distribution detection in multimodality long dialogue. arXiv
      preprint arXiv:2410.23883, 2024.
[309] Yihan Mei, Xinyu Wang, Dell Zhang, and Xiaoling Wang. Multi-label out-of-distribution detection with spectral normalized joint energy. pages
      31–45. Springer, 2024.
[310] Galadrielle Humblot-Renaux, Sergio Escalera, and Thomas B Moeslund. A noisy elephant in the room: Is your out-of-distribution detector robust
      to label noise? arXiv preprint arXiv:2404.01775, 2024.
[311] Harit Vishwakarma, Heguang Lin, and Ramya Korlakai Vinayak. Taming false positives in out-of-distribution detection with human feedback. In
      Proc. AISTATS, 2024.
[312] Haoyue Bai, Xuefeng Du, Katie Rainey, Shibin Parameswaran, and Yixuan Li. Out-of-distribution learning with human feedback. arXiv preprint
      arXiv:2408.07772, 2024.
[313] Haoyue Bai, Jifan Zhang, and Robert Nowak. Aha: Human-assisted out-of-distribution generalization and detection. arXiv preprint arXiv:2410.08000,
      2024.
[314] Atsuyuki Miyai, Qing Yu, Go Irie, and Kiyoharu Aizawa. Zero-shot in-distribution detection in multi-object settings using vision-language
      foundation models. arXiv preprint arXiv:2304.04521, 2023.


Received 22 May 2024; revised 19 June 2025; revised 23 July 2025; accepted 04 August 2025

                                                                                                                        Manuscript submitted to ACM
