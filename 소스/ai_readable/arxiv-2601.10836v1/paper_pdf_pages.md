# One Model, Many Behaviors: Training-Induced Effects on Out-of-Distribution Detection - page-anchored PDF text

- Source ID: `arxiv-2601.10836v1`
- arXiv ID: `2601.10836v1`
- Original PDF: `소스/One Model, Many Behaviors_ Training-Induced Effects on Out-of-Distribution Detection.pdf`
- PDF pages: 21
- Extracted with: WSL poppler `pdftotext -f N -l N -layout` on 2026-05-13T17:01:27+09:00
- Evidence note: use this file for page-anchored claims; verify equations/tables against the original PDF when exact layout matters.

## Page 1

One Model, Many Behaviors: Training-Induced Effects on Out-of-Distribution
                                                                         Detection

                                                                Gerhard Krumpl* 1,2          Henning Avenhaus2                           Horst Possegger1
                                                            1
                                                                Institute of Visual Computing, Graz University of Technology, Austria
                                                                                    2
                                                                                      KESTRELEYE GmbH, Austria
arXiv:2601.10836v1 [cs.CV] 15 Jan 2026




                                                                   Abstract                                                  90.0

                                                                                                                             87.5
                                            Out-of-distribution (OOD) detection is crucial for de-
                                         ploying robust and reliable machine-learning systems in                             85.0




                                                                                                            Mean AUROC (%)
                                         open-world settings. Despite steady advances in OOD
                                         detectors, their interplay with modern training pipelines                           82.5
                                         that maximize in-distribution (ID) accuracy and general-
                                                                                                                             80.0
                                         ization remains under-explored. We investigate this link                                             Training category
                                         through a comprehensive empirical study. Fixing the ar-                             77.5                 Baseline
                                         chitecture to the widely adopted ResNet-50, we benchmark                                                 Adversarial Training
                                         21 post-hoc, state-of-the-art OOD detection methods across                          75.0                 Augmentations
                                         56 ImageNet-trained models obtained via diverse training                                                 SSL
                                                                                                                             72.5                 Improved Training
                                         strategies and evaluate them on eight OOD test sets. Con-
                                                                                                                                                  Freezing
                                         trary to the common assumption that higher ID accuracy                              70.0
                                                                                                                                    55   60       65         70          75   80
                                         implies better OOD detection performance, we uncover a
                                                                                                                                          ID classification accuracy (%)
                                         non-monotonic relationship: OOD performance initially
                                         improves with accuracy but declines once advanced train-           Figure 1. Top-1 classification accuracy is an unreliable indica-
                                         ing recipes push accuracy beyond the baseline. More-               tor for OOD detection performance. This figure shows the re-
                                         over, we observe a strong interdependence between training         lationship between in-distribution (ID) classification accuracy and
                                         strategy, detector choice, and resulting OOD performance,          out-of-distribution (OOD) detection performance for 56 ResNet-
                                         indicating that no single method is universally optimal.           50 models, all sharing the same architecture but trained with dif-
                                                                                                            ferent strategies. Each point represents one model and reports the
                                                                                                            mean AUROC (Area Under the Receiver Operating Characteristic
                                         1. Introduction                                                    Curve) over 21 OOD detection methods and eight OOD datasets.
                                                                                                            Color indicates the model’s training category, while the marker
                                         The robust deployment of machine learning (ML) sys-                shape uniquely identifies each model within that category.
                                         tems in real-world environments depends on models that
                                         not only perform well on in-distribution (ID) data but can
                                         also reliably detect inputs that differ from their training            Post-hoc OOD detection methods offer a promising so-
                                         distribution—such as novel object categories, unexpected           lution. These methods are appealing in practice: i) they
                                         environmental conditions, or sensor corruptions. When ex-          are simple to deploy on pretrained models, avoiding costly
                                         posed to such out-of-distribution (OOD) inputs, modern             retraining and preserving ID accuracy, and ii) they require
                                         ML models often produce high-confidence yet incorrect              no prior exposure to OOD data, which, by definition, are
                                         predictions [27, 50, 52]. This behavior poses significant          unknown before deployment. A wide variety of methods
                                         risks in safety-critical settings, such as autonomous driv-        have emerged, ranging from simple confidence scores [28]
                                         ing, healthcare, or industrial quality assurance. For exam-        to more sophisticated approaches based on feature-space
                                         ple, an autonomous car might find itself in an unexpected          distances [40, 56, 62] and model enhancements [15, 61].
                                         emergency scene, or a food-sorting stream could encounter              Concurrently, modern classification training pipelines
                                         unexpected (potentially allergy-inducing) foreign materials.       have become increasingly diverse and powerful. Augmen-
                                           * Correspondence: gerhard.krumpl@icg.tugraz.at                   tations such as MixUp [78] or CutMix [77], regularization


                                                                                                        1

## Page 2

techniques like label smoothing [63] or Exponential Mov-              improve OOD robustness [17, 33, 44]. Recent training-
ing Averages (EMA), contrastive/self-supervised pretrain-             based works also focus on improving OOD generalization
ing, and extended training schedules have all been shown to           and detection [2, 83]. In contrast, post-hoc methods
improve ID accuracy and generalization [26, 67, 71]. Yet,             operate on pretrained models without requiring retraining
surprisingly, the impact of such training strategies on post-         or access to OOD samples. We focus on post-hoc, sample-
hoc OOD detection has received little attention. Most prior           free methods due to their practical advantages: they are
evaluations continue to benchmark OOD detection methods               model-agnostic, preserve the model’s in-distribution (ID)
on models trained with vanilla strategies—failing to reflect          performance, and scale efficiently to large deployments,
the diversity of models used in practice.                             where retraining is costly or infeasible [81].
    A widely held assumption is that higher ID accuracy nat-             Post-hoc methods can be further grouped into score-
urally translates into stronger OOD detection [19, 35, 66].           based and model enhancement methods. Score-based
This belief has shaped experimental design and practical              methods assign an OOD score to each input based on
implementation alike: train a more accurate classifier and            model outputs or internal activations. Within this category,
get stronger OOD detection performance for free. However,             classification-based methods were pioneered by Hendrycks
this assumption is rarely questioned—and its validity under           et al. [28], who introduced Maximum Softmax Probabil-
diverse training strategies remains unclear.                          ity (MSP), which uses the highest softmax output as a
    In this work, we revisit this and other assump-                   baseline OOD score. Subsequent variants, such as Max-
tions through the first large-scale, architecture- and data-          imum Logit Score (MLS) [31] and Energy-Based OOD
controlled study on the effects of training strategies on post-       (EBO) [46], refine this approach by operating directly on
hoc OOD detection. By fixing the architecture (ResNet-                the model’s logits or their transformations. ODIN [42] fur-
50 [24]) and dataset (ImageNet [14]), we isolate the im-              ther improves separation by combining temperature scaling
pact of training strategy alone. We evaluate 21 post-                 with input perturbations. Feature-based methods instead
hoc OOD detection methods across 56 models spanning                   operate on internal representations: Mahalanobis distance
four training families: data augmentation, contrastive/self-          score (MDS) [40], Simplified Hopfield Energy (SHE) [79],
supervised learning, adversarial training, and improved               and KNN-based methods [62] compute distances in fea-
training schedules—as summarized in Fig. 1. Performance               ture space, while GRAM [58] and GradNorm [34] lever-
is assessed using eight benchmark OOD datasets that cover             age intermediate statistics or input gradients. Model en-
near-, far-, extreme-, and synthetic-OOD categories.                  hancement methods modify inference-time activations to
    This study provides new insights into how training influ-         improve OOD detection robustness without changing model
ences OOD detection and offers guidance for developing                weights. ReAct [61] improves robustness by clipping ab-
more robust, generalizable OOD detection methods. We                  normal activations, ASH [15] prunes and rescales fea-
summarize our contribution as follows:                                ture activations to down-weight irrelevant neurons, and
• We conduct a large-scale, architecture- and data-                   SCALE [73] shapes internal representations. These are typ-
   controlled study on the influence of training strategies on        ically paired with energy-based scoring and have demon-
   post-hoc OOD detection, covering 56 models, 21 OOD                 strated strong performance in large-scale benchmarks [81].
   detectors, and eight OOD test sets. We further analyze
   the correlation between ID accuracy and OOD detection
   performance across methods and training strategies.                ID accuracy vs. OOD detection performance. Prior
• We show that the commonly assumed correlation between               works [19, 35, 66] have reported a positive correlation be-
   ID accuracy and OOD robustness is not generally valid.             tween ID accuracy and OOD detection performance. Vaze
   Prior studies often focus on low-to-moderate accuracy              et al. [66] found a significant correlation between closed-
   regimes or vary model architectures, where capacity ef-            set classification accuracy and open-set recognition perfor-
   fects confound the relationship.                                   mance. Similarly, Galil et al. [19] noted this correlation
• We identify substantial variance in method robustness:              across various pre-trained deep neural networks when using
   many post-hoc detectors overfit to vanilla training recipes        Maximum Softmax Probability (MSP) as the OOD score.
   and degrade under diverse strategies, while methods that           Humbold-Renaux et al. [35] also identified a connection be-
   utilize richer internal representations generalize better.         tween ID and OOD performance, specifically examining the
                                                                      impact of label noise on OOD detection within a small-scale
2. Related Work                                                       benchmark. More recently, Wang et al. [69] extended this
                                                                      line of inquiry by analyzing the distinction between closed-
Out-of-distribution (OOD) detection methods are                       and open-set recognition, highlighting how auxiliary OOD
broadly categorized into training-based and post-hoc                  data influence performance. Notably, they reported a neg-
approaches [74]. Training-based methods modify model                  ative correlation between closed- and open-set recognition
architectures, loss functions, or regularization strategies to        in large-scale settings. Complementary to these findings, in

                                                                  2

## Page 3

the field of OOD generalization, OoD-Bench [76] studies               semantic OOD regime, where the ID and OOD label spaces
diversity and correlation shifts under a fixed label space (no        are disjoint: YID = Y and YOOD ∩YID = ∅. At deployment,
label shift), showing how models that learn spurious cues             the model encounters a test-time data stream drawn from the
(e.g., in Colored MNIST [1]) achieve high accuracy with-              unknown mixture Ptest (x) = (1 − π)PID (x) + πPOOD (x)
out robustness to distribution shifts.                                with an unknown OOD prior π ∈ [0, 1]. A post-hoc OOD
    The works in [19, 66] evaluate a wide variety of pre-             detector therefore provides a scoring function S : X →− R,
trained deep neural networks, spanning various architec-              that, given a threshold λ, implements the decision rule
tures and focusing primarily on MSP or a limited subset                                        (
of baseline OOD detection methods. Similarly, [35] in-                                           ID     if S(x) ≥ λ
                                                                                    G(x, λ) =                        .        (1)
vestigates the impact of label noise while varying both ID                                       OOD otherwise
datasets and model architectures. Varying multiple fac-
                                                                      This decision function must i) accept and maintain high
tors simultaneously, such as architecture, model capacity,
                                                                      classification accuracy on ID samples from PID and ii) re-
and ID dataset, introduces potential confounding effects,
                                                                      ject OOD samples from POOD —all post-hoc, i.e., without
making it difficult to disentangle the individual impact of
                                                                      retraining fθ or relying on OOD samples during calibration.
training-related factors.
    To address this, we take a more controlled experimental           3.2. Experimental Setup
design: we fix the model architecture and the ID dataset.
                                                                      Datasets. Evaluating OOD detection performance typi-
This allows us to isolate the impact of different training
                                                                      cally involves selecting a single ID dataset and multiple
strategies and to reevaluate the relationship between ID ac-
                                                                      disjoint OOD datasets that do not share semantic classes
curacy and OOD detection performance across a wide range
                                                                      with the ID data. Following established large-scale bench-
of post-hoc methods and diverse OOD test distributions.
                                                                      mark protocols [5, 73, 75, 81], we use ImageNet [14] as
                                                                      our ID dataset and adopt a diverse set of standard OOD
3. Investigating Training-induced Effects                             datasets for evaluation. To systematically capture different
This section addresses our central question: How do dif-              degrees of complexity, the OOD datasets are grouped into
ferent training strategies influence the effectiveness of post-       four categories: i) near (SSB-Hard [66], NINCO [5]), ii)
hoc OOD detectors? While prior studies [35, 66, 81] often             far (iNaturalist [65], Textures [11], OpenImage-O [68])—
vary the architecture or in-distribution (ID) dataset, which          two groupings commonly used in prior work to represent
confounds the effects of training, we isolate the impact of           increasing semantic dissimilarity to the ID dataset; iii) ex-
the training strategy by fixing both: a ResNet-50 architec-           treme (MNIST [39], Fashion-MNIST [72]), which repre-
ture trained on ImageNet.                                             sent visually and structurally distant domains, and iv) syn-
   Our study evaluates 21 sample-free post-hoc OOD de-                thetic, synthetic unit-test datasets introduced by NINCO [5]
tection methods on 56 ResNet-50 models, each following a              to probe specific OOD weaknesses such as sensor failures.
distinct training strategy.Evaluation covers eight OOD test           This setup ensures compatibility with established bench-
datasets grouped into four categories: near-, far-, extreme-,         marks [81] while enabling a more fine-grained analysis of
and synthetic-OOD. This design enables a systematic inves-            model behavior across a wide spectrum of data shifts.
tigation of how training choices affect OOD detection.
   The following subsections detail the problem setting,              Models. Our analysis is conducted using models based on
datasets, models, metrics, and detection methods that un-             the ResNet-50 architecture [24], all of which are trained on
derpin our study.                                                     ImageNet [14]. Fixing the architecture is a central design
                                                                      choice, as each network family exhibits unique inductive
3.1. Problem Setting                                                  biases, activation statistics, and feature distributions, all of
Ideally, a deep neural network deployed in the open world             which can impact OOD detection performance. By hold-
should know what it does not know: it must flag unfamil-              ing the architecture constant, we isolate the effect of the
iar out-of-distribution (OOD) inputs while maintaining high           training strategy alone, thereby avoiding confounding fac-
accuracy on familiar in-distribution (ID) inputs.                     tors such as architecture and model capacity, which can in-
    Formally, let X ⊂ RC×H×W denote the input space and               dependently influence both ID accuracy and OOD detec-
Y = {class1 , · · · , classK } the label set of a K-way clas-         tion performance, thereby complicating the interpretation
sification problem. The network fθ : X →     − RK is trained          of their relationship.
once—using a specific training recipe r that fixes augmen-                Restricting the study to ResNet-50 provides further ben-
tations, loss, schedule, and regularizes—on an ID dataset             efits: i) it is one of the most widely used architectures in
DID = {(xi , yi )}N   i=1 drawn from the joint distribution           both research and practice, serving as the de facto stan-
PID (x, y). Everything outside the resulting support XID ⊂            dard for evaluating post-hoc OOD detection methods—
X defines the OOD space XOOD = X \ XID . We target the                particularly in large-scale settings [31, 62, 73, 79, 81], and


                                                                  3

## Page 4

Near OOD                   Far OOD                    Extreme OOD                      Synthetic OOD
Mean AUROC (%)


                 90

                 80

                 70
                                                                             Baseline                Augmentations   Improved Training
                                                                             Adversarial Training    SSL             Freezing
                 60
                      60   70     80         60        70        80          60         70          80         60        70         80
                                                  ID classification accuracy (%)

Figure 2. Relationship between ID accuracy and OOD detection performance (AUROC) across the OOD categories (near, far, extreme,
synthetic). Each point represents one of the 56 ResNet-50 models, averaged over 21 OOD detection methods. Color indicates the model’s
training category, while the marker shape uniquely identifies each model within that category.


ii) it remains a practical choice in many real-world appli-           rics isolate the OOD detection performance on correctly and
cations where large-scale models are infeasible due to con-           incorrectly classified ID samples, respectively. Ideally, a
straints on both training and inference resources.                    robust OOD detector should perform well on both, signal-
    All models are trained on ImageNet to avoid OOD con-              ing that its scores are not merely correlated with classifica-
taminations in our benchmarks. Importantly, none of the               tion correctness but reflect true distributional shifts. Finally,
models were optimized for OOD detection, allowing us to               we also report the AUROCcorrect vs. incorrect , which measures
investigate how the training strategy alone influences OOD            a method’s ability to flag misclassified ID samples [3, 35].
detection performance, independent of any OOD-specific
tuning. In total, we evaluate 56 ResNet-50 models trained
using a diverse set of strategies that affect both ID classi-         Test time and evaluation. At test time, the inputs are
fication accuracy and generalization characteristics. These           resized to 256 × 256 and center-cropped to 224 × 224.
strategies are grouped into the following six categories: i)          We evaluate 21 approaches using the OpenOOD bench-
a baseline model with the original training [24], ii) ad-             mark [75, 80, 81], a comprehensive and standardized open-
versarial training [48, 57] against PGD adversary, iii)               source framework for benchmarking state-of-the-art OOD
different augmentation methods [12, 13, 18, 22, 29, 32,               detection methods. To facilitate analysis, we catego-
36, 41, 43, 49, 51, 55, 77, 78, 82], iv) contrastive learn-           rize the OOD detection methods into five groups based
ing [6–9, 37] with supervised finetuning of the classifica-           on the type of information they leverage to compute the
tion head, v) improved training recipes [54, 67, 70, 71],             OOD score: i)         classification-based methods primar-
and vi) model with randomly weighted spatial convolu-                 ily use model outputs such as softmax probabilities or
tional filters [21]. To distinguish individual models within          logits (MSP [28], MLS [31], EBO [46], ODIN [42],
each group, we use distinct marker shapes. The combina-               TempScale [23], KLM [31], GEN [47]); ii)             feature-
tion of marker and color yields a unique identifier for each          based methods derive the score from the penultimate layer
model configuration. This visual encoding is consistently             (MDS [40], RMDS [56], KNN [62], SHE [79]) iii) hy-
used throughout our study, and a complete legend is avail-            brid methods combine information from both the output
able in the supplementary material.                                   and penultimate layer (ViM [68], ASH [15], ReAct [61],
                                                                      DICE [60], SCALE [73], NNGuide [53], fDBD [45]), iv)
                                                                         intermediate-feature methods use information from shal-
Evaluation metrics. Following standard OOD evaluation                 low to deep layers (GRAM [58], ATS [38]), and v) gradi-
protocols, we primarily report the Area Under the Receiver            ents methods utilize gradients (GradNorm [34]). For meth-
Operating Characteristic curve (AUROC) to measure the                 ods that require ID data to compute statistics or calibra-
separability between ID and OOD samples. For compara-                 tion parameters, we use the training split of the ID dataset.
bility and completeness, we also report the False Positive            For methods involving hyperparameter tuning, we rely on a
Rate at a 95% True Positive Rate (FPR95) in the supple-               held-out validation set comprising both ID and OOD sam-
mentary material. To gain a deeper understanding of model             ples to ensure fair and robust evaluation. Further implemen-
behavior, we also follow the diagnostic metrics used in [35]:         tation details, including tuning procedures and parameter
AUROCcorrect vs. OOD and AUROCincorrect vs. OOD . These met-          settings, are provided in the supplementary material.


                                                                 4

## Page 5

MSP              MLS               EBO                   ODIN              TempScale                     KLM                   GEN

                 80

                 60
                                                                                                           Training category
                                                                                                             Baseline                Augmentations   Improved Training
                 40
                                                                                                             Adversarial Training    SSL             Freezing

                           KNN              MDS               RMDS                      SHE               ViM                       ASH                   ReAct
Mean AUROC (%)




                 80

                 60

                 40


                           DICE             SCALE         NNGuide                   fDBD                  GRAM                      ATS               GradNorm

                 80

                 60

                 40

                      60     70   80   60     70    80   60     70       80    60        70   80     60     70      80        60      70     80      60      70      80
                                                                     ID classification accuracy (%)

Figure 3. Relationship between ID accuracy and OOD detection performance (AUROC) for each of the 21 OOD detection methods. Each
point represents one of the 56 ResNet-50 models, averaged over eight OOD datasets. Color indicates the model’s training category, while
the marker shape uniquely identifies each model within that category. Best viewed on screen.


4. Analysis                                                                              non-monotony, yielding a coefficient near zero (Spearman’s
                                                                                         ρ = 0.04, p ≪ 0.001). This rise-then-fall pattern is consis-
Building on our large-scale experimental setup, we charac-                               tent across all OOD dataset categories (Fig. 2), with near-
terize the key determinants of OOD detection performance                                 OOD showing a smaller min–max spread, while extreme-
by addressing four key questions: i) Does higher ID accu-                                /synthetic-OOD exhibit larger spreads across methods and
racy imply better OOD detection? ii) Are OOD detectors                                   models (Supp. Tab. 4).
merely identifying misclassified samples? iii) Where does                                   At the OOD detection method level, the picture is even
AUROC variance originate from? and iv) How robust are                                    more diverse (Fig. 3): the rise phase is steep for confidence-
detection methods across training variants?                                              based scores such as MSP and GEN, and modest for dis-
                                                                                         tance metrics like KNN, and fDBD, while the fall ranges
Does higher ID accuracy imply better OOD detection?                                      from pronounced (SCALE, ReAct) to negligible (GEN,
Our analysis reveals that the relationship between ID ac-                                GRAM) or even non-existent (KLM). Such heterogeneity
curacy and OOD detection performance, averaged across                                    demonstrates that OOD performance depends strongly on
all OOD detection methods, is not monotonic but rather a                                 the interaction between the OOD detection method heuristic
distinct rise-then-fall pattern (Fig. 1), challenging the com-                           and the training strategy, highlighting that the relationship
mon assumption of a simple positive correlation. Initially,                              between OOD detection and ID accuracy is complex.
as accuracy improves from lower levels up to the vanilla                                    The downturn at higher accuracies indicates that training
ResNet-50 baseline (76.19%), a positive correlation exists                               strategies designed to maximize ID accuracy and general-
(Spearman’s ρ = 0.38, p ≪ 0.001). This region is primar-                                 ization can undermine post-hoc OOD detection; as shown in
ily populated by models subjected to adversarial training                                the supplementary, aggressive regularization (MixUp, Cut-
(orange cluster), which degrades ID accuracy while com-                                  Mix, TorchVision 2) compresses and sparsifies the feature
pressing logit margins. This trend aligns with the findings                              space and narrows max-logit distributions, increasing ID-
of prior work [35], which observed similar behavior when                                 OOD overlap. Consequently, methods that rely on model
analyzing the impact of label noise, where models also fell                              output or specific activation characteristics degrade more,
into a low-to-baseline accuracy region.                                                  whereas geometry/statistics-based methods remain compar-
    However, once accuracy reaches or exceeds the base-                                  atively stable (see Supp. App. B and Supp. Fig. 18–20).
line band, the relationship reverses and becomes weakly                                     We contend that these complex effects were previously
negative—AUROC declines slightly as accuracy rises                                       overlooked because prior studies, which often did not in-
(Spearman’s ρ = −0.082, p ≪ 0.001). A global cor-                                        clude models trained with diverse strategies, focused on a
relation over all models and methods fails to reveal this                                limited subset of OOD methods or had results confounded


                                                                                    5

## Page 6

MSP                      KNN            GRAM
            100                                                                                       90




                                                                                                                                            AUROCincorrect vs. OOD
                                                                               AUROCcorrect vs. OOD
                                                                                                      85
AUROC (%)



             80

                                                                                                      80
             60

                                                                                                      75   Training category
             40                                                                                              Baseline
                                                                                                      70     Adversarial Training
             20                                                                                              Augmentations
                                                                                                             SSL
                  AUROCcorrect vs. incorrect     AUROCcorrect vs. OOD                                 65     Improved Training
                  AUROCID vs. OOD                AUROCincorrect vs. OOD                                      Freezing

                                                                                                                60            70       80                            60   70   80
                                                                                                                           ID classification accuracy (%)
Figure 4.       Comparing the performance of MSP, KNN,
and GRAM across multiple AUROC-based evaluation metrics.
AUROCcorrect vs. incorrect evaluates failure prediction on ID data only,       Figure 5. Relationship between ID classification accuracy and
distinguishing between correctly and incorrectly classified sam-               OOD detection performance. We distinguish between the abil-
ples. The remaining metrics assess OOD detection, either across                ity of OOD detectors to separate correctly classified ID samples
all ID samples, only correctly classified ones, or only misclassified          from OOD samples (left), and incorrectly classified ID samples
ones. Each boxplot shows the distribution over 56 models and four              from OOD samples (right). Each point represents one of the 56
OOD categories.                                                                ResNet-50 models, averaged over 21 OOD detection methods and
                                                                               four OOD categories. Color show the model’s training category;
                                                                               marker shapes uniquely identify models within each category.
by variations in model architecture and capacity.
                                                                               all detectors: methods that derive their scores from richer
Are OOD detectors merely identifying misclassified                             feature representations (e.g., KNN, NNGuide, and GRAM)
samples? Prior work [35] suggested that the correlation                        are less sensitive to misclassified ID samples, whereas
between ID accuracy and OOD performance might arise                            confidence-based approaches that rely solely on the model
because detectors simply distinguish correctly classified ID                   output experience the largest performance drop, with their
inputs from OOD samples, while misclassifications are of-                      sensitivity varying substantially across training recipes.
ten confused with OOD. We test this hypothesis by analyz-
ing OOD detector performance separately on correctly and                       Where does AUROC variance originate from? Figs. 1
incorrectly classified ID subsets.                                             and 3 reveals large performance variations across OOD de-
    Fig. 5 shows that AUROCcorrect vs. OOD and                                 tectors and training strategies, prompting a formal variance
AUROCincorrect vs. OOD follow the same rise-then-fall pattern                  analysis. To quantify factors contributing to AUROC vari-
observed in our global analysis (Fig. 1). The two metrics                      ability, we run a three-way ANOVA with factors model (i.e.,
are strongly correlated (Spearman’s ρ = 0.87, p ≪ 0.001),                      training strategy), method (i.e., OOD detection method),
indicating that methods effective at distinguishing correctly                  and OOD dataset category (i.e., OOD category).
classified ID samples from OOD data generally remain                               Fig. 6 shows that the OOD category explains the largest
robust even when the ID sample is misclassified. A                             share of variance (34.22%), reflecting the intrinsic diffi-
persistent gap of roughly 10% separates the two scores,                        culty gap between the OOD categories (near, far, extreme,
with the largest discrepancy for confidence-based methods                      and synthetic). The model–method interaction ranks sec-
that compute their OOD scores directly from model output.                      ond (21.05%), far exceeding the main effects of either factor
    Fig. 4 contrasts three representative methods: MSP,                        alone; detector performance thus depends strongly on how
KNN, and GRAM. MSP is highly sensitive to correctness:                         its heuristic aligns with the feature statistics induced by a
its failure-detection score is high (AUROCcorrect vs. incorrect =              given training recipe. A smaller yet non-trivial three-way
84.75%), and its OOD AUROC falls sharply from                                  interaction (8.66%) and a residual term (10.62%) indicate
AUROCcorrect vs. OOD = 90.72% on correctly predicted ID                        systematic rather than random effects.
samples to AUROCincorrect vs. OOD = 67.70% on misclassi-                           Robustness checks confirm this picture. Omitting one
fied ones. KNN shows a lower failure detection AUROC                           OOD category at a time (details in the supplementary ma-
(63.15%) with a much narrower gap (correct: 89.13%, in-                        terial) shows that removing the toughest category (near-
correct: 83.96%), resulting in a higher overall AUROC of                       OOD) cuts the OOD category term and boosts the model–
87.93%. GRAM achieves virtually identical scores on both                       method interaction to 33.4%. The coupling between detec-
subsets (correct: 89.67%, incorrect: 89.66%), with a near-                     tor and model is therefore partially masked when all meth-
random failure-detection AUROC (50.03%), demonstrating                         ods struggle uniformly on the most challenging OOD test
robustness to ID classification quality.                                       data. Across all runs, the interaction term remains a major
    The supplementary material confirms this pattern across                    contributor, underscoring its central importance.


                                                                           6

## Page 7

17.5
                      Method                                                       15.0
                        Model                                                      12.5




                                                                       Mean Rank
                OOD Category                                                       10.0                              OOD Method Category
              Model × Method                                                        7.5                                  Classification-based
                                                                                                                         Feature-based
        Model × OOD Category                                                        5.0                                  Hybrid
                                                                                                                         Intermediate Features
      Method × OOD Category                                                         2.5
                                                                                                                         Gradients
Model × Method × OOD Group                                                          0.0




                                                                                             GRAM




                                                                                               GEN
                                                                                               ViM
                                                                                            SCALE



                                                                                               ASH
                                                                                           NNGuide

                                                                                              fDBD
                                                                                               KNN
                                                                                              ReAct



                                                                                              ODIN
                                                                                              DICE
                                                                                               MLS
                                                                                               ATS
                                                                                             RMDS

                                                                                               SHE
                                                                                               KLM
                                                                                               EBO
                                                                                               MSP

                                                                                               MDS
                                                                                          GradNorm
                                                                                          TempScale
                      Residual

                                 0     10       20        30
                                     Explained Variance (%)
                                                                       Figure 7. Robustness of OOD detection methods measured by
Figure 6. Proportions of AUROC variance explained by each fac-         model-averaged mean rank across 56 models. Methods are ranked
tor in the three-way ANOVA (56 models × 21 OOD detection               per model by mean AUROC, and the overall rank is the average
methods × 4 OOD categories).                                           across models. Error bars show 95% CIs; lower rank is better.


   In summary, OOD performance depends as much on the                  mance with the baseline model. This pattern suggests an
alignment between the OOD detector and the model as on                 over-specialization to the characteristics of vanilla training,
the OOD set itself, underscoring the critical role of the train-       on which they were likely developed and benchmarked.
ing strategy.                                                             These two observations highlight a trade-off between
                                                                       high-performing specialists and more consistent generalists.
                                                                       The specialists, such as ReAct and SCALE, are effective
How robust are detection methods across training vari-                 because their heuristics are finely tuned to specific char-
ants? A central finding of this work is that the perfor-               acteristics of vanilla-trained models, for instance, clipping
mance of an OOD detector is conditional: it depends not                high-activation features. This explains both their SOTA per-
only on the OOD data but also on the training strategy used            formance on the baseline and their high variance across di-
to obtain the model. OOD detection methods are typically               verse training strategies, as advanced training recipes alter
validated on a single training strategy, often the ResNet-50           the very activation patterns on which these methods depend.
baseline [24]. This means that there is no universally best            Generalists leverage more invariant properties. GRAM and
method, but it also implies that selecting the most suitable           fDBD illustrate this: they remain among the top methods
method for a given task can be a challenge. Robustness,                (Fig. 7), show only modest performance drops from base-
defined here as delivering consistently strong OOD detec-              line (GRAM: −1.74%(±3.24), fDBD: −2.17%(±2.38)),
tion performance across heterogeneous training strategies,             and maintain tight score distributions (Fig. 8). GRAM’s
therefore becomes as important as raw performance.                     higher-order statistics and fDBD’s distance metric depend
   To consider both aspects, Fig. 7 orders all OOD detection           less on absolute activation magnitudes, making them less
methods by their mean rank over the 56 models. A Fried-                sensitive to training-induced changes.
man test confirms statistically significant differences among             OOD robustness hinges primarily on one key factor: how
methods (χ2 = 467.97, p ≪ 0.001), validating this anal-                well a detector’s scoring rule aligns with the feature repre-
ysis. Model enhancement OOD detection methods domi-                    sentations produced by a given training recipe. Stability
nate: SCALE leads (mean rank 5.15, 95% CI [4.08, 6.22]),               is achieved not by richer features alone, but by combin-
followed by NNGuide, ASH, and fDBD. The intermediate-                  ing them with training-agnostic heuristics (e.g., statistical
feature-based method GRAM, introduced in 2020, also                    summaries or distance-based measures) that remain effec-
proves to be a strong and consistent performer, ranking                tive across the diverse characteristics and activation patterns
second with a tight confidence interval (mean rank 6.54,               induced by varying training strategies.
95% CI [5.84, 7.24]). In contrast, several classification-
based methods, such as MSP, KLM, and TempScale, oc-
                                                                       5. Discussion and Conclusion
cupy lower ranks on average.
   Figures 3 and 8 reveal two further insights. First, many            Our study challenges, extends, and aligns with previous
of the top-ranked methods exhibit high variance, meaning               findings on the relationship between ID classification ac-
that while their performance can be excellent, it can also             curacy and OOD detection performance. Contrary to sim-
be mediocre when paired with the wrong training strategy.              plified claims of a universally positive or negative cor-
Second, many models, such as ASH, SCALE, and ReAct,                    relation, our large-scale, controlled analysis demonstrates
as well as some of the weaker performers, including DICE               a non-monotonic and method-dependent relationship for
and GradNorm, exhibit their best or nearly optimal perfor-             OOD detection. While prior work has often reported a pos-


                                                                   7

## Page 8

90
                                                                       OOD generalization result instead from a method’s ability
                                                                       to combine rich, training-induced features with patterns that
Mean AUROC (%)



                 80

                 70                                                    remain stable across diverse training strategies.
                 60
                      OOD Method Category
                        Classification-based
                 50     Feature-based                                  Guidance for Practitioners. Practitioners should be
                        Hybrid
                 40     Intermediate Features
                        Gradients
                                                                       aware that while achieving baseline levels of ID accu-
                 30
                                                                       racy generally supports better OOD detection, further ac-
                         GRAM




                           GEN
                          fDBD

                           KNN




                           KLM
                        SCALE




                           MLS
                           ATS
                           ASH
                           MSP
                           SHE
                         RMDS
                          ReAct
                           EBO
                          DICE

                           MDS
                       NNGuide

                          ODIN
                           ViM




                      GradNorm
                      TempScale
                                                                       curacy improvements through sophisticated training meth-
                                                                       ods do not universally guarantee enhanced OOD detec-
                                                                       tion performance, and can instead harm it. When utilizing
Figure 8. Mean AUROC distribution for each OOD detection               standard, vanilla models, top-ranked methods from estab-
method, across 56 models and four OOD categories. Diamond              lished benchmarks typically remain reliable. However, for
markers denote the mean for each method; the red horizontal            models customized with advanced training or strong reg-
line indicates that method’s performance on the baseline ResNet-       ularization, generalist OOD detectors should be preferred
50 [24]. Methods are sorted by overall mean AUROC.                     that do not rely on specific activation patterns—in prac-
                                                                       tice, geometry/statistics-based methods that use distances or
                                                                       higher-order feature statistics (e.g., KNN, GRAM, RMDS)
itive correlation between ID accuracy and OOD detection                are typically more stable; otherwise, tailored evaluations
performance, these findings frequently stem from studies               should be performed to select the OOD detector best aligned
with significant potential confounders. For instance, studies          with the model’s learned representations.
[19, 66] evaluated a broad range of architectures combined
with only a small subset of OOD detection methods, while               Limitations and Future Work. A central limitation of
[35] varied both datasets and architectures. In contrast, a re-        our study is the deliberate restriction to the ResNet-50 archi-
cent study [69] reported negative correlations for large-scale         tecture. This methodological decision was necessary to iso-
benchmarks, examining narrowly optimized model sets in                 late training-induced effects from architectural or capacity-
the context of outlier exposure and thus overlooking the ini-          based confounding factors, a challenge in prior studies.
tial positive trend at lower accuracy ranges. Our results              While this design provides clear and definitive insights,
clarify these apparent contradictions by revealing them as             our findings also support recent hypotheses [81] that OOD
different perspectives on the same nuanced relationship.               methods are implicitly tuned to CNN-based architectures,
    Critically, we reveal a strong interaction effect between          explaining why alternatives such as Vision Transformers
models and OOD detection methods, which alone accounts                 (ViTs) [16], despite higher ID accuracy, often show weaker
for over 20% of OOD detection performance variance. This               OOD performance. While a preliminary ablation study
suggests that the effectiveness of an OOD detection method             (App. C) confirms our main findings also for ViTs, a simi-
is not an independent property, but rather is fundamentally            larly detailed examination of the model-method interactions
coupled with the specific, training-induced characteristics            is a valuable future direction.
of the model. This high degree of interaction has direct and
critical implications for our field. Evaluating OOD methods            Conclusion. We have identified important shortcomings
on a single model configuration captures only a limited slice          in previous studies that show correlations between ID ac-
of a more extensive interaction space, potentially rewarding           curacy and OOD performance. Our large-scale, carefully
overly specialized methods. Our findings issue a clear call            designed study shows that while correlations exist between
for a shift in evaluation culture. Moving forward, robust              certain ID metrics and OOD performance under specific
benchmarking must adopt multi-model evaluations across a               conditions, OOD robustness ultimately depends on the in-
wide spectrum of training strategies to establish a method’s           teraction between the model and the detection method, and
true generalization, reveal its limitations and hidden depen-          these correlations do not reflect the broader dynamics of
dencies, and ensure the community is not simply optimizing             generalization. This makes OOD performance too com-
for methods that are brittle and over-specialized to a single,         plex to be inferred from a single ID measure. This calls
default training condition.                                            for a shift toward systematic evaluation across diverse mod-
    In summary, our findings demonstrate that a simple                 els and the development of diagnostics that capture model–
monotonic correlation does not hold, confirming that ID ac-            method compatibility—essential for reliable OOD detec-
curacy alone is not a reliable indicator for OOD detection             tion and trustworthy pre-deployment assessment.
performance. Thus, the assumption that improved ID ac-                    Acknowledgments This work was supported by
curacy automatically translates into better (or worse) OOD             KESTRELEYE GmbH, whose financial contribution and
detection performance is not justified. Stability and robust           commitment made this research possible.


                                                                   8

## Page 9

References                                                                    Worth 16x16 Words: Transformers for Image Recognition at
                                                                              Scale. In Proc. ICLR, 2021. 8, 19
 [1] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David
                                                                         [17] Xuefeng Du, Zhaoning Wang, Mu Cai, and Yixuan Li. Vos:
     Lopez-Paz. Invariant Risk Minimization. arXiv preprint
                                                                              Learning what you don’t know by virtual outlier synthesis.
     arXiv:1907.02893, 2020. 3
                                                                              In Proc. ICLR, 2022. 2
 [2] Haoyue Bai, Gregory Canal, Xuefeng Du, Jeongyeol Kwon,
                                                                         [18] Benjamin Erichson, Soon Hoe Lim, Winnie Xu, Francisco
     Robert D Nowak, and Yixuan Li. Feed Two Birds with One
                                                                              Utrera, Ziang Cao, and Michael Mahoney. NoisyMix:
     Scone: Exploiting Wild Data for Both Out-of-Distribution
                                                                              Boosting model robustness to common corruptions. In Proc.
     Generalization and Detection. In International Conference
                                                                              AISTATS, 2024. 4, 13
     on Machine Learning, 2023. 2
 [3] Mélanie Bernhardt, Fabio De Sousa Ribeiro, and Ben                 [19] Ido Galil, Mohammed Dabbah, and Ran El-Yaniv. A frame-
     Glocker. Failure detection in medical image classification:              work for benchmarking class-out-of-distribution detection
     A reality check and benchmarking testbed. In TMLR, 2022.                 and its application to ImageNet. In Proc. ICLR, 2023. 2,
     4                                                                        3, 8
 [4] Julian Bitterwolf, Alexander Meinke, and Matthias Hein.             [20] Paul Gavrikov and Janis Keuper. Can biases in imagenet
     Certifiably Adversarially Robust Detection of Out-of-                    models explain generalization? In Proc. CVPR, 2024. 12
     Distribution Data. In NeurIPS, 2020. 12                             [21] Paul Gavrikov and Janis Keuper. The Power of Linear Com-
 [5] Julian Bitterwolf, Maximilian Mueller, and Matthias Hein.                binations: Learning with Random Convolutions, 2024. 4,
     In or Out? Fixing ImageNet Out-of-Distribution Detection                 13
     Evaluation. In Proc. ICLR Workshops, 2023. 3                        [22] Robert Geirhos, Patricia Rubisch, Claudio Michaelis,
 [6] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Pi-             Matthias Bethge, Felix A. Wichmann, and Wieland Brendel.
     otr Bojanowski, and Armand Joulin. Unsupervised learn-                   Imagenet-trained CNNs are biased towards texture; increas-
     ing of visual features by contrasting cluster assignments. In            ing shape bias improves accuracy and robustness. In Proc.
     NeurIPS, 2020. 4, 13                                                     ICLR, 2019. 4, 13
 [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou,           [23] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger.
     Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-               On Calibration of Modern Neural Networks. In Proc. ICML,
     ing properties in self-supervised vision transformers. In                2017. 4, 12
     Proc. ICCV, 2021. 13                                                [24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
 [8] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad                      Deep Residual Learning for Image Recognition. In Proc.
     Norouzi, and Geoffrey E Hinton. Big self-supervised models               CVPR, 2016. 2, 3, 4, 7, 8, 13, 15
     are strong semi-supervised learners. In NeurIPS, 2020. 13           [25] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr
 [9] Xinlei Chen, Saining Xie, and Kaiming He. An Empirical                   Dollár, and Ross Girshick. Masked Autoencoders Are Scal-
     Study of Training Self-Supervised Vision Transformers . In               able Vision Learners. In Proc. CVPR, 2022. 19
     Proc. ICCV, 2021. 4, 13                                             [26] Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Jun-
[10] Xiangning Chen, Cho-Jui Hsieh, and Boqing Gong. When                     yuan Xie, and Mu Li. Bag of Tricks for Image Classification
     Vision Transformers Outperform ResNets without Pre-                      with Convolutional Neural Networks. Proc. CVPR, 2018. 2
     training or Strong Data Augmentations. In Proc. ICLR, 2022.         [27] Matthias Hein, Maksym Andriushchenko, and Julian Bitter-
     19                                                                       wolf. Why ReLU networks yield high-confidence predic-
[11] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy                   tions far away from the training data and how to mitigate the
     Mohamed, and Andrea Vedaldi. Describing Textures in the                  problem. Proc. CVPR, 2019. 1
     Wild. In Proc. CVPR, 2014. 3                                        [28] Dan Hendrycks and Kevin Gimpel. A Baseline for Detect-
[12] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Va-                    ing Misclassified and Out-of-Distribution Examples in Neu-
     sudevan, and Quoc V. Le. Autoaugment: Learning augmen-                   ral Networks. In Proc. ICLR, 2017. 1, 2, 4, 12
     tation strategies from data. In Proc. CVPR, 2019. 4, 13             [29] Dan Hendrycks*, Norman Mu*, Ekin Dogus Cubuk, Barret
[13] Ekin Dogus Cubuk, Barret Zoph, Jon Shlens, and Quoc Le.                  Zoph, Justin Gilmer, and Balaji Lakshminarayanan. Aug-
     Randaugment: Practical automated data augmentation with                  mix: A simple method to improve robustness and uncertainty
     a reduced search space. In NeurIPS, 2020. 4, 13                          under data shift. In Proc. ICLR, 2020. 4, 13
[14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,              [30] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kada-
     and Li Fei-Fei. ImageNet: A large-scale hierarchical image               vath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu,
     database. In Proc. CVPR, 2009. 2, 3                                      Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt,
[15] Andrija Djurisic, Nebojsa Bozanic, Arjun Ashok, and                      and Justin Gilmer. The many faces of robustness: A critical
     Rosanne Liu. Extremely Simple Activation Shaping for Out-                analysis of out-of-distribution generalization. Proc. ICCV,
     of-Distribution Detection. In Proc. ICLR, 2023. 1, 2, 4, 12              2021. 13
[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,              [31] Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou,
     Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,                      Joseph Kwon, Mohammadreza Mostajabi, Jacob Steinhardt,
     Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-                 and Dawn Song. Scaling Out-of-Distribution Detection for
     vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is               Real-World Settings. In Proc. ICML, 2022. 2, 3, 4, 12


                                                                     9

## Page 10

[32] Dan Hendrycks, Andy Zou, Mantas Mazeika, Leonard Tang,                [49] Apostolos Modas, Rahul Rade, Guillermo Ortiz-Jiménez,
     Bo Li, Dawn Song, and Jacob Steinhardt. Pixmix: Dreamlike                  Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard.
     pictures comprehensively improve safety measures. Proc.                    Prime: A few primitives can boost robustness to common
     CVPR, 2022. 4, 13                                                          corruptions. In Proc. ECCV, 2022. 4, 13
[33] Yen-Chang Hsu, Yilin Shen, Hongxia Jin, and Zsolt Kira.               [50] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar
     Generalized odin: Detecting out-of-distribution image with-                Fawzi, and Pascal Frossard. Universal adversarial perturba-
     out learning from out-of-distribution data. In Proc. CVPR,                 tions. In Proc. CVPR, 2017. 1
     2020. 2                                                               [51] Patrick Müller, Alexander Braun, and Margret Keuper. Clas-
[34] Rui Huang, Andrew Geng, and Yixuan Li. On the Impor-                       sification robustness to common optical aberrations. In Proc.
     tance of Gradients for Detecting Distributional Shifts in the              ICCV Workshops, 2023. 4, 13
     Wild. In NeurIPS, 2021. 2, 4, 12                                      [52] Anh M Nguyen, Jason Yosinski, and Jeff Clune. Deep Neural
[35] Galadrielle Humblot-Renaux, Sergio Escalera, and                           Networks are Easily Fooled: High Confidence Predictions
     Thomas B. Moeslund. A noisy elephant in the room:                          for Unrecognizable Images. Proc. CVPR, 2015. 1
     Is your out-of-distribution detector robust to label noise? In        [53] Jaewoo Park, Yoon Gyo Jung, and Andrew Beng Jin Teoh.
     Proc. CVPR, 2024. 2, 3, 4, 5, 6, 8, 12, 14                                 Nearest neighbor guidance for out-of-distribution detection.
[36] Priyank Jaini, Kevin Clark, and Robert Geirhos. Intriguing                 In Proc. ICCV, 2023. 4, 12
     properties of generative classifiers. In Proc. ICLR, 2024. 4,         [54] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
     13                                                                         James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
[37] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna,                    Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison,
     Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and                 Andreas Kopf, Edward Yang, Zachary DeVito, Martin Rai-
     Dilip Krishnan. Supervised contrastive learning. In NeurIPS,               son, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
     2020. 4, 13                                                                Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An
[38] Gerhard Krumpl, Henning Avenhaus, Horst Possegger, and                     Imperative Style, High-Performance Deep Learning Library.
     Horst Bischof. ATS: Adaptive Temperature Scaling for En-                   In NeurIPS, 2019. 4, 13, 15
     hancing Out-of-Distribution Detection Methods. In Proc.               [55] Francesco Pinto, Harry Yang, Ser-Nam Lim, Philip Torr, and
     WACV, 2024. 4, 12, 15                                                      Puneet K. Dokania. Using mixup as a regularizer can sur-
[39] Yann Lecun, Lé’on Bottou, Yoshua Bengio, and Patrick                      prisingly improve accuracy & out-of-distribution robustness.
     Haffner. Gradient-based Learning Applied to Document                       In NeurIPS, 2022. 4, 13
     Recognition. IEEE, 86(11):2278–2324, 1998. 3, 12                      [56] Jie Jessie Ren, Stanislav Fort, Jeremiah Zhe Liu, Abhi-
[40] Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo                              jit Guha Roy, Shreyas Padhy, and Balaji Lakshminarayanan.
     Shin. A Simple Unified Framework for Detecting Out-of-                     A Simple Fix to Mahalanobis Distance for Improving Near-
     Distribution Samples and Adversarial Attacks. In NeurIPS,                  OOD Detection. arXiv preprint arXiv:2106.09022, 2021. 1,
     2018. 1, 2, 4, 12                                                          4, 12
[41] Yingwei Li, Qihang Yu, Mingxing Tan, Jieru Mei, Peng                  [57] Hadi Salman, Andrew Ilyas, Logan Engstrom, Ashish
     Tang, Wei Shen, Alan Yuille, and cihang xie. Shape-texture                 Kapoor, and Aleksander Madry. Do Adversarially Robust
     debiased neural network training. In Proc. ICLR, 2021. 4,                  ImageNet Models Transfer Better? In NeurIPS, 2020. 4, 13
     13                                                                    [58] Chandramouli Shama Sastry and Sageev Oore. Detecting
[42] Shiyu Liang, Yixuan Li, and R. Srikant. Enhancing The Reli-                Out-of-Distribution Examples with Gram Matrices. In Proc.
     ability of Out-of-distribution Image Detection in Neural Net-              ICML, 2020. 2, 4, 12
     works. In Proc. ICLR, 2018. 2, 4, 12                                  [59] Andreas Peter Steiner, Alexander Kolesnikov, Xiaohua Zhai,
[43] Sungbin Lim, Ildoo Kim, Taesup Kim, Chiheon Kim, and                       Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to
     Sungwoong Kim. Fast autoaugment. In NeurIPS, 2019. 4,                      train your ViT? Data, Augmentation, and Regularization in
     13                                                                         Vision Transformers. In TMLR, 2022. 19
[44] Ziqian Lin, Sreya Dutta Roy, and Yixuan Li. Mood: Multi-              [60] Yiyou Sun and Yixuan Li. DICE: Leveraging Sparsification
     level out-of-distribution detection. In Proc. CVPR, 2021. 2                for Out-of-Distribution Detection. In Proc. ECCV, 2022. 4,
[45] Litian Liu and Yao Qin. Fast decision boundary based out-                  12
     of-distribution detector. ICML, 2024. 4, 12                           [61] Yiyou Sun, Chuan Guo, and Yixuan Li. ReAct: Out-
[46] Weitang Liu, Xiaoyun Wang, John D. Owens, and Yixuan                       of-distribution Detection With Rectified Activations. In
     Li. Energy-based Out-of-distribution Detection. In NeurIPS,                NeurIPS, 2021. 1, 2, 4, 12, 18
     2020. 2, 4, 12                                                        [62] Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-
[47] Xixi Liu, Yaroslava Lochman, and Zach Christopher. Gen:                    of-distribution Detection with Deep Nearest Neighbors. In
     Pushing the limits of softmax-based out-of-distribution de-                Proc. ICML, 2022. 1, 2, 3, 4, 12
     tection. In Proc. CVPR, 2023. 4, 12                                   [63] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon
[48] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,                      Shlens, and Zbigniew Wojna. Rethinking the inception ar-
     Dimitris Tsipras, and Adrian Vladu. Towards Deep Learn-                    chitecture for computer vision. In Proc. CVPR, 2016. 2
     ing Models Resistant to Adversarial Attacks. In Proc. ICLR,           [64] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco
     2018. 4, 13                                                                Massa, Alexandre Sablayrolles, and Herve Jegou. Training


                                                                      10

## Page 11

data-efficient image transformers & distillation through at-               of-Distribution Detection based on In-Distribution Data Pat-
     tention. In Proc. ICML, 2021. 19                                           terns Memorization with Modern Hopfield Energy. In Proc.
[65] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui,                       ICLR, 2023. 2, 3, 4, 12
     Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and              [80] Jingyang Zhang, Jingkang Yang, Pengyun Wang, Haoqi
     Serge Belongie. The INaturalist Species Classification and                 Wang, Yueqian Lin, Haoran Zhang, Yiyou Sun, Xuefeng Du,
     Detection Dataset. In Proc. CVPR, 2018. 3                                  Yixuan Li, Ziwei Liu, Yiran Chen, and Hai Li. Openood
[66] Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisser-                    v1.5: Enhanced benchmark for out-of-distribution detection.
     man. Open-Set Recognition: a Good Closed-Set Classifier is                 arXiv preprint arXiv:2306.09301, 2023. 4, 12
     All You Need? In Proc. ICLR, 2022. 2, 3, 8                            [81] Jingyang Zhang, Jingkang Yang, Pengyun Wang, Haoqi
[67] Vasilis Vryniotis. How to Train State-Of-The-Art Mod-                      Wang, Yueqian Lin, Haoran Zhang, Yiyou Sun, Xuefeng Du,
     els Using TorchVision’s Latest Primitives, 2023. https:                    Yixuan Li, Ziwei Liu, Yiran Chen, and Hai Li. OpenOOD
     //pytorch.org/blog/how- to- train- state-                                  v1.5: Enhanced benchmark for out-of-distribution detection.
     of - the - art - models - using - torchvision -                            J. of DMLR, 2024. 2, 3, 4, 8, 12, 13, 19
     latest- primitives/ [Accessed: 2025-07-01]. 2, 4,                     [82] Yu Zheng, Zhi Zhang, Shen Yan, and Mi Zhang. Deep au-
     13, 15                                                                     toaugment. In Proc. ICLR, 2022. 4
[68] Haoqi Wang, Zhizhong Li, Litong Feng, and Wayne Zhang.                [83] Lin Zhu, Yifeng Yang, Qinying Gu, Xinbing Wang, Chenghu
     ViM: Out-Of-Distribution with Virtual-logit Matching. In                   Zhou, and Nanyang Ye. CRoFT: Robust fine-tuning with
     Proc. CVPR, 2022. 3, 4, 12                                                 concurrent optimization for OOD generalization and open-
[69] Hongjun Wang, Sagar Vaze, and Kai Han. Dissecting out-                     set OOD detection. In Proc. ICML, 2024. 2
     of-distribution detection and open-set recognition: A critical
     analysis of methods and benchmarks. IJCV, 2024. 2, 8
[70] Ross Wightman.         Pytorch image models.        https :
     //github.com/huggingface/pytorch- image-
     models, 2019. 4, 13
[71] Ross Wightman, Hugo Touvron, and Herve Jegou. ResNet
     strikes back: An improved training procedure in timm. In
     NeurIPS Workshops, 2021. 2, 4, 13
[72] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-
     MNIST: a Novel Image Dataset for Benchmarking Machine
     Learning Algorithms. arXiv preprint arXiv:1708.07747,
     2017. 3, 12
[73] Kai Xu, Rongyu Chen, Gianni Franchi, and Angela Yao.
     Scaling for Training Time and Post-hoc Out-of-distribution
     Detection Enhancement. In Proc. ICLR, 2024. 2, 3, 4, 12
[74] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu.
     Generalized Out-of-Distribution Detection: A Survey. arXiv
     preprint arXiv:2110.11334, 2021. 2
[75] Jingkang Yang, Pengyun Wang, Dejian Zou, Zitang
     Zhou, Kunyuan Ding, WENXUAN PENG, Haoqi Wang,
     Guangyao Chen, Bo Li, Yiyou Sun, Xuefeng Du, Kaiyang
     Zhou, Wayne Zhang, Dan Hendrycks, Yixuan Li, and Zi-
     wei Liu. OpenOOD: Benchmarking Generalized Out-of-
     Distribution Detection. In NeurIPS Datasets and Bench-
     marks Track, 2022. 3, 4, 12
[76] Nanyang Ye, Kaican Li, Haoyue Bai, Runpeng Yu, Lanqing
     Hong, Fengwei Zhou, Zhenguo Li, and Jun Zhu. OoD-
     Bench: Quantifying and Understanding Two Dimensions of
     Out-of-Distribution Generalization. In Proc. CVPR, 2022. 3
[77] Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Seong Joon
     Oh, Youngjoon Yoo, and Junsuk Choe. CutMix: Regular-
     ization Strategy to Train Strong Classifiers With Localizable
     Features . In Proc. ICCV, 2019. 1, 4, 13, 15
[78] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and
     David Lopez-Paz. mixup: Beyond empirical risk minimiza-
     tion. In Proc. ICLR, 2018. 1, 4, 13, 15
[79] Jinsong Zhang, Qiang Fu, Xu Chen, Lun Du, Zelin Li, Gang
     Wang, Xiaoguang Liu, Shi Han, and Dongmei Zhang. Out-


                                                                      11

## Page 12

One Model, Many Behaviors: Training-Induced Effects on Out-of-Distribution
                                Detection
                                               Supplementary Material
   This supplementary document expands on the main                       Method                  Hyperparameter Search Space
manuscript. It provides full experimental details (Ap-                   MSP [28]
pendix A), comprehensive results that support and extend                 MLS [31]
our analyses (Appendix B), and additional experiments with               EBO [46]
                                                                                          temperature T ∈ {1, 10, 100, 1000}
Vision Transformers (ViT) (Appendix C).                                  ODIN [42]
                                                                                          perturbation mag. σ ∈ {0.0014, 0.0028}
                                                                         TempScale [23]
A. Experimental Details                                                  KLM [31]
                                                                                          gamma ∈ {0.01, 0.1, 0.5, 1, 2, 5, 10}
                                                                         GEN [47]
A.1. Implementation Details                                                                top-M classes ∈ {10, 50, 100, 200, 500, 1000}
                                                                         KNN [62]         K ∈ {50, 100, 200, 500, 1000}
Software stack. Our experimental framework is built                      MDS [40]
upon the OpenOOD [75, 80, 81] framework. Specifically,                   RMDS [56]
we utilize the public fork from Humblot-Renaux [35], as                  SHE [79]
                                                                         ViM [68]
it provides a GRAM [58] implementation that follows the                  ASH [15]         percentile ∈ {65, 70, 75, 80, 85, 90, 95}
official implementation details, which also leverages infor-             ReAct [61]       percentile ∈ {70, 80, 85, 90, 95, 99}
mation from intermediate layers. We extend the model zoo                 DICE [60]        percentile ∈ {10, 30, 50, 70, 90}
by integrating 56 ImageNet checkpoints, adapted from the                 SCALE [73]       percentile ∈ {65, 70, 75, 80, 85, 90, 95}
                                                                         NNGuide [53]
collection provided by [20]. The full list of all models used            fDBD [45]        normalization ∈ {true, false}
in this study, along with their training categories and perfor-          GRAM [58]
mance metrics, is detailed in Tab. 2. All evaluations in this            ATS [38]
study are executed within this unified framework.                        GradNorm [34]
    To broaden the evaluation scope, we also enrich the data
layer with two additional OOD categories: (i) extreme-                 Table 1. Overview of hyperparameter search space for all consid-
                                                                       ered OOD detection methods.
OOD including MNIST [39] and Fashion-MNIST [72], and
(ii) synthetic-OOD including the unit-test data provided by
NINCO [4]. This setup ensures reproducibility and fair
                                                                       therefore, provides a fair test across the diverse training
comparison with a broad set of diverse training strategies,
                                                                       strategies evaluated here. The exact settings and hyperpa-
OOD test sets, and existing state-of-the-art OOD detection
                                                                       rameter search spaces adopted for each method are detailed
methods.
                                                                       in Tab. 1.

Hardware and system configuration. All experiments
were executed on a workstation equipped with an Intel Core
                                                                       B. Detailed Results
i9-9900X (10 cores, 3.5 GHz) and two NVIDIA GPUs                       Does higher ID accuracy imply better OOD detection?
(RTX 2080Ti + RTX 3090). The software environment                      To validate that our findings are not an artifact of the AU-
consisted of Ubuntu 22.04, Python 3.10, PyTorch 2.0.1, and             ROC metric, we perform an equivalent analysis using the
CUDA 11.8.                                                             False Positive Rate at 95% True Positive Rate (FPR95). As
                                                                       shown in Fig. 9, this analysis plots ID classification accu-
OOD detection methods. Many OOD detection methods                      racy against FPR95, where lower values signify better OOD
require a configuration phase prior to evaluation, for which           detection performance.
we strictly follow the OpenOOD benchmark protocols to                      This analysis quantitatively confirms the visually ob-
ensure comparability. This process includes two types of               served mirrored fall-then-rise pattern. Consistent with the
setup: some methods are calibrated on the ID training set              AUROC results, the overall relationship between accuracy
to compute statistics or other parameters, while others have           and FPR95 yields a weak global correlation (Spearman’s
crucial hyperparameters that are tuned on a held-out vali-             ρ = −0.04, p ≪ 0.001). Similar to the AUROC analy-
dation set containing both ID and OOD samples. Although                sis, in the low-to-baseline accuracy regime, performance is
detectors ship with default hyperparameters, these defaults            primarily driven by adversarially trained models, which ex-
are typically tuned to a vanilla training recipe, which can            hibit a strong negative correlation between ID classification
risk biasing the comparison. Re-optimizing all parameters,             accuracy and FPR95 (OOD performance improves). Con-


                                                                  12

## Page 13

In-Distribution         Out-of-Distribution
                   Model                               Category
                                                                              Accuracy ↑   ECE ↓     AUROC ↑         FPR95 ↓
                   Original Baseline [24]              Baseline                 76.19        3.62   89.07± 9.95    42.05±27.62
                   PGD-AT (l2 , ϵ = 0) [48, 57]        Adversarial Training     75.90       3.50    88.90±10.22    42.27±28.36
                   PGD-AT (l2 , ϵ = 0.01) [48, 57]     Adversarial Training     75.69       3.02    88.50± 9.97    44.42±27.39
                   PGD-AT (l2 , ϵ = 0.03) [48, 57]     Adversarial Training     75.88       2.80    88.37± 9.76    44.96±26.67
                   PGD-AT (l2 , ϵ = 0.05) [48, 57]     Adversarial Training     75.58       2.68    88.86±10.05    42.97±28.03
                   PGD-AT (l2 , ϵ = 0.1) [48, 57]      Adversarial Training     74.86       2.21    88.53±10.54    44.48±28.62
                   PGD-AT (l2 , ϵ = 0.25) [48, 57]     Adversarial Training     74.15       1.79    87.90±10.14    46.97±26.49
                   PGD-AT (l2 , ϵ = 0.5) [48, 57]      Adversarial Training     73.22       1.58    87.50±10.55    48.06±26.35
                   PGD-AT (l2 , ϵ = 1) [48, 57]        Adversarial Training     70.50       3.35    85.99±10.97    51.61±26.49
                   PGD-AT (l2 , ϵ = 3) [48, 57]        Adversarial Training     62.86       9.06    79.85±10.42    65.71±21.19
                   PGD-AT (l2 , ϵ = 5) [48, 57]        Adversarial Training     56.15      12.65    74.45± 9.75    71.33±18.70
                   PGD-AT (l∞ , ϵ = 0.5) [48, 57]      Adversarial Training     73.75       1.23    87.25± 9.86    48.39±23.86
                   PGD-AT (l∞ , ϵ = 1.0) [48, 57]      Adversarial Training     72.13       2.78    85.86± 9.93    53.04±25.30
                   PGD-AT (l∞ , ϵ = 2.0) [48, 57]      Adversarial Training     69.13       4.80    83.42±10.24    58.99±25.04
                   PGD-AT (l∞ , ϵ = 4.0) [48, 57]      Adversarial Training     63.94       8.92    80.26±10.39    65.47±20.56
                   PGD-AT (l∞ , ϵ = 8.0) [48, 57]      Adversarial Training     54.55      13.28    70.62±11.43    73.24±16.29
                   AutoAugment (270Ep) [12]            Augmentations            77.52       2.74    89.54±10.06    40.22±29.92
                   FastAutoAugment (270Ep) [43]        Augmentations            77.69       3.58    88.77± 9.94    42.82±29.14
                   StyleAugment [22, 81]               Augmentations            74.68       1.91    88.35±10.17    43.92±28.09
                   RandAugment (270Ep) [13]            Augmentations            77.65       3.26    88.78± 9.57    43.16±28.80
                   AugMix (180Ep) [29]                 Augmentations            77.63       1.88    89.72± 9.51    40.78±27.29
                   DeepAugment [30]                    Augmentations            76.76       2.37    88.03± 9.49    45.72±25.13
                   DeepAugment + AugMix [30]           Augmentations            75.89       2.82    88.56±11.00    41.10±30.16
                   RegMixup [55]                       Augmentations            76.69       2.94    88.14± 9.30    45.87±26.13
                   Diffusion-like Noise [36]           Augmentations            67.26       1.79    84.24±11.48    53.83±25.72
                   NoisyMix [18]                       Augmentations            77.14      12.92    86.41±10.25    50.67±27.04
                   OpticsAugment [51]                  Augmentations            74.25       3.02    88.88±10.67    41.10±28.71
                   PRIME [49]                          Augmentations            76.99       2.79    88.63± 9.43    44.04±26.30
                   PixMix (90Ep) [32]                  Augmentations            77.43       1.49    88.07± 8.94    44.60±26.33
                   PixMix (180Ep) [32]                 Augmentations            78.18       2.19    87.29± 8.62    46.45±25.65
             ⋆     MixUp [78]                          Augmentations            77.55      20.40    84.13±11.08    52.89±28.08
                   CutMix [77]                         Augmentations            78.62      18.79    79.65±11.57    58.89±24.11
                   ShapeNet (SIN) [22]                 Augmentations            60.22       6.80    84.65±12.28    47.16±34.05
                   ShapeNet (SIN+IN) [22]              Augmentations            76.74       4.82    88.62± 9.23    44.79±26.48
                   ShapeNet (SIN+IN → IN) [22]         Augmentations            74.68       1.91    88.34±10.18    43.96±28.08
                   Texture/Shape debiased [41]         Augmentations            76.92       3.21    87.72±10.02    46.02±27.19
                   Texture/Shape-Shape biased [41]     Augmentations            76.31       2.38    88.26± 9.84    44.72±27.46
                   Texture/Shape-Texture biased [41]   Augmentations            75.31       3.22    89.03±10.05    41.51±28.97
                   Dinov1 [7]                          SSL                      75.32        2.04   87.46±11.91    40.19±32.33
                   MoCo v3 (100Ep) [9]                 SSL                      68.99        3.79   84.11±12.19    50.33±27.68
                   MoCo v3 (300Ep) [9]                 SSL                      72.84        3.44   85.34±11.21    50.55±26.64
                   MoCo v3 (1000Ep) [9]                SSL                      74.62        2.34   85.58±10.73    49.76±26.13
                   SimCLRv2 [8]                        SSL                      74.96        3.55   87.99±11.96    41.76±30.60
                   SwAV [6]                            SSL                      75.33        2.49   84.07±11.18    53.38±27.46
                   SupCon [37]                         SSL                      77.37        6.53   79.02± 6.27    63.43±13.17
                   timm A1 [70, 71]                    Improved Training        80.14       8.71    84.65± 7.91    57.85±19.18
                   timm A1h [70, 71]                   Improved Training        80.15      43.78    75.69± 6.49    69.05±13.40
                   timm A2 [70, 71]                    Improved Training        79.86       8.77    86.74± 9.40    55.11±23.14
                   timm A3 [70, 71]                    Improved Training        77.45       6.60    79.07± 6.64    72.55±11.76
                   timm B1k [70, 71]                   Improved Training        79.25      14.44    82.72±11.73    51.60±29.58
                   timm B2k [70, 71]                   Improved Training        79.30      14.86    83.49±12.16    49.40±30.26
                   timm C1 [70, 71]                    Improved Training        79.78      22.05    83.05±11.71    47.89±29.53
                   timm C2 [70, 71]                    Improved Training        79.97      15.91    83.42±12.01    47.52±29.54
                   timm D [70, 71]                     Improved Training        79.95       2.97    83.47± 7.04    57.26±19.27
                   TorchVision 2 [54, 67]              Improved Training        80.92      41.27    74.33± 7.45    62.90±18.86
                   Frozen Random Filters [21]          Freezing                 74.87        2.91   79.83± 7.48    66.16±17.12

Table 2. Performance summary for the 56 ResNet-50 models evaluated in our study. For each model, the table lists its unique visual
identifier used consistently throughout all figures: color denotes the training Category (e.g., Augmentations), while marker shape identifies
the specific model. We report ID metrics (Accuracy, ECE) and OOD metrics (AUROC, FPR95). OOD performance is shown as mean ±
standard deviation across all 21 OOD detection methods and eight OOD datasets. Arrows (↑/↓) indicate whether higher or lower values are
better, and all values are reported as percentages.
                                                                       13

## Page 14

Training category
                                                                                                          90
                 70                                                                                                 Baseline




                                                                             AUROCincorrect vs. OOD (%)
                                                                                                                    Adversarial Training
                 65                                                                                       85
Mean FPR95 (%)




                                                                                                                    Augmentations
                                                                                                                    SSL
                 60                                                                                       80        Improved Training
                           Training category                                                                        Freezing
                 55          Baseline                                                                     75
                             Adversarial Training
                 50
                             Augmentations                                                                70
                             SSL
                 45
                             Improved Training                                                            65
                             Freezing
                 40
                      55          60           65   70       75    80                                      75.0    77.5     80.0      82.5   85.0   87.5   90.0   92.5
                                  ID classification accuracy (%)                                                           AUROCcorrect vs. OOD (%)

Figure 9. Relationship between ID classification accuracy and                Figure 10.       Relationship between AUROCcorrect vs. OOD and
OOD detection performance, measured by the mean False Positive               AUROCincorrect vs. OOD . Each point represents one of 56 models,
Rate at 95% True Positive Rate (FPR95). Each point represents                with performance averaged across all 21 OOD detection methods
one of 56 ResNet-50 models trained with a diverse strategy. The              and four OOD categories. Color indicates the model’s training
reported FPR95 for each model is the average across all 21 OOD               category, while the marker shape uniquely identifies each model
detection methods and four OOD categories. Color indicates the               within that category.
model’s training category, while the marker shape uniquely iden-
tifies each model within that category.
                                                                             highly sensitive to classification correctness (i.e., high
                                                                             AUROCcorrect vs. incorrect ), exhibit a large performance drop
versely, for high-performing models, advanced augmenta-
                                                                             when evaluating on misclassified samples, since their scores
tions and regularization techniques reverse this relationship,
                                                                             are tightly coupled to prediction confidence, these methods
leading to a degradation in OOD performance (an increase
                                                                             risk confusing hard ID examples with true OOD data. In
in FPR95).
                                                                             contrast, methods that leverage richer feature-space repre-
   This result provides strong evidence that the complex,
                                                                             sentations, like NNGuide and GRAM, show almost no per-
non-monotonic relationship between ID accuracy and OOD
                                                                             formance gap. Their near-chance failure detection perfor-
performance is a general phenomenon, independent of the
                                                                             mance (AUROCcorrect vs. incorrect ≈ 50%) implies their OOD
evaluation metric.
                                                                             scoring is largely decoupled from the correctness of the ID
                                                                             classification.
                                                                                 In Figs. 13 and 14, we correlate the OOD detection
                                                                             performance (AUROC) with the ID classification accuracy.
Are OOD detectors merely identifying misclassified                           This analysis is performed for all ID samples, and we fur-
samples? We revisit the claim that post-hoc detectors suc-                   ther dissect the behavior by also considering the subsets
ceed largely because they separate correctly classified ID                   of correctly and incorrectly classified samples separately
samples from OOD inputs. Fig. 10 confirms the strong pos-                    (Fig. 13). The Spearman correlation coefficients (Fig. 14)
itive correlation between OOD performance on correctly                       reveal a consistently weak or statistically non-significant re-
versus incorrectly classified ID data (Spearman’s ρ = 0.88,                  lationship across all three groups (i.e., all, correct, and in-
p ≪ 0.001). It also makes the consistent performance gap                     correct), echoing the main manuscript finding. This result
visually apparent, as nearly all points lie below the x = y                  diverges from prior work [35]; while they performed a sim-
identity line, showing that performance is systematically                    ilar analysis, they observed a strong overall correlation that
higher on correctly classified samples. A notable exception                  was almost entirely driven by the performance on correctly
are models trained with MixUp or CutMix, where points                        classified ID samples (AUROCcorrect vs. OOD )
for all detectors lie on (or very close to) the identity line,                   While AUROCincorrect vs. OOD can approach random
indicating similar OOD performance when conditioning on                      chance for some model–method pairs, this is not the case
correct vs. incorrect ID predictions. However, the magni-                    for well-matched configurations (Fig. 13). For the base-
tude of this performance gap is highly method-dependent,                     line model—representing the default benchmark setting
as detailed in Fig. 11 and Fig. 12.                                          where the model-method fit is strong—every single de-
    Classification-based methods like MSP, which are                         tector performs significantly better than random guess-


                                                                        14

## Page 15

Effect                       F-value    p-value   Variance Share (%)        pronounced and unpredictable impact on a method’s effec-
 Method                        156.69   ≪ 0.001                  7.08        tiveness when the domain shift is large but structurally sim-
 Model                          77.90   ≪ 0.001                  9.69
 OOD Category                 5045.22   ≪ 0.001                 34.22        ple (e.g., ImageNet vs. MNIST).
 Model × Method                  8.47   ≪ 0.001                 21.05           This highlights a critical aspect of robustness: a method
 Model × OOD Category            9.10   ≪ 0.001                  3.39
 Method × OOD Category          39.05   ≪ 0.001                  5.30        that appears stable and effective on near-OOD data may be-
 Model × Method × OOD Group      1.16   ≪ 0.001                  8.66        come unreliable on other types of shifts, and vice versa.
 Residual                          —         —                  10.62
                                                                             For example, the high variance of some model enhance-
Table 3. Three-way ANOVA decomposition of AUROC variance
                                                                             ment methods on extreme- and synthetic-OOD data may not
across models, OOD detection methods, and OOD dataset cat-                   just stem from a sensitivity to low-level statistics, but also
egories. The table reports the F-statistic, significance level (p-           from operating on final-layer features where discriminative
value), and proportion of explained variance for each main effect            information for structurally simple OOD data might be di-
and interaction.                                                             minished. This hypothesis is supported by prior work [38],
                                                                             which showed that simpler OOD tasks are often more easily
                                                                             solved in a model’s earlier layers. The notable robustness
ing. This demonstrates a genuine ability to distinguish                      of GRAM, which leverages intermediate features, on these
true OOD samples from a model’s own most challenging                         same categories lends further support to this idea, suggest-
ID examples, proving that—while misclassifications impair                    ing that access to earlier representations is key for handling
performance—these methods are fundamentally more than                        such shifts. This underscores the necessity of benchmark-
mere failure detectors.                                                      ing on a wide range of OOD test sets to gain a complete
                                                                             picture of a method’s generalization capabilities.
Where does the AUROC variance come from? Tab. 3
lists the complete F-values, p-values, and variance shares of                Relationship with Model Calibration To investigate if
the three-way ANOVA; all main effects and interactions are                   other ID metrics are better predictors of OOD performance
significant (p ≪ 0.001). To rule out a single OOD category                   than accuracy, we analyzed the relationship between Ex-
artifact, we reran the ANOVA four times, each time omit-                     pected Calibration Error (ECE) and the OOD detection per-
ting one OOD category. Tab. 4 shows the variance shares.                     formance (Fig. 16). Globally, we observe a weak negative
Leaving out the hardest split (near-OOD) drops the OOD                       correlation (Spearman’s ρ = −0.17, p ≪ 0.001), which,
category main effect to 6.59%, but the model × method                        while more consistent than the correlation with ID classifi-
interaction increases to 33.36%, revealing model-detector                    cation accuracy (ρ = 0.04), remains a poor proxy for OOD
coupling that had been masked by uniformly low AUROC                         detection performance.
on the toughest OOD category. When far- or extreme-                              A breakdown by training category (Fig. 17) reveals that
OOD is omitted, the OOD-category term remains dominant                       this global correlation is a misleading artifact. The trend
(≈ 40%) while the interaction never falls below 16%. The                     is driven almost entirely by the adversarial training regime
residual variance is stable across all runs. Thus, no sin-                   (ρ = −0.33). At the same time, models trained with aug-
gle dataset dictates the conclusions; indeed, model–OOD                      mentations, SSL, or improved recipes show little to no cor-
method compatibility becomes more salient once the most                      relation between their calibration and OOD detection per-
challenging category is removed, underscoring the need for                   formance.
a diverse OOD benchmark.                                                         This finding underscores that OOD detection perfor-
                                                                             mance is too complex to be reliably predicted by a single
How robust are detection methods across training vari-                       ID metric, such as accuracy or calibration. While corre-
ants? The robustness of OOD detection methods also de-                       lations may appear within specific subgroups (e.g., training
pends on the nature of the distributional shift. Fig. 15 shows               strategies or OOD detection methods), such as adversarially
the OOD detection performance for each method across the                     trained models, they do not imply causality and fail to gen-
four OOD categories, revealing several key insights.                         eralize across the diverse landscape of training strategies,
    First, as expected, performance is generally lowest for                  making them unreliable as universal proxies.
the most challenging near-OOD datasets, where the seman-
tic similarity with the ID data is highest. Most methods                     Feature-Space Analysis and Robustness of OOD Detec-
struggle to achieve high AUROC scores in this setting, con-                  tion Methods. To better understand why advanced train-
firming the difficulty of this benchmark.                                    ing recipes degrade OOD detection, we analyze feature-
    Second, and more surprisingly, the variance in perfor-                   space statistics for four ResNet-50 models: the base-
mance across our 56 models is often highest for the suppos-                  line [24], MixUp [78], CutMix [77], and the TorchVi-
edly easier extreme- and synthetic-OOD categories. This                      sion 2 recipe [54, 67] (which includes MixUp, CutMix to-
suggests that the choice of training strategy can have a more                gether with additional regularization such as label smooth-


                                                                        15

## Page 16

MSP                  MLS                  EBO                     ODIN                TempScale               KLM                    GEN
                             100

                              75

                              50

                              25
AUROCincorrect vs. OOD (%)




                                         KNN                  MDS                RMDS                       SHE                    ViM                 ASH                   ReAct
                             100

                              75

                              50

                              25

                                         DICE               SCALE               NNGuide                    fDBD                  GRAM                  ATS                  GradNorm
                             100

                              75

                              50                                               Training category
                                                                                 Baseline               Augmentations   Improved Training
                              25                                                 Adversarial Training   SSL             Freezing

                                   25   50     75     25    50      75    25     50      75        25     50      75     25      50         75   25   50     75        25    50     75
                                                                                              AUROCcorrect vs. OOD (%)

Figure 11. Relation between OOD Detection Performance on correct versus incorrect ID samples for each OOD detection method. Each
point represents one of the 56 ResNet-50 models, averaged over eight OOD datasets. Color indicates the model’s training category, while
the marker shape uniquely identifies each model within that category.

                                         MSP                 MLS                  EBO                      ODIN                TempScale               KLM                    GEN
                             100

                              75

                              50

                              25

                               0
                                         KNN                 MDS                 RMDS                      SHE                     ViM                 ASH                   ReAct
                             100
AUROC (%)




                              75

                              50

                              25

                               0
                                        DICE                SCALE               NNGuide                    fDBD                  GRAM                  ATS                  GradNorm
                             100

                              75

                              50

                              25

                               0

                                             AUROCcorrect vs. incorrect               AUROCID vs. OOD                    AUROCcorrect vs. OOD                     AUROCincorrect vs. OOD

Figure 12.      Performance comparison of all 21 OOD detection methods across multiple AUROC-based evaluation metrics.
AUROCcorrect vs. incorrect evaluates failure prediction on ID data only, distinguishing between correctly and incorrectly classified samples.
The remaining metrics assess OOD detection, either across all ID samples, only correctly classified ones, or only misclassified ones. Each
boxplot shows the distribution over 56 models and four OOD categories.


ing, stronger augmentation, and EMA; see Fig. 18). We re-                                                         While MixUp, CutMix, and TorchVision 2 achieve
port five complementary metrics: total variance (spread of                                                     higher ID accuracy than the baseline, their internal repre-
embeddings), participation ratio (effective dimensionality),                                                   sentations become progressively more compressed. From
sparsity (fraction of near-zero activations), and the mean                                                     the baseline through MixUp and CutMix to TorchVision 2,
and standard deviation of feature norms.                                                                       we observe a clear progression. MixUp reduces variance


                                                                                                        16

## Page 17

MSP                    MLS                  EBO                         ODIN                    TempScale                KLM                   GEN

                            80

                            60

                            40
AUROCcorrect vs. OOD (%)




                                      KNN                    MDS                 RMDS                          SHE                        ViM                  ASH                  ReAct

                            80

                            60

                            40


                                      DICE                  SCALE             NNGuide                         fDBD                        GRAM                 ATS              GradNorm

                            80

                            60
                                                                             Training category
                            40                                                   Baseline                Augmentations         Improved Training
                                                                                 Adversarial Training    SSL                   Freezing

                                 60     70    80       60      70      80   60        70       80       60          70    80         60      70      80   60    70     80      60      70       80
                                                                                           ID classification accuracy (%)
                                                                                                             (a)

                                      MSP                    MLS                  EBO                         ODIN                    TempScale                KLM                   GEN

                            75

                            50

                            25
AUROCincorrect vs. OOD(%)




                                      KNN                    MDS                 RMDS                          SHE                        ViM                  ASH                  ReAct

                            75

                            50

                            25

                                      DICE                  SCALE             NNGuide                         fDBD                        GRAM                 ATS              GradNorm

                            75

                            50                                               Training category
                                                                                 Baseline                Augmentations         Improved Training
                            25                                                   Adversarial Training    SSL                   Freezing

                                 60     70    80       60      70      80   60        70       80       60          70    80         60      70      80   60    70     80      60      70       80
                                                                                           ID classification accuracy (%)
                                                                                                         (b)

Figure 13. Relation between ID classification accuracy and OOD detection performance. Subfigure (a) shows the AUROC for distinguish-
ing correctly classified ID samples from OOD samples, while (b) focuses on incorrectly classified ID samples. Each point represents one
of the 56 ResNet-50 models, averaged over eight OOD datasets. Color indicates the model’s training category, while the marker shape
uniquely identifies each model within that category.

           Left-Out                   Model   Method        OOD Category    Model×Method            Model×OOD Category                Method×OOD Category        3-Way Interaction    Residual
           Near                       16.66   10.02             6.78              33.87                            4.33                            6.41                11.43            10.55
           Far                         9.59    5.16            42.15              16.51                            3.44                            6.25                 8.45             8.44
           Extreme                     8.41    8.62            40.23              19.79                            2.17                            2.60                 6.46            11.67
           Synthetic                  11.49    8.04            31.49              22.49                            2.76                            4.63                 6.39            12.77


Table 4. Explained variance from leave-one-out 3-way ANOVA (factors: model, method, OOD category). Each row excludes one OOD
group and recomputes variance proportions. All reported values are in percentage and statistically significant (p ≪ 0.001).




                                                                                                         17

## Page 18

0.4                                                  0.4                                                     0.4
Spearman ρ




                                                     Spearman ρ




                                                                                                             Spearman ρ
              0.2                                                  0.2                                                     0.2

              0.0                                                  0.0                                                     0.0

             −0.2                                                 −0.2                                                    −0.2

             −0.4                                                 −0.4                                                    −0.4
                             MSP
                             MLS
                             EBO
                            ODIN

                             KLM
                             GEN
                             KNN
                             MDS
                           RMDS
                             SHE
                             ViM
                             ASH
                            ReAct
                            DICE



                           GRAM
                          SCALE



                             ATS




                                                                              MSP
                                                                              MLS
                                                                              EBO
                                                                             ODIN

                                                                              KLM
                                                                              GEN
                                                                              KNN
                                                                              MDS
                                                                            RMDS
                                                                              SHE
                                                                              ViM
                                                                              ASH
                                                                             ReAct
                                                                             DICE
                                                                           SCALE




                                                                                                                                      MSP




                                                                                                                                      MDS
                        TempScale




                         NNGuide
                            fDBD


                        GradNorm




                                                                          NNGuide
                                                                             fDBD
                                                                            GRAM
                                                                              ATS
                                                                         GradNorm




                                                                                                                                      MLS
                                                                                                                                      EBO
                                                                                                                                     ODIN

                                                                                                                                      KLM
                                                                                                                                      GEN
                                                                                                                                      KNN

                                                                                                                                    RMDS
                                                                                                                                      SHE
                                                                                                                                      ViM
                                                                                                                                      ASH
                                                                                                                                     ReAct
                                                                                                                                     DICE



                                                                                                                                    GRAM
                                                                                                                                   SCALE



                                                                                                                                      ATS
                                                                         TempScale




                                                                                                                                 TempScale




                                                                                                                                  NNGuide
                                                                                                                                     fDBD


                                                                                                                                 GradNorm
(a) Correlation between ID classification accu-     (b) Correlation between ID classification accu-          (c) Correlation between ID classification accu-
racy and AUROC.                                     racy and AUROCcorrect vs. OOD .                          racy and AUROCincorrect vs. OOD .

Figure 14. Relationship between ID classification accuracy and OOD detection performance. Spearman rank correlation (ρ) between ID
classification accuracy and OOD-detection AUROC for each detector: (a) all ID samples, (b) only correctly classified ID samples, and (c)
only misclassified ID samples. Bars are sorted and color-coded according to the method’s OOD detection category ( classification-based,
  feature-based, hybrid, intermediate-feature, gradients). Non-significant correlations (p ≥ 0.05) are shown with reduced opacity.
Statistics are computed over 56 models and four OOD categories.

                         MSP        MLS                           EBO           ODIN             TempScale                       KLM          GEN
               100

                75

                50

                25

                    0
                         KNN       MDS                    RMDS                   SHE                ViM                           ASH        ReAct
               100
 AUROC (%)




                75

                50

                25

                    0
                         DICE     SCALE             NNGuide                     fDBD                GRAM                          ATS     GradNorm
               100

                75

                50

                25

                    0
                                                  Near                    Far             Extreme                         Synthetic

Figure 15. OOD detection performance across different OOD categories (near, far, extreme, and synthetic). Each boxplot shows the
distribution over 56 models.


and feature norms while lowering the participation ratio,                            inates it (77.52%). Likewise, the penultimate-layer activa-
suggesting a lower-rank embedding. CutMix shows simi-                                tion distributions show that the characteristic pattern de-
lar but slightly stronger effects, with variance/norm reduced                        scribed by Sun et al. [61]—a near-constant mean activa-
further and sparsity moderately increased. TorchVision 2                             tion for ID samples and lower but more variable activa-
amplifies these trends: variance and norms collapse, spar-                           tions for OOD samples, which ReAct exploits via activa-
sity increases by more than two orders of magnitude, and                             tion clipping—progressively changes under MixUp, Cut-
the representation is flattened. Thus, while all three ad-                           Mix, and TorchVision 2. As a result, activation-shaping
vanced recipes achieve higher ID accuracy than the base-                             detectors such as ReAct—whose efficacy depends on clip-
line, they also progressively compress and sparsify the em-                          ping high activations—lose discriminative power, reflected
bedding space.                                                                       in a significant performance drop: FPR95 increases from
    These shifts are also mirrored in the logit and em-                              16.75% (baseline) to 40.46% (MixUp), 58.51% (CutMix),
bedding space (see Figs. 19 and 20). The max-logit                                   and 88.55% (TorchVision 2).
distributions become narrower and show increasing ID-                                   In contrast, feature-based methods (e.g., KNN, GRAM,
OOD overlap: baseline leaves a clear margin (FPR95 =                                 RMD) that leverage distances or higher-order statistics
30.62%), MixUp reduces separation (57.70%), CutMix                                   rather than specific activation characteristics, and therefore
worsens it further (70.93%), and TorchVision 2 nearly elim-                          remain comparatively robust under increasing regulariza-


                                                                                18

## Page 19

90.0                                                                                                         4.5
                                                                                                                                    4.20
                                                                   Training category                                                                                                                                                                     Orig. Baseline
                                                                                                                              4.0
                                                                     Baseline                                                                                                                                                                            MixUp




                                                                                                  Relative to TorchVision 2
                 87.5
                                                                                                                              3.5                                                                                                                        CutMix
                                                                     Adversarial Training
                                                                                                                                                                                                                           2.91                          TorchVision 2
                                                                                                                              3.0
Mean AUROC (%)




                 85.0                                                Augmentations
                                                                     SSL                                                      2.5          2.39
                                                                                                                                                                                                                                  2.22
                 82.5                                                Improved Training                                        2.0
                                                                     Freezing                                                                                                                                                            1.40                 1.48
                 80.0                                                                                                         1.5                   1.34                                                                                               1.32
                                                                                                                                                              1.00                          1.00                    1.00                        1.00                 1.071.00
                                                                                                                              1.0
                 77.5                                                                                                                                                0.49
                                                                                                                              0.5                                           0.370.42
                                                                                                                                                                                                   0.010.010.07
                 75.0                                                                                                         0.0




                                                                                                                                                                              Part. ratio



                                                                                                                                                                                                         Sparsity



                                                                                                                                                                                                                                    Mean norm



                                                                                                                                                                                                                                                                Std. norm
                                                                                                                                             Total variance
                 72.5

                 70.0
                        0.0             0.1           0.2           0.3          0.4
                                                       ECE
                                                                                                 Figure 18. Feature-space metrics for ResNet-50 baseline, MixUp,
Figure 16. Relationship between the expected calibration error                                   CutMix, and TorchVision 2 on the ImageNet test set, com-
(ECE) and the OOD detection performance. Each point represents                                   puted from penultimate-layer embeddings and shown relative to
one of 56 models, with performance averaged across all 21 OOD                                    TorchVision 2 (set to 1.0). We report five complementary statis-
detection methods and eight OOD datasets.                                                        tics: i) total variance, the trace of the covariance matrix measuring
                                                                                                 overall spread of embeddings; ii) participation ratio, the effective
                                                                                                 dimensionality of the feature space; iii) sparsity, the fraction of ac-
                                                      AUROC
                                                                                                 tivations below 10−3 ; iv) mean feature norm, the average l2 -norm
Accuracy                       0.04           0.38          0.00      0.00        0.00           of embedding vectors; and v) standard deviation of feature norms,
                                                                                                 capturing variability in embedding magnitudes.

                  ECE          -0.17          -0.33         0.00      -0.09       -0.07
                                                                                                 els [16], that originate from AugReg [59], Masked
                              Overall         AT            Aug.      SSL     Imp. Train.        Autoencoders (MAE) [25], Data-Efficient Image Trans-
                                                                                                 formers (DeiT) [64], and Sharpness-Aware Minimization
Figure 17. Spearman correlation coefficients (ρ) between OOD                                     (SAM) [10]. As with ResNet, all models are trained exclu-
detection performance (AUROC) and in-distribution (ID) perfor-
                                                                                                 sively on the ILSVRC2012 subset of ImageNet to prevent
mance metrics (accuracy and ECE), computed across all mod-
els, OOD methods, and OOD categories (Overall), and separately
                                                                                                 OOD contamination.
for adversarial training (AT), data augmentations (Aug.), self-                                      Consistent with our ResNet results, ViTs achieve higher
supervised learning (SSL), and improved training recipes (Imp.                                   ID accuracy but do not exhibit improved OOD detection
Train.). Non-significant correlations (p ≥ 0.05) are set to 0. Note                              performance (see Fig. 21). At the OOD detection method
that Spearman r reflects monotonic relationships and may not cap-                                level (Fig. 22), we again observe a clear dichotomy: feature-
ture non-monotonic trends.                                                                       based methods that rely on distances or higher-order statis-
                                                                                                 tics (e.g., KNN, RMDS, GRAM) remain comparatively ro-
                                                                                                 bust, while model-enhancement methods that depend on
tion. Altogether, these results show that although MixUp,                                        shaping specific activation patterns degrade substantially.
CutMix, and TorchVision 2 improve ID accuracy, they also                                             These findings reinforce our central claim that better ID
systematically reshape the feature space in ways that disad-                                     accuracy does not guarantee better OOD detection, even for
vantage activation-based detectors while leaving geometry-                                       more modern, higher-capacity architectures. They also sup-
based or magnitude-agnostic approaches more stable. This                                         port recent evidence [81] that many OOD detection methods
provides further evidence for our central finding that im-                                       have been implicitly tuned to CNN-style representations,
provements in ID accuracy do not necessarily yield better                                        and may overfit to the activation characteristics of ResNets
OOD detection, underscoring the strong dependency be-                                            rather than transfer robustly to other architectures.
tween the underlying model and the effectiveness of a given
OOD detection method.

C. Results on ViT
To test whether our findings extend beyond ResNets,
we also evaluate Vision Transformer (ViT-B/16) mod-


                                                                                            19

## Page 20

Orig. Baseline                               MixUp                         CutMix                   TorchVision 2
                  0.6
                                                                                                                                                      ID
                                                                                                                                                      OOD
                  0.4
Density




                                           FPR95: 30.62%                          FPR95: 57.70%                 FPR95: 70.93%                 FPR95: 77.52%
                  0.2



                  0.0
                        0            10          20     30           0       10     20     30          0   10       20   30          0   10      20       30
                                                                                            Max logit

Figure 19. Max-logit distributions for ResNet-50 baseline, MixUp, CutMix, and TorchVision 2 with ImageNet (ID) and iNaturalist (OOD).
The plots show the distribution of the maximum predicted logit for ID and OOD samples, together with the corresponding FPR95 values.

                                      Orig. Baseline                              MixUp                         CutMix                   TorchVision 2
                                                                                                                                                         ID
                  0.6                                                                                                                                    OOD
Unit Activation




                  0.4


                  0.2


                  0.0
                            0               1000             2000        0        1000          2000   0         1000         2000   0        1000         2000
                                                                                          Unit Indices

Figure 20. Distribution of per-unit activations in the penultimate layer for ImageNet (ID) and iNaturalist (OOD) across ResNet-50 baseline,
MixUp, CutMix, and TorchVision 2.



                  90.0
                                          ResNet-50
                  87.5                    ViT DeiT
                                          ViT AugReg
 Mean AUROC (%)




                  85.0
                                          ViT SAM
                  82.5                    ViT MAE

                  80.0

                  77.5

                  75.0

                  72.5

                  70.0
                                55          60         65           70       75      80
                                            ID classification accuracy (%)

Figure 21. Relationship between ID accuracy and OOD detec-
tion performance (AUROC) for 56 ResNet-50 and four ViT-B/16
models. Each point corresponds to a specific training strategy on
ImageNet (ID), with OOD performance averaged over 21 detec-
tion methods and eight OOD datasets.



                                                                                                20

## Page 21

MSP                MLS                 EBO                   ODIN                 TempScale                 KLM                   GEN

                 0.8

                 0.6

                 0.4


                             KNN                MDS                 RMDS                  SHE                      ViM                   ASH                   ReAct
Mean AUROC (%)




                 0.8

                 0.6

                 0.4


                             DICE               SCALE           NNGuide                   fDBD                     GRAM                  ATS              GradNorm

                 0.8

                 0.6

                 0.4
                                                                                   Architecture
                                                                                    ResNet-50     ViT DeiT         ViT AugReg    ViT SAM       ViT MAE
                       0.6          0.8   0.6           0.8   0.6            0.8    0.6           0.8        0.6           0.8     0.6           0.8     0.6           0.8
                                                                          ID classification accuracy (%)

Figure 22. Relationship between ID accuracy and OOD detection performance (AUROC) for ResNet-50 and ViT-B/16 across individual
OOD detection methods. Each panel shows one OOD detection method, with points corresponding to different training strategies (21
ResNet-50 and four ViT-B/16 models trained on ImageNet). OOD performance is averaged across eight OOD datasets.




                                                                                      21
