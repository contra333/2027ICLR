# Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data - page-anchored PDF text

- Source ID: `arxiv-2602.05935v1`
- arXiv ID: `2602.05935v1`
- Original PDF: `소스/Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data.pdf`
- PDF pages: 11
- Extracted with: WSL poppler `pdftotext -f N -l N -layout` on 2026-05-13T17:01:27+09:00
- Evidence note: use this file for page-anchored claims; verify equations/tables against the original PDF when exact layout matters.

## Page 1

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data


                                                   Sudeepta Mondal 1 Xinyi Mary Xie 2 Ruxiao Duan 2 Alex Wong 2 Ganesh Sundaramoorthi 1


                                                                 Abstract
                                             Existing out-of-distribution (OOD) detectors are
                                             often tuned by a separate dataset deemed OOD
                                             with respect to the training distribution of a neural
arXiv:2602.05935v1 [cs.LG] 5 Feb 2026




                                             network (NN). OOD detectors process the activa-
                                             tions of NN layers and score the output, where
                                             parameters of the detectors are determined by fit-
                                             ting to an in-distribution (training) set and the                Figure 1. Performance of OOD detectors varies with tuning set.
                                             aforementioned dataset chosen adhocly. At detec-                 Several state-of-the-art OOD detectors parameters are tuned using
                                             tor training time, this adhoc dataset may not be                 different predefined tuning datasets deemed OOD in the OpenOOD
                                                                                                              benchmark. The mean and standard deviation of the detectors’
                                             available or difficult to obtain, and even when it’s             FPR95 (y-axis) over tuning sets for each test OOD datasets (x-
                                             available, it may not be representative of actual                axis) is shown. Depending on the tuning dataset selected, there
                                             OOD data, which is often ”unknown unknowns.”                     can be significant variability in performance of the same detectors
                                             Current benchmarks may specify some left-out                     (vertical bars), so much so that the rankings of detectors can change.
                                             set from test OOD sets. We show that there can                   Results on two different networks (DenseNet101 - left and ResNet-
                                                                                                              18 - right) are shown. Our approach avoids the need for given OOD
                                             be significant variance in performance of detec-                 data, which may be difficult to obtain in practice, and variance
                                             tors based on the adhoc dataset chosen in current                associated with the choice, and still has comparable performance
                                             literature, and thus even if such a dataset can be               to given test OOD sets as OOD tuning data in state-of-the-art
                                             collected, the performance of the detector may                   benchmarks (see Table 3).
                                             be highly dependent on the choice. In this paper,
                                             we introduce and formalize the often neglected
                                             problem of tuning OOD detectors without a given                  detectors, including score-based methods (Liu et al., 2021),
                                             “OOD” dataset. To this end, we present strong                    density models (Zong et al., 2018; He et al., 2015), and
                                             baselines as an attempt to approach this problem.                learned shaping functions (Sun et al., 2021; Djurisic et al.,
                                             Furthermore, we propose a new generic approach                   2022; Mondal et al., 2025).
                                             to OOD detector tuning that does not require any                 Despite this progress, an important practical challenge has
                                             extra data other than those used to train the NN.                received surprisingly little systematic attention: how to tune
                                             We show that our approach improves over base-                    the parameters of OOD detectors. In nearly all modern OOD
                                             line methods consistently across higher-parameter                methods, performance depends critically on one or more
                                             OOD detector families, while being comparable                    parameters that control, for example, thresholds or shaping
                                             across lower-parameter families.                                 functions. Existing benchmarks and experimental protocols
                                                                                                              typically assume access to a held-out dataset deemed OOD
                                                                                                              with respect to the training data of a neural network for a
                                        1. Introduction                                                       task of interest, which we refer to as a “task NN”. This OOD
                                        When machine learning models encounter inputs that differ             tuning dataset is used to determine the parameters of a OOD
                                        in distribution from their training distribution, their predic-       detector before deployment onto the task NN for evaluation.
                                        tions can become unreliable or unsafe, making accurate                Because it is difficult to foresee the data to be encountered
                                        OOD detection crucial, especially in safety-critical systems          at test time, the choice of this tuning dataset is typically
                                        (Yang et al., 2022). As a result, a large body of research            adhoc. While this is convenient for benchmarking, it masks
                                        has focused on designing increasingly sophisticated OOD               a serious gap between experimental practice and real-world
                                                                                                              deployment.
                                           1
                                             RTX 2 Yale University. Correspondence to: Ganesh Sun-
                                        daramoorthi <ganesh.sundaramoorthi@rtx.com>.                          In practice, a suitable tuning dataset is rarely available.
                                                                                                              What constitutes “out-of-distribution” depends not only on
                                        Preprint. February 6, 2026.                                           the data domain but also on the specific trained model, mak-

                                                                                                          1

## Page 2

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

ing it difficult to predefine representative OOD samples.             1.1. Related Work
Moreover, collecting or generating realistic OOD data can
                                                                      We briefly review related work; the reader is referred to
be costly, sensitive, or operationally infeasible in many ap-
                                                                      (Yang et al., 2022) for a survey. Our focus is on post-hoc
plications. Even when some OOD samples are available,
                                                                      inference methods of OOD detection (Zhang et al., 2024),
there is no guarantee that they sufficiently cover the space
                                                                      which are applied to pre-trained models without additional
of possible OOD inputs, leading to substantial variability
                                                                      training. These methods construct scoring functions that
in the subsequent detection outcome. Figure 1 illustrates
                                                                      aim to distinguish in-distribution (ID) from OOD inputs
this issue on the OpenOOD benchmark (Zhang et al., 2024),
                                                                      using properties of the model’s outputs or intermediate rep-
where the relative ranking of state-of-the-art OOD detectors
                                                                      resentations. Prominent examples include confidence-based
changes depending on which OOD tuning set is used.
                                                                      scores (Hendrycks & Gimpel, 2018; Zhang & Xiang, 2023;
In this work, we investigate the problem of tuning OOD                Liang et al., 2020), energy-based metrics (Liu et al., 2021;
detectors without being given a dataset. To the best of our           Wu et al., 2023; Elflein et al., 2021), and distance-based mea-
knowledge, the only attempt of tuning OOD detectors with-             sures in feature space (Lee et al., 2018; Sun et al., 2022).
out an adhoc tuning set is through the use of Gaussian noise
                                                                      Early work such as MSP (Hendrycks & Gimpel, 2018) used
images (Kirichenko et al., 2020). While not in the OOD
                                                                      the maximum softmax probability as a confidence score,
detection literature, we hypothesize that additive adversarial
                                                                      while ODIN (Liang et al., 2020) improved upon this by
perturbations (Goodfellow et al., 2015) can also serve as
                                                                      introducing temperature scaling and input perturbations.
a strong baseline. We begin by benchmarking these base-
                                                                      Likelihood-based approaches using deep generative mod-
line methods as a means to generate synthetic or simulated
                                                                      els have also been explored, although raw likelihoods are
tuning sets for determining the parameters of OOD detec-
                                                                      known to be unreliable for OOD detection (Kirichenko
tors. We note that these choices, too, are largely adhoc and
                                                                      et al., 2020), motivating alternatives such as likelihood
found that they are only effective as tuning sets in limited
                                                                      ratios (Ren et al., 2019). Distance-based methods mea-
settings, e.g., Gaussian noise images were specifically used
                                                                      sure deviation from class-conditional feature distributions,
to tune OOD detectors on ImageNet (Sun et al., 2021; Xu
                                                                      for example using Mahalanobis distance (Lee et al., 2018)
et al., 2023). Hence, we address this gap by introducing
                                                                      or non-parametric nearest-neighbor schemes (Sun et al.,
a new paradigm for OOD detector tuning that eliminates
                                                                      2022). Energy-based scoring (Liu et al., 2021) provides
the need for a tuning dataset altogether. Our key insight
                                                                      a unified alternative to softmax confidence by interpreting
is that the structure of the in-distribution (used to train the
                                                                      logits through the Helmholtz free energy, and has since been
task NN) itself contains sufficient information to calibrate
                                                                      adopted by several feature-based OOD detectors.
OOD detectors, if it is exploited in the right way. This leads
to a principled and fully self-contained tuning procedure             Feature-shaping approaches to OOD detection: Several
that depends only on the data already available during the            OOD detection methods operate by transforming intermedi-
training of the neural network on which the OOD detector              ate network features prior to scoring (Sun et al., 2021; Zhao
will be deployed.                                                     et al., 2024). ReAct (Sun et al., 2021) applies an element-
                                                                      wise clipping operation to the penultimate layer activations,
Contributions.                                                        motivated by the empirical observation that OOD inputs of-
                                                                      ten induce unusually large activation values. ASH (Djurisic
                                                                      et al., 2022) performs feature shaping through sparsification,
  1. We introduce and formalize the problem of the de-                setting small activations to zero while optionally rescaling
     terming the parameters of OOD detectors without ac-              larger ones. DICE (Sun & Li, 2022) follows a similar spar-
     cess to a separate tuning dataset.                               sification principle. Unlike purely element-wise transforma-
                                                                      tions, ASH additionally applies a vector-level normalization
  2. We systematically evaluate baselines for this problem,           step before computing the detection score. In contrast, VRA
     and find no consistent best performer across standard            (Xu et al., 2023) and related work (Zhang et al.) construct
     benchmark datasets for multiple NN architectures.                element-wise shaping functions via an explicit optimization
                                                                      procedure.
  3. We present a novel OOD parameter tuning method that
     uses only in-distribution training data of the task NN           Across the broad literature, we find that OOD detectors often
     and does not require any adhoc tuning dataset.                   rely on adhoc tuning dataset(s) for determining their param-
                                                                      eters. These OOD tuning datasets are also typically paired
  4. Amongst the evaluated methods, we show that our                  with a specific target dataset, e.g., TIN, MNIST and SVHN
     approach consistently outperforms baselines in higher-           for CIFAR10 and CIFAR100, and iNaturalist, OpenImage-
     parameter OOD detector families, while all methods               O and NINCO with ImageNet-200 and ImageNet-1K. At-
     are comparable in lower-parameter families.                      tempts to depart from the reliance of these tuning datasets

                                                                  2

## Page 3

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

have been limited, and are also adhoc and specific to the                conjecture that tuning an OOD detector on such a simulated
target dataset, e.g., Gaussian noise images for ImageNet-1K              dataset(s) will learn the true OOD/ID boundary.
(Sun et al., 2021; Xu et al., 2023); we treat these as base-
                                                                         Generating Tuning Sets for OOD Detector Training: We
lines. Our work aims to present the first general framework
                                                                         now specify the procedure to tune the OOD detector. Given
for OOD tuning detectors without the need for any prede-
                                                                         an OOD detector dϕ with parameters ϕ that operates on a
fined OOD tuning dataset. Our method is broadly applicable
                                                                         task neural network, fθ , e.g., shaping of the penultimate
across task NN training datasets and OOD detectors.
                                                                         layer features. We now sample S simulated ID/OOD tuning
                                                                                M               M
                                                                         sets Did,sim,i,j and Dood,sim,i,j from the simulated ID data
2. Our Method                                                              M                                      M
                                                                         Did,sim,i and simulated OOD data Dood,sim,i      . These sets
                                                                                                 M
In this section, we present our method for tuning OOD de-                and the task networks fθi are used to train the OOD detector
                                                                                                  M
tector parameters without being given a tuning dataset. We               dϕ . To train dϕ we use Did,sim,i,j as samples of ID data and
                                                                           M
will develop and illustrate our approach for the classifica-             Dood,sim,i,j as samples of OOD data 1 . The algorithm for
tion problem. We denote the in-distribution training dataset             simulation of the simulated ID/OOD data and the simulated
Dt = {C1t , . . . , Cnt } where n is the number of categories            tuning data is shown in Algorithm 1.
and Cit is the set of data for category i.
The crux of the method lies in observation that out-of-                  Algorithm 1 Simulated ID/OOD Training and OOD Detec-
distribution (OOD) data lies any where outside of the train-             tor Validation Data from ID Classification Training Data
ing (ID) distribution within some latent feature space. Given
                                                                         Require: The network training set Dt = {C1t , . . . , Cnt }
a stationary training dataset, data samples held out from the
                                                                         Ensure: Trained networks {fθM1 , . . . , fθMN }, simulated val-
training process naturally lie “close” to the ID distribution.                                      M              M
                                                                             idation datasets Did,sim,i,j     and Dood,sim,i,j , simulated
Hence, these held-out data can be seen as hard examples. By                                          M            M
tuning the parameters of a function to discriminate between                  ID/OOD datasets Did,sim,i and Dood,sim,i
these hard (OOD) examples and those of the ID distribution,               1: for each M do
we hypothesize that the OOD detector can robustly identify                2:     for i = 1, . . . , N do
both “near” and “far” OOD examples in novel datasets. To                  3:          Randomly sample M simulated OOD categories
                                                                                         M
this end, our method aims to simulate ID and OOD examples                    to form Dood,sim,i
from the training dataset of the task NN, which allows us to              4:          Define remaining classes as simulated ID
                                                                                        M
bypass the need for a separate tuning dataset to determine                   dataset Did,sim,i
the parameters of the OOD detector.                                       5:          Train classification network fθMi on Did,sim,i
                                                                                                                                  M

                                                                          6:          for j = 1, . . . , S do
Simulated ID/OOD Datasets from ID Training Data:                          7:               Sample simulated validation ID dataset
Given a neural network architecture fθ (task NN), where θ                          M
                                                                                 Did,sim,i,j      M
                                                                                             ∼ Did,sim,i
represents the parameters of the network, we train N vari-                8:               Sample simulated validation OOD dataset
ants of fθ for the original task (classification) as follows.                      M
                                                                                 Dood,sim,i,j       M
                                                                                              ∼ Dood,sim,i
Choose M := Mood,sim categories at random from the n                      9:          end for
categories; we refer to these as the simulated OOD cate-                 10:     end for
                                                            M
gories; the data from these categories is denoted Dood,sim       .       11: end for
We refer to the remaining Mid,sim = n − Mood,sim cate-
gories as the simulated ID categories, the data is denoted
  M
Did,sim  . We train the network fθM1 for the task (classifica-           Loss Function for OOD Detector Training: As training
            M
tion) on Did,sim      . This process is repeated N times over            on a single sample i of simulated data and the network fθi
different random choices of simulated ID/OOD categories                  and a single sample j of ID/OOD tuning set is prone to
to create fθM1 , . . . , fθMN variants of the trained network; we        randomness of choice, we tune the detector to detect sim-
                 M             M                                         ulated OOD well on average across all the N simulated
also denote Did,sim,i        (Dood,sim,i ) the corresponding simu-
lated ID (OOD) data used to train fθMi . The networks fθMi               datasets/networks and S of the tuning sets, as follows. Let
will all share similarity to the network trained on all data if          L(dϕ ; fθMi , Did,sim,i,j
                                                                                        M             M
                                                                                                   , Dood,sim,i,j ) denote the loss func-
                                        M
Mood,sim is small. The dataset Dood,sim,i         will now be OOD        tion associated with the loss of dϕ (operating on fθMi ) de-
                     M
to the network fθi as this data is excluded from training the            tecting simulated OOD data from simulated ID/OOD data.
network. Furthermore, we conjecture that that data will be               This could be metrics such as negative AUROC or FPR95
near ID since it’s from the same dataset underlying dataset                  1
                                                                               Note in our choices Mood,sim is much less than Mid,sim and
collected with the same sampling process, but OOD since                  thus there is an imbalance of simulated ID and OOD tuning data.
the network has not ”seen” it or data from that category. We             To alleviate this, we use the OOD tuning and training data (which
                                                                         the network has not seen) to create balanced simulated datasets.

                                                                     3

## Page 4

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

or a combination, etc. We minimize the loss ℓ defined by              Algorithm 2 Optimal OOD Detector Parameters ϕ
                                                                                                                     M
                                                                       Require: Simulated tuning datasets Did,sim,i,j           and
                                                                             M
                 N
             1 X1X
                        S                                                  D ood,sim,i,j
ℓ(ϕ|M ) =                  L(dϕ ; fθMi , Did,sim,i,j
                                          M             M
                                                     , Dood,sim,i,j ), Ensure: Optimal detector parameters ϕ∗M
            N i=1 S j=1
                                                                        1: for each M do
                                                              (1)       2:     ϕ∗M ← arg maxϕ ℓ(ϕ | M )
where i indexes over the sampled left out categories and                                                 (Bayesian Optimization)
j indexes over the sampled simulated ID/OOD tuning sets                 3: end for
(sampled from the held-in/out data respectively). Note the
inner sum estimates an expectation over simulated tuning
                                                                       Algorithm 3 Optimal Hyper-parameter M
sets of a particular choice of ID/OOD simulated categories,
                                                                                                                M         M
and the outer sum estimates an expectation over simulated              Require: Simulated ID/OOD datasets Did,sim,i    , Dood,sim,i
                                                                                                             ∗
ID/OOD categories. These expectations reduce variance and              Ensure: Optimal hyper-parameter M
dependence on a particular simulated dataset choice. We                 1: for each M do
                                                                                          M                M                  M
perform a parameter search to minimize this loss. We choose             2:     Resample Did,sim,i,j and Dood,sim,i,j from Did,sim,i
                                                                                   M
to use Bayesian optimization (Frazier, 2018) to illustrate the             and Dood,sim,i
idea, but any other technique can be used. The algorithm for            3:     Compute validation loss ℓ̂(ϕ∗M | M )
optimization of this loss function is given in Algorithm 2.             4: end for
Choosing the Number of Simulated OOD Categories:                        5: M ∗ ← arg maxM ℓ̂(ϕ∗M | M )
Note the dependence of ℓ on the number M of simulated
OOD categories chosen. This is a hyper-parameter in our op-
timization scheme. Intuitively leaving out too few categories         3. Baseline OOD Detector Tuning Methods
as OOD will not learn enough about the possible OOD distri-
                                                                      In the experiments in the next section, we compare our new
bution, while leaving out too many would mean that fθMi dif-
                                                                      OOD detector tuning method without given OOD data to
fers too much from the network trained on all the true ID cat-
                                                                      other baselines that have been considered in OOD detection
egories. We offer an approach to choose the hyper-parameter
                                                                      papers or related literature in the past. We emphasize though
M , which we use in experiments. We optimize ℓ(ϕ|M ) in
                                                                      that there has been no formal treatment or formulation in the
ϕ to obtain ϕ∗M for each M . We then re-sample datasets
  M                M                 M              M                 literature of the OOD detector tuning problem that does not
Did,sim,i,j and Dood,sim,i,j from Did,sim,i   and Dood,sim,i ,
                     ∗                                                use held out OOD tuning data nor any systematic evaluation
and re-compute ℓ(ϕM |M ). This estimates the loss on inde-
                                                                      of these baselines. One of our contributions in this paper is
pendently generated simulated datasets following the same
                                                                      to formalize the problem, formally introduce these baselines,
distribution as the simulated ID/OOD dataset, hence it can
                                                                      and evaluate them.
be used to validate the hyper-parameter choice. Therefore,
we choose M ∗ as                                                      The first baseline that we consider is using ID validation data
                                                                      from the validation set of the network training set and to use
                                                                      Gaussian noise images as OOD validation used in (Sun et al.,
                 M ∗ = arg max ℓ̂(ϕ∗M |M ),                 (2)       2021). That is Gaussian noise images, which are created by
                            M
                                                                      sampling Gaussian noise iid at each pixel (Kirchheim et al.,
                                                                      2022). The OOD detector parameters are trained with this
where ℓ̂ indicates that (1) is re-computed on new sampled             simulated OOD dataset.
datasets, as discussed previously. The algorithm for hyper-
                                                                      The second baseline is adversarial perturbed ID data. Ad-
parameter selection is given in Algorithm 3.
                                                                      versarial perturbations are visually imperceptible additive
Final Algorithm to Produce Tuned OOD Detector: Start-                 signals optimized by gradient ascent of the task training loss
ing from the original task network training set Dt and the            function. When added to data, these perturbations shift the
trained network fθ on all of Dt , the simulated ID/OOD                sample into low-density regions of the data manifold (Song
datasets and validation sets are generated by Algorithm 1,            et al., 2018; Stutz et al., 2019). Thus, adversarial perturba-
which also produces trained networks fθMi . Candidate op-             tions simulate OOD examples as the resulting data point
timal OOD detectors’ parameters ϕ∗M are obtained using                exhibit lower likelihood under the training distribution. In
Algorithm 2. Then Algorithm 3 is used to find the optimal             our experiments, the ID data is sampled from the tuning data
hyper-parameter M ∗ (number of left-out sets). The final              of the network training dataset. The OOD data is now the
OOD detector to be used in practice is now ϕ∗M ∗ in con-              result of adversarially perturbing the ID tuning data using
junction with the original network fθ . This is shown in              fast gradient sign method (Goodfellow et al., 2015) to create
Algorithm 4.                                                          the OOD tuning data. Notice both the baselines do not rely

                                                                  4

## Page 5

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

Algorithm 4 Tuning OOD Detector Without Given OOD                     4. Experiments
Data
                                                                      4.1. Datasets and Settings
Require: The network training set Dt = {C1t , . . . , Cnt },
    task network fθ , detector dϕ                                     We evaluate our method using protocols and data from the
Ensure: Tuned parameters ϕ∗M ∗                                        OpenOOD benchmark (Zhang et al., 2023). We experiment
 1: Determine simulated ID/OOD sets and tuning sets -                 with ResNet-18(He et al., 2015), ResNet-50(He et al., 2016),
    Algorithm 1                                                       MobileNet-v2 (Sandler et al., 2018) architectures, and the
 2: Determine candidate optimal parameters ϕ∗M - Algo-                following in distribution datasets.
    rithm 2
                                                                      CIFAR-10 We conduct experiments on CIFAR-10 (Krizhevsky
 3: Determine optimal parameter set ϕ∗M ∗ - Algorithm 3
                                                                      et al., 2009), which contains n = 10 categories. To construct
                                                                      simulated OOD categories, we randomly select Mood,sim ∈
                                                                      {1, 2, 3, 4, 5} categories as simulated OOD classes and treat
on ”true” OOD tuning data (often part of OOD benchmarks)              the remaining Mid,sim = n − Mood,sim categories as simu-
or chosen expertly on a per-application basis.                        lated ID categories. For each value of Mood,sim , we re-
Note that both baselines approaches have a hyper-parameter,           peat the random category selection using ten random seeds
in the case of Gaussian noise, the hyper-parameter is the             {0, 1 . . . 9}, resulting in 5 × 10 = 50 simulated ID/OOD
                                                                                        M            M
noise level σ. This is not discussed in the literature and            splits. Using Did,sim,i  and Dood,sim,i , together with additional
                                                                                                t
a hand-picked choice is presumably chosen. Our results                samples drawn from Dood,sim,i when necessary, we construct
show significant performance variation for the level of noise         a balanced ID/OOD training dataset for the OOD detector.
chosen. Similarly, the adversarial perturbation approach
                                                                      CIFAR-100 (Krizhevsky et al., 2009) contains n = 100
also has a parameter, the norm of the perturbation to be
                                                                      categories, and we follow similar experimental setup as
added. This is also often chosen by hand.
                                                                      CIFAR-10, with the only difference being the number of
So there is a significant practical issue left out on these           simulated OOD categories. Here we select eight Mood,sim ∈
baselines in the literature, i.e., that of choosing the hyper-        {5, 10, 15, . . . 40} categories as possible number of sim-
parameter in generating the ”simulated” OOD validation                ulated OOD classes. For each value of Mood,sim , we re-
data. Simply picking a value is not representative of the             peat the random category selection using ten random seeds
performance of the method given the variance in test results.         {0, 1 . . . 9}, following the same procedure as CIFAR-10.
Moreover, in practical applications one rarely has access to
                                                                      ImageNet-1K. (Russakovsky et al., 2015) contains n =
a test set to maximize performance on the test set. There-
                                                                      1000 categories. We select Mood,sim ∈ {50, 100, 150, . . . ,
fore, we propose now a principled hyper-parameter tuning
                                                                      300} categories as simulated OOD classes. For each setting,
approach, analogous to our hyper-parameter tuning method
                                                                      we generate splits using five random seeds {0, 1, . . . , 4} and
introduced in the last section. We generate the ID and OOD
                                                                      follow the same balanced dataset construction procedure as
validation data as specified previously for several values of
                                                                      in CIFAR-10.
the hyper-parameter, which we call h (Gaussian noise level
or norm of perturbation). Then we define the loss                     ImageNet-200 (Zhang et al., 2024) follows a similar ex-
                                                                      perimental setup. We select Mood,sim ∈ {10, 20, 30, 40}
                  S
              1X                 h           h
                                                                      categories as simulated OOD classes and generate splits us-
   ℓ(ϕ|h) =         L(dϕ ; fθ , Did,sim,j , Dood,sim,j ),   (3)       ing three random seeds {0, 1, 2}. The remaining categories
              S j=1
                                                                      are treated as simulated ID classes, and balanced training
                              h             h
                                                                      datasets are constructed following the same procedure as
where we sample S datasets Did,sim,j   and Dood,sim,j from            above.
the ID data and the simulated OOD data (Gaussian noise
or adversarially perturbed images). This loss is optimized            Gaussian Noise We follow (Kirchheim et al., 2022) to gener-
via Bayesian optimization to find ϕ∗h . Similarly to hyper-           ate Gaussian noise images for CIFAR and ImageNet bench-
parameter tuning in the previous section, we resample the             marks. We sample the noise images as N (µ, σ) at each
simulated ID/OOD data and then optimize the new loss                  pixel location, and thereafter clip each pixel to be in the
ℓ(ϕ∗h |h) over h.                                                     range [0, 255]. For CIFAR benchmarks, the noise images
                                                                      generated are 32 × 32 × 3, and 224 × 224 × 3 for the
Therefore, in the comparisons of the baselines with our               ImageNet benchmarks. We fix µ = 128, and consider three
new method in the next section, we use our novel hyper-               levels for the hyperparameter σ : 32, 64 and 128.
parameter tuning approach to choose the hyper-parameter
for all methods with the same principled approach, which              Adversarial Perturbations Given an input image x from
offers a systematic approach to hyper-parameter tuning in a           CIFAR and ImageNet datasets and its ground-truth label y,
practical setting.                                                    adversarial examples are generated using the Fast Gradient

                                                                  5

## Page 6

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

                                                                   CIFAR-10 (ResNet-18)                                                                                                   CIFAR-100 (ResNet-18)
 Method               cifar100             tin                  mnist            svhn            texture       places365         Best (#)         cifar10           tin               mnist               svhn                texture         places365     Best (#)
 ReAct (Gauss.)     85.80 ± 0.81      88.30 ± 0.49       93.20 ± 3.51      88.92 ± 3.66     88.92 ± 1.15       90.51 ± 0.57         0           69.80 ± 2.25    77.28 ± 2.19     66.34 ± 4.46         83.97 ± 6.05     84.95 ± 0.72          78.28 ± 2.60      2
 ReAct (Adv.)       85.51 ± 1.02      88.06 ± 0.69       92.90 ± 3.55      88.14 ± 3.92     88.59 ± 1.43      90.67 ± 0.71          1           75.13 ± 2.74    80.87 ± 1.38     72.96 ± 5.39          83.25 ± 4.17     82.92 ± 3.79         80.20 ± 0.69      0
 ReAct (Ours)       86.42 ± 0.82     88.64 ± 0.64        93.87 ± 2.44      91.02 ± 1.38     89.13 ± 1.36      89.87 ± 1.18          5           78.16 ± 0.05    82.88 ± 0.07     77.28 ± 1.94         83.68 ± 1.28     81.57 ± 0.35          80.49 ± 0.08      4
 ASH (Gauss.)       74.19 ± 3.31      76.69 ± 4.16       87.78 ± 2.94      73.33 ± 5.45     75.33 ± 6.18       80.25 ± 2.18         0           78.42 ± 1.28    80.76 ± 0.83     78.71 ± 5.54         81.10 ± 1.60     77.36 ± 1.71          78.30 ± 1.41      1
 ASH (Adv.)         77.03 ± 2.11     79.36 ± 1.19        89.08 ± 3.29      80.98 ± 3.03     79.55 ± 1.27      79.72 ± 1.28          5           79.17 ± 0.24    80.67 ± 0.68     83.07 ± 0.40         80.00 ± 2.42     75.05 ± 1.46          77.41 ± 0.74      2
 ASH (Ours)         76.77 ± 3.12      79.20 ± 2.67       85.82 ± 3.99      76.28 ± 6.71     78.37 ± 4.84      82.36 ± 1.97          1           76.49 ± 0.22    80.29 ± 0.23     73.81 ± 1.84         84.38 ± 1.82     79.32 ± 0.47          79.80 ± 0.10      3
 KNN (Gauss.)       89.55 ± 0.12      91.34 ± 0.24       93.97 ± 0.36      92.42 ± 0.30     92.91 ± 0.24       91.52 ± 0.23         0           76.24 ± 0.27    82.91 ± 0.18     83.00 ± 1.44          84.00 ± 1.34    84.21 ± 0.80          78.66 ± 0.49      2
 KNN (Adv.)         89.55 ± 0.12      91.34 ± 0.24       93.97 ± 0.36      92.42 ± 0.30     92.91 ± 0.24      91.52 ± 0.23          0           76.24 ± 0.27    82.92 ± 0.18     83.00 ± 1.44          84.00 ± 1.34     84.20 ± 0.80         78.67 ± 0.49      0
 KNN (Ours)         89.60 ± 0.13     91.39 ± 0.25        94.04 ± 0.36      92.48 ± 0.30     92.97 ± 0.24      91.58 ± 0.23          6           76.65 ± 0.27    83.16 ± 0.18     82.64 ± 1.47         84.03 ± 1.18     83.98 ± 0.81          79.11 ± 0.48      4
 PLF (Gauss.)       82.47 ± 3.29      84.88 ± 3.26       89.78 ± 3.51      87.80 ± 4.15     87.61 ± 3.46      83.55 ± 2.24          0           65.44 ± 3.16    72.79 ± 3.60     51.75 ± 7.18         67.61 ± 11.72    74.47 ± 11.72         73.55 ± 4.91      0
 PLF (Adv.)         84.70 ± 0.23      87.02 ± 0.17       90.69 ± 3.66      87.17 ± 4.67     89.17 ± 1.39      86.55 ± 3.20          1           77.99 ± 0.64    82.42 ± 0.31     76.61 ± 3.65          81.84 ± 2.94     80.94 ± 4.19         80.62 ± 1.66      1
 PLF (Ours)         85.55 ± 0.80     87.95 ± 0.45        94.30 ± 2.38      91.59 ± 1.34     88.53 ± 0.65      88.11 ± 0.83          5           78.72 ± 0.10    83.13 ± 0.07     78.71 ± 1.70         83.18 ± 0.57     81.26 ± 0.54          80.58 ± 0.14      5
 VRA (Gauss.)       71.82 ± 9.74     72.49 ± 11.63       78.23 ± 8.32      74.71 ± 8.09     77.96 ± 8.99      71.74 ± 16.20         0           61.08 ± 3.13    68.07 ± 3.29     57.16 ± 4.97          74.79 ± 9.37     80.59 ± 2.28         70.28 ± 2.38      0
 VRA (Adv.)         85.68 ± 0.81      87.94 ± 0.83       92.50 ± 3.37      88.88 ± 3.36     89.46 ± 0.97      89.71 ± 1.75          1           75.07 ± 3.38    80.73 ± 2.17     73.30 ± 5.07          83.40 ± 3.38    83.42 ± 3.20          78.94 ± 0.91      1
 VRA (Ours)         86.22 ± 0.62     88.57 ± 0.37        94.10 ± 2.46      91.49 ± 1.06     89.71 ± 0.78      88.98 ± 0.66          5           77.40 ± 0.24    82.71 ± 0.05     76.89 ± 2.43         84.05 ± 1.52     83.37 ± 0.17          80.15 ± 0.21      5

Table 1. Benchmark results on CIFAR-10 and CIFAR-100 ID datasets using OpenOOD test OOD datasets (Zhang et al., 2024). We report
AUROC (higher is better). Bold indicates the best-performing detector–tuning method for each ID model. Best (#) denotes the number
of test OOD datasets on which a method achieves the highest AUROC. Gauss denotes the Gaussian noise baseline, Adv denotes the
adversarial noise baseline, and ours indicates our tuning method.


                                                 ImageNet-200 (ResNet-18)                                                          ImageNet-1K (ResNet-50)                                                        ImageNet-1K (MobileNet-V2)
 Method            SSB-hard        NINCO          iNaturalist       Textures     OpenImage-O      Best (#)   SSB-hard    NINCO     iNaturalist     Textures    OpenImage-O     Best (#)   SSB-hard       NINCO     iNaturalist    Textures    OpenImage-O    Best (#)
 ReAct (Gauss.)   77.85 ± 2.81   83.59 ± 2.13    94.03 ± 0.51     93.44 ± 0.15    90.88 ± 0.30       3        73.15      81.94          96.07       92.73         91.93           4           60.98       72.29       89.60         91.46         84.45         0
 ReAct (Adv.)     79.25 ± 0.33   84.97 ± 0.28    92.85 ± 0.90     91.50 ± 0.86    88.93 ± 0.16       0        72.48      80.46          92.40       89.73         89.85           0           61.91       73.17       92.91         94.12         86.81         4
 ReAct (Ours)     79.83 ± 0.02   85.17 ± 0.12    92.56 ± 0.50     90.87 ± 0.16    89.25 ± 0.25       2        73.18      81.69          94.88       91.72         91.33           1           61.69       73.51       92.16         93.69         86.56         1
 ASH (Gauss.)     77.06 ± 0.43   81.46 ± 0.76    93.31 ± 1.02     94.61 ± 0.06    89.68 ± 0.85       1        73.61      82.06          95.62       95.03         91.71           1           61.24       75.31       92.89         96.66         88.69         5
 ASH (Adv.)       79.69 ± 0.15   85.40 ± 0.57    95.06 ± 0.63     94.97 ± 0.06    91.93 ± 0.39       5        73.00      83.38          96.88       96.71         93.20           4           61.07       74.81       91.85         96.35         88.05         0
 ASH (Ours)       79.38 ± 0.29   84.51 ± 0.43    94.52 ± 0.41     93.14 ± 0.17    90.42 ± 0.09       0        73.27      83.17          96.60       96.29         92.90           0           61.16       75.08       92.33         96.50         88.34         0
 KNN (Gauss.)     77.67 ± 0.16   85.67 ± 0.13    91.71 ± 0.48     94.86 ± 0.06    88.70 ± 0.36       1        54.97      65.22          63.79       95.33         77.50           0           56.88       66.38       62.05         93.25         76.60         3
 KNN (Adv.)       77.55 ± 0.18   86.12 ± 0.10    93.24 ± 0.39     95.10 ± 0.03    89.63 ± 0.32       4        62.64      79.67          86.37       97.07         87.02           3           56.88       66.38       62.04         93.25         76.59         1
 KNN (Ours)       77.65 ± 0.16   85.52 ± 0.13    91.23 ± 0.51     94.78 ± 0.08    88.45 ± 0.38       0        63.09      79.87          85.99       96.91         86.90           2           55.61       65.65       63.32         94.84         77.25         1
 PLF (Gauss.)     74.74 ± 0.37   80.45 ± 0.19    84.30 ± 1.38     85.38 ± 0.57    82.64 ± 0.80       0        64.50      77.97          93.72       95.36         91.07           2           60.19       72.99       89.40         92.50         85.05         1
 PLF (Adv.)       76.26 ± 0.30   79.61 ± 0.46    85.60 ± 0.80     78.23 ± 0.31    82.00 ± 0.68       0        71.89      78.11          91.56       90.16         89.06           0           59.95       72.15       85.79         88.98         82.27         0
 PLF (Ours)       80.31 ± 0.03   85.07 ± 0.17    92.58 ± 0.53     88.70 ± 0.17    88.61 ± 0.32       5        73.81      82.34          94.43       90.28         90.98           3           62.82       74.90       90.11         90.98         85.31         4
 VRA (Gauss.)     73.73 ± 8.46   81.57 ± 6.39    92.26 ± 1.82     94.31 ± 0.36    88.65 ± 3.65       1        64.44      71.15          92.30       96.03         87.72           2           61.33       74.04       89.87         93.69         86.17         3
 VRA (Adv.)       79.44 ± 0.41   85.05 ± 0.49    92.83 ± 0.19     91.57 ± 2.43    89.64 ± 0.93       1        72.61      79.35          89.97       86.00         87.99           1           61.42       72.84       88.83         93.37         85.43         0
 VRA (Ours)       79.28 ± 0.04   85.37 ± 0.13    93.37 ± 0.48     93.59 ± 0.14    90.73 ± 0.32       3        70.74      79.85          92.16       92.82         90.57           2           61.84       74.55       89.51         93.36         86.02         2

Table 2. Benchmark results on ImageNet-200 and ImageNet-1k ID datasets using OpenOOD test OOD datasets (Zhang et al., 2024). The
models are ResNet-18 for ImageNet-200; ResNet-50 and MobileNet-V2 for ImageNet-1k. We report AUROC (higher is better). Bold
indicates the best-performing detector–tuning method for each ID model. Best (#) denotes the number of test OOD datasets on which a
method achieves the highest AUROC. Gauss denotes the Gaussian noise baseline, Adv denotes the adversarial noise baseline, and ours
indicates our tuning method.



Sign Method (FGSM) (Goodfellow et al., 2015):                                                                                        ASH performs sparsification based on a percentile parameter
                                                                                                                                    that determines how many activations are suppressed. KNN
           xadv = x + ϵ · sign ∇x L(x, y) ,                                                                             (4)          estimates OOD scores using distances in feature space and
                                                                                                                                     requires tuning the number of nearest neighbors used for
where L denotes the Binary Cross-Entropy (BCE) loss and
                                                                                                                                     scoring. On the other hand, VRA and PLF are higher param-
ϵ controls the perturbation magnitude. We consider three
                                                                                                                                     eter piecewise linear family of detectors. VRA (Xu et al.,
levels for the hyperparameter ϵ: 0.005, 0.01, and 0.1. After
                                                                                                                                     2023) is parameterized by three values that control the sup-
applying the perturbation, the adversarial images are clipped
                                                                                                                                     pression of low activations, the amplification of mid-range
to remain within the valid pixel range.
                                                                                                                                     activations, and the clipping of high activations. PLF is char-
We choose to optimize AUROC as our loss function, as is                                                                              acterized by seven parameters which denote a more general
standard for OOD tuning in the literature.                                                                                           piecewise linear family of detectors (Mondal et al., 2025).
                                                                                                                                     More details of the tuning parameters and their ranges used
4.2. OOD Detectors Trained                                                                                                           for optimization are provided in the Appendix A.

We evaluate our tuning framework on a representative set of    4.3. Results
post-hoc OOD detectors: ReAct (Sun et al., 2021), ASH (Djurisic
et al., 2022), KNN (Sun et al., 2022), VRA (Xu et al., 2023),  We report the results of our tuning method on CIFAR 10/100
and PLF (Mondal et al., 2025). These methods are all lead-     benchmarks (Table 1) and the Imagenet 200/1k benchmarks
ing state-of-the-art post-hoc inference methods (Zhang et al., (Table 2) using the OOD detectors mentioned in Section 4.2.
2024) applied to a particular layer of a trained neural net-   We compare our tuning strategy against Gaussian-noise and
work, and operate by transforming or re-weighting inter-       adversarial-perturbation tuning, which serve as baseline
mediate feature representations before computing an OOD        approaches.
score. Of these, ASH, ReAct and KNN have one tuning
                                                               We notice that our approach generally performs better than
parameter each. ReAct applies element-wise activation clip-
                                                               the other tuning methods on the higher parameter OOD
ping and is controlled by a single threshold parameter, while

                                                                                                                              6

## Page 7

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

                     CIFAR-10 (ResNet-18)                 CIFAR-100 (ResNet-18)               ImageNet-200 (ResNet-18)           ImageNet-1K (ResNet-50)
Method        Near-OOD     Far-OOD      ID Acc.    Near-OOD      Far-OOD       ID Acc.   Near-OOD     Far-OOD       ID Acc.     Near-OOD Far-OOD ID Acc.
ReAct         87.11±0.61 90.42±1.41 95.06±0.30 80.77±0.05 80.39±0.49 77.25±0.10 81.87±0.98 92.31±0.56              86.37±0.08    77.38     93.67    76.18
ReAct (Ours) 87.53 ± 0.73 90.97 ± 1.59 95.06±0.30 80.52 ± 0.06 80.76 ± 0.91 77.25±0.10 82.50 ± 0.07 90.89 ± 0.30   86.37±0.08    77.44     92.64    76.18
ASH           75.27±1.04 78.49±2.58 95.06±0.30 78.20±0.15 80.58±0.66 77.25±0.10 82.38±0.19 93.90±0.27              86.37±0.08    78.17     95.74    76.18
ASH (ours)   77.99 ± 2.90 80.71 ± 4.38 95.06±0.30 78.39 ± 0.23 79.33 ± 1.06 77.25±0.10 81.95 ± 0.36 92.69 ± 0.22   86.37±0.08    78.22     95.26    76.18
KNN           90.64±0.20 92.96±0.14 95.06±0.30 80.18±0.15 82.40±0.17 77.25±0.10 81.57±0.17 93.16±0.22              86.37±0.08    71.10     90.18    76.18
KNN (ours)   90.50 ± 0.19 92.77 ± 0.28 95.06±0.30 79.91 ± 0.23 82.44 ± 0.99 77.25±0.10 81.59±0.15 91.49±0.32       86.37±0.08    71.48     89.93    76.18
PLF           89.74±0.53 92.66±0.72 95.06±0.30 80.82±0.13 80.69±0.94 77.25±0.10 82.23±0.09 93.23±0.19              86.37±0.08    76.82     95.27    76.18
PLF (ours)    86.75±0.63 90.63±1.30 95.06±0.30 80.93±0.09 80.43± 0.74 77.25±0.10 82.69±0.10 89.96±0.34             86.37±0.08    78.08     91.90    76.18
VRA           88.91±0.68 92.08±0.67 95.06±0.30 80.63±0.41 81.39±0.83 77.25±0.10 82.22±0.15 93.53±0.23              86.37±0.08    77.75     94.89    76.18
VRA (ours)    87.40±0.50 91.57±1.24 95.06±0.30 80.06±0.15 81.12±1.08 77.25±0.10 82.32±0.06 92.56±0.22              86.37±0.08    75.29     91.85    76.18
Table 3. Performance comparison of SoA OOD detectors tuned with our method versus given test OOD data used in the Open OOD
benchmark. We compare the performance of OOD detectors tuned with our approach to the detectors tuned with given OOD data,
specifically selected in OpenOOD benchmarks to have favorable properties for tuning. Detectors tuned by our method using only
in-distribution data sometimes out-performs the same detectors tuned with given OOD data, and in the cases it doesn’t, our method
is generally comparable in performance. OOD detection results for ReAct, ASH, KNN, VRA and PLF across CIFAR and ImageNet
benchmarks in OpenOOD are shown. In OpenOOD, all the detectors are tuned using validation subsets from TinyImageNet for CIFAR
10/100 and OpenImage-O for ImageNet 200/1k. “Ours” refers to our proposed tuning method for each detector. Near-OOD and Far-OOD
columns correspond to average AUROC of the detection on the identified near and far OOD datasets of each benchmark as reported in
OpenOOD.



detectors - VRA and PLF. For the CIFAR benchmarks our                             the hyperparameters of the OOD detectors are tuned. These
approach achieves the best AUROC on 10 datasets each                              tuning sets are specifically chosen for favorable properties
for VRA and PLF, substantially improving over Gaussian                            as tuning sets. Even though our tuning approach only uses
and adversarial tuning. For PLF, our method achieves the                          in-distribution data, the results in Table 3 show that OOD
best AUROC on all 5 OOD datasets for ImageNet-200                                 detectors tuned with our method sometimes out performs
(ResNet-18) and ImageNet-1K (ResNet-50), and on 4 out                             the detectors tuned with given OOD data, and in cases that
of 5 datasets for ImageNet-1K (MobileNet-V2). These re-                           it doesn’t the results are generally comparable. This sug-
sults indicate that our tuning strategy can be particularly                       gests our method can be used when no OOD tuning data is
effective for detectors with more complex parameteriza-                           available, without an appreciable performance loss.
tions, whereby the detectors may overfit to Gaussian noise
or adversarial perturbations as OOD tuning data, and can                          4.4. Ablation on number of simulated OOD categories
potentially fail to generalize under test settings.
                                                                                  We present an ablation on the number of simulated OOD cat-
For other lower complexity shaping approaches such as Re-                         egories for our experiments on ImageNet-200 and ImageNet-
Act and ASH, our approach remains competitive and often                           1k, in Tables 4 and 5, respectively. For most methods,
achieves best or second-best performance across datasets. In                      AUROC varies only slightly across different holdout sizes.
particular, on CIFAR benchmarks, our method achieves the                          Minor fluctuations are method-dependent, but overall the
highest number of wins for ReAct compared to its Gaussian                         results demonstrate robustness of our tuning approach over a
and adversarial counterparts, and remains competitive for                         range of simulated OOD categories for both the benchmarks.
ASH, often matching or improving upon adversarial tun-                            The observations are similar for the CIFAR benchmarks, and
ing across datasets. The observations are similar for KNN,                        are therefore omitted for brevity.
which relies primarily on fixed feature geometry and has
limited tunable capacity. On CIFAR-10, our method consis-
tently ranks best with KNN, while on CIFAR-100 and Ima-
                                                                                  5. Limitations
geNet benchmarks the results are mixed, with our approach                         Our method involves training N variants of the task neural
frequently ranking first or second and often outperforming                        network; while manageable with standard computational
Gaussian noise tuning. Overall, these results suggest that                        hardware (e.g., workstations equipped with Nvidia RTX
our OOD detector tuning approach yields the largest and                           3090 GPUs) for the classification task, this can be com-
most consistent improvements for higher-capacity detectors                        putationally expensive for tasks that involve dense predic-
such as PLF and VRA, while providing more modest and                              tions, e.g., segmentation, as well as large language models.
less uniform gains for lower-capacity methods.                                    Furthermore, we tested convolutional neural network archi-
We also compare our results with those reported in OpenOOD                        tectures, where each experiment took roughly 30 minutes
benchmark (Zhang et al., 2024) in Table 3. OpenOOD spec-                          on our workstation. We note that these architectures are
ifies a given OOD tuning set from TinyImageNet for CIFAR                          comparably less expensive than transformer-based models.
10/100, and OpenImage-O for ImageNet 200/1k on which                              Future work will aim to address this limitation, and extend
                                                                                  our formulations for other tasks.

                                                                           7

## Page 8

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

                                           ImageNet-200 (ResNet-18)
 Method   Holdout    SSB-hard        NINCO         iNaturalist     Textures     OpenImage-O    AVG
                                                                                                              https://arxiv.org/abs/1807.02811.
 ReAct      10      79.83 ± 0.02   85.17 ± 0.12   92.56 ± 0.50   90.87 ± 0.16   89.25 ± 0.25   87.536
            20
            30
                    79.83 ± 0.02
                    85.17 ± 0.11
                                   85.17 ± 0.11
                                   82.50 ± 0.05
                                                  92.55 ± 0.50
                                                  90.79 ± 0.16
                                                                 90.79 ± 0.16
                                                                 89.23 ± 0.26
                                                                                89.23 ± 0.26
                                                                                90.86 ± 0.21
                                                                                               87.514
                                                                                               87.710
                                                                                                            Goodfellow, I., Shlens, J., and Szegedy, C. Explaining
            40      79.83 ± 0.02   85.17 ± 0.11   92.55 ± 0.50   90.79 ± 0.16   89.23 ± 0.26   87.514         and harnessing adversarial examples. In International
 ASH        10      79.38 ± 0.29   84.51 ± 0.43   94.52 ± 0.41   93.14 ± 0.17   90.42 ± 0.09   88.394
            20      79.74 ± 0.25   85.18 ± 0.51   94.91 ± 0.46   93.86 ± 0.10   91.21 ± 0.14   88.980        Conference on Learning Representations, 2015. URL
            30      79.86 ± 0.21   85.48 ± 0.58   95.04 ± 0.46   94.33 ± 0.08   91.59 ± 0.18   89.260
            40      79.86 ± 0.21   85.48 ± 0.58   95.04 ± 0.46   94.33 ± 0.08   91.59 ± 0.18   89.260         http://arxiv.org/abs/1412.6572.
 KNN        10      77.65 ± 0.16   85.52 ± 0.13   91.23 ± 0.51   94.78 ± 0.08   88.45 ± 0.38   87.526
            20      77.65 ± 0.16   85.52 ± 0.13   91.23 ± 0.51   94.78 ± 0.08   88.45 ± 0.38   87.526
            30
            40
                    77.42 ± 0.20
                    77.54 ± 0.19
                                   86.17 ± 0.10
                                   86.13 ± 0.10
                                                  93.58 ± 0.37
                                                  93.28 ± 0.38
                                                                 95.17 ± 0.02
                                                                 95.11 ± 0.03
                                                                                89.88 ± 0.31
                                                                                89.66 ± 0.32
                                                                                               88.444
                                                                                               88.344
                                                                                                            He, K., Zhang, X., Ren, S., and Sun, J. Deep residual
 PLF        10      80.31 ± 0.03   85.07 ± 0.17   92.58 ± 0.53   88.70 ± 0.17   88.61 ± 0.32   87.054         learning for image recognition, 2015. URL https://
            20      79.99 ± 0.02   85.10 ± 0.11   92.46 ± 0.48   89.54 ± 0.17   88.89 ± 0.24   87.196
            30      79.85 ± 0.10   85.47 ± 0.14   93.37 ± 0.38   91.29 ± 0.17   89.89 ± 0.24   87.974         arxiv.org/abs/1512.03385.
            40      80.05 ± 0.01   85.57 ± 0.08   93.02 ± 0.45   91.01 ± 0.16   89.69 ± 0.18   87.868
 VRA        10      79.25 ± 0.06   85.03 ± 0.16   92.84 ± 0.53   92.56 ± 0.15   90.06 ± 0.36   87.948
            20      79.26 ± 0.04   85.40 ± 0.13   93.40 ± 0.48   93.48 ± 0.14   90.67 ± 0.33   88.442       He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings
            30      79.22 ± 0.05   85.30 ± 0.14   93.27 ± 0.49   93.26 ± 0.13   90.51 ± 0.33   88.312
            40      79.28 ± 0.04   85.37 ± 0.13   93.37 ± 0.48   93.59 ± 0.14   90.73 ± 0.32   88.468         in deep residual networks. In Leibe, B., Matas, J., Sebe,
Table 4. AUROC on ImageNet-200 (ResNet-18) under Different                                                    N., and Welling, M. (eds.), Computer Vision – ECCV
Holdout-Class Settings                                                                                        2016, pp. 630–645, Cham, 2016. Springer International
                                                                                                              Publishing.
                                                                                                            Hendrycks, D. and Gimpel, K. A baseline for detecting
6. Conclusion                                                                                                 misclassified and out-of-distribution examples in neural
                                                                                                              networks, 2018. URL https://arxiv.org/abs/
We showed through extensive benchmarking that OOD de-
                                                                                                              1610.02136.
tectors, especially larger parameter detectors, can exhibit
large variance across OOD tuning sets, in particular those                                                  Kirchheim, K., Filax, M., and Ortmeier, F. Pytorch-ood: A
chosen in existing benchmarks. This makes the OOD detec-                                                      library for out-of-distribution detection based on pytorch.
tors sensitive to the predefined tuning set. Moreover, obtain-                                                In Proceedings of the IEEE/CVF Conference on Com-
ing such a tuning set in practice may be difficult. We formu-                                                 puter Vision and Pattern Recognition, pp. 4351–4360,
lated the new problem of tuning OOD parameters without                                                        2022.
a given (real) OOD tuning data. Compared to strong base-
lines that OOD practioners might use to tune OOD detectors                                                  Kirichenko, P., Izmailov, P., and Wilson, A. G. Why normal-
without collected tuning data, our new method out-performs                                                    izing flows fail to detect out-of-distribution data. Ad-
on higher parameter families of detectors and are compara-                                                    vances in neural information processing systems, 33:
ble in lower parameter detectors. This was demonstrated                                                       20578–20589, 2020.
on numerous datasets and OOD detectors. Our method                                                          Krizhevsky, A., Hinton, G., et al. Learning multiple layers
thus provides a path to deploying reliable OOD detectors                                                      of features from tiny images. 2009.
in practical applications. Furthermore, this work suggests
that further research in tuning OOD detectors without given                                                 Lee, K., Lee, K., Lee, H., and Shin, J. A simple unified
OOD data is needed.                                                                                           framework for detecting out-of-distribution samples and
                                                                                                              adversarial attacks. Advances in neural information pro-
                                                                                                              cessing systems, 31, 2018.
Impact Statement
                                                                                                            Liang, S., Li, Y., and Srikant, R. Enhancing the relia-
This paper presents work whose goal is to advance the field
                                                                                                              bility of out-of-distribution image detection in neural
of Machine Learning. There are many potential societal
                                                                                                              networks, 2020. URL https://arxiv.org/abs/
consequences of our work, none which we feel must be
                                                                                                              1706.02690.
specifically highlighted here.
                                                                                                            Liu, W., Wang, X., Owens, J. D., and Li, Y. Energy-based
References                                                                                                    out-of-distribution detection, 2021. URL https://
                                                                                                              arxiv.org/abs/2010.03759.
Djurisic, A., Bozanic, N., Ashok, A., and Liu, R. Extremely
  simple activation shaping for out-of-distribution detec-                                                  Mondal, S., Jiang, Z., and Sundaramoorthi, G. A variational
  tion. 2022. URL https://arxiv.org/abs/2209.                                                                information theoretic approach to out-of-distribution de-
  09858.                                                                                                     tection, 2025. URL https://arxiv.org/abs/
                                                                                                             2506.14194.
Elflein, S., Charpentier, B., Zügner, D., and Günnemann,
  S. On out-of-distribution detection with energy-based                                                     Ren, J., Liu, P. J., Fertig, E., Snoek, J., Poplin, R., Depristo,
  models. arXiv preprint arXiv:2107.08785, 2021.                                                              M., Dillon, J., and Lakshminarayanan, B. Likelihood ra-
                                                                                                              tios for out-of-distribution detection. Advances in neural
Frazier, P. I. A tutorial on bayesian optimization, 2018. URL                                                 information processing systems, 32, 2019.

                                                                                                        8

## Page 9

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

                                                                           ImageNet-1K
                                        ResNet-50                                                                MobileNet-V2
 Method   Holdout   SSB-hard   NINCO   iNaturalist   Textures   OpenImage-O     AVG      SSB-hard   NINCO   iNaturalist   Textures   OpenImage-O   AVG
 ReAct     50        73.15     81.76     95.05        91.87        91.44       86.66      61.62     73.49     91.88        93.49        86.38      81.37
           100       72.48     81.39     94.23        91.14        90.92       86.14      61.54     73.41     91.53        93.21        86.12      81.16
           150       73.18     81.69     94.88        91.72        91.33       86.56      61.63     73.50     91.90        93.50        86.39      81.38
           200       73.61     81.65     94.78        91.63        91.27       86.50      61.69     73.51     92.16        93.69        86.56      81.52
           250       73.00     81.55     94.56        91.43        91.13       86.36      61.69     73.51     92.18        93.71        86.58      81.54
           300       73.27     81.76     95.04        91.87        91.43       86.66      61.15     75.01     92.21        96.46        88.27      82.62
 ASH       50        73.15     83.30     96.73        96.48        93.04       88.54      66.99     78.17     92.09        95.73        90.41      84.68
           100       73.20     83.26     96.69        96.41        93.00       88.51      61.04     74.59     91.58        96.26        87.88      82.27
           150       73.05     83.37     96.83        96.63        93.17       88.61      61.19     75.17     92.53        96.56        88.46      82.78
           200       73.11     83.31     96.76        96.54        93.09       88.56      61.16     75.08     92.33        96.50        88.34      82.68
           250       73.27     83.17     96.60        96.29        92.90       88.45      61.15     75.05     92.30        96.49        88.33      82.67
           300       73.05     83.34     96.81        96.61        93.16       88.60      61.73     73.50     92.33        93.81        86.66      81.61
 KNN       50        61.86     79.27     86.78        97.27        87.10       82.46      56.68     66.28     62.30        93.57        76.73      71.11
           100       61.74     79.20     86.82        97.30        87.10       82.43      55.98     65.32     62.93        95.15        77.49      71.38
           150       63.09     79.87     85.99        96.91        86.90       82.55      56.89     66.39     62.05        93.25        76.60      71.03
           200       63.09     79.87     85.99        96.91        86.90       82.55      55.20     65.39     63.63        95.17        77.41      71.36
           250       63.09     79.87     85.99        96.91        86.90       82.55      56.68     66.28     62.30        93.57        76.73      71.11
           300       63.09     79.87     85.99        96.91        86.90       82.55      55.61     65.65     63.32        94.84        77.25      71.33
 PLF       50        73.63     82.90     95.40        92.45        91.98       87.27      60.68     73.04     87.35        89.51        83.25      78.77
           100       73.98     82.65     94.87        91.19        91.45       86.83      62.88     75.13     90.75        91.17        85.70      81.12
           150       73.81     82.34     94.43        90.28        90.98       86.37      62.79     73.74     89.50        88.80        84.01      79.77
           200       73.80     81.47     93.03        87.55        89.59       85.09      62.69     74.46     90.27        90.48        85.12      80.61
           250       73.85     82.16     94.10        89.71        90.71       86.11      63.27     74.29     90.63        87.98        84.06      80.05
           300       73.80     81.47     93.03        87.55        89.59       85.09      62.82     74.90     90.11        90.98        85.31      80.82
 VRA       50        70.33     80.28     92.98        94.00        91.23       85.76      61.84     74.79     89.92        93.73        86.41      81.34
           100       70.34     80.22     92.88        93.87        91.15       85.69      61.88     74.00     88.99        92.35        85.24      80.49
           150       70.74     79.85     92.16        92.82        90.57       85.23      61.96     73.42     88.51        91.06        84.35      79.86
           200       70.70     79.85     92.18        92.86        90.58       85.24      61.84     74.55     89.51        93.36        86.02      81.05
           250       71.05     79.91     92.14        92.62        90.53       85.25      61.97     73.43     88.53        91.08        84.37      79.87
           300       70.94     79.87     92.12        92.65        90.52       85.22      61.96     73.38     88.44        91.00        84.28      79.81
                                Table 5. AUROC on ImageNet-1K under different holdout-class settings.



Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S.,                   Sun, Y., Ming, Y., Zhu, X., and Li, Y. Out-of-distribution de-
  Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein,                        tection with deep nearest neighbors. In International Con-
  M., Berg, A. C., and Fei-Fei, L. Imagenet large scale                          ference on Machine Learning, pp. 20827–20840. PMLR,
  visual recognition challenge, 2015. URL https://                               2022.
  arxiv.org/abs/1409.0575.
                                                                               Wu, Q., Chen, Y., Yang, C., and Yan, J. Energy-based out-
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and                            of-distribution detection for graph neural networks. arXiv
  Chen, L.-C. Mobilenetv2: Inverted residuals and linear                        preprint arXiv:2302.02914, 2023.
  bottlenecks. In 2018 IEEE/CVF Conference on Computer
  Vision and Pattern Recognition, pp. 4510–4520, 2018.                         Xu, M., Lian, Z., Liu, B., and Tao, J. Vra: Variational
  doi: 10.1109/CVPR.2018.00474.                                                  rectified activation for out-of-distribution detection, 2023.
                                                                                 URL https://arxiv.org/abs/2302.11716.
Song, Y., Kim, T., Nowozin, S., Ermon, S., and Kushman, N.
  Pixeldefend: Leveraging generative models to understand                      Yang, J., Zhou, K., Li, Y., and Liu, Z. Generalized out-of-
  and defend against adversarial examples. In International                      distribution detection: A survey, 2022.
  Conference on Learning Representations, 2018.
                                                                               Zhang, J., Yang, J., Wang, P., Wang, H., Lin, Y., Zhang,
Stutz, D., Hein, M., and Schiele, B. Disentangling adversar-                     H., Sun, Y., Du, X., Li, Y., Liu, Z., et al. Openood v1.
  ial robustness and generalization. In Proceedings of the                       5: Enhanced benchmark for out-of-distribution detection.
  IEEE/CVF conference on computer vision and pattern                             arXiv preprint arXiv:2306.09301, 2023.
  recognition, pp. 6976–6987, 2019.
                                                                               Zhang, J., Yang, J., Wang, P., Wang, H., Lin, Y., Zhang,
Sun, Y. and Li, Y. Dice: Leveraging sparsification for out-                      H., Sun, Y., Du, X., Li, Y., Liu, Z., Chen, Y., and Li,
  of-distribution detection. In European Conference on                           H. Openood v1.5: Enhanced benchmark for out-of-
  Computer Vision, 2022.                                                         distribution detection, 2024. URL https://arxiv.
                                                                                 org/abs/2306.09301.
Sun, Y., Guo, C., and Li, Y. React: Out-of-distribution
  detection with rectified activations, 2021. URL https:                       Zhang, Y., Lu, J., Peng, B., Fang, Z., and Cheung, Y.-m.
  //arxiv.org/abs/2111.12797.                                                    Learning to shape in-distribution feature space for out-

                                                                           9

## Page 10

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

  of-distribution detection. In The Thirty-eighth Annual
  Conference on Neural Information Processing Systems.
Zhang, Z. and Xiang, X. Decoupling maxlogit for out-of-
  distribution detection. In 2023 IEEE/CVF Conference
  on Computer Vision and Pattern Recognition (CVPR),
  pp. 3388–3397, 2023. doi: 10.1109/CVPR52729.2023.
  00330.

Zhao, Q., Xu, M., Gupta, K., Asthana, A., Zheng, L.,
  and Gould, S. Towards optimal feature-shaping meth-
  ods for out-of-distribution detection. arXiv preprint
  arXiv:2402.00865, 2024.

Zong, B., Song, Q., Min, M. R., Cheng, W., Lumezanu, C.,
  ki Cho, D., and Chen, H. Deep autoencoding gaussian
  mixture model for unsupervised anomaly detection. In
  International Conference on Learning Representations,
  2018. URL https://api.semanticscholar.
  org/CorpusID:51805340.




                                                           10

## Page 11

Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

A. Detector Definitions and Parameter Selection
ReAct.    ReAct (Sun et al., 2021) clips penultimate-layer feature activations element-wise:
                                                    ReAct(z)i = min(zi , τ ).
The single parameter τ is optimized by Bayesian optimization. The search interval is defined from the range of activations
from the penultimate layer obtained with the ID data, similar to the setup of (Sun et al., 2021).

ASH. ASH (Djurisic et al., 2022) sparsifies intermediate activations by retaining the top-k elements and suppressing the
rest, where k is determined by a percentile parameter p. The retained activations are set to a constant proportional to the
activation energy. The only tunable parameter is p, which is optimized by Bayesian optimization over a valid percentile
range on simulated ID/OOD splits. We use the ASH-B variant of ASH in our benchmarks.

VRA.     VRA+ (Xu et al., 2023) applies a piecewise shaping function:
                                                      
                                                      0,
                                                                z < α,
                                           VRA(z) = z + γ, α ≤ z ≤ β,
                                                      
                                                        β,       z > β.
                                                      

Rather than optimizing α and β directly, we parameterize them by quantiles of the empirical ID feature distribution:
                                                      −1                       −1
                                                α = FID  (ηα ),          β = FID  (ηβ ).
The optimization variables are (ηα , u, γ) with
                                           ηα ∈ [0.1, 0.8],      u ∈ [0, 1],       γ ∈ [0, 5],
and ηβ = ηα + δ, where
                             δ = δmin + u(δmax − δmin ),           δmin = 0.10,        δmax = 0.99 − ηα ,
ensuring ηβ > ηα . Bayesian optimization selects (ηα , ηβ , γ) on simulated ID/OOD splits.

PLF. PLF applies a seven-parameter piecewise linear shaping function to features (Mondal et al., 2025), where f (·)
consists of three linear segments defined by breakpoints x1 , x2 and slopes m1 , m2 with vertical offsets (ystart , yend , y1 ). The
breakpoints are defined by quantiles of the absolute ID features:
                                                     −1                         −1
                                              x1 = F|ZID |
                                                           (q1 ),        x2 = F|ZID |
                                                                                      (q2 ),

with q2 = q1 + δ. The optimization variables are
                                                 (ystart , yend , ∆y, q1 , u, m1 , m2 ),
with
                                         ystart ∈ [−5, 0],    yend ∈ [0, 5],       ∆y ∈ [0, 5],
                                 q1 ∈ [0.1, 0.8],    u ∈ [0, 1],     m1 ∈ [0, 5],          m2 ∈ [−5, 5],
and y1 = yend + ∆y. The upper quantile is defined as
                              δ = δmin + u(δmax − δmin ),          δmin = 0.10,        δmax = 0.99 − q1 ,
so that q2 = q1 + δ and q2 > q1 . Parameters are selected by Bayesian optimization on simulated ID/OOD splits.

KNN. KNN (Sun et al., 2022) computes OOD scores using the negative distance to the K-th nearest neighbor in normalized
feature space. The only tunable parameter is K. We search over integer K ∈ [1, Kmax ] using Bayesian optimization, where
distances are precomputed from a feature index built on ID training data. We choose Kmax = 500

Optimization Protocol. For all detectors, parameters are selected using Bayesian optimization with a Gaussian process
surrogate on simulated ID/OOD splits. The objective is to maximize AUROC computed from energy-based scores after
applying the corresponding feature transformation.

                                                                    11
