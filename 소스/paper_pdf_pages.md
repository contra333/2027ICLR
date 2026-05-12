# PDF Page Text: arXiv 2602.16642v3

Source PDF: `C:\Users\User\Desktop\Wiki\20_원본자료\논문\arxiv-2602.16642\versions\v3\paper.pdf`
Extraction: `pdftotext -layout -f N -l N` via WSL poppler.
Evidence role: primary PDF page context. Use this when a GPT Project needs page-level anchors, surrounding context, or PDF wording checks. Layout artifacts may remain, so verify equation/table details against `paper.pdf` or TeX source when exactness matters.

## Page 001

```text
                                         Published as a conference paper at ICLR 2026




                                         O PTIMIZER C HOICE M ATTERS F OR T HE E MERGENCE
                                         OF N EURAL C OLLAPSE

                                          Jim Zhao∗, Tin Sum Cheng                                     Wojciech Masarczyk
                                          University of Basel                                          Warsaw University of Technology
                                          {jim.zhao, tinsum.cheng}@unibas.ch                           IDEAS Research Institute
                                                                                                       wojciech.masarczyk@gmail.com

                                          Aurelien Lucchi
                                          University of Basel
                                          aurelien.lucchi@unibas.ch
arXiv:2602.16642v3 [cs.LG] 25 Feb 2026




                                                                                               A BSTRACT

                                                      Neural Collapse (NC) refers to the emergence of highly symmetric geometric
                                                      structures in the representations of deep neural networks during the terminal phase
                                                      of training. Despite its prevalence, the theoretical understanding of NC remains
                                                      limited. Existing analyses largely ignore the role of the optimizer, thereby suggest-
                                                      ing that NC is universal across optimization methods. In this work, we challenge
                                                      this assumption and demonstrate that the choice of optimizer plays a critical role in
                                                      the emergence of NC. The phenomenon is typically quantified through NC metrics,
                                                      which, however, are difficult to track and analyze theoretically. To overcome this
                                                      limitation, we introduce a novel diagnostic metric, NC0, whose convergence to
                                                      zero is a necessary condition for NC. Using NC0, we provide theoretical evidence
                                                      that NC cannot emerge under decoupled weight decay in adaptive optimizers, as
                                                      implemented in AdamW. Concretely, we prove that SGD, SignGD with coupled
                                                      weight decay (a special case of Adam), and SignGD with decoupled weight decay
                                                      (a special case of AdamW) exhibit qualitatively different NC0 dynamics. Also,
                                                      we show the accelerating effect of momentum on NC (beyond convergence of
                                                      train loss) when trained with SGD, being the first result concerning momentum in
                                                      the context of NC. Finally, we conduct extensive empirical experiments consist-
                                                      ing of 3,900 training runs across various datasets, architectures, optimizers, and
                                                      hyperparameters, confirming our theoretical results. This work provides the first
                                                      theoretical explanation for optimizer-dependent emergence of NC and highlights
                                                      the overlooked role of weight-decay coupling in shaping the implicit biases of
                                                      optimizers.


                                         1       I NTRODUCTION

                                         Neural networks have driven many of the recent breakthroughs in artificial intelligence, yet the
                                         mechanisms underlying their success remain only partially understood. A key empirical clue is
                                         neural collapse (NC) – first documented by Papyan et al. (2020) – in which the last-layer feature
                                         vectors and classifier weights self-organise into a highly symmetric configuration during the terminal
                                         phase of training (TPT). While the reasons for the emergence of NC are still not fully understood, its
                                         impact on the behavior of a model is evident. For instance, Liu et al. (2023) induce NC to improve
                                         generalization in class-imbalanced training and Galanti et al. (2021) show that the emergence of NC
                                         improves transfer learning as well. Furthermore, the presence of NC has been connected to better
                                         out-of-distribution detection (Liu & Qin, 2023).
                                         Theoretical explanations for NC have primarily relied on simplified models and assumptions (Mixon
                                         et al., 2022; Zhu et al., 2021) that have largely ignored the role of the optimizer, thereby suggesting
                                         that NC is universal across optimization methods. In this work, we challenge this assumption and
                                             ∗
                                                 First two authors share equal contribution.


                                                                                                   1
```

## Page 002

```text
Published as a conference paper at ICLR 2026




demonstrate that the choice of optimizer plays a critical role in the emergence of NC. Concretely,
we show that training with AdamW (Loshchilov & Hutter, 2019) does not lead to an NC solution,
whereas training with SGD or Adam (Kingma & Ba, 2014) does. Through extensive experiments, we
trace this back to how weight decay is applied in both optimizer and identify the coupling of weight
decay as a necessity for the emergence of NC.
One major challenge in studying NC lies in the original metrics, which are difficult to track and
analyze theoretically. These metrics were designed to quantify the progressive geometric alignment
associated with NC and are expected to converge to zero in the idealized setting where NC holds as
training time approaches infinity. However, under realistic training regimes, such as finite training
epochs and learning rate decay, these metrics typically plateau at small but nonzero values. As a
result, there is no rigorous criterion for determining whether NC has truly occurred.
This limitation motivates us to introduce a novel diagnostic metric, NC0, whose convergence to zero
is necessary (though not sufficient) for NC. Unlike previous metrics, NC0 enables a more definitive
assessment: if NC0 diverges during training, we can conclude that NC can not occur—even in
cases where other NC metrics misleadingly converge to small positive values, creating an illusion
of collapse. We discuss the peculiarity of interpreting NC metrics in practice later in Section 4.1.
Furthermore, NC0 allows us to go beyond loss landscape analysis and theoretically derive convergence
rates with which NC0 converges to zero.

Contribution In this paper, we conduct extensive experiments – spanning over 3,900 training
runs – to investigate the role of coupled weight decay in the emergence of NC. We identify coupled
weight decay as a key driver of NC in realistic settings, extending recent theoretical insights (Pan
& Cao, 2024; Jacot et al., 2024) that were limited to quasi-optimal solutions in simplified models.
In particular, we show that the form of weight decay used in adaptive optimizers such as Adam
(Kingma & Ba, 2014) and AdamW (Loshchilov & Hutter, 2019) critically affects whether NC
emerges. Strikingly, while networks trained with Adam often exhibit NC, AdamW – despite its
algorithmic similarity –fails to produce NC, with the corresponding metrics failing to converge to
zero over time (Figure 1). This subtle yet consequential distinction has been largely overlooked in
prior work. An overview of our theoretical contributions can be found in Table 1
In summary, we make the following contributions:

      1. Across a wide range of experiments, we find that coupled weight decay is a necessary
         condition for NC to emerge in adaptive optimizers, such as Adam and Signum.
      2. Furthermore, we show the accelerating effect of momentum on NC (beyond convergence
         of train loss) when trained with SGD, being the first result concerning momentum in the
         context of NC.
      3. We support our empirical findings with the following theoretical statements on the new NC0
         metric:
            • with SGD (with both coupled or decoupled weight decay), NC0 converges to zero at an
               exponential rate proportional to the weight decay;
            • with sign gradient descent (SignGD) with decoupled weight decay, a special case of
              AdamW, NC0 converges to some positive constant;
            • with SignGD with coupled weight decay, a special case of Adam, NC0 exhibits a
               non-monotonic trajectory, increasing before eventually decreasing. Using learning rate
               decreasing to zero, we show that NC0 also vanishes.

Organization This paper is organized as follows. In Section 2, we recapitulate the four properties
to characterize NC and introduce a novel NC property NC0. In Section 3 we present our main
experimental results with theoretical support. Finally, Section 4 provides insights and discussions on
the implications of our results.

Notation We use [K] = {1, 2, . . . , K} to denote the index set for any integer K ∈ N. For a
matrix W, we let Vec(W) denote the vectorization of W obtained by stacking its columns. The
Frobenius inner product between two matrices W, W′ is denoted by ⟨W, W′ ⟩ = Tr(W⊤ W′ ).
With slight abuse of notation, we write ∥W∥ = ∥W∥F for the Frobenius norm when W is a matrix,
and ∥v∥ = ∥v∥2 for the Euclidean norm when v is a vector. In other words, ∥W∥ = ∥ Vec(W)∥.


                                                  2
```

## Page 003

```text
Published as a conference paper at ICLR 2026




                                                                                                       Figure 1: NC0 (left) and NC3 (right) met-
                                                                                                       rics at the end of training. Lower values in-
                                                        1.0
                                                                                                       dicate stronger NC. AdamW shows consis-
                                                                                        optimizer      tently higher metrics than Adam. Averages
       101                                                                                 AdamW
                                                        0.8                                Adam        computed over runs with varying learning
       100                                              0.6                                            rates and momentum; shaded regions show




                                                  NC3
NC0




                                                        0.4
                                                                                                       ±1 standard deviation. X-axis is log-scaled.
      10 1    optimizer
                 AdamW                                                                                 Note that there are no values for Adam for
                 Adam                                   0.2
                                                                                                       WD larger than 0.05 as the model did not
             0 10 5 10 4 10 3 10 2 10 1 100 101                0 10 5 10 4 10 3 10 2 10 1 100 101
                          Weight decay                                   Weight decay                  train due to over regularization.


We denote by I the identity matrix, by 1 the all-ones column vector, and by J the all-ones matrix, i.e.,
J = 11⊤ .
                                          Table 1: Overview of our theoretical results on NC0.

              Result                                          Optimizers           Model             Convergence to 0?        learning rate
              Theorem 3.1                   SGD with DWD                                 Any           yes, exponential           constant
              Theorem 3.2                   SGD with CWD                                 Any           yes, exponential           constant
              Theorem 3.3                SignGD with DWD                                UFM                        yes     step-wise decay
              Theorem 3.4                SignGD with CWD                                UFM                         no                   -



2            N EURAL C OLLAPSE
Neural collapse (NC), observed during the terminal phase of training (TPT) in deep neural networks
(DNN), manifests itself through several geometric properties involving the last-layer features and
weights in the K-class classification task:
                                                  N
                                                  X                                                 λ       λ
                                         min                  ℓ(Whθ (xn ), yn ) +                     ∥W∥2 + ∥Vec(θ)∥2                          (1)
                                         W,θ
                                                  n=1
                                                                                                    2       2

where (xn , yn )N         D
                n=1 ⊂ R × [K] is the training set, W ∈ R
                                                             K×P
                                                                  is the last-layer weights, hθ (xn ) ∈
R is the last-layer feature as the output of some backbone parameterized by θ, ℓ : RK × [K] →
  P

[0, ∞) is the loss function, and λ > 0 is the L2-regularization constant.
These properties, formalized by their corresponding metrics in the original paper Papyan et al. (2020),
are:

             1. NC1 - Variability Collapse: Features collapse to their respective class means, indicating
                that within-class variability vanishes.
             2. NC2 - Convergence of Centered Class Means to Simplex ETF: Centered Class means
                converge to a simplex equiangular tight frame (ETF).
             3. NC3 - Convergence to Self-Duality: Rows of the last-layer weight W ∈ RK×P align with
                the columns of the class means, creating a dual relationship between weights and features.
             4. NC4 - Simplification to Nearest-Class-Center: The classifier’s decision boundaries are
                simplified to those of a nearest-class-mean (NCC) classifier.

A solution satisfying all of these properties is referred to as a NC solution. In addition to these
prior NC properties, we introduce another novel NC property NC0, whose convergence to zero is a
necessary condition (though not sufficient) for NC.
NC0 - Zero Row Sum of Last-Layer Weight: The row sum of the last-layer weight W in the model
converges to zero.
The first observation is that NC0 is a necessary condition for NC2 and NC3:
Proposition 2.1. NC2 and NC3 implies NC0.


                                                                                            3
```

## Page 004

```text
Published as a conference paper at ICLR 2026




Proof. For each class k ∈ [K], we define the class mean µk = |{n:yn1 =k}| n:yn =k hθ (xn ) ∈ RP
                                                                         P
                                             PN
and the centered class mean µ̄k = µk − N1 n=1 hθ (xn ). We concatenate them into a matrix
                     P ×K
M = (µ̄k )K k=1 ∈ R        with M1 = 0, since we centered the class means. By NC2, M converge
to a simplex ETF in the ambient space RP , meaning M/∥M∥F → QM∗ where M∗ ∈ RK×K is a
unit matrix with columns forming a K-simplex EFT in RK and Q ∈ RP ×K is the isometric injection
map into the ambient space. Since M1 = 0 and Q is injective, the unit matrix M∗ has to be in
                def.
the form: M∗ = P √K−1    1       1
                                    
                              I− K J for some orthogonal matrix P. But it can be absorbed into
Q as the matrix QP is still an isometric injection. Hence, without loss of generality, we assume
     def.
M∗ = √K−1  1          1
                         
                 I− K   J and hence

                           M⊤ M/∥M⊤ M∥2F → (QM∗ )⊤ QM∗ = (M∗ )2 = M∗ .
On the other hand, NC3 states that M/∥M∥ − W⊤ /∥W∥ → 0 as t → ∞. Hence we have
WW⊤
∥W∥2F
       − M∗ → 0 as t → ∞. Now note that 1⊤ M∗ 1 = 0, hence ∥W⊤ 1∥2 = 1⊤ WW⊤ 1 → 0.
Note that the last line holds if and only if NC0 holds.

NC0 offers two key advantages. First, it serves as a diagnostic tool: if NC0 does not converge, then
at least one of NC2 or NC3 must fail, providing a clear signal that neural collapse cannot occur.
Second, NC0 is more mathematically tractable than the original NC metrics, whose dynamics are
difficult to analyze and remain underexplored. As we demonstrate in Section 3, NC0’s evolution
during training can be reliably tracked and used to explain empirical trends observed across different
optimizers. In addition, our extensive experiments also show that NC0 is correlating well with prior
NC metrics, particularly for small learning rates (see Figure 2). For a more detailed explanation and
formal definitions of NC properties and their metrics, we refer the reader to Section B.

                  lr = 0.001          lr = 0.005          lr = 0.01   lr = 0.0679     optimizer
           1                                                                            AdamW
                                                                                        Adam
        NC3




                                                                                        SGDW
           0                                                                            SGD
                                                                                        SignumW
                 10 2 102            10 2 102         10 2 102        10 2 102          Signum
                    NC0                 NC0              NC0             NC0

Figure 2: NC0 weakly correlates with NC3 across different optimizers and learning rates. Details on
the regression fit can be found in Section D.3


3       M AIN R ESULT
3.1       E XPERIMENTAL S ETUP

We conducted extensive experiments training a ResNet9 and VGG9 using various optimizers, includ-
ing Adam, AdamW, SGD, SGD with decoupled weight decay (SGDW), Signum (Bernstein et al.,
2018), and Signum with decoupled weight decay (SignumW) trained on MNIST, FashionMNIST and
Cifar10. Every optimizer is trained with three different learning rates (LR), six different values of
momentum, and six different values of weight decay to also control the effect of hyperparameters on
the emergence of NC. This resulted in a total of 2 × 3 × 6 × 108 = 3, 888 training runs. Note that
we only keep runs with reasonably high training accuracy. Too large weight decay over regularize
the model and the model does not train anymore. Thus, the number of valid training runs is actually
smaller than 3,888. All networks were trained for 200 epochs using a batch size of 128, with the
learning rate being decayed by a factor of 10 after one-third and two-thirds of the training duration,
as described in the original work by Papyan et al. (2020). In addition, we conducted ablation studies
to control for the number of training epochs and to verify that the results also hold for unconstrained
feature models (UFM)1 , leading to a total of over 3,900+ training runs. Further details and all
experimental results can be found in Section D. Ablation studies on the effect of training epochs can
be found in Section D.4.1
    1
        see Section C.5 for an introduction to UFM.


                                                      4
```

## Page 005

```text
Published as a conference paper at ICLR 2026




Table 2: Final NC metrics for the same setting as in Figure 6, following the setup of Papyan
et al. (2020). Lower values (↓) indicate stronger neural collapse. Values in parentheses represent
percentages relative to the metric at initialization.

            Optimizer                                           NC0↓                        NC1↓                                 NC2↓                            NC3↓
            SGD                      2.14e-04 (< −99.5%)                        0.05 (−99.3%)                    0.29 (−63.0%)                     0.35 (−75.1%)
            SGDW                            0.55 (−68.9%)                       0.26 (−96.3%)                    0.46 (−42.4%)                     0.80 (−43.5%)
            Adam                            0.34 (−80.6%)                       0.04 (−99.5%)                    0.29 (−63.9%)                     0.29 (−79.5%)
            AdamW                          5.33 (≫ 100%)                        0.20 (−97.2%)                    0.54 (−32.4%)                     0.78 (−45.2%)
            Signum                          0.78 (−55.3%)                       0.13 (−98.1%)                    0.50 (−36.8%)                     0.58 (−59.0%)
            SignumW                     3185.69 (≫ 100%)                        0.30 (−95.7%)                    1.15 (+44.2%)                      1.40 (−1.2%)


3.2            W EIGHT D ECAY IS E SSENTIAL AND M OMENTUM ACCELERATES NC

Our experiments show that weight decay is necessary to reduce the NC metric across all optimizers
and hyperparameter settings, as shown in Figure 3 for Signum and SGD, and earlier in Figure 1 for
Adam and AdamW as well as in our ablation studies in Section D.4.1 and Section D.4.6. While the
experiments cannot fully exclude the possibility that NC can be achieved eventually in the asymptotic
limit without weight decay, we argue that WD is essential to observe the emergence of NC in practical
finite-length training settings on realistic models2 .

                                                 1.4                                                                                         1.0
      104                                                                                        100                                                                        optimizer
                                                 1.2                                                                                         0.8                                SGDW
      103                                                                                       10 1                                                                            SGD
                       optimizer                 1.0
                                                                                                10 2                                         0.6
                                           NC3




                                                                                                                                       NC3
                          SignumW
NC0




      102
                                                                                          NC0




                          Signum                 0.8
                                                                                                10 3                                         0.4
      101                                        0.6     optimizer
                                                            SignumW                             10 4
                                                                                                                SGDW                         0.2
      100                                        0.4        Signum                              10 5            SGD
            0 10 5   10 4 10 3 10 2 10 1               0 10 5   10 4 10 3 10 2 10 1                    0 10 5    10 4 10 3 10 2 10 1               0 10 5   10 4 10 3 10 2 10 1
                      Weight decay                               Weight decay                                     Weight decay                               Weight decay


Figure 3: NC0 and NC3 metrics at the end of training for a ResNet9 trained on FashionMNIST
for Signum and SignumW (left side) and SGD and SGDW (right side). Shaded area refers to one
standard deviation across all trainings run with corresponding optimizer. Note that there are fewer
values for Signum and SGD as the model did not train due to over regularization for too large WD.

From the figures, we can conclude that larger weight decay leads to a stronger decrease of NC
metrics. In particular, we show that adaptive optimizers with decoupled weight decay have much
larger NC metrics, which are strictly away from zero, showing no sign of NC. In addition, we show
empirically that momentum amplifies the effect of weight decay on the decrease of NC metrics in
SGD, as shown in the heatmap in Figure 5. This implies that one achieves a decrease in the NC
metrics both by increasing weight decay for fixed momentum or by increasing momentum for fixed
non-zero weight decay. The effect of momentum on the NC metrics becomes larger for larger values
of weight decay. We remark that this goes beyond the acceleration of convergence of the train loss, as
we study in an ablation study in Section D.4.5. In particular, we show in Figure 4 that two training
runs with different momentum and otherwise same hyperparameters can reach the same train loss,
while reaching different NC metrics. This indicates that they have converged to solutions with very
different geometric structure.
The experimental results are complemented by Theorem 3.1 and Theorem 3.2 showing that NC0
converges to 0 with an exponential rate trained with SGD, which is proportional to momentum and
weight decay, highlighting that NC cannot be achieved without weight decay and that momentum
accelerates the convergence of NC metrics.
Theorem 3.1 (SGD with decoupled weight decay promotes NC0). Assume a model of the form
f (W, θ, x) = Whθ (x) is trained using cross-entropy loss with stochastic gradient descent (SGD)
and momentum β ∈ [0, 1), weight decay λ ∈ [0, 1), and learning rate η > 0 on all parameters θ, W.
    2
      We note that Ji et al. (2021) show both theoretically and empirically the emergence of NC on the uncon-
strained layer-peeled model (ULPM) objective under gradient flow without weight decay.


                                                                                      5
```

## Page 006

```text
Published as a conference paper at ICLR 2026




                                                                     0.7                       1.00                                                                                                                                      0.7                                                                     0.7
                       100                                           0.9                                                                                                                                                                 0.9                 1.25                                                0.9
                                                                                               0.95                                                                              100


    train_loss




                                                                                       train_acc
                   10 1                                                                                                                                                                                                                                      1.00




                                                                                                                                                                    NC0




                                                                                                                                                                                                                                                         NC3
                                                                                               0.90
                                                                                                                                                  mom                        6 × 10 1                                                                        0.75
                   10 2                                                                        0.85                           0.7
                                                                                                                              0.9                                            4 × 10 1                                                                        0.50
                                                                                               0.80                                                                          3 × 10 1
                             0         50        100 150 200                                           0            50 100 150 200                                                        0            50    100 150 200                                                   0            50       100 150 200
                                                epoch                                                                  epoch                                                                                epoch                                                                               epoch

Figure 4: Train loss, train accuracy and NC metrics for fixed WD=0.005 and mom=0.7 and 0.9.
Although both runs converge to almost exactly the same train loss, the final NC metrics differ
considerably. Plots including NC1 and NC2 can be found in Figure 24.

                                    NC0, LR=0.001                                                                                              NC2, LR=0.001                                                                                          NC3, LR=0.001
                                                                                                                                                                                                             0.6                                                                                                       0.7
                       1.75 1.75 1.75 1.75 1.72 1.69 1.61 1.48                                                                    0.62 0.62 0.62 0.62 0.62 0.61 0.60 0.58                                                                0.72 0.72 0.72 0.72 0.72 0.71 0.70 0.67
      0.0




                                                                                                                 0.0




                                                                                                                                                                                                                        0.0
                                                                                                      1.6

                       1.75 1.75 1.75 1.74 1.69 1.64 1.48 1.24                                        1.4                         0.59 0.60 0.59 0.59 0.59 0.58 0.55 0.51                                                                0.70 0.70 0.70 0.69 0.69 0.68 0.65 0.60                                       0.6
      0.5




                                                                                                                 0.5




                                                                                                                                                                                                                        0.5
                                                                                                                                                                                                             0.5
                                                                                                      1.2
                       1.75 1.75 1.74 1.73 1.66 1.56 1.32 0.99                                                                    0.58 0.58 0.58 0.58 0.57 0.56 0.51 0.42                                                                0.68 0.68 0.68 0.68 0.67 0.65 0.60 0.52
      0.7




                                                                                                                 0.7




                                                                                                                                                                                                                        0.7
                                                                                                                                                                                                                                                                                                                       0.5
 momentum




                                                                                                            momentum




                                                                                                                                                                                                                   momentum
                                                                                                      1.0                                                                                                    0.4
                       1.75 1.74 1.72 1.69 1.48 1.24 0.74 0.31                                                                    0.56 0.56 0.56 0.55 0.52 0.47 0.33 0.22                                                                0.65 0.65 0.65 0.64 0.61 0.57 0.43 0.30
      0.9




                                                                                                                 0.9




                                                                                                                                                                                                                        0.9
                                                                                                      0.8                                                                                                                                                                                                              0.4
                       1.75 1.72 1.69 1.64 1.24 0.88 0.31 0.06                                                                    0.57 0.56 0.55 0.55 0.47 0.37 0.22 0.15                                                                0.63 0.63 0.63 0.62 0.55 0.46 0.30 0.19
      0.98 0.97 0.95




                                                                                                                 0.98 0.97 0.95




                                                                                                                                                                                                                        0.98 0.97 0.95
                                                                                                      0.6                                                                                                    0.3
                                                                                                      0.4                                                                                                                                                                                                              0.3
                       1.75 1.70 1.66 1.56 0.99 0.56 0.10 0.01                                                                    0.56 0.55 0.54 0.53 0.40 0.27 0.16 0.14                                                                0.62 0.61 0.60 0.59 0.48 0.37 0.20 0.17
                                                                                                      0.2                                                                                                    0.2
                       1.75 1.68 1.61 1.48 0.74 0.31 0.02 0.00                                                                    0.56 0.55 0.55 0.52 0.32 0.21 0.15 0.13                                                                0.62 0.62 0.61 0.58 0.42 0.31 0.16 0.16                                       0.2
                       0.0
                             2.5e-05
                                       5e-05
                                                0.0001
                                                         0.0005
                                                                  0.001
                                                                           0.0025
                                                                                    0.005




                                                                                                                                  0.0
                                                                                                                                        2.5e-05
                                                                                                                                                  5e-05
                                                                                                                                                           0.0001
                                                                                                                                                                    0.0005
                                                                                                                                                                              0.001
                                                                                                                                                                                      0.0025
                                                                                                                                                                                               0.005




                                                                                                                                                                                                                                         0.0
                                                                                                                                                                                                                                               2.5e-05
                                                                                                                                                                                                                                                         5e-05
                                                                                                                                                                                                                                                                  0.0001
                                                                                                                                                                                                                                                                               0.0005
                                                                                                                                                                                                                                                                                        0.001
                                                                                                                                                                                                                                                                                                0.0025
                                                                                                                                                                                                                                                                                                         0.005
                                               weight_decay                                                                                               weight_decay                                                                                           weight_decay


Figure 5: Heatmap of NC0, NC2 and NC3 for varying values of momentum and weight decay on
ResNet9 trained on FashionMNIST with SGD.


For instance, the last-layer weights W are updated according to:
                                                                                                            Vt+1 = βVt + ∇Wt LCE ,
                                                                                                            Wt+1 = (1 − ηλ)Wt − ηVt+1 .
                                         1
If 0 < ηλ < 2, then the NC0 metric αt := K ∥Wt⊤ 1∥22 decays exponentially to zero in t.

Proof. The key observation is that the row sum of the loss gradient ∇LCE (Wt )⊤ 1K is zero, which
largely simplifies the NC0 metric to only be dependent on the weight decay λ and momentum β. For
the details of the proof, please refer to Subsection E in the Appendix.
Theorem 3.2 (SGD with coupled weight decay promotes NC0). Assume a model of the form
f (W, θ, x) = Whθ (x) is trained using cross-entropy loss with stochastic gradient descent (SGD)
and momentum β ∈ [0, 1), weight decay λ ∈ [0, 1), and learning rate η > 0 on all parameters θ, W.
For instance, the last-layer weights W are updated according to:
                                                                                                       Vt+1 = βVt + ∇Wt LCE + λWt ,
                                                                                                       Wt+1 = Wt − ηVt+1 .
                                                1
If 0 < ηλ < 2(1 + β), then the NC0 metric αt := K ∥Wt⊤ 1∥22 decays exponentially to zero in t.

Proof. Similar to the proof of Theorem 3.1 For the details of the proof, please refer to Subsection E
in the Appendix.

3.3                      W EIGHT D ECAY C OUPLING M ATTERS

While weight decay has been theoretically shown to be essential for NC in prior works (Pan & Cao,
2024; Jacot et al., 2024), these works ignore how weight decay is applied by treating L2 -regularization
of the gradient and applying weight decay directly on parameters as equivalent. However, we note that
this equivalency only holds for vanilla SGD and not for adaptive optimizers, such as Adam or AdamW,
nor when momentum is applied. In particular, our experiments reveal that NC does not emerge under
SignumW and AdamW under realistic settings. This highlights the crucial role of coupled weight


                                                                                                                                                                    6
```

## Page 007

```text
Published as a conference paper at ICLR 2026




             102                                            101
                                                                                                      1.0
                                                            100                                                                       1.0



       NC0




                                                      NC1




                                                                                    NC2




                                                                                                                                NC3
             100
                                                           10 1                                       0.5                             0.5
         10 2
                   0    100                          200          0    100 200                                0    100       200             0    100 200
                       epoch                                          epoch                                       epoch                          epoch
                        SGD                                SGDW          AdamW                                Adam           Signum              SignumW

        Figure 6: NC metrics throughout training on a ResNet9 trained on FashionMNIST.


decay – that is L2 -regularization applied directly within the gradient update – as a requirement for
NC. This subtle yet important distinction has been largely overlooked in prior literature.
Importantly, tracking the evolution of the NC metrics (Figure 6) and the singular values of centered
class means M and the last-layer weight W (Figure 7) throughout training (here shown for a ResNet9
trained on FashionMNIST), one can see that using adaptive optimizers with decoupled weight decay
leads to fundamentally different dynamics of the NC metrics and singular values despite all models
reaching TPT, where training error is (almost) zero.

                                             15                                                   40
                          Singular values of W




                                                                               Singular values of M
                                             10                                                   30
                                                                                                  20
                                                 5
                                                                                                  10
                                                 0                                                    0
                                                      0     50     100 150   200                          0       50      100 150      200
                                                                  Epoch                                                  Epoch
                                                           SGD        SGDW    AdamW                               Adam       Signum

Figure 7: Singular values of last-layer weights W (left) and centered class means M (right) through-
out training. The dotted line corresponds to the smallest singular value and the full line corresponds
to the average singular value, excluding the smallest singular value. Singular values for SignumW
are out-of-range and are shown in Figure 29 in the appendix.

Specifically, Figure 7 shows that the smallest singular value of W increases during training with
AdamW and SignumW, indicating failure to satisfy NC3. Additionally, NC0 and the nonzero singular
values of M grow throughout training and exhibit high variance, suggesting that NC2 is also less
well-fulfilled in these settings.
In Figure 6, we further observe that SGD and Adam achieve the lowest NC metric values, while
AdamW, SignumW, and SGDW saturate early at much higher levels. Although the NC metrics for
Signum are slightly larger than for SGD and Adam, they continue to decrease over time, suggesting
potential convergence to NC under longer training.
Finally, our experiments in Figure 1 and Figure 3 demonstrate that the NC0 and NC3 metrics of
AdamW and SignumW remain significantly larger than those of Adam and Signum, even when
using weight decay several orders of magnitudes higher. This indicates that models trained with
AdamW or SignumW are consistently farther from achieving NC. Note that the NC metrics for
SGD and SGDW remain relatively close, consistent with our theoretical results in Theorem 3.1 and
Theorem 3.2, while the gap between coupled and decoupled weight decay has a more pronounced
effect in adaptive optimizers than in SGD. This suggests the effect is not simply due to greater weight
decay accumulation through momentum but stems from a deeper interaction with the optimization
dynamics.

3.4   I NTERPOLATING A DAM W AND A DAM

To further investigate why AdamW fails to exhibit neural collapse (NC) while Adam does, we con-
ducted an ablation study by “interpolating” between the two optimizers. Specifically, we implemented
a variant that combines both coupled weight decay (as in Adam) and decoupled weight decay (as


                                                                                7
```

## Page 008

```text
Published as a conference paper at ICLR 2026




in AdamW). For each run, we varied the strength of the coupled weight decay while adjusting the
decoupled component such that the total weight decay remained fixed at 0.0005. The momentum was
set to 0.9 across all configurations.
As shown in Figure 8, increasing the coupled component leads to a smooth improvement in NC
metrics—particularly NC0, NC2, and NC3—while the validation accuracy remains largely unaffected.
This experiment suggests that coupled weight decay is a critical factor in enabling neural collapse,
yet it is not strictly necessary for achieving strong generalization performance, as all configurations
yield similar validation accuracy. This strengthens a point raised earlier about the limitations of NC
to understand generalization Hui et al. (2022).
                        1.000
  validation accuracy




                                                                                                            0.8                                     0.8
                        0.995
                                                        NC0   100                                           0.6                                     0.6




                                                                                                      NC2




                                                                                                                                              NC3
                        0.990
                                                                                                            0.4                                     0.4
                        0.985
                                                                                                            0.2                                     0.2
                        0.980
                                0   50    100    150   200           0       50    100     150      200           0       50    100    150   200          0   50    100    150   200
                                         epoch                                    epoch                                        epoch                               epoch
                                                                                                 coupled weight decay
                                                                    0.0000        0.0001           0.0002             0.0003      0.0004      0.0005


Figure 8: Interpolating Adam and AdamW by varying the coupled and decoupled weight decay. Total
weight decay was fixed to 0.0005. Note that coupled weight decay = 0 is equivalent to AdamW and
coupled weight decay = 0.0005 is equivalent to Adam. Experiments trained on ResNet9 with MNIST.

This observation is supported by our theoretical results in Theorem 3.3 and Theorem 3.4, which show
that SignGD with decoupled weight decay fails to satisfy NC0 and therefore cannot converge to a
neural collapse solution, whereas SignGD with coupled weight decay exhibits different behaviour.
We note that SignGD corresponds to a special case of Adam and AdamW when the parameters β1 ,
β2 , and ε are set to zero.
Theorem 3.3 (Sign GD with decoupled weight decay avoids NC0). Consider sign GD with
(decoupled) weight decay λ > 0 and step size η > 0 on the UFM loss LCE (WH, I) =
PN                                                   ∗
   n=1 LCE (Whn , en ), where the feature H = M is fixed to an NC solution and only the weight
W is trained:
                            Wt+1 = Wt − η(sign(∇Wt LCE ) + λWt )
Then the NC0 metric α = ∥Wt⊤ 1K ∥22 increases monotonically from zero to the limit:
                                                                                                       (K − 2)2
                                                                                  lim αt =                      .
                                                                                  t→∞                     λ2
In particular, αt does not vanish as t → ∞.

Proof idea: The key observation is that the signed loss gradient sign(∇LCE (Wt )) in this setting is
constant in t, simplifying the following computation. See Section E for the full proof.           □

Theorem 3.4 (Sign GD with coupled weight decay can lead to NC0). Consider sign
GD with (coupled) weight decay λ > 0 and step size η > 0 on the UFM loss
                   PN                                            ∗
LCE (WH, I) =        n=1 LCE (Whn , en ), where the feature H = M is fixed to an NC solu-
tion and only the weight W is trained:
                                                             Wt+1 = Wt − η(sign(∇Wt LCE + λWt ))
We initialize W0 = 0 ∈ RK×K and define the covariance matrix Ct = Wt Wt⊤ and the scalar αt =
⟨Ct , Ĵ⟩F where Ĵ = K 1
                          11⊤ . Then there exists a learning rate decay scheme η = η(t) −−−→ 0
                                                                                         t→∞
such that αt −−−→ 0.
                                     t→∞


Proof. See Section E.

The key difference between the results of Theorem 3.3 and Theorem 3.4 lies in how coupled weight
decay affects the signed gradient during training. As the weight norm ∥W∥ increases, the coupled
decay term can eventually flip the sign of the gradient, altering the trajectory of the NC0 metric αt .


                                                                                                     8
```

## Page 009

```text
Published as a conference paper at ICLR 2026




                            SGD                             Adam                                  AdamW                              SignSGD                            SignSGDW
           9.7 × 10 4                            10 2                                                                   10 2           wd=5e-04
                                                                                      2
          9.65 × 10 4                                                            10
                                                                                                                                       wd=1e-03
           9.6 × 10 4       wd=5e-04                        wd=5e-04                                                    10 3           wd=5e-03              10 2


    NC0




                                           NC0




                                                                           NC0




                                                                                                                  NC0




                                                                                                                                                       NC0
          9.55 × 10 4       wd=1e-03                        wd=1e-03
                                                 10 3                                              wd=5e-04                                                               wd=5e-04
           9.5 × 10 4       wd=5e-03                        wd=5e-03                                                    10 4
          9.45 × 10 4                                                                              wd=1e-03                                                               wd=1e-03
           9.4 × 10 4                                                                              wd=5e-03             10   5                                            wd=5e-03
                                                                                 10 3                                                                        10 3
                        0   5000   10000                0   5000   10000                  0        5000   10000                  0    5000     10000                0     5000     10000
                            Step                            Step                                   Step                               Step                                Step



Figure 9: Training dynamic of NC0 with optimizers SGD, Adam, AdamW, Adam0 (β1 = β2 = 0),
AdamW0 (β1 = β2 = 0). For AdamW and SignSGD the inlay shows the NC0 metric more detailed
for the last 2000 steps. Note that 5 steps correspond to one training epoch.


Initially, αt grows at a similar rate in both cases, but their behaviors diverge once the decay term
becomes dominant.
To illustrate this effect, we conducted a small-scale experiment using a simple MLP on a separable
dataset with various optimizers. As shown in Figure 9, SignSGD displays non-monotonic dynamics
in αt , while SignSGDW exhibits steady convergence to a positive value. Similar patterns appear in
Adam and AdamW, though more smoothed due to their adaptive updates.


4          D ISCUSSION AND L IMITATIONS

In this section, we discuss new insights, additional considerations and limitations from the main
results in Section 3. Additionally, we explore potential follow-up research directions that could
provide theoretical explanations or extend our experiments to broader settings.

4.1          I NTERPRETING NC M ETRICS IN P RACTICE

While NC is defined by the convergence of all NC metrics to zero in the limit, practical experiments
never achieve exact zeros. Since NC is inherently a continuous rather than discrete phenomenon, it
becomes necessary to define what constitutes the presence of NC in practice. This important issue
has not been thoroughly addressed in the existing literature.
A further complication is that different NC metrics operate on different scales and these scales vary
across settings of architectures and datasets. For example, in our experiments, the smallest observed
values for NC2 and NC3 are on the order of 0.1, whereas NC1 can reach values an order of magnitude
smaller.
In this work, we therefore refer to the emergence of NC in terms of relative strength. Specifically, we
use the NC metric values at initialization as a baseline for models that do not exhibit NC, and use the
smallest values achieved across all experiments as a reference point for models that do. This framing
allows us to discuss the strength of NC emergence across different optimizers and settings.

4.2          T HE R EDUNDANT NC4 PROPERTY

Readers may notice that we omit NC4 from the results in Section 3. This is because we observed
that NC4 is consistently satisfied whenever the training accuracy approaches 100%, regardless of
whether the other NC metrics (NC1–NC3) exhibit collapse. As shown in Figure 57, NC4 is largely
uncorrelated with the other metrics. To maintain a clearer and more focused presentation, we therefore
exclude NC4 from our main analysis.

4.3          PARTIAL N EURAL C OLLAPSE

Another subtlety we observe is what we term partial neural collapse. As shown in Table 3, AdamW
can achieve minimal values for NC1 and NC2 among all optimizers, even while NC0 diverges and
NC3 is not satisfied. This indicates that NC properties may not always emerge jointly, contrary to the
original claim in Papyan et al. (2020). Understanding the theoretical conditions under which only a
subset of NC properties holds remains an intriguing open question.


                                                                                              9
```

## Page 010

```text
Published as a conference paper at ICLR 2026




Table 3: Final NC metrics for the run with the smallest absolute NC3 metric and > 99% training
accuracy for each optimizer. Lower values (↓) indicate stronger neural collapse. Values in parentheses
represent percentages relative to the metric at initialization. Hyperparameters used for each optimizer
can be found in Table 5.

    Optimizer                    NC0↓                  NC1↓               NC2↓              NC3↓
    SGD         1.53e-05 (< −99.5%)        0.02 (< −99.5%)      0.19 (−75.8%)      0.13 (−90.9%)
    SGDW        1.54e-04 (< −99.5%)        0.01 (< −99.5%)      0.15 (−81.7%)      0.10 (−92.7%)
    Adam            0.12 (< −93.2%)           0.04 (−99.5%)     0.23 (−71.6%)      0.17 (−88.2%)
    AdamW              8.09 (≫100%)        0.01 (< −99.5%)      0.14 (−82.1%)      0.49 (−65.1%)



4.4   L IMITATIONS OF T HEORETICAL S UPPORT

Our experiments on Adam and AdamW are conducted on realistic models and datasets, whereas our
theoretical results (Theorem 3.3, Theorem 3.4) focus on a simplified setting: SignGD applied to the
unconstrained feature model. While this restricted setup already demonstrates that AdamW fails to
achieve NC, it does not fully capture the complexity of deep neural networks or adaptive optimizers
in practice. Nevertheless, we believe our proof techniques could be extended to explain why Adam
may lead to NC in more general settings. Moreover, our theoretical analysis is limited to the training
dynamics of NC0, chosen for its analytical tractability and strong empirical correlation with other
NC metrics. A full theoretical understanding of NC1–NC3 under realistic optimization dynamics
remains an open challenge, and we leave this direction for future work.

4.5   F UTURE R ESEARCH

Other than the topic we have discussed in the previous subsections, our findings also open other
intriguing avenues for future research.

       • Empirical studies should be expanded to include larger models, such as Vision Transformers
         (ViTs) and DenseNets, as well as more diverse datasets, to assess the broader generality of
         our findings. Our preliminary results on ViT are available in Section D.4.10, and largely
         confirm our findings also extend to Transformers.
       • Due to computational constraints, our study only analyzed NC properties in the last layer.
         However, previous works (Masarczyk et al., 2023; Rangamani et al., 2023) suggest that
         these properties may also manifest in intermediate layers. Investigating NC behavior across
         different depths could provide further insights into hierarchical feature representations.
       • In addition to the optimizers (SGD, Adam, AdamW, Signum) studied in this work, novel
         first-order methods such as Lion (Chen et al., 2023) and Mars (Yuan et al., 2024), and second-
         order methods, such as Shampoo (Gupta et al., 2018), SOAP (Vyas et al., 2024) and Muon
         (Jordan et al.) demonstrated promising improvements in convergence and generalization.
         However, their effects on NC remain largely unexplored.


5     C ONCLUSION

In this paper we have conducted an extensive number of experiments to elucidate the role of the
optimization algorithm in the emergence of the neural collapse (NC) phenomenon. In particular,
our experiments consistently show that coupled weight decay is necessary for achieving small NC
metrics. While the role of weight decay in the context of NC has been studied in the literature before,
this is the first paper distinguishing between coupled and decoupled weight decay. Moreover, our
theoretical results show that the resulting training dynamics differ considerably and one needs to take
this into account. These findings underscore the limitations of existing theoretical frameworks, which
have studied NC mainly under gradient flow or gradient descent, and highlight the need for further
investigation into the interplay between optimizers and NC.


                                                  10
```

## Page 011

```text
Published as a conference paper at ICLR 2026




ACKNOWLEDGMENTS
WO acknowledges that this research was partially funded by National Science Centre, Poland grant
no 2022/45/N/ST6/04098.

R EFERENCES
Mouïn Ben Ammar, Nacim Belkhir, Sebastian Popescu, Antoine Manzanera, and Gianni Franchi.
 Neco: Neural collapse based out-of-distribution detection, 2024.
Tina Behnia, Ganesh Ramachandra Kini, Vala Vakilian, and Christos Thrampoulidis. On the implicit
  geometry of cross-entropy parameterizations for label-imbalanced data. In Francisco Ruiz, Jennifer
  Dy, and Jan-Willem van de Meent (eds.), Proceedings of The 26th International Conference on
  Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research,
  pp. 10815–10838. PMLR, 25–27 Apr 2023.
Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signsgd:
   Compressed optimisation for non-convex problems. In International Conference on Machine
  Learning, pp. 560–569. PMLR, 2018.
Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong,
  Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V Le. Symbolic discovery of optimization algo-
  rithms. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances
  in Neural Information Processing Systems, volume 36, pp. 49205–49233. Curran Associates, Inc.,
  2023.
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
   hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition,
   pp. 248–255. Ieee, 2009.
Tomer Galanti, András György, and Marcus Hutter. On the role of neural collapse in transfer learning.
  arXiv preprint arXiv:2112.15121, 2021.
Connall Garrod and Jonathan P. Keating. The persistence of neural collapse despite low-rank bias:
  An analytic perspective through unconstrained features, 2024.
Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimiza-
  tion. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference
  on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1842–1850.
  PMLR, 10–15 Jul 2018.
X. Y. Han, Vardan Papyan, and David L. Donoho. Neural collapse under mse loss: Proximity to and
  dynamics on the central path, 2022.
Md Yousuf Harun, Jhair Gallardo, and Christopher Kanan. Controlling neural collapse enhances
 out-of-distribution detection and transfer learning, 2025.
Like Hui, Mikhail Belkin, and Preetum Nakkiran. Limitations of neural collapse for understanding
  generalization in deep learning. arXiv preprint arXiv:2202.08384, 2022.
Arthur Jacot, Peter Súkeník, Zihan Wang, and Marco Mondelli. Wide neural networks trained with
  weight decay provably exhibit neural collapse, 2024.
Wenlong Ji, Yiping Lu, Yiliang Zhang, Zhun Deng, and Weijie J Su. An unconstrained layer-peeled
 perspective on neural collapse. arXiv preprint arXiv:2110.02796, 2021.
Jiachen Jiang, Jinxin Zhou, Peng Wang, Qing Qu, Dustin Mixon, Chong You, and Zhihui Zhu.
   Generalized neural collapse for a large number of classes. arXiv preprint arXiv:2310.05351, 2023.
K Jordan, Y Jin, V Boza, Y Jiacheng, F Cecista, L Newhouse, and J Bernstein. Muon: An optimizer
  for hidden layers in neural networks, 2024b. URL https://kellerjordan. github. io/posts/muon.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
  arXiv:1412.6980, 2014.


                                                11
```

## Page 012

```text
Published as a conference paper at ICLR 2026




Vignesh Kothapalli. Neural collapse: A review on modelling principles and generalization, 2023.
Litian Liu and Yao Qin. Detecting out-of-distribution through the lens of neural collapse. arXiv
  preprint arXiv:2311.01479, 2023.
Xuantong Liu, Jianfeng Zhang, Tianyang Hu, He Cao, Yuan Yao, and Lujia Pan. Inducing neural
  collapse in deep long-tailed learning. In International conference on artificial intelligence and
  statistics, pp. 11534–11544. PMLR, 2023.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.
Wojciech Masarczyk, Mateusz Ostaszewski, Ehsan Imani, Razvan Pascanu, Piotr Miłoś, and Tomasz
 Trzciński. The tunnel effect: Building data representations in deep neural networks. In A. Oh,
 T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural
 Information Processing Systems, volume 36, pp. 76772–76805. Curran Associates, Inc., 2023.
Dustin G Mixon, Hans Parshall, and Jianzong Pi. Neural collapse with unconstrained features.
 Sampling Theory, Signal Processing, and Data Analysis, 20(2):11, 2022.
Kaouther Mouheb, Marawan Elbatel, Stefan Klein, and Esther E Bron. Evaluating the fairness of
  neural collapse in medical image classification. In International Conference on Medical Image
  Computing and Computer-Assisted Intervention, pp. 286–296. Springer, 2024.
Leyan Pan and Xinyuan Cao. Towards understanding neural collapse: The effects of batch normaliza-
  tion and weight decay, 2024.
Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal
  phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):
  24652–24663, 2020.
Akshay Rangamani, Marius Lindegaard, Tomer Galanti, and Tomaso A Poggio. Feature learning
  in deep classifiers through intermediate neural collapse. In Andreas Krause, Emma Brunskill,
  Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of
  the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine
 Learning Research, pp. 28729–28745. PMLR, 23–29 Jul 2023.
Christos Thrampoulidis, Ganesh Ramachandra Kini, Vala Vakilian, and Tina Behnia. Imbalance
  trouble: Revisiting neural-collapse geometry. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave,
  K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp.
  27225–27238. Curran Associates, Inc., 2022.
Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas
  Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam. arXiv preprint
  arXiv:2409.11321, 2024.
Robert Wu and Vardan Papyan. Linguistic collapse: Neural collapse in (large) language models.
  arXiv preprint arXiv:2405.17767, 2024.
Huizhuo Yuan, Yifeng Liu, Shuang Wu, Xun Zhou, and Quanquan Gu. Mars: Unleashing the power
  of variance reduction for training large models, 2024.
Zhihui Zhu, Tianyu Ding, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, and Qing Qu. A
  geometric analysis of neural collapse with unconstrained features. In M. Ranzato, A. Beygelzimer,
  Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing
  Systems, volume 34, pp. 29820–29834. Curran Associates, Inc., 2021.




                                                12
```

## Page 013

```text
Published as a conference paper at ICLR 2026




                                               Appendix

A     LLM USAGE STATEMENT

We disclaim that we have used Large Language Models to refine a few sentences and additionally as
a proxy of a search engine to retrieve additional related work.

The appendix is organized as follows. In Section B, we formally define the neural collapse (NC)
phenomenon and introduce the metrics used in the experiments presented in the main text. In
Section C, we review prior works related to our paper. Section D provides detailed descriptions and
additional observations from our experiments. In Section E, we present the full proof of the theorems
stated in the main text.


B    NC M ETRICS

Neural collapse (NC), discovered by Papyan et al. (2020), is a striking phenomenon observed during
the terminal phase of training (TPT) deep neural networks (DNN) for multi-class classification tasks,
particularly when trained with cross-entropy (CE) loss. Formally, let the (trained) last-layer features
of the DNN be denoted by hn , and concatenate them into a matrix H ∈ Rp×N , where p is the
width of the last layer and N is the number of training samples indexed by n. The output logits of
the network are then computed as WL H ∈ RK×N , where WL ∈ RK×p is the last-layer weight,
b ∈ RK is the bias vector, and K is the number of classes. 3
The DNN is trained using the CE loss computed on the logits:
                                                N
                                                                                   !
                                                X               exp(WL hn )yn
                          CE(WL , H) = −              log      PK                      ,
                                                n=1             k=1 exp(WL hn )k

                                                                                           def.
where yn ∈ [K] denotes the class label index of the feature vector hn . Let Ck = n ∈ [N ] : yn = k
be the index set of data points belonging to class k ∈ [K]. In this paper, we assume that the classes
are balanced, i.e., |Ck | is equal for all k ∈ [K]. For the effects of class imbalance on NC, we refer the
reader to Han et al. (2022); Thrampoulidis et al. (2022); Behnia et al. (2023).
         def.
Let µk = |C1k | n∈Ck hn be the class mean for each class k. The global mean of all classes is
                 P
                    1
                      PK
given by µG = K            k=1 µk and centered class means are defined as µ̄k = µk − µG . Let the
between-class covariance ΣB ∈ Rp×p and the within-class covariance ΣW ∈ Rp×p be:
                                           K
                                       1 X
                               ΣB =        µ̄k µ̄⊤
                                                 k,
                                       K
                                          k=1
                                                K     N
                                       1 1 XX k
                              ΣW =             (hn − µk )(hkn − µk )⊤ ,
                                       KN   n=1k=1


where hkn correspond to the feature vectors of class k.
                                                                       def.
We also concatenate the centered class means into a matrix M = (µ̄1 , ..., µ̄K ) ∈ Rp×K .
With these definitions in place, we now conceptually outline the NC properties and introduce
corresponding metrics to quantitatively measure these properties in our experiments.
    3
      For simplicity, we interchangeably refer to an input x ∈ Rd and its corresponding last-layer feature h ∈ Rp
after the parameters of the network have converged during TPT and the mapping x 7→ h is fixed.


                                                          13
```

## Page 014

```text
Published as a conference paper at ICLR 2026




NC1 - Variability Collapse The first property of neural collapse (NC1) describes the collapse of
features to their respective class means. Formally, this means that the distance between a feature
vector hn and its corresponding class mean µk approaches zero:
                               ∥hn − µk ∥2 → 0, ∀k ∈ [K], n ∈ Ck .
A corresponding metric is defined as Zhu et al. (2021); Kothapalli (2023); Ammar et al. (2024):
                                           def. 1
                                     N C1 =       Tr[ΣW Σ†B ]                                   (2)
                                                K
where † denotes the Moore-Penrose pseudo-inverse.

NC2 - Convergence of Class Means to Simplex ETF The second property of neural collapse
(NC2) describes the convergence of class means to a simplex equiangular tight frame (ETF), where
the angles between the means are maximally symmetric. Formally, this property can be expressed as:
                       (
                         ∥
                         Dµ̄j ∥2 − ∥µ̄k ∥E2 → 0
                             µ̄j      µ̄k       K         1    ∀j, k ∈ [K].
                           ∥µ̄j ∥ , ∥µ̄k ∥  → K−1 δjk − K−1  ,
                               2       2

To measure this property, we define two metrics capturing the equinormality and equiangularity of
the centered class means Papyan et al. (2020); Ammar et al. (2024):
                                 stdk {∥µ̄k ∥2 }
                        N C2n =                  ;                                                (3)
                                avgk {∥µ̄k ∥2 }
                                                            
                                                µ̄k    µ̄k′        1
                        N C2a = avgk̸=k′            ,          +      .                           (4)
                                             ∥µ̄k ∥2 ∥µ̄k ∥2
                                                         ′       K −1
Here, std• (·) and avg• (·) denote the standard deviation and mean, respectively, over the specified
index.
An alternative metric for NC2, introduced by Kothapalli (2023), directly measures the deviation of
the centered class means from a simplex ETF:
                                     def.   1     M⊤ M
                               N C2 =                    − M∗                                     (5)
                                            K2   ∥M⊤ M∥F      F
where                                                      
                                     def.1            1
                                   M∗ = √      I K − JK ,
                                        K −1         K
      K×K                                 K×K
IK ∈ R    is the identity matrix and J ∈ R    is the matrix of ones. Note that N C2n , N C2a →
0 ⇐⇒ N C2 → 0.

NC2W - Convergence of Weight Rows to Simplex ETF In addition to NC2, we define a related
property, NC2W, which describes the convergence of the rows of the last-layer weights WL ∈ RK×p
to a simplex ETF. If the third NC property, NC3 (described later), holds, then NC2 and NC2W are
equivalent. However, to study partial NC, it is essential to decouple these properties and measure
NC2 and NC2W separately.
To measure NC2W, Zhu et al. (2021) introduced the following metric:
                                                      ⊤
                                     def.   1    WL WL
                            N C2W =                  ⊤
                                                          − M∗           .                        (6)
                                            K2   WL WL  F            F

While this metric measures the overall alignment of WL with a simplex ETF, it does not account for
the equinormality and equiangularity of the rows of WL . To address this, we introduce the following
metrics:
                                    stdk {∥wk ∥2 }
                       N C2Wn =                                                                   (7)
                                   avgk {∥wk ∥2 }
                                                              
                                                 wk      wk′            1
                       N C2Wa = avgk̸=k′              ,          +                                (8)
                                                ∥wk ∥2 ∥wk′ ∥2       K −1
where wk⊤ ∈ Rp is the k-th row of WL .


                                                  14
```

## Page 015

```text
Published as a conference paper at ICLR 2026




NC2M - Convergence of Product to Simplex ETF Finally, Zhu et al. (2021); Kothapalli (2023)
proposed a metric that interpolates between NC2 and NC2W: 4
                                                   def.   1        WL M
                                 N C2M =                                  − M∗ .                             (9)
                                                          K2      ∥WL M∥F     F

Note that N C2, N C2W → 0 =⇒ N C2M → 0 but the converse does not hold.

NC3 - Convergence to Self-Duality The third property of neural collapse (NC3) describes that the
rows of the last-layer weight align with the column of the class means, that is,
                                             WL       M⊤
                                                   −                          → 0;
                                            ∥WL ∥F   ∥M⊤ ∥F               F

the corresponding metric is an obvious one Papyan et al. (2020); Garrod & Keating (2024):
                                            def.     1  WL       M⊤
                                  N C3 =                      −                                             (10)
                                                    Kp ∥WL ∥F   ∥M⊤ ∥F               F

NC4 - Simplification of Nearest-Class-Center (NCC) The fourth property of neural collapse
(NC4) describes that the classifier decision boundaries become equivalent to those derived by a
nearest-class-mean classifier, that is,
                                 arg max⟨wk , h⟩ → arg min ∥h − µk ∥2
                                      k

                                                                                         N test
for any test feature h ∈ Rp ; hence we can fix a test set of features {htest
                                                                        n }n=1 define the metric:
                                    test
                             N
                        1 X
                        def.
                 N C4 = test   1{arg max⟨wk , htest           test
                                               n ⟩ = arg min hn − µk 2 }                                    (11)
                       N n=1         k                  k

where 1 is the indicator function.
The above NC properties hold if their corresponding metrics approach zero (except for NC4, which
approach one) as the training step t → ∞. A solution WL , H satisfying these properties is referred
to as an NC solution.
To observe the interpolation between partial and full NC, we introduce a weaker property:

NC0 - Zero Row Sum of Last-Layer Weight This new property describes that the rows of the
last-layer weight WL sums up to zero with the corresponding metric
                                                           def.   1  ⊤
                                                   N C0 =           WL 1 2,                                 (12)
                                                                  p
Note that N C2W → 0 =⇒ N C0 → 0 but the converse does not hold.
The analogous property for the last-layer features, Zero Column Sum of Last-Layer Features,
holds automatically because the columns of M are centered class means:
                                           K
                                           X               K
                                                           X
                                                 µ̄k =            (µk − µG ) = 0.
                                           k=1             k=1


Thus, NC0 for the last-layer weights already represents a form of duality similar to NC3.




  4
    In the original works, this metric was used to evaluate self-duality. However, in this paper, we decouple the
NC properties to study the effects of implicit biases on each individually.


                                                                  15
```

## Page 016

```text
Published as a conference paper at ICLR 2026




C     A DDITIONAL R ELATED WORK

C.1   W EIGHT D ECAY AND N EURAL C OLLAPSE

Weight Decay has been shown to be essential for NC in prior works, like (Zhu et al., 2021; Pan &
Cao, 2024; Jacot et al., 2024). However, their statements on weight decay are for (quasi-)optimal
solutions in oversimplified models, which ignore the complex interaction between non-convex loss
landscape and optimizers. Please see Section C.5 for an example.

C.2   E MPIRICAL STUDIES ON THE E MERGENCE OF N EURAL C OLLAPSE

Neural collapse has also been studied beyond the original problem setting, which assumes few
balanced classes as well as noise-free labels. Notably, Wu & Papyan (2024) studied the occurrence
of NC for large language models, which do not satisfy any of the original assumption. Jiang et al.
(2023) studied neural collapse for a large number of classes, while Mouheb et al. (2024) studied the
influence of imbalanced in medical image classification on NC.

C.3   A PPLICATIONS OF N EURAL C OLLAPSE

The observation of neural collapse (NC) has inspired a growing body of follow-up work that applies
NC metrics across various settings. In the context of out-of-distribution (OOD) detection, Ammar
et al. (2024) propose a novel post-hoc detection method based on the geometric properties of NC,
while Harun et al. (2025) show that explicitly controlling for NC1 can enhance OOD detection
performance. Notably, the latter also claim that AdamW leads to NC, based on empirical results
where NC3 values hover around 0.5 across different models—mirroring the misleading metrics
reported in Table 3. As we demonstrate in the main text, however, this does not indicate true NC.
This discrepancy underscores the need for a more precise and systematic framework for evaluating
NC – one of the central contributions of this work.
In a separate line of inquiry, Liu et al. (2023) study the impact of class imbalance on NC and
propose explicit feature regularization terms to induce NC under imbalanced distributions, resulting
in improved model performance.

C.4   C OUPLED W EIGHT D ECAY IN THE CONTEXT OF N EURAL C OLLAPSE

To the best of our knowledge, no prior work has investigated the role of optimizer choice in the
context of NC. When minimizing the objective in Equation (1) or Equation (13), the weight decay
induced by the L2-regularization parameter λ is coupled with the training loss. However, with the
introduction of AdamW Loshchilov & Hutter (2019), decoupled weight decay has become the default
in many modern optimizers. This paper aims to bridge this gap by systematically examining the
impact of coupled versus decoupled weight decay on the emergence of NC.

C.5   U NCONSTRAINED F EATURE M ODEL

The unconstrained feature model (UFM) Mixon et al. (2022); Zhu et al. (2021) is a simplified
theoretical framework commonly used to study neural collapse (NC). In UFM, the last layer feature is
replaced by a trainable matrix H = (hn )N n=1 , referred to as the unconstrained feature, which mimics
the role of feature extraction layers in deep neural networks (DNN). For analytical simplicity, the
layer following the unconstrained feature is often assumed to be linear W, making UFM a special
case of deep linear networks (DLN):
                                 N
                                 X                      λ       λ
                           min         ℓ(Whn , yn ) +     ∥W∥2 + ∥H∥2 ,                          (13)
                           W,H
                                 n=1
                                                        2       2

simplifying the minimization problem in Equation (1). In this paper, the loss ℓ is always assumed to
be the cross-entropy (CE) loss, because it is the standard loss used in multi-classification tasks.
Zhu et al. (2021) has reported positive results on NC using UFM. Informally it holds that:


                                                  16
```

## Page 017

```text
Published as a conference paper at ICLR 2026




Theorem C.1 (Theorem 3.1 and 3.2 in Zhu et al. (2021)). Any global optimal solution of UFM is an
NC solution, while all other critical points are strict saddles. As a result, for random initialization, it
is almost surely that gradient descent finds an NC solution.

Zhu et al. (2021) also experimented NC on realistic models with optimizers like SGD and Adam,
concluding the universality of NC across different optimizers.




                                                    17
```

## Page 018

```text
Published as a conference paper at ICLR 2026




D     E XPERIMENT

The experiments of this work, particularly regarding computing the NC metrics, were based on
code in Wu & Papyan (2024), which can be found at Github repository https://github.
com/rhubarbwu/neural-collapse, which was published under the MIT license. The
implementation of VGG9 was based on Code taken from https://github.com/jerett/
PyTorch-CIFAR10. The author granted explicit permission to use the code.
An overview of the experiments that were conducted in this work can be found in Table 4, which
resulted in a total number of 36 different experimental settings of (architecture × optimizer × dataset)
combinations. Each optimizer optimizer was trained using three different learning rates, six different
values of momentum and six different values of weight decay, resulting in 108 training runs per
optimizer and 3.888 training runs in total. Some of the runs diverged or only achieved suboptimal
training performance, which were then discarded. In total we had 2.500 “valid” training runs, which
reached at least 99% training accuracy, which were considered for for the subsequent data analysis.

                      Table 4: Overview of experiments conducted in this work.

      Architectures        Optimizers                        Datasets
                              SGD, SGDW, Adam,
      ResNet9, VGG9                                          MNIST, FashionMNIST, CIFAR10
                           AdamW, Signum, SignumW



D.1    D ETAILS ON C HOICE OF H YPERPARAMETERS

Every model was trained over 200 epochs with a batch size of 128. The learning rate λ was chosen to
be in λ ∈ {0.001, 0.01, 0.0679} for SGD and SGDW (the last learning rate was also reported in the
original work by Papyan et al. (2020)) and λ ∈ {0.001, 0.005, 0.01} for Adam, AdamW, Signum, and
SignumW because most trainings diverged with larger learning rates during initial experimental train-
ing runs. The learning rate was decayed by a factor of 10 after one third and two third of training as has
been done in original work by Papyan et al. (2020). Momentum µ (or β1 for Adam, AdamW, Signum,
and SignumW) was chosen to be in the range µ ∈ {0, 0.5, 0.7, 0.9, 0.95, 0.98} for all optimizers and
weight decay WD was chosen to be in the range WD ∈ {0, 5e−5 , 5e−4 , 5e−3 , 0.05, 0.5} for SGD,
SGDW, Adam, and Signum and WD ∈ {0, 5e−4 , 0.05, 0.5, 5, 10} for SignumW and AdamW. The
main motivation for using AdamW and Signum W with much larger weight decay values was based
on the hypothesis that the effect of weight decay is reduced due to decoupling. The β2 parameter in
Adam and AdamW was left to its default value of 0.999.


D.2    D ETAILS ON C OMPUTATIONAL R ESOURCES

All experiments, including preliminary experiments as well as the final 3.888 experiments were
run on 5 NVIDIA RTX4090 GPUs with 24 GB RAM. Since the models and the batch size was
comparably small, actually only 3 GB GPU memory per training was required. Each training took
between 8 and 16 minutes, leading to a total of 500-1000 GPU hours of training.


Table 5: Hyperparameters for each optimizer to achieve the smallest NC3 metric shown in Table 3.

                    Optimizer     Learning rate    Momentum/β1          Weight decay
                    SGD                    0.01                0.9              0.05
                    SGDW                0.0679                 0.5              0.05
                    Adam                 0.005                0.98              0.05
                    AdamW                0.005                0.95                 5
                    Signum               0.001                 0.9              0.05
                    SignumW              0.001                0.98                10


                                                   18
```

## Page 019

```text
Published as a conference paper at ICLR 2026




                       Table 6: Summary of regression fit between NC3 and NC0

 Experiment       n         β̂   SE(β̂)       t-value   p-value         95 % CI      R2 / Adj R2      F-statistic
 LR=0.001      170     0.1903    0.008        24.262     0.000     [0.175, 0.206]   0.778 / 0.777          588.6
 LR=0.005       74     0.2017    0.012        16.252     0.000     [0.177, 0.226]   0.786 / 0.783          264.1
 LR=0.01       114     0.1439    0.007        19.892     0.000      [0.13, 0.158]   0.779 / 0.777          395.7
 LR=0.0679      41     0.1771    0.012        14.367     0.000     [0.152, 0.202]   0.841 / 0.837          206.4
 all           399     0.1582    0.005        32.760     0.000     [0.149, 0.168]   0.730 / 0.729           1073



D.3     D ETAILS ON R EGRESSION F IT BETWEEN NC3 AND NC0

In this subsection we provide additional details regarding the regression fit between NC3 and NC0.
For the sake of completeness, we show the regression fit in Figure 10 again below. In addition, we
have also computed a regression fit across all training runs, which converged, and all learning rates,
shown in Figure 11. A summary of the regression fit can be found in Table 6, showing that more than
70% of the variation in NC3 can be explained by NC0.

          lr = 0.001             lr = 0.005               lr = 0.01            lr = 0.0679          optimizer
   1                                                                                                  AdamW
                                                                                                      Adam
NC3




                                                                                                      SGDW
   0                                                                                                  SGD
                                                                                                      SignumW
          10 2 102               10 2 102                10 2 102              10 2 102               Signum
             NC0                    NC0                     NC0                   NC0
Figure 10: Figure 2 shown again for ease of reading. NC0 weakly correlates with NC3 across
different optimizers and learning rates (here shown for ResNet9 trained on FashionMNIST).



                                                                            AdamW
                         1.0                                                Adam
                                                                            SGDW
                      NC3




                         0.5                                                SGD
                                                                            SignumW
                         0.0                                                Signum
                                     10 3 100                103
                                          NC0
Figure 11: NC0 correlates with NC3 even when considered across all learning rates together (here
shown for ResNet9 trained on FashionMNIST).


D.4     A DDITIONAL E XPERIMENTAL R ESULTS

D.4.1    A BLATION S TUDY ON T RAINING E POCHS

As Neural collapse occurs at the terminal phase of training, it is natural to control for the effect that
the number of training epochs has on the final NC metrics. After all, it is possible that the emergence
of NC occurs at different speeds for different optimizers.
For this reason, we conducted two ablation studies, in which we prolong the training in two settings:
We train a ResNet9 in FashionMNIST, which corresponds to the setting which is shown in Figure 1,
for 2000 epochs with LR=0.0005 and momentum=0.9 for both optimizers. We note that in this setting,
AdamW reaches 100% training accuracy already after around 700 epochs for all training runs with
WD ≤ 0.05. The results can be found in Figure 14 While this leads to some improvement of the final


                                                        19
```

## Page 020

```text
Published as a conference paper at ICLR 2026




                               lr = 0.001                                          lr = 0.005                                                 lr = 0.01                                    lr = 0.0679
       103
       102
                                                                                                                                                                                                                            optimizer
       101                                                                                                                                                                                                                    AdamW
NC1                                                                                                                                                                                                                           Adam
       100                                                                                                                                                                                                                    SGDW
                                                                                                                                                                                                                              SGD
      10 1                                                                                                                                                                                                                    SignumW
                                                                                                                                                                                                                              Signum
      10 2
                     10 4 10 2 100              102        104               10 4 10 2 100           102       104                10 4 10 2 100            102     104         10 4 10 2 100                102   104
                                  NC0                                                      NC0                                                  NC0                                              NC0

Figure 12: NC0 vs. NC1 across different optimizers and learning rates (here shown for ResNet9
trained on FashionMNIST).


      1.2
                              lr = 0.001                                          lr = 0.005                                                  lr = 0.01                                    lr = 0.0679
      1.0
                                                                                                                                                                                                                            optimizer
      0.8                                                                                                                                                                                                                     AdamW
                                                                                                                                                                                                                              Adam
NC2




      0.6                                                                                                                                                                                                                     SGDW
                                                                                                                                                                                                                              SGD
      0.4                                                                                                                                                                                                                     SignumW
      0.2                                                                                                                                                                                                                     Signum

                    10 4 10 2 100              102        104            10 4 10 2 100               102       104                10 4 10 2 100            102    104          10 4 10 2 100                102   104
                                NC0                                                    NC0                                                      NC0                                              NC0

Figure 13: NC0 vs. NC2 across different optimizers and learning rates (here shown for ResNet9
trained on FashionMNIST).



NC1 and NC2 metric for AdamW for some values of weight decay, this has barely an effect on NC0
and NC3.
Furthermore we extend training to up to 2000 epochs for selected runs from Figure 5. Concretely,
these runs trained with a LR of 0.001 and the following combination of WD and momentum (mom,
WD) ∈ {(0, 0), (0.97, 5e−5 ), (0, 5e−4 ), (0.9, 5e−4 ), (0.9, 0), (0.95, 0.0025)}, which corresponds to
different parts in the heatmap. The results can be found in Figure 15. While one can observe a
general decrease of the NC metrics in all cases, the overall trend for increasing weight decay remains
unchanged. Both figures indicate that training the models considered in this work for 200 epochs is
sufficient to draw the conclusions that we make about the necessity of coupled WD for the emergence
of full NC.

                                                                       1.0                                                                                                                 0.8
      10
            1                                                                                                                       0.3
                                                                       0.8                                                                                                                 0.6
NC0




                                                                 NC3




                                                                                                                              NC1




                                                                                                                                                                                     NC2




      10
            0
                                                                       0.6                                                          0.2
                                                                                                                                                                                           0.4
            1
      10                                                               0.4                                                          0.1                                                    0.2
                0    10
                          4      3
                               10 10 10
                                           2          1
                                                            10
                                                                 0           0    10
                                                                                       4         3
                                                                                            10 10 10
                                                                                                           2         1
                                                                                                                         10
                                                                                                                              0           0      10
                                                                                                                                                      4      3
                                                                                                                                                           10 10 10
                                                                                                                                                                       2   1
                                                                                                                                                                                10
                                                                                                                                                                                     0           0     10
                                                                                                                                                                                                            4      3    2
                                                                                                                                                                                                                 10 10 10
                                                                                                                                                                                                                                 1
                                                                                                                                                                                                                                     10
                                                                                                                                                                                                                                          0
                              weight_decay                                                 weight_decay                                                   weight_decay                                          weight_decay
                                                                 optimizer                 AdamW                     Adam                 epochs                 200           2000


            Figure 14: ResNet9 trained on FashionMNIST with Adam and AdamW for more epochs.



D.4.2                U NCONSTRAINED F EATURE M ODEL

We also validated our results on the unconstrained feature model (UFM) (see Section C.5 for reference)
with width d = 512, K = 10 classes and N = 10.000 samples. The UFM was trained with Adam,
AdamW and SGDMW with momentum=0.9 and varying lr∈ {0.1, 0.3, 0.5, 1.0} and weight decay
ranging from 0.0 to 0.05. We then filtered the results, by only including models which achieved 100%
training accuracy. The results in can be found in Figure 16. The plots show that the NC metrics, in
particular NC0 and NC3 remain at least one magnitude of order larger than the same metrics for
Adam and SGDMW, highlighting that AdamW converges to a different solution than Adam, which is
not NC.


                                                                                                                         20
```

## Page 021

```text
Published as a conference paper at ICLR 2026




                                                                                                                                         0.15                                                              0.6
                 1                                                         0.6
           10                                                                                                                            0.10                                                              0.4
 NC0




                                                                   NC3




                                                                                                                                 NC1




                                                                                                                                                                                                     NC2
                                                                           0.4
                 3                                                                                                                       0.05                                                              0.2
           10                                                              0.2
                      0             10
                                        4
                                              10
                                                     3
                                                              10
                                                                       2         0       10
                                                                                              4
                                                                                                   10
                                                                                                               3
                                                                                                                            10
                                                                                                                                     2            0          10
                                                                                                                                                                4
                                                                                                                                                                       10
                                                                                                                                                                              3
                                                                                                                                                                                                10
                                                                                                                                                                                                     2           0           10
                                                                                                                                                                                                                                    4
                                                                                                                                                                                                                                       10
                                                                                                                                                                                                                                                 3
                                                                                                                                                                                                                                                           10
                                                                                                                                                                                                                                                                2
                                    weight_decay                                         weight_decay                                                        weight_decay                                                    weight_decay
                                                                                     epochs                200                           2000            optimizer                SGDMW


                                    Figure 15: Selected runs from Figure 5 trained for more number of epochs.

                                                                       0                                                                                                                        0
                                                                  10                                                             1                                                         10
                                                                                                                            10                                                                                                                            optimizer
      10
           1
                                                                                                                                 0                                                                                                                        AdamW
                                                                  10
                                                                       1                                                    10                                                             10
                                                                                                                                                                                                1                                                         Adam
                                                                                                                                 1                                                                                                                        SGDMW
                                                                                                                            10
NC0




                                                          NC3




                                                                                                                    NC1




                                                                                                                                                                                   NC2
           1
      10                                                               2                                                                                                                                                                                  lr
                                                                  10                                                        10
                                                                                                                                 2
                                                                                                                                                                                           10
                                                                                                                                                                                                2                                                         0.1
           3                                                                                                                                                                                                                                              0.3
      10                                                               3                                                    10
                                                                                                                                 3
                                                                                                                                                                                                                                                          0.5
                                                                  10                                                                                                                            3                                                         1.0
                                                                                                                                                                                           10
                 0 10 5         10
                                    4
                                        10
                                            3
                                                10
                                                     2
                                                         10
                                                              1            0 10 5    10
                                                                                        4
                                                                                             10
                                                                                                  3
                                                                                                      10
                                                                                                           2
                                                                                                                   10
                                                                                                                        1                0 10 5       10
                                                                                                                                                         4
                                                                                                                                                              10
                                                                                                                                                                 3
                                                                                                                                                                     10
                                                                                                                                                                          2
                                                                                                                                                                                  10
                                                                                                                                                                                       1             0 10 5          10
                                                                                                                                                                                                                        4
                                                                                                                                                                                                                             10
                                                                                                                                                                                                                                3
                                                                                                                                                                                                                                        10
                                                                                                                                                                                                                                             2
                                                                                                                                                                                                                                                 10
                                                                                                                                                                                                                                                      1
                                weight_decay                                         weight_decay                                                     weight_decay                                                   weight_decay


Figure 16: NC0 (left), NC3 (center left), NC1 (center right), and NC2 (right) for increasing weight
decay.


D.4.3                         T RAINING DYNAMICS OF MINIMAL NC3 RUNS

In this section we provide the dynamics of the NC metrics as well as the singular values from the
training runs which reached the smallest final NC3 metric as reported in Table 5. The purpose is to
disentangle the effect of using first-order optimizers (such as SGD and SGDW) vs. second-order like
optimizers (such as Adam and AdamW) from the effect of applying coupled vs. decoupled weight
decay. The main question we try to answer here is: Is the difference between Adam and SGD with
respect to the emergence of NC larger than the difference between AdamW and Adam? Figure 17
shows that all runs reach a perfect train accuracy well before the end of training, such that they have
reached the terminal-phase of training (TPT) at epoch 200. Looking at the NC1-NC3 metrics in
Figure 18 and Figure 20, one can see that the NC metrics for SGD and SGDW are close to each other.
It is harder to judge whether AdamW or Adam are closer to NC, as NC3 is considerably larger for
AdamW, while NC1 is slightly larger for Adam, compared to the other optimizers. Nonetheless, the
NC0 metric in Figure 18 and the evolution of the singular values of W in Figure 19 (left) indicate that
AdamW has considerably different training dynamics than Adam, as both NC0 as well as the smallest
singular value increase instead of converging to zero for AdamW, but not for Adam. While the NC0
metric of Adam is still orders of magnitude larger than for SGD and SGDW and the smallest singular
value of W converges to a small, but non-zero value, Adam shares similar trends as SGD and SGDW
and as such converges to a solution which is arguably closer to NC3 than AdamW. Whether the
solution found by Adam can already be classified as NC or not is an inherent problem of interpreting
the NC metrics in practical settings, as we have also discussed in Section 4.1.


                                                                                      SGD                          SGDW                                 Adam                               AdamW
                             101                                                                                                             1.00
                             10 1
                                                                                                                                             0.95
                train_loss




                             10 3
                                                                                                                                 train_acc




                                                                                                                                             0.90
                             10 5
                             10 7                                                                                                            0.85
                                                                                                                                             0.80
                                        0                50                  100              150                       200                              0                50                         100                    150                  200
                                                                            epoch                                                                                                                   epoch
Figure 17: Train loss (left) and train accuracy (right) for training runs with smallest final NC3 metric
for different optimizers.


                                                                                                                             21
```

## Page 022

```text
Published as a conference paper at ICLR 2026




                                                     SGD          SGDW                      Adam          AdamW
                         101

                        10 1                                                         1.0
            NC0




                                                                         NC3
                        10 3
                                                                                     0.5
                        10 5
                                                                                     0.0
                               0      50      100          150    200                       0        50      100    150     200
                                             epoch                                                          epoch
Figure 18: NC0 (left) and NC3 (right) for training runs with smallest final NC3 metric for different
optimizers.
 Singular values of W




                    2                                               Singular values of M
                                                                                       30
                                                                                       20
                    1                                                                  10
                                                                                        0
                    0
                           0                100                  200                            0            100            200
                                           Epoch                                                            Epoch
                                   SGD             Adam                                             SGD             Adam
                                   SGDW            AdamW                                            SGDW            AdamW
Figure 19: Singular values of last-layer weights W (left) and centered class means M (right)
throughout training for runs corresponding to Table 5. The dotted line corresponds to the smallest
singular value and the full line corresponds to the average singular value, excluding the smallest
singular value.



                                                     SGD          SGDW                      Adam          AdamW
                        0.10
                        0.08                                                        1.0
                                                                                    0.8
                        0.06
                                                                                    0.6
            NC1




                                                                        NC2




                        0.04
                                                                                    0.4
                        0.02                                                        0.2
                               0      50      100          150    200                       0        50      100    150     200
                                             epoch                                                          epoch
Figure 20: NC1 (left) and NC2 (right) for training runs with smallest final NC3 metric for different
optimizers.



                                                                    22
```

## Page 023

```text
Published as a conference paper at ICLR 2026




            101
                                                                           10 1




                                                          NC0_normalized
           10 1
                                                                           10 3
    NC0    10 3

           10 5                                                            10 5

                    0     50          100      150     200                                0          50      100        150         200
                                     epoch                                                                  epoch
Figure 21: NC0 (left) and normalized NC0 (right) for training runs with smallest final NC3 metric
for different optimizers.

                    mom        0.0       0.7     0.9         0.95                         0.98        n_epochs        200         2000

           1.7544                                                                  100




                                                                  NC0_normalized
           1.7542
     NC0




           1.7540

           1.7538
                                                                                   10 1
           1.7536
                          0    100       101    102      103                                     0    100       101         102    103
                                      epoch                                                                  epoch


Figure 22: NC0 (left) and normalized NC0 (right) for training runs with zero weight decay from the
ablation study in Section D.4.6. Note that the x-axis is in logarithmic scale and that the point at epoch
-1 corresponds to the model at initialization.


D.4.4       A BLATION STUDY ON NORMALIZING THE NC0 METRIC
We evaluate whether measuring a normalized NC0 metric affects the conclusions that we draw in our
work. Concretely, we compute the normalization as
                                                         1   ⊤
                                      NC0normalized :=     ∥WL 1∥2 /∥WL ∥F .                                                              (14)
                                                         p
We compute both NC0 as well as normalized NC0 for the setting of minimal NC3 that we studied in
Section D.4.3, which we show in Figure 21. While the absolute values differ slightly between NC0
and NC0normalized , both the trends as well as the final values are almost the same.

For zero weight decay, one would expect to see more difference between the dynamics of
NC0 and normalized NC0, which we show in Figure 22. While one can observe the monotontic
effect of momentum on normalized NC0, but not on NC0, we point out that in this case normalized
NC0 does not correlate with NC1-NC3 anymore. On the contrary, NC1-NC3, while still comparably
large, are smaller with less momentum.
As the dynamics of NC0 and normalized NC0 are almost the same for larger values of WD or
normalized NC0 is not consistent with NC1-NC3 for zero WD, we are tentative to conclude that the
normalization will not affect the conclusions that we draw in this work.

D.4.5       A BLATION STUDY ON EFFECT OF MOMENTUM ON NC EMERGENCE
We conduct another ablation study to further evaluate the effect of momentum on the NC emergence.
The main question that we try to answer with this ablation is whether the effect of momentum on
smaller NC metrics can be simply traced back to the fact that momentum accelerates convergence or
if it affects the emergence of NC beyond this.
Concretely, we track the evolution of the NC metrics together with the train loss and accuracy for the
same setting as in Figure 5 over time for a fixed value fo weight decay=0.005 and varying values
of momentum. This is because the final train loss value varies for different values of WD due to


                                                          23
```

## Page 024

```text
Published as a conference paper at ICLR 2026




                                               0.0                1.00
              100                              0.5                                              mom
                                               0.7                0.95                            0.0
 train_loss                                    0.9                                                0.5




                                                      train_acc
              10 1                             0.95                                               0.7
                                               0.97               0.90
                                                                                                  0.9
                                               0.98                                               0.95
              10 2                                                0.85                            0.97
                                                                                                  0.98
                                                                  0.80
                     0      50    100    150     200                     0   50    100    150       200
                                 epoch                                            epoch
              100                                                                                 0.0
                                                                  1.25                            0.5
                         0.0                                                                      0.7
              10 1                                                1.00                            0.9
                         0.5
                                                                                                  0.95
 NC0




                                                      NC3
                         0.7                                      0.75
              10 2       0.9                                                                      0.97
                         0.95                                     0.50                            0.98
              10 3       0.97
                         0.98                                     0.25
                     0      50    100    150     200                     0   50    100    150       200
                                 epoch                                            epoch
                                               0.0                 1.0                            0.0
                                               0.5                                                0.5
                                               0.7                 0.8                            0.7
              100                              0.9                                                0.9
                                               0.95                0.6                            0.95
 NC1




                                                         NC2




                                               0.97                                               0.97
              10 1                             0.98                0.4                            0.98

                                                                   0.2
              10 2
                     0      50    100    150     200                     0   50    100    150       200
                                 epoch                                            epoch
Figure 23: Train loss, train accuracy and NC metrics for fixed WD=0.005 and different values of
momentum on a ResNet9 trained with SGD with otherwise same hyperparameters as in Figure 5.


its regularizing effect. The results can be seen in Figure 23. There are two things to be observed:
While the accelerating effect of momentum is mainly visible in the early phase of training (up to
50-100 epochs), modulo some loss spikes for high momentum, the final train loss is not smallest for
the largest value of momentum. While this is not surprising per se, as too large momentum can lead
to a overshooting of the training trajectory, the NC metrics show a clear monotonic behavior with
respect to the momentum. Furthermore, while the training runs with momentum=0.7 and 0.9 reach
almost the exact same final train loss, the disparity in NC metrics indicates that they converged to
solutions with very different geometric structure. This can be seen more clearly in Figure 24. Both
observations suggest that momentum affects the emergence of NC beyond simply accelerating the
speed of convergence. To the best of our knowledge, connecting the magnitude of momentum to NC
is novel and not been discussed in prior work.

D.4.6            A BLATION STUDY ON NC EMERGENCE UNDER ZERO WEIGHT DECAY

To investigate whether WD is necessary or not for the emergence of NC, we track the NC metrics while
training a ResNet9 on FashionMNIST (Note that this is the same problem setting as in Section D.4.5.)
using SGD with zero WD and varying values of momentum with an initial LR=0.01 for 200 epochs.
Additionally, we train the model also with zero momentum and high momentum=0.98 for 2000
epochs, with LR decay after 1/3 and 2/3 of training. Importantly, all training runs reach perfect
train accuracy after 40 epochs. The training dynamics can be found in Figure 25. We draw two
conclusions from this ablation study:


                                                      24
```

## Page 025

```text
Published as a conference paper at ICLR 2026




                                               0.7               1.00
                 100                           0.9
                                                                 0.95
    train_loss




                                                     train_acc
                 10 1                                            0.90
                 10 2                                                                          mom
                                                                 0.85                            0.7
                                                                                                 0.9
                                                                 0.80
                        0   50    100    150   200                      0   50    100    150      200
                                 epoch                                           epoch
                                               0.7                1.4                            0.7
                                               0.9                1.2                            0.9
                 100
                                                                  1.0
 NC0




                                                        NC3




    6 × 10 1                                                      0.8
    4 × 10 1                                                      0.6
    3 × 10 1                                                      0.4
                        0   50    100    150   200                      0   50    100    150      200
                                 epoch                                           epoch
                                               0.7                                               0.7
                                               0.9                0.8                            0.9
                 100                                              0.7
    NC1




                                                        NC2




                                                                  0.6
                 10 1                                             0.5
                                                                  0.4
                        0   50    100    150   200                      0   50    100    150      200
                                 epoch                                           epoch
Figure 24: Train loss, train accuracy and NC metrics for fixed WD=0.005 and mom=0.7 and 0.9.
Although both runs converge to almost exactly the same train loss, the final NC metrics differ
considerably.




                                                25
```

## Page 026

```text
Published as a conference paper at ICLR 2026




Table 7: Smallest NC metrics achieved with and without weight decay for training a ResNet9 on
FashionMNIST.

                           SGD with ...          NC1       NC2        NC3
                       no WD (2000 epochs)      ≈ 0.2     ≈ 0.55     ≈ 0.7
                         WD (200 epochs)        ≈ 0.02    ≈ 0.2      ≈ 0.13


     1. The final NC metrics NC0-NC3 after 2000 epochs are slightly smaller than after 200 epochs,
        consistent with our ablation study in D.4.1. that longer training reduces the NC metrics.
        This decrease is however fairly small.
     2. The final NC metrics (both for 200 epochs and 2000 epochs of training) remain considerably
        higher than what is achieved by the "best" run of SGD in terms of NC metrics with 200
        epochs of training for all NC metrics, even with 10 times longer training. See Figure 18 and
        Figure 20 for a comparison.

The final NC1-NC3 metrics achieved with WD after 200 epochs and without WD after 2000 epochs
can be found in Table 7. While the experiments cannot fully exclude the possibility that NC can be
achieved eventually in the asymptotic limit, we argue that WD is essential to observe the emergence
of NC in practical finite-length training settings.

D.4.7   M ORE DETAILED PLOTS ON COUPLED VS . DECOUPLED WEIGHT DECAY
As we average across different values of momentum and learning rates in Figure 1 and Figure 3, we
provide more detailed plots here in Figure 26, Figure 27, and Figure 28. It can be seen that for the
adaptive optimizers and SGDW the variance for varying values of momentum is comparably small
for each fixed learning rate, with the variance generally increasing with larger weight decay. For
SGD the variance for NC0 is higher for large values of weight decay, consistent with what is shown
in Figure 3 (right) and what is shown in Figure 5.

D.4.8   M ISSING PLOT: S INGULAR VALUE OF W AND M WITH S IGNUM W
The missing plot of the evolution of the singular values of the last-layer weights W and feature
matrix M can be found in Figure 29.

D.4.9   C OUPLED VS . DECOUPLED DECAY ON OTHER DATASETS
The comparison between coupled and decoupled decay on SGD, Adam, and Signum on other
combinations of (architecture × dataset) can be found in the following pages below, which confirm
our observations made earlier on the ResNet9 trained on FashionMNIST. While NC0 (visually)
correlates well with NC3, it correlates considerably less with NC1 and NC2, although a general trend
is still visible across all experiments.

ResNet50 on ImageNet1K We also conducted experiments on a ResNet50 trained on ImageNet1K
Deng et al. (2009). The model was trained with Adam and AdamW for 90 epochs. We left out other
optimizers due to limited resources. For both optimizers the learning rate was chosen as 0.0003
with a step-wise decay after 1/3 and 2/3 of training, momentum was chosen from {0.0, 0.5, 0.9} and
weight decay was chosen from {0.0, 1e−5 , 1e−4 , 1e−3 }. The resulting NC metrics can be found in
Figure 30 and Figure 31, and confirm the conclusion that AdamW does not have full NC emergence.

VGG9 on FashionMNIST The comparison between coupled and decoupled weight decay on SGD,
Adam, and Signum on a VGG9 trained on FashionMNIST can be found in Figure 32 and Figure 33.
The relation between NC0 and NC3 can be found in Figure 36, between NC0 and NC1 in Figure 34,
and between NC0 and NC2 in Figure 35.

ResNet9 on Cifar10 The comparison between coupled and decoupled weight decay on SGD,
Adam, and Signum on a ResNet9 trained on Cifar10 can be found in Figure 37 and Figure 38. The
relation between NC0 and NC3 can be found in Figure 41, between NC0 and NC1 in Figure 39, and
between NC0 and NC2 in Figure 40.


                                                26
```

## Page 027

```text
Published as a conference paper at ICLR 2026




                             mom    0.0    0.7      0.9     0.95                 0.98   n_epochs   200      2000
                                                                               1.000
                      100
                                                                               0.975
                      10 1                                                     0.950

                      10 2                                                     0.925
         train_loss




                                                                   train_acc
                                                                               0.900
                      10 3
                                                                               0.875
                      10 4                                                     0.850
                      10 5                                                     0.825
                                                                               0.800
                                   0 100      101     102    103                           0 100      101    102   103
                                           epoch                                                   epoch
                             mom    0.0    0.7      0.9     0.95                 0.98   n_epochs   200      2000

    1.7544 × 100                                                                 1.4
    1.7543 × 100                                                                 1.3
    1.7542 × 100                                                                 1.2
    1.7541 × 100
                                                                                 1.1
 NC0




                                                                         NC3




       1.754 × 100
                                                                                 1.0
    1.7539 × 100
                                                                                 0.9
    1.7538 × 100
                                                                                 0.8
    1.7537 × 100
                                                                                 0.7
    1.7536 × 100
                                   0 100      101     102    103                           0 100      101    102   103
                                           epoch                                                   epoch
                             mom    0.0    0.7      0.9     0.95                 0.98   n_epochs   200      2000
                       1.0
                                                                                 1.1
                       0.8
                                                                                 1.0
                       0.6                                                       0.9
             NC1




                                                                         NC2




                       0.4                                                       0.8

                                                                                 0.7
                       0.2
                                                                                 0.6
                       0.0
                                   0 100      101     102    103                           0 100      101    102   103
                                           epoch                                                   epoch

Figure 25: Training loss and train accuracy (top row), NC0 and NC3 (middle row), and NC1 and
NC2 (bottom row) for a ResNet9 trained on FashionMNIST with SGD without weight decay for
varying values of momentum and number of epochs. Note that the x-axis is in log-scale to improve
readability.




                                                             27
```

## Page 028

```text
Published as a conference paper at ICLR 2026




                           lr = 0.001                         lr = 0.005                          lr = 0.01
                                                                                                                           optimizer
           101                                                                                                                AdamW
                                                                                                                              Adam
                                                                                                                           momentum
           100
    NC0



                                                                                                                              0.0
                                                                                                                              0.5
                                                                                                                              0.7
          10 1                                                                                                                0.9
                                                                                                                              0.95
                                                                                                                              0.98
                  010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101
                          weight_decay                      weight_decay                       Weight decay
                          lr = 0.001                         lr = 0.005                         lr = 0.01
          1.0                                                                                                             optimizer
                                                                                                                              AdamW
          0.8                                                                                                                 Adam
                                                                                                                          momentum
    NC3




          0.6                                                                                                                 0.0
                                                                                                                              0.5
          0.4                                                                                                                 0.7
                                                                                                                              0.9
          0.2                                                                                                                 0.95
                                                                                                                              0.98
                 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101
                         weight_decay                       weight_decay                       Weight decay

Figure 26: NC0 metric (top) and NC3 metric (bottom for different values of weight decay, momentum
and LR for Adam vs. AdamW.




                           lr = 0.001                        lr = 0.005                          lr = 0.01
           104                                                                                                            optimizer
                                                                                                                             SignumW
           103                                                                                                               Signum
           102
                                                                                                                          momentum
    NC0




                                                                                                                             0.0
           101                                                                                                               0.5
                                                                                                                             0.7
           100                                                                                                               0.9
                                                                                                                             0.95
          10 1                                                                                                               0.98
                  010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101
                          weight_decay                      weight_decay                      Weight decay
                          lr = 0.001                         lr = 0.005                        lr = 0.01
          1.4
                                                                                                                          optimizer
          1.2                                                                                                                SignumW
                                                                                                                             Signum
          1.0                                                                                                             momentum
    NC3




          0.8                                                                                                                0.0
                                                                                                                             0.5
          0.6                                                                                                                0.7
                                                                                                                             0.9
          0.4                                                                                                                0.95
                                                                                                                             0.98
                 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101
                        weight_decay                       weight_decay                       Weight decay

Figure 27: NC0 metric (top) and NC3 metric (bottom for different values of weight decay, momentum
and LR for Signum vs. SignumW.




                                                                    28
```

## Page 029

```text
Published as a conference paper at ICLR 2026




                                           lr = 0.001                       lr = 0.01                                  lr = 0.0679
           100                                                                                                                              optimizer
          10 1                                                                                                                                 SGDW
                                                                                                                                               SGD
          10 2                                                                                                                              momentum
    NC0   10 3                                                                                                                                 0.0
                                                                                                                                               0.5
          10 4                                                                                                                                 0.7
                                                                                                                                               0.9
          10 5                                                                                                                                 0.95
                                                                                                                                               0.98
                 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101
                                          weight_decay                  weight_decay                                  Weight decay
                                           lr = 0.001                     lr = 0.01                                    lr = 0.0679
           100                                                                                                                              optimizer
                                                                                                                                               SGDW
                                                                                                                                               SGD
                                                                                                                                            momentum
    NC3




                                                                                                                                               0.0
                                                                                                                                               0.5
                                                                                                                                               0.7
                                                                                                                                               0.9
          10 1                                                                                                                                 0.95
                                                                                                                                               0.98
                 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101 010 5 10 4 10 3 10 2 10 1 100 101
                                          weight_decay                  weight_decay                                  Weight decay

Figure 28: NC0 metric (top) and NC3 metric (bottom for different values of weight decay, momentum
and LR for SGD vs. SGDW.

                                      600                                                            200
                                                                       SignumW                                                   SignumW
                   Singular values of W




                                                                                  Singular values of M



                                      400                                                            100
                                      200                                                                0
                                           0
                                               0         50    100    150      200                           0   50    100      150   200
                                                              Epoch                                                   Epoch

Figure 29: Singular values of last-layer weights W (left) and feature matrix M (right) throughout
training for SignumW on ResNet9 trained on FashionMNIST. Dotted line corresponds do smallest
singular value and full line corresponds to the average singular value excluding the smallest singular
value.


VGG9 on Cifar10 The comparison between coupled and decoupled weight decay on SGD, Adam,
and Signum can be found in Figure 42 and Figure 43. The relation between NC0 and NC3 can be
found in Figure 46, between NC0 and NC1 in Figure 44, and between NC0 and NC2 in Figure 45.

ResNet9 on MNIST The comparison between coupled and decoupled weight decay on SGD,
Adam, and Signum on a ResNet9 trained on MNIST can be found in Figure 47 and Figure 48. The
relation between NC0 and NC3 can be found in Figure 51, between NC0 and NC1 in Figure 49, and
between NC0 and NC2 in Figure 50.

VGG9 on MNIST The comparison between coupled and decoupled weight decay on SGD, Adam,
and Signum on a VGG9 trained on MNIST can be found in Figure 52 and Figure 53. The relation
between NC0 and NC3 can be found in Figure 56, between NC0 and NC1 in Figure 54, and between
NC0 and NC2 in Figure 55.




                                                                                  29
```

## Page 030

```text
Published as a conference paper at ICLR 2026




                                                                          1.0
                                                                          0.9
                    102                                                   0.8




                                                                    NC3
              NC0
                                                                          0.7
                                  optimizer                                          optimizer
                                     AdamW                                0.6           AdamW
                                     Adam                                               Adam
                                                                          0.5
                              0      10 5             10 4   10 3                0      10 5             10 4     10 3
                                              Weight decay                                       Weight decay

Figure 30: NC0 (left) and NC3 (right) metrics plotted against weight decay on a ResNet50 trained on
ImageNet1K for Adam and AdamW. Shaded area refers to one standard deviation across all trainings
run with corresponding optimizer.


                    6
                                                                                                         AdamW   Adam
                    5                                                     1.33
                    4                                                     1.32
                                                                          1.31
              NC1




                    3
                                                                    NC2   1.30
                    2         optimizer
                                    AdamW                                 1.29
                    1               Adam                                  1.28
                          0         10 5             10 4    10 3                0       10 5             10 4    10 3
                                            Weight decay                                         Weight decay

Figure 31: NC1 (left) and NC2 (right) metrics plotted against weight decay on a ResNet50 trained on
ImageNet1K for Adam and AdamW. Shaded area refers to one standard deviation across all trainings
run with corresponding optimizer.



D.4.10    P RELIMINARY EXPERIMENTAL RESULTS ON V ISION T RANSFORMER

We have also conducted preliminary experiments pretraining small Vision Transformers (ViT) on
Cifar10 from scratch. Given that training ViTs is computationally much more expensive given the
larger size of the model, we had to limit ourselves to a more restricted number of experiments.
Specifically, we chose to train the ViT with Adam, AdamW, and SGD for 200 epochs with a
batch size of 512 with momentum µ in the range µ ∈ {0, 0.8, 0.9, 0.95} and weight decay WD
∈ {0, 1e−5 , 1e−4 , 5e−4 , 1e−3 , 0.05, 0.5} for Adam and SGD and WD ∈ {0, 1e−4 , 0.05, 0.5, 1, 2, 4}
for AdamW. We discarded all runs, which did not achieve a training accuracy of at least 50%. This
mainly corresponded to training runs of SGD and Adam either with momentum=0 or WD≥ 0.05.
The ViT implementation is based on code from https://github.com/tintn/
vision-transformer-from-scratch/tree/main, which is published under the
MIT license. Specifically, the transformer model was chosen with a hidden dimension of 512, 6
hidden layers, and 8 attention heads, with no dropout applied.
Compared to the training procedure used in other settings, we employ a cosine-decay learning rate
schedule with warm-up, where 5% of the total training steps are allocated to warm-up, and the base
learning rate is set to 1 × 10−3 . Weight decay is applied to all layers except for LayerNorm and
biases, which is standard practice.
The highest final test accuracy across all trainings was achieved by AdamW (β1 = 0.95, WD = 0.5)
with 83.67%, with a final test loss of 0.895. Notably, higher accuracy levels can be attained by
increasing the network size and applying data augmentation or by using a pre-trained model as in
Ammar et al. (2024). However, to ensure consistency with the experiments in the main study, we
do not perform data augmentation due to limited computational resources. This likely explains the
relatively lower test accuracy. Investigating the impact of data augmentation on the convergence to
NC remains an interesting avenue for future work.


                                                                30
```

## Page 031

```text
Published as a conference paper at ICLR 2026




                                                                                                               1.0                                     optimizer
                                                                                                                                                          AdamW
                                                                                                               0.8                                        Adam
                                      101
                                                                                                               0.6




                                                                                                         NC3
                              NC0
                                      100                                          AdamW
                                                                                   Adam                        0.4

                                     10 1                                                                      0.2
                                                  0 10 5 10 4 10 3 10 2 10 1 100 101                                   0 10 5 10 4 10 3 10 2 10 1 100 101
                                                                  Weight decay                                                         Weight decay

Figure 32: NC0 (left) and NC3 (right) metrics plotted against weight decay on a VGG9 trained
on FashionMNIST for Adam and AdamW. Shaded area refers to one standard deviation across all
trainings run with corresponding optimizer.

                                                           1.4                                                                                                                               optimizer
      104                                                                                                        100                                          0.8
                                                           1.2                                                                                                                                   SGDW
                                                                                                                10 1                                                                             SGD
      103                                                  1.0                                                                                                0.6
                        optimizer                                                        optimizer              10 2
                                                                                               SignumW
                                                     NC3




                                                                                                                                                        NC3
                            SignumW
NC0




      102                                                  0.8




                                                                                                          NC0
                            Signum                                                             Signum                                                         0.4
                                                                                                                10 3
      101                                                  0.6
                                                                                                                10 4                                          0.2
      100                                                  0.4                                                                  SGDW
                                                                                                                10 5            SGD
            0 10 5   10 4   10 3    10 2   10 1                  0 10 5   10 4   10 3   10 2    10 1                   0 10 5    10 4 10 3 10 2 10 1                0 10 5   10 4 10 3 10 2 10 1
                      Weight decay                                         Weight decay                                           Weight decay                                Weight decay


Figure 33: NC0 and NC3 metrics plotted against weight decay on a VGG9 trained on FashionMNIST
for Signum and SignumW (left side) and SGD and SGDW (right side). Shaded area refers to one
standard deviation across all trainings run with corresponding optimizer.


While we observe the general trend of decreasing NC metrics with increasing values of weight decay
for SGD (Figure 58a), we note that in the case of ViTs the NC0 metric for both Adam and AdamW
first increases before decreasing (Figure 58b, left), while the NC3 metric for both Adam and AdamW
has a U-shape (Figure 58b, right). We also note that the ViT is much more sensitive to the choice of
weight decay and the training and validation accuracy degrades quickly due to overregularization, as
can be seen in Figure 58c. A further investigation of these observations is left for future work.




                                                                                                         31
```

## Page 032

```text
Published as a conference paper at ICLR 2026




                              lr = 0.001                              lr = 0.005                            lr = 0.01                          lr = 0.0679                   optimizer
             100                                                                                                                                                                AdamW
NC1                                                                                                                                                                             Adam
            10 1                                                                                                                                                                SGDW
                                                                                                                                                                                SGD
            10 2                                                                                                                                                                SignumW
                             10 2 102                             10 2 102                              10 2 102                               10 2 102                         Signum
                                NC0                                  NC0                                   NC0                                    NC0
Figure 34: NC0 vs. NC1 on VGG9 trained on FashionMNIST. Note that the x-axis is plotted in
log-scale.

                             lr = 0.001                          lr = 0.005                                 lr = 0.01                          lr = 0.0679                   optimizer
            1.0                                                                                                                                                                 AdamW
                                                                                                                                                                                Adam
NC2




            0.5                                                                                                                                                                 SGDW
                                                                                                                                                                                SGD
                                                                                                                                                                                SignumW
                        10 2 102                                10 2 102                                10 2 102                               10 2 102                         Signum
                           NC0                                     NC0                                     NC0                                    NC0
Figure 35: NC0 vs. NC2 on VGG9 trained on FashionMNIST. Note that the x-axis is plotted in
log-scale.

                             lr = 0.001                          lr = 0.005                                 lr = 0.01                          lr = 0.0679                   optimizer
                                                                                                                                                                                AdamW
            1.0                                                                                                                                                                 Adam
NC3




            0.5                                                                                                                                                                 SGDW
                                                                                                                                                                                SGD
            0.0                                                                                                                                                                 SignumW
                        10 2 102                                10 2 102                                10 2 102                               10 2 102                         Signum
                           NC0                                     NC0                                     NC0                                    NC0
Figure 36: NC0 vs. NC3 on VGG9 trained on FashionMNIST. Note that the x-axis is plotted in
log-scale.

                                                                                                                                                     optimizer
                                                                                                             0.8                                            AdamW
                                      101
                                                                                                                                                            Adam
                                                                                                             0.7
                                                                                                       NC3
                                NC0




                                                                                                             0.6
                                                                                 AdamW
                                                                                 Adam
                                                                                                             0.5

                                             0 10 5          10 4 10 3 10 2 10 1                                     0 10 5        10 4 10 3 10 2 10 1
                                                              Weight decay                                                           Weight decay

Figure 37: NC0 (left) and NC3 (right) metrics plotted against weight decay on a ResNet9 trained on
Cifar10 for Adam and AdamW. Shaded area refers to one standard deviation across all trainings run
with corresponding optimizer.

      104                                              1.4
                                                                                                               100                                         0.8                            optimizer
                                                       1.3                                                                                                                                    SGDW
                                                       1.2                                                    10 1                                                                            SGD
      103                                                                                                                                                  0.6
                                                       1.1                                                    10 2
                                                 NC3




                                                                                                                                                     NC3
NC0




                                                                                                        NC0




                                                       1.0                                                    10 3                                         0.4
      102                                              0.9                             optimizer
                                                                                             SignumW          10 4                                         0.2
                                      SignumW          0.8                                                                    SGDW
                                      Signum                                                 Signum           10 5            SGD
             0 10 5   10 4    10 3   10 2 10 1               0 10 5     10 4   10 3   10 2    10 1                   0 10 5    10 4 10 3 10 2 10 1               0 10 5   10 4 10 3 10 2 10 1
                       Weight decay                                      Weight decay                                           Weight decay                               Weight decay


Figure 38: NC0 and NC3 metrics plotted against weight decay on a ResNet9 trained on Cifar10
for Signum and SignumW (left side) and SGD and SGDW (right side). Shaded area refers to one
standard deviation across all trainings run with corresponding optimizer.


                                                                                                       32
```

## Page 033

```text
Published as a conference paper at ICLR 2026




             100           lr = 0.001                              lr = 0.005                   lr = 0.01                          lr = 0.0679                   optimizer
                                                                                                                                                                    AdamW
NC1                                                                                                                                                                 Adam
            10 1                                                                                                                                                    SGDW
                                                                                                                                                                    SGD
                                                                                                                                                                    SignumW
                          10 2 102                             10 2 102                         10 2 102                           10 2 102                         Signum
                             NC0                                  NC0                              NC0                                NC0
Figure 39: NC0 vs. NC1 on ResNet9 trained on Cifar10. Note that the x-axis is plotted in log-scale.

                         lr = 0.001                           lr = 0.005                        lr = 0.01                          lr = 0.0679                   optimizer
            1.0                                                                                                                                                     AdamW
                                                                                                                                                                    Adam
NC2




            0.5                                                                                                                                                     SGDW
                                                                                                                                                                    SGD
                                                                                                                                                                    SignumW
                        10 2 102                             10 2 102                       10 2 102                               10 2 102                         Signum
                           NC0                                  NC0                            NC0                                    NC0
Figure 40: NC0 vs. NC2 on ResNet9 trained on Cifar10. Note that the x-axis is plotted in log-scale.

                         lr = 0.001                           lr = 0.005                        lr = 0.01                          lr = 0.0679                   optimizer
                                                                                                                                                                    AdamW
            1.0                                                                                                                                                     Adam
NC3




            0.5                                                                                                                                                     SGDW
                                                                                                                                                                    SGD
                                                                                                                                                                    SignumW
                        10 2 102                             10 2 102                       10 2 102                               10 2 102                         Signum
                           NC0                                  NC0                            NC0                                    NC0
Figure 41: NC0 vs. NC3 on ResNet9 trained on Cifar10. Note that the x-axis is plotted in log-scale.

                                                                                                 0.9
                                                                                                 0.8
                                                                                                 0.7
                                   101
                                                                                                 0.6
                                                                                           NC3
                             NC0




                                                                             AdamW               0.5
                                                                             Adam                                                        optimizer
                                                                                                 0.4                                            AdamW
                                                                                                                                                Adam
                                                                                                 0.3
                                          0 10 5          10 4 10 3 10 2 10 1                            0 10 5        10 4 10 3 10 2 10 1
                                                           Weight decay                                                  Weight decay

Figure 42: NC0 (left) and NC3 (right) metrics plotted against weight decay on a VGG9 trained on
Cifar10 for Adam and AdamW. Shaded area refers to one standard deviation across all trainings run
with corresponding optimizer.

                                                    1.4                                                                                        0.8                            optimizer
                                                                                                   100
      104                                           1.3                                                                                                                           SGDW
                                                    1.2                                           10 1                                         0.6                                SGD

      103                                           1.1                                           10 2
                                              NC3




                                                                                                                                         NC3
NC0




                                                                                            NC0




                                                    1.0                                                                                        0.4
                                                                                                  10 3
                                                    0.9     optimizer
      102                                                      SignumW                            10 4                                         0.2
                                   SignumW          0.8                                                           SGDW
                                   Signum                      Signum                             10 5            SGD
                                                    0.7
             0 10 5   10 4 10 3   10 2 10 1               0 10 5     10 4 10 3 10 2 10 1                 0 10 5    10 4 10 3 10 2 10 1               0 10 5   10 4 10 3 10 2 10 1
                       Weight decay                                   Weight decay                                  Weight decay                               Weight decay


Figure 43: NC0 and NC3 metrics plotted against weight decay on a VGG9 trained on Cifar10 for
Signum and SignumW (left side) and SGD and SGDW (right side). Shaded area refers to one standard
deviation across all trainings run with corresponding optimizer.


                                                                                           33
```

## Page 034

```text
Published as a conference paper at ICLR 2026




                         lr = 0.001                              lr = 0.005                           lr = 0.01                          lr = 0.0679                   optimizer
         102                                                                                                                                                              AdamW
NC1                                                                                                                                                                       Adam
         100                                                                                                                                                              SGDW
                                                                                                                                                                          SGD
                                                                                                                                                                          SignumW
                        10 2 102                                10 2 102                          10 2 102                               10 2 102                         Signum
                           NC0                                     NC0                               NC0                                    NC0
 Figure 44: NC0 vs. NC1 on VGG9 trained on Cifar10. Note that the x-axis is plotted in log-scale.

                         lr = 0.001                              lr = 0.005                           lr = 0.01                          lr = 0.0679                   optimizer
         1.0                                                                                                                                                              AdamW
                                                                                                                                                                          Adam
NC2




         0.5                                                                                                                                                              SGDW
                                                                                                                                                                          SGD
                                                                                                                                                                          SignumW
                       10 2 102                                 10 2 102                          10 2 102                               10 2 102                         Signum
                          NC0                                      NC0                               NC0                                    NC0
 Figure 45: NC0 vs. NC2 on VGG9 trained on Cifar10. Note that the x-axis is plotted in log-scale.

                         lr = 0.001                              lr = 0.005                           lr = 0.01                          lr = 0.0679                   optimizer
                                                                                                                                                                          AdamW
         1.0                                                                                                                                                              Adam
NC3




         0.5                                                                                                                                                              SGDW
                                                                                                                                                                          SGD
         0.0                                                                                                                                                              SignumW
                       10 2 102                                 10 2 102                          10 2 102                               10 2 102                         Signum
                          NC0                                      NC0                               NC0                                    NC0
 Figure 46: NC0 vs. NC3 on VGG9 trained on Cifar10. Note that the x-axis is plotted in log-scale.


                                                                                                                                               optimizer
                                      101                                                              0.8                                            AdamW
                                                                                                                                                      Adam
                                                                                                       0.6
                                      100
                                                                                                 NC3
                             NC0




                                                                                                       0.4
                                     10 1
                                             optimizer                                                 0.2
                                                   AdamW
                                     10 2          Adam
                                            010 5 10 4 10 3 10 2 10 1 100 101                                  010 5 10 4 10 3 10 2 10 1 100 101
                                                               Weight decay                                                    Weight decay

Figure 47: NC0 (left) and NC3 (right) metrics plotted against weight decay on a ResNet9 trained on
MNIST for Adam and AdamW. Shaded area refers to one standard deviation across all trainings run
with corresponding optimizer.

                                                       1.4                                                                                                                          optimizer
       104                                                                                               100                                         0.8
                                                       1.2                                                                                                                              SGDW
       103                                                                                              10 1                                                                            SGD
                                                       1.0                                                                                           0.6
                         optimizer                     0.8
                                                                         optimizer                      10 2
       102                                                                 SignumW
                                                 NC3




                                                                                                                                               NC3




                           SignumW
NC0




                                                                                                  NC0




                           Signum                      0.6                 Signum                       10 3                                         0.4
       101
                                                       0.4                                              10 4
       100                                                                                                                                           0.2
                                                       0.2                                                              SGDW
                                                                                                        10 5            SGD
      10 1
             010 5 10 4 10 3 10 2 10 1 100 101               010 5 10 4 10 3 10 2 10 1 100 101                 0 10 5    10 4 10 3 10 2 10 1               0 10 5   10 4 10 3 10 2 10 1
                       Weight decay                                    Weight decay                                       Weight decay                               Weight decay


Figure 48: NC0 and NC3 metrics plotted against weight decay on a ResNet9 trained on MNIST
for Signum and SignumW (left side) and SGD and SGDW (right side). Shaded area refers to one
standard deviation across all trainings run with corresponding optimizer.


                                                                                                 34
```

## Page 035

```text
Published as a conference paper at ICLR 2026




                           lr = 0.001                              lr = 0.005                          lr = 0.01                          lr = 0.0679                   optimizer
             100                                                                                                                                                           AdamW
NC1                                                                                                                                                                        Adam
         10 1                                                                                                                                                              SGDW
                                                                                                                                                                           SGD
         10 2                                                                                                                                                              SignumW
                          10 2   103                              10 2   103                          10 2   103                          10 2   103                       Signum
                             NC0                                     NC0                                 NC0                                 NC0
Figure 49: NC0 vs. NC1 on ResNet9 trained on MNIST. Note that the x-axis is plotted in log-scale.


                         lr = 0.001                              lr = 0.005                           lr = 0.01                          lr = 0.0679                   optimizer
         1.0                                                                                                                                                              AdamW
                                                                                                                                                                          Adam
NC2




         0.5                                                                                                                                                              SGDW
                                                                                                                                                                          SGD
         0.0                                                                                                                                                              SignumW
                        10 2   103                              10 2   103                        10 2   103                             10 2   103                       Signum
                           NC0                                     NC0                               NC0                                    NC0
Figure 50: NC0 vs. NC2 on ResNet9 trained on MNIST. Note that the x-axis is plotted in log-scale.


                      lr = 0.001                               lr = 0.005                         lr = 0.01                              lr = 0.0679                   optimizer
                                                                                                                                                                          AdamW
         1                                                                                                                                                                Adam
NC3




                                                                                                                                                                          SGDW
                                                                                                                                                                          SGD
         0                                                                                                                                                                SignumW
                      10 2   103                              10 2   103                          10 2   103                             10 2   103                       Signum
                         NC0                                     NC0                                 NC0                                    NC0
Figure 51: NC0 vs. NC3 on ResNet9 trained on MNIST. Note that the x-axis is plotted in log-scale.

                                                                                                       1.0
                                                                                                                                               optimizer
                                                                                                       0.8                                            AdamW
                                      101                                                                                                             Adam
                                                                                                       0.6
                                      100
                                                                                                 NC3
                             NC0




                                                                                                       0.4
                                     10 1    optimizer
                                                   AdamW                                               0.2
                                     10 2          Adam
                                                                                                       0.0
                                            010 5 10 4 10 3 10 2 10 1 100 101                                  010 5 10 4 10 3 10 2 10 1 100 101
                                                               Weight decay                                                    Weight decay

Figure 52: NC0 (left) and NC3 (right) metrics plotted against weight decay on a VGG9 trained on
MNIST for Adam and AdamW. Shaded area refers to one standard deviation across all trainings run
with corresponding optimizer.

                                                       1.4                                                                                           0.8                            optimizer
       104                                                                                               100
                                                       1.2                                                                                                                              SGDW
                                                       1.0                                              10 1                                         0.6                                SGD
       103
                         optimizer                     0.8               optimizer                      10 2
       102                                                                 SignumW
                                                 NC3




                                                                                                                                               NC3




                           SignumW
NC0




                                                                                                  NC0




                                                       0.6                                              10 3                                         0.4
       101                 Signum                                          Signum
                                                       0.4                                              10 4                                         0.2
       100
                                                       0.2                                              10 5            SGDW
      10 1                                                                                                              SGD
                                                       0.0                                                                                           0.0
             010 5 10 4 10 3 10 2 10 1 100 101               010 5 10 4 10 3 10 2 10 1 100 101                 0 10 5    10 4 10 3 10 2 10 1               0 10 5   10 4 10 3 10 2 10 1
                       Weight decay                                    Weight decay                                       Weight decay                               Weight decay


Figure 53: NC0 and NC3 metrics plotted against weight decay on a VGG9 trained on MNIST for
Signum and SignumW (left side) and SGD and SGDW (right side). Shaded area refers to one standard
deviation across all trainings run with corresponding optimizer.


                                                                                                 35
```

## Page 036

```text
Published as a conference paper at ICLR 2026




               lr = 0.001         lr = 0.005                lr = 0.01           lr = 0.0679               optimizer
      101
                                                                                                            AdamW
                                                                                                            Adam
NC1

   10 1                                                                                                     SGDW
                                                                                                            SGD
                                                                                                            SignumW
              10 2   103          10 2   103                10 2   103          10 2   103                  Signum
                 NC0                 NC0                       NC0                 NC0
Figure 54: NC0 vs. NC1 on VGG9 trained on MNIST. Note that the x-axis is plotted in log-scale.




             lr = 0.001          lr = 0.005                 lr = 0.01          lr = 0.0679                optimizer
      1.0                                                                                                   AdamW
                                                                                                            Adam
NC2




      0.5                                                                                                   SGDW
                                                                                                            SGD
      0.0                                                                                                   SignumW
             10 2   103         10 2   103                10 2   103           10 2   103                   Signum
                NC0                NC0                       NC0                  NC0
Figure 55: NC0 vs. NC2 on VGG9 trained on MNIST. Note that the x-axis is plotted in log-scale.




            lr = 0.001          lr = 0.005                lr = 0.01            lr = 0.0679              optimizer
                                                                                                            AdamW
      1                                                                                                     Adam
NC3




                                                                                                            SGDW
                                                                                                            SGD
      0                                                                                                     SignumW
            10 2   103          10 2   103                10 2   103           10 2   103                   Signum
               NC0                 NC0                       NC0                  NC0
Figure 56: NC0 vs. NC3 on VGG9 trained on MNIST. Note that the x-axis is plotted in log-scale.




      1.0                                   1.0                                1.0
      0.9                                   0.9                                0.9
NC4




                                      NC4




                                                                         NC4




      0.8                                   0.8                                0.8
             10 1         101   103               0.25 0.50 0.75 1.00                         0.5          1.0
                     NC1                                  NC2                                       NC3
                      AdamW       Adam             SGDW          SGD     SignumW               Signum

Figure 57: NC4 is largely uncorrelated with NC1-3 across different optimizers and learning rates.




                                                       36
```

## Page 037

```text
  Published as a conference paper at ICLR 2026




                                                                             0.8
                 1.4                                                                                           optimizer
                                                                                                                    SGD
                                                                             0.7
NC0




                                                           NC3
                 1.3
                        optimizer                                            0.6
                 1.2         SGD
                        0       10 4                    10 3                           0       10 4                  10 3
                                    weight_decay                                                weight_decay
  (a) NC0 (left) and NC3 (right) metric for varying values of weight decay on a ViT trained with SGD on
  Cifar10.

                 4                            optimizer
                                                   AdamW                  1.0
                 3                                 Adam                   0.8
NC0




                                                           NC3




                 2
                 1                                                        0.6

                       0 10 5 10 4 10 3 10 2 10 1 100                                  0 10 5 10 4 10 3 10 2 10 1 100
                                weight_decay                                                    weight_decay
  (b) NC0 (left) and NC3 (right) metric for varying values of weight decay on a ViT trained with Adam and
  AdamW on Cifar10.
                 1.0                                                             1.0
                                                           Validation accuracy
Train accuracy




                 0.9                                                             0.9
                 0.8                                                             0.8
                 0.7                                                             0.7
                 0.6                                                             0.6
                        0 10 5 10 4 10 3 10 2 10 1 100                                 0 10 5 10 4 10 3 10 2 10 1 100
                                    weight_decay                                                weight_decay
                                               AdamW                             SGD          Adam
  (c) Training accuracy (left) and validation accuracy (right) for varying values of weight decay on a ViT
  trained on Cifar10.




                                                                      37
```

## Page 038

```text
Published as a conference paper at ICLR 2026




E    P ROOFS
In this section, we will present the proof which is omitted in the main text.
Theorem E.1 (Effect of decoupled SGD update on NC0). Assume a model of the form f (W, θ, x) =
Whθ (x) is trained using cross-entropy loss with SGD with decoupled weight decay for all parameters
W, θ. For instance, the last layer weight W has the following update rule:
                                 Vt+1 = βVt + ∇Wt LCE ,
                                 Wt+1 = (1 − ηλ) Wt − ηVt+1 ,
where β ∈ [0, 1), η > 0, and λ ∈ R. Define the NC0 metric
                                                1       2
                                        αt :=     Wt⊤ 1 2 .
                                                K
Then, for all t ≥ 0,
                                        αt = (1 − ηλ)2t α0 .
In particular, if 0 < ηλ < 2, then αt decays exponentially to zero:
                                   αt = (1 − ηλ)2t α0 −−−→ 0.
                                                            t→∞


Proof. We track the evolution of the row sums of Wt and Vt . Define
                           mt := Wt⊤ 1 ∈ RK ,          qt := Vt⊤ 1 ∈ RK .
By definition of αt we have
                                                 1
                                          αt =     ∥mt ∥22 .
                                                 K
Note that by Lemma E.5, the cross-entropy gradient with respect to the last layer satisfies
                                                 ⊤
                                       ∇Wt LCE 1 = 0
for all Wt . Consider the momentum update
                                      Vt+1 = βVt + ∇Wt LCE .
Multiplying on the right by 1 and using the above result, we obtain
                  ⊤
                                              ⊤                    ⊤
        qt+1 = Vt+1   1 = βVt + ∇Wt LCE 1 = βVt⊤ 1 + ∇Wt LCE 1 = βqt + 0.
Thus qt+1 = βqt , and by induction
                                            qt = β t q0 .
Since V0 = 0, we have q0 = 0, hence
                                      qt = 0       for all t ≥ 0.
Consider now the decoupled weight update
                                 Wt+1 = (1 − ηλ) Wt − ηVt+1 .
Multiplying on the right by 1 gives
        ⊤
                                           ⊤
mt+1 = Wt+1 1 = (1−ηλ)Wt −ηVt+1                 1 = (1−ηλ)Wt⊤ 1−ηVt+1
                                                                  ⊤
                                                                      1 = (1−ηλ)mt −ηqt+1 .
Using qt+1 = 0 for all t, we obtain the simple linear recursion
                                       mt+1 = (1 − ηλ) mt .
Solving this recursion yields
                                       mt = (1 − ηλ)t m0 .
Substituting the expression for mt into the definition of αt gives
                1             1                 2          1
         αt =      ∥mt ∥22 =    (1 − ηλ)t m0 2 = (1 − ηλ)2t ∥m0 ∥22 = (1 − ηλ)2t α0 .
               K             K                             K
This establishes the exact formula claimed in the theorem.



                                                  38
```

## Page 039

```text
Published as a conference paper at ICLR 2026




Theorem E.2 (Effect of SGD update with coupled weight decay on NC0). Assume a model of the
form f (W, θ, x) = Whθ (x) is trained using cross-entropy loss with stochastic gradient descent
(SGD) and momentum β ∈ [0, 1), weight decay λ ∈ [0, 1), and learning rate η > 0 sufficiently small.
The last-layer weights Wt are updated according to:
                                    Vt+1 = βVt + ∇Wt LCE + λWt ,
                                                                                               (15)
                                    Wt+1 = Wt − ηVt+1 ,
where β ∈ [0, 1), η > 0, and λ ∈ R. Then there exists a constant C ≥ 1 such that
                                1       2
                             αt =  mt 2 ≤ C ρ2t α0             for all t ≥ 0,                  (16)
                               K
where ρ := max{|r+ |, |r− |} and r± are the roots of
                                      r2 − (1 + β − ηλ) r + β = 0.                             (17)

In particular: if ηλ < 2(1 + β), then ρ < 1 and the NC0 metric αt decays exponentially in t.

Proof. We follow the same strategy as in the decoupled case: track the evolution of the row sums of
Vt and Wt .
From (15),
                               Vt+1 = βVt + ∇Wt LCE + λWt .
Right-multiplying by 1 and using Lemma (E.5), we get
                                      ⊤
                              qt+1 = Vt+1 1
                                                           ⊤
                                     = βVt + ∇Wt LCE + λWt 1
                                                       ⊤
                                     = βVt⊤ 1 + ∇Wt LCE 1 + λWt⊤ 1
                                     = β qt + λ mt .
Thus
                                          qt+1 = β qt + λ mt .                                 (18)

From the weight update
                                         Wt+1 = Wt − ηVt+1 ,
we obtain
                                      ⊤
                                                                 ⊤
                              mt+1 = Wt+1 1 = Wt − ηVt+1              1
                                      = Wt⊤ 1 − ηVt+1
                                                  ⊤
                                                      1 = mt − η qt+1 .
Using (18) this becomes
                                               
                      mt+1 = mt − η β qt + λ mt = (1 − ηλ) mt − ηβ qt .                        (19)

We also have, from the weight update at time t,
                                          mt = mt−1 − η qt ,
which is just (19) with index shifted by one. Hence
                                                1          
                                         qt =     mt−1 − mt .                                  (20)
                                                η
Substitute (20) into (19):
                                                        1           
                             mt+1 = (1 − ηλ) mt − ηβ ·    mt−1 − mt
                                                        η
                                                                 
                                     = (1 − ηλ) mt − β mt−1 − mt
                                     = (1 − ηλ) mt − βmt−1 + βmt
                                     = (1 + β − ηλ) mt − β mt−1 .


                                                   39
```

## Page 040

```text
Published as a conference paper at ICLR 2026




We are given m0 = W0⊤ 1 and q0 = V0⊤ 1 = 0. Then
                                        q1 = βq0 + λm0 = λm0 ,
and hence from the weight update
                         m1 = m0 − ηq1 = m0 − ηλm0 = (1 − ηλ) m0 .

The recurrence is linear and homogeneous with constant coefficients. For each coordinate of mt , say
(mt )k , we have a scalar second-order recursion
                           (mt+1 )k = (1 + β − ηλ) (mt )k − β (mt−1 )k .
The characteristic polynomial is
                                       r2 − (1 + β − ηλ) r + β = 0,
with roots r+ and r− given by
                                                    p
                                   1 + β − ηλ ±         (1 + β − ηλ)2 − 4β
                            r± =                                           .
                                                         2
Thus each coordinate can be written as
                                                       t         t
                                        (mt )k = c+,k r+ + c−,k r− ,
for some coefficients c+,k , c−,k determined by (m0 )k and (m1 )k . Let
                                           ρ := max{|r+ |, |r− |}
be the spectral radius of the recursion. Then there exists a constant C ≥ 1 (depending only on β, λ, η)
such that
                                           mt 2 ≤ C ρt m0 2 ,
and therefore
                                   1       2            1        2
                            αt =       mt 2 ≤ C 2 ρ2t       m0 2 = C ′ ρ2t α0
                                   K                    K
for some C ′ ≥ 1, which is (16). Finally, for a general quadratic equation r2 + br + c = 0, the roots
are in the unit circle if |c| < 1, 1 + b + c > 0 and 1 − b + c > 0. Thus it is not difficult to check from
the characteristic polynomial that ηλ < 2(1 + β) implies ρ < 1.



Note that the above Theorem holds for any model f (W, θ, x) = Whθ (x) with last layer as linear
classifier and with any backbone hθ parameterized by θ.
However, the dynamics of Adam is more complicated, hence we further restrict the setting to SignGD,
a special case of Adam, training a UFM.
Here, we assume a balanced dataset with only one element in each class k ∈ [K]. It is obvious to
extend our result to multiple elements per class. Hence the total input N = K is equal to the number
of classes and the UFM loss can be written as
                                                    N
                                                    X
                                 LCE (WH, I) =            LCE (Whn , en ),
                                                    n=1

where we can decouple the regularization λ2 ∥W∥2 + λ2 ∥H∥2 into weight decay.
By Zhu et al. (2021), we know that the UFM
                                 N
                                 X                         λ       λ
                           max         LCE (Whn , yn ) +     ∥W∥2 + ∥H∥2 ,
                           W,H
                                 n=1
                                                           2       2

has unique global minimum W, H and no strict saddle points. In particular, H = UM∗ for some
orthogonal matrix U ∈ O(P ). To further simplify the analysis, we assume that P = N = K with
H = M∗ . Then we have the followings:


                                                    40
```

## Page 041

```text
Published as a conference paper at ICLR 2026




Theorem E.3. Consider sign GD with (decoupled) weight decay λ > 0 and step size η > 0 on the
UFM loss
                                          XN
                           LCE (WH, I) =      LCE (Whn , en ),
                                               n=1
where the feature H = M∗ is fixed to an NC solution and only the weight W is trained:
                           Wt+1 = Wt − η(sign(∇Wt LCE ) + λWt )
with initialization W0 = 0 ∈ RK×K . We define the covariance matrix Ct = Wt Wt⊤ and the scalar
αt = ⟨Ct , Ĵ⟩F where Ĵ = K  1
                                11⊤ . Then αt will increase monotonically from zero to the limit:
                                                 (K − 2)2
                                      lim αt =            .
                                     t→∞            λ2
In particular, αt does not vanish as t → ∞.

Proof. By Lemma E.5, we have ∇LCE (W) = N1 (S − Y)H⊤ = N1 (S − I) · √K−1       1         1
                                                                                    (I − K J) =
  √1      (softmax(WH) − I) since (softmax(WH) − I)J = 0. Since softmax has range between 0
N K−1
and 1, we have
                                   sign (∇LCE (WH)) = J − 2I,
that is, the signed gradient is −1 on the diagonal and +1 elsewhere. Note that this holds for all
W ∈ RK×K . The sign GD updates can hence be written as:
                                               h                 i
                              Wt+1 = Wt − η       J − 2I +λWt .
                                                  | {z }                                    (21)
                                               sign(∇Wt LCE )
                       
Since sign ∇LCE (Wt ) is constant, the dynamics collapse onto a scalar wt :
                                                       
                                        Wt = wt J − 2I ,
which has the following recursive form:
                               wt+1 = (1 − ηλ)wt − η, w0 = 0.
Solve it and obtain
                                           1h             i
                                    wt = − 1 − (1 − ηλ)t .
                                           λ
Recall the definition:
                                             1
                       Ct = Wt Wt⊤ Ĵ = 11⊤ and αt = ⟨Ct , Ĵ⟩F .
                                             K
                ⊤
Since ∥ J − 2I 1∥2 = (K − 2)2 K and the factor of 1/K gives (K − 2)2 , we have
                                       αt = (K − 2)2 wt2
Therefore
                       
                         1            2   (K − 2)2 h           t i 2
                          2          t
          αt = (K − 2) − 1 − (1 − ηλ)      =           1 − 1 − ηλ        .
                         λ                      λ2
                t
As t → ∞, 1 − ηλ → 0, so
                                    (K − 2)2
                              α∞ =           .
                                        λ2

Theorem E.4. Consider sign GD with (coupled) weight decay λ > 0 and step size η > 0 on the
UFM loss
                                          XN
                           LCE (WH, I) =      LCE (Whn , en ),
                                               n=1
where the feature H = M∗ is fixed to an NC solution and only the weight W is trained :
                          Wt+1 = Wt − η(sign(∇Wt LCE + λWt ))
with initialization W0 = 0 ∈ RK×K . We define the covariance matrix Ct = Wt Wt⊤ and the
scalar αt = ⟨Ct , Ĵ⟩F where Ĵ = K     1
                                          11⊤ . Then there exists a learning rate decay scheme
η = η(t) −−−→ 0 such that αt −−−→ 0.
            t→∞                t→∞


                                               41
```

## Page 042

```text
Published as a conference paper at ICLR 2026




Proof. Throughout the training, we apply mathematical induction on the structure of Wt : for all t,
there exists at , bt ≥ 0 such that
                                    Wt = (at + bt )I − bt J.
It is not hard to see that α = N1 (at − (K − 1)bt )2 . Note that for t = 0, the signed gradient is the
same as in the case with decoupled weight decay in Theorem 3.3:
          sign(∇Wt LCE + λWt ) = sign(∇W0 LCE ) = sign(softmax(0) − I) = J − 2I.
Hence, W1 = η(2I − J) where a1 = b1 = η. Since H = M∗ = √K−1
                                                         1
                                                             (I − J/k),

                                                   1
                 WH = ((at + bt )I − bt J) · √         (I − J/k)
                                                  K −1
                           1
                                 (at + bt )I − bt J − (at + bt )J/k + (bt /k)J2
                                                                                
                       =√
                          K −1
                        at + bt
                       =√       H = γt H
                          K −1

where we define γt = √atK−1
                        +bt
                            .By Lemma E.5 and the above expression, the loss gradient becomes:

                                             1
                            ∇Wt LCE =       √    (softmax(WH) − I)
                                        N K −1
                                             1
                                      = √        (softmax(γt H) − I)
                                        N K −1
                                      = ψt (−KI + J)

where ψt = N √1K−1 · eγt /√K−11 +(K−1) = N √1K−1 · e(at +bt )/(K−1)
                                                                1
                                                                    +(K−1)
                                                                           . Hence the update weight
will also of form
                                 Wt+1 = (at+1 + bt+1 )I − bt+1 J.
Hence the update rule of the signed GD with coupled weight decay can be written as:
                              at+1 = at + η · sign ((K − 1)ψt − λat )
                              bt+1 = bt + η · sign(ψt − λbt )
Then for each fixed η > 0, starting from t = 0, let ∆t = (at+1 − at , bt+1 − bt ), the training can be
divided into three phases:

      1. ∆t = (+η, +η) as long as (K − 1)ψt ≥ λat and ψt ≥ λbt . Note that ψt ∝
                      1
         e(at +bt )/(K−1) +(K−1)
                                 hence ψt+1 < ψt as ∆t = (+η, +η). Since ψt is strictly decreasing
         with at + bt , assume η is small enough, there exists a constant T1 such that ∆T1 = (+η, +η)
         but ∆T1 +1 = (+η, −η) where (K − 1)ψT1 ≥ λat ≥ λbt > ψT1 .
      2. ∆t = (+η, ±η) indicating at increases striclty in each step and bt starts to oscillate as long
         as (K − 1)ψt ≥ λat : each time ∆t−1 = (+η, −η), we have at + bt = at−1 + bt−1 and
         thus ψt = ψt−1 . Hence ψt decreases monotonically but not strictly. Similar to above, there
         exists a constant T2 > T1 such that (K − 1)ψT2 ≥ λaT2 but (K − 1)ψT2 +1 < λaT2 +1 .
      3. For t > T2 , ∆t = (±η, ±η) where i) ψt becomes constant for ∆t oscillates between
         (+η, −η) and (−η, +η) or ii) ψt oscillate for ∆t oscillates between (+η, +η) and (−η, −η).
         In wither case, we have maxt>T2 {|(K − 1)ψt − λat |, |ψt − λbt |} < λη as each update will
         flip the sign.

Hence for each η, we update T2 = T2 (η) steps until maxt>T2 {|(K − 1)ψt − λat |, |ψt − λbt |} < λη.
Next, we apply learning rate decay to η ′ so that λη ′ < min{|(K − 1)ψT2 +1 − λaT2 +1 |, |ψT2 +1 −
λbT2 +1 |} < λη. Repeat the above argument and find a T2′ > 0 such that ψt oscillates or remains
constant after t > T2 + T2′ , and hence maxt>T2 +T2′ {|(K − 1)ψt − λat |, |ψt − λbt |} < λη ′ . Induction


                                                  42
```

## Page 043

```text
Published as a conference paper at ICLR 2026




on this argument shows that there exists a learning rate decay scheme η = η(t) → 0 such that
maxt {|(K − 1)ψt − λat |, |ψt − λbt |} −−−→ 0, in which case:
                                                   t→∞
                                         2
        αt = (at − (K − 1)bt )
                                                   2
           = λ−2 (λat − (K − 1)λbt )
                                                                                            2
           ≤ λ−2 ((K − 1)ψt + |(K − 1)ψt − λat | − (K − 1)ψt + (K − 1)|ψt − λbt |)
                                                                                 2
           = λ−2 (|(K − 1)ψt − λat | + (K − 1)|ψt − λbt |)
           ≤ λ−2 K 2 max{|(K − 1)ψt − λat |, |ψt − λbt |} −−−→ 0.
                           t                                                 t→∞
                                    2
Hence αt = (at − (K − 1)bt ) −−−→ 0.
                                             t→∞


E.1   T ECHNICAL L EMMATA

Lemma E.5. Let (X, Y) ∈ Rd×N × RK×N be a dataset where the labels Y are written in columns
of one-hot vectors. For each pair (x, y) ∈ RD × RK , and a weight W1 ∈ RK×d , define the
cross-entropy as:
                                                                               
                     K
                     X                                    X
                def.
         ℓ(W1 ) = −     yk log (softmax(W1 x)) = log 1 +     exp(wk − wy )⊤ xi 
                                                                    k
                         k=1                                                         k̸=y

where y = arg maxk∈[K] [y]k is the class index of x. Let L1 (W1 ) = CE(W1 X, Y) be the average
cross-entropy loss of the dataset (X, Y). Then the loss gradient ∇L1 (W1 ) is
                                                                   1
                                              ∇L1 (W1 ) =            (S − Y)X⊤
                                                                   N
where S = (s1 , ...sN ) and si = softmax(W1 xi ) for each i. In particular, 1⊤
                                                                             K ∇L1 (W1 ) = 0.


Proof. The expression of the loss gradient comes from simple calculus. The second statement comes
from the fact that the L1 norms of a post-softmax vector and an one-hot vector are both equal to 1,
that is,
                                       1⊤       ⊤
                                        K si = 1K yi = 1 ∀i.



Lemma E.6. Assume the weight Wt is updated as follows:
                                              Vt+1 = βVt + Gt + λWt
                                              Wt+1 = Wt − ηVt+1 ,
where Gt depends on Wt . Define
                                                       def.   1
                                                   α=           ∥Wt⊤ 1∥22 ≥ 0.
                                                              K
Then we have the expression:
                               1
                                 (αt+1 − αt ) = −2βωt − 2γt − 2λαt + ηνt+1
                               η
         def.                     def.                            def.
where ωt = ⟨Vt Wt⊤ , Ĵ⟩, γt = ⟨Gt Wt⊤ , Ĵ⟩, νt = ⟨Vt Vt⊤ , Ĵ⟩.

                def.
Proof. Let Ct = Wt Wt⊤ be the covariance matrix. Notice that αt = ⟨Ct , Ĵ⟩ where Ĵ = K
                                                                                       1
                                                                                         11⊤ .
By update rule of Wt and Vt :
                       1                1
                                          (Wt − ηVt+1 )(Wt − ηVt+1 )⊤ − Ct
                                                                             
                         (Ct+1 − Ct ) =
                       η                η
                                      = −(Vt+1 Wt⊤ + Wt Vt+1
                                                         ⊤              ⊤
                                                             ) + ηVt+1 Vt+1 .


                                                                  43
```

## Page 044

```text
Published as a conference paper at ICLR 2026




                                                                          def.                def.
Applying the dot product ⟨·, Ĵ⟩F on both sides, and denote ωt = ⟨Vt Wt⊤ , Ĵ⟩, γt = ⟨Gt Wt⊤ , Ĵ⟩,
   def.
νt = ⟨Vt Vt⊤ , Ĵ⟩, we have
                     1
                       (αt+1 − αt ) = −2⟨Vt+1 Wt⊤ , Ĵ⟩ + η⟨Vt+1 Vt+1
                                                                  ⊤
                                                                      , Ĵ⟩
                     η
                                          = −2⟨(βVt + Gt + λWt )Wt⊤ , Ĵ⟩ + ηνt+1
                                          = −2βωt − 2γt − 2λαt + ηνt+1                               (22)
where in the first line we use the fact that Ĵ is symmetric.


Lemma E.7. Assume λ, β ∈ (0, 1) such that log2λ
                                              β −1 < 1. The solution of the following ODE:
                                          Z t               
                                                  t−τ
                              α̇(t) = −λ         β α(τ )dτ                                 (23)
                                                          0
with initial condition α(0) = α0 > 0 admits the following bound:
                                                                
                                                         λ
                                 α(t) ≤ Cα0 exp −              t
                                                      log β −1
for some absolute constant C > 1.

Proof. Observe that we can write the integral in convolution:
                     Z t
                         β t−τ α(τ )dτ = ϕ ∗ α (t), where                   ϕ(t) = β t .
                                                  
                              0
Hence (23) can be written as                                  
                                              α̇(t) = −λ ϕ ∗ α (t).
                     Z ∞
Let L{ψ(t)}(s) =              e−st ψ(t)dt denote the Laplace transform. Denote
                         0
                                  A(s) = L{α(t)}(s),           F (s) = L{ϕ(t)}(s).
Taking the Laplace transform of both sides:
                                       L{α̇(t)}(s) = −λL{(ϕ ∗ α)(t)}(s).                             (24)
And by integration by part and the property of convolution,
                L{α̇(t)}(s) = sA(s) − α(0)               and    L{(ϕ ∗ α)(t)}(s) = F (s)A(s).
Hence
                                         sA(s) − α(0) = −λF (s)A(s).
        t     (log β)t
Since β = e              , we get
                                                                       1
               F (s) = L{β t }(s) = L e(log β)t (s) =
                                     
                                                                                 for s > log(β).
                                                                  s − log(β)
Substitute this back to Eq. (24) and we get:
                                                                       1
                                         sA(s) − α(0) = −λ                   A(s)
                                                                  s − log(β)
                                          λ
                             sA(s) +            A(s) = α(0)
                                     s − log(β)
                                             λ     
                              A(s) s +                = α(0)
                                        s − log(β)
                                    |      {z     }
                                        s2 −s log(β)+λ
                                           s−log(β)
                                                                                  
                                                                         s − log(β)
                                                    A(s) = α(0) ·
                                                                     s2 − s log(β) + λ
                                                                     |       {z      }
                                                                        (s−r1 )(s−r2 )



                                                          44
```

## Page 045

```text
Published as a conference paper at ICLR 2026



                        r        2
                  log(β)±    log(β)    −4λ
where r1 , r2 =              2               . We do partial fractions and matching coefficients gives:
           s − log(β)          A        B
                           =        +        =⇒ A + B = 1,                           − log(β) = −Ar2 − Br1 .
        (s − r1 )(s − r2 )   s − r1   s − r2
Since r1 + r2 = log(β), one finds
                                                 r2                         r1
                                       A=             ,        B=−               .
                                              r2 − r1                    r2 − r1
Thus                                                                   
                                           r2     1         r1     1
                            A(s) = α(0)                −                  .
                                        r2 − r1 s − r1   r2 − r1 s − r2

Recall the inverse of Laplacian transform: L−1 { s−r
                                                  1
                                                     }(t) = ert . Therefore,
                                                                                  
                            −1                         r2    r1 t      r1     r2 t
                   α(t) = L {A(s)}(t) = α(0)                e −             e        .
                                                    r2 − r1         r2 − r1
Equivalently,
                              h               i                            r2             r1
                   α(t) = α(0) Aer1 t + Ber2 t ,                   A=           ,B = −         ,               (25)
                                                                        r2 − r1        r2 − r1
where                                                         q      2
                                                 log(β) ±       log(β) − 4λ
                                  r1 , r2 =                        .
                                                     2
Since β ∈ (0, 1), set L = − log(β) > 0. By the first order approximation,
                                                                      2
                       p
                                2
                                         p
                                             2
                                                             2λ        λ
                         (log β) − 4λ = L − 4λ = L −             +O
                                                              L         L
Hence
                                           −L ± L − 2λ
                                                            2
                                                    L        λ
                                 r1 , r2 =               +O      .
                                                2            L
This gives:
                                                 λ2                                        λ2
                                                                                             
                                λ                                      λ
                       r1 = − + O                         ,   r2 = −L + + O                         .
                                L                L                     L                   L
Plugging r1 , r2 into Eq. (25):
                                                                    
                                 α(t) ≤ Cα(0)er1 t = Cα(0) exp − L
                                                                 λ
                                                                   t

for some absolute constant C > 1. Plug in L = − log(β) = log β −1 to finish the proof.




                                                              45
```

