# Improving Generalization Performance by Switching from Adam to SGD - page-anchored PDF text

- Source ID: `arxiv-1712.07628v1`
- arXiv ID: `1712.07628v1`
- Original PDF: `소스/Improving Generalization Performance by Switching from Adam to SGD.pdf`
- PDF pages: 10
- Extracted with: MiKTeX poppler `pdftotext -f N -l N -layout -enc UTF-8` on 2026-05-20T16:15:58+09:00
- Evidence note: use this file for page-anchored claims; verify equations/tables against the original PDF when exact layout matters.

## Page 1

                                                Improving Generalization Performance by Switching from Adam to SGD


                                                                                   Nitish Shirish Keskar 1 Richard Socher 1


                                                                   Abstract                              lowing non-convex optimization problem,
                                                Despite superior training outcomes, adaptive op-
                                                timization methods such as Adam, Adagrad or                                       min       f (w),
                                                                                                                                 w∈Rn
arXiv:1712.07628v1 [cs.LG] 20 Dec 2017




                                                RMSprop have been found to generalize poorly
                                                compared to Stochastic gradient descent (SGD).           where f is a loss function. The iterations of SGD can be
                                                These methods tend to perform well in the initial        described as:
                                                portion of training but are outperformed by SGD
                                                at later stages of training. We investigate a hy-                                       ˆ (wk−1 ),
                                                                                                                       wk = wk−1 − αk−1 ∇f
                                                brid strategy that begins training with an adaptive
                                                method and switches to SGD when appropriate.             where wk denotes the k th iterate, αk is a (tuned) step size
                                                Concretely, we propose SWATS, a simple strat-            sequence, also called the learning rate, and ∇f  ˆ (wk ) de-
                                                egy which Switches from Adam to SGD when                 notes the stochastic gradient computed at wk . A variant of
                                                a triggering condition is satisfied. The condi-          SGD (SGDM), that uses the inertia of the iterates to accel-
                                                tion we propose relates to the projection of Adam        erate the training process, has also found to be successful in
                                                steps on the gradient subspace. By design, the           practice (Sutskever et al., 2013). The iterations of SGDM
                                                monitoring process for this condition adds very          can be described as:
                                                little overhead and does not increase the num-
                                                ber of hyperparameters in the optimizer. We re-                                        ˆ (wk−1 )
                                                                                                                          vk = βvk−1 + ∇f
                                                port experiments on several standard benchmarks                          wk = wk−1 − αk−1 vk ,
                                                such as: ResNet, SENet, DenseNet and Pyramid-
                                                Net for the CIFAR-10 and CIFAR-100 data sets,            where β ∈ [0, 1) is a momentum parameter and v0 is ini-
                                                ResNet on the tiny-ImageNet data set and lan-            tialized to 0.
                                                guage modeling with recurrent networks on the
                                                PTB and WT2 data sets. The results show that             One disadvantage of SGD is that it scales the gradient uni-
                                                our strategy is capable of closing the generaliza-       formly in all directions; this can be particularly detrimental
                                                tion gap between SGD and Adam on a majority              for ill-scaled problems. This also makes the process of tun-
                                                of the tasks.                                            ing the learning rate α circumstantially laborious.
                                                                                                         To correct for these shortcomings, several adaptive meth-
                                                                                                         ods have been proposed which diagonally scale the gradi-
                                         1. Introduction                                                 ent via estimates of the function’s curvature. Examples of
                                         Stochastic gradient descent (SGD) (Robbins & Monro,             such methods include Adam (Kingma & Ba, 2015), Ada-
                                         1951) has emerged as one of the most used training al-          grad (Duchi et al., 2011) and RMSprop (Tieleman & Hin-
                                         gorithms for deep neural networks. Despite its simplic-         ton, 2012). These methods can be interpreted as methods
                                         ity, SGD performs well empirically across a variety of          that use a vector of learning rates, one for each parameter,
                                         applications but also has strong theoretical foundations.       that are adapted as the training algorithm progresses. This
                                         This includes, but is not limited to, guarantees of saddle      is in contrast to SGD and SGDM which use a scalar learn-
                                         point avoidance (Lee et al., 2016), improved generalization     ing rate uniformly for all parameters.
                                         (Hardt et al., 2015; Wilson et al., 2017) and interpretations   Adagrad takes steps of the form
                                         as Bayesian inference (Mandt et al., 2017).
                                                                                                                                   ˆ (wk−1 )
                                                                                                                                   ∇f
                                         Training neural networks is equivalent to solving the fol-               wk = wk−1 − αk−1 √          ,         where      (1)
                                                                                                                                     vk−1 + 
                                            1
                                             Salesforce Research, Palo Alto, CA – 94301. Correspon-                      k−1
                                         dence to: Nitish Shirish Keskar <nkeskar@salesforce.com>.                       X
                                                                                                                               ˆ (wj )2 .
                                                                                                                vk−1 =         ∇f
                                                                                                                         j=1

## Page 2

                           Improving Generalization Performance by Switching from Adam to SGD

RMSProp uses the same update rule as (1), but instead            to SGD. The switch is designed to be automatic and one
of accumulating vk in a monotonically increasing fashion,        that does not introduce any more hyper-parameters. The
uses an RMS-based approximation instead, i.e.,                   choice of not adding additional hyperparameters is deliber-
                                                                 ate since it allows for a fair comparison between Adam and
                                ˆ (wk−1 )2 .
          vk−1 = βvk−2 + (1 − β)∇f                               SWATS. Our experiments on several architectures and data
                                                                 sets suggest that such a strategy is indeed effective.
In both Adagrad and RMSProp, the accumulator v is ini-
tialized to 0. Owing to the fact that vk is monotonically        Several attempts have been made at improving the con-
increasing in each dimension for Adagrad, the scaling fac-       vergence and generalization performance of Adam. The
         ˆ (wk−1 ) monotonically decreases leading to slow
tor for ∇f                                                       closest to our proposed approach is (Zhang et al., 2017) in
progress. RMSProp corrects for this behavior by employ-          which the authors propose ND-Adam, a variant of Adam
ing an average scale instead of a cumulative scale. How-         which preserves the gradient direction by a nested opti-
ever, because v is initialized to 0, the initial updates tend    mization procedure. This, however, introduces an addi-
to be noisy given that the scaling estimate is biased by its     tional hyperparameter along with the (α, β1 , β2 ) used in
initialization. This behavior is rectified in Adam by em-        Adam. Further, empirically, this adaptation sacrifices the
ploying a bias correction. Further, it uses an exponential       rapid initial progress typically observed for Adam. In
moving average for the step in lieu of the gradient. Math-       Anonymous (2018), the authors investigate Adam and as-
ematically, the Adam update equation can be represented          cribe the poor generalization performance to training issues
as:                                                              arising from the non-monotonic nature of the steps. The
                           p                                     authors propose a variant of Adam called AMSGrad which
                             1 − β2k      mk−1
    wk = wk−1 − αk−1 ·                ·√           , where       monotonically reduces the step sizes and possesses theoret-
                            1 − β1k      vk−1 +                 ical convergence guarantees. Despite these guarantees, we
                                                           (2)   empirically found the generalization performance of AMS-
mk−1 = β1 mk−2 + (1 − β1 )∇f   ˆ (wk−1 ),                        Grad to be similar to that of Adam on problems where a
                                                                 generalization gap exists between Adam and SGD. We note
                           ˆ (wk−1 )2 .
 vk−1 = β2 vk−2 + (1 − β2 )∇f                             (3)    that in the context of the hypothesis of Wilson et al. (2017),
                                                                 all of the aforementioned methods would still yield poor
Adam has been used in many applications owing to its com-        generalization given that the scaling of the gradient is non-
petitive performance and its ability to work well despite        uniform.
minimal tuning (Karpathy, 2017). Recent work, however,
highlights the possible inability of adaptive methods to per-    The idea of switching from an adaptive method to SGD
form on par with SGD when measured by their ability to           is not novel and has been explored previously in the con-
generalize (Wilson et al., 2017).                                text of machine translation (Wu et al., 2016) and ImageNet
                                                                 training (Akiba et al., 2017). Wu et al. (2016) use such a
Furthermore, the authors also show that for even simple          mixed strategy for training and tune both the switchover
quadratic problems, adaptive methods find solutions that         point and the learning rate for SGD after the switch. Akiba
can be orders-of-magnitude worse at generalization than          et al. (2017) use a similar strategy but use a convex com-
those found by SGD(M).                                           bination of RMSProp and SGD steps whose contributions
Indeed, for several state-of-the-art results in language mod-    and learning rates are tuned.
eling and computer vision, the optimizer of choice is SGD        In our strategy, the switchover point and the SGD learn-
(Merity et al., 2017; Loshchilov & Hutter, 2016; He et al.,      ing rate are both learned as a part of the training process.
2015). Interestingly however, in these and other instances,      We monitor a projection of the Adam step on the gradi-
Adam outperforms SGD in both training and generaliza-            ent subspace and use its exponential average as an estimate
tion metrics in the initial portion of the training, but then    for the SGD learning rate after the switchover. Further, the
the performance stagnates. This motivates the investigation      switchover is triggered when no change in this monitored
of a strategy that combines the benefits of Adam, viz. good      quantity is detected. We describe this strategy in detail in
performance with default hyperparameters and fast initial        Section 2. In Section 3, we describe our experiments com-
progress, and the generalization properties of SGD. Given        paring Adam, SGD and SWATS on a host of benchmark
the insights of Wilson et al. (2017) which suggest that the      problems. Finally, in Section 4, we present ideas for future
lack of generalization performance of adaptive methods           research and concluding remarks. We conclude this section
stems from the non-uniform scaling of the gradient, a nat-       by emphasizing the goal of this work is less to propose a
ural hybrid strategy would begin the training process with       new training algorithm but rather to empirically investigate
Adam and switch to SGD when appropriate. To investi-             the viability of hybrid training for improving generaliza-
gate this further, we propose SWATS, a simple strategy that      tion.
combines the best of both worlds by Switching from Adam

## Page 3

                          Improving Generalization Performance by Switching from Adam to SGD

2. SWATS                                                                    25.0
                                                                                              SGD             Clip-(1, )
To investigate the generalization gap between Adam and
SGD, let us consider the training of the CIFAR-10 data
                                                                            22.5              Adam            Clip-(0,1)
set (Krizhevsky & Hinton, 2009) on the DenseNet archi-
tecture (Iandola et al., 2014). This is an example of an
                                                                            20.0
                                                                            17.5




                                                                Testing Error
instance where a significant generalization gap exists be-
tween Adam and SGD. We plot the performance of Adam
and SGD on this task but also consider a variant of Adam                    15.0
which we call Adam-Clip(p, q). Given (p, q) such that
p < q, the iterates for this variant take on the form                       12.5
wk = wk−1 −
       p                                         !                          10.0
          1 − β2k αk−1
  clip           √          , p · αsgd , q · αsgd mk−1 .
        1 − β1k    vk−1 +                                                      7.5
Here, αsgd is the tuned value of the learning rate for SGD                      5.0
that leads to the best performance for the same task. The                             0 25 50 75 100 125 150 175 200
function clip(x, a, b) clips the vector x element-wise such
that the output is constrained to be in [a, b]. Note that
                                                                                                 Epochs
Adam-Clip(1, 1) would correspond to SGD. The network
is trained using Adam, SGD and two variants: Adam-                Figure 1. Training the DenseNet architecture on the CIFAR-10
Clip(1, ∞), Adam-Clip(0, 1) with tuned learning rates for         data set with four optimizers: SGD, Adam, Adam-Clip(1, ∞) and
200 epochs, reducing the learning rate by 10 after 150            Adam-Clip(0, 1). SGD achieves the best testing accuracy while
epochs. The goal of this experiment is to investigate the         training with Adam leads to a generalization gap of roughly 2%.
effect of constraining the large                                  Setting a minimum learning rate for each parameter of Adam par-
                                √ and small step sizes that       tially closes the generalization gap.
                                1−β k
Adam implicitly learns, i.e., 1−β k2 √vαk−1
                                         k−1
                                            + , on the gen-
                                  1
eralization performance of the network. We present the re-
sults in Figure 1.
As seen from Figure 1, SGD converges to the expected test-        rapid initial progress. This raises two questions: (a) when
ing error of ≈ 5% while Adam stagnates in performance at          to switch over from Adam to SGD, and (b) what learn-
around ≈ 7% error. We note that fine-tuning of the learning       ing rate to use for SGD after the switch. Assuming that
rate schedule (primarily the initial value, reduction amount      the learning rate of SGD after the switchover is tuned, we
and the timing) did not lead to better performance. Also,         found that switching too late does not yield generalization
note that the rapid initial progress of Adam relative to SGD.     improvements while switching too early may cause the hy-
This experiment is in agreement with the experimental ob-         brid optimizer to not benefit from Adam’s initial progress.
servations of Wilson et al. (2017). Interestingly, Adam-          Indeed, as shown in Figure 2, switching after 10 epochs
Clip(0, 1) has no tangible effect on the final generalization     leads to a learning curve very similar to that of SGD, while
performance while Adam-Clip(1, ∞) partially closes the            switching after 80 epochs leads to inferior testing accuracy
generalization gap by achieving a final accuracy of ≈ 6%.         of ≈ 6.5%. To investigate the efficacy of a hybrid strat-
We observe similar results for several architectures, data        egy whilst ensuring no increase in the number of hyperpa-
sets and modalities whenever a generalization gap exists          rameters (a necessity for fair comparison with Adam), we
between SGD and Adam. This stands as evidence that the            propose SWATS, a strategy that automates the process of
step sizes learned by Adam could circumstantially be too          switching over by determining both the switchover point
small for effective convergence. This observation regarding       and the learning rate of SGD after the switch.
the need to lower-bound the step sizes of Adam is similar
to the one made in Anonymous (2018), where the authors            2.1. Learning rate for SGD after the switch
devise a one-dimensional example in which infrequent but
                                                                  Consider an iterate wk with a stochastic gradient gk and a
large gradients are not emphasized sufficiently causing the
                                                                  step computed by Adam, pk . For the sake of simplicity,
non-convergence of Adam.
                                                                  assume that pk 6= 0 and pTk gk < 0. This is a common
Given the potential insufficiency of Adam, even when con-         requirement imposed on directions to derive convergence
straining one side of the accumulator, we consider switch-        (Nocedal & Wright, 2006). In the case when β1 = 0 for
ing to SGD once we have reaped the benefits of Adam’s             Adam, i.e., no first-order exponential averaging is used, this

## Page 4

                                     Improving Generalization Performance by Switching from Adam to SGD

                                                                        implies the above equality.
            14                                                          Geometrically, this can be interpreted as the scaling neces-
                                                                        sary for the gradient that leads to its projection on the Adam
                                                                        step pk to be pk itself; see Figure 3. Note that this is not
            12                                                          the same as an orthogonal projection of pk on −gk . Empir-
Testing Error


                                                                        ically, we found that an orthogonal projection consistently
                                                                        underestimates the SGD learning rate necessary, leading to
            10                                                          much smaller SGD steps. Indeed, the `2 norm of an orthog-
                                                                        onally projected step will always be lesser than or equal to
                                                                        that of pk , which is undesirable given our needs. The non-
                8    Sw@10                                              orthogonal projection proposed above does not suffer from
                     Sw@40                                              this problem, and empirically we found that it estimates the
                                                                                                                                    kpk

                6    Sw@80                                              SGD learning rate well. A simple scaling rule of γk = kgk
                     SGD                                                was also not found to be successful. We attribute this to the
                                                                        fact that a scaling rule of this form ignores the relative im-
                     Adam                                               portance of the coordinate directions and tends to amplify
                40 25 50 75 100 125 150 175 200                         the importance of directions with a large step p but small
                                     Epochs                             first-order importance g, and vice versa.
                                                                        Note again that if no momentum (β1 = 0) is employed in
                                                                        Adam, then necessarily γk > 0 since Hk  0. We should
  Figure 2. Training the DenseNet architecture on the CIFAR-10          mention in passing that in this case γk is equivalent to the
  data set using Adam and switching to SGD with learning rate with      reciprocal of the Rayleigh Quotient of Hk−1 with respect to
  learning rate 0.1 and momentum 0.9 after (10, 40, 80) epochs;
                                                                        the vector pk .
  the switchover point is denoted by Sw@ in the figure. Switching
  early enables the model to achieve testing accuracy comparable        Since γk is a noisy estimate of the scaling needed, we main-
  to SGD but switching too late in the training process leads to a      tain an exponential average initialized at 0, denoted by λk
  generalization gap similar to Adam.                                   such that
                                                                                        λk = β2 λk−1 + (1 − β2 )γk .
  is trivially true since
                                                                        We use β2 of Adam, see (3), as the averaging coefficient
                         q
                                                                        since this reuse avoids another hyperparameter and also be-
                                  1 − β2k+1 αk
                    pk = −                 √        gk , t              cause the performance is relatively invariant to fine-grained
                                 1 − β1k+1   vk +                      specification of this parameter.
                             |           {z         }
                                     :=diag(Hk )
                                                                        2.2. Switchover Point
  with Hk  0 where diag(A) denotes the vector constructed
  from the diagonal of A. Ordinarily, to train using Adam, we           Having answered the question of what learning rate λk to
  would update the iterate as:                                          choose for SGD after the switch, we now discuss when to
                                                                        switch. We propose checking a simple, yet powerful, crite-
                             wk+1 = wk + pk .                           rion:

  To determine a feasible learning rate for SGD, γk , we pro-                                   λk
                                                                                                      − γk < ,                    (4)
  pose solving the subproblem for finding γk                                                  1 − β2k

                             proj−γk gk pk = pk                         at every iteration with k > 1. The condition compares the
                                                                        bias-corrected exponential averaged value and the current
  where proja b denotes the orthogonal projection of a onto b.          value (γk ). The bias correction is necessary to prevent the
  This scalar optimization problem can be solved in closed              influence of the zero initialization during the initial portion
  form to yield:                                                        of training. Once this condition is true, we switch over
                                pT pk                                   to SGD with learning rate Λ := (1−β    λk
                                                                                                                  k . We also experi-
                         γk = kT ,                                                                                2)
                               −pk gk                                   mented with more complex criteria including those involv-
  since                                                                 ing monitoring of gradient norms. However, we found that
                                            g T pk                      this simple un-normalized criterion works well across a va-
                    pk = proj−γk gk pk = −γk kT pk
                                            pk pk                       riety of different applications.

## Page 5

                             Improving Generalization Performance by Switching from Adam to SGD

          pk                                                          Algorithm 1 SWATS
                                                                      Inputs: Objective function f , initial point w0 , learn-
                                                                      ing rate α = 10−3 , accumulator coefficients (β1 , β2 ) =
                                                                      (0.9, 0.999),  = 10−9 , phase=Adam.
                                                                       1: Initialize k ← 0, mk ← 0, ak ← 0, λk ← 0
                                                                       2: while stopping criterion not met do
                                                                       3:    k =k+1
               −gk                                                     4:                                     ˆ (wk−1 )
                                                                             Compute stochastic gradient gk = ∇f
                                                                       5:    if phase = SGD then
                         −γk gk                                                 vk = β1 vk−1 + gk
wk                                                                     6:
                                                                       7:       wk = wk−1 − (1 − β1 )Λvk
                                                                       8:       continue
Figure 3. Illustrating the learning rate for SGD (γk ) estimated by    9:    end if
our proposed projection given an iterate wk , a stochastic gradient   10:    mk = β1 mk−1 + (1 − β1 )gk
                                                                                                       2
                                                                                        √ + (1 − β2 )gk
gk and the Adam step pk .                                             11:    ak = β2 ak−1
                                                                                             1−β k
                                                                      12:    pk = −αk 1−β k2 √amkk+
                                                                                              1
                                                                      13:    wk = wk + pk
In the case when β1 > 0, we switch to SGDM with learning              14:    if pTk gk 6= 0 then
rate (1 − β1 )Λ and momentum parameter β1 . The (1 −                                  pT p
                                                                      15:      γk = −pkT gkk
β1 ) factor is the common momentum correction. Refer to                                 k
                                                                      16:      λk = β2 λk−1 + (1 − β2 )γk
Algorithm 1 for a unified view of the algorithm. The text                                       λk
                                                                      17:      if k > 1 and | (1−β k − γk | <  then
in blue denotes operations that are also present in Adam.                                          2)
                                                                      18:         phase = SGD
                                                                      19:         vk = 0
3. Numerical Results                                                  20:         Λ = λk /(1 − β2k )
To demonstrate the efficacy of our approach, we present nu-           21:      end if
merical experiments comparing the proposed strategy with              22:   else
Adam and SGD. We consider the problems of image clas-                 23:      λk = λk−1
sification and language modeling.                                     24:   end if
                                                                      25: end while
For the former, we experiment with four architectures:                return wk
ResNet-32 (He et al., 2015), DenseNet (Iandola et al.,
2014), PyramidNet (Han et al., 2016), and SENet (Hu
et al., 2017) on the CIFAR-10 and CIFAR-100 data sets
(Krizhevsky & Hinton, 2009). The goal is to classify im-              broad importance, the inherent difficulties that arise due
ages into one of 10 classes for CIFAR-10 and 100 classes              to long term dependencies (Hochreiter & Schmidhuber,
for CIFAR-100. The data sets contain 50000 32 × 32 RGB                1997), and since it is a proxy for other sequence learning
images in the training set and 10000 images in the test-              tasks such as machine translation (Bahdanau et al., 2014).
ing set. We choose these architectures given their superior           We use the Penn Treebank (PTB) (Mikolov et al., 2011) and
performance on several image classification benchmarking              the larger WikiText-2 (WT-2) (Merity et al., 2016) data sets
tasks. For a large-scale image classification experiment,             and experimented with the AWD-LSTM and AWD-QRNN
we experiment with the Tiny-ImageNet data set1 on the                 architectures. In the case of SGD, we clip the gradients to a
ResNet-18 architecture (He et al., 2015). This data set is            norm of 0.25 while we perform no such clipping for Adam
a subset of the ILSVRC 2012 data set (Deng et al., 2009)              and SWATS. We found that the performance of SGD deteri-
and contains 200 classes with 500 224 × 224 RGB images                orates without clipping and that of Adam and SWATS with.
per class in the training set and 50 per class in the valida-         The AWD-LSTM architecture uses a multi-layered LSTM
tion and testing sets. We choose this data set given that it is       network with learned embeddings while the AWD-QRNN
a good proxy for the performance on the larger ImageNet               replaces the expensive LSTM layer by the cheaper QRNN
data set.                                                             layer (Bradbury et al., 2016) which uses convolutions in-
We also present results for word-level language modeling              stead of recurrences. The model is regularized with Drop-
where the task is to take as inputs a sequence of words and           Connect (Wan et al., 2013) on the hidden-to-hidden con-
predict the next word. We choose this task because of its             nections as well as other strategies such as weight decay,
                                                                      embedding-softmax weight tying, activity regularization
   1
       https://tiny-imagenet.herokuapp.com/                           and temporal activity regularization. We refer the reader

## Page 6

                          Improving Generalization Performance by Switching from Adam to SGD

to (Merity et al., 2016) for additional details regarding the   pens within the first 20 epochs for most CIFAR data sets
data sets including the sizes of the training, validation and   and at epoch 49 for Tiny-ImageNet. Curiously, in the case
testing sets, size of the vocabulary, and source of the data.   of the Tiny-ImageNet problem, the switch from Adam to
                                                                SGD leads to significant but temporary degradation in per-
For our experiments, we tuned the learning rate of all
                                                                formance. Despite the testing accuracy dropping from 80%
optimizers, and report the best-performing configuration
                                                                to 52% immediately after the switch, the model recovers
in terms of generalization. The learning rate of Adam
                                                                and achieves a better peak testing accuracy compared to
and SWATS were chosen from a grid of {0.0005, 0.0007,
                                                                Adam. We observed similar outcomes for several other ar-
0.001, 0.002, 0.003, 0.004, 0.005}. For both optimizers,
                                                                chitectures on this data set.
we use the (default) recommended values (β1 , β2 ) =
(0.9, 0.999). Note that this implies that, in all cases, we     In the language modeling tasks, Adam outperforms SGD
switch from Adam to SGDM with a momentum coefficient            not only in final generalization performance but also in
of 0.9. For tuning the learning rate for the SGD(M) op-         the number of epochs necessary to attain that performance.
timizer, we first coarsely tune the learning rate on a log-     This is not entirely surprising given that Merity et al. (2017)
arithmic scale from 10−3 to 102 and then fine-tune the          required iterate averaging for SGD to achieve state-of-the-
learning rate. For all cases, we experiment with and with-      art performance despite gradient clipping or learning rate
out employing momentum but don’t tune this parameter            decay rules. In this case, SWATS switches over to SGD,
(β = 0.9). We found this overall procedure to perform           albeit later in the training process, but achieves compara-
better than a generic grid-search or hyperparameter opti-       ble generalization performance to Adam as measured by
mization given the vastly different scales of learning rates    the lowest validation perplexity achieved in the experiment.
needed for different modalities. For instance, SGD with         Again, as in the case of the Tiny-ImageNet experiment
learning rate 0.7 performed best for the DenseNet task on       (Figure 5), the switch may cause a temporary degradation
CIFAR-10 but for the PTB language modeling task using           in performance from which the model is able to recover.
the LSTM architecture, a learning rate of 50 for SGD was
                                                                These experiments suggest that it is indeed possible to com-
necessary. Hyperparameters such as batch size, dropout
                                                                bine the best of both worlds for these tasks: in all the tasks
probability, `2 -norm decay etc. were chosen to match the
                                                                described, SWATS performs almost as well as the best al-
recommendations of the respective base architectures. We
                                                                gorithm amongst SGD and Adam, and in several cases
trained all networks for a total of 300 epochs and reduced
                                                                achieves a good initial decrease in the error metric.
the learning rate by 10 on epochs 150, 225 and 262. This
scheme was surprisingly powerful at obtaining good per-         Figure 7 shows that the estimated learning rate for SGD
formance across the different modalities and architectures.     (γk ) is noisy but convergent (in mean), and that it converges
The experiments were coded in PyTorch2 and conducted            to a value of similar scale as the value obtained by tuning
using job scheduling on 16 NVIDIA Tesla K80 GPUs for            the SGD optimizer (see Table 1). We emphasize that other
roughly 3 weeks.                                                than the learning rate, no other hyperparameters were tuned
                                                                between the experiments.
The experiments comparing SGD, Adam and SWATS on
the CIFAR and Tiny-ImageNet data sets are presented in
Figures 4 and 5, respectively. The experiments compar-          4. Discussion and Conclusion
ing the optimizers on the language modeling tasks are pre-
                                                                Wilson et al. (2017) pointed to the insufficiency of adaptive
sented in Figure 6. In Table 1, we summarize the meta-data
                                                                methods, such as Adam, Adagrad and RMSProp, at gener-
concerning our experiments including the learning rates
                                                                alizing in a fashion comparable to that of SGD. In the case
that achieved the best performance, and, in the case of
                                                                of a convex quadratic function, the authors demonstrate that
SWATS, the number of epochs before the switch occurred
                                                                adaptive methods provably converge to a point with orders-
and the learning rate (Λ) for SGD after the switch. Finally,
                                                                of-magnitude worse generalization performance than SGD.
in Figure 7, we depict the evolution of the estimated SGD
                                                                The authors attribute this generalization gap to the scal-
learning rate (γk ) as the algorithm progresses on two rep-
                                                                ing of the per-variable learning rates definitive of adaptive
resentative tasks.
                                                                methods as we explain below.
With respect to the image classification data sets, it is ev-
                                                                Nevertheless, adaptive methods are important given their
ident that, across different architectures, on all three data
                                                                rapid initial progress, relative insensitivity to hyperparam-
sets, Adam fails to find solutions that generalize well de-
                                                                eters, and ability to deal with ill-scaled problems. Several
spite making good initial progress. This is in agreement
                                                                recent papers have attempted to explain and improve adap-
with the findings of (Wilson et al., 2017). As can be
                                                                tive methods (Loshchilov & Hutter, 2017; Anonymous,
seen from Table 1, the switch from Adam to SGD hap-
                                                                2018; Zhang et al., 2017). However, given that they retain
   2                                                            the adaptivity and non-uniform gradient scaling, they too
       pytorch.org

## Page 7

                                        Improving Generalization Performance by Switching from Adam to SGD



             30                                             25                                             30
                                     SGD                                            SGD                                            SGD                    70                     SGD
             25                      Adam                   20                      Adam                   25                      Adam                                          Adam
                                                                                                                                                          60                     SWATS
                                     SWATS                                          SWATS                                          SWATS
             20                                                                                            20                                             50




                                                                                                                                              Testing Error
 Testing Error




                                                Testing Error




                                                                                               Testing Error
                                                            15
             15                                                                                            15                                             40
                                                            10                                                                                            30
             10                                                                                            10
                                                                                                                                                          20
                 5                                              5                                              5
                                                                                                                                                          10
                 0 0   50 100 150 200 250 300                   0 0   50 100 150 200 250 300                   0 0   50 100 150 200 250 300                    0   50 100 150 200 250 300
                            Epochs                                         Epochs                                         Epochs                                        Epochs
            (a) ResNet-32 — CIFAR-10                        (b) DenseNet — CIFAR-10                    (c) PyramidNet — CIFAR-10                              (d) SENet — CIFAR-10


             70                                             50                                             50                                             70
             65                      SGD                                            SGD                    45
                                                                                                                                   SGD                    65                     SGD
                                     Adam                   45                      Adam                                           Adam                                          Adam
             60                      SWATS                                          SWATS                  40                      SWATS                  60                     SWATS
             55                                             40                                                                                            55
 Testing Error




                                                Testing Error




                                                                                               Testing Error




                                                                                                                                              Testing Error
                                                                                                           35
             50                                                                                                                                           50
                                                            35                                             30
             45                                                                                                                                           45
                                                                                                           25
             40                                             30                                                                                            40
             35                                                                                            20                                             35
                                                            25
             30                                                                                            15                                             30
             25 0      50 100 150 200 250 300               20 0      50 100 150 200 250 300               10 0      50 100 150 200 250 300               25 0     50 100 150 200 250 300
                            Epochs                                         Epochs                                         Epochs                                        Epochs
         (e) ResNet-32 — CIFAR-100                        (f) DenseNet — CIFAR-100                  (g) PyramidNet — CIFAR-100                                (h) SENet — CIFAR-100

Figure 4. Numerical experiments comparing SGD(M), Adam and SWATS with tuned learning rates on the ResNet-32, DenseNet, Pyra-
midNet and SENet architectures on CIFAR-10 and CIFAR-100 data sets.




                         Model           Data Set                          SGDM       Adam                     SWATS         Λ      Switchover Point (epochs)
                         ResNet-32       CIFAR-10                            0.1       0.001                    0.001      0.52                                          1.37
                         DenseNet        CIFAR-10                            0.1       0.001                    0.001      0.79                                         11.54
                         PyramidNet      CIFAR-10                            0.1       0.001                   0.0007      0.85                                          4.94
                         SENet           CIFAR-10                            0.1       0.001                    0.001      0.54                                         24.19
                         ResNet-32       CIFAR-100                           0.3       0.002                    0.002      1.22                                         10.42
                         DenseNet        CIFAR-100                           0.1       0.001                    0.001      0.51                                         11.81
                         PyramidNet      CIFAR-100                           0.1       0.001                    0.001      0.76                                         18.54
                         SENet           CIFAR-100                           0.1       0.001                    0.001      1.39                                          2.04
                         LSTM            PTB                                 55†       0.003                   0.003       7.52                                        186.03
                         QRNN            PTB                                 35†       0.002                   0.002       4.61                                        184.14
                         LSTM            WT-2                                60†       0.003                   0.003       1.11                                        259.47
                         QRNN            WT-2                                60†       0.003                   0.004       14.46                                       295.71
                         ResNet-18       Tiny-ImageNet                       0.2       0.001                   0.0007      1.71                                         48.91

Table 1. Summarizing the optimal hyperparameters for SGD(M), Adam and SWATS for all experiments and, in the case of SWATS, the
value of the estimated learning rate for SGD after the switch and the switchover point in epochs. † denotes that no momentum was
employed for SGDM.

## Page 8

                             Improving Generalization Performance by Switching from Adam to SGD

               85                                                   in the column space of the X, and that only one optimum
                                                                    exists in that column space, viz. the minimum-norm so-
               80                                                   lution. On the other hand, adaptive methods do not nec-
                                                                    essarily stay in the column space of X. Similar arguments
                                                                    can be constructed for logistic regression problems (Soudry
               75
Testing Accuracy


                                                                    et al., 2017), but an analogous treatment for deep networks
                                                                    is, to the best of our knowledge, an open question. We
               70                                                   hypothesize that a successful implementation of a hybrid
                                                                    strategy, such as SWATS, suggests that in the case of deep
               65                                                   networks, despite training for few epochs before switching
                                                                    to SGD, the model is able to navigate towards a basin with
                                                                    better generalization performance. However, further em-
               60                                SGD                pirical and theoretical evidence is necessary to buttress this

               55                                Adam               hypothesis, and is a topic of future research.

                                                 SWATS              While the focus of this work has been on Adam, the strat-
                                                                    egy proposed is generally applicable and can be analo-
               50 0   50 100 150 200 250 300                        gously employed to other adaptive methods such as Ada-
                               Epochs                               grad and RMSProp. A viable research direction includes
                                                                    exploring the possibility of switching back-and-forth, as
                                                                    needed, from Adam to SGD. Indeed, in our preliminary
  Figure 5. Numerical experiments comparing SGD(M), Adam and        experiments, we found that switching back from SGD to
  SWATS with tuned learning rates on the ResNet-18 architecture     Adam at the end of a 300 epoch run for any of the ex-
  on the Tiny-ImageNet data set.                                    periments on the CIFAR-10 data set yielded slightly better
                                                                    performance. Along the same line, a future research direc-
                                                                    tion includes a smoother transition from Adam to SGD as
  are expected to suffer from similar generalization issues as      opposed to the hard switch proposed in this paper, which
  Adam. Motivated by this observation, we investigate the           may cause short-term performance degradation. This can
  question of using a hybrid training strategy that starts with     be achieved by using a convex combination of the SGD and
  an adaptive method and switches to SGD. By design, both           Adam directions as in the case of Akiba et al. (2017), and
  the switchover point and the learning rate for SGD after          gradually increasing the weight for the SGD contribution
  the switch, are determined as a part of the algorithm and         by a criterion. Finally, we note that the strategy proposed
  as such require no added tuning effort. We demonstrate            in this paper does not preclude the use of those proposed in
  the efficacy of this approach on several standard bench-          Zhang et al. (2017); Loshchilov & Hutter (2017); Anony-
  marks, including a host of architectures, on the PennTree         mous (2018). We plan to investigate the performance of
  Bank, WikiText-2, Tiny-ImageNet, CIFAR-10 and CIFAR-              the algorithm obtained by mixing these strategies, such
  100 data sets. In summary, our results show that the pro-         as monotonic increase guarantees of the second-order mo-
  posed strategy leads to results comparable to SGD while           ment, cosine-annealing, `2 -norm correction, in the future.
  retaining the beneficial properties of Adam such as hyper-
  parameter insensitivity and rapid initial progress.               References
  The success of our strategy motivates a deeper exploration        Akiba, T., Suzuki, S., and Fukuda, K. Extremely large
  into the interplay between the dynamics of the optimizer            minibatch SGD: Training resnet-50 on ImageNet in 15
  and the generalization performance. Recent theoretical              minutes. arXiv preprint arXiv:1711.04325, 2017.
  work analyzing generalization for deep learning suggests
  coupling generalization arguments with the training pro-          Anonymous. On the convergence of Adam and be-
  cess (Soudry et al., 2017; Hardt et al., 2015; Zhang et al.,        yond. International Conference on Learning Represen-
  2016; Wilson et al., 2017). The optimizers choose differ-           tations, 2018. URL https://openreview.net/
  ent trajectories in the parameter space and are attracted to        forum?id=ryQu7f-RZ.
  different basins of attractions, with vastly different general-
  ization performance. Even for a simple least-squares prob-        Bahdanau, D., Cho, K., and Bengio, Y. Neural machine
  lem: minw kXw − yk22 with w0 = 0, SGD recovers the                  translation by jointly learning to align and translate.
  minimum-norm solution, with its associated margin bene-             arXiv preprint arXiv:1409.0473, 2014.
  fits, whereas adaptive methods do not. The fundamental
  reason for this is that SGD ensures that the iterates remain      Bradbury, J., Merity, S., Xiong, C., and Socher, R.

## Page 9

                                                            Improving Generalization Performance by Switching from Adam to SGD

                     100                                                             100                                                       100                                                     100
                                                       SGD                               95
                                                                                                                   SGD                             95
                                                                                                                                                                        SGD                                95
                                                                                                                                                                                                                                SGD
                         90                            Adam                                                        Adam                                                 Adam                                                    Adam
                                                       SWATS                                                       SWATS                                                SWATS                                                   SWATS
 Validation Perplexity




                                                                 Validation Perplexity




                                                                                                                           Validation Perplexity




                                                                                                                                                                                   Validation Perplexity
                                                                                         90                                                        90                                                      90
                         80                                                              85                                                        85                                                      85
                                                                                         80                                                        80                                                      80
                         70                                                              75                                                        75                                                      75
                                                                                         70                                                        70                                                      70
                         60
                                                                                         65                                                        65                                                      65
                         50 0       50 100 150 200 250 300                               60 0     50 100 150 200 250 300                           60 0   50 100 150 200 250 300                           60 0   50 100 150 200 250 300
                                           Epochs                                                        Epochs                                                Epochs                                                  Epochs
                                (a) LSTM — PTB                                                (b) LSTM — WT2                                            (c) QRNN — PTB                                          (d) QRNN — WT2

Figure 6. Numerical experiments comparing SGD(M), Adam and SWATS with tuned learning rates on the AWD-LSTM and AWD-
QRNN architectures on PTB and WT-2 data sets.


                                                                        100
                                                                                                                                             Hu, J., Shen, L., and Sun, G. Squeeze-and-excitation net-
            1.8
                                                                             80                                                                works. arXiv preprint arXiv:1709.01507, 2017.
            1.6
            1.4                                                                                                                              Iandola, F., Moskewicz, M., Karayev, S., Girshick, R.,
                                                                             60
            1.2                                                                                                                                Darrell, T., and Keutzer, K. Densenet: Implementing
            1.0                                                              40                                                                efficient convnet descriptor pyramids. arXiv preprint
            0.8                                                                                                                                arXiv:1404.1869, 2014.
                                                                             20
            0.6
                                                                                   0 0                                                       Karpathy, A. A Peek at Trends in Machine Learn-
                          0     2     4     6      8   10   12                                  25 50 75 100 125 150 175
                                          Epochs                                                       Epochs                                  ing.     https://medium.com/@karpathy/a-
                                                                                                                                               peek-at-trends-in-machine-learning-
      (a) DenseNet — CIFAR-100                                                            (b) QRNN — PTB
                                                                                                                                               ab8a1085a106, 2017. [Online; accessed 12-Dec-
                                                                                                                                               2017].
Figure 7. Evolution of the estimated SGD learning rate (γk ) on
two representative tasks.                                                                                                                    Kingma, D. and Ba, J. Adam: A method for stochastic
                                                                                                                                               optimization. In International Conference on Learning
            Quasi-Recurrent Neural Networks.                                                          arXiv preprint                           Representations (ICLR 2015), 2015.
            arXiv:1611.01576, 2016.                                                                                                          Krizhevsky, A. and Hinton, G. Learning multiple layers of
                                                                                                                                               features from tiny images. 2009.
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-
  Fei, L. ImageNet: A Large-Scale Hierarchical Image                                                                                         Lee, J., Simchowitz, M., Jordan, M. I, and Recht, B. Gradi-
  Database. In CVPR09, 2009.                                                                                                                   ent descent converges to minimizers. University of Cal-
                                                                                                                                               ifornia, Berkeley, 1050:16, 2016.
Duchi, J., Hazan, E., and Singer, Y. Adaptive subgradient
  methods for online learning and stochastic optimization.                                                                                   Loshchilov, I. and Hutter, F. SGDR: Stochastic gradient
  The Journal of Machine Learning Research, 12:2121–                                                                                           descent with warm restarts. 2016.
  2159, 2011.
                                                                                                                                             Loshchilov, I. and Hutter, F. Fixing Weight Decay Regu-
Han, D., Kim, J., and Kim, J. Deep pyramidal residual                                                                                          larization in Adam. ArXiv e-prints, November 2017.
  networks. arXiv preprint arXiv:1610.02915, 2016.
                                                                                                                                             Mandt, S., Hoffman, M. D., and Blei, D. M. Stochastic
Hardt, M., Recht, B., and Singer, Y. Train faster, generalize                                                                                 Gradient Descent as Approximate Bayesian Inference.
  better: Stability of stochastic gradient descent. arXiv                                                                                     ArXiv e-prints, April 2017.
  preprint arXiv:1509.01240, 2015.
                                                                                                                                             Merity, S., Xiong, C., Bradbury, J., and Socher, R.
He, K., Zhang, X., Ren, S., and Sun, J. Deep resid-                                                                                           Pointer sentinel mixture models.     arXiv preprint
  ual learning for image recognition. arXiv preprint                                                                                          arXiv:1609.07843, 2016.
  arXiv:1512.03385, 2015.
                                                                                                                                             Merity, S., Keskar, N., and Socher, R. Regularizing and
Hochreiter, S. and Schmidhuber, J. Long short-term mem-                                                                                       Optimizing LSTM Language Models. arXiv preprint
  ory. Neural computation, 9(8):1735–1780, 1997.                                                                                              arXiv:1708.02182, 2017.

## Page 10

                         Improving Generalization Performance by Switching from Adam to SGD

Mikolov, T., Kombrink, S., Deoras, A., Burget, L., and Cer-
 nocky, J. RNNLM-recurrent neural network language
 modeling toolkit. In Proc. of the 2011 ASRU Workshop,
 pp. 196–201, 2011.
Nocedal, J. and Wright, S. Numerical optimization.
  Springer Science & Business Media, 2006.

Robbins, Herbert and Monro, Sutton. A stochastic approx-
  imation method. The annals of mathematical statistics,
  pp. 400–407, 1951.
Soudry, D., Hoffer, E., and Srebro, N. The implicit bias
  of gradient descent on separable data. arXiv preprint
  arXiv:1710.10345, 2017.
Sutskever, I., Martens, J., Dahl, G., and Hinton, G. On
  the importance of initialization and momentum in deep
  learning. In International conference on machine learn-
  ing, pp. 1139–1147, 2013.

Tieleman, T. and Hinton, G. Lecture 6.5-RMSProp: Divide
  the gradient by a running average of its recent magni-
  tude. COURSERA: Neural Networks for Machine Learn-
  ing, 4, 2012.
Wan, L., Zeiler, M., Zhang, S., LeCun, Y, and Fergus, R.
 Regularization of neural networks using dropconnect. In
 Proceedings of the 30th international conference on ma-
 chine learning (ICML-13), pp. 1058–1066, 2013.
Wilson, A. C., Roelofs, R., Stern, M., Srebro, N., and
 Recht, B. The Marginal Value of Adaptive Gradient
 Methods in Machine Learning. ArXiv e-prints, May
 2017.
Wu, Y., Schuster, M., Chen, Z., Le, Q., Norouzi, M.,
 Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey,
 K., et al. Google’s neural machine translation system:
 Bridging the gap between human and machine transla-
 tion. arXiv preprint arXiv:1609.08144, 2016.
Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals,
  O. Understanding deep learning requires rethinking gen-
  eralization. arXiv preprint arXiv:1611.03530, 2016.

Zhang, Z., Ma, L., Li, Z., and Wu, C.         Nor-
  malized direction-preserving Adam. arXiv preprint
  arXiv:1709.04546, 2017.

