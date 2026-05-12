# Local alphaXiv Copy: ViM: Out-Of-Distribution with Virtual-logit Matching

- Source ID: `arxiv-2203.10807v1`
- arXiv ID: `2203.10807v1`
- Evidence note: alphaXiv content is a navigation aid only; verify exact claims against `paper_pdf_pages.md` and the original PDF.

---

## Research Paper Report: ViM: Out-Of-Distribution with Virtual-logit Matching

**Paper Title:** ViM: Out-Of-Distribution with Virtual-logit Matching
**Authors:** Haoqi Wang, Zhizhong Li, Litong Feng, Wayne Zhang
**arXiv ID:** arXiv:2203.10807v1 [cs.CV]

---

### 1. Authors, Institution(s), and Notable Context About the Research Group

The research paper "ViM: Out-Of-Distribution with Virtual-logit Matching" is authored by Haoqi Wang, Zhizhong Li, Litong Feng, and Wayne Zhang. The primary affiliation for all authors is **SenseTime Research**. Additionally, Wayne Zhang, identified as the corresponding author, also holds an affiliation with the **Qing Yuan Research Institute, Shanghai Jiao Tong University**. This dual affiliation is noteworthy as it suggests a collaborative bridge between a leading industrial AI research powerhouse (SenseTime) and a prominent academic institution (Shanghai Jiao Tong University).

SenseTime is a globally recognized artificial intelligence company, particularly renowned for its advancements in computer vision, deep learning, and facial recognition technologies. Their research output frequently pushes the boundaries of applied AI, and contributions from SenseTime Research often have a strong focus on practical applicability and real-world deployment challenges. The fact that this work directly addresses the "open-world" problem in AI deployment aligns well with SenseTime's industrial focus.

Haoqi Wang and Zhizhong Li are noted as contributing equally to the work, indicating a significant collaborative effort in the development and execution of the research. The acknowledgment section further provides context regarding support for this work. It mentions funding from the Innovation and Technology Commission of the Hong Kong Special Administrative Region, China, under the Enterprise Support Scheme, and support for Haoqi Wang through the Technology Leaders of Tomorrow (TLT) Programme of HKSTP InnoAcademy. This highlights the strong ties of the research group to the vibrant innovation and technology ecosystem in Hong Kong, emphasizing governmental and institutional backing for cutting-edge AI research.

### 2. How This Work Fits into the Broader Research Landscape

This research is situated within the critical and rapidly evolving field of **Out-Of-Distribution (OOD) detection** in deep learning. Modern deep learning models are typically trained in a "closed-world" setting, meaning they are exposed only to data from predefined categories. However, when these models are deployed in real-world "open-world" environments, they inevitably encounter inputs that differ significantly from their training distribution. Without proper OOD detection mechanisms, these models can make confident, yet catastrophically wrong, predictions on unfamiliar data, undermining their reliability and trust in sensitive applications like autonomous driving, medical diagnosis, and industrial inspection.

The existing landscape of OOD detection methods can broadly be categorized into two main types:
*   **A Posteriori Scoring Methods:** These methods typically compute an OOD score from a pre-trained neural network without requiring any re-training or access to OOD data during the training phase. They derive scores from different sources within the network, primarily:
    *   **Softmax Probabilities:** e.g., Maximum Softmax Probability (MSP) [13], KL-divergence methods [12].
    *   **Logits:** e.g., MaxLogit, Energy score [25].
    *   **Feature Space:** e.g., methods utilizing feature norms, residuals, or Mahalanobis distances [23, 27].
    The key limitation of these approaches, as highlighted by the authors, is their reliance on a *single* input source. The paper convincingly argues that the immense diversity of OOD examples makes single-source methods fragile; an OOD sample easily identifiable in feature space might be hard to distinguish in logit space, and vice versa. This observation forms the core motivation for ViM.
*   **Training-Time Modification Methods:** These approaches aim to make the network inherently more OOD-aware by modifying the training objective, adding regularization losses, or exposing the model to auxiliary OOD data (e.g., Outlier Exposure [14]). While often effective, these methods require re-training the original network, which can be computationally expensive, time-consuming, and difficult to apply to already deployed large-scale models.

ViM (Virtual-logit Matching) positions itself as an advancement in the **a posteriori scoring methods** by directly addressing the fragility of single-source approaches. It proposes a novel score that intelligently **combines information from both feature space and logit space**, thereby leveraging both class-agnostic and class-dependent insights from the model. This makes ViM a powerful, lightweight, and easily deployable solution, as it does not require model re-training or additional OOD data for its operation.

Beyond algorithmic contributions, the paper also addresses a critical bottleneck in the OOD detection research landscape: the **shortage of large-scale, clean, and realistic OOD datasets**. Current datasets are often curated from existing public datasets with predefined tag lists, leading to potential biases, noise, and "hackability" due to their narrow coverage. The introduction of OpenImage-O, a manually annotated, diverse, and significantly larger OOD dataset for ImageNet-1K, is a direct response to this need. It aims to provide a more robust and realistic benchmark for evaluating large-scale OOD detection algorithms, thereby accelerating research progress in the field.

In summary, this work contributes to the OOD detection field by offering a more robust and practical scoring algorithm through multi-source information fusion, while simultaneously providing a much-needed, high-quality large-scale dataset to enable more reliable future research and evaluation.

### 3. Key Objectives and Motivation

The core motivation behind this research stems from a critical observed limitation in existing Out-Of-Distribution (OOD) detection algorithms: their **fragility due to dependence on a single input source**. The authors empirically demonstrate, particularly through Figure 1 and 2, that OOD examples exhibit immense diversity. Some OOD samples are easily detectable in the feature space of a neural network (e.g., via their deviation from learned data manifolds), while being indistinguishable in the logit space (i.e., yielding high confidence scores for in-distribution classes). Conversely, other OOD samples might exhibit low confidence in the logit space but not necessarily stand out in the feature space. This "one-size-fits-all" approach, relying solely on features, logits, or softmax probabilities, limits the robustness and generalizability of current OOD detectors.

This observation directly leads to the paper's primary objectives:

1.  **Develop a novel and robust OOD scoring method (ViM) that effectively combines information from multiple sources.** Specifically, the goal is to intelligently fuse:
    *   **Class-agnostic information from the feature space:** This captures general deviations of an input from the learned data manifold, often residing in the null space or orthogonal complement of principal components.
    *   **Class-dependent information from the logits:** This reflects the model's confidence and similarity to specific in-distribution classes.
    The aim is to overcome the limitations of single-source methods without requiring model re-training or exposure to auxiliary OOD data, thereby ensuring lightweight computation and ease of deployment.

2.  **Address the critical shortage of high-quality, large-scale OOD datasets for evaluating large-scale semantic spaces.** The authors identify that existing datasets, often derived from other public datasets by tag lists, suffer from issues like:
    *   **Noise and indistinguishability:** Images selected by class labels may still contain instances that are not truly OOD (e.g., "bubbly texture" overlapping with "bubble" class in ImageNet).
    *   **Small coverage and "hackability":** Datasets with a narrow scope (e.g., only textures) might be easily targeted by specific detectors, leading to biased performance comparisons that don't reflect real-world diversity.
    Therefore, a key objective is to **curate a new, human-annotated OOD benchmark for ImageNet-1K, named OpenImage-O**, that is significantly larger, more diverse, and reflects natural class statistics, thus fostering more rigorous and realistic OOD detection research.

3.  **Conduct comprehensive experiments to demonstrate the effectiveness and robustness of the proposed ViM score.** This includes evaluating ViM across various deep learning architectures (including both traditional CNNs like ResNet/BiT and modern Vision Transformers like ViT/Swin) and against a wide range of existing OOD benchmarks, validating its superior performance and broad applicability.

By pursuing these objectives, the research aims to significantly advance the reliability and practicality of OOD detection systems for real-world applications of deep learning.

### 4. Methodology and Approach

The proposed methodology for ViM (Virtual-logit Matching) revolves around a novel OOD scoring function that strategically combines insights from both the feature space and the logit space of a pre-trained classification model. The approach can be broken down into three main steps:

1.  **Principal Subspace and Residual Extraction (Feature Space Analysis):**
    *   **Bias-Free Feature Space:** The first step involves transforming the feature space by offsetting it with a vector `o = -(W^T)^+ b`, where `W` and `b` are the classification layer's weights and biases, respectively. This ensures that the logits become purely a function of the inner product with `W^T` times the new feature, simplifying geometric interpretations.
    *   **Principal Subspace Learning:** A D-dimensional principal subspace `P` is learned from a large sample of the in-distribution (ID) training set. This is achieved by performing eigendecomposition on the covariance matrix `X^T X` of the ID features (in the new bias-free coordinate system), where `X` is the ID data matrix. The principal subspace `P` is spanned by the eigenvectors corresponding to the `D` largest eigenvalues. This subspace is presumed to capture the most significant variations and directions of the ID data.
    *   **Residual Calculation:** For any input feature `x` (also transformed to the bias-free system), its "residual" `x_P_perp` is computed. This residual is the projection of `x` onto `P_perp`, the orthogonal complement of the principal subspace `P`. Intuitively, features that deviate significantly from the principal space (i.e., have a large `||x_P_perp||`) are more likely to be OOD examples, as ID data is assumed to largely reside within or close to this principal subspace.

2.  **Virtual-logit Matching (Bridging Feature and Logit Spaces):**
    *   **Virtual Logit Generation:** The norm of the residual, `||x_P_perp||`, which is a class-agnostic measure of OOD-ness from the feature space, is converted into a "virtual logit" (`l_0`). This virtual logit represents a hypothetical "OOD class."
    *   **Scale Matching:** A crucial aspect is to correctly scale this virtual logit. A scaling constant `alpha` is introduced such that `l_0 = alpha * ||x_P_perp||`. This `alpha` is determined by matching the average `||x_P_perp||` over a large sample of training examples to the average *maximum logit* (`max(l_j)`) of those same training examples. This ensures that, on average, the virtual logit has a scale comparable to the dominant in-distribution logits, preventing it from being trivially overwhelmed or dominating in the subsequent softmax computation. This matching process is done once, offline, on the training set.

3.  **ViM Score Computation (Probability Space):**
    *   **Softmax Augmentation:** The calculated virtual logit `l_0` is appended to the original `C` logits (`l_1, ..., l_C`) produced by the classification network.
    *   **OOD Probability as Score:** A softmax function is then applied to this augmented `C+1` dimensional vector of logits. The probability corresponding to the virtual logit `l_0` is defined as the ViM score. A higher ViM score indicates a higher likelihood of the input being OOD.
    *   **Underlying Principle:** The formula `ViM(x) = exp(l_0) / (sum(exp(l_i)) + exp(l_0))` implicitly captures the desired behavior: if original logits `l_i` are high (strong ID confidence) relative to `l_0`, the ViM score will be low. Conversely, if `l_0` is high (large residual, strong feature-level OOD signal) and `l_i` are low, the ViM score will be high. The authors also note an equivalent expression where `alpha*||x_P_perp|| - log(sum(exp(l_i)))` is used, highlighting a connection to the Energy score.
    *   **Efficiency:** The computational overhead of ViM is minimal, comparable to that of the final fully-connected layer in the classification network, making it suitable for real-time inference.

**Dataset Construction: OpenImage-O**
To provide a robust evaluation platform, the authors meticulously curated a new OOD dataset called OpenImage-O for ImageNet-1K.
*   **Source:** Images were sourced from the test set of OpenImage-v3, a large dataset collected without predefined class names, aiming for natural class statistics.
*   **Annotation Process:** Human labelers were tasked with classifying images as OOD or ID relative to ImageNet-1K. To simplify, labelers were shown the image and the top 10 categories predicted by an ImageNet-1K model, along with visually similar ID images for reference. They could choose "yes" (OOD), "no" (ID), or "difficult."
*   **Quality Control:** Each image was labeled independently by at least two labelers, and only images with consensus were included. Random inspections were also performed.
*   **Scale and Diversity:** This process resulted in 17,632 manually filtered OOD images, making it 7.8 times larger than the previous ImageNet-O dataset and highly diverse, addressing the shortcomings of prior OOD benchmarks.

### 5. Main Findings and Results

The extensive experimental evaluation demonstrates the significant effectiveness and robustness of the proposed ViM score across various models and diverse OOD datasets.

*   **Overall Superior Performance:**
    *   **BiT Model (CNN-based):** ViM achieved an impressive average AUROC of 90.91% across four difficult OOD benchmarks (OpenImage-O, Texture, iNaturalist, ImageNet-O). This represents a substantial improvement of 4.29% over the best baseline (Mahalanobis, 86.62%) and also boasts the lowest average FPR95.
    *   **ViT Model (Transformer-based):** ViM continues to be a top performer, with an average AUROC of 96.23%, closely competitive with Mahalanobis (96.02%). The results show that Vision Transformers, often pre-trained on larger datasets (like ImageNet-21K for ViT), generally yield higher OOD detection performance than CNNs, though ViT surprisingly performed less competitively on the Texture dataset.
    *   **Other Architectures:** ViM consistently maintained its leading performance across a wider range of CNN-based (RepVGG, ResNet-50d) and Transformer-based (Swin Transformer, DeiT) models, affirming its architectural robustness.

*   **Effectiveness of Information Fusion:** The results strongly validate the core hypothesis that combining feature-space (class-agnostic) and logit-space (class-dependent) information is crucial. ViM significantly outperformed methods relying solely on features (e.g., Residual) or logits/probabilities (e.g., Energy, KL Matching). For instance, on the Texture dataset with BiT, feature-based methods like Residual (97.66% AUROC) and Mahalanobis (97.33%) far surpassed logit/softmax-based methods like KL Matching (86.92%). ViM (98.92%) successfully leveraged the strengths of both, showing marked improvements over Residual and Energy individually, indicating a non-trivial and effective fusion.

*   **Dataset Characteristics and Impact:**
    *   **OpenImage-O Validation:** The newly curated OpenImage-O dataset proved to be a challenging and effective benchmark, with its natural class distribution and large scale. The smaller performance gap between the best and average methods on OpenImage-O (5.61%) compared to Texture (10.52%) and ImageNet-O (14.39%) suggests it is less "hackable" and provides a more realistic evaluation of general OOD capabilities.
    *   **Dataset-Specific Performance:** The paper noted that ViM showed slightly less performance gain on iNaturalist compared to other datasets. This was hypothesized to be related to the smaller average residual norm in iNaturalist samples, suggesting that for certain fine-grained OOD types, the feature space residual might carry less distinct information.

*   **Robustness to Hyperparameters and Model Selection:**
    *   **Principal Space Dimension (D):** ViM demonstrated robustness to the choice of the principal space dimension D, with performance remaining stable across a wide range of values.
    *   **Matching Parameter (α):** The calculated `alpha` value (from matching average residual norm to average max logit) proved to be an empirically good choice, as perturbing it by a multiplication factor generally led to performance degradation. This suggests the proposed scaling mechanism effectively balances the contributions of feature and logit information.

*   **Efficiency:** ViM is highly efficient. When compared to other competitive methods (Mahalanobis, KL Matching, Residual) after feature extraction, ViM's score computation time was the fastest, comparable to Residual and significantly faster than Mahalanobis (orders of magnitude) and KL Matching.

*   **Comparison with Grouping Methods:** While grouping information (as in MOS or MaxGroup) can sometimes improve OOD detection, ViM (and its group-aware variant ViM+Group) consistently outperformed group-based baselines, including MOS (which requires fine-tuning). This underscores the fundamental strength of ViM's information fusion mechanism. Grouping was notably less effective for ViT models.

In summary, the results highlight ViM as a state-of-the-art, robust, and efficient OOD detection method that effectively mitigates the fragility of single-source approaches by intelligently combining feature-level and logit-level information. The OpenImage-O dataset provides a much-needed, high-quality benchmark for future research.

### 6. Significance and Potential Impact

The contributions of "ViM: Out-Of-Distribution with Virtual-logit Matching" carry significant implications for both academic research and the practical deployment of deep learning models.

1.  **Advancement in OOD Detection Robustness:** ViM directly addresses a fundamental weakness of existing OOD detection methods: their fragility when relying on a single information source. By effectively combining class-agnostic information from the feature space and class-dependent information from the logit space, ViM offers a more comprehensive and robust OOD scoring mechanism. This multi-modal fusion approach represents a significant step forward in making OOD detectors more resilient to the immense and unpredictable diversity of real-world out-of-distribution data.

2.  **Enhanced Practicality and Deployability:** A major strength of ViM is its lightweight nature. It operates *a posteriori* on a pre-trained model, requiring no re-training or exposure to auxiliary OOD data. This eliminates the substantial computational overhead and complexity associated with retraining-based methods, making ViM highly practical for integration into existing deep learning pipelines and deployed systems. Its fast inference time further ensures that it can be applied in real-time scenarios without introducing significant latency. This ease of adoption lowers the barrier for developers to incorporate robust OOD detection into their applications.

3.  **Critical Contribution to Research Benchmarking:** The introduction of OpenImage-O is a vital contribution to the OOD detection research community. The dataset addresses several shortcomings of previous benchmarks:
    *   **Scale:** Its significantly larger size (8.8x ImageNet-O) provides a more comprehensive evaluation ground.
    *   **Quality:** Being human-annotated at the image level reduces noise and ambiguity inherent in tag-based or automatically generated datasets.
    *   **Realism:** Its natural class statistics, derived from a broad source like OpenImage-v3, mitigate biases and "hackability" often seen in narrowly focused or adversarially curated datasets.
    OpenImage-O will serve as a more reliable and challenging benchmark, encouraging the development of truly generalizable OOD detection algorithms and accelerating progress in the field.

4.  **Improved Model Reliability and Safety:** For critical applications such as autonomous driving, medical imaging, industrial quality control, and fraud detection, accurately identifying OOD inputs is paramount. A model confidently misclassifying an unknown object or medical condition can lead to severe consequences. By improving OOD detection performance, ViM directly contributes to building safer, more reliable, and more trustworthy AI systems, allowing them to effectively signal "I don't know" when confronted with unfamiliar scenarios.

5.  **Foundation for Future Research:** The success of ViM's information fusion paradigm opens new avenues for future research. It encourages further exploration into how different levels and types of information within a neural network can be synergistically combined for OOD detection, potentially leading to even more sophisticated and robust methods. The comprehensive benchmarking across CNNs and Vision Transformers also provides valuable insights into the OOD characteristics of different model architectures, guiding future model design.

In conclusion, ViM is not just an incremental improvement but a significant step towards more robust, practical, and dependable out-of-distribution detection. Coupled with the valuable OpenImage-O dataset, this work creates a strong foundation for advancing the trustworthiness and real-world applicability of deep learning models.
