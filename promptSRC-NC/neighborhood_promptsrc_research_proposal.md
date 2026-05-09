# Research Proposal: Neighborhood-Consistent PromptSRC

## Working title

**Neighborhood-Consistent PromptSRC: Using Unlabeled Target Geometry for Few-Shot CLIP Prompt Learning**

## Executive summary

This project studies a narrow, technically simple question:

> Can a completed few-shot PromptSRC solution be improved by a short second-stage adaptation that uses unlabeled target images only through frozen-CLIP nearest-neighbor relations?

The proposed method starts from the official PromptSRC algorithm, implemented here as standalone PyTorch/OpenCLIP code for Modal cloud runs. PromptSRC first learns deep visual and textual prompts from the labeled few-shot split while keeping CLIP’s pretrained image and text encoders frozen. We then add a lightweight second stage: compute frozen-CLIP image embeddings for unlabeled training images, form reliable nearest-neighbor pairs, and continue prompt optimization with a symmetric prediction-consistency loss over those pairs. The unlabeled images never receive pseudo-labels. They provide only local relational structure: “these two images are close under frozen CLIP, so the prompted classifier should not predict them very differently.”

The minimal experimental design uses three variants:

1. **PromptSRC**: official baseline, trained as in the paper and repository.
2. **PromptSRC-NC**: the proposed method, using real frozen-CLIP neighbor pairs.
3. **PromptSRC-NC shuffled-neighbor control**: same second-stage loss, same unlabeled images, same extra training time, but with the meaningful frozen-CLIP neighborhood topology destroyed.

This directly tests the central scientific claim. If the real-neighbor variant improves over PromptSRC and over the shuffled-neighbor control, the evidence supports the claim that useful information comes from the geometry of the unlabeled target space, rather than from extra optimization time or arbitrary unlabeled smoothing.

The target datasets are **Flowers102**, **EuroSAT**, and **Stanford Cars**. These are deliberately diverse: Flowers102 and Stanford Cars are fine-grained recognition datasets, while EuroSAT is satellite land-use/land-cover classification. This lets the project test the method under different CLIP-neighborhood reliability regimes.

---

## 1. Motivation

### 1.1 CLIP and prompt learning

CLIP learns a joint image-text representation by training an image encoder and a text encoder on large-scale image-text pairs. At inference time, class names can be placed inside natural-language templates such as “a photo of a {class},” encoded by the text encoder, and compared to an image embedding through cosine similarity. This gives CLIP strong zero-shot recognition ability without dataset-specific model training.

Prompt learning adapts CLIP more efficiently than full fine-tuning by freezing the pretrained encoders and learning a small number of prompt vectors. These prompt vectors can be appended to the language input, the visual input, or both. This is attractive for few-shot learning because only a tiny number of parameters are updated.

However, few-shot prompt tuning has a core failure mode: the prompts can overfit the labeled few-shot examples. The prompted model may improve on the supervised classes while drifting away from CLIP’s generalizable pretrained geometry.

### 1.2 What PromptSRC contributes

PromptSRC is a self-regularized prompt learning framework designed to address prompt overfitting. The paper states the problem clearly: prompts trained with task-specific cross-entropy can overfit downstream data distributions and lose CLIP’s original generalization capability. PromptSRC tries to learn prompts that capture both task-specific and task-agnostic general representations.

PromptSRC has three components:

1. **Mutual agreement maximization**  
   Prompted visual/text features and prompted logits are regularized toward the corresponding frozen-CLIP visual/text features and frozen-CLIP logits.

2. **Gaussian weighted prompt aggregation (GPA)**  
   Prompts learned at different epochs are aggregated with Gaussian weights. The intuition is that early prompts are immature/noisy, very late prompts may be overly task-specific, and mid-training prompts may capture a useful balance.

3. **Textual diversity**  
   Frozen CLIP text features are computed using an ensemble of multiple prompt templates, rather than a single class-name template, to reduce the mismatch between the visual branch’s many image samples and the text branch’s limited class-name samples.

In the official code, PromptSRC builds an independent vision-language prompt design over CLIP. It learns deep visual and text prompts while freezing the underlying CLIP encoders. For base-to-novel and few-shot settings, the official implementation uses ViT-B/16 CLIP, four visual prompt tokens, four text prompt tokens, prompt depth nine for both branches, SGD with learning rate 0.0025, and PromptSRC losses with image and text self-consistency weights 10 and 25 respectively.

PromptSRC is therefore an excellent baseline for this project because it already solves the “do not forget CLIP” side of the problem. The missing side is different:

> PromptSRC preserves the pretrained model prior, but it does not explicitly use the geometry of the wider unlabeled target distribution.

That is the precise gap this project targets.

---

## 2. Research question

### 2.1 Main research question

**Can unlabeled target images improve few-shot PromptSRC by providing local neighborhood geometry, without pseudo-labeling, teacher-student distillation, or changing the CLIP backbone?**

### 2.2 More precise hypothesis

Let \(L\) be the labeled few-shot set and \(U\) be an unlabeled image pool from the same dataset split family. PromptSRC learns prompts from \(L\), regularized by frozen CLIP. We hypothesize that frozen-CLIP neighborhoods over \(U\) provide additional information about the local geometry of the target distribution. Regularizing the prompted classifier to be locally smooth over those neighborhoods should improve generalization, especially when the few-shot labeled set is too sparse to cover the data manifold.

The operative assumption is intentionally weak:

> Frozen-CLIP nearest neighbors are, on average, more semantically related than random pairs.

The method does not assume that frozen CLIP perfectly clusters all classes. It only assumes that local CLIP neighborhoods are meaningful enough to serve as weak relational constraints.

---

## 3. Proposed method

### 3.1 Name

Recommended method name:

**PromptSRC-NC: Neighborhood-Consistent PromptSRC**

Shorter internal name:

**PromptSRC-NC**

Full descriptive name:

**PromptSRC-NC: Neighborhood-Consistent PromptSRC for Unlabeled Geometry-Aware Few-Shot CLIP Adaptation**

Avoid naming the method “graph PromptSRC,” “manifold PromptSRC,” “distribution-aware prompt tuning,” “unlabeled PromptSRC,” or “semi-supervised PromptSRC.” The implementation uses local frozen-CLIP nearest-neighbor pairs, but the conceptual method is simpler and more modest: neighboring unlabeled images should have similar prompted predictions. The name should not imply full manifold recovery, class discovery, or pseudo-labeling.

Short method-section wording:

> We propose PromptSRC-NC, a neighborhood-consistency extension of PromptSRC. After training PromptSRC on the few-shot labeled set, we construct frozen-CLIP nearest-neighbor pairs over unlabeled target images and continue prompt optimization with a symmetric prediction-consistency loss over those pairs.

### 3.2 High-level method

The method has three stages.

#### Stage 0: Frozen-CLIP neighbor construction

Before PromptSRC training, embed unlabeled images using the **unprompted frozen CLIP image encoder**:

\[
z_i^0 = E_v^0(u_i)
\]

Normalize features:

\[
\bar{z}_i^0 = \frac{z_i^0}{\|z_i^0\|_2}
\]

Construct a fixed set of neighbor pairs:

\[
\mathcal{P}_{real} = \{(u_i, u_j)\}
\]

using mutual nearest neighbors or mutual top-\(k\) neighbors. The simplest default is mutual top-1; if too few pairs are obtained, use mutual top-5. The neighbor structure is fixed and is not refreshed after PromptSRC.

This design avoids circularity. The geometry used for adaptation is CLIP’s pretrained geometry over the unlabeled target images, not geometry already shaped by the few-shot labels.

#### Stage 1: Official PromptSRC training

Train PromptSRC exactly as in the official paper/repository on the labeled few-shot data.

The Stage 1 objective is:

\[
L_{\text{PromptSRC}}
=
L_{\text{CE}}
+
L_{\text{SCL}}
\]

where

\[
L_{\text{SCL}}
=
\lambda_{image} L_{\text{SCL-image}}
+
\lambda_{text} L_{\text{SCL-text}}
+
L_{\text{SCL-logits}}
\]

and GPA is applied over the training trajectory. At inference and checkpointing, the final GPA-aggregated prompt weights should be used.

This checkpoint is the main baseline.

#### Stage 2: Unlabeled geometry adaptation

Initialize from the completed PromptSRC checkpoint. Continue optimizing the same prompt parameters using:

\[
L_{\text{Stage2}}
=
L_{\text{PromptSRC}}
+
\lambda_{\text{NN}}L_{\text{NN}}
\]

where

\[
L_{\text{NN}}
=
\frac{1}{|\mathcal{P}_B|}
\sum_{(i,j)\in\mathcal{P}_B}
JS(p_\theta(u_i),p_\theta(u_j)).
\]

Here:

- \(\mathcal{P}_B\) is a minibatch of unlabeled neighbor pairs.
- \(p_\theta(u)\) is the prompted classifier’s softmax distribution over the dataset classes.
- \(JS(\cdot,\cdot)\) is Jensen-Shannon divergence.
- No pseudo-labels are created.
- The same labeled few-shot batch used by PromptSRC remains present in Stage 2 to anchor the semantics.

The Stage 2 interpretation is:

> PromptSRC has learned a few-shot, CLIP-preserving prompt solution. The second stage asks the solution to become locally smooth over the wider unlabeled training distribution.

---

## 4. Why this is not pseudo-labeling

This method should not be described as pseudo-labeling.

Pseudo-labeling assigns a class target to an unlabeled image:

\[
\hat{y}_i = \arg\max_c p_\theta(c|u_i)
\]

and then trains the model as if \(\hat{y}_i\) were a label.

PromptSRC-NC does not do this. It creates no hard or soft class target for any unlabeled image. It only imposes a relational constraint:

\[
u_i \sim u_j
\Rightarrow
p_\theta(u_i) \approx p_\theta(u_j).
\]

The unlabeled data contributes pairwise geometry, not semantic class supervision. A useful slogan:

> Pseudo-labeling invents labels for nodes. This method uses frozen CLIP to provide edges between nodes.

This distinction is important because pseudo-labeling has different failure modes: confidence miscalibration, class imbalance, confirmation bias, and incorrect early labels. The proposed method makes a weaker and cleaner assumption.

---

## 5. Why this is scientifically justifiable

### 5.1 Relation to manifold regularization

Classical manifold regularization argues that unlabeled data can help supervised learning by revealing the geometry of the marginal distribution. The standard graph-Laplacian intuition is that a learned function should vary smoothly along high-density regions of the data manifold.

The proposed method is a lightweight CLIP-prompt analogue of this principle. It does not attempt to estimate a full manifold. It constructs reliable local neighbor pairs from frozen CLIP and asks the prompted classifier to be smooth across those local relations.

The claim is deliberately local:

> We do not recover the true data manifold. We use frozen CLIP nearest-neighbor pairs as a finite-sample approximation of local target-space geometry.

### 5.2 Relation to PromptSRC

PromptSRC is model-intrinsic regularization. It tells the prompt learner not to drift too far from frozen CLIP’s representations and logits. This is essential, but it does not explicitly use where the unlabeled downstream images are located.

The proposed second stage adds data-distributional information:

- PromptSRC: preserve CLIP’s prior while learning from labels.
- Our method: preserve PromptSRC’s learned task semantics while enforcing smoothness over unlabeled target neighborhoods.

This is a complementary regularizer, not a replacement for PromptSRC.

### 5.3 Relation to PromptKD

PromptKD uses unlabeled domain images through teacher-student prompt distillation. It pretrains a larger CLIP teacher using few-shot labels and distills into a lighter target model using unlabeled images.

This project intentionally avoids that route:

- no larger teacher;
- no smaller student;
- no capacity asymmetry;
- no teacher logits;
- same CLIP backbone;
- unlabeled data used as geometry, not as a teacher-distillation medium.

This makes the contribution cleaner for a capstone:

> Can unlabeled geometry itself help PromptSRC, without entangling the effect with model compression or teacher quality?

### 5.4 Relation to TPT and consistency methods

Test-Time Prompt Tuning (TPT) optimizes prompts at test time by minimizing prediction entropy under multiple augmented views of a single sample. That demonstrates that unlabeled/test-time information can adapt prompts, but TPT operates per sample and uses entropy/confidence mechanisms.

The proposed method instead operates over the dataset-level unlabeled neighborhood structure. It does not minimize entropy and does not push predictions to be confident by itself. It only asks neighboring unlabeled images to have similar class distributions.

### 5.5 Relation to graph-based semi-supervised learning

The proposed method is graph-like but intentionally minimal. A full graph-Laplacian method would involve choices about normalization, weights, graph connectivity, refresh schedules, thresholds, and global spectral properties. This project avoids most of those knobs by using only a fixed edge set and a symmetric pairwise consistency loss.

This makes the method easier to implement, easier to debug, and easier to interpret.

---

## 6. Datasets

### 6.1 Flowers102

Flowers102 contains 102 flower categories. Classes have substantial variation in pose, scale, lighting, and within-class appearance, and some categories are visually similar. This makes it a useful fine-grained testbed for CLIP-neighborhood consistency: if CLIP’s local neighborhoods respect species-level similarity, the method may help; if CLIP confuses similar flower categories, the method may over-smooth.

PromptSRC’s official repository uses the dataset name `oxford_flowers` and the registered dataset class `OxfordFlowers`.

### 6.2 EuroSAT

EuroSAT is a land-use/land-cover dataset based on Sentinel-2 satellite imagery. The RGB version contains 27,000 labeled images across 10 classes. PromptSRC maps raw folder names into more natural class names such as “Annual Crop Land,” “Highway or Road,” “Industrial Buildings,” and “Sea or Lake.”

EuroSAT is interesting because it differs strongly from natural-object datasets. CLIP’s pretrained web-scale representation may be less natively aligned with remote-sensing semantics, so unlabeled local geometry may be especially informative—or unreliable. This dataset is a good stress test.

PromptSRC’s official repository uses the dataset name `eurosat` and the registered dataset class `EuroSAT`.

### 6.3 Stanford Cars

Stanford Cars contains 16,185 images of 196 car classes, typically defined by make, model, and year. The official split has 8,144 training images and 8,041 test images. This is a fine-grained recognition problem in which visually similar cars may belong to different classes.

Stanford Cars is a critical risk test for the proposed method. Nearest-neighbor consistency may connect visually similar but label-distinct car models. The shuffled-neighbor control and edge-agreement diagnostics are therefore particularly important.

PromptSRC’s official repository uses the dataset name `stanford_cars` and the registered dataset class `StanfordCars`.

---

## 7. Evaluation framework

The project should follow PromptSRC’s evaluation conventions as closely as possible. There are two relevant PromptSRC settings.

### 7.1 Primary recommended setting: few-shot all-class learning

This is the cleanest setting for the research goal.

For each dataset and each shot count \(K\):

1. Use PromptSRC’s official few-shot split generation.
2. Train on \(K\) labeled examples per class.
3. Use the remaining training-split images as unlabeled data for Stage 0/2.
4. Evaluate on the official test split.
5. Report top-1 accuracy.

This setting directly matches the phrase “few-shot learning aware of the wider unlabeled space.” It avoids using test images. It also avoids the ambiguity of whether unlabeled novel-class images are allowed in base-to-novel evaluation.

Recommended shot counts:

- **Minimal capstone**: \(K=16\)
- **Better if compute allows**: \(K \in \{1, 4, 16\}\)
- **Full PromptSRC few-shot setting**: \(K \in \{1,2,4,8,16\}\)

Given compute constraints, use \(K=16\) first. If the method works, run \(K=1\) and \(K=4\) on one or more datasets to test whether the gain is larger in lower-label regimes.

### 7.2 Secondary setting: base-to-novel generalization

PromptSRC’s flagship benchmark is base-to-novel class generalization:

1. Split classes into base and novel halves.
2. Train on 16-shot labeled examples from base classes only.
3. Evaluate on base and novel classes separately.
4. Report base accuracy, novel accuracy, and harmonic mean.

This is valuable because it measures whether prompt learning overfits base classes and harms novel class generalization.

For this project, base-to-novel must be handled carefully. There are two possible unlabeled-pool policies:

#### Strict policy

Use only extra unlabeled images from base classes during Stage 2. This preserves the original base-to-novel restriction: no novel-class examples are seen during training, even unlabeled.

#### Transductive policy

Use unlabeled images from both base and novel classes, but no labels. This more directly tests whether wider target-space geometry helps novel classes, but it is no longer the original PromptSRC base-to-novel protocol. It must be reported as a semi-supervised/transductive extension.

Recommendation:

- Use **few-shot all-class** as the primary setting.
- Use **strict base-to-novel** only if compute permits.
- Do not mix strict and transductive base-to-novel results without clearly labeling them.

---

## 8. Cloud execution and cost discipline

The implementation should stand alone inside `promptSRC-NC/`. It should include its own data preprocessing, PyTorch/OpenCLIP model code, Modal app, training code, evaluation code, diagnostics, and aggregation. The official PromptSRC repository is the reference for method behavior, not the cloud runtime environment.

Local development should use `uv` only. Full training should run on Modal. Do not use conda in the project workflow.

Modal functions should be separate and inspectable:

1. `prepare_data`: download or verify datasets, write manifests and splits to Modal Volumes.
2. `smoke_test`: verify imports, data loading, one Stage 1 step, tiny neighbor construction, one Stage 2 real/shuffled step, and one evaluation batch.
3. `profile_gpu_cost`: measure T4 versus L4 seconds per step, GPU memory, and estimated cost before the full matrix.
4. `build_neighbors`: compute frozen-CLIP features and real/shuffled pairs.
5. `train_stage1`: train PromptSRC baseline.
6. `train_stage2`: train PromptSRC-NC real or shuffled variant.
7. `evaluate`: run val/test evaluation and diagnostics.
8. `aggregate_results`: produce JSON/CSV summaries for plots and tables.

Smoke tests and cost profiling serve different purposes. Smoke tests should run first and answer whether the code path works. The profiling function should run after smoke tests and answer which GPU is cheapest per completed experiment.

With Modal pricing checked for this plan, T4 is the cheapest hourly GPU while L4 is 1.35x more expensive per hour. L4 is cheaper per completed run only if it is more than 1.35x faster end-to-end. Therefore the first cloud sequence should be:

1. run `prepare_data`;
2. run `smoke_test` on T4;
3. run `profile_gpu_cost` on T4 and L4 for Stanford Cars 16-shot seed 1;
4. choose the lower measured cost-per-step GPU for the main matrix.

Do not reduce the official labeled batch size from 4 to 2 merely to fit a cheaper GPU, because that doubles optimizer steps per epoch and changes the training schedule. If memory is tight, reduce Stage 2 pair batch size from 8 to 4 first. If the full `ViT-B/16` matrix is too expensive, run a clearly labeled `ViT-B/32` budget study while preserving the same PromptSRC, PromptSRC-NC, and shuffled-control comparisons.

Every cloud run should write structured logs for runtime, losses, evaluation, diagnostics, GPU memory, seconds per step, and estimated cost. These logs must be rich enough to reconstruct accuracy tables, uplift graphs, runtime plots, and cost comparisons without scraping console output.

---

## 9. Experimental variants

Use exactly three variants.

### Variant A: PromptSRC

Standalone PyTorch/OpenCLIP PromptSRC implementation, matched as closely as possible to the official paper and repository.

This answers:

> What does the few-shot, CLIP-preserving prompt learner achieve without unlabeled geometry?

### Variant B: PromptSRC-NC

Stage 1 is identical to PromptSRC. Stage 2 uses real frozen-CLIP neighbor pairs.

This answers:

> Does local geometry from unlabeled target data improve a completed PromptSRC prompt solution?

### Variant C: PromptSRC-NC shuffled-neighbor control

Stage 1 is identical to PromptSRC. Stage 2 uses the same number of unlabeled pairs, the same loss, and the same training schedule, but the neighbor topology is destroyed by degree-preserving shuffling.

This answers:

> Is the improvement due to meaningful frozen-CLIP geometry, or merely due to extra training/unlabeled smoothing?

The most important comparison is not only B vs A. It is B vs C.

---

## 10. Expected outcomes

### 10.1 Best-case outcome

If results look like:

\[
\text{PromptSRC-NC}
>
\text{PromptSRC-NC shuffled}
\approx
\text{PromptSRC},
\]

then the interpretation is strong:

> Meaningful frozen-CLIP unlabeled geometry helps refine few-shot PromptSRC beyond generic extra training.

### 10.2 Ambiguous outcome

If results look like:

\[
\text{PromptSRC-NC}
\approx
\text{PromptSRC-NC shuffled}
>
\text{PromptSRC},
\]

then the geometry claim is weak. The result may indicate that extra regularization or extra training helps, but not specifically real neighborhood structure.

### 10.3 Negative outcome

If real-neighbor adaptation hurts, possible explanations include:

- CLIP nearest neighbors are unreliable for the dataset.
- The unlabeled pool contains class-ambiguous or out-of-distribution samples.
- Fine-grained classes are connected incorrectly.
- The neighbor-consistency weight is too high.
- The Stage 2 learning rate or duration is too aggressive.
- PromptSRC already saturates performance on the dataset.

A negative outcome is still scientifically interpretable, especially on Stanford Cars, where fine-grained label distinctions may conflict with visual-neighborhood smoothness.

---

## 11. Diagnostics

Even with only three trained variants, add low-cost diagnostics.

### 11.1 Edge agreement

For real neighbor pairs:

\[
\text{EdgeDisagree}
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
\mathbb{1}[\arg\max p_\theta(u_i)\neq \arg\max p_\theta(u_j)].
\]

Report this for PromptSRC before Stage 2 and after real-neighbor adaptation. If accuracy improves and edge disagreement decreases, the mechanism is plausible.

### 11.2 Mean JS on real edges

Report:

\[
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
JS(p_\theta(u_i),p_\theta(u_j)).
\]

This directly measures whether the Stage 2 objective is being optimized.

### 11.3 Prediction entropy

Report mean entropy on the unlabeled pool:

\[
\frac{1}{|U|}
\sum_{u\in U}
H(p_\theta(u)).
\]

The method should not simply collapse into overconfident predictions. If entropy collapses and accuracy does not improve, Stage 2 is too strong.

### 11.4 Neighbor-pair sanity check

Before training, compute and log:

- number of unlabeled images;
- number of neighbor pairs;
- mean cosine similarity of real pairs;
- mean cosine similarity of shuffled pairs;
- distribution of node degrees;
- percentage of mutual nearest-neighbor pairs retained.

These are cheap and make the paper/report more credible.

### 11.5 Runtime and cost diagnostics

Also report:

- GPU type;
- backbone;
- labeled batch size;
- pair batch size;
- seconds per Stage 1 step;
- seconds per Stage 2 real step;
- seconds per Stage 2 shuffled step;
- max GPU memory allocated;
- estimated GPU cost per run;
- estimated GPU cost for the full matrix.

This is necessary because the project has a fixed budget and because the T4-vs-L4 choice should be based on measured cost per completed experiment, not only hourly price.

---

## 12. Risks and mitigations

### 12.1 Fine-grained wrong edges

Stanford Cars and Flowers102 may have nearest-neighbor pairs that are visually close but label-distinct. This can make prediction smoothing harmful.

Mitigation:

- Use mutual nearest neighbors.
- Keep the loss symmetric and weak.
- Keep Stage 2 short.
- Use the shuffled control to diagnose whether meaningful edges help or hurt.
- Inspect a small sample of nearest-neighbor image pairs qualitatively.

### 12.2 Over-smoothing

The neighbor loss can reduce useful class boundaries if applied too strongly.

Mitigation:

- Keep \(L_{\text{PromptSRC}}\) active during Stage 2.
- Use a short Stage 2.
- Use a conservative \(\lambda_{\text{NN}}\).
- Do not use entropy minimization or pseudo-labeling in the core method.

### 12.3 Confounded comparison from extra training

Stage 2 adds compute after PromptSRC. The shuffled-neighbor control is designed specifically to handle this.

The conclusion should be based on:

\[
\text{real-neighbor Stage 2} - \text{shuffled-neighbor Stage 2}.
\]

### 12.4 Unlabeled pool leakage

The unlabeled pool must not include test images in the primary setting. Use only the official training split, excluding the labeled few-shot examples.

If any transductive experiment uses test-domain images, it must be separately labeled as transductive and not compared as if it were the same setting as PromptSRC.

### 12.5 Cloud cost drift

GPU availability, pricing, and runtime can change. Record the Modal GPU type, listed price used for estimates, measured seconds per step, and estimated cost in every profile and training run. If pricing changes, recompute cost from raw seconds and the current price rather than rerunning experiments.

---

## 13. Contribution statement

The final report should make a modest, defensible claim:

> We propose a lightweight, teacher-free unlabeled adaptation stage for PromptSRC. Instead of assigning pseudo-labels, we use frozen CLIP to construct nearest-neighbor pairs over unlabeled training images and regularize the prompted classifier to be locally consistent over those pairs. This tests whether unlabeled target-space geometry can refine a few-shot prompt solution beyond PromptSRC’s model-intrinsic self-regularization.

The contribution is not “new state of the art.” The contribution is a clean experiment around a precise question:

> Does the local geometry of unlabeled target images contain useful information for few-shot CLIP prompt learning?

---

## 14. Primary references

1. **PromptSRC** — Muhammad Uzair Khattak et al., “Self-regulating Prompts: Foundational Model Adaptation without Forgetting,” ICCV 2023. Official repo: `https://github.com/muzairkhattak/PromptSRC`
2. **CLIP** — Alec Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021.
3. **CoOp** — Kaiyang Zhou et al., “Learning to Prompt for Vision-Language Models,” IJCV 2022.
4. **CoCoOp** — Kaiyang Zhou et al., “Conditional Prompt Learning for Vision-Language Models,” CVPR 2022.
5. **MaPLe** — Muhammad Uzair Khattak et al., “MaPLe: Multi-Modal Prompt Learning,” CVPR 2023.
6. **PromptKD** — Zheng Li et al., “PromptKD: Unsupervised Prompt Distillation for Vision-Language Models,” CVPR 2024.
7. **TPT** — Manli Shu et al., “Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models,” NeurIPS 2022.
8. **Manifold regularization** — Mikhail Belkin, Partha Niyogi, Vikas Sindhwani, “Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples,” JMLR 2006.
9. **Flowers102** — Maria-Elena Nilsback and Andrew Zisserman, “Automated Flower Classification over a Large Number of Classes,” ICVGIP 2008.
10. **EuroSAT** — Patrick Helber et al., “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,” J-STARS 2019 / arXiv 2017.
11. **Stanford Cars** — Jonathan Krause et al., “3D Object Representations for Fine-Grained Categorization,” ICCV Workshops 2013.
12. **Modal** — Apps, Functions, GPU acceleration, Images, Volumes, Secrets, and GPU Metrics documentation: `https://modal.com/docs/`
13. **OpenCLIP** — ML Foundations OpenCLIP implementation: `https://github.com/mlfoundations/open_clip`
