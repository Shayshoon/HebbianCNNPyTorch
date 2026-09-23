# Q&A preparation for the final defense

This document is designed for the approximately 10-minute question period after the presentation. The short answers are meant to be spoken in roughly 15-40 seconds. The backup material is for follow-up questions, not for the first response.

## Evidence hierarchy for the defense

Use the frozen final report, frozen CSVs, corrected analytical scripts, and the active notebook branches as the quantitative authority. The two papers establish the published architecture, algorithms, training settings, and accuracy context. `mathematical_architectural_audit.md` was an important early derivation, but a few of its statements were superseded by the later audits and must not be quoted as final results; Question 30 explains the differences.

When an answer depends on an assumption, say so immediately. A strong answer is: "That part is an analytical assumption, and here is why we chose it," rather than presenting it as a fact from a paper.

## Numbers we should know without looking

### Architecture and training

| Item | Value |
|---|---:|
| Batch size | 100 |
| Convolutional channels | 3 -> 100 -> 196 -> 400 |
| Kernels | 5x5, 3x3, 3x3; valid, stride 1 |
| Post-convolution shapes | 100x28x28, 196x12x12, 400x4x4 per image |
| Post-pooling shapes | 100x14x14, 196x6x6, 400x2x2 per image |
| Final feature dimension | 1,600 |
| Dense convolutional weights | 889,500 |
| BP parameters including classifier | 905,510 |
| Seeded unmasked weight coordinates by layer | 7,500; 1,771; 7,142 |
| Deterministic active-weight total | 16,413 |
| Hebbian training | 20 epochs, 500 batches/epoch, 10,000 batches |
| Burn-in | 201 batches; 9,799 weight-update batches |
| Hebbian learning rates | 0.0001, 0.0001, 0.001 |
| Threshold adaptation rate | 3.0 |
| Structural masking | Layer 1 dense; Layers 2 and 3 approximately 99% masked |

The deterministic test batch had 78,400, 14,400, and 1,600 positive WTA entries in Layers 1-3: one at every spatial column in that particular batch. That is not guaranteed by the generic WTA code.

### Matched training-core compute

| Method | One batch | Equal 20 epochs |
|---|---:|---:|
| A1 direct dense Hebbian | 8.667 billion | 86.666 trillion |
| A2 conditional sparse-aware | 1.380 billion | 13.800 trillion |
| B surrogate/autograd Hebbian | 17.268 billion | 172.677 trillion |
| Standard BP | 24.430 billion | 244.305 trillion |

One-batch ratios to memorize:

- BP/A1 = 2.819x.
- BP/A2 conditional = 17.699x.
- BP/B = 1.415x.
- B/A1 is about 1.99x.

Configured training-core budgets are Hebbian at 20 epochs versus BP at 100 epochs. BP at 100 epochs is 1,221.524 trillion operations. The five-epoch Hebbian sensitivity values are A1 21.661 trillion, A2 conditional 3.447 trillion, and B 43.169 trillion. These are not accuracy-matched comparisons.

### Full method-level compute

| Method and budget | Total modeled scalar operations |
|---|---:|
| A1, 20 epochs | 112.590 trillion |
| A2 conditional, 20 epochs | 35.365 trillion |
| B, 20 epochs | 198.609 trillion |
| BP, 20 epochs | 245.158 trillion |
| BP, 100 epochs | 1,222.377 trillion |

The method-level BP value differs slightly from the training-core value because it includes the test prediction/evaluation pass. The Hebbian totals also include known ZCA construction and application arithmetic, training, feature extraction, Ridge fitting, and evaluation. The ZCA eigensolver is a separate sensitivity estimate.

### Memory and movement

| Method | Persistent state | Logical movement per compared update batch |
|---|---:|---:|
| A1 | 7.089 MB | 239.950 MB |
| A2 conditional, CSR32 | 0.106 MB | 139.038 MB |
| B | 10.677 MB | 377.999 MB |
| BP with Adam | 14.488 MB | 366.434 MB |

Peak references worth recognizing, but not describing as measured peaks:

- A1 optimized layer-local core: 49.743 MB; with persistent ZCA: 87.492 MB.
- A2 conditional optimized layer-local core: 42.760 MB; with ZCA: 80.509 MB.
- B optimized layer-local reference: 106.299 MB; notebook named tensors plus ZCA: 278.557 MB.
- BP: 72.233 MB retained-forward baseline, 103.593 MB one-gradient-buffer reference, and 134.953 MB conservative two-buffer envelope.
- One-time ZCA construction references: 221.256 MB, 296.778 MB, and 372.276 MB; the retained ZCA matrix is 37.749 MB.

### Accuracy and uncertainty

- Original Hebbian paper: 64.55% +/- 0.4 over 10 runs in Table 1. The nearby prose says +/- 0.5, so use the table value and acknowledge the discrepancy if asked.
- Saved notebook: 64.65% for one saved run.
- Approximate comparison-paper graph reads, all over 10 runs: Hebbian about 66% at 20 and 100 epochs; BP about 57% at 20 epochs and 70% at 100 epochs.
- The comparison paper says Hebbian stabilizes in roughly 5 epochs while BP takes around 100 epochs. This describes convergence behavior, not equal final accuracy.

## Likely questions and defensible answers

### 1. What question did your project actually answer?

**Short answer:** We asked whether one specific three-layer Hebbian CNN has lower modeled computational and memory requirements than a clearly defined BP reference. We answered that for several implementation models and scopes. We did not test whether Hebbian learning is universally better, faster, or more energy efficient.

**Backup:** The strongest result is the matched training-core comparison: same batch size and equalized architecture, with preprocessing excluded from both sides. We also reported source-appropriate method-level workflows, but those are broader pipeline comparisons rather than identical algorithms.

### 2. In simple terms, what does Grossberg Instar do?

**Short answer:** When a filter wins for an input patch, Instar moves that filter toward the patch. Mathematically, the update is proportional to `winner * (input patch - current weight)`. Competition chooses which filters learn, and normalization prevents their magnitudes from growing without control.

**Backup:** For filter `c`, the batched update is `eta * [sum of patches won by c - number of wins times w_c]`. The rule is local in the algorithmic sense because it needs the receptive-field input, the local winner signal, and that filter's weights, not a label error propagated from the output layer.

### 3. Is the complete Hebbian pipeline really unsupervised?

**Short answer:** The convolutional feature learning is unsupervised because it does not use CIFAR-10 labels. The final evaluation is supervised: Ridge is fitted from the frozen convolutional features to one-hot class targets. So the accurate description is unsupervised representation learning followed by a supervised linear readout.

**Backup:** Labels enter only after the 20 Hebbian learning epochs, during Ridge fitting. This is one reason the Hebbian and BP accuracy protocols are not identical: BP trains its classifier and feature extractor jointly from label-derived gradients.

### 4. If the notebook calls `loss.backward()`, why do you call the rule Hebbian rather than backpropagation?

**Short answer:** `backward()` is being used as an implementation tool, not as a supervised global credit signal. The surrogate expression is designed so that its gradient equals the local Instar update after its forward value is replaced by the WTA activity. There is no end-to-end classification loss during Hebbian feature learning.

**Backup:** The symbolic term is `w^T x - 0.5||w||^2`, whose derivative is `x-w`. The surrogate loss contributes `-y(x-w)` to the gradient, so SGD adds `eta*y(x-w)`. Model B pays for reverse-mode autograd even though the mathematical learning signal is local.

### 5. Does top-1 WTA always produce exactly one winner?

**Short answer:** No. The code keeps every response at the top cutoff and then activates only strictly positive values. A non-positive maximum gives zero winners, a unique positive maximum gives one, and a positive tie at the cutoff can give multiple winners.

**Backup:** The deterministic equivalence batch happened to produce one positive winner at every spatial column, which is why its counts equal the numbers of columns. We use that only as a representative winner assumption when scaling winner-dependent costs; we do not claim it is a universal invariant.

### 6. Why are WTA and triangle activation both needed?

**Short answer:** They serve different purposes. Sparse binary WTA decides which filters receive Hebbian plasticity, while the denser triangle activation carries information to the next layer and to the final readout. The original paper found that separating sparse plasticity from denser representation improved higher-layer accuracy.

**Backup:** Triangle activation subtracts the channel mean from each thresholded response and rectifies negatives, then applies 2x2 average pooling. It does not replace WTA in the update rule.

### 7. What exactly are A1, A2 conditional, B, and BP?

**Short answer:** A1 is the verified direct Instar update with dense forward convolutions. A2 conditional uses the same local mathematics but assumes true sparse storage and kernels for the masked higher layers. B is the notebook's dense surrogate-loss/autograd implementation. The BP model is our same convolutional architecture/geometry BP reference, with a learned linear classifier and Adam.

**Backup:** A1 is the strongest optimized model directly supported by the equivalence test. A2 is an opportunity model, not an implemented result. B distinguishes the cost of the current framework mechanism from the cost of the underlying local rule.

### 8. Is the comparison fair if Hebbian and BP have different learning and evaluation pipelines?

**Short answer:** No single scope makes them identical, so we report separate scopes. The matched training-core table equalizes architecture, batch size, and work while excluding preprocessing from both sides. The method-level table instead follows each method's actual pipeline and is explicitly not presented as a controlled accuracy comparison.

**Backup:** We retain one-batch, equal-20-epoch, configured 20-versus-100-epoch, and method-level views. This prevents us from hiding shared or method-specific costs while also avoiding invented BP preprocessing. The tradeoff is that method-level totals answer a workflow-cost question, not a strict algorithm-isolation question.

### 9. Which BP details came from the comparison paper, and which did you assume?

**Short answer:** The paper directly gives CIFAR-10, channels 100/196/400, a final linear layer, Adam, batch size 100, initial learning rate `1e-5`, StepLR with step size 1, 20 or 100 epochs, and a vanilla network without batch normalization or dropout. We assumed the Hebbian kernel and pooling geometry, ReLU, no convolution bias, classifier bias, mean cross-entropy, PyTorch Adam defaults, and StepLR gamma 0.95.

**Backup:** Adam is explicitly stated in Appendix A of arXiv:2212.04614; it is not inferred from an older repository. The paper does not specify the omitted details, so `backprop_reference.py` labels them as assumptions. StepLR gamma changes the learning-rate trajectory but not our per-parameter Adam arithmetic count.

### 10. Why reuse the Hebbian kernels and pooling for BP?

**Short answer:** The comparison paper fixes the channel counts but not those geometric details. Reusing the active Hebbian 5x5/3x3/3x3 convolutions and 2x2 pooling gives both methods the same tensor shapes and avoids inventing a larger or smaller BP network. It is a fairness assumption, not a paper-derived fact.

**Backup:** The resulting final tensor is 400x2x2, and the BP classifier has 16,000 weights plus 10 biases. The total BP parameter count is 905,510, compared with 889,500 convolutional coordinates in the Hebbian stack.

### 11. What does the direct equivalence test prove mathematically and numerically?

**Short answer:** It proves that, for one deterministic post-burn-in batch and the active three-layer branches, the explicit Instar update and the surrogate/autograd update produce matching weights and thresholds within `atol=1e-6` and `rtol=1e-5`. All three layers matched, with zero reported maximum and mean differences in that run.

**Backup:** Both methods share the same initial FP32 weights, masks, thresholds, input, learning rates, WTA result, masking, filter normalization, pooling path, and threshold update. The algebra establishes the update identity; the test checks that our direct implementation realizes it correctly for the exercised branches.

### 12. Why is one synthetic batch enough, and what does it not prove?

**Short answer:** It is enough to validate the implementation of the local update on all three active layer shapes, but it is not enough to validate full training. The input is a deterministic random tensor with the shape of a ZCA-processed batch, not an actual CIFAR-10 batch. It does not prove identical stochastic trajectories, final accuracy, or A2 sparse-kernel behavior.

**Backup:** A full validation would run many real batches with identical data order and randomness and compare states throughout training. That was outside this analytical project. We therefore combine a one-batch numerical check with the algebraic derivation and keep the claim deliberately limited.

### 13. Is the `.data` overwrite in the notebook safe or recommended?

**Short answer:** It is an intentional mechanism used by the original implementation, but `.data` mutation is generally unsafe style in modern PyTorch because it can bypass autograd safeguards. Our claim is narrower: for this specific graph and tested environment, it produced the expected Instar gradient, which the algebra and numerical test both support.

**Backup:** The loss is constructed after the surrogate tensor's value is replaced, so its local chain-rule multiplier is the WTA value while the derivative path remains `x-w`. A production implementation should avoid relying on fragile in-place autograd behavior and use an explicit local kernel or a carefully defined custom operation.

### 14. What do you mean by an "operation," and is one operation one FLOP?

**Short answer:** We count scalar multiplications, additions/subtractions, comparisons, divisions, square roots, and BP transcendental evaluations separately. Basic FLOPs are only multiplications plus additions; one multiply-accumulate therefore contributes two basic FLOPs in our convention. The displayed total scalar operations adds all categories with unit weight as a bookkeeping total, not as a latency or energy equivalence.

**Backup:** A square root and a comparison clearly do not have equal hardware cost. Keeping the categories separate lets a future accelerator study apply hardware-specific weights or throughput models without changing the underlying formulas.

### 15. How did you handle burn-in, masking, and weight normalization?

**Short answer:** The counter starts at -1 and weights update only when it is greater than 200, so there are 201 burn-in batches and 9,799 update batches in 20 epochs. Forward, WTA, triangle, pooling, autograd loss/backward, thresholds, masking for Layers 2-3, and normalization still execute during burn-in. Layer 1 is not charged a structural-mask multiplication because its `PROBAMASKS` value is zero and that branch is skipped.

**Backup:** The notebook still allocates an all-one Layer 1 mask, which B's storage model includes, but it does not multiply Layer 1 weights by that mask in the training branch. A1's optimized storage omits that redundant all-one mask. These distinctions are why compute and storage mask accounting are not identical.

### 16. Are the 20-epoch Hebbian totals exact?

**Short answer:** Architecture-fixed dense forward and other shape-dependent terms are exact under our scalar formulas, but winner-dependent A1 and A2 update terms are representative estimates. We scaled the deterministic batch's winner counts across training. A2 also assumes its observed pairing between winners and active mask coordinates remains representative.

**Backup:** The structural mask counts are exact for the seeded reference masks, not for every possible random initialization. The generic WTA code can produce zero or tied winners. Therefore the totals are analytical estimates under explicit reference assumptions, not a trace of all 10,000 real batches.

### 17. Why is A2 labeled conditional rather than simply "sparse Hebbian"?

**Short answer:** Because zeros in a dense tensor do not make dense `conv2d` cheaper. A2's savings require sparse storage, sparse convolution, sparse update accumulation, and sparse normalization that skip masked coordinates. None of those end-to-end kernels was implemented or benchmarked in this project.

**Backup:** The deterministic masks leave 1,771 Layer 2 and 7,142 Layer 3 coordinates active, in addition to 7,500 dense Layer 1 coordinates. CSR32 storage includes values, column indices, and row pointers. Irregular indexing, load imbalance, and poor utilization could reduce or eliminate the theoretical runtime advantage even if the arithmetic count is lower.

### 18. Why is B almost twice A1, yet still cheaper in operations than BP?

**Short answer:** B recovers the local Instar rule through a dense surrogate graph, a dense convolution weight-gradient calculation, and gradient storage, which A1 avoids. BP additionally propagates activation gradients through the whole network, trains the classifier, computes cross-entropy, and performs Adam updates. That places B between direct A1 and full BP in arithmetic.

**Backup:** Data movement tells a different story: B is modeled at 377.999 MB per update batch versus BP at 366.434 MB. This is why we keep arithmetic and movement separate; fewer scalar operations do not guarantee less traffic or faster execution.

### 19. What is included in persistent memory?

**Short answer:** Persistent state is what survives across batches: parameters, structural masks or sparse metadata, parameter gradients, optimizer moments, and Hebbian thresholds. A1 is 7.089 MB, A2 conditional CSR32 is 0.106 MB, B is 10.677 MB, and BP with Adam is 14.488 MB, using FP32 values and decimal megabytes.

**Backup:** BP has 3.622 MB of parameters, the same amount of gradients, and 7.244 MB of two Adam moment tensors. B has no momentum state because its SGD uses zero momentum, but it keeps dense gradients. A2's result includes CSR metadata and thresholds rather than pretending nonzero values alone are sufficient.

### 20. Are your peak-memory values predictions of actual GPU memory usage?

**Short answer:** No. They are analytical live-tensor payload references. Actual PyTorch or cuDNN peaks also depend on saved tensors, convolution workspaces, allocator caching, fusion, Python objects, and release timing, so we explicitly avoid calling them measured GPU peaks.

**Backup:** We give alternative BP references rather than adding them: 72.233 MB retained-forward baseline, 103.593 MB with one recycled gradient buffer, and 134.953 MB with two buffers. A1 also shows why implementation matters: an optimized layer-local reference is 49.743 MB, while explicitly materializing unfolded patches raises it to 143.708 MB.

### 21. What does your logical data-movement model mean?

**Short answer:** It counts idealized whole-tensor reads and writes under explicit reuse assumptions. It is useful for showing which algorithms require weight, gradient, activation, or optimizer-state traffic, but it is not physical DRAM traffic. Caches, tiling, fusion, sparse formats, and transfers can change the real number substantially.

**Backup:** The per-batch values are A1 239.950 MB, A2 conditional 139.038 MB, B 377.999 MB, and BP 366.434 MB. The surprising B-above-BP result is a reminder that autograd's local mathematical rule can still create substantial dense tensor traffic.

### 22. What exactly does the notebook do for ZCA, and is it standard ZCA?

**Short answer:** It takes one shuffled batch of 9,000 images, normalizes each image, forms a 3,072x3,072 matrix as `T^T T / 3072`, regularizes it, eigendecomposes it, and builds an inverse-square-root transform. Because it divides by 3,072 features rather than 9,000 samples, it is a scaled scatter-matrix construction rather than the conventional sample covariance. It also differs from the paper's statement that ZCA uses the entire training set.

**Backup:** Our model follows the active code rather than silently correcting it. The notebook also constructs an experimental spatial-filter object even though the active per-batch path uses the full image matrix; because that construction code executes, its known arithmetic is included in the method-level subtotal.

### 23. How did you count the 3,072x3,072 eigendecomposition?

**Short answer:** We did not invent an exact count for `torch.symeig`, because the driver, reduction strategy, convergence behavior, and backend are not fixed. We report a separate broad sensitivity range from `4/3 F^3` to `10 F^3`: about 38.655 to 289.910 billion basic operations. Adding it changes the Hebbian method totals by only about 0.019% to 0.820%, so it does not change the ordering.

**Backup:** The known surrounding work—normalization, scatter-matrix formation, regularization, eigenvalue postprocessing, reconstruction, and ZCA applications—is included. The range is a reference sensitivity interval, not a formal runtime bound.

### 24. Is the Ridge cost exact if the notebook just calls `Ridge()`?

**Short answer:** No. The repository does not pin the scikit-learn or SciPy version, so default solver dispatch and low-level kernels are implementation dependent. We use a dense primal Cholesky path as a defensible analytical reference because the input is dense and has 50,000 samples by 1,600 features, but we label it as an assumption.

**Backup:** The reference Ridge fit is 259.176 billion modeled operations, plus 320.110 million for prediction and evaluation. SVD has the same broad `O(NF^2 + F^3)` structure but a different constant, while iterative solvers depend on iteration count. This uncertainty is too small relative to the full method totals to reverse the main ordering, but we should not call the solver count exact.

### 25. What actually happens during the two post-training feature passes?

**Short answer:** Convolutional weights stop updating, but the complete forward path still runs: normalization, ZCA application, convolution, threshold subtraction, WTA, triangle activation, pooling, and adaptive-threshold updates. In B, the notebook also constructs `yforgrad`, although it does not build a loss or call backward. Ridge uses the flattened final 400x2x2 output.

**Backup:** The notebook also computes quadrant features for all three layers even though the active classifier selects `trainouts` and `testouts`, not the quadrant lists. We count that source-visible work in the method-level pipeline. This makes the model faithful to the notebook but also exposes an optimization opportunity. Because thresholds keep adapting, the feature extractor is not completely frozen, and train/test representations are order dependent.

### 26. Does "5 Hebbian epochs versus 100 BP epochs" mean equal accuracy?

**Short answer:** Absolutely not. It is a paper-reported approximate convergence-budget scenario: Hebbian accuracy stabilizes early, while BP continues improving. In the comparison paper's full-data graph, Hebbian is about 66% and BP reaches about 70% at 100 epochs, so the favorable compute ratio is not an equal-accuracy result.

**Backup:** We retain equal one-batch and equal-20-epoch comparisons separately. The five-versus-100 view only shows how training budget affects cumulative arithmetic if one accepts the paper's stabilization description.

### 27. Can the accuracy values be compared directly with your resource totals?

**Short answer:** Only as context, not as one controlled benchmark. The original paper, saved notebook, comparison paper, and our BP reference differ in preprocessing, readout, randomness, and some architectural or optimizer details. We therefore show accuracy evidence by source and never combine it with compute into a single efficiency score.

**Backup:** The original paper's exact table value is 64.55% +/- 0.4 over 10 runs; the saved notebook's 64.65% is one run. The approximately 66%, 57%, 66%, and 70% comparison-paper values are visual reads from plots, not printed table values. The notebook also applies a random horizontal flip to the test transform and continues threshold adaptation during feature collection.

### 28. Why did you not measure power or runtime, and what is still relevant to hardware accelerators?

**Short answer:** We had no controlled device, kernel implementation, profiler, or power-sampling setup, so converting operations into joules or claiming speed would be false precision. The hardware relevance is structural: direct local learning can remove global activation-gradient propagation, gradient buffers, and Adam state, and it can update and discard information layer by layer.

**Backup:** Whether those structural savings become lower time or energy depends on dense versus sparse kernel efficiency, memory hierarchy, data reuse, fusion, precision, and utilization. Our operation categories, storage sizes, and logical movement provide a reproducible starting point for a real accelerator mapping, not a substitute for one.

### 29. What experiment should come next to validate the hardware claim?

**Short answer:** Implement A1 and A2 end to end, use one controlled data and evaluation protocol for Hebbian and BP, and repeat training across seeds. Then measure accuracy distributions, wall-clock time, allocator peak, hardware-counter traffic, and energy on specified hardware while reporting kernel and sparse-format details.

**Backup:** A useful study would compare dense A1 against B first, because that isolates autograd overhead without depending on sparse hardware. A2 should then test at least one concrete format such as CSR or block sparsity and report index overhead, utilization, and whether dense-to-sparse conversion is required.

### 30. The early mathematical audit says exactly one WTA winner and an exact same trajectory. Is that consistent with the final project?

**Short answer:** Those statements were corrected by the later independent audits. The active code can produce zero or tied winners, the equivalence test covers one deterministic batch rather than a full trajectory, and the final seeded mask counts are 1,771 and 7,142 in Layers 2 and 3 rather than the audit's expected 1% counts. All frozen final numbers come from the corrected scripts and CSVs, not from those superseded sentences.

**Backup:** The early audit is still useful for architecture and the algebraic Instar derivation, but its claims of "exactly one" winner, guaranteed identical final accuracy, and strictly minimal memory traffic are too strong. The final report explicitly limits equivalence, labels representative assumptions, and treats memory movement as analytical rather than physical. Being candid about this revision demonstrates that the project actually audited its own assumptions.

## Claims we must NOT make

- Do not say we measured electricity, joules, watts, or energy savings.
- Do not say we measured runtime, speedup, GPU utilization, or kernel latency.
- Do not call analytical tensor payloads a measured GPU memory peak.
- Do not call logical tensor movement physical DRAM traffic.
- Do not claim A2 is the current PyTorch execution. It is conditional on genuine sparse storage and kernels.
- Do not say a 99% zero mask automatically accelerates dense `conv2d`.
- Do not say top-1 always creates exactly one active winner.
- Do not call the one-batch equivalence test a proof of full-training, final-weight, or final-accuracy equivalence.
- Do not say the deterministic input was an actual CIFAR-10/ZCA batch.
- Do not call every 20-epoch Hebbian total exact; winner-dependent terms use a representative deterministic batch.
- Do not say five Hebbian epochs equal 100 BP epochs in accuracy.
- Do not claim the Hebbian and BP accuracy protocols are identical.
- Do not present the approximate comparison-paper graph reads as exact table values.
- Do not claim the Ridge implementation count is exact across scikit-learn versions.
- Do not claim an exact operation count for `torch.symeig`.
- Do not say the active notebook computes conventional full-training-set ZCA; it uses 9,000 images and divides `T^T T` by 3,072.
- Do not describe feature extraction as completely frozen: convolutional weights are frozen, but adaptive thresholds continue updating.
- Do not say local learning automatically guarantees lower peak memory; an explicit-unfold implementation can be larger than an optimized BP reference.
- Do not claim Hebbian learning is universally more accurate, more efficient, more biologically plausible in every respect, or superior for all hardware.

## Hardest five questions

1. **Question 8: Is the comparison fair?** The answer must distinguish matched training core from source-appropriate method-level workflows without pretending the protocols are identical.
2. **Question 12: What does one-batch equivalence really prove?** Avoid extending a correct local validation into a claim about full training or accuracy.
3. **Question 16: Are the scaled Hebbian totals exact?** Be ready to separate shape-derived exact terms from representative winner-dependent terms.
4. **Questions 23-24: How defensible are the ZCA eigensolver and Ridge costs?** The correct response is to acknowledge backend uncertainty, state the chosen sensitivity/reference models, and explain why the conclusions are not reversed.
5. **Question 30: Why does the early audit contain stronger or different statements?** Treat this as evidence of self-correction: the frozen final results use the corrected scripts, actual seeded counts, and narrower claims.

## Weaknesses Bayan and Shay should be especially ready to explain

The largest scientific weakness is that the project is an analytical comparison across partially different protocols, not a controlled end-to-end hardware benchmark. The BP architecture contains clearly documented assumptions, while the most favorable A2 result depends on unimplemented sparse kernels. The equivalence evidence is strong for the exercised update but limited to one deterministic synthetic batch. Winner-dependent training totals extrapolate one reference pattern. Ridge and eigensolver details are not version/backend fixed. Finally, the active notebook has unusual pipeline details—a 9,000-image scaled-scatter ZCA construction, random test flipping, unused quadrant calculations, and thresholds that continue adapting during feature extraction—that make its saved accuracy and method-level behavior less clean than a modern controlled benchmark.

The safest final defense position is therefore: **we established a reproducible, implementation-specific resource advantage under explicit analytical models, and we also identified exactly what must be implemented and measured before making a hardware performance or energy claim.**
