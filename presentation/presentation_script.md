# Presentation script

## Slide 1 — Hebbian CNNs versus backpropagation

**Speaker:** Bayan  
**Estimated time:** 0:35

**Exact narration:**

Hello, we are Bayan and Shay. Our project studies the computational and memory cost of a concrete Hebbian convolutional network for CIFAR-10, and compares it with backpropagation. We focus on the published HebbianCNNPyTorch implementation. The question is deliberately narrow: for this architecture and training workflow, what resources does each learning method require? We do not claim a universal winner, and we do not estimate electricity without controlled hardware measurements.

## Slide 2 — Research question and evidence

**Speaker:** Bayan  
**Estimated time:** 1:00

**Exact narration:**

Backpropagation computes a loss, propagates derivatives through the complete network, stores parameter gradients, and often keeps optimizer state. Hebbian learning can update a layer from information available locally: its input patch, its winning output, and its current weights. That locality may remove global backward work and some stored state.

Our research question was whether this advantage appears in the actual three-layer repository, rather than only in a simplified equation. We had no controlled CPU or GPU experiment with power sampling, so electricity would have been speculation. Instead, we built a hardware-independent model. We count scalar operations by type, estimate tensor storage, and estimate logical tensor movement. These metrics describe algorithmic resource demand. They do not predict joules, wall-clock time, or measured GPU memory.

## Slide 3 — Architecture and four execution models

**Speaker:** Bayan  
**Estimated time:** 1:10

**Exact narration:**

The active network has three valid convolutions with 100, 196, and 400 output channels. Their kernels are five by five, three by three, and three by three. Each layer uses two by two average pooling. The final tensor has 400 channels at two by two spatial positions, so the frozen representation contains 1,600 features. The Hebbian workflow then fits a Ridge classifier.

The local rule is Grossberg Instar. At each spatial position, WTA activates the positive top-cutoff winner or winners, if any. A non-positive maximum produces no winner, while a positive tie can activate more than one filter. The repository expresses this update through a surrogate loss and PyTorch autograd. We created a direct formula and tested one deterministic batch through all three layers. In that deterministic test batch, every updated weight and threshold matched within the stated floating-point tolerance.

That verification lets us separate four models: A1 is direct dense local Hebbian learning. A2 conditional assumes true sparse storage and sparse kernels. B is the current dense surrogate and autograd implementation. BP is an equal-architecture analytical reference with a learned linear classifier.

## Slide 4 — Matched-work compute

**Speaker:** Bayan  
**Estimated time:** 1:15

**Exact narration:**

This slide compares one batch of learning work with batch size 100. A1 requires 8.667 billion modeled scalar operations. A2 conditional requires 1.380 billion, but only if the structural sparsity is genuinely exploited. The current surrogate implementation B requires 17.268 billion. The backpropagation reference requires 24.430 billion.

So BP is about 2.819 times A1, 17.699 times A2 conditional, and 1.415 times B. The main point is not that Hebbian convolution is free. Every method still performs large forward convolutions. The direct update saves the global backward path and optimizer work. B recovers the same local rule, but autograd adds computational overhead. A2 saves much more because it skips structurally masked higher-layer work. Ordinary dense PyTorch convolution would not realize that saving. The ordering remains the same when each method is scaled to twenty epochs.

## Slide 5 — Full pipeline and persistent state

**Speaker:** Shay  
**Estimated time:** 1:15

**Exact narration:**

The matched batch isolates the learning core. The fuller Hebbian method-level subtotal includes known ZCA preprocessing arithmetic, including the modeled construction and application work, twenty training epochs, post-training feature extraction with convolutional weights frozen while adaptive thresholds continue updating, Ridge fitting, and evaluation. The eigensolver remains a separate sensitivity estimate. Under that scope, A1 totals 112.590 trillion operations, A2 conditional 35.365 trillion, and B 198.609 trillion. BP at twenty epochs totals 245.158 trillion. At the comparison paper's one-hundred-epoch budget, BP reaches 1,222.377 trillion.

Persistent state shows a related effect. A1 uses 7.089 megabytes. B uses 10.677 megabytes because autograd needs dense gradients. BP with Adam uses 14.488 megabytes for parameters, gradients, and two moment tensors. A2 conditional uses 0.106 megabytes with the stated CSR32 representation, including sparse metadata and thresholds. These are analytical tensor payloads, not measured allocator or GPU peaks.

## Slide 6 — Accuracy and convergence context

**Speaker:** Shay  
**Estimated time:** 1:05

**Exact narration:**

Compute cannot be interpreted without accuracy context, but these sources do not form one controlled experiment. The original Hebbian paper reports 64.55 percent plus or minus 0.4 across ten runs. The saved notebook records 64.65 percent for one run, which is consistent but is not a replication study.

The comparison paper gives approximate graph readings. Hebbian learning is around 66 percent at both twenty and one hundred epochs. Backpropagation is around 57 percent at twenty epochs and 70 percent at one hundred epochs. This supports the observation that the Hebbian representation stabilizes earlier, while BP continues improving and eventually exceeds it in that experiment.

Therefore, a five-epoch Hebbian versus one-hundred-epoch BP budget can illustrate convergence sensitivity, but it cannot be called an equal-accuracy comparison. We also avoid a combined efficiency score because the training protocols, classifiers, and evidence types differ.

## Slide 7 — Supported findings and limits

**Speaker:** Shay  
**Estimated time:** 1:00

**Exact narration:**

The strongest supported finding is implementation-specific. For this architecture and our frozen counting rules, direct local Instar uses less modeled arithmetic than the equal-architecture BP reference. The current autograd method captures only part of that potential because its graph and gradients remain dense. A2 conditional gives the largest modeled savings, but it is an opportunity estimate that requires real sparse storage and kernels.

Local execution can also reduce persistent learning state and global-gradient traffic. However, lower counts do not guarantee lower runtime or energy. Kernel fusion, cache behavior, sparse irregularity, and library choices all matter on real hardware. The BP reference also contains documented assumptions where the comparison paper omits details. Finally, the accuracy protocols are not identical. These boundaries are central to the conclusion, rather than footnotes to remove.

## Slide 8 — Conclusion and Q&A

**Speaker:** Shay  
**Estimated time:** 0:35

**Exact narration:**

We conclude that the verified direct Hebbian rule can be implemented considerably more cheaply than conventional backpropagation under this analytical model. Implementation matters: A1 is much cheaper than the current surrogate method B, and A2 conditional could reduce cost further with genuine sparse execution. This is a resource-efficiency result for one concrete network. It does not establish equal accuracy, measured energy savings, or universal Hebbian superiority. Thank you. We are happy to take questions.

## Timing summary

| Speaker | Slides | Estimated time |
|---|---:|---:|
| Bayan | 1–4 | 4:00 |
| Shay | 5–8 | 3:55 |
| **Total** | **8 slides** | **7:55** |

Target narration length: approximately 900–1,050 spoken words.
