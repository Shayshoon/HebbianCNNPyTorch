# Accuracy and performance context

## Status and scope

This document connects the frozen analytical compute/memory models to the
available CIFAR-10 accuracy evidence. It does not treat lower arithmetic cost as
proof of a better learning method, does not combine compute and accuracy into an
artificial score, and does not convert operations or bytes into runtime or
energy.

The technical computational methodology is **FROZEN** after the final checks
described below. The original notebooks were not modified.

## Sources and evidence classes

- **Original paper:** T. Miconi, *Hebbian learning with gradients: Hebbian
  convolutional neural networks with modern deep learning frameworks*,
  `2107.01729v2.pdf`.
- **Comparison paper:** M. Gupta et al., *Is Bio-Inspired Learning Better than
  Backprop? Benchmarking Bio Learning vs. Backprop*, `2212.04614.pdf`.
- **Repository implementation:** `HebbGrad_Github.ipynb`, including its saved
  outputs.
- **Frozen analytical models:** `hebbian_compute_cost.py`,
  `backprop_compute_cost.py`, `method_level_compute_cost.py`, and
  `memory_resource_cost.py`.

Evidence labels used below:

- **EXACT PAPER:** a numerical value printed in paper text or a table.
- **EXACT SAVED RUN:** a numerical value printed in the saved notebook output;
  it is exact for that one saved run, not a multi-run estimate.
- **APPROXIMATE GRAPH READ:** a value read visually from a plotted mean curve.
  Values are deliberately rounded to about one percentage point.

## 1. Original Hebbian implementation and its readout

The active repository setup corresponds closely to the original paper's
"network with triangle method & pruning, trained weights" condition:

- CIFAR-10, using all 50,000 training images and 10,000 test images.
- Three valid convolutions with 100, 196, and 400 channels; kernels 5x5, 3x3,
  and 3x3; 2x2 average pooling after each layer.
- Twenty unsupervised Hebbian training epochs.
- Grossberg Instar learning, binarized top-1 WTA for plasticity, triangle
  activations for propagated/readout features, adaptive thresholds, per-filter
  L2 normalization, whole-image ZCA, and random 99% masks in layers 2 and 3.
- Final representation: 400x2x2 = 1,600 features.

The architecture, 20-epoch budget, freezing step, and linear evaluation are
described in the original paper on page 4, Sections 2.3-2.4. The exact result is
reported on page 6, Table 1.

### Exact evaluation procedure in `HebbGrad_Github.ipynb`

After the 20 Hebbian epochs, convolutional weights stop updating. The notebook
then makes one pass over the 50,000-example training set and one pass over the
10,000-example test set. It collects the final pooled tensor `x`, not the
quadrant alternatives, because the active assignments are
`zetrainouts = trainouts` and `zetestouts = testouts`.

Each 400x2x2 tensor is flattened to 1,600 features. Training labels are converted
to ten-dimensional one-hot targets. The notebook fits
`sklearn.linear_model.Ridge()` with defaults (therefore `alpha=1.0` and
`fit_intercept=True` for the usual sklearn API), predicts ten real-valued scores
for each test example, takes `argmax`, and compares that class with the integer
test label.

The saved notebook output is **64.65%** test accuracy for this single run. This
is consistent with the paper's multi-run result of **64.55% +/- 0.4 percentage
points** in Table 1. The prose immediately below Table 1 instead says +/- 0.5;
this document uses the structured table value and records the source's internal
0.1-point discrepancy as a limitation.

Two implementation details prevent calling the post-training extractor fully
frozen: adaptive thresholds continue changing during both feature passes, and
the test transform includes a random horizontal flip. The active notebook also
constructs ZCA from one shuffled batch of 9,000 images, whereas the original
paper says on page 3 that its ZCA matrix is computed over the entire training
set.

## 2. Defensible CIFAR-10 accuracy results

### Exact values

| Source and location | Algorithm / implementation | Data fraction | Epoch budget | Mean/test accuracy | Standard deviation | Evidence |
|---|---|---:|---:|---:|---:|---|
| Original paper, p. 6, Table 1 | Hebbian, triangle + pruning, final 400x2x2 output with linear regression readout | 100% | 20 | 64.55% | 0.4 percentage points | EXACT PAPER; mean and SD over 10 runs |
| `HebbGrad_Github.ipynb`, saved classifier/evaluation cells | Active Hebbian repository run, 1,600-feature default Ridge readout | 100% | 20 | 64.65% | Not applicable | EXACT SAVED RUN; one run, not a reported mean |

The original paper's Table 1 also gives exact layerwise quadrant-readout results,
but they are not used as the main method accuracy because the active notebook's
selected classifier input is the final 1,600-feature output.

### Approximate comparison-paper values

The comparison paper does not print numerical CIFAR-10 endpoint values in a
table. Page 3, Figure 2 plots mean curves and shaded standard deviations over 10
runs. The values below are therefore visual reads, rounded to avoid false
precision. A numerical standard deviation cannot be recovered defensibly from
the shading.

| Source and location | Algorithm | Dataset | Data fraction | Epoch budget | Mean accuracy | Standard deviation | Evidence |
|---|---|---|---:|---:|---:|---|---|
| Comparison paper, p. 3, Fig. 2(b) | Hebbian (HB) | CIFAR-10 | 100% | 20 | approximately 66% | Shown as shading; not numerically reported | APPROXIMATE GRAPH READ |
| Comparison paper, p. 3, Fig. 2(b) | Backprop (BP) | CIFAR-10 | 100% | 20 | approximately 57% | Shown as shading; not numerically reported | APPROXIMATE GRAPH READ |
| Comparison paper, p. 3, Fig. 2(d) | Hebbian (HB) | CIFAR-10 | 100% | 100 | approximately 66% | Shown as shading; not numerically reported | APPROXIMATE GRAPH READ |
| Comparison paper, p. 3, Fig. 2(d) | Backprop (BP) | CIFAR-10 | 100% | 100 | approximately 70% | Shown as shading; not numerically reported | APPROXIMATE GRAPH READ |

These four graph reads are the only comparison-paper CIFAR-10 endpoint values
used here. The paper's numerical Table I on page 6 concerns accuracy *after 95%
post-training magnitude pruning* and excludes Hebbian learning; it is not a
clean endpoint table for the present Hebbian-vs-BP comparison.

## 3. What "about 5 vs about 100 epochs" means

The comparison paper makes the convergence statement in several places:

- Abstract, page 1: Hebbian learning is said to learn in about 5 epochs, compared
  with about 100 for BP.
- Page 3, Figure 2 caption: Hebbian is said to converge in around 5 epochs.
- Page 5, Section IV: with the full dataset and full budget, BP needs the full
  100 epochs while Hebbian finishes learning within about 5 epochs.

Figure 2 supports a **stabilization-speed** interpretation: the orange Hebbian
curve becomes nearly flat early, while the blue BP curve keeps improving for
many more epochs. It does **not** establish an equal-accuracy comparison. In the
100%-data panels, Hebbian is around 66% at both 20 and 100 epochs, while BP rises
from around 57% at 20 epochs to around 70% at 100 epochs. Thus "5 vs 100" must
not be rewritten as "Hebbian reaches BP's 100-epoch accuracy in 5 epochs."

The frozen compute model includes a 5-epoch Hebbian sensitivity view because it
matches the paper's convergence-budget claim, but that view is not a
matched-accuracy result.

## 4. Comparability with the analytical models

| Dimension | Accuracy experiments | Frozen analytical model | Consequence |
|---|---|---|---|
| Channel architecture | Both papers use three convolutional layers with 100/196/400 channels | Exact same channel counts | Strong match |
| Hebbian implementation | Comparison paper says it uses the original implementation; Appendix A says ZCA, pruning, and triangle | Model traces the active notebook's branches | Good lineage, but not proof of identical commit/runtime/configuration |
| Kernels, pooling, BP activations | Original notebook specifies 5/3/3 valid convolutions and 2x2 average pooling; comparison paper does not fully specify BP geometry or activation | BP assumes the Hebbian geometry and ReLU for an equal-architecture reference | Explicit fairness assumption, not paper-direct BP detail |
| Preprocessing | Original paper says full-training-set ZCA; active notebook uses one 9,000-image batch; comparison paper says it uses ZCA | Hebbian method-level model follows the active 9,000-image notebook path; BP core excludes unspecified preprocessing | Not identical across every source |
| Readout | Hebbian: frozen features + Ridge; BP: trained linear output layer | Ridge is modeled separately; BP classifier is trained end-to-end | Method-level comparison includes different, source-appropriate readouts |
| Ridge solver | Repository calls default `Ridge()` without a version lock | Dense primal Cholesky is the main analytical reference | Reasonable modern-sklearn path, not a universal exact library count |
| Optimizer | Hebbian notebook uses SGD with zero momentum as a surrogate implementation device; comparison BP uses Adam | BP model uses Adam defaults plus the paper's learning-rate/StepLR facts | Optimizer-state and arithmetic asymmetry is intentional |
| Learning rate | Active notebook's effective layer rates are 1e-4, 1e-4, 1e-3; comparison Appendix lists HB 1e-5 | Arithmetic counts are rate-value independent | Accuracy protocols are not numerically identical |
| Epochs | Active notebook: 20; comparison paper: 20 or 100, with an about-5-epoch convergence observation | Reports equal-batch, equal-20, configured 20-vs-100, and 5-vs-100 sensitivity views separately | No single scope should replace the others |
| Evaluation randomness | Active test transform includes random horizontal flip; thresholds continue adapting in feature passes | Compute model counts the active branches but does not model statistical variance | Saved 64.65% is not a deterministic benchmark |
| Sparse execution | Notebook stores dense tensors and uses dense `conv2d` | A2 assumes CSR-like storage and specialized sparse kernels | A2 is conditional/hypothetical, not current runtime behavior |

## 5. Frozen compute context

All counts are analytical scalar-operation counts under the scripts' printed
formulas, not measured hardware instructions. "Total scalar operations" adds
multiplications, additions/subtractions, comparisons, divisions, square roots,
and (for BP) transcendental evaluations as separately counted categories.

### Learning/training core

| Method | One representative batch | Equal 20 epochs | Equal 100 epochs |
|---|---:|---:|---:|
| A1 direct dense Hebbian | 8,667,340,084 | 86,666,040,782,800 | 433,359,644,142,800 |
| A2 conditional sparse-aware Hebbian | 1,380,339,683 | 13,799,879,559,140 | 69,013,466,879,140 |
| B current surrogate/autograd Hebbian | 17,267,771,387 | 172,677,356,291,000 | 863,388,211,771,000 |
| Standard BP | 24,430,482,934 | 244,304,829,340,020 | 1,221,524,146,700,100 |

At equal batch/equal 20-epoch scope, modeled BP cost is about 2.819x A1,
17.699x A2, and 1.415x B. A2's ratio is conditional on real sparse storage and
kernels; it does not describe the current dense notebook.

At the configured 20-epoch Hebbian vs 100-epoch BP budgets, BP's modeled
training-core count is about 14.095x A1, 88.517x A2, and 7.074x B. These are
budget comparisons, not equal-accuracy comparisons.

Applying the comparison paper's approximate convergence budgets gives 5-epoch
Hebbian core totals of 21,660,990,152,800 (A1), 3,447,331,936,640 (A2), and
43,169,070,888,500 (B), versus 1,221,524,146,700,100 for 100-epoch BP. The
resulting ratios (56.393x, 354.339x, and 28.296x) are sensitivity results only;
the paper does not show equal accuracy at those endpoints.

### Fuller method-level modeled subtotal

Hebbian subtotals include known ZCA construction arithmetic, 20 Hebbian epochs,
one 50,000-example training-feature pass, one 10,000-example test-feature pass,
Ridge fit, and Ridge evaluation. BP includes supervised training and one
10,000-example test pass.

| Method and budget | Modeled subtotal |
|---|---:|
| A1 direct dense Hebbian, 20 epochs | 112,590,041,834,287 |
| A2 conditional sparse-aware Hebbian, 20 epochs | 35,365,252,130,627 |
| B current surrogate/autograd Hebbian, 20 epochs | 198,609,206,182,487 |
| Standard BP, 20 epochs | 245,157,703,690,020 |
| Standard BP, 100 epochs | 1,222,377,021,050,100 |

The dense symmetric ZCA eigendecomposition is intentionally not hidden inside
those point totals. Its non-exact sensitivity range is 38,654,705,664 to
289,910,292,480 basic operations (4/3 F^3 to 10 F^3 for F=3,072). Adding that
range changes the Hebbian subtotals by only 0.019%-0.820%, depending on model;
it does not change their ordering. This is a reference range, not an exact
`torch.symeig` operation count.

## 6. Frozen memory/resource context

All values below are decimal MB of tensor payload or idealized logical tensor
movement. They are not allocator measurements, DRAM traffic, runtime, or energy.

### Persistent learning state

| Method | Total persistent payload |
|---|---:|
| A1 direct dense | 7.089 MB |
| A2 sparse-aware CSR32 | 0.106 MB |
| B surrogate/autograd | 10.677 MB |
| Standard BP + Adam | 14.488 MB |

A2's 0.106 MB includes FP32 nonzero values, int32 column indices, and CSR row
pointers. Alternative complete representations range from 0.104 MB (CSR32) to
0.351 MB (generic 4-D COO with int64 indices).

### Peak and one-time construction references

- A1 optimized method-level plus persistent ZCA: 87.492 MB.
- A2 optimized method-level plus persistent ZCA: 80.509 MB.
- Current B notebook named tensors plus persistent ZCA: 278.557 MB; allocator,
  autograd, and convolution workspaces can raise this.
- BP retained-forward inventory: 72.233 MB before transient activation-gradient
  buffers.
- BP optimized one-gradient-buffer reference: 103.593 MB.
- BP conservative two-gradient-buffer envelope: 134.953 MB. This assumes no
  release of retained forward saves and is deliberately not claimed as an exact
  PyTorch/cuDNN peak.
- One-time ZCA construction named-tensor peak: 372.276 MB. This includes the
  9,000x3,072 source and normalized clone plus the source-visible 3,072x3,072
  covariance/eigen/reconstruction matrices. Eigensolver and GEMM workspaces are
  unknown and excluded. Ordinary per-batch Hebbian training retains the final
  37.749 MB ZCA matrix, not the full construction peak.

### Idealized per-update-batch logical tensor movement

| Method | Logical tensor movement |
|---|---:|
| A1 direct dense | 239.950 MB |
| A2 sparse-aware CSR32 | 139.038 MB |
| B surrogate/autograd | 377.999 MB |
| Standard BP + Adam | 366.434 MB |

These movement estimates assume idealized whole-tensor passes and perfect reuse
inside kernels. Physical DRAM traffic can differ substantially.

## WHAT THE EVIDENCE SUPPORTS

- The active repository configuration has a saved single-run CIFAR-10 accuracy
  of 64.65% with a 1,600-feature default Ridge readout, and the closely matching
  original-paper condition reports 64.55% +/- 0.4 over 10 runs.
- In the comparison paper's full-data CIFAR-10 plots, Hebbian is approximately
  66% at both 20 and 100 epochs; BP is approximately 57% at 20 epochs and 70% at
  100 epochs.
- The comparison paper supports the qualitative statement that Hebbian accuracy
  stabilizes around 5 epochs while BP continues improving toward 100 epochs.
- Under the frozen formulas, all three Hebbian compute models have lower
  learning-core scalar-operation counts than the equal-architecture BP
  reference at equal batch and equal 20-epoch scope. The current dense
  surrogate/autograd implementation B still models lower core compute than BP,
  but by a much smaller factor than hypothetical direct A1/A2.
- Adding the ZCA eigensolver sensitivity range does not materially change the
  method-level compute ordering.
- Direct layer-local Hebbian implementations permit earlier disposal of local
  learning intermediates; BP requires a global backward graph. A2 can also have
  much smaller persistent state and logical movement if genuine sparse kernels
  and representations are used.

## WHAT THE EVIDENCE DOES NOT SUPPORT

- It does not support the claim that lower compute automatically makes Hebbian
  learning the better method.
- It does not support an equal-accuracy "Hebbian 5 epochs vs BP 100 epochs"
  claim. The full-data 100-epoch plot has BP above Hebbian.
- It does not show that A1 or A2 reproduce the reported accuracy in an actual
  end-to-end implementation. Their update rule is verified algebraically and
  numerically for the deterministic test, but A2 remains a conditional sparse
  design.
- It does not establish exact BP architecture, preprocessing, activation, loss,
  Adam defaults, or StepLR gamma from the comparison paper. Those are explicit
  analytical assumptions where the paper is silent.
- It does not justify treating the Ridge Cholesky count as universally exact;
  solver dispatch, backend, dtype, and copies depend on sklearn/SciPy versions.
- It does not justify an exact `torch.symeig` count or an exact
  PyTorch/cuDNN/allocator memory peak.
- It does not support claims about wall-clock speed, energy, GPU cycles, or
  physical DRAM traffic.
- It does not justify a single combined efficiency/accuracy score or hiding the
  protocol mismatches between the original notebook, original paper, comparison
  paper, and analytical BP reference.

## Remaining limitations and readiness

The strongest remaining limitation is that the comparison paper's key CIFAR-10
accuracy endpoints exist only as plotted curves. Exact per-epoch means and
standard deviations would require author data or code. The current notebook is
also stochastic, version-unpinned, and differs from the original paper's stated
full-training-set ZCA construction.

Subject to those explicit limits, the technical and research methodology is
ready for graphs, a report, and a presentation. Any later visual must keep exact
and approximate accuracy values visually distinct, preserve the compute scopes,
label A2 as conditional, and avoid presenting the 5-vs-100 observation as an
equal-accuracy result.
