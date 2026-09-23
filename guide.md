# Agent Instruction Manual: Mathematical Modeling & Empirical Profiling of Hebbian CNNs

### Mission Objective

Using the local clone of `HebbianCNNPyTorch` (supporting Miconi, 2022 / [arXiv:2212.04614](https://arxiv.org/pdf/2212.04614?utm_source=gemini)), construct a closed-form mathematical cost model (FLOPs, DRAM memory traffic, and peak VRAM) for the Hebbian convolutional network. Instrument and run the Hebbian training pipeline to measure empirical hardware metrics (hardware counters, NVML energy, and memory footprint), and evaluate how accurately the theoretical mathematical model predicts real-world execution across the paper's experiments.

*(Note: No empirical Backpropagation baseline implementation is required. Theoretical comparisons to standard Backprop will use standard analytical formulas).*

---

## 1. Local Codebase Ingestion & Architectural Auditing

1. **Inspect Network Architecture:**
* Review the network definitions and training scripts in the cloned repository to extract layer parameters:
* Layer index $l \in \{1, \dots, L\}$.
* Input and output channels ($C_{\text{in}}^{(l)}, C_{\text{out}}^{(l)}$).
* Spatial kernel dimensions ($K^{(l)} \times K^{(l)}$), stride ($S^{(l)}$), padding ($P^{(l)}$).
* Spatial input and output resolutions ($H_{\text{in}}^{(l)}, W_{\text{in}}^{(l)} \to H_{\text{out}}^{(l)}, W_{\text{out}}^{(l)}$).
* Linear readout dimensions ($d_{\text{in}}^{\text{readout}} \to C_{\text{classes}}$).




2. **Audit Core Mathematical Operations in Code:**
* Locate the forward convolutive pass (e.g., `F.unfold` followed by matrix multiplication or custom convolution).
* Verify the Triangle competition logic:

$$\mu_m = \frac{1}{C_{\text{out}}} \sum_{k=1}^{C_{\text{out}}} z_{m, k}, \quad y_{m, k} = \max(0, \; z_{m, k} - \mu_m)$$


* Verify the Grossberg Instar update and normalization steps:

$$\Delta W = \frac{\eta}{M} \left( Y^T X_{\text{unfold}} - \text{diag}(S) W \right), \quad w_k \leftarrow \frac{w_k}{\Vert{}w_k\Vert{}_2}$$


* Identify how dead neurons or duplicate filters are detected and re-seeded.



---

## 2. Parameterized Mathematical Cost Model

Construct a standalone module or notebook (`hebbian_theoretical_model.py`) that implements the closed-form complexity equations.

### 2.1 Dimensional Variables per Layer $l$

* $B$: Batch size
* $D^{(l)} = C_{\text{in}}^{(l)} \cdot (K^{(l)})^2$: Dimension of an unrolled receptive field
* $M^{(l)} = B \cdot H_{\text{out}}^{(l)} \cdot W_{\text{out}}^{(l)}$: Total patch vectors per mini-batch
* $N_{\text{params}}^{(l)} = C_{\text{out}}^{(l)} \cdot D^{(l)}$: Trainable weights
* $p$: Bytes per float ($p = 4$ for FP32)
* $\alpha$: Activation density ratio ($\frac{\text{nnz}(Y)}{M \cdot C_{\text{out}}}$, empirically measured from Triangle competition, typically $\approx 0.35$)

### 2.2 Per-Step Closed-Form Equations

#### Arithmetic Operations (FLOPs)

For each unsupervised Hebbian convolutional layer $l$:


$$\Phi_{\text{Hebb}}^{(l)} = \underbrace{2 M^{(l)} N_{\text{params}}^{(l)}}_{\text{Forward Dot-Product}} + \underbrace{2 M^{(l)} C_{\text{out}}^{(l)}}_{\text{Triangle Mean \& Rectification}} + \underbrace{2 \alpha M^{(l)} N_{\text{params}}^{(l)}}_{\text{Instar Correlation } Y^T X} + \underbrace{6 N_{\text{params}}^{(l)}}_{\text{Instar Decay, Add, \& L2 Norm}}$$

For standard dense BLAS execution where sparsity is not exploited at the hardware instruction level:


$$\Phi_{\text{Hebb, dense}}^{(l)} \approx 4 M^{(l)} N_{\text{params}}^{(l)} + 6 N_{\text{params}}^{(l)}$$

#### Theoretical DRAM Memory Traffic (Bytes)

Assuming an idealized kernel execution model where layer activations and updates reside in SRAM/registers during computation:


$$Q_{\text{Hebb}}^{(l)} = p \cdot \left( \underbrace{M^{(l)} D^{(l)}}_{\text{Read Patches } X} + \underbrace{N_{\text{params}}^{(l)}}_{\text{Read Weights } W} + \underbrace{\alpha M^{(l)} C_{\text{out}}^{(l)}}_{\text{Write Activations } Y} + \underbrace{N_{\text{params}}^{(l)}}_{\text{Write Updated } W} \right)$$

If modeling standard PyTorch execution where intermediate tensors trip-round through global VRAM:


$$Q_{\text{Hebb, un-fused}}^{(l)} \approx p \cdot \Big( 6 N_{\text{params}}^{(l)} + 2 M^{(l)} D^{(l)} + (1 + \alpha) M^{(l)} C_{\text{out}}^{(l)} \Big)$$

#### Peak Working Memory Footprint (VRAM)

Under greedy layer-wise training, preceding layers are frozen and intermediate activations are discarded immediately after updating weights:


$$M_{\text{peak}}^{(l)} = p \cdot \left( 2 N_{\text{params}}^{(l)} + B \cdot C_{\text{in}}^{(l)} \cdot H_{\text{in}}^{(l)} \cdot W_{\text{in}}^{(l)} \right)$$

### 2.3 Cumulative Training Budget Model

Given $E_{\text{Hebb}}^{(l)}$ epochs to train layer $l$, $E_{\text{linear}}$ epochs for the supervised readout head, and $N_{\text{batches}}$ iterations per epoch:


$$\Phi_{\text{total}} = N_{\text{batches}} \cdot \left[ \sum_{l=1}^{L-1} \left( E_{\text{Hebb}}^{(l)} \cdot \Phi_{\text{Hebb}}^{(l)} \right) + E_{\text{linear}} \cdot \left( \sum_{l=1}^{L-1} \Phi_{\text{fwd}}^{(l)} + \Phi_{\text{BP}}^{(\text{readout})} \right) \right]$$

---

## 3. Empirical Instrumentation & Profiling Setup

Instrument the existing repository code to collect hardware data during training without altering training dynamics.

### 3.1 NVML Energy Logger (`energy_tracker.py`)

Implement a background polling thread using `pynvml` to integrate GPU power over time:

```python
import time
import threading
import pynvml

class EnergyTracker:
    def __init__(self, device_index=0, interval=0.01):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval = interval
        self.running = False
        self.total_joules = 0.0

    def _monitor(self):
        last_time = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            # Power reported in milliwatts
            power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            self.total_joules += power_w * dt
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.total_joules = 0.0
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        return self.total_joules

```

### 3.2 Peak VRAM Tracking

Wrap training steps with PyTorch memory hooks:

```python
torch.cuda.reset_peak_memory_stats()
# ... execute layer forward and instar update ...
peak_bytes = torch.cuda.max_memory_allocated()

```

### 3.3 Hardware Counter Verification (Nsight Compute)

Profile an isolated single-step execution of the Hebbian layer using `ncu` to measure actual hardware instructions and memory transactions:

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum \
    --target-processes all \
    python profile_hebbian_step.py --batch-size 128

```

---

## 4. Experiment Execution & Replication Matrix

Replicate the core experiments from the paper using the local Hebbian codebase, logging both theoretical model predictions and measured values.

### Experiment 1: Full-Dataset Convergence (CIFAR-10)

* **Execution:**
1. Run offline ZCA whitening preprocessing.
2. Train Layer 1 unsupervised ($E_1 \approx 2\text{--}5$ epochs).
3. Propagate activations, train Layer 2 unsupervised ($E_2 \approx 2\text{--}5$ epochs).
4. Train linear readout layer on extracted features ($E_{\text{linear}} \approx 10\text{--}20$ epochs).


* **Metrics to Log per Epoch:**
* Test accuracy.
* Theoretical cumulative FLOPs (from Section 2).
* Empirical cumulative Joules (via NVML).
* Peak resident VRAM.



### Experiment 2: Few-Shot / Small-Data Regimes

* Replicate the low-sample experiments from the paper (e.g., $N \in \{10, 50, 100, 500\}$ labeled samples per class).
* Evaluate compute required to reach target accuracy when fine-tuning the readout head on limited data.

---

## 5. Model Verification & Discrepancy Analysis

The final deliverable is an analytical report (`COMPUTE_EVALUATION.md`) detailing how well the theoretical model matches hardware reality.

### 5.1 Verification Checklist

1. **FLOP Scaling:** Plot theoretical FLOPs vs. batch size ($B \in [16, 256]$) and channel width ($C_{\text{out}} \in [32, 256]$). Compare directly against the FMA/FADD/FMUL instruction counts extracted by `ncu`.
2. **Memory Traffic Analysis:** Compare theoretical $Q_{\text{Hebb}}$ against `dram__bytes_read.sum + dram__bytes_write.sum`. Detail the degree of memory amplification caused by PyTorch's `F.unfold` allocating intermediate patch buffers.
3. **Analytical Comparison to Backprop:**
* Contrast the theoretical Hebbian FLOP formula ($\Phi_{\text{Hebb}} \approx 4 M \cdot N_{\text{params}}$) against the standard theoretical Backprop formula ($\Phi_{\text{BP}} \approx 6 M \cdot N_{\text{params}}$).
* Evaluate the convergence multiplier: show how convergence in fewer epochs ($E_{\text{Hebb}} \ll E_{\text{BP}}$) translates to net cumulative energy and FLOP savings despite un-fused PyTorch runtime overheads.