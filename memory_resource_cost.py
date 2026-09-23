"""Hardware-independent tensor storage and data-movement model.

This companion to the arithmetic cost scripts deliberately keeps bytes and
operations separate.  It reports exact architecture-derived tensor payloads,
then labels peak-memory and per-batch movement models as analytical estimates.

Scopes
------
* A1: verified direct Instar update with dense weights.
* A2: conditional sparse-aware direct Instar update.  This requires kernels
  and a sparse representation that do not materialize masked zeros.
* B: the notebook's surrogate-loss/autograd Hebbian implementation.
* BP: the equal-architecture supervised reference with Adam.

FP32 values use four bytes.  Sparse metadata assumptions are stated in the
printed report.  Python objects, allocator caching, CUDA/cuDNN workspaces,
kernel fusion, cache lines, transfers between devices, and library temporaries
cannot be derived from the source and are not included in exact payloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import backprop_compute_cost as bp_cost
import backprop_reference as bp
import hebbian_compute_cost as hebb


FP32_BYTES = 4
INT32_BYTES = 4
INT64_BYTES = 8

IMAGE_VECTOR = bp.INPUT_CHANNELS * bp.INPUT_SIZE * bp.INPUT_SIZE
ZCA_MATRIX_ELEMENTS = IMAGE_VECTOR * IMAGE_VECTOR
ZCA_BUILD_SAMPLES = 9_000
HEBB_CHANNEL_STATE = sum(bp.CHANNELS)
CLASSIFIER_PARAMETERS = bp.CHANNELS[-1] * 2 * 2 * bp.NUM_CLASSES + bp.NUM_CLASSES
RIDGE_FEATURES = bp.CHANNELS[-1] * 2 * 2


@dataclass(frozen=True)
class StorageRow:
    name: str
    parameters: int
    mask_or_sparse_metadata: int
    gradients: int
    optimizer_state: int
    hebbian_thresholds: int

    @property
    def total(self) -> int:
        return (
            self.parameters
            + self.mask_or_sparse_metadata
            + self.gradients
            + self.optimizer_state
            + self.hebbian_thresholds
        )


@dataclass(frozen=True)
class MovementRow:
    name: str
    weights: int
    masks_or_indices: int
    activations_and_update_buffers: int
    gradients: int
    optimizer_state: int

    @property
    def total(self) -> int:
        return (
            self.weights
            + self.masks_or_indices
            + self.activations_and_update_buffers
            + self.gradients
            + self.optimizer_state
        )


def decimal_mb(value: int) -> float:
    return value / 1_000_000


def mib(value: int) -> float:
    return value / (1024 * 1024)


def fmt_bytes(value: int) -> str:
    return f"{value:,} B ({decimal_mb(value):,.3f} MB; {mib(value):,.3f} MiB)"


def sparse_parameter_storage(facts: list[hebb.LayerFacts]) -> dict[str, int]:
    """A2 storage alternatives: L1 dense, L2/L3 sparse by output-filter row."""

    layer1_dense = facts[0].dense_weights
    sparse_nonzeros = sum(layer.active_weights for layer in facts[1:])
    row_pointer_entries = sum(layer.output_channels + 1 for layer in facts[1:])

    values = (layer1_dense + sparse_nonzeros) * FP32_BYTES
    csr32 = (
        values
        + sparse_nonzeros * INT32_BYTES
        + row_pointer_entries * INT32_BYTES
    )
    csr64 = (
        values
        + sparse_nonzeros * INT64_BYTES
        + row_pointer_entries * INT64_BYTES
    )
    # COO with (filter,row-within-flattened-kernel) coordinates.
    coo_flat_int64 = values + sparse_nonzeros * 2 * INT64_BYTES
    # Generic four-dimensional COO: out-channel, in-channel, kernel-y, kernel-x.
    coo_4d_int64 = values + sparse_nonzeros * 4 * INT64_BYTES
    return {
        "CSR int32": csr32,
        "CSR int64": csr64,
        "COO flat int64": coo_flat_int64,
        "COO 4-D int64": coo_4d_int64,
    }


def static_storage_rows(facts: list[hebb.LayerFacts]) -> list[StorageRow]:
    dense_weights = sum(layer.dense_weights for layer in facts)
    active_branch_mask = sum(
        layer.dense_weights for layer in facts if layer.training_mask_enabled
    )
    bp_parameters = dense_weights + CLASSIFIER_PARAMETERS
    sparse_csr32 = sparse_parameter_storage(facts)["CSR int32"]
    sparse_values = sum(layer.active_weights for layer in facts) * FP32_BYTES

    return [
        StorageRow(
            "A1 direct dense",
            parameters=dense_weights * FP32_BYTES,
            # Optimized storage omits the all-one L1 mask; the actual notebook
            # allocates a dense mask for L1 too and is reported separately.
            mask_or_sparse_metadata=active_branch_mask * FP32_BYTES,
            gradients=0,
            optimizer_state=0,
            hebbian_thresholds=HEBB_CHANNEL_STATE * FP32_BYTES,
        ),
        StorageRow(
            "A2 sparse-aware (CSR32)",
            parameters=sparse_values,
            mask_or_sparse_metadata=sparse_csr32 - sparse_values,
            gradients=0,
            optimizer_state=0,
            hebbian_thresholds=HEBB_CHANNEL_STATE * FP32_BYTES,
        ),
        StorageRow(
            "B surrogate/autograd",
            parameters=dense_weights * FP32_BYTES,
            # Exact notebook allocation: masks are created for all layers,
            # including an all-one L1 mask.
            mask_or_sparse_metadata=dense_weights * FP32_BYTES,
            gradients=dense_weights * FP32_BYTES,
            optimizer_state=0,  # SGD(momentum=0) has no moment tensor.
            hebbian_thresholds=HEBB_CHANNEL_STATE * FP32_BYTES,
        ),
        StorageRow(
            "Standard BP + Adam",
            parameters=bp_parameters * FP32_BYTES,
            mask_or_sparse_metadata=0,
            gradients=bp_parameters * FP32_BYTES,
            # First and second moments. Per-parameter-group step scalars are
            # tens of bytes and intentionally left as implementation detail.
            optimizer_state=2 * bp_parameters * FP32_BYTES,
            hebbian_thresholds=0,
        ),
    ]


def direct_optimized_workspace(
    facts: list[hebb.LayerFacts], sparse: bool
) -> int:
    """Layer-local streaming workspace, excluding persistent model storage.

    One input, one reusable preactivation/triangle buffer, pooled output, top-1
    cutoff, winner coordinates, and one update accumulator are retained.  It
    assumes a streaming Instar kernel, not ``torch.nn.functional.unfold``.
    """

    peaks = []
    for layer in facts:
        update_values = layer.active_weights if sparse else layer.dense_weights
        peaks.append(
            layer.input_elements * FP32_BYTES
            + layer.activations * FP32_BYTES
            + layer.pooled_elements * FP32_BYTES
            + layer.spatial_columns * FP32_BYTES  # top-1 cutoff
            + layer.spatial_columns * 3 * INT64_BYTES  # worst-case coordinates
            + update_values * FP32_BYTES
        )
    return max(peaks)


def explicit_unfold_direct_workspace(facts: list[hebb.LayerFacts]) -> int:
    """Approximate peak payload of the verified direct kernel's intermediates."""

    peaks = []
    for layer in facts:
        patch_elements = layer.spatial_columns * layer.input_dimension
        peaks.append(
            layer.input_elements * FP32_BYTES
            + 2 * layer.activations * FP32_BYTES  # preactivation and real WTA
            + 2 * patch_elements * FP32_BYTES  # unfold plus selected patches
            + layer.spatial_columns * 3 * INT64_BYTES
            + layer.spatial_columns * FP32_BYTES  # top-1 cutoff
            + 3 * layer.dense_weights * FP32_BYTES  # accum/delta/updated weight
        )
    return max(peaks)


def surrogate_optimized_workspace(facts: list[hebb.LayerFacts]) -> int:
    """Layer-local lower-reference payload for the surrogate graph."""

    return max(
        layer.input_elements * FP32_BYTES
        + 3 * layer.activations * FP32_BYTES  # prelim, real, yforgrad
        + layer.spatial_columns * FP32_BYTES
        for layer in facts
    )


def surrogate_notebook_named_tensor_workspace(
    facts: list[hebb.LayerFacts],
) -> int:
    """Approximate peak of explicitly named live tensors in the notebook loop.

    Prior layers retain ``xs``, ``realys`` and ``prelimys``.  At the current
    layer the source holds x/xs, prelimy, prelimysav, realy, realys, prelimys,
    yforgrad, triangle outy, pooled output, and the top-k cutoff.  Autograd
    internals, allocator caching, ZCA temporaries and library workspaces remain
    outside this named-tensor inventory.
    """

    peaks = []
    persistent = 0
    for layer in facts:
        current = (
            2 * layer.input_elements * FP32_BYTES
            + 7 * layer.activations * FP32_BYTES
            + layer.pooled_elements * FP32_BYTES
            + layer.spatial_columns * FP32_BYTES
        )
        peaks.append(persistent + current)
        persistent += (
            layer.input_elements + 2 * layer.activations
        ) * FP32_BYTES
    return max(peaks)


def bp_retained_forward_payload(facts: list[bp_cost.ConvFacts]) -> int:
    """Conventional retained forward tensors for one global backward graph."""

    input_batch = bp.BATCH_SIZE * bp.INPUT_CHANNELS * bp.INPUT_SIZE**2
    activations = sum(layer.output_activations for layer in facts)
    pooled = sum(layer.pooled_elements for layer in facts)
    logits = bp.BATCH_SIZE * bp.NUM_CLASSES
    return (input_batch + activations + pooled + logits) * FP32_BYTES


def bp_activation_gradient_buffer(facts: list[bp_cost.ConvFacts]) -> int:
    """Largest single FP32 activation-gradient tensor in the BP network."""

    return max(layer.output_activations for layer in facts) * FP32_BYTES


def zca_construction_payloads() -> dict[str, int]:
    """Source-level payloads that coexist while the notebook constructs ZCA.

    The active path retains ``xorig`` and its normalized clone ``t``.  It then
    retains ``covt``, the eigenvectors, eigenvalue vectors, and temporaries from
    ``E @ diag(d) @ E.T``.  The reconstruction peak has four F-by-F matrices
    live: covt, E, one diagonal/product temporary, and the product/result.
    Library eigensolver/GEMM workspaces and DataLoader host-side prefetch are
    intentionally excluded because the source does not determine them.
    """

    sample_batch = ZCA_BUILD_SAMPLES * IMAGE_VECTOR * FP32_BYTES
    matrix = ZCA_MATRIX_ELEMENTS * FP32_BYTES
    eigenvalue_vector = IMAGE_VECTOR * FP32_BYTES
    targets = ZCA_BUILD_SAMPLES * INT64_BYTES
    return {
        "Normalized sample pair (xorig + t)": 2 * sample_batch + targets,
        "Covariance/eigensystem named payload": (
            2 * sample_batch
            + 2 * matrix
            + 2 * eigenvalue_vector
            + targets
        ),
        "ZCA reconstruction named-tensor peak": (
            2 * sample_batch
            + 4 * matrix
            + 2 * eigenvalue_vector
            + targets
        ),
        "Persistent zcamat after construction": matrix,
    }


def peak_memory_rows(
    facts: list[hebb.LayerFacts], bp_facts: list[bp_cost.ConvFacts]
) -> list[tuple[str, int, str]]:
    storage = {row.name: row.total for row in static_storage_rows(facts)}
    zca = ZCA_MATRIX_ELEMENTS * FP32_BYTES
    bp_forward = bp_retained_forward_payload(bp_facts)
    bp_grad_buffer = bp_activation_gradient_buffer(bp_facts)
    return [
        (
            "A1 optimized layer-local core",
            storage["A1 direct dense"] + direct_optimized_workspace(facts, False),
            "theoretical streaming minimum; preprocessing excluded",
        ),
        (
            "A1 explicit-unfold reference core",
            storage["A1 direct dense"] + explicit_unfold_direct_workspace(facts),
            "approximate intermediates used by verified direct kernel",
        ),
        (
            "A1 optimized method-level + ZCA",
            storage["A1 direct dense"]
            + direct_optimized_workspace(facts, False)
            + zca,
            "adds persistent 3072x3072 FP32 ZCA matrix",
        ),
        (
            "A2 optimized layer-local core",
            storage["A2 sparse-aware (CSR32)"]
            + direct_optimized_workspace(facts, True),
            "conditional CSR32 sparse kernels; no dense materialization",
        ),
        (
            "A2 optimized method-level + ZCA",
            storage["A2 sparse-aware (CSR32)"]
            + direct_optimized_workspace(facts, True)
            + zca,
            "conditional sparse kernels plus persistent ZCA matrix",
        ),
        (
            "B optimized layer-local core",
            storage["B surrogate/autograd"]
            + surrogate_optimized_workspace(facts),
            "lower-reference graph payload; PyTorch internals excluded",
        ),
        (
            "B notebook named tensors + ZCA",
            storage["B surrogate/autograd"]
            + surrogate_notebook_named_tensor_workspace(facts)
            + zca,
            "named tensors only; allocator/autograd workspaces can raise peak",
        ),
        (
            "BP retained-forward inventory",
            storage["Standard BP + Adam"] + bp_forward,
            "persistent state plus forward saves; no activation-gradient buffer",
        ),
        (
            "BP optimized one-grad-buffer reference",
            storage["Standard BP + Adam"] + bp_forward + bp_grad_buffer,
            "adds one recycled largest activation-gradient buffer",
        ),
        (
            "BP conservative two-grad-buffer envelope",
            storage["Standard BP + Adam"] + bp_forward + 2 * bp_grad_buffer,
            "unfused backward overlap; assumes no forward-save release",
        ),
    ]


def notebook_auxiliary_payload(facts: list[hebb.LayerFacts]) -> dict[str, int]:
    dense_weights = sum(layer.dense_weights for layer in facts)
    channel_square = sum(layer.output_channels**2 for layer in facts)
    batches = hebb.TOTAL_BATCHES

    # trainouts/testouts plus all three quadrant lists, all float32 payloads.
    train_features = 50_000 * (1_600 + 400 + 784 + 1_600) * FP32_BYTES
    test_features = 10_000 * (1_600 + 400 + 784 + 1_600) * FP32_BYTES
    return {
        "Persistent ZCA matrix": ZCA_MATRIX_ELEMENTS * FP32_BYTES,
        "Unused-but-allocated wflat filters": (
            75 * 75 + 900 * 900 + 1_764 * 1_764
        )
        * FP32_BYTES,
        "Four dense channel-square state families": 4
        * channel_square
        * FP32_BYTES,
        "Four monitoring vectors (threshold excluded)": 4
        * HEBB_CHANNEL_STATE
        * FP32_BYTES,
        "20-epoch allthres/allfirings payload": 3
        * batches
        * HEBB_CHANNEL_STATE
        * FP32_BYTES,
        "Eight periodic dense weight snapshots": 8
        * dense_weights
        * FP32_BYTES,
        "Collected train features + unused quadrants": train_features,
        "Collected test features + unused quadrants": test_features,
        "Vstacked final train features for Ridge": 50_000
        * RIDGE_FEATURES
        * FP32_BYTES,
        "Vstacked final test features": 10_000
        * RIDGE_FEATURES
        * FP32_BYTES,
    }


def movement_rows(facts: list[hebb.LayerFacts]) -> list[MovementRow]:
    """Idealized compulsory whole-tensor traffic for one update batch.

    This is a tensor-pass model, not DRAM traffic.  It assumes perfect reuse
    inside convolution kernels and counts explicit logical reads/writes once.
    Cache behavior, im2col lowering, fusion and backend workspaces can change
    physical movement substantially.
    """

    dense_weights = sum(layer.dense_weights for layer in facts)
    active_weights = sum(layer.active_weights for layer in facts)
    masked_dense = sum(
        layer.dense_weights for layer in facts if layer.training_mask_enabled
    )
    activations = sum(layer.activations for layer in facts)
    pooled = sum(layer.pooled_elements for layer in facts)
    layer_inputs = sum(layer.input_elements for layer in facts)
    nonfirst_inputs = sum(layer.input_elements for layer in facts[1:])
    input_batch = facts[0].input_elements
    logits = bp.BATCH_SIZE * bp.NUM_CLASSES

    # Produce and consume each forward tensor once.  Pool outputs that become
    # later inputs are not counted a second time here.
    hebb_forward_stream = (
        input_batch + 2 * (activations + pooled)
    ) * FP32_BYTES
    bp_forward_stream = (
        input_batch + 2 * (activations + pooled + logits)
    ) * FP32_BYTES

    dense_selected_patch_reads = sum(
        layer.winners * layer.input_dimension for layer in facts
    )
    sparse_selected_patch_reads = sum(
        layer.winner_weight_coordinates for layer in facts
    )

    sparse_info = sparse_parameter_storage(facts)
    sparse_values = active_weights * FP32_BYTES
    csr32_metadata = sparse_info["CSR int32"] - sparse_values

    a1 = MovementRow(
        "A1 direct dense",
        # forward read; update read/write; norm square/read/write; mask
        # read/write for L2/L3.
        weights=(6 * dense_weights + 2 * masked_dense) * FP32_BYTES,
        masks_or_indices=masked_dense * FP32_BYTES,
        activations_and_update_buffers=(
            hebb_forward_stream
            + dense_selected_patch_reads * FP32_BYTES
            + 2 * dense_weights * FP32_BYTES
        ),
        gradients=0,
        optimizer_state=0,
    )
    a2 = MovementRow(
        "A2 sparse-aware (CSR32)",
        weights=6 * active_weights * FP32_BYTES,
        # Three metadata passes: sparse forward, update and normalization.
        masks_or_indices=3 * csr32_metadata,
        activations_and_update_buffers=(
            hebb_forward_stream
            + sparse_selected_patch_reads * FP32_BYTES
            + 2 * active_weights * FP32_BYTES
        ),
        gradients=0,
        optimizer_state=0,
    )
    surrogate = MovementRow(
        "B surrogate/autograd",
        # Dense forward, norm-expression forward/backward, SGD, mask, norm.
        weights=(8 * dense_weights + 3 * masked_dense) * FP32_BYTES,
        masks_or_indices=masked_dense * FP32_BYTES,
        activations_and_update_buffers=(
            hebb_forward_stream
            + 4 * activations * FP32_BYTES
            + layer_inputs * FP32_BYTES
            + 2 * nonfirst_inputs * FP32_BYTES
        ),
        gradients=2 * dense_weights * FP32_BYTES,  # write then SGD read
        optimizer_state=0,
    )

    bp_parameters = dense_weights + CLASSIFIER_PARAMETERS
    backward_weight_reads = (
        sum(layer.dense_weights for layer in facts[1:])
        + bp.CHANNELS[-1] * 2 * 2 * bp.NUM_CLASSES
    )
    bp_activation_traffic = (
        bp_forward_stream
        + 2 * logits * FP32_BYTES
        + 4 * activations * FP32_BYTES
        + 2 * nonfirst_inputs * FP32_BYTES
        + layer_inputs * FP32_BYTES
        + 3 * facts[-1].pooled_elements * FP32_BYTES
    )
    standard_bp = MovementRow(
        "Standard BP + Adam",
        weights=(
            bp_parameters + backward_weight_reads + 2 * bp_parameters
        )
        * FP32_BYTES,
        masks_or_indices=0,
        activations_and_update_buffers=bp_activation_traffic,
        gradients=2 * bp_parameters * FP32_BYTES,
        optimizer_state=4 * bp_parameters * FP32_BYTES,
    )
    return [a1, a2, surrogate, standard_bp]


def print_static_storage(facts: list[hebb.LayerFacts]) -> None:
    print("\n=== Exact persistent learning-state tensor payloads ===")
    print(
        f"{'Method':30} {'Parameters':>15} {'Mask/index':>15} "
        f"{'Gradients':>15} {'Optimizer':>15} {'Threshold':>13} {'Total':>15}"
    )
    print("-" * 125)
    for row in static_storage_rows(facts):
        print(
            f"{row.name:30} {decimal_mb(row.parameters):15.3f} "
            f"{decimal_mb(row.mask_or_sparse_metadata):15.3f} "
            f"{decimal_mb(row.gradients):15.3f} "
            f"{decimal_mb(row.optimizer_state):15.3f} "
            f"{decimal_mb(row.hebbian_thresholds):13.6f} "
            f"{decimal_mb(row.total):15.3f}"
        )
    print("All table values are decimal MB; tensor values are FP32.")

    dense_weights = sum(layer.dense_weights for layer in facts)
    print(
        "Actual notebook dense mask allocation (including all-one L1): "
        f"{fmt_bytes(dense_weights * FP32_BYTES)}. A1's table row omits that "
        "redundant L1 mask; B retains it because it models current code."
    )


def print_sparse_storage(facts: list[hebb.LayerFacts]) -> None:
    print("\n=== A2 sparse parameter representation alternatives ===")
    print(
        "Layer 1 stays dense; layers 2/3 store only seeded nonzeros. Values "
        "are FP32. CSR rows are output filters and columns flatten each filter."
    )
    for name, value in sparse_parameter_storage(facts).items():
        print(f"  {name:18}: {fmt_bytes(value)}")
    print(
        "These totals include values and indices/pointers. They do not claim "
        "that PyTorch dense conv2d uses the sparse representation."
    )


def print_peak_memory(
    facts: list[hebb.LayerFacts], bp_facts: list[bp_cost.ConvFacts]
) -> None:
    print("\n=== Approximate peak batch-100 training tensor memory ===")
    print(f"{'Model/scope':42} {'Peak payload':>19}  Basis")
    print("-" * 122)
    for name, value, basis in peak_memory_rows(facts, bp_facts):
        print(f"{name:42} {decimal_mb(value):15.3f} MB  {basis}")
    print(
        "These are tensor-payload inventories, not measured allocator peaks. "
        "BP must retain forward information until global backward; direct A1/A2 "
        "can update a layer and discard its local intermediates before moving on."
    )
    print(
        "The two BP additions explicitly expose transient activation-gradient "
        "storage. The one-buffer row is an optimized recycling reference; the "
        "two-buffer row is a conservative source-level overlap envelope. Actual "
        "PyTorch/cuDNN peaks may be lower through release/reuse or higher through "
        "autograd, convolution, allocator, and backend workspaces."
    )


def print_zca_construction_memory() -> None:
    print("\n=== One-time ZCA construction tensor payloads ===")
    print(f"{'Construction stage':45} {'Payload':>19}  Basis")
    print("-" * 118)
    bases = {
        "Normalized sample pair (xorig + t)": (
            "9,000x3,072 FP32 source plus normalized clone and int64 targets"
        ),
        "Covariance/eigensystem named payload": (
            "adds covt, eigenvectors, and two length-3,072 vectors"
        ),
        "ZCA reconstruction named-tensor peak": (
            "four 3,072x3,072 matrices coexist with xorig/t"
        ),
        "Persistent zcamat after construction": (
            "ordinary per-batch Hebbian phases retain only the final matrix"
        ),
    }
    for name, value in zca_construction_payloads().items():
        print(f"{name:45} {decimal_mb(value):15.3f} MB  {bases[name]}")
    print(
        "This is a defensible source-level named-tensor peak for the one-time "
        "preprocessing phase, not a measured allocator peak. torch.symeig and "
        "matrix multiplication may require additional backend workspaces whose "
        "sizes are not fixed by the repository. It is separate from ordinary "
        "per-batch Hebbian training memory."
    )


def print_auxiliary_memory(facts: list[hebb.LayerFacts]) -> None:
    print("\n=== Current notebook auxiliary/method-level payloads ===")
    print(
        "These are separate from the optimized learning-state table. Several "
        "are diagnostics or unused alternatives, not algorithmic necessities."
    )
    for name, value in notebook_auxiliary_payload(facts).items():
        print(f"  {name:49} {decimal_mb(value):11.3f} MB")
    print(
        "Feature arrays above sum payloads, not simultaneous exact peaks. NumPy "
        "vstack, sklearn validation/centering, float64 targets, and solver "
        "copies can add hundreds of MB and depend on library/version."
    )


def print_movement(facts: list[hebb.LayerFacts]) -> None:
    print("\n=== Idealized per-update-batch tensor data movement ===")
    print(
        f"{'Method':30} {'Weights':>13} {'Mask/index':>13} {'Activation':>13} "
        f"{'Gradient':>13} {'Opt state':>13} {'Total':>13}"
    )
    print("-" * 120)
    for row in movement_rows(facts):
        print(
            f"{row.name:30} {decimal_mb(row.weights):13.3f} "
            f"{decimal_mb(row.masks_or_indices):13.3f} "
            f"{decimal_mb(row.activations_and_update_buffers):13.3f} "
            f"{decimal_mb(row.gradients):13.3f} "
            f"{decimal_mb(row.optimizer_state):13.3f} "
            f"{decimal_mb(row.total):13.3f}"
        )
    print(
        "This is a logical whole-tensor-pass lower-reference, not physical DRAM "
        "traffic. A1/A2 winner-dependent patch reads use the deterministic "
        "representative batch. A2 savings require sparse kernels; dense conv2d "
        "does not receive them."
    )


def validate(
    facts: list[hebb.LayerFacts], bp_facts: list[bp_cost.ConvFacts]
) -> None:
    dense_weights = sum(layer.dense_weights for layer in facts)
    active_weights = sum(layer.active_weights for layer in facts)
    assert dense_weights == 889_500
    assert active_weights == 16_413
    assert CLASSIFIER_PARAMETERS == 16_010
    assert ZCA_MATRIX_ELEMENTS == 9_437_184
    assert ZCA_MATRIX_ELEMENTS * FP32_BYTES == 37_748_736
    assert bp_retained_forward_payload(bp_facts) == 57_744_800
    assert bp_activation_gradient_buffer(bp_facts) == 31_360_000
    assert zca_construction_payloads() == {
        "Normalized sample pair (xorig + t)": 221_256_000,
        "Covariance/eigensystem named payload": 296_778_048,
        "ZCA reconstruction named-tensor peak": 372_275_520,
        "Persistent zcamat after construction": 37_748_736,
    }
    assert HEBB_CHANNEL_STATE == 696
    assert sparse_parameter_storage(facts) == {
        "CSR int32": 103_696,
        "CSR int64": 141_740,
        "COO flat int64": 208_260,
        "COO 4-D int64": 350_868,
    }
    rows = static_storage_rows(facts)
    assert rows[0].parameters == 3_558_000
    assert rows[2].gradients == 3_558_000
    assert rows[3].parameters == 3_622_040
    assert rows[3].optimizer_state == 7_244_080
    assert len(bp_facts) == 3
    assert all(row.total > 0 for row in movement_rows(facts))


def main() -> None:
    facts = hebb.collect_verified_layer_facts()
    hebb.validate_model(facts)
    bp_facts = bp_cost.collect_conv_facts()
    validate(facts, bp_facts)

    print("Hardware-independent memory/resource model")
    print("Internal consistency checks: PASS")
    print(
        "Exact tensor sizes come from batch=100, channels=(100,196,400), "
        "kernels=(5,3,3), valid stride-1 convolutions, 2x2 average pooling, "
        "and the seeded masks used by the verified equivalence test."
    )
    print_static_storage(facts)
    print_sparse_storage(facts)
    print_peak_memory(facts, bp_facts)
    print_zca_construction_memory()
    print_auxiliary_memory(facts)
    print_movement(facts)

    print(
        "\nEXACTNESS CLASSES:\n"
        "  Exact from architecture/code: element counts, FP32 payloads, dense "
        "parameter/mask/gradient tensors, Adam moment tensors, thresholds, ZCA "
        "matrix size, and sparse representation byte formulas.\n"
        "  Representative assumption: winner-dependent A1/A2 update movement "
        "uses the deterministic synthetic batch.\n"
        "  Analytical estimates: peak liveness and logical tensor-pass movement.\n"
        "  Implementation-dependent and excluded: PyTorch autograd saved-tensor "
        "choices, allocator caching, convolution lowering/workspaces, cache/DRAM "
        "traffic, device transfers, Python-object overhead, and sklearn copies."
    )


if __name__ == "__main__":
    main()
