"""Analytical compute-cost model for the verified Hebbian CNN implementation.

This script models only the active three-layer Hebbian training path from
HebbGrad_Github.ipynb. It does not contain or model a conventional
backpropagation baseline.

Three Hebbian execution models are reported:

1. Direct/local (dense storage, verified rule): the Instar update implemented
   and tested in direct_hebb_equivalence_test.py, with the active notebook's
   conditional mask branch. Forward convolution remains dense and masked-out
   coordinates are discarded only in layers 2 and 3.
2. Direct/local (sparse-aware, conditional): the same mathematical update, but
   assuming a sparse representation and kernels that skip every masked weight.
   These savings do not occur automatically in PyTorch dense conv2d.
3. Current surrogate/autograd estimate: the notebook's dense forward
   convolution, surrogate expression/loss, dense convolution weight-gradient,
   SGD step, masking, and normalization.

The model counts scalar arithmetic only. It does not estimate memory traffic,
kernel launches, electricity, joules, processor cycles, or wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

import direct_hebb_equivalence_test as reference


BATCHES_PER_EPOCH = 50_000 // reference.BATCH_SIZE
HEBBIAN_EPOCHS = 20
TOTAL_BATCHES = BATCHES_PER_EPOCH * HEBBIAN_EPOCHS

# nbbatches starts at -1 and the notebook steps only when nbbatches > 200.
BURN_IN_BATCHES = 201
FIRST_EPOCH_UPDATE_BATCHES = BATCHES_PER_EPOCH - BURN_IN_BATCHES
TOTAL_UPDATE_BATCHES = TOTAL_BATCHES - BURN_IN_BATCHES


@dataclass(frozen=True)
class OperationCounts:
    multiplications: int = 0
    additions_subtractions: int = 0
    comparisons: int = 0
    divisions: int = 0
    square_roots: int = 0

    def __add__(self, other: "OperationCounts") -> "OperationCounts":
        return OperationCounts(
            self.multiplications + other.multiplications,
            self.additions_subtractions + other.additions_subtractions,
            self.comparisons + other.comparisons,
            self.divisions + other.divisions,
            self.square_roots + other.square_roots,
        )

    def scale(self, factor: int) -> "OperationCounts":
        return OperationCounts(
            self.multiplications * factor,
            self.additions_subtractions * factor,
            self.comparisons * factor,
            self.divisions * factor,
            self.square_roots * factor,
        )

    @property
    def basic_flops(self) -> int:
        """Multiplications plus additions/subtractions only."""

        return self.multiplications + self.additions_subtractions

    @property
    def total_scalar_operations(self) -> int:
        return (
            self.basic_flops
            + self.comparisons
            + self.divisions
            + self.square_roots
        )


ZERO = OperationCounts()


@dataclass(frozen=True)
class LayerFacts:
    layer: int
    input_shape: tuple[int, ...]
    conv_output_shape: tuple[int, ...]
    pooled_output_shape: tuple[int, ...]
    input_elements: int
    input_dimension: int
    output_channels: int
    spatial_columns: int
    activations: int
    dense_weights: int
    active_weights: int
    nonempty_filters: int
    training_mask_enabled: bool
    winners: int
    winner_weight_coordinates: int
    pooled_elements: int


ComponentMap = dict[str, OperationCounts]


def product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def collect_verified_layer_facts() -> list[LayerFacts]:
    """Recreate the seeded masks and one verified batch from the prior test."""

    reference.make_deterministic()
    weights, masks, thresholds = reference.initialize_state()
    generator = torch.Generator(device=reference.DEVICE).manual_seed(
        reference.SEED + 1
    )
    layer_input = torch.randn(
        (
            reference.BATCH_SIZE,
            reference.INPUT_CHANNELS,
            reference.INPUT_SIZE,
            reference.INPUT_SIZE,
        ),
        generator=generator,
        dtype=reference.DTYPE,
        device=reference.DEVICE,
    )

    facts: list[LayerFacts] = []
    with torch.no_grad():
        for layer in range(reference.NL):
            layer_input = reference.normalize_per_sample(layer_input)
            preliminary_output = F.conv2d(
                layer_input, weights[layer], stride=reference.STRIDES[layer]
            )
            real_output = reference.binarized_wta(
                preliminary_output,
                thresholds[layer],
                reference.WINNERS[layer],
            )
            pooled_output = reference.triangle_and_pool(
                preliminary_output, thresholds[layer], layer
            )

            output_channels = weights[layer].shape[0]
            input_dimension = weights[layer][0].numel()
            dense_weights = weights[layer].numel()
            mask_by_filter = masks[layer].reshape(output_channels, -1).sum(dim=1)
            wins_by_filter = real_output.sum(dim=(0, 2, 3))

            active_weights = int(mask_by_filter.sum().item())
            nonempty_filters = int(torch.count_nonzero(mask_by_filter).item())
            winners = int(wins_by_filter.sum().item())
            winner_weight_coordinates = int(
                torch.dot(mask_by_filter, wins_by_filter).item()
            )
            spatial_columns = (
                preliminary_output.shape[0]
                * preliminary_output.shape[2]
                * preliminary_output.shape[3]
            )

            facts.append(
                LayerFacts(
                    layer=layer + 1,
                    input_shape=tuple(layer_input.shape),
                    conv_output_shape=tuple(preliminary_output.shape),
                    pooled_output_shape=tuple(pooled_output.shape),
                    input_elements=layer_input.numel(),
                    input_dimension=input_dimension,
                    output_channels=output_channels,
                    spatial_columns=spatial_columns,
                    activations=preliminary_output.numel(),
                    dense_weights=dense_weights,
                    active_weights=active_weights,
                    nonempty_filters=nonempty_filters,
                    training_mask_enabled=(
                        reference.MASK_PROBABILITIES[layer] > 0.0
                    ),
                    winners=winners,
                    winner_weight_coordinates=winner_weight_coordinates,
                    pooled_elements=pooled_output.numel(),
                )
            )
            layer_input = pooled_output

    return facts


def normalization_cost(facts: LayerFacts) -> OperationCounts:
    """Two-pass per-sample mean/std normalization.

    For I tensor elements and B samples:
      mean sums: I-B additions; mean scaling: B divisions
      centering: I subtractions
      sum of centered squares: I multiplications + I-B additions
      variance scaling: B divisions; standard deviation: B square roots
      epsilon: B additions; final scaling: I divisions

    A fused/Welford framework kernel can use a different instruction sequence;
    this is the explicit scalar algorithm used consistently for all models.
    """

    elements = facts.input_elements
    batch = reference.BATCH_SIZE
    return OperationCounts(
        multiplications=elements,
        additions_subtractions=3 * elements - batch,
        divisions=elements + 2 * batch,
        square_roots=batch,
    )


def pre_zca_normalization_cost(facts: LayerFacts) -> OperationCounts:
    """Count the notebook's two whole-image normalizations before ZCA.

    HebbGrad_Github.ipynb normalizes the RGB batch once at lines 422 and again
    after flattening it inside the active ZCA='IMAGE' branch at lines 445-446.
    Both have the same number of elements as the layer-1 input. This component
    is attached to layer 1 only; the ZCA matrix multiplication itself remains
    outside the requested component set and is explicitly excluded.
    """

    if facts.layer != 1:
        return ZERO
    return normalization_cost(facts).scale(2)


def dense_forward_convolution_cost(facts: LayerFacts) -> OperationCounts:
    """A outputs, D products per output: A*D mult and A*(D-1) add."""

    return OperationCounts(
        multiplications=facts.activations * facts.input_dimension,
        additions_subtractions=facts.activations * (facts.input_dimension - 1),
    )


def sparse_forward_convolution_cost(facts: LayerFacts) -> OperationCounts:
    """Conditional sparse kernel using the actual nonzero mask coordinates."""

    return OperationCounts(
        multiplications=facts.spatial_columns * facts.active_weights,
        additions_subtractions=facts.spatial_columns
        * (facts.active_weights - facts.nonempty_filters),
    )


def threshold_subtraction_cost(facts: LayerFacts) -> OperationCounts:
    return OperationCounts(additions_subtractions=facts.activations)


def wta_cost(facts: LayerFacts, include_redundant_clamp: bool) -> OperationCounts:
    """Analytical top-1 plus the notebook's cutoff and positivity tests.

    Each spatial column needs C-1 comparisons to find the maximum, C comparisons
    to zero values below the cutoff, and C comparisons to binarize with > 0.
    The notebook also clamps the already-binary result to at most 50; that adds
    one redundant comparison per activation in the surrogate implementation.
    """

    comparisons = facts.spatial_columns * (facts.output_channels - 1)
    comparisons += 2 * facts.activations
    if include_redundant_clamp:
        comparisons += facts.activations
    return OperationCounts(comparisons=comparisons)


def direct_instar_dense_cost(facts: LayerFacts) -> OperationCounts:
    """Cost of the verified explicit update with dense patch coordinates.

    Winner accumulation: W*D additions
    Winner counts: W integer additions
    n*w and learning-rate scaling: 2P multiplications
    delta subtraction and weight addition: 2P additions/subtractions
    torch.nonzero winner scan: A comparisons
    """

    return OperationCounts(
        multiplications=2 * facts.dense_weights,
        additions_subtractions=(
            facts.winners * facts.input_dimension
            + facts.winners
            + 2 * facts.dense_weights
        ),
        comparisons=facts.activations,
    )


def direct_instar_sparse_cost(facts: LayerFacts) -> OperationCounts:
    """Conditional update that touches only actual active mask coordinates."""

    return OperationCounts(
        multiplications=2 * facts.active_weights,
        additions_subtractions=(
            facts.winner_weight_coordinates
            + facts.winners
            + 2 * facts.active_weights
        ),
        comparisons=facts.activations,
    )


def dense_mask_cost(facts: LayerFacts) -> OperationCounts:
    # The notebook executes this branch only when PROBAMASKS[layer] > 0.
    if not facts.training_mask_enabled:
        return ZERO
    return OperationCounts(multiplications=facts.dense_weights)


def sparse_mask_cost(_: LayerFacts) -> OperationCounts:
    # Conditional on the fixed mask being encoded in the sparse data structure.
    return ZERO


def dense_filter_normalization_cost(facts: LayerFacts) -> OperationCounts:
    """Square, reduce, sqrt, add epsilon, and divide every dense weight."""

    return OperationCounts(
        multiplications=facts.dense_weights,
        additions_subtractions=facts.dense_weights,
        divisions=facts.dense_weights,
        square_roots=facts.output_channels,
    )


def sparse_filter_normalization_cost(facts: LayerFacts) -> OperationCounts:
    """Conditional norm over stored nonzeros while retaining one norm/filter."""

    reduction_additions = facts.active_weights - facts.nonempty_filters
    epsilon_additions = facts.output_channels
    return OperationCounts(
        multiplications=facts.active_weights,
        additions_subtractions=reduction_additions + epsilon_additions,
        divisions=facts.active_weights,
        square_roots=facts.output_channels,
    )


def triangle_activation_cost(facts: LayerFacts) -> OperationCounts:
    """Threshold, channel mean, mean subtraction, and ReLU.

    The triangle path subtracts the threshold independently of the plasticity
    path, so this threshold subtraction is counted here rather than reused.
    """

    channel_mean_additions = facts.spatial_columns * (
        facts.output_channels - 1
    )
    return OperationCounts(
        additions_subtractions=(
            facts.activations  # threshold subtraction
            + channel_mean_additions
            + facts.activations  # subtract channel mean
        ),
        comparisons=facts.activations,  # ReLU clamp
        divisions=facts.spatial_columns,
    )


def average_pooling_cost(facts: LayerFacts) -> OperationCounts:
    """Each 2x2 pooled result uses three additions and one division."""

    return OperationCounts(
        additions_subtractions=3 * facts.pooled_elements,
        divisions=facts.pooled_elements,
    )


def adaptive_threshold_cost(facts: LayerFacts) -> OperationCounts:
    """Re-test activity, reduce firing rate, scale error, and update threshold."""

    channels = facts.output_channels
    return OperationCounts(
        multiplications=channels,
        additions_subtractions=(facts.activations - channels) + 2 * channels,
        comparisons=facts.activations,
        divisions=channels,
    )


def surrogate_expression_cost(facts: LayerFacts) -> OperationCounts:
    """prelim - 0.5*sum(w^2), including broadcast subtraction."""

    return OperationCounts(
        multiplications=facts.dense_weights + facts.output_channels,
        additions_subtractions=(
            facts.dense_weights - facts.output_channels + facts.activations
        ),
    )


def surrogate_loss_cost(facts: LayerFacts) -> OperationCounts:
    """sum(-0.5*y^2): two multiplies/activation and one global reduction."""

    return OperationCounts(
        multiplications=2 * facts.activations,
        additions_subtractions=facts.activations - 1,
    )


def zero_l1_branch_cost(facts: LayerFacts) -> OperationCounts:
    """Analytical eager-execution cost of ``0 * sum(abs(weight))``.

    The active notebook leaves this graph branch in place even though every
    L1PEN coefficient is zero.  For a scalar reference implementation we count
    one sign test for each forward absolute value, P-1 reduction additions,
    one scalar multiplication by zero, and one addition into the loss.  Its
    backward uses one scalar multiplication by zero plus a sign test and a
    multiplication for each absolute-value derivative.  PyTorch kernel details
    can differ, so this remains an explicitly labelled analytical estimate.
    """

    weights = facts.dense_weights
    return OperationCounts(
        multiplications=weights + 2,
        additions_subtractions=weights,
        comparisons=2 * weights,
    )


def autograd_dense_instar_gradient_cost(facts: LayerFacts) -> OperationCounts:
    """Graph-level estimate of the notebook's dense autograd backward pass.

    Backward through -0.5*(y*y): 3A multiplications and A additions for the
    two product branches. Dense convolution weight correlation: A*D
    multiplications and P*(Q-1) additions. The broadcast norm branch adds an
    A-C reduction, C scale multiplications, 2P weight multiplications, and P
    additions. Combining the convolution and norm gradient branches adds P.

    This describes scalar arithmetic, not PyTorch kernel-launch or graph-engine
    overhead. It intentionally receives no savings from binary WTA or masks.
    """

    return OperationCounts(
        multiplications=(
            3 * facts.activations
            + facts.activations * facts.input_dimension
            + facts.output_channels
            + 2 * facts.dense_weights
        ),
        additions_subtractions=(
            facts.activations
            + facts.dense_weights * (facts.spatial_columns - 1)
            + (facts.activations - facts.output_channels)
            + 2 * facts.dense_weights
        ),
    )


def sgd_step_cost(facts: LayerFacts) -> OperationCounts:
    return OperationCounts(
        multiplications=facts.dense_weights,
        additions_subtractions=facts.dense_weights,
    )


def direct_components(
    facts: LayerFacts, sparse_aware: bool
) -> ComponentMap:
    convolution: Callable[[LayerFacts], OperationCounts]
    instar: Callable[[LayerFacts], OperationCounts]
    mask: Callable[[LayerFacts], OperationCounts]
    normalization: Callable[[LayerFacts], OperationCounts]

    if sparse_aware:
        convolution = sparse_forward_convolution_cost
        instar = direct_instar_sparse_cost
        mask = sparse_mask_cost
        normalization = sparse_filter_normalization_cost
    else:
        convolution = dense_forward_convolution_cost
        instar = direct_instar_dense_cost
        mask = dense_mask_cost
        normalization = dense_filter_normalization_cost

    return {
        "Pre-ZCA input normalizations (2x)": pre_zca_normalization_cost(facts),
        "Input/per-layer normalization": normalization_cost(facts),
        "Forward convolution": convolution(facts),
        "Threshold subtraction": threshold_subtraction_cost(facts),
        "WTA/top-1 competition": wta_cost(
            facts, include_redundant_clamp=False
        ),
        "Direct Instar update": instar(facts),
        "Structural mask application": mask(facts),
        "Per-filter L2 normalization": normalization(facts),
        "Triangle activation": triangle_activation_cost(facts),
        "Average pooling": average_pooling_cost(facts),
        "Adaptive threshold update": adaptive_threshold_cost(facts),
    }


def surrogate_components(facts: LayerFacts) -> ComponentMap:
    return {
        "Pre-ZCA input normalizations (2x)": pre_zca_normalization_cost(facts),
        "Input/per-layer normalization": normalization_cost(facts),
        "Forward convolution": dense_forward_convolution_cost(facts),
        "Threshold subtraction": threshold_subtraction_cost(facts),
        "WTA/top-1 competition": wta_cost(
            facts, include_redundant_clamp=True
        ),
        "Surrogate expression": surrogate_expression_cost(facts),
        "Surrogate loss": surrogate_loss_cost(facts),
        "Zero-valued L1 branch (estimate)": zero_l1_branch_cost(facts),
        "Autograd dense Instar gradient": autograd_dense_instar_gradient_cost(
            facts
        ),
        "SGD step": sgd_step_cost(facts),
        "Structural mask application": dense_mask_cost(facts),
        "Per-filter L2 normalization": dense_filter_normalization_cost(facts),
        "Triangle activation": triangle_activation_cost(facts),
        "Average pooling": average_pooling_cost(facts),
        "Adaptive threshold update": adaptive_threshold_cost(facts),
    }


def sum_counts(counts: ComponentMap) -> OperationCounts:
    total = ZERO
    for count in counts.values():
        total = total + count
    return total


def component_factor(
    component: str,
    scope: str,
    is_surrogate: bool,
) -> int:
    if scope == "batch":
        return 1
    if scope == "first_epoch":
        all_batches = BATCHES_PER_EPOCH
        update_batches = FIRST_EPOCH_UPDATE_BATCHES
    elif scope == "full_training":
        all_batches = TOTAL_BATCHES
        update_batches = TOTAL_UPDATE_BATCHES
    else:
        raise ValueError(scope)

    if is_surrogate:
        # The notebook constructs the surrogate and calls backward during burn-in;
        # only optimizer.step is skipped.
        return update_batches if component == "SGD step" else all_batches

    # This is an explicit optimized-control-flow assumption: a local training
    # loop can skip delta construction when the notebook suppresses the step.
    return update_batches if component == "Direct Instar update" else all_batches


def scale_components(
    components: ComponentMap,
    scope: str,
    is_surrogate: bool,
) -> ComponentMap:
    return {
        name: count.scale(component_factor(name, scope, is_surrogate))
        for name, count in components.items()
    }


def format_integer(value: int) -> str:
    return f"{value:,}"


def print_operation_header() -> None:
    print(
        f"{'Component':34} {'Mult':>18} {'Add/Sub':>18} "
        f"{'Basic FLOPs':>18} {'Compare':>18} {'Divide':>14} "
        f"{'Sqrt':>10} {'Total scalar':>18}"
    )
    print("-" * 168)


def print_operation_row(name: str, count: OperationCounts) -> None:
    print(
        f"{name:34} "
        f"{format_integer(count.multiplications):>18} "
        f"{format_integer(count.additions_subtractions):>18} "
        f"{format_integer(count.basic_flops):>18} "
        f"{format_integer(count.comparisons):>18} "
        f"{format_integer(count.divisions):>14} "
        f"{format_integer(count.square_roots):>10} "
        f"{format_integer(count.total_scalar_operations):>18}"
    )


def print_component_tables(
    title: str,
    facts: list[LayerFacts],
    builder: Callable[[LayerFacts], ComponentMap],
) -> None:
    print(f"\n=== {title}: one representative post-burn-in batch ===")
    for layer in facts:
        components = builder(layer)
        print(f"\nLayer {layer.layer}")
        print_operation_header()
        for name, count in components.items():
            print_operation_row(name, count)
        print_operation_row("TOTAL", sum_counts(components))


def print_scaled_totals(
    title: str,
    facts: list[LayerFacts],
    builder: Callable[[LayerFacts], ComponentMap],
    is_surrogate: bool,
) -> None:
    print(f"\n=== {title}: scaled totals by layer ===")
    for scope, label in (
        ("batch", "One representative post-burn-in batch"),
        (
            "first_epoch",
            "Estimated first epoch: 500 batches, 201-batch optimizer burn-in",
        ),
        (
            "full_training",
            "Estimated 20 epochs: 10,000 batches, 9,799 optimizer steps",
        ),
    ):
        print(f"\n{label}")
        print_operation_header()
        network_total = ZERO
        for layer in facts:
            scaled = scale_components(
                builder(layer), scope=scope, is_surrogate=is_surrogate
            )
            layer_total = sum_counts(scaled)
            network_total = network_total + layer_total
            print_operation_row(f"Layer {layer.layer} total", layer_total)
        print_operation_row("NETWORK TOTAL", network_total)


def print_component_scalar_scaling(
    title: str,
    facts: list[LayerFacts],
    builder: Callable[[LayerFacts], ComponentMap],
    is_surrogate: bool,
) -> None:
    """Compact component-level totals for all requested training scopes."""

    print(f"\n=== {title}: total scalar operations by component and scope ===")
    print(
        f"{'Layer / component':38} {'Representative batch':>20} "
        f"{'Estimated first epoch':>24} {'Estimated 20 epochs':>26}"
    )
    print("-" * 114)
    for layer in facts:
        batch = builder(layer)
        epoch = scale_components(
            batch, scope="first_epoch", is_surrogate=is_surrogate
        )
        full = scale_components(
            batch, scope="full_training", is_surrogate=is_surrogate
        )
        for component in batch:
            print(
                f"L{layer.layer} {component:35} "
                f"{format_integer(batch[component].total_scalar_operations):>20} "
                f"{format_integer(epoch[component].total_scalar_operations):>24} "
                f"{format_integer(full[component].total_scalar_operations):>26}"
            )


def print_formulas() -> None:
    print("Hebbian CNN analytical compute-cost model")
    print(
        f"batch={reference.BATCH_SIZE}, batches/epoch={BATCHES_PER_EPOCH}, "
        f"epochs={HEBBIAN_EPOCHS}, total_batches={TOTAL_BATCHES}"
    )
    print(
        f"burn-in batches={BURN_IN_BATCHES}, first-epoch updates="
        f"{FIRST_EPOCH_UPDATE_BATCHES}, total updates={TOTAL_UPDATE_BATCHES}"
    )
    print(
        "Basic FLOPs = multiplications + additions/subtractions. Comparisons, "
        "divisions, and square roots are reported separately."
    )
    print("\nSymbols per layer:")
    print("  I = input tensor elements")
    print("  D = input channels * kernel_height * kernel_width")
    print("  C = output channels")
    print("  Q = batch * output_height * output_width (columns per channel)")
    print("  A = Q*C convolution activations")
    print("  P = C*D dense weights")
    print("  Z = actual nonzero weights in the seeded structural mask")
    print("  W = positive WTA winner entries in the deterministic reference batch")
    print("  R = pooled output elements")
    print("\nCore formulas:")
    print("  pre-ZCA normalization: two times the layer-1 normalization cost")
    print("  normalization: mult=I; add/sub=3I-B; div=I+2B; sqrt=B")
    print("  dense convolution: mult=A*D; add=A*(D-1)")
    print(
        "  conditional sparse convolution: mult=Q*Z; "
        "add=Q*sum_c(max(nnz_c-1,0))"
    )
    print("  threshold subtraction: add/sub=A")
    print("  direct dense Instar: mult=2P; add/sub=W*D+W+2P; compare=A")
    print(
        "  direct sparse Instar: mult=2Z; add/sub="
        "sum_c(wins_c*nnz_c)+W+2Z; compare=A"
    )
    print("  dense mask: mult=P; sparse-encoded fixed mask: zero arithmetic")
    print("  dense L2 norm: mult=P; add/sub=P; div=P; sqrt=C")
    print(
        "  triangle: add/sub=2A+Q*(C-1); compare=A; div=Q"
    )
    print("  2x2 average pooling: add/sub=3R; div=R")
    print(
        "  threshold update: mult=C; add/sub=A+C; compare=A; div=C"
    )
    print(
        "  zero-valued L1 graph branch (B only): mult=P+2; add/sub=P; "
        "compare=2P (analytical eager-execution estimate)"
    )
    print(
        "  graph-level dense autograd backward: mult=3A+A*D+C+2P; "
        "add/sub=A+P*(Q-1)+(A-C)+2P"
    )


def print_layer_facts(facts: list[LayerFacts]) -> None:
    print("\nDeterministic reference-batch facts and seeded mask realization")
    print(
        f"{'Layer':>5} {'Input':>20} {'Conv output':>22} {'D':>7} "
        f"{'P dense':>12} {'Z active':>12} {'Density':>11} "
        f"{'Mask/train':>11} {'W winners':>12} {'sum(wins*nnz)':>16}"
    )
    print("-" * 144)
    for layer in facts:
        density = layer.active_weights / layer.dense_weights
        print(
            f"{layer.layer:>5} {str(layer.input_shape):>20} "
            f"{str(layer.conv_output_shape):>22} "
            f"{layer.input_dimension:>7,} {layer.dense_weights:>12,} "
            f"{layer.active_weights:>12,} {density:>10.6%} "
            f"{str(layer.training_mask_enabled):>11} "
            f"{layer.winners:>12,} "
            f"{layer.winner_weight_coordinates:>16,}"
        )


def validate_model(facts: list[LayerFacts]) -> None:
    """Fail loudly if the model drifts from the verified reference setup."""

    expected_inputs = (
        (100, 3, 32, 32),
        (100, 100, 14, 14),
        (100, 196, 6, 6),
    )
    expected_outputs = (
        (100, 100, 28, 28),
        (100, 196, 12, 12),
        (100, 400, 4, 4),
    )
    expected_active_weights = (7_500, 1_771, 7_142)
    expected_winners = (78_400, 14_400, 1_600)

    assert len(facts) == reference.NL
    for index, layer in enumerate(facts):
        assert layer.input_shape == expected_inputs[index]
        assert layer.conv_output_shape == expected_outputs[index]
        assert layer.active_weights == expected_active_weights[index]
        assert layer.winners == expected_winners[index]
        assert layer.nonempty_filters == layer.output_channels
        assert layer.training_mask_enabled == (index > 0)

        dense_direct = direct_components(layer, sparse_aware=False)
        sparse_direct = direct_components(layer, sparse_aware=True)
        surrogate = surrogate_components(layer)
        for components in (dense_direct, sparse_direct, surrogate):
            for count in components.values():
                assert count.multiplications >= 0
                assert count.additions_subtractions >= 0
                assert count.comparisons >= 0
                assert count.divisions >= 0
                assert count.square_roots >= 0

        assert (
            dense_direct["Forward convolution"]
            == surrogate["Forward convolution"]
        )
        assert sum_counts(sparse_direct).total_scalar_operations <= sum_counts(
            dense_direct
        ).total_scalar_operations
        assert sum_counts(dense_direct).total_scalar_operations < sum_counts(
            surrogate
        ).total_scalar_operations


def main() -> None:
    print_formulas()
    facts = collect_verified_layer_facts()
    validate_model(facts)
    print("\nInternal consistency checks: PASS")
    print_layer_facts(facts)
    print(
        "\nCOUNTING-CONVENTION NOTE: convolution is modeled as direct dot "
        "products, normalization as the documented centered two-pass scalar "
        "algorithm, top-1 as C-1 comparisons per spatial column, and average "
        "pooling as three additions plus one division per output. PyTorch may "
        "use fused, Welford, reciprocal-multiply, or other kernel algorithms, "
        "so these are analytical scalar-operation counts rather than measured "
        "hardware instruction counts."
    )
    print(
        "EXACTNESS NOTE: tensor shapes, dense parameter counts, batch schedule, "
        "burn-in/step counts, and active notebook branches are code-derived. "
        "Seeded mask counts and WTA winners are exact only for the deterministic "
        "reference scenario; epoch/full-training winner-dependent totals are "
        "representative estimates."
    )

    direct_dense_builder = lambda layer: direct_components(
        layer, sparse_aware=False
    )
    direct_sparse_builder = lambda layer: direct_components(
        layer, sparse_aware=True
    )

    print_component_tables(
        "A1. Direct/local Hebbian - dense storage and dense conv, verified update rule",
        facts,
        direct_dense_builder,
    )
    print_scaled_totals(
        "A1. Direct/local Hebbian - dense storage and dense conv, verified update rule",
        facts,
        direct_dense_builder,
        is_surrogate=False,
    )
    print_component_scalar_scaling(
        "A1. Direct/local Hebbian - dense storage and dense conv, verified update rule",
        facts,
        direct_dense_builder,
        is_surrogate=False,
    )

    print(
        "\nCONDITIONAL SPARSITY NOTE: A2 assumes unstructured sparse storage and "
        "specialized kernels that skip masked coordinates in convolution, "
        "Instar accumulation, masking, and normalization. Dense PyTorch "
        "conv2d does not obtain these savings."
    )
    print(
        "WINNER ASSUMPTION: the deterministic reference batch had one positive "
        "WTA winner at every spatial column. Estimated epoch and 20-epoch "
        "direct-update counts assume that winner count repeats. Conditional "
        "sparse Instar estimates additionally assume the reference batch's "
        "per-filter winner/mask pairing repeats. These are not measured or "
        "exact full-training winner totals."
    )
    print_component_tables(
        "A2. Direct/local Hebbian - conditional sparse-aware model",
        facts,
        direct_sparse_builder,
    )
    print_scaled_totals(
        "A2. Direct/local Hebbian - conditional sparse-aware model",
        facts,
        direct_sparse_builder,
        is_surrogate=False,
    )
    print_component_scalar_scaling(
        "A2. Direct/local Hebbian - conditional sparse-aware model",
        facts,
        direct_sparse_builder,
        is_surrogate=False,
    )

    print(
        "\nAUTOGRAD NOTE: B is an analytical scalar-arithmetic estimate. It "
        "excludes framework graph-engine overhead and memory traffic, and it "
        "does not grant dense convolution or dense weight-gradient kernels any "
        "benefit from masks or binary WTA."
    )
    print_component_tables(
        "B. Current PyTorch surrogate-loss/autograd estimate",
        facts,
        surrogate_components,
    )
    print_scaled_totals(
        "B. Current PyTorch surrogate-loss/autograd estimate",
        facts,
        surrogate_components,
        is_surrogate=True,
    )
    print_component_scalar_scaling(
        "B. Current PyTorch surrogate-loss/autograd estimate",
        facts,
        surrogate_components,
        is_surrogate=True,
    )

    print(
        "\nSCOPE NOTE: the two active pre-ZCA normalizations are included, but "
        "the ZCA matrix multiplication is excluded because it was not one of "
        "the requested components. Also excluded: data loading, tensor-copy "
        "and memory traffic, notebook monitoring/plotting/diagnostic code, "
        "one-time initialization, feature-collection epochs, ridge fitting, "
        "hardware runtime, and energy. No conventional backpropagation "
        "baseline is included."
    )


if __name__ == "__main__":
    main()
