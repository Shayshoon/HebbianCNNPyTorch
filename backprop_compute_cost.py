"""Analytical BP cost model and fair comparison with the Hebbian models.

The BP architecture and assumptions are defined in backprop_reference.py.  The
counting conventions match hebbian_compute_cost.py: scalar multiplications,
additions/subtractions, comparisons, divisions, square roots, basic FLOPs, and
total scalar operations are kept separate.  Cross-entropy necessarily adds
exp/log operations, so this script reports them in one extra ``Transcendental``
column rather than hiding them inside FLOPs.

The direct comparison excludes all input/per-layer preprocessing from both
sides.  In particular, the Hebbian pre-ZCA and per-layer normalizations are
removed from the comparison totals, and the ZCA matrix multiply remains out of
scope.  This is the least assumption-heavy comparison because arXiv:2212.04614
does not specify BP preprocessing and explicitly associates ZCA with its HB
setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import backprop_reference as bp
import hebbian_compute_cost as hebb


BATCHES_PER_EPOCH = 50_000 // bp.BATCH_SIZE
EPOCH_SCOPES = (1, 20, 100)
PREPROCESS_COMPONENTS = {
    "Pre-ZCA input normalizations (2x)",
    "Input/per-layer normalization",
}


@dataclass(frozen=True)
class OperationCounts:
    multiplications: int = 0
    additions_subtractions: int = 0
    comparisons: int = 0
    divisions: int = 0
    square_roots: int = 0
    transcendental: int = 0

    def __add__(self, other: "OperationCounts") -> "OperationCounts":
        return OperationCounts(
            self.multiplications + other.multiplications,
            self.additions_subtractions + other.additions_subtractions,
            self.comparisons + other.comparisons,
            self.divisions + other.divisions,
            self.square_roots + other.square_roots,
            self.transcendental + other.transcendental,
        )

    def scale(self, factor: int) -> "OperationCounts":
        return OperationCounts(
            self.multiplications * factor,
            self.additions_subtractions * factor,
            self.comparisons * factor,
            self.divisions * factor,
            self.square_roots * factor,
            self.transcendental * factor,
        )

    @property
    def basic_flops(self) -> int:
        return self.multiplications + self.additions_subtractions

    @property
    def total_scalar_operations(self) -> int:
        return (
            self.basic_flops
            + self.comparisons
            + self.divisions
            + self.square_roots
            + self.transcendental
        )


ZERO = OperationCounts()


@dataclass(frozen=True)
class ConvFacts:
    layer: int
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    pooled_shape: tuple[int, int, int, int]
    input_elements: int
    output_activations: int
    pooled_elements: int
    input_dimension: int
    spatial_columns: int
    weights: int
    output_channels: int


ComponentMap = dict[str, OperationCounts]


def product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def collect_conv_facts() -> list[ConvFacts]:
    height = bp.INPUT_SIZE
    width = bp.INPUT_SIZE
    input_channels = bp.INPUT_CHANNELS
    facts: list[ConvFacts] = []
    for layer, (output_channels, kernel, stride) in enumerate(
        zip(bp.CHANNELS, bp.KERNEL_SIZES, bp.STRIDES), start=1
    ):
        output_height = (height - kernel) // stride + 1
        output_width = (width - kernel) // stride + 1
        pooled_height = output_height // 2
        pooled_width = output_width // 2
        input_shape = (bp.BATCH_SIZE, input_channels, height, width)
        output_shape = (
            bp.BATCH_SIZE,
            output_channels,
            output_height,
            output_width,
        )
        pooled_shape = (
            bp.BATCH_SIZE,
            output_channels,
            pooled_height,
            pooled_width,
        )
        input_dimension = input_channels * kernel * kernel
        spatial_columns = bp.BATCH_SIZE * output_height * output_width
        facts.append(
            ConvFacts(
                layer=layer,
                input_shape=input_shape,
                output_shape=output_shape,
                pooled_shape=pooled_shape,
                input_elements=product(input_shape),
                output_activations=product(output_shape),
                pooled_elements=product(pooled_shape),
                input_dimension=input_dimension,
                spatial_columns=spatial_columns,
                weights=output_channels * input_dimension,
                output_channels=output_channels,
            )
        )
        height = pooled_height
        width = pooled_width
        input_channels = output_channels
    return facts


def forward_convolution(facts: ConvFacts) -> OperationCounts:
    return OperationCounts(
        multiplications=facts.output_activations * facts.input_dimension,
        additions_subtractions=facts.output_activations
        * (facts.input_dimension - 1),
    )


def relu_forward(facts: ConvFacts) -> OperationCounts:
    return OperationCounts(comparisons=facts.output_activations)


def average_pool_forward(facts: ConvFacts) -> OperationCounts:
    return OperationCounts(
        additions_subtractions=3 * facts.pooled_elements,
        divisions=facts.pooled_elements,
    )


def average_pool_backward(facts: ConvFacts) -> OperationCounts:
    # Non-overlapping 2x2 pooling distributes each output gradient to 4 inputs.
    return OperationCounts(divisions=4 * facts.pooled_elements)


def relu_backward(facts: ConvFacts) -> OperationCounts:
    # Re-test/retain the positive mask and multiply the upstream gradient.
    return OperationCounts(
        multiplications=facts.output_activations,
        comparisons=facts.output_activations,
    )


def convolution_weight_gradient(facts: ConvFacts) -> OperationCounts:
    # Each weight correlates over Q=batch*output_height*output_width samples.
    return OperationCounts(
        multiplications=facts.weights * facts.spatial_columns,
        additions_subtractions=facts.weights
        * (facts.spatial_columns - 1),
    )


def convolution_input_gradient(facts: ConvFacts) -> OperationCounts:
    # Conv1's raw input does not require a gradient.  For later layers, every
    # valid-convolution input element receives at least one contribution.
    if facts.layer == 1:
        return ZERO
    products = facts.output_activations * facts.input_dimension
    return OperationCounts(
        multiplications=products,
        additions_subtractions=products - facts.input_elements,
    )


def adam_parameter_update(parameters: int, include_group_state: bool) -> OperationCounts:
    """Reference Adam arithmetic with explicit bias-corrected moments.

    Per parameter: m update (2 mult, 1 add), v update (3 mult, 1 add),
    m/v bias correction (2 div), sqrt(v_hat), epsilon add, quotient (1 div),
    learning-rate scale (1 mult), and parameter subtraction (1 add).
    Two group-level multiplications and two subtractions recurrently maintain
    beta1**t, beta2**t, and their correction factors once per optimizer step.
    """

    return OperationCounts(
        multiplications=6 * parameters + (2 if include_group_state else 0),
        additions_subtractions=4 * parameters
        + (2 if include_group_state else 0),
        divisions=3 * parameters,
        square_roots=parameters,
    )


def convolution_components(facts: ConvFacts) -> ComponentMap:
    return {
        "Forward convolution": forward_convolution(facts),
        "ReLU forward": relu_forward(facts),
        "Average pooling forward": average_pool_forward(facts),
        "Average pooling backward": average_pool_backward(facts),
        "ReLU backward": relu_backward(facts),
        "Convolution weight gradient": convolution_weight_gradient(facts),
        "Convolution input gradient": convolution_input_gradient(facts),
        "Adam parameter update": adam_parameter_update(
            facts.weights, include_group_state=False
        ),
    }


def classifier_components() -> ComponentMap:
    batch = bp.BATCH_SIZE
    classes = bp.NUM_CLASSES
    features = bp.CHANNELS[-1] * 2 * 2
    weight_parameters = features * classes
    bias_parameters = classes
    parameters = weight_parameters + bias_parameters
    logits = batch * classes

    # Bias-enabled linear forward: F multiplies and F-1 reductions plus one
    # bias addition for every logit, hence F additions per logit.
    linear_forward = OperationCounts(
        multiplications=logits * features,
        additions_subtractions=logits * features,
    )

    # Stable mean cross-entropy/log-sum-exp:
    # max, shift, exp, reduction, log, add max, subtract target, batch mean.
    cross_entropy_forward = OperationCounts(
        additions_subtractions=(2 * logits + 2 * batch - 1),
        comparisons=batch * (classes - 1),
        divisions=1,
        transcendental=batch * (classes + 1),
    )

    # dL/dlogits=(softmax-one_hot)/B; exp values and sums are reused.  One
    # division forms each probability and a second applies mean-loss scaling.
    cross_entropy_backward = OperationCounts(
        additions_subtractions=batch,
        divisions=2 * logits,
    )

    # Weight/bias gradients and the gradient propagated to the 1600 features.
    linear_backward = OperationCounts(
        multiplications=(2 * batch * classes * features),
        additions_subtractions=(
            weight_parameters * (batch - 1)
            + bias_parameters * (batch - 1)
            + batch * features * (classes - 1)
        ),
    )

    return {
        "Linear classifier forward": linear_forward,
        "Mean cross-entropy forward": cross_entropy_forward,
        "Cross-entropy logits gradient": cross_entropy_backward,
        "Linear classifier backward": linear_backward,
        "Adam parameter update": adam_parameter_update(
            parameters, include_group_state=True
        ),
    }


def sum_counts(counts: ComponentMap) -> OperationCounts:
    total = ZERO
    for count in counts.values():
        total = total + count
    return total


def bp_per_batch_by_layer(
    facts: list[ConvFacts],
) -> list[tuple[str, ComponentMap]]:
    result = [
        (f"Conv {layer.layer}", convolution_components(layer)) for layer in facts
    ]
    result.append(("Classifier/loss", classifier_components()))
    return result


def bp_batch_total(facts: list[ConvFacts]) -> OperationCounts:
    total = ZERO
    for _, components in bp_per_batch_by_layer(facts):
        total = total + sum_counts(components)
    return total


def bp_epoch_total(batch_total: OperationCounts, epochs: int) -> OperationCounts:
    batches = BATCHES_PER_EPOCH * epochs
    # StepLR updates one group learning rate once per epoch.
    return batch_total.scale(batches) + OperationCounts(
        multiplications=epochs
    )


def from_hebbian(count: hebb.OperationCounts) -> OperationCounts:
    return OperationCounts(
        multiplications=count.multiplications,
        additions_subtractions=count.additions_subtractions,
        comparisons=count.comparisons,
        divisions=count.divisions,
        square_roots=count.square_roots,
    )


def fair_hebbian_total(
    facts: list[hebb.LayerFacts],
    builder: Callable[[hebb.LayerFacts], hebb.ComponentMap],
    is_surrogate: bool,
    batches: int | None,
) -> OperationCounts:
    """Hebbian total with preprocessing removed for the shared fair scope.

    ``batches=None`` means one representative post-burn-in update batch.
    For multi-batch scopes, the notebook's one-time 201-batch burn-in is used.
    """

    total = ZERO
    for layer in facts:
        for name, count in builder(layer).items():
            if name in PREPROCESS_COMPONENTS:
                continue
            if batches is None:
                factor = 1
            else:
                update_batches = max(0, batches - hebb.BURN_IN_BATCHES)
                if is_surrogate and name == "SGD step":
                    factor = update_batches
                elif not is_surrogate and name == "Direct Instar update":
                    factor = update_batches
                else:
                    factor = batches
            total = total + from_hebbian(count.scale(factor))
    return total


def format_integer(value: int) -> str:
    return f"{value:,}"


def print_header(name_width: int = 36) -> None:
    print(
        f"{'Component':{name_width}} {'Mult':>18} {'Add/Sub':>18} "
        f"{'Basic FLOPs':>18} {'Compare':>16} {'Divide':>14} "
        f"{'Sqrt':>12} {'Transcendental':>16} {'Total scalar':>20}"
    )
    print("-" * (name_width + 136))


def print_row(name: str, count: OperationCounts, name_width: int = 36) -> None:
    print(
        f"{name:{name_width}} "
        f"{format_integer(count.multiplications):>18} "
        f"{format_integer(count.additions_subtractions):>18} "
        f"{format_integer(count.basic_flops):>18} "
        f"{format_integer(count.comparisons):>16} "
        f"{format_integer(count.divisions):>14} "
        f"{format_integer(count.square_roots):>12} "
        f"{format_integer(count.transcendental):>16} "
        f"{format_integer(count.total_scalar_operations):>20}"
    )


def print_formulas() -> None:
    print("BP analytical compute-cost model")
    print(
        f"batch={bp.BATCH_SIZE}, batches/epoch={BATCHES_PER_EPOCH}, "
        "full CIFAR-10 training set"
    )
    print("Basic FLOPs = multiplications + additions/subtractions.")
    print("Formulas:")
    print("  conv forward: mult=A*D; add=A*(D-1)")
    print("  ReLU forward: compare=A")
    print("  2x2 average pool forward: add=3R; div=R")
    print("  pool backward: div=4R")
    print("  ReLU backward: mult=A; compare=A")
    print("  conv weight grad: mult=P*Q; add=P*(Q-1)")
    print("  conv input grad: mult=A*D; add=A*D-I (not computed for Conv1)")
    print("  linear forward with bias: mult=B*K*F; add=B*K*F")
    print(
        "  linear backward: weight/input-gradient mult=2*B*K*F; "
        "explicit reduction additions"
    )
    print(
        "  stable mean cross-entropy: max/shift/exp/sum/log/target subtraction; "
        "dlogits=(softmax-one_hot)/B"
    )
    print("  Adam per parameter: mult=6; add/sub=4; div=3; sqrt=1")
    print("  StepLR: one scalar multiplication per epoch")


def print_architecture(facts: list[ConvFacts]) -> None:
    print("\nEqual-architecture BP facts")
    print(
        f"{'Layer':<9} {'Input':>23} {'Conv output':>25} "
        f"{'Pooled output':>25} {'D':>8} {'Weights':>12}"
    )
    print("-" * 112)
    for layer in facts:
        print(
            f"Conv {layer.layer:<4} {str(layer.input_shape):>23} "
            f"{str(layer.output_shape):>25} {str(layer.pooled_shape):>25} "
            f"{layer.input_dimension:>8,} {layer.weights:>12,}"
        )
    print(
        "Classifier: flatten 400x2x2=1,600 -> 10, with 16,000 weights "
        "and 10 biases"
    )


def print_bp_tables(facts: list[ConvFacts]) -> OperationCounts:
    print("\n=== Standard BP: one training batch by layer/component ===")
    batch_total = ZERO
    for name, components in bp_per_batch_by_layer(facts):
        print(f"\n{name}")
        print_header()
        for component, count in components.items():
            print_row(component, count)
        layer_total = sum_counts(components)
        print_row("TOTAL", layer_total)
        batch_total = batch_total + layer_total

    print("\n=== Standard BP: aggregate scopes ===")
    print_header()
    print_row("One batch", batch_total)
    for epochs in EPOCH_SCOPES:
        print_row(
            f"{epochs} epoch{'s' if epochs != 1 else ''}",
            bp_epoch_total(batch_total, epochs),
        )
    return batch_total


def print_comparison(
    hebbian_facts: list[hebb.LayerFacts], bp_batch: OperationCounts
) -> None:
    models: list[
        tuple[
            str,
            Callable[[hebb.LayerFacts], hebb.ComponentMap] | None,
            bool,
        ]
    ] = [
        (
            "A1 direct dense Hebbian",
            lambda layer: hebb.direct_components(layer, sparse_aware=False),
            False,
        ),
        (
            "A2 sparse-aware Hebbian",
            lambda layer: hebb.direct_components(layer, sparse_aware=True),
            False,
        ),
        ("B surrogate/autograd Hebbian", hebb.surrogate_components, True),
        ("Standard BP", None, False),
    ]

    print(
        "\n=== Learning/training-core computational comparison: preprocessing "
        "excluded from all methods ==="
    )
    print(
        f"{'Method':32} {'Representative batch':>24} {'1 epoch':>24} "
        f"{'20 equal epochs':>24} {'100 equal epochs':>24}"
    )
    print("-" * 132)
    totals: dict[str, dict[int | str, OperationCounts]] = {}
    for name, builder, is_surrogate in models:
        if builder is None:
            batch_count = bp_batch
            scoped = {
                epoch: bp_epoch_total(bp_batch, epoch) for epoch in EPOCH_SCOPES
            }
        else:
            batch_count = fair_hebbian_total(
                hebbian_facts, builder, is_surrogate, batches=None
            )
            scoped = {
                epoch: fair_hebbian_total(
                    hebbian_facts,
                    builder,
                    is_surrogate,
                    batches=BATCHES_PER_EPOCH * epoch,
                )
                for epoch in EPOCH_SCOPES
            }
        totals[name] = {"batch": batch_count, **scoped}
        print(
            f"{name:32} "
            f"{format_integer(batch_count.total_scalar_operations):>24} "
            f"{format_integer(scoped[1].total_scalar_operations):>24} "
            f"{format_integer(scoped[20].total_scalar_operations):>24} "
            f"{format_integer(scoped[100].total_scalar_operations):>24}"
        )

    print("\n=== Paper training-budget comparison: Hebbian 20 epochs vs BP 100 ===")
    print(f"{'Method / budget':44} {'Total scalar operations':>28}")
    print("-" * 74)
    for name, _, _ in models[:3]:
        print(
            f"{(name + ' / 20 epochs'):44} "
            f"{format_integer(totals[name][20].total_scalar_operations):>28}"
        )
    bp_name = "Standard BP"
    print(
        f"{(bp_name + ' / 100 epochs'):44} "
        f"{format_integer(totals[bp_name][100].total_scalar_operations):>28}"
    )


def validate(
    facts: list[ConvFacts],
    hebbian_facts: list[hebb.LayerFacts],
    batch_total: OperationCounts,
) -> None:
    assert [layer.output_shape for layer in facts] == [
        (100, 100, 28, 28),
        (100, 196, 12, 12),
        (100, 400, 4, 4),
    ]
    assert [layer.pooled_shape for layer in facts] == [
        (100, 100, 14, 14),
        (100, 196, 6, 6),
        (100, 400, 2, 2),
    ]
    assert sum(layer.weights for layer in facts) == 889_500
    assert sum(layer.weights for layer in facts) + 16_000 + 10 == 905_510
    assert batch_total.total_scalar_operations > batch_total.basic_flops

    a1 = fair_hebbian_total(
        hebbian_facts,
        lambda layer: hebb.direct_components(layer, sparse_aware=False),
        False,
        batches=None,
    )
    a2 = fair_hebbian_total(
        hebbian_facts,
        lambda layer: hebb.direct_components(layer, sparse_aware=True),
        False,
        batches=None,
    )
    surrogate = fair_hebbian_total(
        hebbian_facts,
        hebb.surrogate_components,
        True,
        batches=None,
    )
    assert a2.total_scalar_operations < a1.total_scalar_operations
    assert a1.total_scalar_operations < surrogate.total_scalar_operations


def main() -> None:
    print_formulas()
    facts = collect_conv_facts()
    print_architecture(facts)
    batch_total = print_bp_tables(facts)

    # Recreate the deterministic masks/winner scenario used by the corrected
    # Hebbian cost model instead of duplicating its constants here.
    hebbian_facts = hebb.collect_verified_layer_facts()
    hebb.validate_model(hebbian_facts)
    validate(facts, hebbian_facts, batch_total)
    print("\nInternal consistency checks: PASS")
    print_comparison(hebbian_facts, batch_total)

    print(
        "\nEXACTNESS: channel counts, batch size, Adam/LR/StepLR step-size, and "
        "20/100 epochs are paper-derived. Tensor shapes and parameter counts are "
        "exact consequences of the explicitly assumed equal architecture. "
        "Scalar counts are exact under the printed reference formulas, not "
        "measured PyTorch kernel instructions."
    )
    print(
        "ASSUMPTIONS: kernels/pooling copied from the active Hebbian notebook; "
        "ReLU; mean cross-entropy; conv bias off; classifier bias on; Adam "
        "defaults; StepLR gamma=0.95; dense BP training; no raw-input gradient."
    )
    print(
        "SPARSITY: the paper's 95% BP pruning is post-training global-magnitude "
        "pruning with accuracy remeasurement, not sparse training. It therefore "
        "does not reduce the dense BP training totals above. A2 savings remain "
        "conditional on specialized sparse storage/kernels."
    )
    print(
        "SCOPE: full 50,000-example CIFAR-10 set, 500 batches/epoch. Shared "
        "preprocessing, ZCA, data augmentation/loading, memory traffic, kernel "
        "launches, and the Hebbian post-training ridge fit are excluded."
    )


if __name__ == "__main__":
    main()
