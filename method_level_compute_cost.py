"""Extended scope audit for Hebbian-vs-BP computational comparisons.

This file preserves the existing learning/training-core comparison and adds:
  * a fuller method-level Hebbian pipeline (20 training epochs, active
    normalizations, per-batch image ZCA, frozen train/test feature collection,
    ridge fitting, and ridge prediction/evaluation),
  * BP supervised training plus one test prediction/evaluation pass, and
  * the paper-reported approximate 5-epoch HB vs 100-epoch BP budget view.

The one-time ZCA construction is split into analytically modeled operations and
a deliberately non-exact reference range for the symmetric eigendecomposition.
The range is used only as a sensitivity check; it is not folded into the exact
operation-category tables because ``torch.symeig`` does not fix a scalar kernel.
"""

from __future__ import annotations

from typing import Callable

import backprop_compute_cost as bp_cost
import backprop_reference as bp
import hebbian_compute_cost as hebb


Counts = bp_cost.OperationCounts
ZERO = bp_cost.ZERO

TRAIN_SAMPLES = 50_000
TEST_SAMPLES = 10_000
TRAIN_BATCHES = TRAIN_SAMPLES // bp.BATCH_SIZE
TEST_BATCHES = TEST_SAMPLES // bp.BATCH_SIZE
HEBB_CONFIGURED_EPOCHS = 20
BP_PAPER_BUDGET_EPOCHS = 100
HEBB_APPROX_CONVERGENCE_EPOCHS = 5

IMAGE_VECTOR = bp.INPUT_CHANNELS * bp.INPUT_SIZE * bp.INPUT_SIZE
ZCA_BUILD_SAMPLES = 9_000
RIDGE_FEATURES = bp.CHANNELS[-1] * 2 * 2


def symmetric_eigensolver_reference_range() -> tuple[int, int]:
    """Reference basic-operation range for a dense symmetric eigensystem.

    LAPACK-style dense symmetric eigensolvers first reduce the matrix to
    tridiagonal form, solve the tridiagonal problem, and (when eigenvectors are
    requested, as in the notebook) back-transform those vectors.  Exact work
    depends on the selected driver, blocked/two-stage reduction, convergence,
    and backend.  We therefore use a deliberately broad 4/3*F^3 to 10*F^3
    sensitivity range, rather than claiming an exact count for ``symeig``.

    The lower endpoint is the standard leading-order reference for the dense
    symmetric-to-tridiagonal reduction alone; the upper endpoint allows for
    the eigensolve and eigenvector back-transformation with implementation
    variation.  This is an analytical reference range, not a formal bound.
    """

    dimension_cubed = IMAGE_VECTOR**3
    return (4 * dimension_cubed // 3, 10 * dimension_cubed)


def zca_application_cost(batch_size: int = bp.BATCH_SIZE) -> Counts:
    """Dense ``zcamat @ t.T`` for an F-by-F matrix and F-by-B input."""

    dimension = IMAGE_VECTOR
    outputs = batch_size * dimension
    return Counts(
        multiplications=outputs * dimension,
        additions_subtractions=outputs * (dimension - 1),
    )


def arbitrary_normalization_cost(elements: int, samples: int) -> Counts:
    """The same centered two-pass scalar convention as hebbian_compute_cost."""

    return Counts(
        multiplications=elements,
        additions_subtractions=3 * elements - samples,
        divisions=elements + 2 * samples,
        square_roots=samples,
    )


def known_zca_construction_cost() -> Counts:
    """Known one-time arithmetic around the unmodeled eigendecomposition.

    The active notebook normalizes 9,000 images, forms a dense F-by-F covariance,
    adds a dense diagonal-regularizer tensor, invokes ``torch.symeig``, converts
    eigenvalues to inverse square roots, reconstructs ZCA with two dense F-cube
    matrix multiplies, makes three matrix-vector projections for the unused
    spatial-filter object, and spatially centers its 3x3 filters.

    ``torch.symeig`` itself is intentionally absent: its backend algorithm and
    reduction strategy are not fixed by the notebook, so a scalar total would
    be implementation-dependent rather than defensible as an exact code count.
    """

    dimension = IMAGE_VECTOR
    samples = ZCA_BUILD_SAMPLES
    matrix_entries = dimension * dimension

    normalization = arbitrary_normalization_cost(
        elements=samples * dimension,
        samples=samples,
    )
    covariance = Counts(
        multiplications=matrix_entries * samples,
        additions_subtractions=matrix_entries * (samples - 1),
        divisions=matrix_entries,
    )
    regularizer = Counts(
        multiplications=dimension,
        # PyTorch adds the materialized dense diagonal matrix elementwise.
        additions_subtractions=matrix_entries,
    )
    eigenvalue_postprocess = Counts(
        # One positivity test per eigenvalue plus the torch.all reduction.
        comparisons=(2 * dimension - 1),
        divisions=dimension,
        square_roots=dimension,
    )
    reconstruct = Counts(
        multiplications=2 * dimension**3,
        additions_subtractions=2 * matrix_entries * (dimension - 1),
    )
    spatial_filter_projections = Counts(
        multiplications=bp.INPUT_CHANNELS * matrix_entries,
        additions_subtractions=bp.INPUT_CHANNELS
        * dimension
        * (dimension - 1),
    )
    spatial_filter_centering = Counts(
        # Nine spatial filters, each with 32x32 values.
        additions_subtractions=(9 * (1_024 - 1)) + (9 * 1_024),
        divisions=9,
    )
    return (
        normalization
        + covariance
        + regularizer
        + eigenvalue_postprocess
        + reconstruct
        + spatial_filter_projections
        + spatial_filter_centering
    )


def quadrant_feature_cost() -> Counts:
    """Notebook's four quadrant means for L1, L2, and L3 per feature batch."""

    additions = 0
    divisions = 0
    # Tensors are Bx100x14x14, Bx196x6x6, and Bx400x2x2.
    for channels, side in ((100, 14), (196, 6), (400, 2)):
        quadrant_side = side // 2
        values_per_mean = quadrant_side * quadrant_side
        means = bp.BATCH_SIZE * channels * 4
        additions += means * (values_per_mean - 1)
        divisions += means
    return Counts(additions_subtractions=additions, divisions=divisions)


def feature_collection_batch(
    facts: list[hebb.LayerFacts],
    sparse_aware: bool,
    current_surrogate_notebook: bool,
) -> Counts:
    """One frozen-weight feature batch, following the active notebook branches."""

    total = zca_application_cost()
    for layer in facts:
        total += bp_cost.from_hebbian(hebb.pre_zca_normalization_cost(layer))
        total += bp_cost.from_hebbian(hebb.normalization_cost(layer))
        convolution = (
            hebb.sparse_forward_convolution_cost(layer)
            if sparse_aware
            else hebb.dense_forward_convolution_cost(layer)
        )
        total += bp_cost.from_hebbian(convolution)
        total += bp_cost.from_hebbian(hebb.threshold_subtraction_cost(layer))
        # The notebook's frozen passes still execute top-k, binarization, and
        # the redundant clamp, although ridge features use the triangle output.
        total += bp_cost.from_hebbian(
            hebb.wta_cost(layer, include_redundant_clamp=True)
        )
        if current_surrogate_notebook:
            # yforgrad is constructed outside ``if UNLAB`` even though no loss
            # or backward pass follows in the train/test feature epochs.
            total += bp_cost.from_hebbian(hebb.surrogate_expression_cost(layer))
        total += bp_cost.from_hebbian(hebb.triangle_activation_cost(layer))
        total += bp_cost.from_hebbian(hebb.average_pooling_cost(layer))
        # Weights are frozen, but the notebook continues adapting thresholds.
        total += bp_cost.from_hebbian(hebb.adaptive_threshold_cost(layer))
    return total + quadrant_feature_cost()


def hebbian_training_total(
    facts: list[hebb.LayerFacts],
    builder: Callable[[hebb.LayerFacts], hebb.ComponentMap],
    is_surrogate: bool,
    epochs: int,
) -> Counts:
    """Training with existing normalizations plus the previously omitted ZCA."""

    batches = TRAIN_BATCHES * epochs
    total = zca_application_cost().scale(batches)
    for layer in facts:
        for name, count in builder(layer).items():
            update_batches = max(0, batches - hebb.BURN_IN_BATCHES)
            if is_surrogate and name == "SGD step":
                factor = update_batches
            elif not is_surrogate and name == "Direct Instar update":
                factor = update_batches
            else:
                factor = batches
            total += bp_cost.from_hebbian(count.scale(factor))
    return total


def ridge_fit_cost() -> Counts:
    """Primal dense Ridge fit using an explicitly assumed Cholesky solve.

    The notebook calls ``linear_model.Ridge()`` with library defaults.  We model
    alpha=1, fit_intercept=True and the dense N>F primal path: center X/Y, form
    full X^T X and X^T Y, add alpha to the diagonal, Cholesky-factor the F-by-F
    SPD matrix, solve for ten right-hand sides, and recover the intercept.
    Current scikit-learn resolves ``solver='auto'`` to Cholesky for a dense
    NumPy input such as the notebook's ``np.vstack`` result.  The repository,
    however, pins no scikit-learn/SciPy version; older dispatch, a numerical
    fallback to SVD, BLAS/LAPACK kernels, and dtype/copy behavior are not fixed.
    This is therefore the most defensible reference path, not an exact library
    kernel count.
    """

    samples = TRAIN_SAMPLES
    features = RIDGE_FEATURES
    classes = bp.NUM_CLASSES

    centering = Counts(
        additions_subtractions=(features + classes) * (2 * samples - 1),
        divisions=features + classes,
    )
    gram = Counts(
        multiplications=samples * features * features,
        additions_subtractions=(samples - 1) * features * features,
    )
    cross = Counts(
        multiplications=samples * features * classes,
        additions_subtractions=(samples - 1) * features * classes,
    )
    diagonal_regularizer = Counts(additions_subtractions=features)

    cholesky_dot_terms = features * (features - 1) * (features + 1) // 6
    cholesky = Counts(
        multiplications=cholesky_dot_terms,
        additions_subtractions=cholesky_dot_terms,
        divisions=features * (features - 1) // 2,
        square_roots=features,
    )
    triangular_solves = Counts(
        multiplications=classes * features * (features - 1),
        additions_subtractions=classes * features * (features - 1),
        divisions=2 * classes * features,
    )
    intercept = Counts(
        multiplications=features * classes,
        additions_subtractions=features * classes,
    )
    return (
        centering
        + gram
        + cross
        + diagonal_regularizer
        + cholesky
        + triangular_solves
        + intercept
    )


def ridge_prediction_evaluation_cost() -> Counts:
    samples = TEST_SAMPLES
    features = RIDGE_FEATURES
    classes = bp.NUM_CLASSES
    return Counts(
        multiplications=samples * features * classes,
        additions_subtractions=(samples * features * classes) + (samples - 1),
        # argmax comparisons plus equality tests against all labels.
        comparisons=samples * (classes - 1) + samples,
        divisions=1,
    )


def bp_test_prediction_evaluation_cost(facts: list[bp_cost.ConvFacts]) -> Counts:
    per_batch = ZERO
    for layer in facts:
        per_batch += bp_cost.forward_convolution(layer)
        per_batch += bp_cost.relu_forward(layer)
        per_batch += bp_cost.average_pool_forward(layer)

    classifier = bp_cost.classifier_components()["Linear classifier forward"]
    per_batch += classifier
    total = per_batch.scale(TEST_BATCHES)
    return total + Counts(
        additions_subtractions=TEST_SAMPLES - 1,
        comparisons=TEST_SAMPLES * (bp.NUM_CLASSES - 1) + TEST_SAMPLES,
        divisions=1,
    )


def format_count(value: int) -> str:
    return f"{value:,}"


def print_count_table(rows: list[tuple[str, Counts]]) -> None:
    print(
        f"{'Component':44} {'Mult':>19} {'Add/Sub':>19} {'Compare':>16} "
        f"{'Divide':>15} {'Sqrt':>13} {'Trans':>10} {'Total':>21}"
    )
    print("-" * 166)
    for name, count in rows:
        print(
            f"{name:44} {format_count(count.multiplications):>19} "
            f"{format_count(count.additions_subtractions):>19} "
            f"{format_count(count.comparisons):>16} "
            f"{format_count(count.divisions):>15} "
            f"{format_count(count.square_roots):>13} "
            f"{format_count(count.transcendental):>10} "
            f"{format_count(count.total_scalar_operations):>21}"
        )


def core_total(
    facts: list[hebb.LayerFacts],
    builder: Callable[[hebb.LayerFacts], hebb.ComponentMap],
    is_surrogate: bool,
    epochs: int | None,
) -> Counts:
    batches = None if epochs is None else TRAIN_BATCHES * epochs
    return bp_cost.fair_hebbian_total(
        facts, builder, is_surrogate, batches=batches
    )


def print_learning_core_views(
    facts: list[hebb.LayerFacts], bp_batch: Counts
) -> None:
    models = [
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
    ]

    print(
        "\n=== Learning/training-core computational comparison "
        "(preprocessing excluded) ==="
    )
    print(
        f"{'Method':32} {'One batch':>22} {'Equal 20 epochs':>23} "
        f"{'Equal 100 epochs':>24}"
    )
    print("-" * 106)
    for name, builder, surrogate in models:
        batch = core_total(facts, builder, surrogate, None)
        twenty = core_total(facts, builder, surrogate, 20)
        hundred = core_total(facts, builder, surrogate, 100)
        print(
            f"{name:32} {format_count(batch.total_scalar_operations):>22} "
            f"{format_count(twenty.total_scalar_operations):>23} "
            f"{format_count(hundred.total_scalar_operations):>24}"
        )
    print(
        f"{'Standard BP':32} {format_count(bp_batch.total_scalar_operations):>22} "
        f"{format_count(bp_cost.bp_epoch_total(bp_batch, 20).total_scalar_operations):>23} "
        f"{format_count(bp_cost.bp_epoch_total(bp_batch, 100).total_scalar_operations):>24}"
    )

    print("\nConfigured budget view: Hebbian 20 epochs vs BP 100 epochs")
    for name, builder, surrogate in models:
        count = core_total(facts, builder, surrogate, 20)
        print(f"  {name}: {format_count(count.total_scalar_operations)}")
    print(
        "  Standard BP: "
        f"{format_count(bp_cost.bp_epoch_total(bp_batch, 100).total_scalar_operations)}"
    )

    print(
        "\nPaper-reported approximate convergence-budget scenario: "
        "Hebbian 5 epochs vs BP 100 epochs"
    )
    for name, builder, surrogate in models:
        count = core_total(
            facts, builder, surrogate, HEBB_APPROX_CONVERGENCE_EPOCHS
        )
        print(f"  {name} / 5 epochs: {format_count(count.total_scalar_operations)}")
    print(
        "  Standard BP / 100 epochs: "
        f"{format_count(bp_cost.bp_epoch_total(bp_batch, 100).total_scalar_operations)}"
    )
    print(
        "  This applies the paper's approximate convergence budgets; it is not "
        "an equal-accuracy theorem or a measured runtime comparison."
    )


def print_method_level_view(
    hebbian_facts: list[hebb.LayerFacts],
    bp_facts: list[bp_cost.ConvFacts],
    bp_batch: Counts,
) -> None:
    ridge_fit = ridge_fit_cost()
    ridge_eval = ridge_prediction_evaluation_cost()
    zca_build = known_zca_construction_cost()
    bp_test = bp_test_prediction_evaluation_cost(bp_facts)

    models = [
        (
            "A1 direct dense Hebbian",
            lambda layer: hebb.direct_components(layer, sparse_aware=False),
            False,
            False,
            False,
        ),
        (
            "A2 sparse-aware Hebbian",
            lambda layer: hebb.direct_components(layer, sparse_aware=True),
            False,
            True,
            False,
        ),
        (
            "B surrogate/autograd Hebbian",
            hebb.surrogate_components,
            True,
            False,
            True,
        ),
    ]

    print("\n=== Frozen feature-pass costs (one batch) ===")
    feature_batches: dict[str, Counts] = {}
    for name, _, _, sparse, surrogate_notebook in models:
        count = feature_collection_batch(
            hebbian_facts,
            sparse_aware=sparse,
            current_surrogate_notebook=surrogate_notebook,
        )
        feature_batches[name] = count
        print(f"{name:32} {format_count(count.total_scalar_operations):>22}")

    print("\n=== One-time ZCA construction: modeled part only ===")
    print_count_table([("Known arithmetic excluding symeig", zca_build)])
    eig_low, eig_high = symmetric_eigensolver_reference_range()
    print(
        "NON-EXACT EIGENSOLVER SENSITIVITY RANGE: "
        f"{format_count(eig_low)} to {format_count(eig_high)} basic operations "
        "(4/3*F^3 to 10*F^3 for F=3,072). The notebook delegates the dense "
        "symmetric eigensystem to torch.symeig; driver, reduction strategy, "
        "back-transformation, and backend are not fixed by the source."
    )

    print("\n=== Ridge classifier analytical model ===")
    print_count_table(
        [
            ("Ridge fit (primal Cholesky assumption)", ridge_fit),
            ("Ridge test prediction/evaluation", ridge_eval),
        ]
    )
    print(
        "Ridge solver status: dense solver='auto' selects Cholesky in current "
        "scikit-learn, but this repository has no version lock. The displayed "
        "fit is a Cholesky reference model, not an exact sklearn count. SVD "
        "would still be O(N*F^2 + F^3); iterative solvers depend on iterations."
    )

    print(
        "\n=== Fuller method-level modeled subtotal ===\n"
        "Hebbian rows include known ZCA construction arithmetic, 20 training "
        "epochs, one frozen 50k training-feature pass, one frozen 10k test-"
        "feature pass, ridge fit, and ridge test prediction/evaluation. The "
        "unmodeled eigendecomposition is not included. BP rows include the "
        "supervised training path and one 10k test prediction/evaluation pass."
    )
    print(f"{'Method / budget':38} {'Modeled subtotal':>25}")
    print("-" * 66)

    method_totals: dict[str, Counts] = {}
    for name, builder, surrogate, _, _ in models:
        training = hebbian_training_total(
            hebbian_facts,
            builder,
            is_surrogate=surrogate,
            epochs=HEBB_CONFIGURED_EPOCHS,
        )
        train_features = feature_batches[name].scale(TRAIN_BATCHES)
        test_features = feature_batches[name].scale(TEST_BATCHES)
        total = (
            zca_build
            + training
            + train_features
            + test_features
            + ridge_fit
            + ridge_eval
        )
        method_totals[name] = total
        print(f"{(name + ' / 20 epochs'):38} {format_count(total.total_scalar_operations):>25}")

    for epochs in (20, 100):
        total = (
            bp_cost.bp_epoch_total(bp_batch, epochs)
            + bp_test
        )
        method_totals[f"BP{epochs}"] = total
        print(
            f"{('Standard BP / ' + str(epochs) + ' epochs'):38} "
            f"{format_count(total.total_scalar_operations):>25}"
        )

    print("\nHebbian subtotal sensitivity after adding the eigensolver range")
    print(f"{'Method / 20 epochs':38} {'Low reference':>25} {'High reference':>25} {'Increase':>18}")
    print("-" * 110)
    for name, _, _, _, _ in models:
        subtotal = method_totals[name].total_scalar_operations
        low_total = subtotal + eig_low
        high_total = subtotal + eig_high
        print(
            f"{name:38} {format_count(low_total):>25} "
            f"{format_count(high_total):>25} "
            f"{(100.0 * eig_low / subtotal):6.3f}%.."
            f"{(100.0 * eig_high / subtotal):6.3f}%"
        )

    print("\nMethod-level Hebbian breakdowns")
    for name, builder, surrogate, _, _ in models:
        training = hebbian_training_total(
            hebbian_facts, builder, surrogate, HEBB_CONFIGURED_EPOCHS
        )
        train_features = feature_batches[name].scale(TRAIN_BATCHES)
        test_features = feature_batches[name].scale(TEST_BATCHES)
        rows = [
            ("Known one-time ZCA construction", zca_build),
            ("20-epoch Hebbian training", training),
            ("Frozen training-feature pass", train_features),
            ("Frozen test-feature pass", test_features),
            ("Ridge fit", ridge_fit),
            ("Ridge prediction/evaluation", ridge_eval),
            ("MODELED SUBTOTAL", method_totals[name]),
        ]
        print(f"\n{name}")
        print_count_table(rows)

    for epochs in (20, 100):
        training = bp_cost.bp_epoch_total(bp_batch, epochs)
        rows = [
            (f"{epochs}-epoch supervised BP training", training),
            ("BP test prediction/evaluation", bp_test),
            ("MODELED SUBTOTAL", method_totals[f"BP{epochs}"]),
        ]
        print(f"\nStandard BP / {epochs} epochs")
        print_count_table(rows)

    assert all(
        method_totals[name].total_scalar_operations
        > hebbian_training_total(
            hebbian_facts, builder, surrogate, HEBB_CONFIGURED_EPOCHS
        ).total_scalar_operations
        for name, builder, surrogate, _, _ in models
    )
    assert method_totals["BP100"].total_scalar_operations > method_totals[
        "BP20"
    ].total_scalar_operations


def validate(
    hebbian_facts: list[hebb.LayerFacts],
    bp_facts: list[bp_cost.ConvFacts],
    bp_batch: Counts,
) -> None:
    assert TRAIN_BATCHES == 500
    assert TEST_BATCHES == 100
    assert IMAGE_VECTOR == 3_072
    assert RIDGE_FEATURES == 1_600
    assert zca_application_cost() == Counts(
        multiplications=943_718_400,
        additions_subtractions=943_411_200,
    )
    quadrants = quadrant_feature_cost()
    assert quadrants.additions_subtractions == 2_547_200
    assert quadrants.divisions == 278_400
    assert len(hebbian_facts) == 3
    assert len(bp_facts) == 3
    assert bp_batch.total_scalar_operations == 24_430_482_934

    fit = ridge_fit_cost()
    prediction = ridge_prediction_evaluation_cost()
    assert fit.total_scalar_operations > prediction.total_scalar_operations
    assert known_zca_construction_cost().total_scalar_operations > 0
    eig_low, eig_high = symmetric_eigensolver_reference_range()
    assert eig_low == 38_654_705_664
    assert eig_high == 289_910_292_480
    assert eig_low < eig_high


def main() -> None:
    hebbian_facts = hebb.collect_verified_layer_facts()
    hebb.validate_model(hebbian_facts)
    bp_facts = bp_cost.collect_conv_facts()
    bp_batch = bp_cost.bp_batch_total(bp_facts)
    bp_cost.validate(bp_facts, hebbian_facts, bp_batch)
    validate(hebbian_facts, bp_facts, bp_batch)

    print("Final comparison-scope extension")
    print("Internal consistency checks: PASS")
    print(
        "Source attribution: Adam is stated directly in Appendix A of "
        "arXiv:2212.04614. No BP implementation is explicitly cited."
    )
    print_learning_core_views(hebbian_facts, bp_batch)
    print_method_level_view(hebbian_facts, bp_facts, bp_batch)

    print(
        "\nSOURCE CLASSES:\n"
        "  Paper-direct: 100/196/400 channels, BP linear layer, Adam, batch "
        "100, LR 1e-5, StepLR step size 1, 20/100 epochs, and no BN/dropout.\n"
        "  Code-direct: Hebbian kernels/pooling, ZCA=IMAGE, three active "
        "normalization stages before Conv1, layer normalizations, 9,000-image "
        "ZCA build, frozen 50k/10k feature passes, Ridge(), and feature size "
        "1,600.\n"
        "  Cited-implementation inference: none for BP; the equal BP geometry "
        "borrows the explicitly cited Hebbian implementation only as a fairness "
        "assumption.\n"
        "  Analytical assumptions: ReLU/cross-entropy/bias choices, Adam "
        "defaults, StepLR gamma=0.95, scalar kernel formulas, sparse A2 kernels, "
        "and a primal-Cholesky Ridge reference path."
    )
    print(
        "SCOPE LIMITS: data loading/augmentation, tensor copies and transfers, "
        "memory traffic, weight/mask initialization, notebook diagnostic moving "
        "averages and periodic statistics, and plotting remain outside the "
        "modeled subtotal. The ZCA eigensolver is shown only as a separate "
        "non-exact sensitivity range."
    )


if __name__ == "__main__":
    main()
