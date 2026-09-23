"""Single-batch equivalence test for the active HebbGrad_Github.ipynb setup.

Method A reproduces the notebook's surrogate-loss/autograd implementation.
Method B applies Grossberg's Instar rule directly, without autograd:

    delta_w = sum_m y_m * (x_m - w)

The test starts with a deterministic float32 tensor having the same shape as a
ZCA-preprocessed CIFAR-10 batch. ZCA is common preprocessing and is deliberately
outside this update-equivalence test. All learning-relevant operations from the
active three-layer configuration are retained: per-sample normalization,
binarized 1-WTA plasticity, triangle activations, average pooling, SGD learning
rates, structural masks, per-filter L2 normalization, and adaptive thresholds.
The tested batch represents a post-burn-in batch on which the notebook performs
an optimizer step; testing an initial burn-in batch would produce no weight
update and would therefore not establish the requested equivalence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


SEED = 210701729
DTYPE = torch.float32
DEVICE = torch.device("cpu")

# Active configuration from HebbGrad_Github.ipynb.
NL = 3
BATCH_SIZE = 100
INPUT_CHANNELS = 3
INPUT_SIZE = 32
STRIDES = (1, 1, 1)
POOL_STRIDES = (2, 2, 2)
POOL_DIAMETERS = (2, 2, 2)
KERNEL_SIZES = (5, 3, 3)
CHANNELS = (100, 196, 400)
WINNERS = (1, 1, 1)
MASK_PROBABILITIES = (0.0, 0.99, 0.99)
LEARNING_RATES = (
    0.001 / BATCH_SIZE * 10.0,
    0.001 / BATCH_SIZE * 10.0,
    10.0 * 0.001 / BATCH_SIZE * 10.0,
)
THRESHOLD_RATE = 3.0
TARGET_RATES = tuple(WINNERS[i] / CHANNELS[i] for i in range(NL))
EPSILON = 1e-10

# Float32 accumulation order differs slightly between conv2d backward and the
# direct index_add accumulation. These tolerances test numerical, not bitwise,
# equivalence.
ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-5


@dataclass
class LayerResult:
    layer: int
    weight_shape: tuple[int, ...]
    conv_output_shape: tuple[int, ...]
    wta_nonzero: int
    mask_nonzero: int
    mask_total: int
    weight_max_abs_difference: float
    weight_mean_abs_difference: float
    threshold_max_abs_difference: float
    threshold_mean_abs_difference: float
    weights_match: bool
    thresholds_match: bool


def make_deterministic() -> None:
    """Fix every random source used by this test."""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Match the notebook's normalization over channels and spatial positions."""

    x = x - torch.mean(x, dim=(1, 2, 3), keepdim=True)
    return x / (EPSILON + torch.std(x, dim=(1, 2, 3), keepdim=True))


def normalize_filters(weight: torch.Tensor) -> torch.Tensor:
    """Match the notebook's per-output-filter Euclidean normalization."""

    norms = torch.sqrt(torch.sum(weight * weight, dim=(1, 2, 3), keepdim=True))
    return weight / (EPSILON + norms)


def initialize_state() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Create weights, structural masks, and thresholds as in the notebook."""

    weights: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []

    for layer in range(NL):
        input_channels = INPUT_CHANNELS if layer == 0 else CHANNELS[layer - 1]
        shape = (
            CHANNELS[layer],
            input_channels,
            KERNEL_SIZES[layer],
            KERNEL_SIZES[layer],
        )

        weight = torch.randn(shape, dtype=DTYPE, device=DEVICE) * 0.01
        mask = (
            torch.rand(shape, dtype=DTYPE, device=DEVICE)
            > MASK_PROBABILITIES[layer]
        ).to(DTYPE)
        if layer == 0:
            mask.fill_(1.0)

        weight = normalize_filters(weight * mask)
        threshold = torch.zeros(
            (1, CHANNELS[layer], 1, 1), dtype=DTYPE, device=DEVICE
        )

        weights.append(weight)
        masks.append(mask)
        thresholds.append(threshold)

    return weights, masks, thresholds


def binarized_wta(
    preliminary_output: torch.Tensor,
    threshold: torch.Tensor,
    winners: int,
) -> torch.Tensor:
    """Reproduce the notebook's thresholding and k-WTA behavior exactly."""

    real_output = preliminary_output.detach() - threshold
    top_values = torch.topk(real_output, winners, dim=1, largest=True)[0]
    cutoff = top_values[:, -1, :, :][:, None, :, :]
    real_output[real_output < cutoff] = 0.0
    return (real_output > 0.0).to(DTYPE)


def triangle_and_pool(
    preliminary_output: torch.Tensor,
    threshold: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Produce the active triangle representation and 2x2 average pooling."""

    output = preliminary_output.detach() - threshold
    output = output - torch.mean(output, dim=1, keepdim=True)
    output = torch.clamp(output, min=0.0)
    return F.avg_pool2d(
        output,
        POOL_DIAMETERS[layer],
        stride=POOL_STRIDES[layer],
    ).detach()


def update_threshold(
    threshold: torch.Tensor,
    real_output: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Apply the notebook's homeostatic threshold update."""

    firing_rate = torch.mean(
        (real_output > 0.0).to(DTYPE), dim=(0, 2, 3), keepdim=True
    )
    return threshold + THRESHOLD_RATE * (firing_rate - TARGET_RATES[layer])


def surrogate_autograd_update(
    layer_input: torch.Tensor,
    initial_weight: torch.Tensor,
    mask: torch.Tensor,
    threshold: torch.Tensor,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Method A: reproduce the notebook's surrogate-loss update."""

    weight = initial_weight.detach().clone().requires_grad_(True)
    preliminary_output = F.conv2d(
        layer_input, weight, stride=STRIDES[layer]
    )
    real_output = binarized_wta(
        preliminary_output, threshold, WINNERS[layer]
    )

    squared_norm = torch.sum(weight * weight, dim=(1, 2, 3))
    output_for_gradient = (
        preliminary_output - 0.5 * squared_norm[None, :, None, None]
    )

    # This is intentionally the same value-substitution technique used by the
    # original notebook: the numerical value becomes the WTA output while the
    # computational graph remains the Instar surrogate expression above.
    output_for_gradient.data = real_output.data
    loss = torch.sum(-0.5 * output_for_gradient * output_for_gradient)
    loss.backward()

    with torch.no_grad():
        updated_weight = weight - LEARNING_RATES[layer] * weight.grad
        updated_weight = updated_weight * mask
        updated_weight = normalize_filters(updated_weight)
        updated_threshold = update_threshold(threshold, real_output, layer)

    return (
        updated_weight.detach(),
        updated_threshold.detach(),
        preliminary_output.detach(),
        real_output.detach(),
    )


def explicit_instar_delta(
    layer_input: torch.Tensor,
    weight: torch.Tensor,
    real_output: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Compute sum_m y_m * (x_m - w) directly, without autograd."""

    kernel_size = KERNEL_SIZES[layer]
    patches = F.unfold(
        layer_input,
        kernel_size=kernel_size,
        stride=STRIDES[layer],
    )
    batch_size, patch_size, positions = patches.shape
    output_channels = weight.shape[0]
    flattened_output = real_output.reshape(batch_size, output_channels, positions)

    # The active configuration uses binarized 1-WTA. Gathering only winners is
    # algebraically identical to multiplying every patch by a dense 0/1 tensor.
    winner_locations = torch.nonzero(flattened_output, as_tuple=False)
    batch_indices = winner_locations[:, 0]
    channel_indices = winner_locations[:, 1]
    position_indices = winner_locations[:, 2]

    selected_patches = patches[batch_indices, :, position_indices]
    accumulated_inputs = torch.zeros(
        (output_channels, patch_size), dtype=DTYPE, device=DEVICE
    )
    accumulated_inputs.index_add_(0, channel_indices, selected_patches)

    winner_counts = torch.bincount(
        channel_indices, minlength=output_channels
    ).to(DTYPE)[:, None]
    flattened_weight = weight.reshape(output_channels, patch_size)
    delta = accumulated_inputs - winner_counts * flattened_weight
    return delta.reshape_as(weight)


def direct_instar_update(
    layer_input: torch.Tensor,
    initial_weight: torch.Tensor,
    mask: torch.Tensor,
    threshold: torch.Tensor,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Method B: apply the explicit Instar update without autograd."""

    with torch.no_grad():
        weight = initial_weight.detach().clone()
        preliminary_output = F.conv2d(
            layer_input, weight, stride=STRIDES[layer]
        )
        real_output = binarized_wta(
            preliminary_output, threshold, WINNERS[layer]
        )
        delta = explicit_instar_delta(
            layer_input, weight, real_output, layer
        )

        updated_weight = weight + LEARNING_RATES[layer] * delta
        updated_weight = updated_weight * mask
        updated_weight = normalize_filters(updated_weight)
        updated_threshold = update_threshold(threshold, real_output, layer)

    return (
        updated_weight,
        updated_threshold,
        preliminary_output,
        real_output,
    )


def tensor_differences(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[float, float]:
    difference = torch.abs(first - second)
    return float(torch.max(difference)), float(torch.mean(difference))


def tensors_match(first: torch.Tensor, second: torch.Tensor) -> bool:
    return bool(
        torch.allclose(
            first,
            second,
            atol=ABSOLUTE_TOLERANCE,
            rtol=RELATIVE_TOLERANCE,
        )
    )


def run_test() -> list[LayerResult]:
    make_deterministic()
    initial_weights, masks, initial_thresholds = initialize_state()

    # This tensor represents the output of the common ZCA preprocessing stage.
    input_generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    common_input = torch.randn(
        (BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        generator=input_generator,
        dtype=DTYPE,
        device=DEVICE,
    )
    input_autograd = common_input.clone()
    input_direct = common_input.clone()

    results: list[LayerResult] = []

    for layer in range(NL):
        input_autograd = normalize_per_sample(input_autograd)
        input_direct = normalize_per_sample(input_direct)

        if not torch.equal(input_autograd, input_direct):
            raise AssertionError(f"Layer {layer + 1} inputs are not identical")

        (
            weight_autograd,
            threshold_autograd,
            preliminary_autograd,
            real_autograd,
        ) = surrogate_autograd_update(
            input_autograd,
            initial_weights[layer],
            masks[layer],
            initial_thresholds[layer],
            layer,
        )
        (
            weight_direct,
            threshold_direct,
            preliminary_direct,
            real_direct,
        ) = direct_instar_update(
            input_direct,
            initial_weights[layer],
            masks[layer],
            initial_thresholds[layer],
            layer,
        )

        if not torch.equal(preliminary_autograd, preliminary_direct):
            raise AssertionError(
                f"Layer {layer + 1} preliminary outputs are not identical"
            )
        if not torch.equal(real_autograd, real_direct):
            raise AssertionError(f"Layer {layer + 1} WTA outputs are not identical")

        weight_max, weight_mean = tensor_differences(
            weight_autograd, weight_direct
        )
        threshold_max, threshold_mean = tensor_differences(
            threshold_autograd, threshold_direct
        )

        results.append(
            LayerResult(
                layer=layer + 1,
                weight_shape=tuple(weight_autograd.shape),
                conv_output_shape=tuple(preliminary_autograd.shape),
                wta_nonzero=int(torch.count_nonzero(real_autograd)),
                mask_nonzero=int(torch.count_nonzero(masks[layer])),
                mask_total=masks[layer].numel(),
                weight_max_abs_difference=weight_max,
                weight_mean_abs_difference=weight_mean,
                threshold_max_abs_difference=threshold_max,
                threshold_mean_abs_difference=threshold_mean,
                weights_match=tensors_match(weight_autograd, weight_direct),
                thresholds_match=tensors_match(
                    threshold_autograd, threshold_direct
                ),
            )
        )

        # The notebook computes triangle activations from the pre-update
        # convolution output and pre-update threshold, then passes those pooled
        # activations to the next layer.
        input_autograd = triangle_and_pool(
            preliminary_autograd, initial_thresholds[layer], layer
        )
        input_direct = triangle_and_pool(
            preliminary_direct, initial_thresholds[layer], layer
        )

    return results


def print_results(results: list[LayerResult]) -> None:
    print("Direct Grossberg Instar equivalence test")
    print(f"seed={SEED} device={DEVICE} dtype={DTYPE} batch_size={BATCH_SIZE}")
    print(
        "tolerance: "
        f"atol={ABSOLUTE_TOLERANCE:.1e}, rtol={RELATIVE_TOLERANCE:.1e}"
    )

    for result in results:
        density = result.mask_nonzero / result.mask_total
        print(f"\nLayer {result.layer}")
        print(f"  weight shape: {result.weight_shape}")
        print(f"  convolution output shape: {result.conv_output_shape}")
        print(f"  active WTA entries: {result.wta_nonzero}")
        print(
            "  mask nonzero: "
            f"{result.mask_nonzero}/{result.mask_total} ({density:.6%})"
        )
        print(
            "  weight difference: "
            f"max={result.weight_max_abs_difference:.9e}, "
            f"mean={result.weight_mean_abs_difference:.9e}, "
            f"match={result.weights_match}"
        )
        print(
            "  threshold difference: "
            f"max={result.threshold_max_abs_difference:.9e}, "
            f"mean={result.threshold_mean_abs_difference:.9e}, "
            f"match={result.thresholds_match}"
        )

    overall_match = all(
        result.weights_match and result.thresholds_match for result in results
    )
    print(f"\nOVERALL MATCH: {overall_match}")
    if not overall_match:
        raise AssertionError("Direct Instar and surrogate-autograd updates differ")


if __name__ == "__main__":
    print_results(run_test())
