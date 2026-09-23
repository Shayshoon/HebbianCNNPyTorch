"""Deterministic reference model for the BP side of the Hebbian comparison.

Facts stated directly by arXiv:2212.04614:
  * CIFAR-10, three convolutional layers with 100, 196, and 400 channels.
  * A final linear classifier for BP.
  * Vanilla network: no batch normalization, dropout, or other training tricks.
  * Adam, batch size 100, initial learning rate 1e-5, StepLR step size 1.
  * Experiments use either 20 or 100 epochs.

Adam is not inferred from an older implementation: Appendix A explicitly says
that the authors "use the Adam optimizer" for BP.  The paper does not cite a
BP implementation.  Its implementation citations concern HB, FA/DFA, DDTP,
and PC.

The paper does not specify convolution kernels/strides, pooling, activations,
classifier input geometry, classifier/conv biases, loss, Adam betas/epsilon,
StepLR gamma, or BP preprocessing.  For the equal-architecture comparison we
therefore make the following minimum explicit assumptions:
  * Reuse HebbGrad_Github.ipynb's active 5x5, 3x3, 3x3 valid convolutions,
    stride 1, and 2x2 stride-2 average pooling after every convolution.
  * Use ReLU after every convolution (the conventional vanilla-CNN choice).
  * Match the Hebbian convolutions by omitting convolution biases.
  * Flatten the final 400x2x2 representation into a bias-enabled 10-way linear
    classifier and train with mean cross-entropy.
  * Use PyTorch Adam defaults beta1=0.9, beta2=0.999, epsilon=1e-8, no weight
    decay/AMSGrad, and assume StepLR gamma=0.95.  Gamma affects the learning-rate
    values but not the per-parameter operation-count formulas.
  * Accept already-loaded float CIFAR-10 tensors.  ZCA and all input/per-layer
    normalization are excluded from both sides of the fair compute table.

This file is a reference/smoke test, not a full dataset-training program.  It
does not modify the original notebooks or download training data.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 221204614
INPUT_CHANNELS = 3
INPUT_SIZE = 32
NUM_CLASSES = 10
BATCH_SIZE = 100

CHANNELS = (100, 196, 400)
KERNEL_SIZES = (5, 3, 3)
STRIDES = (1, 1, 1)
POOL_KERNELS = (2, 2, 2)
POOL_STRIDES = (2, 2, 2)

LEARNING_RATE = 1e-5
TRAINING_EPOCH_OPTIONS = (20, 100)
STEP_SIZE = 1
STEP_GAMMA_ASSUMPTION = 0.95
ADAM_BETAS_ASSUMPTION = (0.9, 0.999)
ADAM_EPSILON_ASSUMPTION = 1e-8


def make_deterministic() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


class EqualArchitectureBP(nn.Module):
    """Three-convolution CIFAR-10 BP model under the assumptions above."""

    def __init__(self) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList(
            (
                nn.Conv2d(3, 100, 5, stride=1, padding=0, bias=False),
                nn.Conv2d(100, 196, 3, stride=1, padding=0, bias=False),
                nn.Conv2d(196, 400, 3, stride=1, padding=0, bias=False),
            )
        )
        self.classifier = nn.Linear(400 * 2 * 2, NUM_CLASSES, bias=True)

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        shapes: list[tuple[int, ...]] = []
        for convolution in self.convolutions:
            x = convolution(x)
            shapes.append(tuple(x.shape))
            x = F.relu(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            shapes.append(tuple(x.shape))
        return x, shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.forward_features(x)
        return self.classifier(torch.flatten(features, start_dim=1))


def build_training_objects(
    model: nn.Module,
) -> tuple[torch.optim.Adam, torch.optim.lr_scheduler.StepLR]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS_ASSUMPTION,
        eps=ADAM_EPSILON_ASSUMPTION,
        weight_decay=0.0,
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=STEP_GAMMA_ASSUMPTION,
    )
    return optimizer, scheduler


def run_smoke_test() -> None:
    make_deterministic()
    model = EqualArchitectureBP()
    optimizer, scheduler = build_training_objects(model)

    smoke_batch_size = 2
    generator = torch.Generator().manual_seed(SEED + 1)
    inputs = torch.randn(
        smoke_batch_size,
        INPUT_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
        generator=generator,
    )
    targets = torch.tensor((0, 1), dtype=torch.long)

    expected_feature_shapes = [
        (2, 100, 28, 28),
        (2, 100, 14, 14),
        (2, 196, 12, 12),
        (2, 196, 6, 6),
        (2, 400, 4, 4),
        (2, 400, 2, 2),
    ]
    features, feature_shapes = model.forward_features(inputs)
    assert feature_shapes == expected_feature_shapes
    assert tuple(features.shape) == (2, 400, 2, 2)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    assert tuple(logits.shape) == (2, NUM_CLASSES)
    loss = F.cross_entropy(logits, targets, reduction="mean")
    assert torch.isfinite(loss)
    loss.backward()

    parameters = list(model.parameters())
    assert len(parameters) == 5  # 3 conv weights + classifier weight and bias.
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in parameters)

    parameters_before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    scheduler.step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters)
    )

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert trainable_parameters == 905_510

    print("Deterministic BP reference smoke test: PASS")
    print(f"seed={SEED} smoke_batch={smoke_batch_size}")
    print(f"feature_shapes={feature_shapes}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"trainable_parameters={trainable_parameters:,}")
    print(f"loss={float(loss.detach()):.9f}")
    print(
        "paper-direct settings: optimizer=Adam batch=100 initial_lr=1e-5 "
        "StepLR(step_size=1) epochs=(20,100)"
    )
    print(
        "assumptions: ReLU, Hebbian kernel/pool geometry, conv_bias=False, "
        "linear_bias=True, mean cross-entropy, Adam defaults, StepLR gamma=0.95, "
        "shared preprocessing excluded"
    )


if __name__ == "__main__":
    run_smoke_test()
