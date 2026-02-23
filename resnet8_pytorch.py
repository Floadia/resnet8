"""Simple PyTorch implementation of a ResNet-8 style CIFAR-10 model.

Layout:
- Conv 3x3 (16)
- Residual stack x2 (16)
- Residual stack x2 (32, stride=2, projection shortcut)
- Residual stack x2 (64, stride=2, projection shortcut)
- Global average pooling + linear classifier
"""

from __future__ import annotations

import argparse

import torch
from torch import nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding used in CIFAR-style ResNet."""

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        else:
            self.downsample_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None and self.downsample_bn is not None:
            identity = self.downsample(identity)
            identity = self.downsample_bn(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet8(nn.Module):
    """Small CIFAR-10 ResNet-8 model."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self._in_channels = 16

        self.stem = _conv3x3(3, self._in_channels)
        self.stem_bn = nn.BatchNorm2d(self._in_channels)
        self.stem_relu = nn.ReLU(inplace=True)

        self.block1 = self._make_layer(16, num_blocks=2, stride=1)
        self.block2 = self._make_layer(32, num_blocks=2, stride=2)
        self.block3 = self._make_layer(64, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * _BasicBlock.expansion, num_classes)

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []

        for block_stride in strides:
            layers.append(_BasicBlock(self._in_channels, out_channels, block_stride))
            self._in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Supports input in NCHW or NHWC format."""

        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D tensor")

        # Accept both: NCHW (N, C, H, W) and NHWC (N, H, W, C)
        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self.stem(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet8(num_classes: int = 10) -> ResNet8:
    """Factory for the simple ResNet-8 model."""

    return ResNet8(num_classes=num_classes)


def load_model_from_pt(model_path: str, num_classes: int = 10) -> nn.Module:
    """Load a model from .pt checkpoint/TorchScript and set eval mode."""

    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        return model
    except RuntimeError:
        pass

    loaded = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(loaded, nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        if "model" in loaded and isinstance(loaded["model"], nn.Module):
            model = loaded["model"]
        elif "state_dict" in loaded:
            model = resnet8(num_classes=num_classes)
            missing, unexpected = model.load_state_dict(
                loaded["state_dict"], strict=False
            )
            if missing or unexpected:
                raise ValueError(
                    "state_dict keys do not match ResNet8 definition. "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected one of: "
                "TorchScript module, nn.Module, dict with 'model', "
                "or dict with 'state_dict'."
            )
    else:
        raise ValueError(f"Unsupported object type in checkpoint: {type(loaded)!r}")

    model.eval()
    return model


def _looks_like_probability(outputs: torch.Tensor) -> bool:
    if outputs.dim() != 2:
        return False
    if torch.any(outputs < 0) or torch.any(outputs > 1):
        return False
    sums = outputs.sum(dim=1)
    ones = torch.ones_like(sums)
    return torch.allclose(sums, ones, atol=1e-3, rtol=1e-3)


def run_dummy_inference(model: nn.Module, batch_size: int, layout: str) -> None:
    """Run one inference pass with random CIFAR-like input and print summary."""

    if layout == "nhwc":
        dummy = torch.randint(0, 256, (batch_size, 32, 32, 3), dtype=torch.uint8)
    else:
        dummy = torch.randint(0, 256, (batch_size, 3, 32, 32), dtype=torch.uint8)

    input_tensor = dummy.to(torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)

    if not isinstance(outputs, torch.Tensor):
        raise TypeError(f"Model output is not a tensor: {type(outputs)!r}")

    scores = outputs if _looks_like_probability(outputs) else torch.softmax(outputs, -1)
    top1_scores, top1_classes = torch.max(scores, dim=1)

    print(f"Input shape:  {tuple(input_tensor.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")
    print(f"Top-1 class (sample 0): {int(top1_classes[0])}")
    print(f"Top-1 score (sample 0): {float(top1_scores[0]):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple ResNet8 definition + quick .pt inference smoke test"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8.pt",
        help="Path to .pt file for inference (default: models/resnet8.pt)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Dummy batch size (default: 2)"
    )
    parser.add_argument(
        "--layout",
        choices=["nhwc", "nchw"],
        default="nhwc",
        help="Dummy input layout (default: nhwc)",
    )
    parser.add_argument(
        "--use-simple-model",
        action="store_true",
        help="Use simple ResNet8 definition without loading --model",
    )
    args = parser.parse_args()

    if args.use_simple_model:
        model = resnet8()
        print("Using simple ResNet8 definition (randomly initialized).")
    else:
        model = load_model_from_pt(args.model)
        print(f"Loaded model from: {args.model}")

    run_dummy_inference(model=model, batch_size=args.batch_size, layout=args.layout)
