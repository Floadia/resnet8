#!/usr/bin/env python3
"""Extract intermediate activations from a ResNet8 PyTorch checkpoint.

Examples:
  # list available layer names
  python scripts/get_resnet8_intermediate.py --list-layers

  # print one layer tensor (from CIFAR-10 test sample 0)
  python scripts/get_resnet8_intermediate.py \
    --model models/resnet8.pt \
    --layer "model_1/conv2d_1/BiasAdd"

  # save a layer activation to NPZ
  python scripts/get_resnet8_intermediate.py \
    --model models/resnet8.pt \
    --layer-index 2 \
    --save logs/resnet8_layer_2.npz
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def load_cifar10_test_sample(data_dir: str, index: int) -> tuple[np.ndarray, int]:
    test_path = os.path.join(data_dir, "test_batch")
    with open(test_path, "rb") as f:
        test_data = pickle.load(f, encoding="bytes")

    images = test_data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    labels = np.array(test_data[b"labels"], dtype=np.int64)

    if index < 0 or index >= len(images):
        raise ValueError(f"--index out of range: {index}")

    sample = images[index : index + 1]  # (1, 32, 32, 3)
    return sample, int(labels[index])


def load_model(path: Path | str, device: torch.device) -> torch.nn.Module:
    try:
        scripted = torch.jit.load(path, map_location=device)
        return scripted
    except (RuntimeError, OSError, ValueError):
        pass

    candidate = torch.load(path, map_location=device, weights_only=False)

    # TorchScript path used for quantized/int8 artifacts
    if isinstance(candidate, torch.jit.ScriptModule):
        return candidate

    if isinstance(candidate, torch.nn.Module):
        return candidate

    if isinstance(candidate, dict):
        model_obj = candidate.get("model")
        if isinstance(model_obj, torch.nn.Module):
            return model_obj

        if isinstance(candidate.get("state_dict"), dict):
            raise ValueError(
                "Loaded checkpoint contains only 'state_dict'. "
                "Hooking requires a full nn.Module object. "
                "Save/export with a full model (as used in models/resnet8.pt)."
            )

    raise TypeError("Unsupported checkpoint format for this script")


def get_model_layers(model: torch.nn.Module) -> List[str]:
    return [name for name, _ in model.named_modules() if name]


def normalize_input(sample: np.ndarray, device: torch.device) -> torch.Tensor:
    # Models in this repo are N,H,W,C (32x32x3)
    tensor = torch.from_numpy(sample).to(device)
    return tensor


def collect_named_tensors(value: Any, prefix: str) -> Iterable[Tuple[str, np.ndarray]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value.detach().cpu().numpy()
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from collect_named_tensors(item, f"{prefix}_{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from collect_named_tensors(item, f"{prefix}_{idx}")
    else:
        return


def run_with_hook(model: torch.nn.Module, layer_name: str, x: torch.Tensor) -> Any:
    captured = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        captured["value"] = output

    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break

    if handle is None:
        raise KeyError(f"Layer not found: {layer_name}")

    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        handle.remove()

    return captured.get("value")


def print_tensor_summary(name: str, arr: np.ndarray, max_items: int = 16) -> None:
    print(f"\n{name}")
    print(f"  shape={arr.shape} dtype={arr.dtype} min={arr.min():.6f} "
          f"max={arr.max():.6f} mean={arr.mean():.6f} std={arr.std():.6f}")
    flat = arr.reshape(-1)
    print(f"  first_{max_items}: {flat[:max_items]}")


def save_activation(values: Iterable[Tuple[str, np.ndarray]], path: Path) -> None:
    data = {name: arr for name, arr in values}

    # use npz when multiple arrays are present
    if len(data) == 1:
        only_arr = next(iter(data.values()))
        np.save(path, only_arr)
    else:
        np.savez(path, **data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one intermediate activation from a ResNet8 .pth/.pt file"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8.pth",
        help="Path to checkpoint (supports .pt/.pth with a full nn.Module)",
    )
    parser.add_argument(
        "--layer",
        help="Exact layer name from `model.named_modules()` to capture",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        help="Alternative: capture layer by index from printed layer list",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print all layer names and exit",
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py",
        help="CIFAR-10 test folder (test_batch) when using a real sample",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Test sample index from CIFAR-10 batch",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random input instead of CIFAR-10 sample",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"),
        help="Run on cpu/gpu",
    )
    parser.add_argument(
        "--save",
        help="Optional output path to save activation (.npy/.npz)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model = load_model(args.model, device).to(device).eval()
    layers = get_model_layers(model)

    if args.list_layers:
        print("Available layers:")
        for idx, name in enumerate(layers):
            print(f"[{idx}] {name}")
        if not args.layer and args.layer_index is None:
            return

    target_layer = args.layer
    if target_layer is None and args.layer_index is not None:
        if args.layer_index < 0 or args.layer_index >= len(layers):
            raise IndexError(f"layer index out of range: {args.layer_index}")
        target_layer = layers[args.layer_index]

    if not target_layer:
        raise ValueError(
            "No --layer provided. Use --layer <name> or --layer-index <i> "
            "(or pass --list-layers to see choices)."
        )

    if args.random:
        sample = np.random.random((1, 32, 32, 3)).astype(np.float32)
        label = None
    else:
        sample, label = load_cifar10_test_sample(args.data_dir, args.index)

    x = normalize_input(sample, device)

    value = run_with_hook(model, target_layer, x)

    named = list(collect_named_tensors(value, target_layer))
    if not named:
        raise RuntimeError(f"Captured value is not tensor-like for layer '{target_layer}'")

    for name, arr in named:
        print_tensor_summary(name, arr)

    if args.save:
        save_activation(named, Path(args.save))
        print(f"\nSaved activation to: {args.save}")

    if not args.random:
        print(f"\nSample label: {label}")


if __name__ == "__main__":
    main()
