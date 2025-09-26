"""Export 4-bit quantized models to the BitNet ternary multiplane format."""

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM  # type: ignore

    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TRANSFORMERS = False
    AutoModelForCausalLM = None  # type: ignore
    print("Warning: transformers not installed")


@dataclass
class QuantizationConfig:
    bits: int = 4
    group_size: int = 128
    symmetric: bool = True


def quantize_tensor_to_4bit(
    tensor: torch.Tensor, config: QuantizationConfig
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Quantize a tensor into a 4-bit representation."""

    original_shape = tensor.shape
    tensor_flat = tensor.flatten()
    num_elements = tensor_flat.numel()
    num_groups = (num_elements + config.group_size - 1) // config.group_size

    quantized = torch.zeros_like(tensor_flat, dtype=torch.int8)
    scales: List[float] = []

    for group_idx in range(num_groups):
        start = group_idx * config.group_size
        end = min((group_idx + 1) * config.group_size, num_elements)
        group = tensor_flat[start:end]

        abs_max = group.abs().max()
        scale = abs_max / 7.0 if abs_max > 0 else 1.0

        q_group = torch.round(group / scale).to(torch.int8)
        q_group = torch.clamp(q_group, -8, 7)

        quantized[start:end] = q_group
        scales.append(float(scale))

    metadata: Dict[str, torch.Tensor] = {
        "scales": torch.tensor(scales, dtype=torch.float32),
        "group_size": torch.tensor(config.group_size, dtype=torch.int32),
        "shape": torch.tensor(original_shape, dtype=torch.int32),
    }

    return quantized.reshape(original_shape), metadata


def int4_to_balanced_ternary(value: int) -> List[int]:
    """Convert a 4-bit integer into its three-plane ternary coefficients."""

    mapping = {
        -8: [-1, 0, 1],
        -7: [-1, 1, -1],
        -6: [-1, 1, 0],
        -5: [-1, 1, 1],
        -4: [0, -1, -1],
        -3: [0, -1, 0],
        -2: [0, -1, 1],
        -1: [0, 0, -1],
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, -1],
        3: [0, 1, 0],
        4: [0, 1, 1],
        5: [1, -1, -1],
        6: [1, -1, 0],
        7: [1, -1, 1],
    }
    clamped = int(np.clip(value, -8, 7))
    return mapping[clamped]


class TernaryModelExporter:
    """Convert and export a model in ternary multiplane format."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name
        self.config = QuantizationConfig()
        self.layers_converted = 0
        self.total_weights = 0
        self.total_bytes = 0

    def convert_linear_layer(self, layer: nn.Linear) -> Dict[str, np.ndarray]:
        """Convert a single linear layer to the ternary multiplane format."""

        if layer.in_features == 0 or layer.out_features == 0:
            return {}

        weight_4bit, metadata = quantize_tensor_to_4bit(layer.weight.data, self.config)

        out_features, in_features = weight_4bit.shape
        num_weights = out_features * in_features
        packed_size = (num_weights + 7) // 8

        planes_data = []
        weight_flat = weight_4bit.flatten().cpu().numpy()

        for plane_idx in range(3):
            pos_mask = np.zeros(packed_size, dtype=np.uint8)
            neg_mask = np.zeros(packed_size, dtype=np.uint8)

            for i, weight in enumerate(weight_flat):
                ternary_coeffs = int4_to_balanced_ternary(int(weight))
                coeff = ternary_coeffs[plane_idx]

                byte_idx = i // 8
                bit_idx = i % 8

                if coeff == 1:
                    pos_mask[byte_idx] |= 1 << bit_idx
                elif coeff == -1:
                    neg_mask[byte_idx] |= 1 << bit_idx

            planes_data.append({"pos_mask": pos_mask, "neg_mask": neg_mask})

        weight_bytes = packed_size * 2 * 3
        scale_bytes = int(metadata["scales"].numel()) * 4
        total_bytes = weight_bytes + scale_bytes + 16

        self.layers_converted += 1
        self.total_weights += num_weights
        self.total_bytes += total_bytes

        bias = layer.bias.data.cpu().numpy() if layer.bias is not None else None

        return {
            "shape": (out_features, in_features),
            "group_size": int(metadata["group_size"].item()),
            "group_scales": metadata["scales"].cpu().numpy(),
            "plane_scales": np.array([9.0, 3.0, 1.0], dtype=np.float32),
            "planes": planes_data,
            "bias": bias,
        }

    def export_to_file(self, model: nn.Module, output_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Convert all linear layers and persist them to disk."""

        print(f"Exporting model to {output_path}...")
        converted_layers: Dict[str, Dict[str, np.ndarray]] = {}

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if "wte" in name or "wpe" in name or "ln" in name:
                print(f"  Skipping {name} (embedding/norm layer)")
                continue

            print(f"  Converting {name}... ", end="")
            layer_data = self.convert_linear_layer(module)
            if layer_data:
                converted_layers[name] = layer_data
                print(f"✓ ({layer_data['shape'][0]}×{layer_data['shape'][1]})")
            else:
                print("skipped")

        with open(output_path, "wb") as f:
            f.write(b"TERN")
            f.write(struct.pack("I", 1))
            f.write(struct.pack("I", len(converted_layers)))

            for name, layer_data in converted_layers.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("I", len(name_bytes)))
                f.write(name_bytes)

                rows, cols = layer_data["shape"]
                f.write(struct.pack("II", rows, cols))
                f.write(struct.pack("I", layer_data["group_size"]))
                f.write(struct.pack("I", len(layer_data["group_scales"])))

                layer_data["plane_scales"].tofile(f)
                layer_data["group_scales"].tofile(f)

                for plane in layer_data["planes"]:
                    plane["pos_mask"].tofile(f)
                    plane["neg_mask"].tofile(f)

                has_bias = layer_data["bias"] is not None
                f.write(struct.pack("B", 1 if has_bias else 0))
                if has_bias:
                    layer_data["bias"].astype(np.float32).tofile(f)

        print("\n✓ Export complete!")
        print(f"  Layers converted: {self.layers_converted}")
        print(f"  Total weights: {self.total_weights:,}")
        print(f"  File size: {self.total_bytes / 1024 / 1024:.2f} MB")

        metadata_path = Path(output_path).with_suffix(".json")
        metadata = {
            "model_name": self.model_name,
            "format": "ternary_multiplane",
            "version": 1,
            "layers_converted": self.layers_converted,
            "total_weights": self.total_weights,
            "total_bytes": self.total_bytes,
            "config": {
                "bits": self.config.bits,
                "group_size": self.config.group_size,
                "plane_scales": [9.0, 3.0, 1.0],
            },
            "layers": {
                name: {
                    "shape": layer["shape"],
                    "has_bias": layer["bias"] is not None,
                }
                for name, layer in converted_layers.items()
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata saved to {metadata_path}")
        return converted_layers

    def verify_export(self, export_path: str) -> None:
        """Sanity-check that an exported file can be read back."""

        print(f"\nVerifying export {export_path}...")
        with open(export_path, "rb") as f:
            magic = f.read(4)
            assert magic == b"TERN", f"Invalid magic number: {magic}"

            version = struct.unpack("I", f.read(4))[0]
            assert version == 1, f"Unknown version: {version}"

            num_layers = struct.unpack("I", f.read(4))[0]
            print(f"  Found {num_layers} layers")

            for _ in range(num_layers):
                name_len = struct.unpack("I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                rows, cols = struct.unpack("II", f.read(8))
                print(f"    Layer '{name}': {rows}×{cols}")

                group_size = struct.unpack("I", f.read(4))[0]
                n_groups = struct.unpack("I", f.read(4))[0]

                f.seek(3 * 4, 1)
                f.seek(n_groups * 4, 1)

                packed_size = (rows * cols + 7) // 8
                f.seek(packed_size * 2 * 3, 1)

                has_bias = struct.unpack("B", f.read(1))[0]
                if has_bias:
                    f.seek(rows * 4, 1)

        print("  ✓ File structure verified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to ternary format")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--output", type=str, default="model.ternary", help="Output file path")
    parser.add_argument("--verify", action="store_true", help="Verify the export after writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not HAS_TRANSFORMERS:
        print("Error: transformers library required")
        print("Install with: pip install transformers torch")
        return

    assert AutoModelForCausalLM is not None

    print("=" * 70)
    print("TERNARY MODEL EXPORTER FOR BITNET.CPP")
    print("=" * 70)

    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")

    exporter = TernaryModelExporter(args.model)
    exporter.export_to_file(model, args.output)

    if args.verify:
        exporter.verify_export(args.output)

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nModel exported to: {args.output}")
    print(f"Metadata saved to: {Path(args.output).with_suffix('.json')}")
    print("\nTo use with BitNet.cpp:")
    print("1. Copy the .ternary file to your BitNet.cpp models directory")
    print("2. Update BitNet.cpp with the multiplane kernel (see documentation)")
    print("3. Run inference with: ./bitnet -m model.ternary")
    print("\nNote: BitNet.cpp must be modified to support the 3-plane format")
    print("See the integration guide for details.")


if __name__ == "__main__":
    main()

