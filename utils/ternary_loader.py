"""Utilities for loading and materializing ternary multiplane weights."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TernaryPlane:
    """Bit-packed ternary mask data for one plane."""

    pos_mask: np.ndarray
    neg_mask: np.ndarray


@dataclass
class TernaryLayer:
    """In-memory representation of a ternary multiplane layer."""

    name: str
    shape: Tuple[int, ...]
    group_size: int
    group_scales: np.ndarray
    plane_scales: np.ndarray
    planes: List[TernaryPlane]
    bias: Optional[np.ndarray]

    @property
    def rows(self) -> int:
        return int(self.shape[0]) if self.shape else 0

    @property
    def cols(self) -> int:
        if len(self.shape) >= 2:
            return int(np.prod(self.shape[1:]))
        return 1 if self.shape else 0


@dataclass
class TernaryModel:
    """Container for an exported ternary model."""

    layers: Dict[str, TernaryLayer]
    metadata: Dict[str, object]


def _read_uint32(handle) -> int:
    data = handle.read(4)
    if len(data) != 4:
        raise EOFError("Unexpected end of file while reading uint32")
    return struct.unpack("<I", data)[0]


def _read_float32_array(handle, count: int) -> np.ndarray:
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    buffer = handle.read(4 * count)
    if len(buffer) != 4 * count:
        raise EOFError("Unexpected end of file while reading float array")
    return np.frombuffer(buffer, dtype=np.float32).copy()


def _read_mask(handle, size: int) -> np.ndarray:
    buffer = handle.read(size)
    if len(buffer) != size:
        raise EOFError("Unexpected end of file while reading mask")
    return np.frombuffer(buffer, dtype=np.uint8).copy()


def _load_metadata(path: Path) -> Dict[str, object]:
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as descriptor:
            return json.load(descriptor)
    return {}


def is_ternary_file(path: Path) -> bool:
    """Return True when the file looks like a ternary export."""

    try:
        with Path(path).open("rb") as handle:
            return handle.read(4) == b"TERN"
    except OSError:
        return False


def _read_shape(handle, dims: int) -> Tuple[int, ...]:
    shape: List[int] = []
    for _ in range(dims):
        shape.append(_read_uint32(handle))
    return tuple(shape)


def _load_version1_layers(handle) -> Dict[str, TernaryLayer]:
    layers: Dict[str, TernaryLayer] = {}

    layer_count = _read_uint32(handle)

    for _ in range(layer_count):
        name_length = _read_uint32(handle)
        name_bytes = handle.read(name_length)
        if len(name_bytes) != name_length:
            raise EOFError("Unexpected end of file while reading layer name")
        name = name_bytes.decode("utf-8")

        rows = _read_uint32(handle)
        cols = _read_uint32(handle)
        group_size = _read_uint32(handle)
        n_groups = _read_uint32(handle)

        plane_scales = _read_float32_array(handle, 3)
        group_scales = _read_float32_array(handle, n_groups)

        packed_size = (rows * cols + 7) // 8
        planes: List[TernaryPlane] = []
        for _plane in range(3):
            pos_mask = _read_mask(handle, packed_size)
            neg_mask = _read_mask(handle, packed_size)
            planes.append(TernaryPlane(pos_mask=pos_mask, neg_mask=neg_mask))

        bias_flag = handle.read(1)
        if len(bias_flag) != 1:
            raise EOFError("Unexpected end of file while reading bias flag")
        has_bias = struct.unpack("<B", bias_flag)[0] == 1
        bias = None
        if has_bias:
            bias = _read_float32_array(handle, rows)

        layers[name] = TernaryLayer(
            name=name,
            shape=(rows, cols),
            group_size=group_size,
            group_scales=group_scales,
            plane_scales=plane_scales,
            planes=planes,
            bias=bias,
        )

    return layers


def _load_version3_layers(handle) -> Tuple[Dict[str, TernaryLayer], Dict[str, object]]:
    layers: Dict[str, TernaryLayer] = {}

    layer_count = _read_uint32(handle)

    format_length = _read_uint32(handle)
    format_bytes = handle.read(format_length)
    if len(format_bytes) != format_length:
        raise EOFError("Unexpected end of file while reading format name")
    format_name = format_bytes.decode("utf-8") if format_bytes else ""

    metadata_length = _read_uint32(handle)
    metadata_blob = handle.read(metadata_length)
    if len(metadata_blob) != metadata_length:
        raise EOFError("Unexpected end of file while reading metadata block")
    metadata: Dict[str, object] = {}
    if metadata_blob:
        metadata = json.loads(metadata_blob.decode("utf-8"))

    if format_name and "source_format" not in metadata:
        metadata["source_format"] = format_name

    for _ in range(layer_count):
        name_length = _read_uint32(handle)
        name_bytes = handle.read(name_length)
        if len(name_bytes) != name_length:
            raise EOFError("Unexpected end of file while reading layer name")
        name = name_bytes.decode("utf-8")

        ndim = _read_uint32(handle)
        shape = _read_shape(handle, ndim)
        if not shape:
            total_weights = 0
        else:
            total_weights = int(np.prod(shape))

        group_size = _read_uint32(handle)
        n_groups = _read_uint32(handle)

        plane_scales = _read_float32_array(handle, 3)
        group_scales = _read_float32_array(handle, n_groups)

        packed_size = (total_weights + 7) // 8
        planes: List[TernaryPlane] = []
        for _plane in range(3):
            pos_mask = _read_mask(handle, packed_size)
            neg_mask = _read_mask(handle, packed_size)
            planes.append(TernaryPlane(pos_mask=pos_mask, neg_mask=neg_mask))

        layers[name] = TernaryLayer(
            name=name,
            shape=shape,
            group_size=group_size,
            group_scales=group_scales,
            plane_scales=plane_scales,
            planes=planes,
            bias=None,
        )

    return layers, metadata


def load_ternary_model(path: Path) -> TernaryModel:
    """Parse a ternary model export into an in-memory structure."""

    path = Path(path)

    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"TERN":
            raise ValueError("File is not a ternary export (missing magic header)")

        version = _read_uint32(handle)

        if version == 1:
            layers = _load_version1_layers(handle)
            metadata = _load_metadata(path)
        elif version == 3:
            layers, metadata = _load_version3_layers(handle)
            external_metadata = _load_metadata(path)
            if external_metadata:
                metadata = {**metadata, **external_metadata}
        else:
            raise ValueError(f"Unsupported ternary export version: {version}")

    return TernaryModel(layers=layers, metadata=metadata)


def materialize_layer(layer: TernaryLayer) -> np.ndarray:
    """Convert a ternary layer into floating-point weights."""

    if layer.shape:
        total_weights = int(np.prod(layer.shape))
    else:
        total_weights = 0
    if total_weights == 0:
        return np.zeros(layer.shape, dtype=np.float32)

    accumulator = np.zeros(total_weights, dtype=np.float32)

    for plane_index, plane in enumerate(layer.planes):
        pos_bits = np.unpackbits(plane.pos_mask, bitorder="little", count=total_weights)
        neg_bits = np.unpackbits(plane.neg_mask, bitorder="little", count=total_weights)
        coeff = pos_bits.astype(np.int8) - neg_bits.astype(np.int8)
        accumulator += coeff.astype(np.float32) * layer.plane_scales[plane_index]

    if layer.group_scales.size > 0:
        group_indices = np.arange(total_weights) // max(layer.group_size, 1)
        group_indices = np.clip(group_indices, 0, layer.group_scales.size - 1)
        accumulator *= layer.group_scales[group_indices]

    return accumulator.reshape(layer.shape)

