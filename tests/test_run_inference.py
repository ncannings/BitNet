from pathlib import Path
import sys

import numpy as np
import torch

from transformers import LlamaConfig, LlamaForCausalLM

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_inference import _apply_ternary_weights
from utils.ternary_loader import TernaryLayer, TernaryModel, TernaryPlane


def _make_ternary_layer(
    name: str, weights: np.ndarray, bias: np.ndarray | None = None
) -> TernaryLayer:
    weights = np.asarray(weights, dtype=np.float32)
    flat = weights.reshape(-1)
    if weights.ndim == 0:
        shape = ()
        group_size = 1
    else:
        shape = weights.shape
        group_size = shape[-1] if shape[-1] > 0 else 1

    total = flat.size
    if group_size <= 0:
        group_size = 1
    n_groups = int(np.ceil(total / group_size)) if total else 0
    group_scales = np.ones((n_groups,), dtype=np.float32)

    def pack(bits: np.ndarray) -> np.ndarray:
        if bits.size == 0:
            return np.zeros((0,), dtype=np.uint8)
        return np.packbits(bits.astype(np.uint8), bitorder="little")

    positive = pack(flat > 0)
    negative = pack(flat < 0)
    zero_mask = np.zeros_like(positive)

    planes = [
        TernaryPlane(pos_mask=positive, neg_mask=negative),
        TernaryPlane(pos_mask=zero_mask, neg_mask=zero_mask),
        TernaryPlane(pos_mask=zero_mask, neg_mask=zero_mask),
    ]

    plane_scales = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return TernaryLayer(
        name=name,
        shape=shape,
        group_size=int(group_size),
        group_scales=group_scales,
        plane_scales=plane_scales,
        planes=planes,
        bias=None if bias is None else np.asarray(bias, dtype=np.float32),
    )


def test_apply_ternary_weights_resolves_llama_cpp_aliases():
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = LlamaForCausalLM(config)

    q_weight = np.array(
        [
            [1, -1, 0, 1],
            [-1, 1, 1, -1],
            [0, 1, -1, 1],
            [1, 0, -1, -1],
        ],
        dtype=np.float32,
    )
    norm_weight = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)
    lm_head_weight = np.array(
        [
            [1, -1, 0, 1, -1, 1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0],
            [-1, 1, 1, -1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 1, -1],
            [0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
            [1, 0, -1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0],
        ],
        dtype=np.float32,
    )

    layers = {
        "blk.0.attn_q.weight": _make_ternary_layer("blk.0.attn_q.weight", q_weight),
        "blk.0.attn_norm.weight": _make_ternary_layer("blk.0.attn_norm.weight", norm_weight),
        "output.weight": _make_ternary_layer("output.weight", lm_head_weight),
    }

    ternary_model = TernaryModel(layers=layers, metadata={})

    _apply_ternary_weights(model, ternary_model)

    q_proj_weight = (
        model.model.layers[0].self_attn.q_proj.weight.detach().cpu().numpy()
    )
    assert np.allclose(q_proj_weight, q_weight)

    norm_tensor = (
        model.model.layers[0].input_layernorm.weight.detach().cpu().numpy()
    )
    assert np.allclose(norm_tensor, norm_weight)

    lm_head_tensor = model.lm_head.weight.detach().cpu().numpy()
    assert np.allclose(lm_head_tensor, lm_head_weight.T)


def test_apply_ternary_weights_handles_direct_hf_names():
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = LlamaForCausalLM(config)

    up_weight = np.array(
        [
            [1, -1, 0, 1],
            [-1, 1, 1, 0],
            [0, 1, -1, -1],
            [1, 0, 1, -1],
            [-1, -1, 0, 1],
            [1, 1, -1, 0],
            [0, -1, 1, 1],
            [-1, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    down_weight = np.array(
        [
            [1, 0, -1, 1, -1, 1, 0, -1],
            [-1, 1, 0, -1, 1, -1, 0, 1],
            [0, -1, 1, 0, -1, 1, -1, 0],
            [1, 1, -1, 0, 0, 1, 1, -1],
        ],
        dtype=np.float32,
    )
    norm_weight = np.array([1, -1, 0, 1], dtype=np.float32)

    layers = {
        "blk.0.ffn_up.weight": _make_ternary_layer("blk.0.ffn_up.weight", up_weight),
        "model.layers.0.mlp.down_proj.weight": _make_ternary_layer(
            "model.layers.0.mlp.down_proj.weight", down_weight
        ),
        "blk.0.attn_norm.weight": _make_ternary_layer(
            "blk.0.attn_norm.weight", norm_weight
        ),
    }

    ternary_model = TernaryModel(layers=layers, metadata={})

    _apply_ternary_weights(model, ternary_model)

    up_proj = model.model.layers[0].mlp.up_proj
    assert np.allclose(up_proj.weight.detach().cpu().numpy(), up_weight)
    down_proj = model.model.layers[0].mlp.down_proj
    assert np.allclose(down_proj.weight.detach().cpu().numpy(), down_weight)

    attn_norm = model.model.layers[0].input_layernorm
    assert np.allclose(attn_norm.weight.detach().cpu().numpy(), norm_weight)


def test_apply_ternary_weights_updates_bias_when_present():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(3, 3, bias=True)

    model = DummyModel()

    weight = np.array(
        [[1, -1, 0], [-1, 1, 1], [0, -1, 1]], dtype=np.float32
    )
    bias = np.array([0.25, -0.5, 1.0], dtype=np.float32)

    layers = {
        "dense.weight": _make_ternary_layer("dense.weight", weight, bias=bias)
    }

    ternary_model = TernaryModel(layers=layers, metadata={})

    _apply_ternary_weights(model, ternary_model)

    assert np.allclose(model.dense.weight.detach().cpu().numpy(), weight)
    assert np.allclose(model.dense.bias.detach().cpu().numpy(), bias)

