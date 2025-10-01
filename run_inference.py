import argparse
import os
import platform
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from utils.ternary_loader import (
    TernaryModel,
    is_ternary_file,
    load_ternary_model,
    materialize_layer,
)


def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error occurred while running command: {exc}")
        sys.exit(1)


def _resolve_llama_cli() -> str:
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")
    return main_path


def run_llama_cli(args):
    main_path = _resolve_llama_cli()
    command = [
        f"{main_path}",
        "-m",
        args.model,
        "-n",
        str(args.n_predict),
        "-t",
        str(args.threads),
        "-p",
        args.prompt,
        "-ngl",
        "0",
        "-c",
        str(args.ctx_size),
        "--temp",
        str(args.temperature),
        "-b",
        "1",
    ]
    if args.conversation:
        command.append("-cnv")
    run_command(command)


def _load_transformers():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        print(
            "Error: ternary inference requires the 'torch' and 'transformers' packages. "
            "Install them with `pip install torch transformers`."
        )
        raise SystemExit(1) from exc

    return AutoModelForCausalLM, AutoTokenizer


_LLAMA_CPP_STATIC_ALIASES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "token_embd.bias": "model.embed_tokens.bias",
    "output.weight": "lm_head.weight",
    "output.bias": "lm_head.bias",
    "output_norm.weight": "model.norm.weight",
    "output_norm.bias": "model.norm.bias",
}


_LLAMA_CPP_PATTERN_ALIASES = [
    (
        re.compile(r"^blk\.(?P<i>\d+)\.attn_q\.(?P<param>weight|bias)$"),
        "model.layers.{i}.self_attn.q_proj.{param}",
    ),
    (
        re.compile(r"^blk\.(?P<i>\d+)\.attn_k\.(?P<param>weight|bias)$"),
        "model.layers.{i}.self_attn.k_proj.{param}",
    ),
    (
        re.compile(r"^blk\.(?P<i>\d+)\.attn_v\.(?P<param>weight|bias)$"),
        "model.layers.{i}.self_attn.v_proj.{param}",
    ),
    (
        re.compile(r"^blk\.(?P<i>\d+)\.attn_output\.(?P<param>weight|bias)$"),
        "model.layers.{i}.self_attn.o_proj.{param}",
    ),
    (
        re.compile(r"^blk\.(?P<i>\d+)\.attn_norm\.(?P<param>weight|bias)$"),
        "model.layers.{i}.input_layernorm.{param}",
    ),
    (
        re.compile(r"^blk\.(?P<i>\d+)\.ffn_norm\.(?P<param>weight|bias)$"),
        "model.layers.{i}.post_attention_layernorm.{param}",
    ),
    (
        re.compile(r"^blk\.(?P<i>\d+)\.ffn_up\.(?P<param>weight|bias)$"),
        "model.layers.{i}.mlp.up_proj.{param}",
    ),
    (
        re.compile(r"^blk\.(?P<i>\d+)\.ffn_gate\.(?P<param>weight|bias)$"),
        "model.layers.{i}.mlp.gate_proj.{param}",
    ),
    (
        re.compile(r"^blk\.(?P<i>\d+)\.ffn_down\.(?P<param>weight|bias)$"),
        "model.layers.{i}.mlp.down_proj.{param}",
    ),
]


def _alias_hf_name(layer_name: str) -> Optional[str]:
    if layer_name in _LLAMA_CPP_STATIC_ALIASES:
        return _LLAMA_CPP_STATIC_ALIASES[layer_name]
    for pattern, target in _LLAMA_CPP_PATTERN_ALIASES:
        match = pattern.match(layer_name)
        if match:
            return target.format(**match.groupdict())
    return None



def _resolve_module(
    named_modules: dict,
    named_parameters: dict,
    named_buffers: dict,
    layer_name: str,
) -> Tuple[Optional[object], Optional[str]]:
    """Return the owning module and attribute for a ternary layer."""

    def _resolve_candidate(candidate: str) -> Tuple[Optional[object], Optional[str]]:
        module = named_modules.get(candidate)
        if module is not None and hasattr(module, "weight"):
            return module, "weight"

        if candidate in named_parameters or candidate in named_buffers:
            if "." in candidate:
                module_name, attribute = candidate.rsplit(".", 1)
                module = named_modules.get(module_name)
                if module is not None and hasattr(module, attribute):
                    return module, attribute
        elif "." in candidate:
            module_name, attribute = candidate.rsplit(".", 1)
            module = named_modules.get(module_name)
            if module is not None and hasattr(module, attribute):
                return module, attribute

        return None, None

    for candidate in (layer_name, _alias_hf_name(layer_name)):
        if not candidate:
            continue
        module, attribute = _resolve_candidate(candidate)
        if module is not None:
            return module, attribute

    return None, None


def _prepare_hf_components(model_metadata: dict, hf_override: Optional[str]):
    AutoModelForCausalLM, AutoTokenizer = _load_transformers()
    model_name = hf_override or model_metadata.get("model_name")
    if not model_name:
        print(
            "Error: metadata for the ternary export is missing the base model name. "
            "Pass --hf-base to specify the Hugging Face identifier manually."
        )
        sys.exit(1)

    print(f"Loading Hugging Face model '{model_name}' for ternary inference...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _apply_ternary_weights(model, ternary_model: TernaryModel):
    import torch

    named_modules = dict(model.named_modules())
    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    for layer_name, layer in ternary_model.layers.items():
        module, attribute = _resolve_module(
            named_modules, named_parameters, named_buffers, layer_name
        )

        if module is None or attribute is None:
            print(f"Warning: layer '{layer_name}' not found in base model; skipping.")
            continue

        weights = materialize_layer(layer)
        target_param = getattr(module, attribute, None)
        if target_param is None:
            print(
                "Warning: resolved attribute '"
                f"{attribute}' for layer '{layer_name}' missing on base model; skipping."
            )
            continue

        weight_tensor = torch.from_numpy(weights).to(
            dtype=target_param.dtype, device=target_param.device
        )

        if weight_tensor.shape != target_param.shape:
            if (
                weight_tensor.ndim == target_param.ndim == 2
                and weight_tensor.t().shape == target_param.shape
            ):
                weight_tensor = weight_tensor.t().contiguous()
            else:
                print(
                    "Warning: layer '"
                    f"{layer_name}' resolved to '{attribute}' with mismatched shape "
                    f"{weight_tensor.shape} != {tuple(target_param.shape)}; skipping."
                )
                continue

        target_param.data.copy_(weight_tensor)

        if (
            attribute != "bias"
            and layer.bias is not None
            and getattr(module, "bias", None) is not None
        ):
            bias_tensor = torch.from_numpy(layer.bias).to(
                dtype=module.bias.dtype, device=module.bias.device
            )

            if bias_tensor.shape != module.bias.shape:
                print(
                    "Warning: bias for layer '"
                    f"{layer_name}' has mismatched shape {bias_tensor.shape} "
                    f"!= {tuple(module.bias.shape)}; skipping bias update."
                )
            else:
                module.bias.data.copy_(bias_tensor)



def run_ternary_inference(args):
    ternary_path = Path(args.model)
    print(f"Detected ternary export at '{ternary_path}'. Using Python inference path.")
    ternary_model = load_ternary_model(ternary_path)
    model, tokenizer = _prepare_hf_components(
        ternary_model.metadata, hf_override=args.hf_base
    )

    _apply_ternary_weights(model, ternary_model)

    try:
        import torch
    except ImportError:  # pragma: no cover - defensive guard
        print("Error: torch is required for ternary inference.")
        sys.exit(1)

    torch.set_num_threads(args.threads)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    max_new_tokens = max(args.n_predict, 1)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        output_ids = model.generate(**encoded, **generation_kwargs)

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


def run_inference(args):
    if args.model.lower().endswith(".ternary") or is_ternary_file(args.model):
        run_ternary_inference(args)
    else:
        run_llama_cli(args)


def signal_handler(sig, frame):  # pragma: no cover - CLI guard
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to model file",
        required=False,
        default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf",
    )
    parser.add_argument(
        "-n",
        "--n-predict",
        type=int,
        help="Number of tokens to predict when generating text",
        required=False,
        default=128,
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to generate text from",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads to use",
        required=False,
        default=2,
    )
    parser.add_argument(
        "-c",
        "--ctx-size",
        type=int,
        help="Size of the prompt context",
        required=False,
        default=2048,
    )
    parser.add_argument(
        "-temp",
        "--temperature",
        type=float,
        help="Temperature, a hyperparameter that controls the randomness of the generated text",
        required=False,
        default=0.8,
    )
    parser.add_argument(
        "--hf-base",
        type=str,
        help=(
            "Override the Hugging Face model identifier to use when hydrating ternary exports. "
            "This is useful if the export metadata is missing or you want to use a local path."
        ),
    )
    parser.add_argument(
        "-cnv",
        "--conversation",
        action="store_true",
        help="Whether to enable chat mode or not (for instruct models.)",
    )

    parsed_args = parser.parse_args()
    run_inference(parsed_args)