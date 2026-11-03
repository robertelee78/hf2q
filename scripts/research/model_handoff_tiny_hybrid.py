#!/usr/bin/env python3
"""Research-only random-weight hybrid model control; never a serving backend.

Requires torch and transformers, but downloads no weights. Demonstrates complete
same-model state restoration, the effect of dropping recurrent state, and the
upstream SwaT cache reader's failure on a real hybrid cache object. This is NOT
pretrained-model quality, hf2q/Metal parity, or cross-model transfer evidence.
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import transformers
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swat-checkout", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, partial_rotary_factor=0.5,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        max_position_embeddings=128,
    )
    model = Qwen3_5ForCausalLM(cfg).eval()
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), use_cache=True)
        anchor = output.past_key_values
        restored = copy.deepcopy(anchor)
        incomplete = copy.deepcopy(anchor)
        for layer in incomplete.layers[:3]:
            layer.recurrent_states.zero_()
            layer.conv_states.zero_()
        token = torch.tensor([[9]])
        native = model(input_ids=token, past_key_values=anchor, use_cache=True).logits
        resumed = model(input_ids=token, past_key_values=restored, use_cache=True).logits
        damaged = model(input_ids=token, past_key_values=incomplete, use_cache=True).logits
    assert torch.equal(native, resumed)
    delta = (native - damaged).abs().max().item()
    assert delta > 0

    source = args.swat_checkout / "multiagent/model/unaware_comm.py"
    tree = ast.parse(source.read_text())
    selected = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cache_layer")
    # Execute the upstream reader unchanged; only its type-annotation binding is supplied.
    namespace = {"Tensor": torch.Tensor}
    exec(compile(ast.Module(body=[selected], type_ignores=[]), str(source), "exec"), namespace)
    error = None
    try:
        namespace["cache_layer"](restored, 0)
    except AttributeError as exc:
        error = str(exc)
    assert error and "keys" in error
    result = {
        "schema": "hf2q.model-handoff-tiny-hybrid.v1",
        "seed": 7, "device": "cpu", "torch": torch.__version__,
        "transformers": transformers.__version__,
        "swat_commit": subprocess.check_output(
            ["git", "-C", str(args.swat_checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_reader_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fixture": "random-weight four-layer Qwen3.5, eight-token prefix, one-token continuation",
        "same_model_complete_restore_max_abs_logit_difference": (native - resumed).abs().max().item(),
        "kv_only_max_abs_logit_difference": delta,
        "upstream_cache_reader_error": error,
        "scope": "CPU mechanics and negative control only; no pretrained weights, cross-model mapping, or timing claim",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
