#!/usr/bin/env python3
"""Generate DFlash2 parity tensors from the pinned Apache-2.0 reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, quants
from safetensors.torch import save_file

REFERENCE_COMMIT = "95c8aeca5e4b4c4f9c0c967c05ab89fa3ed24f4c"
REFERENCE_MODEL_SHA256 = "f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e"
SCHEMA = "hf2q.dflash2-reference-fixture.v2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_reference(path: Path):
    actual = digest(path)
    if actual != REFERENCE_MODEL_SHA256:
        raise RuntimeError(
            f"reference model.py SHA256 {actual} != pinned {REFERENCE_MODEL_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("hf2q_pinned_dflash_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gguf_matrix(reader: GGUFReader, name: str) -> torch.Tensor:
    tensor = next((item for item in reader.tensors if item.name == name), None)
    if tensor is None:
        raise RuntimeError(f"GGUF tensor {name!r} is missing")
    values = quants.dequantize(tensor.data, tensor.tensor_type)
    shape = tuple(reversed([int(value) for value in tensor.shape]))
    return torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(shape).copy())


def deterministic_target_hidden(
    batch: int,
    rows: int,
    width: int,
    offset: int,
    lane_offset: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dimensions = torch.arange(width, dtype=torch.float32, device=device)
    positions = torch.arange(offset, offset + rows, dtype=torch.float32, device=device)
    lanes = torch.arange(batch, dtype=torch.float32, device=device) + lane_offset
    values = (
        torch.sin(dimensions[None, None, :] * 0.0017 + positions[None, :, None] * 0.031)
        + torch.cos(dimensions[None, None, :] * 0.0009 + lanes[:, None, None] * 0.17)
    ) * 0.125
    return values.to(dtype)


def install_captures(model):
    captures: dict[str, torch.Tensor] = {}
    handles = []

    for index, layer in enumerate(model.layers):
        def layer_hook(_module, _inputs, output, *, index=index):
            captures[f"layer.{index}.output"] = output.detach().clone()

        handles.append(layer.register_forward_hook(layer_hook))
        for label, convolution in (
            ("attention", layer.attention_conv),
            ("mlp", layer.mlp_conv),
        ):
            original_prepare = convolution.prepare
            original_finish = convolution.finish

            def prepare(self, hidden, *, original=original_prepare, index=index, label=label):
                output, dynamic = original(hidden)
                captures[f"layer.{index}.{label}.prepare"] = output.detach().clone()
                captures[f"layer.{index}.{label}.dynamic"] = dynamic.detach().clone()
                return output, dynamic

            def finish(self, hidden, dynamic, *, original=original_finish, index=index, label=label):
                output = original(hidden, dynamic)
                captures[f"layer.{index}.{label}.finish"] = output.detach().clone()
                return output

            convolution.prepare = types.MethodType(prepare, convolution)
            convolution.finish = types.MethodType(finish, convolution)
    return captures, handles


def install_hf2q_attention_precision(model, reference):
    """Match hf2q's F32 draft graph with BF16 packed attention and KV rings."""

    def forward(
        self,
        hidden_states,
        target_hidden,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        batch, query_length = hidden_states.shape[:-1]
        context_length = target_hidden.shape[1]
        query = self.q_proj(hidden_states)
        query = query.view(batch, query_length, -1, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key = torch.cat(
            [self.k_proj(target_hidden), self.k_proj(hidden_states)], dim=1
        )
        value = torch.cat(
            [self.v_proj(target_hidden), self.v_proj(hidden_states)], dim=1
        )
        key = key.view(batch, context_length + query_length, -1, self.head_dim)
        value = value.view(batch, context_length + query_length, -1, self.head_dim)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cosine, sine = position_embeddings
        query, key = reference.apply_rotary_pos_emb(query, key, cosine, sine)
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sine,
                "cos": cosine,
                "cache_position": cache_position,
            }
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs
            )
        if attention_mask is None and (
            self.is_causal or self.sliding_window is not None
        ):
            attention_mask = reference._attention_mask(
                query,
                key,
                is_causal=self.is_causal,
                sliding_window=self.sliding_window,
            )
        output, weights = reference.ALL_ATTENTION_FUNCTIONS["sdpa"](
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        output = output.float().reshape(batch, query_length, -1)
        return self.o_proj(output), weights

    for layer in model.layers:
        layer.self_attn.forward = types.MethodType(forward, layer.self_attn)


def selector_tables(model, hidden: torch.Tensor, logits: torch.Tensor, anchors: torch.Tensor):
    selector = model.candidate_selector
    unary, candidates = torch.topk(logits, selector.top_k, dim=-1, sorted=False)
    projected = selector.hidden_projection(hidden)
    batch, positions, top_k = candidates.shape
    edges = torch.zeros(
        (batch, positions, top_k, top_k), dtype=torch.float32, device=hidden.device
    )
    for position in range(positions):
        predecessors = anchors[:, None] if position == 0 else candidates[:, position - 1]
        predecessor = selector.predecessor_codebook(predecessors)
        successor = selector.successor_codebook(candidates[:, position])
        context = predecessor * projected[:, position, None, :]
        edges[:, position] = torch.einsum("bpr,bsr->bps", context.float(), successor.float())
    path, _, _ = selector.select(hidden, logits, anchors, 0.0)
    confidence = []
    predecessor_index = torch.zeros(batch, dtype=torch.long, device=hidden.device)
    for position in range(positions):
        score = unary[:, position] + edges[
            torch.arange(batch, device=hidden.device), position, predecessor_index
        ]
        probability = torch.softmax(score.float(), dim=-1)
        selected = torch.argmax(score, dim=-1)
        confidence.append(probability.gather(-1, selected[:, None])[:, 0])
        predecessor_index = selected
    return {
        "selector_hidden": projected,
        "candidates": candidates,
        "unary": unary,
        "edges": edges,
        "path": path,
        "confidence": torch.stack(confidence, dim=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf-sha256", required=True)
    parser.add_argument(
        "--case",
        choices=("short", "wrap", "multi", "restore", "single"),
        required=True,
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float32"),
        default="float32",
        help="Reference activation precision; float32 matches hf2q's runtime",
    )
    parser.add_argument("--lane-offset", type=int, default=0)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("DFlash2 reference fixtures require Apple MPS")
    reference = load_reference(args.reference_model)
    config_path = args.checkpoint / "config.json"
    weights_path = args.checkpoint / "model.safetensors"
    for path in (config_path, weights_path, args.gguf):
        if not path.is_file():
            raise RuntimeError(f"required input is missing: {path}")

    device = torch.device("mps")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.compute_dtype]
    model = reference.DFlash2DraftModel.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=dtype
    ).to(device).eval()
    if dtype == torch.float32:
        install_hf2q_attention_precision(model, reference)
    reader = GGUFReader(args.gguf)
    embedding = gguf_matrix(reader, "token_embd.weight").to(device=device, dtype=dtype)
    try:
        output_head_weight = gguf_matrix(reader, "output.weight")
    except RuntimeError:
        output_head_weight = embedding.float().cpu()
    output_head = torch.nn.Linear(
        output_head_weight.shape[1], output_head_weight.shape[0], bias=False,
        device=device, dtype=dtype,
    )
    output_head.weight.data.copy_(output_head_weight.to(device=device, dtype=dtype))
    del output_head_weight, reader

    captures, handles = install_captures(model)
    case_geometry = {
        "short": (1, 3, 0),
        "wrap": (1, 2048, 0),
        "multi": (2, 17, 0),
        "restore": (1, 12, 0),
        "single": (1, 17, 0),
    }
    batch, context_rows, offset = case_geometry[args.case]
    anchors = torch.tensor(
        [17 + (args.lane_offset + lane) * 101 for lane in range(batch)],
        device=device,
    )
    row_tokens = anchors[:, None].repeat(1, model.block_size)
    row_tokens[:, 1:] = model.mask_token_id
    noise = torch.nn.functional.embedding(row_tokens, embedding)
    target_width = len(model.target_layer_ids) * model.config.hidden_size
    target_hidden = deterministic_target_hidden(
        batch, context_rows, target_width, offset, args.lane_offset, device, dtype
    )
    initial_target_hidden = target_hidden
    position_ids = torch.arange(
        offset, offset + context_rows + model.block_size, device=device
    ).unsqueeze(0).expand(batch, -1)

    try:
        cache = reference._make_cache(model.config)
    except AttributeError as error:
        if "activate_past_recording" not in str(error):
            raise
        # transformers 5.5 records DynamicCache updates unconditionally;
        # the pinned reference targets 5.15, where recording is explicit.
        cache = reference.DynamicCache(config=model.config)
    with torch.inference_mode():
        final_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        reference._crop_to(cache, offset + context_rows)
        if args.case == "restore":
            restore_hidden = deterministic_target_hidden(
                batch, 3, target_width, 12, args.lane_offset, device, dtype
            )
            restore_positions = torch.arange(12, 12 + 3 + model.block_size, device=device)[None]
            target_hidden = restore_hidden
            final_hidden = model(
                target_hidden=restore_hidden,
                noise_embedding=noise,
                position_ids=restore_positions,
                past_key_values=cache,
                use_cache=True,
            )
            reference._crop_to(cache, 15)
            offset, context_rows = 12, 3
        proposal_hidden = final_hidden[:, 1:]
        logits = model.compute_logits(proposal_hidden, output_head)
        selector = selector_tables(model, proposal_hidden, logits, anchors)

    tensors: dict[str, torch.Tensor] = {
        "input.target_hidden": target_hidden,
        "input.noise_embedding": noise,
        "input.anchor": anchors.to(torch.int64),
        "output.final_hidden": final_hidden,
        "output.logits": logits,
    }
    if args.case == "restore":
        tensors["input.initial_target_hidden"] = initial_target_hidden
    tensors.update({f"output.{name}": value for name, value in selector.items()})
    tensors.update({f"output.{name}": value for name, value in captures.items()})
    tensors = {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
    for handle in handles:
        handle.remove()

    tolerances = (
        {
            "input_max_abs": 0.03,
            "layer_relative_max_abs": 0.05,
            "final_hidden_max_abs": 2.5,
            "logits_max_abs": 2.0,
            "selector_max_abs": 1.1,
            "candidate_min_overlap": 10,
            "path_min_agreement": 4,
        }
        if dtype == torch.float32
        else {"mode": "diagnostic_only"}
    )
    metadata = {
        "schema": SCHEMA,
        "case": args.case,
        "reference_commit": REFERENCE_COMMIT,
        "reference_model_sha256": REFERENCE_MODEL_SHA256,
        "checkpoint_config_sha256": digest(config_path),
        "checkpoint_weights_sha256": digest(weights_path),
        "target_gguf_sha256": args.gguf_sha256,
        "attention_precision": "bfloat16" if dtype == torch.float32 else args.compute_dtype,
        "context_start": str(offset),
        "compute_dtype": args.compute_dtype,
        "context_rows": str(context_rows),
        "lane_offset": str(args.lane_offset),
        "tolerances": json.dumps(tolerances, sort_keys=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata=metadata)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": digest(args.output),
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
