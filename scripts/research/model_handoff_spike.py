#!/usr/bin/env python3
"""Bounded ADR-064 research probes; no model inference or production imports.

Run with --swat-checkout pointing to the pinned upstream SwaT checkout.
The toy counterexample is mathematical evidence against a generic lossless
translation promise, NOT a measurement of Qwen or DeepSeek model quality.
"""

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace


def git_head(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def function_from_source(path, name):
    """Execute one upstream pure function unchanged, without importing PyTorch."""
    tree = ast.parse(path.read_text())
    selected = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(compile(ast.Module(body=[selected], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def adapter_probe(swat):
    source = swat / "multiagent/model/unaware_comm.py"
    indices = function_from_source(source, "kv_layer_indices")
    kinds = ["linear_attention"] * 3 + ["full_attention"]
    cfg = SimpleNamespace(num_hidden_layers=4, layer_types=kinds)
    selected = indices(cfg)
    wrong = [i for i in selected if kinds[i] != "full_attention"]
    assert wrong == [0, 1, 2]
    return {
        "upstream_commit": git_head(swat),
        "upstream_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fixture": "four-layer synthetic Qwen-style 3:1 hybrid schedule",
        "selected_kv_layers": selected,
        "non_kv_layers_selected": wrong,
        "verdict": "unchanged layer enumeration is not a hybrid-state adapter",
        "scope": "executes upstream pure enumeration; does not execute a model",
    }


def collision_probe():
    # Simplified finite scalar gated delta recurrence, alpha=1, beta=1/2,
    # k=1: s' = (s + v)/2. Last value models a one-token conv history.
    # Two different prefixes can have identical source persistent state,
    # while a target retaining token-indexed history can distinguish them.
    seen = {}
    collision = None
    for values in itertools.product(range(3), repeat=4):
        state = 0.0
        for value in values:
            state = (state + value) / 2
        key = (state, values[-1])
        if key in seen and seen[key][0] != values[0]:
            collision = (seen[key], values, state)
            break
        seen[key] = values
    assert collision is not None
    first, second, state = collision
    return {
        "prefixes": [first, second],
        "identical_source_state": {"recurrent": state, "last_value": first[-1]},
        "target_query": "return the first historical value",
        "different_target_answers": [first[0], second[0]],
        "verdict": "source state alone need not determine target history",
        "scope": "toy counterexample, not an impossibility proof for the actual model pair",
        "implication": "retaining text permits replay; translating state cannot guarantee exact recovery",
    }


def exact_return_probe():
    # Executable protocol specification: match the receiver's *rendered*
    # token ledger, never a message count or a token count in another model.
    def plan(owner, current_owner, anchor, rendered):
        if owner != current_owner:
            return 0, len(rendered), "identity_mismatch"
        if tuple(rendered[:len(anchor)]) != tuple(anchor):
            return 0, len(rendered), "prefix_mismatch"
        return len(anchor), len(rendered) - len(anchor), "native_anchor_suffix"

    identity = ("session-a", "qwen-hash", "template-hash", "config-hash", "exact")
    prefix = tuple(range(1000))
    continuation = prefix + tuple(range(1000, 1064))
    good = plan(identity, identity, prefix, continuation)
    assert good == (1000, 64, "native_anchor_suffix")
    failures = []
    for field in range(len(identity)):
        changed = list(identity)
        changed[field] = "incompatible"
        result = plan(identity, tuple(changed), prefix, continuation)
        assert result[:2] == (0, 1064)
        failures.append(field)
    rewritten = (9999,) + continuation[1:]
    assert plan(identity, identity, prefix, rewritten) == (0, 1064, "prefix_mismatch")
    # A state obtained through learned transport cannot become an exact anchor.
    approximate = (*identity[:-1], "translated")
    assert plan(approximate, identity, prefix, continuation)[0] == 0
    return {
        "fixture": "1000 receiver tokens before switch, 64 appended receiver tokens",
        "reused_tokens": good[0], "suffix_prefill_tokens": good[1],
        "identity_mutations_rejected": len(failures),
        "rewritten_prefix_rejected": True,
        "translated_state_rejected_as_exact": True,
        "verdict": "exact return avoids old-prefix replay only with a complete native anchor",
        "scope": "synthetic protocol assertions, no cache serialization or runtime validation",
    }


def economics_probe():
    # These are assumptions for sensitivity analysis, never measured bandwidth.
    # State copies, allocator, warmup and suffix costs still need real timings.
    rows = []
    for weight_gib in (15.66, 100.05):
        for seconds in (10, 30):
            rows.append({
                "artifact_gib_rounded_from_static_info": weight_gib,
                "total_switch_target_seconds": seconds,
                "minimum_full_read_gib_per_second_if_no_other_cost": weight_gib / seconds,
            })
    return {
        "rows": rows,
        "mapper_bf16_gib_if_300m_parameters": 300_000_000 * 2 / 2**30,
        "dense_score_matrix_gib_64k_16heads_bf16_if_materialized": 65536**2 * 16 * 2 / 2**30,
        "scope": "arithmetic sensitivity only; mmap need not read every byte at activation",
        "implication": "measure first useful output with actual faults; do not extrapolate CUDA mapper latency",
    }


def source_probe(repo):
    files = {
        "qwen": "src/inference/models/qwen35/kv_cache.rs",
        "deepseek": "src/inference/models/deepseek4/cache.rs",
        "deepseek_disk": "src/inference/models/deepseek4/cache/anchor_image.rs",
        "chat": "src/chat/client.rs",
    }
    text = {key: (repo / path).read_text() for key, path in files.items()}
    assert "self.transcript.reset();" in text["chat"].split("fn set_model", 1)[1].split("fn reset", 1)[0]
    assert all(name in text["qwen"] for name in ("conv_state", "recurrent"))
    names = ("window_kv", "compressed_kv", "indexer_kv", "main_kv_state",
             "main_score_state", "indexer_kv_state", "indexer_score_state")
    assert all(name in text["deepseek"] for name in names)
    assert "append-only" in text["deepseek_disk"]
    return {
        "hf2q_head": git_head(repo),
        "files_sha256": {
            files[key]: hashlib.sha256(value.encode()).hexdigest()
            for key, value in text.items()
        },
        "chat_switch_resets_transcript": True,
        "qwen_non_attention_components": ["conv_state", "recurrent"],
        "deepseek_components": names,
        "deepseek_restart_requires_append_only_rows": True,
        "scope": "source assertions at recorded hashes; not a runtime restore test",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swat-checkout", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    result = {
        "schema": "hf2q.model-handoff-research.v1",
        "source_inventory": source_probe(repo),
        "upstream_adapter": adapter_probe(args.swat_checkout),
        "state_information_counterexample": collision_probe(),
        "exact_return_protocol": exact_return_probe(),
        "latency_sensitivity": economics_probe(),
        "not_run": ["learned full-state transport", "model inference", "Metal",
                    "end-to-end client switching", "runtime cache restore", "training"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
