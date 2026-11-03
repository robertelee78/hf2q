#!/usr/bin/env python3
"""Offline token-prefix survey, including synthetic generation and tool boundaries.

No weights, inference, live clients or GGUF-native tokenizer parity. DeepSeek
rendering executes hf2q's unchanged Rust encoder; Qwen uses its source Jinja.
Tokenization uses explicitly supplied local HF tokenizer.json files. Synthetic
emitted token sequences are mechanics controls, not observed model generations.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import tokenizers
import transformers
from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

from model_handoff_render_probe import compare, render_deepseek


def token_compare(old, new):
    common = 0
    for left, right in zip(old, new):
        if left != right:
            break
        common += 1
    return {
        "old_tokens": len(old), "new_tokens": len(new),
        "common_prefix_tokens": common,
        "surviving_token_fraction": common / len(old) if old else 1.0,
        "is_token_prefix": new[:len(old)] == old,
        "first_old_mismatch": old[common:common + 8],
        "first_new_mismatch": new[common:common + 8],
    }


def fixtures():
    base = [
        {"role": "system", "content": "Keep exact src/λ.rs, identifier id_007 and every tool result."},
        {"role": "user", "content": "Inspect src/λ.rs; preserve `let id_007 = 42;`."},
    ]
    for name, content, reasoning in (
        ("text", "Keep `let id_007 = 42;` in src/λ.rs.", "TRACE_007"),
        ("whitespace", "\n  Keep `let id_007 = 42;` in src/λ.rs.  \n", "  TRACE_007  "),
        ("tool_result", "", "TRACE_007"),
    ):
        assistant = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if name == "tool_result":
            assistant["tool_calls"] = [{
                "id": "call_007", "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path":"src/λ.rs"}'}}]
            suffix = [{"role": "tool", "tool_call_id": "call_007", "content": "let id_007 = 42;\n"}]
        else:
            suffix = [{"role": "user", "content": "Continue; keep the exact code and identifier."}]
        yield name, base, assistant, suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-source", type=Path, required=True)
    parser.add_argument("--deepseek-source", type=Path, required=True)
    args = parser.parse_args()
    template_file = args.qwen_source / "chat_template.jinja"
    template = _compile_jinja_template(template_file.read_text())
    sources = {"qwen": args.qwen_source, "deepseek": args.deepseek_source}
    engines = {name: Tokenizer.from_file(str(path / "tokenizer.json")) for name, path in sources.items()}
    tool = {"type": "function", "function": {"name": "read_file", "description": "Read a source file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    work = []
    deepseek_requests = []
    for family in sources:
        for tools in (False, True):
            for preserve in ((False, True) if family == "qwen" else (None,)):
                for fixture, base, assistant, suffix in fixtures():
                    if fixture == "tool_result" and not tools:
                        continue
                    ledgers = [copy.deepcopy(base), copy.deepcopy(base + [assistant]), copy.deepcopy(base + [assistant] + suffix)]
                    row = {"family": family, "tools": tools, "preserve_thinking": preserve, "fixture": fixture}
                    if family == "qwen":
                        # hf2q's serving boundary also parses OpenAI argument JSON
                        # before passing it to the Qwen source template.
                        for ledger in ledgers:
                            for message in ledger:
                                for call in message.get("tool_calls", []):
                                    call["function"]["arguments"] = json.loads(call["function"]["arguments"])
                        row["rendered"] = [template.render(messages=ledger, tools=[tool] if tools else [],
                            add_generation_prompt=index != 1, enable_thinking=True, preserve_thinking=preserve)
                            for index, ledger in enumerate(ledgers)]
                    else:
                        for ledger in ledgers:
                            if tools:
                                ledger[0]["tools"] = [tool]
                        row["render_index"] = len(deepseek_requests)
                        deepseek_requests.extend(ledgers)
                    row["assistant"] = assistant
                    work.append(row)
    deepseek_rendered, lock_sha = render_deepseek(deepseek_requests)
    results = []
    for row in work:
        family = row["family"]
        if family == "deepseek":
            index = row.pop("render_index")
            row["rendered"] = deepseek_rendered[index:index + 3]
        prompt, completed, following = row.pop("rendered")
        assistant = row.pop("assistant")
        encode = lambda text: engines[family].encode(text, add_special_tokens=False).ids
        prompt_ids, completed_ids, following_ids = map(encode, (prompt, completed, following))
        if row["fixture"] == "tool_result":
            # Native tool spelling comes from the encoder, not a second port.
            assert completed.startswith(prompt), "tool fixture cannot extend its generation prompt"
            emitted = completed[len(prompt):]
            row["emission_source"] = "synthetic template-derived tool completion"
        elif family == "qwen":
            emitted = assistant["reasoning_content"] + "\n</think>\n\n" + assistant["content"] + "<|im_end|>\n"
            row["emission_source"] = "synthetic raw Qwen completion retaining whitespace"
        else:
            emitted = assistant["reasoning_content"] + "</think>" + assistant["content"] + "<｜end▁of▁sentence｜>"
            row["emission_source"] = "synthetic raw DeepSeek completion retaining whitespace"
        generated_ids = prompt_ids + encode(emitted)
        # Verify the synthetic IDs actually encode the claimed text, including
        # special tokens. This is not a native model's generation receipt.
        assert engines[family].decode(generated_ids, skip_special_tokens=False) == prompt + emitted
        row.update({
            "completed_render_to_next": {**compare(completed, following), **token_compare(completed_ids, following_ids)},
            "pre_generation_anchor_to_next": token_compare(prompt_ids, following_ids),
            "synthetic_emitted_history_to_next": token_compare(generated_ids, following_ids),
            "synthetic_emitted_history_to_completed_render": token_compare(generated_ids, completed_ids),
            "joint_retokenization_matches_emitted_ids": encode(prompt + emitted) == generated_ids,
            "render_sha256": {key: hashlib.sha256(text.encode()).hexdigest() for key, text in (
                ("prompt", prompt), ("completed", completed), ("following", following))},
        })
        results.append(row)
    root = Path(__file__).resolve().parents[2]
    paths = {"qwen_template": template_file, **{family + "_tokenizer": path / "tokenizer.json" for family, path in sources.items()}}
    for name in ("mod.rs", "types.rs", "parser.rs"):
        paths["deepseek_encoder_" + name] = root / "src/core/deepseek_v4_encoding" / name
    print(json.dumps({
        "schema": "hf2q.model-handoff-prefix-probe.v1",
        "scope": __doc__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "isolated_cargo_lock_sha256": lock_sha, "cases": results,
    }, indent=2))


if __name__ == "__main__":
    main()
