#!/usr/bin/env python3
"""Offline template stability probe; no model weights, Metal or live clients.

Executes hf2q's actual Rust DeepSeek encoder in an isolated minimal Cargo
package, and the supplied Qwen source Jinja with Transformers' template helper.
Reports UTF-8 byte-prefix survival, NOT token-prefix survival or live-client
compatibility. Runtime/client/tokenizer qualification remains required.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from transformers.utils.chat_template_utils import _compile_jinja_template


def compare(old, new):
    a, b = old.encode(), new.encode()
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return {
        "old_bytes": len(a), "new_bytes": len(b), "common_prefix_bytes": common,
        "surviving_byte_fraction": common / len(a), "is_byte_prefix": b.startswith(a),
        "old_reasoning_present": "PRIVATE_REASONING_MARKER" in old,
        "new_reasoning_present": "PRIVATE_REASONING_MARKER" in new,
    }


def render_deepseek(requests):
    """Execute the unchanged native encoder, with one isolated Cargo target."""
    root = Path(__file__).resolve().parents[2]
    module = root / "src/core/deepseek_v4_encoding/mod.rs"
    rust = r'''
#![allow(dead_code, unused_imports)]
#[path = MODULE_PATH]
mod encoder;
fn main() {
    let requests: Vec<serde_json::Value> =
        serde_json::from_reader(std::io::stdin()).unwrap();
    let rows: Vec<String> = requests.iter().map(|value| {
        let opts = encoder::EncodeOptions {
            thinking_mode: encoder::ThinkingMode::Thinking,
            ..Default::default()
        };
        encoder::encode_json(&value.to_string(), opts).unwrap()
    }).collect();
    println!("{}", serde_json::to_string(&rows).unwrap());
}
'''.replace("MODULE_PATH", json.dumps(str(module)))
    with tempfile.TemporaryDirectory(prefix="hf2q-render-probe-") as directory:
        package = Path(directory)
        (package / "src").mkdir()
        (package / "Cargo.toml").write_text('''[package]
name = "hf2q-handoff-render-probe"
version = "0.0.0"
edition = "2021"
[dependencies]
serde = { version = "=1.0.228", features = ["derive"] }
serde_json = { version = "=1.0.149", features = ["preserve_order"] }
thiserror = "=2.0.18"
''')
        (package / "src/main.rs").write_text(rust)
        subprocess.run(["cargo", "generate-lockfile", "--offline"], cwd=package, check=True)
        completed = subprocess.run(
            ["cargo", "run", "--offline", "--locked", "--quiet", "--target-dir", str(package / "target")],
            cwd=package, input=json.dumps(requests), text=True, capture_output=True, check=True,
        )
        return json.loads(completed.stdout), hashlib.sha256((package / "Cargo.lock").read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-template", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    module = root / "src/core/deepseek_v4_encoding/mod.rs"
    messages = [
        {"role": "system", "content": "Use exact file src/a.rs and preserve id 007."},
        {"role": "user", "content": "Inspect the bug."},
        {"role": "assistant", "reasoning_content": "PRIVATE_REASONING_MARKER",
         "content": "The next step is testing."},
    ]
    following = messages + [{"role": "user", "content": "Continue with the fix."}]
    tool = {"type": "function", "function": {"name": "read_file",
            "description": "Read a source file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"]}}}
    cases = {}
    template = _compile_jinja_template(args.qwen_template.read_text())
    for preserve in (True, False):
        options = dict(add_generation_prompt=False, enable_thinking=True,
                       preserve_thinking=preserve, tools=[tool])
        before = template.render(messages=messages, **options)
        after = template.render(messages=following, **options)
        cases[f"qwen_preserve_thinking_{str(preserve).lower()}"] = compare(before, after)

    requests = []
    for tools in (False, True):
        for messages_to_render in (messages, following):
            copied = json.loads(json.dumps(messages_to_render))
            if tools:
                copied[0]["tools"] = [tool]
            requests.append(copied)
    rendered, lock_sha = render_deepseek(requests)
    for index, tools in enumerate((False, True)):
        cases[f"deepseek_thinking_tools_{str(tools).lower()}"] = compare(
            rendered[index * 2], rendered[index * 2 + 1])
    assert cases["qwen_preserve_thinking_true"]["is_byte_prefix"]
    assert not cases["qwen_preserve_thinking_false"]["is_byte_prefix"]
    assert not cases["deepseek_thinking_tools_false"]["is_byte_prefix"]
    assert cases["deepseek_thinking_tools_true"]["is_byte_prefix"]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [args.qwen_template, module, module.parent / "types.rs", module.parent / "parser.rs"]}
    print(json.dumps({
        "schema": "hf2q.model-handoff-render-probe.v1", "cases": cases,
        "source_sha256": hashes, "isolated_cargo_lock_sha256": lock_sha,
        "scope": "actual Rust DeepSeek encoder and source Qwen Jinja; bytes only, not actual client requests or tokenization",
    }, indent=2))


if __name__ == "__main__":
    main()
