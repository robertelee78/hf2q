#!/usr/bin/env python3
"""Deterministic divergent-history variant for ADR-063 C2.

At request turns 3 and 5, replace the tool-result message from turn t-2 with
its first 16 lines followed by the pinned elision marker. The edit is applied
once, immediately before request serialization, and composes with --instance.

    python3 coding_divergent_edit.py estimate
    python3 coding_divergent_edit.py derive-case q5_cases/qwen38.json
"""
import hashlib
import json
from pathlib import Path
import sys

from qwen38_n1_verdict import EXPECTED, WIRE_LIMIT, loads, require, wire

ID = "divergent-edit-v1"
TURNS = (3, 5)
KEEP_LINES = 16
FIXTURE = Path(__file__).with_name("fixtures") / f"{ID}.txt"
FIXTURE_REPO_PATH = f"{{REPO}}/scripts/four_agent/fixtures/{ID}.txt"
FIXTURE_SHA256 = "b6a671e3c52adc8bb0bbdd62584973f3720fd97c7a92b2c2f4eaadbd20849991"
MODULES = ("coding_long_context.py", "coding_divergent_edit.py", "test_coding_divergent_edit.py")
CLIENT = "{REPO}/scripts/four_agent/qwen_n4_live_smoke.py"


class Variant:
    """A message-history transform with no fixture-file replacement."""

    id = ID
    composes_with_instance = True
    modules = MODULES

    def __init__(self, marker, fixture_sha256):
        self.marker = marker
        self.fixture_sha256 = fixture_sha256

    def binding(self):
        return {"workload_variant": self.id, "workload_fixture_sha256": self.fixture_sha256}

    def record(self):
        return {"id": self.id, "fixture_sha256": self.fixture_sha256,
                "edit_turns": list(TURNS), "keep_lines": KEEP_LINES}

    def before_request(self, messages, turn):
        """Mutate the t-2 tool result before request t; return an audit row."""
        if turn not in TURNS:
            return None
        tool_indexes = [index for index, message in enumerate(messages)
                        if message.get("role") == "tool"]
        target_turn = turn - 2
        require(target_turn < len(tool_indexes),
                f"divergent edit turn {turn} lacks tool result for turn {target_turn}")
        index = tool_indexes[target_turn]
        message = messages[index]
        content = message.get("content")
        require(type(content) is str, "divergent edit target tool content must be text")
        lines = content.splitlines()
        edited = "\n".join(lines[:KEEP_LINES] + [self.marker])
        message["content"] = edited
        return {"turn": turn, "target_turn": target_turn, "message_index": index,
                "original_bytes": len(content.encode()), "edited_bytes": len(edited.encode()),
                "original_lines": len(lines), "kept_lines": min(len(lines), KEEP_LINES)}


def select(name=ID, path=FIXTURE):
    require(name == ID, f"unknown divergent workload variant {name!r}")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == FIXTURE_SHA256, "divergent-edit fixture drift")
    marker = raw.decode("utf8", "strict").rstrip("\n")
    require(marker and "divergent-edit-v1" in marker, "divergent-edit marker identity")
    return Variant(marker, digest)


def estimate(variant=None, family="qwen"):
    """Return the seven per-turn request wire sizes after deterministic edits."""
    from coding_long_context import estimate as canonical_estimate
    from qwen38_n1_smoke import build_request, initial_messages
    from qwen38_n1_tools import IMMUTABLE, INITIAL
    from qwen_n4_live_smoke import identity

    if variant is None:
        variant = select()
    fixture = loads((Path(__file__).with_name("fixtures") /
                     "canonical-coding-requests.json").read_bytes())
    initial = fixture[0]["request"]
    messages = initial_messages(initial["messages"], identity("agent-0", "divergent_edit_estimate_0"))
    schemas = initial["tools"]
    for tool in schemas:
        tool["function"]["strict"] = False
    files = {**IMMUTABLE, "src/ports.rs": INITIAL}
    fixed = ("pub fn parse_port(s: &str) -> Result<u16, String> {\n"
             "    let value: u32 = s.parse().map_err(|e| format!(\"invalid port {s:?}: {e}\"))?;\n"
             "    if value == 0 || value > 65535 { return Err(\"out of range\".into()); }\n"
             "    Ok(value as u16)\n}\n")
    report = "\n".join(f"test output line {line:02d}: ok" for line in range(1, 25))
    rows = []
    for turn in range(7):
        edit = variant.before_request(messages, turn)
        body = wire(build_request("Qwen3.8 27B", messages, schemas, turn, family), family)
        rows.append({"turn": turn, "request_bytes": len(body),
                     "prompt_tokens_estimate": len(body) // 4, "edit": edit})
        reasoning = f"Step {turn + 1}. " + "I check the previous result. " * 22
        if turn == 6:
            messages.append({"role": "assistant", "content": "All tests passed.",
                             "reasoning_content": reasoning})
            break
        name, args = EXPECTED[turn]
        if name == "write_file":
            args = {"path": "src/ports.rs", "content": fixed}
            files["src/ports.rs"] = fixed
        cid = f"call_{turn}"
        messages.append({"role": "assistant", "content": "", "reasoning_content": reasoning,
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": name,
                                                      "arguments": json.dumps(args)}}]})
        if name == "read_file":
            result = files[args["path"]]
        elif name == "write_file":
            result = "write complete; sha256=" + hashlib.sha256(fixed.encode()).hexdigest()
        else:
            result = json.dumps({"target": args["target"], "passed": True,
                                 "source_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
                                 "output": report})
        messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": result})
    return {"canonical": canonical_estimate(family=family), ID: rows,
            "wire_limit_bytes": WIRE_LIMIT}


def derive_case(case, name=ID):
    """Derive a separately admitted Q5 case for this workload."""
    import q5_case
    q5_case.validate(case)
    require(name == ID, f"unknown workload variant {name!r}")
    derived = json.loads(json.dumps(case))
    client = derived["client"]
    require(CLIENT in client["argv"] and "--workload" not in client["argv"],
            "case must use the canonical four-agent client without --workload")
    approval = client.get("approval")
    require(approval is not None and "workload_variant" not in approval["set"],
            "case approval already selects a workload")
    sources = approval["set"].get("sources")
    require(isinstance(sources, dict) and set(sources) == {"sha256_of"},
            "case approval must freeze sources")
    client["argv"] += ["--workload", name]
    approval["set"].update({"workload_variant": name,
                             "workload_fixture_sha256": {"sha256_of": FIXTURE_REPO_PATH}})
    sources["sha256_of"] += [f"{{REPO}}/scripts/four_agent/{module}" for module in MODULES]
    derived["workload"]["id"] += f"-{name}"
    if isinstance(derived["workload"].get("client"), str):
        derived["workload"]["client"] += f" --workload {name}"
    derived["notes"] = list(derived.get("notes", [])) + [
        "Derived by coding_divergent_edit.py: request turns 3 and 5 truncate the turn t-2 "
        "tool-result message to 16 lines plus the pinned elision marker."]
    return q5_case.validate(derived)


def main(argv):
    if argv == ["estimate"]:
        print(json.dumps(estimate(), indent=2))
    elif len(argv) == 2 and argv[0] == "derive-case":
        import q5_case
        case = q5_case.loads(Path(argv[1]).read_bytes())
        print(json.dumps(derive_case(case), indent=2, ensure_ascii=False))
    else:
        sys.stderr.write(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
