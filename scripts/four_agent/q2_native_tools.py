"""Explicit OpenCode-native protocol policy; never executes a tool or client.
A protocol-chain audit cannot certify native tool execution, quality or primary.
"""
import copy
import hashlib
import json
from pathlib import Path
from q2_client_profile import phase_controls
from qwen38_n1_verdict import DECODE_LIMIT, loads, require

TOOLS = {"edit", "glob", "grep", "question", "read", "shell", "skill", "subagent",
         "webfetch", "websearch", "write", "execute"}
STEPS = ("read", "read", "write", "shell", "read", "shell")
SCHEMA = "hf2q.q2.native-tool-workload-policy.v1"
WORKLOAD = "adr063-opencode-native-parse-port-v1"


def ordered_digest(value):
    """Prompt-significant object/list order. NOT the old sorted semantic hash."""
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_policy(policy):
    require(set(policy) == {"schema", "family", "model", "client_version", "client_binary_sha256",
            "profile_sha256", "tools_order_sha256", "fixture_root", "shell_commands", "workload_id", "task_sha256", "model_sha256"},
            "closed externally frozen native policy")
    require(policy["schema"] == SCHEMA and policy["workload_id"] == WORKLOAD and policy["family"] in {"qwen38", "qwen36", "deepseek4", "gemma4"}
            and isinstance(policy["model"], str) and policy["model"], "explicit native family/model")
    require(isinstance(policy["client_version"], str) and policy["client_version"], "actual version, no wildcard")
    for key in ("client_binary_sha256", "profile_sha256", "tools_order_sha256", "task_sha256", "model_sha256"):
        require(isinstance(policy[key], str) and len(policy[key]) == 64
                and all(c in "0123456789abcdef" for c in policy[key]), "native identity SHA")
    root = Path(policy["fixture_root"])
    require(root.is_absolute() and ".." not in root.parts, "owned absolute fixture locator")
    commands = policy["shell_commands"]
    require(isinstance(commands, list) and len(commands) == 2 and len(set(commands)) == 2
            and all(isinstance(x, str) and 0 < len(x.encode()) <= 4096 and "\x00" not in x for x in commands),
            "two exact external reviewed test commands; never shell-evaluated here")


def validate_profile(profile, policy):
    validate_policy(policy)
    require(profile.get("schema") == "hf2q.q2.production-client-wire-profile.v1"
            and profile.get("model") == policy["model"]
            and profile.get("client_version") == policy["client_version"]
            and profile.get("artifact_sha256") == policy["model_sha256"]
            and profile.get("projection", {}).get("family") == policy["family"]
            and profile.get("client_binary", {}).get("sha256") == policy["client_binary_sha256"],
            "actual captured client/version/model differs")
    require(profile.get("model_inference") is False and profile.get("scripted_stub_responses") is True
            and profile.get("campaign_primary_eligible") is False, "capture is controls-only, not model quality")
    tools = profile.get("actual_tools")
    require(isinstance(tools, list) and len(tools) == 12
            and {t["function"]["name"] for t in tools} == TOOLS
            and ordered_digest(tools) == policy["tools_order_sha256"], "exact ordered native12 schemas")
    require(len(profile.get("phase_controls", [])) == 7 and len(profile.get("raw_request_refs", [])) == 7,
            "actual complete seven-phase profile required")
    return tools


def validate_request(request, profile, policy, phase):
    require(type(phase) is int and 0 <= phase < 7, "native phase")
    validate_profile(profile, policy)
    require(request.get("model") == policy["model"] and request.get("stream") is True,
            "actual production all-SSE lane, no invented unary primary")
    require(ordered_digest(request.get("tools")) == policy["tools_order_sha256"],
            "schema/key-order change affects prompt")
    require(phase_controls(request) == profile["phase_controls"][phase],
            "actual per-phase controls/omissions differ; no default inference")
    require(request.get("max_completion_tokens", request.get("max_tokens")) == 8192,
            "captured full output allowance")
    require(isinstance(request.get("messages"), list) and request["messages"], "actual messages")
    return request


def native_assistant(policy):
    validate_policy(policy)
    root = Path(policy["fixture_root"])
    def validate(message, finish, phase, seen_ids):
        require(isinstance(message, dict) and message.get("role") == "assistant", "native assistant")
        for key in ("content", "reasoning_content"):
            require(message.get(key) is None or isinstance(message[key], str), "actual reasoning/content")
        calls = message.get("tool_calls") or []
        require(isinstance(calls, list), "native tool array")
        total = sum(len((message.get(k) or "").encode()) for k in ("content", "reasoning_content"))
        for item in calls:
            arguments = item.get("function", {}).get("arguments", "")
            require(isinstance(arguments, str), "native raw arguments string")
            total += len(arguments.encode())
        require(total <= 6 * DECODE_LIMIT, "same mature structured-output envelope")
        if phase == 6:
            require(not calls and finish == "stop" and bool((message.get("content") or "").strip()),
                    "native final summary missing/extra tool")
            return None
        require(type(phase) is int and 0 <= phase < 6 and finish == "tool_calls" and len(calls) == 1,
                "one genuine native call per frozen round")
        call = calls[0]; cid = call.get("id")
        require(call.get("type") == "function" and isinstance(cid, str)
                and 0 < len(cid.encode()) <= 256 and cid not in seen_ids, "actual unique native call ID")
        function = call.get("function", {}); name = function.get("name")
        require(name == STEPS[phase], "unsupported/external tool or native workflow drift")
        args = loads(function.get("arguments", "")); require(isinstance(args, dict), "native arguments object")
        if name == "read":
            expected = str(root / ("tests/ports.rs" if phase == 1 else "src/ports.rs"))
            require(args == {"path": expected}, "exact owned full native read, no slicing/clipping")
        elif name == "write":
            require(set(args) == {"path", "content"} and args["path"] == str(root / "src/ports.rs")
                    and isinstance(args["content"], str) and len(args["content"].encode()) <= 65536,
                    "native write only owned source, whole64KiB bound")
        else:
            # This validates a request; it NEVER invokes command/eval/subprocess.
            command = policy["shell_commands"][0 if phase == 3 else 1]
            require(args == {"command": command, "workdir": str(root)},
                    "only exact reviewed foreground compiler commands; no background/timeout override")
        return cid, name, args
    return validate


def extend_actual_history(messages, assistant, call, next_messages):
    """Use actual client result wrapper verbatim; no fabricated three-tool result."""
    prefix = messages + [assistant]
    require(isinstance(next_messages, list) and next_messages[:len(prefix)] == prefix
            and len(next_messages) == len(prefix) + 1, "assistant/reasoning/history changed")
    result = next_messages[-1]
    require(isinstance(result, dict) and set(result) <= {"role", "tool_call_id", "name", "content"}
            and result.get("role") == "tool" and result.get("tool_call_id") == call[0]
            and result.get("name", call[1]) == call[1] and isinstance(result.get("content"), str)
            and len(result["content"].encode()) <= 65536, "whole genuine native tool-result wire join")
    # Content format/execution provenance is deliberately NOT certified here.
    return copy.deepcopy(next_messages), result["content"]
