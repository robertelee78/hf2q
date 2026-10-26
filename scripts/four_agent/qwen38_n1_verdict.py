"""Pure, fail-closed protocol boundaries for the prospective N1 diagnostic."""
import json
import re

WIRE_LIMIT = 256 * 1024
FILE_LIMIT = 64 * 1024
DECODE_LIMIT = 3 * 1024 * 1024
RAW_LIMIT = 32 * 1024 * 1024
PROFILE = "adr063-qwen38-n1-live-v2"
FIXTURE_SHA = "1643af68f5be6118ab2c83a386027564fcd9954ab284d1cc1fbcba544fe053ff"
TOOLS = {"read_file", "write_file", "run_tests"}
WORKFLOW = (
    "Work only in this isolated Rust fixture. The paths are src/ports.rs and "
    "tests/ports.rs. Complete exactly these six steps, one tool call per response, "
    "waiting for its real result before the next step: (1) read_file src/ports.rs; "
    "(2) read_file tests/ports.rs; (3) write_file src/ports.rs with the fix; "
    "(4) run_tests target ports; (5) read_file src/ports.rs to review the final file; "
    "(6) run_tests target all. Then give a final summary without another tool call, "
    "describing the behavior and actual test results. Do not change the tests. "
    "Fix parse_port to reject zero and values above 65535, accept valid u16 ports, "
    "and return a Result<u16, String> for invalid input rather than panic."
)
EXPECTED = [("read_file", {"path": "src/ports.rs"}),
            ("read_file", {"path": "tests/ports.rs"}),
            ("write_file", None), ("run_tests", {"target": "ports"}),
            ("read_file", {"path": "src/ports.rs"}),
            ("run_tests", {"target": "all"})]


class Invalid(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Invalid(reason)


def integer(value, name, maximum=None):
    require(type(value) is int and value >= 0, f"invalid {name}")
    require(maximum is None or value <= maximum, f"excess {name}")
    return value


def loads(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f"duplicate JSON key {key}")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(Invalid(x)))


def wire(request, family="qwen"):
    require(family in ("qwen", "deepseek4", "gemma4"), "unsupported request family")
    require(request.get("max_completion_tokens") == 8192, "output allowance drift")
    require(request.get("tool_choice") == "auto", "tool choice drift")
    require(request.get("logprobs") is False and "response_format" not in request,
            "constraint settings drift")
    for key in ("top_p", "top_k", "min_p", "seed",
                "frequency_penalty", "presence_penalty", "repetition_penalty"):
        require(key not in request, f"captured omission changed: {key}")
    if family in ("qwen", "gemma4"):
        require("temperature" not in request, "captured omission changed: temperature")
        require("reasoning_effort" not in request, "family reasoning control drift")
        if family == "gemma4":
            require(all(k not in request for k in ("chat_template_kwargs", "thinking_budget",
                    "thinking_token_budget", "hf2q_enable_thinking")), "Gemma default reasoning drift")
    else:
        require(type(request.get("temperature")) in (int, float) and
                request["temperature"] == 0, "DeepSeek greedy temperature required")
        require(request.get("reasoning_effort") == "max", "DeepSeek max reasoning required")
        require(all(key not in request for key in ("chat_template_kwargs", "thinking_budget",
                    "thinking_token_budget", "hf2q_enable_thinking")),
                "DeepSeek must use native reasoning controls")
    require(request.get("hf2q_overflow_policy") == "reject", "overflow policy drift")
    require(len(request.get("tools", [])) == 3, "tool schema count drift")
    require({t["function"]["name"] for t in request["tools"]} == TOOLS, "tool names")
    require(all(t["function"].get("strict") is False for t in request["tools"]),
            "strict false required")
    result = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    require(len(result) <= WIRE_LIMIT, "complete request exceeds 256KiB; no truncation")
    return result


def usage(value, turn, engine_kind="hf2q"):
    require(engine_kind in ("hf2q", "peer"), "engine kind")
    if engine_kind == "peer":
        from qwen_n4_peer import usage as peer_usage
        return peer_usage(value, turn)
    require(isinstance(value, dict), "missing usage")
    prompt = integer(value.get("prompt_tokens"), "prompt tokens", 262144)
    completion = integer(value.get("completion_tokens"), "completion tokens", 8192)
    require(integer(value.get("total_tokens"), "total tokens") == prompt + completion,
            "usage total contradiction")
    cached = integer(value.get("prompt_tokens_details", {}).get("cached_tokens"),
                     "cached tokens", prompt)
    require(prompt + 8192 <= 262144, "context plus full output exceeds context")
    require(cached == 0 if turn == 0 else cached > 0, "prefix reuse not proven")
    return value


def assistant(message, finish, turn, seen_ids):
    require(isinstance(message, dict) and message.get("role") == "assistant", "role")
    for key in ("content", "reasoning_content"):
        require(message.get(key) is None or isinstance(message[key], str), key)
    total = sum(len((message.get(k) or "").encode()) for k in ("content", "reasoning_content"))
    calls = message.get("tool_calls") or []
    require(isinstance(calls, list), "tool calls not array")
    total += sum(len(c.get("function", {}).get("arguments", "").encode()) for c in calls)
    # Structured tool serialization can expand raw decoded text; separately bounded raw wire.
    require(total <= 6 * DECODE_LIMIT, "decoded/structured output envelope")
    if turn == 6:
        require(not calls and finish == "stop" and bool((message.get("content") or "").strip()),
                "final summary missing or extra tool continuation")
        return None
    require(finish == "tool_calls" and len(calls) == 1, "exactly one live call required")
    call = calls[0]
    require(call.get("type") == "function", "tool call type")
    cid = call.get("id")
    require(isinstance(cid, str) and 0 < len(cid.encode()) <= 256 and cid not in seen_ids,
            "missing/duplicate/oversized tool id")
    function = call.get("function", {})
    name = function.get("name")
    require(name == EXPECTED[turn][0], "initial six-step workflow not followed")
    args = loads(function.get("arguments", ""))
    require(isinstance(args, dict), "tool arguments not object")
    if name == "write_file":
        require(set(args) == {"path", "content"} and args["path"] == "src/ports.rs"
                and isinstance(args["content"], str), "write arguments")
        require(len(args["content"].encode()) <= FILE_LIMIT, "source exceeds 64KiB")
    else:
        require(args == EXPECTED[turn][1], "tool arguments/workflow drift")
    return cid, name, args


def unary(raw, model, turn, seen, engine_kind="hf2q"):
    obj = loads(raw)
    require(isinstance(obj, dict) and "error" not in obj, "API error")
    require(obj.get("model") == model and isinstance(obj.get("id"), str), "response identity")
    choices = obj.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and choices[0].get("index") == 0,
            "choice count/index")
    message = choices[0].get("message")
    call = assistant(message, choices[0].get("finish_reason"), turn, seen)
    return message, call, usage(obj.get("usage"), turn, engine_kind)


class SSE:
    """Consume every complete SSE data frame, including usage and terminal marker."""
    def __init__(self, model, turn, seen, engine_kind="hf2q"):
        self.engine_kind = engine_kind
        self.model, self.turn, self.seen = model, turn, seen
        self.identity = None
        self.message = {"role": "assistant", "content": "", "reasoning_content": ""}
        self.calls = {}
        self.finish = None
        self.stats = None
        self.done = False
        self.frames = 0

    def feed(self, data):
        require(not self.done, "data after DONE")
        self.frames += 1
        require(self.frames <= 65536, "SSE frame envelope")
        if data == "[DONE]":
            require(self.finish is not None and self.stats is not None, "premature DONE")
            self.done = True
            return False
        obj = loads(data)
        require(isinstance(obj, dict) and "error" not in obj, "SSE API error")
        require(obj.get("model") == self.model and isinstance(obj.get("id"), str), "SSE identity")
        if self.identity is None:
            self.identity = obj["id"]
        require(obj["id"] == self.identity, "changing SSE id")
        choices = obj.get("choices")
        require(isinstance(choices, list), "SSE choices")
        if not choices:
            require(self.finish is not None and self.stats is None, "misordered usage")
            self.stats = usage(obj.get("usage"), self.turn, self.engine_kind)
            return False
        require(self.finish is None and len(choices) == 1 and choices[0].get("index") == 0,
                "duplicate/postfinish choice")
        choice = choices[0]
        delta = choice.get("delta")
        require(isinstance(delta, dict), "SSE delta")
        require(set(delta) <= {"role", "content", "reasoning_content", "tool_calls"}, "unknown delta")
        require(delta.get("role", "assistant") == "assistant", "SSE role")
        semantic = False
        for key in ("content", "reasoning_content"):
            if delta.get(key) is not None:
                require(isinstance(delta[key], str), "nonstring delta")
                self.message[key] += delta[key]
                semantic |= key == "content" and bool(delta[key])
        for part in delta.get("tool_calls") or []:
            require(part.get("index") == 0, "tool index")
            call = self.calls.setdefault(0, {"id": None, "type": "function",
                                             "function": {"name": "", "arguments": ""}})
            if part.get("id") is not None:
                require(call["id"] is None, "repeated tool id field")
                call["id"] = part["id"]
            require(part.get("type", "function") == "function", "delta tool type")
            function = part.get("function", {})
            require(set(function) <= {"name", "arguments"}, "unknown tool delta")
            for key, value in function.items():
                require(isinstance(value, str), "tool delta not string")
                call["function"][key] += value
                semantic |= bool(value)
        if choice.get("finish_reason") is not None:
            self.finish = choice["finish_reason"]
        if obj.get("usage") is not None:
            require(self.finish is not None and self.stats is None, "usage before/duplicate finish")
            self.stats = usage(obj["usage"], self.turn, self.engine_kind)
        return semantic

    def result(self):
        require(self.done, "missing DONE")
        if self.calls:
            self.message["tool_calls"] = [self.calls[0]]
        call = assistant(self.message, self.finish, self.turn, self.seen)
        return self.message, call, self.stats


def test_harness(stdout, expected=2):
    """Exit zero alone is insufficient: consume the real Rust harness result marker."""
    text = stdout.decode("utf8", "strict")
    markers = re.findall(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; "
                         r"(\d+) ignored; (\d+) measured; (\d+) filtered out;.*$", text, re.M)
    require(len(markers) == 1, "missing/duplicate actual test result")
    outcome, passed, failed, ignored, measured, filtered = markers[0]
    require(outcome == "ok" and tuple(map(int, (passed, failed, ignored, measured, filtered))) ==
            (expected, 0, 0, 0, 0), "actual test count/outcome mismatch")
    return expected


def qualify(turns, tools, cleanup, source_hash, final_hash):
    require(len(turns) == 7 and [t["turn"] for t in turns] == list(range(7)), "seven turns")
    require([t["stream"] for t in turns] == [False, True, False, True, False, True, False],
            "transport matrix")
    require(len(tools) == 6 and [t["name"] for t in tools] == [e[0] for e in EXPECTED], "six tools")
    require(all(t.get("cleanup") is True for t in tools) and cleanup is True, "cleanup unknown")
    require(source_hash == final_hash and tools[-1].get("passed") is True, "final tests/source")
    require(tools[-1].get("source_sha256") == final_hash, "stale test artifact")
    return "ProtocolCompleteAwaitingIndependentPatchReview"
