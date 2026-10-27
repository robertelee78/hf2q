"""Actual-client profile capture boundaries, preserving absent request controls.
Four local-artifact substitutions are explicit derived profiles, never evidence
that the operator already configured all four. No launch/import side effects.
"""
import copy
import hashlib
import json
from urllib.parse import urlsplit
from qwen38_n1_verdict import loads, require

FAMILIES = {"qwen38", "qwen36", "deepseek4", "gemma4"}
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens",
            "max_completion_tokens", "frequency_penalty", "presence_penalty",
            "repetition_penalty", "reasoning_effort", "thinking_budget",
            "thinking_token_budget", "chat_template_kwargs", "hf2q_enable_thinking")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def projected_config(raw, family, model_id, endpoint):
    require(family in FAMILIES and isinstance(model_id, str) and 0 < len(model_id) < 200,
            "explicit local family/model")
    url = urlsplit(endpoint)
    require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port is not None
            and url.path == "/v1" and not url.query and not url.fragment and url.username is None, "loopback endpoint")
    source = loads(raw)
    require(set(source) <= {"$schema", "model", "providers"} and
            isinstance(source.get("model"), str) and source["model"].startswith("hf2q/") and
            set(source.get("providers", {})) == {"hf2q"}, "reviewed actual local configuration shape")
    provider = source["providers"]["hf2q"]
    require(set(provider) == {"name", "package", "settings", "models"} and
            provider["package"] == "aisdk:@ai-sdk/openai-compatible" and
            set(provider["settings"]) <= {"baseURL", "apiKey"}, "unsupported provider/hook settings")
    old_id = source["model"].split("/", 1)[1]
    require(old_id in provider["models"], "selected actual model configuration missing")
    model = copy.deepcopy(provider["models"][old_id])
    require(set(model) <= {"modelID", "name", "capabilities", "compatibility", "cost", "limit", "options", "variants"},
            "unreviewed model configuration extension")
    require(model.get("limit") == {"context": 262144, "output": 8192}, "production context/output changed")
    # Do not import arbitrary provider secrets, plugins, commands, MCP or hooks.
    # Actual model options/variants/capabilities are preserved byte-semantically.
    def no_secrets(value):
        if isinstance(value, dict):
            require(not any(str(k).lower().replace("_", "") in {"apikey", "authorization", "accesstoken", "clientsecret", "cookie", "password", "credentials", "headers"} for k in value), "credential/header in model options")
            for child in value.values(): no_secrets(child)
        elif isinstance(value, list):
            for child in value: no_secrets(child)
    no_secrets(model)
    model["modelID"] = model_id
    model["name"] = model_id + " via hf2q (derived capture profile)"
    result = {"model": "hf2q/" + model_id, "providers": {"hf2q": {
        "name": provider["name"], "package": provider["package"],
        "settings": {"baseURL": endpoint, "apiKey": "q2-loopback-no-credential"},
        "models": {model_id: model}}}}
    return result, {"source_config_sha256": sha(raw), "actual_selected_model": old_id,
                    "family": family, "derived_model_id": model_id,
                    "artifact_substitution_explicit": True, "credential_replaced_before_persistence": True,
                    "safe_projection_sha256": sha(encoded(result)),
                    "configured_all_four_in_original": False}


def phase_controls(request):
    require(isinstance(request, dict), "actual request object")
    return {"present": {k: copy.deepcopy(request[k]) for k in SAMPLING if k in request},
            "absent": [k for k in SAMPLING if k not in request],
            "other_controls": {k: copy.deepcopy(v) for k, v in request.items()
                               if k not in {*SAMPLING, "messages", "tools", "model"}},
            "tools_sha256": sha(encoded(request.get("tools"))),
            "tool_names": [x["function"]["name"] for x in request.get("tools", [])]}


def validate_capture_request(request, model_id, previous_call=None, previous_path=None):
    require(request.get("model") == model_id and request.get("stream") is True,
            "actual selected model/stream drift")
    tools = request.get("tools")
    require(isinstance(tools, list) and len(tools) == 12 and
            len({t["function"]["name"] for t in tools}) == 12, "actual twelve-tool catalog required")
    read = next((t for t in tools if t["function"]["name"] == "read"), None)
    require(read is not None and "path" in read["function"]["parameters"].get("properties", {}),
            "actual built-in read tool shape unknown")
    if previous_call is not None:
        messages = request.get("messages", [])
        # Actual client result formats are retained; do not replace them with
        # custom three-tool fixture messages or guess a renderer normalization.
        calls = [c for m in messages if m.get("role") == "assistant" for c in m.get("tool_calls", [])
                 if c.get("id") == previous_call]
        results = [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == previous_call]
        require(len(calls) == len(results) == 1, "real built-in tool continuation missing/ambiguous")
        call = calls[0]
        require(call["function"]["name"] == "read" and
                loads(call["function"]["arguments"]) == {"path": previous_path}, "scripted owned read drift")
        require(isinstance(results[0].get("content"), str) and "Q2_OWNED_CAPTURE" in results[0]["content"],
                "actual owned file contents absent from tool result")
    return phase_controls(request)


def capture_profile(records, model_id):
    require(len(records) == 7 and [r["phase"] for r in records] == list(range(7)), "whole seven-phase capture")
    names = [r["controls"]["tool_names"] for r in records]
    require(all(n == names[0] for n in names) and
            all(r["controls"]["tools_sha256"] == records[0]["controls"]["tools_sha256"] for r in records),
            "native tool catalog/schema changed between phases")
    return {"schema": "hf2q.q2.production-client-wire-profile.v1", "model": model_id,
            "sampling_lane": "captured_production_client_controls_only",
            "phase_controls": [r["controls"] for r in records], "actual_tools": records[0]["tools"],
            "raw_request_refs": [r["request"] for r in records], "model_inference": False,
            "scripted_stub_responses": True, "actual_builtin_read_calls": 6,
            "coding_quality_qualified": False, "campaign_primary_eligible": False,
            "pending": "independent capture review and native12-tool workload/raw-quality adapter"}
