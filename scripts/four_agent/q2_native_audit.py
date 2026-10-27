"""Bounded offline native-client protocol audit, NOT tool/quality certification.
No subprocess/client/tool/HTTP/import side effects. Actual execution sidecars and
independent QE remain required before a production-client primary can qualify.
"""
from pathlib import Path
import hashlib
from q2_certify import BoundReader, replay_response, require_clock
from q2_client_profile import phase_controls
from q2_native_tools import (native_assistant, ordered_digest, validate_profile,
                             validate_request, extend_actual_history)
from qwen38_n1_verdict import loads, require


def audit_native_agent(manifest_path, expected_policy):
    require(Path(manifest_path).is_absolute() and Path(manifest_path).stat().st_size <= 1048576,
            "bounded external native raw manifest")
    manifest = loads(Path(manifest_path).read_bytes())
    require(set(manifest) == {"schema", "profile", "initial_request", "turns"}
            and manifest["schema"] == "hf2q.q2.native-agent-chain.v1", "separate native raw schema")
    reader = BoundReader()
    require(manifest["profile"]["sha256"] == expected_policy["profile_sha256"], "externally frozen source profile")
    profile = reader.obj(manifest["profile"])
    validate_profile(profile, expected_policy)
    # Verify native schemas against EACH raw captured request, not only a copied
    # profile flag or sorted semantic digest. Captures are controls, not quality.
    for phase, ref in enumerate(profile["raw_request_refs"]):
        captured = reader.obj(ref, 2 * 1024 * 1024)
        require(ordered_digest(captured["tools"]) == expected_policy["tools_order_sha256"]
                and phase_controls(captured) == profile["phase_controls"][phase],
                "actual captured native schema/order/omissions join")
    reader.path(profile["client_binary"], 512 * 1024 * 1024)
    initial = reader.raw(manifest["initial_request"], 262144)
    initial_value = loads(initial)
    require(initial_value["messages"][-1].get("role") == "user"
            and isinstance(initial_value["messages"][-1].get("content"), str)
            and hashlib.sha256(initial_value["messages"][-1]["content"].encode()).hexdigest()
                == expected_policy["task_sha256"], "externally frozen actual native coding task")
    require(len(manifest["turns"]) == 7, "native seven actual requests")
    expected_messages, pending, seen, observations = None, None, set(), []
    validate = native_assistant(expected_policy)
    for phase, refs in enumerate(manifest["turns"]):
        require(set(refs) == {"request", "response", "receipt", "assembled"}, "closed native raw turn")
        wire = reader.raw(refs["request"], 262144)
        request = validate_request(loads(wire), profile, expected_policy, phase)
        if phase == 0:
            require(wire == initial, "actual initial wire changed")
        else:
            require(pending is not None, "missing previous native tool call")
            extend_actual_history(expected_messages, pending[0], pending[1], request["messages"])
        frames = []
        raw = reader.raw(refs["response"])
        message, call, usage = replay_response(raw, request, phase, seen, "hf2q", frames,
                                               assistant_policy=validate)
        require(reader.obj(refs["assembled"]) == message, "assembled native message/raw response differ")
        receipt = reader.obj(refs["receipt"])
        require_clock(receipt)
        require(receipt.get("turn") == phase and receipt.get("stream") is True
                and receipt.get("http_status") == 200 and receipt.get("request_bytes") == len(wire)
                and receipt.get("request_sha256") == refs["request"]["sha256"]
                and receipt.get("response_sha256") == refs["response"]["sha256"]
                and receipt.get("response_bytes") == len(raw) and receipt.get("usage") == usage,
                "actual native usage/raw request/response joins")
        from f2_semantic import semantic_timing_v3
        require(len(receipt["semantic_event_monotonic_ns"]) == len(frames)
                and receipt["semantic_timing_v3"] == semantic_timing_v3(
                    receipt["send_monotonic_ns"], receipt["semantic_event_monotonic_ns"],
                    receipt["validated_terminal_monotonic_ns"], True, True),
                "actual native SSE predicate/timestamp clock join")
        observations.append({"phase": phase, "call": call, "usage": usage,
                             "timing": receipt["semantic_timing_v3"]})
        if call is not None:
            seen.add(call[0]); pending = message, call
        else: pending = None
        expected_messages = request["messages"]
    require(pending is None and len(seen) == 6, "six genuine unique calls and final summary")
    reader.verify()
    return {"schema": "hf2q.q2.native-protocol-audit.v1",
            "status": "ProtocolChainAuditedAwaitingNativeExecutionAndIndependentQE",
            "workload_id": expected_policy["workload_id"], "family": expected_policy["family"],
            "profile_sha256": expected_policy["profile_sha256"],
            "tools_order_sha256": expected_policy["tools_order_sha256"], "observations": observations,
            "verified_refs": reader.refs, "actual_client_all_sse": True,
            "native_tool_execution_qualified": False, "coding_quality_qualified": False,
            "campaign_primary_eligible": False, "whole_cohort_trace_qualified": False, "server_identity_qualified": False,
            "effective_sampling_qualified": False,
            "pending": ["actual native filesystem/process/tool-result sidecar capability",
                        "actual compiler artifacts/positive harness counts/final immutable source",
                        "actual client transport clock-source proof", "actual server/readiness/effective-defaults joins", "independent QE",
                        "reviewed full-cohort native trace/window coverage policy"]}
