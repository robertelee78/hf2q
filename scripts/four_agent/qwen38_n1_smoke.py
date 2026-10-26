#!/usr/bin/env python3
"""Live coding agent for N1 or one of four N4 clients; one external server only."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import time
from urllib.parse import urlsplit

from qwen38_n1_verdict import (FIXTURE_SHA, PROFILE, RAW_LIMIT, WORKFLOW, Invalid, SSE,
                              loads, qualify, require, unary, wire)
from qwen38_n1_tools import Tools
import qwen38_n1_process as child
import coding_family as family_contract

PREFLIGHT = {"source_binary_bound", "model_identity_bound", "ctx262144_slots1_verified",
             "auto_tq8_text_only_verified", "profile24g_ack_verified",
             "no_glp_graft_disk_persistence", "greedy_omissions_rep105_thinking2048_512",
             "auto_lazy_tool_grammar_charged", "cpu_owned_io_3g_tool2g_client128m",
             "external_memory_guard_running", "sandbox_confinement_verified"}
INTERRUPTED = False


def interrupted(signum, _frame):
    global INTERRUPTED
    INTERRUPTED = True
    raise KeyboardInterrupt(f"signal {signum}")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def new_output(path):
    path = Path(path).absolute()
    require(not path.exists(), "output must be new")
    for ancestor in (path.parent, *path.parents):
        require(not (ancestor / ".git").exists(), "private output must be outside Git")
    require(path.parent.is_dir() and not path.parent.is_symlink(), "output parent")
    path.mkdir(mode=0o700)
    return path.resolve()


def approval(args):
    require(child.digest(args.approval) == args.approval_sha256, "approval hash")
    packet = loads(Path(args.approval).read_bytes())
    n4 = args.diagnostic_slots == 4
    if n4:
        from qwen_n4_live_smoke import preflight, validate_agent_binding
        validate_agent_binding(packet, args.agent_id, args.run_id, args.server_pid,
                               args.barrier_dir, args.barrier_token, args.output, args.family, args.engine_kind)
    else:
        require(packet.get("schema") == family_contract.schema(args.family, "approval", 1), "approval schema")
        require(packet.get("profile") == family_contract.profile(args.family, 1), "operating packet identity")
    family_contract.validate_selection(args.family, args.engine_kind, args.diagnostic_slots)
    require(packet.get("engine_kind", "hf2q") == args.engine_kind, "engine approval mismatch")
    family_contract.validate_binding(packet, args.family, args.diagnostic_slots)
    require(packet.get("base_url") == args.base_url and packet.get("model") == args.model and
            packet.get("server_pid") == args.server_pid, "server identity drift")
    keys = family_contract.preflight(preflight(PREFLIGHT) if n4 else PREFLIGHT, args.family)
    if args.engine_kind == "peer":
        from qwen_n4_peer import preflight as peer_preflight
        keys = peer_preflight(keys)
    require(set(packet.get("preflight", {})) == keys and
            all(v is True for v in packet["preflight"].values()), "preflight unknown")
    require(packet.get("sandbox_proof_sha256") == args.sandbox_proof_sha256,
            "sandbox proof binding")
    require(packet.get("cargo_sha256") == child.digest(args.cargo), "compiler provenance")
    require(packet.get("sandbox_exec_sha256") == child.digest(args.sandbox_exec), "sandbox executable")
    require(packet.get("fixture_sha256") == FIXTURE_SHA and packet.get("workflow_sha256") ==
            hashlib.sha256(WORKFLOW.encode()).hexdigest(), "new N1 workload identity")
    source_paths = [Path(__file__).with_name(n) for n in (
        "qwen38_n1_smoke.py", "qwen38_n1_tools.py", "qwen38_n1_process.py",
        "qwen38_n1_verdict.py", "test_qwen38_n1_verdict.py",
        "coding_family.py", "test_coding_family.py")]
    if n4:
        source_paths += [Path(__file__).with_name(n) for n in
                         ("qwen_n4_live_smoke.py", "test_qwen_n4_live_smoke.py",
                          "qwen_n4_peer.py", "test_qwen_n4_peer.py")]
    require(packet.get("sources") == {p.name: child.digest(p) for p in source_paths}, "client source drift")
    require(packet.get("approved") is True, "explicit internal operating approval absent")
    proof = child.validate_proof(args.sandbox_proof, args.sandbox_proof_sha256)
    require(proof.get("cargo_sha256") == packet["cargo_sha256"] and
            proof.get("sandbox_exec_sha256") == packet["sandbox_exec_sha256"],
            "sandbox capability executable mismatch")
    sdk = Path(packet.get("sdk_root", "")).resolve()
    require(sdk.is_absolute() and sdk.name.endswith(".sdk") and
            str(sdk).startswith(("/Library/Developer/", "/Applications/")), "exact trusted SDK root")
    require(proof.get("sdk_root") == str(sdk) and
            proof.get("toolchain_root") == str(Path(args.cargo).resolve().parent.parent),
            "sandbox toolchain/SDK proof mismatch")
    os.kill(args.server_pid, 0)
    return packet


def request_timeout(_signum, _frame):
    raise TimeoutError("absolute request deadline")


def build_request(model, messages, schemas, turn, family="qwen"):
    """Actual shared request construction, before the exchange wire boundary."""
    request = {"model": model, "messages": messages, "tools": schemas,
               "tool_choice": "auto", "max_completion_tokens": 8192,
               "logprobs": False, "hf2q_overflow_policy": "reject", "stream": turn % 2 == 1}
    family_contract.request_controls(request, family)
    if request["stream"]:
        request["stream_options"] = {"include_usage": True}
    return request


def exchange(base, request, output, turn, seen, deadline=900, engine_kind="hf2q", family="qwen"):
    raw_request = wire(request, family)
    (output / "request.json").write_bytes(raw_request)
    endpoint = urlsplit(base)
    require(endpoint.scheme == "http" and endpoint.hostname == "127.0.0.1" and endpoint.port and
            endpoint.path == "/v1" and not endpoint.query and not endpoint.fragment and
            endpoint.username is None, "only bound loopback /v1 endpoint allowed")
    connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=900)
    started = time.monotonic()
    previous_alarm = signal.signal(signal.SIGALRM, request_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, deadline)
    record = {"turn": turn, "stream": request["stream"], "request_bytes": len(raw_request),
              "request_sha256": hashlib.sha256(raw_request).hexdigest(), "semantic_ttft_seconds": None,
              "first_reasoning_seconds": None, "first_frame_seconds": None}
    total = 0
    try:
        require(not INTERRUPTED, "interrupted before request")
        connection.request("POST", "/v1/chat/completions", raw_request,
                           {"Content-Type": "application/json", "Connection": "close"})
        response = connection.getresponse()
        record["http_status"] = response.status
        record["headers"] = response.getheaders()
        save(output / "response-headers.json", record)
        require(response.status == 200, "API/context reject; no retry or summary")
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip()
        require(content_type == ("text/event-stream" if request["stream"] else "application/json"),
                "response media type")
        with (output / "response.bin").open("xb") as archive:
            if not request["stream"]:
                chunks = []
                while True:
                    require(time.monotonic() - started <= deadline and not INTERRUPTED, "request deadline")
                    if connection.sock is not None:
                        connection.sock.settimeout(max(0.001, deadline - (time.monotonic() - started)))
                    block = response.read1(16384)
                    require(time.monotonic() - started <= deadline, "absolute postread deadline")
                    if not block:
                        break
                    archive.write(block)
                    total += len(block)
                    require(total <= RAW_LIMIT, "response wire32MiB envelope")
                    chunks.append(block)
                message, call, stats = unary(b"".join(chunks), request["model"], turn, seen, engine_kind)
            else:
                parser = SSE(request["model"], turn, seen, engine_kind)
                frame, pending = [], bytearray()
                while True:
                    require(time.monotonic() - started <= deadline and not INTERRUPTED, "request deadline")
                    block = response.read1(16384)
                    require(time.monotonic() - started <= deadline, "absolute postread deadline")
                    if not block:
                        break
                    archive.write(block)
                    total += len(block)
                    require(total <= RAW_LIMIT, "SSE raw32MiB envelope")
                    pending.extend(block)
                    while b"\n" in pending:
                        raw_line, _, rest = pending.partition(b"\n")
                        pending = bytearray(rest)
                        line = raw_line[:-1] if raw_line.endswith(b"\r") else raw_line
                        if line.startswith(b":"):
                            continue
                        if line:
                            require(line.startswith(b"data: "), "unexpected SSE field")
                            frame.append(line[6:])
                        else:
                            require(len(frame) <= 1, "unexpected multiline SSE data")
                            if not frame:
                                continue
                            elapsed = time.monotonic() - started
                            if record["first_frame_seconds"] is None:
                                record["first_frame_seconds"] = elapsed
                            before = len(parser.message["reasoning_content"])
                            semantic = parser.feed(frame[0].decode("utf8", "strict"))
                            if semantic and record["semantic_ttft_seconds"] is None:
                                record["semantic_ttft_seconds"] = elapsed
                            if len(parser.message["reasoning_content"]) > before and record["first_reasoning_seconds"] is None:
                                record["first_reasoning_seconds"] = elapsed
                            frame = []
                require(not frame and not pending, "incomplete SSE line/frame")
                message, call, stats = parser.result()
                require(record["semantic_ttft_seconds"] is not None, "no semantic SSE progress")
        record.update({"usage": stats, "elapsed_seconds": time.monotonic() - started,
                       "response_bytes": total, "response_sha256": child.digest(output / "response.bin")})
        save(output / "assembled-assistant.json", message)
        save(output / "receipt.json", record)
        return message, call, record
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("base-url", "model", "output", "cargo", "toolchain", "sandbox-exec",
                   "sandbox-proof", "sandbox-proof-sha256", "approval", "approval-sha256"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--server-pid", required=True, type=int)
    parser.add_argument("--diagnostic-slots", type=int, choices=(1, 4), default=1)
    for name in ("agent-id", "run-id", "barrier-dir", "barrier-token"):
        parser.add_argument("--" + name)
    parser.add_argument("--family", choices=family_contract.FAMILIES, default="qwen")
    parser.add_argument("--engine-kind", choices=("hf2q", "peer"), default="hf2q")
    args = parser.parse_args()
    family_contract.validate_selection(args.family, args.engine_kind, args.diagnostic_slots)
    os.umask(0o077)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    output = new_output(args.output)
    summary = {"schema": family_contract.schema(args.family, "result", 1),
               "profile": family_contract.profile(args.family, 1), "family": args.family,
               "status": "Pending", "cleanup": False, "model_qualified": False,
               "turns": [], "tools": [], "independent_patch_review_required": True}
    n4 = args.diagnostic_slots == 4
    if n4:
        from qwen_n4_live_smoke import PROFILE as n4_profile
        summary.update(schema=family_contract.schema(args.family, "agent-result", 4),
                       profile=family_contract.profile(args.family, 4),
                       agent_id=args.agent_id, run_id=args.run_id, server_pid=args.server_pid,
                       diagnostic_slots=4, full_prefix_reuse="PendingPostOracle", barrier={"status": "Pending"})
    tools = None
    try:
        packet = approval(args)
        summary["engine_kind"] = args.engine_kind
        if args.family == "deepseek4": summary["family_controls"] = packet["family_controls"]
        save(output / "approval-copy.json", packet)
        require(Path(args.cargo).is_absolute() and Path(args.cargo).is_file(), "verified absolute cargo")
        require(Path(args.sandbox_exec).is_absolute(), "verified absolute sandbox executable")
        require(args.toolchain == packet.get("toolchain"), "toolchain identity")
        fixture_path = Path(__file__).with_name("fixtures") / "canonical-coding-requests.json"
        require(child.digest(fixture_path) == FIXTURE_SHA, "canonical fixture drift")
        fixture = loads(fixture_path.read_bytes())
        initial = fixture[0]["request"]
        messages = [dict(m) for m in initial["messages"]]
        require(len(messages) == 2 and messages[-1]["role"] == "user", "initial task")
        if n4:
            from qwen_n4_live_smoke import identity
            prompt_identity = identity(args.agent_id, args.run_id)
            messages[0]["content"] = prompt_identity + "\n\n" + messages[0]["content"]
            summary["initial_identity_sha256"] = hashlib.sha256(prompt_identity.encode()).hexdigest()
        messages[-1]["content"] += "\n\n" + WORKFLOW
        initial_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                                               separators=(",", ":")).encode()).hexdigest()
        summary["initial_messages_sha256"] = initial_hash
        schemas = initial["tools"]
        for tool in schemas:
            tool["function"]["strict"] = False
        tools = Tools(output / "tools", args.cargo, args.sandbox_exec, packet["sdk_root"])
        seen = set()
        if n4:
            from qwen_n4_live_smoke import wait_barrier
            save(output / "summary.json", summary)
            summary["barrier"] = wait_barrier(args.barrier_dir, args.agent_id, args.run_id,
                                               args.server_pid, args.barrier_token, output,
                                               initial_hash, args.approval_sha256)
        started = time.monotonic()
        for turn in range(7):
            require(not INTERRUPTED and time.monotonic() - started < 7200, "whole client deadline")
            os.kill(args.server_pid, 0)
            request = build_request(args.model, messages, schemas, turn, args.family)
            turn_output = output / f"turn-{turn}"
            turn_output.mkdir(mode=0o700)
            message, call, receipt = exchange(args.base_url, request, turn_output, turn, seen,
                                               deadline=min(900, 7200 - (time.monotonic() - started)), engine_kind=args.engine_kind, family=args.family)
            summary["turns"].append(receipt)
            messages.append(message)
            if call is not None:
                cid, name, arguments = call
                seen.add(cid)
                result = tools.invoke(name, arguments)
                messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": result})
                summary["tools"] = tools.receipts
            save(output / "transcript.json", messages)
            save(output / "summary.json", summary)
        tools.invariant()
        final_hash = child.digest(tools.fixture / "src/ports.rs")
        summary["status"] = qualify(summary["turns"], tools.receipts, True,
                                    tools.source_hash, final_hash)
        summary["final_source_sha256"] = final_hash
        summary["workflow_sha256"] = hashlib.sha256(WORKFLOW.encode()).hexdigest()
        summary["full_prefix_reuse"] = "PendingPostOracle"
        summary["runtime_qualified"] = False
    except BaseException as exc:
        summary["status"] = "Failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if tools is not None:
            tools.close()
        summary["tools"] = [] if tools is None else tools.receipts
        # No server ownership: outer supervisor alone proves server/listener teardown.
        summary["cleanup"] = all(x.get("cleanup") is True for x in summary["tools"])
        summary["server_cleanup_owned_by_outer_supervisor"] = True
        save(output / "summary.json", summary)
    return 0 if summary["status"] == "ProtocolCompleteAwaitingIndependentPatchReview" else 1


if __name__ == "__main__":
    raise SystemExit(main())
