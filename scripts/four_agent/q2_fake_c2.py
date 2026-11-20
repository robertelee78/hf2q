"""Synthetic approval/log chain for CPU C2 client controls; never a model/Q5 pass."""
import hashlib
import json
from pathlib import Path

import c2_observation
import coding_long_context
import qwen_n4_live_smoke as coordinator

HERE = Path(__file__).resolve().parent


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def context(client_dir, depth, run_id, server_pid):
    assert type(depth) is int and depth in (1, 4)
    client = Path(client_dir).resolve()
    client.mkdir(parents=True, exist_ok=True)
    log = client.parent / "server.stderr"
    with log.open("x") as stream:
        stream.write(" INFO hf2q::serve::api::engine: Qwen35 multi-depth anchor store configured "
                     f"anchor_depth={depth} anchor_bytes_per_slot_max=Some(1048576)\n")
    return c2_observation.cache_control(str(log), client, server_pid, run_id)


def seal(client_dir, summary, control):
    """Materialize explicitly synthetic authority joins around real loopback exchanges."""
    client = Path(client_dir).resolve()
    variant = coding_long_context.select(summary["workload_variant"]["id"])
    names = set(coordinator.MODULES) | set(variant.modules)
    sources = {name: coordinator.child.digest(HERE / name) for name in names}
    observed_sources = {"c2_observation.py": coordinator.child.digest(HERE / "c2_observation.py")}
    request = json.loads((client / "agent-0/turn-0/request.json").read_bytes())
    model, base = request["model"], "http://127.0.0.1:1/v1"
    packet = {
        "schema": coordinator.family_contract.schema("qwen", "coordinator-approval", 4),
        "approved": True, "server_pid": summary["server_pid"], "run_id": summary["run_id"],
        "base_url": base, "model": model, "agent_template": {}, "sources": sources,
        "c2_server_log": control["server_log"], "c2_observation_sources": observed_sources,
        "fixture_scope": "ModelFreeSyntheticApprovalChainNotOperatingApproval",
    }
    approval = client.parent / "client-approval.json"
    write_new(approval, packet)
    summary["c2_cache_control"] = control
    for name, result in summary["agents"].items():
        result["c2_cache_control"] = control
        child = {"schema": coordinator.family_contract.schema("qwen", "agent-approval", 4),
                 "approved": True, "agent_id": name, "run_id": summary["run_id"],
                 "server_pid": summary["server_pid"], "base_url": base, "model": model,
                 "sources": sources, "c2_observation_sources": observed_sources, "c2_cache_control": control}
        child_path = client / f"{name}.approval.json"
        write_new(child_path, child)
        digest = coordinator.child.digest(child_path)
        write_new(client / f"{name}.command.json", {"argv": [
            "ModelFreeQ2Fixture", "--approval", str(child_path), "--approval-sha256", digest]})
        write_new(client / name / "approval-copy.json", child)
        summary["barrier"]["ready"][name] = {"approval_sha256": digest}
        (client / name / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    receipt = {"models_loaded": 0, "verdict": {"verdict": "ModelFreeFixtureNotQ5Pass"}, "client_argv": [
        "ModelFreeQ2Fixture", "--approval", str(approval), "--approval-sha256",
        hashlib.sha256(approval.read_bytes()).hexdigest(), "--output", str(client),
        "--server-pid", str(summary["server_pid"]), "--base-url", base, "--model", model,
        "--c2-server-log", control["server_log"]]}
    write_new(client.parent / "c2-fixture-receipt.json", receipt)
    (client / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return receipt
