"""Re-derive C2 observations from raw files; external owned guard verdicts remain separate."""
import json
from pathlib import Path
import re

import c2_metrics
import c2_observation as observation
import coding_long_context
import q2_run_audit
import q5_admission
import q5_case
import quality_campaign as qc
import q5_calibration
import q5_policy
import qwen_n4_live_smoke as coordinator

HERE = Path(__file__).resolve().parent


def require(ok, reason):
    observation.require(ok, reason)


def load(path):
    return json.loads(Path(path).read_bytes())


def bound_json(reference):
    require(isinstance(reference, dict) and isinstance(reference.get("path"), str) and
            isinstance(reference.get("sha256"), str), "source-bound JSON path and sha256 required")
    path = Path(reference["path"])
    require(path.is_absolute() and observation.sha(path) == reference["sha256"], "bound evidence changed")
    return load(path)


def run_identity(run, receipt, case, identity, host, overrides, repo_root, admission_reference=None):
    """Bind the actual launch, complete client-source packet and calibrated live cap."""
    run = Path(run).resolve()
    artifact = identity.get("artifact") or {}
    require(re.fullmatch(r"[0-9a-f]{64}", artifact.get("sha256") or "") is not None and
            (receipt.get("artifact") or {}).get("sha256") == artifact["sha256"],
            "C2 requires the exact artifact SHA256, not size/mtime identity")
    require((receipt.get("binary") or {}).get("sha256") == identity["binary"]["sha256"] and
            receipt.get("host") == host and receipt.get("workload") == case["workload"],
            "C2 binary, complete host identity or workload differs")
    require(bound_json(receipt.get("case")) == case, "C2 recorded case differs")
    expected = {name: observation.sha(HERE / name) for name in q5_calibration.SOURCES}
    require(receipt.get("supervisor_sources") == expected, "C2 supervisor sources differ")
    require(receipt.get("inputs") == identity["inputs"], "C2 client/input source files differ")

    packet = load(run / "client-approval.json")
    sources = packet.get("sources")
    names = set(coordinator.MODULES)
    argv = case["client"]["argv"]
    if "--workload" in argv:
        names.update(coding_long_context.select(argv[argv.index("--workload") + 1]).modules)
    require(isinstance(sources, dict) and names <= sources.keys() and
            all(isinstance(name, str) and Path(name).name == name for name in sources),
            "complete coordinator/transitive source packet required")
    expected_sources = {name: observation.sha(HERE / name) for name in sources}
    require(sources == expected_sources, "C2 rendered coordinator transitive sources differ")
    for agent in coordinator.AGENTS:
        agent_packet = load(run / "client" / f"{agent}.approval.json")
        require(agent_packet.get("sources") == sources, "C2 retained agent transitive sources differ")
    observation.activation(packet, str(run / "server.stderr"), "qwen", "hf2q", None)

    builtins = q5_calibration.builtins(run)
    builtins["REPO"] = repo_root
    planned = q5_case.build_plan(case, builtins, qc.arm_environ({}, overrides))
    require(planned["complete"] and receipt.get("serve_argv") == planned["serve"]["argv"] and
            receipt.get("serve_env") == planned["serve"]["env"], "C2 resolved serving controls differ")
    policy = planned["policy"]
    if admission_reference is None:
        require(receipt.get("run_kind") == "calibration", "fresh C2 baseline must be a calibration run")
    else:
        prior = bound_json(admission_reference)
        admission = q5_admission.admission_from(prior, {
            "family": case["family"], "workload": case["workload"],
            "binary_sha256": identity["binary"]["sha256"], "artifact": artifact,
            "physical_bytes": host["physical_bytes"]})
        floor_inputs = receipt["floor_inputs"]
        require(floor_inputs.get("seed_rate_bytes_per_s") == admission["seed_rate_bytes_per_s"],
                "C2 admission floor seed differs")
        initial = q5_policy.effective_floor(policy["free_cached_floor_bytes"],
                                           admission["seed_rate_bytes_per_s"],
                                           policy["interval_seconds"], floor_inputs["latency_ns"])
        decision = q5_admission.admit(admission, receipt["baseline"]["free_plus_cached_bytes"], initial)
        require(prior.get("host") == host and receipt.get("run_kind") == "admitted" and
                decision["admitted"] is True and receipt.get("admission") ==
                {**admission, "receipt": admission_reference["path"],
                 "receipt_sha256": admission_reference["sha256"], "decision": decision},
                "C2 measured cohort lacks its exact admitted calibration/decision")
        require(receipt.get("growth_cap") == q5_admission.growth_cap(
            admission, host["physical_bytes"], policy["free_cached_floor_bytes"]),
            "C2 calibrated live growth cap differs")
        policy = q5_policy.effective_policy({**policy, "footprint_cap_bytes": admission["footprint_cap_bytes"]})
    require(receipt.get("policy") == policy, "C2 effective runtime policy differs")


def client_observation(run, summary):
    """Replay both HTTP snapshots, tagged dispatches and the exact original stderr slice."""
    run = Path(run).resolve()
    client = run / "client"
    reference = summary.get("c2_observation")
    require(isinstance(reference, dict) and Path(reference.get("path", "")).resolve() == client / "c2-observation.json",
            "C2 observation must belong to this owned client output")
    record = bound_json(reference)
    require(record.get("schema") == observation.SCHEMA and record.get("status") == "Observed" and
            record.get("run_id") == summary.get("run_id") and record.get("server_pid") == summary.get("server_pid") and
            record.get("collector_sha256") == observation.sha(HERE / "c2_observation.py"),
            "C2 observation identity/source/status mismatch")
    require(record.get("c2_cache_control") == summary.get("c2_cache_control"),
            "C2 startup/dispatch observation context differs")
    before = observation.metrics((client / "c2-before.prom").read_text())
    after = observation.metrics((client / "c2-after.prom").read_text())
    require(record.get("before") == before and record.get("after") == after, "C2 metric snapshot contradiction")
    trace = client / "c2-trace.log"
    require(record.get("trace_sha256") == observation.sha(trace), "C2 retained trace changed")
    start, end = record.get("start_offset"), record.get("end_offset")
    require(type(start) is int and type(end) is int and 0 <= start <= end and
            end - start == trace.stat().st_size and Path(record.get("server_log", "")).resolve() == run / "server.stderr",
            "C2 original log extent mismatch")
    counts = {key: 0 for key in observation.COUNTERS.values()}
    routes = {}
    with trace.open("rb") as retained, (run / "server.stderr").open("rb") as original:
        original.seek(start)
        for raw in retained:
            require(original.read(len(raw)) == raw and raw.endswith(b"\n"), "C2 trace is not original complete log data")
            row = observation.trace_row(raw.decode("utf-8"))
            if row is not None:
                for key in counts:
                    counts[key] += row[key]
                route = row["route"] + ":" + row["path"]
                routes[route] = routes.get(route, 0) + row["joint_lane_steps"] + row["serial_lane_steps"]
    actual = observation.eligibility(before, after, counts)
    require(actual == record.get("eligibility") == reference.get("eligibility") and
            routes == record.get("route_lane_steps"), "C2 derived eligibility/route contradiction")
    return actual, routes


def configured_depth(run):
    with observation.regular_evidence(Path(run) / "server.stderr") as stream:
        return observation.scan_anchor_config(stream)[0]


def run_evidence(run, receipt, summary, depth, kind):
    require(configured_depth(run) == depth, "C2 requested depth differs from actual worker depth")
    if kind == "divergent":
        require(type(summary.get("c2_cache_control")) is dict and
                summary["c2_cache_control"].get("anchor_depth") == depth,
                "C2 divergent run requires its actual observed arm context")
    audit, failure = q2_run_audit.check_run(receipt, Path(run) / "client", summary, None, "hf2q", require_width=True)
    require(failure is None, "C2 raw client audit failed: " + str(failure))
    eligibility, routes = client_observation(run, summary)
    extractor = c2_metrics.divergent_metrics if kind == "divergent" else c2_metrics.append_metrics
    return {"metrics": extractor(summary), "eligibility": eligibility, "routes": routes,
            "peak_footprint_bytes": q5_admission.host_charge(receipt)["peak_footprint_bytes"],
            "audit": audit, "observation_sha256": summary["c2_observation"]["sha256"],
            "server_stderr_sha256": observation.sha(Path(run) / "server.stderr")}


def calibration(reference, case, identity, host, depth, kind, overrides, repo_root):
    """No age guess: fresh means the exact source/binary/artifact/workload/footprint and host."""
    receipt = bound_json(reference)
    q5_admission.admission_from(receipt, {
        "family": case["family"], "workload": case["workload"],
        "binary_sha256": identity["binary"]["sha256"], "artifact": identity["artifact"],
        "physical_bytes": host["physical_bytes"]})
    run = Path(reference["path"]).parent
    run_identity(run, receipt, case, identity, host, overrides, repo_root)
    summary = load(run / "client/summary.json")
    data = run_evidence(run, receipt, summary, depth, kind)
    require(all(value == 0 for value in data["eligibility"]["not_eligible"].values()),
            "calibration observed anchor-caused joint refusal")
    return {"reference": reference, "peak_footprint_bytes": data["peak_footprint_bytes"],
            "binary_sha256": identity["binary"]["sha256"], "artifact": identity["artifact"],
            "workload": case["workload"], "observation_sha256": data["observation_sha256"],
            "routes": data["routes"]}
