"""Join quality evidence to the frozen arm's actual Q5 launch."""
import hashlib
import json
from pathlib import Path

import q5_calibration
import q5_case


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_json(path, digest):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "quality launch input changed")
    return json.loads(raw)


def validate(case, overrides, record, receipt, summary, summary_path):
    """Rebuild the scheduled arm, using only its launch-bound runtime identities."""
    output = Path(record["output"]).resolve()
    case_path = Path(record["case"]).resolve()
    require(Path(receipt["case"]["path"]).resolve() == case_path and
            receipt["case"]["sha256"] == record["case_sha256"], "quality receipt case differs")
    require(bound_json(case_path, record["case_sha256"]) == case, "quality derived case differs")
    expected_summary = Path(q5_case.expand(case["client"]["summary"], {"OUT": str(output)})).resolve()
    require(Path(summary_path).resolve() == expected_summary and
            Path(receipt["client"]["summary_path"]).resolve() == expected_summary,
            "quality summary is not the launched client's output")

    approval_path = output / "client-approval.json"
    approval_raw = approval_path.read_bytes()
    approval_sha = hashlib.sha256(approval_raw).hexdigest()
    approval = json.loads(approval_raw)
    builtins = q5_calibration.builtins(output)
    builtins.update(RUN_ID=approval["run_id"], NONCE32=approval["barrier_token"])
    environ = {q5_case.OVERRIDE_PREFIX + key: value for key, value in overrides.items()}
    plan = q5_case.build_plan(case, builtins, environ)
    require(plan["complete"] and hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            == receipt["plan_sha256"], "quality launch differs from the frozen arm configuration")
    require(receipt["serve_argv"] == plan["serve"]["argv"] and
            receipt["serve_env"] == plan["serve"]["env"], "quality server launch differs")
    pid = receipt["server"]["pid"]
    require(type(pid) is int and pid > 0 and summary["server_pid"] == approval["server_pid"] == pid and
            summary["run_id"] == approval["run_id"], "quality client/server run identity differs")
    values, missing = q5_case.resolve_variables(case, builtins, environ)
    require(not missing, "quality launch has unresolved inputs")
    values.update(SERVED_MODEL=plan["served_model"], SERVER_PID=pid,
                  CLIENT_APPROVAL=str(approval_path), CLIENT_APPROVAL_SHA256=approval_sha)
    require(receipt["client_argv"] == q5_case.expand(plan["client"]["argv"], values),
            "quality client launch/approval differs")
    spec = plan["client"]["approval"]
    expected = bound_json(spec["template"], receipt["inputs"][spec["template"]])
    require("peer_supervised_launch" not in expected and "peer_supervised_launch" not in spec["set"],
            "quality peer launch cannot be supplied by the approval template")
    expected.update({key: q5_calibration.resolve_set(value, values) for key, value in spec["set"].items()})
    if expected.get("peer_reference") is not None:
        expected["peer_supervised_launch"] = {
            "schema": "hf2q.peer.supervised-launch.v1", "server_pid": pid,
            "base_url": expected["base_url"], "binary": plan["binary"],
            "binary_sha256": receipt["binary"]["sha256"], "serve_argv": receipt["serve_argv"]}
    require(approval == expected, "quality approval differs from the launched frozen case")
