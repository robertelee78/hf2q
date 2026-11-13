"""ADR-063's exact canonical Qwen3.8 invalid-Splash scope, not a family-wide exemption.

Reduce the existing sweep's real Q5 runs, and freeze its raw inputs in the Q3
plan. This never manufactures an approval or a replacement evidence receipt.
"""
from pathlib import Path
import hashlib

import native_peer_splash as sp
import native_peer_splash_audit as audit
import native_peer_splash_ready as ready
import q5_case
import splash_sweep
import qwen38_n1_smoke as n1
import qwen_n4_live_smoke as n4
from qwen38_n1_verdict import FIXTURE_SHA, WORKFLOW, loads, require

HERE = Path(__file__).resolve().parent
# The target of ADR-063's documented strict-parity sweep. Another quantization,
# model revision or workload requires new reviewed evidence, not this exemption.
ARTIFACT = {"sha256": "cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7",
            "bytes": 16810714976}
# PR843 a9364ec9 changes only peer readiness/attestation. These exact variants
# do not execute on hf2q/Splash; the earlier canonical-stream-gating change is
# deliberately NOT in this whitelist. Independent PR review covers this diff.
SOURCE_CONTINUITY_WHITELIST = {
    "qwen_n4_live_smoke.py": frozenset((
        "2bc55641ad41ea06a52209431710c704728aefdc93383ffa72ae19132b68f242",
        "500320afad124c4e6785aa9ad42cf4f069fbd16292c7a7f0b05995a2e46983c7")),
    "family_peer.py": frozenset((
        "73a3d44884c79d7212921fe8816f8bf4d9a6c6b298181bff679c75c0a8f2e888",
        "fe6b9c4ea6df277a30dc37dfad21e56ba7716e9a8b8af9226123ca4f46957a30")),
}


def source_matches(name, recorded):
    current = ready.sha256(HERE / name)
    return recorded == current or {recorded, current} <= SOURCE_CONTINUITY_WHITELIST.get(name, frozenset())


def initial_requests(run, summary):
    """Content-check the archived actual request, not merely a workload label."""
    path = HERE / "fixtures/canonical-coding-requests.json"
    require(ready.sha256(path) == FIXTURE_SHA, "canonical fixture changed")
    fixture = read(path)[0]["request"]
    for tool in fixture["tools"]:
        tool["function"]["strict"] = False
    for agent in n4.AGENTS:
        directory = run / "client" / agent
        result = read(directory / "summary.json")
        identity = n4.identity(agent, summary["run_id"])
        messages = n1.initial_messages(fixture["messages"], identity)
        expected = n1.build_request(sp.MODEL_IDS["qwen38"], messages, fixture["tools"], 0, "qwen", "splash")
        require(read(directory / "turn-0/request.json") == expected
                and result["initial_identity_sha256"] == hashlib.sha256(identity.encode()).hexdigest()
                and result["initial_messages_sha256"] == hashlib.sha256(
                    n1.json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                "invalid-Splash raw initial request/identity/workflow/sampling/tool contract mismatch")


def read(path):
    return loads(Path(path).read_bytes())


def canonical(case):
    expected = read(HERE / "q5_cases/qwen38.json")
    require(case["family"] == "qwen38" and case["workload"] == expected["workload"]
            and case["client"]["argv"] == expected["client"]["argv"]
            and case["client"]["env"] == expected["client"]["env"]
            and q5_case.repair_turns(case) == 0 and q5_case.request_setup(case) is None,
            "invalid-Splash substitution is only bound to the canonical Qwen3.8 n4 workload")
    return expected


def freeze(evidence, case):
    """Validate before any model runs; bind the actual selection, cases and raw run trees."""
    expected = canonical(case)
    require(evidence, "missing documented invalid-Splash selection evidence")
    source_names = {Path(p).name for p in expected["client"]["approval"]["set"]["sources"]["sha256_of"]
                    if not Path(p).name.startswith("test_")}
    # Exactly the declared dense-family one-factor sweep (including matched);
    # verify-all is a MoE-only point, refused by the existing preset parser.
    configs = {sp.preset(point, "qwen38") for point in sp.PRESETS if point != "verify-all"}
    files, all_runs = set(), set()
    for bound in evidence:
        path = Path(bound["path"])
        require(ready.sha256(path) == bound["sha256"], "invalid-Splash selection evidence changed")
        selection = read(path)
        require(isinstance(selection, dict) and selection.get("schema") == "hf2q.splash-sweep-selection.v1"
                and selection.get("choice") is None and selection.get("pass_floor") == splash_sweep.PASS_FLOOR,
                "Splash evidence is not an entirely invalid sweep selection")
        table = selection.get("configurations")
        require(isinstance(table, dict) and set(table) == configs,
                "invalid-Splash evidence lacks the complete canonical Qwen3.8 configuration sweep")
        records = []
        files.add(path.resolve())
        for config, row in table.items():
            require(isinstance(row, dict) and row.get("valid") is False
                    and isinstance(row.get("runs"), list) and len(row["runs"]) >= 2,
                    "invalid-Splash evidence requires every configuration's retained Q5 runs")
            for name in row["runs"]:
                run = Path(name).resolve()
                require(run.is_absolute() and run not in all_runs, "duplicate invalid-Splash Q5 run")
                all_runs.add(run)
                receipt_path, summary_path = run / "receipt.json", run / "client/summary.json"
                receipt, summary = read(receipt_path), read(summary_path)
                saved_case = Path(receipt["case"]["path"])
                require(ready.sha256(saved_case) == receipt["case"]["sha256"],
                        "invalid-Splash retained Q5 case changed")
                native_case = q5_case.loads(saved_case.read_bytes())
                argv = list(expected["client"]["argv"])
                argv[-1] = "splash"
                workload = native_case["workload"]
                require(receipt["schema"] == "hf2q.q5.calibration-receipt.v1"
                        and receipt["family"] == native_case["family"] == "qwen38"
                        and receipt["workload"] == workload
                        and workload["id"].startswith("adr063-q5-splash-qwen38-n4-coding-ctx262144-sweep-")
                        and workload["slots"] == 4 and workload["context_per_slot"] == 262144
                        and workload["client"] == expected["workload"]["client"] + " --engine-kind splash"
                        and native_case["client"]["argv"] == argv and native_case["client"]["env"] == {}
                        and q5_case.repair_turns(native_case) == 0 and q5_case.request_setup(native_case) is None,
                        "invalid-Splash retained Q5 workload differs from the canonical matched workload")
                require(all(receipt["artifact"].get(k) == v for k, v in ARTIFACT.items()),
                        "invalid-Splash retained Q5 artifact differs from ADR-063's canonical target")
                require(receipt.get("cleanup_proven") is True,
                        "invalid-Splash retained Q5 cleanup is unproven")
                approval_path = run / "client-approval.json"
                approval = read(approval_path)
                spec = ready.native_spec(approval["native_splash"])
                native = summary["native_readiness"]
                require(summary["engine_kind"] == "splash" and native["q3_family"] == "qwen38"
                        and native["source_revision"] == spec["commit"] == sp.SPLASH_COMMIT
                        and native["build_id"] == spec["build_id"] == sp.BUILD_ID
                        and native["config"] == spec["config"] == config
                        and native["controls"] == approval["family_controls"] == sp.controls("qwen38", config)
                        and spec["model_sha256"] == ARTIFACT["sha256"]
                        and receipt["binary"]["sha256"] == spec["binary_sha256"]
                        and all(native["identity"]["model"].get(k) == v for k, v in ARTIFACT.items()),
                        "invalid-Splash source/configuration/model readiness binding mismatch")
                require(native["chat_template"]["hf2q_reviewed_sha256"] == sp.ARTIFACTS["qwen38"]["template_sha256"],
                        "invalid-Splash reviewed target template mismatch")
                initial_requests(run, summary)
                source = native["identity"]["source"]
                require(source["commit"] == spec["commit"] and source["build_id"] == spec["build_id"]
                        and all(source[k] == spec[k] for k in ("binary_sha256", "metallib_sha256", "host_step_sha256"))
                        and source["toolchain"]["sha256"] == spec["toolchain_sha256"]
                        and native["identity"]["venv"]["freeze_sha256"] == spec["venv_freeze_sha256"]
                        and native["identity"]["draft"]["revision"] == sp.ARTIFACTS["qwen38"]["draft_revision"]
                        and native["identity"]["draft"]["config_sha256"] == sp.ARTIFACTS["qwen38"]["draft_config_sha256"]
                        and native["identity"]["draft"]["weights_sha256"] == sp.ARTIFACTS["qwen38"]["draft_weights_sha256"],
                        "invalid-Splash retained binary/draft identity differs from its approval")
                client_argv = receipt["client_argv"]
                require(len(client_argv) == len(argv) and client_argv[1:4] == argv[1:4]
                        and Path(client_argv[4]).name == "qwen_n4_live_smoke.py"
                        and client_argv[5::2] == argv[5::2] and client_argv[-4:] == argv[-4:]
                        and client_argv[client_argv.index("--model") + 1] == sp.MODEL_IDS["qwen38"]
                        and approval["agent_template"]["fixture_sha256"] == FIXTURE_SHA
                        and approval["agent_template"]["workflow_sha256"] == hashlib.sha256(WORKFLOW.encode()).hexdigest(),
                        "invalid-Splash actual client/fixture/workflow binding mismatch")
                # Workload-source continuity is essential: the same argv does not
                # prove an old invalid arm remains invalid after client changes.
                sources = {Path(p).name: sha for p, sha in approval["sources"].items()}
                require(source_names <= sources.keys(), "invalid-Splash evidence lacks canonical workload sources")
                drift = sorted(n for n in source_names if not source_matches(n, sources[n]))
                require(not drift, "invalid-Splash workload source drift; reviewed continuity or fresh matched evidence required: "
                        + ", ".join(drift))
                files.update((receipt_path, summary_path, saved_case, approval_path))
                # No models or native assembly paths are traversed. Include every
                # archived client/parity byte, even when an earlier audit fails:
                # changing raw evidence must never preserve this frozen proof.
                for directory in (run / "client", run / audit.PARITY_DIR):
                    files.update(p.resolve() for p in directory.rglob("*") if p.is_file())
                for name in ("server.stdout", "server.stderr", "splash/launch.json"):
                    files.add(run / name)
                record = audit.run_record(run)
                require(record["prompt_parity"] == "PassedByteIdentical",
                        "invalid-Splash evidence lacks the bound prompt-parity post-oracle; missing artifacts are not invalid quality")
                records.append(record)
        require(splash_sweep.select(records) == selection,
                "invalid-Splash selection differs from independently reduced retained Q5 evidence")
    return {"artifact": ARTIFACT, "workload": expected["workload"],
            "inputs": [{"path": str(p), "sha256": ready.sha256(p)} for p in sorted(files)],
            "runs": len(all_runs)}


def check(plan, case, artifact):
    """Called both before launch and in the independent final report."""
    binding = plan.get("qwen38_splash_substitution")
    require(isinstance(binding, dict), "missing source-bound invalid-Splash substitution")
    require(artifact is not None and all(artifact.get(k) == v for k, v in ARTIFACT.items()),
            "invalid-Splash substitution candidate artifact differs from the canonical target")
    require(freeze(plan.get("splash_unavailable_evidence"), case) == binding,
            "invalid-Splash raw evidence changed after freezing")
    return binding


def current_inputs(binding):
    """Q3's existing per-position identity-drift check covers every frozen raw input."""
    return {row["path"]: ready.sha256(row["path"]) if Path(row["path"]).is_file() else None
            for row in binding["inputs"]}
