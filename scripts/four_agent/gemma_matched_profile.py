"""Opt-in benchmark tool wire; every semantic result derives from real raw proof.

No process, fixture or host access on import. Canonical clients do not import this
module unless the explicit CLI/approval profile is selected.
"""
import json
import re
from pathlib import Path

from qwen38_n1_tools import Tools
from qwen38_n1_verdict import FILE_LIMIT, loads, require, test_harness
import qwen38_n1_process as child

PROFILE = "gemma-matched-v1"
EXTRA_MODULES = ("gemma_matched_profile.py", "gemma_matched_stats.py",
                 "test_gemma_matched_profile.py", "test_gemma_matched_stats.py")
TESTS = {"ports": ("accepts_u16_ports", "rejects_invalid_ports"),
         "qwen38_n1_fixture": ("tests::boundary_ports_accepted", "tests::zero_and_overflow_rejected")}


def validate_activation(profile, packet, family, engine, slots):
    require(profile == PROFILE and packet.get("benchmark_profile") == PROFILE,
            "explicit matched benchmark CLI and approval required")
    require((family, engine, slots) == ("gemma4", "hf2q", 4),
            "matched profile is Gemma HF four-slot only")


def bounded_read(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= FILE_LIMIT,
            "bounded regular raw proof required")
    return path.read_bytes()


def semantic_tests(detail, compiler_stdout, test_runs):
    """Pure actual-used boundary. Never infer test success from an exit alone."""
    target = detail.get("target")
    expected = {"ports"} if target == "ports" else set(TESTS) if target == "all" else set()
    require(expected and detail.get("passed") is True and detail.get("cleanup") is True,
            "actual successful compiler/tests/cleanup required")
    source = detail.get("source_sha256")
    require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{64}", source), "source digest")
    processes = detail.get("processes", [])
    require(len(processes) == len(expected) + 1 and
            all(p.get("exit_code") == 0 and p.get("cleanup") is True and
                p.get("cleanup_receipt", {}).get("verified") is True for p in processes),
            "every actual process must succeed and be cleaned")
    artifacts, finished, warnings = {}, [], 0
    for line in compiler_stdout.splitlines():
        item = loads(line)
        reason = item.get("reason")
        if reason == "build-finished":
            finished.append(item.get("success"))
        elif reason == "compiler-message":
            level = item.get("message", {}).get("level")
            require(level not in ("error", "failure-note"), "compiler error cannot become semantic pass")
            warnings += int(level == "warning")
        elif reason == "compiler-artifact" and item.get("executable"):
            name = item.get("target", {}).get("name")
            require(name in expected and name not in artifacts and
                    item.get("profile", {}).get("test") is True, "actual test target set")
            artifacts[name] = item["executable"]
    require(finished == [True] and set(artifacts) == expected, "complete compiler proof")
    require(len(test_runs) == len(expected), "one real run per test artifact")
    accepted = []
    seen = set()
    for run in test_runs:
        name = run["target"]
        require(name in expected and name not in seen and
                run["executable"] == artifacts[name], "test run must use actual compiler artifact")
        seen.add(name)
        require(run["stderr"] == b"", "unexpected test stderr retained; no semantic concealment")
        test_harness(run["stdout"], 2)
        names = re.findall(r"^test ([^\s]+) \.\.\. ok$", run["stdout"].decode("utf8", "strict"), re.M)
        require(len(names) == 2 and sorted(names) == sorted(TESTS[name]), "fixed real test names/counts")
        accepted.append({"target": name, "tests": [{"name": n, "status": "ok"} for n in sorted(names)]})
    require(seen == expected, "missing test target")
    return json.dumps({"schema": "hf2q.gemma-matched-tool-result.v1", "target": target,
                       "passed": True, "source_sha256": source, "compiler_warnings": warnings,
                       "artifacts": sorted(accepted, key=lambda a: a["target"])},
                      sort_keys=True, separators=(",", ":"))


def load_real_proof(detail, log):
    """Read the production Tools.tests raw outputs; retain all original receipts."""
    log = Path(log)
    compiler = bounded_read(log / "logs/compile/stdout.bin")
    processes = detail["processes"]
    runs = []
    # Executable-to-target provenance comes from actual Cargo JSON, never filename guessing.
    by_executable = {}
    for line in compiler.splitlines():
        item = loads(line)
        if item.get("reason") == "compiler-artifact" and item.get("executable"):
            require(item["executable"] not in by_executable, "duplicate compiler executable")
            by_executable[item["executable"]] = item.get("target", {}).get("name")
    for index, process in enumerate(processes):
        raw_receipt = loads(bounded_read(log / "logs" / ("compile" if index == 0 else f"test-{index-1}") / "receipt.json"))
        require(all(raw_receipt.get(k) == process.get(k) for k in
                    ("pid", "argv", "exit_code", "cleanup", "cleanup_receipt")), "raw process receipt drift")
        if index == 0:
            continue
        argv = process["argv"]
        require(len(argv) >= 2 and argv[-1] == "--test-threads=1", "real test command")
        executable = argv[-2]
        path = Path(executable)
        require(executable in by_executable and not path.is_symlink() and path.is_file() and
                path.resolve().is_relative_to((log / "target").resolve()), "foreign/changed artifact")
        require(process.get("actual_tests_passed") == 2, "actual existing harness proof required")
        require(child.digest(path) in detail["artifact_sha256"], "actual artifact hash drift")
        directory = log / "logs" / f"test-{index-1}"
        runs.append({"target": by_executable[executable], "executable": executable,
                     "stdout": bounded_read(directory / "stdout.bin"),
                     "stderr": bounded_read(directory / "stderr.bin")})
    return compiler, runs


def invoke_semantic(tools, name, args, real_invoke):
    """Actual-used wrapper; errors preserve raw output and fail the conversation."""
    raw_result = real_invoke(name, args)
    if name != "run_tests":
        return raw_result
    receipt = tools.receipts[-1]
    log = tools.root / f"tool-{tools.calls:02d}"
    receipt["raw_model_result"] = raw_result
    try:
        require(receipt.get("name") == name and receipt.get("arguments") == args and
                child.digest(tools.fixture / "src/ports.rs") == receipt["source_sha256"],
                "actual tool/source binding")
        compiler, runs = load_real_proof(receipt, log)
        result = semantic_tests(receipt, compiler, runs)
        receipt.update(result=result, benchmark_profile=PROFILE)
        return result
    except BaseException as exc:
        receipt.update(status="Failed", semantic_error=str(exc))
        raise
    finally:
        (log / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")


class BenchmarkTools(Tools):
    def invoke(self, name, args):
        return invoke_semantic(self, name, args, super().invoke)
