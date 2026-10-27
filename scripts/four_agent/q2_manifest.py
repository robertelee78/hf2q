"""Post-unload canonical-client raw manifest assembler; no launch or inference.
Hashing existing files does not certify them. q2_certify replays and validates
all evidence; the resulting raw audit still awaits independent QE.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from qwen38_n1_verdict import loads, require
from q2_certify import BoundReader, digest


def bound(path, maximum=32 * 1024 * 1024):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
            and path.is_file() and path.stat().st_size <= maximum, "bounded canonical raw file")
    return {"path": str(path), "sha256": digest(path)}


def process_refs(directory, binary=None):
    return {"receipt": bound(directory / "receipt.json"),
            "stdout": bound(directory / "stdout.bin"),
            "stderr": bound(directory / "stderr.bin"),
            "binary": None if binary is None else bound(binary, 128 * 1024 * 1024)}


def test_group(agent, tool_number):
    # Actual Tools.tests layout, not files guessed from model text.
    logs = agent / "tools" / f"tool-{tool_number:02d}" / "logs"
    compile_refs = process_refs(logs / "compile")
    records = loads(Path(compile_refs["receipt"]["path"]).read_bytes())
    require(records.get("exit_code") == 0, "failed compile retained, cannot assemble positive cohort")
    artifacts = []
    for line in Path(compile_refs["stdout"]["path"]).read_bytes().splitlines():
        record = loads(line)
        if record.get("reason") == "compiler-artifact" and record.get("executable"):
            artifacts.append(Path(record["executable"]))
    require(len(artifacts) == (1 if tool_number == 4 else 2), "actual fixed compiler artifact count")
    return {"compile": compile_refs, "executions": [process_refs(logs / f"test-{i}", executable)
                                                   for i, executable in enumerate(artifacts)]}


def agent_refs(root, name, cardinality):
    agent = root / name if cardinality == 4 else root
    fixture = agent / "tools" / "fixture"
    import qwen38_n1_tools as tools
    return {"summary": bound(agent / "summary.json"),
            "ready": bound(root / "barrier" / f"{name}.ready.json") if cardinality == 4 else None,
            "turns": [{"request": bound(agent / f"turn-{i}" / "request.json"),
                       "response": bound(agent / f"turn-{i}" / "response.bin"),
                       "receipt": bound(agent / f"turn-{i}" / "receipt.json"),
                       "assembled": bound(agent / f"turn-{i}" / "assembled-assistant.json")}
                      for i in range(7)],
            "tool_results": [bound(agent / "tools" / f"tool-{i:02d}" / "result.json") for i in range(1, 7)],
            "test_groups": {"3": test_group(agent, 4), "5": test_group(agent, 6)},
            "fixture_files": {name: bound(fixture / name) for name in (*tools.IMMUTABLE, "src/ports.rs")}}


def assemble(spec):
    require(set(spec) == {"schema", "expected_plan", "client_root", "server_receipt", "readiness", "trace"}
            and spec["schema"] == "hf2q.q2.manifest-input.v1", "closed assembler inputs")
    reader = BoundReader()
    plan = reader.obj(spec["expected_plan"], 1048576)
    count = plan["required_cardinality"]
    require(type(count) is int and count in (1, 4), "declared cardinality")
    root = Path(spec["client_root"])
    require(root.is_absolute() and root == root.resolve() and root.is_dir() and not root.is_symlink(),
            "actual canonical client root")
    agents = {f"agent-{i}": agent_refs(root, f"agent-{i}", count) for i in range(count)}
    # Inputs refer to the exact retained raw server/readiness/trace files; no
    # favorable summary rewriting, cache-count substitution or trace invention.
    for key in ("server_receipt", "readiness", "trace"):
        reader.path(spec[key], 256 * 1024 * 1024 if key == "trace" else 32 * 1024 * 1024)
    manifest = {"schema": "hf2q.q2.cohort.v1",
                **{key: plan[key] for key in ("cohort_id", "identity", "family", "engine_kind", "model", "server_pid", "run_id")},
                "logical_clients": count, "server_receipt": spec["server_receipt"],
                "readiness": spec["readiness"], "client_summary": bound(root / "summary.json"),
                "agents": agents, "release": bound(root / "barrier" / "release.json") if count == 4 else None,
                "trace": spec["trace"]}
    reader.verify()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(digest(args.spec) == args.spec_sha256 and Path(args.spec).stat().st_size <= 1048576,
            "frozen bounded assembler specification")
    result = assemble(loads(Path(args.spec).read_bytes()))
    destination = Path(args.output)
    require(destination.is_absolute() and destination.parent == destination.parent.resolve()
            and not any(os.path.lexists(p / ".git") for p in (destination.parent, *destination.parent.parents)),
            "canonical new private manifest outside Git")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
