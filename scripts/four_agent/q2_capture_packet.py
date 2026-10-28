"""Closed model-free OpenCode2.1.1 capture packet; never launches a client.
The existing capture runner remains the only process/socket owner. A generated
approval is source/control authorization, never native tool or model quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from q2_client_profile import FAMILIES
from q2_capture_preflight import service_identity
from qwen38_n1_verdict import loads, require

SHA = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_CLOSURE = {"q2_capture_client.py", "q2_capture_packet.py", "q2_client_profile.py",
                  "q2_certify.py", "q2_capture_preflight.py", "qwen38_n1_process.py", "qwen38_n1_verdict.py",
                  "qwen38_n1_tools.py", "qwen_n4_live_smoke.py", "coding_family.py", "f2_semantic.py"}
KEYS = {"schema", "approved_capture_only", "source_files", "client_binary", "expected_version",
        "source_config", "sandbox_binary", "port", "profiles"}


def bound_shape(value):
    require(type(value) is dict and set(value) == {"path", "sha256"}
            and type(value["path"]) is str and Path(value["path"]).is_absolute()
            and ".." not in Path(value["path"]).parts
            and type(value["sha256"]) is str and SHA.fullmatch(value["sha256"]), "complete frozen BoundFile")


def validate_capture_approval(value):
    require(type(value) is dict and set(value) in (KEYS, KEYS | {"existing_client_service"})
            and value["schema"] == "hf2q.q2.client-capture-approval.v1"
            and value["approved_capture_only"] is True, "explicit model-free capture approval")
    require(value["expected_version"] == "2.1.1", "exact native OpenCode2.1.1, not a wildcard/other client")
    sources = value["source_files"]
    require(type(sources) is dict and SOURCE_CLOSURE <= sources.keys()
            and all(type(n) is str and Path(n).name == n and n.endswith(".py")
                    and type(h) is str and SHA.fullmatch(h) for n, h in sources.items()), "actual capture import-source closure")
    for key in ("client_binary", "source_config", "sandbox_binary"): bound_shape(value[key])
    require(value["sandbox_binary"]["path"] == "/usr/bin/sandbox-exec", "exact sandbox executable")
    require(type(value["port"]) is int and 1024 <= value["port"] <= 65535, "exact owned loopback port")
    require(type(value["profiles"]) is dict and set(value["profiles"]) == FAMILIES, "explicit four derived family profiles")
    for family, profile in value["profiles"].items():
        require(type(profile) is dict and set(profile) == {"model_id", "artifact_sha256", "effective_hf_defaults"}
                and type(profile["model_id"]) is str and 0 < len(profile["model_id"]) < 200
                and type(profile["artifact_sha256"]) is str and SHA.fullmatch(profile["artifact_sha256"]), "closed actual family/artifact identity")
        bound_shape(profile["effective_hf_defaults"])
    service_identity(value.get("existing_client_service"), value["client_binary"])
    return value


def approval_from_packet(packet):
    require(type(packet) is dict and set(packet) == {"schema", "review", "approval"}
            and packet["schema"] == "hf2q.q2.native-capture-packet.v1", "closed disabled capture packet")
    # None is deliberately unexecutable; root supplies an independent review,
    # not a flag inferred from source compilation or scripted responses.
    bound_shape(packet["review"])
    review = Path(packet["review"]["path"])
    require(review.is_file() and not review.is_symlink() and review.stat().st_size <= 1048576
            and hashlib.sha256(review.read_bytes()).hexdigest() == packet["review"]["sha256"], "independent source review bound")
    return validate_capture_approval(packet["approval"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", required=True); p.add_argument("--packet-sha256", required=True)
    p.add_argument("--output-approval", required=True); a = p.parse_args()
    source = Path(a.packet)
    require(source.is_absolute() and source.is_file() and not source.is_symlink()
            and source.stat().st_size <= 1048576, "bounded frozen packet")
    raw = source.read_bytes(); require(hashlib.sha256(raw).hexdigest() == a.packet_sha256, "packet SHA")
    approval = approval_from_packet(loads(raw)); output = Path(a.output_approval)
    require(output.is_absolute() and output.parent == output.parent.resolve()
            and not any((x / ".git").exists() for x in (output.parent, *output.parent.parents)), "new private approval outside Git")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream: json.dump(approval, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__": main()
