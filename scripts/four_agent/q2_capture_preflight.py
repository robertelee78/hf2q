"""Actual-used model-free capture boundaries; no process/import side effects."""
from pathlib import Path
import re
from qwen38_n1_verdict import require


def service_identity(value, binary):
    if value is None: return None
    require(type(value) is dict and set(value) == {"pid", "ppid", "start", "argv", "binary"}
            and type(value["pid"]) is int and value["pid"] > 1
            and type(value["ppid"]) is int and value["ppid"] == 1
            and type(value["start"]) is str and 0 < len(value["start"]) < 128
            and value["binary"] == binary
            and value["argv"] == [binary["path"], "serve", "--service"],
            "only one exact bound same-client background service")
    return value


def quiet_inventory(raw, service, binary):
    """One exact identity exception, not a generic PID/name allowlist."""
    service_identity(service, binary)
    require(type(raw) is bytes and len(raw) <= 4 * 1024 * 1024, "quiet inventory envelope")
    allowed = None
    for line in raw.decode("utf8", "strict").splitlines():
        fields = line.strip().split(None, 8)
        require(len(fields) == 9 and fields[0].isdecimal() and fields[1].isdecimal(), "typed process inventory")
        pid, ppid = int(fields[0]), int(fields[1]); start = " ".join(fields[2:7])
        name, args = fields[7], fields[8]
        if service is not None and pid == service["pid"]:
            require(allowed is None and ppid == service["ppid"] and start == service["start"]
                    and name == Path(binary["path"]).name and args == " ".join(service["argv"]),
                    "preexisting service PID/start/argv/binary drift")
            allowed = {"pid": pid, "ppid": ppid, "start": start, "ucomm": fields[7], "args": args}
            continue
        require(name not in {"hf2q", "llama-server", "llama-cli", "test-binary", "helper-binary", "cargo", "rustc", "clang", "clang++", "opencode"}
                and not name.startswith(("hf2q-", "mlx_native-", "noalloc-size", "live-wired", "owned-overlap")),
                "known competing model/test/compiler/client worker")
    require(service is None or allowed is not None, "bound preexisting service disappeared; re-review identity")
    return {"raw_ps": raw.decode("utf8"), "allowed_preexisting_service": allowed,
            "scope": "isolated standalone metadata capture; not performance quietness"}


def standalone_capture_argv(argv, sandbox, profile, binary):
    require(type(argv) is list and argv[:6] == [str(sandbox), "-f", str(profile), str(binary), "run", "--standalone"],
            "native capture must use standalone client, never shared service")
    return argv


def default_resolution(value, family, reader):
    require(type(value) is dict and value.get("family") == family
            and type(value.get("server_source")) is str and re.fullmatch("[a-f0-9]{40}", value["server_source"])
            and type(value.get("source_files")) is dict and value["source_files"], "source-bound HF default evidence")
    for path, digest in value["source_files"].items(): reader.path({"path": path, "sha256": digest})
    if value.get("schema") == "hf2q.q2.effective-hf-defaults.v1":
        require([x.get("phase") for x in value.get("per_phase", [])] == list(range(7)), "actual per-phase default receipt")
        return {"kind": "retained_actual_default_receipt", "effective_sampling_qualified": False,
                "status": "PendingActualServerJoin"}
    require(set(value) == {"schema", "family", "server_source", "source_files", "resolution_scope",
                          "effective_sampling_qualified", "status"}
            and value["schema"] == "hf2q.q2.hf-default-resolution-recipe.v1"
            and value["resolution_scope"] == "source_rules_only_no_actual_request_or_server_resolution"
            and value["effective_sampling_qualified"] is False and value["status"] == "PendingActualServerJoin",
            "recipe is rules-only, never fabricated resolved sampling")
    return {"kind": "source_default_resolution_recipe_only", "effective_sampling_qualified": False,
            "status": "PendingActualServerJoin"}


def exact_client_version(raw, expected):
    require(type(raw) is bytes and expected == "2.1.1", "bound native client version")
    text = raw.decode("utf8", "strict").strip()
    require(text in {expected, "opencode v" + expected}, "actual native client version changed")
    return expected
