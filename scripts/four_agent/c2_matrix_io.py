"""Bound C2 matrix metadata/payload IO. Offline only; never opens model weights."""
import hashlib
import json
import math
import os
from pathlib import Path

from c2_observation import regular_evidence, require

META_LIMIT = 16 * 1024 * 1024
SCALARS = {"F32": 4, "U32": 4, "U8": 1}


def integer(value, label, minimum=0, maximum=(1 << 64) - 1):
    require(type(value) is int and minimum <= value <= maximum, f"invalid {label}")
    return value


def digest(value):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "invalid SHA-256 binding")
    return value


def stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read_json(path, expected_sha=None, limit=META_LIMIT):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "canonical absolute evidence path required")
    with regular_evidence(path) as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= limit, f"metadata exceeds finite bound: {path}")
        raw = stream.read(limit + 1)
        require(len(raw) == before.st_size and stamp(os.fstat(stream.fileno())) == stamp(before),
                f"metadata changed while reading: {path}")
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None:
        require(sha == digest(expected_sha), f"metadata digest differs: {path}")
    return json.loads(raw), sha


def resolve(root, reference, absolute=False):
    require(type(reference) is dict and set(reference) == {"path", "sha256"}, "closed file binding required")
    digest(reference["sha256"])
    require(type(reference["path"]) is str, "bound path must be text")
    path = Path(reference["path"])
    require(path.parts and all(part not in ("..", ".") for part in path.parts), "invalid bound path components")
    if path.is_absolute():
        require(absolute, "scope-relative binding required")
    else:
        path = Path(root) / path
    require(path.is_relative_to(root) and path.resolve(strict=True) == path, "bound file aliases/escapes its scope")
    return path


def bound(root, reference, absolute=False, limit=META_LIMIT):
    path = resolve(root, reference, absolute)
    value, _ = read_json(path, reference["sha256"], limit)
    return value, path


def acquisition(value, expected, output):
    fields = {"packet_sha256", "nonce", "output_root", "actual_binary_sha256", "composition_sha256",
              "independent_review_sha256", "opened_model", "observed_model_opens", "required_model_opens",
              "device_registry_id", "queue_identity", "model_instance", "provenance"}
    require(type(value) is dict and type(expected) is dict and set(value) == set(expected) == fields,
            "acquisition schema differs")
    stable = set(expected) - {"output_root", "observed_model_opens"}
    require(all(value[key] == expected[key] for key in stable) and value["output_root"] == str(output),
            "acquisition source/nonce/model/device/queue/root differs")
    observed = integer(value["observed_model_opens"], "observed model opens")
    require(value["required_model_opens"] <= observed <= expected["observed_model_opens"],
            "scope model-open counter is outside its actual lifetime")
    require(value["provenance"] == "IndependentRootReviewedComposition", "synthetic/preparation is not native acquisition")
    for key in ("packet_sha256", "nonce", "actual_binary_sha256", "composition_sha256", "independent_review_sha256"):
        digest(value[key])
    for key in ("device_registry_id", "queue_identity", "model_instance", "required_model_opens"):
        integer(value[key], key, 1)
    require(integer(value["observed_model_opens"], "observed model opens") >= value["required_model_opens"],
            "actual loader did not consume its bound model descriptor")
    require(integer(value["opened_model"]["inode"], "model inode", 1)
            and integer(value["opened_model"]["bytes"], "model bytes", 1), "opened model identity absent")
    descriptor = value["opened_model"]
    require(type(descriptor) is dict and set(descriptor) == {"device", "inode", "bytes", "mtime", "mtime_ns", "ctime", "ctime_ns"},
            "actual opened-model descriptor is incomplete")
    integer(descriptor["device"], "opened model device")
    for key in ("mtime", "mtime_ns", "ctime", "ctime_ns"):
        integer(descriptor[key], "opened model " + key, -(1 << 63), (1 << 63) - 1)


def coordinate(value):
    require(type(value) is dict and set(value) == {"logical_step", "layer", "stage", "agent", "logical_row_start", "segment"},
            "closed Q1 coordinate required")
    for key in ("logical_step", "layer", "logical_row_start", "segment"):
        integer(value[key], key)
    require(type(value["stage"]) is str and type(value["agent"]) is str, "Q1 coordinate strings absent")
    return tuple(value[key] for key in ("stage", "agent", "logical_row_start", "layer", "segment", "logical_step"))


def extents(values):
    require(type(values) is list and 0 < len(values) <= 65536, "finite nonempty strict Q1 extents required")
    result = {}
    for value in values:
        require(type(value) is dict and set(value) == {"at", "scalar", "shape"}, "closed expected tensor required")
        key = coordinate(value["at"])
        require(key not in result and value["scalar"] in SCALARS, "duplicate/unsupported expected tensor")
        require(type(value["shape"]) is list and value["shape"], "expected tensor shape absent")
        for width in value["shape"]:
            integer(width, "tensor dimension", 1)
        result[key] = value
    return result


def payload(path, sha, length):
    require(path.resolve(strict=True) == path, "payload aliases its source root")
    with regular_evidence(path) as stream:
        before = os.fstat(stream.fileno())
        require(before.st_size == length, "payload length differs from exact source geometry")
        result = hashlib.sha256()
        count = 0
        while block := stream.read(65536):
            count += len(block)
            require(count <= length, "payload grew during verification")
            result.update(block)
        require(count == length and stamp(os.fstat(stream.fileno())) == stamp(before), "payload changed during verification")
    require(result.hexdigest() == digest(sha), "payload digest differs")


class Scope:
    def __init__(self, packet, packet_sha, registration, registration_sha, actual):
        self.packet, self.identity = packet, packet["identity"]
        self.root = Path(packet["output"])
        require(self.root.is_absolute() and self.root.resolve(strict=True) == self.root, "private scope root is not canonical")
        self.registration, self.registration_sha, self.actual = registration, registration_sha, actual
        require(actual["packet_sha256"] == packet_sha and actual["actual_binary_sha256"] == self.identity["binary_sha256"],
                "actual acquisition packet/image differs")
        acquisition(actual, actual, self.root)

    def header(self, value, case, arm, name="acquisition", output=None, frozen=None):
        require(value["identity"] == self.identity and value["case"] == case and value["arm"] == arm
                and value["slot"] == 0 and value["registration_sha256"] == self.registration_sha,
                "source/registration/case/arm/slot custody differs")
        acquisition(value[name], self.actual, self.root if output is None else output)
        integer(value["slot"], "source physical slot", 0, 0)
        if frozen is not None:
            require(value[name] == frozen, "pre-call evidence changed its owning request acquisition")
        if "agent" in value:
            require(value["agent"] == f"{case}/slot0", "physical-slot agent binding differs")

    def trace(self, reference, expected):
        value, path = bound(self.root, reference)
        require(value["schema"] == "hf2q.adr063.invariance.v2" and value["identity"] == self.identity
                and value["comparison_key"] == self.identity, "strict Q1 trace identity/schema differs")
        acquisition(value["acquisition"], self.actual, path.parent)
        coverage = value["coverage"]
        require(not coverage["errors"] and not coverage["missing_native_capabilities"], "Q1 capture is incomplete")
        wanted = extents(expected)
        require(extents(coverage["expected"]) == wanted, "Q1 expected geometry differs from pre-call source registration")
        require(len(coverage["captured"]) == len(wanted), "Q1 captured extent differs")
        total = sum(math.prod(item["shape"]) * SCALARS[item["scalar"]] for item in wanted.values())
        require(total <= integer(self.packet["host_payload_limit"], "per-store payload bound", 1)
                and total <= self.registration["grants"]["payload_bytes"], "strict store exceeds admitted payload bounds")
        seen = set()
        for row in coverage["captured"]:
            key = coordinate(row["at"])
            require(key in wanted and key not in seen, "Q1 row unplanned or duplicated")
            seen.add(key)
            descriptor = wanted[key]
            require(row["scalar"] == descriptor["scalar"] and row["shape"] == descriptor["shape"], "Q1 row representation differs")
            length = math.prod(row["shape"]) * SCALARS[row["scalar"]]
            require(row["logical_bytes"] == length and row["checked_drain"], "Q1 exact bytes/checked terminal absent")
            data = resolve(path.parent, {"path": row["file"], "sha256": row["sha256"]})
            payload(data, row["sha256"], length)
        return path, value


def input_binding(value, rows):
    require(type(value) is dict and len(value) == 1, "closed pre-call input binding required")
    if set(value) == {"Host"}:
        require(type(value["Host"]) is dict and set(value["Host"]) == {"sha256"}, "closed host input binding required")
        digest(value["Host"]["sha256"])
    else:
        require(set(value) == {"GpuCheckedCompletion"}, "unknown native input source")
        source = value["GpuCheckedCompletion"]
        require(type(source) is dict and set(source) == {"buffer_identity", "buffer_tokens", "first", "rows"},
                "closed actual GPU input descriptor required")
        integer(source["buffer_identity"], "actual GPU input identity", 1)
        capacity = integer(source["buffer_tokens"], "actual GPU token capacity", 1)
        first = integer(source["first"], "actual GPU token offset")
        require(integer(source["rows"], "actual GPU input rows", 1) == rows and first + rows <= capacity,
                "actual GPU input range differs from required prefix")
    return value


def input_digest(tokens):
    require(type(tokens) is list, "actual native input token sequence required")
    value = hashlib.sha256(b"hf2q.c2.native-input.le-u32.v1\0")
    value.update(len(tokens).to_bytes(8, "little"))
    for token in tokens:
        value.update(integer(token, "actual native input token", 0, (1 << 32) - 1).to_bytes(4, "little"))
    return value.hexdigest()
