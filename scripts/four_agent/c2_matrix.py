#!/usr/bin/env python3
"""Register complete C2 cases and independently verify their source-bound evidence.

Offline only. Numeric evidence is not native-root, admission, quality, timing,
whole-matrix or release qualification. No model is opened and no retry is run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from c2_matrix_io import Scope, bound, digest, integer, read_json, require, resolve
from c2_matrix_plans import execution
from c2_matrix_evidence import (canonical_arm, capability_markers, equal_semantic_state, final_state, logical_prompt,
                                raw_arm, same_raw_twins, saved_witnesses, semantic_state_pair, settlement_proof, transactions)
from q1_decision_policy import exact
from qwen38_n1_verdict import Invalid

GRANTS = {"events", "child_stores", "records", "payload_bytes", "metadata_bytes",
          "retained_raw_bytes", "state_readback_bytes", "live_ram_bytes"}


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def catalog(path, sha, identity):
    fields = {"family", "source_commit", "binary_sha256", "native_version", "native_checksum", "model_sha256",
              "tokenizer_sha256", "template_sha256", "config_sha256", "token_vectors_sha256"}
    require(type(identity) is dict and set(identity) == fields and identity["family"] in ("Qwen38", "Qwen36"),
            "closed Qwen C2 identity required")
    require(type(identity["source_commit"]) is str and len(identity["source_commit"]) == 40
            and all(char in "0123456789abcdef" for char in identity["source_commit"]), "invalid source commit")
    for key in fields - {"family", "source_commit", "native_version"}:
        digest(identity[key])
    value, _ = read_json(path, sha)
    require(value["native_version"] == identity["native_version"], "catalog Native pin differs from source identity")
    require(value["schema"] == "hf2q.c2.compiled-matrix-catalog.v1"
            and value["actual_exporter_binary_sha256"] == digest(identity["binary_sha256"]),
            "catalog exporter differs from the actual packet image")
    require(value["model_opened"] is False and value["device_created"] is False
            and value["acquisition_attested"] is False, "catalog export invented runtime acquisition")
    require(value["mode"] in ("lifecycle", "o1") and type(value["graft"]) is bool, "catalog mode/bank absent")
    cases = value["full_case_catalog"]
    require(type(cases) is list and cases and all(type(case) is str and case for case in cases)
            and len(set(cases)) == len(cases), "compiled full case catalog is empty or duplicated")
    require(hashlib.sha256(compact(cases)).hexdigest() == value["catalog_sha256"], "compiled full catalog digest differs")
    artifacts = value["required_artifacts"]
    require(type(artifacts) is dict and set(artifacts) == set(cases), "compiled required-artifact catalog omits cases")
    keys = set()
    for case in cases:
        required = artifacts[case]
        require(type(required) is list and required and len(set(required)) == len(required)
                and all(type(key) is str and key.startswith(case + "-") and key not in keys for key in required),
                "compiled required artifacts repeat or escape their case")
        keys.update(required)
    return value


def grants(value):
    require(type(value) is dict and set(value) == GRANTS, "closed finite matrix grants required")
    for key in GRANTS:
        integer(value[key], key, 1, 65536 if key == "events" else (1 << 64) - 1)
    require(value["retained_raw_bytes"] <= value["live_ram_bytes"]
            and value["state_readback_bytes"] <= value["live_ram_bytes"], "raw/state grants exceed live RAM")
    return value


def write_new(path, value):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve(strict=True) == path.parent, "canonical output parent required")
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def register(args):
    identity, _ = read_json(args.identity)
    source = catalog(args.catalog, args.catalog_sha256, identity)
    limits, _ = read_json(args.grants)
    grants(limits)
    require(args.case and len(set(args.case)) == len(args.case)
            and all(case in source["full_case_catalog"] for case in args.case), "select unique complete compiled cases")
    value = {"schema": "hf2q.c2.matrix-registration.v1", "identity": identity,
             "mode": source["mode"], "catalog_sha256": source["catalog_sha256"], "cases": args.case, "grants": limits}
    return {"registration": write_new(args.output, value), "model_opened": False, "acquisition_admitted": False}


def semantic_tokens(scope, case, bundle, raw):
    result = []
    for index, request in bundle["decisions"].items():
        key = ("SpecCanonicalIds", f"{case}/slot0", index, 0, 0, index)
        record = raw["rows"][key]
        path = resolve(raw["path"].parent, {"path": record["file"], "sha256": record["sha256"]})
        from c2_observation import regular_evidence
        with regular_evidence(path) as stream:
            value = stream.read(5)
        require(len(value) == 4 and hashlib.sha256(value).hexdigest() == record["sha256"], "semantic selector changed")
        token = int.from_bytes(value, "little")
        require(token < request["vocab"], "actual semantic selector is outside requested vocabulary")
        result.append(token)
    return result


def classification(scope, case, value, authority_binding, arms, raw, comparisons):
    if "prefix_script_sha256" in value:
        require(value["prefix_script_sha256"] == authority_binding["sha256"], "classification changed its input authority")
    schema = value["schema"]
    wanted_schema = ("hf2q.c2.o1.v1" if scope.registration["mode"] == "o1" else
                     "hf2q.c2.cancel-matrix.v1" if case.startswith("cancel-") else
                     "hf2q.c2.failed-restore.v1" if case.startswith("failure-") else
                     "hf2q.c2.graft-identity.v1" if case == "graft-identity-change" else
                     "hf2q.c2.exactness-case.v1")
    require(schema == wanted_schema, "classification relabelled its preregistered case kind")
    negative = False
    if schema == "hf2q.c2.o1.v1":
        require(type(value["shallower"]) is bool and type(value["pruning_disabled"]) is bool,
                "typed O1 sensitivity branch required")
        negative = value["shallower"] and value["pruning_disabled"]
        require(case == f"o1-shallower{str(value['shallower']).lower()}-pruningdisabled{str(value['pruning_disabled']).lower()}",
                "O1 classification changed the registered sensitivity branch")
        require(value["b_completed_high_cursor"] is True and value["second_fault"] is False
                and value["twin_image_restores"] == 0, "O1 control skipped its original high-cursor/cold-twin boundary")
        if not value["pruning_disabled"]:
            integer(value["tombstoned"], "actual O1 tombstones", 1)
        if not negative:
            integer(value["restore_refused"], "actual O1 refusal", 1)
        if not value["shallower"] and value["pruning_disabled"]:
            require(value["cold_reset_negative_control"] == "RefusedByIndependentCacheGeneration",
                    "cold reset omitted its independent generation refusal")
        require(value["classification"] == ("DivergedExactInvariant" if negative else "ExactEqual"),
                "O1 intentional negative relabelled as a positive")
    elif schema == "hf2q.c2.cancel-matrix.v1":
        require(case.startswith("cancel-") and value["state"] == "ExactEqual" and value["pending"] is False,
                "cancellation recovery left state/pending work unfinished")
        recovered = integer(value["recovered_tokens"], "classified recovery depth")
        require(recovered == arms["actual"]["outcome"]["Cancelled"]["recovered_tokens"]
                == arms["twin"]["outcome"]["Reference"]["recovered_tokens"],
                "cancelled/reference/classified recovery depths differ")
    elif schema == "hf2q.c2.failed-restore.v1":
        require(case == "failure-" + value["fault"] and type(value["actual_refusal"]) is str and value["actual_refusal"]
                and value["poisoned_after_fault"] is (value["fault"] == "Encode")
                and value["cold_twin"] == "ExactEqual", "restore fault did not preserve refusal/poison/cold semantics")
    elif schema == "hf2q.c2.graft-identity.v1":
        require(case == "graft-identity-change" and value["old_identity"] != value["new_identity"]
                and value["old_restore"] == "RefusedBeforeMutation" and value["cold_twin"] == "ExactEqual",
                "changed bank did not refuse the original anchor")
    else:
        require(schema == "hf2q.c2.exactness-case.v1" and value["class"] == "ClassA" and value["case"] == case
                and value["route"] == scope.packet["route"] and value["same_concurrency_raw_twin"] is True
                and value["state_exact"] is True and value["single_slot_twin"] is True,
                "lifecycle classification changed the required parity contract")
        rank = integer(value["rank"], "selected anchor rank", 0, 2)
        require(type(value["equality"]) is bool
                and case == f"rank{rank}-eq{str(value['equality']).lower()}-peers{str(case.endswith('-peerstrue')).lower()}",
                "classification changed the registered rank/equality branch")
        capability_markers(scope, value, raw)
        actual_tokens = semantic_tokens(scope, case, arms["actual"], raw["actual"])
        count = arms["actual"]["outcome"]["Completed"]["emitted_tokens"]
        require(value["target_tokens"] == actual_tokens[:count], "reported output differs from actual semantic selectors")
        peers = value["peer_restore_state_digests"]
        require(type(peers) is list and len(peers) == 3, "peer restore state digest proof omitted")
        require([integer(peer["slot"], "peer physical slot", 1, 3) for peer in peers] == [1, 2, 3],
                "peer restore proof omits/repeats physical slots")
        for peer in peers:
            require(set(peer) == {"slot", "before", "after"} and digest(peer["before"]) == digest(peer["after"]),
                    "slot zero restore mutated a peer state")
    if negative:
        require(any(item["classification"] == "DivergedExactInvariant" for item in comparisons),
                "O1 negative lost its actual raw/state sensitivity")
    else:
        require(all(item["classification"] == "ExactEqual" and item["common_records"] > 0 for item in comparisons),
                "positive complete-case pair is not ExactEqual")
    return negative


def verify_case(scope, case, artifacts, required):
    def binding(key):
        return artifacts[f"{case}-{key}"]["binding"]
    authority_binding = binding("authority")
    authority, _ = bound(scope.root, authority_binding)
    value, _ = bound(scope.root, binding("classification"))
    arm_names = [arm for arm in ("actual", "twin", "single") if f"{case}-{arm}-execution" in required]
    arms, states, streams, raw = {}, {}, {}, {}
    for arm in arm_names:
        has_raw = f"{case}-{arm}" in required
        arms[arm] = execution(scope, case, arm, binding(arm + "-execution"), has_raw)
        states[arm] = final_state(scope, case, arm, binding(arm + "-state"))
        streams[arm] = transactions(scope, case, arm, binding(arm + "-transactions"), arms[arm])
        if has_raw:
            raw[arm] = raw_arm(scope, case, arm, binding(arm), arms[arm])
            semantic_tokens(scope, case, arms[arm], raw[arm])
            saved_witnesses(scope, case, arms[arm], raw[arm], authority)
    comparisons = [settlement_proof(states[arm], streams[arm], arms[arm]["outcome"]) for arm in arm_names]
    state_comparison = exact if case == "cancel-fresh-reset" else semantic_state_pair
    comparisons.append(state_comparison(states["actual"][0], states["twin"][0]))
    if "actual" in raw and "twin" in raw:
        comparisons.append(exact(raw["actual"]["path"], raw["twin"]["path"]))
    if "single" in arms:
        comparisons.append(semantic_state_pair(states["actual"][0], states["single"][0]))
        prompt = logical_prompt(scope, case, authority)
        require(value["checkpoint_depth"] == len(prompt) - (0 if value["equality"] else 16),
                "canonical input authority changed the selected checkpoint depth")
        canonical = []
        for arm, source in (("actual", "actual"), ("twin", "single")):
            graft = integer(arms[source]["origin"] - len(prompt) + 1, "source-bound graft offset")
            require(scope.packet["graft"] is not None or graft == 0, "ungrafted canonical origin invented physical prefix rows")
            tokens = semantic_tokens(scope, case, arms[source], raw[source])
            canonical.append(canonical_arm(scope, case, arm, source, binding("single-slot-parity-" + arm),
                                           arms[source], raw[source], prompt, tokens, graft))
        comparisons.append(exact(canonical[0][0], canonical[1][0]))
    negative = classification(scope, case, value, authority_binding, arms, raw, comparisons)
    if not negative and "actual" in raw and "twin" in raw:
        same_raw_twins(raw["actual"], raw["twin"])
    if not case.startswith("cancel-") and not negative:
        require(len(streams["actual"]) == len(streams["twin"]), "positive state event streams omit a transaction")
        comparisons.extend(equal_semantic_state(left[0], right[0]) for left, right in zip(streams["actual"][:-1], streams["twin"][:-1]))
    return {"case": case, "numeric_complete": True, "intentional_negative": negative,
            "comparisons": comparisons, "required_state_events": {arm: len(stream) for arm, stream in streams.items()}}


def verify(args):
    packet, packet_sha = read_json(args.packet, args.packet_sha256)
    source = catalog(args.catalog, args.catalog_sha256, packet["identity"])
    registration_binding = packet["registration"]
    require(type(registration_binding) is dict and set(registration_binding) == {"path", "sha256"},
            "closed preregistered manifest binding required")
    registration, _ = read_json(registration_binding["path"], registration_binding["sha256"], 1024 * 1024)
    require(packet["output"] == args.output_root and source["graft"] is (packet["graft"] is not None),
            "packet output or actual bank differs from compiled catalog")
    require(registration["schema"] == "hf2q.c2.matrix-registration.v1" and registration["identity"] == packet["identity"]
            and registration["mode"] == source["mode"] and registration["catalog_sha256"] == source["catalog_sha256"],
            "registered complete cases differ from compiled source catalog")
    limits = grants(registration["grants"])
    selected = registration["cases"]
    require(type(selected) is list and selected and len(set(selected)) == len(selected)
            and all(case in source["full_case_catalog"] for case in selected), "invalid complete-case shard selection")
    shard, _ = read_json(args.shard, args.shard_sha256)
    require(shard["schema"] == "hf2q.c2.matrix-shard.v1" and shard["identity"] == packet["identity"]
            and shard["mode"] == source["mode"] and shard["full_case_catalog"] == source["full_case_catalog"]
            and shard["catalog_sha256"] == source["catalog_sha256"] and shard["selected_complete_cases"] == selected
            and shard["all_selected_cases_completed"] is True and shard["grants"] == limits
            and shard["whole_catalog_in_this_scope"] is (len(selected) == len(source["full_case_catalog"]))
            and shard["route"] == packet["route"] and shard["independent_full_matrix_qualified"] is False
            and shard["timing_quality_release_qualified"] is False, "incomplete, substituted or overclaimed matrix shard")
    spent = shard["spent_before_index"]
    for key in ("child_stores", "records", "payload_bytes", "metadata_bytes", "retained_raw_bytes", "live_ram_bytes"):
        require(integer(spent[key], "spent " + key) <= limits[key], "matrix exhausted its aggregate grant")
    for key in ("peak_live_ram_bytes", "preflight_case_peak_bytes"):
        require(integer(spent[key], key, 1) <= limits["live_ram_bytes"], "matrix live RAM exceeds predeclared bound")
    scope = Scope(packet, packet_sha, registration, packet["registration"]["sha256"], shard["acquisition"])
    required = {key for case in selected for key in source["required_artifacts"][case]}
    artifacts = shard["artifacts"]
    require(type(artifacts) is dict and set(artifacts) == required, "whole required case artifact set was omitted/replaced")
    for index, case in enumerate(selected):
        for key in source["required_artifacts"][case]:
            require(type(artifacts[key]) is dict and set(artifacts[key]) == {"case_index", "binding"}
                    and integer(artifacts[key]["case_index"], "artifact case index") == index,
                    "artifact changed its preregistered complete case")
    results = [verify_case(scope, case, artifacts, required) for case in selected]
    result = {"schema": "hf2q.c2.complete-shard-numeric-review.v1", "packet_sha256": packet_sha,
              "shard_sha256": args.shard_sha256, "catalog_binding": {"path": args.catalog, "sha256": args.catalog_sha256},
              "registration": packet["registration"], "cases": results, "all_selected_numeric_cases_complete": True,
              "native_root_independently_attested_by_this_tool": False, "whole_matrix_qualified": False,
              "admission_quality_timing_release_qualified": False}
    return {"review": write_new(args.output, result), "cases": len(results), "release_qualified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "verify-shard"):
        command = commands.add_parser(name)
        command.add_argument("--catalog", required=True)
        command.add_argument("--catalog-sha256", required=True)
        command.add_argument("--output", required=True)
        if name == "register":
            command.add_argument("--identity", required=True)
            command.add_argument("--grants", required=True)
            command.add_argument("--case", action="append", required=True)
        else:
            for flag in ("packet", "packet-sha256", "output-root", "shard", "shard-sha256"):
                command.add_argument("--" + flag, required=True)
    args = parser.parse_args(argv)
    try:
        result = register(args) if args.command == "register" else verify(args)
    except (Invalid, ValueError, TypeError, KeyError, OSError) as error:
        print(f"C2 evidence refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
