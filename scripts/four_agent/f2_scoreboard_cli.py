#!/usr/bin/env python3
"""Audit all retained raw cohorts through Q2, then compute the frozen F2 score.

This command cannot launch a runtime or replace failed cohorts. Its frozen plan
must bind Q2's independent QE attestations and every raw cohort manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from f2_scoreboard import paired_scoreboard, require


def bound_json(ref, cap=4*1024*1024):
    require(set(ref) == {"path", "sha256"}, "closed BoundFile")
    p = Path(ref["path"])
    require(p.is_absolute() and p.resolve() == p and not p.is_symlink() and
            p.is_file() and p.stat().st_size <= cap, "bounded canonical artifact")
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "artifact digest mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def audit_scoreboard(plan, cohort_records):
    from q2_certify import certify_cohort
    require(len(cohort_records) == 18, "retain all eighteen fixed cohorts")
    # Profile bytes must be retained; Q2 additionally verifies raw capture/default
    # joins. This does not turn the canonical greedy builder into a primary arm.
    bound_json(plan["production_client_profile"])
    audited, certificates = [], []
    for cohort in cohort_records:
        cid = cohort["cohort_id"]
        expected = plan["certification_expectations"][cid]
        require(expected["cohort_id"] == cid and expected["required_cardinality"] == 4 and
                expected["identity"] == plan["arms"][cohort["arm"]], "externally frozen N4 expectation")
        # Digest-check before handing the exact path to the actual raw auditor.
        ref = cohort["cohort_manifest"]
        bound_json(ref)
        try:
            certificate = certify_cohort(ref["path"], expected)
        except Exception as error:
            # Retain a rejected outcome, never omit or replace its elapsed time.
            certificate = {"schema":"hf2q.q2.certification.v1", "cohort_id":cid,
                "identity":expected["identity"], "required_cardinality":4,
                "status":"Rejected", "failure_class":"infrastructure",
                "failure":str(error), "verified_refs":{"manifest":ref}}
        # Refuse a changed manifest after the auditor reads it.
        bound_json(ref)
        certificates.append(certificate)
        audited.append({**cohort, "certification":certificate,
            "trace_status":"Qualified" if certificate.get("trace", {}).get("qualified") is True
            and certificate.get("phase_window", {}).get("status") == "Qualified" else "Unqualified"})
    result = paired_scoreboard(plan, audited)
    result["raw_certifications"] = certificates
    result["calculation_only"] = False
    result["raw_certification_performed"] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--cohorts", required=True)
    parser.add_argument("--cohorts-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = bound_json({"path":args.plan,"sha256":args.plan_sha256})
    cohorts = bound_json({"path":args.cohorts,"sha256":args.cohorts_sha256})
    result = audit_scoreboard(plan, cohorts)
    output = Path(args.output)
    require(output.is_absolute() and output.parent.resolve() == output.parent and
            not os.path.lexists(output) and not any(os.path.lexists(p/".git")
            for p in (output.parent, *output.parent.parents)), "new private output outside Git")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["status"] == "Qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
