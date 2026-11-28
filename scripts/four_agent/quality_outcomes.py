"""Q1 task instances are label × agent, adjudicated with the existing patch-review format."""
import re

from qwen_n4_live_smoke import AGENTS

PATCH_REVIEW_SCHEMA = "hf2q.adr063.q3-patch-review.v1"
OUTCOMES_SCHEMA = "hf2q.q1.paired-task-outcomes.v1"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def adjudicate(manifest, review):
    """Return (paired document, pending source hashes); never score an unreviewed pass."""
    plan = manifest["plan"]
    labels = plan["instances"]
    require(type(labels) is int and labels > 0, "quality label count must be positive")
    floor_only = plan["reference_kind"] is None
    arms = ("subject",) if floor_only else ("subject", "reference")
    expected = {(label, arm) for label in range(labels) for arm in arms}
    runs = {}
    for run in manifest["runs"]:
        key = (run["instance"], run["arm"])
        require(key in expected and key not in runs, "duplicate or foreign quality run")
        require(run["outcome"] in ("Pass", "Fail"), "quality run has no task outcome")
        require(bool(run["attempts"]), "quality task has no attempt")
        agents = run["attempts"][-1].get("agents")
        require(isinstance(agents, dict) and set(agents) == set(AGENTS), "quality agent set differs")
        runs[key] = agents
    require(set(runs) == expected, "quality run set is incomplete")
    verdicts = {}
    if review is not None:
        require(isinstance(review, dict) and review.get("schema") == PATCH_REVIEW_SCHEMA,
                "unexpected independent patch-review schema")
        verdicts = review.get("verdicts")
        require(isinstance(verdicts, dict) and all(
            isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) and verdict in ("Pass", "Fail")
            for sha, verdict in verdicts.items()), "malformed patch-review verdicts")
    pending, instances = set(), []
    for label in range(labels):
        for agent in AGENTS:
            row = {"id": f"i{label:02d}-{agent}"}
            for arm in arms:
                outcome = runs[(label, arm)][agent]
                require(outcome.get("outcome") in ("Pass", "Fail"), "unsettled agent task")
                passed = False
                if outcome["outcome"] == "Pass":
                    sha = outcome.get("final_source_sha256")
                    require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha),
                            "completed task has no final source identity")
                    if sha not in verdicts:
                        pending.add(sha)
                    else:
                        passed = verdicts[sha] == "Pass"
                row[arm] = passed
            if floor_only:
                row["reference"] = None
            instances.append(row)
    if pending:
        return None, sorted(pending)
    return {"schema": OUTCOMES_SCHEMA, "reference_kind": plan["reference_kind"],
            "instances": instances}, []
