"""Pure controls for the long-context workload variant and the unchanged canonical workload."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import coding_long_context as lc
import q2_fake_service as fake
import q2_run_audit as audit
import q5_calibration
import q5_case
from qwen38_n1_smoke import build_request, initial_messages
from qwen38_n1_tools import IMMUTABLE, INITIAL, Tools
from qwen38_n1_verdict import FILE_LIMIT, WIRE_LIMIT, Invalid, loads, wire
from qwen_n4_live_smoke import AGENTS, MODULES, aggregate, identity, workload_variant
from test_coding_instances import CANONICAL_FILES, CANONICAL_N4, RUN as INSTANCE_RUN, digest, fixture_messages

HERE = Path(__file__).resolve().parent
RUN = "long_context_fixture_01"
# sha256 of the actual wire bytes of the canonical N4 turn-0/turn-1 requests (before any reply),
# computed from the pre-variant client sources: the default path must stay byte-identical.
CANONICAL_REQUESTS = {("agent-0", 0): "5bdcdd545ac56d4c11c877ff841222451da1b28e9613a3142de669c44a63b30c",
                      ("agent-0", 1): "1e400b2cc1d09c6429258999bd1942fc08f6d56b95ea7246d9d02e4b5ba016d5",
                      ("agent-3", 0): "eb98a9cd9af676c5e873955c08dd47eabbf2f09e2e302c7dfa32ef1a50e0315d",
                      ("agent-3", 1): "8dfe14260766735bb8f3cf9b24c875bf4e1a89c3c0b354d311ea1720b073cd71"}
PASSING_RECEIPT = {"verdict": {"verdict": "CalibrationPass"}, "cleanup": {"proven": True}, "cleanup_proven": True}


def schemas():
    tools = json.loads(json.dumps(loads((HERE / "fixtures" / "canonical-coding-requests.json").read_bytes())
                                  [0]["request"]["tools"]))
    for tool in tools:
        tool["function"]["strict"] = False
    return tools


def packet(variant, agent="agent-0"):
    return {"schema": "hf2q.qwen-n4-agent-approval.v1", "profile": "adr063-qwen-n4-live-v1",
            "diagnostic_slots": 4, "agent_id": agent, "run_id": RUN, "server_pid": 1,
            "barrier_dir": "/private/b", "barrier_token": "a" * 32, "agent_output": "/private/o",
            "initial_identity_sha256": hashlib.sha256(identity(agent, RUN).encode()).hexdigest(),
            **({} if variant is None else variant.binding())}


class CanonicalUnchanged(unittest.TestCase):
    def test_canonical_requests_are_byte_identical(self):
        for (agent, turn), pinned in CANONICAL_REQUESTS.items():
            messages = initial_messages(fixture_messages(), identity(agent, RUN))
            body = wire(build_request("Qwen3.8 27B", messages, schemas(), turn, "qwen"), "qwen")
            self.assertEqual(hashlib.sha256(body).hexdigest(), pinned, (agent, turn))

    def test_canonical_files_prompts_and_approval_unchanged(self):
        self.assertEqual(hashlib.sha256(json.dumps({**IMMUTABLE, "src/ports.rs": INITIAL}, sort_keys=True)
                                        .encode()).hexdigest(), CANONICAL_FILES)
        for agent in AGENTS:
            self.assertEqual(digest(initial_messages(fixture_messages(), identity(agent, INSTANCE_RUN))),
                             CANONICAL_N4[agent])
        self.assertIsNone(workload_variant({}, None))
        for key in ("workload_variant", "workload_fixture_sha256"):
            with self.subTest(key=key), self.assertRaises(Invalid):
                workload_variant({key: "x"}, None)

    def test_the_variant_leaves_prompts_and_first_two_requests_unchanged(self):
        canonical, variant = lc.estimate(), lc.estimate(lc.select(lc.ID))
        self.assertEqual(canonical[:2], variant[:2])
        self.assertTrue(all(v["request_bytes"] > c["request_bytes"] for c, v in zip(canonical[2:], variant[2:])))


class Fixture(unittest.TestCase):
    def test_rendering_is_deterministic_and_pinned(self):
        raw = lc.FIXTURE.read_bytes()
        self.assertEqual(lc.render(), lc.render())
        self.assertEqual(lc.render().encode(), raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), lc.FIXTURE_SHA256)
        self.assertEqual(lc.select(lc.ID).fixture_sha256, lc.FIXTURE_SHA256)

    def test_only_tests_ports_rs_grows_and_the_test_count_is_unchanged(self):
        variant = lc.select(lc.ID)
        self.assertEqual({k: v for k, v in variant.immutable.items() if k != "tests/ports.rs"},
                         {k: v for k, v in IMMUTABLE.items() if k != "tests/ports.rs"})
        self.assertEqual(variant.initial, INITIAL)
        tests = variant.immutable["tests/ports.rs"]
        self.assertTrue(tests.startswith(IMMUTABLE["tests/ports.rs"]))
        support = tests[len(IMMUTABLE["tests/ports.rs"]):]
        self.assertEqual(support, lc.FIXTURE.read_text())
        self.assertEqual(tests.count("#[test]"), IMMUTABLE["tests/ports.rs"].count("#[test]"))
        self.assertNotIn("parse_port", support.replace("`parse_port`", ""))
        self.assertIn("#[allow(dead_code)]\nmod netconf {", support)
        self.assertEqual(support.count("{"), support.count("}"))
        self.assertTrue(128 * 1024 <= len(tests.encode()) <= lc.READ_LIMIT)

    def test_fixture_drift_and_unknown_variants_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drift.rs"
            path.write_bytes(lc.FIXTURE.read_bytes() + b"\n")
            with self.assertRaises(Invalid):
                lc.select(lc.ID, path)
        for name in ("long-context-v2", "", None):
            with self.subTest(name=name), self.assertRaises(Invalid):
                lc.select(name)

    def test_schema_validation(self):
        bad = (("listener", [("backlog", "u16", 1, 70000, 2, "x", "y")]),)
        for schema in (bad, (("listener", [("backlog", "u8", 5, 4, 4, "x", "y")]),),
                       (("listener", [("backlog", "u8", 1, 4, 2, "x", 'say "hi"')]),),
                       (("listener", [("a", "u8", 1, 4, 2, "x", "y"), ("a", "u8", 1, 4, 2, "x", "y")]),)):
            with self.subTest(schema=schema), self.assertRaises(Invalid):
                lc.render(schema)


class Binding(unittest.TestCase):
    def test_approval_binds_variant_name_and_fixture_sha256(self):
        variant = lc.select(lc.ID)
        self.assertEqual(variant.binding(), {"workload_variant": lc.ID, "workload_fixture_sha256": lc.FIXTURE_SHA256})
        self.assertEqual(variant.record(), {"id": lc.ID, "fixture_sha256": lc.FIXTURE_SHA256})
        self.assertEqual(workload_variant(packet(variant), lc.ID).record(), variant.record())
        for change in ({"workload_variant": "long-context-v2"}, {"workload_fixture_sha256": "0" * 64},
                       {"workload_variant": None}):
            with self.subTest(change=change), self.assertRaises(Invalid):
                workload_variant({**packet(variant), **change}, lc.ID)
        with self.assertRaises(Invalid):  # a canonical approval cannot run the variant
            workload_variant(packet(None), lc.ID)
        with self.assertRaises(Invalid):  # nor can a variant approval run the canonical files
            workload_variant(packet(variant), None)

    def test_aggregate_refuses_agents_from_another_workload(self):
        from test_qwen_n4_live_smoke import PID, RUN as N4_RUN, results
        record = lc.select(lc.ID).record()
        values = results()
        for agent in AGENTS:
            values[agent]["workload_variant"] = record
        ok = dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True)
        self.assertEqual(aggregate(values, *ok, PID, N4_RUN, variant_record=record),
                         "FourAgentProtocolCompleteAwaitingIndependentReview")
        for bad in (None, {**record, "fixture_sha256": "0" * 64}):
            with self.subTest(record=bad), self.assertRaises(Invalid):
                aggregate(values, *ok, PID, N4_RUN, variant_record=bad)
        values["agent-1"].pop("workload_variant")
        with self.assertRaises(Invalid):
            aggregate(values, *ok, PID, N4_RUN, variant_record=record)

    def test_agent_command_forwards_the_variant(self):
        from test_qwen_n4_live_smoke import RUN as N4_RUN, TOKEN, client_args
        from qwen_n4_live_smoke import agent_command
        with tempfile.TemporaryDirectory() as tmp:
            approval = Path(tmp) / "agent-0.approval.json"
            approval.write_text("{}")
            args = client_args()
            canonical = agent_command(args, Path("/m"), "agent-0", Path("/r/a"), approval, Path("/r/b"), N4_RUN, TOKEN)
            args.workload = lc.ID
            command = agent_command(args, Path("/m"), "agent-0", Path("/r/a"), approval, Path("/r/b"), N4_RUN, TOKEN)
            self.assertEqual(command, canonical + ["--workload", lc.ID])

    def test_owned_fixture_reads_the_long_tests_file_only_with_the_variant_limit(self):
        variant = lc.select(lc.ID)
        with tempfile.TemporaryDirectory() as directory:
            for name, limit, readable in (("variant", variant.read_limit, True), ("canonical", FILE_LIMIT, False)):
                tools = Tools(Path(directory) / name, "/not/executed/bin/cargo", "/not/executed/sandbox",
                              "/Library/Developer/SDKs/Mock.sdk", immutable=variant.immutable,
                              initial=variant.initial, read_limit=limit)
                try:
                    self.assertEqual(tools.read("src/ports.rs"), INITIAL)
                    if readable:
                        self.assertEqual(tools.read("tests/ports.rs"), variant.immutable["tests/ports.rs"])
                    else:
                        with self.assertRaises(Invalid):
                            tools.read("tests/ports.rs")
                    tools.invariant()
                finally:
                    tools.close()
                    tools.cwd.rmdir()
            with self.assertRaises(Invalid):
                Tools(Path(directory) / "low", "/c", "/s", "/Library/Developer/SDKs/Mock.sdk", read_limit=FILE_LIMIT - 1)


class PromptGrowth(unittest.TestCase):
    def test_offline_estimate_reaches_long_context_within_the_wire_bound(self):
        canonical, variant = lc.estimate(), lc.estimate(lc.select(lc.ID))
        self.assertEqual([r["turn"] for r in variant], list(range(7)))
        self.assertTrue(2000 <= canonical[-1]["prompt_tokens_estimate"] <= 4000)  # the ~2.8k canonical history
        self.assertTrue(all(r["prompt_tokens_estimate"] < 1000 for r in variant[:2]))
        self.assertTrue(all(32000 <= r["prompt_tokens_estimate"] <= 64000 for r in variant[2:]))
        self.assertEqual(sorted(r["request_bytes"] for r in variant), [r["request_bytes"] for r in variant])
        self.assertLessEqual(variant[-1]["request_bytes"], WIRE_LIMIT - 64 * 1024)  # headroom for longer reasoning
        self.assertEqual([r["prompt_tokens_estimate"] for r in variant], [r["request_bytes"] // 4 for r in variant])

    def test_client_exchange_carries_the_long_history_and_the_audit_rederives_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = Path(tmp) / "client"
            summary = fake.honest_run(client, variant=lc.ID)
            record = lc.select(lc.ID).record()
            self.assertEqual(summary["workload_variant"], record)
            for agent in AGENTS:
                result = summary["agents"][agent]
                self.assertEqual(result["workload_variant"], record)
                self.assertEqual(result["tools"][1]["result"], lc.select(lc.ID).immutable["tests/ports.rs"])
                prompts = [t["usage"]["prompt_tokens"] for t in result["turns"]]  # fake usage: request bytes / 4
                self.assertTrue(all(p < 1000 for p in prompts[:2]) and all(32000 <= p <= 64000 for p in prompts[2:]))
                self.assertTrue(all(t["request_bytes"] <= WIRE_LIMIT for t in result["turns"]))
            report, reason = audit.check_run(PASSING_RECEIPT, client, summary, None, "hf2q", True)
            self.assertIsNone(reason)
            self.assertEqual(report["progress"]["width"], 4)
            for forged in ({**summary, "workload_variant": None},
                           {k: v for k, v in summary.items() if k != "workload_variant"}):
                _, reason = audit.check_run(PASSING_RECEIPT, client, forged, None, "hf2q", True)
                self.assertEqual(reason, "run audit: aggregate workload variant")


class Cases(unittest.TestCase):
    def test_checked_in_cases_are_the_derived_canonical_cases(self):
        for base, derived in (("qwen38.json", "qwen38-long-context.json"),
                              ("peer-qwen38.json", "peer-qwen38-long-context.json")):
            with self.subTest(case=derived):
                case = q5_case.loads((HERE / "q5_cases" / base).read_bytes())
                checked = q5_case.loads((HERE / "q5_cases" / derived).read_bytes())
                self.assertEqual(checked, lc.derive_case(case))
                self.assertEqual(checked["client"]["argv"], case["client"]["argv"] + ["--workload", lc.ID])
                self.assertEqual(checked["serve"], case["serve"])
                self.assertEqual(checked["workload"]["id"], case["workload"]["id"] + "-" + lc.ID)
                values = {"REPO": str(HERE.parent.parent)}
                approval = checked["client"]["approval"]["set"]
                self.assertEqual(approval["workload_variant"], lc.ID)
                self.assertEqual(q5_calibration.resolve_set(approval["workload_fixture_sha256"], values),
                                 lc.FIXTURE_SHA256)
                self.assertEqual(set(q5_calibration.resolve_set(approval["sources"], values)),
                                 set(MODULES) | set(lc.MODULES))
                plan = q5_case.build_plan(checked, q5_calibration.builtins(Path("/private/tmp/long-context-dry")), {})
                self.assertEqual(plan["client"]["argv"][-2:], ["--workload", lc.ID])

    def test_tensorfold_d2_arms_derive_the_same_way(self):
        import native_peer_tensorfold as tf
        for name in ("native-tensorfold-qwen38.json", "native-tensorfold-qwen38-dflash2.json"):
            with self.subTest(case=name):
                case = q5_case.loads((HERE / "q5_cases" / name).read_bytes())
                derived = lc.derive_case(case)
                self.assertEqual(derived["client"]["argv"][-2:], ["--workload", lc.ID])
                self.assertEqual({Path(s).name for s in derived["client"]["approval"]["set"]["sources"]["sha256_of"]},
                                 set(MODULES) | set(tf.CLIENT_MODULES) | set(lc.MODULES))
                self.assertEqual(derived["serve"], case["serve"])

    def test_refuses_cases_that_already_select_a_workload(self):
        case = q5_case.loads((HERE / "q5_cases" / "qwen38.json").read_bytes())
        for mutate in (lambda c: c["client"]["argv"].extend(["--instance", "1"]),
                       lambda c: c["client"]["argv"].extend(["--phase-plan", "2+2"]),
                       lambda c: c["client"]["approval"]["set"].update(workload_variant=lc.ID),
                       lambda c: c["client"]["approval"]["set"].update(instance=1),
                       lambda c: c["client"]["argv"].__setitem__(4, "{REPO}/scripts/other_client.py"),
                       lambda c: c["client"]["approval"]["set"].pop("sources")):
            bad = json.loads(json.dumps(case))
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(Invalid):
                lc.derive_case(bad)
        with self.assertRaises(Invalid):
            lc.derive_case(lc.derive_case(case))
        self.assertTrue(re.fullmatch(r"[a-z0-9-]+", lc.derive_case(case)["workload"]["id"]))


if __name__ == "__main__":
    unittest.main()
