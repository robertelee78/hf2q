"""Pure controls for the opt-in Qwen vendor request setup. No HTTP/server/model."""
import json
import tempfile
import unittest
from pathlib import Path

import family_peer as fp
import q2_certify
import quality_campaign as qc
import q5_case
import request_setup as rs
from qwen38_n1_smoke import PREFLIGHT, build_request
from qwen38_n1_verdict import Invalid, wire
from qwen_n4_live_smoke import AGENTS, agent_command, aggregate, preflight as n4_preflight, validate_preflight
from test_family_peer import chain, schemas
from test_qwen_n4_live_smoke import PID, RUN, TOKEN, client_args, results

HERE = Path(__file__).resolve().parent
SETUP = rs.NAME
MODEL = "Qwen3.6 35B A3B"
VENDOR = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
          "repetition_penalty": 1.0}


class Requests(unittest.TestCase):
    def test_hf2q_request_carries_the_model_card_sampling_and_nothing_else(self):
        for turn in range(7):
            canonical = build_request(MODEL, chain(), schemas(), turn, "qwen")
            vendor = build_request(MODEL, chain(), schemas(), turn, "qwen", "hf2q", SETUP)
            self.assertEqual({k: v for k, v in vendor.items() if k not in VENDOR}, canonical)
            self.assertEqual({k: vendor[k] for k in VENDOR}, VENDOR)
            self.assertEqual(wire(vendor, "qwen", SETUP), json.dumps(vendor, ensure_ascii=False,
                                                                       separators=(",", ":")).encode())
            with self.assertRaises(Invalid):  # the canonical wire still refuses sampling fields
                wire(vendor, "qwen")
            with self.assertRaises(Invalid):  # and the setup wire refuses the canonical omissions
                wire(canonical, "qwen", SETUP)

    def test_peer_request_renames_the_penalty_and_sends_no_budget(self):
        for turn in range(7):
            hf = build_request(MODEL, chain(), schemas(), turn, "qwen", "hf2q", SETUP)
            peer = build_request(MODEL, chain(), schemas(), turn, "qwen", "peer", SETUP)
            self.assertNotIn("reasoning_budget_tokens", peer)
            self.assertNotIn("repetition_penalty", peer)
            self.assertNotIn("hf2q_overflow_policy", peer)
            self.assertEqual(peer["repeat_penalty"], 1.0)
            self.assertEqual(fp.canonical(peer, "qwen", SETUP), hf)
            wire(fp.canonical(peer, "qwen", SETUP), "qwen", SETUP)
            raw = json.dumps(peer, ensure_ascii=False, separators=(",", ":")).encode()
            q2_certify.request_wire(peer, raw, "qwen", "peer", SETUP)
            with self.assertRaises(Invalid):  # a canonical peer request carries its budget
                fp.canonical({**peer, "reasoning_budget_tokens": 512}, "qwen", SETUP)

    def test_drift_refuses(self):
        request = build_request(MODEL, chain(), schemas(), 1, "qwen", "hf2q", SETUP)
        changes = [("temperature", 0.0), ("temperature", 1.0), ("top_k", 20.0), ("top_p", 1), ("min_p", 0),
                   ("repetition_penalty", 1.05), ("presence_penalty", 1.5)]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                wire({**request, key: value}, "qwen", SETUP)
        for key in VENDOR:
            with self.subTest(missing=key), self.assertRaises(Invalid):
                wire({k: v for k, v in request.items() if k != key}, "qwen", SETUP)
        for key, value in (("seed", 1), ("frequency_penalty", 0.0), ("reasoning_effort", "high")):
            with self.subTest(extra=key), self.assertRaises(Invalid):
                wire({**request, key: value}, "qwen", SETUP)
        peer = build_request(MODEL, chain(), schemas(), 1, "qwen", "peer", SETUP)
        with self.assertRaises(Invalid):
            fp.canonical({**peer, "repeat_penalty": 1.05}, "qwen", SETUP)
        with self.assertRaises(Invalid):
            fp.canonical({k: v for k, v in peer.items() if k != "repeat_penalty"}, "qwen", SETUP)

    def test_setup_is_qwen_hf2q_or_peer_only(self):
        for family in ("gemma4", "deepseek4"):
            with self.subTest(family=family), self.assertRaises(Invalid):
                build_request("m", chain(), schemas(), 0, family, "hf2q", SETUP)
        with self.assertRaises(Invalid):
            build_request(MODEL, chain(), schemas(), 0, "qwen", "omlx", SETUP)
        with self.assertRaises(Invalid):
            build_request(MODEL, chain(), schemas(), 0, "qwen", "hf2q", "qwen36-vendor-coding-v2")
        with self.assertRaises(Invalid):
            rs.apply({"temperature": 0}, SETUP)  # never overwrites


class Binding(unittest.TestCase):
    def test_activation_needs_cli_and_approval(self):
        rs.validate_activation(SETUP, {"request_setup": SETUP}, "qwen", "peer", 4)
        for packet, family, engine, slots in (({}, "qwen", "hf2q", 4), ({"request_setup": "x"}, "qwen", "hf2q", 4),
                                              ({"request_setup": SETUP}, "gemma4", "hf2q", 4),
                                              ({"request_setup": SETUP}, "qwen", "omlx", 4),
                                              ({"request_setup": SETUP}, "qwen", "hf2q", 1)):
            with self.subTest(packet=packet, family=family, engine=engine, slots=slots), self.assertRaises(Invalid):
                rs.validate_activation(SETUP, packet, family, engine, slots)

    def test_preflight_replaces_only_the_greedy_claim(self):
        for engine in ("hf2q", "peer"):
            canonical = n4_preflight(PREFLIGHT) if engine == "hf2q" else fp.preflight(n4_preflight(PREFLIGHT), "qwen")
            keys = {k: True for k in rs.preflight(canonical, SETUP)}
            self.assertEqual(set(keys) - {rs.SETUP_PREFLIGHT[SETUP]}, set(canonical) - {rs.GREEDY_PREFLIGHT})
            self.assertIn(rs.SETUP_PREFLIGHT[SETUP], keys)
            validate_preflight(keys, PREFLIGHT, engine, "qwen", SETUP)
            with self.assertRaises(Invalid):
                validate_preflight(keys, PREFLIGHT, engine, "qwen")

    def test_aggregate_refuses_mixed_setups(self):
        values = results()
        for agent in AGENTS:
            values[agent]["request_setup"] = SETUP
        ok = dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True)
        self.assertEqual(aggregate(values, *ok, PID, RUN, request_setup=SETUP),
                         "FourAgentProtocolCompleteAwaitingIndependentReview")
        with self.assertRaises(Invalid):
            aggregate(values, *ok, PID, RUN)
        values["agent-2"].pop("request_setup")
        with self.assertRaises(Invalid):
            aggregate(values, *ok, PID, RUN, request_setup=SETUP)

    def test_agent_command_forwards_the_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval = Path(tmp) / "agent-0.approval.json"
            approval.write_text("{}")
            args = client_args()
            canonical = agent_command(args, Path("/m"), "agent-0", Path("/r/a"), approval, Path("/r/b"), RUN, TOKEN)
            args.request_setup = SETUP
            command = agent_command(args, Path("/m"), "agent-0", Path("/r/a"), approval, Path("/r/b"), RUN, TOKEN)
            self.assertEqual(command, canonical + ["--request-setup", SETUP])


class Cases(unittest.TestCase):
    def load(self, name):
        return q5_case.loads((HERE / "q5_cases" / name).read_bytes())

    def test_vendor_cases_differ_from_their_bases_only_in_setup_fields(self):
        for name, base in (("qwen36-vendor.json", "qwen36.json"), ("peer-qwen36-vendor.json", "peer-qwen36.json")):
            case, original = self.load(name), self.load(base)
            client = case["client"]
            self.assertEqual(client["argv"], original["client"]["argv"] + ["--request-setup", SETUP])
            approval = client["approval"]["set"]
            self.assertEqual(approval["request_setup"], SETUP)
            self.assertEqual(approval["sources"]["sha256_of"], original["client"]["approval"]["set"]["sources"]
                             ["sha256_of"] + [f"{{REPO}}/scripts/four_agent/{m}" for m in rs.EXTRA_MODULES])
            self.assertEqual(case["variables"], original["variables"])
            self.assertNotEqual(case["workload"]["id"], original["workload"]["id"])
            derived = qc.derive_case(case, 3)
            self.assertEqual(derived["client"]["argv"][-4:], ["--request-setup", SETUP, "--instance", "3"])

    def test_hf2q_vendor_launch_disables_both_budgets_and_the_penalty(self):
        case, base = self.load("qwen36-vendor.json"), self.load("qwen36.json")
        env = case["serve"]["env"]
        self.assertEqual((env["REP_PENALTY"], env["THINKING_TOKEN_BUDGET"], env["TOOL_THINKING_TOKEN_BUDGET"]),
                         ("1.0", "0", "0"))
        self.assertEqual({k: v for k, v in env.items() if k not in ("REP_PENALTY", "THINKING_TOKEN_BUDGET",
                                                                     "TOOL_THINKING_TOKEN_BUDGET")},
                         {k: v for k, v in base["serve"]["env"].items() if k not in (
                             "REP_PENALTY", "THINKING_TOKEN_BUDGET", "TOOL_THINKING_TOKEN_BUDGET")})
        checks = {c["field"]: c["equals"] for c in case["readiness"]["checks"]}
        self.assertEqual(checks["measurement_snapshot.sampling_defaults.thinking_token_budget"], 0)
        self.assertEqual(checks["measurement_snapshot.sampling_defaults.tool_thinking_token_budget"], 0)

    def test_peer_vendor_launch_uses_the_same_sampling_and_no_budget(self):
        argv, base = self.load("peer-qwen36-vendor.json")["serve"]["argv"], self.load("peer-qwen36.json")["serve"]["argv"]
        flags = {"--temp": "0.6", "--top-k": "20", "--top-p": "0.95", "--min-p": "0", "--repeat-penalty": "1.0",
                 "--presence-penalty": "0", "--reasoning-budget": "-1"}
        for flag, value in flags.items():
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        stripped = [a for i, a in enumerate(argv) if a not in flags and (i == 0 or argv[i - 1] not in flags)]
        base_stripped = [a for i, a in enumerate(base) if a not in flags and (i == 0 or base[i - 1] not in flags)]
        self.assertEqual(stripped, base_stripped)


if __name__ == "__main__":
    unittest.main()
