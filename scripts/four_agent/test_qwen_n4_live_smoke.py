"""Pure controls for actual-used N4 identity/approval/barrier/aggregation boundaries."""
import hashlib
import unittest
from pathlib import Path

from qwen38_n1_verdict import Invalid, EXPECTED
from qwen_n4_live_smoke import (AGENTS, PROFILE, aggregate, barrier_release, identity,
                               preflight, validate_agent_binding, validate_ready, validate_release, validate_preflight)

RUN = "n4_source_fixture_20261025"
TOKEN = "a" * 32
PID = 12345


def ready():
    return {a: {"schema": "hf2q.n4-agent-ready.v1", "agent_id": a, "run_id": RUN,
                "server_pid": PID, "barrier_token": TOKEN, "output": f"/private/run/{a}",
                "identity_sha256": hashlib.sha256(identity(a, RUN).encode()).hexdigest(),
                "initial_messages_sha256": hashlib.sha256(a.encode()).hexdigest()}
            for a in AGENTS}


def results():
    return {a: {"schema": "hf2q.qwen-n4-agent-result.v1", "profile": PROFILE,
                "agent_id": a, "run_id": RUN, "server_pid": PID, "diagnostic_slots": 4,
                "status": "ProtocolCompleteAwaitingIndependentPatchReview", "cleanup": True,
                "barrier": {"status": "Released"}, "full_prefix_reuse": "PendingPostOracle",
                "runtime_qualified": False, "model_qualified": False,
                "turns": [{"turn": i, "stream": i % 2 == 1} for i in range(7)],
                "tools": [{"name": e[0], "cleanup": True, "passed": True, "source_sha256": "hash"}
                          for e in EXPECTED],
                "final_source_sha256": "hash", "observed_final_source_sha256": "hash",
                "initial_identity_sha256": hashlib.sha256(identity(a, RUN).encode()).hexdigest(),
                "initial_messages_sha256": hashlib.sha256(a.encode()).hexdigest()}
            for a in AGENTS}


class FourAgent(unittest.TestCase):
    def test_identity_unique_and_frozen(self):
        self.assertEqual(len({identity(a, RUN) for a in AGENTS}), 4)
        self.assertEqual(identity(AGENTS[0], RUN), identity(AGENTS[0], RUN + "x"))

    def test_unknown_or_injected_identity_rejected(self):
        for agent, run in (("agent-4", RUN), (AGENTS[0], "short"), (AGENTS[0], "../../private/escape")):
            with self.subTest(agent=agent), self.assertRaises(Invalid):
                identity(agent, run)

    def test_four_slot_preflight_never_relabels_one_slot(self):
        value = preflight({"ctx262144_slots1_verified", "cpu_owned_io_3g_tool2g_client128m", "other"})
        self.assertNotIn("ctx262144_slots1_verified", value)
        self.assertIn("ctx262144_slots4_verified", value)
        self.assertIn("dedicated_host_live_pressure_availablememory", value)
        self.assertIn("other", value)

    def test_n4_pool_selection_key_truthful_for_both_modes(self):
        base = {"ctx262144_slots1_verified", "profile24g_ack_verified", "other"}
        expected = preflight(base)
        self.assertNotIn("profile24g_ack_verified", expected)
        self.assertIn("pool_profile_selection_verified", expected)
        validate_preflight(dict.fromkeys(expected, True), base)
        old = dict.fromkeys((expected - {"pool_profile_selection_verified"}) |
                            {"profile24g_ack_verified"}, True)
        with self.assertRaises(Invalid):
            validate_preflight(old, base)
        with self.assertRaises(Invalid):
            validate_preflight(dict.fromkeys(expected - {"pool_profile_selection_verified"}, True), base)
        bad = dict.fromkeys(expected, True)
        bad["pool_profile_selection_verified"] = False
        with self.assertRaises(Invalid):
            validate_preflight(bad, base)

    def test_actual_agent_approval_boundary(self):
        agent = AGENTS[0]
        packet = {"schema": "hf2q.qwen-n4-agent-approval.v1", "profile": PROFILE,
                  "diagnostic_slots": 4, "agent_id": agent, "run_id": RUN, "server_pid": PID,
                  "barrier_dir": str(Path("/private/run/barrier").resolve()), "barrier_token": TOKEN,
                  "agent_output": str(Path("/private/run/agent-0").resolve()),
                  "initial_identity_sha256": hashlib.sha256(identity(agent, RUN).encode()).hexdigest()}
        validate_agent_binding(packet, agent, RUN, PID, "/private/run/barrier", TOKEN, "/private/run/agent-0")
        for key, wrong in (("diagnostic_slots", 1), ("agent_id", AGENTS[1]), ("server_pid", PID + 1),
                           ("run_id", RUN + "x"), ("barrier_token", "b" * 32)):
            with self.subTest(key=key), self.assertRaises(Invalid):
                validate_agent_binding({**packet, key: wrong}, agent, RUN, PID,
                                       "/private/run/barrier", TOKEN, "/private/run/agent-0")

    def test_barrier_requires_all_four(self):
        records = ready()
        del records[AGENTS[3]]
        with self.assertRaises(Invalid):
            validate_ready(records, RUN, PID, TOKEN)

    def test_barrier_rejects_same_prompt_or_output(self):
        for key in ("initial_messages_sha256", "output"):
            records = ready()
            records[AGENTS[1]][key] = records[AGENTS[0]][key]
            with self.subTest(key=key), self.assertRaises(Invalid):
                validate_ready(records, RUN, PID, TOKEN)

    def test_barrier_rejects_other_server_or_stale_run(self):
        for key, bad in (("server_pid", PID + 1), ("run_id", RUN + "x"), ("barrier_token", "b" * 32)):
            records = ready()
            records[AGENTS[2]][key] = bad
            with self.subTest(key=key), self.assertRaises(Invalid):
                validate_ready(records, RUN, PID, TOKEN)

    def test_release_binds_four_ready_hashes(self):
        hashes = {a: hashlib.sha256(a.encode()).hexdigest() for a in AGENTS}
        release = barrier_release(ready(), RUN, PID, TOKEN, hashes)
        self.assertEqual(release["ready_sha256"], hashes)
        self.assertEqual(release["agents"], list(AGENTS))
        del hashes[AGENTS[1]]
        with self.assertRaises(Invalid):
            barrier_release(ready(), RUN, PID, TOKEN, hashes)

    def test_actual_release_boundary_rejects_stale_or_incomplete(self):
        hashes = {a: hashlib.sha256(a.encode()).hexdigest() for a in AGENTS}
        value = barrier_release(ready(), RUN, PID, TOKEN, hashes)
        validate_release(value, AGENTS[0], RUN, PID, TOKEN, hashes[AGENTS[0]])
        for changed in ({**value, "server_pid": PID + 1}, {**value, "run_id": RUN + "x"},
                        {**value, "ready_sha256": {AGENTS[0]: hashes[AGENTS[0]]}}):
            with self.assertRaises(Invalid):
                validate_release(changed, AGENTS[0], RUN, PID, TOKEN, hashes[AGENTS[0]])
        with self.assertRaises(Invalid):
            validate_release(value, AGENTS[0], RUN, PID, TOKEN, "b" * 64)

    def test_all_four_protocol_success_still_pending_independent_review(self):
        self.assertEqual(aggregate(results(), dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True), PID, RUN),
                         "FourAgentProtocolCompleteAwaitingIndependentReview")

    def test_no_partial_success_missing_exit_or_cleanup(self):
        for category in ("result", "exit", "cleanup"):
            data, exits, cleanup = results(), dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True)
            del {"result": data, "exit": exits, "cleanup": cleanup}[category][AGENTS[0]]
            with self.subTest(category=category), self.assertRaises(Invalid):
                aggregate(data, exits, cleanup, PID, RUN)

    def test_one_failure_or_pending_cannot_pass(self):
        for key, bad in (("status", "Pending"), ("cleanup", False), ("server_pid", PID + 1),
                         ("diagnostic_slots", 1), ("full_prefix_reuse", "Passed"),
                         ("model_qualified", True), ("barrier", {"status": "Pending"}),
                         ("observed_final_source_sha256", "changed")):
            data = results()
            data[AGENTS[3]][key] = bad
            with self.subTest(key=key), self.assertRaises(Invalid):
                aggregate(data, dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True), PID, RUN)

    def test_nonzero_exit_and_unverified_cleanup_rejected(self):
        for exits, cleanup in (({**dict.fromkeys(AGENTS, 0), AGENTS[0]: 1}, dict.fromkeys(AGENTS, True)),
                               (dict.fromkeys(AGENTS, 0), {**dict.fromkeys(AGENTS, True), AGENTS[1]: False})):
            with self.assertRaises(Invalid):
                aggregate(results(), exits, cleanup, PID, RUN)

    def test_agent_swap_or_transcript_reuse_rejected(self):
        for key in ("agent_id", "initial_identity_sha256", "initial_messages_sha256"):
            data = results()
            data[AGENTS[1]][key] = data[AGENTS[0]][key]
            with self.subTest(key=key), self.assertRaises(Invalid):
                aggregate(data, dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True), PID, RUN)

    def test_all28_requests_and24_tools_required(self):
        for key in ("turns", "tools"):
            data = results()
            data[AGENTS[2]][key] = data[AGENTS[2]][key][:-1]
            with self.subTest(key=key), self.assertRaises(Invalid):
                aggregate(data, dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True), PID, RUN)


if __name__ == "__main__":
    unittest.main()
