"""Pure controls for actual-used N4 identity/approval/barrier/aggregation boundaries."""
import argparse
import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from qwen38_n1_verdict import Invalid, EXPECTED
import qwen38_n1_process as child
from qwen_n4_live_smoke import (AGENTS, PROFILE, active_agents, agent_command, agent_phase, aggregate,
                               barrier_release, first_decode_marker, identity, phase_cohorts, phase_trigger,
                               preflight, release_name, validate_agent_binding, validate_ready, validate_release,
                               validate_preflight, wait_barrier)

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


def client_args(phase_plan=None):
    return argparse.Namespace(base_url="http://127.0.0.1:18238/v1", model="Qwen3.8 27B", cargo="/x/cargo",
                              toolchain="stable", sandbox_exec="/usr/bin/sandbox-exec", sandbox_proof="/x/proof.json",
                              sandbox_proof_sha256="b" * 64, server_pid=PID, family="qwen", engine_kind="hf2q",
                              benchmark_profile=None, instance=None, phase_plan=phase_plan)


class PhasePlans(unittest.TestCase):
    """ADR-063 Q3 --phase-plan controls; absent, the canonical workload is unchanged."""

    def test_canonical_child_argv_is_byte_identical_without_a_phase_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval = Path(tmp) / "agent-1.approval.json"
            approval.write_text("{}")
            module_dir, barrier, path = Path("/m"), Path("/r/barrier"), Path("/r/agent-1")
            legacy = [sys.executable, "-E", "-s", "-B", "/m/qwen38_n1_smoke.py",
                      "--base-url", "http://127.0.0.1:18238/v1", "--model", "Qwen3.8 27B", "--cargo", "/x/cargo",
                      "--toolchain", "stable", "--sandbox-exec", "/usr/bin/sandbox-exec",
                      "--sandbox-proof", "/x/proof.json", "--sandbox-proof-sha256", "b" * 64,
                      "--server-pid", str(PID), "--output", "/r/agent-1", "--approval", str(approval),
                      "--approval-sha256", child.digest(approval), "--diagnostic-slots", "4", "--family", "qwen",
                      "--engine-kind", "hf2q", "--agent-id", "agent-1", "--run-id", RUN,
                      "--barrier-dir", "/r/barrier", "--barrier-token", TOKEN]
            self.assertEqual(agent_command(client_args(), module_dir, "agent-1", path, approval, barrier, RUN, TOKEN),
                             legacy)
            staged = agent_command(client_args("3+1"), module_dir, "agent-3", path, approval, barrier, RUN, TOKEN)
            self.assertEqual(staged[-4:], ["--phase-plan", "3+1", "--phase-cohort", "2"])

    def test_canonical_release_and_results_carry_no_phase_fields(self):
        hashes = {a: hashlib.sha256(a.encode()).hexdigest() for a in AGENTS}
        release = barrier_release(ready(), RUN, PID, TOKEN, hashes)
        self.assertEqual(list(release), ["schema", "run_id", "server_pid", "barrier_token", "agents", "ready_sha256"])
        data = results()
        data[AGENTS[1]]["phase"] = {"plan": "3+1", "cohort": 1}
        with self.assertRaises(Invalid):
            aggregate(data, dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True), PID, RUN)

    def test_cohorts_and_release_files(self):
        self.assertEqual(phase_cohorts(None), (AGENTS,))
        self.assertEqual(phase_cohorts("1"), (("agent-0",),))
        self.assertEqual(phase_cohorts("3+1"), (AGENTS[:3], AGENTS[3:]))
        self.assertEqual(phase_cohorts("1+3"), (AGENTS[:1], AGENTS[1:]))
        self.assertEqual(agent_phase("2+2", "agent-2"), {"plan": "2+2", "cohort": 2})
        self.assertIsNone(agent_phase(None, "agent-2"))
        self.assertEqual((release_name(1), release_name(2)), ("release.json", "release-2.json"))
        for bad in ("4", "2+1", "0+4"):
            with self.subTest(plan=bad), self.assertRaises(Invalid):
                phase_cohorts(bad)
        with self.assertRaises(Invalid):
            agent_phase("1", "agent-1")  # the one-agent control runs agent-0 only

    def test_one_agent_control_binds_only_agent_zero(self):
        agents = active_agents("1")
        records = {a: r for a, r in ready().items() if a in agents}
        hashes = {a: hashlib.sha256(a.encode()).hexdigest() for a in agents}
        release = barrier_release(records, RUN, PID, TOKEN, hashes, agents=agents)
        self.assertEqual(release["agents"], ["agent-0"])
        validate_release(release, "agent-0", RUN, PID, TOKEN, hashes["agent-0"], agents)
        with self.assertRaises(Invalid):  # a four-agent child cannot accept a one-agent release
            validate_release(release, "agent-0", RUN, PID, TOKEN, hashes["agent-0"])
        with self.assertRaises(Invalid):
            barrier_release(ready(), RUN, PID, TOKEN, hashes, agents=agents)
        data = {a: {**r, "phase": {"plan": "1", "cohort": 1}} for a, r in results().items() if a in agents}
        ok = dict.fromkeys(agents, 0), dict.fromkeys(agents, True)
        self.assertEqual(aggregate(data, *ok, PID, RUN, phase_plan="1"),
                         "FourAgentProtocolCompleteAwaitingIndependentReview")
        with self.assertRaises(Invalid):  # missing phase record
            aggregate({a: results()[a] for a in agents}, *ok, PID, RUN, phase_plan="1")

    def test_staggered_results_need_each_agents_cohort(self):
        data = {a: {**r, "phase": agent_phase("2+2", a)} for a, r in results().items()}
        ok = dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True)
        self.assertEqual(aggregate(data, *ok, PID, RUN, phase_plan="2+2"),
                         "FourAgentProtocolCompleteAwaitingIndependentReview")
        data["agent-3"]["phase"] = {"plan": "2+2", "cohort": 1}
        with self.assertRaises(Invalid):
            aggregate(data, *ok, PID, RUN, phase_plan="2+2")

    def test_trigger_waits_for_every_first_cohort_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = phase_cohorts("3+1")[0]
            for agent, ns in zip(first[:2], (5, 7)):
                first_decode_marker(tmp, agent, RUN, TOKEN, ns)
            self.assertIsNone(phase_trigger(tmp, first, RUN, TOKEN))
            first_decode_marker(tmp, first[2], RUN, TOKEN, 9)
            self.assertEqual(phase_trigger(tmp, first, RUN, TOKEN), {"agent-0": 5, "agent-1": 7, "agent-2": 9})
            with self.assertRaises(Invalid):
                phase_trigger(tmp, first, RUN, "c" * 32)  # a marker from another barrier
            with self.assertRaises(FileExistsError):
                first_decode_marker(tmp, first[0], RUN, TOKEN, 11)  # published once

    def test_later_cohort_waits_for_its_own_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            barrier, agent = Path(tmp), "agent-3"
            phase = agent_phase("3+1", agent)

            def coordinator():
                own = barrier / f"{agent}.ready.json"
                while not own.exists():
                    time.sleep(0.01)
                hashes = {a: hashlib.sha256(a.encode()).hexdigest() for a in AGENTS}
                hashes[agent] = child.digest(own)
                base = {"schema": "hf2q.n4-barrier-release.v1", "run_id": RUN, "server_pid": PID,
                        "barrier_token": TOKEN, "agents": list(AGENTS), "ready_sha256": hashes, "phase_plan": "3+1"}
                (barrier / "release.json").write_text(json.dumps({**base, "cohort": 1, "release_monotonic_ns": 1}))
                time.sleep(0.1)
                (barrier / "release-2.json").write_text(json.dumps({**base, "cohort": 2, "release_monotonic_ns": 2}))

            thread = threading.Thread(target=coordinator)
            thread.start()
            released = wait_barrier(barrier, agent, RUN, PID, TOKEN, barrier / agent, "d" * 64, "e" * 64, phase=phase)
            thread.join()
            self.assertEqual(released["release_monotonic_ns"], 2)
            self.assertEqual(released["release_sha256"], child.digest(barrier / "release-2.json"))

    def test_first_token_callback_fires_once_at_first_decoded_token(self):
        from qwen38_n1_smoke import exchange
        from test_q2_socket_controls import FakeSocket, frame
        from test_qwen38_n1_verdict import request, chunk, stats
        value = request()
        value["stream"] = True
        parts = [(0, frame(chunk({"role": "assistant"}))),
                 (.05, frame(chunk({"reasoning_content": "think"}))),
                 (.10, frame(chunk({"tool_calls": [{"index": 0, "id": "live", "type": "function",
                                                    "function": {"name": "read_file"}}]}))),
                 (.01, frame(chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path":"src/ports.rs"}'}}]}))),
                 (.01, frame(chunk({}, "tool_calls", stats()))), (0, frame("[DONE]"))]
        calls = []
        with tempfile.TemporaryDirectory() as tmp, FakeSocket(parts) as server:
            out = Path(tmp).resolve() / "turn"
            out.mkdir()
            _, _, receipt = exchange(server.url, value, out, 0, set(), deadline=2, on_first_token=calls.append)
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(calls[0], receipt["send_monotonic_ns"])
        self.assertLess(calls[0], receipt["semantic_event_monotonic_ns"][0])  # reasoning precedes the tool name


if __name__ == "__main__":
    unittest.main()
