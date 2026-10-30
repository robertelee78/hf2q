"""Pure controls for quality-campaign task instances and the unchanged canonical workload."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import coding_instances
from coding_instances import MAXIMUM, select
from qwen38_n1_tools import IMMUTABLE, INITIAL, Tools
from qwen38_n1_verdict import WORKFLOW, Invalid, loads
from qwen38_n1_smoke import initial_messages
from qwen_n4_live_smoke import AGENTS, aggregate, identity, task_instance, validate_agent_binding

RUN = "quality_instance_fixture_01"
FIXTURE = Path(__file__).with_name("fixtures") / "canonical-coding-requests.json"
# initial_messages_sha256 of the canonical N4 agents, as recorded by retained
# receipts (e.g. $E/qwen38-template-order-normal-n4-20261028T034811Z/client/agent-*/summary.json).
CANONICAL_N4 = {"agent-0": "93ed522748fd05a31657e270bb26c4ad2136dc8dffc13d4d342745f8d559e163",
                "agent-1": "266c654215137d58d9ff5707e0c32b573dfd154133d32bdd2f3cb2748341324f",
                "agent-2": "eb843928b2eb27e61b28735187856bb980064173c2b5c7ee555d892450432833",
                "agent-3": "7ac6cf84cb79e4aef43a6bb1beaf612deecb7dec20bb035154cc7bee8b2931d7"}
CANONICAL_N1 = "3587958dde84e90fd106d9268c4aaf0484427fc86e46a1b91bda6ca794b0fea4"
CANONICAL_FILES = "3e33aff27ab8e9ea7603f6719b1e4491edf4b7f506cc2452984e3ce854ea7152"
CANONICAL_WORKFLOW = "65d08e5cb9077d3ba28d4ac435da46bdb65242ecccd86f8c662d969650329965"
# Every rendered instance (prompt, workflow and fixture files). A change here
# changes the frozen campaign workload: bump the table schema, never rescore.
RENDERED = "d1f0df2aedd49e242e471910e9ee8d49732339156ce3afd3c2a0d20fb54c5304"


def digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def fixture_messages():
    return loads(FIXTURE.read_bytes())[0]["request"]["messages"]


def rendered(workload):
    return {"user": workload.user, "workflow": workload.workflow, "initial": workload.initial,
            "files": workload.immutable, "binding": workload.binding()}


def rust_parse(text, kind):
    """Rust `str::parse::<uN>()`: optional '+', then ASCII digits, within the type."""
    if re.fullmatch(r"\+?[0-9]+", text) is None:
        return None
    value = int(text)
    return value if value <= MAXIMUM[kind] else None


def table(rows):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"schema": coding_instances.SCHEMA, "difficulty_class": "x", "instances": rows}, handle)
    handle.close()
    return handle.name


class CanonicalUnchanged(unittest.TestCase):
    def test_canonical_prompts_match_retained_receipts(self):
        for agent in AGENTS:
            self.assertEqual(digest(initial_messages(fixture_messages(), identity(agent, RUN))), CANONICAL_N4[agent])
        self.assertEqual(digest(initial_messages(fixture_messages())), CANONICAL_N1)

    def test_canonical_fixture_files_and_workflow_unchanged(self):
        self.assertEqual(hashlib.sha256(json.dumps({**IMMUTABLE, "src/ports.rs": INITIAL}, sort_keys=True)
                                        .encode()).hexdigest(), CANONICAL_FILES)
        self.assertEqual(hashlib.sha256(WORKFLOW.encode()).hexdigest(), CANONICAL_WORKFLOW)

    def test_canonical_approval_cannot_carry_an_instance(self):
        self.assertIsNone(task_instance({}, None))
        for packet in ({"instance": 0}, {"instance_table_sha256": "0" * 64}):
            with self.subTest(packet=packet), self.assertRaises(Invalid):
                task_instance(packet, None)


class Instances(unittest.TestCase):
    def test_rendering_is_deterministic_and_pinned(self):
        rows, sha = coding_instances.load()
        self.assertEqual(sha, hashlib.sha256(coding_instances.TABLE.read_bytes()).hexdigest())
        first = [rendered(select(i)) for i in range(len(rows))]
        self.assertEqual(first, [rendered(select(i)) for i in range(len(rows))])
        self.assertEqual(hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest(), RENDERED)

    def test_twenty_four_distinct_problems_and_prompts(self):
        rows, _ = coding_instances.load()
        self.assertGreaterEqual(len(rows), 24)
        workloads = [select(i) for i in range(len(rows))]
        for field in ("function", "user", "workflow", "initial"):
            self.assertEqual(len({getattr(w, field) for w in workloads}), len(rows), field)
        for name in ("src/lib.rs", "tests/ports.rs"):
            self.assertEqual(len({w.immutable[name] for w in workloads} | {IMMUTABLE[name]}), len(rows) + 1, name)
        self.assertEqual(len({(r["type"], r["min"], r["max"]) for r in rows}), len(rows))
        prompts = {digest(initial_messages(fixture_messages(), identity(a, RUN, w.instance), w))
                   for w in workloads for a in AGENTS}
        self.assertEqual(len(prompts), 4 * len(rows))
        self.assertFalse(prompts & set(CANONICAL_N4.values()))

    def test_every_instance_is_the_same_difficulty_class(self):
        for workload in (select(i) for i in range(len(coding_instances.load()[0]))):
            row = coding_instances.load()[0][workload.instance]
            kind, low, high, name = row["type"], row["min"], row["max"], workload.function
            with self.subTest(instance=workload.instance):
                self.assertEqual(workload.initial, f"pub fn {name}(s: &str) -> {kind} {{ s.parse().unwrap() }}\n")
                self.assertTrue(0 < low < high < MAXIMUM[kind])
                tests, library = workload.immutable["tests/ports.rs"], workload.immutable["src/lib.rs"]
                self.assertEqual(set(workload.immutable), set(IMMUTABLE))
                for name_ in ("Cargo.toml", "Cargo.lock"):
                    self.assertEqual(workload.immutable[name_], IMMUTABLE[name_])
                self.assertEqual(len(re.findall(r"^#\[test\]$", tests, re.M)), 2)
                self.assertEqual(len(re.findall(r"^    #\[test\]$", library, re.M)), 2)
                rejects = re.findall(r'"([^"]*)"', re.search(r"for s in \[(.*)\]", tests).group(1))
                rejects += re.findall(rf'{name}\("([^"]*)"\)\.is_err\(\)', library)
                accepts = re.findall(rf'{name}\("([^"]*)"\)\.unwrap\(\), (\d+)\)', tests + library)
                # Both bounds are tested from both sides; nothing passes by parse alone.
                self.assertTrue({str(low - 1), str(high + 1)} <= set(rejects))
                self.assertEqual({a for a, _ in accepts}, {str(low), str(high)})
                for text in rejects:
                    value = rust_parse(text, kind)
                    self.assertTrue(value is None or not low <= value <= high, text)
                self.assertIn(str(MAXIMUM[kind] + 1), rejects)
                for text, expected in accepts:
                    self.assertEqual(rust_parse(text, kind), int(expected))
                self.assertIn(f"below {low} and above {high}", workload.user)
                self.assertTrue(workload.workflow.endswith(
                    f"accept valid {kind} {row['nouns']}, and return a Result<{kind}, String> "
                    "for invalid input rather than panic."))
                self.assertTrue(workload.workflow.startswith(WORKFLOW[:WORKFLOW.index("Fix parse_port")]))

    def test_table_refuses_ambiguous_or_easier_rows(self):
        good = {"id": 0, "function": "parse_a", "type": "u8", "min": 1, "max": 10,
                "nouns": "as", "slug": "as", "word": "x"}
        bad = [[{**good, "min": 0}], [{**good, "max": 255}], [{**good, "id": 1}],
               [{**good, "function": "parse_port"}], [{**good, "type": "i32"}], [{**good, "extra": 1}],
               [good, {**good, "id": 1, "slug": "bs"}], [good, {**good, "id": 1, "function": "parse_b"}]]
        for rows in bad:
            path = table(rows)
            with self.subTest(rows=rows), self.assertRaises(Invalid):
                coding_instances.load(path)
            Path(path).unlink()
        for instance in (-1, 24, "0", True):
            with self.subTest(instance=instance), self.assertRaises(Invalid):
                select(instance)


class InstanceBinding(unittest.TestCase):
    def packet(self, workload, agent="agent-0"):
        return {"schema": "hf2q.qwen-n4-agent-approval.v1", "profile": "adr063-qwen-n4-live-v1",
                "diagnostic_slots": 4, "agent_id": agent, "run_id": RUN, "server_pid": 1,
                "barrier_dir": "/private/b", "barrier_token": "a" * 32, "agent_output": "/private/o",
                "initial_identity_sha256": hashlib.sha256(
                    identity(agent, RUN, None if workload is None else workload.instance).encode()).hexdigest(),
                **({} if workload is None else workload.binding())}

    def test_approval_binds_instance_and_table(self):
        workload = select(5)
        self.assertEqual(rendered(task_instance(self.packet(workload), 5)), rendered(workload))
        validate_agent_binding(self.packet(workload), "agent-0", RUN, 1, "/private/b", "a" * 32, "/private/o",
                               instance=5)
        for change in ({"instance": 6}, {"instance": 5.0}, {"instance_table_sha256": "0" * 64}):
            with self.subTest(change=change), self.assertRaises(Invalid):
                task_instance({**self.packet(workload), **change}, 5)
        with self.assertRaises(Invalid):  # canonical identity cannot pass for the instance
            validate_agent_binding(self.packet(None), "agent-0", RUN, 1, "/private/b", "a" * 32, "/private/o",
                                   instance=5)
        self.assertNotEqual(identity("agent-0", RUN, 5), identity("agent-0", RUN))
        self.assertNotEqual(identity("agent-0", RUN, 5), identity("agent-0", RUN, 6))

    def test_aggregate_refuses_agents_from_another_instance(self):
        from test_qwen_n4_live_smoke import PID, RUN as N4_RUN, results
        record = select(3).record()
        values = results()
        for agent in AGENTS:
            values[agent].update(instance=record, initial_identity_sha256=hashlib.sha256(
                identity(agent, N4_RUN, 3).encode()).hexdigest())
        exits, cleanup = dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True)
        self.assertEqual(aggregate(values, exits, cleanup, PID, N4_RUN, instance_record=record),
                         "FourAgentProtocolCompleteAwaitingIndependentReview")
        values["agent-2"]["instance"] = select(4).record()
        with self.assertRaises(Invalid):
            aggregate(values, exits, cleanup, PID, N4_RUN, instance_record=record)
        with self.assertRaises(Invalid):
            aggregate(results(), exits, cleanup, PID, N4_RUN, instance_record=record)

    def test_owned_fixture_holds_the_instance_files(self):
        workload = select(11)
        with tempfile.TemporaryDirectory() as directory:
            tools = Tools(Path(directory) / "owned", "/not/executed/bin/cargo", "/not/executed/sandbox",
                          "/Library/Developer/SDKs/Mock.sdk", immutable=workload.immutable, initial=workload.initial)
            try:
                self.assertEqual(tools.read("src/ports.rs"), workload.initial)
                self.assertEqual(tools.read("tests/ports.rs"), workload.immutable["tests/ports.rs"])
                tools.invariant()
                path = tools.fixture / "tests/ports.rs"
                path.chmod(0o600)
                path.write_text(IMMUTABLE["tests/ports.rs"])  # canonical tests are not this instance's
                with self.assertRaises(Invalid):
                    tools.invariant()
            finally:
                tools.close()
                tools.cwd.rmdir()


if __name__ == "__main__":
    unittest.main()
