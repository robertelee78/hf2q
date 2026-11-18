#!/usr/bin/env python3
"""Model-free schema3 evidence fixtures and orchestration regression tests.

Synthetic receipts below are test data only: the production CLI has no import,
manual-success or alternate-workload option. make_evidence is also used by the
portable archive contract test. It writes manifest.json and its checksum.
"""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import subprocess
import json
import fcntl
import signal
import socket
import sys
import tarfile
import tempfile
import unittest
import time
from unittest import mock

import cache_release_orchestrator as gate


def put(path: Path, content: bytes | str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    if isinstance(content, dict):
        content = gate.canonical(content)
    if isinstance(content, str):
        content = content.encode()
    path.write_bytes(content)


def fixture_record(root: Path, work: str, plan_sha: str | None, previous: str | None,
                   when: int, artifacts: dict, initial: dict | None = None,
                   build_inputs: dict | None = None) -> dict:
    """Write one synthetic immutable attempt; intentionally outside production."""
    start_path = gate.attempt_path(root, work, "start")
    ticket_path = gate.attempt_path(root, work, "ticket")
    end_path = gate.attempt_path(root, work, "end")
    deadline = gate.WALL_SECONDS
    start = {"schema_version": 3, "work": work, "out_root": str(root), "began_ns": when,
             "driver": {"pid": 900001, "birth": "synthetic test driver"},
             "capability_sha256": gate.digest_bytes(b"test-data-not-a-capability"),
             "plan_sha256": plan_sha, "previous_end_sha256": previous,
             "build_inputs": build_inputs,
             "initial_artifacts": initial or {}, "argv": ["bash", "/offline-fixture-source/scripts/run_agentic_cache_release_gate.sh", "_work", work]}
    gate.exclusive(start_path, start)
    ticket = {"schema_version": 3, "work": work, "start_sha256": gate.sha256(start_path),
              "ticket_process": {"pid": 900002, "birth": "synthetic test ticket"},
              "began_ns": when + 1, "deadline_ns": when + 1 + deadline * 10**9,
              "term_after_seconds": gate.TERM_SECONDS,
              "wall_limit_seconds": deadline, "hostq_class": "build" if work in gate.BUILDS else "measure"}
    gate.exclusive(ticket_path, ticket)
    log = root / gate.STATE / "logs" / f"{work}.child.log"
    put(log, f"canonical child fixture {work}: exit 0\n")
    hostq_log = root / gate.STATE / "logs" / f"{work}.hostq.log"
    put(hostq_log, f"hostq {work} child exited and owned cleanup completed\n")
    telemetry = None
    if work in gate.PHASES:
        rows = [{"at_ns": when + n, "thermal": "nominal", "source": "ac", "mode_code": "2",
                 "pmset_battery": "Now drawing from 'AC Power'", "pmset_live": " powermode 2\n"}
                for n in (2, 8)]
        path = root / gate.STATE / "logs" / f"{work}.telemetry.jsonl"
        put(path, b"".join(gate.canonical(row) for row in rows))
        telemetry = {"path": path.relative_to(root).as_posix(), **gate.descriptor(path)}
    end = {"schema_version": 3, "work": work, "status": "pass", "exit_code": 0, "reason": "",
           "plan_sha256": plan_sha, "start_sha256": gate.sha256(start_path),
           "ticket_sha256": gate.sha256(ticket_path), "began_ns": when + 1,
           "ended_ns": when + 9, "elapsed_seconds": 8 / 10**9,
           "deadline_ns": ticket["deadline_ns"], "artifacts": artifacts,
           "child_log": gate.descriptor(log), "child_started_ns": when + 3,
           "child_ended_ns": when + 7, "telemetry": telemetry, "conflicts": [],
           "cleanup": {"clean": True, "term_sent": False, "kill_sent": False,
                       "owned_processes": [{"pid": 900003, "birth": "synthetic test child"}],
                       "observed_listeners": [], "remaining_processes": [], "remaining_listeners": [],
                       "direct_child_reaped": True}}
    gate.exclusive(end_path, end)
    gate.exclusive(gate.attempt_path(root, work, "dispatch"), {
        "schema_version": 3, "work": work, "exit_code": 0, "interrupted": False,
        "completed_ns": when + 10, "hostq_log": gate.descriptor(hostq_log),
        "end_sha256": gate.sha256(end_path)})
    return gate.verify_record(root, work, plan_sha, {})


def make_evidence(root: Path) -> dict:
    """Create a complete portable synthetic evidence tree, returning its manifest."""
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path("/offline-fixture-source")
    sources = {"Cargo.toml": b'[package]\nname="hf2q"\nversion="0.1.30"\n',
               "Cargo.lock": b"version = 4\n", "src/main.rs": b"fn main() {}\n",
               "scripts/run_agentic_cache_release_gate.sh": b"#!/bin/bash\nexit 0\n",
               "scripts/macos_thermal_probe.swift": b'import Foundation\nprint("nominal")\n',
               "tests/lcp_partial_prefill_byte_identity.rs": b"#[test] fn identity() {}\n"}
    frozen = {}
    source_inventory = {}
    for name, data in sources.items():
        path = root / gate.STATE / "inputs" / "source" / name
        put(path, data)
        source_inventory[name] = gate.descriptor(path)
        frozen[path.relative_to(root).as_posix()] = source_inventory[name]
    binary = root / "standalone" / "hf2q"
    put(binary, b"synthetic signed binary bytes\n")
    binary_sha = gate.sha256(binary)
    crate_path = root / gate.STATE / "inputs" / "candidate" / "candidate.crate"
    crate_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(crate_path, "w:gz") as archive:
        info_bytes = gate.canonical({"git": {"sha1": "a" * 40, "dirty": False}})
        entry = tarfile.TarInfo("hf2q-0.1.30/.cargo_vcs_info.json")
        entry.size = len(info_bytes)
        archive.addfile(entry, io.BytesIO(info_bytes))
        for name, content in sources.items():
            entry = tarfile.TarInfo(f"hf2q-0.1.30/{name}")
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    identity = {"EXPECTED_SHA": "a" * 40, "EXPECTED_VERSION": "0.1.30",
                "STANDALONE_CANDIDATE_RUN_ID": "1234567", "CRATE_SHA256": gate.sha256(crate_path),
                "EXPECTED_BINARY_SHA256": binary_sha}
    proof = {"kind": "hf2q.standalone-apple-release-proof", "schema_version": 1,
             "package": "hf2q", "source_sha": identity["EXPECTED_SHA"], "version": "0.1.30",
             "target": "aarch64-apple-darwin", "asset": {"name": "hf2q-aarch64-apple-darwin",
             "sha256": binary_sha, "size": binary.stat().st_size},
             "notarization": {"status": "Accepted"}, "verification": {"codesign": "accepted",
             "online_notarization": "accepted", "notary_ticket_cdhash_matches": True}}
    proof["signing"] = {"identifier": "us.hf2q.cli", "team_id": "TEST000001",
                        "authority": "Developer ID Application", "hardened_runtime": True,
                        "secure_timestamp": True, "cdhash": "d" * 40}
    support_paths = []
    for filename, section, key, content in (
        ("notary-submission.json", "notarization", "submission_sha256", {"id": "synthetic-test-submission"}),
        ("notary-wait.json", "notarization", "wait_sha256", {"status": "Accepted"}),
        ("notary-log.json", "notarization", "log_sha256", {"status": "Accepted", "issues": [],
          "ticketContents": [{"digestAlgorithm": "SHA-256", "cdhash": "d" * 40}]}),
        ("codesign.txt", "verification", "codesign_log_sha256", "synthetic accepted signature log\n"),
        ("notarization-check.txt", "verification", "online_notarization_log_sha256", "synthetic accepted notary log\n"),
    ):
        path = crate_path.parent / filename
        put(path, content)
        proof[section][key] = gate.sha256(path)
        support_paths.append(path)
    proof_path = crate_path.parent / "signed-proof.json"
    put(proof_path, proof)
    candidate_files = {}
    for path in (crate_path, proof_path, *support_paths):
        desc = gate.descriptor(path)
        candidate_files[path.name] = {"path": str(path), **desc}
        frozen[path.relative_to(root).as_posix()] = desc
    initial = {"standalone/hf2q": gate.descriptor(binary)}
    put(root / "build" / "power-policy.json", {"initial_source": "ac", "mode_name": "high", "mode_code": "2"})
    put(root / "build" / "candidate-signature-check.txt", "synthetic signature evidence, test data only\n")
    probe = root / "build" / "macos-thermal-probe"
    put(probe, "synthetic compiled thermal probe\n")
    probe_identity = {"path": str(probe), "sha256": gate.sha256(probe),
                      "source": str(source_root / "scripts/macos_thermal_probe.swift"),
                      "source_sha256": source_inventory["scripts/macos_thermal_probe.swift"]["sha256"],
                      "compiler": "/usr/bin/swiftc", "compiler_sha256": "b" * 64,
                      "compiler_version": "Apple Swift version (synthetic fixture)"}
    previous = None
    builds = []
    clock = 1_000_000_000_000
    # The first record covers probe artifacts but not the pre-existing signed binary.
    artifacts = gate.snapshot_artifacts(root, gate.BUILDS[0], initial)
    builds.append(fixture_record(root, gate.BUILDS[0], None, previous, clock, artifacts, initial,
        {"source_files": source_inventory, "source_sha": identity["EXPECTED_SHA"],
         "identity": identity, "configuration": {}}))
    previous = builds[-1]["end_sha256"]
    test_rows = []
    for work, role, filename in zip(gate.BUILDS[1:4], ("debug-bin", "release-bin", "release-integration"),
                                    ("hf2q-debug-tests", "hf2q-release-tests", "lcp-partial-prefill-byte-identity-tests")):
        before = gate.inventory(root)
        path = root / "build" / filename
        put(path, f"synthetic prebuilt executable {role}\n")
        build_log = root / "build" / f"{role}.jsonl"
        put(build_log, {"reason": "compiler-artifact", "executable": str(path)})
        test_rows.append({"role": role, "path": str(path), "sha256": gate.sha256(path),
                          "build_log": str(build_log), "build_log_sha256": gate.sha256(build_log)})
        clock += 100
        builds.append(fixture_record(root, work, None, previous, clock, gate.snapshot_artifacts(root, work, before)))
        previous = builds[-1]["end_sha256"]
    before = gate.inventory(root)
    fixture = root / "fixtures" / "public-347.json"
    put(fixture, {"messages": [{"role": "user", "content": "synthetic model-free fixture"}]})
    prebuilt = {"schema_version": 1, "test_binaries": test_rows, "thermal_probe": probe_identity,
                "fixtures": [{"path": str(fixture), "sha256": gate.sha256(fixture)}],
                "release_binary": {"path": str(binary), "sha256": binary_sha}}
    put(root / "build" / "inventory.json", prebuilt)
    models = {}
    for family, (_, architecture) in gate.FAMILIES.items():
        model_sha = gate.digest_bytes(f"{family} synthetic weights".encode())
        model_path = f"/not-present-on-consumer/{family}.gguf"
        receipt = {"schema_version": 1, "path": model_path, "sha256": model_sha,
                   "file_snapshot": "1:2:2048:10:10", "content_hash_verified": True}
        put(root / family / "model-verification.json", receipt)
        put(root / family / "model-info.txt", f"Architecture: {architecture}\n")
        put(root / family / "model.sha256", f"{model_sha}  {model_path}\n")
        models[family] = {"path": model_path, "sha256": model_sha, "bytes": 2048,
                          "architecture": architecture, "file_snapshot": {"device": 1, "inode": 2, "bytes": 2048,
                          "mtime_ns": 10**10, "ctime_ns": 10**10},
                          "verification": gate.descriptor(root / family / "model-verification.json"),
                          "info": gate.descriptor(root / family / "model-info.txt"),
                          "hash_record": gate.descriptor(root / family / "model.sha256")}
    dependencies = {}
    for name, content in (("Cargo.lock", "version = 4\n"), ("cargo-metadata.json", {"packages": []}),
                          ("receipt.json", {"schema_version": 1, "status": "pass"})):
        path = root / "dependency-provenance" / name
        put(path, content)
        destination = root / gate.STATE / "inputs" / "dependency" / name
        dependencies[name] = gate.copy_exclusive(path, destination)
        frozen[destination.relative_to(root).as_posix()] = dependencies[name]
    clock += 100
    builds.append(fixture_record(root, gate.BUILDS[-1], None, previous, clock,
                                 gate.snapshot_artifacts(root, gate.BUILDS[-1], before)))
    previous = builds[-1]["end_sha256"]
    plan = {"schema_version": 3, "kind": "hf2q.cache-release-plan", "created_ns": clock + 20,
            "out_root": str(root), "source_root": str(source_root), "source_sha": identity["EXPECTED_SHA"],
            "source_files": source_inventory, "candidate": {"identity": identity, "proof": proof, "files": candidate_files},
            "models": models, "dependency": dependencies, "build_inventory": prebuilt,
            "configuration": {}, "source_argv": ["bash", str(source_root / "scripts/run_agentic_cache_release_gate.sh"), "_work"],
            "hostq": {"path": "/not-present-on-consumer/hostq", "sha256": "c" * 64, "bytes": 1},
            "fit": {phase: {"status": "unknown", "note": "Synthetic test has no fit claim"} for phase in gate.PHASES},
            "required_phases": list(gate.PHASES), "dag": gate.phase_dag(), "coverage": gate.COVERAGE,
            "wall_limit_seconds": gate.WALL_SECONDS, "term_after_seconds": gate.TERM_SECONDS,
            "build_end_sha256": previous, "build_attempts": builds, "frozen_files": frozen}
    plan_path = root / gate.STATE / "plan.json"
    gate.exclusive(plan_path, plan)
    plan_sha = gate.sha256(plan_path)
    for phase in gate.PHASES:
        before = gate.inventory(root)
        for prefix in gate.COVERAGE[phase]:
            path = root / prefix
            if path.suffix:
                put(path, f"child-produced synthetic {phase} output\n")
            else:
                put(path / "raw-receipt.json", {"status": "pass", "phase": phase})
        for name in ("power.log", "thermal.log", "power-assertions.before.txt", "power-assertions.after.txt"):
            put(root / "phases" / phase / name, f"{phase} canonical {name}\n")
        clock += 100
        record = fixture_record(root, phase, plan_sha, previous, clock, gate.snapshot_artifacts(root, phase, before))
        previous = record["end_sha256"]
    binding = gate.verify_evidence(root)
    manifest = {"kind": "hf2q.cache-lifecycle-release-manifest", "schema_version": 3, "status": "pass",
                "source_sha": identity["EXPECTED_SHA"], "version": identity["EXPECTED_VERSION"],
                "standalone_candidate_run_id": identity["STANDALONE_CANDIDATE_RUN_ID"],
                "crate_sha256": identity["CRATE_SHA256"], "binary_sha256": binary_sha,
                "phase_ledger": binding, "models": {family: {
                    key: model[key] for key in ("path", "bytes", "sha256", "architecture")}
                    | {"model_info_sha256": model["info"]["sha256"]} for family, model in models.items()}}
    before = gate.inventory(root)
    put(root / "manifest.json", manifest)
    put(root / "manifest.json.sha256", f"{gate.sha256(root / 'manifest.json')}  manifest.json\n")
    clock += 100
    fixture_record(root, "finalize", plan_sha, previous, clock, gate.snapshot_artifacts(root, "finalize", before))
    gate.verify_evidence(root, manifest)
    return manifest


class PortableEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "evidence"
        self.manifest = make_evidence(self.root)

    def rejects(self):
        with self.assertRaises((gate.Refusal, OSError, KeyError, TypeError, ValueError)):
            gate.verify_evidence(self.root, self.manifest)

    def change_end(self, phase: str, change) -> None:
        path = gate.attempt_path(self.root, phase, "end")
        value = gate.load(path)
        change(value)
        put(path, value)
        # Rebind the dispatch record to exercise semantic validation, not only hashes.
        dispatch = gate.attempt_path(self.root, phase, "dispatch")
        receipt = gate.load(dispatch)
        receipt["end_sha256"] = gate.sha256(path)
        put(dispatch, receipt)

    def test_credentials_cannot_be_frozen_into_public_evidence(self):
        for name in ("CARGO_REGISTRY_TOKEN", "CARGO_REGISTRIES_PRIVATE_TOKEN", "HF2Q_API_KEY"):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: "private-value"}, clear=True):
                with self.assertRaises(gate.Refusal):
                    gate.configuration()

    def test_complete_downloaded_root_needs_no_live_models_or_source(self):
        destination = Path(self.temp.name) / "relocated"
        self.root.rename(destination)
        self.root = destination
        self.assertEqual(gate.verify_evidence(destination, self.manifest), self.manifest["phase_ledger"])

    def test_missing_end_is_interrupted_not_resumable(self):
        gate.attempt_path(self.root, gate.PHASES[2], "end").unlink()
        self.rejects()

    def test_failed_phase_is_not_a_pass(self):
        self.change_end(gate.PHASES[0], lambda row: row.update(status="failed", exit_code=1, reason="oracle failure"))
        self.rejects()

    def test_dirty_cleanup_is_not_a_pass(self):
        self.change_end(gate.PHASES[0], lambda row: row["cleanup"].update(clean=False, remaining_processes=[{"pid": 42}]))
        self.rejects()

    def test_owned_listener_blocks_pass(self):
        self.change_end(gate.PHASES[0], lambda row: row["cleanup"].update(remaining_listeners=["p42", "n127.0.0.1:18080"]))
        self.rejects()

    def test_kill_cleanup_is_not_success(self):
        self.change_end(gate.PHASES[0], lambda row: row["cleanup"].update(kill_sent=True))
        self.rejects()

    def test_over_deadline_is_not_success(self):
        self.change_end(gate.PHASES[0], lambda row: row.update(ended_ns=row["deadline_ns"] + 1, elapsed_seconds=541))
        self.rejects()

    def test_raw_snapshot_tamper_rejected(self):
        end = gate.load(gate.attempt_path(self.root, gate.PHASES[0], "end"))
        path = self.root / next(iter(end["artifacts"].values()))["snapshot"]
        put(path, "tampered\n")
        self.rejects()

    def test_current_raw_tamper_rejected(self):
        put(self.root / "gemma/parity/n8.log", "tampered\n")
        self.rejects()

    def test_extra_unrecorded_model_output_rejected(self):
        put(self.root / "deepseek/unrecorded/request.json", {"status": "pass"})
        self.rejects()

    def test_missing_phase_coverage_rejected_even_with_rebound_end(self):
        self.change_end(gate.PHASES[0], lambda row: row["artifacts"].clear())
        self.rejects()

    def test_duplicate_phase_record_rejected(self):
        put(self.root / gate.STATE / "attempts" / "deepseek-cooperative.retry.end.json", {"status": "pass"})
        self.rejects()

    def test_plan_tamper_rejected(self):
        path = self.root / gate.STATE / "plan.json"
        plan = gate.load(path)
        plan["configuration"]["HF2Q_USE_DENSE"] = "1"
        put(path, plan)
        self.rejects()

    def test_selected_subset_rejected(self):
        path = self.root / gate.STATE / "plan.json"
        plan = gate.load(path)
        plan["required_phases"] = plan["required_phases"][:-1]
        put(path, plan)
        self.rejects()

    def test_manifest_binary_identity_rejected(self):
        self.manifest["binary_sha256"] = "0" * 64
        self.rejects()

    def test_probe_tamper_rejected_without_executing_probe(self):
        put(self.root / "build/macos-thermal-probe", "arbitrary override\n")
        self.rejects()

    def test_prebuilt_inventory_tamper_rejected(self):
        put(self.root / "build/hf2q-release-tests", "different candidate\n")
        self.rejects()

    def test_telemetry_tamper_rejected(self):
        put(self.root / gate.STATE / "logs" / f"{gate.PHASES[0]}.telemetry.jsonl", "{}\n")
        self.rejects()

    def test_hot_thermal_sample_rejected_even_with_rebound_digest(self):
        phase = gate.PHASES[0]
        path = self.root / gate.STATE / "logs" / f"{phase}.telemetry.jsonl"
        rows = [gate.json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["thermal"] = "serious"
        put(path, b"".join(gate.canonical(row) for row in rows))
        self.change_end(phase, lambda row: row["telemetry"].update(gate.descriptor(path)))
        self.rejects()

    def test_low_power_sample_rejected_even_with_rebound_digest(self):
        phase = gate.PHASES[0]
        path = self.root / gate.STATE / "logs" / f"{phase}.telemetry.jsonl"
        rows = [gate.json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["mode_code"] = "1"
        put(path, b"".join(gate.canonical(row) for row in rows))
        self.change_end(phase, lambda row: row["telemetry"].update(gate.descriptor(path)))
        self.rejects()

    def test_stale_output_is_refused_before_child_launch(self):
        with self.assertRaisesRegex(gate.Refusal, "pre-existing phase output"):
            gate.require_fresh_outputs(self.root, gate.PHASES[0])

    def test_unknown_immutable_metadata_file_rejected(self):
        put(self.root / gate.STATE / "raw" / "manual-pass.json", {"status": "pass"})
        self.rejects()

    def test_live_model_stat_drift_is_detected_without_inference(self):
        environment = {}
        for family, (prefix, _) in gate.FAMILIES.items():
            path = Path(self.temp.name) / f"{family}.gguf"
            put(path, f"model-free {family} bytes")
            st = path.stat()
            digest = gate.sha256(path)
            environment[f"{prefix}_MODEL"] = str(path)
            environment[f"{prefix}_MODEL_SHA256"] = digest
            put(self.root / family / "model-verification.json", {
                "schema_version": 1, "path": str(path), "sha256": digest, "content_hash_verified": True,
                "file_snapshot": f"{st.st_dev}:{st.st_ino}:{st.st_size}:{int(st.st_mtime)}:{int(st.st_ctime)}"})
        with mock.patch.dict(os.environ, environment):
            gate.model_inputs(self.root)
            put(Path(environment["DEEPSEEK_MODEL"]), "replacement with a different size")
            with self.assertRaisesRegex(gate.Refusal, "model stat changed"):
                gate.model_inputs(self.root)

    def test_cancelled_driver_cannot_finalize(self):
        gate.exclusive(self.root / gate.STATE / "cancel.json", {"signal": 15})
        self.rejects()

    def test_append_only_creation_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):
            gate.exclusive(gate.attempt_path(self.root, gate.PHASES[0], "start"), {})

    def test_standalone_phase_replay_rejected_before_dispatch(self):
        with self.assertRaisesRegex(gate.Refusal, "retry/replay"):
            gate.dispatch(self.root, gate.PHASES[0], gate.canonical_script(None))

    def test_manually_exported_internal_bit_does_not_authorize(self):
        with mock.patch.dict(os.environ, {"CACHE_RELEASE_INTERNAL": "1", "CACHE_RELEASE_INTERNAL_RECEIPT": "/tmp/manual.json"}):
            with self.assertRaises(gate.Refusal):
                gate.authorize(self.root, gate.PHASES[0])

    def test_unknown_fit_is_uncertainty_and_known_oversize_refuses(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(all(row["status"] == "unknown" for row in gate.fit_records().values()))
        evidence = Path(self.temp.name) / "duration.log"
        put(evidence, "real recorded wall 700 seconds\n")
        record = Path(self.temp.name) / "fit.json"
        put(record, {"schema_version": 1, "phases": {gate.PHASES[0]: [{"unit": "unchanged indivisible workload",
             "wall_seconds": 700, "evidence": str(evidence), "evidence_sha256": gate.sha256(evidence)}]}})
        with mock.patch.dict(os.environ, {"CACHE_RELEASE_FIT_RECORD": str(record)}):
            with self.assertRaisesRegex(gate.Refusal, "exceeds 540"):
                gate.fit_records()

    def test_arbitrary_script_override_refused(self):
        with self.assertRaisesRegex(gate.Refusal, "canonical"):
            gate.canonical_script("/tmp/arbitrary-work.sh")

    def test_global_lock_excludes_other_output_roots(self):
        lock = Path(self.temp.name) / "machine-driver.lock"
        with mock.patch.object(gate, "GLOBAL_LOCK", lock):
            with gate.driver_lock():
                with self.assertRaisesRegex(gate.Refusal, "another cache release driver"):
                    with gate.driver_lock():
                        self.fail("second machine-wide driver admitted")


class ProcessOwnershipTests(unittest.TestCase):
    def test_tracking_descendants_in_another_process_group_does_not_claim_foreign(self):
        rows = {10: {"pid": 10, "ppid": 1, "pgid": 10, "birth": "leader"},
                11: {"pid": 11, "ppid": 10, "pgid": 11, "birth": "child"},
                12: {"pid": 12, "ppid": 11, "pgid": 12, "birth": "escaped grandchild"},
                13: {"pid": 13, "ppid": 1, "pgid": 13, "birth": "foreign"}}
        owned = {}
        with mock.patch.object(os, "getsid", side_effect=lambda pid: 10 if pid == 10 else pid):
            self.assertEqual(gate.track_owned(rows, 10, owned), {10, 11, 12})
        self.assertNotIn(13, owned)

    def test_reused_pid_is_never_owned(self):
        identity = {"pid": 10, "birth": "original"}
        self.assertFalse(gate.same_process(identity, {10: {"birth": "replacement"}}))

    def test_signal_refuses_reused_kernel_birth(self):
        with mock.patch.object(gate, "kernel_process", return_value={"birth": "replacement"}), \
                mock.patch.object(os, "kill") as kill:
            self.assertFalse(gate.signal_owned({"pid": 10, "birth": "original"}, signal.SIGKILL))
            kill.assert_not_called()

    def test_unavailable_kernel_birth_is_not_process_absence(self):
        with mock.patch.object(gate, "kernel_process", side_effect=PermissionError("inspection denied")), \
                mock.patch.object(os, "kill") as kill:
            with self.assertRaises(PermissionError):
                gate.signal_owned({"pid": 10, "birth": "original"}, signal.SIGKILL)
            kill.assert_not_called()

    def test_inaccessible_foreign_birth_preserves_conflicts_not_owned_absence(self):
        output = ("1 0 1 Mon Jan 1 00:00:00 2026 /usr/bin/metal\n"
                  "10 1 10 Mon Jan 1 00:00:00 2026 /usr/bin/python\n")

        def identity(pid):
            if pid == 1:
                raise PermissionError("foreign system process")
            return {"ppid": 1, "pgid": 10, "birth": "owned"}

        with mock.patch.object(subprocess, "check_output", return_value=output), \
                mock.patch.object(gate, "kernel_process", side_effect=identity):
            table = gate.process_table()
        with mock.patch.object(os, "getsid", side_effect=lambda pid: pid):
            self.assertEqual(gate.track_owned(table, 10, {}), {10})
            with self.assertRaisesRegex(gate.Refusal, "owned descendant"):
                gate.track_owned(table, 1, {})
        self.assertEqual([row["pid"] for row in gate.conflicts(gate.PHASES[0], table)], [1])
        with self.assertRaisesRegex(gate.Refusal, "owned process identity"):
            gate.same_process({"pid": 1, "birth": "previously owned"}, table)

    def test_actual_process_group_is_owned_and_reaped(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            table = gate.process_table()
            owned = {}
            self.assertIn(process.pid, gate.track_owned(table, process.pid, owned))
            process.terminate()
            process.wait(timeout=3)
            self.assertFalse(gate.same_process(owned[process.pid], gate.process_table()))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()



def isolated_lease_fixture(root: Path, scenario: str, work: str) -> None:
    """Test-only short clocks and fault injection, in a disposable process."""
    gate.WALL_SECONDS, gate.TERM_SECONDS, gate.KILL_SECONDS = 3, 0.8, 1.4
    driver = gate.process_identity(os.getpid())
    put(gate.attempt_path(root, work, "start"), {})
    os.environ["OUT_ROOT"] = str(root)
    patches = []

    def hang(*_args, **_kwargs):
        put(root / "entered", {"pid": os.getpid(), "birth": gate.kernel_process(os.getpid())["birth"]})
        while True:
            time.sleep(0.05)

    def authorization(*_args, **_kwargs):
        put(root / "authorization-entered", {"at_ns": time.time_ns()})
        if scenario in ("authorization", "descendants"):
            if scenario == "descendants":
                subprocess.Popen([sys.executable, "-c", DESCENDANT_FIXTURE, str(root)], start_new_session=True)
            return hang()
        return {"plan_sha256": None, "driver": driver}

    patches.append(mock.patch.object(gate, "authorize", side_effect=authorization))
    # Admission is not the subject of these timeout injections; actual CLI
    # rejection and live secret/lock lineage are exercised separately below.
    if scenario != "conflicts":
        patches.append(mock.patch.object(gate, "conflicts", return_value=[]))
    target = {"canonical": "canonical_script", "inventory": "inventory", "conflicts": "conflicts",
              "signature": "candidate_inputs", "telemetry": "telemetry_sample",
              "sealing": "snapshot_artifacts"}.get(scenario)
    if target:
        patches.append(mock.patch.object(gate, target, side_effect=hang))
    if target or scenario in ("authorization", "descendants"):
        # Start the injected clock when the blocked phase is actually reached,
        # not while a contended runner is still scheduling worker setup. The
        # real outer subprocess timeout still catches a phase outside the lease.
        phase_started = None
        real_monotonic = time.monotonic

        def phase_monotonic():
            nonlocal phase_started
            if phase_started is None:
                if not (root / "entered").exists():
                    return 0.0
                phase_started = real_monotonic()
            return real_monotonic() - phase_started

        phase_time = mock.Mock(wraps=time)
        phase_time.monotonic.side_effect = phase_monotonic
        patches.append(mock.patch.object(gate, "time", phase_time))
    if scenario == "canonical-argv":
        original_popen = subprocess.Popen

        def launch(argv, *args, **kwargs):
            if argv[:2] == ["bash", str(gate.canonical_script(None))]:
                put(root / "launch-contract", {"argv": argv, "cwd": str(kwargs["cwd"]),
                                                "work": kwargs["env"]["CACHE_RELEASE_WORK"]})
                return original_popen([sys.executable, "-c", "print('model-free canonical launch observation')"],
                                      *args, **kwargs)
            return original_popen(argv, *args, **kwargs)

        patches.append(mock.patch.object(subprocess, "Popen", side_effect=launch))
        gate.WALL_SECONDS, gate.TERM_SECONDS, gate.KILL_SECONDS = 4, 2, 2.5
    if scenario == "stranded":
        original_table = gate.process_table

        def unavailable_table():
            if not (root / "inspection-restored").exists():
                raise gate.Refusal("injected unavailable cleanup proof")
            return original_table()

        patches.append(mock.patch.object(gate, "process_table", side_effect=unavailable_table))
    if scenario == "success":
        gate.WALL_SECONDS, gate.TERM_SECONDS, gate.KILL_SECONDS = 4, 2, 2.5
        probe_fd = os.open(root / "guardian-only-fd", os.O_RDWR | os.O_CREAT, 0o600)

        def successful_work(_work, _script, _began, _mono):
            try:
                os.fstat(probe_fd)
            except OSError:
                put(root / "descriptor-closed", "worker did not inherit guardian descriptor\n")
            else:
                raise AssertionError("worker inherited guardian-only descriptor")
            child = subprocess.Popen([sys.executable, "-c",
                                      "import socket,time; from pathlib import Path; import sys; "
                                      "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
                                      "Path(sys.argv[1]).write_text(str(s.getsockname()[1])); time.sleep(.2)",
                                      str(root / "listener-port")])
            put(root / "entered", {"pid": os.getpid(), "birth": gate.kernel_process(os.getpid())["birth"]})
            child.wait(timeout=2)
            return 0

        patches.append(mock.patch.object(gate, "ticket_work", side_effect=successful_work))
    lock_fd = os.open(root / "machine.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            result = gate.main(["_ticket", work])
    finally:
        os.close(lock_fd)
        if scenario == "success":
            os.close(probe_fd)
    print(json.dumps({"result": result}), flush=True)


DESCENDANT_FIXTURE = r"""
import json, os, signal, socket, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
def record(name):
    with (root / name).open("a") as f:
        f.write(str(os.getpid()) + "\n")
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
(root / "listener-port").write_text(str(listener.getsockname()[1]))
record("descendant-pids")
spawned = False
def term(*_):
    global spawned
    record("descendant-terms")
    if not spawned:
        spawned = True
        subprocess.Popen([sys.executable, "-c", '''
import os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
(root / "descendant-pids").open("a").write(str(os.getpid()) + "\\n")
def term(*_):
    (root / "descendant-terms").open("a").write(str(os.getpid()) + "\\n")
signal.signal(signal.SIGTERM, term)
while True: time.sleep(.05)
''', str(root)], start_new_session=True)
signal.signal(signal.SIGTERM, term)
while True: time.sleep(.05)
"""


class WholeLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, scenario, work="build-probe"):
        command = [sys.executable, "-c",
                   "import sys; from pathlib import Path; "
                   "from test_cache_release_orchestrator import isolated_lease_fixture; "
                   "isolated_lease_fixture(Path(sys.argv[1]), sys.argv[2], sys.argv[3])",
                   str(self.root), scenario, work]
        began = time.monotonic()
        result = subprocess.run(command, cwd=Path(__file__).parent, text=True, capture_output=True, timeout=6)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result"], 1, result.stderr)
        self.assertLess(time.monotonic() - began, 4, result.stderr)
        self.assertTrue((self.root / "entered").exists(), result.stderr)
        entered = gate.load(self.root / "entered")
        self.assertFalse(gate.same_process(entered, gate.process_table()))
        end = gate.attempt_path(self.root, work, "end")
        if end.exists():
            self.assertNotEqual(gate.load(end)["status"], "pass")
        return result

    def test_authorization_is_inside_build_lease(self):
        self.fixture("authorization")
        self.assertFalse(gate.attempt_path(self.root, "build-probe", "ticket").exists())

    def test_finalize_authorization_has_the_same_finite_lease(self):
        self.fixture("authorization", "finalize")

    def test_path_resolution_is_inside_lease(self):
        self.fixture("canonical")

    def test_initial_inventory_is_inside_lease(self):
        self.fixture("inventory")
        ticket = gate.load(gate.attempt_path(self.root, "build-probe", "ticket"))
        self.assertLessEqual(ticket["began_ns"], gate.load(self.root / "authorization-entered")["at_ns"],
                             "authorization must not reset the lease clock")

    def test_initial_conflict_scan_is_inside_lease(self):
        self.fixture("conflicts")

    def test_signed_candidate_check_is_inside_lease(self):
        self.fixture("signature")

    def test_initial_telemetry_is_inside_lease(self):
        self.fixture("telemetry", gate.PHASES[0])

    def test_evidence_sealing_is_inside_lease(self):
        self.fixture("sealing", "build-debug")

    def test_preflight_and_term_spawned_descendants_leave_no_listener(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.fixture("descendants")
            pids = {int(row) for row in (self.root / "descendant-pids").read_text().splitlines()}
            terms = {int(row) for row in (self.root / "descendant-terms").read_text().splitlines()}
            self.assertEqual(len(pids), 2, "TERM must spawn a late owned descendant")
            self.assertEqual(terms, pids, "each descendant must receive TERM before KILL")
            table = gate.process_table()
            self.assertTrue(pids.isdisjoint(table), "owned descendants survived lease return")
            port = int((self.root / "listener-port").read_text())
            with socket.socket() as client:
                self.assertNotEqual(client.connect_ex(("127.0.0.1", port)), 0)
            self.assertIsNone(foreign.poll(), "foreign process must never be signaled")
        finally:
            foreign.terminate()
            foreign.wait(timeout=3)

    def test_actual_cli_rejects_unauthenticated_ticket_and_work(self):
        environment = dict(os.environ, OUT_ROOT=str(self.root), CACHE_RELEASE_INTERNAL="1",
                           CACHE_RELEASE_INTERNAL_RECEIPT=str(self.root / "forged.json"))
        script = gate.canonical_script(None)
        for command in ([sys.executable, str(Path(gate.__file__)), "_ticket", "build-probe"],
                        ["bash", str(script), "_work", "build-probe"]):
            with self.subTest(command=command):
                result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("driver capability", result.stderr)
                self.assertFalse((self.root / "build").exists())

    def test_actual_cli_rejects_alternate_command_authority(self):
        result = subprocess.run([sys.executable, str(Path(gate.__file__)), "--script", "/tmp/arbitrary.sh",
                                 "_ticket", "build-probe"],
                                env=dict(os.environ, OUT_ROOT=str(self.root)), text=True,
                                capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical workload", result.stderr)

    def test_actual_cli_has_no_arbitrary_command_entry(self):
        for arguments in (["_ticket", "echo"], ["_ticket", "build-probe", "echo"],
                          ["--command", "echo", "_ticket", "build-probe"]):
            result = subprocess.run([sys.executable, str(Path(gate.__file__)), *arguments],
                                    env=dict(os.environ, OUT_ROOT=str(self.root)), text=True,
                                    capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(gate.attempt_path(self.root, "build-probe", "ticket").exists())

    def test_success_requires_owned_exit_and_closes_guardian_only_descriptors(self):
        command = [sys.executable, "-c",
                   "import sys; from pathlib import Path; "
                   "from test_cache_release_orchestrator import isolated_lease_fixture; "
                   "isolated_lease_fixture(Path(sys.argv[1]), 'success', 'build-probe')", str(self.root)]
        result = subprocess.run(command, cwd=Path(__file__).parent, text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result"], 0, result.stderr)
        self.assertTrue((self.root / "descriptor-closed").exists())
        self.assertFalse(gate.same_process(gate.load(self.root / "entered"), gate.process_table()))
        with socket.socket() as client:
            self.assertNotEqual(client.connect_ex(("127.0.0.1", int((self.root / "listener-port").read_text()))), 0)

    def test_canonical_work_argv_and_environment_are_preserved(self):
        command = [sys.executable, "-c",
                   "import sys; from pathlib import Path; "
                   "from test_cache_release_orchestrator import isolated_lease_fixture; "
                   "isolated_lease_fixture(Path(sys.argv[1]), 'canonical-argv', 'build-debug')", str(self.root)]
        result = subprocess.run(command, cwd=Path(__file__).parent, text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result"], 0, result.stderr)
        script = gate.canonical_script(None)
        self.assertEqual(gate.load(self.root / "launch-contract"),
                         {"argv": ["bash", str(script), "_work", "build-debug"],
                          "cwd": str(script.parent.parent), "work": "build-debug"})

    def test_stranded_proof_failure_retains_lock_and_never_reports_success(self):
        command = [sys.executable, "-c",
                   "import sys; from pathlib import Path; "
                   "from test_cache_release_orchestrator import isolated_lease_fixture; "
                   "isolated_lease_fixture(Path(sys.argv[1]), 'stranded', 'build-probe')", str(self.root)]
        error_path = self.root / "guardian.stderr"
        with error_path.open("w") as errors:
            process = subprocess.Popen(command, cwd=Path(__file__).parent, text=True,
                                       stdout=subprocess.PIPE, stderr=errors)
            try:
                until = time.monotonic() + 5
                while "STRANDED / FAIL-CLOSED" not in error_path.read_text() and time.monotonic() < until:
                    self.assertIsNone(process.poll(), error_path.read_text())
                    time.sleep(0.02)
                self.assertIn("STRANDED / FAIL-CLOSED", error_path.read_text())
                self.assertIsNone(process.poll())
                with mock.patch.object(gate, "GLOBAL_LOCK", self.root / "machine.lock"):
                    with self.assertRaisesRegex(gate.Refusal, "another cache release driver"):
                        with gate.driver_lock():
                            self.fail("stranded guardian dropped its lock")
            finally:
                put(self.root / "inspection-restored", "operator restored inspection\n")
                stdout, _ = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(json.loads(stdout)["result"], 1)

if __name__ == "__main__":
    unittest.main()
