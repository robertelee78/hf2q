#!/usr/bin/env python3
"""Pinned, validation-only Qwen3.8 matched-reference producer.

The input is emitted by hf2q's source-teacher operator and already contains
the exact token ids and prediction schedule. This program never tokenizes or
renders prompts. It writes the same bounded F32 row framing as hf2q, plus a
non-authoritative evidence JSON that the Rust comparator independently
reconstructs and verifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

TRANSFORMERS_REPOSITORY = "https://github.com/huggingface/transformers"
TRANSFORMERS_COMMIT = "945dac9117cb54196888c0e6c08035792a98c485"
TARGET_MAGIC = b"hf2q-exact-teacher-targets-v1\0"
ROW_MAGIC = b"ROW1"
GREEDY_TOKEN_COUNT = 32


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
            length += len(chunk)
    return length, digest.hexdigest()


def prefix_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"hf2q-teacher-prefix-token-ids-v1")
    digest.update(struct.pack("<Q", len(tokens)))
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def rendered_tokens_sha256(stable_id: str, tokens: list[int]) -> str:
    encoded_id = stable_id.encode("utf-8")
    framed = bytearray(b"hf2q-token-ids-v1")
    framed.extend(struct.pack("<I", len(encoded_id)))
    framed.extend(encoded_id)
    framed.extend(struct.pack("<Q", len(tokens)))
    for token in tokens:
        framed.extend(struct.pack("<I", token))
    return sha256_bytes(framed)


def trajectory_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"hf2q-exact-teacher-greedy-tokens-v1")
    digest.update(struct.pack("<Q", len(tokens)))
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def reference_input_hash_view(reference_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": reference_input["schema_version"],
        "profile": reference_input["profile"],
        "prediction_plan": reference_input["prediction_plan"],
        "vocabulary_size": reference_input["vocabulary_size"],
        "target_limits": reference_input["target_limits"],
        "examples": reference_input["examples"],
        "source_teacher_authority": reference_input["source_teacher_authority"],
        "sensitivity_authority": reference_input["sensitivity_authority"],
        "allocator_authority": reference_input["allocator_authority"],
        "selector_authority": reference_input["selector_authority"],
        "autoquant_authority": reference_input["autoquant_authority"],
        "runtime_dependency": reference_input["runtime_dependency"],
    }


def validate_reference_input(reference_input: dict[str, Any]) -> None:
    expected = sha256_bytes(canonical_json(reference_input_hash_view(reference_input)))
    if reference_input.get("reference_input_sha256") != expected:
        raise ValueError("reference input SHA-256 does not reproduce")
    if reference_input.get("profile") != "exact_teacher_reference_input_v1":
        raise ValueError("unsupported reference input profile")
    vocabulary = int(reference_input["vocabulary_size"])
    examples = reference_input["examples"]
    receipts = reference_input["prediction_plan"]["examples"]
    if vocabulary <= 0 or len(examples) != len(receipts):
        raise ValueError("reference input dimensions are invalid")
    retained: dict[str, list[int]] = {}
    for example, receipt in zip(examples, receipts, strict=True):
        tokens = [int(token) for token in example["token_ids"]]
        if (
            example["stable_id"] != receipt["stable_id"]
            or example["render_mode"] != receipt["render_mode"]
            or len(tokens) != receipt["token_count"]
            or rendered_tokens_sha256(example["stable_id"], tokens)
            != receipt["token_ids_sha256"]
            or any(token < 0 or token >= vocabulary for token in tokens)
        ):
            raise ValueError("reference example differs from its plan receipt")
        retained[example["stable_id"]] = tokens
    for point in reference_input["prediction_plan"]["prediction_points"]:
        tokens = retained[point["stable_id"]]
        prefix_count = int(point["prefix_token_count"])
        if (
            prefix_count > len(tokens)
            or prefix_sha256(tokens[:prefix_count])
            != point["prefix_token_ids_sha256"]
        ):
            raise ValueError("prediction point prefix differs from retained tokens")
        if point["kind"] == "teacher_forced":
            target_index = int(point["target_token_index"])
            if tokens[target_index] != int(point["target_token_id"]):
                raise ValueError("teacher-forced target differs from retained tokens")


def evidence_hash_view(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": evidence["schema_version"],
        "profile": evidence["profile"],
        "reference_input_sha256": evidence["reference_input_sha256"],
        "prediction_plan_sha256": evidence["prediction_plan_sha256"],
        "target_artifact": evidence["target_artifact"],
        "greedy_trajectories": evidence["greedy_trajectories"],
        "implementation": evidence["implementation"],
        "external_reference": evidence["external_reference"],
        "runtime_dependency": evidence["runtime_dependency"],
        "source_teacher_authority": evidence["source_teacher_authority"],
        "sensitivity_authority": evidence["sensitivity_authority"],
        "allocator_authority": evidence["allocator_authority"],
        "selector_authority": evidence["selector_authority"],
        "autoquant_authority": evidence["autoquant_authority"],
        "dwq": evidence["dwq"],
    }


class TargetWriter:
    def __init__(self, output: Path, vocabulary_size: int, point_count: int) -> None:
        if output.exists():
            raise FileExistsError(f"target already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.external.", dir=output.parent
        )
        self.output = output
        self.temporary = Path(temporary)
        self.file = os.fdopen(descriptor, "wb")
        self.vocabulary_size = vocabulary_size
        self.point_count = point_count
        self.next_ordinal = 0
        self.file.write(TARGET_MAGIC)

    def write(self, point: dict[str, Any], logits: Any) -> int:
        import numpy as np

        ordinal = int(point["point_ordinal"])
        row = np.asarray(logits, dtype="<f4")
        if ordinal != self.next_ordinal or row.shape != (self.vocabulary_size,):
            raise ValueError("reference row order or vocabulary differs from the plan")
        if not np.isfinite(row).all():
            raise ValueError("reference logits contain a non-finite value")
        payload = row.tobytes(order="C")
        self.file.write(ROW_MAGIC)
        self.file.write(struct.pack("<Q", ordinal))
        self.file.write(struct.pack("<Q", self.vocabulary_size))
        self.file.write(bytes.fromhex(point["prefix_token_ids_sha256"]))
        self.file.write(struct.pack("<Q", len(payload)))
        self.file.write(payload)
        self.next_ordinal += 1
        return int(np.argmax(row))

    def finish(self) -> tuple[int, str]:
        if self.next_ordinal != self.point_count:
            raise ValueError("reference target is missing prediction rows")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        expected = len(TARGET_MAGIC) + self.point_count * (
            60 + self.vocabulary_size * 4
        )
        length, digest = sha256_file(self.temporary)
        if length != expected:
            raise ValueError("reference target length differs from checked framing")
        os.link(self.temporary, self.output)
        self.temporary.unlink()
        return length, digest

    def abort(self) -> None:
        try:
            if not self.file.closed:
                self.file.close()
        finally:
            self.temporary.unlink(missing_ok=True)


def last_logits(model: Any, torch: Any, device: Any, tokens: list[int], cache: Any) -> tuple[Any, Any]:
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    past_length = 0 if cache is None else int(cache.get_seq_length())
    attention_mask = torch.ones((1, past_length + len(tokens)), dtype=torch.long, device=device)
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    logits = result.logits[0, -1].to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return logits, result.past_key_values


def execute_plan(reference_input: dict[str, Any], writer: TargetWriter, model: Any, torch: Any, device: Any) -> list[dict[str, Any]]:
    plan = reference_input["prediction_plan"]
    points_by_example: dict[str, list[dict[str, Any]]] = {}
    for point in plan["prediction_points"]:
        points_by_example.setdefault(point["stable_id"], []).append(point)
    prompts = {prompt["stable_id"]: prompt for prompt in plan["greedy_prompts"]}
    trajectories: list[dict[str, Any]] = []

    for example in reference_input["examples"]:
        stable_id = example["stable_id"]
        tokens = [int(token) for token in example["token_ids"]]
        points = points_by_example.get(stable_id, [])
        if not points:
            continue
        cache = None
        first_prefix = int(points[0]["prefix_token_count"])
        logits, cache = last_logits(model, torch, device, tokens[:first_prefix], cache)
        current_prefix = first_prefix
        point_index = 0
        while point_index < len(points):
            point = points[point_index]
            desired_prefix = int(point["prefix_token_count"])
            while current_prefix < desired_prefix:
                logits, cache = last_logits(
                    model, torch, device, [tokens[current_prefix]], cache
                )
                current_prefix += 1
            if current_prefix != desired_prefix:
                raise ValueError("prediction prefixes are not monotonically ordered")
            first_token = writer.write(point, logits)
            point_index += 1

        prompt = prompts.get(stable_id)
        if prompt is not None:
            if len(points) != 1 or points[0]["kind"] != "generation_next":
                raise ValueError("generation example has a non-canonical point schedule")
            generated = [first_token]
            for _ in range(1, GREEDY_TOKEN_COUNT):
                logits, cache = last_logits(model, torch, device, [generated[-1]], cache)
                generated.append(int(logits.argmax()))
            trajectories.append(
                {
                    "stable_id": stable_id,
                    "prompt_token_ids_sha256": prompt["prefix_token_ids_sha256"],
                    "token_ids": generated,
                    "token_ids_sha256": trajectory_sha256(generated),
                }
            )
    if len(trajectories) != len(plan["greedy_prompts"]):
        raise ValueError("reference execution did not produce every greedy trajectory")
    return trajectories


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(canonical_json(value))
            destination.write(b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-teacher-summary", type=Path, required=True)
    parser.add_argument("--output-target", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.source_teacher_summary.read_text(encoding="utf-8"))
    if not summary.get("executed") or not summary.get("structural_target_receipt"):
        raise ValueError("source-teacher summary is not a completed native execution")
    reference_input = summary["reference_input"]
    validate_reference_input(reference_input)
    if summary["prediction_plan_sha256"] != reference_input["prediction_plan"]["manifest_sha256"]:
        raise ValueError("source-teacher summary and reference input plans differ")

    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the pinned external reference requires an available MPS device")
    device = torch.device(args.device)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    vocabulary = int(reference_input["vocabulary_size"])
    if int(model.config.text_config.vocab_size) != vocabulary:
        raise ValueError("external model vocabulary differs from hf2q's exact plan")

    writer = TargetWriter(
        args.output_target,
        vocabulary,
        len(reference_input["prediction_plan"]["prediction_points"]),
    )
    try:
        trajectories = execute_plan(reference_input, writer, model, torch, device)
        target_length, target_sha256 = writer.finish()
    except BaseException:
        writer.abort()
        raise
    del model
    if args.device == "mps":
        torch.mps.empty_cache()

    lock_path = Path(__file__).with_name("uv.lock")
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "profile": "external_exact_teacher_reference_target_v1",
        "reference_input_sha256": reference_input["reference_input_sha256"],
        "prediction_plan_sha256": reference_input["prediction_plan"]["manifest_sha256"],
        "target_artifact": {
            "artifact_id": "external_exact_teacher_logits",
            "role": "external_full_vocabulary_f32_target_rows",
            "byte_len": target_length,
            "sha256": target_sha256,
        },
        "greedy_trajectories": trajectories,
        "implementation": {
            "name": "huggingface_transformers.Qwen3_5ForConditionalGeneration",
            "repository_url": TRANSFORMERS_REPOSITORY,
            "repository_commit": TRANSFORMERS_COMMIT,
            "package_version": transformers.__version__,
            "dependency_lock_sha256": sha256_file(lock_path)[1],
            "python_version": platform.python_version(),
            "framework_version": torch.__version__,
            "device": str(device),
            "source_dtype": "bfloat16",
            "logit_dtype": "f32_le",
            "attention_implementation": "eager",
            "cache_enabled": True,
        },
        "external_reference": True,
        "runtime_dependency": False,
        "source_teacher_authority": False,
        "sensitivity_authority": False,
        "allocator_authority": False,
        "selector_authority": False,
        "autoquant_authority": False,
        "dwq": False,
        "evidence_sha256": "",
    }
    evidence["evidence_sha256"] = sha256_bytes(canonical_json(evidence_hash_view(evidence)))
    atomic_write_json(args.output_evidence, evidence)
    print(canonical_json(evidence).decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"qwen38 reference failed: {error}", file=sys.stderr)
        raise
