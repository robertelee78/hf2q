#!/usr/bin/env python3
"""Read-only, source-bound inventory; no build, model load, or working-tree reads.

Run from any hf2q checkout:
  python3 docs/research/affine-retirement-2026-10-06/measure.py --repo .
Physical lines include comments/blanks inside named items. This is not SLOC,
an AST reachability analysis, a deletion estimate, or a performance measurement.
"""
import argparse
import hashlib
import json
import re
import subprocess

BASE = "cca6425860c1fb770c6223c2bbdfa5d2d769dc70"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--ref", default=BASE)
    args = parser.parse_args()

    def git(*argv):
        return subprocess.check_output(["git", "-C", args.repo, *argv])

    commit = git("rev-parse", args.ref + "^{commit}").decode().strip()
    paths = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
    files = {}

    def read(path):
        if path not in files:
            raw = git("show", commit + ":" + path)
            files[path] = (raw, raw.decode().splitlines())
        return files[path][1]

    def matches(pattern, roots):
        result = subprocess.run(
            ["git", "-C", args.repo, "grep", "-ilE", pattern, commit, "--", *roots],
            capture_output=True, text=True, check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        found = [line.split(":", 1)[1] for line in result.stdout.splitlines()]
        return {"file_count": len(found), "files": found}

    items = []

    def item(path, anchor, disposition, occurrence=0):
        lines = read(path)
        starts = [i for i, line in enumerate(lines) if re.search(anchor, line)]
        if len(starts) <= occurrence:
            raise ValueError(f"Missing item: {path}: {anchor}")
        start = starts[occurrence]
        indent = len(lines[start]) - len(lines[start].lstrip())
        # Source is rustfmt-formatted. End at the matching indentation's
        # standalone closing brace; no nested blocks share this indentation.
        end = next(i for i in range(start + 1, len(lines))
                   if lines[i] == " " * indent + "}")
        items.append({"path": path, "anchor": lines[start].strip(),
                      "start": start + 1, "end": end + 1,
                      "physical_lines": end - start + 1,
                      "disposition": disposition})

    loader = "src/core/mlx_safetensors_loader.rs"
    for anchor in [r"^pub struct MlxQuantConfig", r"^impl MlxQuantConfig",
                   r"^pub struct MlxAffineLinear \{", r"^impl MlxAffineLinear \{",
                   r"^pub struct RoundTripDrift", r"^pub fn unpack_u32_packed",
                   r"^pub fn pack_u32_codes", r"^pub fn write_floats_from_f32",
                   r"^pub struct MlxAffineLinearBytes", r"^impl MlxAffineLinearBytes"]:
        item(loader, anchor, "retire")
    item(loader, r"^impl MlxAffineLinear \{", "retire", occurrence=1)
    for name in ("read_floats_to_f32", "discover_shards"):
        item(loader, rf"^pub fn {name}\(", "preserve-and-relocate")
    for line in read(loader):
        match = re.match(r"    fn (\w+)\(", line)
        if match:
            name = match[1]
            disposition = ("preserve-and-relocate" if name.startswith(
                ("read_floats_to_f32_", "discover_shards_")) else "retire")
            item(loader, rf"^    fn {name}\(", disposition)

    shared = "src/serve/forward_mlx_shared.rs"
    for anchor in [r"^pub struct MlxAffineExtra", r"^pub struct MlxAffineMoeStack",
                   r"^    pub fn from_mlx_affine_linear", r"^pub enum DwqOverlayRole",
                   r"^pub fn parse_dwq_overlay_metadata", r"^pub fn parse_dwq_overlay_role",
                   r"^pub enum MoeBaseRole", r"^pub fn parse_dwq_moe_expert_role",
                   r"^mod ac5_iter_b_affine_qweight_roundtrip"]:
        item(shared, anchor, "retire")
    item("src/inference/models/gemma4/model.rs", r"^    pub fn apply_dwq_overlay", "retire")
    qwen = "src/inference/models/qwen35/model.rs"
    for anchor in [r"^    pub fn apply_dwq_overlay", r"^enum AttnRole",
                   r"^enum DenseFfnRole", r"^fn dwq_to_native_q4_0_f32",
                   r"^fn overwrite_dense_ffn_q4_0_linear", r"^fn overwrite_full_attn_f32_linear"]:
        item(qwen, anchor, "retire")
    item("src/inference/models/qwen35/gpu_ffn.rs", r"^    pub fn attach_affine_overlay", "retire")
    # Separate cohort found during independent review; do not silently change
    # the original 2,890-line affine-definition measurement.
    for path, anchor in [
        ("src/arch/registry.rs", r"^pub struct QualityThresholds"),
        ("src/arch/registry.rs", r"^impl QualityThresholds"),
        ("src/arch/conformance.rs", r"^pub struct QualityReport"),
        ("src/arch/conformance.rs", r"^impl QualityReport"),
    ]:
        item(path, anchor, "retire-arch-scaffold")

    context_paths = [loader, shared, qwen, "src/inference/models/gemma4/model.rs",
                     "tests/peer_parity_gates.rs", "tests/common/mlx_lm_runner.rs",
                     "scripts/adr020_ac7_boundary_validate.sh", "scripts/adr020_ac8_validate.sh",
                     "scripts/adr020_fetch_wikitext_calibration_jsonl.sh", "scripts/p11_re_emit_dwq.sh",
                     "docs/calibrator-onboarding.md", "docs/converting-qwen35.md",
                     "docs/adr/ADR-046-evidence-driven-apple-auto-quant.md",
                     "docs/research/mlx-affine-support-2026-09-23.md",
                     "scripts/peer_parity_run.sh", "tests/common/llama_cpp_runner.rs",
                     "tests/common/metrics.rs", "tests/common/env_lock.rs",
                     "tests/quality_thresholds.rs", "src/arch/conformance.rs",
                     "src/arch/registry.rs", "src/arch/smoke.rs"]
    contexts = []
    for path in context_paths:
        lines = read(path)
        contexts.append({"path": path, "physical_lines": len(lines),
                         "test_attributes": sum(bool(re.match(r"\s*#\[test\]", l)) for l in lines),
                         "ignore_attributes": sum(bool(re.match(r"\s*#\[ignore", l)) for l in lines),
                         "sha256": hashlib.sha256(files[path][0]).hexdigest()})
    unions = {}
    for entry in items:
        bucket = unions.setdefault(entry["disposition"], {})
        bucket.setdefault(entry["path"], set()).update(range(entry["start"], entry["end"] + 1))
    totals = {kind: {"physical_lines": sum(len(s) for s in by_path.values()),
                     "per_file": {p: len(s) for p, s in sorted(by_path.items())}}
              for kind, by_path in unions.items()}
    output = {
        "schema": "hf2q.affine-retirement-inventory.v1", "commit": commit,
        "measurement": "selected nonoverlapping physical source spans; incomplete removal footprint",
        "keyword_discovery_only": {
            "affine_src": matches("affine", ["src"]),
            "overlay_src": matches("dwq[_-]overlay", ["src"]),
            "overlay_scripts": matches("dwq[_-]overlay", ["scripts"]),
            "dwq_arch": matches("dwq", ["src/arch"]),
        },
        "selected_item_totals": totals, "selected_items": items, "whole_file_context": contexts,
        "absent_retired_producers": {p: p not in paths for p in [
            "src/backends/safetensors_out.rs", "src/calibrate/dwq_loop.rs",
            "src/quantize/dwq_activation.rs", "src/calibrate/imatrix_xvalidate.rs",
            "tests/safetensors_mlx_lm_round_trip.rs"]},
        "unmeasured": ["compiled bytes", "clean build seconds", "startup seconds", "resident bytes",
                       "prefill tokens/second", "decode tokens/second", "time to first token"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
