#!/usr/bin/env python3
"""phantom-kv pipeline ported to the hybrid Qwen3.5 architecture (ADR-059).

The reference implementation (phantom-kv, /opt/phantom-kv) trains and
serves KV grafts on standard-attention models: the bank covers EVERY
layer and splices into a DynamicCache at every layer index. Qwen3.5 is
hybrid — full attention at 8 of 32 layers (interval 4), DeltaNet
recurrent state elsewhere — which is exactly the hf2q graft surface
(ADR-059 site `full_attn_kv`).

This port keeps their pipeline verbatim — donor text, suites, judge,
run-file format, target distillation, the three-role objective
(ce 50% / sup 25% / kl 25%), hyperparameters — and adapts exactly one
seam: the bank is [n_full_attn_layers, n_slots, H, D] and every cache
splice lands at the full-attention layer indices {3, 7, ..., 31},
matching hf2q's serve-time splice semantics (graft below, real
conversation above, DeltaNet layers untouched).

Subcommands (the pipeline in order):
  base-run   ungrafted scoreboard run -> their run-file JSON
  v1-build   derive the v1 donor bank (their donor text) -> phantom.bin
  v1-run     grafted scoreboard run under the v1 bank -> run-file JSON
  distill    base + v1 runs -> training targets JSONL (their constitution)
  train      train the K/V bank (their objective/hparams) -> ckpt_final.pt
  eval       grafted scoreboard run under the trained bank -> run-file JSON
  to-gguf    their container -> hf2q graft.* GGUF (serves in hf2q)

Usage: python3 phantom_port.py <subcommand> [options]
(env: /opt/phantom-kv/.venv — torch + transformers + phantom_kv -e)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/opt/phantom-kv/src")

from phantom_kv.eval.refusal import is_refusal, matched_pattern  # noqa: E402
from phantom_kv.graft.build import (  # noqa: E402
    load_prefill_source,
    shape_prefill,
    user_turn_suffix,
)
from phantom_kv.graft.format import (  # noqa: E402
    DIRECTKV_KIND,
    GRAFT_KIND,
    save_graft,
    validate_graft,
)
from phantom_kv.model import load_model  # noqa: E402
from phantom_kv.train.softprompt import (  # noqa: E402
    ROLE_WEIGHTS,
    _base_logprobs,
    _collate,
    _tokenize_row,
)
from safetensors.torch import load_file as safetensors_load_file  # noqa: E402

SUITES = Path("/opt/phantom-kv/data/suites")
DONOR_SOURCE = "/opt/phantom-kv/data/grafts/v1_prefill.json"

# Multi-model registry. "template" selects the donor-render adapter:
#   "split"  — phantom's generic chat-template prefix/suffix split
#              (works when the template renders turns append-only).
#   "gemma4" — hand-shaped render for Gemma-4's loop-based template
#              (its re-rendering defeats the generic split): turns are
#              <|turn>role\n...<turn|>\n, the system block is native,
#              the generation header is <|turn>model\n plus a closed
#              thought channel, and the eot is <turn|> (token 106).
MODELS = {
    "qwen35-4b": {
        "path": "/opt/hf2q/models/sources/Qwen3.5-4B",
        "eot_token": "<|im_end|>",
        "template": "split",
        "art": "/opt/hf2q/artifacts/grafts/qwen35-4b",
        "gguf_identity_name": "Qwen3.5-4B",
        "gguf_facts": {
            "rope_theta": 1e7, "rotary_dim": 64, "n_kv_heads": 4,
            "head_dim": 256, "layers": [3, 7, 11, 15, 19, 23, 27, 31],
        },
    },
    "gemma4-26b": {
        "path": "/opt/hf2q/models/sources/gemma-4-26B-A4B-it",
        "eot_token": "<turn|>",
        "template": "gemma4",
        "art": "/opt/hf2q/artifacts/grafts/gemma4-26b",
        "gguf_identity_name": "Gemma-4-26B-A4B-It",
        "gguf_facts": None,  # filled from the model config at to-gguf time
    },
}
DEFAULT_MODEL = "qwen35-4b"


def model_facts(key: str) -> dict:
    return MODELS[key]


def shape_prefill_for(key: str, tokenizer, system: str, ack: str) -> str:
    kind = MODELS[key]["template"]
    if kind == "split":
        return shape_prefill(tokenizer, system, ack)
    if kind == "gemma4":
        return (
            "<bos><|turn>system\n" + system.strip() + "<turn|>\n"
            "<|turn>model\n" + ack.strip() + "<turn|>\n"
        )
    raise ValueError(f"unknown template adapter {kind!r}")


def user_turn_suffix_for(key: str, tokenizer, prompt: str) -> str:
    kind = MODELS[key]["template"]
    if kind == "split":
        return user_turn_suffix(tokenizer, prompt)
    if kind == "gemma4":
        return (
            "<|turn>user\n" + prompt.strip() + "<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>"
        )
    raise ValueError(f"unknown template adapter {kind!r}")


def chat_prompt_for(key: str, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def eot_id_for(key: str, tokenizer) -> int:
    return tokenizer.convert_tokens_to_ids(MODELS[key]["eot_token"])


def tokenize_row_for(key: str, tokenizer, row: dict, eot_id: int) -> dict:
    """Their _tokenize_row, parameterized on the model's suffix/eot."""
    prompt_ids = tokenizer(
        user_turn_suffix_for(key, tokenizer, row["prompt"]),
        add_special_tokens=False,
    )["input_ids"]
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    if row["role"] in ("ce", "sup"):
        completion_ids = completion_ids + [eot_id]
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"ids": prompt_ids + completion_ids, "labels": labels, "start": len(prompt_ids)}


def full_attention_indices(config) -> list[int]:
    types = getattr(config, "layer_types", None)
    if types is None:
        types = getattr(getattr(config, "text_config", None), "layer_types", None)
    if types is None:
        raise ValueError("config exposes no layer_types (directly or via text_config)")
    return [i for i, t in enumerate(types) if t == "full_attention"]


def load_suite(name: str) -> list[dict]:
    rows = []
    for line in (SUITES / f"{name}.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ── Hybrid graft: bank over the full-attention layers only ─────────────


class GraftHybrid:
    """Bank [n_full, n_slots, H, D]; splices at the full-attention indices."""

    def __init__(self, k, v, meta, path, full_idx, config):
        self.k = k
        self.v = v
        self.meta = meta
        self.path = Path(path)
        self.full_idx = full_idx
        self.config = config

    @property
    def n_slots(self) -> int:
        return self.k.shape[1]

    @property
    def sha256_12(self) -> str:
        return self.meta["sha256"][:12]

    def new_cache(self, batch_size: int = 1):
        from transformers.cache_utils import DynamicCache

        dtype = self.k.dtype
        cache = DynamicCache(config=self.config)
        for j, i in enumerate(self.full_idx):
            k = self.k[j].permute(1, 0, 2).unsqueeze(0)
            v = self.v[j].permute(1, 0, 2).unsqueeze(0)
            if batch_size != 1:
                k = k.expand(batch_size, -1, -1, -1)
                v = v.expand(batch_size, -1, -1, -1)
            cache.update(k.to(dtype), v.to(dtype), i)
        return cache


def load_graft_hybrid(path: str, config, device: str, dtype) -> GraftHybrid:
    p = Path(path)
    meta = json.loads(p.with_suffix(".json").read_text())
    tensors = safetensors_load_file(str(p))
    validate_graft(meta, {k: tuple(v.shape) for k, v in tensors.items()})
    full_idx = full_attention_indices(config)
    assert meta["n_layers"] == len(full_idx), (
        f"bank covers {meta['n_layers']} layers, model has {len(full_idx)} full-attn"
    )
    return GraftHybrid(
        tensors["k"].to(device=device, dtype=dtype),
        tensors["v"].to(device=device, dtype=dtype),
        meta, p, full_idx, config,
    )


# ── Scoreboard runs (their runner semantics, hybrid cache) ─────────────


def generate_completion(model, tokenizer, device, prompt, max_new_tokens, graft, model_key):
    if graft is None:
        prompt_tensor = tokenizer(
            chat_prompt_for(model_key, tokenizer, prompt), return_tensors="pt"
        ).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                prompt_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
    else:
        prompt_tensor = tokenizer(
            user_turn_suffix_for(model_key, tokenizer, prompt),
            return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(device)
        mask = torch.ones(
            1, prompt_tensor.shape[1] + graft.n_slots, dtype=torch.long, device=device
        )
        with torch.no_grad():
            out = model.generate(
                prompt_tensor,
                past_key_values=graft.new_cache(),
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
    new_ids = out[0, prompt_tensor.shape[1]:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run_scoreboard(model, tokenizer, device, config, graft, graft_label, out_path, model_key,
                   harmful="harmful_seed", harmless="harmless_seed", max_new_tokens=128):
    started = time.perf_counter()
    results = {}
    for suite, name in ((harmful, "harmful"), (harmless, "harmless")):
        rows = []
        for item in load_suite(suite):
            text = generate_completion(
                model, tokenizer, device, item["prompt"], max_new_tokens, graft, model_key
            )
            rows.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "completion": text,
                "refusal": is_refusal(text),
                "pattern": matched_pattern(text),
            })
            print(f"[run] {graft_label} {name} {item['id']}: refusal={rows[-1]['refusal']}")
        refusals = sum(r["refusal"] for r in rows)
        print(f"[run] {graft_label} {name}: {refusals}/{len(rows)} refusals")
        results[name] = rows
    report = {
        "env": {"model_id": MODELS[model_key]["path"], "device": str(device)},
        "graft": None if graft is None else {
            "path": str(graft.path),
            "kind": graft.meta["kind"],
            "n_slots": graft.n_slots,
            "sha256_12": graft.sha256_12,
            "alias": graft_label,
        },
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"[run] wrote {out_path}")
    return report


# ── v1 build: donor prefill -> bank (hybrid extraction) ────────────────


def cmd_v1_build(args):
    key = args.model
    model, tokenizer, info = load_model(MODELS[key]["path"])
    device = info["device"]
    config = model.config
    full_idx = full_attention_indices(config)
    system, ack = load_prefill_source(DONOR_SOURCE)
    prefill_text = shape_prefill_for(key, tokenizer, system, ack)
    ids = tokenizer(
        prefill_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    n_slots = ids.shape[1]
    print(f"[v1] prefill tokens: {n_slots} | full-attn layers: {full_idx}")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    k = torch.stack([cache.layers[i].keys for i in full_idx]).squeeze(1).permute(0, 2, 1, 3).contiguous()
    v = torch.stack([cache.layers[i].values for i in full_idx]).squeeze(1).permute(0, 2, 1, 3).contiguous()
    print(f"[v1] bank: k {tuple(k.shape)} v {tuple(v.shape)}")
    meta = save_graft(
        args.out, k.float().cpu(), v.float().cpu(),
        {
            "kind": GRAFT_KIND,
            "model_id": MODELS[key]["path"],
            "n_layers": len(full_idx),
            "n_slots": n_slots,
            "n_kv_heads": k.shape[2],
            "head_dim": k.shape[-1],
            "dtype": "fp32",
            "source_sha256_12": __import__("hashlib").sha256(
                Path(DONOR_SOURCE).read_bytes()
            ).hexdigest()[:12],
            "prefill_text": prefill_text,
        },
    )
    print(f"[v1] wrote {args.out} (+ .json) sha256_12={meta['sha256'][:12]}")


# ── Distill: their constitution, our counts ────────────────────────────


def cmd_distill(args):
    base = json.loads(Path(args.base_run).read_text())
    v1 = json.loads(Path(args.v1_run).read_text())
    base_harmful = {r["id"]: r for r in base["results"]["harmful"]}
    v1_harmful = {r["id"]: r for r in v1["results"]["harmful"]}
    assert set(base_harmful) == set(v1_harmful), "harmful id sets differ"
    rows = []
    for pid, brow in base_harmful.items():
        vrow = v1_harmful[pid]
        if brow["refusal"] and not vrow["refusal"]:
            rows.append({"id": pid, "role": "ce", "prompt": brow["prompt"],
                         "completion": vrow["completion"]})
        elif not brow["refusal"]:
            rows.append({"id": pid, "role": "ce", "prompt": brow["prompt"],
                         "completion": brow["completion"]})
        if brow["refusal"]:
            rows.append({"id": pid, "role": "sup", "prompt": brow["prompt"],
                         "completion": brow["completion"]})
    for r in base["results"]["harmless"]:
        rows.append({"id": r["id"], "role": "kl", "prompt": r["prompt"],
                     "completion": r["completion"]})
    counts = {}
    for row in rows:
        counts[row["role"]] = counts.get(row["role"], 0) + 1
    print(f"[distill] counts: {counts} (their constitution: ce=flips+complies, sup=base refusals, kl=harmless)")
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[distill] wrote {args.out} ({len(rows)} rows)")


# ── Train: their objective verbatim, hybrid cache ──────────────────────


class DirectKVParamsHybrid:
    def __init__(self, warm_path, device, config):
        graft = load_graft_hybrid_raw(warm_path)
        k0 = graft["k"].float().to(device)
        v0 = graft["v"].float().to(device)
        self.k = torch.nn.Parameter(k0.clone().detach())
        self.v = torch.nn.Parameter(v0.clone().detach())
        self.k_anchor = k0.detach()
        self.v_anchor = v0.detach()
        self.meta = graft["meta"]
        self.path = str(warm_path)

    @property
    def n_slots(self):
        return self.k.shape[1]

    def anchor_loss(self):
        dk = (self.k - self.k_anchor).pow(2)
        dv = (self.v - self.v_anchor).pow(2)
        return torch.cat([dk.flatten(), dv.flatten()]).mean()

    def new_cache(self, dtype, batch_size, full_idx, config):
        from transformers.cache_utils import DynamicCache

        kb, vb = self.k.to(dtype), self.v.to(dtype)
        cache = DynamicCache(config=config)
        for j, i in enumerate(full_idx):
            k = kb[j].permute(1, 0, 2).unsqueeze(0)
            v = vb[j].permute(1, 0, 2).unsqueeze(0)
            if batch_size != 1:
                k = k.expand(batch_size, -1, -1, -1)
                v = v.expand(batch_size, -1, -1, -1)
            cache.update(k, v, i)
        return cache


def load_graft_hybrid_raw(path):
    p = Path(path)
    meta = json.loads(p.with_suffix(".json").read_text())
    tensors = safetensors_load_file(str(p))
    return {"k": tensors["k"], "v": tensors["v"], "meta": meta}


def cmd_train(args):
    key = args.model
    model, tokenizer, info = load_model(MODELS[key]["path"])
    model.requires_grad_(False)
    device = info["device"]
    config = model.config
    full_idx = full_attention_indices(config)
    eot_id = eot_id_for(key, tokenizer)
    pdtype = model.get_input_embeddings().weight.dtype

    params = DirectKVParamsHybrid(args.warm_graft, device, config)
    print(f"[train] bank: {tuple(params.k.shape)} | warm {args.warm_graft}")

    # Gradient-flow proof (their hard gate), hybrid-adapted.
    ids = tokenizer("proof ok", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    mask = torch.ones(1, params.n_slots + ids.shape[1], dtype=torch.long, device=device)
    cache = params.new_cache(pdtype, 1, full_idx, config)
    model(input_ids=ids, past_key_values=cache, attention_mask=mask).logits.float().mean().backward()
    ok = params.k.grad is not None and bool((params.k.grad != 0).any()) \
        and params.v.grad is not None and bool((params.v.grad != 0).any())
    print(f"[proof] gradient flow: {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient-flow proof failed"
    params.k.grad = None
    params.v.grad = None

    opt = torch.optim.AdamW([params.k, params.v], lr=args.lr)
    rng = random.Random(args.seed)

    groups = {}
    for line in Path(args.targets).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            groups.setdefault(row["role"], []).append(row)
    seqs = {role: [tokenize_row_for(key, tokenizer, row, eot_id) for row in rows]
            for role, rows in groups.items()}
    print(f"[train] KL base log-probs precompute ({len(seqs['kl'])} rows)...")
    base_lp = _base_logprobs(model, seqs["kl"], device)

    queues = {}

    def take(role, n):
        pool = queues.setdefault(role, [])
        while len(pool) < n:
            pool += rng.sample(range(len(seqs[role])), len(seqs[role]))
        batch, queues[role] = pool[:n], pool[n:]
        return batch

    losses = {"ce": [], "sup": [], "kl": [], "anchor": []}
    roles, weights = zip(*ROLE_WEIGHTS.items())
    log_path = Path(args.out_dir) / "train.log"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"# model={MODELS[key]['path']} targets={args.targets} warm={args.warm_graft}"
                  f" seed={args.seed} steps={args.steps} lr={args.lr}"
                  f" micro_batch={args.micro_batch} arm=kv-hybrid\n")
        for step in range(1, args.steps + 1):
            role = rng.choices(roles, weights=weights)[0]
            idx = take(role, args.micro_batch)
            rows = [seqs[role][i] for i in idx]
            input_ids, labels, token_mask = _collate(rows, tokenizer.pad_token_id, device)
            mask = torch.cat(
                [torch.ones(len(rows), params.n_slots, dtype=torch.long, device=device),
                 token_mask], dim=1,
            )
            cache = params.new_cache(pdtype, len(rows), full_idx, config)
            logits = model(
                input_ids=input_ids, past_key_values=cache, attention_mask=mask
            ).logits.float()
            shifted = logits[:, :-1]
            shifted_labels = labels[:, 1:]
            if role == "kl":
                base = torch.zeros(len(rows), labels.shape[1], logits.shape[-1])
                for i, (row, j) in enumerate(zip(rows, idx)):
                    lp = base_lp[j]
                    base[i, row["start"]:row["start"] + lp.shape[0]] = lp.cpu()
                base = base.to(device)[:, 1:]
                cand = F.log_softmax(shifted, dim=-1)
                p = base.exp()
                terms = torch.where(p > 0, p * (base - cand), torch.zeros_like(base))
                valid = (shifted_labels != -100).to(shifted.dtype)
                core = (terms.sum(-1) * valid).sum() / valid.sum().clamp(min=1)
            else:
                ce = F.cross_entropy(
                    shifted.reshape(-1, shifted.shape[-1]), shifted_labels.reshape(-1),
                    ignore_index=-100, reduction="none",
                ).reshape_as(shifted_labels)
                if role == "sup":
                    valid = shifted_labels != -100
                    mean_logp = -(ce.sum(-1) / valid.sum(-1).clamp(min=1))
                    core = torch.relu(mean_logp + args.sup_margin).mean()
                else:
                    core = ce.sum() / (shifted_labels != -100).sum().clamp(min=1)
            anchor_loss = params.anchor_loss()
            loss = core + args.anchor * anchor_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            value = core.item()
            losses[role].append(value)
            losses["anchor"].append(anchor_loss.item())
            if step % 10 == 0 or step == 1:
                line = f"step={step} role={role} loss={value:.4f} anchor={anchor_loss.item():.3e}"
                log.write(line + "\n")
                log.flush()
                print(f"[train] {line}")

    ckpt = Path(args.out_dir) / "ckpt_final.pt"
    torch.save({
        "k": params.k.detach().cpu(),
        "v": params.v.detach().cpu(),
        "meta": {
            "model_id": MODELS[key]["path"], "graft_kind": DIRECTKV_KIND, "arm": "kv-hybrid",
            "seed": args.seed, "steps": args.steps, "lr": args.lr,
            "micro_batch": args.micro_batch, "sup_margin": args.sup_margin,
            "anchor": args.anchor, "n_slots": params.n_slots,
            "targets_path": str(args.targets), "warm_graft": params.path,
            "losses": {r: {"mean_first10": sum(v[:10]) / len(v[:10]) if v else 0.0,
                           "mean_last10": sum(v[-10:]) / len(v[-10:]) if v else 0.0}
                       for r, v in losses.items()},
        },
    }, ckpt)
    print(f"[train] wrote {ckpt}")


def cmd_compile(args):
    """Trained ckpt -> their container (bf16) with trained=True provenance."""
    ckpt = torch.load(args.ckpt, map_location="cpu")
    k, v, meta = ckpt["k"], ckpt["v"], ckpt["meta"]
    warm = load_graft_hybrid_raw(meta["warm_graft"])
    save_graft(
        args.out, k.to(torch.bfloat16), v.to(torch.bfloat16),
        {
            "kind": DIRECTKV_KIND,
            "model_id": ckpt["meta"]["model_id"],
            "n_layers": k.shape[0],
            "n_slots": meta["n_slots"],
            "n_kv_heads": k.shape[2],
            "head_dim": k.shape[3],
            "dtype": "bf16",
            "source_sha256_12": warm["meta"].get("source_sha256_12", "0" * 12),
            "prefill_text": warm["meta"].get("prefill_text", ""),
            "trained": True,
            "arm": "kv-hybrid",
            "warm_graft_sha256_12": warm["meta"]["sha256"][:12],
            "steps": meta["steps"], "seed": meta["seed"],
        },
    )
    print(f"[compile] wrote {args.out} (+ .json)")


# ── to-gguf: their container -> hf2q graft.* GGUF ──────────────────────


def cmd_to_gguf(args):
    import struct

    key = args.model
    facts = MODELS[key]["gguf_facts"]
    if facts is None:
        # Derive from the HF config: full-attn layer indices + the
        # GLOBAL (full-attention) geometry + rope theta.
        import json as _json
        cfg = _json.load(open(MODELS[key]["path"] + "/config.json"))
        text = cfg.get("text_config", cfg)
        layer_types = text["layer_types"]
        layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        facts = {
            "rope_theta": text.get("rope_theta_global", text.get("rope_theta", 1e6)),
            "rotary_dim": int(text.get("rope_local_base_freq", 0) and 0) or text.get(
                "rope_embedding_head_dim", text.get("head_dim", 256)),
            "n_kv_heads": text.get("num_global_key_value_heads", text.get("num_key_value_heads")),
            "head_dim": text.get("global_head_dim", text.get("head_dim")),
            "layers": layers,
        }
        MODELS[key]["gguf_facts"] = facts
    graft = load_graft_hybrid_raw(args.bank)
    k = graft["k"].float()
    v = graft["v"].float()
    meta = graft["meta"]
    n_slots, heads, head_dim = k.shape[1], k.shape[2], k.shape[3]
    layers = facts["layers"]
    assert k.shape[0] == len(layers), f"bank layers {k.shape[0]} != full-attn {len(layers)}"
    assert heads == facts["n_kv_heads"] and head_dim == facts["head_dim"], (
        f"bank geometry {heads}x{head_dim} != model {facts['n_kv_heads']}x{facts['head_dim']}"
    )

    def kv_string(key, value):
        return (struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 8)
                + struct.pack("<Q", len(value)) + value.encode())

    def kv_u32(key, value):
        return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 4) + struct.pack("<I", value)

    def kv_f32(key, value):
        return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 6) + struct.pack("<f", value)

    def kv_bool(key, value):
        return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 7) + struct.pack("<B", 1 if value else 0)

    metadata = [
        kv_string("graft.mode", "splice_prefix"),
        kv_u32("graft.spec_version", 1),
        kv_string("graft.hook_point", "full_attn_kv"),
        kv_string("graft.kind", "direct_kv"),
        kv_u32("graft.n_slots", n_slots),
        kv_f32("graft.rope_theta", float(facts["rope_theta"])),
        kv_u32("graft.rotary_dim", int(facts["rotary_dim"])),
        kv_u32("graft.position_base", 0),
        kv_bool("graft.mrope_interleaved", True),
        kv_string("graft.quant_lane", "f32"),
        kv_string("general.base_model.0.name", MODELS[key]["gguf_identity_name"]),
    ]
    tensors = []
    for j, layer in enumerate(layers):
        tensors.append((f"graft.k.{layer}", [n_slots, heads, head_dim],
                        k[j].contiguous().numpy().tobytes()))
        tensors.append((f"graft.v.{layer}", [n_slots, heads, head_dim],
                        v[j].contiguous().numpy().tobytes()))
    out = bytearray(b"GGUF" + struct.pack("<I", 3)
                    + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata)))
    for kv in metadata:
        out += kv
    offset = 0
    for name, dims, _ in tensors:
        out += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", 0) + struct.pack("<Q", offset)
        offset += 4 * dims[0] * dims[1] * dims[2]
    while len(out) % 32:
        out += b"\0"
    for _, _, payload in tensors:
        out += payload
    Path(args.out).write_bytes(bytes(out))
    print(f"[to-gguf] wrote {args.out}: n_slots={n_slots} layers={layers} "
          f"trained={meta.get('trained', False)} bytes={len(out)}")


# ── CLI ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("base-run")
    p.add_argument("--out", default=None)

    p = sub.add_parser("v1-build")
    p.add_argument("--out", default=None)

    p = sub.add_parser("v1-run")
    p.add_argument("--bank", default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("distill")
    p.add_argument("--base-run", default=None)
    p.add_argument("--v1-run", default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("train")
    p.add_argument("--targets", default=None)
    p.add_argument("--warm-graft", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sup-margin", type=float, default=3.0)
    p.add_argument("--anchor", type=float, default=1e-2)

    p = sub.add_parser("compile")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("eval")
    p.add_argument("--bank", default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("to-gguf")
    p.add_argument("--bank", default=None)
    p.add_argument("--out", default=None)

    for name, subp in sub.choices.items():
        subp.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    args = parser.parse_args()
    ART = Path(MODELS[args.model]["art"])
    ART.mkdir(parents=True, exist_ok=True)
    args.art = ART
    D = {
        "base-run": {"out": "run_base.json"},
        "v1-build": {"out": "v1_prefill_kv.bin"},
        "v1-run": {"bank": "v1_prefill_kv.bin", "out": "run_v1.json"},
        "distill": {"base_run": "run_base.json", "v1_run": "run_v1.json", "out": "targets.jsonl"},
        "train": {"targets": "targets.jsonl", "warm_graft": "v1_prefill_kv.bin",
                  "out_dir": "train"},
        "compile": {"ckpt": "train/ckpt_final.pt", "out": "trained_kv.bin"},
        "eval": {"bank": "trained_kv.bin", "out": "run_trained.json"},
        "to-gguf": {"bank": "trained_kv.bin", "out": "trained.graft.gguf"},
    }
    for field, default in D.get(args.cmd, {}).items():
        if getattr(args, field) is None:
            setattr(args, field, str(ART / default))

    if args.cmd == "base-run":
        model, tokenizer, info = load_model(MODELS[args.model]["path"])
        run_scoreboard(model, tokenizer, info["device"], model.config,
                       None, "base", args.out, args.model)
    elif args.cmd == "v1-build":
        cmd_v1_build(args)
    elif args.cmd == "v1-run":
        model, tokenizer, info = load_model(MODELS[args.model]["path"])
        graft = load_graft_hybrid(args.bank, model.config, info["device"], model.dtype)
        run_scoreboard(model, tokenizer, info["device"], model.config,
                       graft, "v1", args.out, args.model)
    elif args.cmd == "distill":
        cmd_distill(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "compile":
        cmd_compile(args)
    elif args.cmd == "eval":
        model, tokenizer, info = load_model(MODELS[args.model]["path"])
        graft = load_graft_hybrid(args.bank, model.config, info["device"], model.dtype)
        run_scoreboard(model, tokenizer, info["device"], model.config,
                       graft, "trained", args.out, args.model)
    elif args.cmd == "to-gguf":
        cmd_to_gguf(args)


if __name__ == "__main__":
    main()
