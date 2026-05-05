# /cfa ↔ hf2q integration — v0.1 design

- **Date:** 2026-05-05
- **Status:** Design ready; cfa.md edit ships separately
- **Cross-refs:** ADR-017 (Phase D Closed-Shipped), ADR-017 Phase E option (a) research dossier `docs/research/adr017-phase-e-option-a-2026-05-05.md`

## TL;DR

- **Integration shape:** option (c) — `hf2q serve --kv-persist` sidecar per /cfa session. /cfa Phase 0 launches it; Phase 5 tears it down via `POST /shutdown` + SIGTERM drain (graceful spill flushes KV cache to disk).
- **Worker invocation:** workers call the sidecar via `Bash: curl http://localhost:$PORT/v1/chat/completions ...` (mirrors the existing Codex backend's "external process per backend" pattern).
- **Default flag:** `--kv-persist` is ON for /cfa-launched sidecars (cache_dir = `~/.cache/cfa/kv-persist/`, budget = 4 GiB). Global `hf2q serve` default-OFF stays unchanged per ADR-017 R-F1.
- **What /cfa gets:** R-P6 (4-agent shared system prompt → 1.00× single-agent prefill) is automatic via in-process PromptCache replay (Phase E option b, shipped). R-P5 (44,500× cold-process resume) applies across CFA invocations when same system prompt is reused.
- **What /cfa does NOT get yet:** LCP partial-prefill resume (Phase E option a, iter-1 landed `9beb906` as standalone substrate; iter-2/2.5/3 still pending). Until E.a iter-3 lands, /cfa workers with shared-prefix-different-suffix prompts (different role descriptions on top of same system prompt) get no cache hit — only fully-identical prompt fan-outs hit.

## Motivation

Today /cfa already wires two model backends:
- **Claude:** via `Task()` → Claude Code's Anthropic-API path
- **Codex:** via `Bash: codex exec --json` → OpenAI-API through codex CLI

A third backend — **hf2q** — would let /cfa run inference against a local Gemma 4 / Qwen 3.5 / Qwen 3-VL model without an external API. The headline win is ADR-017's R-P6 ship-gate: 4 concurrent agents sharing a 4 K system prompt → aggregate prefill ≤ 1.25× single-agent prefill (measured at 1.00× — exactly what /cfa Phase 2 fan-out is shaped like).

## Why option (c) sidecar, not (a) HTTP-only / (b) library / (d) replace-Codex

| Option | /cfa fit | Why |
|---|---|---|
| **(a) HTTP to existing `hf2q serve`** | Conditional | Requires the user to have one running. Most won't. /cfa would need a "if not, start one" branch — that's just (c) by another name. |
| **(b) Library binding** | Bad | Each worker spawns in-process → each worker has its own KV state → R-P6 mechanism never fires (no shared cache). Loses the headline. |
| **(c) Sidecar per session** | **Best** | Self-contained; R-P6 fires automatically (all workers share one server's PromptCache); aligns with /cfa's existing "external process per backend" pattern. |
| **(d) Replace Codex** | Different scope | Interesting future direction (Claude + hf2q-Gemma4 dual-mode) but conflates two concerns. /cfa-Codex value is the second-opinion review, not the model. Replacing it loses that. |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ /cfa Phase 0 launcher                                       │
│   ├─ pick random free port                                  │
│   ├─ Bash run_in_background:true:                           │
│   │   hf2q serve --model <user-spec'd>                      │
│   │                --kv-persist ~/.cache/cfa/kv-persist/    │
│   │                --host 127.0.0.1                         │
│   │                --port $RANDOM_PORT                      │
│   │     env HF2Q_KV_PERSIST_BUDGET_BYTES=4294967296 (4 GiB) │
│   ├─ until grep -q "hf2q serving on" stderr; do sleep 0.5;  │
│   ├─ memory_store HF2Q_SERVE_URL = "http://127.0.0.1:$PORT" │
│   └─ continue with hive-mind_init / queen spawn / ...       │
│                                                              │
│ /cfa Phase 2 workers (Claude Task or Codex Bash):           │
│   workers read HF2Q_SERVE_URL from shared memory then:      │
│   curl -sS $HF2Q_SERVE_URL/v1/chat/completions \            │
│        -H 'Content-Type: application/json' \                │
│        -d '{"model":"...","messages":[...],"stream":true}'  │
│   (or use any existing OpenAI-compat client lib)            │
│                                                              │
│ /cfa Phase 5 teardown (launcher):                           │
│   ├─ Bash: curl -X POST $HF2Q_SERVE_URL/shutdown            │
│   │       (returns 202 Accepted; raises SIGTERM internally) │
│   ├─ wait_for graceful_exit (drain flushes KV cache to disk)│
│   └─ kill -9 $BG_PID  (only if graceful drain timed out)    │
└──────────────────────────────────────────────────────────────┘
```

## What about R-P6 in actual /cfa fan-out?

The R-P6 ship-gate measures aggregate prefill on **byte-identical prompts**. Real /cfa Phase 2 fan-out sends *shared-prefix-different-suffix* prompts (workers get `[SYSTEM] [QUEEN_SPEC] [your role: architect; produce ...]`, `[your role: coder; implement ...]`, etc. — same prefix, divergent suffix per worker).

| Pattern | shipped E.b helps? | E.a iter-3 will help? |
|---|---|---|
| 4 workers send identical prompt (R-P6 fixture) | ✅ in-process PromptCache full-equality replay → 1.00× | ✅ same |
| 4 workers send same prefix, different suffix (real /cfa) | ❌ misses | ✅ partial-prefix LCP hit on `[SYSTEM] [QUEEN_SPEC]` shared chunk |
| Same exact prompt across CFA sessions (cache survives restart) | ✅ R-P5 cross-process replay → 44,500× speedup | ✅ same |

So:
- **At v0.1 ship time:** /cfa-hf2q gets the *idealized* R-P6 case (R-P6 fixture happy path) and the cross-session R-P5 case. Real shared-prefix-different-suffix fan-outs are *cache-miss* per worker on E.b alone.
- **After E.a iter-3:** /cfa gets the *generalized* R-P6 case — the shared-prefix portion of every worker's prompt is reused regardless of suffix divergence. THIS is when the R-P6 marketing claim holds for production /cfa workloads.

## Default flags — global `hf2q serve` vs /cfa-launched sidecar

| Flag | Global `hf2q serve` | /cfa-launched sidecar | Rationale |
|---|---|---|---|
| `--kv-persist` | OFF (R-F1) | **ON** | /cfa controls cache_dir + budget + lifecycle; cross-CFA-session cache reuse is /cfa's headline. |
| `--kv-persist=PATH` | n/a | `~/.cache/cfa/kv-persist/` | Stable per-user path; survives across CFA sessions until LRU-evicted. |
| `HF2Q_KV_PERSIST_BUDGET_BYTES` | 0 (unlimited) | **4 GiB** (`4294967296`) | Matches ADR-017 R-F5 spec default. Bounded growth; LRU eviction wired in iter-11 commit `c2eeecd`. |
| Port | user-specified | random free | /cfa picks a free port to allow multiple concurrent CFA sessions. |
| `--host` | configurable | `127.0.0.1` | Sidecar is /cfa-internal; never bind 0.0.0.0. |

This keeps R-F1 honest at the global level (default-off) while giving /cfa the cross-session benefit when the user reuses system prompts across CFA invocations.

## Cleanup discipline

`POST /shutdown` was added in ADR-017 closure iter-2 (commit `b2d0cda`) specifically to support this lifecycle. Internally it `libc::raise(SIGTERM)`; the `shutdown_signal()` future fires; axum gracefully drains in-flight HTTP responses; `drain_loaded_models_to_disk` walks the pool's loaded engines and fires `pre_evict` per entry to push block-aligned KV state into the async writer queue; the writer drains; the worker thread exits.

If graceful drain takes > 30 s (or fails for any reason), Phase 5 falls back to `kill -9 $BG_PID`. KV cache state may be incomplete in that case — operator-visible but not fatal (next CFA session's recovery scan rebuilds the index from whatever survived).

## What does the user need to do?

**Nothing.** /cfa Phase 0 launches the sidecar; Phase 5 tears it down. The user just runs `/cfa <task>` like always. Optional knobs:

| Env var | Default | When to set |
|---|---|---|
| `HF2Q_SERVE_MODEL` | `~/.cache/hf2q/models/...latest...` | Pin to a specific model GGUF path. |
| `HF2Q_SERVE_PORT_BASE` | random | Force a deterministic port (e.g. for tooling that expects 8080). |
| `CFA_HF2Q_DISABLE` | unset | Set to `1` to disable the hf2q sidecar entirely (fall back to Claude+Codex only). |
| `CFA_HF2Q_KV_PERSIST_BYTES` | `4294967296` | Override the 4 GiB default cache budget. |

## Operator visibility

The sidecar exposes the standard hf2q observability surface:

- `GET /v1/models` — list of loaded model families with bytes_resident
- `GET /metrics` — Prometheus exposition; ADR-017 §R-F7 counters:
  - `hf2q_pool_kv_spills_total{repo,quant,outcome=enqueued|skipped|error}`
  - `hf2q_pool_kv_restores_total{repo,quant,outcome=restored|skipped|error}`
  - `hf2q_kv_cache_bytes_on_disk` (gauge)
  - `hf2q_kv_cache_blocks_total` (gauge)
  - `hf2q_kv_cache_evictions_total{trigger=budget_overflow|...}`
- `POST /shutdown` — graceful drain (returns 202, takes effect via SIGTERM)
- `GET /readyz` — liveness probe
- stderr: `[stress]`-style structured log lines

/cfa Phase 5's `session_save` should snapshot `/metrics` so the user can see "your CFA session benefited from N cache hits" post-hoc.

## What's NOT in v0.1

- **B-hybrid families** (Qwen 3.5, Qwen 3-VL): blocked on ADR-013. /cfa-hf2q at v0.1 = Gemma 4 dense only. Document forward-pointer in cfa.md.
- **LCP partial-prefill (E.a iter-3)**: blocks the production R-P6 case for shared-prefix-different-suffix fan-outs. Until iter-3 lands, /cfa fan-outs with role-divergent suffixes get cache-MISS per worker. (E.a iter-1 landed standalone substrate at commit `9beb906`; iter-2 + iter-2.5 + iter-3 are next.)
- **TurboQuant payload variant (B-tq)**: blocked on ADR-007. ~4× cache footprint reduction when it lands.
- **Multi-model concurrent loading**: hf2q's HotSwapManager supports up to 3 models in pool by default, but /cfa v0.1 ships single-model-per-sidecar. Multi-model = future.

## v0.1 ship checklist

- [ ] Add the new "## hf2q sidecar (optional model backend)" section to `~/.claude/commands/cfa.md` (between Phase 0 Bootstrap and Phase 1 Queen sections, since it's a launcher-side concern).
- [ ] Add Phase 0 sidecar-launch logic (port pick, run_in_background spawn, /readyz wait, memory_store the URL).
- [ ] Add Phase 5 sidecar-teardown logic (`POST /shutdown` then wait_for_exit then SIGKILL fallback).
- [ ] Add the `CFA_HF2Q_DISABLE=1` short-circuit so users can opt out.
- [ ] Update Step 4 plan presentation to include "hf2q sidecar: <port> | disabled" line.
- [ ] Update fallbacks section: "hf2q binary missing or `hf2q serve --version` fails → set `CFA_HF2Q_DISABLE=1` automatically and warn in plan."
- [ ] Document the cross-session cache (`~/.cache/cfa/kv-persist/` 4 GiB LRU) in the cfa.md preamble.

## v0.2+ roadmap

- E.a iter-2/2.5/3 — production R-P6 win for shared-prefix-different-suffix fan-outs.
- B-hybrid — Qwen 3.5 / Qwen 3-VL family support (post ADR-013).
- B-tq — TurboQuant cache codec (post ADR-007); ~4× footprint reduction.
- Per-CFA-session metrics snapshot in Phase 5 final report.
- Multi-model concurrent loading (architect + coder workers run on different model sizes).

## Cross-references

- ADR-017 Phase D Closed-Shipped (commits `4830353` ... `9beb906`): R-C4/R-P4/R-P5/R-P6/R-C6/K1/K2/K3/stress 24h smoke/R-F5 all GREEN.
- ADR-017 Phase E option (a) research dossier `docs/research/adr017-phase-e-option-a-2026-05-05.md`: ~1,060 LOC scope, 7 iters, iter-1 = `LcpRegistry` (landed `9beb906`).
- `POST /shutdown` HTTP endpoint: `src/serve/api/handlers.rs::shutdown` (commit `b2d0cda`).
- `drain_loaded_models_to_disk` graceful-drain helper: `src/serve/mod.rs` (commit `b2d0cda`).
- `HF2Q_KV_PERSIST_BUDGET_BYTES`: `src/serve/mod.rs:2887-2902` (commit `c2eeecd`).
- R-F5 LRU eviction: `src/serve/kv_persist/writer.rs::process_job` (commit `c2eeecd`).
- ADR-017 R-F1 ("Default OFF until Phase D ships"): preserved at global level.
