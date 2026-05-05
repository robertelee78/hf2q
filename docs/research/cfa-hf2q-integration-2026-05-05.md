# /cfa ↔ hf2q integration — v0.1 design

- **Date:** 2026-05-05 (rev 2 — singleton + server-side state)
- **Status:** Design ready; cfa.md edit + `src/serve/` API additions ship as part of v0.1
- **Cross-refs:** ADR-017 (Phase D Closed-Shipped + Phase E option b shipped), ADR-017 Phase E option (a) research dossier `docs/research/adr017-phase-e-option-a-2026-05-05.md`, `mantra.txt` (no shortcuts, no fallback, no stub), memory `feedback_no_client_side_state_machines.md`

## TL;DR

- **Integration shape:** option (c.2) — `hf2q serve --kv-persist` **shared singleton sidecar**. The first /cfa session spawns it under a brief `flock`; subsequent /cfa sessions ATTACH. Each /cfa session is tracked server-side via `POST /sessions/register` → background heartbeat → `DELETE /sessions/<id>`. The sidecar self-shuts via an idle-timeout watcher when `sessions == 0 && in_flight == 0 && idle_since > N`. **/cfa launchers never fire `/shutdown` directly.**
- **OOM bound:** one model's working set, NOT N × model. Two concurrent /cfa sessions on the same model share the sidecar's KV cache. Two concurrent /cfa sessions on different models trigger HotSwapManager (ADR-005:5712) — paid once per swap, amortized via ADR-017 R-P4 cache-hit (≤ 0.20× no-cache TTFT).
- **State authority:** all coordination state lives in the sidecar. /cfa launchers do client-side `flock` only around the discover-or-spawn decision; everything else (refcount, in-flight tracking, /shutdown gating, idle reaping, model arbitration) is a server-side mutex inside `hf2q serve`. Per `feedback_no_client_side_state_machines.md`: client-side state machines for distributed coordination are always asking for trouble.
- **Worker invocation:** workers call the sidecar via `Bash: curl http://localhost:$PORT/v1/chat/completions ...` — same shape /cfa already uses for Codex.
- **Defaults for /cfa-launched sidecar:** `--kv-persist` ON, cache_dir `~/.cache/cfa/kv-persist/`, budget 4 GiB, `--idle-timeout-seconds 60`. Global `hf2q serve` defaults unchanged per ADR-017 R-F1 (default-OFF at the global level).
- **What /cfa gets at v0.1:** R-P5 cross-process resume (44,500× cold-start speedup) for repeat invocations with byte-identical system prompts; the synthetic R-P6 fixture (4 byte-identical prompts → 1.00× aggregate prefill) hits via Phase E option b PromptCache replay. The natural /cfa Phase 2 fan-out shape (shared `[SYSTEM][QUEEN_SPEC]` prefix + per-role suffix) gets cache-MISS on workers 2-4 until E.a iter-3 lands.
- **What /cfa gets after E.a iter-3:** R-P6 1.00× generalizes to actual /cfa workloads via LCP partial-prefill resume. **Until then, public docs / launch post must NOT claim "R-P6 1.00× for real /cfa fan-outs"** — the synthetic-fixture number is correct as an ADR-017 ship-gate measurement but is not what /cfa workloads actually see at v0.1.

## Motivation

Today /cfa already wires two model backends:
- **Claude:** via `Task()` → Claude Code's Anthropic-API path
- **Codex:** via `Bash: codex exec --json` → OpenAI-API through codex CLI

A third backend — **hf2q** — would let /cfa run inference against a local Gemma 4 / Qwen 3.5 / Qwen 3-VL model without an external API. ADR-017 ship-gates R-P5 / R-P6 demonstrate the underlying mechanism is in place; the remaining work is the integration shape.

Two non-trivial constraints shape the design:

1. **Multiple concurrent /cfa sessions must coexist on a 128 GiB M5 Max.** A 30B-class model resident in RAM is ~20-30 GiB; per-session sidecars would OOM at 3-4 concurrent sessions.
2. **The lifecycle must be safe under crashes.** /cfa launchers can SIGKILL (panic, OOM, unplugged laptop). The sidecar must reap stale sessions; /cfa launchers must detect orphan sidecars.

Both push to a singleton with server-side authoritative state.

## Why option (c.2) singleton sidecar — not per-session, not (a)/(b)/(d)

| Option | /cfa fit | Why |
|---|---|---|
| **(a) HTTP to existing `hf2q serve`** | Conditional | Requires the user to have one running. Most won't. /cfa would need a "if not, start one" branch — that's just (c) by another name. |
| **(b) Library binding** | Bad | Each worker spawns in-process → each worker has its own KV state → no shared cache → R-P6 mechanism never fires. Also: no graceful drain, no central in-flight tracking, no place to refuse `/shutdown` while requests are streaming. |
| **(c.1) Sidecar per /cfa session** | Wrong | Two concurrent /cfa = 2× model RAM (OOM on 30B-class). Crash of /cfa-A's launcher shuts down the sidecar /cfa-B is using. Per-session is what we tried first; it had four named race windows (two-attach collide on empty-refs, /shutdown mid-streaming, stale-PID recycle, launcher-crash leaks ref forever) which drove the redesign to (c.2). |
| **(c.2) Singleton sidecar, server-tracked sessions** | **Best** | OOM bounded by 1× model's working set. Concurrent /cfa sessions share state via server-side `/sessions` refcount. `/shutdown` returns 409 if any session active OR any request in-flight. Idle-timeout watcher self-shuts when truly idle. Per `feedback_no_client_side_state_machines.md`. |
| **(d) Replace Codex** | Different scope | Interesting future direction (Claude + hf2q-Gemma4 dual-mode) but conflates two concerns. /cfa-Codex value is the second-opinion review, not the model. Replacing it loses that. |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ /cfa Phase 0 launcher (any /cfa session)                           │
│                                                                     │
│  1. Discover-or-spawn (under flock):                               │
│     exec 9>~/.cache/cfa/sidecar.lock                                │
│     flock 9                                                         │
│     PORT=$(cat ~/.cache/cfa/sidecar.port 2>/dev/null)               │
│     if ! curl -sf http://localhost:$PORT/readyz 2>/dev/null; then   │
│       hf2q serve --port 0 \                                         │
│                  --pidfile  ~/.cache/cfa/sidecar.pid \              │
│                  --portfile ~/.cache/cfa/sidecar.port \             │
│                  --kv-persist ~/.cache/cfa/kv-persist/ \            │
│                  --idle-timeout-seconds 60 \                        │
│                  --model "$MODEL" &                                 │
│       until [ -f ~/.cache/cfa/sidecar.port ] && \                   │
│             curl -sf http://localhost:$(cat ~/.cache/cfa/sidecar.port)/readyz; do
│         sleep 1                                                     │
│       done                                                          │
│       PORT=$(cat ~/.cache/cfa/sidecar.port)                         │
│     fi                                                              │
│     flock -u 9                                                      │
│                                                                     │
│  2. Ensure model (no sync auto-swap surprises):                    │
│     curl -sX POST http://localhost:$PORT/switch_model \             │
│            -d "{\"model\":\"$MODEL\"}"                              │
│     until [ "$(curl -s :$PORT/modelinfo | jq -r .loading)" = "false" ]; do
│       sleep 2                                                       │
│     done                                                            │
│                                                                     │
│  3. Register session:                                               │
│     SESSION=$(curl -sX POST :$PORT/sessions/register \              │
│                    -d "{\"slug\":\"$SLUG\"}" | jq -r .session_id)   │
│     ( while sleep 60; do \                                          │
│         curl -sX POST :$PORT/sessions/$SESSION/heartbeat; done ) &  │
│     HB_PID=$!                                                       │
│     memory_store HF2Q_SERVE_URL  "http://localhost:$PORT"           │
│     memory_store HF2Q_SESSION_ID "$SESSION"                         │
│     memory_store HF2Q_HB_PID     "$HB_PID"                          │
│                                                                     │
│ /cfa Phase 2 workers:                                               │
│     curl -sS $HF2Q_SERVE_URL/v1/chat/completions ...                │
│                                                                     │
│ /cfa Phase 5 teardown (any /cfa session):                          │
│     kill $HF2Q_HB_PID                                               │
│     curl -sX DELETE $HF2Q_SERVE_URL/sessions/$HF2Q_SESSION_ID       │
│     # /cfa is done. NO /shutdown call.                             │
│     # Sidecar self-shuts on idle-timeout if no other session.       │
└─────────────────────────────────────────────────────────────────────┘

   ┌───────────────────────────────────────────────────────────────┐
   │ hf2q serve (singleton; survives across /cfa sessions)         │
   │                                                               │
   │  Authoritative state:                                         │
   │    sessions:    Mutex<HashMap<SessionId, SessionMeta>>        │
   │    in_flight:   AtomicUsize  (already exists for /metrics)    │
   │    last_activity: Instant                                     │
   │                                                               │
   │  Endpoints — all atomic against `sessions` mutex:             │
   │    POST   /sessions/register       → 201 { id, ttl_seconds }  │
   │    POST   /sessions/<id>/heartbeat → 204                      │
   │    DELETE /sessions/<id>           → 204                      │
   │    GET    /sessions                → 200 [...]                │
   │    POST   /switch_model            → 200 / 202 / 423          │
   │    GET    /modelinfo               → 200 { loaded, loading }  │
   │    POST   /shutdown                → 200 if idle, 409 if not  │
   │                                                               │
   │  Idle watcher (tokio::spawn, 30s tick):                       │
   │    if sessions.is_empty() && in_flight == 0                   │
   │       && Instant::since(last_activity) > idle_timeout         │
   │    then graceful_self_shutdown                                │
   │      (drains KV cache via ADR-017 iter-2 b2d0cda path)        │
   │                                                               │
   │  TTL reaper (tokio::spawn, 30s tick):                         │
   │    for s in sessions {                                        │
   │      if now - s.last_heartbeat > s.ttl                        │
   │      then remove (launcher crashed; 600s default)             │
   │    }                                                          │
   └───────────────────────────────────────────────────────────────┘
```

## API surface additions on `hf2q serve` (v0.1 scope)

These do not exist today. Required for the /cfa integration to be correct.

### Sessions module (~300 LOC src + ~150 LOC tests)

```
POST /sessions/register { slug: "..." }
  → 201 { session_id, ttl_seconds: 600 }

POST /sessions/<id>/heartbeat
  → 204
  → 404 if session reaped or never registered

DELETE /sessions/<id>
  → 204

GET /sessions
  → 200 [{ id, slug, registered_at, last_heartbeat, ttl_remaining }]
```

Internal: `Arc<Mutex<HashMap<SessionId, SessionMeta>>>`. SessionId = UUID v4 (opaque). TTL reaper task wakes every 30s, removes sessions whose `last_heartbeat` is older than `ttl_seconds`. /cfa launcher heartbeats every 60s (TTL = 600s = ~10 keepalive intervals of headroom).

### Model swap module (~200 LOC src + ~80 LOC tests)

```
POST /switch_model { model: "Y" }
  → 200 if already at Y, no work
  → 202 { task_id } if swap kicked off
  → 423 Locked if HotSwapManager refuses (e.g., concurrent in-flight blocks swap)

GET /modelinfo
  → 200 {
      model: "Y" | "X→Y",
      loaded: bool,
      loading: bool,
      progress_pct: u8,        // 0-100; rough
      stage: "tensor_load" | "kv_restore" | "ready"
    }
```

Replaces the implicit sync-auto-swap-on-first-request shape (which has hidden long blocks + HTTP timeout risk). /cfa launcher polls `/modelinfo` with `sleep 2` until `loading == false`.

ETA expectations on M5 Max for client-side `MAX_WAIT` budget:

| Model + quant | Cold load | KV-restore (R-P4 ≤ 0.20×) | Total swap |
|---|---|---|---|
| Gemma4-26b Q4_0 | ~5-10s | ~50-100ms | ~10s |
| Qwen3.6-30B-A3B-DWQ48 | ~10-15s | ~100-200ms | ~15s |
| Qwen3.5-MoE 35B-DWQ46 | ~15-25s | ~200-400ms | ~25s |

`MAX_WAIT=60` generous; `MAX_WAIT=120` paranoid-safe.

### `/shutdown` semantics change (~30 LOC)

`POST /shutdown` was added in ADR-017 closure iter-2 (commit `b2d0cda`) as 202-always-accepts + internal SIGTERM. v0.1 makes it refcount-aware:

```
POST /shutdown
  → 200 if sessions.is_empty() && in_flight == 0
       (proceeds with graceful drain via existing iter-2 path)
  → 409 { active_sessions: [...], in_flight: N, advice: "DELETE sessions or wait" }
```

Idempotent on 200. The 409 path lets a tooling caller (or human running `curl /shutdown`) see what's blocking shutdown without surprise-cutting other clients' streaming responses.

### New CLI flags (~50 LOC)

```
--pidfile <path>             atomic write on /readyz; cleanup on graceful shutdown
--portfile <path>            same lifecycle; bound port (relevant for --port 0 random)
--idle-timeout-seconds <N>   default 0 (disabled). /cfa passes 60.
                             When > 0, sidecar self-shuts when sessions == 0 &&
                             in_flight == 0 && Instant::since(last_activity) > N.
```

### Idle watcher + TTL reaper tasks (~50 LOC)

`tokio::spawn` at `cmd_serve` startup if `--idle-timeout-seconds > 0`:
- Idle watcher wakes every 30s; checks the three conditions; on satisfaction, fires the existing `b2d0cda` graceful-shutdown path (`drain_loaded_models_to_disk` → SIGTERM internally → axum drain).
- TTL reaper wakes every 30s; removes sessions whose `last_heartbeat` is older than their `ttl_seconds`.

## R-P6 in actual /cfa fan-out

The R-P6 ship-gate measures aggregate prefill on **byte-identical prompts** (synthetic fixture). Real /cfa Phase 2 fan-out sends *shared-prefix-different-suffix* prompts: each worker gets `[SYSTEM] [QUEEN_SPEC] [your role: architect; produce ...]`, `[your role: coder; implement ...]`, etc. — same prefix, divergent suffix per worker.

| Pattern | Phase E.b (shipped) | Phase E.a iter-3 (post-v0.1?) |
|---|---|---|
| 4 workers send identical prompt (R-P6 fixture) | ✅ in-process PromptCache full-equality replay → 1.00× | ✅ same |
| 4 workers send same prefix, different suffix (real /cfa) | ❌ misses on workers 2-4 | ✅ partial-prefix LCP hit on `[SYSTEM][QUEEN_SPEC]` shared chunk |
| Same exact prompt across CFA sessions (cache survives restart) | ✅ R-P5 cross-process replay → 44,500× speedup | ✅ same |

So:
- **At v0.1 ship time:** /cfa-hf2q gets the *idealized* R-P6 case (byte-identical fan-outs) + the cross-session R-P5 case. Real shared-prefix-different-suffix fan-outs are cache-miss per worker.
- **After E.a iter-3:** /cfa gets the *generalized* R-P6 case — the shared prefix portion of every worker's prompt is reused regardless of suffix divergence. THIS is when "R-P6 1.00× for /cfa workloads" is honest marketing.

**Public docs / launch post discipline:** until E.a iter-3 lands, **DO NOT claim "R-P6 1.00× for /cfa"**. The synthetic-fixture number is a correct ADR-017 ship-gate measurement but is not what /cfa fan-outs actually see at v0.1. The honest v0.1 line is "R-P5 44,500× cross-session resume + R-P6 mechanism in place; per-fan-out amortization generalizes in v0.2 via E.a."

## Default flags — global `hf2q serve` vs /cfa-launched sidecar

| Flag | Global `hf2q serve` | /cfa-launched sidecar | Rationale |
|---|---|---|---|
| `--kv-persist` | OFF (R-F1) | **ON** | /cfa controls cache_dir + budget + lifecycle; cross-CFA-session cache reuse is /cfa's headline. |
| `--kv-persist=PATH` | n/a | `~/.cache/cfa/kv-persist/` | Stable per-user path; survives across CFA sessions until LRU-evicted. |
| `HF2Q_KV_PERSIST_BUDGET_BYTES` | 0 (unlimited) | **4 GiB** (`4294967296`) | ADR-017 R-F5 spec default. LRU eviction wired in iter-11 commit `c2eeecd`. |
| `--port` | user-specified | **0** (random free) | First /cfa session picks a free port; subsequent sessions ATTACH via `~/.cache/cfa/sidecar.port`. |
| `--host` | configurable | `127.0.0.1` | Sidecar is /cfa-internal; never bind 0.0.0.0. |
| `--pidfile` / `--portfile` | n/a | `~/.cache/cfa/sidecar.{pid,port}` | Discover-or-spawn coordination across concurrent /cfa launchers. |
| `--idle-timeout-seconds` | n/a | **60** | Sidecar self-shuts when sessions == 0 + in_flight == 0 + idle > 60s. |

## Cleanup discipline (singleton + server-side)

/cfa launchers do NOT fire `POST /shutdown`. They `DELETE /sessions/<id>`, kill the heartbeat bg loop, and exit. The sidecar handles its own lifecycle:

- **Last /cfa exits cleanly:** sidecar's session count drops to 0; idle-watcher (every 30s) checks `(sessions == 0 && in_flight == 0 && Instant::since(last_activity) > idle_timeout)`; on satisfaction, fires `drain_loaded_models_to_disk` (ADR-017 iter-2 `b2d0cda` graceful path) and exits. `--pidfile` and `--portfile` are removed at exit.

- **/cfa launcher crashes (SIGKILL, panic, unplugged laptop):** session entry stays in the sidecar with a stale `last_heartbeat`. TTL reaper task (every 30s) removes sessions where `now - last_heartbeat > ttl_seconds` (default 600s). After reaping, idle-watcher activates as in the clean-exit case.

- **Sidecar process crashes:** `~/.cache/cfa/sidecar.pid` is stale; next /cfa launcher's discover-or-spawn check (`curl /readyz`) fails → respawns under `flock`. KV cache state on disk survives; ADR-017 startup recovery rebuilds the `BlockIndex` on the new sidecar's first /readyz publish.

- **Concurrent /cfa launchers race to spawn:** `flock 9` serializes the discover-or-spawn decision. First in spawns; second in finds the spawn complete + /readyz green, falls through to ATTACH.

- **Two /cfa sessions want different models:** first /cfa Phase 0's `POST /switch_model` already loaded model X. Second /cfa Phase 0's `POST /switch_model { model: Y }` returns 202; HotSwapManager swap-loads Y; both /cfa sessions point at model Y for the rest of their work. Note: this evicts X mid-session for any /cfa-A worker that hasn't fired its request yet — they'll pay the swap latency on first request. **For v0.1, this is the documented behavior.** Per-session model pinning is v0.2 scope.

- **Two /cfa sessions, one streaming, second tries `/shutdown`:** the second's request returns 409 with `{ active_sessions: [...], in_flight: 1, advice: "..." }`. No surprise-cutting of the first's response. (In practice /cfa launchers don't fire /shutdown anyway, but human operators can — the 409 prevents foot-gun.)

## Operator visibility

- `GET /v1/models` — list of loaded model families with bytes_resident
- `GET /sessions` — active /cfa sessions (helpful for "why won't this sidecar shut down?")
- `GET /modelinfo` — model state + load progress
- `GET /metrics` — Prometheus exposition; ADR-017 §R-F7 counters:
  - `hf2q_pool_kv_spills_total{repo,quant,outcome=enqueued|skipped|error}`
  - `hf2q_pool_kv_restores_total{repo,quant,outcome=restored|skipped|error}`
  - `hf2q_kv_cache_bytes_on_disk` (gauge)
  - `hf2q_kv_cache_blocks_total` (gauge)
  - `hf2q_kv_cache_evictions_total{trigger=budget_overflow|...}`
  - **NEW v0.1** `hf2q_serve_active_sessions` (gauge)
  - **NEW v0.1** `hf2q_serve_in_flight_requests` (gauge)
- `POST /shutdown` — 200 if zero sessions + zero in-flight, else 409 with body
- `GET /readyz` — liveness probe
- stderr: `[stress]`-style structured log lines

/cfa Phase 5's `session_save` snapshots `/metrics` so the user can see "your CFA session benefited from N cache hits" post-hoc.

## What the user does

**Nothing different.** /cfa Phase 0 attach-or-spawns the singleton; Phase 5 deregisters its session. The user just runs `/cfa <task>` like always. With **two concurrent** /cfa sessions: both share one sidecar; finishing one doesn't shut down the other; OOM is bounded by one model's working set.

Optional knobs:

| Env var | Default | When to set |
|---|---|---|
| `HF2Q_SERVE_MODEL` | `~/.cache/hf2q/models/...latest...` | Pin to a specific model GGUF path. |
| `CFA_HF2Q_DISABLE` | unset | Set to `1` to disable the hf2q sidecar entirely (fall back to Claude+Codex only). |
| `CFA_HF2Q_KV_PERSIST_BYTES` | `4294967296` | Override the 4 GiB default cache budget. |
| `CFA_HF2Q_IDLE_TIMEOUT_SECONDS` | `60` | Tune sidecar idle-shutdown after last /cfa. |
| `CFA_HF2Q_SESSION_TTL_SECONDS` | `600` | Tune launcher-crash detection window. |

## What's NOT in v0.1

- **B-hybrid families** (Qwen 3.5, Qwen 3-VL): blocked on ADR-013. /cfa-hf2q at v0.1 = Gemma 4 dense only. Document forward-pointer in cfa.md.
- **LCP partial-prefill (E.a iter-3)**: blocks the production R-P6 case for shared-prefix-different-suffix fan-outs. Until iter-3 lands, /cfa fan-outs with role-divergent suffixes get cache-MISS per worker. (E.a iter-1 landed standalone substrate at commit `9beb906`; iter-2 + iter-2.5 + iter-3 are next.)
- **TurboQuant payload variant (B-tq)**: blocked on ADR-007. ~4× cache footprint reduction when it lands.
- **Per-session model pinning**: /cfa-A's session does NOT keep model X resident if /cfa-B's session requests Y — HotSwapManager arbitrates globally. Per-session pinning is v0.2.
- **Multi-model concurrent loading**: hf2q's HotSwapManager supports up to 3 models in pool by default, but /cfa v0.1 ships single-active-model-at-a-time. Multi-model concurrency = future.

## Scope impact (vs original §4.2 estimate)

The v0.1 release plan §4.2 (`examples/cfa-team-member.md` integration test) estimated ~80 LOC. With the singleton + server-side state architecture, /cfa-hf2q integration scope is:

| Layer | LOC | Notes |
|---|---|---|
| `src/serve/sessions.rs` + handler wiring | ~300 | new module |
| `src/serve/model_swap.rs` (POST /switch_model + GET /modelinfo) | ~200 | new module |
| Sessions + model_swap unit tests | ~150 | TDD-first |
| `--pidfile / --portfile / --idle-timeout-seconds` plumbing | ~50 | atomic write-on-/readyz, cleanup-on-graceful-shutdown |
| Idle watcher + TTL reaper tasks in `src/serve/mod.rs` | ~50 | tokio::spawn, 30s tick |
| `/shutdown` 409-aware refactor | ~30 | refcount check before existing b2d0cda path |
| `/metrics` two new gauges | ~20 | active_sessions + in_flight_requests |
| Model_swap concurrency tests (2 launchers) | ~80 | integration tests |
| `~/.claude/commands/cfa.md` lifecycle | ~120 | flock + register + heartbeat + DELETE + comments |
| Phase 4 §4.2 integration test (concurrent two-/cfa) | ~100 | covers attach-or-spawn race, model arbitration, last-out shutdown, TTL reap |
| **Total** | **~1,100** | vs §4.2's ~80 LOC estimate |

Working-day estimate in v0.1 release plan §4.2 ("1-2 working days") is now off by ~3-5×. v0.1 plan amendment is owed (likely 5-8 working days for /cfa-hf2q sub-phase alone, given Codex audit will be required for the `src/serve/` mutex + tokio::spawn additions per `feedback_codex_review_catches_unified_memory_races`).

## v0.1 ship checklist

### Sidecar (`src/serve/`)

- [ ] Implement `POST /sessions/register | /heartbeat | DELETE | GET /sessions`. ~300 LOC + ~150 LOC tests.
- [ ] Implement `POST /switch_model + GET /modelinfo`. ~200 LOC + ~80 LOC tests.
- [ ] Refactor `POST /shutdown` to 409-aware. ~30 LOC.
- [ ] Add `--pidfile / --portfile / --idle-timeout-seconds` flags + atomic-write-on-/readyz + cleanup-on-graceful-shutdown. ~50 LOC.
- [ ] Add idle watcher task (tokio::spawn, 30s tick) + TTL reaper task. ~50 LOC.
- [ ] Add `hf2q_serve_active_sessions` + `hf2q_serve_in_flight_requests` gauges to `/metrics`.
- [ ] Codex audit on the mutex + tokio::spawn additions before merge.

### cfa.md lifecycle (`~/.claude/commands/cfa.md`)

- [ ] Add "## hf2q sidecar (optional model backend)" section.
- [ ] Add Phase 0 discover-or-spawn-under-flock + register-session + heartbeat-bg-loop logic.
- [ ] Add Phase 5 deregister-session + heartbeat-kill logic. **No /shutdown call.**
- [ ] Add `CFA_HF2Q_DISABLE=1` short-circuit.
- [ ] Update Step 4 plan presentation to include "hf2q sidecar: <port> | disabled" line.
- [ ] Update fallbacks: "hf2q binary missing or `hf2q serve --version` fails → set `CFA_HF2Q_DISABLE=1` automatically and warn in plan."

### Phase 4 §4.2 integration test

- [ ] Two concurrent /cfa-launcher processes — verify they share the sidecar.
- [ ] Finishing one /cfa session does NOT shut down the sidecar while the other is registered.
- [ ] Last /cfa session detaches → idle-watcher fires within `idle_timeout` → sidecar exits gracefully.
- [ ] Crashed /cfa launcher (SIGKILL) — TTL reaper removes its session within `ttl_seconds`; sidecar continues serving the surviving /cfa.
- [ ] /cfa-A wants Gemma4, /cfa-B wants Qwen3.6 — model swap happens; both sessions get correct results.
- [ ] `POST /shutdown` while a session is registered returns 409 with body listing the active session.

### Documentation

- [ ] Update v0.1 release plan §4.2 LOC estimate to ~1,100; phase duration to 5-8 working days.
- [ ] Document the cross-session cache (`~/.cache/cfa/kv-persist/` 4 GiB LRU) in cfa.md preamble.
- [ ] Document the singleton sidecar shape in cfa.md preamble (concurrent-/cfa-friendly, OOM-bounded).
- [ ] Public-docs discipline: do NOT claim "R-P6 1.00× for /cfa" until E.a iter-3 lands.

## v0.2+ roadmap

- **E.a iter-2/2.5/3** — production R-P6 win for shared-prefix-different-suffix fan-outs.
- **B-hybrid** — Qwen 3.5 / Qwen 3-VL family support (post ADR-013).
- **B-tq** — TurboQuant cache codec (post ADR-007); ~4× cache footprint reduction.
- **Per-session model pinning** — /cfa-A's session keeps model X resident even if /cfa-B requests Y, modulo HotSwapManager pool capacity.
- **/cfa-aware queue** — priority for actively-running /cfa sessions over background batches.
- **Multi-model concurrent loading** — architect + coder workers running on different model sizes within one /cfa session.

## Cross-references

- ADR-017 Phase D Closed-Shipped + Phase E option (b) shipped (commits `4830353` ... `9beb906`)
- ADR-017 Phase E option (a) research dossier `docs/research/adr017-phase-e-option-a-2026-05-05.md` (~1,060 LOC scope, 7 iters; iter-1 `LcpRegistry` landed `9beb906`)
- `feedback_no_client_side_state_machines.md` (memory): why state authority lives server-side
- `feedback_codex_review_catches_unified_memory_races.md` (memory): why Codex audit is required for `src/serve/` mutex + tokio::spawn additions
- `mantra.txt`: no shortcuts, no fallback, no v0.2-deferred stubs
- `POST /shutdown` HTTP endpoint: `src/serve/api/handlers.rs::shutdown` (commit `b2d0cda`; v0.1 makes it 409-aware)
- `drain_loaded_models_to_disk` graceful-drain helper: `src/serve/mod.rs` (commit `b2d0cda`)
- `HF2Q_KV_PERSIST_BUDGET_BYTES`: `src/serve/mod.rs:2887-2902` (commit `c2eeecd`)
- R-F5 LRU eviction: `src/serve/kv_persist/writer.rs::process_job` (commit `c2eeecd`)
- ADR-017 R-F1 ("Default OFF until Phase D ships"): preserved at global level
