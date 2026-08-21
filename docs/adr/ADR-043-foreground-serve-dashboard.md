# ADR-043: Foreground serve dashboard

- **Status:** Accepted 0.1.5 candidate; immutable main/package publication pending
- **Date:** 2026-08-09
- **Related:** ADR-040, ADR-042

## Context

Foreground serving currently prints one INFO record per bounded prefill
transaction. A 99,007-token Qwen request produced 49 near-identical lines; a
107,045-token DeepSeek request did the same while the only actionable fact—a
short peer receiving one token every 6–7 seconds—was difficult to see. The log
stream is useful as an evidence ledger, but it is a poor live operator surface.

The server must still preserve plain logs for launchd, CI, pipes, JSON ingest,
and incident receipts. Rendering must never backpressure the single model
worker, and the UI must not expose prompt, tool, path, or generated content.

## Decision

`hf2q serve` gains `--operator-ui auto|dashboard|plain`, defaulting to `auto`.

- `auto` enables a stable alternate-screen dashboard only when text stderr is
  an interactive terminal and the process is not running under CI.
- `dashboard` requires that environment and fails before model loading when it
  is unavailable or JSON logging is selected.
- `plain` always retains the traditional tracing stream. Non-TTY, CI, and JSON
  output remain plain automatically.

Family workers publish lifecycle/progress values through a bounded
`try_send`-only channel. A full or slow UI drops presentation events rather
than delaying scheduler, cache, or GPU work. The tracing writer sends complete
runtime log records to a small recent-events pane while the dashboard is
active; it preserves normal stderr before startup and whenever the dashboard
is absent or disconnected.

Each request row contains only operational metadata:

- request/slot identity and family-owned execution mode;
- queued, prefill, decode, complete, cancelled, or failed phase;
- prompt tokens, verified cached tokens, and new prefill work;
- prefill progress, percentage, rate, and ETA;
- generated tokens versus the maximum budget, decode rate, and elapsed time.

Decode does not display a false “percent remaining”: EOS is unknown, so
`max_tokens` is shown explicitly as a budget. A poisoned engine replaces the
green ready state with an unhealthy/restart-required status using the same
one-way supervisor signal that backs `/readyz`.

## Acceptance

Before acceptance and release:

1. pure state/render tests must bind cached suffix progress, percentages,
   clamping, completion retention, and unhealthy status;
2. CLI tests must prove `auto` stays plain for pipes/JSON and explicit
   dashboard mode fails before model load on unsupported terminals;
3. a pseudo-terminal integration run must prove stable redraw and terminal
   restoration after normal shutdown and SIGINT;
4. Qwen, Gemma, and DeepSeek real-model gates must show correct cached/new
   counts and progress without changing response bytes, scheduler accounting,
   TTFT, or throughput beyond measurement noise;
5. plain logs and release receipts remain byte-parseable by the existing
   checked-in harnesses.

Until these gates pass on an immutable packed artifact, this ADR describes a
candidate implementation rather than released availability.

## Candidate validation (2026-08-09)

The exact 0.1.5 source-candidate binary
`cd8867820898eb33beb5523894084ed5af5a8cdbba92c4aaa8ca4bbb48150784`
passed the following pre-publication gates:

- six pure dashboard state/render tests, including cached/new prefill
  accounting, percentage/ETA rendering, completion retention, and unhealthy
  state;
- a forced `xterm-256color` pseudo-terminal run against the production Qwen
  launcher, proving alternate-screen entry, stable redraw, a real validated
  552-token SSE request, visible lifecycle events, clean SIGINT shutdown, and
  cursor/alternate-screen restoration;
- the plain-mode Qwen exact overlap/cache/cancellation/cumulative gates,
  including 87,965 cached tokens on the tool-result continuation and flat
  CFString, autorelease-pool, and command-buffer populations across two
  measured four-agent waves;
- the DeepSeek public 94,576-token/347-tool overlap, where the short lane
  generated 49 tokens during the first reporting window under a 256-token
  mixed prefill slice; and
- the Gemma agentic gate, which reused 8,691/8,698 tokens and preserved unary,
  SSE, tool-result, and source-response semantics.

The full hosted-safe and family source filters passed with Qwen 689/0, Gemma
415/0, DeepSeek 95/0 (three hardware-only ignores), dashboard 6/0, and zero
failures. Plain log receipts remained parseable throughout. Publication is
still fail-closed on the final commit, exact-main CI, packed-crate tests, and
registry-byte verification; those state gates do not reopen this dashboard
design unless the immutable artifact differs.
