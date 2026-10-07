# hf2q QE playbook

Hands-on quality engineering: use hf2q the way a person does and write down
what happened. ADR-061 D2 governs it.

The key words MUST, MUST NOT, SHOULD, and MAY are to be interpreted as
described in RFC 2119.

- This is **ad hoc**. An agent runs it when the owner asks ("run the QE pass
  on this build"), against the build the owner names. It is **not** a release
  step and MUST NOT be wired into CI, a workflow, or a required check.
- It is not a test suite. Unit tests and batteries already exist. The point
  is the experience: does the product do what the guide says, does it feel
  right, does anything hang, confuse, or lie.
- Record everything in `docs/qe/<label>.md` (template at the end), where
  `<label>` is the version or a short build name such as `0.1.21` or
  `main-4e705f2c`.

## Before you start

1. **Name the build.** Record `hf2q --version`, the binary path, and its
   SHA-256 (`shasum -a 256 <path>`). For a standalone release use the signed
   binary from the GitHub release, not a local build.
2. **Check the machine.** No other `hf2q`, `llama-server`, or `llama-cli`
   process (`pgrep -fl 'hf2q|llama-'`), and enough free memory for the model
   (`vm_stat`; about 1.5x the GGUF size). Load one model at a time.
3. **Isolate state.** Use a scratch home so the run cannot touch the owner's
   configuration:

   ```bash
   export QE=$(mktemp -d /private/tmp/hf2q-qe.XXXXXX)
   export QE_HOME=$QE/home; mkdir -p "$QE_HOME"
   # run hf2q with HOME="$QE_HOME" unless a journey says otherwise
   ```

   Use `/private/tmp`, not `/tmp`: `/tmp` is a symlink and `hf2q setup`
   refuses a home path that contains one.

   Model files MAY be reused from the owner's managed directory
   (`~/.local/share/hf2q/models/`) by passing their absolute path.
4. **Pick the models.** The default text and vision model is the guide pair:
   `jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M` (GGUF plus its mmproj).
   Record the exact GGUF path and SHA-256 for every model used.
5. **Clean up.** Every server you start MUST be stopped before the next
   journey and at the end (`kill` the PID you recorded; confirm with `pgrep`).

## Driving the interactive chat

`hf2q chat` is an interactive terminal program. Drive it in tmux so you can
type, wait, and read the screen like a person:

```bash
tmux new-session -d -s qe -x 200 -y 50
tmux send-keys -t qe "HOME=$QE_HOME hf2q chat --url http://127.0.0.1:8081" Enter
tmux send-keys -t qe "hi" Enter
sleep 5; tmux capture-pane -t qe -p | tail -40
```

Poll `capture-pane` every few seconds while a reply streams and note when the
text stops changing. Save the final pane text for the record.

To measure streaming objectively, use the API with per-chunk timestamps:

```bash
curl -sN http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<id>","stream":true,"max_tokens":768,"messages":[{"role":"user","content":"Write a detailed 600-word explanation of how TCP congestion control works."}]}' \
  | while IFS= read -r line; do printf '%s %s\n' "$(python3 -c 'import time;print(f"{time.time():.3f}")')" "$line"; done > "$QE/stream.log"
```

The largest gap between consecutive `data:` lines is the longest silence.

## Journeys

Run them in order unless the owner asks for a subset. Each has a pass bar.
"Fail" includes hangs, crashes, wrong answers, misleading messages, and
documentation that does not match behavior.

### J1 Install

Fresh scratch home. For a published release:
`HOME=$QE_HOME sh -c 'curl -fsSL https://hf2q.us/install.sh | sh'`.
For an unpublished candidate, use the release's rendered `install.sh` in
installer test mode against a local copy of the assets:
`HF2Q_INSTALL_TEST_MODE=1 HF2Q_RELEASE_BASE_URL=file:///path/to/assets`.
Then `hf2q setup --accept-defaults`, `hf2q doctor`, `hf2q --version`.

Pass: installs without errors, `doctor` is healthy, version is the expected
one, shell completion files exist.

### J2 Chat

Start the server (`hf2q serve <model>`) and wait for the ready message. In
`hf2q chat`:

1. A first message under sixteen tokens ("hi").
2. A conversation of at least ten turns that refers back to earlier turns.
3. One request for a long answer (512+ tokens).
4. `/status`, then `/quit`.

Also run the timestamped streaming request above.

Pass: every turn completes or errors cleanly within ten minutes; no silence
over sixty seconds between tokens; answers are coherent and remember context.
Record time to first token and decode rate (from `/status` or the stream
log).

### J3 Agentic coding

Configure OpenCode as in `docs/getting-started.md` section 7 (back up the
config first; restore it after). In a scratch git repo, ask for a small real
task, for example: "Add a function `slugify(s)` to `util.py` with three
tests, run the tests, and fix any failure."

Pass: the transcript shows real tool calls with correct tools and arguments,
the results are used, the task is completed, and follow-up turns reuse the
prompt cache: `usage.prompt_tokens_details.cached_tokens` on later turns
covers most of the prompt instead of recomputing it (watch the server log or
replay a turn with `curl` to read the usage block).

### J4 API

One unary and one streaming `/v1/chat/completions` request with `curl`.

Pass: the unary response has `choices[0].message.content` and `usage`; the
stream ends with `data: [DONE]` and its chunks reassemble to a sensible
answer.

### J5 Convert

Convert a small Hugging Face source model from a supported family and use
it (plain `qwen3` is not a supported conversion architecture; Qwen3.5 is).
The 2B model converts in about a minute and also produces an mmproj:

```bash
HOME=$QE_HOME hf2q convert Qwen/Qwen3.5-2B --quant q4_k_m --output $QE/qwen35-2b-q4km.gguf
HOME=$QE_HOME hf2q serve $QE/qwen35-2b-q4km.gguf
```

Ask three plain factual questions in `hf2q chat` (capital of France, 12 x 12,
what is HTTP).

Pass: conversion completes; answers are correct and fluent. Emitting tokens
is not a pass.

### J6 GCD

Restart the server with each of:

- `--gcd`
- `--gcd-schema examples/recon-opportunities.schema.json`
- `--gcd-schema examples/recon-opportunities.schema.json --gcd-schema-locked`

For each: one `hf2q chat` turn and one `curl` request. Under the locked
schema also send a request with tool definitions and `tool_choice: "auto"`
(expected: rejected before streaming) and one with `tool_choice: "none"`
(expected: accepted).

Pass: output follows the grammar or schema the README describes; documented
rejections happen with a clear message; nothing hangs.

### J7 GLP

Needs an artifact that matches the served checkpoint exactly. No GLP
artifact is published for the guide's abliterated SFT checkpoint, so use the
stock model and its published artifact:

```bash
HOME=$QE_HOME hf2q serve Qwen/Qwen3.8-27B:Q4_K_M --glp msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49
```

If no stock Q4_K_M is available, convert one first with `hf2q convert
Qwen/Qwen3.8-27B --quant q4_k_m`. Chat with it; restart with `--glp-alpha 0.5`;
then restart without `--glp`. A bind refusal against the abliterated SFT
checkpoint is expected behavior, not a finding.

Pass: loads or fails with the documented typed error; outputs change when
steering is on; returning to baseline restores baseline behavior.

### J8 Update and uninstall

On a scratch home holding the previous standalone release:
`hf2q update --check`, `hf2q update`, `hf2q --version`,
`hf2q update --rollback`, `hf2q --version`, `hf2q update` again, then
`hf2q uninstall --yes`.

Pass: each step does what it says; config and model data survive uninstall.

### J9 Vision

Run the red-image check from `docs/getting-started.md` section 5 against the
served guide pair. `hf2q chat` has no image or attachment input, so vision is
exercised through the API only; record that gap in the record.

Pass: the reply says red.

## GLP and GCD deep pass

When the owner asks for it, go beyond J6 and J7 and cover every documented
path: `--gcd` (and the hidden `--uncensor` alias), `--gcd-schema`,
`--gcd-schema-locked`, `--glp` with a local path, `--glp` with a Hub
repository or `huggingface.co` file URL, bare `--glp` discovery,
`--glp-alpha`, `--gcd --glp` together, `hf2q calibrate` (DeepSeek-V4 only),
and returning to baseline; through chat, the API, and OpenCode; on Qwen
3.5/3.6/3.8 and DeepSeek-V4 where the docs claim support.

Expected, not findings: a GLP bind refusal for a mismatched checkpoint, site,
or width; a bare `--glp` ambiguity rejection; chat GCD/GLP flags having no
effect on a chat attached with `--url`.

## Record template

```markdown
# QE pass: <label>

- Date, operator (agent and model):
- Binary: path, `hf2q --version`, SHA-256
- Machine: chip, memory, macOS version
- Models: path and SHA-256 for each

| Journey | Result | Notes |
|---|---|---|
| J1 Install | pass/fail/skipped | |
| ... | | |

## Findings

1. What happened, how to reproduce, expected vs actual, severity, issue link.

## Impressions

Anything that felt slow, confusing, or wrong even if it did not fail.
```
