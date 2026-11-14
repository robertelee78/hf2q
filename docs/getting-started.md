# Getting started: hf2q + omp

Run Qwen3.8 locally, then code with [omp](https://omp.sh). One model server,
four inference slots, parallel agents.

**Before you start:** Apple Silicon, macOS 14 or newer, and at least 20 GiB
free for the roughly 17 GiB first download. The documented validation host
has 128 GiB unified memory; this is not a minimum-memory guarantee. Four
long conversations need more memory than one.

## 1. Install hf2q

```bash
curl -fsSL https://hf2q.us/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
hf2q setup --accept-defaults \
  --serve-host 127.0.0.1 --serve-port 8081 \
  --serve-scheduler inflight-batched --serve-max-slots 4
hf2q doctor
```

This saves a private, four-slot configuration, including on a setup rerun.
Read any failures reported by `doctor` before continuing.

## 2. Start the model · terminal 1

```bash
hf2q serve jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M
```

hf2q downloads the compatible model if needed and reuses it on later starts.
Wait until the server is listening on `127.0.0.1:8081`. Leave this terminal
open.

The four slots share model weights, not a divided context window.

## 3. Try a conversation · terminal 2

Open a new terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
hf2q chat --url http://127.0.0.1:8081
```

Ask “What is 12 × 13?” Wait for the answer, then type `/quit`.
Keep terminal 1 running for the remaining steps.

To inspect downloaded models later, use `hf2q serve list`.

## 4. Install omp

```bash
curl -fsSL https://omp.sh/install | sh
```

No separate Node.js, Bun, Agentic Kit, or research services to install.

## 5. Connect omp to hf2q

```bash
curl -fsSL https://hf2q.us/omp.sh | sh
```

This creates a separate `hf2q` profile and the `omp-hf2q` launcher. It
detects the loaded Qwen model and its context window, routes coding roles
to it, runs task batches with up to four workers, and caps hf2q requests
at four. Workers cannot spawn nested workers.
Your default omp profile stays unchanged; changed hf2q-profile files are
backed up.

## 6. Start coding

From your project directory:

```bash
omp-hf2q
```

Try:

> Use four parallel read-only agents to inspect this project: structure,
> dependencies, tests, and documentation. Have each use file or shell tools,
> then combine their findings. Do not change files.

Look for four workers and actual tool results, not just a claim that work
was delegated. Four is the request ceiling, not a promise that every task
keeps every slot busy. omp includes web search and URL reading; no separate
search-stack setup is needed.

Next time, repeat step 2 in one terminal and run `omp-hf2q` in another.
Stop the server with **Ctrl-C** in terminal 1.

## If something stops you

- **Cannot connect:** check terminal 1. The guide uses `127.0.0.1:8081`
  throughout; don't start a second model server.
- **Command not found:** run `export PATH="$HOME/.local/bin:$PATH"` in that
  terminal.
- **Changed the served Qwen model or context limit:** rerun step 5 to refresh the
  omp profile.
- **Memory pressure:** shorten conversations or choose a smaller model.
  Four full-context conversations are not a hardware-independent guarantee.

For advanced serving controls, see [setup](setup.md) and
[GCD / GLP](gcd-in-hf2q.md).
