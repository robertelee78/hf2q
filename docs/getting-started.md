# Getting started: hf2q + omp

Run official Qwen3.8 locally with GLP, GCD, and PhantomKV, then code with
[omp](https://omp.sh). One model server, four inference slots, parallel agents.

**Before you start:** Apple Silicon, macOS 14 or newer, and 80 GiB free
for the first source download and conversion. Validation used 128 GiB
unified memory; four long conversations need more memory than one.

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

The first run downloads and converts the exact official checkpoint that
matches the GLP vector. hf2q handles downloading and caching:

```bash
hf2q convert Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --quant q4_k_m --text-only \
  --output "$HOME/.local/share/hf2q/models/qwen3.8.gguf"
```

Download the small PhantomKV bank once:

```bash
curl -fL https://huggingface.co/jenerallee78/Qwen3.8-27B-hf2q-Q4_K_M/resolve/840dc4196706714939679630d137b04fb33c0c9f/Qwen3.8-27B-phantom-kv.gguf \
  -o "$HOME/.local/share/hf2q/models/qwen3.8-phantom-kv.gguf"
```

Start the server. Reuse this command next time:

```bash
hf2q serve "$HOME/.local/share/hf2q/models/qwen3.8.gguf" \
  --gcd \
  --glp https://huggingface.co/msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49/resolve/5f1be31cfde653f24fe74c71047433012b8ede9e/Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf \
  --kv-graft "$HOME/.local/share/hf2q/models/qwen3.8-phantom-kv.gguf"
```

These runtime modifiers target refusal suppression without editing model
weights. Omit the three flags for stock behavior. The setup is text-only.

Wait until the server is listening on `127.0.0.1:8081`. Leave this terminal
open.

## 3. Try a conversation · terminal 2

Open a new terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
hf2q chat --url http://127.0.0.1:8081
```

Ask “What is 12 × 13?” Wait for the answer, then type `/quit`.
Keep terminal 1 running for the remaining steps.

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

omp requests normal text so the default GCD prose grammar cannot block
tools. Native tool-call grammar stays active; omp's free-form prose is
not grammar-constrained.

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

Next time, run the serve command in step 2, then `omp-hf2q` in another terminal.
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
