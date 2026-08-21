# Get started with hf2q, Qwen3.8, OpenCode, and Agentic Kit

There is one supported end-to-end setup for this model:

**[hf2q + Qwen3.8 + AK + search/fetch: complete setup](hf2q+qwen3.8+ak+search-fetch-setup.md)**

Follow that guide from beginning to end. It:

- installs and configures hf2q;
- downloads the exact published hf2q Q4_K_M text GGUF and its matching F16
  projector from an immutable Hugging Face revision;
- verifies both SHA-256 hashes;
- serves the pair with `--mmproj` and explicit qualified runtime settings;
- proves unary, SSE, and real image generation before opening OpenCode;
- installs full Agentic Kit with `ak setup` and `ak setup --opencode`;
- preserves stock OpenCode Build, Bash, file, task, skill, and MCP tools;
- installs and proves search, fetch, crawl, and extraction; and
- provides disable, re-enable, status, server-stop, and uninstall cleanup.

This page intentionally does not define a second, reduced setup. In
particular, do not substitute an on-the-fly source conversion, a text-only
server, the older Q5 artifact, or a minimal/tool-disabled OpenCode agent for
the tested path above.

For general hf2q conversion documentation unrelated to this qualified setup,
see [Converting a model](converting-a-model.md).
