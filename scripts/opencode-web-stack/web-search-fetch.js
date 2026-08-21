/**
 * web-search-fetch — local SearXNG search + Crawl4AI fetch tools for opencode.
 *
 * Backends (autostart via launchd, loopback only, $0 / no API keys):
 *   SearXNG   http://127.0.0.1:8888  (metasearch over ~70 engines, JSON API)
 *   Crawl4AI  http://127.0.0.1:11235 (headless-Chromium markdown fetch, static fast-path)
 *
 * Tools:
 *   web_search  — canonical search front door; searches, then reads top pages
 *   web_fetch   — fetches one user-supplied or search-discovered URL
 *   web_crawl   — bounded multi-page crawl with domain/relevance filters
 *   web_extract — structured CSS or semantic extraction from one URL
 *
 * Permission defaults are injected via the config hook so these tools do not
 * prompt. The built-in fetch-only `webfetch` is removed from the catalog so a
 * search request cannot degrade into guessed URLs; the replacement `web_fetch`
 * retains and expands that capability. Bash, files, tasks, skills, and MCP
 * tools remain untouched.
 */

import { tool } from "@opencode-ai/plugin";

const SEARXNG = process.env.OPENCODE_SEARXNG_URL || "http://127.0.0.1:8888";
const FETCH = process.env.OPENCODE_FETCH_URL || "http://127.0.0.1:11235";

const FETCH_TIMEOUT_MS = 150_000;
const SEARCH_TIMEOUT_MS = 20_000;

// Known junk/spam/clickbait domains filtered from search results.
const JUNK_DOMAINS = new Set([
  "99designs.com",
  "anyrgb.com",
  "boredpanda.com",
  "buzzfeed.com",
  "clickbait.com",
  "dailymail.co.uk",
  "digg.com",
  "eonline.com",
  "eskipaper.com",
  "fanpop.com",
  "fotofacil.com.br",
  "grabcad.com",
  "hobbylark.com",
  "pinterest.com",
  "pinterest.co.uk",
  "promopanda.com",
  "quotefancy.com",
  "redbubble.com",
  "shutterstock.com",
  "slideshare.net",
  "spongebob.com",
  "taringa.net",
  "teepublic.com",
  "thefamouspeople.com",
  "themogh.org",
  "wallpaperaccess.com",
  "wallpaperflare.com",
  "wattpad.com",
]);

const BETTER_ENGINES = {
  general: "google,duckduckgo,mojeek,yahoo",
  academic: "arxiv,google scholar,semantic scholar,pubmed,crossref,openalex",
  tech: "github,stackoverflow,mdn,docker hub,arch linux wiki,gentoo",
  news: "google,duckduckgo,mastodon hashtags",
  wikidata: "wikipedia,wikidata,wiktionary",
};

function isJunkUrl(url) {
  try {
    return JUNK_DOMAINS.has(new URL(url).hostname.replace(/^www\./, ""));
  } catch {
    return false;
  }
}

function dedupByUrl(results) {
  const seen = new Set();
  const out = [];
  for (const r of results) {
    let key;
    try {
      key = new URL(r.url).hostname + new URL(r.url).pathname;
    } catch {
      key = r.url;
    }
    if (!seen.has(key)) {
      seen.add(key);
      out.push(r);
    }
  }
  return out;
}

async function postJSON(url, body, timeoutMs) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${url}`);
  return res.json();
}

function clip(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + " …" : s;
}

function assertHttpUrl(url) {
  const parsed = new URL(url);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`URL must use http or https: ${url}`);
  }
}

function commaList(value) {
  if (!value) return undefined;
  const values = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return values.length ? values : undefined;
}

async function searchExecute(args) {
  const pages = Math.min(Math.max(Math.trunc(args.pages ?? 2), 1), 3);
  const excerpt = Math.min(Math.max(Math.trunc(args.excerpt_chars ?? 4000), 500), 8000);
  try {
    const engines = args.engines || BETTER_ENGINES[args.category] || BETTER_ENGINES.general;
    const data = await searx(args.query, {
      time_range: args.time_range,
      engines,
      language: args.language,
    });
    const results = dedupByUrl(data.results || [])
      .filter((r) => !isJunkUrl(r.url))
      .slice(0, pages);
    if (!results.length) return `No search results for: ${args.query}`;
    const sections = await Promise.all(
      results.map(async (result, index) => {
        const engines = (result.engines || [result.engine]).filter(Boolean).join(",");
        const date = result.publishedDate ? ` (${result.publishedDate.slice(0, 10)})` : "";
        const searchEvidence = clip(result.content, 500);
        try {
          const page = await fetchPage(result.url, { mode: "auto", max_chars: excerpt });
          const body = page.ok ? clip(page.markdown, excerpt) : `(page read failed: ${page.error})`;
          return `## Source ${index + 1}: ${result.title}${date}\n${result.url} [${engines}]\n\nSearch excerpt: ${searchEvidence}\n\n${body}`;
        } catch (error) {
          return `## Source ${index + 1}: ${result.title}${date}\n${result.url} [${engines}]\n\nSearch excerpt: ${searchEvidence}\n\n(page read failed: ${error.message})`;
        }
      }),
    );
    return `# Web research: "${args.query}"\n\n${sections.join("\n\n---\n\n")}`;
  } catch (error) {
    return `WEB_SEARCH_FAILED: ${error.message}. Do not guess URLs or retry with web_fetch.`;
  }
}

async function fetchExecute(args) {
  try {
    assertHttpUrl(args.url);
    const maxChars = Math.min(Math.max(Math.trunc(args.max_chars ?? 12000), 500), 40000);
    const page = await fetchPage(args.url, { ...args, max_chars: maxChars });
    if (!page.ok) return `WEB_FETCH_FAILED for ${args.url}: ${page.error || "empty content"}`;
    const head = `# ${page.title || page.url}\n${page.url}  (via ${page.via}${page.truncated ? ", truncated" : ""})\n\n`;
    return head + (page.markdown || "(no text content extracted)");
  } catch (error) {
    return `WEB_FETCH_FAILED: ${error.message}`;
  }
}

async function crawlExecute(args) {
  try {
    assertHttpUrl(args.url);
    const maxDepth = Math.min(Math.max(Math.trunc(args.max_depth ?? 2), 1), 5);
    const maxPages = Math.min(Math.max(Math.trunc(args.max_pages ?? 6), 1), 20);
    const maxChars = Math.min(Math.max(Math.trunc(args.max_chars ?? 6000), 500), 20000);
    const result = await postJSON(
      `${FETCH}/crawl`,
      {
        url: args.url,
        max_depth: maxDepth,
        max_pages: maxPages,
        allowed_domains: commaList(args.allowed_domains),
        blocked_domains: commaList(args.blocked_domains),
        include_external: args.include_external ?? false,
        query: args.query,
        max_chars: maxChars,
      },
      FETCH_TIMEOUT_MS,
    );
    if (!result.ok) return `WEB_CRAWL_FAILED for ${args.url}: ${result.error || "crawl failed"}`;
    const pages = (result.pages || []).map(
      (page, index) =>
        `## Page ${index + 1}: ${page.title || page.url}\n${page.url} (depth ${page.depth ?? "?"})\n\n${page.markdown || "(no text extracted)"}`,
    );
    return `# Web crawl: ${args.url}\nCrawled ${result.crawled ?? pages.length} page(s).\n\n${pages.join("\n\n---\n\n")}`;
  } catch (error) {
    return `WEB_CRAWL_FAILED: ${error.message}`;
  }
}

export function normalizeJsonCssSchema(schema) {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return schema;

  const normalizeField = (name, value) => {
    if (typeof value === "string") return { name, selector: value, type: "text" };
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return { name, type: "text", ...value };
    }
    return value;
  };

  if (schema.fields && !Array.isArray(schema.fields) && typeof schema.fields === "object") {
    return {
      ...schema,
      baseSelector: schema.baseSelector || "body",
      fields: Object.entries(schema.fields).map(([name, value]) => normalizeField(name, value)),
    };
  }
  if (Array.isArray(schema.fields)) {
    return { ...schema, baseSelector: schema.baseSelector || "body" };
  }

  // Accept the compact shape models naturally emit, e.g. {"heading":"h1"}.
  // Reserved JSON-CSS metadata remains metadata; all other keys become fields.
  const reserved = new Set(["name", "baseSelector", "baseFields"]);
  const entries = Object.entries(schema).filter(([name]) => !reserved.has(name));
  if (entries.length && entries.every(([, value]) => typeof value === "string" || (value && typeof value === "object"))) {
    return {
      name: schema.name,
      baseSelector: schema.baseSelector || "body",
      baseFields: schema.baseFields,
      fields: entries.map(([name, value]) => normalizeField(name, value)),
    };
  }
  return schema;
}

async function extractExecute(args) {
  try {
    assertHttpUrl(args.url);
    let schema;
    if (args.schema_json) schema = JSON.parse(args.schema_json);
    // Qwen-family tool callers occasionally place a JSON-CSS schema in the
    // adjacent `query` field. Accept that unambiguous shape instead of burning
    // a failed tool round-trip; ordinary semantic queries remain untouched.
    if (!schema && (args.strategy ?? "json_css") === "json_css" && args.query?.trim().startsWith("{")) {
      schema = JSON.parse(args.query);
    }
    schema = normalizeJsonCssSchema(schema);
    if ((args.strategy ?? "json_css") === "json_css" && !schema) {
      return "WEB_EXTRACT_REJECTED: json_css requires schema_json";
    }
    const result = await postJSON(
      `${FETCH}/extract`,
      {
        url: args.url,
        strategy: args.strategy ?? "json_css",
        schema,
        query: args.query,
        max_chars: Math.min(Math.max(Math.trunc(args.max_chars ?? 40000), 500), 100000),
      },
      FETCH_TIMEOUT_MS,
    );
    if (!result.ok) return `WEB_EXTRACT_FAILED for ${args.url}: ${result.error || "extraction failed"}`;
    return `# Web extraction: ${args.url}\n\n${
      typeof result.data === "string" ? result.data : JSON.stringify(result.data, null, 2)
    }`;
  } catch (error) {
    return `WEB_EXTRACT_FAILED: ${error.message}`;
  }
}

async function searx(query, opts = {}) {
  const params = new URLSearchParams({ q: query, format: "json" });
  if (opts.categories) params.set("categories", opts.categories);
  if (opts.engines) params.set("engines", opts.engines);
  if (opts.time_range) params.set("time_range", opts.time_range);
  if (opts.language) params.set("language", opts.language);
  const res = await fetch(`${SEARXNG}/search?${params}`, {
    signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} from SearXNG`);
  const data = await res.json();
  // If the requested engines were all rate-limited and returned nothing,
  // retry once with the broad engine set before giving up.
  if (!(data.results || []).length && opts.engines && opts.engines !== BETTER_ENGINES.general) {
    params.delete("engines");
    const retry = await fetch(`${SEARXNG}/search?${params}`, {
      signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS),
    });
    if (retry.ok) return retry.json();
  }
  return data;
}

async function fetchPage(url, opts = {}) {
  return postJSON(
    `${FETCH}/fetch`,
    {
      url,
      mode: opts.mode || "auto",
      css_selector: opts.css_selector,
      max_chars: opts.max_chars,
    },
    FETCH_TIMEOUT_MS,
  );
}

/** @type {import("@opencode-ai/plugin").Plugin} */
export default async function webSearchFetch() {
  return {
    config(cfg) {
      cfg.tools = {
        ...(cfg.tools || {}),
        webfetch: false,
      };
      cfg.permission = {
        web_search: "allow",
        web_fetch: "allow",
        web_crawl: "allow",
        web_extract: "allow",
        WebSearch: "allow",
        WebFetch: "allow",
        WebCrawl: "allow",
        WebExtract: "allow",
        ...(cfg.permission || {}),
        webfetch: "deny",
      };
    },

    tool: {
      web_search: tool({
        description:
          "Canonical front door for every request to search, research, find, look up, or get current web information. " +
          "Searches via local SearXNG, then reads the top pages in parallel via local Crawl4AI. " +
          "Always call this before web_fetch; never guess a URL. One call should normally complete the research.",
        args: {
          query: tool.schema.string().describe("The search or research query"),
          category: tool.schema
            .enum(Object.keys(BETTER_ENGINES))
            .optional()
            .describe("Engine set: general, academic, tech, news, or wikidata"),
          engines: tool.schema
            .string()
            .optional()
            .describe("Explicit comma-separated engine list (overrides category)"),
          language: tool.schema
            .string()
            .optional()
            .describe("Search language code, e.g. 'en'"),
          pages: tool.schema.number().optional().describe("Top pages to read in parallel (default 2, max 3)"),
          time_range: tool.schema
            .enum(["day", "week", "month", "year"])
            .optional()
            .describe("Optional recency window"),
          excerpt_chars: tool.schema
            .number()
            .optional()
            .describe("Content characters per page (default 4000, range 500-8000)"),
        },
        execute: searchExecute,
      }),

      web_fetch: tool({
        description:
          "Fetch one URL supplied by the user or returned by web_search. Never invent or guess a URL. " +
          "Returns clean LLM-ready markdown via local Crawl4AI. " +
          "mode='auto' (default) uses static, then Chromium, then a Cloudflare-aware stealth browser only when needed.",
        args: {
          url: tool.schema.string().describe("Exact URL from the user or web_search"),
          mode: tool.schema
            .enum(["auto", "static", "browser", "stealth"])
            .optional()
            .describe("Fetch strategy (default: auto)"),
          css_selector: tool.schema
            .string()
            .optional()
            .describe("Optional CSS selector, such as 'article' or '.docs-content'"),
          max_chars: tool.schema
            .number()
            .optional()
            .describe("Maximum markdown characters (default 12000, range 500-40000)"),
        },
        execute: fetchExecute,
      }),

      web_crawl: tool({
        description:
          "Crawl multiple pages starting at one exact URL. Use after web_search when a task needs a bounded section of a site, " +
          "not for one-page reading. Supports domain and relevance filters and returns clean markdown from each page.",
        args: {
          url: tool.schema.string().describe("Exact starting URL from the user or web_search"),
          max_depth: tool.schema.number().optional().describe("Link depth (default 2, max 5)"),
          max_pages: tool.schema.number().optional().describe("Page limit (default 6, max 20)"),
          allowed_domains: tool.schema.string().optional().describe("Optional comma-separated domain allowlist"),
          blocked_domains: tool.schema.string().optional().describe("Optional comma-separated domain blocklist"),
          include_external: tool.schema.boolean().optional().describe("Follow external links (default false)"),
          query: tool.schema.string().optional().describe("Optional semantic relevance query"),
          max_chars: tool.schema.number().optional().describe("Characters retained per page (default 6000)"),
        },
        execute: crawlExecute,
      }),

      web_extract: tool({
        description:
          "Extract structured data from one exact URL. Use json_css with schema_json, for example " +
          "{\"baseSelector\":\"body\",\"fields\":[{\"name\":\"heading\",\"selector\":\"h1\",\"type\":\"text\"}]}; " +
          "or use cosine with a semantic query.",
        args: {
          url: tool.schema.string().describe("Exact URL from the user or web_search"),
          strategy: tool.schema.enum(["json_css", "cosine"]).optional().describe("Extraction strategy; defaults to json_css"),
          schema_json: tool.schema
            .string()
            .optional()
            .describe("Required for json_css: put the JSON-CSS schema string here, never in query"),
          query: tool.schema
            .string()
            .optional()
            .describe("Only for cosine: semantic filter text; never put JSON schema here"),
          max_chars: tool.schema.number().optional().describe("Maximum extracted characters"),
        },
        execute: extractExecute,
      }),

      WebSearch: tool({
        description:
          "Alias of web_search. Canonical front door for every request to search, research, find, look up, or get " +
          "current web information. Searches via local SearXNG, then reads the top pages in parallel via local " +
          "Crawl4AI. Always call this before WebFetch; never guess a URL.",
        args: {
          query: tool.schema.string().describe("The search or research query"),
          category: tool.schema
            .enum(Object.keys(BETTER_ENGINES))
            .optional()
            .describe("Engine set: general, academic, tech, news, or wikidata"),
          engines: tool.schema
            .string()
            .optional()
            .describe("Explicit comma-separated engine list (overrides category)"),
          language: tool.schema
            .string()
            .optional()
            .describe("Search language code, e.g. 'en'"),
          pages: tool.schema.number().optional().describe("Top pages to read in parallel (default 2, max 3)"),
          time_range: tool.schema
            .enum(["day", "week", "month", "year"])
            .optional()
            .describe("Optional recency window"),
          excerpt_chars: tool.schema
            .number()
            .optional()
            .describe("Content characters per page (default 4000, range 500-8000)"),
        },
        execute: searchExecute,
      }),

      WebFetch: tool({
        description:
          "Alias of web_fetch. Fetch one URL supplied by the user or returned by WebSearch. Never invent or guess a URL. " +
          "Returns clean LLM-ready markdown via the local fetch stack. mode='auto' (default) uses static, then " +
          "Chromium, then a Cloudflare-aware stealth browser only when needed.",
        args: {
          url: tool.schema.string().describe("Exact URL from the user or WebSearch"),
          mode: tool.schema
            .enum(["auto", "static", "browser", "stealth"])
            .optional()
            .describe("Fetch strategy (default: auto)"),
          css_selector: tool.schema
            .string()
            .optional()
            .describe("Optional CSS selector, such as 'article' or '.docs-content'"),
          max_chars: tool.schema
            .number()
            .optional()
            .describe("Maximum markdown characters (default 12000, range 500-40000)"),
        },
        execute: fetchExecute,
      }),

      WebCrawl: tool({
        description: "Alias of web_crawl for Ruflo workflows. Bounded multi-page crawl from one exact URL.",
        args: {
          url: tool.schema.string().describe("Exact starting URL from the user or WebSearch"),
          max_depth: tool.schema.number().optional().describe("Link depth (default 2, max 5)"),
          max_pages: tool.schema.number().optional().describe("Page limit (default 6, max 20)"),
          allowed_domains: tool.schema.string().optional().describe("Optional comma-separated domain allowlist"),
          blocked_domains: tool.schema.string().optional().describe("Optional comma-separated domain blocklist"),
          include_external: tool.schema.boolean().optional().describe("Follow external links (default false)"),
          query: tool.schema.string().optional().describe("Optional semantic relevance query"),
          max_chars: tool.schema.number().optional().describe("Characters retained per page (default 6000)"),
        },
        execute: crawlExecute,
      }),

      WebExtract: tool({
        description:
          "Alias of web_extract for Ruflo workflows. Structured or semantic extraction from one exact URL. " +
          "For json_css, pass schema_json with baseSelector and a fields array.",
        args: {
          url: tool.schema.string().describe("Exact URL from the user or WebSearch"),
          strategy: tool.schema.enum(["json_css", "cosine"]).optional().describe("Extraction strategy; defaults to json_css"),
          schema_json: tool.schema
            .string()
            .optional()
            .describe("Required for json_css: put the JSON-CSS schema string here, never in query"),
          query: tool.schema
            .string()
            .optional()
            .describe("Only for cosine: semantic filter text; never put JSON schema here"),
          max_chars: tool.schema.number().optional().describe("Maximum extracted characters"),
        },
        execute: extractExecute,
      }),
    },
  };
}
