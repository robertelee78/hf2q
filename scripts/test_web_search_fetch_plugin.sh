#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d -t hf2q-web-plugin.XXXXXX)"
cleanup() {
    rm -r -- "$test_root"
}
trap cleanup EXIT

mkdir -p "$test_root/node_modules/@opencode-ai/plugin"
printf '%s\n' \
    '{"type":"module","exports":"./index.js"}' \
    > "$test_root/node_modules/@opencode-ai/plugin/package.json"
printf '%s\n' \
    'export const tool = {};' \
    > "$test_root/node_modules/@opencode-ai/plugin/index.js"
cp "$repo_root/scripts/opencode-web-stack/web-search-fetch.js" \
    "$test_root/web-search-fetch.mjs"

# JavaScript receives its file path through process.argv; shell expansion here
# would be a bug.
# shellcheck disable=SC2016
node --input-type=module -e '
const mod = await import(`file://${process.argv[1]}`);
const compact = mod.normalizeJsonCssSchema({heading: "h1"});
const keyed = mod.normalizeJsonCssSchema({
  name: "Example",
  baseSelector: "body",
  fields: {heading: {selector: "h1", type: "text"}},
});
for (const schema of [compact, keyed]) {
  if (schema.baseSelector !== "body") throw new Error(JSON.stringify(schema));
  if (!Array.isArray(schema.fields)) throw new Error(JSON.stringify(schema));
  const field = schema.fields[0];
  if (field.name !== "heading" || field.selector !== "h1" || field.type !== "text") {
    throw new Error(JSON.stringify(schema));
  }
}

if (!mod.resultsLookRelevant("what is the price of gold today", [
  {title: "Gold Price Today", url: "https://example.com/gold-price", content: "Live spot price"},
])) throw new Error("gold relevance false negative");
if (!mod.resultsLookRelevant("tell me about the company IOActive", [
  {title: "IOActive", url: "https://ioactive.com/about", content: "Security consultancy"},
])) throw new Error("IOActive relevance false negative");
if (mod.resultsLookRelevant("who wrote unicornscan", [
  {title: "WROTE Definition", url: "https://dictionary.example/wrote", content: "past tense"},
])) throw new Error("Unicornscan relevance false positive");
const laptopGoldJunk = [
  {
    title: "Price.com: Save with Cash Back, Coupons & Price Comparison",
    url: "https://price.com/",
    content: "Offers for more than 100,000 brands.",
  },
  {
    title: "Home - Price Industries",
    url: "https://priceindustries.com/",
    content: "A market leader in supplying air distribution products.",
  },
  {
    title: "Priceline.com - Hotels, Flights and Rental Cars",
    url: "https://www.priceline.com/",
    content: "Members get our best travel price.",
  },
];
if (mod.resultsLookRelevant("what is the price of gold today", laptopGoldJunk)) {
  throw new Error("exact laptop junk passed relevance");
}
if (mod.resultsLookRelevant("what is the price of gold today", [{
  title: "Goldman Sachs market outlook",
  url: "https://example.com/goldman-outlook",
  content: "Current equity prices",
}])) throw new Error("substring matched gold without a token boundary");
if (!mod.resultsLookRelevant("what is the price of gold today", [{
  title: "Gold prices today",
  url: "https://example.com/metals",
  content: "Latest bullion quote",
}])) throw new Error("plural prices did not match singular price query term");
if (!mod.resultsLookRelevant("what are the prices of gold today", [{
  title: "Gold price today",
  url: "https://example.com/metals",
  content: "Latest bullion quote",
}])) throw new Error("singular price did not match plural prices query term");
if (!mod.resultsLookRelevant("rust async runtime", [{
  title: "Rust runtime design",
  url: "https://example.com/runtime",
  content: "A systems guide",
}])) throw new Error("two of three query terms did not satisfy the intended threshold");
if (mod.resultsLookRelevant("who", [{
  title: "Anything",
  url: "https://example.com/",
  content: "",
}])) throw new Error("query without identifying terms did not fail closed");

const response = (payload, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => payload,
});
const source = {
  title: "Example",
  url: "https://example.com/docs",
  content: "evidence",
  engines: ["bing"],
};

let calls = [];
globalThis.fetch = async (url, options = {}) => {
  calls.push({url: String(url), options});
  if (String(url).includes(":8888/search")) {
    return response({results: [source], unresponsive_engines: []});
  }
  if (String(url).endsWith("/fetch")) {
    const body = JSON.parse(options.body);
    if (!body.public_only || body.mode !== "static") throw new Error(JSON.stringify(body));
    return response({ok: true, url: body.url, markdown: "page", via: "static"});
  }
  throw new Error(`unexpected URL ${url}`);
};
let output = await mod.searchExecute({query: "example", pages: 1});
if (!output.includes("Search route: SearXNG primary")) throw new Error(output);
if (calls.some(({url}) => url.endsWith("/search-fallback"))) throw new Error("fallback ran after primary success");

calls = [];
globalThis.fetch = async (url, options = {}) => {
  calls.push({url: String(url), options});
  if (String(url).includes(":8888/search")) throw new Error("primary unavailable");
  if (String(url).endsWith("/search-fallback")) {
    return response({
      ok: true,
      provider: "bing-browser-fallback",
      via: "browser",
      results: [{...source, engines: ["bing-browser-fallback"]}],
    });
  }
  if (String(url).endsWith("/fetch")) {
    return response({ok: true, url: source.url, markdown: "page", via: "static"});
  }
  throw new Error(`unexpected URL ${url}`);
};
output = await mod.searchExecute({query: "example", pages: 1});
if (!output.includes("Search route: bing-browser-fallback via browser")) throw new Error(output);
if (!output.includes("[bing-browser-fallback]")) throw new Error(output);
if (calls.filter(({url}) => url.endsWith("/search-fallback")).length !== 1) {
  throw new Error("forced outage did not make exactly one fallback request");
}

calls = [];
globalThis.fetch = async (url) => {
  calls.push({url: String(url)});
  if (String(url).includes(":8888/search")) {
    return response({results: laptopGoldJunk, unresponsive_engines: []});
  }
  if (String(url).endsWith("/search-fallback")) {
    return response({
      ok: true,
      provider: "bing-browser-fallback",
      via: "browser",
      results: laptopGoldJunk,
    });
  }
  if (String(url).endsWith("/fetch")) throw new Error("irrelevant laptop URL was fetched");
  throw new Error(`unexpected URL ${url}`);
};
output = await mod.searchExecute({query: "what is the price of gold today", pages: 3});
if (!output.includes("WEB_SEARCH_FAILED")) throw new Error(output);
if (!output.includes("Do not guess URLs")) throw new Error(output);
if (output.includes("Price.com") || output.includes("Priceline")) throw new Error(output);

calls = [];
globalThis.fetch = async (url, options = {}) => {
  calls.push({url: String(url), options});
  if (String(url).includes(":8888/search")) {
    return response({
      results: [
        laptopGoldJunk[0],
        {
          title: "Gold Price Today",
          url: "https://www.kitco.com/charts/gold",
          content: "Live gold price per ounce.",
          engines: ["bing"],
        },
      ],
      unresponsive_engines: [],
    });
  }
  if (String(url).endsWith("/fetch")) {
    const body = JSON.parse(options.body);
    if (body.url !== "https://www.kitco.com/charts/gold") throw new Error(options.body);
    return response({ok: true, url: body.url, markdown: "Gold price evidence", via: "static"});
  }
  throw new Error(`unexpected URL ${url}`);
};
output = await mod.searchExecute({query: "what is the price of gold today", pages: 3});
if (!output.includes("Gold Price Today") || output.includes("Price.com")) throw new Error(output);
if (calls.some(({url}) => url.endsWith("/search-fallback"))) throw new Error("fallback ran despite relevant primary result");

calls = [];
globalThis.fetch = async (url, options = {}) => {
  calls.push({url: String(url), options});
  if (String(url).includes(":8888/search")) {
    return response({
      results: [{title: "WROTE Definition", url: "https://dictionary.example/wrote", content: "past tense"}],
      unresponsive_engines: [],
    });
  }
  if (String(url).endsWith("/search-fallback")) {
    return response({
      ok: true,
      provider: "bing-browser-fallback",
      via: "stealth",
      results: [{
        title: "About Unicornscan",
        url: "https://unicornscan.org/about",
        content: "Creator Jack C. Louis",
        engines: ["bing-browser-fallback"],
      }],
    });
  }
  if (String(url).endsWith("/fetch")) {
    return response({ok: true, url: "https://unicornscan.org/about", markdown: "Jack C. Louis", via: "static"});
  }
  throw new Error(`unexpected URL ${url}`);
};
output = await mod.searchExecute({query: "who wrote unicornscan", pages: 1});
if (!output.includes("Search route: bing-browser-fallback via stealth")) throw new Error(output);
if (!output.includes("About Unicornscan")) throw new Error(output);
if (calls.filter(({url}) => url.endsWith("/search-fallback")).length !== 1) {
  throw new Error("low-relevance primary did not make exactly one fallback request");
}

calls = [];
globalThis.fetch = async (url, options = {}) => {
  calls.push({url: String(url), options});
  if (String(url).includes(":8888/search")) {
    return response({results: [], unresponsive_engines: [["arxiv", "timeout"]]});
  }
  throw new Error(`constraint was relaxed through ${url}`);
};
output = await mod.searchExecute({query: "paper", engines: "arxiv", time_range: "year"});
if (!output.includes("fallback skipped to preserve explicit constraints")) throw new Error(output);
if (calls.length !== 1 || !calls[0].url.includes("engines=arxiv") || !calls[0].url.includes("time_range=year")) {
  throw new Error(JSON.stringify(calls));
}

calls = [];
globalThis.fetch = async (url) => {
  calls.push({url: String(url)});
  if (String(url).includes(":8888/search")) {
    return response({results: [], unresponsive_engines: [["arxiv", "timeout"]]});
  }
  throw new Error(`category constraint was relaxed through ${url}`);
};
output = await mod.searchExecute({query: "paper", category: "academic"});
if (!output.includes("fallback skipped to preserve explicit constraints")) throw new Error(output);
if (calls.length !== 1 || !calls[0].url.includes("engines=arxiv")) throw new Error(JSON.stringify(calls));

calls = [];
globalThis.fetch = async (url, options = {}) => {
  calls.push({url: String(url), options});
  if (String(url).includes(":8888/search")) {
    if (!String(url).includes("language=fr-FR")) throw new Error(`language missing from primary: ${url}`);
    return response({results: [], unresponsive_engines: [["bing", "timeout"]]});
  }
  if (String(url).endsWith("/search-fallback")) {
    const body = JSON.parse(options.body);
    if (body.language !== "fr-FR") throw new Error(`language missing from fallback: ${options.body}`);
    return response({
      ok: true,
      provider: "bing-browser-fallback",
      via: "browser",
      results: [{...source, engines: ["bing-browser-fallback"]}],
    });
  }
  if (String(url).endsWith("/fetch")) {
    return response({ok: true, url: source.url, markdown: "page", via: "static"});
  }
  throw new Error(`unexpected URL ${url}`);
};
output = await mod.searchExecute({query: "example", language: "fr-FR", pages: 1});
if (!output.includes("Search route: bing-browser-fallback via browser")) throw new Error(output);

console.log("web-search-fetch reliability and schema contracts passed");
' "$test_root/web-search-fetch.mjs"
