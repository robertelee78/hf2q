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
console.log("web-search-fetch plugin schema contract passed");
' "$test_root/web-search-fetch.mjs"
