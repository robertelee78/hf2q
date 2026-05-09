#!/usr/bin/env bash
# ADR-020 AC#7 calibration corpus fetcher.
#
# Downloads WikiText-2 (raw, v1) and emits a JSONL file with one
# `{"text": "..."}` record per article — the schema consumed by
# `hf2q dwq-train --full-model-teacher --calibration-data <path>`.
#
# Mirrors mlx-lm's `CompletionsDataset` convention at
# `mlx_lm/tuner/datasets.py:25-29` (one record per line, "text" key).
#
# Reuses the canonical source URL + SHA256 + zip member established
# by `scripts/fetch_wikitext2.sh` (the perplexity-benchmark fetcher,
# which emits a different binary token-ID format for a different
# downstream consumer).  Single source of truth for the corpus.

set -euo pipefail

SCRIPT_NAME="${0##*/}"

DEFAULT_OUTPUT="tests/fixtures/calibrate/wikitext2-train.jsonl"
DEFAULT_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hf2q/wikitext2"

# --- Source spec — KEEP IN SYNC with scripts/fetch_wikitext2.sh ---
# We use the TRAIN split here (calibration corpus typically uses
# the larger split, vs perplexity benchmarks which use test).
WIKITEXT2_URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT2_SHA256="ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
WIKITEXT2_ZIP_NAME="wikitext-2-raw-v1.zip"
WIKITEXT2_TRAIN_MEMBER="wikitext-2-raw/wiki.train.raw"

# Validation: the train split must produce at least this many JSONL
# records (articles).  WikiText-2 train has ~600 articles; floor at
# 100 catches gross corruption.
MIN_JSONL_LINES=100
MIN_TOTAL_BYTES=1048576

die() {
    printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
usage: $SCRIPT_NAME [options]

Downloads wikitext-2-raw-v1 train split and emits one
{"text": "..."} JSONL record per article — the schema consumed by
\`hf2q dwq-train --full-model-teacher --calibration-data\`.

Options:
  --output <path>     output .jsonl path
                      (default: $DEFAULT_OUTPUT)
  --cache-dir <path>  cache directory for zip + extracted text
                      (default: $DEFAULT_CACHE_DIR)
  --force             rebuild even when output/cache files exist
  --help              show this help

Source URL:
  $WIKITEXT2_URL

Locked SHA-256:
  $WIKITEXT2_SHA256
EOF
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "missing sha256sum or shasum"
    fi
}

file_size_bytes() {
    if stat -c %s "$1" >/dev/null 2>&1; then
        stat -c %s "$1"
    else
        stat -f %z "$1"
    fi
}

validate_jsonl_file() {
    local path="$1"
    local size lines

    [[ -f "$path" ]] || return 1
    size="$(file_size_bytes "$path")" || return 1
    [[ "$size" =~ ^[0-9]+$ ]] || return 1
    (( size >= MIN_TOTAL_BYTES )) || return 1
    lines="$(wc -l < "$path" | awk '{print $1}')"
    (( lines >= MIN_JSONL_LINES )) || return 1
    # Verify FIRST line is well-formed JSON with a "text" key.
    head -n 1 "$path" | python3 -c '
import sys, json
line = sys.stdin.readline()
obj = json.loads(line)
if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
    sys.exit(1)
' || return 1
    printf '%s %s\n' "$lines" "$size"
}

output="$DEFAULT_OUTPUT"
cache_dir="$DEFAULT_CACHE_DIR"
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || die "--output requires a path"
            output="$2"
            shift 2
            ;;
        --cache-dir)
            [[ $# -ge 2 ]] || die "--cache-dir requires a path"
            cache_dir="$2"
            shift 2
            ;;
        --force)
            force=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

output_parent="$(dirname "$output")"
mkdir -p "$output_parent"

if [[ "$force" -eq 0 && -f "$output" ]]; then
    if read -r cached_lines cached_size < <(validate_jsonl_file "$output"); then
        cached_sha="$(sha256_file "$output")"
        printf 'cached, skipping: %s has %s lines (%s bytes); SHA-256 %s\n' \
            "$output" "$cached_lines" "$cached_size" "$cached_sha"
        exit 0
    fi
    die "existing output failed validation: $output (use --force to rebuild)"
fi

require_cmd curl
require_cmd unzip
require_cmd python3

mkdir -p "$cache_dir"
zip_path="$cache_dir/$WIKITEXT2_ZIP_NAME"
text_path="$cache_dir/wiki.train.raw"

if [[ "$force" -eq 1 ]]; then
    rm -f "$zip_path" "$text_path"
fi

if [[ ! -f "$zip_path" ]]; then
    tmp_zip="$zip_path.tmp.$$"
    rm -f "$tmp_zip"
    if ! curl -L --fail --retry 3 -o "$tmp_zip" "$WIKITEXT2_URL"; then
        rm -f "$tmp_zip"
        die "download failed: $WIKITEXT2_URL"
    fi
    mv "$tmp_zip" "$zip_path"
fi

actual_sha="$(sha256_file "$zip_path")"
if [[ "$actual_sha" != "$WIKITEXT2_SHA256" ]]; then
    die "SHA-256 mismatch for $zip_path: expected $WIKITEXT2_SHA256, got $actual_sha"
fi

if [[ ! -f "$text_path" || "$zip_path" -nt "$text_path" ]]; then
    tmp_text="$text_path.tmp.$$"
    rm -f "$tmp_text"
    if ! unzip -p "$zip_path" "$WIKITEXT2_TRAIN_MEMBER" >"$tmp_text"; then
        rm -f "$tmp_text"
        die "failed to extract $WIKITEXT2_TRAIN_MEMBER from $zip_path"
    fi
    if [[ ! -s "$tmp_text" ]]; then
        rm -f "$tmp_text"
        die "extracted text is empty: $WIKITEXT2_TRAIN_MEMBER"
    fi
    if ! python3 -c 'import sys; open(sys.argv[1], "rb").read().decode("utf-8")' "$tmp_text"; then
        rm -f "$tmp_text"
        die "extracted text is not valid UTF-8: $WIKITEXT2_TRAIN_MEMBER"
    fi
    mv "$tmp_text" "$text_path"
fi

# Article splitting: WikiText-2 separates articles by " = Title = "
# (single =, level-1 heading) on its own line.  Lower-level headings
# use 2+ equals signs and stay within an article.
#
# Pure Python (jq doesn't ship a way to chunk text by regex on
# multi-line input).  Emit one JSONL record per article; preserve
# inline whitespace + newlines via JSON's string-escape semantics.
tmp_output="$output.tmp.$$"
rm -f "$tmp_output"
if ! line_count="$(
    python3 - "$text_path" "$tmp_output" <<'PY'
import json
import re
import sys

text_path, out_path = sys.argv[1], sys.argv[2]

with open(text_path, "r", encoding="utf-8") as f:
    raw = f.read()

# Article boundary: " = Title = " on its own line (single =).  Use a
# regex that requires the surrounding whitespace and forbids 2+ = on
# either side (those are sub-headings within an article).
boundary_re = re.compile(r"^\s*=\s[^=].*\s=\s*$", re.MULTILINE)

articles = []
last_end = 0
for m in boundary_re.finditer(raw):
    # Article TEXT runs from previous boundary's end to this match's
    # start.  The first article (preamble before any heading) is
    # usually empty/whitespace; skip if so.
    chunk = raw[last_end:m.start()].strip()
    if chunk:
        articles.append(chunk)
    last_end = m.start()
# Final article runs from the last boundary to EOF.
final = raw[last_end:].strip()
if final:
    articles.append(final)

with open(out_path, "w", encoding="utf-8") as out:
    for a in articles:
        out.write(json.dumps({"text": a}, ensure_ascii=False))
        out.write("\n")

print(len(articles))
PY
)"; then
    rm -f "$tmp_output"
    die "JSONL conversion failed"
fi

[[ "$line_count" =~ ^[0-9]+$ ]] || die "internal error: non-numeric line count: $line_count"
if (( line_count < MIN_JSONL_LINES )); then
    rm -f "$tmp_output"
    die "JSONL line-count validation failed: got $line_count, expected at least $MIN_JSONL_LINES"
fi

output_size="$(file_size_bytes "$tmp_output")"
if (( output_size < MIN_TOTAL_BYTES )); then
    rm -f "$tmp_output"
    die "file-size validation failed: got $output_size bytes, expected at least $MIN_TOTAL_BYTES"
fi

# Schema check: every line must parse as JSON with a string "text" field.
if ! python3 - "$tmp_output" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            sys.stderr.write(f"line {i}: not valid JSON: {e}\n")
            sys.exit(1)
        if not isinstance(obj, dict):
            sys.stderr.write(f"line {i}: JSON value is not an object\n")
            sys.exit(1)
        if "text" not in obj or not isinstance(obj["text"], str):
            sys.stderr.write(f"line {i}: missing or non-string 'text' field\n")
            sys.exit(1)
        if not obj["text"].strip():
            sys.stderr.write(f"line {i}: 'text' is empty/whitespace\n")
            sys.exit(1)
PY
then
    rm -f "$tmp_output"
    die "JSONL schema validation failed"
fi

mv "$tmp_output" "$output"
output_sha="$(sha256_file "$output")"
printf 'OK wrote %s articles (%s bytes) to %s; SHA-256 %s\n' \
    "$line_count" "$output_size" "$output" "$output_sha"
