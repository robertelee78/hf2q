#!/usr/bin/env bash
# Install the durable, loopback-only OpenCode search/fetch/crawl/extract stack.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This installer currently supports macOS launchd hosts only." >&2
    exit 2
fi

ACTION="${1:-install}"
case "$ACTION" in
    install|--install) ACTION="install" ;;
    --disable) ACTION="disable" ;;
    --enable) ACTION="enable" ;;
    --uninstall) ACTION="uninstall" ;;
    --status) ACTION="status" ;;
    *)
        echo "usage: $0 [--install|--disable|--enable|--status|--uninstall]" >&2
        exit 2
        ;;
esac

STATE_DIR="$HOME/.local/state"
FETCH_DIR="$HOME/.local/opt/crawl4ai-server"
SEARX_DIR="$HOME/.local/opt/searxng"
PLUGIN_DIR="$HOME/.config/opencode/plugins"
PLUGIN_PATH="$PLUGIN_DIR/web-search-fetch.js"
DISABLED_PLUGIN_PATH="$PLUGIN_PATH.disabled"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
SEARX_PLIST="$LAUNCH_DIR/com.opencode.searxng.plist"
FETCH_PLIST="$LAUNCH_DIR/com.opencode.crawl4ai.plist"
USER_DOMAIN="gui/$(id -u)"

stop_services() {
    launchctl bootout "$USER_DOMAIN" "$SEARX_PLIST" >/dev/null 2>&1 || true
    launchctl bootout "$USER_DOMAIN" "$FETCH_PLIST" >/dev/null 2>&1 || true
}

if [[ "$ACTION" == "disable" ]]; then
    stop_services
    if [[ -f "$PLUGIN_PATH" ]]; then
        [[ ! -e "$DISABLED_PLUGIN_PATH" ]] || {
            echo "refusing to overwrite existing $DISABLED_PLUGIN_PATH" >&2
            exit 2
        }
        mv "$PLUGIN_PATH" "$DISABLED_PLUGIN_PATH"
    fi
    echo "OpenCode web stack disabled. Restart OpenCode to unload its tools."
    exit 0
fi

if [[ "$ACTION" == "enable" ]]; then
    if [[ -f "$DISABLED_PLUGIN_PATH" ]]; then
        [[ ! -e "$PLUGIN_PATH" ]] || {
            echo "refusing to overwrite existing $PLUGIN_PATH" >&2
            exit 2
        }
        mv "$DISABLED_PLUGIN_PATH" "$PLUGIN_PATH"
    fi
    [[ -f "$SEARX_PLIST" && -f "$FETCH_PLIST" ]] || {
        echo "LaunchAgents are missing; run the installer without a flag first" >&2
        exit 2
    }
    stop_services
    launchctl bootstrap "$USER_DOMAIN" "$SEARX_PLIST"
    launchctl bootstrap "$USER_DOMAIN" "$FETCH_PLIST"
    echo "OpenCode web stack enabled. Restart OpenCode to load its tools."
    exit 0
fi

if [[ "$ACTION" == "status" ]]; then
    if [[ -f "$PLUGIN_PATH" ]]; then
        echo "plugin: enabled ($PLUGIN_PATH)"
    elif [[ -f "$DISABLED_PLUGIN_PATH" ]]; then
        echo "plugin: disabled ($DISABLED_PLUGIN_PATH)"
    else
        echo "plugin: not installed"
    fi
    launchctl print "$USER_DOMAIN/com.opencode.searxng" >/dev/null 2>&1 \
        && echo "searxng: running" || echo "searxng: stopped"
    launchctl print "$USER_DOMAIN/com.opencode.crawl4ai" >/dev/null 2>&1 \
        && echo "crawl4ai: running" || echo "crawl4ai: stopped"
    exit 0
fi

if [[ "$ACTION" == "uninstall" ]]; then
    stop_services
    TRASH_DIR="$HOME/.Trash/hf2q-opencode-web-stack-$(date +%Y%m%d%H%M%S)"
    mkdir -p "$TRASH_DIR"
    shopt -s nullglob
    LEGACY_BACKUPS=(
        "$PLUGIN_PATH".*.bak
        "$DISABLED_PLUGIN_PATH".*.bak
        "$SEARX_PLIST".*.bak
        "$FETCH_PLIST".*.bak
    )
    for path in \
        "$PLUGIN_PATH" \
        "$DISABLED_PLUGIN_PATH" \
        "$SEARX_PLIST" \
        "$FETCH_PLIST" \
        "$SEARX_DIR" \
        "$FETCH_DIR" \
        "$STATE_DIR/searxng.log" \
        "$STATE_DIR/crawl4ai.log" \
        "${LEGACY_BACKUPS[@]}"
    do
        if [[ -e "$path" ]]; then
            mv "$path" "$TRASH_DIR/"
        fi
    done
    echo "OpenCode web stack moved to: $TRASH_DIR"
    echo "Restart OpenCode to unload its tools. The moved files remain recoverable."
    exit 0
fi

for command in curl git jq launchctl node openssl plutil; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "required command not found: $command" >&2
        exit 2
    }
done

if [[ ! -d "/Applications/Google Chrome.app" ]]; then
    echo "Google Chrome is required for the bounded stealth fallback." >&2
    echo "Install it with: brew install --cask google-chrome" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..." >&2
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || {
    echo "uv installation completed but uv is not on PATH" >&2
    exit 2
}

if SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; then
    :
else
    SCRIPT_DIR="$PWD"
fi
LOCAL_ASSET_DIR="$SCRIPT_DIR/opencode-web-stack"
ASSET_TMP=""
if [[ -d "$LOCAL_ASSET_DIR" ]]; then
    ASSET_DIR="$LOCAL_ASSET_DIR"
else
    ASSET_TMP="$(mktemp -d -t hf2q-web-stack.XXXXXX)"
    ASSET_DIR="$ASSET_TMP"
    RAW_BASE="${HF2Q_WEB_STACK_ASSET_BASE:-https://raw.githubusercontent.com/robertelee78/hf2q/main/scripts/opencode-web-stack}"
    for asset in \
        web-search-fetch.js \
        server.py \
        stealth_fetch.py \
        requirements.txt \
        test_server.py \
        searxng-settings.yml
    do
        curl -fsSL "$RAW_BASE/$asset" -o "$ASSET_DIR/$asset"
    done
fi
cleanup() {
    if [[ -n "$ASSET_TMP" && -d "$ASSET_TMP" ]]; then
        rm -r -- "$ASSET_TMP"
    fi
}
trap cleanup EXIT

for asset in \
    web-search-fetch.js \
    server.py \
    stealth_fetch.py \
    requirements.txt \
    test_server.py \
    searxng-settings.yml
do
    [[ -s "$ASSET_DIR/$asset" ]] || {
        echo "installer asset missing or empty: $asset" >&2
        exit 2
    }
done

BACKUP_TAG="$(date +%Y%m%d%H%M%S)"
BACKUP_DIR="$FETCH_DIR/backups/$BACKUP_TAG"
backup_if_changed() {
    local source_path="$1"
    local target_path="$2"
    if [[ -f "$target_path" ]] && ! cmp -s "$source_path" "$target_path"; then
        mkdir -p "$BACKUP_DIR"
        cp -p "$target_path" "$BACKUP_DIR/$(basename "$target_path")"
    fi
}

mkdir -p "$STATE_DIR" "$FETCH_DIR" "$PLUGIN_DIR" "$LAUNCH_DIR"

backup_if_changed "$ASSET_DIR/server.py" "$FETCH_DIR/server.py"
backup_if_changed "$ASSET_DIR/stealth_fetch.py" "$FETCH_DIR/stealth_fetch.py"
backup_if_changed "$ASSET_DIR/requirements.txt" "$FETCH_DIR/requirements.txt"
backup_if_changed "$ASSET_DIR/test_server.py" "$FETCH_DIR/test_server.py"
backup_if_changed "$ASSET_DIR/web-search-fetch.js" "$PLUGIN_DIR/web-search-fetch.js"
if [[ -f "$DISABLED_PLUGIN_PATH" ]]; then
    mkdir -p "$BACKUP_DIR"
    mv "$DISABLED_PLUGIN_PATH" "$BACKUP_DIR/web-search-fetch.js.disabled"
fi
install -m 0644 "$ASSET_DIR/server.py" "$FETCH_DIR/server.py"
install -m 0644 "$ASSET_DIR/stealth_fetch.py" "$FETCH_DIR/stealth_fetch.py"
install -m 0644 "$ASSET_DIR/requirements.txt" "$FETCH_DIR/requirements.txt"
install -m 0644 "$ASSET_DIR/test_server.py" "$FETCH_DIR/test_server.py"
install -m 0644 "$ASSET_DIR/web-search-fetch.js" "$PLUGIN_DIR/web-search-fetch.js"

SEARX_REV="b023a28bab8839dba9eac96e9a51cc91bbd0a267"
if [[ ! -d "$SEARX_DIR/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
        https://github.com/searxng/searxng.git "$SEARX_DIR"
fi
if ! git -C "$SEARX_DIR" diff --quiet --ignore-submodules --; then
    echo "tracked local changes exist in $SEARX_DIR; refusing to overwrite them" >&2
    exit 2
fi
git -C "$SEARX_DIR" fetch --depth 1 origin "$SEARX_REV"
git -C "$SEARX_DIR" switch --detach FETCH_HEAD

SEARX_SECRET="$(openssl rand -hex 32)"
SETTINGS_TMP="$(mktemp -t hf2q-searx-settings.XXXXXX)"
sed "s/__SEARXNG_SECRET__/$SEARX_SECRET/" \
    "$ASSET_DIR/searxng-settings.yml" > "$SETTINGS_TMP"
backup_if_changed "$SETTINGS_TMP" "$SEARX_DIR/settings.yml"
install -m 0600 "$SETTINGS_TMP" "$SEARX_DIR/settings.yml"
rm "$SETTINGS_TMP"

uv python install 3.13
if [[ ! -x "$SEARX_DIR/.venv/bin/python" ]]; then
    uv venv --python 3.13 "$SEARX_DIR/.venv"
fi
uv pip install --python "$SEARX_DIR/.venv/bin/python" \
    -r "$SEARX_DIR/requirements.txt" \
    -r "$SEARX_DIR/requirements-server.txt"
# SearXNG's legacy editable build imports runtime modules (including msgspec)
# from searx/__init__.py while evaluating setup.py. Its declared requirements
# are installed immediately above, so use that exact environment instead of a
# fresh PEP 517 build sandbox that cannot import them.
uv pip install --python "$SEARX_DIR/.venv/bin/python" \
    --no-build-isolation \
    -e "$SEARX_DIR"

if [[ ! -x "$FETCH_DIR/.venv/bin/python" ]]; then
    uv venv --python 3.13 "$FETCH_DIR/.venv"
fi
uv pip install --python "$FETCH_DIR/.venv/bin/python" \
    -r "$FETCH_DIR/requirements.txt"
uv pip check --python "$FETCH_DIR/.venv/bin/python"
BROWSER_SETUP_REV="$({
    uv pip freeze --python "$FETCH_DIR/.venv/bin/python" 2>/dev/null
} | shasum -a 256 | awk '{print $1}')"
BROWSER_SETUP_MARKER="$FETCH_DIR/.browser-setup-$BROWSER_SETUP_REV"
if [[ ! -f "$BROWSER_SETUP_MARKER" ]]; then
    "$FETCH_DIR/.venv/bin/crawl4ai-setup"
    touch "$BROWSER_SETUP_MARKER"
fi

PLIST_TMP="$(mktemp -t hf2q-launch-agent.XXXXXX)"

jq -n \
    --arg home "$HOME" \
    --arg python "$SEARX_DIR/.venv/bin/python" \
    --arg workdir "$SEARX_DIR" \
    --arg settings "$SEARX_DIR/settings.yml" \
    --arg log "$STATE_DIR/searxng.log" '{
      Label: "com.opencode.searxng",
      ProgramArguments: [$python, "-m", "searx.webapp"],
      WorkingDirectory: $workdir,
      EnvironmentVariables: {SEARXNG_SETTINGS_PATH: $settings, HOME: $home},
      RunAtLoad: true,
      KeepAlive: true,
      StandardOutPath: $log,
      StandardErrorPath: $log
    }' > "$PLIST_TMP"
plutil -convert xml1 -o "$PLIST_TMP.xml" "$PLIST_TMP"
backup_if_changed "$PLIST_TMP.xml" "$SEARX_PLIST"
install -m 0644 "$PLIST_TMP.xml" "$SEARX_PLIST"

jq -n \
    --arg home "$HOME" \
    --arg python "$FETCH_DIR/.venv/bin/python" \
    --arg server "$FETCH_DIR/server.py" \
    --arg workdir "$FETCH_DIR" \
    --arg log "$STATE_DIR/crawl4ai.log" '{
      Label: "com.opencode.crawl4ai",
      ProgramArguments: [$python, $server],
      WorkingDirectory: $workdir,
      EnvironmentVariables: {FETCH_LOG_LEVEL: "INFO", PYTHONUNBUFFERED: "1", HOME: $home},
      RunAtLoad: true,
      KeepAlive: true,
      StandardOutPath: $log,
      StandardErrorPath: $log
    }' > "$PLIST_TMP"
rm -f "$PLIST_TMP.xml"
plutil -convert xml1 -o "$PLIST_TMP.xml" "$PLIST_TMP"
backup_if_changed "$PLIST_TMP.xml" "$FETCH_PLIST"
install -m 0644 "$PLIST_TMP.xml" "$FETCH_PLIST"
rm -f "$PLIST_TMP" "$PLIST_TMP.xml"

launchctl bootout "$USER_DOMAIN" "$SEARX_PLIST" >/dev/null 2>&1 || true
launchctl bootout "$USER_DOMAIN" "$FETCH_PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "$USER_DOMAIN" "$SEARX_PLIST"
launchctl bootstrap "$USER_DOMAIN" "$FETCH_PLIST"

FETCH_READY=0
SEARX_READY=0
for _ in $(seq 1 120); do
    if curl --connect-timeout 1 --max-time 2 -fsS \
        http://127.0.0.1:11235/healthz >/dev/null 2>&1; then
        FETCH_READY=1
    fi
    if curl --connect-timeout 1 --max-time 2 -fsS \
        http://127.0.0.1:8888/ >/dev/null 2>&1; then
        SEARX_READY=1
    fi
    [[ "$FETCH_READY" -eq 1 && "$SEARX_READY" -eq 1 ]] && break
    sleep 2
done
[[ "$FETCH_READY" -eq 1 ]] || {
    echo "fetch service did not become ready; inspect $STATE_DIR/crawl4ai.log" >&2
    exit 1
}
[[ "$SEARX_READY" -eq 1 ]] || {
    echo "SearXNG did not become ready; inspect $STATE_DIR/searxng.log" >&2
    exit 1
}

"$FETCH_DIR/.venv/bin/python" -m py_compile \
    "$FETCH_DIR/server.py" "$FETCH_DIR/stealth_fetch.py" "$FETCH_DIR/test_server.py"
(
    cd "$FETCH_DIR"
    .venv/bin/python -m unittest -v test_server.py
)
node --check "$PLUGIN_DIR/web-search-fetch.js"
plutil -lint "$SEARX_PLIST" "$FETCH_PLIST"

curl -fsS -X POST http://127.0.0.1:11235/fetch \
    -H 'Content-Type: application/json' \
    -d '{"url":"https://example.com/","mode":"auto","max_chars":2000}' \
    | jq -e '.ok == true and (.markdown | length > 0)' >/dev/null
curl -fsS --get http://127.0.0.1:8888/search \
    --data-urlencode 'q=hf2q local inference' \
    --data 'format=json' \
    | jq -e '.results | type == "array"' >/dev/null

echo
echo "OpenCode web stack installed and verified."
echo "Restart OpenCode, then use: web_search, web_fetch, web_crawl, web_extract."
echo "Ruflo aliases are also present: WebSearch, WebFetch, WebCrawl, WebExtract."
