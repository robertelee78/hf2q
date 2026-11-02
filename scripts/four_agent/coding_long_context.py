#!/usr/bin/env python3
"""Long-context variant of the canonical four-agent coding workload (ADR-063 D2, #365).

`qwen_n4_live_smoke.py --workload long-context-v1` runs the canonical parse_port task:
the same prompts, tools, six-step workflow, transport matrix, src/ files, test counts
and grading. The one difference is tests/ports.rs. After its canonical two integration
tests it carries a large read-only Rust module, the configuration schema of the daemon
the port parser feeds (fixtures/long-context-v1.rs, rendered by `render()` and pinned by
FIXTURE_SHA256). Step (2) of the workflow reads tests/ports.rs, so every request from
turn 2 on carries that module in its history.

Without --workload nothing here is imported, and the canonical workload is byte-identical
to earlier receipts.

    python3 coding_long_context.py render > fixtures/long-context-v1.rs
    python3 coding_long_context.py estimate
    python3 coding_long_context.py derive-case q5_cases/<case>.json > <case>-long-context.json
"""
import hashlib
import json
from pathlib import Path
import re
from string import Template
import sys

from qwen38_n1_tools import IMMUTABLE, INITIAL
from qwen38_n1_verdict import EXPECTED, WIRE_LIMIT, require

ID = "long-context-v1"
WORKLOADS = (ID,)
FIXTURE = Path(__file__).with_name("fixtures") / f"{ID}.rs"
FIXTURE_REPO_PATH = f"{{REPO}}/scripts/four_agent/fixtures/{ID}.rs"
FIXTURE_SHA256 = "cfa2a052035de194fd16965a6493601b2383c9ede219235c061d98fbe10ad96b"
# Approval sources added to the client's source freeze when --workload is used.
MODULES = ("coding_long_context.py", "test_coding_long_context.py")
# Per-read bound for this workload's tests/ports.rs. Every other tool result keeps the
# canonical 64 KiB bound, and every request keeps the canonical 256 KiB wire bound.
READ_LIMIT = 192 * 1024
CLIENT = "{REPO}/scripts/four_agent/qwen_n4_live_smoke.py"
# Client options a derived case must not already carry (they select another workload).
FOREIGN_OPTIONS = {"--workload", "--instance", "--phase-plan", "--benchmark-profile"}
FOREIGN_KEYS = {"workload_variant", "workload_fixture_sha256", "instance", "instance_table_sha256",
                "benchmark_profile"}
MAXIMUM = {"u8": 2**8 - 1, "u16": 2**16 - 1, "u32": 2**32 - 1}

# (section, [(key, type, min, max, default, unit, description)]): the daemon's configuration schema.
SCHEMA = (
    ("listener", [
        ("backlog", "u32", 16, 65535, 1024, "connections", "pending connections the kernel queues before accept"),
        ("max_connections", "u32", 1, 1000000, 10000, "connections", "open client connections across all workers"),
        ("accept_batch", "u16", 1, 1024, 64, "connections", "connections accepted per readiness event"),
        ("read_buffer_kib", "u32", 4, 16384, 64, "KiB", "per-connection receive buffer"),
        ("write_buffer_kib", "u32", 4, 16384, 64, "KiB", "per-connection send buffer"),
        ("idle_timeout_secs", "u32", 1, 86400, 300, "seconds", "time an idle keep-alive connection stays open"),
        ("header_timeout_ms", "u32", 100, 120000, 10000, "milliseconds", "time allowed to receive the request head"),
        ("max_header_kib", "u16", 1, 1024, 16, "KiB", "largest accepted request head"),
    ]),
    ("upstream", [
        ("connect_timeout_ms", "u32", 10, 120000, 3000, "milliseconds", "time allowed to open an upstream connection"),
        ("request_timeout_ms", "u32", 100, 600000, 30000, "milliseconds", "time allowed for a whole upstream exchange"),
        ("max_idle_per_host", "u16", 0, 4096, 32, "connections", "idle pooled connections kept per upstream host"),
        ("pool_size", "u16", 1, 4096, 64, "connections", "upstream connections opened per host"),
        ("keepalive_secs", "u32", 1, 7200, 60, "seconds", "TCP keep-alive probe interval on upstream sockets"),
        ("retries", "u8", 0, 16, 2, "attempts", "extra attempts for idempotent upstream requests"),
        ("retry_backoff_ms", "u32", 1, 60000, 250, "milliseconds", "base delay between upstream retries"),
        ("weight", "u16", 1, 1000, 100, "share", "relative load-balancing weight of this upstream"),
    ]),
    ("health_check", [
        ("interval_secs", "u32", 1, 3600, 10, "seconds", "time between active health probes"),
        ("timeout_ms", "u32", 10, 60000, 2000, "milliseconds", "time allowed for one health probe"),
        ("healthy_threshold", "u8", 1, 32, 2, "probes", "consecutive passes that mark a host healthy"),
        ("unhealthy_threshold", "u8", 1, 32, 3, "probes", "consecutive failures that mark a host unhealthy"),
        ("expected_status", "u16", 100, 599, 200, "HTTP status", "status code a passing probe returns"),
        ("jitter_percent", "u8", 0, 100, 10, "percent", "random spread added to the probe interval"),
    ]),
    ("tls", [
        ("handshake_timeout_ms", "u32", 100, 60000, 5000, "milliseconds", "time allowed for a TLS handshake"),
        ("session_cache_entries", "u32", 0, 1000000, 20480, "sessions", "resumable sessions kept in memory"),
        ("session_ticket_lifetime_secs", "u32", 60, 604800, 7200, "seconds", "lifetime of issued session tickets"),
        ("min_version_code", "u16", 769, 772, 771, "protocol code", "oldest accepted protocol version (0x0301 to 0x0304)"),
        ("max_cert_chain_depth", "u8", 1, 16, 8, "certificates", "longest accepted client certificate chain"),
        ("ocsp_refresh_secs", "u32", 60, 604800, 3600, "seconds", "time between stapled OCSP response refreshes"),
        ("record_size_limit", "u16", 64, 16385, 16385, "bytes", "largest TLS record this side sends"),
    ]),
    ("cache", [
        ("capacity_mib", "u32", 1, 1048576, 512, "MiB", "memory reserved for cached responses"),
        ("max_object_kib", "u32", 1, 1048576, 1024, "KiB", "largest response body that is cached"),
        ("default_ttl_secs", "u32", 0, 2592000, 300, "seconds", "freshness lifetime when the origin sets none"),
        ("stale_while_revalidate_secs", "u32", 0, 86400, 30, "seconds", "time a stale entry is served while it refreshes"),
        ("shards", "u16", 1, 1024, 16, "shards", "independently locked cache partitions"),
        ("eviction_batch", "u16", 1, 4096, 128, "entries", "entries evicted per pass when the cache is full"),
        ("admission_window_percent", "u8", 1, 100, 1, "percent", "share of capacity used as the admission window"),
    ]),
    ("rate_limit", [
        ("requests_per_second", "u32", 1, 1000000, 100, "requests", "sustained request rate per client"),
        ("burst", "u32", 1, 1000000, 200, "requests", "requests a client may send above the sustained rate"),
        ("window_secs", "u32", 1, 3600, 60, "seconds", "accounting window of the limiter"),
        ("max_tracked_clients", "u32", 1, 10000000, 100000, "clients", "clients tracked before the oldest are dropped"),
        ("penalty_secs", "u32", 0, 86400, 30, "seconds", "time a client stays blocked after exceeding its burst"),
        ("status_code", "u16", 400, 599, 429, "HTTP status", "status code returned to a limited client"),
    ]),
    ("dns", [
        ("resolver_timeout_ms", "u32", 10, 30000, 2000, "milliseconds", "time allowed for one resolver query"),
        ("attempts", "u8", 1, 10, 2, "attempts", "queries sent per name before resolution fails"),
        ("cache_entries", "u32", 0, 1000000, 4096, "records", "resolved records kept in memory"),
        ("min_ttl_secs", "u32", 0, 86400, 5, "seconds", "lower bound applied to record lifetimes"),
        ("max_ttl_secs", "u32", 1, 604800, 3600, "seconds", "upper bound applied to record lifetimes"),
        ("negative_ttl_secs", "u32", 0, 86400, 30, "seconds", "lifetime of a cached failed lookup"),
        ("ndots", "u8", 0, 15, 1, "dots", "dots in a name before it is tried as absolute first"),
    ]),
    ("runtime", [
        ("worker_threads", "u16", 1, 1024, 8, "threads", "threads that run connection tasks"),
        ("blocking_threads", "u16", 1, 4096, 512, "threads", "threads reserved for blocking file and DNS work"),
        ("queue_depth", "u32", 1, 1048576, 1024, "tasks", "tasks queued per worker before new work is refused"),
        ("stack_kib", "u32", 64, 65536, 2048, "KiB", "stack size of each worker thread"),
        ("shutdown_grace_secs", "u32", 0, 3600, 30, "seconds", "time in-flight requests get after a shutdown signal"),
        ("steal_batch", "u16", 1, 256, 32, "tasks", "tasks moved per work-stealing attempt"),
    ]),
)

HEADER = '''
/// Configuration schema of the daemon that the port parser above feeds.
///
/// The integration tests in this file cover `parse_port` only. This module is
/// reviewed together with them so that schema changes and parser changes land in
/// one place; it has no tests of its own and nothing above uses it.
#[allow(dead_code)]
mod netconf {
    use std::collections::BTreeMap;
    use std::fmt;

    /// Why one configuration value was refused.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ConfigError {
        /// The key is present but its value is empty.
        Empty { key: &'static str },
        /// The value is not a decimal number of the key's type.
        NotANumber { key: &'static str, raw: String },
        /// The value parsed but lies outside the key's inclusive bounds.
        OutOfRange { key: &'static str, min: u64, max: u64, value: u64 },
        /// The key, section or line is not part of the schema.
        Unknown { key: String },
    }

    impl fmt::Display for ConfigError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ConfigError::Empty { key } => write!(f, "{key}: empty value"),
                ConfigError::NotANumber { key, raw } => write!(f, "{key}: {raw:?} is not a decimal number"),
                ConfigError::OutOfRange { key, min, max, value } => {
                    write!(f, "{key}: {value} is outside {min}..={max}")
                }
                ConfigError::Unknown { key } => write!(f, "unknown setting {key:?}"),
            }
        }
    }

    impl std::error::Error for ConfigError {}

    /// Splits one `key = value` line; text after `#` is a comment.
    pub fn split_line(line: &str) -> Option<(&str, &str)> {
        let content = match line.find('#') {
            Some(index) => &line[..index],
            None => line,
        };
        let (key, value) = content.split_once('=')?;
        let key = key.trim();
        if key.is_empty() {
            return None;
        }
        Some((key, value.trim()))
    }

    /// Parses a whole configuration file into its `[section]` tables.
    pub fn parse_file(text: &str) -> Result<BTreeMap<String, BTreeMap<String, String>>, ConfigError> {
        let mut sections: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        let mut current = String::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
                current = name.trim().to_string();
                if !sections.contains_key(&current) {
                    sections.insert(current.clone(), BTreeMap::new());
                }
                continue;
            }
            match split_line(line) {
                Some((key, value)) => {
                    let table = sections.entry(current.clone()).or_insert_with(BTreeMap::new);
                    table.insert(key.to_string(), value.to_string());
                }
                None => return Err(ConfigError::Unknown { key: line.to_string() }),
            }
        }
        Ok(sections)
    }
'''

SETTING = Template('''
    /// Lower bound accepted for `$section.$key` ($unit).
    pub const ${CONST}_MIN: $type = $min;
    /// Upper bound accepted for `$section.$key` ($unit).
    pub const ${CONST}_MAX: $type = $max;
    /// Value used when `$section.$key` is absent.
    pub const ${CONST}_DEFAULT: $type = $default;

    /// Parses `$section.$key`: $description.
    ///
    /// Accepts decimal values in `$min..=$max` ($unit); surrounding whitespace is ignored.
    pub fn parse_${name}(raw: &str) -> Result<$type, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "$section.$key" });
        }
        let value: $type = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "$section.$key",
            raw: trimmed.to_string(),
        })?;
        if !(${CONST}_MIN..=${CONST}_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "$section.$key",
                min: u64::from(${CONST}_MIN),
                max: u64::from(${CONST}_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `$key` from a `[$section]` table, falling back to its default.
    pub fn ${name}_or_default(settings: &BTreeMap<String, String>) -> Result<$type, ConfigError> {
        match settings.get("$key") {
            Some(raw) => parse_${name}(raw),
            None => Ok(${CONST}_DEFAULT),
        }
    }

    /// Renders `$key` the way the configuration file stores it.
    pub fn format_${name}(value: $type) -> String {
        format!("$key = {} # $unit", value)
    }

    /// One-line help text for `$section.$key`.
    pub fn describe_${name}() -> &'static str {
        "$section.$key: $description ($unit, $min to $max, default $default)"
    }
''')

SECTION = Template('''
    /// The `[$section]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct $struct {
$fields    }

    impl Default for $struct {
        fn default() -> Self {
            Self {
$defaults            }
        }
    }

    impl $struct {
        /// Keys accepted in `[$section]`.
        pub const KEYS: &'static [&'static str] = &[$keys];

        /// Parses every key of `[$section]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("$section.{}", key) });
                }
            }
            Ok(Self {
$parsed            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[$section]\\n");
$rendered            out
        }
    }
''')

FOOTER = Template('''
    /// Sections of the daemon configuration, in file order.
    pub const SECTIONS: &[&str] = &[$sections];

    /// Validates one `[section]` table and renders it back in canonical form.
    pub fn normalize_section(name: &str, settings: &BTreeMap<String, String>) -> Result<String, ConfigError> {
        match name {
$arms            other => Err(ConfigError::Unknown { key: other.to_string() }),
        }
    }

    /// Validates a whole configuration file and renders it back in canonical form.
    pub fn normalize_file(text: &str) -> Result<String, ConfigError> {
        let sections = parse_file(text)?;
        let mut out = String::new();
        for (name, settings) in &sections {
            out.push_str(&normalize_section(name, settings)?);
            out.push('\\n');
        }
        Ok(out)
    }
}
''')

IDENT = re.compile(r"[a-z][a-z0-9_]*")
WORDS = re.compile(r"[A-Za-z0-9 ,()-]+")


def validate_schema(schema=SCHEMA):
    names = set()
    for section, settings in schema:
        require(IDENT.fullmatch(section) is not None and settings, "schema section")
        require(len({s[0] for s in settings}) == len(settings), "duplicate schema key")
        for key, kind, low, high, default, unit, description in settings:
            require(IDENT.fullmatch(key) is not None and f"{section}_{key}" not in names, "schema key")
            names.add(f"{section}_{key}")
            require(kind in MAXIMUM and all(type(v) is int for v in (low, high, default))
                    and 0 <= low <= default <= high <= MAXIMUM[kind] and low < high, "schema bounds")
            require(WORDS.fullmatch(unit) is not None and WORDS.fullmatch(description) is not None,
                    "schema text must need no Rust escaping")


def render(schema=SCHEMA):
    """The fixture text: deterministic, a pure function of SCHEMA and the templates above."""
    validate_schema(schema)
    parts = [HEADER]
    for section, settings in schema:
        for key, kind, low, high, default, unit, description in settings:
            name = f"{section}_{key}"
            parts.append(SETTING.substitute(section=section, key=key, name=name, CONST=name.upper(), type=kind,
                                            min=low, max=high, default=default, unit=unit,
                                            description=description))
        keys = [s[0] for s in settings]
        parts.append(SECTION.substitute(
            section=section, struct="".join(w.capitalize() for w in section.split("_")) + "Section",
            fields="".join(f"        pub {k}: {s[1]},\n" for k, s in zip(keys, settings)),
            defaults="".join(f"                {k}: {section.upper()}_{k.upper()}_DEFAULT,\n" for k in keys),
            keys=", ".join(f'"{k}"' for k in keys),
            parsed="".join(f"                {k}: {section}_{k}_or_default(settings)?,\n" for k in keys),
            rendered="".join(f"            out.push_str(&format_{section}_{k}(self.{k}));\n"
                             "            out.push('\\n');\n" for k in keys)))
    parts.append(FOOTER.substitute(
        sections=", ".join(f'"{s}"' for s, _ in schema),
        arms="".join(f'            "{s}" => Ok({"".join(w.capitalize() for w in s.split("_"))}Section::from_map'
                     f"(settings)?.render()),\n" for s, _ in schema)))
    return "".join(parts)


class Variant:
    """The selected workload variant: fixture files and its approval/summary binding."""

    def __init__(self, name, fixture_text, fixture_sha256):
        self.id, self.fixture_sha256 = name, fixture_sha256
        self.immutable = {**IMMUTABLE, "tests/ports.rs": IMMUTABLE["tests/ports.rs"] + fixture_text}
        self.initial = INITIAL
        self.read_limit = READ_LIMIT
        require(len(self.immutable["tests/ports.rs"].encode()) <= READ_LIMIT, "long-context tests exceed the read limit")

    def binding(self):
        """Approval keys; derived q5 cases set them."""
        return {"workload_variant": self.id, "workload_fixture_sha256": self.fixture_sha256}

    def record(self):
        """Client summary record."""
        return {"id": self.id, "fixture_sha256": self.fixture_sha256}


def select(name, path=FIXTURE):
    require(name in WORKLOADS, f"unknown workload variant {name!r} (known: {', '.join(WORKLOADS)})")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == FIXTURE_SHA256, "long-context fixture drift (regenerate and re-pin FIXTURE_SHA256)")
    return Variant(name, raw.decode("utf8", "strict"), digest)


def estimate(variant=None, family="qwen"):
    """Offline per-turn request size of one agent that follows the six-step workflow.

    Builds the seven actual requests (canonical initial messages, the variant's real file
    contents as read results, a canonical fix as the write, a passing test report, ~1 KiB of
    reasoning per reply) and returns [{"turn", "request_bytes", "prompt_tokens_estimate"}],
    the estimate being wire bytes / 4. Real tokenizers count Rust at about 3-3.5 bytes per
    token, so served prompt_tokens are expected 15-30% above the estimate."""
    from qwen38_n1_smoke import build_request, initial_messages
    from qwen38_n1_verdict import loads, wire
    from qwen_n4_live_smoke import identity
    files = {**IMMUTABLE, "src/ports.rs": INITIAL} if variant is None else \
        {**variant.immutable, "src/ports.rs": variant.initial}
    fixture = loads((Path(__file__).with_name("fixtures") / "canonical-coding-requests.json").read_bytes())
    initial = fixture[0]["request"]
    messages = initial_messages(initial["messages"], identity("agent-0", "long_context_estimate_0"))
    schemas = initial["tools"]
    for tool in schemas:
        tool["function"]["strict"] = False
    fixed = ('pub fn parse_port(s: &str) -> Result<u16, String> {\n'
             '    let value: u32 = s.parse().map_err(|e| format!("invalid port {s:?}: {e}"))?;\n'
             '    if value == 0 || value > 65535 {\n'
             '        return Err(format!("port {value} out of range 1..=65535"));\n'
             '    }\n'
             '    Ok(value as u16)\n'
             '}\n')
    report = "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
    rows = []
    for turn in range(7):
        body = wire(build_request("Qwen3.8 27B", messages, schemas, turn, family), family)
        rows.append({"turn": turn, "request_bytes": len(body), "prompt_tokens_estimate": len(body) // 4})
        reasoning = f"Step {turn + 1} of the workflow. " + "I check the previous result against the task. " * 22
        if turn == 6:
            messages.append({"role": "assistant", "content": "parse_port now returns Result<u16, String>; "
                             "ports and all tests passed.", "reasoning_content": reasoning})
            break
        name, args = EXPECTED[turn]
        if name == "write_file":
            args = {"path": "src/ports.rs", "content": fixed}
            files["src/ports.rs"] = fixed
        cid = f"call_{turn}"
        messages.append({"role": "assistant", "content": "", "reasoning_content": reasoning,
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": name, "arguments": json.dumps(args)}}]})
        if name == "read_file":
            result = files[args["path"]]
        elif name == "write_file":
            result = "write complete; sha256=" + hashlib.sha256(fixed.encode()).hexdigest()
        else:
            result = json.dumps({"target": args["target"], "passed": True,
                                 "source_sha256": hashlib.sha256(fixed.encode()).hexdigest(), "output": report})
        messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": result})
    return rows


def case_selection(case):
    """(client --workload value, approval workload_variant) of a Q5 case; (None, None) for the canonical files."""
    argv = case["client"]["argv"]
    flag = argv[argv.index("--workload") + 1] if "--workload" in argv[:-1] else None
    approved = ((case["client"].get("approval") or {}).get("set") or {}).get("workload_variant")
    return flag, approved


def derive_case(case, name=ID):
    """A Q5 case running `case`'s server and client on this workload variant; `case` is a canonical N4 case."""
    import q5_case
    q5_case.validate(case)
    require(name in WORKLOADS, f"unknown workload variant {name!r}")
    derived = json.loads(json.dumps(case))
    client = derived["client"]
    require(CLIENT in client["argv"] and not FOREIGN_OPTIONS & set(client["argv"]),
            "case client must be the canonical four-agent client without workload/instance/phase/profile options")
    approval = client.get("approval")
    require(approval is not None and not FOREIGN_KEYS & set(approval["set"]),
            "case approval must not already select a workload")
    sources = approval["set"].get("sources")
    require(isinstance(sources, dict) and set(sources) == {"sha256_of"} and isinstance(sources["sha256_of"], list),
            "case approval must freeze client sources with sha256_of")
    client["argv"] = client["argv"] + ["--workload", name]
    approval["set"]["workload_variant"] = name
    approval["set"]["workload_fixture_sha256"] = {"sha256_of": FIXTURE_REPO_PATH}
    sources["sha256_of"] = sources["sha256_of"] + [f"{{REPO}}/scripts/four_agent/{m}" for m in MODULES]
    derived["workload"]["id"] = f"{case['workload']['id']}-{name}"
    if isinstance(derived["workload"].get("client"), str):
        derived["workload"]["client"] += f" --workload {name}"
    derived["notes"] = list(case.get("notes", [])) + [
        f"Derived by coding_long_context.py derive-case: the client runs --workload {name}, the canonical parse_port "
        f"task whose tests/ports.rs also carries fixtures/{name}.rs, so every request from turn 2 on carries about "
        "40k prompt tokens of history (wire bytes / 4; served counts are expected higher, as Rust tokenizes at "
        "about 3-3.5 bytes per token). The workload id differs from the base case, so it calibrates separately."]
    return q5_case.validate(derived)


def main(argv):
    if argv == ["render"]:
        sys.stdout.write(render())
    elif argv == ["estimate"]:
        canonical, variant = estimate(), estimate(select(ID))
        print(json.dumps({"canonical": canonical, ID: variant, "wire_limit_bytes": WIRE_LIMIT}, indent=2))
    elif len(argv) == 2 and argv[0] == "derive-case":
        import q5_case
        case = q5_case.loads(Path(argv[1]).read_bytes())
        sys.stdout.write(json.dumps(derive_case(case), indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stderr.write(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
