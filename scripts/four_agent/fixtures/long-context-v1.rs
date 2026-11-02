
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

    /// Lower bound accepted for `listener.backlog` (connections).
    pub const LISTENER_BACKLOG_MIN: u32 = 16;
    /// Upper bound accepted for `listener.backlog` (connections).
    pub const LISTENER_BACKLOG_MAX: u32 = 65535;
    /// Value used when `listener.backlog` is absent.
    pub const LISTENER_BACKLOG_DEFAULT: u32 = 1024;

    /// Parses `listener.backlog`: pending connections the kernel queues before accept.
    ///
    /// Accepts decimal values in `16..=65535` (connections); surrounding whitespace is ignored.
    pub fn parse_listener_backlog(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "listener.backlog" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "listener.backlog",
            raw: trimmed.to_string(),
        })?;
        if !(LISTENER_BACKLOG_MIN..=LISTENER_BACKLOG_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "listener.backlog",
                min: u64::from(LISTENER_BACKLOG_MIN),
                max: u64::from(LISTENER_BACKLOG_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `backlog` from a `[listener]` table, falling back to its default.
    pub fn listener_backlog_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("backlog") {
            Some(raw) => parse_listener_backlog(raw),
            None => Ok(LISTENER_BACKLOG_DEFAULT),
        }
    }

    /// Renders `backlog` the way the configuration file stores it.
    pub fn format_listener_backlog(value: u32) -> String {
        format!("backlog = {} # connections", value)
    }

    /// One-line help text for `listener.backlog`.
    pub fn describe_listener_backlog() -> &'static str {
        "listener.backlog: pending connections the kernel queues before accept (connections, 16 to 65535, default 1024)"
    }

    /// Lower bound accepted for `listener.max_connections` (connections).
    pub const LISTENER_MAX_CONNECTIONS_MIN: u32 = 1;
    /// Upper bound accepted for `listener.max_connections` (connections).
    pub const LISTENER_MAX_CONNECTIONS_MAX: u32 = 1000000;
    /// Value used when `listener.max_connections` is absent.
    pub const LISTENER_MAX_CONNECTIONS_DEFAULT: u32 = 10000;

    /// Parses `listener.max_connections`: open client connections across all workers.
    ///
    /// Accepts decimal values in `1..=1000000` (connections); surrounding whitespace is ignored.
    pub fn parse_listener_max_connections(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "listener.max_connections" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "listener.max_connections",
            raw: trimmed.to_string(),
        })?;
        if !(LISTENER_MAX_CONNECTIONS_MIN..=LISTENER_MAX_CONNECTIONS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "listener.max_connections",
                min: u64::from(LISTENER_MAX_CONNECTIONS_MIN),
                max: u64::from(LISTENER_MAX_CONNECTIONS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `max_connections` from a `[listener]` table, falling back to its default.
    pub fn listener_max_connections_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("max_connections") {
            Some(raw) => parse_listener_max_connections(raw),
            None => Ok(LISTENER_MAX_CONNECTIONS_DEFAULT),
        }
    }

    /// Renders `max_connections` the way the configuration file stores it.
    pub fn format_listener_max_connections(value: u32) -> String {
        format!("max_connections = {} # connections", value)
    }

    /// One-line help text for `listener.max_connections`.
    pub fn describe_listener_max_connections() -> &'static str {
        "listener.max_connections: open client connections across all workers (connections, 1 to 1000000, default 10000)"
    }

    /// Lower bound accepted for `listener.accept_batch` (connections).
    pub const LISTENER_ACCEPT_BATCH_MIN: u16 = 1;
    /// Upper bound accepted for `listener.accept_batch` (connections).
    pub const LISTENER_ACCEPT_BATCH_MAX: u16 = 1024;
    /// Value used when `listener.accept_batch` is absent.
    pub const LISTENER_ACCEPT_BATCH_DEFAULT: u16 = 64;

    /// Parses `listener.accept_batch`: connections accepted per readiness event.
    ///
    /// Accepts decimal values in `1..=1024` (connections); surrounding whitespace is ignored.
    pub fn parse_listener_accept_batch(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "listener.accept_batch" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "listener.accept_batch",
            raw: trimmed.to_string(),
        })?;
        if !(LISTENER_ACCEPT_BATCH_MIN..=LISTENER_ACCEPT_BATCH_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "listener.accept_batch",
                min: u64::from(LISTENER_ACCEPT_BATCH_MIN),
                max: u64::from(LISTENER_ACCEPT_BATCH_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `accept_batch` from a `[listener]` table, falling back to its default.
    pub fn listener_accept_batch_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("accept_batch") {
            Some(raw) => parse_listener_accept_batch(raw),
            None => Ok(LISTENER_ACCEPT_BATCH_DEFAULT),
        }
    }

    /// Renders `accept_batch` the way the configuration file stores it.
    pub fn format_listener_accept_batch(value: u16) -> String {
        format!("accept_batch = {} # connections", value)
    }

    /// One-line help text for `listener.accept_batch`.
    pub fn describe_listener_accept_batch() -> &'static str {
        "listener.accept_batch: connections accepted per readiness event (connections, 1 to 1024, default 64)"
    }

    /// Lower bound accepted for `listener.read_buffer_kib` (KiB).
    pub const LISTENER_READ_BUFFER_KIB_MIN: u32 = 4;
    /// Upper bound accepted for `listener.read_buffer_kib` (KiB).
    pub const LISTENER_READ_BUFFER_KIB_MAX: u32 = 16384;
    /// Value used when `listener.read_buffer_kib` is absent.
    pub const LISTENER_READ_BUFFER_KIB_DEFAULT: u32 = 64;

    /// Parses `listener.read_buffer_kib`: per-connection receive buffer.
    ///
    /// Accepts decimal values in `4..=16384` (KiB); surrounding whitespace is ignored.
    pub fn parse_listener_read_buffer_kib(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "listener.read_buffer_kib" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "listener.read_buffer_kib",
            raw: trimmed.to_string(),
        })?;
        if !(LISTENER_READ_BUFFER_KIB_MIN..=LISTENER_READ_BUFFER_KIB_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "listener.read_buffer_kib",
                min: u64::from(LISTENER_READ_BUFFER_KIB_MIN),
                max: u64::from(LISTENER_READ_BUFFER_KIB_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `read_buffer_kib` from a `[listener]` table, falling back to its default.
    pub fn listener_read_buffer_kib_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("read_buffer_kib") {
            Some(raw) => parse_listener_read_buffer_kib(raw),
            None => Ok(LISTENER_READ_BUFFER_KIB_DEFAULT),
        }
    }

    /// Renders `read_buffer_kib` the way the configuration file stores it.
    pub fn format_listener_read_buffer_kib(value: u32) -> String {
        format!("read_buffer_kib = {} # KiB", value)
    }

    /// One-line help text for `listener.read_buffer_kib`.
    pub fn describe_listener_read_buffer_kib() -> &'static str {
        "listener.read_buffer_kib: per-connection receive buffer (KiB, 4 to 16384, default 64)"
    }

    /// Lower bound accepted for `listener.write_buffer_kib` (KiB).
    pub const LISTENER_WRITE_BUFFER_KIB_MIN: u32 = 4;
    /// Upper bound accepted for `listener.write_buffer_kib` (KiB).
    pub const LISTENER_WRITE_BUFFER_KIB_MAX: u32 = 16384;
    /// Value used when `listener.write_buffer_kib` is absent.
    pub const LISTENER_WRITE_BUFFER_KIB_DEFAULT: u32 = 64;

    /// Parses `listener.write_buffer_kib`: per-connection send buffer.
    ///
    /// Accepts decimal values in `4..=16384` (KiB); surrounding whitespace is ignored.
    pub fn parse_listener_write_buffer_kib(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "listener.write_buffer_kib" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "listener.write_buffer_kib",
            raw: trimmed.to_string(),
        })?;
        if !(LISTENER_WRITE_BUFFER_KIB_MIN..=LISTENER_WRITE_BUFFER_KIB_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "listener.write_buffer_kib",
                min: u64::from(LISTENER_WRITE_BUFFER_KIB_MIN),
                max: u64::from(LISTENER_WRITE_BUFFER_KIB_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `write_buffer_kib` from a `[listener]` table, falling back to its default.
    pub fn listener_write_buffer_kib_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("write_buffer_kib") {
            Some(raw) => parse_listener_write_buffer_kib(raw),
            None => Ok(LISTENER_WRITE_BUFFER_KIB_DEFAULT),
        }
    }

    /// Renders `write_buffer_kib` the way the configuration file stores it.
    pub fn format_listener_write_buffer_kib(value: u32) -> String {
        format!("write_buffer_kib = {} # KiB", value)
    }

    /// One-line help text for `listener.write_buffer_kib`.
    pub fn describe_listener_write_buffer_kib() -> &'static str {
        "listener.write_buffer_kib: per-connection send buffer (KiB, 4 to 16384, default 64)"
    }

    /// Lower bound accepted for `listener.idle_timeout_secs` (seconds).
    pub const LISTENER_IDLE_TIMEOUT_SECS_MIN: u32 = 1;
    /// Upper bound accepted for `listener.idle_timeout_secs` (seconds).
    pub const LISTENER_IDLE_TIMEOUT_SECS_MAX: u32 = 86400;
    /// Value used when `listener.idle_timeout_secs` is absent.
    pub const LISTENER_IDLE_TIMEOUT_SECS_DEFAULT: u32 = 300;

    /// Parses `listener.idle_timeout_secs`: time an idle keep-alive connection stays open.
    ///
    /// Accepts decimal values in `1..=86400` (seconds); surrounding whitespace is ignored.
    pub fn parse_listener_idle_timeout_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "listener.idle_timeout_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "listener.idle_timeout_secs",
            raw: trimmed.to_string(),
        })?;
        if !(LISTENER_IDLE_TIMEOUT_SECS_MIN..=LISTENER_IDLE_TIMEOUT_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "listener.idle_timeout_secs",
                min: u64::from(LISTENER_IDLE_TIMEOUT_SECS_MIN),
                max: u64::from(LISTENER_IDLE_TIMEOUT_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `idle_timeout_secs` from a `[listener]` table, falling back to its default.
    pub fn listener_idle_timeout_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("idle_timeout_secs") {
            Some(raw) => parse_listener_idle_timeout_secs(raw),
            None => Ok(LISTENER_IDLE_TIMEOUT_SECS_DEFAULT),
        }
    }

    /// Renders `idle_timeout_secs` the way the configuration file stores it.
    pub fn format_listener_idle_timeout_secs(value: u32) -> String {
        format!("idle_timeout_secs = {} # seconds", value)
    }

    /// One-line help text for `listener.idle_timeout_secs`.
    pub fn describe_listener_idle_timeout_secs() -> &'static str {
        "listener.idle_timeout_secs: time an idle keep-alive connection stays open (seconds, 1 to 86400, default 300)"
    }

    /// Lower bound accepted for `listener.header_timeout_ms` (milliseconds).
    pub const LISTENER_HEADER_TIMEOUT_MS_MIN: u32 = 100;
    /// Upper bound accepted for `listener.header_timeout_ms` (milliseconds).
    pub const LISTENER_HEADER_TIMEOUT_MS_MAX: u32 = 120000;
    /// Value used when `listener.header_timeout_ms` is absent.
    pub const LISTENER_HEADER_TIMEOUT_MS_DEFAULT: u32 = 10000;

    /// Parses `listener.header_timeout_ms`: time allowed to receive the request head.
    ///
    /// Accepts decimal values in `100..=120000` (milliseconds); surrounding whitespace is ignored.
    pub fn parse_listener_header_timeout_ms(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "listener.header_timeout_ms" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "listener.header_timeout_ms",
            raw: trimmed.to_string(),
        })?;
        if !(LISTENER_HEADER_TIMEOUT_MS_MIN..=LISTENER_HEADER_TIMEOUT_MS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "listener.header_timeout_ms",
                min: u64::from(LISTENER_HEADER_TIMEOUT_MS_MIN),
                max: u64::from(LISTENER_HEADER_TIMEOUT_MS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `header_timeout_ms` from a `[listener]` table, falling back to its default.
    pub fn listener_header_timeout_ms_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("header_timeout_ms") {
            Some(raw) => parse_listener_header_timeout_ms(raw),
            None => Ok(LISTENER_HEADER_TIMEOUT_MS_DEFAULT),
        }
    }

    /// Renders `header_timeout_ms` the way the configuration file stores it.
    pub fn format_listener_header_timeout_ms(value: u32) -> String {
        format!("header_timeout_ms = {} # milliseconds", value)
    }

    /// One-line help text for `listener.header_timeout_ms`.
    pub fn describe_listener_header_timeout_ms() -> &'static str {
        "listener.header_timeout_ms: time allowed to receive the request head (milliseconds, 100 to 120000, default 10000)"
    }

    /// Lower bound accepted for `listener.max_header_kib` (KiB).
    pub const LISTENER_MAX_HEADER_KIB_MIN: u16 = 1;
    /// Upper bound accepted for `listener.max_header_kib` (KiB).
    pub const LISTENER_MAX_HEADER_KIB_MAX: u16 = 1024;
    /// Value used when `listener.max_header_kib` is absent.
    pub const LISTENER_MAX_HEADER_KIB_DEFAULT: u16 = 16;

    /// Parses `listener.max_header_kib`: largest accepted request head.
    ///
    /// Accepts decimal values in `1..=1024` (KiB); surrounding whitespace is ignored.
    pub fn parse_listener_max_header_kib(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "listener.max_header_kib" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "listener.max_header_kib",
            raw: trimmed.to_string(),
        })?;
        if !(LISTENER_MAX_HEADER_KIB_MIN..=LISTENER_MAX_HEADER_KIB_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "listener.max_header_kib",
                min: u64::from(LISTENER_MAX_HEADER_KIB_MIN),
                max: u64::from(LISTENER_MAX_HEADER_KIB_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `max_header_kib` from a `[listener]` table, falling back to its default.
    pub fn listener_max_header_kib_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("max_header_kib") {
            Some(raw) => parse_listener_max_header_kib(raw),
            None => Ok(LISTENER_MAX_HEADER_KIB_DEFAULT),
        }
    }

    /// Renders `max_header_kib` the way the configuration file stores it.
    pub fn format_listener_max_header_kib(value: u16) -> String {
        format!("max_header_kib = {} # KiB", value)
    }

    /// One-line help text for `listener.max_header_kib`.
    pub fn describe_listener_max_header_kib() -> &'static str {
        "listener.max_header_kib: largest accepted request head (KiB, 1 to 1024, default 16)"
    }

    /// The `[listener]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ListenerSection {
        pub backlog: u32,
        pub max_connections: u32,
        pub accept_batch: u16,
        pub read_buffer_kib: u32,
        pub write_buffer_kib: u32,
        pub idle_timeout_secs: u32,
        pub header_timeout_ms: u32,
        pub max_header_kib: u16,
    }

    impl Default for ListenerSection {
        fn default() -> Self {
            Self {
                backlog: LISTENER_BACKLOG_DEFAULT,
                max_connections: LISTENER_MAX_CONNECTIONS_DEFAULT,
                accept_batch: LISTENER_ACCEPT_BATCH_DEFAULT,
                read_buffer_kib: LISTENER_READ_BUFFER_KIB_DEFAULT,
                write_buffer_kib: LISTENER_WRITE_BUFFER_KIB_DEFAULT,
                idle_timeout_secs: LISTENER_IDLE_TIMEOUT_SECS_DEFAULT,
                header_timeout_ms: LISTENER_HEADER_TIMEOUT_MS_DEFAULT,
                max_header_kib: LISTENER_MAX_HEADER_KIB_DEFAULT,
            }
        }
    }

    impl ListenerSection {
        /// Keys accepted in `[listener]`.
        pub const KEYS: &'static [&'static str] = &["backlog", "max_connections", "accept_batch", "read_buffer_kib", "write_buffer_kib", "idle_timeout_secs", "header_timeout_ms", "max_header_kib"];

        /// Parses every key of `[listener]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("listener.{}", key) });
                }
            }
            Ok(Self {
                backlog: listener_backlog_or_default(settings)?,
                max_connections: listener_max_connections_or_default(settings)?,
                accept_batch: listener_accept_batch_or_default(settings)?,
                read_buffer_kib: listener_read_buffer_kib_or_default(settings)?,
                write_buffer_kib: listener_write_buffer_kib_or_default(settings)?,
                idle_timeout_secs: listener_idle_timeout_secs_or_default(settings)?,
                header_timeout_ms: listener_header_timeout_ms_or_default(settings)?,
                max_header_kib: listener_max_header_kib_or_default(settings)?,
            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[listener]\n");
            out.push_str(&format_listener_backlog(self.backlog));
            out.push('\n');
            out.push_str(&format_listener_max_connections(self.max_connections));
            out.push('\n');
            out.push_str(&format_listener_accept_batch(self.accept_batch));
            out.push('\n');
            out.push_str(&format_listener_read_buffer_kib(self.read_buffer_kib));
            out.push('\n');
            out.push_str(&format_listener_write_buffer_kib(self.write_buffer_kib));
            out.push('\n');
            out.push_str(&format_listener_idle_timeout_secs(self.idle_timeout_secs));
            out.push('\n');
            out.push_str(&format_listener_header_timeout_ms(self.header_timeout_ms));
            out.push('\n');
            out.push_str(&format_listener_max_header_kib(self.max_header_kib));
            out.push('\n');
            out
        }
    }

    /// Lower bound accepted for `upstream.connect_timeout_ms` (milliseconds).
    pub const UPSTREAM_CONNECT_TIMEOUT_MS_MIN: u32 = 10;
    /// Upper bound accepted for `upstream.connect_timeout_ms` (milliseconds).
    pub const UPSTREAM_CONNECT_TIMEOUT_MS_MAX: u32 = 120000;
    /// Value used when `upstream.connect_timeout_ms` is absent.
    pub const UPSTREAM_CONNECT_TIMEOUT_MS_DEFAULT: u32 = 3000;

    /// Parses `upstream.connect_timeout_ms`: time allowed to open an upstream connection.
    ///
    /// Accepts decimal values in `10..=120000` (milliseconds); surrounding whitespace is ignored.
    pub fn parse_upstream_connect_timeout_ms(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "upstream.connect_timeout_ms" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "upstream.connect_timeout_ms",
            raw: trimmed.to_string(),
        })?;
        if !(UPSTREAM_CONNECT_TIMEOUT_MS_MIN..=UPSTREAM_CONNECT_TIMEOUT_MS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "upstream.connect_timeout_ms",
                min: u64::from(UPSTREAM_CONNECT_TIMEOUT_MS_MIN),
                max: u64::from(UPSTREAM_CONNECT_TIMEOUT_MS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `connect_timeout_ms` from a `[upstream]` table, falling back to its default.
    pub fn upstream_connect_timeout_ms_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("connect_timeout_ms") {
            Some(raw) => parse_upstream_connect_timeout_ms(raw),
            None => Ok(UPSTREAM_CONNECT_TIMEOUT_MS_DEFAULT),
        }
    }

    /// Renders `connect_timeout_ms` the way the configuration file stores it.
    pub fn format_upstream_connect_timeout_ms(value: u32) -> String {
        format!("connect_timeout_ms = {} # milliseconds", value)
    }

    /// One-line help text for `upstream.connect_timeout_ms`.
    pub fn describe_upstream_connect_timeout_ms() -> &'static str {
        "upstream.connect_timeout_ms: time allowed to open an upstream connection (milliseconds, 10 to 120000, default 3000)"
    }

    /// Lower bound accepted for `upstream.request_timeout_ms` (milliseconds).
    pub const UPSTREAM_REQUEST_TIMEOUT_MS_MIN: u32 = 100;
    /// Upper bound accepted for `upstream.request_timeout_ms` (milliseconds).
    pub const UPSTREAM_REQUEST_TIMEOUT_MS_MAX: u32 = 600000;
    /// Value used when `upstream.request_timeout_ms` is absent.
    pub const UPSTREAM_REQUEST_TIMEOUT_MS_DEFAULT: u32 = 30000;

    /// Parses `upstream.request_timeout_ms`: time allowed for a whole upstream exchange.
    ///
    /// Accepts decimal values in `100..=600000` (milliseconds); surrounding whitespace is ignored.
    pub fn parse_upstream_request_timeout_ms(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "upstream.request_timeout_ms" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "upstream.request_timeout_ms",
            raw: trimmed.to_string(),
        })?;
        if !(UPSTREAM_REQUEST_TIMEOUT_MS_MIN..=UPSTREAM_REQUEST_TIMEOUT_MS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "upstream.request_timeout_ms",
                min: u64::from(UPSTREAM_REQUEST_TIMEOUT_MS_MIN),
                max: u64::from(UPSTREAM_REQUEST_TIMEOUT_MS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `request_timeout_ms` from a `[upstream]` table, falling back to its default.
    pub fn upstream_request_timeout_ms_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("request_timeout_ms") {
            Some(raw) => parse_upstream_request_timeout_ms(raw),
            None => Ok(UPSTREAM_REQUEST_TIMEOUT_MS_DEFAULT),
        }
    }

    /// Renders `request_timeout_ms` the way the configuration file stores it.
    pub fn format_upstream_request_timeout_ms(value: u32) -> String {
        format!("request_timeout_ms = {} # milliseconds", value)
    }

    /// One-line help text for `upstream.request_timeout_ms`.
    pub fn describe_upstream_request_timeout_ms() -> &'static str {
        "upstream.request_timeout_ms: time allowed for a whole upstream exchange (milliseconds, 100 to 600000, default 30000)"
    }

    /// Lower bound accepted for `upstream.max_idle_per_host` (connections).
    pub const UPSTREAM_MAX_IDLE_PER_HOST_MIN: u16 = 0;
    /// Upper bound accepted for `upstream.max_idle_per_host` (connections).
    pub const UPSTREAM_MAX_IDLE_PER_HOST_MAX: u16 = 4096;
    /// Value used when `upstream.max_idle_per_host` is absent.
    pub const UPSTREAM_MAX_IDLE_PER_HOST_DEFAULT: u16 = 32;

    /// Parses `upstream.max_idle_per_host`: idle pooled connections kept per upstream host.
    ///
    /// Accepts decimal values in `0..=4096` (connections); surrounding whitespace is ignored.
    pub fn parse_upstream_max_idle_per_host(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "upstream.max_idle_per_host" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "upstream.max_idle_per_host",
            raw: trimmed.to_string(),
        })?;
        if !(UPSTREAM_MAX_IDLE_PER_HOST_MIN..=UPSTREAM_MAX_IDLE_PER_HOST_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "upstream.max_idle_per_host",
                min: u64::from(UPSTREAM_MAX_IDLE_PER_HOST_MIN),
                max: u64::from(UPSTREAM_MAX_IDLE_PER_HOST_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `max_idle_per_host` from a `[upstream]` table, falling back to its default.
    pub fn upstream_max_idle_per_host_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("max_idle_per_host") {
            Some(raw) => parse_upstream_max_idle_per_host(raw),
            None => Ok(UPSTREAM_MAX_IDLE_PER_HOST_DEFAULT),
        }
    }

    /// Renders `max_idle_per_host` the way the configuration file stores it.
    pub fn format_upstream_max_idle_per_host(value: u16) -> String {
        format!("max_idle_per_host = {} # connections", value)
    }

    /// One-line help text for `upstream.max_idle_per_host`.
    pub fn describe_upstream_max_idle_per_host() -> &'static str {
        "upstream.max_idle_per_host: idle pooled connections kept per upstream host (connections, 0 to 4096, default 32)"
    }

    /// Lower bound accepted for `upstream.pool_size` (connections).
    pub const UPSTREAM_POOL_SIZE_MIN: u16 = 1;
    /// Upper bound accepted for `upstream.pool_size` (connections).
    pub const UPSTREAM_POOL_SIZE_MAX: u16 = 4096;
    /// Value used when `upstream.pool_size` is absent.
    pub const UPSTREAM_POOL_SIZE_DEFAULT: u16 = 64;

    /// Parses `upstream.pool_size`: upstream connections opened per host.
    ///
    /// Accepts decimal values in `1..=4096` (connections); surrounding whitespace is ignored.
    pub fn parse_upstream_pool_size(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "upstream.pool_size" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "upstream.pool_size",
            raw: trimmed.to_string(),
        })?;
        if !(UPSTREAM_POOL_SIZE_MIN..=UPSTREAM_POOL_SIZE_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "upstream.pool_size",
                min: u64::from(UPSTREAM_POOL_SIZE_MIN),
                max: u64::from(UPSTREAM_POOL_SIZE_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `pool_size` from a `[upstream]` table, falling back to its default.
    pub fn upstream_pool_size_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("pool_size") {
            Some(raw) => parse_upstream_pool_size(raw),
            None => Ok(UPSTREAM_POOL_SIZE_DEFAULT),
        }
    }

    /// Renders `pool_size` the way the configuration file stores it.
    pub fn format_upstream_pool_size(value: u16) -> String {
        format!("pool_size = {} # connections", value)
    }

    /// One-line help text for `upstream.pool_size`.
    pub fn describe_upstream_pool_size() -> &'static str {
        "upstream.pool_size: upstream connections opened per host (connections, 1 to 4096, default 64)"
    }

    /// Lower bound accepted for `upstream.keepalive_secs` (seconds).
    pub const UPSTREAM_KEEPALIVE_SECS_MIN: u32 = 1;
    /// Upper bound accepted for `upstream.keepalive_secs` (seconds).
    pub const UPSTREAM_KEEPALIVE_SECS_MAX: u32 = 7200;
    /// Value used when `upstream.keepalive_secs` is absent.
    pub const UPSTREAM_KEEPALIVE_SECS_DEFAULT: u32 = 60;

    /// Parses `upstream.keepalive_secs`: TCP keep-alive probe interval on upstream sockets.
    ///
    /// Accepts decimal values in `1..=7200` (seconds); surrounding whitespace is ignored.
    pub fn parse_upstream_keepalive_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "upstream.keepalive_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "upstream.keepalive_secs",
            raw: trimmed.to_string(),
        })?;
        if !(UPSTREAM_KEEPALIVE_SECS_MIN..=UPSTREAM_KEEPALIVE_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "upstream.keepalive_secs",
                min: u64::from(UPSTREAM_KEEPALIVE_SECS_MIN),
                max: u64::from(UPSTREAM_KEEPALIVE_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `keepalive_secs` from a `[upstream]` table, falling back to its default.
    pub fn upstream_keepalive_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("keepalive_secs") {
            Some(raw) => parse_upstream_keepalive_secs(raw),
            None => Ok(UPSTREAM_KEEPALIVE_SECS_DEFAULT),
        }
    }

    /// Renders `keepalive_secs` the way the configuration file stores it.
    pub fn format_upstream_keepalive_secs(value: u32) -> String {
        format!("keepalive_secs = {} # seconds", value)
    }

    /// One-line help text for `upstream.keepalive_secs`.
    pub fn describe_upstream_keepalive_secs() -> &'static str {
        "upstream.keepalive_secs: TCP keep-alive probe interval on upstream sockets (seconds, 1 to 7200, default 60)"
    }

    /// Lower bound accepted for `upstream.retries` (attempts).
    pub const UPSTREAM_RETRIES_MIN: u8 = 0;
    /// Upper bound accepted for `upstream.retries` (attempts).
    pub const UPSTREAM_RETRIES_MAX: u8 = 16;
    /// Value used when `upstream.retries` is absent.
    pub const UPSTREAM_RETRIES_DEFAULT: u8 = 2;

    /// Parses `upstream.retries`: extra attempts for idempotent upstream requests.
    ///
    /// Accepts decimal values in `0..=16` (attempts); surrounding whitespace is ignored.
    pub fn parse_upstream_retries(raw: &str) -> Result<u8, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "upstream.retries" });
        }
        let value: u8 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "upstream.retries",
            raw: trimmed.to_string(),
        })?;
        if !(UPSTREAM_RETRIES_MIN..=UPSTREAM_RETRIES_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "upstream.retries",
                min: u64::from(UPSTREAM_RETRIES_MIN),
                max: u64::from(UPSTREAM_RETRIES_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `retries` from a `[upstream]` table, falling back to its default.
    pub fn upstream_retries_or_default(settings: &BTreeMap<String, String>) -> Result<u8, ConfigError> {
        match settings.get("retries") {
            Some(raw) => parse_upstream_retries(raw),
            None => Ok(UPSTREAM_RETRIES_DEFAULT),
        }
    }

    /// Renders `retries` the way the configuration file stores it.
    pub fn format_upstream_retries(value: u8) -> String {
        format!("retries = {} # attempts", value)
    }

    /// One-line help text for `upstream.retries`.
    pub fn describe_upstream_retries() -> &'static str {
        "upstream.retries: extra attempts for idempotent upstream requests (attempts, 0 to 16, default 2)"
    }

    /// Lower bound accepted for `upstream.retry_backoff_ms` (milliseconds).
    pub const UPSTREAM_RETRY_BACKOFF_MS_MIN: u32 = 1;
    /// Upper bound accepted for `upstream.retry_backoff_ms` (milliseconds).
    pub const UPSTREAM_RETRY_BACKOFF_MS_MAX: u32 = 60000;
    /// Value used when `upstream.retry_backoff_ms` is absent.
    pub const UPSTREAM_RETRY_BACKOFF_MS_DEFAULT: u32 = 250;

    /// Parses `upstream.retry_backoff_ms`: base delay between upstream retries.
    ///
    /// Accepts decimal values in `1..=60000` (milliseconds); surrounding whitespace is ignored.
    pub fn parse_upstream_retry_backoff_ms(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "upstream.retry_backoff_ms" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "upstream.retry_backoff_ms",
            raw: trimmed.to_string(),
        })?;
        if !(UPSTREAM_RETRY_BACKOFF_MS_MIN..=UPSTREAM_RETRY_BACKOFF_MS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "upstream.retry_backoff_ms",
                min: u64::from(UPSTREAM_RETRY_BACKOFF_MS_MIN),
                max: u64::from(UPSTREAM_RETRY_BACKOFF_MS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `retry_backoff_ms` from a `[upstream]` table, falling back to its default.
    pub fn upstream_retry_backoff_ms_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("retry_backoff_ms") {
            Some(raw) => parse_upstream_retry_backoff_ms(raw),
            None => Ok(UPSTREAM_RETRY_BACKOFF_MS_DEFAULT),
        }
    }

    /// Renders `retry_backoff_ms` the way the configuration file stores it.
    pub fn format_upstream_retry_backoff_ms(value: u32) -> String {
        format!("retry_backoff_ms = {} # milliseconds", value)
    }

    /// One-line help text for `upstream.retry_backoff_ms`.
    pub fn describe_upstream_retry_backoff_ms() -> &'static str {
        "upstream.retry_backoff_ms: base delay between upstream retries (milliseconds, 1 to 60000, default 250)"
    }

    /// Lower bound accepted for `upstream.weight` (share).
    pub const UPSTREAM_WEIGHT_MIN: u16 = 1;
    /// Upper bound accepted for `upstream.weight` (share).
    pub const UPSTREAM_WEIGHT_MAX: u16 = 1000;
    /// Value used when `upstream.weight` is absent.
    pub const UPSTREAM_WEIGHT_DEFAULT: u16 = 100;

    /// Parses `upstream.weight`: relative load-balancing weight of this upstream.
    ///
    /// Accepts decimal values in `1..=1000` (share); surrounding whitespace is ignored.
    pub fn parse_upstream_weight(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "upstream.weight" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "upstream.weight",
            raw: trimmed.to_string(),
        })?;
        if !(UPSTREAM_WEIGHT_MIN..=UPSTREAM_WEIGHT_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "upstream.weight",
                min: u64::from(UPSTREAM_WEIGHT_MIN),
                max: u64::from(UPSTREAM_WEIGHT_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `weight` from a `[upstream]` table, falling back to its default.
    pub fn upstream_weight_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("weight") {
            Some(raw) => parse_upstream_weight(raw),
            None => Ok(UPSTREAM_WEIGHT_DEFAULT),
        }
    }

    /// Renders `weight` the way the configuration file stores it.
    pub fn format_upstream_weight(value: u16) -> String {
        format!("weight = {} # share", value)
    }

    /// One-line help text for `upstream.weight`.
    pub fn describe_upstream_weight() -> &'static str {
        "upstream.weight: relative load-balancing weight of this upstream (share, 1 to 1000, default 100)"
    }

    /// The `[upstream]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct UpstreamSection {
        pub connect_timeout_ms: u32,
        pub request_timeout_ms: u32,
        pub max_idle_per_host: u16,
        pub pool_size: u16,
        pub keepalive_secs: u32,
        pub retries: u8,
        pub retry_backoff_ms: u32,
        pub weight: u16,
    }

    impl Default for UpstreamSection {
        fn default() -> Self {
            Self {
                connect_timeout_ms: UPSTREAM_CONNECT_TIMEOUT_MS_DEFAULT,
                request_timeout_ms: UPSTREAM_REQUEST_TIMEOUT_MS_DEFAULT,
                max_idle_per_host: UPSTREAM_MAX_IDLE_PER_HOST_DEFAULT,
                pool_size: UPSTREAM_POOL_SIZE_DEFAULT,
                keepalive_secs: UPSTREAM_KEEPALIVE_SECS_DEFAULT,
                retries: UPSTREAM_RETRIES_DEFAULT,
                retry_backoff_ms: UPSTREAM_RETRY_BACKOFF_MS_DEFAULT,
                weight: UPSTREAM_WEIGHT_DEFAULT,
            }
        }
    }

    impl UpstreamSection {
        /// Keys accepted in `[upstream]`.
        pub const KEYS: &'static [&'static str] = &["connect_timeout_ms", "request_timeout_ms", "max_idle_per_host", "pool_size", "keepalive_secs", "retries", "retry_backoff_ms", "weight"];

        /// Parses every key of `[upstream]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("upstream.{}", key) });
                }
            }
            Ok(Self {
                connect_timeout_ms: upstream_connect_timeout_ms_or_default(settings)?,
                request_timeout_ms: upstream_request_timeout_ms_or_default(settings)?,
                max_idle_per_host: upstream_max_idle_per_host_or_default(settings)?,
                pool_size: upstream_pool_size_or_default(settings)?,
                keepalive_secs: upstream_keepalive_secs_or_default(settings)?,
                retries: upstream_retries_or_default(settings)?,
                retry_backoff_ms: upstream_retry_backoff_ms_or_default(settings)?,
                weight: upstream_weight_or_default(settings)?,
            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[upstream]\n");
            out.push_str(&format_upstream_connect_timeout_ms(self.connect_timeout_ms));
            out.push('\n');
            out.push_str(&format_upstream_request_timeout_ms(self.request_timeout_ms));
            out.push('\n');
            out.push_str(&format_upstream_max_idle_per_host(self.max_idle_per_host));
            out.push('\n');
            out.push_str(&format_upstream_pool_size(self.pool_size));
            out.push('\n');
            out.push_str(&format_upstream_keepalive_secs(self.keepalive_secs));
            out.push('\n');
            out.push_str(&format_upstream_retries(self.retries));
            out.push('\n');
            out.push_str(&format_upstream_retry_backoff_ms(self.retry_backoff_ms));
            out.push('\n');
            out.push_str(&format_upstream_weight(self.weight));
            out.push('\n');
            out
        }
    }

    /// Lower bound accepted for `health_check.interval_secs` (seconds).
    pub const HEALTH_CHECK_INTERVAL_SECS_MIN: u32 = 1;
    /// Upper bound accepted for `health_check.interval_secs` (seconds).
    pub const HEALTH_CHECK_INTERVAL_SECS_MAX: u32 = 3600;
    /// Value used when `health_check.interval_secs` is absent.
    pub const HEALTH_CHECK_INTERVAL_SECS_DEFAULT: u32 = 10;

    /// Parses `health_check.interval_secs`: time between active health probes.
    ///
    /// Accepts decimal values in `1..=3600` (seconds); surrounding whitespace is ignored.
    pub fn parse_health_check_interval_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "health_check.interval_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "health_check.interval_secs",
            raw: trimmed.to_string(),
        })?;
        if !(HEALTH_CHECK_INTERVAL_SECS_MIN..=HEALTH_CHECK_INTERVAL_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "health_check.interval_secs",
                min: u64::from(HEALTH_CHECK_INTERVAL_SECS_MIN),
                max: u64::from(HEALTH_CHECK_INTERVAL_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `interval_secs` from a `[health_check]` table, falling back to its default.
    pub fn health_check_interval_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("interval_secs") {
            Some(raw) => parse_health_check_interval_secs(raw),
            None => Ok(HEALTH_CHECK_INTERVAL_SECS_DEFAULT),
        }
    }

    /// Renders `interval_secs` the way the configuration file stores it.
    pub fn format_health_check_interval_secs(value: u32) -> String {
        format!("interval_secs = {} # seconds", value)
    }

    /// One-line help text for `health_check.interval_secs`.
    pub fn describe_health_check_interval_secs() -> &'static str {
        "health_check.interval_secs: time between active health probes (seconds, 1 to 3600, default 10)"
    }

    /// Lower bound accepted for `health_check.timeout_ms` (milliseconds).
    pub const HEALTH_CHECK_TIMEOUT_MS_MIN: u32 = 10;
    /// Upper bound accepted for `health_check.timeout_ms` (milliseconds).
    pub const HEALTH_CHECK_TIMEOUT_MS_MAX: u32 = 60000;
    /// Value used when `health_check.timeout_ms` is absent.
    pub const HEALTH_CHECK_TIMEOUT_MS_DEFAULT: u32 = 2000;

    /// Parses `health_check.timeout_ms`: time allowed for one health probe.
    ///
    /// Accepts decimal values in `10..=60000` (milliseconds); surrounding whitespace is ignored.
    pub fn parse_health_check_timeout_ms(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "health_check.timeout_ms" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "health_check.timeout_ms",
            raw: trimmed.to_string(),
        })?;
        if !(HEALTH_CHECK_TIMEOUT_MS_MIN..=HEALTH_CHECK_TIMEOUT_MS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "health_check.timeout_ms",
                min: u64::from(HEALTH_CHECK_TIMEOUT_MS_MIN),
                max: u64::from(HEALTH_CHECK_TIMEOUT_MS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `timeout_ms` from a `[health_check]` table, falling back to its default.
    pub fn health_check_timeout_ms_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("timeout_ms") {
            Some(raw) => parse_health_check_timeout_ms(raw),
            None => Ok(HEALTH_CHECK_TIMEOUT_MS_DEFAULT),
        }
    }

    /// Renders `timeout_ms` the way the configuration file stores it.
    pub fn format_health_check_timeout_ms(value: u32) -> String {
        format!("timeout_ms = {} # milliseconds", value)
    }

    /// One-line help text for `health_check.timeout_ms`.
    pub fn describe_health_check_timeout_ms() -> &'static str {
        "health_check.timeout_ms: time allowed for one health probe (milliseconds, 10 to 60000, default 2000)"
    }

    /// Lower bound accepted for `health_check.healthy_threshold` (probes).
    pub const HEALTH_CHECK_HEALTHY_THRESHOLD_MIN: u8 = 1;
    /// Upper bound accepted for `health_check.healthy_threshold` (probes).
    pub const HEALTH_CHECK_HEALTHY_THRESHOLD_MAX: u8 = 32;
    /// Value used when `health_check.healthy_threshold` is absent.
    pub const HEALTH_CHECK_HEALTHY_THRESHOLD_DEFAULT: u8 = 2;

    /// Parses `health_check.healthy_threshold`: consecutive passes that mark a host healthy.
    ///
    /// Accepts decimal values in `1..=32` (probes); surrounding whitespace is ignored.
    pub fn parse_health_check_healthy_threshold(raw: &str) -> Result<u8, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "health_check.healthy_threshold" });
        }
        let value: u8 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "health_check.healthy_threshold",
            raw: trimmed.to_string(),
        })?;
        if !(HEALTH_CHECK_HEALTHY_THRESHOLD_MIN..=HEALTH_CHECK_HEALTHY_THRESHOLD_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "health_check.healthy_threshold",
                min: u64::from(HEALTH_CHECK_HEALTHY_THRESHOLD_MIN),
                max: u64::from(HEALTH_CHECK_HEALTHY_THRESHOLD_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `healthy_threshold` from a `[health_check]` table, falling back to its default.
    pub fn health_check_healthy_threshold_or_default(settings: &BTreeMap<String, String>) -> Result<u8, ConfigError> {
        match settings.get("healthy_threshold") {
            Some(raw) => parse_health_check_healthy_threshold(raw),
            None => Ok(HEALTH_CHECK_HEALTHY_THRESHOLD_DEFAULT),
        }
    }

    /// Renders `healthy_threshold` the way the configuration file stores it.
    pub fn format_health_check_healthy_threshold(value: u8) -> String {
        format!("healthy_threshold = {} # probes", value)
    }

    /// One-line help text for `health_check.healthy_threshold`.
    pub fn describe_health_check_healthy_threshold() -> &'static str {
        "health_check.healthy_threshold: consecutive passes that mark a host healthy (probes, 1 to 32, default 2)"
    }

    /// Lower bound accepted for `health_check.unhealthy_threshold` (probes).
    pub const HEALTH_CHECK_UNHEALTHY_THRESHOLD_MIN: u8 = 1;
    /// Upper bound accepted for `health_check.unhealthy_threshold` (probes).
    pub const HEALTH_CHECK_UNHEALTHY_THRESHOLD_MAX: u8 = 32;
    /// Value used when `health_check.unhealthy_threshold` is absent.
    pub const HEALTH_CHECK_UNHEALTHY_THRESHOLD_DEFAULT: u8 = 3;

    /// Parses `health_check.unhealthy_threshold`: consecutive failures that mark a host unhealthy.
    ///
    /// Accepts decimal values in `1..=32` (probes); surrounding whitespace is ignored.
    pub fn parse_health_check_unhealthy_threshold(raw: &str) -> Result<u8, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "health_check.unhealthy_threshold" });
        }
        let value: u8 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "health_check.unhealthy_threshold",
            raw: trimmed.to_string(),
        })?;
        if !(HEALTH_CHECK_UNHEALTHY_THRESHOLD_MIN..=HEALTH_CHECK_UNHEALTHY_THRESHOLD_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "health_check.unhealthy_threshold",
                min: u64::from(HEALTH_CHECK_UNHEALTHY_THRESHOLD_MIN),
                max: u64::from(HEALTH_CHECK_UNHEALTHY_THRESHOLD_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `unhealthy_threshold` from a `[health_check]` table, falling back to its default.
    pub fn health_check_unhealthy_threshold_or_default(settings: &BTreeMap<String, String>) -> Result<u8, ConfigError> {
        match settings.get("unhealthy_threshold") {
            Some(raw) => parse_health_check_unhealthy_threshold(raw),
            None => Ok(HEALTH_CHECK_UNHEALTHY_THRESHOLD_DEFAULT),
        }
    }

    /// Renders `unhealthy_threshold` the way the configuration file stores it.
    pub fn format_health_check_unhealthy_threshold(value: u8) -> String {
        format!("unhealthy_threshold = {} # probes", value)
    }

    /// One-line help text for `health_check.unhealthy_threshold`.
    pub fn describe_health_check_unhealthy_threshold() -> &'static str {
        "health_check.unhealthy_threshold: consecutive failures that mark a host unhealthy (probes, 1 to 32, default 3)"
    }

    /// Lower bound accepted for `health_check.expected_status` (HTTP status).
    pub const HEALTH_CHECK_EXPECTED_STATUS_MIN: u16 = 100;
    /// Upper bound accepted for `health_check.expected_status` (HTTP status).
    pub const HEALTH_CHECK_EXPECTED_STATUS_MAX: u16 = 599;
    /// Value used when `health_check.expected_status` is absent.
    pub const HEALTH_CHECK_EXPECTED_STATUS_DEFAULT: u16 = 200;

    /// Parses `health_check.expected_status`: status code a passing probe returns.
    ///
    /// Accepts decimal values in `100..=599` (HTTP status); surrounding whitespace is ignored.
    pub fn parse_health_check_expected_status(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "health_check.expected_status" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "health_check.expected_status",
            raw: trimmed.to_string(),
        })?;
        if !(HEALTH_CHECK_EXPECTED_STATUS_MIN..=HEALTH_CHECK_EXPECTED_STATUS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "health_check.expected_status",
                min: u64::from(HEALTH_CHECK_EXPECTED_STATUS_MIN),
                max: u64::from(HEALTH_CHECK_EXPECTED_STATUS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `expected_status` from a `[health_check]` table, falling back to its default.
    pub fn health_check_expected_status_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("expected_status") {
            Some(raw) => parse_health_check_expected_status(raw),
            None => Ok(HEALTH_CHECK_EXPECTED_STATUS_DEFAULT),
        }
    }

    /// Renders `expected_status` the way the configuration file stores it.
    pub fn format_health_check_expected_status(value: u16) -> String {
        format!("expected_status = {} # HTTP status", value)
    }

    /// One-line help text for `health_check.expected_status`.
    pub fn describe_health_check_expected_status() -> &'static str {
        "health_check.expected_status: status code a passing probe returns (HTTP status, 100 to 599, default 200)"
    }

    /// Lower bound accepted for `health_check.jitter_percent` (percent).
    pub const HEALTH_CHECK_JITTER_PERCENT_MIN: u8 = 0;
    /// Upper bound accepted for `health_check.jitter_percent` (percent).
    pub const HEALTH_CHECK_JITTER_PERCENT_MAX: u8 = 100;
    /// Value used when `health_check.jitter_percent` is absent.
    pub const HEALTH_CHECK_JITTER_PERCENT_DEFAULT: u8 = 10;

    /// Parses `health_check.jitter_percent`: random spread added to the probe interval.
    ///
    /// Accepts decimal values in `0..=100` (percent); surrounding whitespace is ignored.
    pub fn parse_health_check_jitter_percent(raw: &str) -> Result<u8, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "health_check.jitter_percent" });
        }
        let value: u8 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "health_check.jitter_percent",
            raw: trimmed.to_string(),
        })?;
        if !(HEALTH_CHECK_JITTER_PERCENT_MIN..=HEALTH_CHECK_JITTER_PERCENT_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "health_check.jitter_percent",
                min: u64::from(HEALTH_CHECK_JITTER_PERCENT_MIN),
                max: u64::from(HEALTH_CHECK_JITTER_PERCENT_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `jitter_percent` from a `[health_check]` table, falling back to its default.
    pub fn health_check_jitter_percent_or_default(settings: &BTreeMap<String, String>) -> Result<u8, ConfigError> {
        match settings.get("jitter_percent") {
            Some(raw) => parse_health_check_jitter_percent(raw),
            None => Ok(HEALTH_CHECK_JITTER_PERCENT_DEFAULT),
        }
    }

    /// Renders `jitter_percent` the way the configuration file stores it.
    pub fn format_health_check_jitter_percent(value: u8) -> String {
        format!("jitter_percent = {} # percent", value)
    }

    /// One-line help text for `health_check.jitter_percent`.
    pub fn describe_health_check_jitter_percent() -> &'static str {
        "health_check.jitter_percent: random spread added to the probe interval (percent, 0 to 100, default 10)"
    }

    /// The `[health_check]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct HealthCheckSection {
        pub interval_secs: u32,
        pub timeout_ms: u32,
        pub healthy_threshold: u8,
        pub unhealthy_threshold: u8,
        pub expected_status: u16,
        pub jitter_percent: u8,
    }

    impl Default for HealthCheckSection {
        fn default() -> Self {
            Self {
                interval_secs: HEALTH_CHECK_INTERVAL_SECS_DEFAULT,
                timeout_ms: HEALTH_CHECK_TIMEOUT_MS_DEFAULT,
                healthy_threshold: HEALTH_CHECK_HEALTHY_THRESHOLD_DEFAULT,
                unhealthy_threshold: HEALTH_CHECK_UNHEALTHY_THRESHOLD_DEFAULT,
                expected_status: HEALTH_CHECK_EXPECTED_STATUS_DEFAULT,
                jitter_percent: HEALTH_CHECK_JITTER_PERCENT_DEFAULT,
            }
        }
    }

    impl HealthCheckSection {
        /// Keys accepted in `[health_check]`.
        pub const KEYS: &'static [&'static str] = &["interval_secs", "timeout_ms", "healthy_threshold", "unhealthy_threshold", "expected_status", "jitter_percent"];

        /// Parses every key of `[health_check]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("health_check.{}", key) });
                }
            }
            Ok(Self {
                interval_secs: health_check_interval_secs_or_default(settings)?,
                timeout_ms: health_check_timeout_ms_or_default(settings)?,
                healthy_threshold: health_check_healthy_threshold_or_default(settings)?,
                unhealthy_threshold: health_check_unhealthy_threshold_or_default(settings)?,
                expected_status: health_check_expected_status_or_default(settings)?,
                jitter_percent: health_check_jitter_percent_or_default(settings)?,
            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[health_check]\n");
            out.push_str(&format_health_check_interval_secs(self.interval_secs));
            out.push('\n');
            out.push_str(&format_health_check_timeout_ms(self.timeout_ms));
            out.push('\n');
            out.push_str(&format_health_check_healthy_threshold(self.healthy_threshold));
            out.push('\n');
            out.push_str(&format_health_check_unhealthy_threshold(self.unhealthy_threshold));
            out.push('\n');
            out.push_str(&format_health_check_expected_status(self.expected_status));
            out.push('\n');
            out.push_str(&format_health_check_jitter_percent(self.jitter_percent));
            out.push('\n');
            out
        }
    }

    /// Lower bound accepted for `tls.handshake_timeout_ms` (milliseconds).
    pub const TLS_HANDSHAKE_TIMEOUT_MS_MIN: u32 = 100;
    /// Upper bound accepted for `tls.handshake_timeout_ms` (milliseconds).
    pub const TLS_HANDSHAKE_TIMEOUT_MS_MAX: u32 = 60000;
    /// Value used when `tls.handshake_timeout_ms` is absent.
    pub const TLS_HANDSHAKE_TIMEOUT_MS_DEFAULT: u32 = 5000;

    /// Parses `tls.handshake_timeout_ms`: time allowed for a TLS handshake.
    ///
    /// Accepts decimal values in `100..=60000` (milliseconds); surrounding whitespace is ignored.
    pub fn parse_tls_handshake_timeout_ms(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "tls.handshake_timeout_ms" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "tls.handshake_timeout_ms",
            raw: trimmed.to_string(),
        })?;
        if !(TLS_HANDSHAKE_TIMEOUT_MS_MIN..=TLS_HANDSHAKE_TIMEOUT_MS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "tls.handshake_timeout_ms",
                min: u64::from(TLS_HANDSHAKE_TIMEOUT_MS_MIN),
                max: u64::from(TLS_HANDSHAKE_TIMEOUT_MS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `handshake_timeout_ms` from a `[tls]` table, falling back to its default.
    pub fn tls_handshake_timeout_ms_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("handshake_timeout_ms") {
            Some(raw) => parse_tls_handshake_timeout_ms(raw),
            None => Ok(TLS_HANDSHAKE_TIMEOUT_MS_DEFAULT),
        }
    }

    /// Renders `handshake_timeout_ms` the way the configuration file stores it.
    pub fn format_tls_handshake_timeout_ms(value: u32) -> String {
        format!("handshake_timeout_ms = {} # milliseconds", value)
    }

    /// One-line help text for `tls.handshake_timeout_ms`.
    pub fn describe_tls_handshake_timeout_ms() -> &'static str {
        "tls.handshake_timeout_ms: time allowed for a TLS handshake (milliseconds, 100 to 60000, default 5000)"
    }

    /// Lower bound accepted for `tls.session_cache_entries` (sessions).
    pub const TLS_SESSION_CACHE_ENTRIES_MIN: u32 = 0;
    /// Upper bound accepted for `tls.session_cache_entries` (sessions).
    pub const TLS_SESSION_CACHE_ENTRIES_MAX: u32 = 1000000;
    /// Value used when `tls.session_cache_entries` is absent.
    pub const TLS_SESSION_CACHE_ENTRIES_DEFAULT: u32 = 20480;

    /// Parses `tls.session_cache_entries`: resumable sessions kept in memory.
    ///
    /// Accepts decimal values in `0..=1000000` (sessions); surrounding whitespace is ignored.
    pub fn parse_tls_session_cache_entries(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "tls.session_cache_entries" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "tls.session_cache_entries",
            raw: trimmed.to_string(),
        })?;
        if !(TLS_SESSION_CACHE_ENTRIES_MIN..=TLS_SESSION_CACHE_ENTRIES_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "tls.session_cache_entries",
                min: u64::from(TLS_SESSION_CACHE_ENTRIES_MIN),
                max: u64::from(TLS_SESSION_CACHE_ENTRIES_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `session_cache_entries` from a `[tls]` table, falling back to its default.
    pub fn tls_session_cache_entries_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("session_cache_entries") {
            Some(raw) => parse_tls_session_cache_entries(raw),
            None => Ok(TLS_SESSION_CACHE_ENTRIES_DEFAULT),
        }
    }

    /// Renders `session_cache_entries` the way the configuration file stores it.
    pub fn format_tls_session_cache_entries(value: u32) -> String {
        format!("session_cache_entries = {} # sessions", value)
    }

    /// One-line help text for `tls.session_cache_entries`.
    pub fn describe_tls_session_cache_entries() -> &'static str {
        "tls.session_cache_entries: resumable sessions kept in memory (sessions, 0 to 1000000, default 20480)"
    }

    /// Lower bound accepted for `tls.session_ticket_lifetime_secs` (seconds).
    pub const TLS_SESSION_TICKET_LIFETIME_SECS_MIN: u32 = 60;
    /// Upper bound accepted for `tls.session_ticket_lifetime_secs` (seconds).
    pub const TLS_SESSION_TICKET_LIFETIME_SECS_MAX: u32 = 604800;
    /// Value used when `tls.session_ticket_lifetime_secs` is absent.
    pub const TLS_SESSION_TICKET_LIFETIME_SECS_DEFAULT: u32 = 7200;

    /// Parses `tls.session_ticket_lifetime_secs`: lifetime of issued session tickets.
    ///
    /// Accepts decimal values in `60..=604800` (seconds); surrounding whitespace is ignored.
    pub fn parse_tls_session_ticket_lifetime_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "tls.session_ticket_lifetime_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "tls.session_ticket_lifetime_secs",
            raw: trimmed.to_string(),
        })?;
        if !(TLS_SESSION_TICKET_LIFETIME_SECS_MIN..=TLS_SESSION_TICKET_LIFETIME_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "tls.session_ticket_lifetime_secs",
                min: u64::from(TLS_SESSION_TICKET_LIFETIME_SECS_MIN),
                max: u64::from(TLS_SESSION_TICKET_LIFETIME_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `session_ticket_lifetime_secs` from a `[tls]` table, falling back to its default.
    pub fn tls_session_ticket_lifetime_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("session_ticket_lifetime_secs") {
            Some(raw) => parse_tls_session_ticket_lifetime_secs(raw),
            None => Ok(TLS_SESSION_TICKET_LIFETIME_SECS_DEFAULT),
        }
    }

    /// Renders `session_ticket_lifetime_secs` the way the configuration file stores it.
    pub fn format_tls_session_ticket_lifetime_secs(value: u32) -> String {
        format!("session_ticket_lifetime_secs = {} # seconds", value)
    }

    /// One-line help text for `tls.session_ticket_lifetime_secs`.
    pub fn describe_tls_session_ticket_lifetime_secs() -> &'static str {
        "tls.session_ticket_lifetime_secs: lifetime of issued session tickets (seconds, 60 to 604800, default 7200)"
    }

    /// Lower bound accepted for `tls.min_version_code` (protocol code).
    pub const TLS_MIN_VERSION_CODE_MIN: u16 = 769;
    /// Upper bound accepted for `tls.min_version_code` (protocol code).
    pub const TLS_MIN_VERSION_CODE_MAX: u16 = 772;
    /// Value used when `tls.min_version_code` is absent.
    pub const TLS_MIN_VERSION_CODE_DEFAULT: u16 = 771;

    /// Parses `tls.min_version_code`: oldest accepted protocol version (0x0301 to 0x0304).
    ///
    /// Accepts decimal values in `769..=772` (protocol code); surrounding whitespace is ignored.
    pub fn parse_tls_min_version_code(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "tls.min_version_code" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "tls.min_version_code",
            raw: trimmed.to_string(),
        })?;
        if !(TLS_MIN_VERSION_CODE_MIN..=TLS_MIN_VERSION_CODE_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "tls.min_version_code",
                min: u64::from(TLS_MIN_VERSION_CODE_MIN),
                max: u64::from(TLS_MIN_VERSION_CODE_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `min_version_code` from a `[tls]` table, falling back to its default.
    pub fn tls_min_version_code_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("min_version_code") {
            Some(raw) => parse_tls_min_version_code(raw),
            None => Ok(TLS_MIN_VERSION_CODE_DEFAULT),
        }
    }

    /// Renders `min_version_code` the way the configuration file stores it.
    pub fn format_tls_min_version_code(value: u16) -> String {
        format!("min_version_code = {} # protocol code", value)
    }

    /// One-line help text for `tls.min_version_code`.
    pub fn describe_tls_min_version_code() -> &'static str {
        "tls.min_version_code: oldest accepted protocol version (0x0301 to 0x0304) (protocol code, 769 to 772, default 771)"
    }

    /// Lower bound accepted for `tls.max_cert_chain_depth` (certificates).
    pub const TLS_MAX_CERT_CHAIN_DEPTH_MIN: u8 = 1;
    /// Upper bound accepted for `tls.max_cert_chain_depth` (certificates).
    pub const TLS_MAX_CERT_CHAIN_DEPTH_MAX: u8 = 16;
    /// Value used when `tls.max_cert_chain_depth` is absent.
    pub const TLS_MAX_CERT_CHAIN_DEPTH_DEFAULT: u8 = 8;

    /// Parses `tls.max_cert_chain_depth`: longest accepted client certificate chain.
    ///
    /// Accepts decimal values in `1..=16` (certificates); surrounding whitespace is ignored.
    pub fn parse_tls_max_cert_chain_depth(raw: &str) -> Result<u8, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "tls.max_cert_chain_depth" });
        }
        let value: u8 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "tls.max_cert_chain_depth",
            raw: trimmed.to_string(),
        })?;
        if !(TLS_MAX_CERT_CHAIN_DEPTH_MIN..=TLS_MAX_CERT_CHAIN_DEPTH_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "tls.max_cert_chain_depth",
                min: u64::from(TLS_MAX_CERT_CHAIN_DEPTH_MIN),
                max: u64::from(TLS_MAX_CERT_CHAIN_DEPTH_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `max_cert_chain_depth` from a `[tls]` table, falling back to its default.
    pub fn tls_max_cert_chain_depth_or_default(settings: &BTreeMap<String, String>) -> Result<u8, ConfigError> {
        match settings.get("max_cert_chain_depth") {
            Some(raw) => parse_tls_max_cert_chain_depth(raw),
            None => Ok(TLS_MAX_CERT_CHAIN_DEPTH_DEFAULT),
        }
    }

    /// Renders `max_cert_chain_depth` the way the configuration file stores it.
    pub fn format_tls_max_cert_chain_depth(value: u8) -> String {
        format!("max_cert_chain_depth = {} # certificates", value)
    }

    /// One-line help text for `tls.max_cert_chain_depth`.
    pub fn describe_tls_max_cert_chain_depth() -> &'static str {
        "tls.max_cert_chain_depth: longest accepted client certificate chain (certificates, 1 to 16, default 8)"
    }

    /// Lower bound accepted for `tls.ocsp_refresh_secs` (seconds).
    pub const TLS_OCSP_REFRESH_SECS_MIN: u32 = 60;
    /// Upper bound accepted for `tls.ocsp_refresh_secs` (seconds).
    pub const TLS_OCSP_REFRESH_SECS_MAX: u32 = 604800;
    /// Value used when `tls.ocsp_refresh_secs` is absent.
    pub const TLS_OCSP_REFRESH_SECS_DEFAULT: u32 = 3600;

    /// Parses `tls.ocsp_refresh_secs`: time between stapled OCSP response refreshes.
    ///
    /// Accepts decimal values in `60..=604800` (seconds); surrounding whitespace is ignored.
    pub fn parse_tls_ocsp_refresh_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "tls.ocsp_refresh_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "tls.ocsp_refresh_secs",
            raw: trimmed.to_string(),
        })?;
        if !(TLS_OCSP_REFRESH_SECS_MIN..=TLS_OCSP_REFRESH_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "tls.ocsp_refresh_secs",
                min: u64::from(TLS_OCSP_REFRESH_SECS_MIN),
                max: u64::from(TLS_OCSP_REFRESH_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `ocsp_refresh_secs` from a `[tls]` table, falling back to its default.
    pub fn tls_ocsp_refresh_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("ocsp_refresh_secs") {
            Some(raw) => parse_tls_ocsp_refresh_secs(raw),
            None => Ok(TLS_OCSP_REFRESH_SECS_DEFAULT),
        }
    }

    /// Renders `ocsp_refresh_secs` the way the configuration file stores it.
    pub fn format_tls_ocsp_refresh_secs(value: u32) -> String {
        format!("ocsp_refresh_secs = {} # seconds", value)
    }

    /// One-line help text for `tls.ocsp_refresh_secs`.
    pub fn describe_tls_ocsp_refresh_secs() -> &'static str {
        "tls.ocsp_refresh_secs: time between stapled OCSP response refreshes (seconds, 60 to 604800, default 3600)"
    }

    /// Lower bound accepted for `tls.record_size_limit` (bytes).
    pub const TLS_RECORD_SIZE_LIMIT_MIN: u16 = 64;
    /// Upper bound accepted for `tls.record_size_limit` (bytes).
    pub const TLS_RECORD_SIZE_LIMIT_MAX: u16 = 16385;
    /// Value used when `tls.record_size_limit` is absent.
    pub const TLS_RECORD_SIZE_LIMIT_DEFAULT: u16 = 16385;

    /// Parses `tls.record_size_limit`: largest TLS record this side sends.
    ///
    /// Accepts decimal values in `64..=16385` (bytes); surrounding whitespace is ignored.
    pub fn parse_tls_record_size_limit(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "tls.record_size_limit" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "tls.record_size_limit",
            raw: trimmed.to_string(),
        })?;
        if !(TLS_RECORD_SIZE_LIMIT_MIN..=TLS_RECORD_SIZE_LIMIT_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "tls.record_size_limit",
                min: u64::from(TLS_RECORD_SIZE_LIMIT_MIN),
                max: u64::from(TLS_RECORD_SIZE_LIMIT_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `record_size_limit` from a `[tls]` table, falling back to its default.
    pub fn tls_record_size_limit_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("record_size_limit") {
            Some(raw) => parse_tls_record_size_limit(raw),
            None => Ok(TLS_RECORD_SIZE_LIMIT_DEFAULT),
        }
    }

    /// Renders `record_size_limit` the way the configuration file stores it.
    pub fn format_tls_record_size_limit(value: u16) -> String {
        format!("record_size_limit = {} # bytes", value)
    }

    /// One-line help text for `tls.record_size_limit`.
    pub fn describe_tls_record_size_limit() -> &'static str {
        "tls.record_size_limit: largest TLS record this side sends (bytes, 64 to 16385, default 16385)"
    }

    /// The `[tls]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct TlsSection {
        pub handshake_timeout_ms: u32,
        pub session_cache_entries: u32,
        pub session_ticket_lifetime_secs: u32,
        pub min_version_code: u16,
        pub max_cert_chain_depth: u8,
        pub ocsp_refresh_secs: u32,
        pub record_size_limit: u16,
    }

    impl Default for TlsSection {
        fn default() -> Self {
            Self {
                handshake_timeout_ms: TLS_HANDSHAKE_TIMEOUT_MS_DEFAULT,
                session_cache_entries: TLS_SESSION_CACHE_ENTRIES_DEFAULT,
                session_ticket_lifetime_secs: TLS_SESSION_TICKET_LIFETIME_SECS_DEFAULT,
                min_version_code: TLS_MIN_VERSION_CODE_DEFAULT,
                max_cert_chain_depth: TLS_MAX_CERT_CHAIN_DEPTH_DEFAULT,
                ocsp_refresh_secs: TLS_OCSP_REFRESH_SECS_DEFAULT,
                record_size_limit: TLS_RECORD_SIZE_LIMIT_DEFAULT,
            }
        }
    }

    impl TlsSection {
        /// Keys accepted in `[tls]`.
        pub const KEYS: &'static [&'static str] = &["handshake_timeout_ms", "session_cache_entries", "session_ticket_lifetime_secs", "min_version_code", "max_cert_chain_depth", "ocsp_refresh_secs", "record_size_limit"];

        /// Parses every key of `[tls]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("tls.{}", key) });
                }
            }
            Ok(Self {
                handshake_timeout_ms: tls_handshake_timeout_ms_or_default(settings)?,
                session_cache_entries: tls_session_cache_entries_or_default(settings)?,
                session_ticket_lifetime_secs: tls_session_ticket_lifetime_secs_or_default(settings)?,
                min_version_code: tls_min_version_code_or_default(settings)?,
                max_cert_chain_depth: tls_max_cert_chain_depth_or_default(settings)?,
                ocsp_refresh_secs: tls_ocsp_refresh_secs_or_default(settings)?,
                record_size_limit: tls_record_size_limit_or_default(settings)?,
            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[tls]\n");
            out.push_str(&format_tls_handshake_timeout_ms(self.handshake_timeout_ms));
            out.push('\n');
            out.push_str(&format_tls_session_cache_entries(self.session_cache_entries));
            out.push('\n');
            out.push_str(&format_tls_session_ticket_lifetime_secs(self.session_ticket_lifetime_secs));
            out.push('\n');
            out.push_str(&format_tls_min_version_code(self.min_version_code));
            out.push('\n');
            out.push_str(&format_tls_max_cert_chain_depth(self.max_cert_chain_depth));
            out.push('\n');
            out.push_str(&format_tls_ocsp_refresh_secs(self.ocsp_refresh_secs));
            out.push('\n');
            out.push_str(&format_tls_record_size_limit(self.record_size_limit));
            out.push('\n');
            out
        }
    }

    /// Lower bound accepted for `cache.capacity_mib` (MiB).
    pub const CACHE_CAPACITY_MIB_MIN: u32 = 1;
    /// Upper bound accepted for `cache.capacity_mib` (MiB).
    pub const CACHE_CAPACITY_MIB_MAX: u32 = 1048576;
    /// Value used when `cache.capacity_mib` is absent.
    pub const CACHE_CAPACITY_MIB_DEFAULT: u32 = 512;

    /// Parses `cache.capacity_mib`: memory reserved for cached responses.
    ///
    /// Accepts decimal values in `1..=1048576` (MiB); surrounding whitespace is ignored.
    pub fn parse_cache_capacity_mib(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "cache.capacity_mib" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "cache.capacity_mib",
            raw: trimmed.to_string(),
        })?;
        if !(CACHE_CAPACITY_MIB_MIN..=CACHE_CAPACITY_MIB_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "cache.capacity_mib",
                min: u64::from(CACHE_CAPACITY_MIB_MIN),
                max: u64::from(CACHE_CAPACITY_MIB_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `capacity_mib` from a `[cache]` table, falling back to its default.
    pub fn cache_capacity_mib_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("capacity_mib") {
            Some(raw) => parse_cache_capacity_mib(raw),
            None => Ok(CACHE_CAPACITY_MIB_DEFAULT),
        }
    }

    /// Renders `capacity_mib` the way the configuration file stores it.
    pub fn format_cache_capacity_mib(value: u32) -> String {
        format!("capacity_mib = {} # MiB", value)
    }

    /// One-line help text for `cache.capacity_mib`.
    pub fn describe_cache_capacity_mib() -> &'static str {
        "cache.capacity_mib: memory reserved for cached responses (MiB, 1 to 1048576, default 512)"
    }

    /// Lower bound accepted for `cache.max_object_kib` (KiB).
    pub const CACHE_MAX_OBJECT_KIB_MIN: u32 = 1;
    /// Upper bound accepted for `cache.max_object_kib` (KiB).
    pub const CACHE_MAX_OBJECT_KIB_MAX: u32 = 1048576;
    /// Value used when `cache.max_object_kib` is absent.
    pub const CACHE_MAX_OBJECT_KIB_DEFAULT: u32 = 1024;

    /// Parses `cache.max_object_kib`: largest response body that is cached.
    ///
    /// Accepts decimal values in `1..=1048576` (KiB); surrounding whitespace is ignored.
    pub fn parse_cache_max_object_kib(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "cache.max_object_kib" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "cache.max_object_kib",
            raw: trimmed.to_string(),
        })?;
        if !(CACHE_MAX_OBJECT_KIB_MIN..=CACHE_MAX_OBJECT_KIB_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "cache.max_object_kib",
                min: u64::from(CACHE_MAX_OBJECT_KIB_MIN),
                max: u64::from(CACHE_MAX_OBJECT_KIB_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `max_object_kib` from a `[cache]` table, falling back to its default.
    pub fn cache_max_object_kib_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("max_object_kib") {
            Some(raw) => parse_cache_max_object_kib(raw),
            None => Ok(CACHE_MAX_OBJECT_KIB_DEFAULT),
        }
    }

    /// Renders `max_object_kib` the way the configuration file stores it.
    pub fn format_cache_max_object_kib(value: u32) -> String {
        format!("max_object_kib = {} # KiB", value)
    }

    /// One-line help text for `cache.max_object_kib`.
    pub fn describe_cache_max_object_kib() -> &'static str {
        "cache.max_object_kib: largest response body that is cached (KiB, 1 to 1048576, default 1024)"
    }

    /// Lower bound accepted for `cache.default_ttl_secs` (seconds).
    pub const CACHE_DEFAULT_TTL_SECS_MIN: u32 = 0;
    /// Upper bound accepted for `cache.default_ttl_secs` (seconds).
    pub const CACHE_DEFAULT_TTL_SECS_MAX: u32 = 2592000;
    /// Value used when `cache.default_ttl_secs` is absent.
    pub const CACHE_DEFAULT_TTL_SECS_DEFAULT: u32 = 300;

    /// Parses `cache.default_ttl_secs`: freshness lifetime when the origin sets none.
    ///
    /// Accepts decimal values in `0..=2592000` (seconds); surrounding whitespace is ignored.
    pub fn parse_cache_default_ttl_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "cache.default_ttl_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "cache.default_ttl_secs",
            raw: trimmed.to_string(),
        })?;
        if !(CACHE_DEFAULT_TTL_SECS_MIN..=CACHE_DEFAULT_TTL_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "cache.default_ttl_secs",
                min: u64::from(CACHE_DEFAULT_TTL_SECS_MIN),
                max: u64::from(CACHE_DEFAULT_TTL_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `default_ttl_secs` from a `[cache]` table, falling back to its default.
    pub fn cache_default_ttl_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("default_ttl_secs") {
            Some(raw) => parse_cache_default_ttl_secs(raw),
            None => Ok(CACHE_DEFAULT_TTL_SECS_DEFAULT),
        }
    }

    /// Renders `default_ttl_secs` the way the configuration file stores it.
    pub fn format_cache_default_ttl_secs(value: u32) -> String {
        format!("default_ttl_secs = {} # seconds", value)
    }

    /// One-line help text for `cache.default_ttl_secs`.
    pub fn describe_cache_default_ttl_secs() -> &'static str {
        "cache.default_ttl_secs: freshness lifetime when the origin sets none (seconds, 0 to 2592000, default 300)"
    }

    /// Lower bound accepted for `cache.stale_while_revalidate_secs` (seconds).
    pub const CACHE_STALE_WHILE_REVALIDATE_SECS_MIN: u32 = 0;
    /// Upper bound accepted for `cache.stale_while_revalidate_secs` (seconds).
    pub const CACHE_STALE_WHILE_REVALIDATE_SECS_MAX: u32 = 86400;
    /// Value used when `cache.stale_while_revalidate_secs` is absent.
    pub const CACHE_STALE_WHILE_REVALIDATE_SECS_DEFAULT: u32 = 30;

    /// Parses `cache.stale_while_revalidate_secs`: time a stale entry is served while it refreshes.
    ///
    /// Accepts decimal values in `0..=86400` (seconds); surrounding whitespace is ignored.
    pub fn parse_cache_stale_while_revalidate_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "cache.stale_while_revalidate_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "cache.stale_while_revalidate_secs",
            raw: trimmed.to_string(),
        })?;
        if !(CACHE_STALE_WHILE_REVALIDATE_SECS_MIN..=CACHE_STALE_WHILE_REVALIDATE_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "cache.stale_while_revalidate_secs",
                min: u64::from(CACHE_STALE_WHILE_REVALIDATE_SECS_MIN),
                max: u64::from(CACHE_STALE_WHILE_REVALIDATE_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `stale_while_revalidate_secs` from a `[cache]` table, falling back to its default.
    pub fn cache_stale_while_revalidate_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("stale_while_revalidate_secs") {
            Some(raw) => parse_cache_stale_while_revalidate_secs(raw),
            None => Ok(CACHE_STALE_WHILE_REVALIDATE_SECS_DEFAULT),
        }
    }

    /// Renders `stale_while_revalidate_secs` the way the configuration file stores it.
    pub fn format_cache_stale_while_revalidate_secs(value: u32) -> String {
        format!("stale_while_revalidate_secs = {} # seconds", value)
    }

    /// One-line help text for `cache.stale_while_revalidate_secs`.
    pub fn describe_cache_stale_while_revalidate_secs() -> &'static str {
        "cache.stale_while_revalidate_secs: time a stale entry is served while it refreshes (seconds, 0 to 86400, default 30)"
    }

    /// Lower bound accepted for `cache.shards` (shards).
    pub const CACHE_SHARDS_MIN: u16 = 1;
    /// Upper bound accepted for `cache.shards` (shards).
    pub const CACHE_SHARDS_MAX: u16 = 1024;
    /// Value used when `cache.shards` is absent.
    pub const CACHE_SHARDS_DEFAULT: u16 = 16;

    /// Parses `cache.shards`: independently locked cache partitions.
    ///
    /// Accepts decimal values in `1..=1024` (shards); surrounding whitespace is ignored.
    pub fn parse_cache_shards(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "cache.shards" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "cache.shards",
            raw: trimmed.to_string(),
        })?;
        if !(CACHE_SHARDS_MIN..=CACHE_SHARDS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "cache.shards",
                min: u64::from(CACHE_SHARDS_MIN),
                max: u64::from(CACHE_SHARDS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `shards` from a `[cache]` table, falling back to its default.
    pub fn cache_shards_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("shards") {
            Some(raw) => parse_cache_shards(raw),
            None => Ok(CACHE_SHARDS_DEFAULT),
        }
    }

    /// Renders `shards` the way the configuration file stores it.
    pub fn format_cache_shards(value: u16) -> String {
        format!("shards = {} # shards", value)
    }

    /// One-line help text for `cache.shards`.
    pub fn describe_cache_shards() -> &'static str {
        "cache.shards: independently locked cache partitions (shards, 1 to 1024, default 16)"
    }

    /// Lower bound accepted for `cache.eviction_batch` (entries).
    pub const CACHE_EVICTION_BATCH_MIN: u16 = 1;
    /// Upper bound accepted for `cache.eviction_batch` (entries).
    pub const CACHE_EVICTION_BATCH_MAX: u16 = 4096;
    /// Value used when `cache.eviction_batch` is absent.
    pub const CACHE_EVICTION_BATCH_DEFAULT: u16 = 128;

    /// Parses `cache.eviction_batch`: entries evicted per pass when the cache is full.
    ///
    /// Accepts decimal values in `1..=4096` (entries); surrounding whitespace is ignored.
    pub fn parse_cache_eviction_batch(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "cache.eviction_batch" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "cache.eviction_batch",
            raw: trimmed.to_string(),
        })?;
        if !(CACHE_EVICTION_BATCH_MIN..=CACHE_EVICTION_BATCH_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "cache.eviction_batch",
                min: u64::from(CACHE_EVICTION_BATCH_MIN),
                max: u64::from(CACHE_EVICTION_BATCH_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `eviction_batch` from a `[cache]` table, falling back to its default.
    pub fn cache_eviction_batch_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("eviction_batch") {
            Some(raw) => parse_cache_eviction_batch(raw),
            None => Ok(CACHE_EVICTION_BATCH_DEFAULT),
        }
    }

    /// Renders `eviction_batch` the way the configuration file stores it.
    pub fn format_cache_eviction_batch(value: u16) -> String {
        format!("eviction_batch = {} # entries", value)
    }

    /// One-line help text for `cache.eviction_batch`.
    pub fn describe_cache_eviction_batch() -> &'static str {
        "cache.eviction_batch: entries evicted per pass when the cache is full (entries, 1 to 4096, default 128)"
    }

    /// Lower bound accepted for `cache.admission_window_percent` (percent).
    pub const CACHE_ADMISSION_WINDOW_PERCENT_MIN: u8 = 1;
    /// Upper bound accepted for `cache.admission_window_percent` (percent).
    pub const CACHE_ADMISSION_WINDOW_PERCENT_MAX: u8 = 100;
    /// Value used when `cache.admission_window_percent` is absent.
    pub const CACHE_ADMISSION_WINDOW_PERCENT_DEFAULT: u8 = 1;

    /// Parses `cache.admission_window_percent`: share of capacity used as the admission window.
    ///
    /// Accepts decimal values in `1..=100` (percent); surrounding whitespace is ignored.
    pub fn parse_cache_admission_window_percent(raw: &str) -> Result<u8, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "cache.admission_window_percent" });
        }
        let value: u8 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "cache.admission_window_percent",
            raw: trimmed.to_string(),
        })?;
        if !(CACHE_ADMISSION_WINDOW_PERCENT_MIN..=CACHE_ADMISSION_WINDOW_PERCENT_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "cache.admission_window_percent",
                min: u64::from(CACHE_ADMISSION_WINDOW_PERCENT_MIN),
                max: u64::from(CACHE_ADMISSION_WINDOW_PERCENT_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `admission_window_percent` from a `[cache]` table, falling back to its default.
    pub fn cache_admission_window_percent_or_default(settings: &BTreeMap<String, String>) -> Result<u8, ConfigError> {
        match settings.get("admission_window_percent") {
            Some(raw) => parse_cache_admission_window_percent(raw),
            None => Ok(CACHE_ADMISSION_WINDOW_PERCENT_DEFAULT),
        }
    }

    /// Renders `admission_window_percent` the way the configuration file stores it.
    pub fn format_cache_admission_window_percent(value: u8) -> String {
        format!("admission_window_percent = {} # percent", value)
    }

    /// One-line help text for `cache.admission_window_percent`.
    pub fn describe_cache_admission_window_percent() -> &'static str {
        "cache.admission_window_percent: share of capacity used as the admission window (percent, 1 to 100, default 1)"
    }

    /// The `[cache]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct CacheSection {
        pub capacity_mib: u32,
        pub max_object_kib: u32,
        pub default_ttl_secs: u32,
        pub stale_while_revalidate_secs: u32,
        pub shards: u16,
        pub eviction_batch: u16,
        pub admission_window_percent: u8,
    }

    impl Default for CacheSection {
        fn default() -> Self {
            Self {
                capacity_mib: CACHE_CAPACITY_MIB_DEFAULT,
                max_object_kib: CACHE_MAX_OBJECT_KIB_DEFAULT,
                default_ttl_secs: CACHE_DEFAULT_TTL_SECS_DEFAULT,
                stale_while_revalidate_secs: CACHE_STALE_WHILE_REVALIDATE_SECS_DEFAULT,
                shards: CACHE_SHARDS_DEFAULT,
                eviction_batch: CACHE_EVICTION_BATCH_DEFAULT,
                admission_window_percent: CACHE_ADMISSION_WINDOW_PERCENT_DEFAULT,
            }
        }
    }

    impl CacheSection {
        /// Keys accepted in `[cache]`.
        pub const KEYS: &'static [&'static str] = &["capacity_mib", "max_object_kib", "default_ttl_secs", "stale_while_revalidate_secs", "shards", "eviction_batch", "admission_window_percent"];

        /// Parses every key of `[cache]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("cache.{}", key) });
                }
            }
            Ok(Self {
                capacity_mib: cache_capacity_mib_or_default(settings)?,
                max_object_kib: cache_max_object_kib_or_default(settings)?,
                default_ttl_secs: cache_default_ttl_secs_or_default(settings)?,
                stale_while_revalidate_secs: cache_stale_while_revalidate_secs_or_default(settings)?,
                shards: cache_shards_or_default(settings)?,
                eviction_batch: cache_eviction_batch_or_default(settings)?,
                admission_window_percent: cache_admission_window_percent_or_default(settings)?,
            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[cache]\n");
            out.push_str(&format_cache_capacity_mib(self.capacity_mib));
            out.push('\n');
            out.push_str(&format_cache_max_object_kib(self.max_object_kib));
            out.push('\n');
            out.push_str(&format_cache_default_ttl_secs(self.default_ttl_secs));
            out.push('\n');
            out.push_str(&format_cache_stale_while_revalidate_secs(self.stale_while_revalidate_secs));
            out.push('\n');
            out.push_str(&format_cache_shards(self.shards));
            out.push('\n');
            out.push_str(&format_cache_eviction_batch(self.eviction_batch));
            out.push('\n');
            out.push_str(&format_cache_admission_window_percent(self.admission_window_percent));
            out.push('\n');
            out
        }
    }

    /// Lower bound accepted for `rate_limit.requests_per_second` (requests).
    pub const RATE_LIMIT_REQUESTS_PER_SECOND_MIN: u32 = 1;
    /// Upper bound accepted for `rate_limit.requests_per_second` (requests).
    pub const RATE_LIMIT_REQUESTS_PER_SECOND_MAX: u32 = 1000000;
    /// Value used when `rate_limit.requests_per_second` is absent.
    pub const RATE_LIMIT_REQUESTS_PER_SECOND_DEFAULT: u32 = 100;

    /// Parses `rate_limit.requests_per_second`: sustained request rate per client.
    ///
    /// Accepts decimal values in `1..=1000000` (requests); surrounding whitespace is ignored.
    pub fn parse_rate_limit_requests_per_second(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "rate_limit.requests_per_second" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "rate_limit.requests_per_second",
            raw: trimmed.to_string(),
        })?;
        if !(RATE_LIMIT_REQUESTS_PER_SECOND_MIN..=RATE_LIMIT_REQUESTS_PER_SECOND_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "rate_limit.requests_per_second",
                min: u64::from(RATE_LIMIT_REQUESTS_PER_SECOND_MIN),
                max: u64::from(RATE_LIMIT_REQUESTS_PER_SECOND_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `requests_per_second` from a `[rate_limit]` table, falling back to its default.
    pub fn rate_limit_requests_per_second_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("requests_per_second") {
            Some(raw) => parse_rate_limit_requests_per_second(raw),
            None => Ok(RATE_LIMIT_REQUESTS_PER_SECOND_DEFAULT),
        }
    }

    /// Renders `requests_per_second` the way the configuration file stores it.
    pub fn format_rate_limit_requests_per_second(value: u32) -> String {
        format!("requests_per_second = {} # requests", value)
    }

    /// One-line help text for `rate_limit.requests_per_second`.
    pub fn describe_rate_limit_requests_per_second() -> &'static str {
        "rate_limit.requests_per_second: sustained request rate per client (requests, 1 to 1000000, default 100)"
    }

    /// Lower bound accepted for `rate_limit.burst` (requests).
    pub const RATE_LIMIT_BURST_MIN: u32 = 1;
    /// Upper bound accepted for `rate_limit.burst` (requests).
    pub const RATE_LIMIT_BURST_MAX: u32 = 1000000;
    /// Value used when `rate_limit.burst` is absent.
    pub const RATE_LIMIT_BURST_DEFAULT: u32 = 200;

    /// Parses `rate_limit.burst`: requests a client may send above the sustained rate.
    ///
    /// Accepts decimal values in `1..=1000000` (requests); surrounding whitespace is ignored.
    pub fn parse_rate_limit_burst(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "rate_limit.burst" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "rate_limit.burst",
            raw: trimmed.to_string(),
        })?;
        if !(RATE_LIMIT_BURST_MIN..=RATE_LIMIT_BURST_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "rate_limit.burst",
                min: u64::from(RATE_LIMIT_BURST_MIN),
                max: u64::from(RATE_LIMIT_BURST_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `burst` from a `[rate_limit]` table, falling back to its default.
    pub fn rate_limit_burst_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("burst") {
            Some(raw) => parse_rate_limit_burst(raw),
            None => Ok(RATE_LIMIT_BURST_DEFAULT),
        }
    }

    /// Renders `burst` the way the configuration file stores it.
    pub fn format_rate_limit_burst(value: u32) -> String {
        format!("burst = {} # requests", value)
    }

    /// One-line help text for `rate_limit.burst`.
    pub fn describe_rate_limit_burst() -> &'static str {
        "rate_limit.burst: requests a client may send above the sustained rate (requests, 1 to 1000000, default 200)"
    }

    /// Lower bound accepted for `rate_limit.window_secs` (seconds).
    pub const RATE_LIMIT_WINDOW_SECS_MIN: u32 = 1;
    /// Upper bound accepted for `rate_limit.window_secs` (seconds).
    pub const RATE_LIMIT_WINDOW_SECS_MAX: u32 = 3600;
    /// Value used when `rate_limit.window_secs` is absent.
    pub const RATE_LIMIT_WINDOW_SECS_DEFAULT: u32 = 60;

    /// Parses `rate_limit.window_secs`: accounting window of the limiter.
    ///
    /// Accepts decimal values in `1..=3600` (seconds); surrounding whitespace is ignored.
    pub fn parse_rate_limit_window_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "rate_limit.window_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "rate_limit.window_secs",
            raw: trimmed.to_string(),
        })?;
        if !(RATE_LIMIT_WINDOW_SECS_MIN..=RATE_LIMIT_WINDOW_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "rate_limit.window_secs",
                min: u64::from(RATE_LIMIT_WINDOW_SECS_MIN),
                max: u64::from(RATE_LIMIT_WINDOW_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `window_secs` from a `[rate_limit]` table, falling back to its default.
    pub fn rate_limit_window_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("window_secs") {
            Some(raw) => parse_rate_limit_window_secs(raw),
            None => Ok(RATE_LIMIT_WINDOW_SECS_DEFAULT),
        }
    }

    /// Renders `window_secs` the way the configuration file stores it.
    pub fn format_rate_limit_window_secs(value: u32) -> String {
        format!("window_secs = {} # seconds", value)
    }

    /// One-line help text for `rate_limit.window_secs`.
    pub fn describe_rate_limit_window_secs() -> &'static str {
        "rate_limit.window_secs: accounting window of the limiter (seconds, 1 to 3600, default 60)"
    }

    /// Lower bound accepted for `rate_limit.max_tracked_clients` (clients).
    pub const RATE_LIMIT_MAX_TRACKED_CLIENTS_MIN: u32 = 1;
    /// Upper bound accepted for `rate_limit.max_tracked_clients` (clients).
    pub const RATE_LIMIT_MAX_TRACKED_CLIENTS_MAX: u32 = 10000000;
    /// Value used when `rate_limit.max_tracked_clients` is absent.
    pub const RATE_LIMIT_MAX_TRACKED_CLIENTS_DEFAULT: u32 = 100000;

    /// Parses `rate_limit.max_tracked_clients`: clients tracked before the oldest are dropped.
    ///
    /// Accepts decimal values in `1..=10000000` (clients); surrounding whitespace is ignored.
    pub fn parse_rate_limit_max_tracked_clients(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "rate_limit.max_tracked_clients" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "rate_limit.max_tracked_clients",
            raw: trimmed.to_string(),
        })?;
        if !(RATE_LIMIT_MAX_TRACKED_CLIENTS_MIN..=RATE_LIMIT_MAX_TRACKED_CLIENTS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "rate_limit.max_tracked_clients",
                min: u64::from(RATE_LIMIT_MAX_TRACKED_CLIENTS_MIN),
                max: u64::from(RATE_LIMIT_MAX_TRACKED_CLIENTS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `max_tracked_clients` from a `[rate_limit]` table, falling back to its default.
    pub fn rate_limit_max_tracked_clients_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("max_tracked_clients") {
            Some(raw) => parse_rate_limit_max_tracked_clients(raw),
            None => Ok(RATE_LIMIT_MAX_TRACKED_CLIENTS_DEFAULT),
        }
    }

    /// Renders `max_tracked_clients` the way the configuration file stores it.
    pub fn format_rate_limit_max_tracked_clients(value: u32) -> String {
        format!("max_tracked_clients = {} # clients", value)
    }

    /// One-line help text for `rate_limit.max_tracked_clients`.
    pub fn describe_rate_limit_max_tracked_clients() -> &'static str {
        "rate_limit.max_tracked_clients: clients tracked before the oldest are dropped (clients, 1 to 10000000, default 100000)"
    }

    /// Lower bound accepted for `rate_limit.penalty_secs` (seconds).
    pub const RATE_LIMIT_PENALTY_SECS_MIN: u32 = 0;
    /// Upper bound accepted for `rate_limit.penalty_secs` (seconds).
    pub const RATE_LIMIT_PENALTY_SECS_MAX: u32 = 86400;
    /// Value used when `rate_limit.penalty_secs` is absent.
    pub const RATE_LIMIT_PENALTY_SECS_DEFAULT: u32 = 30;

    /// Parses `rate_limit.penalty_secs`: time a client stays blocked after exceeding its burst.
    ///
    /// Accepts decimal values in `0..=86400` (seconds); surrounding whitespace is ignored.
    pub fn parse_rate_limit_penalty_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "rate_limit.penalty_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "rate_limit.penalty_secs",
            raw: trimmed.to_string(),
        })?;
        if !(RATE_LIMIT_PENALTY_SECS_MIN..=RATE_LIMIT_PENALTY_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "rate_limit.penalty_secs",
                min: u64::from(RATE_LIMIT_PENALTY_SECS_MIN),
                max: u64::from(RATE_LIMIT_PENALTY_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `penalty_secs` from a `[rate_limit]` table, falling back to its default.
    pub fn rate_limit_penalty_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("penalty_secs") {
            Some(raw) => parse_rate_limit_penalty_secs(raw),
            None => Ok(RATE_LIMIT_PENALTY_SECS_DEFAULT),
        }
    }

    /// Renders `penalty_secs` the way the configuration file stores it.
    pub fn format_rate_limit_penalty_secs(value: u32) -> String {
        format!("penalty_secs = {} # seconds", value)
    }

    /// One-line help text for `rate_limit.penalty_secs`.
    pub fn describe_rate_limit_penalty_secs() -> &'static str {
        "rate_limit.penalty_secs: time a client stays blocked after exceeding its burst (seconds, 0 to 86400, default 30)"
    }

    /// Lower bound accepted for `rate_limit.status_code` (HTTP status).
    pub const RATE_LIMIT_STATUS_CODE_MIN: u16 = 400;
    /// Upper bound accepted for `rate_limit.status_code` (HTTP status).
    pub const RATE_LIMIT_STATUS_CODE_MAX: u16 = 599;
    /// Value used when `rate_limit.status_code` is absent.
    pub const RATE_LIMIT_STATUS_CODE_DEFAULT: u16 = 429;

    /// Parses `rate_limit.status_code`: status code returned to a limited client.
    ///
    /// Accepts decimal values in `400..=599` (HTTP status); surrounding whitespace is ignored.
    pub fn parse_rate_limit_status_code(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "rate_limit.status_code" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "rate_limit.status_code",
            raw: trimmed.to_string(),
        })?;
        if !(RATE_LIMIT_STATUS_CODE_MIN..=RATE_LIMIT_STATUS_CODE_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "rate_limit.status_code",
                min: u64::from(RATE_LIMIT_STATUS_CODE_MIN),
                max: u64::from(RATE_LIMIT_STATUS_CODE_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `status_code` from a `[rate_limit]` table, falling back to its default.
    pub fn rate_limit_status_code_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("status_code") {
            Some(raw) => parse_rate_limit_status_code(raw),
            None => Ok(RATE_LIMIT_STATUS_CODE_DEFAULT),
        }
    }

    /// Renders `status_code` the way the configuration file stores it.
    pub fn format_rate_limit_status_code(value: u16) -> String {
        format!("status_code = {} # HTTP status", value)
    }

    /// One-line help text for `rate_limit.status_code`.
    pub fn describe_rate_limit_status_code() -> &'static str {
        "rate_limit.status_code: status code returned to a limited client (HTTP status, 400 to 599, default 429)"
    }

    /// The `[rate_limit]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RateLimitSection {
        pub requests_per_second: u32,
        pub burst: u32,
        pub window_secs: u32,
        pub max_tracked_clients: u32,
        pub penalty_secs: u32,
        pub status_code: u16,
    }

    impl Default for RateLimitSection {
        fn default() -> Self {
            Self {
                requests_per_second: RATE_LIMIT_REQUESTS_PER_SECOND_DEFAULT,
                burst: RATE_LIMIT_BURST_DEFAULT,
                window_secs: RATE_LIMIT_WINDOW_SECS_DEFAULT,
                max_tracked_clients: RATE_LIMIT_MAX_TRACKED_CLIENTS_DEFAULT,
                penalty_secs: RATE_LIMIT_PENALTY_SECS_DEFAULT,
                status_code: RATE_LIMIT_STATUS_CODE_DEFAULT,
            }
        }
    }

    impl RateLimitSection {
        /// Keys accepted in `[rate_limit]`.
        pub const KEYS: &'static [&'static str] = &["requests_per_second", "burst", "window_secs", "max_tracked_clients", "penalty_secs", "status_code"];

        /// Parses every key of `[rate_limit]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("rate_limit.{}", key) });
                }
            }
            Ok(Self {
                requests_per_second: rate_limit_requests_per_second_or_default(settings)?,
                burst: rate_limit_burst_or_default(settings)?,
                window_secs: rate_limit_window_secs_or_default(settings)?,
                max_tracked_clients: rate_limit_max_tracked_clients_or_default(settings)?,
                penalty_secs: rate_limit_penalty_secs_or_default(settings)?,
                status_code: rate_limit_status_code_or_default(settings)?,
            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[rate_limit]\n");
            out.push_str(&format_rate_limit_requests_per_second(self.requests_per_second));
            out.push('\n');
            out.push_str(&format_rate_limit_burst(self.burst));
            out.push('\n');
            out.push_str(&format_rate_limit_window_secs(self.window_secs));
            out.push('\n');
            out.push_str(&format_rate_limit_max_tracked_clients(self.max_tracked_clients));
            out.push('\n');
            out.push_str(&format_rate_limit_penalty_secs(self.penalty_secs));
            out.push('\n');
            out.push_str(&format_rate_limit_status_code(self.status_code));
            out.push('\n');
            out
        }
    }

    /// Lower bound accepted for `dns.resolver_timeout_ms` (milliseconds).
    pub const DNS_RESOLVER_TIMEOUT_MS_MIN: u32 = 10;
    /// Upper bound accepted for `dns.resolver_timeout_ms` (milliseconds).
    pub const DNS_RESOLVER_TIMEOUT_MS_MAX: u32 = 30000;
    /// Value used when `dns.resolver_timeout_ms` is absent.
    pub const DNS_RESOLVER_TIMEOUT_MS_DEFAULT: u32 = 2000;

    /// Parses `dns.resolver_timeout_ms`: time allowed for one resolver query.
    ///
    /// Accepts decimal values in `10..=30000` (milliseconds); surrounding whitespace is ignored.
    pub fn parse_dns_resolver_timeout_ms(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "dns.resolver_timeout_ms" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "dns.resolver_timeout_ms",
            raw: trimmed.to_string(),
        })?;
        if !(DNS_RESOLVER_TIMEOUT_MS_MIN..=DNS_RESOLVER_TIMEOUT_MS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "dns.resolver_timeout_ms",
                min: u64::from(DNS_RESOLVER_TIMEOUT_MS_MIN),
                max: u64::from(DNS_RESOLVER_TIMEOUT_MS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `resolver_timeout_ms` from a `[dns]` table, falling back to its default.
    pub fn dns_resolver_timeout_ms_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("resolver_timeout_ms") {
            Some(raw) => parse_dns_resolver_timeout_ms(raw),
            None => Ok(DNS_RESOLVER_TIMEOUT_MS_DEFAULT),
        }
    }

    /// Renders `resolver_timeout_ms` the way the configuration file stores it.
    pub fn format_dns_resolver_timeout_ms(value: u32) -> String {
        format!("resolver_timeout_ms = {} # milliseconds", value)
    }

    /// One-line help text for `dns.resolver_timeout_ms`.
    pub fn describe_dns_resolver_timeout_ms() -> &'static str {
        "dns.resolver_timeout_ms: time allowed for one resolver query (milliseconds, 10 to 30000, default 2000)"
    }

    /// Lower bound accepted for `dns.attempts` (attempts).
    pub const DNS_ATTEMPTS_MIN: u8 = 1;
    /// Upper bound accepted for `dns.attempts` (attempts).
    pub const DNS_ATTEMPTS_MAX: u8 = 10;
    /// Value used when `dns.attempts` is absent.
    pub const DNS_ATTEMPTS_DEFAULT: u8 = 2;

    /// Parses `dns.attempts`: queries sent per name before resolution fails.
    ///
    /// Accepts decimal values in `1..=10` (attempts); surrounding whitespace is ignored.
    pub fn parse_dns_attempts(raw: &str) -> Result<u8, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "dns.attempts" });
        }
        let value: u8 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "dns.attempts",
            raw: trimmed.to_string(),
        })?;
        if !(DNS_ATTEMPTS_MIN..=DNS_ATTEMPTS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "dns.attempts",
                min: u64::from(DNS_ATTEMPTS_MIN),
                max: u64::from(DNS_ATTEMPTS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `attempts` from a `[dns]` table, falling back to its default.
    pub fn dns_attempts_or_default(settings: &BTreeMap<String, String>) -> Result<u8, ConfigError> {
        match settings.get("attempts") {
            Some(raw) => parse_dns_attempts(raw),
            None => Ok(DNS_ATTEMPTS_DEFAULT),
        }
    }

    /// Renders `attempts` the way the configuration file stores it.
    pub fn format_dns_attempts(value: u8) -> String {
        format!("attempts = {} # attempts", value)
    }

    /// One-line help text for `dns.attempts`.
    pub fn describe_dns_attempts() -> &'static str {
        "dns.attempts: queries sent per name before resolution fails (attempts, 1 to 10, default 2)"
    }

    /// Lower bound accepted for `dns.cache_entries` (records).
    pub const DNS_CACHE_ENTRIES_MIN: u32 = 0;
    /// Upper bound accepted for `dns.cache_entries` (records).
    pub const DNS_CACHE_ENTRIES_MAX: u32 = 1000000;
    /// Value used when `dns.cache_entries` is absent.
    pub const DNS_CACHE_ENTRIES_DEFAULT: u32 = 4096;

    /// Parses `dns.cache_entries`: resolved records kept in memory.
    ///
    /// Accepts decimal values in `0..=1000000` (records); surrounding whitespace is ignored.
    pub fn parse_dns_cache_entries(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "dns.cache_entries" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "dns.cache_entries",
            raw: trimmed.to_string(),
        })?;
        if !(DNS_CACHE_ENTRIES_MIN..=DNS_CACHE_ENTRIES_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "dns.cache_entries",
                min: u64::from(DNS_CACHE_ENTRIES_MIN),
                max: u64::from(DNS_CACHE_ENTRIES_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `cache_entries` from a `[dns]` table, falling back to its default.
    pub fn dns_cache_entries_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("cache_entries") {
            Some(raw) => parse_dns_cache_entries(raw),
            None => Ok(DNS_CACHE_ENTRIES_DEFAULT),
        }
    }

    /// Renders `cache_entries` the way the configuration file stores it.
    pub fn format_dns_cache_entries(value: u32) -> String {
        format!("cache_entries = {} # records", value)
    }

    /// One-line help text for `dns.cache_entries`.
    pub fn describe_dns_cache_entries() -> &'static str {
        "dns.cache_entries: resolved records kept in memory (records, 0 to 1000000, default 4096)"
    }

    /// Lower bound accepted for `dns.min_ttl_secs` (seconds).
    pub const DNS_MIN_TTL_SECS_MIN: u32 = 0;
    /// Upper bound accepted for `dns.min_ttl_secs` (seconds).
    pub const DNS_MIN_TTL_SECS_MAX: u32 = 86400;
    /// Value used when `dns.min_ttl_secs` is absent.
    pub const DNS_MIN_TTL_SECS_DEFAULT: u32 = 5;

    /// Parses `dns.min_ttl_secs`: lower bound applied to record lifetimes.
    ///
    /// Accepts decimal values in `0..=86400` (seconds); surrounding whitespace is ignored.
    pub fn parse_dns_min_ttl_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "dns.min_ttl_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "dns.min_ttl_secs",
            raw: trimmed.to_string(),
        })?;
        if !(DNS_MIN_TTL_SECS_MIN..=DNS_MIN_TTL_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "dns.min_ttl_secs",
                min: u64::from(DNS_MIN_TTL_SECS_MIN),
                max: u64::from(DNS_MIN_TTL_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `min_ttl_secs` from a `[dns]` table, falling back to its default.
    pub fn dns_min_ttl_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("min_ttl_secs") {
            Some(raw) => parse_dns_min_ttl_secs(raw),
            None => Ok(DNS_MIN_TTL_SECS_DEFAULT),
        }
    }

    /// Renders `min_ttl_secs` the way the configuration file stores it.
    pub fn format_dns_min_ttl_secs(value: u32) -> String {
        format!("min_ttl_secs = {} # seconds", value)
    }

    /// One-line help text for `dns.min_ttl_secs`.
    pub fn describe_dns_min_ttl_secs() -> &'static str {
        "dns.min_ttl_secs: lower bound applied to record lifetimes (seconds, 0 to 86400, default 5)"
    }

    /// Lower bound accepted for `dns.max_ttl_secs` (seconds).
    pub const DNS_MAX_TTL_SECS_MIN: u32 = 1;
    /// Upper bound accepted for `dns.max_ttl_secs` (seconds).
    pub const DNS_MAX_TTL_SECS_MAX: u32 = 604800;
    /// Value used when `dns.max_ttl_secs` is absent.
    pub const DNS_MAX_TTL_SECS_DEFAULT: u32 = 3600;

    /// Parses `dns.max_ttl_secs`: upper bound applied to record lifetimes.
    ///
    /// Accepts decimal values in `1..=604800` (seconds); surrounding whitespace is ignored.
    pub fn parse_dns_max_ttl_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "dns.max_ttl_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "dns.max_ttl_secs",
            raw: trimmed.to_string(),
        })?;
        if !(DNS_MAX_TTL_SECS_MIN..=DNS_MAX_TTL_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "dns.max_ttl_secs",
                min: u64::from(DNS_MAX_TTL_SECS_MIN),
                max: u64::from(DNS_MAX_TTL_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `max_ttl_secs` from a `[dns]` table, falling back to its default.
    pub fn dns_max_ttl_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("max_ttl_secs") {
            Some(raw) => parse_dns_max_ttl_secs(raw),
            None => Ok(DNS_MAX_TTL_SECS_DEFAULT),
        }
    }

    /// Renders `max_ttl_secs` the way the configuration file stores it.
    pub fn format_dns_max_ttl_secs(value: u32) -> String {
        format!("max_ttl_secs = {} # seconds", value)
    }

    /// One-line help text for `dns.max_ttl_secs`.
    pub fn describe_dns_max_ttl_secs() -> &'static str {
        "dns.max_ttl_secs: upper bound applied to record lifetimes (seconds, 1 to 604800, default 3600)"
    }

    /// Lower bound accepted for `dns.negative_ttl_secs` (seconds).
    pub const DNS_NEGATIVE_TTL_SECS_MIN: u32 = 0;
    /// Upper bound accepted for `dns.negative_ttl_secs` (seconds).
    pub const DNS_NEGATIVE_TTL_SECS_MAX: u32 = 86400;
    /// Value used when `dns.negative_ttl_secs` is absent.
    pub const DNS_NEGATIVE_TTL_SECS_DEFAULT: u32 = 30;

    /// Parses `dns.negative_ttl_secs`: lifetime of a cached failed lookup.
    ///
    /// Accepts decimal values in `0..=86400` (seconds); surrounding whitespace is ignored.
    pub fn parse_dns_negative_ttl_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "dns.negative_ttl_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "dns.negative_ttl_secs",
            raw: trimmed.to_string(),
        })?;
        if !(DNS_NEGATIVE_TTL_SECS_MIN..=DNS_NEGATIVE_TTL_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "dns.negative_ttl_secs",
                min: u64::from(DNS_NEGATIVE_TTL_SECS_MIN),
                max: u64::from(DNS_NEGATIVE_TTL_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `negative_ttl_secs` from a `[dns]` table, falling back to its default.
    pub fn dns_negative_ttl_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("negative_ttl_secs") {
            Some(raw) => parse_dns_negative_ttl_secs(raw),
            None => Ok(DNS_NEGATIVE_TTL_SECS_DEFAULT),
        }
    }

    /// Renders `negative_ttl_secs` the way the configuration file stores it.
    pub fn format_dns_negative_ttl_secs(value: u32) -> String {
        format!("negative_ttl_secs = {} # seconds", value)
    }

    /// One-line help text for `dns.negative_ttl_secs`.
    pub fn describe_dns_negative_ttl_secs() -> &'static str {
        "dns.negative_ttl_secs: lifetime of a cached failed lookup (seconds, 0 to 86400, default 30)"
    }

    /// Lower bound accepted for `dns.ndots` (dots).
    pub const DNS_NDOTS_MIN: u8 = 0;
    /// Upper bound accepted for `dns.ndots` (dots).
    pub const DNS_NDOTS_MAX: u8 = 15;
    /// Value used when `dns.ndots` is absent.
    pub const DNS_NDOTS_DEFAULT: u8 = 1;

    /// Parses `dns.ndots`: dots in a name before it is tried as absolute first.
    ///
    /// Accepts decimal values in `0..=15` (dots); surrounding whitespace is ignored.
    pub fn parse_dns_ndots(raw: &str) -> Result<u8, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "dns.ndots" });
        }
        let value: u8 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "dns.ndots",
            raw: trimmed.to_string(),
        })?;
        if !(DNS_NDOTS_MIN..=DNS_NDOTS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "dns.ndots",
                min: u64::from(DNS_NDOTS_MIN),
                max: u64::from(DNS_NDOTS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `ndots` from a `[dns]` table, falling back to its default.
    pub fn dns_ndots_or_default(settings: &BTreeMap<String, String>) -> Result<u8, ConfigError> {
        match settings.get("ndots") {
            Some(raw) => parse_dns_ndots(raw),
            None => Ok(DNS_NDOTS_DEFAULT),
        }
    }

    /// Renders `ndots` the way the configuration file stores it.
    pub fn format_dns_ndots(value: u8) -> String {
        format!("ndots = {} # dots", value)
    }

    /// One-line help text for `dns.ndots`.
    pub fn describe_dns_ndots() -> &'static str {
        "dns.ndots: dots in a name before it is tried as absolute first (dots, 0 to 15, default 1)"
    }

    /// The `[dns]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct DnsSection {
        pub resolver_timeout_ms: u32,
        pub attempts: u8,
        pub cache_entries: u32,
        pub min_ttl_secs: u32,
        pub max_ttl_secs: u32,
        pub negative_ttl_secs: u32,
        pub ndots: u8,
    }

    impl Default for DnsSection {
        fn default() -> Self {
            Self {
                resolver_timeout_ms: DNS_RESOLVER_TIMEOUT_MS_DEFAULT,
                attempts: DNS_ATTEMPTS_DEFAULT,
                cache_entries: DNS_CACHE_ENTRIES_DEFAULT,
                min_ttl_secs: DNS_MIN_TTL_SECS_DEFAULT,
                max_ttl_secs: DNS_MAX_TTL_SECS_DEFAULT,
                negative_ttl_secs: DNS_NEGATIVE_TTL_SECS_DEFAULT,
                ndots: DNS_NDOTS_DEFAULT,
            }
        }
    }

    impl DnsSection {
        /// Keys accepted in `[dns]`.
        pub const KEYS: &'static [&'static str] = &["resolver_timeout_ms", "attempts", "cache_entries", "min_ttl_secs", "max_ttl_secs", "negative_ttl_secs", "ndots"];

        /// Parses every key of `[dns]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("dns.{}", key) });
                }
            }
            Ok(Self {
                resolver_timeout_ms: dns_resolver_timeout_ms_or_default(settings)?,
                attempts: dns_attempts_or_default(settings)?,
                cache_entries: dns_cache_entries_or_default(settings)?,
                min_ttl_secs: dns_min_ttl_secs_or_default(settings)?,
                max_ttl_secs: dns_max_ttl_secs_or_default(settings)?,
                negative_ttl_secs: dns_negative_ttl_secs_or_default(settings)?,
                ndots: dns_ndots_or_default(settings)?,
            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[dns]\n");
            out.push_str(&format_dns_resolver_timeout_ms(self.resolver_timeout_ms));
            out.push('\n');
            out.push_str(&format_dns_attempts(self.attempts));
            out.push('\n');
            out.push_str(&format_dns_cache_entries(self.cache_entries));
            out.push('\n');
            out.push_str(&format_dns_min_ttl_secs(self.min_ttl_secs));
            out.push('\n');
            out.push_str(&format_dns_max_ttl_secs(self.max_ttl_secs));
            out.push('\n');
            out.push_str(&format_dns_negative_ttl_secs(self.negative_ttl_secs));
            out.push('\n');
            out.push_str(&format_dns_ndots(self.ndots));
            out.push('\n');
            out
        }
    }

    /// Lower bound accepted for `runtime.worker_threads` (threads).
    pub const RUNTIME_WORKER_THREADS_MIN: u16 = 1;
    /// Upper bound accepted for `runtime.worker_threads` (threads).
    pub const RUNTIME_WORKER_THREADS_MAX: u16 = 1024;
    /// Value used when `runtime.worker_threads` is absent.
    pub const RUNTIME_WORKER_THREADS_DEFAULT: u16 = 8;

    /// Parses `runtime.worker_threads`: threads that run connection tasks.
    ///
    /// Accepts decimal values in `1..=1024` (threads); surrounding whitespace is ignored.
    pub fn parse_runtime_worker_threads(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "runtime.worker_threads" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "runtime.worker_threads",
            raw: trimmed.to_string(),
        })?;
        if !(RUNTIME_WORKER_THREADS_MIN..=RUNTIME_WORKER_THREADS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "runtime.worker_threads",
                min: u64::from(RUNTIME_WORKER_THREADS_MIN),
                max: u64::from(RUNTIME_WORKER_THREADS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `worker_threads` from a `[runtime]` table, falling back to its default.
    pub fn runtime_worker_threads_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("worker_threads") {
            Some(raw) => parse_runtime_worker_threads(raw),
            None => Ok(RUNTIME_WORKER_THREADS_DEFAULT),
        }
    }

    /// Renders `worker_threads` the way the configuration file stores it.
    pub fn format_runtime_worker_threads(value: u16) -> String {
        format!("worker_threads = {} # threads", value)
    }

    /// One-line help text for `runtime.worker_threads`.
    pub fn describe_runtime_worker_threads() -> &'static str {
        "runtime.worker_threads: threads that run connection tasks (threads, 1 to 1024, default 8)"
    }

    /// Lower bound accepted for `runtime.blocking_threads` (threads).
    pub const RUNTIME_BLOCKING_THREADS_MIN: u16 = 1;
    /// Upper bound accepted for `runtime.blocking_threads` (threads).
    pub const RUNTIME_BLOCKING_THREADS_MAX: u16 = 4096;
    /// Value used when `runtime.blocking_threads` is absent.
    pub const RUNTIME_BLOCKING_THREADS_DEFAULT: u16 = 512;

    /// Parses `runtime.blocking_threads`: threads reserved for blocking file and DNS work.
    ///
    /// Accepts decimal values in `1..=4096` (threads); surrounding whitespace is ignored.
    pub fn parse_runtime_blocking_threads(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "runtime.blocking_threads" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "runtime.blocking_threads",
            raw: trimmed.to_string(),
        })?;
        if !(RUNTIME_BLOCKING_THREADS_MIN..=RUNTIME_BLOCKING_THREADS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "runtime.blocking_threads",
                min: u64::from(RUNTIME_BLOCKING_THREADS_MIN),
                max: u64::from(RUNTIME_BLOCKING_THREADS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `blocking_threads` from a `[runtime]` table, falling back to its default.
    pub fn runtime_blocking_threads_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("blocking_threads") {
            Some(raw) => parse_runtime_blocking_threads(raw),
            None => Ok(RUNTIME_BLOCKING_THREADS_DEFAULT),
        }
    }

    /// Renders `blocking_threads` the way the configuration file stores it.
    pub fn format_runtime_blocking_threads(value: u16) -> String {
        format!("blocking_threads = {} # threads", value)
    }

    /// One-line help text for `runtime.blocking_threads`.
    pub fn describe_runtime_blocking_threads() -> &'static str {
        "runtime.blocking_threads: threads reserved for blocking file and DNS work (threads, 1 to 4096, default 512)"
    }

    /// Lower bound accepted for `runtime.queue_depth` (tasks).
    pub const RUNTIME_QUEUE_DEPTH_MIN: u32 = 1;
    /// Upper bound accepted for `runtime.queue_depth` (tasks).
    pub const RUNTIME_QUEUE_DEPTH_MAX: u32 = 1048576;
    /// Value used when `runtime.queue_depth` is absent.
    pub const RUNTIME_QUEUE_DEPTH_DEFAULT: u32 = 1024;

    /// Parses `runtime.queue_depth`: tasks queued per worker before new work is refused.
    ///
    /// Accepts decimal values in `1..=1048576` (tasks); surrounding whitespace is ignored.
    pub fn parse_runtime_queue_depth(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "runtime.queue_depth" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "runtime.queue_depth",
            raw: trimmed.to_string(),
        })?;
        if !(RUNTIME_QUEUE_DEPTH_MIN..=RUNTIME_QUEUE_DEPTH_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "runtime.queue_depth",
                min: u64::from(RUNTIME_QUEUE_DEPTH_MIN),
                max: u64::from(RUNTIME_QUEUE_DEPTH_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `queue_depth` from a `[runtime]` table, falling back to its default.
    pub fn runtime_queue_depth_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("queue_depth") {
            Some(raw) => parse_runtime_queue_depth(raw),
            None => Ok(RUNTIME_QUEUE_DEPTH_DEFAULT),
        }
    }

    /// Renders `queue_depth` the way the configuration file stores it.
    pub fn format_runtime_queue_depth(value: u32) -> String {
        format!("queue_depth = {} # tasks", value)
    }

    /// One-line help text for `runtime.queue_depth`.
    pub fn describe_runtime_queue_depth() -> &'static str {
        "runtime.queue_depth: tasks queued per worker before new work is refused (tasks, 1 to 1048576, default 1024)"
    }

    /// Lower bound accepted for `runtime.stack_kib` (KiB).
    pub const RUNTIME_STACK_KIB_MIN: u32 = 64;
    /// Upper bound accepted for `runtime.stack_kib` (KiB).
    pub const RUNTIME_STACK_KIB_MAX: u32 = 65536;
    /// Value used when `runtime.stack_kib` is absent.
    pub const RUNTIME_STACK_KIB_DEFAULT: u32 = 2048;

    /// Parses `runtime.stack_kib`: stack size of each worker thread.
    ///
    /// Accepts decimal values in `64..=65536` (KiB); surrounding whitespace is ignored.
    pub fn parse_runtime_stack_kib(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "runtime.stack_kib" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "runtime.stack_kib",
            raw: trimmed.to_string(),
        })?;
        if !(RUNTIME_STACK_KIB_MIN..=RUNTIME_STACK_KIB_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "runtime.stack_kib",
                min: u64::from(RUNTIME_STACK_KIB_MIN),
                max: u64::from(RUNTIME_STACK_KIB_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `stack_kib` from a `[runtime]` table, falling back to its default.
    pub fn runtime_stack_kib_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("stack_kib") {
            Some(raw) => parse_runtime_stack_kib(raw),
            None => Ok(RUNTIME_STACK_KIB_DEFAULT),
        }
    }

    /// Renders `stack_kib` the way the configuration file stores it.
    pub fn format_runtime_stack_kib(value: u32) -> String {
        format!("stack_kib = {} # KiB", value)
    }

    /// One-line help text for `runtime.stack_kib`.
    pub fn describe_runtime_stack_kib() -> &'static str {
        "runtime.stack_kib: stack size of each worker thread (KiB, 64 to 65536, default 2048)"
    }

    /// Lower bound accepted for `runtime.shutdown_grace_secs` (seconds).
    pub const RUNTIME_SHUTDOWN_GRACE_SECS_MIN: u32 = 0;
    /// Upper bound accepted for `runtime.shutdown_grace_secs` (seconds).
    pub const RUNTIME_SHUTDOWN_GRACE_SECS_MAX: u32 = 3600;
    /// Value used when `runtime.shutdown_grace_secs` is absent.
    pub const RUNTIME_SHUTDOWN_GRACE_SECS_DEFAULT: u32 = 30;

    /// Parses `runtime.shutdown_grace_secs`: time in-flight requests get after a shutdown signal.
    ///
    /// Accepts decimal values in `0..=3600` (seconds); surrounding whitespace is ignored.
    pub fn parse_runtime_shutdown_grace_secs(raw: &str) -> Result<u32, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "runtime.shutdown_grace_secs" });
        }
        let value: u32 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "runtime.shutdown_grace_secs",
            raw: trimmed.to_string(),
        })?;
        if !(RUNTIME_SHUTDOWN_GRACE_SECS_MIN..=RUNTIME_SHUTDOWN_GRACE_SECS_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "runtime.shutdown_grace_secs",
                min: u64::from(RUNTIME_SHUTDOWN_GRACE_SECS_MIN),
                max: u64::from(RUNTIME_SHUTDOWN_GRACE_SECS_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `shutdown_grace_secs` from a `[runtime]` table, falling back to its default.
    pub fn runtime_shutdown_grace_secs_or_default(settings: &BTreeMap<String, String>) -> Result<u32, ConfigError> {
        match settings.get("shutdown_grace_secs") {
            Some(raw) => parse_runtime_shutdown_grace_secs(raw),
            None => Ok(RUNTIME_SHUTDOWN_GRACE_SECS_DEFAULT),
        }
    }

    /// Renders `shutdown_grace_secs` the way the configuration file stores it.
    pub fn format_runtime_shutdown_grace_secs(value: u32) -> String {
        format!("shutdown_grace_secs = {} # seconds", value)
    }

    /// One-line help text for `runtime.shutdown_grace_secs`.
    pub fn describe_runtime_shutdown_grace_secs() -> &'static str {
        "runtime.shutdown_grace_secs: time in-flight requests get after a shutdown signal (seconds, 0 to 3600, default 30)"
    }

    /// Lower bound accepted for `runtime.steal_batch` (tasks).
    pub const RUNTIME_STEAL_BATCH_MIN: u16 = 1;
    /// Upper bound accepted for `runtime.steal_batch` (tasks).
    pub const RUNTIME_STEAL_BATCH_MAX: u16 = 256;
    /// Value used when `runtime.steal_batch` is absent.
    pub const RUNTIME_STEAL_BATCH_DEFAULT: u16 = 32;

    /// Parses `runtime.steal_batch`: tasks moved per work-stealing attempt.
    ///
    /// Accepts decimal values in `1..=256` (tasks); surrounding whitespace is ignored.
    pub fn parse_runtime_steal_batch(raw: &str) -> Result<u16, ConfigError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ConfigError::Empty { key: "runtime.steal_batch" });
        }
        let value: u16 = trimmed.parse().map_err(|_| ConfigError::NotANumber {
            key: "runtime.steal_batch",
            raw: trimmed.to_string(),
        })?;
        if !(RUNTIME_STEAL_BATCH_MIN..=RUNTIME_STEAL_BATCH_MAX).contains(&value) {
            return Err(ConfigError::OutOfRange {
                key: "runtime.steal_batch",
                min: u64::from(RUNTIME_STEAL_BATCH_MIN),
                max: u64::from(RUNTIME_STEAL_BATCH_MAX),
                value: u64::from(value),
            });
        }
        Ok(value)
    }

    /// Reads `steal_batch` from a `[runtime]` table, falling back to its default.
    pub fn runtime_steal_batch_or_default(settings: &BTreeMap<String, String>) -> Result<u16, ConfigError> {
        match settings.get("steal_batch") {
            Some(raw) => parse_runtime_steal_batch(raw),
            None => Ok(RUNTIME_STEAL_BATCH_DEFAULT),
        }
    }

    /// Renders `steal_batch` the way the configuration file stores it.
    pub fn format_runtime_steal_batch(value: u16) -> String {
        format!("steal_batch = {} # tasks", value)
    }

    /// One-line help text for `runtime.steal_batch`.
    pub fn describe_runtime_steal_batch() -> &'static str {
        "runtime.steal_batch: tasks moved per work-stealing attempt (tasks, 1 to 256, default 32)"
    }

    /// The `[runtime]` table of the daemon configuration.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RuntimeSection {
        pub worker_threads: u16,
        pub blocking_threads: u16,
        pub queue_depth: u32,
        pub stack_kib: u32,
        pub shutdown_grace_secs: u32,
        pub steal_batch: u16,
    }

    impl Default for RuntimeSection {
        fn default() -> Self {
            Self {
                worker_threads: RUNTIME_WORKER_THREADS_DEFAULT,
                blocking_threads: RUNTIME_BLOCKING_THREADS_DEFAULT,
                queue_depth: RUNTIME_QUEUE_DEPTH_DEFAULT,
                stack_kib: RUNTIME_STACK_KIB_DEFAULT,
                shutdown_grace_secs: RUNTIME_SHUTDOWN_GRACE_SECS_DEFAULT,
                steal_batch: RUNTIME_STEAL_BATCH_DEFAULT,
            }
        }
    }

    impl RuntimeSection {
        /// Keys accepted in `[runtime]`.
        pub const KEYS: &'static [&'static str] = &["worker_threads", "blocking_threads", "queue_depth", "stack_kib", "shutdown_grace_secs", "steal_batch"];

        /// Parses every key of `[runtime]`; a key outside the schema is refused.
        pub fn from_map(settings: &BTreeMap<String, String>) -> Result<Self, ConfigError> {
            for key in settings.keys() {
                if !Self::KEYS.contains(&key.as_str()) {
                    return Err(ConfigError::Unknown { key: format!("runtime.{}", key) });
                }
            }
            Ok(Self {
                worker_threads: runtime_worker_threads_or_default(settings)?,
                blocking_threads: runtime_blocking_threads_or_default(settings)?,
                queue_depth: runtime_queue_depth_or_default(settings)?,
                stack_kib: runtime_stack_kib_or_default(settings)?,
                shutdown_grace_secs: runtime_shutdown_grace_secs_or_default(settings)?,
                steal_batch: runtime_steal_batch_or_default(settings)?,
            })
        }

        /// Renders the table in configuration-file syntax.
        pub fn render(&self) -> String {
            let mut out = String::from("[runtime]\n");
            out.push_str(&format_runtime_worker_threads(self.worker_threads));
            out.push('\n');
            out.push_str(&format_runtime_blocking_threads(self.blocking_threads));
            out.push('\n');
            out.push_str(&format_runtime_queue_depth(self.queue_depth));
            out.push('\n');
            out.push_str(&format_runtime_stack_kib(self.stack_kib));
            out.push('\n');
            out.push_str(&format_runtime_shutdown_grace_secs(self.shutdown_grace_secs));
            out.push('\n');
            out.push_str(&format_runtime_steal_batch(self.steal_batch));
            out.push('\n');
            out
        }
    }

    /// Sections of the daemon configuration, in file order.
    pub const SECTIONS: &[&str] = &["listener", "upstream", "health_check", "tls", "cache", "rate_limit", "dns", "runtime"];

    /// Validates one `[section]` table and renders it back in canonical form.
    pub fn normalize_section(name: &str, settings: &BTreeMap<String, String>) -> Result<String, ConfigError> {
        match name {
            "listener" => Ok(ListenerSection::from_map(settings)?.render()),
            "upstream" => Ok(UpstreamSection::from_map(settings)?.render()),
            "health_check" => Ok(HealthCheckSection::from_map(settings)?.render()),
            "tls" => Ok(TlsSection::from_map(settings)?.render()),
            "cache" => Ok(CacheSection::from_map(settings)?.render()),
            "rate_limit" => Ok(RateLimitSection::from_map(settings)?.render()),
            "dns" => Ok(DnsSection::from_map(settings)?.render()),
            "runtime" => Ok(RuntimeSection::from_map(settings)?.render()),
            other => Err(ConfigError::Unknown { key: other.to_string() }),
        }
    }

    /// Validates a whole configuration file and renders it back in canonical form.
    pub fn normalize_file(text: &str) -> Result<String, ConfigError> {
        let sections = parse_file(text)?;
        let mut out = String::new();
        for (name, settings) in &sections {
            out.push_str(&normalize_section(name, settings)?);
            out.push('\n');
        }
        Ok(out)
    }
}
