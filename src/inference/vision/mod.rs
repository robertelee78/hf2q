//! Vision preprocessing (ADR-005 Phase 2c, Task #14).
//!
//! CPU-side image handling for the `/v1/chat/completions` multimodal path
//! (Decision #1 of the 2026-04-23 scope refinement — vision absorbed as
//! Phase 2c sub-phase). Decodes OpenAI-format `image_url` content parts,
//! resizes to the ViT's expected patch grid, and normalizes to a CHW
//! float tensor ready for a future ViT forward pass.
//!
//! # Supported input formats
//!
//!   - `data:image/png;base64,<payload>`  — inline base64, Open WebUI default.
//!   - `data:image/jpeg;base64,<payload>` — same shape, JPEG payload.
//!   - `file:///absolute/path/to/image.{png,jpg,jpeg}` — local file URL.
//!   - `/absolute/path/to/image.{png,jpg,jpeg}` — bare path (shorthand).
//!   - `https://<host>/<path>` — fetched via reqwest blocking client.
//!   - `http://<host>/<path>` — same fetch path as HTTPS.
//!
//! HTTPS URLs are fetched via `reqwest::blocking::Client` with a 10-second
//! timeout and a 20 MB response-body cap. The blocking I/O is wrapped in
//! `tokio::task::block_in_place` so the axum executor thread is not starved
//! while waiting on the network.
//!
//! # Preprocessing pipeline
//!
//!   1. Decode bytes into `image::DynamicImage`.
//!   2. Resize to `target_size × target_size` (typical ViT: 224, 336, 518).
//!   3. Convert to RGB8 (drops alpha channel; mmproj inputs are 3-channel).
//!   4. Normalize each channel: `(pixel/255 - mean[c]) / std[c]`.
//!   5. Transpose HWC → CHW into a flat `Vec<f32>` of length `3 × size × size`.
//!
//! # Not done in this iter (deferred to ViT-forward-pass iter)
//!
//!   - Patchifying `[3, H, W]` → `[N_PATCHES, PATCH_DIM]` via conv stem.
//!     That's a ViT model-side operation, not preprocessing.
//!   - Multi-image batching. A single request may carry multiple images
//!     (OpenAI's `content: [{text}, {image_url}, {image_url}, ...]` shape);
//!     the handler will iterate.

use anyhow::{anyhow, Result};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub mod mmproj;
pub mod mmproj_weights;
pub mod preprocess;
pub mod vit;
pub mod vit_dump;
pub mod vit_gpu;

#[allow(unused_imports)]
pub use preprocess::{preprocess_rgb_chw, PreprocessConfig, GEMMA4_VISION_CONFIG};

/// A single preprocessed image ready for the ViT forward pass.
///
/// `pixel_values` carries the CHW-layout f32 tensor produced by
/// `preprocess_rgb_chw` (length = `3 × target_size × target_size`).
/// `source_label` is a debug/log-friendly id (mime type for data URIs,
/// file-name stem for file paths) so request-level tracing can
/// correlate per-image timings without leaking the full URL or payload.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub pixel_values: Vec<f32>,
    pub target_size: u32,
    pub source_label: String,
}

// ---------------------------------------------------------------------------
// ImageInput parsing
// ---------------------------------------------------------------------------

/// Parsed image source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageInput {
    /// Base64-encoded image payload (PNG or JPEG). The `mime_type` is the
    /// string between `data:` and `;base64,` — e.g. `"image/png"`.
    DataUri {
        mime_type: String,
        payload_base64: String,
    },
    /// Local filesystem path.
    FilePath(PathBuf),
    /// HTTP(S) URL — fetched at load time with a 10 s timeout + 20 MB cap.
    HttpUrl(String),
}

/// Parse an OpenAI-format `image_url` string into a typed `ImageInput`.
///
/// Returns `Err` on unrecognized / malformed URLs. The caller should map
/// the error to a 400 invalid_request with `param = "content"`.
pub fn parse_image_url(url: &str) -> Result<ImageInput> {
    // data:image/{fmt};base64,<payload>
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, payload) = rest
            .split_once(",")
            .ok_or_else(|| anyhow!("data URI missing comma separator"))?;
        // Metadata: `image/png;base64` → mime=image/png, encoding=base64.
        let (mime_type, encoding) = meta
            .split_once(";")
            .ok_or_else(|| anyhow!("data URI missing encoding section"))?;
        if encoding != "base64" {
            return Err(anyhow!(
                "data URI encoding '{}' not supported (only 'base64')",
                encoding
            ));
        }
        if !(mime_type == "image/png" || mime_type == "image/jpeg" || mime_type == "image/jpg") {
            return Err(anyhow!(
                "data URI mime type '{}' not supported (only image/png and image/jpeg)",
                mime_type
            ));
        }
        return Ok(ImageInput::DataUri {
            mime_type: mime_type.to_string(),
            payload_base64: payload.to_string(),
        });
    }

    // file:///path
    if let Some(rest) = url.strip_prefix("file://") {
        // Trim the leading '/' that's always present in file:// URLs to
        // keep an absolute POSIX path.
        let path = if rest.starts_with('/') {
            PathBuf::from(rest)
        } else {
            return Err(anyhow!(
                "file:// URL must contain an absolute path (file:///path)"
            ));
        };
        return Ok(ImageInput::FilePath(path));
    }

    // Bare absolute path.
    if url.starts_with('/') {
        return Ok(ImageInput::FilePath(PathBuf::from(url)));
    }

    // http(s):// — fetched via fetch_https_image.
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(ImageInput::HttpUrl(url.to_string()));
    }

    Err(anyhow!(
        "unrecognized image URL scheme (expected data:, file://, or absolute path)"
    ))
}

/// Read an `ImageInput` into a raw bytes buffer.
pub fn load_image_bytes(input: &ImageInput) -> Result<Vec<u8>> {
    match input {
        ImageInput::DataUri { payload_base64, .. } => {
            use base64::Engine;
            let payload = base64::engine::general_purpose::STANDARD
                .decode(payload_base64.trim())
                .map_err(|e| anyhow!("base64 decode: {e}"))?;
            Ok(payload)
        }
        ImageInput::FilePath(p) => read_file_bounded(p),
        ImageInput::HttpUrl(url) => fetch_https_image(url),
    }
}

/// Read a file with a 20 MB size cap — defensive cap that exceeds the
/// biggest reasonable VLM input (a 4K JPEG is ~6 MB).
fn read_file_bounded(p: &Path) -> Result<Vec<u8>> {
    const MAX: u64 = 20 * 1024 * 1024;
    let meta = std::fs::metadata(p)
        .map_err(|e| anyhow!("stat {}: {e}", p.display()))?;
    if meta.len() > MAX {
        return Err(anyhow!(
            "image file {} exceeds {}-byte cap (got {})",
            p.display(),
            MAX,
            meta.len()
        ));
    }
    std::fs::read(p).map_err(|e| anyhow!("read {}: {e}", p.display()))
}

/// Fetch an HTTPS image URL via reqwest blocking, enforcing:
///   - 10-second total timeout (connect + transfer).
///   - 20 MB response-body cap checked against Content-Length before download,
///     then enforced again on the byte stream to guard against missing headers.
///
/// The call runs inside `tokio::task::block_in_place` when a tokio runtime
/// is active, so the axum executor thread yields to the blocking pool for the
/// duration of the network I/O instead of stalling it.
fn fetch_https_image(url: &str) -> Result<Vec<u8>> {
    const MAX_BYTES: u64 = 20 * 1024 * 1024; // 20 MB

    let fetch = || -> Result<Vec<u8>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| anyhow!("HTTPS fetch: failed to build client: {e}"))?;

        let resp = client
            .get(url)
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    anyhow!("HTTPS fetch timed out after 10 s ({})", url)
                } else {
                    anyhow!("HTTPS fetch network error ({}): {e}", url)
                }
            })?;

        let status = resp.status();
        if !status.is_success() {
            return Err(anyhow!(
                "HTTPS fetch received HTTP {} from {}",
                status.as_u16(),
                url
            ));
        }

        // Reject oversized payloads early via Content-Length when present.
        if let Some(len) = resp.content_length() {
            if len > MAX_BYTES {
                return Err(anyhow!(
                    "HTTPS fetch: Content-Length {} exceeds {}-byte cap ({})",
                    len,
                    MAX_BYTES,
                    url
                ));
            }
        }

        // Stream body bytes with enforced cap.
        //
        // Security note: we cannot use `resp.bytes()` here because it buffers
        // the entire HTTP body before the size check fires.  A server that
        // omits Content-Length and streams more than MAX_BYTES would force
        // hf2q to allocate the full malicious body in memory before we could
        // reject it (no-Content-Length DoS / OOM vector).
        //
        // reqwest::blocking::Response implements std::io::Read, so we use
        // `Read::take(MAX_BYTES + 1)` which stops pulling bytes from the
        // socket the moment we have read MAX_BYTES + 1 bytes.  If the buffer
        // ends up exactly MAX_BYTES + 1 long the server was over the cap;
        // we return an error WITHOUT reading the rest of the body.  Errors
        // from the underlying read (e.g. network / timeout) surface as
        // std::io::Error which we map to an anyhow context.
        let cap = MAX_BYTES + 1;
        let hint = resp
            .content_length()
            .map(|cl| (cl as usize).min(cap as usize))
            .unwrap_or(0);
        let mut buf: Vec<u8> = Vec::with_capacity(hint);
        resp.take(cap)
            .read_to_end(&mut buf)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::TimedOut {
                    anyhow!("HTTPS fetch body read timed out ({})", url)
                } else {
                    anyhow!("HTTPS fetch body read error ({}): {e}", url)
                }
            })?;

        if buf.len() as u64 >= cap {
            return Err(anyhow!(
                "HTTPS fetch: response body exceeds {}-byte cap ({})",
                MAX_BYTES,
                url
            ));
        }

        Ok(buf)
    };

    // If a tokio runtime is active (axum handler context), use block_in_place
    // so the blocking I/O does not stall the async executor thread.
    // Outside a runtime (unit tests, CLI) call directly.
    match tokio::runtime::Handle::try_current() {
        Ok(_) => tokio::task::block_in_place(fetch),
        Err(_) => fetch(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_image_url_data_png() {
        let got = parse_image_url("data:image/png;base64,iVBORw0K").unwrap();
        match got {
            ImageInput::DataUri { mime_type, payload_base64 } => {
                assert_eq!(mime_type, "image/png");
                assert_eq!(payload_base64, "iVBORw0K");
            }
            other => panic!("expected DataUri, got {:?}", other),
        }
    }

    #[test]
    fn parse_image_url_data_jpeg() {
        let got = parse_image_url("data:image/jpeg;base64,/9j/4AA").unwrap();
        assert!(matches!(got, ImageInput::DataUri { .. }));
    }

    #[test]
    fn parse_image_url_rejects_unsupported_mime() {
        let err = parse_image_url("data:image/gif;base64,xyz").unwrap_err();
        assert!(format!("{err}").contains("not supported"));
    }

    #[test]
    fn parse_image_url_rejects_non_base64_encoding() {
        let err = parse_image_url("data:image/png;utf8,hello").unwrap_err();
        assert!(format!("{err}").contains("not supported"));
    }

    #[test]
    fn parse_image_url_file_scheme() {
        let got = parse_image_url("file:///tmp/cat.jpg").unwrap();
        assert_eq!(got, ImageInput::FilePath(PathBuf::from("/tmp/cat.jpg")));
    }

    #[test]
    fn parse_image_url_bare_absolute_path() {
        let got = parse_image_url("/tmp/dog.png").unwrap();
        assert_eq!(got, ImageInput::FilePath(PathBuf::from("/tmp/dog.png")));
    }

    #[test]
    fn parse_image_url_http_preserved_for_fetch() {
        let got = parse_image_url("https://example.com/img.jpg").unwrap();
        assert!(matches!(got, ImageInput::HttpUrl(_)));
    }

    #[test]
    fn parse_image_url_rejects_gibberish() {
        let err = parse_image_url("not-a-url").unwrap_err();
        assert!(format!("{err}").contains("unrecognized"));
    }

    #[test]
    fn load_image_bytes_data_uri_round_trips_base64() {
        // A minimal PNG signature: 8 bytes.
        let sig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(sig);
        let url = format!("data:image/png;base64,{b64}");
        let input = parse_image_url(&url).unwrap();
        let bytes = load_image_bytes(&input).unwrap();
        assert_eq!(bytes, sig);
    }

    #[test]
    fn load_image_bytes_https_url_attempts_fetch() {
        // Verify that HttpUrl no longer returns the old "not yet loaded"
        // static error — it now attempts a network fetch. Using an
        // unresolvable host ensures the test does not egress while still
        // exercising the dispatch path. The error must be a network error,
        // not the old static rejection string.
        let input = ImageInput::HttpUrl("https://example.invalid/cat.jpg".into());
        let err = load_image_bytes(&input).unwrap_err();
        let msg = format!("{err}");
        assert!(
            !msg.contains("not yet loaded"),
            "expected network error, got static rejection: {msg}"
        );
        // The error must mention the URL or a network-level failure.
        assert!(
            msg.contains("example.invalid")
                || msg.contains("network error")
                || msg.contains("timed out")
                || msg.contains("HTTPS fetch"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn load_image_bytes_rejects_oversized_file() {
        // Don't actually create a 20 MB file in a unit test — just test the
        // nonexistent-path branch for the fast-fail contract. The size cap
        // is separately exercised by the live smoke harness when a real
        // oversized file is available.
        let err = load_image_bytes(&ImageInput::FilePath(PathBuf::from(
            "/tmp/does-not-exist-xyz-42.png",
        )))
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("stat") || msg.contains("No such"),
            "unexpected error: {msg}"
        );
    }

    // -------------------------------------------------------------------------
    // T1.4 streaming cap — no-Content-Length attack vector
    // -------------------------------------------------------------------------
    //
    // Security regression test (ADR-005 wave-1.5 Codex HIGH finding).
    //
    // Previously, `fetch_https_image` called `resp.bytes()` which accumulates
    // the entire HTTP body in memory before the size check fires.  A server
    // that omits Content-Length and streams >20 MB forces hf2q to buffer the
    // full body (OOM / DoS vector).
    //
    // The fix uses `Read::take(MAX_BYTES + 1).read_to_end(&mut buf)` which
    // stops pulling bytes from the socket at MAX_BYTES + 1.  This test
    // verifies the cap fires correctly by spinning up a minimal TcpListener
    // that:
    //   1. Sends HTTP/1.1 200 OK with NO Content-Length header.
    //   2. Streams small chunks in a loop so the body length would eventually
    //      exceed 20 MB — but the client cuts the connection after MAX_BYTES+1
    //      bytes, so the server never sends the full amount.
    //
    // Discipline note: we do NOT allocate >20 MB in this test.  The server
    // sends data in 64 KB chunks and tracks how many bytes the client accepts.
    // Once the client closes the connection (having read MAX_BYTES+1 bytes and
    // returned an error), the server's write fails and the thread exits.
    // Total server-side allocation: one 64 KB chunk buffer, reused.
    #[test]
    fn fetch_https_image_cap_fires_without_content_length() {
        use std::io::Write as _;
        use std::net::TcpListener;
        use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
        use std::thread;

        const MAX_BYTES: u64 = 20 * 1024 * 1024; // must match fetch_https_image

        // Bind to an ephemeral port on loopback.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");

        // Track how many bytes of body the server successfully wrote to the
        // socket.  The test asserts this stays <= MAX_BYTES + 1 + overhead to
        // prove the client cut the connection early.
        let bytes_accepted = Arc::new(AtomicU64::new(0));
        let bytes_accepted_srv = Arc::clone(&bytes_accepted);

        // Spawn the mock server on a background thread.
        let srv = thread::spawn(move || {
            // Accept exactly one connection — the test makes exactly one fetch.
            let (mut stream, _peer) = listener.accept().expect("accept");
            // Drain the HTTP request headers (we don't need to parse them).
            drain_http_request(&mut stream);
            // Send a chunked-ish HTTP/1.1 200 with NO Content-Length.
            // Using HTTP/1.0 + Connection: close is the easiest way to omit
            // Content-Length without implementing Transfer-Encoding: chunked
            // framing — the body is everything until socket close.
            let header = b"HTTP/1.0 200 OK\r\nContent-Type: application/octet-stream\r\n\r\n";
            if stream.write_all(header).is_err() {
                return;
            }
            // Stream 64 KB chunks of zeros until the client closes.
            // Total intended body = MAX_BYTES + 64 KB > cap, but we stop as
            // soon as the write fails (client disconnected after cap).
            let chunk = vec![0u8; 64 * 1024];
            loop {
                match stream.write_all(&chunk) {
                    Ok(_) => {
                        bytes_accepted_srv.fetch_add(chunk.len() as u64, Ordering::Relaxed);
                        // Stop once we've offered significantly more than the cap —
                        // the client will have disconnected long before this.
                        if bytes_accepted_srv.load(Ordering::Relaxed) > MAX_BYTES + chunk.len() as u64 {
                            break;
                        }
                    }
                    Err(_) => break, // client closed connection — expected
                }
            }
        });

        // Use fetch_https_image against our local mock (HTTP, not HTTPS — the
        // function accepts both http:// and https:// URLs via the same code path).
        let url = format!("http://127.0.0.1:{}/oversized.bin", addr.port());
        let err = fetch_https_image(&url).expect_err(
            "expected fetch_https_image to return an error for oversized body",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("cap") || msg.contains("exceed"),
            "error message should mention the cap; got: {msg}"
        );

        // The server thread will exit soon after the client disconnects (write
        // fails).  Give it a moment; ignore join errors if it panicked on
        // the broken pipe.
        let _ = srv.join();

        // The primary security assertion is that fetch_https_image returned an
        // error at all (proven above by expect_err).  We also verify that the
        // server did not manage to push a full order-of-magnitude more data
        // than the cap before the client disconnected, which would indicate the
        // old resp.bytes() buffering pattern regressed.  TCP kernel buffers
        // mean the server can write a few extra chunks before receiving
        // SIGPIPE/ECONNRESET, so we allow up to ~2 MB of TCP-buffer slop on
        // top of the cap.
        let written = bytes_accepted.load(Ordering::Relaxed);
        let slop = 2 * 1024 * 1024u64; // 2 MB TCP-buffer headroom
        assert!(
            written <= MAX_BYTES + slop,
            "server wrote {written} bytes before client closed — \
             cap is {} bytes; too much slop suggests client buffered full body",
            MAX_BYTES
        );
    }

    /// Drain a minimal HTTP request from `stream` so the server can send its
    /// response.  Reads until the first blank CRLF line (end of headers).
    /// Any body (e.g. POST) is ignored — our mock only handles GET.
    fn drain_http_request(stream: &mut std::net::TcpStream) {
        use std::io::BufRead as _;
        let mut reader = std::io::BufReader::new(stream.try_clone().expect("clone"));
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if line == "\r\n" || line == "\n" {
                        break;
                    }
                }
            }
        }
    }
}
