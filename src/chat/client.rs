use anyhow::{bail, Context, Result};
use futures::StreamExt;
use serde::de::DeserializeOwned;

use super::endpoint::Endpoint;
use super::sse::{CompletedResponse, SseDecoder, StreamUpdate};
use super::transcript::Transcript;
use super::wire::{ChatRequest, Model, ModelList, RequestOptions, ThinkingMode};

const MAX_AUXILIARY_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

pub(crate) struct ChatClient {
    http: reqwest::Client,
    endpoint: Endpoint,
    model: String,
    transcript: Transcript,
    options: RequestOptions,
    auth_token: Option<String>,
    hf2q_non_evicting: bool,
}

impl ChatClient {
    pub(crate) fn new(
        endpoint: Endpoint,
        model: String,
        system: Option<String>,
        options: RequestOptions,
        auth_token: Option<String>,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .context("build diagnostic chat HTTP client")?;
        Ok(Self {
            http,
            endpoint,
            model,
            transcript: Transcript::new(system),
            options,
            auth_token,
            hf2q_non_evicting: false,
        })
    }

    pub(crate) fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn thinking(&self) -> ThinkingMode {
        self.options.thinking
    }

    pub(crate) fn set_thinking(&mut self, mode: ThinkingMode) {
        self.options.thinking = mode;
    }

    pub(crate) fn set_hf2q_non_evicting(&mut self, enabled: bool) {
        self.hf2q_non_evicting = enabled;
    }

    pub(crate) fn set_model(&mut self, model: String) {
        if self.model != model {
            self.model = model;
            self.transcript.reset();
        }
    }

    pub(crate) fn reset(&mut self) {
        self.transcript.reset();
    }

    pub(crate) async fn send_turn(
        &mut self,
        user: &str,
        mut observe: impl FnMut(StreamUpdate) -> Result<()>,
    ) -> Result<CompletedResponse> {
        let pending = self.transcript.pending(user);
        let request = ChatRequest::new(&self.model, &pending, &self.options);
        let mut builder = self
            .http
            .post(self.endpoint.route("/v1/chat/completions"))
            .json(&request);
        if self.hf2q_non_evicting {
            builder = builder.header(super::control::NON_EVICTING_HEADER, "1");
        }
        if let Some(token) = &self.auth_token {
            builder = builder.bearer_auth(token);
        }
        let response = builder.send().await.context("send chat request")?;
        let status = response.status();
        if !status.is_success() {
            let detail = read_text_bounded(response, "chat error response").await?;
            bail!("chat endpoint returned HTTP {status}: {}", compact(&detail));
        }

        let mut decoder = SseDecoder::default();
        let mut bytes = response.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            for update in decoder.push(&chunk.context("read chat SSE stream")?)? {
                observe(update)?;
            }
        }
        let completed = decoder.finish()?;
        self.transcript.commit(user, &completed);
        Ok(completed)
    }
}

pub(crate) async fn fetch_models(
    http: &reqwest::Client,
    endpoint: &Endpoint,
    auth_token: Option<&str>,
) -> Result<Vec<Model>> {
    let mut builder = http.get(endpoint.route("/v1/models"));
    if let Some(token) = auth_token {
        builder = builder.bearer_auth(token);
    }
    let response = builder.send().await.context("query endpoint models")?;
    let status = response.status();
    if !status.is_success() {
        let detail = read_text_bounded(response, "model-list error response").await?;
        bail!(
            "model endpoint returned HTTP {status}: {}",
            compact(&detail)
        );
    }
    let models: ModelList = decode_json_bounded(response, "/v1/models response").await?;
    Ok(models.data)
}

pub(crate) async fn decode_json_bounded<T: DeserializeOwned>(
    response: reqwest::Response,
    label: &str,
) -> Result<T> {
    let bytes = read_bytes_bounded(response, label).await?;
    serde_json::from_slice(&bytes).with_context(|| format!("decode {label}"))
}

pub(crate) async fn read_text_bounded(response: reqwest::Response, label: &str) -> Result<String> {
    let bytes = read_bytes_bounded(response, label).await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

async fn read_bytes_bounded(response: reqwest::Response, label: &str) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_AUXILIARY_RESPONSE_BYTES as u64)
    {
        bail!("{label} exceeded the 4 MiB diagnostic-client limit");
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read {label}"))?;
        if body.len().saturating_add(chunk.len()) > MAX_AUXILIARY_RESPONSE_BYTES {
            bail!("{label} exceeded the 4 MiB diagnostic-client limit");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn compact(value: &str) -> String {
    const MAX: usize = 512;
    let one_line = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= MAX {
        one_line
    } else {
        format!("{}…", one_line.chars().take(MAX).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{header, HeaderMap, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use futures::stream;
    use serde_json::Value;
    use tokio::sync::oneshot;

    use super::*;

    #[derive(Clone)]
    struct Recorded(Arc<Mutex<Vec<Value>>>);

    async fn successful_chat(
        State(recorded): State<Recorded>,
        Json(request): Json<Value>,
    ) -> Response<Body> {
        let turn = {
            let mut requests = recorded.0.lock().unwrap();
            requests.push(request);
            requests.len()
        };
        let answer = if turn == 1 { "one" } else { "two" };
        let parts = vec![
            Ok::<_, Infallible>(format!(
                "data: {{\"choices\":[{{\"delta\":{{\"reasoning_content\":\"check {turn}\"}}}}]}}\n\n"
            )),
            Ok(format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{answer}\"}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{turn},\"completion_tokens\":1,\"total_tokens\":{}}},\"x_hf2q_timing\":{{\"time_to_first_token_ms\":3.5}}}}\n\n",
                turn + 1
            )),
            Ok("data: [DONE]\n\n".to_owned()),
        ];
        Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(stream::iter(parts)))
            .unwrap()
    }

    async fn error_chat() -> Response<Body> {
        Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"error\"}]}\n\n",
                "data: [DONE]\n\n"
            )))
            .unwrap()
    }

    async fn truncated_chat() -> Response<Body> {
        Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
            ))
            .unwrap()
    }

    #[derive(Clone)]
    struct AuthSeen(Arc<Mutex<Vec<String>>>);

    #[derive(Clone, Default)]
    struct DiagnosticHeaders(Arc<Mutex<Vec<String>>>);

    fn record_auth(headers: &HeaderMap, seen: &AuthSeen) {
        seen.0.lock().unwrap().push(
            headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_owned(),
        );
    }

    async fn authenticated_models(State(seen): State<AuthSeen>, headers: HeaderMap) -> Json<Value> {
        record_auth(&headers, &seen);
        Json(serde_json::json!({"object":"list","data":[{"id":"model-a"}]}))
    }

    async fn authenticated_chat(
        State(seen): State<AuthSeen>,
        headers: HeaderMap,
    ) -> Response<Body> {
        record_auth(&headers, &seen);
        Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n"
            )))
            .unwrap()
    }

    async fn diagnostic_chat(
        State(seen): State<DiagnosticHeaders>,
        headers: HeaderMap,
    ) -> Response<Body> {
        seen.0.lock().unwrap().push(
            headers
                .get(super::super::control::NON_EVICTING_HEADER)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_owned(),
        );
        Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n"
            )))
            .unwrap()
    }

    async fn serve(router: Router) -> (Endpoint, oneshot::Sender<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (stop_tx, stop_rx) = oneshot::channel();
        tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stop_rx.await;
                })
                .await
                .unwrap();
        });
        (Endpoint::discovered_loopback(port), stop_tx)
    }

    #[tokio::test]
    async fn network_two_turn_request_contains_exact_successful_context() {
        let recorded = Recorded(Arc::new(Mutex::new(Vec::new())));
        let router = Router::new()
            .route("/v1/chat/completions", post(successful_chat))
            .with_state(recorded.clone());
        let (endpoint, stop) = serve(router).await;
        let mut client = ChatClient::new(
            endpoint,
            "model-a".into(),
            None,
            RequestOptions::default(),
            None,
        )
        .unwrap();

        let first = client.send_turn("first", |_| Ok(())).await.unwrap();
        let second = client.send_turn("second", |_| Ok(())).await.unwrap();
        assert_eq!(first.reasoning_content, "check 1");
        assert_eq!(second.content, "two");

        let requests = recorded.0.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0],
            serde_json::json!({
                "model": "model-a",
                "messages": [{"role":"user","content":"first"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            })
        );
        assert_eq!(
            requests[1],
            serde_json::json!({
                "model": "model-a",
                "messages": [
                    {"role":"user","content":"first"},
                    {"role":"assistant","content":"one","reasoning_content":"check 1"},
                    {"role":"user","content":"second"}
                ],
                "stream": true,
                "stream_options": {"include_usage": true}
            })
        );
        drop(requests);
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn network_error_and_truncation_do_not_mutate_transcript() {
        for router in [
            Router::new().route("/v1/chat/completions", post(error_chat)),
            Router::new().route("/v1/chat/completions", post(truncated_chat)),
        ] {
            let (endpoint, stop) = serve(router).await;
            let mut client = ChatClient::new(
                endpoint,
                "model-a".into(),
                None,
                RequestOptions::default(),
                None,
            )
            .unwrap();
            assert!(client.send_turn("discard me", |_| Ok(())).await.is_err());
            assert!(client.transcript.messages().is_empty());
            let _ = stop.send(());
        }
    }

    #[tokio::test]
    async fn bearer_auth_is_used_for_models_and_chat_without_persistence() {
        let seen = AuthSeen(Arc::new(Mutex::new(Vec::new())));
        let router = Router::new()
            .route("/v1/models", get(authenticated_models))
            .route("/v1/chat/completions", post(authenticated_chat))
            .with_state(seen.clone());
        let (endpoint, stop) = serve(router).await;
        let probe = reqwest::Client::new();
        let models = fetch_models(&probe, &endpoint, Some("test-secret"))
            .await
            .unwrap();
        assert_eq!(models[0].id, "model-a");
        let mut client = ChatClient::new(
            endpoint,
            "model-a".into(),
            None,
            RequestOptions::default(),
            Some("test-secret".into()),
        )
        .unwrap();
        client.send_turn("hello", |_| Ok(())).await.unwrap();
        assert_eq!(
            *seen.0.lock().unwrap(),
            vec!["Bearer test-secret", "Bearer test-secret"]
        );
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn hf2q_capability_enables_non_evicting_header_without_changing_body() {
        let seen = DiagnosticHeaders::default();
        let router = Router::new()
            .route("/v1/chat/completions", post(diagnostic_chat))
            .with_state(seen.clone());
        let (endpoint, stop) = serve(router).await;
        let mut client = ChatClient::new(
            endpoint,
            "model-a".into(),
            None,
            RequestOptions::default(),
            None,
        )
        .unwrap();
        client.set_hf2q_non_evicting(true);
        client.send_turn("hello", |_| Ok(())).await.unwrap();
        assert_eq!(*seen.0.lock().unwrap(), vec!["1"]);
        let _ = stop.send(());
    }
}
