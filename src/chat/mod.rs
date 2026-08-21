mod client;
mod control;
pub(crate) mod endpoint;
mod local;
mod render;
mod sse;
mod transcript;
mod wire;

use std::io::{BufRead, Write};

use anyhow::{bail, Context, Result};

use crate::cli::ChatArgs;

use client::{fetch_models, ChatClient};
use control::{looks_like_hub_model, Hf2qControl};
use endpoint::{EndpointResolver, EndpointSession};
use local::AutomaticEndpointResolver;
use render::StreamRenderer;
use wire::{Model, RequestOptions, ThinkingMode};

pub(crate) fn cmd_chat(args: ChatArgs) -> Result<()> {
    let mut resolver = AutomaticEndpointResolver;
    cmd_chat_with_resolver(args, &mut resolver)
}

/// Injection seam used by the discovery/activation integration lane. The
/// resolver supplies a verified endpoint and, only when it launched that
/// endpoint itself, an owned Child handle in the EndpointSession.
pub(crate) fn cmd_chat_with_resolver(
    args: ChatArgs,
    resolver: &mut impl EndpointResolver,
) -> Result<()> {
    let mut session = resolver.resolve(&args)?;
    if args.keep_serving {
        if let Some(log_path) = session.detach()? {
            eprintln!("detached hf2q server log: {}", log_path.display());
        }
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build diagnostic chat runtime")?;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    runtime.block_on(async {
        let auth_token = std::env::var("HF2Q_AUTH_TOKEN").ok();
        let run_result = run_session(
            args,
            &mut session,
            auth_token.clone(),
            &mut input,
            &mut output,
        )
        .await;
        // Cleanup lives outside run_session so model-probe, client-construction,
        // and terminal I/O failures still stop a server this chat launched.
        let cleanup_http = reqwest::Client::new();
        let shutdown_result = session
            .shutdown_if_owned(&cleanup_http, auth_token.as_deref())
            .await;
        match (run_result, shutdown_result) {
            (Err(run_error), Err(shutdown_error)) => Err(anyhow::anyhow!(
                "{run_error:#}; chat-owned server cleanup also failed: {shutdown_error:#}"
            )),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    })
}

async fn run_session(
    args: ChatArgs,
    session: &mut EndpointSession,
    auth_token: Option<String>,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    let probe_http = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .context("build endpoint probe client")?;
    let mut model = match args.model.clone() {
        Some(model) => model,
        None => {
            choose_remote_model(&probe_http, session, auth_token.as_deref(), input, output).await?
        }
    };
    let control = Hf2qControl::detect(
        &probe_http,
        session.endpoint(),
        auth_token.as_deref(),
        session.expects_hf2q_control(),
    )
    .await?;
    if let Some(control) = &control {
        let Some(request_model) = activate_diagnostic_model(
            control,
            &model,
            args.quant.as_deref(),
            args.artifact.as_deref(),
            auth_token.as_deref(),
            input,
            output,
        )
        .await?
        else {
            bail!("model switch declined");
        };
        model = request_model;
    } else if args.quant.is_some() || args.artifact.is_some() {
        bail!("--quant and --artifact require an hf2q hosted-GGUF capability endpoint");
    }
    let options = RequestOptions {
        temperature: args.temperature,
        top_p: args.top_p,
        max_tokens: args.max_tokens,
        seed: args.seed,
        reasoning_effort: args.reasoning_effort,
        thinking: ThinkingMode::Auto,
    };
    let mut client = ChatClient::new(
        session.endpoint().clone(),
        model,
        args.system,
        options,
        auth_token.clone(),
    )?;
    client.set_hf2q_non_evicting(control.is_some());

    writeln!(
        output,
        "hf2q diagnostic chat — {} @ {}",
        client.model(),
        session.endpoint().base_url()
    )?;
    writeln!(
        output,
        "commands: /new /model [id] /thinking auto|on|off /status /detach /quit"
    )?;

    interactive_loop(
        &mut client,
        session,
        control.as_ref(),
        auth_token.as_deref(),
        input,
        output,
    )
    .await
}

async fn activate_diagnostic_model(
    control: &Hf2qControl,
    model: &str,
    quant: Option<&str>,
    artifact: Option<&str>,
    auth_token: Option<&str>,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<Option<String>> {
    if looks_like_hub_model(model) {
        if quant.is_none() && artifact.is_none() {
            if let Some(resident) = control.probe_resident(model, auth_token).await? {
                writeln!(output, "model activation: resident")?;
                return Ok(Some(resident));
            }
        }
        let candidate = control
            .select_hub_gguf(model, quant, artifact, auth_token, input, output)
            .await?
            .context("hf2q did not issue activation authority for the selected artifact")?;
        control
            .ensure_active(model, Some(&candidate), auth_token, input, output)
            .await
    } else {
        control
            .ensure_active(model, None, auth_token, input, output)
            .await
    }
}

async fn interactive_loop(
    client: &mut ChatClient,
    session: &mut EndpointSession,
    control: Option<&Hf2qControl>,
    auth_token: Option<&str>,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    let mut line = String::new();
    loop {
        write!(output, "> ")?;
        output.flush()?;
        line.clear();
        if input.read_line(&mut line)? == 0 {
            writeln!(output)?;
            return Ok(());
        }
        let value = line.trim();
        if value.is_empty() {
            continue;
        }
        if value.starts_with('/') {
            if handle_command(client, session, control, auth_token, value, input, output).await? {
                return Ok(());
            }
            continue;
        }

        let mut renderer = StreamRenderer::new(output, client.model().to_owned());
        match client
            .send_turn(value, |update| renderer.render(update))
            .await
        {
            Ok(response) => {
                renderer.complete(&response)?;
                if let Some(control) = control {
                    control.write_runtime_status(auth_token, output).await?;
                }
            }
            Err(error) => renderer.fail(&error)?,
        }
    }
}

/// Returns true when the caller should leave the chat loop.
async fn handle_command(
    client: &mut ChatClient,
    session: &mut EndpointSession,
    control: Option<&Hf2qControl>,
    auth_token: Option<&str>,
    command: &str,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<bool> {
    let mut words = command.split_whitespace();
    match words.next().unwrap_or_default() {
        "/quit" => Ok(true),
        "/new" => {
            client.reset();
            writeln!(output, "context cleared")?;
            Ok(false)
        }
        "/detach" => {
            if let Some(log_path) = session.detach()? {
                writeln!(
                    output,
                    "detached; the chat-owned server will keep serving (log: {})",
                    log_path.display()
                )?;
            } else {
                writeln!(
                    output,
                    "endpoint is external; chat never owned its lifecycle"
                )?;
            }
            Ok(false)
        }
        "/status" => {
            let lifecycle = if !session.is_owned() {
                "external"
            } else if session.is_detached() {
                "owned, detached"
            } else {
                "owned, stops on exit"
            };
            writeln!(
                output,
                "model={} url={} thinking={} lifecycle={}",
                client.model(),
                session.endpoint().base_url(),
                client.thinking().as_str(),
                lifecycle
            )?;
            if let Some(control) = control {
                control.write_runtime_status(auth_token, output).await?;
            }
            Ok(false)
        }
        "/thinking" => {
            let Some(mode) = words.next().and_then(ThinkingMode::parse) else {
                writeln!(output, "usage: /thinking auto|on|off")?;
                return Ok(false);
            };
            if words.next().is_some() {
                writeln!(output, "usage: /thinking auto|on|off")?;
                return Ok(false);
            }
            client.set_thinking(mode);
            writeln!(output, "thinking={}", mode.as_str())?;
            Ok(false)
        }
        "/model" => {
            let mut model = if let Some(model) = words.next() {
                if words.next().is_some() {
                    writeln!(output, "usage: /model [id]")?;
                    return Ok(false);
                }
                model.to_owned()
            } else {
                choose_remote_model(client.http(), session, auth_token, input, output).await?
            };
            if let Some(control) = control {
                let Some(request_model) = activate_diagnostic_model(
                    control, &model, None, None, auth_token, input, output,
                )
                .await?
                else {
                    writeln!(output, "model unchanged")?;
                    return Ok(false);
                };
                model = request_model;
            }
            client.set_model(model);
            writeln!(output, "model={} (context cleared)", client.model())?;
            Ok(false)
        }
        other => {
            writeln!(output, "unknown command: {other}")?;
            Ok(false)
        }
    }
}

async fn choose_remote_model(
    http: &reqwest::Client,
    session: &EndpointSession,
    auth_token: Option<&str>,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<String> {
    let mut models = fetch_models(http, session.endpoint(), auth_token).await?;
    if models.is_empty() {
        bail!("endpoint advertised no models");
    }
    models.sort_by_key(|model| (!model.loaded.unwrap_or(false), model.id.clone()));
    if models.len() == 1 {
        return Ok(models.remove(0).id);
    }
    pick_model(&models, input, output)
}

fn pick_model(
    models: &[Model],
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<String> {
    writeln!(output, "available models:")?;
    for (index, model) in models.iter().enumerate() {
        let state = if model.loaded.unwrap_or(false) {
            " [loaded]"
        } else {
            ""
        };
        writeln!(output, "  {}. {}{}", index + 1, model.id, state)?;
    }
    write!(output, "model> ")?;
    output.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        bail!("input ended before a model was selected");
    }
    let index: usize = line
        .trim()
        .parse()
        .context("model selection must be a number")?;
    models
        .get(
            index
                .checked_sub(1)
                .context("model selection starts at 1")?,
        )
        .map(|model| model.id.clone())
        .context("model selection is out of range")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::oneshot;

    #[test]
    fn picker_marks_loaded_models_and_returns_numbered_choice() {
        let models = vec![
            Model {
                id: "resident".into(),
                loaded: Some(true),
            },
            Model {
                id: "cached".into(),
                loaded: Some(false),
            },
        ];
        let mut input = std::io::Cursor::new(b"2\n");
        let mut output = Vec::new();
        assert_eq!(
            pick_model(&models, &mut input, &mut output).unwrap(),
            "cached"
        );
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("resident [loaded]"));
    }

    #[derive(Clone)]
    struct ActivationFixture {
        ambiguous_probe: bool,
        catalog_calls: Arc<AtomicUsize>,
    }

    async fn runtime_fixture() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "schema_version":"hf2q.runtime.v1",
            "backend":"mlx-native",
            "capabilities":{
                "model_activation":"hf2q.model-activation.v2",
                "artifact_resolution":"hf2q.artifact-resolution.v2",
                "non_evicting_load":true,
                "explicit_revision_bound_switch":true,
                "request_generation_leases":true,
                "diagnostic_no_evict_header":{"name":"x-hf2q-diagnostic-no-evict","value":"1"}
            },
            "pool":{
                "revision":1,"loaded_count":1,"capacity_models":2,
                "total_resident_bytes":42,"memory_budget_bytes":100,
                "resident":[]
            }
        }))
    }

    async fn catalog_fixture(State(fixture): State<ActivationFixture>) -> Json<serde_json::Value> {
        fixture.catalog_calls.fetch_add(1, Ordering::SeqCst);
        Json(serde_json::json!({
            "schema_version":"hf2q.artifact-resolution.v2",
            "repository":"owner/model",
            "revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "candidates":[{
                "candidate_id":"q6-candidate",
                "filename":"model-q6_k.gguf",
                "bytes":42,
                "quant_hint":"Q6_K",
                "role":"text_model",
                "selectable":true,
                "unavailable_reason":null
            }]
        }))
    }

    async fn activation_fixture(
        State(fixture): State<ActivationFixture>,
        Json(request): Json<serde_json::Value>,
    ) -> axum::response::Response {
        use axum::response::IntoResponse;
        if request["action"] == "probe" {
            if fixture.ambiguous_probe {
                return axum::http::StatusCode::NOT_FOUND.into_response();
            }
            return Json(serde_json::json!({
                "schema_version":"hf2q.model-activation.v2",
                "status":"resident",
                "pool_revision":1,
                "request_model":"hf://owner/model@aaaaaaaa/model-q6_k.gguf#bbbb"
            }))
            .into_response();
        }
        assert_eq!(request["candidate_id"], "q6-candidate");
        Json(serde_json::json!({
            "schema_version":"hf2q.model-activation.v2",
            "status":"resident",
            "pool_revision":1,
            "request_model":"hf://owner/model@aaaaaaaa/model-q6_k.gguf#bbbb"
        }))
        .into_response()
    }

    async fn serve_activation_fixture(
        ambiguous_probe: bool,
    ) -> (endpoint::Endpoint, Arc<AtomicUsize>, oneshot::Sender<()>) {
        let catalog_calls = Arc::new(AtomicUsize::new(0));
        let fixture = ActivationFixture {
            ambiguous_probe,
            catalog_calls: Arc::clone(&catalog_calls),
        };
        let app = Router::new()
            .route("/hf2q/v1/runtime", get(runtime_fixture))
            .route("/hf2q/v1/models/catalog", get(catalog_fixture))
            .route("/hf2q/v1/models/activate", post(activation_fixture))
            .with_state(fixture);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (stop_tx, stop_rx) = oneshot::channel();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stop_rx.await;
                })
                .await
                .unwrap();
        });
        (
            endpoint::Endpoint::discovered_loopback(port),
            catalog_calls,
            stop_tx,
        )
    }

    #[tokio::test]
    async fn unique_resident_repository_uses_zero_catalog_calls() {
        let (endpoint, catalog_calls, stop) = serve_activation_fixture(false).await;
        let control = Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();
        let model = activate_diagnostic_model(
            &control,
            "owner/model",
            None,
            None,
            None,
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            &mut Vec::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            model.as_deref(),
            Some("hf://owner/model@aaaaaaaa/model-q6_k.gguf#bbbb")
        );
        assert_eq!(catalog_calls.load(Ordering::SeqCst), 0);
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn ambiguous_resident_repository_falls_through_to_exact_hosted_candidate() {
        let (endpoint, catalog_calls, stop) = serve_activation_fixture(true).await;
        let control = Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();
        let model = activate_diagnostic_model(
            &control,
            "owner/model",
            None,
            None,
            None,
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            &mut Vec::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            model.as_deref(),
            Some("hf://owner/model@aaaaaaaa/model-q6_k.gguf#bbbb")
        );
        assert_eq!(catalog_calls.load(Ordering::SeqCst), 1);
        let _ = stop.send(());
    }
}
