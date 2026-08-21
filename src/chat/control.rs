//! Client for hf2q's optional ADR-047 diagnostic control plane.

use std::io::{BufRead, Write};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::client::{decode_json_bounded, read_text_bounded};
use super::endpoint::Endpoint;

const RUNTIME_SCHEMA: &str = "hf2q.runtime.v1";
const ACTIVATION_SCHEMA: &str = "hf2q.model-activation.v2";
const ARTIFACT_SCHEMA: &str = "hf2q.artifact-resolution.v2";
pub(crate) const NON_EVICTING_HEADER: &str = "x-hf2q-diagnostic-no-evict";

pub(crate) struct Hf2qControl {
    http: reqwest::Client,
    endpoint: Endpoint,
    artifact_resolution: bool,
}

#[derive(Debug, Deserialize)]
struct RuntimeView {
    schema_version: String,
    capabilities: RuntimeCapabilities,
    pool: RuntimePool,
}

#[derive(Debug, Deserialize)]
struct RuntimeCapabilities {
    model_activation: String,
    #[serde(default)]
    non_evicting_load: bool,
    #[serde(default)]
    diagnostic_no_evict_header: Option<DiagnosticHeaderCapability>,
    #[serde(default)]
    artifact_resolution: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DiagnosticHeaderCapability {
    name: String,
    value: String,
}

#[derive(Debug, Deserialize)]
struct RuntimePool {
    revision: u64,
    loaded_count: usize,
    capacity_models: usize,
    total_resident_bytes: u64,
    memory_budget_bytes: u64,
    #[serde(default)]
    resident: Vec<ActivationVictim>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ActivationVictim {
    pool_key: String,
    quant: String,
    bytes_resident: u64,
    generation: u64,
}

#[derive(Debug, Deserialize)]
struct ActivationSuccess {
    schema_version: String,
    status: String,
    pool_revision: u64,
    #[serde(default)]
    request_model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ActivationConflict {
    schema_version: String,
    status: String,
    pool_revision: u64,
    victims: Vec<ActivationVictim>,
    projected_bytes: u64,
    requires_explicit_switch: bool,
}

#[derive(Debug, Serialize)]
struct ActivationRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    action: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_revision: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    victims: Vec<ActivationVictim>,
}

#[derive(Debug, Deserialize)]
struct ArtifactCatalog {
    schema_version: String,
    candidates: Vec<ArtifactCandidate>,
}

#[derive(Clone, Debug, Deserialize)]
struct ArtifactCandidate {
    candidate_id: Option<String>,
    filename: String,
    bytes: u64,
    quant_hint: Option<String>,
    role: String,
    selectable: bool,
    unavailable_reason: Option<String>,
}

enum ActivationResult {
    Ready(ActivationSuccess),
    Conflict(ActivationConflict),
}

impl Hf2qControl {
    pub(crate) async fn detect(
        http: &reqwest::Client,
        endpoint: &Endpoint,
        auth_token: Option<&str>,
        required: bool,
    ) -> Result<Option<Self>> {
        let mut request = http.get(endpoint.route("/hf2q/v1/runtime"));
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) if !required => {
                tracing::debug!(%error, "explicit endpoint did not answer the optional hf2q runtime probe");
                return Ok(None);
            }
            Err(error) => return Err(error).context("probe hf2q runtime API"),
        };
        if matches!(
            response.status(),
            reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::METHOD_NOT_ALLOWED
        ) {
            if required {
                bail!("the discovered hf2q server does not expose the diagnostic lifecycle API");
            }
            return Ok(None);
        }
        if !response.status().is_success() {
            let status = response.status();
            let detail = read_text_bounded(response, "hf2q runtime error").await?;
            if required {
                bail!(
                    "hf2q runtime API returned HTTP {status}: {}",
                    compact(&detail)
                );
            }
            return Ok(None);
        }
        let view: RuntimeView = match decode_json_bounded(response, "hf2q runtime API").await {
            Ok(view) => view,
            Err(error) if !required => {
                tracing::debug!(%error, "explicit endpoint did not expose a compatible hf2q runtime payload");
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if view.schema_version != RUNTIME_SCHEMA
            || view.capabilities.model_activation != ACTIVATION_SCHEMA
            || !view.capabilities.non_evicting_load
            || !view
                .capabilities
                .diagnostic_no_evict_header
                .as_ref()
                .is_some_and(|header| header.name == NON_EVICTING_HEADER && header.value == "1")
        {
            if required {
                bail!("the discovered hf2q server has an incompatible diagnostic lifecycle API");
            }
            return Ok(None);
        }
        Ok(Some(Self {
            http: http.clone(),
            endpoint: endpoint.clone(),
            artifact_resolution: view.capabilities.artifact_resolution.as_deref()
                == Some(ARTIFACT_SCHEMA),
        }))
    }

    /// Resolve a mixed Hub repository to one exact hosted GGUF before model
    /// activation. Returning `None` means the model string is not a Hub
    /// repository candidate and should retain ordinary activation semantics.
    pub(crate) async fn select_hub_gguf(
        &self,
        model: &str,
        quant: Option<&str>,
        artifact: Option<&str>,
        auth_token: Option<&str>,
        input: &mut impl BufRead,
        output: &mut impl Write,
    ) -> Result<Option<String>> {
        let looks_like_hub = looks_like_hub_model(model);
        if !looks_like_hub && quant.is_none() && artifact.is_none() {
            return Ok(None);
        }
        if !self.artifact_resolution {
            bail!(
                "this hf2q endpoint does not advertise hosted-GGUF selection; use a local GGUF path or upgrade the server"
            );
        }
        let mut request = self
            .http
            .get(self.endpoint.route("/hf2q/v1/models/catalog"))
            .query(&[("model", model)]);
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.context("catalog hosted GGUFs")?;
        if !response.status().is_success() {
            let status = response.status();
            let detail = read_text_bounded(response, "hosted GGUF catalog error").await?;
            bail!(
                "hosted GGUF catalog returned HTTP {status}: {}",
                compact(&detail)
            );
        }
        let catalog: ArtifactCatalog = decode_json_bounded(response, "hosted GGUF catalog").await?;
        if catalog.schema_version != ARTIFACT_SCHEMA {
            bail!("hf2q returned an incompatible hosted-GGUF catalog");
        }
        if let Some(filename) = artifact {
            let selected = catalog
                .candidates
                .iter()
                .find(|candidate| candidate.filename == filename)
                .with_context(|| format!("hosted GGUF artifact `{filename}` was not found"))?;
            ensure_selectable(selected)?;
            return Ok(selected.candidate_id.clone());
        }
        let mut selectable = catalog
            .candidates
            .iter()
            .filter(|candidate| candidate.selectable && candidate.role == "text_model")
            .collect::<Vec<_>>();
        if let Some(quant) = quant {
            selectable.retain(|candidate| {
                candidate
                    .quant_hint
                    .as_deref()
                    .is_some_and(|actual| actual.eq_ignore_ascii_case(quant))
            });
            match selectable.as_slice() {
                [selected] => return Ok(selected.candidate_id.clone()),
                [] => bail!("no selectable hosted GGUF matches --quant {quant}"),
                _ => bail!(
                    "--quant {quant} is ambiguous across {} artifacts; use --artifact",
                    selectable.len()
                ),
            }
        }
        match selectable.as_slice() {
            [] => bail!(
                "repository {model} has no compatible hosted text GGUF; source conversion is not implicit in diagnostic chat"
            ),
            [selected] => Ok(selected.candidate_id.clone()),
            _ => {
                writeln!(output, "hosted GGUF artifacts for {model}:")?;
                for (index, candidate) in selectable.iter().enumerate() {
                    writeln!(
                        output,
                        "  {}. {}  {}  {}",
                        index + 1,
                        candidate.quant_hint.as_deref().unwrap_or("unknown"),
                        human_bytes(candidate.bytes),
                        candidate.filename
                    )?;
                }
                for candidate in catalog
                    .candidates
                    .iter()
                    .filter(|candidate| !candidate.selectable)
                {
                    writeln!(
                        output,
                        "  - {}  {} [{}]",
                        candidate.quant_hint.as_deref().unwrap_or("unknown"),
                        candidate.filename,
                        candidate
                            .unavailable_reason
                            .as_deref()
                            .unwrap_or("unavailable")
                    )?;
                }
                write!(output, "GGUF> ")?;
                output.flush()?;
                let mut answer = String::new();
                if input.read_line(&mut answer)? == 0 {
                    bail!("input ended before a GGUF was selected; nothing was downloaded");
                }
                let index: usize = answer
                    .trim()
                    .parse()
                    .context("GGUF selection must be a number")?;
                let selected = selectable
                    .get(index.checked_sub(1).context("GGUF selection starts at 1")?)
                    .context("GGUF selection is out of range")?;
                Ok(selected.candidate_id.clone())
            }
        }
    }

    pub(crate) async fn probe_resident(
        &self,
        model: &str,
        auth_token: Option<&str>,
    ) -> Result<Option<String>> {
        let body = ActivationRequest {
            model,
            candidate_id: None,
            action: Some("probe"),
            expected_revision: None,
            victims: Vec::new(),
        };
        let mut request = self
            .http
            .post(self.endpoint.route("/hf2q/v1/models/activate"))
            .json(&body);
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.context("probe resident hf2q model")?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            let status = response.status();
            let detail = read_text_bounded(response, "hf2q resident probe error").await?;
            bail!(
                "hf2q resident probe returned HTTP {status}: {}",
                compact(&detail)
            );
        }
        let success: ActivationSuccess =
            decode_json_bounded(response, "hf2q resident probe").await?;
        if success.schema_version != ACTIVATION_SCHEMA || success.status != "resident" {
            bail!("hf2q returned an incompatible resident probe response");
        }
        Ok(success.request_model)
    }

    pub(crate) async fn ensure_active(
        &self,
        model: &str,
        candidate_id: Option<&str>,
        auth_token: Option<&str>,
        input: &mut impl BufRead,
        output: &mut impl Write,
    ) -> Result<Option<String>> {
        match self
            .activate(
                ActivationRequest {
                    model,
                    candidate_id,
                    action: None,
                    expected_revision: None,
                    victims: Vec::new(),
                },
                auth_token,
            )
            .await?
        {
            ActivationResult::Ready(success) => {
                writeln!(
                    output,
                    "model activation: {} (pool revision {})",
                    success.status, success.pool_revision
                )?;
                Ok(Some(
                    success.request_model.unwrap_or_else(|| model.to_owned()),
                ))
            }
            ActivationResult::Conflict(conflict) => {
                if conflict.schema_version != ACTIVATION_SCHEMA
                    || conflict.status != "conflict"
                    || !conflict.requires_explicit_switch
                {
                    bail!("hf2q returned an incompatible activation conflict");
                }
                writeln!(
                    output,
                    "model {model} would require explicit switching (projected pool {})",
                    human_bytes(conflict.projected_bytes)
                )?;
                for victim in &conflict.victims {
                    writeln!(
                        output,
                        "  unload {} generation {} ({})",
                        victim.pool_key,
                        victim.generation,
                        human_bytes(victim.bytes_resident)
                    )?;
                }
                write!(output, "Switch to {model}? [y/N] ")?;
                output.flush()?;
                let mut answer = String::new();
                if input.read_line(&mut answer)? == 0
                    || !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                {
                    return Ok(None);
                }
                let switched = self
                    .activate(
                        ActivationRequest {
                            model,
                            candidate_id,
                            action: Some("switch"),
                            expected_revision: Some(conflict.pool_revision),
                            victims: conflict.victims,
                        },
                        auth_token,
                    )
                    .await?;
                match switched {
                    ActivationResult::Ready(success) if success.status == "switched" => {
                        writeln!(
                            output,
                            "model activation: switched (pool revision {})",
                            success.pool_revision
                        )?;
                        Ok(Some(
                            success.request_model.unwrap_or_else(|| model.to_owned()),
                        ))
                    }
                    ActivationResult::Ready(success) => {
                        bail!("hf2q returned unexpected switch status {}", success.status)
                    }
                    ActivationResult::Conflict(_) => {
                        bail!("the model pool changed before the switch; select the model again")
                    }
                }
            }
        }
    }

    async fn activate(
        &self,
        body: ActivationRequest<'_>,
        auth_token: Option<&str>,
    ) -> Result<ActivationResult> {
        let mut request = self
            .http
            .post(self.endpoint.route("/hf2q/v1/models/activate"))
            .json(&body);
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.context("activate hf2q model")?;
        let status = response.status();
        if status.is_success() {
            let success: ActivationSuccess =
                decode_json_bounded(response, "hf2q activation response").await?;
            if success.schema_version != ACTIVATION_SCHEMA {
                bail!("hf2q returned an incompatible activation response");
            }
            return Ok(ActivationResult::Ready(success));
        }
        if status == reqwest::StatusCode::CONFLICT {
            let conflict: ActivationConflict =
                decode_json_bounded(response, "hf2q activation conflict").await?;
            return Ok(ActivationResult::Conflict(conflict));
        }
        let detail = read_text_bounded(response, "hf2q activation error").await?;
        bail!(
            "hf2q model activation returned HTTP {status}: {}",
            compact(&detail)
        )
    }

    async fn runtime(&self, auth_token: Option<&str>) -> Result<RuntimeView> {
        let mut request = self.http.get(self.endpoint.route("/hf2q/v1/runtime"));
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.context("query hf2q runtime")?;
        let status = response.status();
        if !status.is_success() {
            let detail = read_text_bounded(response, "hf2q runtime error").await?;
            bail!("hf2q runtime returned HTTP {status}: {}", compact(&detail));
        }
        decode_json_bounded(response, "hf2q runtime").await
    }

    pub(crate) async fn write_runtime_status(
        &self,
        auth_token: Option<&str>,
        output: &mut impl Write,
    ) -> Result<()> {
        let view = match self.runtime(auth_token).await {
            Ok(view) => view,
            Err(error) => {
                writeln!(output, "[pool] unavailable: {error:#}")?;
                return Ok(());
            }
        };
        let residents = view
            .pool
            .resident
            .iter()
            .map(|resident| resident.pool_key.as_str())
            .collect::<Vec<_>>();
        writeln!(
            output,
            "[pool] revision={} models={}/{} resident={}/{} entries={}",
            view.pool.revision,
            view.pool.loaded_count,
            view.pool.capacity_models,
            human_bytes(view.pool.total_resident_bytes),
            human_bytes(view.pool.memory_budget_bytes),
            if residents.is_empty() {
                "none".to_owned()
            } else {
                residents.join(",")
            }
        )?;
        Ok(())
    }
}

pub(crate) fn looks_like_hub_model(model: &str) -> bool {
    model.starts_with("https://huggingface.co/")
        || (model.contains('/') && !std::path::Path::new(model).exists())
}

fn ensure_selectable(candidate: &ArtifactCandidate) -> Result<()> {
    if candidate.selectable && candidate.role == "text_model" {
        if candidate.candidate_id.is_some() {
            Ok(())
        } else {
            bail!("hf2q returned a selectable artifact without activation authority")
        }
    } else {
        bail!(
            "hosted GGUF artifact `{}` is unavailable: {}",
            candidate.filename,
            candidate
                .unavailable_reason
                .as_deref()
                .unwrap_or("unknown reason")
        )
    }
}

fn human_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{bytes} B")
    }
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
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use serde_json::Value;
    use tokio::sync::oneshot;

    use super::*;

    #[derive(Clone, Default)]
    struct Recorded(Arc<Mutex<Vec<Value>>>);

    async fn runtime() -> Json<Value> {
        Json(serde_json::json!({
            "schema_version": RUNTIME_SCHEMA,
            "backend": "mlx-native",
            "capabilities": {
                "model_activation": ACTIVATION_SCHEMA,
                "artifact_resolution": ARTIFACT_SCHEMA,
                "non_evicting_load": true,
                "diagnostic_no_evict_header": {
                    "name": NON_EVICTING_HEADER,
                    "value": "1"
                },
                "explicit_revision_bound_switch": true,
                "request_generation_leases": true
            },
            "pool": {
                "revision": 9,
                "loaded_count": 1,
                "capacity_models": 1,
                "total_resident_bytes": 400,
                "memory_budget_bytes": 800,
                "resident": [{
                    "pool_key": "old/model@Q4_K_M",
                    "quant": "Q4_K_M",
                    "bytes_resident": 400,
                    "generation": 7
                }]
            }
        }))
    }

    async fn activate(
        State(recorded): State<Recorded>,
        Json(request): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        let mut requests = recorded.0.lock().unwrap();
        requests.push(request);
        if requests.len() == 1 {
            (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "schema_version": ACTIVATION_SCHEMA,
                    "status": "conflict",
                    "pool_revision": 9,
                    "victims": [{
                        "pool_key": "old/model@Q4_K_M",
                        "quant": "Q4_K_M",
                        "bytes_resident": 400,
                        "generation": 7
                    }],
                    "projected_bytes": 500,
                    "requires_explicit_switch": true
                })),
            )
        } else {
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "schema_version": ACTIVATION_SCHEMA,
                    "status": "switched",
                    "pool_revision": 11
                })),
            )
        }
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

    async fn catalog() -> Json<Value> {
        Json(serde_json::json!({
            "schema_version":ARTIFACT_SCHEMA,
            "repository":"owner/mixed",
            "revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "candidates":[
                {"candidate_id":null,"filename":"gguf/model-q5_k_m.gguf","bytes":19535701568_u64,"quant_hint":"Q5_K_M","role":"text_model","selectable":false,"unavailable_reason":"Q5 policy identity deferred"},
                {"candidate_id":"q6-candidate","filename":"gguf/model-q6_k.gguf","bytes":22431000128_u64,"quant_hint":"Q6_K","role":"text_model","selectable":true,"unavailable_reason":null},
                {"candidate_id":"q8-candidate","filename":"gguf/model-q8_0.gguf","bytes":28000000000_u64,"quant_hint":"Q8_0","role":"text_model","selectable":true,"unavailable_reason":null},
                {"candidate_id":null,"filename":"gguf/model-bf16.gguf","bytes":54657734208_u64,"quant_hint":"BF16","role":"text_model","selectable":false,"unavailable_reason":"BF16 unsupported"},
                {"candidate_id":null,"filename":"gguf/mmproj-f16.gguf","bytes":927607264_u64,"quant_hint":null,"role":"companion","selectable":false,"unavailable_reason":"not a text model"}
            ]
        }))
    }

    #[tokio::test]
    async fn hosted_gguf_quant_and_picker_select_exact_artifact_without_transfer() {
        let router = Router::new()
            .route("/hf2q/v1/runtime", get(runtime))
            .route("/hf2q/v1/models/catalog", get(catalog));
        let (endpoint, stop) = serve(router).await;
        let control = Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();

        let mut unused = std::io::Cursor::new(Vec::<u8>::new());
        let q6 = control
            .select_hub_gguf(
                "owner/mixed",
                Some("q6_k"),
                None,
                None,
                &mut unused,
                &mut Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(q6.as_deref(), Some("q6-candidate"));

        let mut picker = std::io::Cursor::new(b"1\n");
        let mut output = Vec::new();
        let picked = control
            .select_hub_gguf("owner/mixed", None, None, None, &mut picker, &mut output)
            .await
            .unwrap();
        assert_eq!(picked.as_deref(), Some("q6-candidate"));
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("Q5_K_M"));
        assert!(output.contains("BF16 unsupported"));
        assert!(output.contains("not a text model"));
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn activation_returns_authoritative_request_model_to_chat() {
        async fn ready(Json(request): Json<Value>) -> Json<Value> {
            assert_eq!(request["candidate_id"], "q6-candidate");
            Json(serde_json::json!({
                "schema_version": ACTIVATION_SCHEMA,
                "status":"loaded",
                "pool_revision":2,
                "request_model":"hf://owner/mixed@aaaaaaaa/gguf/model-q6_k.gguf#bbbb"
            }))
        }
        let router = Router::new()
            .route("/hf2q/v1/runtime", get(runtime))
            .route("/hf2q/v1/models/activate", post(ready));
        let (endpoint, stop) = serve(router).await;
        let control = Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();
        let selected = control
            .ensure_active(
                "owner/mixed",
                Some("q6-candidate"),
                None,
                &mut std::io::Cursor::new(Vec::<u8>::new()),
                &mut Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            selected.as_deref(),
            Some("hf://owner/mixed@aaaaaaaa/gguf/model-q6_k.gguf#bbbb")
        );
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn explicit_switch_resends_exact_revision_and_victim_receipt() {
        let recorded = Recorded::default();
        let router = Router::new()
            .route("/hf2q/v1/runtime", get(runtime))
            .route("/hf2q/v1/models/activate", post(activate))
            .with_state(recorded.clone());
        let (endpoint, stop) = serve(router).await;
        let http = reqwest::Client::new();
        let control = Hf2qControl::detect(&http, &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();
        let mut input = std::io::Cursor::new(b"yes\n");
        let mut output = Vec::new();
        assert!(control
            .ensure_active(
                "new/model",
                Some("new-candidate"),
                None,
                &mut input,
                &mut output,
            )
            .await
            .unwrap()
            .is_some());
        let requests = recorded.0.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0],
            serde_json::json!({"model":"new/model","candidate_id":"new-candidate"})
        );
        assert_eq!(
            requests[1],
            serde_json::json!({
                "model":"new/model",
                "candidate_id":"new-candidate",
                "action":"switch",
                "expected_revision":9,
                "victims":[{
                    "pool_key":"old/model@Q4_K_M",
                    "quant":"Q4_K_M",
                    "bytes_resident":400,
                    "generation":7
                }]
            })
        );
        drop(requests);
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("Switch to new/model?"));
        assert!(output.contains("model activation: switched"));
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn declined_switch_never_posts_the_switch_action() {
        let recorded = Recorded::default();
        let router = Router::new()
            .route("/hf2q/v1/runtime", get(runtime))
            .route("/hf2q/v1/models/activate", post(activate))
            .with_state(recorded.clone());
        let (endpoint, stop) = serve(router).await;
        let control = Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();
        let mut input = std::io::Cursor::new(b"no\n");
        assert!(control
            .ensure_active(
                "new/model",
                Some("new-candidate"),
                None,
                &mut input,
                &mut Vec::new(),
            )
            .await
            .unwrap()
            .is_none());
        assert_eq!(recorded.0.lock().unwrap().len(), 1);
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn generic_endpoint_without_capability_remains_usable() {
        let (endpoint, stop) = serve(Router::new()).await;
        assert!(
            Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, false)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
                .await
                .is_err()
        );
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn generic_endpoint_with_an_invalid_optional_capability_remains_usable() {
        let router = Router::new().route(
            "/hf2q/v1/runtime",
            get(|| async { (StatusCode::OK, "not-json") }),
        );
        let (endpoint, stop) = serve(router).await;
        assert!(
            Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, false)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
                .await
                .is_err()
        );
        let _ = stop.send(());
    }
}
