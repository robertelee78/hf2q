//! Client for hf2q's optional ADR-047 diagnostic control plane.

use std::io::{BufRead, Write};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::endpoint::Endpoint;

const RUNTIME_SCHEMA: &str = "hf2q.runtime.v1";
const ACTIVATION_SCHEMA: &str = "hf2q.model-activation.v1";
pub(crate) const NON_EVICTING_HEADER: &str = "x-hf2q-diagnostic-no-evict";

pub(crate) struct Hf2qControl {
    http: reqwest::Client,
    endpoint: Endpoint,
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
    action: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_revision: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    victims: Vec<ActivationVictim>,
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
        let response = request.send().await.context("probe hf2q runtime API")?;
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
            let detail = response.text().await.unwrap_or_default();
            if required {
                bail!(
                    "hf2q runtime API returned HTTP {status}: {}",
                    compact(&detail)
                );
            }
            return Ok(None);
        }
        let view: RuntimeView = response.json().await.context("decode hf2q runtime API")?;
        if view.schema_version != RUNTIME_SCHEMA
            || view.capabilities.model_activation != ACTIVATION_SCHEMA
            || !view.capabilities.non_evicting_load
        {
            if required {
                bail!("the discovered hf2q server has an incompatible diagnostic lifecycle API");
            }
            return Ok(None);
        }
        Ok(Some(Self {
            http: http.clone(),
            endpoint: endpoint.clone(),
        }))
    }

    pub(crate) async fn ensure_active(
        &self,
        model: &str,
        auth_token: Option<&str>,
        input: &mut impl BufRead,
        output: &mut impl Write,
    ) -> Result<bool> {
        match self
            .activate(
                ActivationRequest {
                    model,
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
                Ok(true)
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
                    return Ok(false);
                }
                let switched = self
                    .activate(
                        ActivationRequest {
                            model,
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
                        Ok(true)
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
            let success: ActivationSuccess = response
                .json()
                .await
                .context("decode hf2q activation response")?;
            if success.schema_version != ACTIVATION_SCHEMA {
                bail!("hf2q returned an incompatible activation response");
            }
            return Ok(ActivationResult::Ready(success));
        }
        if status == reqwest::StatusCode::CONFLICT {
            let conflict: ActivationConflict = response
                .json()
                .await
                .context("decode hf2q activation conflict")?;
            return Ok(ActivationResult::Conflict(conflict));
        }
        let detail = response.text().await.unwrap_or_default();
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
            let detail = response.text().await.unwrap_or_default();
            bail!("hf2q runtime returned HTTP {status}: {}", compact(&detail));
        }
        response.json().await.context("decode hf2q runtime")
    }

    pub(crate) async fn write_runtime_status(
        &self,
        auth_token: Option<&str>,
        output: &mut impl Write,
    ) -> Result<()> {
        let view = self.runtime(auth_token).await?;
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
                "non_evicting_load": true,
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
            .ensure_active("new/model", None, &mut input, &mut output)
            .await
            .unwrap());
        let requests = recorded.0.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0], serde_json::json!({"model":"new/model"}));
        assert_eq!(
            requests[1],
            serde_json::json!({
                "model":"new/model",
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
        assert!(!control
            .ensure_active("new/model", None, &mut input, &mut Vec::new())
            .await
            .unwrap());
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
}
