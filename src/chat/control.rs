//! Client for hf2q's optional ADR-047 diagnostic control plane.

use std::io::{BufRead, Write};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::client::{decode_json_bounded, read_text_bounded};
use super::endpoint::Endpoint;

const RUNTIME_SCHEMA: &str = "hf2q.runtime.v1";
const ACTIVATION_SCHEMA: &str = "hf2q.model-activation.v2";
const ARTIFACT_SCHEMA: &str = "hf2q.artifact-resolution.v2";
const LOCAL_ARTIFACT_SCHEMA: &str = "hf2q.local-artifact-resolution.v1";
pub(crate) const NON_EVICTING_HEADER: &str = "x-hf2q-diagnostic-no-evict";

pub(crate) struct Hf2qControl {
    http: reqwest::Client,
    endpoint: Endpoint,
    artifact_resolution: bool,
    local_artifact_resolution: bool,
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
    #[serde(default)]
    local_artifact_resolution: Option<String>,
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

#[derive(Debug, Deserialize)]
struct LocalArtifactCatalog {
    schema_version: String,
    candidates: Vec<LocalArtifactCandidate>,
}

#[derive(Clone, Debug, Deserialize)]
struct LocalArtifactCandidate {
    candidate_id: Option<String>,
    repository: String,
    filename: String,
    bytes: u64,
    quant_hint: String,
    origin: String,
    role: String,
    selectable: bool,
    unavailable_reason: Option<String>,
}

#[derive(Clone, Debug)]
struct ArtifactChoice {
    candidate_id: Option<String>,
    repository: Option<String>,
    filename: String,
    bytes: u64,
    quant_hint: Option<String>,
    origin: &'static str,
    role: String,
    selectable: bool,
    unavailable_reason: Option<String>,
}

enum PickerSelection {
    Candidate(String),
    BrowseHosted,
}

pub(crate) struct InitialLocalSelection {
    pub model: String,
    pub candidate_id: String,
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
            local_artifact_resolution: view.capabilities.local_artifact_resolution.as_deref()
                == Some(LOCAL_ARTIFACT_SCHEMA),
        }))
    }

    /// Resolve a repository to a receipt-backed local GGUF first, then to a
    /// hosted GGUF only when needed. No conversion is implicit.
    pub(crate) async fn select_gguf(
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
        if !self.artifact_resolution && !self.local_artifact_resolution {
            bail!(
                "this hf2q endpoint does not advertise GGUF selection; use an explicit local GGUF path or upgrade the server"
            );
        }
        let local = if self.local_artifact_resolution {
            self.fetch_local_artifacts(Some(model), auth_token).await?
        } else {
            Vec::new()
        };
        if let Some(filename) = artifact {
            match exact_local_match(&local, filename)? {
                Some(candidate_id) => return Ok(Some(candidate_id)),
                None => {}
            }
        }
        if let Some(quant) = quant {
            match quant_local_match(&local, quant)? {
                Some(candidate_id) => return Ok(Some(candidate_id)),
                None => {}
            }
        }

        if quant.is_none() && artifact.is_none() && local.iter().any(is_selectable_text_choice) {
            let browse = self.artifact_resolution;
            match pick_artifact(model, &local, browse, input, output)? {
                PickerSelection::Candidate(candidate_id) => return Ok(Some(candidate_id)),
                PickerSelection::BrowseHosted => {}
            }
        }

        if !self.artifact_resolution {
            bail!(
                "no receipt-backed local GGUF matched, and this hf2q endpoint does not advertise hosted-GGUF selection"
            );
        }
        let hosted = match self.fetch_hosted_artifacts(model, auth_token).await {
            Ok(hosted) => hosted,
            Err(error) if local.iter().any(is_selectable_text_choice) => {
                writeln!(
                    output,
                    "hosted artifacts unavailable: {}; local artifacts remain selectable",
                    terminal_safe(&format!("{error:#}"))
                )?;
                return match pick_artifact(model, &local, false, input, output)? {
                    PickerSelection::Candidate(candidate_id) => Ok(Some(candidate_id)),
                    PickerSelection::BrowseHosted => unreachable!("browse row was disabled"),
                };
            }
            Err(error) => return Err(error),
        };
        if let Some(filename) = artifact {
            return exact_hosted_match(&hosted, filename).map(Some);
        }
        if let Some(quant) = quant {
            return quant_hosted_match(&hosted, quant).map(Some);
        }
        let mut combined = local;
        combined.extend(hosted);
        match pick_artifact(model, &combined, false, input, output)? {
            PickerSelection::Candidate(candidate_id) => Ok(Some(candidate_id)),
            PickerSelection::BrowseHosted => unreachable!("browse row was disabled"),
        }
    }

    async fn fetch_local_artifacts(
        &self,
        model: Option<&str>,
        auth_token: Option<&str>,
    ) -> Result<Vec<ArtifactChoice>> {
        let mut request = self
            .http
            .get(self.endpoint.route("/hf2q/v1/models/local-artifacts"));
        if let Some(model) = model {
            request = request.query(&[("model", model)]);
        }
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.context("catalog local hf2q GGUFs")?;
        if !response.status().is_success() {
            let status = response.status();
            let detail = read_text_bounded(response, "local GGUF catalog error").await?;
            bail!(
                "local GGUF catalog returned HTTP {status}: {}",
                compact(&detail)
            );
        }
        let catalog: LocalArtifactCatalog =
            decode_json_bounded(response, "local GGUF catalog").await?;
        if catalog.schema_version != LOCAL_ARTIFACT_SCHEMA {
            bail!("hf2q returned an incompatible local-GGUF catalog");
        }
        Ok(catalog
            .candidates
            .into_iter()
            .filter(|candidate| model.is_none_or(|expected| candidate.repository == expected))
            .map(|candidate| ArtifactChoice {
                candidate_id: candidate.candidate_id,
                repository: Some(candidate.repository),
                filename: candidate.filename,
                bytes: candidate.bytes,
                quant_hint: Some(candidate.quant_hint),
                origin: if candidate.origin == "local_cache" {
                    "local cache"
                } else {
                    "local hf2q"
                },
                role: candidate.role,
                selectable: candidate.selectable,
                unavailable_reason: candidate.unavailable_reason,
            })
            .collect())
    }

    async fn fetch_hosted_artifacts(
        &self,
        model: &str,
        auth_token: Option<&str>,
    ) -> Result<Vec<ArtifactChoice>> {
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
        Ok(catalog
            .candidates
            .into_iter()
            .map(|candidate| ArtifactChoice {
                candidate_id: candidate.candidate_id,
                repository: Some(model.to_owned()),
                filename: candidate.filename,
                bytes: candidate.bytes,
                quant_hint: candidate.quant_hint,
                origin: "hosted",
                role: candidate.role,
                selectable: candidate.selectable,
                unavailable_reason: candidate.unavailable_reason,
            })
            .collect())
    }

    pub(crate) async fn select_initial_local(
        &self,
        auth_token: Option<&str>,
        input: &mut impl BufRead,
        output: &mut impl Write,
    ) -> Result<Option<InitialLocalSelection>> {
        if !self.local_artifact_resolution {
            return Ok(None);
        }
        let candidates = self.fetch_local_artifacts(None, auth_token).await?;
        let selectable = candidates
            .iter()
            .filter(|candidate| is_selectable_text_choice(candidate))
            .collect::<Vec<_>>();
        if selectable.is_empty() {
            return Ok(None);
        }
        let selected = if selectable.len() == 1 {
            selectable[0]
        } else {
            writeln!(output, "local hf2q GGUF artifacts:")?;
            for (index, candidate) in selectable.iter().enumerate() {
                writeln!(
                    output,
                    "  {}. {}  {}  {}  {}",
                    index + 1,
                    terminal_safe(candidate.repository.as_deref().unwrap_or("unknown")),
                    terminal_safe(candidate.quant_hint.as_deref().unwrap_or("unknown")),
                    human_bytes(candidate.bytes),
                    terminal_safe(&candidate.filename)
                )?;
            }
            write!(output, "artifact> ")?;
            output.flush()?;
            let mut answer = String::new();
            if input.read_line(&mut answer)? == 0 {
                bail!("input ended before a local artifact was selected");
            }
            let index: usize = answer
                .trim()
                .parse()
                .context("artifact selection must be a number")?;
            selectable
                .get(
                    index
                        .checked_sub(1)
                        .context("artifact selection starts at 1")?,
                )
                .copied()
                .context("artifact selection is out of range")?
        };
        Ok(Some(InitialLocalSelection {
            model: selected
                .repository
                .clone()
                .context("local artifact omitted its repository")?,
            candidate_id: selectable_id(selected)?,
        }))
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
                    terminal_safe(&success.status),
                    success.pool_revision
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
                        terminal_safe(&victim.pool_key),
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
                    bail!(
                        "hf2q returned unexpected switch status {}",
                        terminal_safe(&success.status)
                    )
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
            .map(|resident| terminal_safe(&resident.pool_key))
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
        || (!model.contains("://")
            && !model.starts_with('.')
            && !std::path::Path::new(model).is_absolute()
            && crate::serve::auto_pipeline::looks_like_hf_repo_id(model))
}

fn is_selectable_text_choice(candidate: &ArtifactChoice) -> bool {
    candidate.selectable && candidate.role == "text_model" && candidate.candidate_id.is_some()
}

fn selectable_id(candidate: &ArtifactChoice) -> Result<String> {
    if !candidate.selectable || candidate.role != "text_model" {
        bail!(
            "GGUF artifact `{}` is unavailable: {}",
            terminal_safe(&candidate.filename),
            terminal_safe(
                candidate
                    .unavailable_reason
                    .as_deref()
                    .unwrap_or("unknown reason")
            )
        );
    }
    candidate
        .candidate_id
        .clone()
        .context("hf2q returned a selectable artifact without activation authority")
}

fn exact_local_match(candidates: &[ArtifactChoice], filename: &str) -> Result<Option<String>> {
    let matches = candidates
        .iter()
        .filter(|candidate| {
            candidate.origin != "hosted"
                && candidate.filename == filename
                && is_selectable_text_choice(candidate)
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [selected] => selectable_id(selected).map(Some),
        _ => bail!(
            "--artifact {} is ambiguous across {} local artifacts; use the numbered picker or an explicit GGUF path",
            terminal_safe(filename),
            matches.len()
        ),
    }
}

fn quant_local_match(candidates: &[ArtifactChoice], quant: &str) -> Result<Option<String>> {
    let matches = candidates
        .iter()
        .filter(|candidate| {
            candidate.origin != "hosted"
                && candidate
                    .quant_hint
                    .as_deref()
                    .is_some_and(|actual| actual.eq_ignore_ascii_case(quant))
                && is_selectable_text_choice(candidate)
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [selected] => selectable_id(selected).map(Some),
        _ => bail!(
            "--quant {} is ambiguous across {} local artifacts; use --artifact or the numbered picker",
            terminal_safe(quant),
            matches.len()
        ),
    }
}

fn exact_hosted_match(candidates: &[ArtifactChoice], filename: &str) -> Result<String> {
    let matches = candidates
        .iter()
        .filter(|candidate| candidate.filename == filename)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => bail!(
            "hosted GGUF artifact `{}` was not found",
            terminal_safe(filename)
        ),
        [selected] => selectable_id(selected),
        _ => bail!(
            "hosted GGUF artifact `{}` is ambiguous across {} entries",
            terminal_safe(filename),
            matches.len()
        ),
    }
}

fn quant_hosted_match(candidates: &[ArtifactChoice], quant: &str) -> Result<String> {
    let matches = candidates
        .iter()
        .filter(|candidate| {
            candidate
                .quant_hint
                .as_deref()
                .is_some_and(|actual| actual.eq_ignore_ascii_case(quant))
                && is_selectable_text_choice(candidate)
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => bail!(
            "no selectable hosted GGUF matches --quant {}",
            terminal_safe(quant)
        ),
        [selected] => selectable_id(selected),
        _ => bail!(
            "--quant {} is ambiguous across {} hosted artifacts; use --artifact",
            terminal_safe(quant),
            matches.len()
        ),
    }
}

fn pick_artifact(
    model: &str,
    candidates: &[ArtifactChoice],
    browse_hosted: bool,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<PickerSelection> {
    let selectable = candidates
        .iter()
        .filter(|candidate| is_selectable_text_choice(candidate))
        .collect::<Vec<_>>();
    if selectable.is_empty() && !browse_hosted {
        bail!(
            "repository {} has no compatible GGUF; source conversion is not implicit in diagnostic chat",
            terminal_safe(model)
        );
    }
    if selectable.len() == 1 && !browse_hosted {
        return Ok(PickerSelection::Candidate(selectable_id(selectable[0])?));
    }
    writeln!(output, "GGUF artifacts for {}:", terminal_safe(model))?;
    for (index, candidate) in selectable.iter().enumerate() {
        writeln!(
            output,
            "  {}. [{}] {}  {}  {}",
            index + 1,
            candidate.origin,
            terminal_safe(candidate.quant_hint.as_deref().unwrap_or("unknown")),
            human_bytes(candidate.bytes),
            terminal_safe(&candidate.filename)
        )?;
    }
    let browse_index = browse_hosted.then_some(selectable.len() + 1);
    if let Some(index) = browse_index {
        writeln!(output, "  {index}. [hosted] Browse hosted artifacts")?;
    }
    for candidate in candidates.iter().filter(|candidate| !candidate.selectable) {
        writeln!(
            output,
            "  - [{}] {}  {} [{}]",
            candidate.origin,
            terminal_safe(candidate.quant_hint.as_deref().unwrap_or("unknown")),
            terminal_safe(&candidate.filename),
            terminal_safe(
                candidate
                    .unavailable_reason
                    .as_deref()
                    .unwrap_or("unavailable")
            )
        )?;
    }
    write!(output, "artifact> ")?;
    output.flush()?;
    let mut answer = String::new();
    if input.read_line(&mut answer)? == 0 {
        bail!("input ended before an artifact was selected; nothing was downloaded");
    }
    let index: usize = answer
        .trim()
        .parse()
        .context("artifact selection must be a number")?;
    if browse_index == Some(index) {
        return Ok(PickerSelection::BrowseHosted);
    }
    let selected = selectable
        .get(
            index
                .checked_sub(1)
                .context("artifact selection starts at 1")?,
        )
        .context("artifact selection is out of range")?;
    Ok(PickerSelection::Candidate(selectable_id(selected)?))
}

pub(crate) fn terminal_safe(value: &str) -> String {
    const MAX_CHARS: usize = 256;
    let mut rendered = String::new();
    for character in value.chars().take(MAX_CHARS) {
        if character.is_control() || character == '\u{1b}' {
            rendered.extend(character.escape_default());
        } else {
            rendered.push(character);
        }
    }
    if value.chars().count() > MAX_CHARS {
        rendered.push('…');
    }
    rendered
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
    let compacted = if one_line.chars().count() <= MAX {
        one_line
    } else {
        format!("{}…", one_line.chars().take(MAX).collect::<String>())
    };
    terminal_safe(&compacted)
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

    async fn runtime_with_local() -> Json<Value> {
        let mut value = runtime().await.0;
        value["capabilities"]["local_artifact_resolution"] =
            Value::String(LOCAL_ARTIFACT_SCHEMA.into());
        Json(value)
    }

    async fn local_catalog() -> Json<Value> {
        Json(serde_json::json!({
            "schema_version": LOCAL_ARTIFACT_SCHEMA,
            "candidates": [
                {"candidate_id":"local-q4","repository":"owner/mixed","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","filename":"model-q4_k_m.gguf","bytes":16810714848_u64,"quant_hint":"Q4_K_M","origin":"local_receipt","role":"text_model","selectable":true,"unavailable_reason":null},
                {"candidate_id":null,"repository":"owner/mixed","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","filename":"model-bf16.gguf","bytes":54000000000_u64,"quant_hint":"BF16","origin":"local_receipt","role":"text_model","selectable":false,"unavailable_reason":"GGUF quant is not supported by the current mlx-native diagnostic loader"}
            ]
        }))
    }

    #[tokio::test]
    async fn local_quant_and_picker_complete_without_hub_catalog_work() {
        #[derive(Clone, Default)]
        struct Calls(Arc<std::sync::atomic::AtomicUsize>);
        async fn hosted(State(calls): State<Calls>) -> Json<Value> {
            calls.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            catalog().await
        }
        let calls = Calls::default();
        let router = Router::new()
            .route("/hf2q/v1/runtime", get(runtime_with_local))
            .route("/hf2q/v1/models/local-artifacts", get(local_catalog))
            .route("/hf2q/v1/models/catalog", get(hosted))
            .with_state(calls.clone());
        let (endpoint, stop) = serve(router).await;
        let control = Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();

        let selected = control
            .select_gguf(
                "owner/mixed",
                Some("q4_k_m"),
                None,
                None,
                &mut std::io::Cursor::new(Vec::<u8>::new()),
                &mut Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(selected.as_deref(), Some("local-q4"));
        assert_eq!(calls.0.load(std::sync::atomic::Ordering::SeqCst), 0);

        let mut output = Vec::new();
        let selected = control
            .select_gguf(
                "owner/mixed",
                None,
                None,
                None,
                &mut std::io::Cursor::new(b"1\n"),
                &mut output,
            )
            .await
            .unwrap();
        assert_eq!(selected.as_deref(), Some("local-q4"));
        assert_eq!(calls.0.load(std::sync::atomic::Ordering::SeqCst), 0);
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("[local hf2q] Q4_K_M"));
        assert!(output.contains("Browse hosted artifacts"));
        assert!(output.contains("BF16"));

        let initial = control
            .select_initial_local(
                None,
                &mut std::io::Cursor::new(Vec::<u8>::new()),
                &mut Vec::new(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(initial.model, "owner/mixed");
        assert_eq!(initial.candidate_id, "local-q4");
        let _ = stop.send(());
    }

    #[test]
    fn terminal_control_sequences_are_rendered_as_text() {
        let rendered = terminal_safe("model\u{1b}]52;c;secret\u{7}\nnext");
        assert!(!rendered.contains('\u{1b}'));
        assert!(!rendered.contains('\u{7}'));
        assert!(!rendered.contains('\n'));
        assert!(rendered.contains("\\u{1b}"));
        assert!(rendered.contains("\\n"));
    }

    #[test]
    fn ambiguous_local_selector_fails_before_any_hosted_fallback() {
        let candidate = |id: &str, filename: &str| ArtifactChoice {
            candidate_id: Some(id.into()),
            repository: Some("owner/model".into()),
            filename: filename.into(),
            bytes: 42,
            quant_hint: Some("Q4_K_M".into()),
            origin: "local hf2q",
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        };
        let candidates = vec![candidate("a", "one.gguf"), candidate("b", "two.gguf")];
        let error = quant_local_match(&candidates, "q4_k_m").unwrap_err();
        assert!(error.to_string().contains("ambiguous"));
    }

    #[test]
    fn missing_local_paths_are_not_reclassified_as_hub_repositories() {
        assert!(looks_like_hub_model("owner/model"));
        assert!(looks_like_hub_model("https://huggingface.co/owner/model"));
        assert!(!looks_like_hub_model("./models/missing.gguf"));
        assert!(!looks_like_hub_model("/models/missing.gguf"));
        assert!(!looks_like_hub_model(
            "local://owner/model@revision/candidate"
        ));
    }

    #[tokio::test]
    async fn hosted_failure_after_browse_preserves_local_selection() {
        async fn unavailable() -> (StatusCode, &'static str) {
            (StatusCode::SERVICE_UNAVAILABLE, "offline")
        }
        let router = Router::new()
            .route("/hf2q/v1/runtime", get(runtime_with_local))
            .route("/hf2q/v1/models/local-artifacts", get(local_catalog))
            .route("/hf2q/v1/models/catalog", get(unavailable));
        let (endpoint, stop) = serve(router).await;
        let control = Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();
        let mut output = Vec::new();
        let selected = control
            .select_gguf(
                "owner/mixed",
                None,
                None,
                None,
                &mut std::io::Cursor::new(b"2\n"),
                &mut output,
            )
            .await
            .unwrap();
        assert_eq!(selected.as_deref(), Some("local-q4"));
        assert!(String::from_utf8(output)
            .unwrap()
            .contains("local artifacts remain selectable"));
        let _ = stop.send(());
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
            .select_gguf(
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
            .select_gguf("owner/mixed", None, None, None, &mut picker, &mut output)
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
    async fn activation_preserves_bounded_safe_loader_diagnostic() {
        async fn rejected() -> (StatusCode, Json<Value>) {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "model load failed: selected GGUF is missing required tensor 'output.weight'",
                        "type": "generation_error"
                    }
                })),
            )
        }
        let router = Router::new()
            .route("/hf2q/v1/runtime", get(runtime))
            .route("/hf2q/v1/models/activate", post(rejected));
        let (endpoint, stop) = serve(router).await;
        let control = Hf2qControl::detect(&reqwest::Client::new(), &endpoint, None, true)
            .await
            .unwrap()
            .unwrap();
        let error = control
            .ensure_active(
                "owner/mixed",
                Some("q8-candidate"),
                None,
                &mut std::io::Cursor::new(Vec::<u8>::new()),
                &mut Vec::new(),
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("output.weight"));
        assert!(error.contains("HTTP 500"));
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
