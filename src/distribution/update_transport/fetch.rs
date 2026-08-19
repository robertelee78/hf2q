use std::io::Read;

use sha2::{Digest, Sha256};

use super::http::{HttpExecutor, HttpResponse, ReqwestExecutor};
use super::origin::{pages_pointer_url, release_asset_url, release_redirect, TargetFetchSpec};
use super::{UpdateTransportError, VerifiedReleaseBundle};
use crate::distribution::install_state::EphemeralArtifactStage;
use crate::distribution::schema::{ReleaseManifestV1, UpdateChannel};
use crate::distribution::update_auth::{
    ArtifactFetchAuthorization, ArtifactPointerBinding, BoundArtifactFetchAuthorization,
};

pub(in crate::distribution) struct StableUpdateTransport {
    executor: ReqwestExecutor,
}

impl StableUpdateTransport {
    pub(in crate::distribution) fn new() -> Result<Self, UpdateTransportError> {
        Ok(Self {
            executor: ReqwestExecutor::new()?,
        })
    }

    pub(in crate::distribution) fn fetch_and_bind_pointer<'a>(
        &self,
        authorization: ArtifactFetchAuthorization<'a>,
    ) -> Result<ArtifactPointerBinding<'a>, UpdateTransportError> {
        fetch_and_bind_pointer(&self.executor, authorization)
    }

    pub(in crate::distribution) fn fetch_release_bundle<'a>(
        &self,
        authorization: BoundArtifactFetchAuthorization<'a>,
    ) -> Result<VerifiedReleaseBundle<'a>, UpdateTransportError> {
        fetch_release_bundle(&self.executor, authorization)
    }
}

fn fetch_and_bind_pointer<'a>(
    executor: &impl HttpExecutor,
    authorization: ArtifactFetchAuthorization<'a>,
) -> Result<ArtifactPointerBinding<'a>, UpdateTransportError> {
    let spec = TargetFetchSpec::from_descriptor(authorization.pointer())?;
    let request = pages_pointer_url(&spec)?;
    let bytes = fetch_direct(executor, &request, &spec, false)?;
    Ok(authorization.bind_pointer(&bytes)?)
}

fn fetch_release_bundle<'a>(
    executor: &impl HttpExecutor,
    mut authorization: BoundArtifactFetchAuthorization<'a>,
) -> Result<VerifiedReleaseBundle<'a>, UpdateTransportError> {
    let manifest_spec = TargetFetchSpec::from_descriptor(authorization.manifest())?;
    let manifest_request = release_asset_url(authorization.version(), &manifest_spec)?;
    let manifest_bytes = fetch_release(executor, &manifest_request, &manifest_spec, false)?;
    let manifest = ReleaseManifestV1::parse_and_validate(&manifest_bytes)?;
    if manifest.version() != authorization.version()
        || manifest.target() != authorization.target()
        || manifest.channel() != UpdateChannel::Stable
    {
        return Err(UpdateTransportError::ManifestIdentity);
    }

    let stage = authorization.create_archive_stage()?;
    let archive_spec = TargetFetchSpec::from_descriptor(authorization.archive())?;
    let archive_request = release_asset_url(authorization.version(), &archive_spec)?;
    let response = open_release_response(executor, &archive_request, true)?;
    validate_response_headers(&response, archive_spec.descriptor().length())?;
    let archive = stream_archive(response, &archive_spec, stage)?;
    let final_authorization = authorization.finalize()?;
    Ok(VerifiedReleaseBundle {
        authorization: final_authorization,
        manifest_bytes: manifest_bytes.into_boxed_slice(),
        manifest,
        archive,
    })
}

pub(super) fn fetch_direct(
    executor: &impl HttpExecutor,
    request: &super::origin::OriginLockedUrl,
    spec: &TargetFetchSpec<'_>,
    large: bool,
) -> Result<Vec<u8>, UpdateTransportError> {
    let response = executor.execute(request, large)?;
    if response.status != 200 || response.location.is_some() {
        return Err(UpdateTransportError::Status);
    }
    validate_response_headers(&response, spec.descriptor().length())?;
    read_bounded_exact(response, spec)
}

pub(super) fn fetch_release(
    executor: &impl HttpExecutor,
    request: &super::origin::OriginLockedUrl,
    spec: &TargetFetchSpec<'_>,
    large: bool,
) -> Result<Vec<u8>, UpdateTransportError> {
    let response = open_release_response(executor, request, large)?;
    validate_response_headers(&response, spec.descriptor().length())?;
    read_bounded_exact(response, spec)
}

fn open_release_response(
    executor: &impl HttpExecutor,
    request: &super::origin::OriginLockedUrl,
    large: bool,
) -> Result<HttpResponse, UpdateTransportError> {
    let response = executor.execute(request, large)?;
    match response.status {
        200 if response.location.is_none() => Ok(response),
        302 => {
            let location = response.location.ok_or(UpdateTransportError::Headers)?;
            let redirected = release_redirect(&location)?;
            let final_response = executor.execute(&redirected, large)?;
            if final_response.status != 200 || final_response.location.is_some() {
                return Err(UpdateTransportError::Status);
            }
            Ok(final_response)
        }
        _ => Err(UpdateTransportError::Status),
    }
}

fn validate_response_headers(
    response: &HttpResponse,
    expected_length: u64,
) -> Result<(), UpdateTransportError> {
    if response
        .content_encoding
        .as_deref()
        .is_some_and(|encoding| !encoding.eq_ignore_ascii_case("identity"))
    {
        return Err(UpdateTransportError::ContentEncoding);
    }
    if response
        .content_length
        .is_some_and(|length| length != expected_length)
    {
        return Err(UpdateTransportError::Length);
    }
    Ok(())
}

fn read_bounded_exact(
    mut response: HttpResponse,
    spec: &TargetFetchSpec<'_>,
) -> Result<Vec<u8>, UpdateTransportError> {
    let expected = spec.descriptor().length();
    let capacity = usize::try_from(expected).map_err(|_| UpdateTransportError::Length)?;
    let mut bytes = Vec::with_capacity(capacity);
    response
        .body
        .by_ref()
        .take(expected.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| UpdateTransportError::BodyRead)?;
    if bytes.len() as u64 != expected {
        return Err(UpdateTransportError::Length);
    }
    if hex::encode(Sha256::digest(&bytes)) != spec.descriptor().sha256().as_str() {
        return Err(UpdateTransportError::Digest);
    }
    Ok(bytes)
}

pub(super) fn stream_archive(
    mut response: HttpResponse,
    spec: &TargetFetchSpec<'_>,
    mut stage: EphemeralArtifactStage,
) -> Result<crate::distribution::install_state::VerifiedArchiveFile, UpdateTransportError> {
    let expected = spec.descriptor().length();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let remaining = expected.saturating_sub(total);
        let limit = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| UpdateTransportError::Length)?;
        let read_limit = if limit == 0 { 1 } else { limit };
        let count = response
            .body
            .read(&mut buffer[..read_limit])
            .map_err(|_| UpdateTransportError::BodyRead)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or(UpdateTransportError::Length)?;
        if total > expected {
            return Err(UpdateTransportError::Length);
        }
        stage.write_chunk(&buffer[..count])?;
    }
    if total != expected {
        return Err(UpdateTransportError::Length);
    }
    // A final one-byte read detects a response that exactly filled the signed
    // length but continues beyond it.
    let mut extra = [0_u8; 1];
    if response
        .body
        .read(&mut extra)
        .map_err(|_| UpdateTransportError::BodyRead)?
        != 0
    {
        return Err(UpdateTransportError::Length);
    }
    Ok(stage.finish(spec.descriptor().sha256())?)
}
