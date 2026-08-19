use std::io::Read;

use super::http::{HttpExecutor, HttpResponse};
use super::origin::pages_metadata_url;
use super::UpdateTransportError;
use crate::distribution::install_state::metadata::{
    MetadataCommitOutcome, MetadataStateAuthorization,
};
use crate::distribution::update_auth::{
    begin_metadata_update, EmbeddedTrustRoot, MetadataFetchResponse, MetadataSessionProgress,
    MetadataUpdateSession,
};

pub(super) fn refresh_metadata(
    executor: &impl HttpExecutor,
    authorization: &MetadataStateAuthorization,
    anchor: &EmbeddedTrustRoot,
) -> Result<MetadataCommitOutcome, UpdateTransportError> {
    let mut session = begin_metadata_update(authorization, anchor)?;
    loop {
        let response = fetch_response(executor, &session)?;
        match session.respond(response)? {
            MetadataSessionProgress::Request(next) => session = next,
            MetadataSessionProgress::Complete(outcome) => return Ok(outcome),
        }
    }
}

fn fetch_response(
    executor: &impl HttpExecutor,
    session: &MetadataUpdateSession<'_>,
) -> Result<MetadataFetchResponse, UpdateTransportError> {
    let spec = session.request();
    let request = pages_metadata_url(spec)?;
    let response = executor.execute(&request, false)?;
    if response.location.is_some() {
        return Err(UpdateTransportError::Status);
    }
    match response.status {
        200 => Ok(MetadataFetchResponse::Found(
            read_bounded_metadata(response, spec.maximum_bytes())?.into_boxed_slice(),
        )),
        404 if spec.accepts_confirmed_not_found() => Ok(MetadataFetchResponse::ConfirmedNotFound),
        _ => Err(UpdateTransportError::Status),
    }
}

fn read_bounded_metadata(
    mut response: HttpResponse,
    maximum_bytes: usize,
) -> Result<Vec<u8>, UpdateTransportError> {
    if response
        .content_encoding
        .as_deref()
        .is_some_and(|encoding| !encoding.eq_ignore_ascii_case("identity"))
    {
        return Err(UpdateTransportError::ContentEncoding);
    }
    if response.content_length.is_some_and(|length| {
        length == 0 || usize::try_from(length).map_or(true, |length| length > maximum_bytes)
    }) {
        return Err(UpdateTransportError::Length);
    }
    let mut bytes = Vec::with_capacity(
        response
            .content_length
            .and_then(|length| usize::try_from(length).ok())
            .unwrap_or(0),
    );
    let read_limit = u64::try_from(maximum_bytes)
        .map_err(|_| UpdateTransportError::Length)?
        .checked_add(1)
        .ok_or(UpdateTransportError::Length)?;
    response
        .body
        .by_ref()
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|_| UpdateTransportError::BodyRead)?;
    if bytes.is_empty()
        || bytes.len() > maximum_bytes
        || response
            .content_length
            .is_some_and(|length| length != bytes.len() as u64)
    {
        return Err(UpdateTransportError::Length);
    }
    Ok(bytes)
}

#[cfg(test)]
pub(super) fn read_bounded_metadata_for_test(
    response: HttpResponse,
    maximum_bytes: usize,
) -> Result<Vec<u8>, UpdateTransportError> {
    read_bounded_metadata(response, maximum_bytes)
}
