use std::io::Read;
use std::time::Duration;

use reqwest::blocking::{Client, Response};
use reqwest::header::{
    HeaderMap, HeaderName, ACCEPT_ENCODING, CACHE_CONTROL, CONTENT_ENCODING, CONTENT_LENGTH,
    LOCATION,
};

use super::origin::{OriginLockedUrl, RequestClass};
use super::UpdateTransportError;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const SMALL_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const ARCHIVE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

pub(super) trait HttpExecutor {
    fn execute(
        &self,
        request: &OriginLockedUrl,
        large: bool,
    ) -> Result<HttpResponse, UpdateTransportError>;
}

pub(super) struct HttpResponse {
    pub(super) status: u16,
    pub(super) content_length: Option<u64>,
    pub(super) content_encoding: Option<String>,
    pub(super) location: Option<String>,
    pub(super) body: Box<dyn Read>,
}

pub(super) struct ReqwestExecutor {
    small: Client,
    archive: Client,
}

impl ReqwestExecutor {
    pub(super) fn new() -> Result<Self, UpdateTransportError> {
        Ok(Self {
            small: build_client(SMALL_REQUEST_TIMEOUT)?,
            archive: build_client(ARCHIVE_REQUEST_TIMEOUT)?,
        })
    }
}

impl HttpExecutor for ReqwestExecutor {
    fn execute(
        &self,
        request: &OriginLockedUrl,
        large: bool,
    ) -> Result<HttpResponse, UpdateTransportError> {
        let client = if large { &self.archive } else { &self.small };
        let request_message = build_request(client, request)?;
        let response = client
            .execute(request_message)
            .map_err(sanitize_network_error)?;
        if response.url() != request.url() {
            return Err(UpdateTransportError::OriginPolicy);
        }
        response_parts(response)
    }
}

pub(super) fn build_request(
    client: &Client,
    request: &OriginLockedUrl,
) -> Result<reqwest::blocking::Request, UpdateTransportError> {
    let mut builder = client
        .get(request.url().clone())
        .header(ACCEPT_ENCODING, "identity");
    if request.class() == RequestClass::Pages {
        builder = builder.header(CACHE_CONTROL, "no-cache");
    }
    builder
        .build()
        .map_err(|error| UpdateTransportError::Network(error.without_url()))
}

fn build_client(timeout: Duration) -> Result<Client, UpdateTransportError> {
    Client::builder()
        .use_rustls_tls()
        .https_only(true)
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .referer(false)
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(timeout)
        .user_agent(concat!("hf2q/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|error| UpdateTransportError::Client(error.without_url()))
}

fn response_parts(response: Response) -> Result<HttpResponse, UpdateTransportError> {
    let (content_length, content_encoding, location) = parse_headers(response.headers())?;
    Ok(HttpResponse {
        status: response.status().as_u16(),
        content_length,
        content_encoding,
        location,
        body: Box::new(response),
    })
}

pub(super) fn parse_headers(
    headers: &HeaderMap,
) -> Result<(Option<u64>, Option<String>, Option<String>), UpdateTransportError> {
    let content_length = singleton_header(headers, &CONTENT_LENGTH)?
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|_| UpdateTransportError::Headers)
        })
        .transpose()?;
    let content_encoding = singleton_header(headers, &CONTENT_ENCODING)?;
    let location = singleton_header(headers, &LOCATION)?;
    Ok((content_length, content_encoding, location))
}

fn singleton_header(
    headers: &HeaderMap,
    name: &HeaderName,
) -> Result<Option<String>, UpdateTransportError> {
    let values: Vec<_> = headers.get_all(name).iter().collect();
    match values.as_slice() {
        [] => Ok(None),
        [value] if value.as_bytes().len() <= 16 * 1024 => value
            .to_str()
            .map(|value| Some(value.to_owned()))
            .map_err(|_| UpdateTransportError::Headers),
        [_] => Err(UpdateTransportError::Headers),
        _ => Err(UpdateTransportError::Headers),
    }
}

fn sanitize_network_error(error: reqwest::Error) -> UpdateTransportError {
    UpdateTransportError::Network(error.without_url())
}
