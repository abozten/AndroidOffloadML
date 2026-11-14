//! HTTP error type shared by all handlers.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use offload_protocol::ErrorResponse;

use crate::runner::RunnerError;

/// An HTTP status plus message, rendered as an [`ErrorResponse`] JSON body.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }
}

impl From<RunnerError> for ApiError {
    fn from(err: RunnerError) -> Self {
        let status = match err {
            RunnerError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            RunnerError::Forbidden(_) => StatusCode::FORBIDDEN,
            RunnerError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
            RunnerError::Process(_) => StatusCode::BAD_GATEWAY,
            RunnerError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, err.to_string())
    }
}

/// Unexpected server-side I/O (disk full, permissions, ...) is a 500.
impl From<std::io::Error> for ApiError {
    fn from(err: std::io::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
    }
}

/// A blocking hash task panicked or was cancelled.
impl From<tokio::task::JoinError> for ApiError {
    fn from(err: tokio::task::JoinError) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
            }),
        )
            .into_response()
    }
}
