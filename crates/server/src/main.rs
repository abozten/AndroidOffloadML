//! `offload-server`: runs ML/compute tasks on an Android phone under Termux.
//!
//! Exposes an HTTP API (`/v1/run`, `/v1/jobs`, `/v1/files/*`) guarded by an optional
//! bearer token, plus a UDP responder so clients can find the phone on the LAN.

mod discovery;
mod error;
mod files;
mod jobs;
mod runner;
mod state;
mod telemetry;

use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    process::ExitCode,
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post, put},
};
use clap::Parser;
use offload_protocol::{DEFAULT_DISCOVERY_PORT, ModelEntry, NodeInfo};
use tracing::{error, info, warn};

use crate::{
    discovery::{DiscoveryConfig, spawn_discovery_responder},
    error::ApiError,
    state::{AppState, ServerConfig},
};

/// musl's allocator is slow under multithreaded contention (`MatMul`/JSON on several
/// worker threads); mimalloc fixes that. Opt-in via the `mimalloc` feature (needs a C cross-compiler).
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Parser)]
#[command(
    name = "offload-server",
    about = "ML task offload server for Termux/Android"
)]
struct Cli {
    /// Address to bind the HTTP server to (e.g. 0.0.0.0:7878 for LAN)
    #[arg(long, env = "OFFLOAD_BIND", default_value = "0.0.0.0:7878")]
    bind: SocketAddr,

    /// Human-readable node name advertised over LAN discovery
    #[arg(long, env = "OFFLOAD_NODE_NAME", default_value = "oneplus-15")]
    name: String,

    /// UDP port for LAN discovery probes
    #[arg(long, env = "OFFLOAD_DISCOVERY_PORT", default_value_t = DEFAULT_DISCOVERY_PORT)]
    discovery_port: u16,

    /// Disable UDP LAN discovery responder
    #[arg(long, env = "OFFLOAD_NO_DISCOVERY", default_value_t = false)]
    no_discovery: bool,

    /// Optional bearer token required on /v1/* endpoints
    #[arg(long, env = "OFFLOAD_TOKEN")]
    token: Option<String>,

    /// Maximum number of ML tasks to run concurrently
    #[arg(long, env = "OFFLOAD_MAX_CONCURRENT", default_value_t = 2)]
    max_concurrent: usize,

    /// Base directory for GGUF models used by `llama_completion` tasks
    #[arg(long, env = "OFFLOAD_MODELS_DIR")]
    models_dir: Option<PathBuf>,

    /// Directory where resumable file uploads are stored (defaults to --models-dir or ./uploads)
    #[arg(long, env = "OFFLOAD_FILES_DIR")]
    files_dir: Option<PathBuf>,

    /// Path or binary name for llama.cpp CLI
    #[arg(long, env = "OFFLOAD_LLAMA_BIN", default_value = "llama-cli")]
    llama_bin: String,

    /// Comma-separated allowlist of external commands permitted for `command` tasks
    #[arg(long, env = "OFFLOAD_ALLOW_CMD", value_delimiter = ',')]
    allow_cmd: Vec<String>,

    /// Default timeout in seconds for subprocess tasks
    #[arg(long, env = "OFFLOAD_TIMEOUT_SECS", default_value_t = 300)]
    default_timeout_secs: u64,

    /// Maximum request body size in megabytes
    #[arg(long, env = "OFFLOAD_MAX_BODY_MB", default_value_t = 64)]
    max_body_mb: usize,
}

fn build_router(state: AppState, max_body_mb: usize) -> Router {
    let api_routes = Router::new()
        .route("/info", get(get_info))
        .route("/models", get(list_models))
        .route("/run", post(jobs::run_sync))
        .route("/jobs", post(jobs::submit_job).get(jobs::list_jobs))
        .route("/jobs/{id}", get(jobs::get_job).delete(jobs::cancel_job))
        .route("/jobs/{id}/stream", get(jobs::stream_job))
        .route("/files/status", get(files::file_status))
        .route("/files/chunk", put(files::upload_chunk))
        .route("/files/download", get(files::download_file))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::new()
        .route("/health", get(health))
        .nest("/v1", api_routes)
        .layer(DefaultBodyLimit::max(max_body_mb.max(1) * 1024 * 1024))
        .with_state(state)
}

async fn auth_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let Some(expected) = state.config.auth_token.as_deref() else {
        return Ok(next.run(req).await);
    };

    let authorized = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|token| constant_time_eq(token.as_bytes(), expected.as_bytes()));

    if !authorized {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token",
        ));
    }

    Ok(next.run(req).await)
}

/// Compares secrets without an early exit, so timing doesn't reveal the matching prefix.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0_u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    std::hint::black_box(diff) == 0
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn get_info(State(state): State<AppState>) -> Json<NodeInfo> {
    let logical_cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let tel = telemetry::read_telemetry();

    Json(NodeInfo {
        service: "offload-server".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        logical_cpus,
        mem_total_mb: tel.mem_total_mb,
        mem_available_mb: tel.mem_available_mb,
        thermal_c: tel.thermal_c,
        max_concurrent_jobs: state.config.max_concurrent_jobs,
        active_jobs: state.active_job_count(),
        models_dir: state
            .config
            .models_dir
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        allowed_commands: state.config.allowed_commands.clone(),
    })
}

async fn list_models(State(state): State<AppState>) -> Result<Json<Vec<ModelEntry>>, ApiError> {
    let Some(dir) = &state.config.models_dir else {
        return Ok(Json(Vec::new()));
    };
    let mut entries = Vec::new();
    let mut rd = match tokio::fs::read_dir(dir).await {
        Ok(rd) => rd,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Json(Vec::new())),
        Err(err) => return Err(err.into()),
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if Path::new(&name)
            .extension()
            .is_some_and(|ext| ext == "part")
        {
            continue;
        }
        entries.push(ModelEntry {
            name,
            size_bytes: meta.len(),
        });
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Json(entries))
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "offload_server=info".into()),
        )
        .init();

    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!(error = %err, "offload-server failed");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> std::io::Result<()> {
    let files_dir = cli
        .files_dir
        .or_else(|| cli.models_dir.clone())
        .unwrap_or_else(|| PathBuf::from("uploads"));

    let config = ServerConfig {
        auth_token: cli.token.filter(|t| !t.is_empty()),
        max_concurrent_jobs: cli.max_concurrent.max(1),
        models_dir: cli.models_dir,
        files_dir,
        llama_bin: cli.llama_bin,
        allowed_commands: cli
            .allow_cmd
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        default_timeout_secs: cli.default_timeout_secs.max(1),
    };

    let state = AppState::new(config.clone());
    let app = build_router(state, cli.max_body_mb);

    let listener = tokio::net::TcpListener::bind(cli.bind)
        .await
        .map_err(|e| std::io::Error::new(e.kind(), format!("failed to bind {}: {e}", cli.bind)))?;
    let http_port = listener.local_addr()?.port();

    if !cli.no_discovery {
        let d_cfg = DiscoveryConfig {
            port: cli.discovery_port,
            http_port,
            node_name: cli.name,
            auth_required: config.auth_token.is_some(),
        };
        if let Err(err) = spawn_discovery_responder(d_cfg).await {
            warn!(error = %err, "failed to start LAN discovery responder");
        }
    }

    info!(
        bind = %cli.bind,
        max_concurrent = config.max_concurrent_jobs,
        auth_enabled = config.auth_token.is_some(),
        models_dir = ?config.models_dir,
        allowed_commands = ?config.allowed_commands,
        "offload-server listening"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::{body::Body, http::Request as HttpRequest};
    use offload_protocol::{
        JobRecord, JobStatus, RunTaskRequest, RunTaskResponse, TaskKind, TaskResult,
    };
    use tower::ServiceExt;
    use uuid::Uuid;

    use super::*;

    fn test_state(token: Option<&str>) -> AppState {
        let dir = std::env::temp_dir().join(format!("offload-test-{}", Uuid::new_v4()));
        AppState::new(ServerConfig {
            auth_token: token.map(str::to_string),
            max_concurrent_jobs: 2,
            models_dir: None,
            files_dir: dir,
            llama_bin: "llama-cli".to_string(),
            allowed_commands: vec![],
            default_timeout_secs: 30,
        })
    }

    #[tokio::test]
    async fn health_is_open_and_v1_requires_token_when_configured() {
        let app = build_router(test_state(Some("secret-token")), 16);

        let health_res = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health_res.status(), StatusCode::OK);

        let unauth_res = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauth_res.status(), StatusCode::UNAUTHORIZED);

        let auth_res = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/info")
                    .header(header::AUTHORIZATION, "Bearer secret-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(auth_res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn run_sync_executes_matmul() {
        let app = build_router(test_state(None), 16);
        let req_body = serde_json::to_vec(&RunTaskRequest {
            task: TaskKind::MatMul {
                rows_a: 2,
                cols_a: 2,
                cols_b: 2,
                a: vec![1.0, 2.0, 3.0, 4.0],
                b: vec![5.0, 6.0, 7.0, 8.0],
            },
        })
        .unwrap();

        let res = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/v1/run")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(req_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(res.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: RunTaskResponse = serde_json::from_slice(&bytes).unwrap();
        match parsed.result {
            TaskResult::MatMul { rows, cols, data } => {
                assert_eq!((rows, cols), (2, 2));
                assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[tokio::test]
    async fn chunked_upload_supports_status_and_resume() {
        let state = test_state(None);
        let files_dir = state.config.files_dir.clone();
        let app = build_router(state, 16);

        // Upload first 5 bytes of an 11-byte payload.
        let put1 = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("PUT")
                    .uri("/v1/files/chunk?path=test.bin&offset=0&total_size=11")
                    .body(Body::from("hello"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(put1.status(), StatusCode::OK);

        // Query status -> should report partial_size=5, exists=false.
        let partial_res = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/files/status?path=test.bin")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let partial_bytes = axum::body::to_bytes(partial_res.into_body(), 4096)
            .await
            .unwrap();
        let status: offload_protocol::FileTransferStatus =
            serde_json::from_slice(&partial_bytes).unwrap();
        assert!(!status.exists);
        assert_eq!(status.partial_size, 5);

        // Resume remaining 6 bytes.
        let put2 = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("PUT")
                    .uri("/v1/files/chunk?path=test.bin&offset=5&total_size=11")
                    .body(Body::from(" world"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(put2.status(), StatusCode::OK);
        let content = std::fs::read_to_string(files_dir.join("test.bin")).unwrap();
        assert_eq!(content, "hello world");

        // Verify blake3 hash on completed status and Range-resumed download.
        let done_res = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/files/status?path=test.bin")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let done_bytes = axum::body::to_bytes(done_res.into_body(), 4096)
            .await
            .unwrap();
        let done_status: offload_protocol::FileTransferStatus =
            serde_json::from_slice(&done_bytes).unwrap();
        assert!(done_status.exists);
        assert_eq!(done_status.final_size, Some(11));
        assert_eq!(
            done_status.blake3.as_deref(),
            Some(blake3::hash(b"hello world").to_hex().as_str())
        );

        let dl = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/files/download?path=test.bin")
                    .header(header::RANGE, "bytes=6-")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(dl.status(), StatusCode::PARTIAL_CONTENT);
        let dl_bytes = axum::body::to_bytes(dl.into_body(), 4096).await.unwrap();
        assert_eq!(&*dl_bytes, b"world");

        let _ = std::fs::remove_dir_all(&files_dir);
    }

    #[tokio::test]
    async fn cancel_job_marks_cancelled() {
        let state = test_state(None);
        let (record, _) = state.create_job().await;
        let app = build_router(state, 16);

        let res = app
            .oneshot(
                HttpRequest::builder()
                    .method("DELETE")
                    .uri(format!("/v1/jobs/{}", record.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
        let updated: JobRecord = serde_json::from_slice(&body).unwrap();
        assert_eq!(updated.status, JobStatus::Cancelled);
    }

    #[tokio::test]
    async fn stream_ends_after_done_even_if_sender_outlives_job() {
        let state = test_state(None);
        // `_leaked_tx` plays a pipe-reader task that still holds a sender after cancel.
        let (record, _leaked_tx) = state.create_job().await;
        let app = build_router(state.clone(), 16);

        let res = app
            .oneshot(
                HttpRequest::builder()
                    .uri(format!("/v1/jobs/{}/stream", record.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        state.cancel_job(record.id).await.unwrap();

        let body = tokio::time::timeout(
            Duration::from_secs(2),
            axum::body::to_bytes(res.into_body(), 64 * 1024),
        )
        .await
        .expect("SSE stream must close after Done")
        .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("cancelled"));
    }
}
