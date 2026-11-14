mod discovery;
mod files;
mod runner;
mod state;
mod telemetry;

use std::{
    convert::Infallible,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post, put},
};
use clap::Parser;
use offload_protocol::{
    DEFAULT_DISCOVERY_PORT, ErrorResponse, JobRecord, JobStatus, JobStreamEvent, ModelEntry,
    NodeInfo, RunTaskRequest, RunTaskResponse,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    discovery::{DiscoveryConfig, spawn_discovery_responder},
    runner::{RunnerError, execute_task},
    state::{AppState, ServerConfig},
};

#[derive(Debug, Parser)]
#[command(name = "offload-server", about = "ML task offload server for Termux/Android")]
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

    /// Base directory for GGUF models used by llama_completion tasks
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

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
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

fn build_router(state: AppState, max_body_mb: usize) -> Router {
    let api_routes = Router::new()
        .route("/info", get(get_info))
        .route("/models", get(list_models))
        .route("/run", post(run_sync))
        .route("/jobs", post(submit_job).get(list_jobs))
        .route("/jobs/{id}", get(get_job).delete(cancel_job))
        .route("/jobs/{id}/stream", get(stream_job))
        .route("/files/status", get(files::file_status))
        .route("/files/chunk", put(files::upload_chunk))
        .route("/files/download", get(files::download_file))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

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

    let auth_header = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    let authorized = auth_header
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

struct ActiveJobGuard(Arc<AtomicUsize>);

impl ActiveJobGuard {
    fn enter(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(counter)
    }
}

impl Drop for ActiveJobGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn get_info(State(state): State<AppState>) -> Json<NodeInfo> {
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
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
        Err(err) => return Err(ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())),
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".part") {
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

async fn run_sync(
    State(state): State<AppState>,
    Json(req): Json<RunTaskRequest>,
) -> Result<Json<RunTaskResponse>, ApiError> {
    let permit = state
        .semaphore
        .acquire()
        .await
        .map_err(|_| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "worker shutting down"))?;

    let _guard = ActiveJobGuard::enter(state.active_jobs.clone());
    let started = Instant::now();
    let result = execute_task(req.task, &state.config, None).await;
    let duration_ms = started.elapsed().as_millis() as u64;
    drop(permit);

    let result = result?;
    Ok(Json(RunTaskResponse {
        job_id: Uuid::new_v4(),
        duration_ms,
        result,
    }))
}

async fn submit_job(
    State(state): State<AppState>,
    Json(req): Json<RunTaskRequest>,
) -> (StatusCode, Json<JobRecord>) {
    let (record, stream_tx) = state.create_job().await;
    let job_id = record.id;
    let worker_state = state.clone();

    let handle = tokio::spawn(async move {
        let Ok(permit) = worker_state.semaphore.acquire().await else {
            worker_state
                .mark_failed(job_id, 0, "worker shutting down".into())
                .await;
            return;
        };

        let _guard = ActiveJobGuard::enter(worker_state.active_jobs.clone());
        worker_state.mark_running(job_id).await;
        let started = Instant::now();
        let outcome = execute_task(req.task, &worker_state.config, Some(stream_tx)).await;
        let duration_ms = started.elapsed().as_millis() as u64;
        drop(permit);

        match outcome {
            Ok(res) => worker_state.mark_succeeded(job_id, duration_ms, res).await,
            Err(err) => {
                error!(%job_id, error = %err, "async job failed");
                worker_state
                    .mark_failed(job_id, duration_ms, err.to_string())
                    .await;
            }
        }
    });

    state.set_abort_handle(job_id, handle.abort_handle()).await;
    (StatusCode::ACCEPTED, Json(record))
}

async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<JobRecord>, ApiError> {
    state
        .get_job(id)
        .await
        .map(Json)
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, format!("job {id} not found")))
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<JobRecord>, ApiError> {
    state
        .cancel_job(id)
        .await
        .map(Json)
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, format!("job {id} not found")))
}

async fn stream_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let (record, mut rx) = state
        .subscribe_job(id)
        .await
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, format!("job {id} not found")))?;

    let (tx, mpsc_rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        // If already terminal, emit Done immediately and close.
        if matches!(
            record.status,
            JobStatus::Succeeded | JobStatus::Failed | JobStatus::Cancelled
        ) {
            let done = JobStreamEvent::Done {
                status: record.status,
                duration_ms: record.duration_ms,
                error: record.error,
            };
            if let Ok(json) = serde_json::to_string(&done) {
                let _ = tx.send(Ok(Event::default().data(json))).await;
            }
            return;
        }

        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let is_done = matches!(ev, JobStreamEvent::Done { .. });
                    if let Ok(json) = serde_json::to_string(&ev)
                        && tx.send(Ok(Event::default().data(json))).await.is_err()
                    {
                        break;
                    }
                    if is_done {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    Ok(Sse::new(ReceiverStream::new(mpsc_rx))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}

async fn list_jobs(State(state): State<AppState>) -> Json<Vec<JobRecord>> {
    Json(state.list_jobs().await)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "offload_server=info".into()),
        )
        .init();

    let cli = Cli::parse();
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
        .unwrap_or_else(|e| panic!("failed to bind {}: {e}", cli.bind));
    let http_port = listener
        .local_addr()
        .map(|a| a.port())
        .unwrap_or_else(|_| cli.bind.port());

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
        .expect("server failed");
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request as HttpRequest};
    use offload_protocol::{TaskKind, TaskResult};
    use tower::ServiceExt;

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
        let st = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/files/status?path=test.bin")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let st_bytes = axum::body::to_bytes(st.into_body(), 4096).await.unwrap();
        let status: offload_protocol::FileTransferStatus =
            serde_json::from_slice(&st_bytes).unwrap();
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
        let st2 = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/files/status?path=test.bin")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let st2_bytes = axum::body::to_bytes(st2.into_body(), 4096).await.unwrap();
        let status2: offload_protocol::FileTransferStatus =
            serde_json::from_slice(&st2_bytes).unwrap();
        assert!(status2.exists);
        assert_eq!(status2.final_size, Some(11));
        assert_eq!(
            status2.blake3.as_deref(),
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
        assert_eq!(&dl_bytes[..], b"world");

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
}
