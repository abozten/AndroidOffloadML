use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Streaming BLAKE3 hex digest. Client and server must agree byte-for-byte, so both use this.
pub fn blake3_hex(mut reader: impl std::io::Read) -> std::io::Result<String> {
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0_u8; 256 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(hasher.finalize().to_hex().to_string());
        }
        hasher.update(&buf[..n]);
    }
}

/// Default UDP port used for zero-config LAN discovery.
pub const DEFAULT_DISCOVERY_PORT: u16 = 7879;

/// Multicast IPv4 group used alongside UDP broadcast for LAN discovery.
pub const DISCOVERY_MULTICAST_ADDR: &str = "239.255.78.79";

/// UDP probe sent by clients to discover offload servers on the local network.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryProbe {
    pub magic: String,
}

impl DiscoveryProbe {
    pub const MAGIC: &'static str = "android-offload-discover-v1";

    pub fn new() -> Self {
        Self {
            magic: Self::MAGIC.to_string(),
        }
    }
}

impl Default for DiscoveryProbe {
    fn default() -> Self {
        Self::new()
    }
}

/// UDP response sent by an offload server in reply to a [`DiscoveryProbe`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryBeacon {
    pub magic: String,
    pub service: String,
    pub version: String,
    pub name: String,
    pub http_port: u16,
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub auth_required: bool,
}

impl DiscoveryBeacon {
    pub const MAGIC: &'static str = "android-offload-beacon-v1";
}

/// A discovered worker node with its resolved HTTP base URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredNode {
    pub url: String,
    pub name: String,
    pub version: String,
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub auth_required: bool,
}

/// Information about the Termux/Android worker node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub service: String,
    pub version: String,
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_total_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_available_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_c: Option<f32>,
    pub max_concurrent_jobs: usize,
    pub active_jobs: usize,
    pub models_dir: Option<String>,
    pub allowed_commands: Vec<String>,
}

/// A model file discovered in `--models-dir`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub size_bytes: u64,
}

/// Request payload for both synchronous (`/v1/run`) and async (`/v1/jobs`) execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTaskRequest {
    pub task: TaskKind,
}

/// Supported ML and compute tasks that can be offloaded to the Android worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskKind {
    /// Lightweight round-trip check.
    Echo { payload: serde_json::Value },

    /// Dense single-precision matrix multiplication: (rows_a x cols_a) * (cols_a x cols_b).
    MatMul {
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
        a: Vec<f32>,
        b: Vec<f32>,
    },

    /// Top-K cosine similarity search over a batch of embedding vectors.
    CosineTopK {
        query: Vec<f32>,
        corpus: Vec<Vec<f32>>,
        top_k: usize,
    },

    /// Run local LLM inference via llama.cpp (`llama-cli` / `llama-completion`) on Termux.
    LlamaCompletion {
        /// Model filename relative to `--models-dir` (or absolute path if `--models-dir` is unset).
        model: String,
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: u32,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        extra_args: Vec<String>,
    },

    /// Run an allowlisted external ML command on Termux (e.g., `python3`, `whisper-cli`).
    Command {
        program: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        stdin: Option<String>,
        #[serde(default)]
        timeout_secs: Option<u64>,
    },
}

fn default_max_tokens() -> u32 {
    256
}

/// Result returned by a completed ML task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskResult {
    Echo {
        payload: serde_json::Value,
    },
    MatMul {
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    },
    CosineTopK {
        matches: Vec<SimilarityMatch>,
    },
    TextOutput {
        text: String,
        exit_code: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        stderr: Option<String>,
    },
}

/// A single similarity match from `CosineTopK`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMatch {
    pub index: usize,
    pub score: f32,
}

/// Response for synchronous task execution (`/v1/run`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTaskResponse {
    pub job_id: Uuid,
    pub duration_ms: u64,
    pub result: TaskResult,
}

/// Lifecycle state of an asynchronous job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// Live SSE event emitted by `/v1/jobs/{id}/stream`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum JobStreamEvent {
    Output {
        stream: String,
        line: String,
    },
    Done {
        status: JobStatus,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

/// Full status record for an asynchronous job (`/v1/jobs` and `/v1/jobs/{id}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: Uuid,
    pub status: JobStatus,
    pub created_at_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<TaskResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Standard JSON error body returned by the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Server-side status of a target file for resumable uploads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTransferStatus {
    pub path: String,
    pub exists: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blake3: Option<String>,
    pub partial_size: u64,
}

/// Response returned after uploading a single binary file chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkUploadResponse {
    pub path: String,
    pub partial_size: u64,
    pub completed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blake3: Option<String>,
}
