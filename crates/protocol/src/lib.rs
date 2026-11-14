//! Wire types shared by `offload-server`, `offload-client`, and `offload-mcp`.
//!
//! Every type here is serialized as JSON over HTTP (or UDP for discovery), so any
//! field rename or enum-tag change is a breaking protocol change for deployed phones.
//!
//! - Discovery: [`DiscoveryProbe`] is broadcast by clients, [`DiscoveryBeacon`] is the reply.
//! - Execution: [`RunTaskRequest`] carries a [`TaskKind`]; results come back as [`TaskResult`].
//! - Async jobs: [`JobRecord`] snapshots and live [`JobStreamEvent`]s over SSE.
//! - File transfer: [`FileTransferStatus`] and [`ChunkUploadResponse`], verified with [`blake3_hex`].

use std::{fmt, time::Duration};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Read buffer for [`blake3_hex`].
///
/// 256 KiB amortizes syscalls on multi-GB model files without a large per-hash allocation.
const HASH_BUF_BYTES: usize = 256 * 1024;

/// Streaming BLAKE3 hex digest of everything `reader` yields.
///
/// Client and server must agree byte-for-byte, so both use this one function.
///
/// # Errors
///
/// Returns the first I/O error raised by `reader`.
pub fn blake3_hex(mut reader: impl std::io::Read) -> std::io::Result<String> {
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0_u8; HASH_BUF_BYTES];
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

/// How long clients wait for discovery beacons by default.
///
/// 1.5 s covers phones waking Wi-Fi from power-save; shorter values miss sleeping
/// devices, longer ones add latency to every CLI/MCP call without `--url`.
pub const DEFAULT_DISCOVERY_WAIT: Duration = Duration::from_millis(1500);

/// Multicast IPv4 group used alongside UDP broadcast for LAN discovery.
pub const DISCOVERY_MULTICAST_ADDR: &str = "239.255.78.79";

/// UDP probe sent by clients to discover offload servers on the local network.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryProbe {
    /// Must equal [`DiscoveryProbe::MAGIC`]; servers ignore anything else.
    pub magic: String,
}

impl DiscoveryProbe {
    /// Protocol tag identifying a v1 discovery probe.
    pub const MAGIC: &'static str = "android-offload-discover-v1";

    /// Creates a probe carrying [`DiscoveryProbe::MAGIC`].
    #[must_use]
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
    /// Must equal [`DiscoveryBeacon::MAGIC`]; clients ignore anything else.
    pub magic: String,
    pub service: String,
    pub version: String,
    /// Human-readable node name (`--name`).
    pub name: String,
    /// TCP port of the HTTP API; the host is the beacon's source address.
    pub http_port: u16,
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    /// Whether `/v1/*` requires a bearer token.
    pub auth_required: bool,
}

impl DiscoveryBeacon {
    /// Protocol tag identifying a v1 discovery beacon.
    pub const MAGIC: &'static str = "android-offload-beacon-v1";
}

/// A discovered worker node with its resolved HTTP base URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredNode {
    /// Base URL, e.g. `http://192.168.1.20:7878`.
    pub url: String,
    pub name: String,
    pub version: String,
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub auth_required: bool,
}

/// Hardware and capability snapshot of a worker node (`GET /v1/info`).
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
    /// Hottest thermal zone in °C, if readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal_c: Option<f32>,
    pub max_concurrent_jobs: usize,
    pub active_jobs: usize,
    pub models_dir: Option<String>,
    /// Programs permitted for [`TaskKind::Command`].
    pub allowed_commands: Vec<String>,
}

/// A model file discovered in `--models-dir`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// File name relative to `--models-dir`.
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

    /// Dense single-precision matrix multiplication: (`rows_a` x `cols_a`) * (`cols_a` x `cols_b`).
    MatMul {
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
        /// Row-major, length `rows_a * cols_a`.
        a: Vec<f32>,
        /// Row-major, length `cols_a * cols_b`.
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
        /// Extra llama.cpp flags; the server only accepts an allowlisted subset.
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
        /// Falls back to the server's `--default-timeout-secs`.
        #[serde(default)]
        timeout_secs: Option<u64>,
    },
}

/// Bounded default: llama.cpp's own `-n` default is unlimited, which would run until the timeout.
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
        /// Row-major, length `rows * cols`.
        data: Vec<f32>,
    },
    CosineTopK {
        /// Best match first.
        matches: Vec<SimilarityMatch>,
    },
    /// Output of a subprocess task (`LlamaCompletion` or `Command`).
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
    /// Index into the request's `corpus`.
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

impl JobStatus {
    /// Whether the job can no longer change state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        })
    }
}

/// Which subprocess pipe a [`JobStreamEvent::Output`] line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputStream {
    Stdout,
    Stderr,
}

/// Live SSE event emitted by `/v1/jobs/{id}/stream`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum JobStreamEvent {
    /// One line of subprocess output, without its trailing newline.
    Output { stream: OutputStream, line: String },
    /// Final event; the stream closes after it.
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
    /// Milliseconds since the Unix epoch.
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
    /// Path relative to `--files-dir`, as requested.
    pub path: String,
    /// Whether the completed file exists.
    pub exists: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_size: Option<u64>,
    /// BLAKE3 hex digest of the completed file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blake3: Option<String>,
    /// Bytes already staged in the `.part` file; uploads resume from here.
    pub partial_size: u64,
}

/// Response returned after uploading a single binary file chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkUploadResponse {
    pub path: String,
    pub partial_size: u64,
    /// Whether this chunk finished the file (it has been renamed into place).
    pub completed: bool,
    /// BLAKE3 hex digest, present once `completed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blake3: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_stream_keeps_its_wire_format() {
        // Older clients compare the raw string, so the enum must serialize to it unchanged.
        let ev = JobStreamEvent::Output {
            stream: OutputStream::Stderr,
            line: "x".into(),
        };
        let json = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["stream"], "stderr");
    }

    #[test]
    fn job_status_display_matches_serde_name() {
        for s in [
            JobStatus::Queued,
            JobStatus::Running,
            JobStatus::Succeeded,
            JobStatus::Failed,
            JobStatus::Cancelled,
        ] {
            assert_eq!(serde_json::to_value(s).unwrap(), s.to_string());
        }
    }
}
