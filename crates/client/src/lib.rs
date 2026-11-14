//! Async client for `offload-server`, the Android/Termux ML offload worker.
//!
//! [`OffloadClient`] wraps the HTTP API: synchronous runs, async jobs with SSE
//! streaming, and resumable BLAKE3-verified file transfer. [`discover`] and
//! [`discover_first`] find servers on the LAN when no URL is configured.
//!
//! ```no_run
//! # async fn demo() -> Result<(), offload_client::ClientError> {
//! use offload_client::{OffloadClient, discover_first};
//! use offload_protocol::{DEFAULT_DISCOVERY_PORT, DEFAULT_DISCOVERY_WAIT, TaskKind};
//!
//! let node = discover_first(DEFAULT_DISCOVERY_PORT, DEFAULT_DISCOVERY_WAIT).await?;
//! let client = OffloadClient::new(node.url, None)?;
//! let res = client.run(TaskKind::Echo { payload: "hi".into() }).await?;
//! println!("{} ms", res.duration_ms);
//! # Ok(()) }
//! ```

use std::{
    collections::HashSet,
    io::{self, Read, SeekFrom},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use if_addrs::IfAddr;
use offload_protocol::{
    ChunkUploadResponse, DISCOVERY_MULTICAST_ADDR, DiscoveredNode, DiscoveryBeacon, DiscoveryProbe,
    ErrorResponse, FileTransferStatus, JobRecord, JobStatus, JobStreamEvent, ModelEntry, NodeInfo,
    RunTaskRequest, RunTaskResponse, TaskKind,
};
use reqwest::{Client, RequestBuilder, StatusCode, header};
use socket2::SockRef;
use thiserror::Error;
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    net::UdpSocket,
};
use uuid::Uuid;

/// Error returned by every fallible [`OffloadClient`] operation.
#[derive(Debug, Error)]
pub enum ClientError {
    /// Connection, TLS, timeout, or response-decoding failure.
    #[error("http transport error: {0}")]
    Transport(#[from] TransportError),
    /// Local file or socket I/O failure.
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    /// Discovery finished without any server replying.
    #[error("no offload servers discovered on LAN")]
    NoNodesDiscovered,
    /// A transferred file's BLAKE3 digest did not match the other side's.
    #[error("integrity check failed: expected blake3 {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    /// The server answered with a non-2xx HTTP `status` code.
    #[error("server returned {status}: {message}")]
    Api { status: u16, message: String },
    /// An async job ended `failed` or `cancelled`.
    #[error("job {0} failed: {1}")]
    JobFailed(Uuid, String),
}

/// Opaque HTTP transport failure; keeps `reqwest` out of this crate's public API.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct TransportError(reqwest::Error);

impl From<reqwest::Error> for ClientError {
    fn from(err: reqwest::Error) -> Self {
        Self::Transport(TransportError(err))
    }
}

/// TCP connect timeout. Phones on Wi-Fi power-save can take a few seconds to answer.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Max gap between bytes on a response, not total request time.
///
/// Sync `/v1/run` llama tasks stay silent until done, so this caps how long a sync
/// request can run; use async jobs for anything longer.
const READ_TIMEOUT: Duration = Duration::from_secs(60);

/// Floor for upload chunk size: smaller chunks make per-request overhead dominate.
const MIN_CHUNK_BYTES: usize = 64 * 1024;

/// Poll interval the CLI and MCP use for [`OffloadClient::run_and_wait`].
///
/// Half a second keeps latency low while costing the phone ~2 tiny requests/s.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Live progress snapshot emitted during a resumable file transfer.
#[derive(Debug, Clone, Copy)]
pub struct TransferProgress {
    pub bytes_done: u64,
    pub total_bytes: u64,
    pub resumed_bytes: u64,
    pub speed_bps: f64,
    pub eta_secs: Option<u64>,
}

/// Final summary returned when a file upload completes (or is skipped as already present).
#[derive(Debug, Clone)]
pub struct TransferSummary {
    pub remote_path: String,
    pub total_bytes: u64,
    pub resumed_bytes: u64,
    pub uploaded_bytes: u64,
    pub elapsed: Duration,
    pub already_existed: bool,
}

/// Async client for communicating with an `offload-server` running on Termux/Android.
#[derive(Debug, Clone)]
pub struct OffloadClient {
    base_url: String,
    token: Option<String>,
    http: Client,
}

impl OffloadClient {
    /// Creates a client for the server at `base_url` (e.g. `http://192.168.1.20:7878`).
    ///
    /// An empty `token` is treated as no token.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] if the HTTP client cannot be built (TLS init).
    pub fn new(base_url: impl Into<String>, token: Option<String>) -> Result<Self, ClientError> {
        let http = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .build()?;
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token: token.filter(|t| !t.is_empty()),
            http,
        })
    }

    /// Checks the unauthenticated `/health` endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] if unreachable, [`ClientError::Api`] on non-2xx.
    pub async fn health(&self) -> Result<(), ClientError> {
        let res = self
            .http
            .get(format!("{}/health", self.base_url))
            .send()
            .await?;
        if res.status().is_success() {
            Ok(())
        } else {
            Err(extract_api_error(res).await)
        }
    }

    /// Fetches hardware and capability info (`GET /v1/info`).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn info(&self) -> Result<NodeInfo, ClientError> {
        let req = self.auth(self.http.get(format!("{}/v1/info", self.base_url)));
        send_json(req).await
    }

    /// Lists model files in the server's `--models-dir`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn list_models(&self) -> Result<Vec<ModelEntry>, ClientError> {
        let req = self.auth(self.http.get(format!("{}/v1/models", self.base_url)));
        send_json(req).await
    }

    /// Runs `task` synchronously and returns its result (`POST /v1/run`).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn run(&self, task: TaskKind) -> Result<RunTaskResponse, ClientError> {
        let req = self
            .auth(self.http.post(format!("{}/v1/run", self.base_url)))
            .json(&RunTaskRequest { task });
        send_json(req).await
    }

    /// Queues `task` as an async job and returns its initial record (`POST /v1/jobs`).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn submit(&self, task: TaskKind) -> Result<JobRecord, ClientError> {
        let req = self
            .auth(self.http.post(format!("{}/v1/jobs", self.base_url)))
            .json(&RunTaskRequest { task });
        send_json(req).await
    }

    /// Fetches the current record of async job `id`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn job_status(&self, id: Uuid) -> Result<JobRecord, ClientError> {
        let req = self.auth(self.http.get(format!("{}/v1/jobs/{id}", self.base_url)));
        send_json(req).await
    }

    /// Lists recent async jobs, newest first.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn list_jobs(&self) -> Result<Vec<JobRecord>, ClientError> {
        let req = self.auth(self.http.get(format!("{}/v1/jobs", self.base_url)));
        send_json(req).await
    }

    /// Cancels async job `id` if still queued or running, returning its record.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn cancel_job(&self, id: Uuid) -> Result<JobRecord, ClientError> {
        let req = self.auth(self.http.delete(format!("{}/v1/jobs/{id}", self.base_url)));
        send_json(req).await
    }

    /// Streams live SSE output and the final `Done` event of async job `id` into `on_event`.
    ///
    /// Returns after `Done`, or when the server closes the stream.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn stream_job<F>(&self, id: Uuid, mut on_event: F) -> Result<(), ClientError>
    where
        F: FnMut(JobStreamEvent),
    {
        let mut res = self
            .auth(
                self.http
                    .get(format!("{}/v1/jobs/{id}/stream", self.base_url)),
            )
            .send()
            .await?;
        if !res.status().is_success() {
            return Err(extract_api_error(res).await);
        }

        let mut buf = Vec::<u8>::new();
        while let Some(chunk) = res.chunk().await? {
            buf.extend_from_slice(&chunk);
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line = String::from_utf8_lossy(&buf[..pos]);
                let line = line.trim_end_matches('\r');
                if let Some(data) = line.strip_prefix("data:") {
                    let payload = data.trim();
                    if let Ok(ev) = serde_json::from_str::<JobStreamEvent>(payload) {
                        let done = matches!(ev, JobStreamEvent::Done { .. });
                        on_event(ev);
                        if done {
                            return Ok(());
                        }
                    }
                }
                buf.drain(..=pos);
            }
        }
        Ok(())
    }

    /// Queries server-side upload status for a path relative to `--files-dir`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on network failure or [`ClientError::Api`] on non-2xx.
    pub async fn file_status(&self, remote_path: &str) -> Result<FileTransferStatus, ClientError> {
        let remote_path = normalize_remote_path(remote_path);
        let req = self
            .auth(self.http.get(format!("{}/v1/files/status", self.base_url)))
            .query(&[("path", remote_path.as_str())]);
        send_json(req).await
    }

    /// Uploads `local_path` to `remote_path` in chunks, resuming any partial upload.
    ///
    /// Skips the transfer when the server already has an identical file (by BLAKE3).
    /// `on_progress` receives throttled speed/ETA snapshots.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Io`] if the local file can't be read or shrinks mid-upload,
    /// [`ClientError::ChecksumMismatch`] if the server's digest differs, and transport or
    /// API errors from the server.
    pub async fn upload_file<F>(
        &self,
        local_path: &Path,
        remote_path: &str,
        chunk_size: usize,
        mut on_progress: F,
    ) -> Result<TransferSummary, ClientError>
    where
        F: FnMut(TransferProgress),
    {
        let remote_path = normalize_remote_path(remote_path);
        let mut file = File::open(local_path).await?;
        let total_bytes = file.metadata().await?.len();
        let started = Instant::now();
        let mut local_hash = BackgroundHash::spawn(local_path.to_path_buf());

        let status = self.file_status(&remote_path).await?;
        if status.exists
            && status.final_size == Some(total_bytes)
            && let Some(remote_hash) = status.blake3.as_deref()
            && remote_hash == local_hash.get().await?
        {
            return Ok(TransferSummary {
                remote_path,
                total_bytes,
                resumed_bytes: total_bytes,
                uploaded_bytes: 0,
                elapsed: started.elapsed(),
                already_existed: true,
            });
        }

        let mut offset = if status.partial_size <= total_bytes {
            status.partial_size
        } else {
            0
        };
        let resumed_bytes = offset;
        file.seek(SeekFrom::Start(offset)).await?;

        let chunk_cap = chunk_size.max(MIN_CHUNK_BYTES) as u64;
        let mut tracker = ProgressTracker::new(total_bytes, resumed_bytes);

        on_progress(tracker.initial());

        loop {
            // A fresh Vec per chunk is moved into the request body: no zeroing, no copy.
            let mut chunk = Vec::new();
            (&mut file).take(chunk_cap).read_to_end(&mut chunk).await?;
            let n = chunk.len() as u64;
            if n == 0 && offset < total_bytes {
                return Err(ClientError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "local file shrunk during upload: read 0 bytes at offset {offset} of {total_bytes}"
                    ),
                )));
            }

            let req = self
                .auth(self.http.put(format!("{}/v1/files/chunk", self.base_url)))
                .query(&[
                    ("path", remote_path.as_str()),
                    ("offset", &offset.to_string()),
                    ("total_size", &total_bytes.to_string()),
                ])
                .body(chunk);

            let resp: ChunkUploadResponse = send_json(req).await?;
            offset = resp.partial_size;

            let done = resp.completed || offset >= total_bytes;
            if let Some(progress) = tracker.update(n, offset, done) {
                on_progress(progress);
            }

            if done {
                if let Some(remote_hash) = resp.blake3 {
                    let expected = local_hash.get().await?;
                    if remote_hash != expected {
                        return Err(ClientError::ChecksumMismatch {
                            expected,
                            actual: remote_hash,
                        });
                    }
                }
                break;
            }
        }

        Ok(TransferSummary {
            remote_path,
            total_bytes,
            resumed_bytes,
            uploaded_bytes: total_bytes.saturating_sub(resumed_bytes),
            elapsed: started.elapsed(),
            already_existed: false,
        })
    }

    /// Downloads `remote_path` from `--files-dir` to `local_path` with Range resume.
    ///
    /// Data is staged in `<local_path>.part` and renamed into place only after the
    /// size and BLAKE3 digest check out. Skips the transfer if `local_path` already matches.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Api`] (404) if the remote file doesn't exist,
    /// [`ClientError::Io`] on local I/O failure or a truncated response,
    /// [`ClientError::ChecksumMismatch`] if the digest differs, and transport errors.
    pub async fn download_file<F>(
        &self,
        remote_path: &str,
        local_path: &Path,
        mut on_progress: F,
    ) -> Result<TransferSummary, ClientError>
    where
        F: FnMut(TransferProgress),
    {
        let remote_path = normalize_remote_path(remote_path);
        let started = Instant::now();
        let status = self.file_status(&remote_path).await?;
        let total_bytes = status.final_size.ok_or_else(|| ClientError::Api {
            status: StatusCode::NOT_FOUND.as_u16(),
            message: format!("remote file '{remote_path}' does not exist"),
        })?;

        if local_matches(local_path, total_bytes, status.blake3.as_deref()).await? {
            return Ok(TransferSummary {
                remote_path,
                total_bytes,
                resumed_bytes: total_bytes,
                uploaded_bytes: 0,
                elapsed: started.elapsed(),
                already_existed: true,
            });
        }

        let part_path = {
            let mut os = local_path.as_os_str().to_os_string();
            os.push(".part");
            PathBuf::from(os)
        };
        if let Some(parent) = part_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).await?;
        }

        let mut offset = fs::metadata(&part_path).await.map(|m| m.len()).unwrap_or(0);
        if offset > total_bytes {
            offset = 0;
        }
        let mut resumed_bytes = offset;

        let mut out_file = OpenOptions::new()
            .create(true)
            .truncate(false) // keep existing bytes: they are the resume point
            .write(true)
            .read(true)
            .open(&part_path)
            .await?;
        out_file.set_len(offset).await?;
        out_file.seek(SeekFrom::Start(offset)).await?;

        let mut req = self
            .auth(
                self.http
                    .get(format!("{}/v1/files/download", self.base_url)),
            )
            .query(&[("path", remote_path.as_str())]);
        if offset > 0 && offset < total_bytes {
            req = req.header(header::RANGE, format!("bytes={offset}-"));
        }

        let mut res = req.send().await?;
        if !res.status().is_success() {
            return Err(extract_api_error(res).await);
        }
        if res.status() == StatusCode::OK && offset > 0 {
            offset = 0;
            resumed_bytes = 0;
            out_file.set_len(0).await?;
            out_file.seek(SeekFrom::Start(0)).await?;
        }

        let mut tracker = ProgressTracker::new(total_bytes, resumed_bytes);
        on_progress(tracker.initial());

        while let Some(chunk) = res.chunk().await? {
            out_file.write_all(&chunk).await?;
            offset += chunk.len() as u64;

            if let Some(progress) =
                tracker.update(chunk.len() as u64, offset, offset >= total_bytes)
            {
                on_progress(progress);
            }
        }

        out_file.flush().await?;
        drop(out_file);

        // A clean early EOF must not be promoted to the final file (matters when the
        // server sent no blake3). The .part stays so the next call resumes from here.
        if offset != total_bytes {
            return Err(ClientError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("download ended at {offset} of {total_bytes} bytes"),
            )));
        }

        if let Some(expected) = status.blake3 {
            let actual = compute_blake3_blocking(&part_path).await?;
            if actual != expected {
                let _ = fs::remove_file(&part_path).await;
                return Err(ClientError::ChecksumMismatch { expected, actual });
            }
        }
        fs::rename(&part_path, local_path).await?;

        Ok(TransferSummary {
            remote_path,
            total_bytes,
            resumed_bytes,
            uploaded_bytes: total_bytes.saturating_sub(resumed_bytes),
            elapsed: started.elapsed(),
            already_existed: false,
        })
    }

    /// Submits `task` as an async job and polls every `poll_interval` until it finishes.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::JobFailed`] if the job fails or is cancelled, plus any
    /// transport or API error from submitting or polling.
    pub async fn run_and_wait(
        &self,
        task: TaskKind,
        poll_interval: Duration,
    ) -> Result<JobRecord, ClientError> {
        let initial = self.submit(task).await?;
        let id = initial.id;
        loop {
            let record = self.job_status(id).await?;
            match record.status {
                JobStatus::Succeeded => return Ok(record),
                JobStatus::Failed | JobStatus::Cancelled => {
                    return Err(ClientError::JobFailed(
                        id,
                        record.error.unwrap_or_else(|| "unknown error".into()),
                    ));
                }
                JobStatus::Queued | JobStatus::Running => {
                    tokio::time::sleep(poll_interval).await;
                }
            }
        }
    }

    fn auth(&self, req: RequestBuilder) -> RequestBuilder {
        match &self.token {
            Some(token) => req.bearer_auth(token),
            None => req,
        }
    }
}

/// Broadcasts/multicasts a discovery probe and collects every node that replies within `wait`.
///
/// # Errors
///
/// Returns [`ClientError::Io`] if the UDP socket can't be bound or read.
pub async fn discover(
    discovery_port: u16,
    wait: Duration,
) -> Result<Vec<DiscoveredNode>, ClientError> {
    discover_nodes(discovery_port, wait, false).await
}

/// Returns the first offload server that replies within `wait`.
///
/// # Errors
///
/// Returns [`ClientError::NoNodesDiscovered`] if none replied, or [`ClientError::Io`]
/// on socket failure.
pub async fn discover_first(
    discovery_port: u16,
    wait: Duration,
) -> Result<DiscoveredNode, ClientError> {
    let mut nodes = discover_nodes(discovery_port, wait, true).await?;
    nodes.pop().ok_or(ClientError::NoNodesDiscovered)
}

/// Beacons are small JSON objects (~250 bytes); 2 KiB leaves room without fragmenting.
const BEACON_BUF_BYTES: usize = 2048;

async fn discover_nodes(
    discovery_port: u16,
    wait: Duration,
    stop_on_first: bool,
) -> Result<Vec<DiscoveredNode>, ClientError> {
    let socket =
        UdpSocket::bind(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0))).await?;
    socket.set_broadcast(true)?;

    let probe_bytes = serde_json::to_vec(&DiscoveryProbe::new()).expect("probe serialize");
    send_discovery_probes(&socket, discovery_port, &probe_bytes).await;

    let deadline = Instant::now() + wait;
    let mut seen_urls = HashSet::new();
    let mut nodes = Vec::new();
    let mut buf = [0_u8; BEACON_BUF_BYTES];

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }

        match tokio::time::timeout(remaining, socket.recv_from(&mut buf)).await {
            Ok(Ok((len, peer))) => {
                if let Ok(beacon) = serde_json::from_slice::<DiscoveryBeacon>(&buf[..len])
                    && beacon.magic == DiscoveryBeacon::MAGIC
                {
                    let url = format!("http://{}:{}", peer.ip(), beacon.http_port);
                    if seen_urls.insert(url.clone()) {
                        nodes.push(DiscoveredNode {
                            url,
                            name: beacon.name,
                            version: beacon.version,
                            os: beacon.os,
                            arch: beacon.arch,
                            logical_cpus: beacon.logical_cpus,
                            auth_required: beacon.auth_required,
                        });
                        if stop_on_first {
                            break;
                        }
                    }
                }
            }
            Ok(Err(err)) if is_transient_udp_recv_error(&err) => {}
            Ok(Err(err)) => return Err(ClientError::Io(err)),
            Err(_) => break,
        }
    }

    Ok(nodes)
}

/// Sends the discovery probe across every non-loopback IPv4 interface (subnet broadcast +
/// `IP_MULTICAST_IF`), falling back to global broadcast/multicast. On Windows in particular,
/// a single `255.255.255.255` or multicast send on `0.0.0.0` only leaves the lowest-metric
/// adapter (often WSL2/Hyper-V/VPN rather than Wi-Fi).
async fn send_discovery_probes(socket: &UdpSocket, discovery_port: u16, probe_bytes: &[u8]) {
    let mut bcast_ips = HashSet::from([Ipv4Addr::BROADCAST]);
    let mut iface_ips = Vec::new();

    if let Ok(ifaces) = if_addrs::get_if_addrs() {
        for iface in ifaces {
            if !iface.is_loopback()
                && let IfAddr::V4(v4) = iface.addr
            {
                let bcast = v4
                    .broadcast
                    .unwrap_or_else(|| subnet_broadcast_v4(v4.ip, v4.netmask));
                bcast_ips.insert(bcast);
                iface_ips.push(v4.ip);
            }
        }
    }

    for ip in bcast_ips {
        let addr = SocketAddr::V4(SocketAddrV4::new(ip, discovery_port));
        let _ = socket.send_to(probe_bytes, addr).await;
    }

    if let Ok(mcast_ip) = DISCOVERY_MULTICAST_ADDR.parse::<Ipv4Addr>() {
        let mcast_addr = SocketAddr::V4(SocketAddrV4::new(mcast_ip, discovery_port));
        if iface_ips.is_empty() {
            let _ = socket.send_to(probe_bytes, mcast_addr).await;
        } else {
            let sock_ref = SockRef::from(socket);
            for ip in iface_ips {
                let _ = sock_ref.set_multicast_if_v4(&ip);
                let _ = socket.send_to(probe_bytes, mcast_addr).await;
            }
        }
    }
}

fn subnet_broadcast_v4(ip: Ipv4Addr, netmask: Ipv4Addr) -> Ipv4Addr {
    Ipv4Addr::from(u32::from(ip) | !u32::from(netmask))
}

fn is_transient_udp_recv_error(err: &io::Error) -> bool {
    // Winsock surfaces ICMP Port Unreachable from an earlier UDP send as WSAECONNRESET (10054)
    // on the next recv_from(); keep listening instead of aborting discovery.
    err.kind() == io::ErrorKind::ConnectionReset
}

fn normalize_remote_path(remote_path: &str) -> String {
    remote_path.replace('\\', "/")
}

/// Progress callbacks fire at most this often (10 Hz), enough for a smooth terminal bar.
const PROGRESS_REPORT_INTERVAL: Duration = Duration::from_millis(100);

/// Weight of the newest sample in the speed EMA. 0.3 smooths Wi-Fi jitter while
/// still tracking a real speed change within about a second of updates.
const SPEED_EMA_ALPHA: f64 = 0.3;

struct ProgressTracker {
    total_bytes: u64,
    resumed_bytes: u64,
    ema_speed_bps: f64,
    last_report: Instant,
    pending_bytes: u64,
}

impl ProgressTracker {
    fn new(total_bytes: u64, resumed_bytes: u64) -> Self {
        Self {
            total_bytes,
            resumed_bytes,
            ema_speed_bps: 0.0,
            last_report: Instant::now(),
            pending_bytes: 0,
        }
    }

    fn initial(&self) -> TransferProgress {
        TransferProgress {
            bytes_done: self.resumed_bytes,
            total_bytes: self.total_bytes,
            resumed_bytes: self.resumed_bytes,
            speed_bps: 0.0,
            eta_secs: None,
        }
    }

    fn update(
        &mut self,
        bytes_delta: u64,
        bytes_done: u64,
        force: bool,
    ) -> Option<TransferProgress> {
        self.pending_bytes += bytes_delta;
        let elapsed = self.last_report.elapsed();
        if !force && elapsed < PROGRESS_REPORT_INTERVAL {
            return None;
        }
        let dt = elapsed.as_secs_f64().max(1e-4);
        let inst_bps = (self.pending_bytes as f64) / dt;
        self.ema_speed_bps = if self.ema_speed_bps == 0.0 {
            inst_bps
        } else {
            SPEED_EMA_ALPHA * inst_bps + (1.0 - SPEED_EMA_ALPHA) * self.ema_speed_bps
        };
        self.pending_bytes = 0;
        self.last_report = Instant::now();

        let remaining = self.total_bytes.saturating_sub(bytes_done);
        let eta_secs = if self.ema_speed_bps > 1.0 && remaining > 0 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "positive finite seconds; `as` saturates on overflow"
            )]
            let secs = (remaining as f64 / self.ema_speed_bps).ceil() as u64;
            Some(secs)
        } else {
            Some(0)
        };

        Some(TransferProgress {
            bytes_done,
            total_bytes: self.total_bytes,
            resumed_bytes: self.resumed_bytes,
            speed_bps: self.ema_speed_bps,
            eta_secs,
        })
    }
}

/// Local-file hash started early so it overlaps network round-trips. Dropping it
/// (early return, error, or hash not needed) stops the read instead of finishing the file.
struct BackgroundHash {
    handle: Option<tokio::task::JoinHandle<io::Result<String>>>,
    result: Option<String>,
    cancel: Arc<AtomicBool>,
}

impl BackgroundHash {
    fn spawn(path: PathBuf) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let handle = tokio::task::spawn_blocking(move || {
            offload_protocol::blake3_hex(StopOnFlag(std::fs::File::open(path)?, flag))
        });
        Self {
            handle: Some(handle),
            result: None,
            cancel,
        }
    }

    async fn get(&mut self) -> Result<String, ClientError> {
        if let Some(hash) = &self.result {
            return Ok(hash.clone());
        }
        let handle = self
            .handle
            .take()
            .ok_or_else(|| io::Error::other("local hash already failed"))?;
        let hash = handle.await.map_err(io::Error::other)??;
        self.result = Some(hash.clone());
        Ok(hash)
    }
}

impl Drop for BackgroundHash {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Reader that fails once the flag is set; how a blocking hash gets cancelled.
struct StopOnFlag<R>(R, Arc<AtomicBool>);

impl<R: Read> Read for StopOnFlag<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.1.load(Ordering::Relaxed) {
            return Err(io::Error::other("hash cancelled"));
        }
        self.0.read(buf)
    }
}

/// Whether `path` already holds exactly `size` bytes with BLAKE3 digest `blake3`.
async fn local_matches(path: &Path, size: u64, blake3: Option<&str>) -> io::Result<bool> {
    let Some(expected) = blake3 else {
        return Ok(false);
    };
    match fs::metadata(path).await {
        Ok(meta) if meta.len() == size => Ok(compute_blake3_blocking(path).await? == expected),
        Ok(_) => Ok(false),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

async fn compute_blake3_blocking(path: &Path) -> Result<String, io::Error> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || offload_protocol::blake3_hex(std::fs::File::open(path)?))
        .await
        .map_err(io::Error::other)?
}

async fn send_json<T: serde::de::DeserializeOwned>(req: RequestBuilder) -> Result<T, ClientError> {
    let res = req.send().await?;
    if res.status().is_success() {
        Ok(res.json::<T>().await?)
    } else {
        Err(extract_api_error(res).await)
    }
}

async fn extract_api_error(res: reqwest::Response) -> ClientError {
    let status = res.status();
    let body = res.text().await.unwrap_or_default();
    let message = match serde_json::from_str::<ErrorResponse>(&body) {
        Ok(e) => e.error,
        Err(_) if body.is_empty() => status.to_string(),
        Err(_) => body,
    };
    ClientError::Api {
        status: status.as_u16(),
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_tracker_throttles_and_forces_final_update() {
        let mut tracker = ProgressTracker::new(1000, 0);
        let init = tracker.initial();
        assert_eq!(init.bytes_done, 0);
        assert_eq!(init.total_bytes, 1000);
        assert!(init.eta_secs.is_none());

        // Immediate non-forced update within 100 ms should be throttled.
        assert!(tracker.update(100, 100, false).is_none());
        // Forced final update should emit immediately with accumulated bytes.
        let final_p = tracker.update(900, 1000, true).expect("forced update");
        assert_eq!(final_p.bytes_done, 1000);
        assert_eq!(final_p.eta_secs, Some(0));
        assert!(final_p.speed_bps > 0.0);
    }

    #[test]
    fn subnet_broadcast_computes_directed_broadcast() {
        assert_eq!(
            subnet_broadcast_v4(
                Ipv4Addr::new(192, 168, 1, 42),
                Ipv4Addr::new(255, 255, 255, 0)
            ),
            Ipv4Addr::new(192, 168, 1, 255)
        );
        assert_eq!(
            subnet_broadcast_v4(Ipv4Addr::new(10, 0, 5, 12), Ipv4Addr::new(255, 255, 0, 0)),
            Ipv4Addr::new(10, 0, 255, 255)
        );
    }

    #[test]
    fn transient_udp_recv_error_ignores_connection_reset() {
        let reset = io::Error::from(io::ErrorKind::ConnectionReset);
        let refused = io::Error::from(io::ErrorKind::PermissionDenied);
        assert!(is_transient_udp_recv_error(&reset));
        assert!(!is_transient_udp_recv_error(&refused));
    }

    #[test]
    fn normalize_remote_path_replaces_backslashes() {
        assert_eq!(
            normalize_remote_path(r"models\qwen\model.gguf"),
            "models/qwen/model.gguf"
        );
        assert_eq!(
            normalize_remote_path("models/qwen/model.gguf"),
            "models/qwen/model.gguf"
        );
    }
}
