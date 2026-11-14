use std::{
    collections::HashSet,
    io::{Read, SeekFrom},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::Path,
    time::{Duration, Instant},
};

use offload_protocol::{
    ChunkUploadResponse, DISCOVERY_MULTICAST_ADDR, DiscoveredNode, DiscoveryBeacon, DiscoveryProbe,
    ErrorResponse, FileTransferStatus, JobRecord, JobStatus, JobStreamEvent, ModelEntry, NodeInfo,
    RunTaskRequest, RunTaskResponse, TaskKind,
};
use reqwest::{Client, RequestBuilder, StatusCode, header};
use thiserror::Error;
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    net::UdpSocket,
};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("http transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("no offload servers discovered on LAN")]
    NoNodesDiscovered,
    #[error("integrity check failed: expected blake3 {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("server returned {status}: {message}")]
    Api { status: StatusCode, message: String },
    #[error("job {0} failed: {1}")]
    JobFailed(Uuid, String),
}

/// Live progress snapshot emitted during a resumable file upload.
#[derive(Debug, Clone)]
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
    pub fn new(base_url: impl Into<String>, token: Option<String>) -> Result<Self, ClientError> {
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token: token.filter(|t| !t.is_empty()),
            http,
        })
    }

    pub async fn health(&self) -> Result<(), ClientError> {
        let res = self.http.get(format!("{}/health", self.base_url)).send().await?;
        if res.status().is_success() {
            Ok(())
        } else {
            Err(extract_api_error(res).await)
        }
    }

    pub async fn info(&self) -> Result<NodeInfo, ClientError> {
        let req = self.auth(self.http.get(format!("{}/v1/info", self.base_url)));
        send_json(req).await
    }

    pub async fn list_models(&self) -> Result<Vec<ModelEntry>, ClientError> {
        let req = self.auth(self.http.get(format!("{}/v1/models", self.base_url)));
        send_json(req).await
    }

    pub async fn run(&self, task: TaskKind) -> Result<RunTaskResponse, ClientError> {
        let req = self
            .auth(self.http.post(format!("{}/v1/run", self.base_url)))
            .json(&RunTaskRequest { task });
        send_json(req).await
    }

    pub async fn submit(&self, task: TaskKind) -> Result<JobRecord, ClientError> {
        let req = self
            .auth(self.http.post(format!("{}/v1/jobs", self.base_url)))
            .json(&RunTaskRequest { task });
        send_json(req).await
    }

    pub async fn job_status(&self, id: Uuid) -> Result<JobRecord, ClientError> {
        let req = self.auth(self.http.get(format!("{}/v1/jobs/{id}", self.base_url)));
        send_json(req).await
    }

    pub async fn list_jobs(&self) -> Result<Vec<JobRecord>, ClientError> {
        let req = self.auth(self.http.get(format!("{}/v1/jobs", self.base_url)));
        send_json(req).await
    }

    pub async fn cancel_job(&self, id: Uuid) -> Result<JobRecord, ClientError> {
        let req = self.auth(self.http.delete(format!("{}/v1/jobs/{id}", self.base_url)));
        send_json(req).await
    }

    /// Stream live SSE output and completion events for an async job.
    pub async fn stream_job<F>(&self, id: Uuid, mut on_event: F) -> Result<(), ClientError>
    where
        F: FnMut(JobStreamEvent),
    {
        let mut res = self
            .auth(self.http.get(format!("{}/v1/jobs/{id}/stream", self.base_url)))
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

    /// Query server-side upload status for a remote relative path.
    pub async fn file_status(&self, remote_path: &str) -> Result<FileTransferStatus, ClientError> {
        let req = self
            .auth(self.http.get(format!("{}/v1/files/status", self.base_url)))
            .query(&[("path", remote_path)]);
        send_json(req).await
    }

    /// Upload a local file in chunks with automatic resume and live speed/ETA callbacks.
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
        let mut file = File::open(local_path).await?;
        let total_bytes = file.metadata().await?.len();
        let started = Instant::now();
        let hash_path = local_path.to_path_buf();
        let mut hash_handle = Some(tokio::task::spawn_blocking(move || {
            compute_blake3_sync(&hash_path)
        }));
        let mut local_hash: Option<String> = None;

        let status = self.file_status(remote_path).await?;
        if status.exists
            && status.final_size == Some(total_bytes)
            && let Some(remote_hash) = status.blake3.as_deref()
        {
            let computed = hash_handle
                .take()
                .expect("hash handle present")
                .await
                .map_err(|e| std::io::Error::other(e.to_string()))??;
            if remote_hash == computed {
                return Ok(TransferSummary {
                    remote_path: remote_path.to_string(),
                    total_bytes,
                    resumed_bytes: total_bytes,
                    uploaded_bytes: 0,
                    elapsed: started.elapsed(),
                    already_existed: true,
                });
            }
            local_hash = Some(computed);
        }

        let mut offset = if status.partial_size <= total_bytes {
            status.partial_size
        } else {
            0
        };
        let resumed_bytes = offset;
        file.seek(SeekFrom::Start(offset)).await?;

        let chunk_cap = chunk_size.max(64 * 1024);
        let mut buf = vec![0_u8; chunk_cap];
        let mut tracker = ProgressTracker::new(total_bytes, resumed_bytes);

        on_progress(tracker.initial());

        loop {
            let n = read_full_chunk(&mut file, &mut buf).await?;
            if n == 0 && offset < total_bytes {
                return Err(ClientError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "local file shrunk during upload: read 0 bytes at offset {offset} of {total_bytes}"
                    ),
                )));
            }

            let req = self
                .auth(self.http.put(format!("{}/v1/files/chunk", self.base_url)))
                .query(&[
                    ("path", remote_path),
                    ("offset", &offset.to_string()),
                    ("total_size", &total_bytes.to_string()),
                ])
                .body(buf[..n].to_vec());

            let resp: ChunkUploadResponse = send_json(req).await?;
            offset = resp.partial_size;

            let done = resp.completed || offset >= total_bytes;
            if let Some(progress) = tracker.update(n as u64, offset, done) {
                on_progress(progress);
            }

            if done {
                if let Some(remote_hash) = resp.blake3 {
                    let expected = match local_hash.take() {
                        Some(h) => h,
                        None => hash_handle
                            .take()
                            .expect("hash handle present")
                            .await
                            .map_err(|e| std::io::Error::other(e.to_string()))??,
                    };
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
            remote_path: remote_path.to_string(),
            total_bytes,
            resumed_bytes,
            uploaded_bytes: total_bytes.saturating_sub(resumed_bytes),
            elapsed: started.elapsed(),
            already_existed: false,
        })
    }

    /// Download a file from `--files-dir` with HTTP Range resume, speed/ETA, and BLAKE3 verification.
    pub async fn download_file<F>(
        &self,
        remote_path: &str,
        local_path: &Path,
        mut on_progress: F,
    ) -> Result<TransferSummary, ClientError>
    where
        F: FnMut(TransferProgress),
    {
        let started = Instant::now();
        let status = self.file_status(remote_path).await?;
        let total_bytes = status.final_size.ok_or_else(|| ClientError::Api {
            status: StatusCode::NOT_FOUND,
            message: format!("remote file '{remote_path}' does not exist"),
        })?;

        if local_path.exists()
            && fs::metadata(local_path).await?.len() == total_bytes
            && let Some(expected) = &status.blake3
        {
            let local_hash = compute_blake3_blocking(local_path).await?;
            if &local_hash == expected {
                return Ok(TransferSummary {
                    remote_path: remote_path.to_string(),
                    total_bytes,
                    resumed_bytes: total_bytes,
                    uploaded_bytes: 0,
                    elapsed: started.elapsed(),
                    already_existed: true,
                });
            }
        }

        let part_path = {
            let mut os = local_path.as_os_str().to_os_string();
            os.push(".part");
            std::path::PathBuf::from(os)
        };
        if let Some(parent) = part_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).await?;
        }

        let mut offset = fs::metadata(&part_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        if offset > total_bytes {
            offset = 0;
        }
        let mut resumed_bytes = offset;

        let mut out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&part_path)
            .await?;
        out_file.set_len(offset).await?;
        out_file.seek(SeekFrom::Start(offset)).await?;

        let mut req = self
            .auth(self.http.get(format!("{}/v1/files/download", self.base_url)))
            .query(&[("path", remote_path)]);
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

            if let Some(progress) = tracker.update(chunk.len() as u64, offset, offset >= total_bytes)
            {
                on_progress(progress);
            }
        }

        out_file.flush().await?;
        drop(out_file);

        if let Some(expected) = status.blake3 {
            let actual = compute_blake3_blocking(&part_path).await?;
            if actual != expected {
                let _ = fs::remove_file(&part_path).await;
                return Err(ClientError::ChecksumMismatch { expected, actual });
            }
        }
        fs::rename(&part_path, local_path).await?;

        Ok(TransferSummary {
            remote_path: remote_path.to_string(),
            total_bytes,
            resumed_bytes,
            uploaded_bytes: total_bytes.saturating_sub(resumed_bytes),
            elapsed: started.elapsed(),
            already_existed: false,
        })
    }

    /// Submit a task asynchronously and poll until it reaches a terminal state.
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

    /// Broadcast/multicast a discovery probe on the LAN and collect all responding nodes.
    pub async fn discover(
        discovery_port: u16,
        wait: Duration,
    ) -> Result<Vec<DiscoveredNode>, ClientError> {
        discover_nodes(discovery_port, wait, false).await
    }

    /// Return the first offload server that responds on the LAN.
    pub async fn discover_first(
        discovery_port: u16,
        wait: Duration,
    ) -> Result<DiscoveredNode, ClientError> {
        let mut nodes = discover_nodes(discovery_port, wait, true).await?;
        nodes.pop().ok_or(ClientError::NoNodesDiscovered)
    }
}

async fn discover_nodes(
    discovery_port: u16,
    wait: Duration,
    stop_on_first: bool,
) -> Result<Vec<DiscoveredNode>, ClientError> {
    let socket = UdpSocket::bind(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0))).await?;
    socket.set_broadcast(true)?;

    let probe_bytes = serde_json::to_vec(&DiscoveryProbe::new()).expect("probe serialize");
    let bcast = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::BROADCAST, discovery_port));
    let mcast = DISCOVERY_MULTICAST_ADDR
        .parse::<Ipv4Addr>()
        .ok()
        .map(|ip| SocketAddr::V4(SocketAddrV4::new(ip, discovery_port)));

    let _ = socket.send_to(&probe_bytes, bcast).await;
    if let Some(mcast_addr) = mcast {
        let _ = socket.send_to(&probe_bytes, mcast_addr).await;
    }

    let deadline = Instant::now() + wait;
    let mut seen_urls = HashSet::new();
    let mut nodes = Vec::new();
    let mut buf = [0_u8; 2048];

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
            Ok(Err(err)) => return Err(ClientError::Io(err)),
            Err(_) => break,
        }
    }

    Ok(nodes)
}

async fn read_full_chunk(file: &mut File, buf: &mut [u8]) -> Result<usize, std::io::Error> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = file.read(&mut buf[filled..]).await?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    Ok(filled)
}

const PROGRESS_REPORT_INTERVAL: Duration = Duration::from_millis(100);

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

    fn update(&mut self, bytes_delta: u64, bytes_done: u64, force: bool) -> Option<TransferProgress> {
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
            0.3 * inst_bps + 0.7 * self.ema_speed_bps
        };
        self.pending_bytes = 0;
        self.last_report = Instant::now();

        let remaining = self.total_bytes.saturating_sub(bytes_done);
        let eta_secs = if self.ema_speed_bps > 1.0 && remaining > 0 {
            Some((remaining as f64 / self.ema_speed_bps).ceil() as u64)
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

fn compute_blake3_sync(path: &Path) -> Result<String, std::io::Error> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0_u8; 256 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

async fn compute_blake3_blocking(path: &Path) -> Result<String, std::io::Error> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || compute_blake3_sync(&path))
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?
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
    let message = serde_json::from_str::<ErrorResponse>(&body)
        .map(|e| e.error)
        .unwrap_or_else(|_| {
            if body.is_empty() {
                status.to_string()
            } else {
                body
            }
        });
    ClientError::Api { status, message }
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
}
