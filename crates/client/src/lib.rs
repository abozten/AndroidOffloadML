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
            .timeout(Duration::from_secs(600))
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

        let mut buf = String::new();
        while let Some(chunk) = res.chunk().await? {
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim_end_matches('\r').to_string();
                buf.drain(..=pos);
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
        let local_hash = compute_blake3_blocking(local_path).await?;

        let status = self.file_status(remote_path).await?;
        if status.exists
            && status.final_size == Some(total_bytes)
            && status.blake3.as_deref() == Some(local_hash.as_str())
        {
            return Ok(TransferSummary {
                remote_path: remote_path.to_string(),
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

        let chunk_cap = chunk_size.max(64 * 1024);
        let mut buf = vec![0_u8; chunk_cap];
        let mut ema_speed_bps = 0.0_f64;

        on_progress(TransferProgress {
            bytes_done: offset,
            total_bytes,
            resumed_bytes,
            speed_bps: 0.0,
            eta_secs: None,
        });

        loop {
            let chunk_start = Instant::now();
            let n = read_full_chunk(&mut file, &mut buf).await?;
            if n == 0 && total_bytes > 0 && offset >= total_bytes {
                break;
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

            let dt = chunk_start.elapsed().as_secs_f64().max(1e-4);
            let inst_bps = (n as f64) / dt;
            ema_speed_bps = if ema_speed_bps == 0.0 {
                inst_bps
            } else {
                0.3 * inst_bps + 0.7 * ema_speed_bps
            };

            let remaining = total_bytes.saturating_sub(offset);
            let eta_secs = if ema_speed_bps > 1.0 && remaining > 0 {
                Some((remaining as f64 / ema_speed_bps).ceil() as u64)
            } else {
                Some(0)
            };

            on_progress(TransferProgress {
                bytes_done: offset,
                total_bytes,
                resumed_bytes,
                speed_bps: ema_speed_bps,
                eta_secs,
            });

            if resp.completed || offset >= total_bytes {
                if let Some(remote_hash) = resp.blake3
                    && remote_hash != local_hash
                {
                    return Err(ClientError::ChecksumMismatch {
                        expected: local_hash,
                        actual: remote_hash,
                    });
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
        let resumed_bytes = offset;

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

        let mut ema_speed_bps = 0.0_f64;
        let mut last_tick = Instant::now();
        on_progress(TransferProgress {
            bytes_done: offset,
            total_bytes,
            resumed_bytes,
            speed_bps: 0.0,
            eta_secs: None,
        });

        while let Some(chunk) = res.chunk().await? {
            out_file.write_all(&chunk).await?;
            offset += chunk.len() as u64;

            let dt = last_tick.elapsed().as_secs_f64().max(1e-4);
            let inst_bps = (chunk.len() as f64) / dt;
            ema_speed_bps = if ema_speed_bps == 0.0 {
                inst_bps
            } else {
                0.3 * inst_bps + 0.7 * ema_speed_bps
            };
            last_tick = Instant::now();

            let remaining = total_bytes.saturating_sub(offset);
            let eta_secs = if ema_speed_bps > 1.0 && remaining > 0 {
                Some((remaining as f64 / ema_speed_bps).ceil() as u64)
            } else {
                Some(0)
            };
            on_progress(TransferProgress {
                bytes_done: offset,
                total_bytes,
                resumed_bytes,
                speed_bps: ema_speed_bps,
                eta_secs,
            });
        }

        out_file.flush().await?;
        drop(out_file);

        if let Some(expected) = status.blake3 {
            let actual = compute_blake3_blocking(&part_path).await?;
            if actual != expected {
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

async fn compute_blake3_blocking(path: &Path) -> Result<String, std::io::Error> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> std::io::Result<String> {
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
    })
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
