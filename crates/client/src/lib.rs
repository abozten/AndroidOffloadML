use std::{
    collections::HashSet,
    io::SeekFrom,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::Path,
    time::{Duration, Instant},
};

use offload_protocol::{
    ChunkUploadResponse, DISCOVERY_MULTICAST_ADDR, DiscoveredNode, DiscoveryBeacon, DiscoveryProbe,
    ErrorResponse, FileTransferStatus, JobRecord, JobStatus, NodeInfo, RunTaskRequest,
    RunTaskResponse, TaskKind,
};
use reqwest::{Client, RequestBuilder, StatusCode};
use thiserror::Error;
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
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

        let status = self.file_status(remote_path).await?;
        if status.exists && status.final_size == Some(total_bytes) {
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
                JobStatus::Failed => {
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
