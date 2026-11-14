use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use offload_protocol::{JobRecord, JobStatus};
use tokio::sync::{RwLock, Semaphore};
use uuid::Uuid;

const MAX_RETAINED_JOBS: usize = 200;

/// Runtime configuration for the offload server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub auth_token: Option<String>,
    pub max_concurrent_jobs: usize,
    pub models_dir: Option<PathBuf>,
    pub files_dir: PathBuf,
    pub llama_bin: String,
    pub allowed_commands: Vec<String>,
    pub default_timeout_secs: u64,
}

/// Shared application state across HTTP handlers and background job workers.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub semaphore: Arc<Semaphore>,
    pub active_jobs: Arc<AtomicUsize>,
    jobs: Arc<RwLock<JobStore>>,
}

#[derive(Default)]
struct JobStore {
    records: HashMap<Uuid, JobRecord>,
    order: VecDeque<Uuid>,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        let max_jobs = config.max_concurrent_jobs.max(1);
        Self {
            config: Arc::new(config),
            semaphore: Arc::new(Semaphore::new(max_jobs)),
            active_jobs: Arc::new(AtomicUsize::new(0)),
            jobs: Arc::new(RwLock::new(JobStore::default())),
        }
    }

    pub fn active_job_count(&self) -> usize {
        self.active_jobs.load(Ordering::Relaxed)
    }

    pub async fn create_job(&self) -> JobRecord {
        let id = Uuid::new_v4();
        let record = JobRecord {
            id,
            status: JobStatus::Queued,
            created_at_ms: now_ms(),
            duration_ms: None,
            result: None,
            error: None,
        };

        let mut store = self.jobs.write().await;
        store.records.insert(id, record.clone());
        store.order.push_back(id);
        while store.order.len() > MAX_RETAINED_JOBS {
            if let Some(oldest) = store.order.pop_front() {
                store.records.remove(&oldest);
            }
        }
        record
    }

    pub async fn mark_running(&self, id: Uuid) {
        let mut store = self.jobs.write().await;
        if let Some(job) = store.records.get_mut(&id) {
            job.status = JobStatus::Running;
        }
    }

    pub async fn mark_succeeded(
        &self,
        id: Uuid,
        duration_ms: u64,
        result: offload_protocol::TaskResult,
    ) {
        let mut store = self.jobs.write().await;
        if let Some(job) = store.records.get_mut(&id) {
            job.status = JobStatus::Succeeded;
            job.duration_ms = Some(duration_ms);
            job.result = Some(result);
            job.error = None;
        }
    }

    pub async fn mark_failed(&self, id: Uuid, duration_ms: u64, error: String) {
        let mut store = self.jobs.write().await;
        if let Some(job) = store.records.get_mut(&id) {
            job.status = JobStatus::Failed;
            job.duration_ms = Some(duration_ms);
            job.result = None;
            job.error = Some(error);
        }
    }

    pub async fn get_job(&self, id: Uuid) -> Option<JobRecord> {
        let store = self.jobs.read().await;
        store.records.get(&id).cloned()
    }

    pub async fn list_jobs(&self) -> Vec<JobRecord> {
        let store = self.jobs.read().await;
        store
            .order
            .iter()
            .rev()
            .filter_map(|id| store.records.get(id).cloned())
            .collect()
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}
