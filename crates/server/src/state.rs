use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use offload_protocol::{JobRecord, JobStatus, JobStreamEvent, TaskResult};
use tokio::{
    sync::{RwLock, Semaphore, broadcast},
    task::AbortHandle,
};
use uuid::Uuid;

const MAX_RETAINED_JOBS: usize = 200;
const STREAM_CHANNEL_CAPACITY: usize = 256;

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

struct JobEntry {
    record: JobRecord,
    tx: broadcast::Sender<JobStreamEvent>,
    abort_handle: Option<AbortHandle>,
}

#[derive(Default)]
struct JobStore {
    entries: HashMap<Uuid, JobEntry>,
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

    pub async fn create_job(&self) -> (JobRecord, broadcast::Sender<JobStreamEvent>) {
        let id = Uuid::new_v4();
        let record = JobRecord {
            id,
            status: JobStatus::Queued,
            created_at_ms: now_ms(),
            duration_ms: None,
            result: None,
            error: None,
        };
        let (tx, _) = broadcast::channel(STREAM_CHANNEL_CAPACITY);

        let mut store = self.jobs.write().await;
        store.entries.insert(
            id,
            JobEntry {
                record: record.clone(),
                tx: tx.clone(),
                abort_handle: None,
            },
        );
        store.order.push_back(id);
        while store.order.len() > MAX_RETAINED_JOBS {
            if let Some(oldest) = store.order.pop_front() {
                store.entries.remove(&oldest);
            }
        }
        (record, tx)
    }

    pub async fn set_abort_handle(&self, id: Uuid, handle: AbortHandle) {
        let mut store = self.jobs.write().await;
        if let Some(entry) = store.entries.get_mut(&id) {
            if entry.record.status == JobStatus::Cancelled {
                handle.abort();
            } else if !is_terminal(entry.record.status) {
                entry.abort_handle = Some(handle);
            }
        }
    }

    pub async fn mark_running(&self, id: Uuid) {
        let mut store = self.jobs.write().await;
        if let Some(entry) = store.entries.get_mut(&id)
            && entry.record.status == JobStatus::Queued
        {
            entry.record.status = JobStatus::Running;
        }
    }

    pub async fn mark_succeeded(&self, id: Uuid, duration_ms: u64, result: TaskResult) {
        let mut store = self.jobs.write().await;
        if let Some(entry) = store.entries.get_mut(&id) {
            if is_terminal(entry.record.status) {
                return;
            }
            entry.record.status = JobStatus::Succeeded;
            entry.record.duration_ms = Some(duration_ms);
            entry.record.result = Some(result);
            entry.record.error = None;
            entry.abort_handle = None;
            let _ = entry.tx.send(JobStreamEvent::Done {
                status: JobStatus::Succeeded,
                duration_ms: Some(duration_ms),
                error: None,
            });
        }
    }

    pub async fn mark_failed(&self, id: Uuid, duration_ms: u64, error: String) {
        let mut store = self.jobs.write().await;
        if let Some(entry) = store.entries.get_mut(&id) {
            if is_terminal(entry.record.status) {
                return;
            }
            entry.record.status = JobStatus::Failed;
            entry.record.duration_ms = Some(duration_ms);
            entry.record.result = None;
            entry.record.error = Some(error.clone());
            entry.abort_handle = None;
            let _ = entry.tx.send(JobStreamEvent::Done {
                status: JobStatus::Failed,
                duration_ms: Some(duration_ms),
                error: Some(error),
            });
        }
    }

    /// Cancel a queued or running job and abort its background task.
    pub async fn cancel_job(&self, id: Uuid) -> Option<JobRecord> {
        let mut store = self.jobs.write().await;
        let entry = store.entries.get_mut(&id)?;
        match entry.record.status {
            JobStatus::Queued | JobStatus::Running => {
                if let Some(handle) = entry.abort_handle.take() {
                    handle.abort();
                }
                entry.record.status = JobStatus::Cancelled;
                entry.record.error = Some("cancelled by client".to_string());
                let _ = entry.tx.send(JobStreamEvent::Done {
                    status: JobStatus::Cancelled,
                    duration_ms: None,
                    error: entry.record.error.clone(),
                });
            }
            JobStatus::Succeeded | JobStatus::Failed | JobStatus::Cancelled => {}
        }
        Some(entry.record.clone())
    }

    pub async fn get_job(&self, id: Uuid) -> Option<JobRecord> {
        let store = self.jobs.read().await;
        store.entries.get(&id).map(|e| e.record.clone())
    }

    pub async fn subscribe_job(
        &self,
        id: Uuid,
    ) -> Option<(JobRecord, broadcast::Receiver<JobStreamEvent>)> {
        let store = self.jobs.read().await;
        store
            .entries
            .get(&id)
            .map(|e| (e.record.clone(), e.tx.subscribe()))
    }

    pub async fn list_jobs(&self) -> Vec<JobRecord> {
        let store = self.jobs.read().await;
        store
            .order
            .iter()
            .rev()
            .filter_map(|id| store.entries.get(id).map(|e| e.record.clone()))
            .collect()
    }
}

fn is_terminal(status: JobStatus) -> bool {
    matches!(
        status,
        JobStatus::Succeeded | JobStatus::Failed | JobStatus::Cancelled
    )
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mark_succeeded_and_failed_do_not_overwrite_cancelled() {
        let state = AppState::new(ServerConfig {
            auth_token: None,
            max_concurrent_jobs: 1,
            models_dir: None,
            files_dir: PathBuf::from("uploads"),
            llama_bin: "llama-cli".into(),
            allowed_commands: vec![],
            default_timeout_secs: 60,
        });

        let (rec, _) = state.create_job().await;
        let cancelled = state.cancel_job(rec.id).await.unwrap();
        assert_eq!(cancelled.status, JobStatus::Cancelled);

        state
            .mark_succeeded(rec.id, 10, TaskResult::Echo { payload: "ok".into() })
            .await;
        state.mark_failed(rec.id, 10, "err".into()).await;

        let after = state.get_job(rec.id).await.unwrap();
        assert_eq!(after.status, JobStatus::Cancelled);
        assert!(after.result.is_none());
    }
}
