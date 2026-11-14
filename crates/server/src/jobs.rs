//! Task execution handlers: synchronous `/v1/run` and the async `/v1/jobs` API.

use std::{
    convert::Infallible,
    time::{Duration, Instant},
};

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
};
use futures_util::Stream;
use offload_protocol::{JobRecord, JobStreamEvent, RunTaskRequest, RunTaskResponse};
use tokio::sync::broadcast;
use tracing::error;
use uuid::Uuid;

use crate::{error::ApiError, runner::execute_task, state::AppState};

/// SSE comment interval. Keeps idle streams (e.g. a model loading) alive through
/// NATs and proxies that drop connections silent for ~30 s or more.
const SSE_KEEPALIVE: Duration = Duration::from_secs(15);

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

pub async fn run_sync(
    State(state): State<AppState>,
    Json(req): Json<RunTaskRequest>,
) -> Result<Json<RunTaskResponse>, ApiError> {
    let permit = state.semaphore.acquire().await.map_err(|_closed| {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "worker shutting down")
    })?;

    let started = Instant::now();
    let result = execute_task(req.task, &state.config, None).await;
    let duration_ms = elapsed_ms(started);
    drop(permit);

    Ok(Json(RunTaskResponse {
        job_id: Uuid::new_v4(),
        duration_ms,
        result: result?,
    }))
}

pub async fn submit_job(
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

        worker_state.mark_running(job_id).await;
        let started = Instant::now();
        let outcome = execute_task(req.task, &worker_state.config, Some(stream_tx)).await;
        let duration_ms = elapsed_ms(started);
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

pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<JobRecord>, ApiError> {
    state
        .get_job(id)
        .await
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("job {id} not found")))
}

pub async fn cancel_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<JobRecord>, ApiError> {
    state
        .cancel_job(id)
        .await
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("job {id} not found")))
}

pub async fn stream_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let (_record, rx) = state
        .subscribe_job(id)
        .await
        .ok_or_else(|| ApiError::not_found(format!("job {id} not found")))?;

    // Close right after `Done`, not when the channel closes: pipe-reader tasks of a
    // cancelled/timed-out job can still hold senders (forever, if a grandchild keeps
    // the pipe open). `take_while` can't do this; it needs one more item to stop.
    let stream = futures_util::stream::unfold(Some(rx), |rx| async move {
        let mut rx = rx?;
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let done = matches!(ev, JobStreamEvent::Done { .. });
                    let Ok(json) = serde_json::to_string(&ev) else {
                        continue;
                    };
                    let next = (!done).then_some(rx);
                    return Some((Ok::<_, Infallible>(Event::default().data(json)), next));
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(SSE_KEEPALIVE)))
}

pub async fn list_jobs(State(state): State<AppState>) -> Json<Vec<JobRecord>> {
    Json(state.list_jobs().await)
}
