use std::{
    io::{Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};

use axum::{
    Json,
    body::{Body, Bytes},
    extract::{Query, State},
    http::{HeaderMap, StatusCode, header},
    response::Response,
};
use offload_protocol::{ChunkUploadResponse, FileTransferStatus};
use serde::Deserialize;
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    sync::mpsc,
};
use tokio_stream::wrappers::ReceiverStream;

use crate::{ApiError, state::AppState};

#[derive(Debug, Deserialize)]
pub struct FileStatusQuery {
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct ChunkUploadQuery {
    pub path: String,
    pub offset: u64,
    pub total_size: u64,
}

#[derive(Debug, Deserialize)]
pub struct DownloadQuery {
    pub path: String,
}

/// Resolve a relative upload/download path strictly within `base_dir`.
pub fn resolve_upload_path(rel_path: &str, base_dir: &Path) -> Result<PathBuf, ApiError> {
    let trimmed = rel_path.trim();
    if trimmed.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "file path must not be empty",
        ));
    }

    let rel = Path::new(trimmed);
    if rel.is_absolute() {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "absolute file paths are disallowed",
        ));
    }

    let mut clean = PathBuf::new();
    for comp in rel.components() {
        match comp {
            Component::Normal(seg) => clean.push(seg),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ApiError::new(
                    StatusCode::FORBIDDEN,
                    "file path must stay within --files-dir",
                ));
            }
        }
    }

    if clean.as_os_str().is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "file path must not be empty",
        ));
    }

    Ok(base_dir.join(clean))
}

fn part_path(final_path: &Path) -> PathBuf {
    let mut os = final_path.as_os_str().to_os_string();
    os.push(".part");
    PathBuf::from(os)
}

/// Hash through an already-open handle. Pair it with `file.metadata()` from the same
/// handle so the digest and the cache stamp describe one inode, even if the path is
/// renamed over concurrently.
async fn hash_handle(file: File) -> Result<String, ApiError> {
    let mut file = file.into_std().await;
    tokio::task::spawn_blocking(move || {
        file.rewind()?;
        offload_protocol::blake3_hex(&file)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

pub async fn file_status(
    State(state): State<AppState>,
    Query(q): Query<FileStatusQuery>,
) -> Result<Json<FileTransferStatus>, ApiError> {
    let final_path = resolve_upload_path(&q.path, &state.config.files_dir)?;
    let staging_path = part_path(&final_path);

    let part_meta = fs::metadata(&staging_path).await.ok();
    let (final_size, blake3) = match File::open(&final_path).await {
        Ok(file) => {
            let meta = file
                .metadata()
                .await
                .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            let hash = match state.get_file_hash(&final_path, &meta).await {
                Some(cached) => cached,
                None => {
                    let hash = hash_handle(file).await?;
                    state.insert_file_hash(final_path, &meta, hash.clone()).await;
                    hash
                }
            };
            (Some(meta.len()), Some(hash))
        }
        Err(_) => (None, None),
    };

    Ok(Json(FileTransferStatus {
        path: q.path,
        exists: final_size.is_some(),
        final_size,
        blake3,
        partial_size: part_meta.map(|m| m.len()).unwrap_or(0),
    }))
}

pub async fn upload_chunk(
    State(state): State<AppState>,
    Query(q): Query<ChunkUploadQuery>,
    body: Bytes,
) -> Result<Json<ChunkUploadResponse>, ApiError> {
    let final_path = resolve_upload_path(&q.path, &state.config.files_dir)?;
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    let staging_path = part_path(&final_path);
    let current_part_size = fs::metadata(&staging_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);

    if q.offset > current_part_size {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            format!(
                "offset {} exceeds current partial size {}",
                q.offset, current_part_size
            ),
        ));
    }

    let new_end = q
        .offset
        .checked_add(body.len() as u64)
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "chunk offset overflow"))?;
    if new_end > q.total_size {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("chunk end {new_end} exceeds total_size {}", q.total_size),
        ));
    }

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open(&staging_path)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    file.seek(SeekFrom::Start(q.offset))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    file.write_all(&body)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    file.set_len(new_end)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    file.flush()
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let completed = new_end == q.total_size;
    let blake3 = if completed {
        fs::rename(&staging_path, &final_path)
            .await
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        // Stat and hash our own handle (post-rename, so ctime is final), never the path:
        // another upload may already have renamed its file over `final_path`.
        let meta = file
            .metadata()
            .await
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let hash = hash_handle(file).await?;
        state.insert_file_hash(final_path, &meta, hash.clone()).await;
        Some(hash)
    } else {
        None
    };

    Ok(Json(ChunkUploadResponse {
        path: q.path,
        partial_size: new_end,
        completed,
        blake3,
    }))
}

pub async fn download_file(
    State(state): State<AppState>,
    Query(q): Query<DownloadQuery>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let final_path = resolve_upload_path(&q.path, &state.config.files_dir)?;
    let meta = fs::metadata(&final_path)
        .await
        .map_err(|_| ApiError::new(StatusCode::NOT_FOUND, format!("file '{}' not found", q.path)))?;
    let total_size = meta.len();

    let start = parse_range_start(&headers, total_size)?;
    let mut file = File::open(&final_path)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    file.seek(SeekFrom::Start(start))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let remaining = total_size.saturating_sub(start);
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);

    tokio::spawn(async move {
        let mut buf = vec![0_u8; 256 * 1024];
        loop {
            match file.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    if tx.send(Ok(Bytes::copy_from_slice(&buf[..n]))).await.is_err() {
                        break;
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                    break;
                }
            }
        }
    });

    let body = Body::from_stream(ReceiverStream::new(rx));
    let mut builder = Response::builder()
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, remaining.to_string())
        .header(header::CONTENT_TYPE, "application/octet-stream");

    if start > 0 && total_size > 0 {
        builder = builder
            .status(StatusCode::PARTIAL_CONTENT)
            .header(
                header::CONTENT_RANGE,
                format!("bytes {start}-{}/{total_size}", total_size - 1),
            );
    } else {
        builder = builder.status(StatusCode::OK);
    }

    builder
        .body(body)
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

fn parse_range_start(headers: &HeaderMap, total_size: u64) -> Result<u64, ApiError> {
    let Some(val) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) else {
        return Ok(0);
    };
    let Some(spec) = val.strip_prefix("bytes=") else {
        return Err(ApiError::new(
            StatusCode::RANGE_NOT_SATISFIABLE,
            "unsupported Range header",
        ));
    };
    let start_str = spec.split('-').next().unwrap_or("");
    let start: u64 = start_str.parse().map_err(|_| {
        ApiError::new(StatusCode::RANGE_NOT_SATISFIABLE, "invalid Range start")
    })?;
    if total_size > 0 && start >= total_size {
        return Err(ApiError::new(
            StatusCode::RANGE_NOT_SATISFIABLE,
            "Range start beyond end of file",
        ));
    }
    Ok(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_path_blocks_traversal() {
        let base = Path::new("/tmp/offload-files");
        assert!(resolve_upload_path("../evil.bin", base).is_err());
        assert!(resolve_upload_path("/etc/passwd", base).is_err());
        assert_eq!(
            resolve_upload_path("models/qwen.gguf", base).unwrap(),
            base.join("models/qwen.gguf")
        );
    }
}
