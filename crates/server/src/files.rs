use std::{
    io::SeekFrom,
    path::{Component, Path, PathBuf},
};

use axum::{
    Json,
    body::Bytes,
    extract::{Query, State},
    http::StatusCode,
};
use offload_protocol::{ChunkUploadResponse, FileTransferStatus};
use serde::Deserialize;
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncSeekExt, AsyncWriteExt},
};

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

/// Resolve a relative upload path strictly within `base_dir`.
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
            "absolute upload paths are disallowed",
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
                    "upload path must stay within --files-dir",
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

pub async fn file_status(
    State(state): State<AppState>,
    Query(q): Query<FileStatusQuery>,
) -> Result<Json<FileTransferStatus>, ApiError> {
    let final_path = resolve_upload_path(&q.path, &state.config.files_dir)?;
    let staging_path = part_path(&final_path);

    let final_meta = fs::metadata(&final_path).await.ok();
    let part_meta = fs::metadata(&staging_path).await.ok();

    Ok(Json(FileTransferStatus {
        path: q.path,
        exists: final_meta.is_some(),
        final_size: final_meta.map(|m| m.len()),
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

    let new_end = q.offset + body.len() as u64;
    if new_end > q.total_size {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("chunk end {new_end} exceeds total_size {}", q.total_size),
        ));
    }

    file.set_len(new_end)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    file.flush()
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    drop(file);

    let completed = new_end == q.total_size;
    if completed {
        fs::rename(&staging_path, &final_path)
            .await
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    Ok(Json(ChunkUploadResponse {
        path: q.path,
        partial_size: new_end,
        completed,
    }))
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
