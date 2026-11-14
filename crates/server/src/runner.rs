use std::{
    cmp::Ordering,
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use offload_protocol::{JobStreamEvent, SimilarityMatch, TaskKind, TaskResult};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::broadcast,
    time::timeout,
};

use crate::state::ServerConfig;

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("execution timed out after {0}s")]
    Timeout(u64),
    #[error("process execution failed: {0}")]
    Process(String),
    #[error("internal task failure: {0}")]
    Internal(String),
}

/// Execute a single ML or compute task using the configured worker environment.
pub async fn execute_task(
    task: TaskKind,
    config: &ServerConfig,
    stream_tx: Option<broadcast::Sender<JobStreamEvent>>,
) -> Result<TaskResult, RunnerError> {
    match task {
        TaskKind::Echo { payload } => Ok(TaskResult::Echo { payload }),

        TaskKind::MatMul {
            rows_a,
            cols_a,
            cols_b,
            a,
            b,
        } => tokio::task::spawn_blocking(move || run_matmul(rows_a, cols_a, cols_b, &a, &b))
            .await
            .map_err(|e| RunnerError::Internal(e.to_string()))?,

        TaskKind::CosineTopK {
            query,
            corpus,
            top_k,
        } => tokio::task::spawn_blocking(move || run_cosine_top_k(&query, &corpus, top_k))
            .await
            .map_err(|e| RunnerError::Internal(e.to_string()))?,

        TaskKind::LlamaCompletion {
            model,
            prompt,
            max_tokens,
            temperature,
            extra_args,
        } => {
            let model_path = resolve_model_path(&model, config.models_dir.as_deref())?;
            let mut args = vec![
                "-m".to_string(),
                model_path.to_string_lossy().into_owned(),
                "-p".to_string(),
                prompt,
                "-n".to_string(),
                max_tokens.to_string(),
            ];
            if let Some(temp) = temperature {
                args.push("--temp".to_string());
                args.push(temp.to_string());
            }
            args.extend(extra_args);

            run_subprocess(
                &config.llama_bin,
                &args,
                None,
                Duration::from_secs(config.default_timeout_secs),
                stream_tx,
            )
            .await
        }

        TaskKind::Command {
            program,
            args,
            stdin,
            timeout_secs,
        } => {
            if !is_command_allowed(&program, &config.allowed_commands) {
                return Err(RunnerError::Forbidden(format!(
                    "command '{program}' is not in --allow-cmd"
                )));
            }
            let secs = timeout_secs.unwrap_or(config.default_timeout_secs).max(1);
            run_subprocess(
                &program,
                &args,
                stdin.as_deref(),
                Duration::from_secs(secs),
                stream_tx,
            )
            .await
        }
    }
}

fn run_matmul(
    rows_a: usize,
    cols_a: usize,
    cols_b: usize,
    a: &[f32],
    b: &[f32],
) -> Result<TaskResult, RunnerError> {
    if rows_a == 0 || cols_a == 0 || cols_b == 0 {
        return Err(RunnerError::InvalidInput(
            "matrix dimensions must be non-zero".into(),
        ));
    }
    let expected_a = rows_a
        .checked_mul(cols_a)
        .ok_or_else(|| RunnerError::InvalidInput("matrix A dimensions overflow".into()))?;
    let expected_b = cols_a
        .checked_mul(cols_b)
        .ok_or_else(|| RunnerError::InvalidInput("matrix B dimensions overflow".into()))?;

    if a.len() != expected_a {
        return Err(RunnerError::InvalidInput(format!(
            "matrix A length {} does not match {}x{}",
            a.len(),
            rows_a,
            cols_a
        )));
    }
    if b.len() != expected_b {
        return Err(RunnerError::InvalidInput(format!(
            "matrix B length {} does not match {}x{}",
            b.len(),
            cols_a,
            cols_b
        )));
    }

    let mut out = vec![0.0_f32; rows_a * cols_b];
    // ikj loop order keeps row slices of B and out contiguous for cache locality.
    for i in 0..rows_a {
        let a_row = &a[i * cols_a..(i + 1) * cols_a];
        let out_row = &mut out[i * cols_b..(i + 1) * cols_b];
        for k in 0..cols_a {
            let aik = a_row[k];
            let b_row = &b[k * cols_b..(k + 1) * cols_b];
            for j in 0..cols_b {
                out_row[j] += aik * b_row[j];
            }
        }
    }

    Ok(TaskResult::MatMul {
        rows: rows_a,
        cols: cols_b,
        data: out,
    })
}

fn run_cosine_top_k(
    query: &[f32],
    corpus: &[Vec<f32>],
    top_k: usize,
) -> Result<TaskResult, RunnerError> {
    if query.is_empty() {
        return Err(RunnerError::InvalidInput(
            "query vector must not be empty".into(),
        ));
    }
    let query_norm = l2_norm(query);
    if query_norm == 0.0 {
        return Err(RunnerError::InvalidInput(
            "query vector norm must be non-zero".into(),
        ));
    }

    let mut scored = Vec::with_capacity(corpus.len());
    for (index, doc) in corpus.iter().enumerate() {
        if doc.len() != query.len() {
            return Err(RunnerError::InvalidInput(format!(
                "corpus vector at index {index} has length {}, expected {}",
                doc.len(),
                query.len()
            )));
        }
        let doc_norm = l2_norm(doc);
        let score = if doc_norm == 0.0 {
            0.0
        } else {
            dot(query, doc) / (query_norm * doc_norm)
        };
        scored.push(SimilarityMatch { index, score });
    }

    scored.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.index.cmp(&rhs.index))
    });
    scored.truncate(top_k);

    Ok(TaskResult::CosineTopK { matches: scored })
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn resolve_model_path(model: &str, models_dir: Option<&Path>) -> Result<PathBuf, RunnerError> {
    if model.trim().is_empty() {
        return Err(RunnerError::InvalidInput("model must not be empty".into()));
    }

    let rel = Path::new(model);
    match models_dir {
        Some(base) => {
            if rel.is_absolute() {
                return Err(RunnerError::Forbidden(
                    "absolute model paths are disallowed when --models-dir is configured".into(),
                ));
            }
            for comp in rel.components() {
                if matches!(comp, Component::ParentDir | Component::RootDir | Component::Prefix(_)) {
                    return Err(RunnerError::Forbidden(
                        "model path must stay within --models-dir".into(),
                    ));
                }
            }
            Ok(base.join(rel))
        }
        None => Ok(rel.to_path_buf()),
    }
}

pub fn is_command_allowed(program: &str, allowed_commands: &[String]) -> bool {
    if program.trim().is_empty() {
        return false;
    }
    allowed_commands.iter().any(|allowed| allowed == program)
}

async fn run_subprocess(
    program: &str,
    args: &[String],
    stdin_data: Option<&str>,
    limit: Duration,
    stream_tx: Option<broadcast::Sender<JobStreamEvent>>,
) -> Result<TaskResult, RunnerError> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if stdin_data.is_some() {
        cmd.stdin(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null());
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| RunnerError::Process(format!("failed to spawn '{program}': {e}")))?;

    if let Some(input) = stdin_data
        && let Some(mut child_stdin) = child.stdin.take()
    {
        let bytes = input.as_bytes().to_vec();
        tokio::spawn(async move {
            let _ = child_stdin.write_all(&bytes).await;
        });
    }

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let tx_out = stream_tx.clone();
    let out_task = tokio::spawn(async move {
        let mut collected = String::new();
        if let Some(pipe) = stdout_pipe {
            let mut lines = BufReader::new(pipe).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(tx) = &tx_out {
                    let _ = tx.send(JobStreamEvent::Output {
                        stream: "stdout".to_string(),
                        line: line.clone(),
                    });
                }
                if !collected.is_empty() {
                    collected.push('\n');
                }
                collected.push_str(&line);
            }
        }
        collected
    });

    let tx_err = stream_tx;
    let err_task = tokio::spawn(async move {
        let mut collected = String::new();
        if let Some(pipe) = stderr_pipe {
            let mut lines = BufReader::new(pipe).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(tx) = &tx_err {
                    let _ = tx.send(JobStreamEvent::Output {
                        stream: "stderr".to_string(),
                        line: line.clone(),
                    });
                }
                if !collected.is_empty() {
                    collected.push('\n');
                }
                collected.push_str(&line);
            }
        }
        collected
    });

    let status = timeout(limit, child.wait())
        .await
        .map_err(|_| RunnerError::Timeout(limit.as_secs()))?
        .map_err(|e| RunnerError::Process(format!("failed waiting on '{program}': {e}")))?;

    let stdout = out_task.await.unwrap_or_default();
    let stderr_raw = err_task.await.unwrap_or_default();
    let exit_code = status.code().unwrap_or(-1);
    let stderr = if stderr_raw.trim().is_empty() {
        None
    } else {
        Some(stderr_raw)
    };

    if !status.success() {
        return Err(RunnerError::Process(format!(
            "'{program}' exited with code {exit_code}: {}",
            stderr.as_deref().unwrap_or("no stderr")
        )));
    }

    Ok(TaskResult::TextOutput {
        text: stdout,
        exit_code,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_computes_expected_product() {
        // (2x3) * (3x2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let res = run_matmul(2, 3, 2, &a, &b).unwrap();
        match res {
            TaskResult::MatMul { rows, cols, data } => {
                assert_eq!((rows, cols), (2, 2));
                assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
            }
            _ => panic!("unexpected result variant"),
        }
    }

    #[test]
    fn cosine_top_k_ranks_matches() {
        let query = vec![1.0, 0.0];
        let corpus = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.7, 0.7]];
        let res = run_cosine_top_k(&query, &corpus, 2).unwrap();
        match res {
            TaskResult::CosineTopK { matches } => {
                assert_eq!(matches.len(), 2);
                assert_eq!(matches[0].index, 1);
                assert!((matches[0].score - 1.0).abs() < 1e-5);
                assert_eq!(matches[1].index, 2);
            }
            _ => panic!("unexpected result variant"),
        }
    }

    #[test]
    fn model_path_rejects_traversal_when_models_dir_set() {
        let base = Path::new("/data/data/com.termux/files/home/models");
        assert!(resolve_model_path("../secret.gguf", Some(base)).is_err());
        assert!(resolve_model_path("/etc/passwd", Some(base)).is_err());
        assert_eq!(
            resolve_model_path("qwen2.5-3b.gguf", Some(base)).unwrap(),
            base.join("qwen2.5-3b.gguf")
        );
    }

    #[test]
    fn command_allowlist_requires_exact_match() {
        let allowed = vec!["python3".to_string(), "whisper-cli".to_string()];
        assert!(is_command_allowed("python3", &allowed));
        assert!(!is_command_allowed("sh", &allowed));
        assert!(!is_command_allowed("", &allowed));
    }
}
