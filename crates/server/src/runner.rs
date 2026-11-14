//! Task execution: in-process compute (`MatMul`, `CosineTopK`) and sandboxed subprocesses.

use std::{
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::Duration,
};

/// Largest `MatMul` output accepted: 64 Mi f32 = 256 MiB, and several times that as JSON.
/// Bigger requests would risk the Android low-memory killer taking the whole server down.
const MAX_MATMUL_OUTPUT_ELEMENTS: usize = 64 * 1024 * 1024;

/// Cap on stdout/stderr kept in a task result (streaming is unaffected). Keeps job
/// records, which are retained in memory, bounded for chatty processes.
const MAX_COLLECTED_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

/// llama.cpp flags that take a value and are safe to expose: none of them read or write
/// files, so they can't be used to escape `--models-dir` or the allowlist.
const ALLOWED_LLAMA_VALUE_FLAGS: &[&str] = &[
    "-c",
    "--ctx-size",
    "-t",
    "--threads",
    "-tb",
    "--threads-batch",
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "--top-k",
    "--top-p",
    "--min-p",
    "--typical",
    "--repeat-penalty",
    "--repeat-last-n",
    "--presence-penalty",
    "--frequency-penalty",
    "--mirostat",
    "--mirostat-lr",
    "--mirostat-ent",
    "-s",
    "--seed",
    "-ngl",
    "--n-gpu-layers",
    "--keep",
    "--temp",
    "-n",
    "--predict",
    "--n-predict",
    "--cache-type-k",
    "--cache-type-v",
    "-fa",
    "--flash-attn",
    "--rope-freq-base",
    "--rope-freq-scale",
];

/// Value-less llama.cpp flags that are safe to expose (see [`ALLOWED_LLAMA_VALUE_FLAGS`]).
const ALLOWED_LLAMA_BOOL_FLAGS: &[&str] = &[
    "--no-display-prompt",
    "--simple-io",
    "--ignore-eos",
    "--no-mmap",
    "--mlock",
    "--color",
    "--no-warmup",
    "-e",
    "--escape",
    "--no-escape",
    "--verbose-prompt",
];

use offload_protocol::{JobStreamEvent, OutputStream, SimilarityMatch, TaskKind, TaskResult};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::broadcast,
    time::timeout,
};

use crate::state::ServerConfig;

/// Why a task could not produce a result; mapped to an HTTP status by `ApiError`.
#[derive(Debug, Error)]
pub enum RunnerError {
    /// Malformed task parameters (400).
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Disallowed command, path, or flag (403).
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// Subprocess exceeded its timeout, in seconds (504).
    #[error("execution timed out after {0}s")]
    Timeout(u64),
    /// Subprocess failed to spawn or exited non-zero (502).
    #[error("process execution failed: {0}")]
    Process(String),
    /// A worker thread panicked or was cancelled (500).
    #[error("internal task failure: {0}")]
    Internal(String),
}

/// Executes one task with the configured worker environment.
///
/// CPU-bound tasks run on the blocking pool; subprocess output is forwarded line by
/// line to `stream_tx` when given.
///
/// # Errors
///
/// Returns a [`RunnerError`] describing why the task was rejected or failed.
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
            validate_llama_extra_args(&extra_args)?;
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

    let out_elems = rows_a
        .checked_mul(cols_b)
        .ok_or_else(|| RunnerError::InvalidInput("output matrix dimensions overflow".into()))?;
    if out_elems > MAX_MATMUL_OUTPUT_ELEMENTS {
        return Err(RunnerError::InvalidInput(format!(
            "output matrix size {out_elems} exceeds limit of {MAX_MATMUL_OUTPUT_ELEMENTS} elements"
        )));
    }

    let mut out = vec![0.0_f32; out_elems];
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

    let cmp = |lhs: &SimilarityMatch, rhs: &SimilarityMatch| {
        rhs.score
            .total_cmp(&lhs.score)
            .then_with(|| lhs.index.cmp(&rhs.index))
    };
    if top_k == 0 {
        scored.clear();
    } else {
        if top_k < scored.len() {
            scored.select_nth_unstable_by(top_k - 1, cmp);
            scored.truncate(top_k);
        }
        scored.sort_by(cmp);
    }

    Ok(TaskResult::CosineTopK { matches: scored })
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Resolves a model name inside `models_dir`, rejecting absolute paths and `..` when set.
///
/// # Errors
///
/// [`RunnerError::InvalidInput`] for an empty name, [`RunnerError::Forbidden`] if it escapes `models_dir`.
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
                if matches!(
                    comp,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                ) {
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

/// Checks every extra llama.cpp argument against the flag allowlists.
///
/// # Errors
///
/// [`RunnerError::Forbidden`] for unknown flags, positional args, or path-like values;
/// [`RunnerError::InvalidInput`] if a value flag is missing its value.
pub fn validate_llama_extra_args(extra_args: &[String]) -> Result<(), RunnerError> {
    let mut i = 0;
    while i < extra_args.len() {
        let arg = &extra_args[i];
        if !arg.starts_with('-') {
            return Err(RunnerError::Forbidden(format!(
                "positional llama extra arg '{arg}' is not allowed"
            )));
        }
        let (flag, inline_val) = match arg.split_once('=') {
            Some((f, v)) => (f, Some(v)),
            None => (arg.as_str(), None),
        };
        if ALLOWED_LLAMA_VALUE_FLAGS.contains(&flag) {
            let val = if let Some(v) = inline_val {
                v
            } else {
                i += 1;
                extra_args.get(i).map(String::as_str).ok_or_else(|| {
                    RunnerError::InvalidInput(format!("llama extra arg '{flag}' requires a value"))
                })?
            };
            if val.is_empty() || val.contains('/') || val.contains('\\') || val.contains("..") {
                return Err(RunnerError::Forbidden(format!(
                    "invalid value for llama extra arg '{flag}'"
                )));
            }
        } else if ALLOWED_LLAMA_BOOL_FLAGS.contains(&flag) && inline_val.is_none() {
            // Allowed boolean flag.
        } else {
            return Err(RunnerError::Forbidden(format!(
                "llama extra arg '{flag}' is not allowed"
            )));
        }
        i += 1;
    }
    Ok(())
}

/// Whether `program` exactly matches an `--allow-cmd` entry (no path or prefix matching).
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

    let out_task = tokio::spawn(read_pipe_lines(
        stdout_pipe,
        OutputStream::Stdout,
        stream_tx.clone(),
    ));
    let err_task = tokio::spawn(read_pipe_lines(
        stderr_pipe,
        OutputStream::Stderr,
        stream_tx,
    ));

    let status = timeout(limit, child.wait())
        .await
        .map_err(|_elapsed| RunnerError::Timeout(limit.as_secs()))?
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

async fn read_pipe_lines<R>(
    pipe: Option<R>,
    stream: OutputStream,
    tx: Option<broadcast::Sender<JobStreamEvent>>,
) -> String
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut collected = String::new();
    let Some(pipe) = pipe else {
        return collected;
    };

    let mut reader = BufReader::new(pipe);
    let mut raw_line = Vec::new();
    let mut truncated = false;

    loop {
        raw_line.clear();
        match reader.read_until(b'\n', &mut raw_line).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }

        if raw_line.last() == Some(&b'\n') {
            raw_line.pop();
            if raw_line.last() == Some(&b'\r') {
                raw_line.pop();
            }
        }

        let line = String::from_utf8_lossy(&raw_line).into_owned();
        if let Some(tx) = &tx {
            let _ = tx.send(JobStreamEvent::Output {
                stream,
                line: line.clone(),
            });
        }

        if !truncated {
            let sep_len = usize::from(!collected.is_empty());
            if collected.len() + sep_len + line.len() <= MAX_COLLECTED_OUTPUT_BYTES {
                if sep_len > 0 {
                    collected.push('\n');
                }
                collected.push_str(&line);
            } else {
                let remaining =
                    MAX_COLLECTED_OUTPUT_BYTES.saturating_sub(collected.len() + sep_len);
                if remaining > 0 {
                    if sep_len > 0 {
                        collected.push('\n');
                    }
                    let mut end = remaining.min(line.len());
                    while end > 0 && !line.is_char_boundary(end) {
                        end -= 1;
                    }
                    collected.push_str(&line[..end]);
                }
                collected.push_str("\n... [output truncated]");
                truncated = true;
            }
        }
    }

    collected
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
        resolve_model_path("../secret.gguf", Some(base)).unwrap_err();
        resolve_model_path("/etc/passwd", Some(base)).unwrap_err();
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

    #[test]
    fn matmul_rejects_oversized_output() {
        let a = vec![1.0_f32; 10_000];
        let b = vec![1.0_f32; 10_000];
        let err = run_matmul(10_000, 1, 10_000, &a, &b).unwrap_err();
        assert!(matches!(err, RunnerError::InvalidInput(_)));
    }

    #[test]
    fn cosine_top_k_handles_nan_without_panicking() {
        let query = vec![1.0_f32, 0.0];
        let corpus = vec![vec![f32::INFINITY, 1.0], vec![1.0, 0.0], vec![0.5, 0.5]];
        let res = run_cosine_top_k(&query, &corpus, 3).unwrap();
        match res {
            TaskResult::CosineTopK { matches } => assert_eq!(matches.len(), 3),
            _ => panic!("unexpected result variant"),
        }
    }

    #[test]
    fn validate_llama_extra_args_blocks_paths_and_model_flags() {
        validate_llama_extra_args(&[
            "--temp".into(),
            "0.7".into(),
            "--ctx-size=4096".into(),
            "--simple-io".into(),
        ])
        .unwrap();
        validate_llama_extra_args(&["-m".into(), "/etc/passwd".into()]).unwrap_err();
        validate_llama_extra_args(&["--model=/etc/passwd".into()]).unwrap_err();
        validate_llama_extra_args(&["--log-file".into(), "/tmp/out".into()]).unwrap_err();
        validate_llama_extra_args(&["/etc/passwd".into()]).unwrap_err();
        validate_llama_extra_args(&["--ctx-size".into(), "../foo".into()]).unwrap_err();
    }

    #[tokio::test]
    async fn read_pipe_lines_handles_non_utf8_bytes() {
        let data: &[u8] = b"first line\nhello \xff\xfe world\nthird line\n";
        let output = read_pipe_lines(Some(data), OutputStream::Stdout, None).await;
        assert_eq!(
            output,
            "first line\nhello \u{FFFD}\u{FFFD} world\nthird line"
        );
    }
}
