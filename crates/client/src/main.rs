//! `offload-client`: command-line front end for `offload-server`.
//!
//! Prints results as JSON on stdout and progress/diagnostics on stderr, so output can be piped.

use std::{
    error::Error,
    io::{Write, stderr},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use offload_client::{
    DEFAULT_POLL_INTERVAL, OffloadClient, TransferProgress, TransferSummary, discover,
    discover_first,
};
use offload_protocol::{
    DEFAULT_DISCOVERY_PORT, DEFAULT_DISCOVERY_WAIT, JobStreamEvent, OutputStream, TaskKind,
};
use uuid::Uuid;

type CliResult = Result<(), Box<dyn Error>>;

#[derive(Debug, Parser)]
#[command(
    name = "offload-client",
    about = "Client CLI for Android/Termux ML offload"
)]
struct Cli {
    /// Base URL of offload-server (if omitted, auto-discovers on LAN)
    #[arg(long, env = "OFFLOAD_URL")]
    url: Option<String>,

    /// UDP port used for LAN auto-discovery
    #[arg(long, env = "OFFLOAD_DISCOVERY_PORT", default_value_t = DEFAULT_DISCOVERY_PORT)]
    discovery_port: u16,

    /// Optional bearer token matching offload-server --token
    #[arg(long, env = "OFFLOAD_TOKEN")]
    token: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Discover offload-server nodes on the local network
    Discover {
        /// How long to wait for replies, in milliseconds
        #[arg(long, default_value_t = DEFAULT_DISCOVERY_WAIT.as_millis().try_into().unwrap_or(u64::MAX))]
        timeout_ms: u64,
    },

    /// Check server health and print worker hardware/capability info
    Info,

    /// List models available in the server's --models-dir
    Models,

    /// Send an echo payload to measure round-trip latency
    Echo {
        #[arg(default_value = "hello-oneplus-15")]
        message: String,
    },

    /// Run a square f32 matrix multiplication benchmark (N x N) on the device
    MatmulBench {
        #[arg(long, default_value_t = 256)]
        size: usize,
    },

    /// Run llama.cpp completion on the device
    Llama {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
        #[arg(long)]
        temp: Option<f32>,
        #[arg(long)]
        r#async: bool,
        #[arg(long)]
        stream: bool,
    },

    /// Run an allowlisted external ML command on the device
    Cmd {
        #[arg(long)]
        program: String,
        #[arg(long)]
        r#async: bool,
        #[arg(long)]
        stream: bool,
        /// Send this local file as the program's stdin (e.g. `--program python3 --stdin-file x.py -- -`)
        #[arg(long)]
        stdin_file: Option<PathBuf>,
        #[arg(last = true)]
        args: Vec<String>,
    },

    /// Fetch status of an async job by UUID
    Job { id: Uuid },

    /// Cancel a queued or running async job by UUID
    Cancel { id: Uuid },

    /// Stream live output of an async job by UUID
    Stream { id: Uuid },

    /// List recent async jobs on the server
    Jobs,

    /// Upload a file (e.g. GGUF model) with automatic resume and speed/ETA progress
    Push {
        /// Local file path to upload
        file: PathBuf,
        /// Remote relative path under the server's --files-dir (defaults to filename)
        #[arg(long)]
        to: Option<String>,
        /// Chunk size in MiB
        #[arg(long, default_value_t = 4)]
        chunk_mb: usize,
    },

    /// Download a file from the server's --files-dir with Range resume and speed/ETA progress
    Pull {
        /// Remote relative path under the server's --files-dir
        remote_path: String,
        /// Local destination path (defaults to filename in current directory)
        #[arg(long)]
        to: Option<PathBuf>,
    },

    /// Execute a raw `TaskKind` JSON payload from a file or stdin (`-`)
    RunJson {
        /// Path to a JSON file, or `-` to read from stdin
        #[arg(default_value = "-")]
        input: String,
        #[arg(long)]
        r#async: bool,
        #[arg(long)]
        stream: bool,
    },
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

async fn run() -> CliResult {
    let cli = Cli::parse();

    if let Commands::Discover { timeout_ms } = cli.command {
        let nodes = discover(cli.discovery_port, Duration::from_millis(timeout_ms)).await?;
        println!("{}", serde_json::to_string_pretty(&nodes)?);
        return Ok(());
    }

    let url = if let Some(u) = cli.url {
        u
    } else {
        let node = discover_first(cli.discovery_port, DEFAULT_DISCOVERY_WAIT).await?;
        eprintln!("discovered '{}' at {}", node.name, node.url);
        node.url
    };

    let client = OffloadClient::new(url, cli.token)?;

    match cli.command {
        Commands::Discover { .. } => unreachable!("handled above"),
        Commands::Info => {
            client.health().await?;
            let info = client.info().await?;
            println!("{}", serde_json::to_string_pretty(&info)?);
        }

        Commands::Models => {
            let models = client.list_models().await?;
            println!("{}", serde_json::to_string_pretty(&models)?);
        }

        Commands::Echo { message } => {
            let started = Instant::now();
            let res = client
                .run(TaskKind::Echo {
                    payload: serde_json::Value::String(message),
                })
                .await?;
            let rtt = started.elapsed().as_millis();
            println!("rtt_ms={rtt} server_duration_ms={}", res.duration_ms);
            println!("{}", serde_json::to_string_pretty(&res)?);
        }

        Commands::MatmulBench { size } => matmul_bench(&client, size).await?,

        Commands::Llama {
            model,
            prompt,
            max_tokens,
            temp,
            r#async,
            stream,
        } => {
            let task = TaskKind::LlamaCompletion {
                model,
                prompt,
                max_tokens,
                temperature: temp,
                extra_args: vec![],
            };
            execute_cli_task(&client, task, r#async, stream).await?;
        }

        Commands::Cmd {
            program,
            r#async,
            stream,
            stdin_file,
            args,
        } => {
            let stdin = stdin_file.map(std::fs::read_to_string).transpose()?;
            let task = TaskKind::Command {
                program,
                args,
                stdin,
                timeout_secs: None,
            };
            execute_cli_task(&client, task, r#async, stream).await?;
        }

        Commands::Job { id } => {
            let job = client.job_status(id).await?;
            println!("{}", serde_json::to_string_pretty(&job)?);
        }

        Commands::Cancel { id } => {
            let job = client.cancel_job(id).await?;
            println!("{}", serde_json::to_string_pretty(&job)?);
        }

        Commands::Stream { id } => {
            stream_to_console(&client, id).await?;
        }

        Commands::Jobs => {
            let jobs = client.list_jobs().await?;
            println!("{}", serde_json::to_string_pretty(&jobs)?);
        }

        Commands::Push { file, to, chunk_mb } => push(&client, &file, to, chunk_mb).await?,
        Commands::Pull { remote_path, to } => pull(&client, &remote_path, to).await?,
        Commands::RunJson {
            input,
            r#async,
            stream,
        } => run_json(&client, &input, r#async, stream).await?,
    }

    Ok(())
}

async fn run_json(client: &OffloadClient, input: &str, r#async: bool, stream: bool) -> CliResult {
    let raw = if input == "-" {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        std::fs::read_to_string(input)?
    };
    let task: TaskKind = serde_json::from_str(&raw)?;
    execute_cli_task(client, task, r#async, stream).await
}

async fn matmul_bench(client: &OffloadClient, size: usize) -> CliResult {
    let len = size * size;
    let started = Instant::now();
    let res = client
        .run(TaskKind::MatMul {
            rows_a: size,
            cols_a: size,
            cols_b: size,
            a: vec![1.0_f32; len],
            b: vec![0.5_f32; len],
        })
        .await?;
    let rtt = started.elapsed().as_millis();
    println!(
        "matmul {size}x{size}: rtt_ms={rtt} server_duration_ms={}",
        res.duration_ms
    );
    Ok(())
}

async fn push(
    client: &OffloadClient,
    file: &Path,
    to: Option<String>,
    chunk_mb: usize,
) -> CliResult {
    let remote_path = match to {
        Some(p) => p,
        None => file
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("local file has no valid UTF-8 filename")?
            .to_string(),
    };
    let chunk_bytes = chunk_mb.max(1) * 1024 * 1024;
    let summary = client
        .upload_file(file, &remote_path, chunk_bytes, render_progress)
        .await?;
    print_summary("completed", &summary.remote_path, &summary);
    Ok(())
}

async fn pull(client: &OffloadClient, remote_path: &str, to: Option<PathBuf>) -> CliResult {
    let local_path = match to {
        Some(p) => p,
        None => Path::new(remote_path)
            .file_name()
            .map(PathBuf::from)
            .ok_or("remote_path has no filename")?,
    };
    let summary = client
        .download_file(remote_path, &local_path, render_progress)
        .await?;
    print_summary("downloaded", &local_path.display().to_string(), &summary);
    Ok(())
}

fn print_summary(verb: &str, name: &str, summary: &TransferSummary) {
    let total = format_bytes(summary.total_bytes as f64);
    if summary.already_existed {
        println!("already up to date: {name} ({total})");
        return;
    }
    eprintln!();
    let secs = summary.elapsed.as_secs_f64();
    let avg_bps = summary.uploaded_bytes as f64 / secs.max(1e-4);
    println!(
        "{verb} {name} ({total}, resumed {}, avg {}/s in {secs:.2}s)",
        format_bytes(summary.resumed_bytes as f64),
        format_bytes(avg_bps),
    );
}

async fn execute_cli_task(
    client: &OffloadClient,
    task: TaskKind,
    use_async: bool,
    stream: bool,
) -> CliResult {
    if stream {
        let job = client.submit(task).await?;
        eprintln!("job_id={}", job.id);
        stream_to_console(client, job.id).await?;
    } else if use_async {
        let record = client.run_and_wait(task, DEFAULT_POLL_INTERVAL).await?;
        println!("{}", serde_json::to_string_pretty(&record)?);
    } else {
        let res = client.run(task).await?;
        println!("{}", serde_json::to_string_pretty(&res)?);
    }
    Ok(())
}

async fn stream_to_console(client: &OffloadClient, id: Uuid) -> CliResult {
    client
        .stream_job(id, |ev| match ev {
            JobStreamEvent::Output { stream, line } => {
                if stream == OutputStream::Stderr {
                    eprintln!("[stderr] {line}");
                } else {
                    println!("{line}");
                }
            }
            JobStreamEvent::Done {
                status,
                duration_ms,
                error,
            } => {
                eprintln!("[done] status={status} duration_ms={duration_ms:?} error={error:?}");
            }
        })
        .await?;
    Ok(())
}

fn render_progress(p: TransferProgress) {
    const BAR_WIDTH: usize = 24;
    let pct = if p.total_bytes == 0 {
        100.0
    } else {
        (p.bytes_done as f64 / p.total_bytes as f64) * 100.0
    };
    let filled = if p.total_bytes == 0 {
        BAR_WIDTH
    } else {
        let cells = p.bytes_done.min(p.total_bytes) * BAR_WIDTH as u64 / p.total_bytes;
        usize::try_from(cells).unwrap_or(BAR_WIDTH)
    };
    let empty = BAR_WIDTH - filled;

    let bar: String = "█".repeat(filled) + &"░".repeat(empty);
    let speed = format!("{}/s", format_bytes(p.speed_bps));
    let eta = p.eta_secs.map_or_else(|| "--:--".to_string(), format_eta);
    let resume_tag = if p.resumed_bytes > 0 {
        format!(" | resumed {}", format_bytes(p.resumed_bytes as f64))
    } else {
        String::new()
    };

    let mut err = stderr();
    let _ = write!(
        err,
        "\r[{bar}] {pct:5.1}% | {} / {} | {speed} | ETA {eta}{resume_tag}   ",
        format_bytes(p.bytes_done as f64),
        format_bytes(p.total_bytes as f64),
    );
    let _ = err.flush();
}

fn format_bytes(bytes: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut val = bytes.max(0.0);
    let mut idx = 0;
    while val >= 1024.0 && idx + 1 < UNITS.len() {
        val /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{val:.0} {}", UNITS[idx])
    } else {
        format!("{val:.2} {}", UNITS[idx])
    }
}

fn format_eta(secs: u64) -> String {
    let mins = secs / 60;
    let rem = secs % 60;
    if mins >= 60 {
        let hrs = mins / 60;
        let m = mins % 60;
        format!("{hrs:02}:{m:02}:{rem:02}")
    } else {
        format!("{mins:02}:{rem:02}")
    }
}
