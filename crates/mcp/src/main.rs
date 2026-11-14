use std::{path::Path, sync::Arc, time::Duration};

use clap::Parser;
use offload_client::OffloadClient;
use offload_protocol::{DEFAULT_DISCOVERY_PORT, TaskKind};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::{OnceCell, mpsc},
};
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(name = "offload-mcp", about = "MCP stdio server for Android/Termux ML offload")]
struct Cli {
    /// Default offload-server URL (if omitted, auto-discovers on LAN)
    #[arg(long, env = "OFFLOAD_URL")]
    url: Option<String>,

    /// Default bearer token for offload-server
    #[arg(long, env = "OFFLOAD_TOKEN")]
    token: Option<String>,

    /// UDP port used for LAN auto-discovery
    #[arg(long, env = "OFFLOAD_DISCOVERY_PORT", default_value_t = DEFAULT_DISCOVERY_PORT)]
    discovery_port: u16,
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

struct McpState {
    cli: Cli,
    default_client: OnceCell<OffloadClient>,
}

impl McpState {
    fn new(cli: Cli) -> Self {
        Self {
            cli,
            default_client: OnceCell::new(),
        }
    }
}

#[tokio::main]
async fn main() {
    let state = Arc::new(McpState::new(Cli::parse()));
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let (tx, mut rx) = mpsc::channel::<JsonRpcResponse>(128);

    let writer_task = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(resp) = rx.recv().await {
            write_response(&mut stdout, &resp).await;
        }
    });

    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(err) => {
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("parse error: {err}"),
                    }),
                };
                if tx.send(resp).await.is_err() {
                    break;
                }
                continue;
            }
        };

        // Notifications have no `id` and expect no response.
        let Some(id) = req.id else {
            continue;
        };

        let state = Arc::clone(&state);
        let tx = tx.clone();
        tokio::spawn(async move {
            let resp = handle_request(&state, id, &req.method, req.params).await;
            let _ = tx.send(resp).await;
        });
    }

    drop(tx);
    let _ = writer_task.await;
}

async fn write_response(stdout: &mut tokio::io::Stdout, resp: &JsonRpcResponse) {
    if let Ok(mut bytes) = serde_json::to_vec(resp) {
        bytes.push(b'\n');
        let _ = stdout.write_all(&bytes).await;
        let _ = stdout.flush().await;
    }
}

async fn handle_request(state: &McpState, id: Value, method: &str, params: Value) -> JsonRpcResponse {
    match method {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "offload-mcp",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            })),
            error: None,
        },
        "ping" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({})),
            error: None,
        },
        "tools/list" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "tools": tool_definitions() })),
            error: None,
        },
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            let outcome = call_tool(state, name, args).await;
            let result = match outcome {
                Ok(val) => json!({
                    "content": [{ "type": "text", "text": serde_json::to_string_pretty(&val).unwrap_or_default() }],
                    "isError": false
                }),
                Err(msg) => json!({
                    "content": [{ "type": "text", "text": msg }],
                    "isError": true
                }),
            };
            JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(result),
                error: None,
            }
        }
        other => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("method not found: {other}"),
            }),
        },
    }
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "discover_nodes",
            "description": "Discover Android/Termux offload-server nodes on the local network via UDP.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "timeout_ms": { "type": "integer", "default": 1500 }
                }
            }
        },
        {
            "name": "node_info",
            "description": "Get hardware, concurrency, models_dir, and allowed_commands for an offload node (auto-discovers if url is omitted).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": { "type": "string" }
                }
            }
        },
        {
            "name": "run_llama",
            "description": "Run local llama.cpp inference on the Android/Termux device.",
            "inputSchema": {
                "type": "object",
                "required": ["model", "prompt"],
                "properties": {
                    "url": { "type": "string" },
                    "model": { "type": "string", "description": "GGUF model filename or path" },
                    "prompt": { "type": "string" },
                    "max_tokens": { "type": "integer", "default": 256 },
                    "temperature": { "type": "number" },
                    "async": { "type": "boolean", "default": true }
                }
            }
        },
        {
            "name": "run_command",
            "description": "Run an allowlisted ML command (e.g. python3, whisper-cli) on the Android/Termux device.",
            "inputSchema": {
                "type": "object",
                "required": ["program"],
                "properties": {
                    "url": { "type": "string" },
                    "program": { "type": "string" },
                    "args": { "type": "array", "items": { "type": "string" }, "default": [] },
                    "stdin": { "type": "string" },
                    "timeout_secs": { "type": "integer" },
                    "async": { "type": "boolean", "default": false }
                }
            }
        },
        {
            "name": "run_task",
            "description": "Run any TaskKind JSON payload (echo, mat_mul, cosine_top_k, llama_completion, command) on the device.",
            "inputSchema": {
                "type": "object",
                "required": ["task"],
                "properties": {
                    "url": { "type": "string" },
                    "task": { "type": "object" },
                    "async": { "type": "boolean", "default": false }
                }
            }
        },
        {
            "name": "get_job",
            "description": "Check the status or result of an async job by UUID.",
            "inputSchema": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "url": { "type": "string" },
                    "id": { "type": "string" }
                }
            }
        },
        {
            "name": "push_file",
            "description": "Upload a local file (e.g. GGUF model or script) to the phone's --files-dir with automatic resume.",
            "inputSchema": {
                "type": "object",
                "required": ["local_path"],
                "properties": {
                    "url": { "type": "string" },
                    "local_path": { "type": "string" },
                    "remote_path": { "type": "string" },
                    "chunk_mb": { "type": "integer", "default": 4 }
                }
            }
        },
        {
            "name": "pull_file",
            "description": "Download a file from the phone's --files-dir with Range resume and BLAKE3 verification.",
            "inputSchema": {
                "type": "object",
                "required": ["remote_path", "local_path"],
                "properties": {
                    "url": { "type": "string" },
                    "remote_path": { "type": "string" },
                    "local_path": { "type": "string" }
                }
            }
        },
        {
            "name": "list_models",
            "description": "List GGUF/model files available in the phone's --models-dir.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": { "type": "string" }
                }
            }
        },
        {
            "name": "cancel_job",
            "description": "Cancel a queued or running async job by UUID.",
            "inputSchema": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "url": { "type": "string" },
                    "id": { "type": "string" }
                }
            }
        }
    ])
}

async fn resolve_client(state: &McpState, explicit_url: Option<&str>) -> Result<OffloadClient, String> {
    if let Some(url) = explicit_url {
        return OffloadClient::new(url, state.cli.token.clone()).map_err(|e| e.to_string());
    }
    let client = state
        .default_client
        .get_or_try_init(|| async {
            let url = match state.cli.url.as_deref() {
                Some(u) => u.to_string(),
                None => OffloadClient::discover_first(
                    state.cli.discovery_port,
                    Duration::from_millis(1500),
                )
                .await
                .map_err(|e| e.to_string())?
                .url,
            };
            OffloadClient::new(url, state.cli.token.clone()).map_err(|e| e.to_string())
        })
        .await?;
    Ok(client.clone())
}

async fn execute_task(client: &OffloadClient, task: TaskKind, use_async: bool) -> Result<Value, String> {
    if use_async {
        let record = client
            .run_and_wait(task, Duration::from_millis(500))
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_value(record).map_err(|e| e.to_string())
    } else {
        let res = client.run(task).await.map_err(|e| e.to_string())?;
        serde_json::to_value(res).map_err(|e| e.to_string())
    }
}

async fn call_tool(state: &McpState, name: &str, args: Value) -> Result<Value, String> {
    let explicit_url = args.get("url").and_then(Value::as_str);

    match name {
        "discover_nodes" => {
            let ms = args.get("timeout_ms").and_then(Value::as_u64).unwrap_or(1500);
            let nodes =
                OffloadClient::discover(state.cli.discovery_port, Duration::from_millis(ms))
                    .await
                    .map_err(|e| e.to_string())?;
            serde_json::to_value(nodes).map_err(|e| e.to_string())
        }
        "node_info" => {
            let client = resolve_client(state, explicit_url).await?;
            let info = client.info().await.map_err(|e| e.to_string())?;
            serde_json::to_value(info).map_err(|e| e.to_string())
        }
        "run_llama" => {
            let client = resolve_client(state, explicit_url).await?;
            let model = args
                .get("model")
                .and_then(Value::as_str)
                .ok_or("missing required 'model'")?
                .to_string();
            let prompt = args
                .get("prompt")
                .and_then(Value::as_str)
                .ok_or("missing required 'prompt'")?
                .to_string();
            let max_tokens = args
                .get("max_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(256) as u32;
            let temperature = args
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|t| t as f32);
            let use_async = args.get("async").and_then(Value::as_bool).unwrap_or(true);

            execute_task(
                &client,
                TaskKind::LlamaCompletion {
                    model,
                    prompt,
                    max_tokens,
                    temperature,
                    extra_args: vec![],
                },
                use_async,
            )
            .await
        }
        "run_command" => {
            let client = resolve_client(state, explicit_url).await?;
            let program = args
                .get("program")
                .and_then(Value::as_str)
                .ok_or("missing required 'program'")?
                .to_string();
            let cmd_args: Vec<String> = args
                .get("args")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| format!("invalid 'args': {e}"))?
                .unwrap_or_default();
            let stdin = args.get("stdin").and_then(Value::as_str).map(str::to_string);
            let timeout_secs = args.get("timeout_secs").and_then(Value::as_u64);
            let use_async = args.get("async").and_then(Value::as_bool).unwrap_or(false);

            execute_task(
                &client,
                TaskKind::Command {
                    program,
                    args: cmd_args,
                    stdin,
                    timeout_secs,
                },
                use_async,
            )
            .await
        }
        "run_task" => {
            let client = resolve_client(state, explicit_url).await?;
            let task_val = args.get("task").cloned().ok_or("missing required 'task'")?;
            let task: TaskKind =
                serde_json::from_value(task_val).map_err(|e| format!("invalid 'task': {e}"))?;
            let use_async = args.get("async").and_then(Value::as_bool).unwrap_or(false);
            execute_task(&client, task, use_async).await
        }
        "get_job" => {
            let client = resolve_client(state, explicit_url).await?;
            let id_str = args
                .get("id")
                .and_then(Value::as_str)
                .ok_or("missing required 'id'")?;
            let id = Uuid::parse_str(id_str).map_err(|e| format!("invalid UUID: {e}"))?;
            let job = client.job_status(id).await.map_err(|e| e.to_string())?;
            serde_json::to_value(job).map_err(|e| e.to_string())
        }
        "push_file" => {
            let client = resolve_client(state, explicit_url).await?;
            let local_str = args
                .get("local_path")
                .and_then(Value::as_str)
                .ok_or("missing required 'local_path'")?;
            let local_path = Path::new(local_str);
            let remote_path = match args.get("remote_path").and_then(Value::as_str) {
                Some(r) => r.to_string(),
                None => local_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or("local_path has no valid UTF-8 filename")?
                    .to_string(),
            };
            let chunk_mb = args.get("chunk_mb").and_then(Value::as_u64).unwrap_or(4) as usize;
            let summary = client
                .upload_file(local_path, &remote_path, chunk_mb.max(1) * 1024 * 1024, |_| {})
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({
                "remote_path": summary.remote_path,
                "total_bytes": summary.total_bytes,
                "resumed_bytes": summary.resumed_bytes,
                "uploaded_bytes": summary.uploaded_bytes,
                "elapsed_secs": summary.elapsed.as_secs_f64(),
                "already_existed": summary.already_existed,
            }))
        }
        "pull_file" => {
            let client = resolve_client(state, explicit_url).await?;
            let remote_path = args
                .get("remote_path")
                .and_then(Value::as_str)
                .ok_or("missing required 'remote_path'")?;
            let local_path = args
                .get("local_path")
                .and_then(Value::as_str)
                .ok_or("missing required 'local_path'")?;
            let summary = client
                .download_file(remote_path, Path::new(local_path), |_| {})
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({
                "remote_path": summary.remote_path,
                "local_path": local_path,
                "total_bytes": summary.total_bytes,
                "resumed_bytes": summary.resumed_bytes,
                "downloaded_bytes": summary.uploaded_bytes,
                "elapsed_secs": summary.elapsed.as_secs_f64(),
                "already_existed": summary.already_existed,
            }))
        }
        "list_models" => {
            let client = resolve_client(state, explicit_url).await?;
            let models = client.list_models().await.map_err(|e| e.to_string())?;
            serde_json::to_value(models).map_err(|e| e.to_string())
        }
        "cancel_job" => {
            let client = resolve_client(state, explicit_url).await?;
            let id_str = args
                .get("id")
                .and_then(Value::as_str)
                .ok_or("missing required 'id'")?;
            let id = Uuid::parse_str(id_str).map_err(|e| format!("invalid UUID: {e}"))?;
            let job = client.cancel_job(id).await.map_err(|e| e.to_string())?;
            serde_json::to_value(job).map_err(|e| e.to_string())
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cli() -> Cli {
        Cli {
            url: None,
            token: None,
            discovery_port: DEFAULT_DISCOVERY_PORT,
        }
    }

    #[tokio::test]
    async fn initialize_and_tools_list_return_valid_mcp_payloads() {
        let state = McpState::new(test_cli());
        let init = handle_request(&state, json!(1), "initialize", json!({})).await;
        assert!(init.error.is_none());
        assert_eq!(init.result.unwrap()["serverInfo"]["name"], "offload-mcp");

        let list = handle_request(&state, json!(2), "tools/list", json!({})).await;
        assert!(list.error.is_none());
        let tools = list.result.unwrap()["tools"].as_array().unwrap().len();
        assert_eq!(tools, 10);
    }
}
