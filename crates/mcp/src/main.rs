//! `offload-mcp`: Model Context Protocol server over stdio for `offload-server`.
//!
//! Reads newline-delimited JSON-RPC 2.0 from stdin and writes responses to stdout.
//! Each `tools/call` maps to one [`OffloadClient`] operation; when no `url` is given,
//! the node is auto-discovered once and cached until a call fails.

use std::{error::Error, path::Path, sync::Arc, time::Duration};

use clap::Parser;
use offload_client::{DEFAULT_POLL_INTERVAL, OffloadClient, discover, discover_first};
use offload_protocol::{DEFAULT_DISCOVERY_PORT, DEFAULT_DISCOVERY_WAIT, TaskKind};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::{Mutex, mpsc},
};
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(
    name = "offload-mcp",
    about = "MCP stdio server for Android/Termux ML offload"
)]
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

/// MCP revision this server implements; clients negotiate against it in `initialize`.
const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC 2.0 error code for unparseable input.
const PARSE_ERROR: i64 = -32700;
/// JSON-RPC 2.0 error code for an unknown method.
const METHOD_NOT_FOUND: i64 = -32601;

/// Responses buffered between request handlers and the single stdout writer.
/// Only back-pressures if the MCP host stops reading stdout.
const RESPONSE_QUEUE_CAPACITY: usize = 128;

/// Tool errors are rendered to text for the MCP host, so any error type will do.
type ToolResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

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

#[derive(Debug)]
struct McpState {
    cli: Cli,
    default_client: Mutex<Option<OffloadClient>>,
}

impl McpState {
    fn new(cli: Cli) -> Self {
        Self {
            cli,
            default_client: Mutex::new(None),
        }
    }
}

#[tokio::main]
async fn main() {
    let state = Arc::new(McpState::new(Cli::parse()));
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let (tx, mut rx) = mpsc::channel::<JsonRpcResponse>(RESPONSE_QUEUE_CAPACITY);

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
                        code: PARSE_ERROR,
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

async fn handle_request(
    state: &McpState,
    id: Value,
    method: &str,
    params: Value,
) -> JsonRpcResponse {
    match method {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
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
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let uses_default = args.get("url").and_then(Value::as_str).is_none();
            let outcome = call_tool(state, name, args).await;
            if outcome.is_err() && uses_default && state.cli.url.is_none() {
                // ponytail: any error drops the discovered node, so the next call rediscovers
                // (the phone may have a new DHCP address). Costs one extra discovery after bad-arg errors.
                *state.default_client.lock().await = None;
            }
            let result = match outcome {
                Ok(val) => json!({
                    "content": [{ "type": "text", "text": serde_json::to_string_pretty(&val).unwrap_or_default() }],
                    "isError": false
                }),
                Err(err) => json!({
                    "content": [{ "type": "text", "text": err.to_string() }],
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
                code: METHOD_NOT_FOUND,
                message: format!("method not found: {other}"),
            }),
        },
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "static JSON schema data, one entry per tool"
)]
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

async fn resolve_client(state: &McpState, explicit_url: Option<&str>) -> ToolResult<OffloadClient> {
    if let Some(url) = explicit_url {
        return Ok(OffloadClient::new(url, state.cli.token.clone())?);
    }
    // Lock held across discovery so concurrent calls share one discovery round.
    let mut cached = state.default_client.lock().await;
    if let Some(client) = cached.as_ref() {
        return Ok(client.clone());
    }
    let url = match state.cli.url.as_deref() {
        Some(u) => u.to_string(),
        None => {
            discover_first(state.cli.discovery_port, DEFAULT_DISCOVERY_WAIT)
                .await?
                .url
        }
    };
    let client = OffloadClient::new(url, state.cli.token.clone())?;
    *cached = Some(client.clone());
    Ok(client)
}

async fn execute_task(
    client: &OffloadClient,
    task: TaskKind,
    use_async: bool,
) -> ToolResult<Value> {
    Ok(if use_async {
        serde_json::to_value(client.run_and_wait(task, DEFAULT_POLL_INTERVAL).await?)?
    } else {
        serde_json::to_value(client.run(task).await?)?
    })
}

fn str_arg<'a>(args: &'a Value, key: &str) -> ToolResult<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing required '{key}'").into())
}

fn bool_arg(args: &Value, key: &str, default: bool) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn uuid_arg(args: &Value) -> ToolResult<Uuid> {
    Ok(Uuid::parse_str(str_arg(args, "id")?).map_err(|e| format!("invalid UUID: {e}"))?)
}

async fn call_tool(state: &McpState, name: &str, args: Value) -> ToolResult<Value> {
    let explicit_url = args.get("url").and_then(Value::as_str);

    if name == "discover_nodes" {
        let wait = args
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_DISCOVERY_WAIT, Duration::from_millis);
        return Ok(serde_json::to_value(
            discover(state.cli.discovery_port, wait).await?,
        )?);
    }

    let client = || resolve_client(state, explicit_url);

    match name {
        "node_info" => Ok(serde_json::to_value(client().await?.info().await?)?),
        "list_models" => Ok(serde_json::to_value(client().await?.list_models().await?)?),
        "get_job" => {
            let id = uuid_arg(&args)?;
            Ok(serde_json::to_value(client().await?.job_status(id).await?)?)
        }
        "cancel_job" => {
            let id = uuid_arg(&args)?;
            Ok(serde_json::to_value(client().await?.cancel_job(id).await?)?)
        }
        "run_llama" => {
            let task = TaskKind::LlamaCompletion {
                model: str_arg(&args, "model")?.to_string(),
                prompt: str_arg(&args, "prompt")?.to_string(),
                max_tokens: args
                    .get("max_tokens")
                    .and_then(Value::as_u64)
                    .map_or(Ok(256), u32::try_from)?,
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "f64 -> f32 temperature; precision loss is harmless"
                )]
                temperature: args
                    .get("temperature")
                    .and_then(Value::as_f64)
                    .map(|t| t as f32),
                extra_args: vec![],
            };
            execute_task(&client().await?, task, bool_arg(&args, "async", true)).await
        }
        "run_command" => {
            let task = TaskKind::Command {
                program: str_arg(&args, "program")?.to_string(),
                args: args
                    .get("args")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|e| format!("invalid 'args': {e}"))?
                    .unwrap_or_default(),
                stdin: args
                    .get("stdin")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                timeout_secs: args.get("timeout_secs").and_then(Value::as_u64),
            };
            execute_task(&client().await?, task, bool_arg(&args, "async", false)).await
        }
        "run_task" => {
            let task_val = args.get("task").cloned().ok_or("missing required 'task'")?;
            let task: TaskKind =
                serde_json::from_value(task_val).map_err(|e| format!("invalid 'task': {e}"))?;
            execute_task(&client().await?, task, bool_arg(&args, "async", false)).await
        }
        "push_file" => push_file(&client().await?, &args).await,
        "pull_file" => pull_file(&client().await?, &args).await,
        other => Err(format!("unknown tool: {other}").into()),
    }
}

async fn push_file(client: &OffloadClient, args: &Value) -> ToolResult<Value> {
    let local_path = Path::new(str_arg(args, "local_path")?);
    let remote_path = match args.get("remote_path").and_then(Value::as_str) {
        Some(r) => r.to_string(),
        None => local_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("local_path has no valid UTF-8 filename")?
            .to_string(),
    };
    let chunk_mb = usize::try_from(args.get("chunk_mb").and_then(Value::as_u64).unwrap_or(4))?;
    let summary = client
        .upload_file(
            local_path,
            &remote_path,
            chunk_mb.max(1) * 1024 * 1024,
            |_| {},
        )
        .await?;
    Ok(json!({
        "remote_path": summary.remote_path,
        "total_bytes": summary.total_bytes,
        "resumed_bytes": summary.resumed_bytes,
        "uploaded_bytes": summary.uploaded_bytes,
        "elapsed_secs": summary.elapsed.as_secs_f64(),
        "already_existed": summary.already_existed,
    }))
}

async fn pull_file(client: &OffloadClient, args: &Value) -> ToolResult<Value> {
    let remote_path = str_arg(args, "remote_path")?;
    let local_path = str_arg(args, "local_path")?;
    let summary = client
        .download_file(remote_path, Path::new(local_path), |_| {})
        .await?;
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
    }

    #[tokio::test]
    async fn every_listed_tool_is_dispatched() {
        // Port 9 (discard) refuses connections, so each call fails fast after dispatch.
        let state = McpState::new(test_cli());
        for tool in tool_definitions().as_array().unwrap() {
            let name = tool["name"].as_str().unwrap();
            let args = json!({ "url": "http://127.0.0.1:9", "timeout_ms": 1 });
            if let Err(err) = call_tool(&state, name, args).await {
                assert!(
                    !err.to_string().starts_with("unknown tool"),
                    "{name} is listed but not handled"
                );
            }
        }
        let err = call_tool(&state, "nope", json!({})).await.unwrap_err();
        assert!(err.to_string().starts_with("unknown tool"));
    }
}
