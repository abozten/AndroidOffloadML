# AndroidOffload

A lightweight Rust server/client workspace for offloading ML and compute tasks from your host machine to **Termux** on Android (e.g., OnePlus 15).

## Architecture

- [`crates/protocol`](crates/protocol/src/lib.rs) (`offload-protocol`): Shared JSON/HTTP wire types (`TaskKind`, `TaskResult`, `JobRecord`, `NodeInfo`).
- [`crates/server`](crates/server/src/main.rs) (`offload-server`): Async Axum HTTP server intended to run inside Termux on Android. Features:
  - Built-in compute tasks (`echo`, `mat_mul`, `cosine_top_k`) executed on `spawn_blocking`.
  - `llama_completion` task to run `llama.cpp` (`llama-cli`) against models in `--models-dir` (with path-traversal protection).
  - `command` task to run allowlisted ML binaries/scripts (`--allow-cmd python3,whisper-cli`).
  - Concurrency limiting (`--max-concurrent`), optional Bearer token auth (`--token`), and both sync (`/v1/run`) and async (`/v1/jobs`) endpoints.
- [`crates/client`](crates/client/src/main.rs) (`offload-client`): Async Rust client library and CLI for your Mac/Linux/Windows host.

## 1. Cross-compile `offload-server` on your Mac

Build a single static ARM64 binary (configured in `.cargo/config.toml` via `rust-lld`):

```bash
rustup target add aarch64-unknown-linux-musl
cargo build --release -p offload-server --target aarch64-unknown-linux-musl
```

This produces a ~4.4 MB statically linked `aarch64` binary at `target/aarch64-unknown-linux-musl/release/offload-server`.

## 2. Copy & run on Termux over LAN (no ADB or Rust needed on phone)

On your **Mac**, serve the binary over LAN for a few seconds:
```bash
python3 -m http.server 8000 --directory target/aarch64-unknown-linux-musl/release
```

In **Termux** on your OnePlus 15, download it from your Mac's LAN IP and start the server:
```bash
curl -O http://<mac-lan-ip>:8000/offload-server
chmod +x offload-server
./offload-server \
  --bind 0.0.0.0:7878 \
  --token "your-secret-token" \
  --max-concurrent 1 \
  --models-dir ~/models \
  --allow-cmd python3,whisper-cli
```
*(You can stop the `python3 -m http.server` on your Mac once `curl` finishes. If you already run `sshd` in Termux, `scp` works too.)*

> Tip: In Termux, run `termux-wake-lock` before starting long ML jobs so Android does not throttle or sleep the process.

## 3. Run `offload-client` from your Mac over LAN (auto-discovery)

`offload-server` listens for UDP discovery probes on port `7879` (broadcast + multicast), so you don't even need to look up your phone's IP:

```bash
# List all discovered devices on the LAN
cargo run -p offload-client -- discover

# Or omit --url / OFFLOAD_URL to auto-discover the first node:
export OFFLOAD_TOKEN="your-secret-token"
cargo run -p offload-client -- info
```

*(You can still pass `--url http://<phone-lan-ip>:7878` or set `OFFLOAD_URL` to target a specific node directly.)*

# Push a large model or file with automatic resume & speed/ETA bar:
cargo run -p offload-client -- push ./qwen2.5-3b-instruct-q4_k_m.gguf

# Run an echo latency test or matrix-multiply benchmark:
```bash
cargo run -p offload-client -- echo "hello oneplus 15"
cargo run -p offload-client -- matmul-bench --size 512
```

Run `llama.cpp` inference on the phone (sync or async):
```bash
cargo run -p offload-client -- llama \
  --model qwen2.5-3b-instruct-q4_k_m.gguf \
  --prompt "Summarize why on-device NPU/CPU offload is useful." \
  --max-tokens 256 \
  --async
```

Run an allowlisted Python ML script on the phone:
```bash
cargo run -p offload-client -- cmd --program python3 -- ~/scripts/run_onnx.py --input data.json
```

## 4. Use with AI Agents (MCP + Skill)

Build the MCP stdio server once on your Mac:
```bash
cargo build --release -p offload-mcp
```

Add it to your agent's MCP config (e.g. `mcp_config.json`):
```json
{
  "mcpServers": {
    "android-offload": {
      "command": "/Users/abozten/Documents/Projects/AndroidOffload/target/release/offload-mcp",
      "env": {
        "OFFLOAD_TOKEN": "your-secret-token"
      }
    }
  }
}
```

This exposes `discover_nodes`, `node_info`, `run_llama`, `run_command`, `run_task`, and `get_job` directly as agent tools. A workspace skill is also included at `.agents/skills/android-offload/SKILL.md`.

If using `--token` on the server, pass `--token "your-secret-token"` (or set `OFFLOAD_TOKEN`) and `--url http://<phone-ip>:7878` (or set `OFFLOAD_URL`).
