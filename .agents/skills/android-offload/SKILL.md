---
name: android-offload
description: Offload ML inference, Python scripts, or compute tasks to an Android/Termux device on the LAN using offload-mcp or offload-client.
---

# Android/Termux ML Offload

Use this skill when offloading local LLM inference (`llama.cpp`), Python/ONNX scripts, or compute tasks to the user's Android/Termux node (e.g., OnePlus 15) on the LAN.

## Preferred: MCP Tools (`offload-mcp`)

If the `android-offload` MCP server is connected, call its tools directly:
1. `discover_nodes` — find active `offload-server` devices on the LAN.
2. `node_info` — inspect `logical_cpus`, `models_dir`, and `allowed_commands`.
3. `run_llama` — run `llama.cpp` inference (`async: true` by default).
4. `run_command` — run an allowlisted command (`python3`, `whisper-cli`, etc.).
5. `run_task` — run a raw `TaskKind` JSON payload (`echo`, `mat_mul`, `cosine_top_k`, `llama_completion`, `command`).

## CLI Fallback (`offload-client`)

If MCP is not configured, use `offload-client` (auto-discovers on LAN when `--url` / `OFFLOAD_URL` is omitted; pass `OFFLOAD_TOKEN` if the server requires auth):

```bash
cargo run -q -p offload-client -- discover
cargo run -q -p offload-client -- info
cargo run -q -p offload-client -- llama --model <model.gguf> --prompt "<prompt>" --async
cargo run -q -p offload-client -- cmd --program python3 -- <script.py> <args...>
```
