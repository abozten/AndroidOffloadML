"""Zero-dependency Python client for AndroidOffload (`offload-server`).

Designed for offloading CPU-bound CASMI / ML kernels (MS/MS spectral similarity,
bit-packed fingerprint Tanimoto & dot-product scoring, dense cosine top-k, matmul,
and allowlisted Python/CLI tasks) to an Android/Termux node over the LAN.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Sequence

DEFAULT_DISCOVERY_PORT = 7879
DISCOVERY_MULTICAST_ADDR = "239.255.78.79"
DISCOVERY_PROBE_MAGIC = "android-offload-discover-v1"
DISCOVERY_BEACON_MAGIC = "android-offload-beacon-v1"


class OffloadError(RuntimeError):
    """Raised when discovery, transport, or remote task execution fails."""


def discover(timeout: float = 1.5, port: int = DEFAULT_DISCOVERY_PORT) -> list[dict[str, Any]]:
    """Broadcast a UDP discovery probe on the LAN and return discovered nodes."""
    probe = json.dumps({"magic": DISCOVERY_PROBE_MAGIC}).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    nodes: dict[str, dict[str, Any]] = {}
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.25)
        for addr in ("255.255.255.255", DISCOVERY_MULTICAST_ADDR):
            try:
                sock.sendto(probe, (addr, port))
            except OSError:
                pass

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data, (ip, _) = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                beacon = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if beacon.get("magic") != DISCOVERY_BEACON_MAGIC:
                continue
            url = f"http://{ip}:{beacon['http_port']}"
            nodes[url] = {
                "url": url,
                "name": beacon.get("name", ""),
                "version": beacon.get("version", ""),
                "os": beacon.get("os", ""),
                "arch": beacon.get("arch", ""),
                "logical_cpus": beacon.get("logical_cpus", 0),
                "auth_required": bool(beacon.get("auth_required", False)),
            }
    finally:
        sock.close()
    return list(nodes.values())


def discover_first(timeout: float = 1.5, port: int = DEFAULT_DISCOVERY_PORT) -> dict[str, Any]:
    """Return the first discovered node or raise OffloadError."""
    nodes = discover(timeout=timeout, port=port)
    if not nodes:
        raise OffloadError(f"no offload-server nodes discovered on UDP port {port}")
    return nodes[0]


class OffloadClient:
    """HTTP client for `offload-server`."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        resolved_url = url or os.environ.get("OFFLOAD_URL")
        if not resolved_url:
            resolved_url = discover_first()["url"]
        self.base_url = resolved_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("OFFLOAD_TOKEN")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OffloadError(f"HTTP {exc.code} on {method} {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OffloadError(f"request failed on {method} {path}: {exc.reason}") from exc

    def info(self) -> dict[str, Any]:
        """Fetch worker hardware and capability info (`GET /v1/info`)."""
        return self._request("GET", "/v1/info")

    def run_task(
        self,
        task: dict[str, Any],
        *,
        async_job: bool = False,
        poll_interval: float = 0.25,
    ) -> dict[str, Any]:
        """Execute a raw `TaskKind` payload synchronously or as an async job."""
        if not async_job:
            resp = self._request("POST", "/v1/run", {"task": task})
            return resp["result"]

        job = self._request("POST", "/v1/jobs", {"task": task})
        job_id = job["id"]
        while job["status"] in ("queued", "running"):
            time.sleep(poll_interval)
            job = self._request("GET", f"/v1/jobs/{job_id}")
        if job["status"] != "succeeded":
            raise OffloadError(f"job {job_id} {job['status']}: {job.get('error')}")
        return job["result"]

    def cosine_top_k(
        self,
        query: Sequence[float],
        corpus: Sequence[Sequence[float]],
        top_k: int = 10,
        *,
        async_job: bool = False,
    ) -> list[dict[str, Any]]:
        """Top-K dense float32 cosine similarity search."""
        task = {
            "kind": "cosine_top_k",
            "query": [float(x) for x in query],
            "corpus": [[float(x) for x in row] for row in corpus],
            "top_k": int(top_k),
        }
        return self.run_task(task, async_job=async_job)["matches"]

    def mat_mul(
        self,
        rows_a: int,
        cols_a: int,
        cols_b: int,
        a: Sequence[float],
        b: Sequence[float],
        *,
        async_job: bool = False,
    ) -> list[float]:
        """Dense float32 matrix multiplication."""
        task = {
            "kind": "mat_mul",
            "rows_a": int(rows_a),
            "cols_a": int(cols_a),
            "cols_b": int(cols_b),
            "a": [float(x) for x in a],
            "b": [float(x) for x in b],
        }
        return self.run_task(task, async_job=async_job)["data"]

    def run_command(
        self,
        program: str,
        args: Sequence[str] = (),
        *,
        stdin: str | None = None,
        timeout_secs: int | None = None,
        async_job: bool = False,
    ) -> dict[str, Any]:
        """Run an allowlisted command (e.g. `python3`) on the device."""
        task: dict[str, Any] = {
            "kind": "command",
            "program": program,
            "args": list(args),
        }
        if stdin is not None:
            task["stdin"] = stdin
        if timeout_secs is not None:
            task["timeout_secs"] = int(timeout_secs)
        return self.run_task(task, async_job=async_job)
