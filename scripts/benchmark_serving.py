"""Asynchronous load-testing benchmark for BentoML adaptive dynamic batching."""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

# Ensure repository root is on sys.path for direct CLI execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import httpx
import numpy as np
import torch
import yaml

from scripts.client_example import load_image_as_base64


def compute_run_metrics(
    latencies_ms: list[float],
    batch_sizes: list[int],
    elapsed_s: float,
) -> dict[str, float]:
    """Computes throughput (samples/s) and latency percentiles for one run."""
    if not latencies_ms or elapsed_s <= 0.0:
        return {
            "num_requests": 0.0,
            "throughput_samples_per_s": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "mean_batch_size": 0.0,
        }

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_bs = float(np.mean(batch_sizes)) if batch_sizes else 1.0
    return {
        "num_requests": float(len(latencies_ms)),
        "throughput_samples_per_s": float(len(latencies_ms) / elapsed_s),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_batch_size": mean_bs,
    }


def aggregate_repeats(runs: list[dict[str, float]]) -> dict[str, float]:
    """Computes mean and standard deviation across repeated benchmark runs."""
    if not runs:
        return {}

    keys = [
        "throughput_samples_per_s",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_batch_size",
    ]
    agg: dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in runs]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return agg


def format_markdown_table(
    summary: dict[str, dict[int, dict[str, float]]],
    concurrency_levels: list[int],
) -> str:
    """Formats the HW2 comparison table in GitHub-flavored Markdown."""
    lines = [
        "| Concurrency | Throughput no batching [samples/s] | Throughput with batching [samples/s] | p95 latency no batching [ms] | p95 latency with batching [ms] |",
        "|---|---|---|---|---|",
    ]
    no_b = summary.get("no_batching", {})
    with_b = summary.get("adaptive_batching", {})

    for c in concurrency_levels:
        nb = no_b.get(c, {})
        wb = with_b.get(c, {})
        t_nb = f"{nb.get('throughput_samples_per_s_mean', 0.0):.2f} ± {nb.get('throughput_samples_per_s_std', 0.0):.2f}"
        t_wb = f"{wb.get('throughput_samples_per_s_mean', 0.0):.2f} ± {wb.get('throughput_samples_per_s_std', 0.0):.2f}"
        p95_nb = f"{nb.get('p95_ms_mean', 0.0):.1f} ± {nb.get('p95_ms_std', 0.0):.1f}"
        p95_wb = f"{wb.get('p95_ms_mean', 0.0):.1f} ± {wb.get('p95_ms_std', 0.0):.1f}"
        lines.append(f"| {c} | {t_nb} | {t_wb} | {p95_nb} | {p95_wb} |")

    return "\n".join(lines)


async def _run_worker(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    stop_time: float,
    latencies_ms: list[float],
    batch_sizes: list[int],
    record: bool,
) -> None:
    """Continuously sends POST requests until stop_time is reached."""
    while time.perf_counter() < stop_time:
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200 and record:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt_ms)
                data = resp.json()
                if isinstance(data, list) and data:
                    batch_sizes.append(int(data[0].get("batch_size_processed", 1)))
        except httpx.HTTPError:
            continue


async def benchmark_window(
    base_url: str,
    payload: dict[str, Any],
    concurrency: int,
    warmup_s: float,
    duration_s: float,
) -> dict[str, float]:
    """Runs warm-up followed by a fixed-duration measurement window."""
    predict_url = f"{base_url}/predict"
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 64),
        max_keepalive_connections=max(concurrency * 2, 64),
    )
    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # 1. Warm-up phase (discarded)
        if warmup_s > 0:
            warmup_end = time.perf_counter() + warmup_s
            await asyncio.gather(
                *[
                    _run_worker(
                        client, predict_url, payload, warmup_end, [], [], record=False
                    )
                    for _ in range(concurrency)
                ]
            )

        # 2. Fixed measurement window
        latencies_ms: list[float] = []
        batch_sizes: list[int] = []
        t_start = time.perf_counter()
        measure_end = t_start + duration_s
        await asyncio.gather(
            *[
                _run_worker(
                    client,
                    predict_url,
                    payload,
                    measure_end,
                    latencies_ms,
                    batch_sizes,
                    record=True,
                )
                for _ in range(concurrency)
            ]
        )
        elapsed_s = time.perf_counter() - t_start

    return compute_run_metrics(latencies_ms, batch_sizes, elapsed_s)


def _wait_for_server(base_url: str, timeout_s: float = 60.0) -> dict[str, Any]:
    """Polls /metadata until the BentoML server is ready."""
    deadline = time.perf_counter() + timeout_s
    with httpx.Client(timeout=5.0) as client:
        while time.perf_counter() < deadline:
            try:
                resp = client.post(f"{base_url}/metadata", json={})
                if resp.status_code == 200:
                    return cast(dict[str, Any], resp.json())
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise TimeoutError(f"BentoML server at {base_url} did not become ready.")


def _spawn_bentoml_server(
    host: str,
    port: int,
    enable_batching: bool,
    max_batch_size: int,
    max_latency_ms: int,
) -> subprocess.Popen[str]:
    """Spawns a BentoML server subprocess with explicit batching env vars."""
    env = os.environ.copy()
    env["POSE_ENABLE_BATCHING"] = "1" if enable_batching else "0"
    env["POSE_MAX_BATCH_SIZE"] = str(max_batch_size)
    env["POSE_MAX_LATENCY_MS"] = str(max_latency_ms)

    cmd = [
        sys.executable,
        "-m",
        "bentoml",
        "serve",
        "src.serving.service:InBedPoseService",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _stop_server(proc: subprocess.Popen[str]) -> None:
    """Gracefully terminates a spawned BentoML server process tree."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def collect_hardware_info() -> dict[str, Any]:
    """Collects CPU, GPU, and Python runtime metadata for reproducibility."""
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only"
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the serving benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark BentoML Pose API with and without dynamic batching."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/serving.yaml",
        help="Path to serving configuration YAML.",
    )
    parser.add_argument(
        "--auto-spawn",
        action="store_true",
        help="Automatically spawn and stop BentoML servers for both modes.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="URL of an already-running BentoML server (when not using --auto-spawn).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["no_batching", "adaptive_batching"],
        default="adaptive_batching",
        help="Mode label when benchmarking a single pre-started --server-url.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=None,
        help="Concurrency levels to sweep (default: 1 8 32).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Measurement window duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=None,
        help="Warm-up duration in seconds (default: 15).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of repeated runs per concurrency level (default: 3).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke benchmark (warmup=2s, duration=5s, repeats=1).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark results JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = cast(dict[str, Any], yaml.safe_load(f))

    serving_cfg = cfg.get("serving", {})
    batch_cfg = serving_cfg.get("batching", {})
    bench_cfg = cfg.get("benchmark", {})

    host = str(serving_cfg.get("host", "127.0.0.1"))
    port = int(serving_cfg.get("port", 3000))
    max_batch_size = int(batch_cfg.get("max_batch_size", 32))
    max_latency_ms = int(batch_cfg.get("max_latency_ms", 50))

    concurrency_levels = args.concurrency or bench_cfg.get(
        "concurrency_levels", [1, 8, 32]
    )
    duration_s = (
        2.0
        if args.quick
        else (args.duration or float(bench_cfg.get("duration_seconds", 60)))
    )
    warmup_s = (
        1.0
        if args.quick
        else (args.warmup or float(bench_cfg.get("warmup_seconds", 15)))
    )
    repeats = 1 if args.quick else (args.repeats or int(bench_cfg.get("repeats", 3)))
    output_json = Path(
        args.output_json
        or bench_cfg.get("output_json", "docs/hw2_benchmark_results.json")
    )

    img_b64 = load_image_as_base64(bench_cfg.get("sample_image_path"))
    payload = {"images_b64": [img_b64]}

    modes_to_run: list[tuple[str, bool]] = (
        [("no_batching", False), ("adaptive_batching", True)]
        if args.auto_spawn
        else [(args.mode, args.mode == "adaptive_batching")]
    )

    all_runs: dict[str, dict[int, list[dict[str, float]]]] = {}
    summary: dict[str, dict[int, dict[str, float]]] = {}
    server_metadata: dict[str, Any] = {}

    for mode_idx, (mode_name, enable_batching) in enumerate(modes_to_run):
        print(f"\n=== Benchmarking mode: {mode_name} ===")
        proc: subprocess.Popen[str] | None = None
        mode_port = port + mode_idx
        base_url = args.server_url or f"http://{host}:{mode_port}"

        try:
            if args.auto_spawn:
                proc = _spawn_bentoml_server(
                    host=host,
                    port=mode_port,
                    enable_batching=enable_batching,
                    max_batch_size=max_batch_size,
                    max_latency_ms=max_latency_ms,
                )
            meta = _wait_for_server(base_url)
            server_metadata[mode_name] = meta
            print(f"Server ready: {meta}")

            all_runs[mode_name] = {}
            summary[mode_name] = {}

            for c in concurrency_levels:
                repeat_results: list[dict[str, float]] = []
                for r_idx in range(1, repeats + 1):
                    print(
                        f"  [concurrency={c:2d}] Repeat {r_idx}/{repeats} "
                        f"(warmup={warmup_s}s, window={duration_s}s)..."
                    )
                    res = asyncio.run(
                        benchmark_window(
                            base_url=base_url,
                            payload=payload,
                            concurrency=c,
                            warmup_s=warmup_s,
                            duration_s=duration_s,
                        )
                    )
                    repeat_results.append(res)
                    print(
                        f"    -> throughput={res['throughput_samples_per_s']:.2f} samples/s, "
                        f"p50={res['p50_ms']:.1f}ms, p95={res['p95_ms']:.1f}ms, "
                        f"p99={res['p99_ms']:.1f}ms, mean_batch={res['mean_batch_size']:.2f}"
                    )
                all_runs[mode_name][c] = repeat_results
                summary[mode_name][c] = aggregate_repeats(repeat_results)
        finally:
            if proc is not None:
                _stop_server(proc)
                time.sleep(2.0)

    table_md = format_markdown_table(summary, concurrency_levels)
    print("\n" + table_md + "\n")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hardware": collect_hardware_info(),
        "batching_params": {
            "max_batch_size": max_batch_size,
            "max_latency_ms": max_latency_ms,
        },
        "methodology": {
            "concurrency_levels": concurrency_levels,
            "warmup_seconds": warmup_s,
            "duration_seconds": duration_s,
            "repeats": repeats,
        },
        "server_metadata": server_metadata,
        "raw_runs": all_runs,
        "summary": summary,
        "markdown_table": table_md,
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved benchmark report to {output_json}")


if __name__ == "__main__":
    main()
