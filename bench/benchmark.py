from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pandas as pd

TOKEN_PRESETS = [100, 512, 1024, 2048, 4096, 8192]

DEFAULT_FEATURE_FLAGS = {
    "prefix_caching": True,
    "staggered_loading": True,
    "continuous_batching": True,
    "paged_kv_cache": True,
    "speculative_decoding": True,
}


@dataclass
class TrialResult:
    concurrency: int
    input_tokens: int
    output_tokens: int
    requests: int
    success: int
    failed: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    throughput_rps: float
    feature_flags: str


def make_prompt(token_count: int) -> str:
    words = max(1, token_count // 2)
    return " ".join(["benchmark"] * words)


async def run_case(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    requests: int,
    feature_flags: dict[str, bool],
) -> TrialResult:
    latencies: list[float] = []
    success = 0
    failed = 0
    sem = asyncio.Semaphore(concurrency)

    async def one_request() -> None:
        nonlocal success, failed
        payload = {
            "model": model,
            "prompt": make_prompt(input_tokens),
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "feature_flags": feature_flags,
        }
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{base_url}/v1/completions", json=payload, timeout=300)
                if r.status_code == 200:
                    success += 1
                    latencies.append((time.perf_counter() - t0) * 1000)
                else:
                    failed += 1
            except Exception:
                failed += 1

    t0 = time.perf_counter()
    await asyncio.gather(*[one_request() for _ in range(requests)])
    elapsed = max(1e-9, time.perf_counter() - t0)

    latencies_sorted = sorted(latencies)

    def pct(p: float) -> float:
        if not latencies_sorted:
            return math.nan
        i = min(len(latencies_sorted) - 1, int(round((p / 100.0) * (len(latencies_sorted) - 1))))
        return latencies_sorted[i]

    return TrialResult(
        concurrency=concurrency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        requests=requests,
        success=success,
        failed=failed,
        p50_ms=pct(50),
        p90_ms=pct(90),
        p99_ms=pct(99),
        throughput_rps=success / elapsed,
        feature_flags=json.dumps(feature_flags, sort_keys=True),
    )


def resolve_feature_flags(args: argparse.Namespace) -> dict[str, bool]:
    flags = dict(DEFAULT_FEATURE_FLAGS)
    if args.disable_prefix_caching:
        flags["prefix_caching"] = False
    if args.disable_staggered_loading:
        flags["staggered_loading"] = False
    if args.disable_continuous_batching:
        flags["continuous_batching"] = False
    if args.disable_paged_kv_cache:
        flags["paged_kv_cache"] = False
    if args.disable_speculative_decoding:
        flags["speculative_decoding"] = False
    return flags


async def main() -> None:
    parser = argparse.ArgumentParser(description="Xeon inference benchmark matrix")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--min-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--concurrency-step", type=int, default=5)
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKEN_PRESETS)
    parser.add_argument("--requests-per-case", type=int, default=100)
    parser.add_argument("--output-dir", default="bench/results")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--disable-staggered-loading", action="store_true")
    parser.add_argument("--disable-continuous-batching", action="store_true")
    parser.add_argument("--disable-paged-kv-cache", action="store_true")
    parser.add_argument("--disable-speculative-decoding", action="store_true")
    args = parser.parse_args()

    feature_flags = resolve_feature_flags(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concurrency_levels = list(range(args.min_concurrency, args.max_concurrency + 1, args.concurrency_step))
    if args.max_concurrency not in concurrency_levels:
        concurrency_levels.append(args.max_concurrency)

    results: list[TrialResult] = []
    async with httpx.AsyncClient() as client:
        for c in concurrency_levels:
            for t in args.tokens:
                print(f"Running concurrency={c}, tokens={t}, features={feature_flags}")
                result = await run_case(
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    concurrency=c,
                    input_tokens=t,
                    output_tokens=t,
                    requests=args.requests_per_case,
                    feature_flags=feature_flags,
                )
                results.append(result)

    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_matrix.csv"
    json_path = output_dir / "benchmark_matrix.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
