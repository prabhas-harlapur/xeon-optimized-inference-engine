# Xeon Optimized Inference Engine

CPU-first GenAI inference platform inspired by vLLM, built for Intel Xeon 6 with explicit hooks for AMX/AVX-512 optimization, continuous batching, model lifecycle control, benchmarking, and Grafana observability.

## What is implemented

- FastAPI-based serving layer with OpenAI-compatible endpoints:
  - `POST /v1/completions`
  - `POST /v1/chat/completions`
- Control-plane endpoints for model lifecycle:
  - `POST /control/models/load`
  - `POST /control/models/unload`
  - `GET /control/models`
  - `POST /control/models/select`
- Runtime architecture for CPU-first scheduling:
  - Dynamic micro-batching hooks
  - KV-cache accounting hooks
  - Xeon optimization profile resolver (AMX/AVX-512, thread pinning config)
- Benchmark runner:
  - Concurrency sweep: `1..100`
  - Token presets: `100, 512, 1024, 2048, 4096, 8192`
  - Latency/throughput report CSV + JSON
- Prometheus + Grafana stack:
  - Engine metrics and host hardware telemetry (CPU/core, memory, disk, NUMA if available)
  - Prebuilt Grafana dashboard JSON and provisioning
- Dockerized local setup via `docker-compose.yml`

## Quick start

1. Create Python environment and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CPU-only PyTorch wheels on Linux Xeon hosts:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install intel-extension-for-pytorch==2.8.0
```

2. Run the engine:

```powershell
uvicorn engine.xeon_inference.main:app --host 0.0.0.0 --port 8000
```

3. Load/select a model:

```powershell
curl -X POST http://localhost:8000/control/models/load -H "Content-Type: application/json" -d '{"model_id":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","alias":"tinyllama","dtype":"bfloat16"}'
curl -X POST http://localhost:8000/control/models/select -H "Content-Type: application/json" -d '{"alias":"tinyllama"}'
```

4. Run benchmark matrix:

```powershell
python bench/benchmark.py --base-url http://localhost:8000 --model tinyllama --max-concurrency 100 --tokens 100 512 1024 2048 4096 8192
```

Feature flags are enabled by default (`prefix_caching`, `staggered_loading`, `continuous_batching`, `paged_kv_cache`, `speculative_decoding`).
Disable any feature per run:

```powershell
python bench/benchmark.py --base-url http://localhost:8000 --model tinyllama --disable-prefix-caching --disable-staggered-loading
```

5. Start observability stack:

```powershell
docker compose up -d
```

Grafana: `http://localhost:3000` (`admin` / `admin`)
