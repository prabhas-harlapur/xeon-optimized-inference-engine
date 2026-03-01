# Xeon 6 Validation Playbook

Use this on the Xeon 6 host for controlled, repeatable validation.

## 1. Environment

```bash
export OMP_NUM_THREADS=<physical_cores_per_socket>
export KMP_AFFINITY=granularity=fine,compact,1,0
export TORCH_INTRA_OP_THREADS=<physical_cores_per_socket>
export TORCH_INTER_OP_THREADS=1
```

Optional single-socket pinning:

```bash
numactl --cpunodebind=0 --membind=0 uvicorn engine.xeon_inference.main:app --host 0.0.0.0 --port 8000
```

## 2. Model lifecycle

```bash
curl -X POST http://localhost:8000/control/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","alias":"tinyllama","dtype":"bfloat16"}'

curl -X POST http://localhost:8000/control/models/select \
  -H "Content-Type: application/json" \
  -d '{"alias":"tinyllama"}'
```

## 3. Benchmark matrix

All advanced features are enabled by default.

```bash
python bench/benchmark.py \
  --base-url http://localhost:8000 \
  --model tinyllama \
  --min-concurrency 1 \
  --max-concurrency 100 \
  --concurrency-step 1 \
  --tokens 100 512 1024 2048 4096 8192
```

Disable specific features when needed:

```bash
python bench/benchmark.py --base-url http://localhost:8000 --model tinyllama \
  --disable-prefix-caching --disable-staggered-loading
```

## 4. Observability

```bash
docker compose up -d
```

Grafana dashboards include throughput, latency, per-core CPU, memory, disk, NUMA memory, and cache size signals.
