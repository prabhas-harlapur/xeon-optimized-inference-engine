from __future__ import annotations

import os
import threading
import time

import psutil
from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter("xeon_requests_total", "Total inference requests", ["endpoint", "model"])
TOKENS_IN = Counter("xeon_tokens_input_total", "Input tokens", ["model"])
TOKENS_OUT = Counter("xeon_tokens_output_total", "Output tokens", ["model"])
LATENCY_SECONDS = Histogram(
    "xeon_request_latency_seconds",
    "Inference request latency",
    ["endpoint", "model"],
    buckets=(0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

CPU_PERCENT = Gauge("xeon_host_cpu_percent", "Host CPU percent")
MEMORY_USED_BYTES = Gauge("xeon_host_memory_used_bytes", "Host memory used bytes")
MEMORY_TOTAL_BYTES = Gauge("xeon_host_memory_total_bytes", "Host memory total bytes")
DISK_USED_BYTES = Gauge("xeon_host_disk_used_bytes", "Host disk used bytes")
DISK_TOTAL_BYTES = Gauge("xeon_host_disk_total_bytes", "Host disk total bytes")
CORE_CPU_PERCENT = Gauge("xeon_core_cpu_percent", "Per-core CPU percent", ["core"])
NUMA_MEMORY_TOTAL_BYTES = Gauge("xeon_numa_memory_total_bytes", "NUMA node total memory bytes", ["node"])
NUMA_MEMORY_FREE_BYTES = Gauge("xeon_numa_memory_free_bytes", "NUMA node free memory bytes", ["node"])
CPU_CACHE_SIZE_BYTES = Gauge("xeon_cpu_cache_size_bytes", "CPU cache size bytes", ["cpu", "index", "level", "type"])


def _parse_mem_kb(line: str) -> float:
    parts = line.split(":")
    if len(parts) != 2:
        return 0.0
    val = parts[1].strip().split(" ")[0]
    try:
        return float(val) * 1024.0
    except ValueError:
        return 0.0


def _sample_numa_memory() -> None:
    base = "/sys/devices/system/node"
    if not os.path.isdir(base):
        return
    for node in os.listdir(base):
        if not node.startswith("node"):
            continue
        node_id = node.replace("node", "")
        meminfo = os.path.join(base, node, "meminfo")
        if not os.path.exists(meminfo):
            continue
        with open(meminfo, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = 0.0
        free = 0.0
        for line in lines:
            if "MemTotal" in line:
                total = _parse_mem_kb(line)
            if "MemFree" in line:
                free = _parse_mem_kb(line)
        NUMA_MEMORY_TOTAL_BYTES.labels(node=node_id).set(total)
        NUMA_MEMORY_FREE_BYTES.labels(node=node_id).set(free)


def _sample_cache_sizes() -> None:
    cpu_root = "/sys/devices/system/cpu"
    if not os.path.isdir(cpu_root):
        return
    for cpu in os.listdir(cpu_root):
        if not cpu.startswith("cpu") or not cpu[3:].isdigit():
            continue
        cache_root = os.path.join(cpu_root, cpu, "cache")
        if not os.path.isdir(cache_root):
            continue
        for index in os.listdir(cache_root):
            if not index.startswith("index"):
                continue
            index_path = os.path.join(cache_root, index)
            level_path = os.path.join(index_path, "level")
            type_path = os.path.join(index_path, "type")
            size_path = os.path.join(index_path, "size")
            if not (os.path.exists(level_path) and os.path.exists(type_path) and os.path.exists(size_path)):
                continue
            with open(level_path, "r", encoding="utf-8") as f:
                level = f.read().strip()
            with open(type_path, "r", encoding="utf-8") as f:
                cache_type = f.read().strip()
            with open(size_path, "r", encoding="utf-8") as f:
                size_txt = f.read().strip().upper()
            factor = 1
            if size_txt.endswith("K"):
                factor = 1024
            elif size_txt.endswith("M"):
                factor = 1024 * 1024
            try:
                size_bytes = float(size_txt[:-1]) * factor
            except ValueError:
                continue
            CPU_CACHE_SIZE_BYTES.labels(cpu=cpu, index=index, level=level, type=cache_type).set(size_bytes)


class HostSampler:
    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sampled_static = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._sampled_static:
                _sample_numa_memory()
                _sample_cache_sizes()
                self._sampled_static = True
            vm = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            CPU_PERCENT.set(psutil.cpu_percent(interval=None))
            MEMORY_USED_BYTES.set(vm.used)
            MEMORY_TOTAL_BYTES.set(vm.total)
            DISK_USED_BYTES.set(disk.used)
            DISK_TOTAL_BYTES.set(disk.total)
            for idx, val in enumerate(psutil.cpu_percent(interval=None, percpu=True)):
                CORE_CPU_PERCENT.labels(core=str(idx)).set(val)
            time.sleep(self.interval_seconds)
