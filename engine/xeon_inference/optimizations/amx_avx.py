import os
import platform
from dataclasses import dataclass


@dataclass
class XeonOptimizationProfile:
    amx_available: bool
    avx512_available: bool
    omp_num_threads: int
    kmp_affinity: str
    intra_op_threads: int
    inter_op_threads: int


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback


def detect_isa_flags() -> tuple[bool, bool]:
    amx = False
    avx512 = False
    cpuinfo_path = "/proc/cpuinfo"
    if os.path.exists(cpuinfo_path):
        with open(cpuinfo_path, "r", encoding="utf-8") as f:
            txt = f.read().lower()
        amx = ("amx_tile" in txt) or ("amx_int8" in txt) or ("amx_bf16" in txt)
        avx512 = "avx512f" in txt
    return amx, avx512


def resolve_profile() -> XeonOptimizationProfile:
    amx, avx512 = detect_isa_flags()
    cores = os.cpu_count() or 1
    return XeonOptimizationProfile(
        amx_available=amx,
        avx512_available=avx512,
        omp_num_threads=_env_int("OMP_NUM_THREADS", cores),
        kmp_affinity=os.getenv("KMP_AFFINITY", "granularity=fine,compact,1,0"),
        intra_op_threads=_env_int("TORCH_INTRA_OP_THREADS", max(1, cores - 2)),
        inter_op_threads=_env_int("TORCH_INTER_OP_THREADS", 1),
    )


def host_summary() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
