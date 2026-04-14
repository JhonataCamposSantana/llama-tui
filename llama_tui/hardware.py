import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class HardwareProfile:
    cpu_logical: int = 0
    cpu_physical: int = 0
    memory_total: int = 0
    memory_available: int = 0
    gpu_name: str = ''
    gpu_memory_total: int = 0
    gpu_memory_free: int = 0
    gpu_error: str = ''

    def has_usable_gpu(self) -> bool:
        return self.gpu_memory_free > 0

    def short_summary(self) -> str:
        total_gib = bytes_to_gib(self.memory_total)
        avail_gib = bytes_to_gib(self.memory_available)
        cpu = f'cpu={self.cpu_physical or "?"}c/{self.cpu_logical or "?"}t'
        ram = f'ram={avail_gib:.1f}/{total_gib:.1f}GiB'
        if self.has_usable_gpu():
            gpu_free = bytes_to_gib(self.gpu_memory_free)
            gpu_total = bytes_to_gib(self.gpu_memory_total)
            gpu = f'gpu={self.gpu_name or "detected"} {gpu_free:.1f}/{gpu_total:.1f}GiB'
        elif self.gpu_error:
            gpu = 'gpu=unavailable'
        else:
            gpu = 'gpu=none'
        return f'{cpu} {ram} {gpu}'

def bytes_to_gib(value: int) -> float:
    return float(value or 0) / float(1024**3)
def read_meminfo_bytes() -> Dict[str, int]:
    meminfo = Path('/proc/meminfo')
    values: Dict[str, int] = {}
    if not meminfo.exists():
        return values
    try:
        for line in meminfo.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                values[parts[0].rstrip(':')] = int(parts[1]) * 1024
    except Exception:
        return {}
    return values
def detect_cpu_counts() -> Tuple[int, int]:
    logical = os.cpu_count() or 1
    cpuinfo = Path('/proc/cpuinfo')
    if not cpuinfo.exists():
        return logical, max(1, logical // 2)

    physical_cores = set()
    current_physical = ''
    current_core = ''
    try:
        for line in cpuinfo.read_text(errors='replace').splitlines() + ['']:
            if not line.strip():
                if current_core:
                    physical_cores.add((current_physical or '0', current_core))
                current_physical = ''
                current_core = ''
                continue
            if ':' not in line:
                continue
            key, value = [part.strip() for part in line.split(':', 1)]
            if key == 'physical id':
                current_physical = value
            elif key == 'core id':
                current_core = value
    except Exception:
        return logical, max(1, logical // 2)

    physical = len(physical_cores) if physical_cores else max(1, logical // 2)
    return logical, max(1, min(physical, logical))
def probe_nvidia_gpu() -> Tuple[str, int, int, str]:
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return '', 0, 0, ''
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                '--query-gpu=name,memory.total,memory.free',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        return '', 0, 0, f'nvidia-smi failed: {exc}'

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or '').strip().splitlines()
        return '', 0, 0, detail[0] if detail else 'nvidia-smi failed'

    best_name = ''
    best_total = 0
    best_free = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            total = int(float(parts[1])) * 1024**2
            free = int(float(parts[2])) * 1024**2
        except ValueError:
            continue
        if free > best_free:
            best_name = parts[0]
            best_total = total
            best_free = free
    return best_name, best_total, best_free, ''
def benchmark_current_hardware() -> HardwareProfile:
    logical, physical = detect_cpu_counts()
    mem = read_meminfo_bytes()
    gpu_name, gpu_total, gpu_free, gpu_error = probe_nvidia_gpu()
    return HardwareProfile(
        cpu_logical=logical,
        cpu_physical=physical,
        memory_total=mem.get('MemTotal', 0),
        memory_available=mem.get('MemAvailable', 0),
        gpu_name=gpu_name,
        gpu_memory_total=gpu_total,
        gpu_memory_free=gpu_free,
        gpu_error=gpu_error,
    )
