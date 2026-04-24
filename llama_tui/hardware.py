import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


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


@dataclass
class ProcessPressureSnapshot:
    timestamp: str = ''
    load_1m: float = 0.0
    load_5m: float = 0.0
    load_15m: float = 0.0
    runnable_processes: int = 0
    total_processes: int = 0
    cpu_logical: int = 0
    load_ratio: float = 0.0
    memory_total: int = 0
    memory_available: int = 0
    gpu_memory_total: int = 0
    gpu_memory_free: int = 0
    process_count: int = 0
    known_processes: Dict[str, int] = field(default_factory=dict)
    known_memory: Dict[str, int] = field(default_factory=dict)
    top_memory: List[Dict[str, object]] = field(default_factory=list)
    top_cpu: List[Dict[str, object]] = field(default_factory=list)
    gpu_processes: List[Dict[str, object]] = field(default_factory=list)
    pressure_score: float = 0.0
    pressure_level: str = 'low'
    detail: str = ''

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


KNOWN_PROCESS_PATTERNS = {
    'browser': ('chrome', 'chromium', 'firefox', 'brave', 'vivaldi', 'edge'),
    'ide': ('code', 'codium', 'cursor', 'pycharm', 'idea', 'webstorm', 'zed'),
    'terminal': ('gnome-terminal', 'konsole', 'kitty', 'alacritty', 'wezterm', 'foot', 'xterm', 'ptyxis'),
    'container': ('docker', 'containerd', 'podman', 'distrobox', 'flatpak'),
    'llama': ('llama-server', 'llama-cli', 'llama.cpp', 'llmfit'),
    'ollama': ('ollama',),
    'vllm': ('vllm',),
    'opencode': ('opencode',),
    'hermes': ('hermes',),
}


def _read_loadavg(proc_root: Path) -> Tuple[float, float, float, int, int]:
    try:
        parts = (proc_root / 'loadavg').read_text(encoding='utf-8', errors='replace').split()
        load_1m = float(parts[0])
        load_5m = float(parts[1])
        load_15m = float(parts[2])
        runnable = 0
        total = 0
        if len(parts) >= 4 and '/' in parts[3]:
            left, _, right = parts[3].partition('/')
            runnable = int(left or 0)
            total = int(right or 0)
        return load_1m, load_5m, load_15m, runnable, total
    except Exception:
        return 0.0, 0.0, 0.0, 0, 0


def _read_process_stat(stat_text: str) -> Tuple[str, str, int]:
    end = stat_text.rfind(')')
    if end == -1:
        return '', '', 0
    start = stat_text.find('(')
    comm = stat_text[start + 1:end] if start != -1 else ''
    rest = stat_text[end + 2:].split()
    state = rest[0] if rest else ''
    cpu_ticks = 0
    try:
        cpu_ticks = int(rest[11]) + int(rest[12])
    except Exception:
        cpu_ticks = 0
    return comm, state, cpu_ticks


def _compact_cmdline(raw: str, fallback: str) -> str:
    text = raw.replace('\0', ' ').strip()
    return text or fallback


def _known_process_bucket(name: str, cmdline: str) -> str:
    text = f'{name} {cmdline}'.lower()
    for bucket, patterns in KNOWN_PROCESS_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return bucket
    return ''


def _read_process_rows(proc_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    page_size = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096
    try:
        entries = list(proc_root.iterdir())
    except Exception:
        return rows
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_text = (entry / 'stat').read_text(encoding='utf-8', errors='replace')
            name, state, cpu_ticks = _read_process_stat(stat_text)
            cmdline = _compact_cmdline((entry / 'cmdline').read_text(encoding='utf-8', errors='replace'), name)
            rss_pages = 0
            try:
                statm = (entry / 'statm').read_text(encoding='utf-8', errors='replace').split()
                if len(statm) >= 2:
                    rss_pages = int(statm[1])
            except Exception:
                rss_pages = 0
            rss_bytes = max(0, rss_pages * page_size)
            bucket = _known_process_bucket(name, cmdline)
            rows.append({
                'pid': pid,
                'name': name,
                'cmdline': cmdline,
                'state': state,
                'rss_bytes': rss_bytes,
                'cpu_ticks': cpu_ticks,
                'bucket': bucket,
            })
        except Exception:
            continue
    return rows


def _probe_nvidia_processes() -> List[Dict[str, object]]:
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return []
    commands = [
        [
            nvidia_smi,
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits',
        ],
        [
            nvidia_smi,
            '--query-accounted-apps=pid,process_name,gpu_util,mem_util,max_memory_usage',
            '--format=csv,noheader,nounits',
        ],
    ]
    rows: List[Dict[str, object]] = []
    seen = set()
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=2)
        except Exception:
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            used_mib = 0
            for value in reversed(parts[2:]):
                try:
                    used_mib = int(float(value))
                    break
                except Exception:
                    continue
            rows.append({
                'pid': pid,
                'name': parts[1],
                'gpu_memory_used': max(0, used_mib) * 1024**2,
            })
    return rows


def _pressure_level(score: float) -> str:
    if score >= 0.80:
        return 'high'
    if score >= 0.45:
        return 'medium'
    return 'low'


def benchmark_current_process_pressure(proc_root: str | Path = '/proc') -> ProcessPressureSnapshot:
    proc = Path(proc_root)
    load_1m, load_5m, load_15m, runnable, total_processes = _read_loadavg(proc)
    cpu_logical = os.cpu_count() or 1
    mem = read_meminfo_bytes() if proc == Path('/proc') else {}
    gpu_name, gpu_total, gpu_free, _gpu_error = probe_nvidia_gpu() if proc == Path('/proc') else ('', 0, 0, '')
    rows = _read_process_rows(proc)
    known_processes: Dict[str, int] = {}
    known_memory: Dict[str, int] = {}
    for row in rows:
        bucket = str(row.get('bucket', '') or '')
        if not bucket:
            continue
        known_processes[bucket] = known_processes.get(bucket, 0) + 1
        known_memory[bucket] = known_memory.get(bucket, 0) + int(row.get('rss_bytes', 0) or 0)
    top_memory = [
        {
            'pid': row['pid'],
            'name': row['name'],
            'rss_bytes': int(row.get('rss_bytes', 0) or 0),
            'bucket': row.get('bucket', ''),
        }
        for row in sorted(rows, key=lambda item: int(item.get('rss_bytes', 0) or 0), reverse=True)[:5]
        if int(row.get('rss_bytes', 0) or 0) > 0
    ]
    top_cpu = [
        {
            'pid': row['pid'],
            'name': row['name'],
            'cpu_ticks': int(row.get('cpu_ticks', 0) or 0),
            'bucket': row.get('bucket', ''),
        }
        for row in sorted(rows, key=lambda item: int(item.get('cpu_ticks', 0) or 0), reverse=True)[:5]
        if int(row.get('cpu_ticks', 0) or 0) > 0
    ]
    load_ratio = load_1m / max(1, cpu_logical)
    ram_pressure = 0.0
    memory_total = int(mem.get('MemTotal', 0) or 0)
    memory_available = int(mem.get('MemAvailable', 0) or 0)
    if memory_total > 0:
        ram_pressure = 1.0 - min(1.0, memory_available / memory_total)
    gpu_pressure = 0.0
    if gpu_total > 0:
        gpu_pressure = 1.0 - min(1.0, gpu_free / gpu_total)
    companion_pressure = min(1.0, sum(known_processes.values()) / 24.0)
    pressure_score = max(
        0.50 * min(1.5, load_ratio) / 1.5 + 0.30 * ram_pressure + 0.20 * companion_pressure,
        gpu_pressure,
    )
    pressure_score = max(0.0, min(1.0, pressure_score))
    snapshot = ProcessPressureSnapshot(
        timestamp=datetime.now().isoformat(timespec='seconds'),
        load_1m=round(load_1m, 2),
        load_5m=round(load_5m, 2),
        load_15m=round(load_15m, 2),
        runnable_processes=runnable,
        total_processes=total_processes,
        cpu_logical=cpu_logical,
        load_ratio=round(load_ratio, 3),
        memory_total=memory_total,
        memory_available=memory_available,
        gpu_memory_total=gpu_total,
        gpu_memory_free=gpu_free,
        process_count=len(rows),
        known_processes=known_processes,
        known_memory=known_memory,
        top_memory=top_memory,
        top_cpu=top_cpu,
        gpu_processes=_probe_nvidia_processes() if proc == Path('/proc') else [],
        pressure_score=round(pressure_score, 3),
        pressure_level=_pressure_level(pressure_score),
    )
    snapshot.detail = process_pressure_label(snapshot)
    return snapshot


def process_pressure_label(snapshot: ProcessPressureSnapshot) -> str:
    level = snapshot.pressure_level or 'low'
    load = f'load={snapshot.load_1m:.2f}/{snapshot.cpu_logical or "?"}'
    ram = ''
    if snapshot.memory_total > 0:
        ram = f'ram_free={bytes_to_gib(snapshot.memory_available):.1f}GiB'
    gpu = ''
    if snapshot.gpu_memory_total > 0:
        gpu = f'vram_free={bytes_to_gib(snapshot.gpu_memory_free):.1f}GiB'
    known = ','.join(
        f'{key}:{value}'
        for key, value in sorted(snapshot.known_processes.items())
        if value > 0
    )
    parts = [f'pressure={level}', load]
    if ram:
        parts.append(ram)
    if gpu:
        parts.append(gpu)
    if known:
        parts.append(f'apps={known}')
    return ' '.join(parts)
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
