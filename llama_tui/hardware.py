import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class HardwareProfile:
    cpu_logical: int = 0
    cpu_physical: int = 0
    cpu_performance: int = 0
    memory_total: int = 0
    memory_available: int = 0
    gpu_name: str = ''
    gpu_memory_total: int = 0
    gpu_memory_free: int = 0
    gpu_error: str = ''
    gpu_temperature: int = 0
    gpu_throttle_active: bool = False

    def has_usable_gpu(self) -> bool:
        return self.gpu_memory_free > 0

    def short_summary(self) -> str:
        total_gib = bytes_to_gib(self.memory_total)
        avail_gib = bytes_to_gib(self.memory_available)
        cpu = f'cpu={self.cpu_physical or "?"}c/{self.cpu_logical or "?"}t'
        if self.cpu_performance and self.cpu_physical and self.cpu_performance < self.cpu_physical:
            cpu += f'({self.cpu_performance}P)'
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


def read_cgroup_memory_limits(
    cgroup_root: Path = Path('/sys/fs/cgroup'),
) -> Dict[str, int]:
    """Return ``{'max': bytes, 'current': bytes}`` from cgroups v2, or ``{}``.

    Inside a Docker/Distrobox/systemd-slice container the kernel still serves
    the host's ``/proc/meminfo``, so a 64 GiB host with a 4 GiB cgroup cap
    looks like 64 GiB to ``read_meminfo_bytes`` and the optimizer happily
    plans for memory that will be killed by the cgroup OOM. Reading
    ``memory.max`` / ``memory.current`` gives the real limit. ``memory.max``
    is either a byte count or the literal string ``max`` (unlimited).
    Cgroups v1 (legacy ``memory.limit_in_bytes``) is intentionally not
    supported; it predates llama-tui's deployment targets.
    """
    out: Dict[str, int] = {}
    max_path = cgroup_root / 'memory.max'
    current_path = cgroup_root / 'memory.current'
    try:
        max_raw = max_path.read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        return out
    if max_raw and max_raw.lower() != 'max':
        try:
            out['max'] = int(max_raw)
        except ValueError:
            pass
    try:
        current_raw = current_path.read_text(encoding='utf-8', errors='replace').strip()
        out['current'] = int(current_raw)
    except (OSError, ValueError):
        pass
    return out


def clamp_memory_to_cgroup(
    memory_total: int,
    memory_available: int,
    cgroup_root: Path = Path('/sys/fs/cgroup'),
) -> Tuple[int, int]:
    """Clamp ``(memory_total, memory_available)`` against the cgroup v2 limit.

    Returns the pair unchanged when no cgroup limit is in effect.
    """
    limits = read_cgroup_memory_limits(cgroup_root)
    cap = int(limits.get('max', 0) or 0)
    if cap <= 0:
        return memory_total, memory_available
    bounded_total = min(memory_total, cap) if memory_total > 0 else cap
    used = int(limits.get('current', 0) or 0)
    cgroup_available = max(0, cap - used)
    if memory_available > 0:
        bounded_available = min(memory_available, cgroup_available)
    else:
        bounded_available = cgroup_available
    return bounded_total, bounded_available


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


def detect_performance_core_count(
    physical: int,
    sysfs_root: str = '/sys/devices/system/cpu',
) -> int:
    """Count the fastest physical cores on a hybrid (P+E) CPU.

    Maps each logical CPU to its physical core and reads that core's max
    frequency from sysfs. The number of distinct physical cores at the top
    frequency tier is the performance-core count. On a homogeneous CPU (all
    cores share one tier) or when sysfs is unavailable, this returns
    ``physical`` so callers behave exactly as before.
    """
    physical = max(1, int(physical or 1))
    root = Path(sysfs_root)
    if not root.exists():
        return physical
    try:
        core_max_freq: Dict[Tuple[str, str], int] = {}
        for cpu_dir in root.glob('cpu[0-9]*'):
            if not cpu_dir.name[3:].isdigit():
                continue
            freq_file = cpu_dir / 'cpufreq' / 'cpuinfo_max_freq'
            pkg_file = cpu_dir / 'topology' / 'physical_package_id'
            core_file = cpu_dir / 'topology' / 'core_id'
            if not (freq_file.exists() and core_file.exists()):
                return physical
            freq = int(freq_file.read_text().strip())
            pkg = (pkg_file.read_text().strip() if pkg_file.exists() else '0')
            core = core_file.read_text().strip()
            key = (pkg, core)
            # Same physical core can appear under multiple HT siblings; keep max.
            if freq > core_max_freq.get(key, 0):
                core_max_freq[key] = freq
        if not core_max_freq:
            return physical
        global_max = max(core_max_freq.values())
        if global_max <= 0:
            return physical
        threshold = global_max * 0.97
        perf = sum(1 for f in core_max_freq.values() if f >= threshold)
        if perf <= 0 or perf >= len(core_max_freq):
            # Homogeneous CPU: no E-cores to avoid.
            return physical
        return max(1, min(perf, physical))
    except (OSError, ValueError):
        return physical
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
def probe_amd_rocm_gpu() -> Tuple[str, int, int, str]:
    """Return ``(name, memory_total_bytes, memory_free_bytes, error)`` from
    ``rocm-smi``. Mirrors :func:`probe_nvidia_gpu` so AMD ROCm hosts get the
    same ``HardwareProfile`` shape as NVIDIA hosts. Returns empty/zero values
    when rocm-smi is unavailable or fails — the caller treats that as "no GPU
    of this kind detected" and may try other backends.
    """
    rocm_smi = shutil.which('rocm-smi')
    if not rocm_smi:
        return '', 0, 0, ''
    try:
        result = subprocess.run(
            [rocm_smi, '--showmeminfo', 'vram', '--showproductname', '--csv'],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        return '', 0, 0, f'rocm-smi failed: {exc}'
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or '').strip().splitlines()
        return '', 0, 0, detail[0] if detail else 'rocm-smi failed'
    best_name = ''
    best_total = 0
    best_free = 0
    # rocm-smi --csv emits one header row then one row per device. We look up
    # columns by header name so column-order changes in future ROCm releases
    # do not silently mis-map values.
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return '', 0, 0, ''
    header_parts = [part.strip() for part in rows[0].split(',')]
    def column(row_parts: List[str], *candidates: str) -> str:
        for candidate in candidates:
            try:
                idx = header_parts.index(candidate)
            except ValueError:
                continue
            if idx < len(row_parts):
                return row_parts[idx].strip()
        return ''
    for line in rows[1:]:
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 2:
            continue
        total_raw = column(parts, 'VRAM Total Memory (B)', 'VRAM Total (B)')
        used_raw = column(parts, 'VRAM Total Used Memory (B)', 'VRAM Used (B)')
        name_raw = column(parts, 'Card series', 'Card model', 'Card SKU', 'GPU')
        try:
            total = int(total_raw)
            used = int(used_raw)
        except (TypeError, ValueError):
            continue
        free = max(0, total - used)
        if free > best_free:
            best_name = name_raw or 'AMD GPU'
            best_total = total
            best_free = free
    return best_name, best_total, best_free, ''


# Apple's Metal driver on M-series typically caps the GPU "Recommended Max
# Working Set Size" near 75% of unified memory. Reading that exact value
# requires a Swift/Objective-C call into MTLDevice; we approximate with the
# fraction so the optimizer at least routes through the GPU code path
# instead of going CPU-only on every macOS launch. Callers that need the
# real ceiling can override by setting LLAMA_TUI_APPLE_METAL_FRACTION.
APPLE_METAL_WORKING_SET_FRACTION = 0.75


def _sysctl_bytes(key: str) -> int:
    sysctl = shutil.which('sysctl')
    if not sysctl:
        return 0
    try:
        result = subprocess.run(
            [sysctl, '-n', key],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if result.returncode != 0:
        return 0
    try:
        return int((result.stdout or '').strip())
    except ValueError:
        return 0


def _sysctl_text(key: str) -> str:
    sysctl = shutil.which('sysctl')
    if not sysctl:
        return ''
    try:
        result = subprocess.run(
            [sysctl, '-n', key],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ''
    if result.returncode != 0:
        return ''
    return (result.stdout or '').strip()


def _apple_metal_fraction() -> float:
    raw = os.environ.get('LLAMA_TUI_APPLE_METAL_FRACTION', '').strip()
    if not raw:
        return APPLE_METAL_WORKING_SET_FRACTION
    try:
        value = float(raw)
    except ValueError:
        return APPLE_METAL_WORKING_SET_FRACTION
    # Keep the fraction physical: at least a tiny sliver, never above 1.0.
    return max(0.05, min(1.0, value))


def _vm_stat_free_bytes() -> int:
    vm_stat = shutil.which('vm_stat')
    if not vm_stat:
        return 0
    try:
        result = subprocess.run(
            [vm_stat],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if result.returncode != 0:
        return 0
    page_size = 4096
    free_pages = 0
    speculative_pages = 0
    for line in (result.stdout or '').splitlines():
        if 'page size of' in line:
            for token in line.split():
                token = token.strip('.,()')
                if token.isdigit():
                    page_size = int(token)
                    break
            continue
        label, sep, value = line.partition(':')
        if not sep:
            continue
        digits = value.strip().rstrip('.').replace(',', '')
        if not digits.isdigit():
            continue
        if label.strip().lower() == 'pages free':
            free_pages = int(digits)
        elif label.strip().lower() == 'pages speculative':
            speculative_pages = int(digits)
    return (free_pages + speculative_pages) * page_size


def probe_apple_silicon_gpu() -> Tuple[str, int, int, str]:
    """Return ``(name, memory_total_bytes, memory_free_bytes, error)`` for
    Apple Silicon Macs.

    M-series Macs use unified memory: there is no separate VRAM pool. We
    report the Metal "recommended working set" fraction of system memory so
    the optimizer can size GPU offload against a realistic ceiling instead
    of falling through to CPU-only. The caller should still subtract the
    weight footprint from the same system_memory budget — both pools alias
    the same physical RAM.
    """
    if sys.platform != 'darwin':
        return '', 0, 0, ''
    total_ram = _sysctl_bytes('hw.memsize')
    if total_ram <= 0:
        return '', 0, 0, ''
    free_ram = _vm_stat_free_bytes()
    if free_ram <= 0:
        # vm_stat absent or unparseable — fall back to a conservative
        # "half free" estimate so the GPU path is still picked.
        free_ram = total_ram // 2
    name = _sysctl_text('machdep.cpu.brand_string') or 'Apple Silicon GPU'
    fraction = _apple_metal_fraction()
    gpu_total = int(total_ram * fraction)
    gpu_free = int(min(free_ram, total_ram) * fraction)
    return name, gpu_total, gpu_free, ''


def probe_nvidia_gpu_thermal() -> Tuple[int, bool]:
    """Return (gpu_temperature_celsius, thermal_throttle_active).

    Queried separately from probe_nvidia_gpu() so an unsupported throttle-reason
    field can never break the memory probe. Returns (0, False) on any failure.
    """
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return 0, False
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                '--query-gpu=temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown,'
                'clocks_throttle_reasons.sw_thermal_slowdown',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return 0, False
    if result.returncode != 0:
        return 0, False
    best_temp = 0
    throttle = False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if not parts or not parts[0]:
            continue
        try:
            best_temp = max(best_temp, int(float(parts[0])))
        except ValueError:
            continue
        for reason in parts[1:]:
            low = reason.lower()
            if 'active' in low and 'not active' not in low:
                throttle = True
    return best_temp, throttle


def benchmark_current_hardware() -> HardwareProfile:
    logical, physical = detect_cpu_counts()
    performance = detect_performance_core_count(physical)
    mem = read_meminfo_bytes()
    memory_total, memory_available = clamp_memory_to_cgroup(
        mem.get('MemTotal', 0),
        mem.get('MemAvailable', 0),
    )
    gpu_name, gpu_total, gpu_free, gpu_error = probe_nvidia_gpu()
    if gpu_total <= 0 and gpu_free <= 0:
        # Fall back to ROCm so AMD users get a populated HardwareProfile.
        # nvidia-smi takes precedence on hosts that have both because the
        # llama.cpp CUDA path is still the dominant deployment target.
        amd_name, amd_total, amd_free, amd_error = probe_amd_rocm_gpu()
        if amd_total > 0 or amd_free > 0:
            gpu_name, gpu_total, gpu_free = amd_name, amd_total, amd_free
            gpu_error = ''
        elif amd_error and not gpu_error:
            gpu_error = amd_error
    if gpu_total <= 0 and gpu_free <= 0:
        # On macOS Apple Silicon there is no nvidia-smi/rocm-smi; report
        # Metal's recommended working-set fraction of unified memory so the
        # optimizer routes through the GPU path instead of going CPU-only.
        apple_name, apple_total, apple_free, apple_error = probe_apple_silicon_gpu()
        if apple_total > 0 or apple_free > 0:
            gpu_name, gpu_total, gpu_free = apple_name, apple_total, apple_free
            gpu_error = ''
        elif apple_error and not gpu_error:
            gpu_error = apple_error
    gpu_temp, gpu_throttle = probe_nvidia_gpu_thermal()
    return HardwareProfile(
        cpu_logical=logical,
        cpu_physical=physical,
        cpu_performance=performance,
        memory_total=memory_total,
        memory_available=memory_available,
        gpu_name=gpu_name,
        gpu_memory_total=gpu_total,
        gpu_memory_free=gpu_free,
        gpu_error=gpu_error,
        gpu_temperature=gpu_temp,
        gpu_throttle_active=gpu_throttle,
    )
