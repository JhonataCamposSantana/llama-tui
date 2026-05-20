"""TQ3 / llama-bench raw-speed subprocess primitives.

Third extraction of audit finding #6 (decompose ``benchmark.py``).
Covers the cohesive, mostly-pure subset of the raw-bench code:

  - the command builder (``tq3_llama_bench_command``) and output parser
  - the subprocess runner (``_run_tq3_raw_process``) + process-group
    termination helper
  - the TQ3 raw-profile selection helpers
  - constants: ``RAW_BENCH_DETERMINISTIC_SEED`` (audit #21),
    ``TQ3_RAW_BENCH_CASES``

The heavy orchestrator ``run_tq3_raw_llama_bench_presearch`` stays in
``benchmark.py`` because it depends on ``BenchmarkDeadline``,
``adaptive_record_from_candidate``, ``emit_benchmark_event``,
``resolve_engine_install``, and ``model_is_moe`` — pulling all of those
along would tangle the module dependency graph. The orchestrator
imports the primitives here and stays a thin coordinator.
"""

import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .control import CancelToken
from .hardware import HardwareProfile
from .models import ModelConfig
from .optimize import model_is_moe
from .runtime_profiles import RuntimeProfile, kv_modes_from_preset


# Pick a stable seed so two consecutive raw-bench runs of the same
# (model, profile) yield reproducible token/sec readings within rounding.
# Audit finding #21.
RAW_BENCH_DETERMINISTIC_SEED = 42


TQ3_RAW_BENCH_CASES: Tuple[Tuple[str, int, int], ...] = (
    ('raw_pp', 1024, 0),
    ('raw_tg', 0, 64),
    ('raw_combined', 1024, 64),
)


def sibling_llama_bench_for_server(server_bin: str) -> str:
    parts = shlex.split(server_bin or '')
    if not parts:
        return 'llama-bench'
    first = os.path.expanduser(parts[0])
    if '/' in first or first.startswith('.'):
        return str(Path(first).expanduser().with_name('llama-bench'))
    return 'llama-bench'


def _tq3_raw_profile_key(runtime_profile: RuntimeProfile) -> Tuple[object, ...]:
    return (
        runtime_profile.gpu_layers,
        runtime_profile.kv_preset,
        bool(runtime_profile.cpu_moe),
        int(runtime_profile.n_cpu_moe or 0),
        tuple(runtime_profile.tensor_overrides or ()),
    )


def _tq3_raw_profile_rank(runtime_profile: RuntimeProfile, original_index: int) -> Tuple[int, int, int]:
    n_cpu_moe = int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0)
    if n_cpu_moe == 32:
        priority = 0
    elif n_cpu_moe == 30:
        priority = 1
    elif n_cpu_moe > 0:
        priority = 20 + abs(n_cpu_moe - 32)
    elif bool(getattr(runtime_profile, 'cpu_moe', False)):
        priority = 100
    else:
        priority = 999
    return priority, original_index, n_cpu_moe


def _tq3_raw_runtime_profiles(runtime_profiles: List[RuntimeProfile], depth: str) -> List[RuntimeProfile]:
    candidates: List[Tuple[Tuple[int, int, int], RuntimeProfile]] = []
    seen = set()
    fast = (depth or '').strip().lower() == 'fast'
    limit = 1 if fast else 2
    for index, profile in enumerate(runtime_profiles):
        if profile.engine_id != 'tq3' or profile.fit:
            continue
        if not (profile.cpu_moe or int(profile.n_cpu_moe or 0) > 0):
            continue
        if str(profile.kv_preset or '') != 'q8_0/q8_0':
            continue
        if tuple(profile.tensor_overrides or ()):
            continue
        if (
            str(getattr(profile, 'reasoning', '') or '').strip()
            or int(getattr(profile, 'reasoning_budget', -1) or -1) >= 0
            or str(getattr(profile, 'reasoning_format', '') or '').strip()
        ):
            continue
        key = _tq3_raw_profile_key(profile)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((_tq3_raw_profile_rank(profile, index), profile))
    candidates.sort(key=lambda item: item[0])
    return [profile for _rank, profile in candidates[:limit]]


def tq3_raw_presearch_case_total(runtime_profiles: List[RuntimeProfile], depth: str) -> int:
    return len(_tq3_raw_runtime_profiles(runtime_profiles, depth)) * len(TQ3_RAW_BENCH_CASES)


def tq3_llama_bench_command(
    llama_bench_bin: str,
    model: ModelConfig,
    runtime_profile: RuntimeProfile,
    prompt_tokens: int,
    generated_tokens: int,
    threads: int,
) -> List[str]:
    cmd = [
        llama_bench_bin,
        '-m', str(getattr(model, 'path', '') or ''),
        '-p', str(int(prompt_tokens)),
        '-n', str(int(generated_tokens)),
        '--seed', str(RAW_BENCH_DETERMINISTIC_SEED),
    ]
    if runtime_profile.gpu_layers is not None:
        cmd += ['-ngl', str(int(runtime_profile.gpu_layers))]
    if runtime_profile.cpu_moe:
        cmd.append('-cmoe')
    elif int(runtime_profile.n_cpu_moe or 0) > 0:
        cmd += ['-ncmoe', str(int(runtime_profile.n_cpu_moe or 0))]
    key_mode, value_mode = kv_modes_from_preset(runtime_profile.kv_preset)
    if key_mode and value_mode:
        cmd += ['-ctk', key_mode, '-ctv', value_mode]
    if int(runtime_profile.batch_size or 0) > 0:
        cmd += ['-b', str(int(runtime_profile.batch_size))]
    if int(runtime_profile.ubatch_size or 0) > 0:
        cmd += ['-ub', str(int(runtime_profile.ubatch_size))]
    if str(runtime_profile.flash_attn or '').strip().lower() in ('on', 'auto', ''):
        cmd += ['-fa', '1']
    if int(threads or 0) > 0:
        cmd += ['-t', str(int(threads))]
    return cmd


def parse_llama_bench_tokens_per_sec(output: str) -> float:
    values = []
    for match in re.finditer(r'([0-9]+(?:\.[0-9]+)?)\s*(?:±[^|\n]+)?\s*tok/s', output or '', re.IGNORECASE):
        try:
            values.append(float(match.group(1)))
        except Exception:
            pass
    return values[-1] if values else 0.0


def _raw_bench_candidate_model(model: ModelConfig, runtime_profile: RuntimeProfile) -> ModelConfig:
    candidate = ModelConfig(**asdict(model))
    candidate.ctx = max(1, int(runtime_profile.ctx_size or candidate.ctx or 1))
    candidate.parallel = max(1, int(runtime_profile.parallel or 1))
    if runtime_profile.gpu_layers is not None:
        candidate.ngl = int(runtime_profile.gpu_layers)
    candidate.moe_placement_strategy = str(runtime_profile.placement_strategy or '')
    candidate.cpu_moe = bool(runtime_profile.cpu_moe)
    candidate.n_cpu_moe = int(runtime_profile.n_cpu_moe or 0)
    candidate.tensor_overrides = [str(item) for item in tuple(runtime_profile.tensor_overrides or ())]
    return candidate


def tq3_moe_cpu_placement_threads(
    model: ModelConfig,
    runtime_profile: Optional[RuntimeProfile] = None,
    hardware: Optional[HardwareProfile] = None,
) -> int:
    current = max(1, int(getattr(model, 'threads', 0) or 1))
    if runtime_profile is not None and str(getattr(runtime_profile, 'engine_id', '') or '') != 'tq3':
        return current
    if not model_is_moe(model):
        return current
    cpu_placement = bool(getattr(model, 'cpu_moe', False)) or int(getattr(model, 'n_cpu_moe', 0) or 0) > 0
    if runtime_profile is not None:
        cpu_placement = (
            cpu_placement
            or bool(getattr(runtime_profile, 'cpu_moe', False))
            or int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0) > 0
        )
    if not cpu_placement:
        return current
    logical = int(getattr(hardware, 'cpu_logical', 0) or 0) if hardware is not None else 0
    logical = logical or (os.cpu_count() or current)
    return max(current, min(max(1, logical), 12))


def _terminate_process_group(process: subprocess.Popen):
    if process.poll() is not None:
        return
    try:
        if hasattr(os, 'killpg'):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.terminate()
        except Exception:
            return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, 'killpg'):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _run_tq3_raw_process(
    cmd: List[str],
    timeout_seconds: float,
    cancel_token: Optional[CancelToken] = None,
) -> Dict[str, object]:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
        )
    except FileNotFoundError as exc:
        return {
            'returncode': 127,
            'stdout': str(exc),
            'seconds': time.monotonic() - started,
            'timed_out': False,
            'cancelled': False,
        }
    except Exception as exc:
        return {
            'returncode': -1,
            'stdout': str(exc),
            'seconds': time.monotonic() - started,
            'timed_out': False,
            'cancelled': False,
        }

    timed_out = False
    cancelled = False
    timeout_seconds = max(1.0, float(timeout_seconds or 1.0))
    while process.poll() is None:
        if cancel_token is not None and cancel_token.is_cancelled():
            cancelled = True
            _terminate_process_group(process)
            break
        if time.monotonic() - started >= timeout_seconds:
            timed_out = True
            _terminate_process_group(process)
            break
        if cancel_token is not None and cancel_token.wait(0.5):
            continue
        time.sleep(0.5)

    try:
        stdout, _stderr = process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        try:
            stdout, _stderr = process.communicate(timeout=1)
        except Exception:
            stdout = ''
    return {
        'returncode': int(process.returncode if process.returncode is not None else -1),
        'stdout': stdout or '',
        'seconds': time.monotonic() - started,
        'timed_out': timed_out,
        'cancelled': cancelled,
    }
