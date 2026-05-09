import json
import os
import re
import selectors
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .benchmark import (
    adaptive_context_upper_bound,
    architecture_payload,
    benchmark_preflight_cleanup,
    candidate_safe_context_estimate,
    configure_adaptive_candidate,
    clone_model_config,
    concise_failure,
    ctx_per_slot,
    current_process_pressure_payload,
    dynamic_context_growth_targets,
    emit_benchmark_event,
    expand_workflow_cache_ram_candidates,
    model_and_runtime_profile_from_measured_profile,
    observed_opencode_context_floor,
    parse_context_requirement,
    sync_opencode_after_tuning,
    workflow_cache_ram_profile_from_record,
    workflow_cache_ram_record_fields,
    workflow_cache_ram_selection_key,
)
from .control import CancelToken, CancelledError, check_cancelled, sleep_with_cancel
from .hardware import HardwareProfile, read_meminfo_bytes
from .memory_guardrail import (
    MemoryGuardrailState,
    memory_guardrail_record_fields,
    start_memory_guardrail_watchdog,
)
from .models import ModelConfig
from .textutil import compact_message

OPENCODE_PREFLIGHT_TIMEOUT = 20
OPENCODE_TASK_TIMEOUT = 300
WORKFLOW_NORMAL_NO_OUTPUT_TIMEOUT = 90
WORKFLOW_NORMAL_IDLE_OUTPUT_TIMEOUT = 180
WORKFLOW_SLOW_NO_OUTPUT_TIMEOUT = 180
WORKFLOW_SLOW_IDLE_OUTPUT_TIMEOUT = 240

OPENCODE_NO_OUTPUT_TIMEOUT = WORKFLOW_NORMAL_NO_OUTPUT_TIMEOUT
OPENCODE_IDLE_OUTPUT_TIMEOUT = WORKFLOW_NORMAL_IDLE_OUTPUT_TIMEOUT
OPENCODE_BENCHMARK_CANDIDATES = 4
OPENCODE_LOG_LEVEL = 'WARN'
OPENCODE_DYNAMIC_CANDIDATE_OBJECTIVE = 'opencode_ready'


@dataclass
class WorkflowTask:
    name: str
    prompt: str
    files: Dict[str, str]


@dataclass(frozen=True)
class WorkflowTimeoutPolicy:
    no_output_timeout: int
    idle_output_timeout: int
    reason: str
    pressure_level: str = ''


OPENCODE_WORKFLOW_TASKS = [
    WorkflowTask(
        name='fix_calc',
        prompt=(
            'Fix this tiny Python project so `python -m unittest -q` passes. '
            'Keep the change minimal, do not use the network, and do not edit files outside this directory.'
        ),
        files={
            'calc.py': (
                'def add_numbers(values):\n'
                '    total = 0\n'
                '    for value in values:\n'
                '        total += value\n'
                '    return total\n'
            ),
            'test_calc.py': (
                'import unittest\n'
                'from calc import add_numbers\n\n'
                'class CalcTests(unittest.TestCase):\n'
                '    def test_adds_numbers_and_numeric_strings(self):\n'
                '        self.assertEqual(add_numbers([1, "2", -3, "4"]), 4)\n\n'
                '    def test_empty_values(self):\n'
                '        self.assertEqual(add_numbers([]), 0)\n\n'
                'if __name__ == "__main__":\n'
                '    unittest.main()\n'
            ),
            'README.md': 'Small benchmark fixture for llama-tui/OpenCode.\n',
        },
    ),
    WorkflowTask(
        name='add_slugify',
        prompt=(
            'Implement the missing slugify function so `python -m unittest -q` passes. '
            'Keep the implementation compact, deterministic, and local to this project.'
        ),
        files={
            'text_tools.py': (
                'def slugify(text):\n'
                '    raise NotImplementedError("slugify is not implemented yet")\n'
            ),
            'test_text_tools.py': (
                'import unittest\n'
                'from text_tools import slugify\n\n'
                'class TextToolTests(unittest.TestCase):\n'
                '    def test_slugify_words(self):\n'
                '        self.assertEqual(slugify("Hello, Local LLM!"), "hello-local-llm")\n\n'
                '    def test_slugify_spaces_and_symbols(self):\n'
                '        self.assertEqual(slugify("  GPUs + CPUs  "), "gpus-cpus")\n\n'
                'if __name__ == "__main__":\n'
                '    unittest.main()\n'
            ),
        },
    ),
]


def detect_vscode_pressure() -> Dict[str, object]:
    count = 0
    rss_bytes = 0
    page_size = os.sysconf('SC_PAGE_SIZE')
    for proc_dir in Path('/proc').iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            comm = (proc_dir / 'comm').read_text(errors='replace').strip().lower()
            raw_cmd = (proc_dir / 'cmdline').read_bytes()
            parts = [part.decode(errors='ignore') for part in raw_cmd.split(b'\0') if part]
            exe_name = Path(parts[0]).name.lower() if parts else comm
            is_code = comm in ('code', 'code-insiders') or exe_name in ('code', 'code-insiders')
            if not is_code:
                continue
            statm = (proc_dir / 'statm').read_text().split()
            rss_pages = int(statm[1]) if len(statm) > 1 else 0
        except Exception:
            continue
        count += 1
        rss_bytes += rss_pages * page_size
    return {
        'present': count > 0,
        'processes': count,
        'rss_mib': round(rss_bytes / 1024**2, 1),
    }


def workflow_timeout_policy(model: ModelConfig, pressure_payload: Optional[Dict[str, object]] = None) -> WorkflowTimeoutPolicy:
    pressure_payload = pressure_payload or {}
    pressure_level = str(pressure_payload.get('process_pressure_level', '') or '').strip().lower()
    is_moe = str(getattr(model, 'architecture_type', '') or '').strip().lower() == 'moe'
    pressure_slow = pressure_level in ('medium', 'high')
    if is_moe or pressure_slow:
        reasons = []
        if is_moe:
            reasons.append('moe')
        if pressure_slow:
            reasons.append(f'{pressure_level}_pressure')
        return WorkflowTimeoutPolicy(
            no_output_timeout=WORKFLOW_SLOW_NO_OUTPUT_TIMEOUT,
            idle_output_timeout=WORKFLOW_SLOW_IDLE_OUTPUT_TIMEOUT,
            reason='+'.join(reasons) or 'slow_path',
            pressure_level=pressure_level,
        )
    return WorkflowTimeoutPolicy(
        no_output_timeout=WORKFLOW_NORMAL_NO_OUTPUT_TIMEOUT,
        idle_output_timeout=WORKFLOW_NORMAL_IDLE_OUTPUT_TIMEOUT,
        reason='normal',
        pressure_level=pressure_level,
    )


def write_fixture(root: Path, task: WorkflowTask):
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in task.files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')


def write_temp_opencode_config(app, model: ModelConfig, home: Path) -> Path:
    config_dir = home / '.config' / 'opencode'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / 'opencode.json'
    provider_key = app.opencode_provider_key(model)
    model_ref = app.opencode_model_ref(model)
    config = {
        '$schema': 'https://opencode.ai/config.json',
        'instructions': [
            'You are running inside a disposable llama-tui OpenCode benchmark fixture.',
            'Only inspect and edit files in the current benchmark directory.',
            'Run python -m unittest -q before finishing and summarize whether tests pass.',
        ],
        'permission': {
            'edit': 'allow',
            'read': 'allow',
            'list': 'allow',
            'glob': 'allow',
            'grep': 'allow',
            'webfetch': 'deny',
            'websearch': 'deny',
            'external_directory': 'deny',
            'bash': {
                '*': 'deny',
                'python *': 'allow',
                'python3 *': 'allow',
                f'{shlex_python()} *': 'allow',
                'ls *': 'allow',
                'cat *': 'allow',
                'grep *': 'allow',
                'sed *': 'allow',
            },
        },
        'provider': {
            provider_key: {
                'npm': '@ai-sdk/openai-compatible',
                'name': f'Benchmark {model.name}',
                'options': {
                    'baseURL': f'http://{model.host}:{model.port}/v1',
                    'timeout': getattr(app.opencode, 'timeout', 600000),
                    'chunkTimeout': getattr(app.opencode, 'chunk_timeout', 60000),
                },
                'models': {
                    model.alias: {
                        'name': model.name,
                        'limit': {
                            'context': max(1, ctx_per_slot(model)),
                            'output': model.output,
                        },
                    },
                },
            },
        },
        'model': model_ref,
        'small_model': model_ref,
        'agent': {
            'build': {
                'model': model_ref,
                'prompt': (
                    'Fix the tiny local fixture with the smallest useful change. '
                    'Use local files only, do not use the network, and run python -m unittest -q.'
                ),
                'tools': {
                    'read': True,
                    'list': True,
                    'glob': True,
                    'grep': True,
                    'edit': True,
                    'write': True,
                    'bash': True,
                    'webfetch': False,
                    'websearch': False,
                    'external_directory': False,
                },
            },
            'plan': {'model': model_ref},
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return config_path


def shlex_python() -> str:
    return Path(sys.executable).name


def isolated_opencode_env(home: Path, config_path: Optional[Path] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env['HOME'] = str(home)
    env['XDG_CONFIG_HOME'] = str(home / '.config')
    env['XDG_DATA_HOME'] = str(home / '.local' / 'share')
    env['XDG_STATE_HOME'] = str(home / '.local' / 'state')
    if config_path is not None:
        env['OPENCODE_CONFIG'] = str(config_path)
    env['OPENCODE_DISABLE_AUTOUPDATE'] = 'true'
    env['OPENCODE_DISABLE_PRUNE'] = 'true'
    env['OPENCODE_DISABLE_MODELS_FETCH'] = 'true'
    env['OPENCODE_CLIENT'] = 'llama-tui-benchmark'
    return env


def opencode_cli_preflight(timeout: int = OPENCODE_PREFLIGHT_TIMEOUT) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix='llama-tui-opencode-preflight-') as home_raw:
        home = Path(home_raw)
        config_path = home / '.config' / 'opencode' / 'opencode.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({'$schema': 'https://opencode.ai/config.json'}, indent=2), encoding='utf-8')
        env = isolated_opencode_env(home, config_path)
        checks = [
            ['opencode', '--version'],
            ['opencode', 'run', '--help'],
        ]
        details = []
        for command in checks:
            try:
                result = subprocess.run(
                    command,
                    cwd=str(home),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, f'{" ".join(command)} did not return within {timeout}s'
            except OSError as exc:
                return False, str(exc)
            output = compact_message((result.stdout or result.stderr or '').strip())
            if result.returncode != 0:
                return False, f'{" ".join(command)} failed ({result.returncode}): {output}'
            if output:
                details.append(output.split()[0])
    return True, 'opencode CLI ready' + (f' ({", ".join(details)})' if details else '')


def opencode_provider_preflight(app, model: ModelConfig, timeout: int = OPENCODE_PREFLIGHT_TIMEOUT) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix='llama-tui-opencode-provider-') as home_raw:
        home = Path(home_raw)
        config_path = write_temp_opencode_config(app, model, home)
        env = isolated_opencode_env(home, config_path)
        provider_key = app.opencode_provider_key(model)
        command = ['opencode', 'models', provider_key]
        try:
            result = subprocess.run(
                command,
                cwd=str(home),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f'opencode models {provider_key} did not return within {timeout}s'
        except OSError as exc:
            return False, str(exc)
        output = compact_message(((result.stdout or '') + ' ' + (result.stderr or '')).strip())
        if result.returncode != 0:
            return False, f'opencode models {provider_key} failed ({result.returncode}): {output}'
        if app.opencode_model_ref(model) not in output:
            return False, f'opencode models {provider_key} did not list {app.opencode_model_ref(model)}: {output}'
        return True, f'opencode provider {provider_key} lists {app.opencode_model_ref(model)}'


def build_opencode_run_command(app, model: ModelConfig, workspace: Path, prompt: str) -> List[str]:
    return [
        'opencode',
        'run',
        '--pure',
        '--model', app.opencode_model_ref(model),
        '--agent', 'build',
        '--format', 'json',
        '--dir', str(workspace),
        '--dangerously-skip-permissions',
        '--print-logs',
        '--log-level', OPENCODE_LOG_LEVEL,
        prompt,
    ]


def opencode_candidate_models(model: ModelConfig, profile) -> List[Tuple[str, str, ModelConfig, str]]:
    candidates: List[Tuple[str, str, ModelConfig, str]] = []
    seen = set()
    observed_floor = max(0, int(observed_opencode_context_floor(model) or 0))

    def add(label: str, tier: str, candidate: ModelConfig, detail: str):
        slot = ctx_per_slot(candidate)
        if observed_floor and slot < observed_floor:
            return
        key = (int(getattr(candidate, 'ctx', 0) or 0), int(getattr(candidate, 'parallel', 1) or 1), tuple(getattr(candidate, 'extra_args', []) or []))
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, tier, candidate, detail))

    for key in ('opencode_ready', 'long_context', 'auto', 'fast_chat'):
        measured, _runtime_profile = model_and_runtime_profile_from_measured_profile(model, key)
        if measured is not None:
            add(key, 'measured', measured, f'measured {key} ctx_per_slot={ctx_per_slot(measured)}')
            if len(candidates) >= OPENCODE_BENCHMARK_CANDIDATES:
                return candidates

    variants = ['default']
    if getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp' and profile.has_usable_gpu():
        variants.append('q8_kv')
    for variant in variants:
        upper = adaptive_context_upper_bound(model, profile, OPENCODE_DYNAMIC_CANDIDATE_OBJECTIVE, parallel=1, variant=variant)
        ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
        ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', upper or ctx_min) or (upper or ctx_min)))
        dynamic_points = dynamic_context_growth_targets(
            model,
            profile,
            ctx_min,
            ctx_max,
            depth='fast',
            observed_floor=observed_floor,
            objective=OPENCODE_DYNAMIC_CANDIDATE_OBJECTIVE,
            variant=variant,
        )
        points = sorted(set([
            ctx_min,
            max(ctx_min, upper // 2),
            upper,
            *dynamic_points,
        ]))
        if (getattr(model, 'architecture_type', '') or '').strip().lower() == 'moe':
            points = sorted(set(points + [
                max(ctx_min, min(upper, 16384)),
                max(ctx_min, min(upper, 32768)),
            ]))
        points = sorted(set(points))
        for ctx in points:
            candidate = configure_adaptive_candidate(model, profile, OPENCODE_DYNAMIC_CANDIDATE_OBJECTIVE, ctx, 1, variant)
            label = OPENCODE_DYNAMIC_CANDIDATE_OBJECTIVE if variant == 'default' else f'{OPENCODE_DYNAMIC_CANDIDATE_OBJECTIVE}_{variant}'
            add(label, 'estimated', candidate, f'estimated {variant} ctx_per_slot={ctx_per_slot(candidate)}')
            if len(candidates) >= OPENCODE_BENCHMARK_CANDIDATES:
                return candidates
    return candidates


def sample_memory(app) -> Dict[str, int]:
    profile = app.hardware_profile(refresh=True)
    mem_available = profile.memory_available or read_meminfo_bytes().get('MemAvailable', 0)
    return {
        'ram_available': int(mem_available or 0),
        'ram_total': int(profile.memory_total or 0),
        'gpu_memory_free': int(profile.gpu_memory_free or 0),
        'gpu_memory_total': int(profile.gpu_memory_total or 0),
    }


def benchmark_record_context(model: ModelConfig) -> Dict[str, object]:
    payload = architecture_payload(model)
    payload.update(current_process_pressure_payload())
    payload.update(workflow_cache_ram_record_fields(model))
    return payload


def is_startup_only_output(line: str) -> bool:
    text = compact_message(str(line or '').strip())
    if not text:
        return True
    lowered = text.lower()
    if lowered.startswith('performing one time database migration'):
        return True
    if lowered.startswith('sqlite-migration:'):
        return True
    if lowered == 'database migration complete.':
        return True
    if 'preparing terminal' in lowered:
        return True
    if lowered.startswith('\u256d') and 'hermes' in lowered:
        return True
    return False


def is_meaningful_process_output(line: str) -> bool:
    text = str(line or '').strip()
    if not text or is_startup_only_output(text):
        return False
    payload = text[5:].strip() if text.startswith('data:') else text
    try:
        event = json.loads(payload)
    except Exception:
        return True
    event_type = str(event.get('type', '') or '').strip().lower()
    if event_type in ('step_start', 'step_finish'):
        return False
    return True


def run_process_with_metrics(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout: int,
    app,
    cancel_token: Optional[CancelToken] = None,
    no_output_timeout: int = OPENCODE_NO_OUTPUT_TIMEOUT,
    idle_output_timeout: int = OPENCODE_IDLE_OUTPUT_TIMEOUT,
    stop_on_context_overflow: bool = True,
) -> Dict[str, object]:
    check_cancelled(cancel_token)
    started = time.monotonic()
    first_output: Optional[float] = None
    first_process_output: Optional[float] = None
    last_meaningful_output_at = started
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    startup_output_seen = False
    min_ram = 0
    min_vram = 0
    timed_out = False
    no_output_timed_out = False
    idle_timed_out = False
    aborted = False
    memory_guardrail_stopped = False
    context_overflow = False
    context_required = 0
    guardrail_state = MemoryGuardrailState()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    if proc.stdout:
        selector.register(proc.stdout, selectors.EVENT_READ, 'stdout')
    if proc.stderr:
        selector.register(proc.stderr, selectors.EVENT_READ, 'stderr')

    def remember_memory():
        nonlocal min_ram, min_vram, memory_guardrail_stopped
        snap = sample_memory(app)
        ram = int(snap.get('ram_available', 0) or 0)
        vram = int(snap.get('gpu_memory_free', 0) or 0)
        min_ram = ram if min_ram <= 0 else min(min_ram, ram)
        if vram > 0:
            min_vram = vram if min_vram <= 0 else min(min_vram, vram)
        decision = guardrail_state.observe(
            HardwareProfile(
                memory_total=int(snap.get('ram_total', 0) or 0),
                memory_available=ram,
                gpu_memory_total=int(snap.get('gpu_memory_total', 0) or 0),
                gpu_memory_free=vram,
            ),
            phase='runtime',
        )
        if decision.should_stop and proc.poll() is None:
            memory_guardrail_stopped = True
            app.terminate_process_group(proc.pid)

    remember_memory()
    try:
        while True:
            if cancel_token is not None and cancel_token.is_cancelled():
                aborted = True
                app.terminate_process_group(proc.pid)
                break
            elapsed = time.monotonic() - started
            if elapsed > timeout and proc.poll() is None:
                timed_out = True
                app.terminate_process_group(proc.pid)
                break
            if first_output is None and no_output_timeout > 0 and elapsed > no_output_timeout and proc.poll() is None:
                timed_out = True
                no_output_timed_out = True
                app.terminate_process_group(proc.pid)
                break
            if (
                first_output is not None
                and idle_output_timeout > 0
                and time.monotonic() - last_meaningful_output_at > idle_output_timeout
                and proc.poll() is None
            ):
                timed_out = True
                idle_timed_out = True
                app.terminate_process_group(proc.pid)
                break
            events = selector.select(timeout=0.25)
            for key, _mask in events:
                line = key.fileobj.readline()
                if not line:
                    try:
                        selector.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue
                now = time.monotonic()
                if first_process_output is None:
                    first_process_output = now - started
                meaningful = is_meaningful_process_output(line)
                if meaningful:
                    if first_output is None:
                        first_output = now - started
                    last_meaningful_output_at = now
                else:
                    startup_output_seen = True
                if key.data == 'stdout':
                    stdout_lines.append(line.rstrip())
                else:
                    stderr_lines.append(line.rstrip())
                if stop_on_context_overflow:
                    required = parse_context_requirement(line)
                    if required:
                        context_required = max(context_required, int(required))
                        context_overflow = True
                        app.terminate_process_group(proc.pid)
                        break
            remember_memory()
            if memory_guardrail_stopped:
                break
            if context_overflow:
                break
            if proc.poll() is not None:
                break
    finally:
        for stream in (proc.stdout, proc.stderr):
            if not stream:
                continue
            try:
                if proc.poll() is not None:
                    rest = stream.read()
                    if rest:
                        target = stdout_lines if stream is proc.stdout else stderr_lines
                        target.extend(rest.splitlines())
            except Exception:
                pass
            try:
                selector.unregister(stream)
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        selector.close()

    try:
        returncode = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        app.terminate_process_group(proc.pid)
        returncode = -9
    elapsed = max(0.001, time.monotonic() - started)
    return {
        'returncode': returncode,
        'timed_out': timed_out,
        'no_output_timeout': no_output_timed_out,
        'idle_output_timeout': idle_timed_out,
        'aborted': aborted,
        'context_overflow': context_overflow,
        'context_required': context_required,
        'memory_guardrail_stopped': memory_guardrail_stopped,
        'elapsed': elapsed,
        'first_output': first_output if first_output is not None else elapsed,
        'first_meaningful_output': first_output if first_output is not None else 0.0,
        'first_process_output': first_process_output if first_process_output is not None else 0.0,
        'startup_output_seen': startup_output_seen,
        'resolved_no_output_timeout': float(no_output_timeout or 0),
        'resolved_idle_output_timeout': float(idle_output_timeout or 0),
        'stdout': stdout_lines[-40:],
        'stderr': stderr_lines[-40:],
        'json_event_tail': json_event_tail(stdout_lines + stderr_lines),
        'raw_event_tail': raw_event_tail(stdout_lines + stderr_lines),
        'min_ram_available': min_ram,
        'min_gpu_memory_free': min_vram,
        **guardrail_state.record_fields(),
    }


def json_event_tail(lines: List[str], limit: int = 20) -> List[str]:
    events: List[str] = []
    for line in lines:
        text = str(line).strip()
        if not text:
            continue
        if text.startswith('data:'):
            text = text[5:].strip()
        if not text or text == '[DONE]':
            continue
        try:
            json.loads(text)
        except Exception:
            continue
        events.append(text)
    return events[-limit:]


def raw_event_tail(lines: List[str], limit: int = 20) -> List[str]:
    raw: List[str] = []
    for line in lines:
        text = compact_message(str(line).strip())
        if text:
            raw.append(text)
    return raw[-limit:]


def detected_unittest_command(lines: List[str]) -> bool:
    pattern = re.compile(r'\bpython(?:3)?(?:\S*)?\s+-m\s+unittest\b|\bunittest\s+-q\b', re.IGNORECASE)
    return any(pattern.search(str(line)) for line in lines)


def verify_fixture(workspace: Path) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'unittest', '-q'],
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=40,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    return result.returncode == 0, compact_message(result.stdout[-1200:])


def run_opencode_task(
    app,
    model: ModelConfig,
    task: WorkflowTask,
    timeout: int = OPENCODE_TASK_TIMEOUT,
    cancel_token: Optional[CancelToken] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='llama-tui-opencode-work-') as workspace_raw:
        with tempfile.TemporaryDirectory(prefix='llama-tui-opencode-home-') as home_raw:
            check_cancelled(cancel_token)
            workspace = Path(workspace_raw)
            home = Path(home_raw)
            write_fixture(workspace, task)
            config_path = write_temp_opencode_config(app, model, home)
            env = isolated_opencode_env(home, config_path)
            command = build_opencode_run_command(app, model, workspace, task.prompt)
            preview = shlex.join(command)
            pressure_payload = current_process_pressure_payload()
            timeout_policy = workflow_timeout_policy(model, pressure_payload)
            app.append_log(model.id, f'OpenCode headless benchmark command: {preview}')
            if progress:
                emit_benchmark_event(
                    progress,
                    'benchmark_phase',
                    model,
                    'opencode',
                    message=f'headless OpenCode command: {preview}',
                    phase='OpenCode benchmark (headless)',
                    candidate=task.name,
                    command=preview,
                )
            run = run_process_with_metrics(
                command,
                workspace,
                env,
                timeout,
                app,
                cancel_token=cancel_token,
                no_output_timeout=timeout_policy.no_output_timeout,
                idle_output_timeout=timeout_policy.idle_output_timeout,
            )
            all_output = list(run.get('stdout', []) or []) + list(run.get('stderr', []) or [])
            unittest_seen = detected_unittest_command(all_output)
            timeout_fields = {
                'resolved_no_output_timeout': int(run.get('resolved_no_output_timeout', timeout_policy.no_output_timeout) or 0),
                'resolved_idle_output_timeout': int(run.get('resolved_idle_output_timeout', timeout_policy.idle_output_timeout) or 0),
                'timeout_policy': timeout_policy.reason,
                'timeout_pressure_level': timeout_policy.pressure_level,
                'startup_output_seen': bool(run.get('startup_output_seen')),
                'first_meaningful_output': float(run.get('first_meaningful_output', 0.0) or 0.0),
                'first_process_output': float(run.get('first_process_output', 0.0) or 0.0),
            }
            guardrail_fields = {
                key: run.get(key)
                for key in (
                    'memory_guardrail_status',
                    'memory_guardrail_reason',
                    'memory_guardrail_action',
                    'memory_guardrail_snapshot',
                    'memory_guardrail_min_ram_available',
                    'memory_guardrail_min_gpu_memory_free',
                    'memory_guardrail_observations',
                )
                if key in run
            }
            if run.get('aborted'):
                return {
                    'task': task.name,
                    'command_preview': preview,
                    'ok': False,
                    'tests_ok': False,
                    'status': 'aborted',
                    'exit_code': int(run.get('returncode', -1)),
                    'timed_out': bool(run.get('timed_out')),
                    'no_output_timeout': bool(run.get('no_output_timeout')),
                    'idle_output_timeout': bool(run.get('idle_output_timeout')),
                    'aborted': True,
                    'elapsed': float(run.get('elapsed', 0.0) or 0.0),
                    'first_output': float(run.get('first_output', 0.0) or 0.0),
                    'min_ram_available': int(run.get('min_ram_available', 0) or 0),
                    'min_gpu_memory_free': int(run.get('min_gpu_memory_free', 0) or 0),
                    'stdout_tail': list(run.get('stdout', []) or [])[-12:],
                    'stderr_tail': list(run.get('stderr', []) or [])[-12:],
                    'json_event_tail': list(run.get('json_event_tail', []) or [])[-12:],
                    'raw_event_tail': list(run.get('raw_event_tail', []) or [])[-12:],
                    'unittest_command_seen': unittest_seen,
                    'detail': 'user requested abort',
                    **timeout_fields,
                    **guardrail_fields,
                }
            check_cancelled(cancel_token)
            tests_ok, test_detail = verify_fixture(workspace)
            stderr = ' | '.join(str(line) for line in run.get('stderr', [])[-8:])
            stdout = ' | '.join(str(line) for line in run.get('stdout', [])[-8:])
            detail = test_detail or stderr or stdout
            context_required = max(
                int(run.get('context_required', 0) or 0),
                int(parse_context_requirement(' | '.join([detail, stderr, stdout])) or 0),
            )
            if context_required:
                status = 'context too small'
                detail = f'OpenCode requested about {context_required} tokens; {detail}'
            elif run.get('memory_guardrail_stopped') or run.get('memory_guardrail_status') == 'memory_guardrail_stopped':
                status = 'memory guardrail stopped'
                detail = f'candidate stopped to protect system memory: {run.get("memory_guardrail_reason", "")}'
            elif run.get('no_output_timeout'):
                status = 'opencode no output timeout'
                detail = f'no meaningful OpenCode output for {timeout_fields["resolved_no_output_timeout"]}s'
            elif run.get('idle_output_timeout'):
                status = 'opencode idle timeout'
                detail = f'no meaningful OpenCode output for {timeout_fields["resolved_idle_output_timeout"]}s after workflow output'
            elif run.get('timed_out'):
                status = 'opencode timed out'
            elif int(run.get('returncode', -1)) != 0:
                status = 'opencode command failed'
            elif tests_ok:
                status = 'tests passed'
            else:
                status = 'tests failed'
            return {
                'task': task.name,
                'command_preview': preview,
                'ok': bool(status == 'tests passed'),
                'tests_ok': tests_ok,
                'status': status,
                'exit_code': int(run.get('returncode', -1)),
                'timed_out': bool(run.get('timed_out')),
                'no_output_timeout': bool(run.get('no_output_timeout')),
                'idle_output_timeout': bool(run.get('idle_output_timeout')),
                'aborted': bool(run.get('aborted')),
                'context_overflow': bool(run.get('context_overflow')),
                'elapsed': float(run.get('elapsed', 0.0) or 0.0),
                'first_output': float(run.get('first_output', 0.0) or 0.0),
                'min_ram_available': int(run.get('min_ram_available', 0) or 0),
                'min_gpu_memory_free': int(run.get('min_gpu_memory_free', 0) or 0),
                'stdout_tail': list(run.get('stdout', []) or [])[-12:],
                'stderr_tail': list(run.get('stderr', []) or [])[-12:],
                'json_event_tail': list(run.get('json_event_tail', []) or [])[-12:],
                'raw_event_tail': list(run.get('raw_event_tail', []) or [])[-12:],
                'unittest_command_seen': unittest_seen,
                'context_required': context_required,
                'detail': concise_failure(detail, limit=500),
                **timeout_fields,
                **guardrail_fields,
            }


def score_opencode_samples(samples: List[Dict[str, object]]) -> float:
    if not samples:
        return 0.0
    total = len(samples)
    passed = sum(1 for sample in samples if sample.get('ok'))
    success_ratio = passed / max(1, total)
    elapsed_values = [float(sample.get('elapsed', 0.0) or 0.0) for sample in samples]
    first_values = [float(sample.get('first_output', 0.0) or 0.0) for sample in samples]
    median_elapsed = statistics.median(elapsed_values) if elapsed_values else 999.0
    median_first = statistics.median(first_values) if first_values else 999.0
    min_ram = min(int(sample.get('min_ram_available', 0) or 0) for sample in samples)
    vram_values = [int(sample.get('min_gpu_memory_free', 0) or 0) for sample in samples if int(sample.get('min_gpu_memory_free', 0) or 0) > 0]
    min_vram = min(vram_values) if vram_values else 0

    score = success_ratio * 1000.0
    score += max(0.0, 240.0 - median_elapsed * 4.0)
    score += max(0.0, 120.0 - median_first * 10.0)
    score += min(80.0, (min_ram / 1024**3) * 8.0)
    if min_vram:
        score += min(80.0, (min_vram / 1024**3) * 25.0)
        if min_vram < 512 * 1024**2:
            score -= 120.0
    if min_ram and min_ram < 1024**3:
        score -= 120.0
    score -= (total - passed) * 180.0
    return round(max(0.0, score), 2)


def summarize_sample_status(samples: List[Dict[str, object]]) -> str:
    if samples and all(sample.get('ok') for sample in samples):
        return 'tests passed'
    statuses = [str(sample.get('status', '') or '') for sample in samples]
    for candidate in (
        'aborted',
        'memory guardrail stopped',
        'context too small',
        'opencode no output timeout',
        'opencode idle timeout',
        'opencode timed out',
        'opencode command failed',
        'tests failed',
    ):
        if candidate in statuses:
            return candidate
    return 'tests failed' if samples else 'failed'


def opencode_failure_summary(records: List[Dict[str, object]]) -> str:
    if not records:
        return 'no candidate completed a task'
    guardrail = next((
        row for row in records
        if isinstance(row, dict)
        and row.get('memory_guardrail_status') in ('memory_guardrail_stopped', 'memory_guardrail_skipped')
    ), None)
    if guardrail:
        return concise_failure(
            f'memory guardrail stopped candidate: {guardrail.get("memory_guardrail_reason", guardrail.get("detail", ""))}',
            limit=500,
        )
    required = max([int(row.get('context_required', 0) or 0) for row in records if isinstance(row, dict)] or [0])
    largest = max([int(row.get('ctx_per_slot', 0) or 0) for row in records if isinstance(row, dict)] or [0])
    if required:
        return f'no candidate completed; OpenCode requested about {required} tokens, largest tested ctx/slot was {largest}'
    best_partial = max(
        (
            row for row in records
            if isinstance(row, dict) and int(row.get('passed', 0) or 0) > 0
        ),
        key=lambda row: (int(row.get('passed', 0) or 0), float(row.get('score', 0.0) or 0.0)),
        default=None,
    )
    if best_partial:
        return concise_failure(
            f'no candidate completed all tasks; best partial passed '
            f'{int(best_partial.get("passed", 0) or 0)}/{int(best_partial.get("tasks", 0) or 0)} '
            f'with status {best_partial.get("status", "failed")}: {best_partial.get("detail", "")}',
            limit=500,
        )
    first_detail = next((str(row.get('detail', '') or '') for row in records if isinstance(row, dict) and row.get('detail')), '')
    first_status = next((str(row.get('status', '') or '') for row in records if isinstance(row, dict) and row.get('status')), 'failed')
    return concise_failure(f'{first_status}: {first_detail}' if first_detail else first_status, limit=500)


def sample_timeout_type(sample: Dict[str, object]) -> str:
    if sample.get('aborted'):
        return 'aborted'
    if sample.get('no_output_timeout'):
        return 'no_output'
    if sample.get('idle_output_timeout'):
        return 'idle'
    if sample.get('timed_out'):
        return 'total'
    return ''


def compact_sample_details(samples: List[Dict[str, object]]) -> List[Dict[str, object]]:
    details: List[Dict[str, object]] = []
    for sample in samples:
        details.append({
            'task': sample.get('task', ''),
            'command_preview': sample.get('command_preview', ''),
            'config_path': sample.get('config_path', ''),
            'status': sample.get('status', ''),
            'ok': bool(sample.get('ok')),
            'tests_ok': bool(sample.get('tests_ok')),
            'exit_code': int(sample.get('exit_code', -1) or -1),
            'timeout_type': sample_timeout_type(sample),
            'timed_out': bool(sample.get('timed_out')),
            'no_output_timeout': bool(sample.get('no_output_timeout')),
            'idle_output_timeout': bool(sample.get('idle_output_timeout')),
            'context_overflow': bool(sample.get('context_overflow')),
            'unittest_command_seen': bool(sample.get('unittest_command_seen')),
            'context_required': int(sample.get('context_required', 0) or 0),
            'resolved_no_output_timeout': int(sample.get('resolved_no_output_timeout', 0) or 0),
            'resolved_idle_output_timeout': int(sample.get('resolved_idle_output_timeout', 0) or 0),
            'timeout_policy': sample.get('timeout_policy', ''),
            'timeout_pressure_level': sample.get('timeout_pressure_level', ''),
            'startup_output_seen': bool(sample.get('startup_output_seen')),
            'first_meaningful_output': float(sample.get('first_meaningful_output', 0.0) or 0.0),
            'first_process_output': float(sample.get('first_process_output', 0.0) or 0.0),
            'memory_guardrail_status': sample.get('memory_guardrail_status', ''),
            'memory_guardrail_reason': sample.get('memory_guardrail_reason', ''),
            'detail': concise_failure(str(sample.get('detail', '')), limit=320),
            'stdout_tail': list(sample.get('stdout_tail', []) or [])[-8:],
            'stderr_tail': list(sample.get('stderr_tail', []) or [])[-8:],
            'json_event_tail': list(sample.get('json_event_tail', []) or [])[-8:],
            'raw_event_tail': list(sample.get('raw_event_tail', []) or [])[-8:],
        })
    return details


def benchmark_opencode_workflow(
    app,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    check_cancelled(cancel_token)
    if not app.command_exists('opencode'):
        return False, '❌ opencode command not found in PATH.'
    cli_ok, cli_msg = opencode_cli_preflight()
    if not cli_ok:
        return False, f'❌ opencode preflight failed: {cli_msg}'
    if progress:
        progress(cli_msg)
    preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'opencode', progress, cancel_token)
    if not preflight_ok:
        return False, preflight_msg

    profile = app.hardware_profile(refresh=True)
    candidates = expand_workflow_cache_ram_candidates(opencode_candidate_models(model, profile), profile)
    vscode = detect_vscode_pressure()
    pressure_payload = current_process_pressure_payload()
    records: List[Dict[str, object]] = []
    results: List[Dict[str, object]] = []
    total_steps = max(1, len(candidates) * max(1, len(OPENCODE_WORKFLOW_TASKS)))
    if not candidates:
        ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
        record = {
            'preset': 'opencode',
            'tier': 'estimated',
            'status': 'context too small',
            'score': 0.0,
            'seconds': 0.0,
            'passed': 0,
            'tasks': len(OPENCODE_WORKFLOW_TASKS),
            'ctx': int(getattr(model, 'ctx', 0) or 0),
            'ctx_per_slot': ctx_per_slot(model),
            'parallel': int(getattr(model, 'parallel', 0) or 0),
            'threads': int(getattr(model, 'threads', 0) or 0),
            'ngl': int(getattr(model, 'ngl', 0) or 0),
            'detail': f'not OpenCode-ready: cannot fit minimum ctx={ctx_min}',
            'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
        }
        record.update(benchmark_record_context(model))
        records.append(record)
        recorded_model = clone_model_config(model)
        recorded_model.last_opencode_benchmark_results = records
        app.add_or_update(recorded_model)
        msg = f'❌ not OpenCode-ready: cannot fit minimum ctx={ctx_min}'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_result',
            model,
            'opencode',
            message=msg,
            phase='candidate discovery',
            completed=0,
            total=1,
            record=record,
        )
        emit_benchmark_event(
            progress,
            'benchmark_error',
            model,
            'opencode',
            message=msg,
            phase='failed',
            records=records,
        )
        return False, msg
    if progress:
        progress(
            f'OpenCode benchmark (headless) started: {len(candidates)} candidate(s), '
            f'vscode={vscode["processes"]} proc/{vscode["rss_mib"]} MiB, '
            f'arch={architecture_payload(model).get("architecture_label", "Unknown")}, '
            f'{profile.short_summary()} {pressure_payload.get("process_pressure_detail", "")}'
        )
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'opencode',
        message=(
            f'OpenCode benchmark (headless) started: {len(candidates)} candidate(s), '
            f'vscode={vscode["processes"]} proc/{vscode["rss_mib"]} MiB, '
            f'arch={architecture_payload(model).get("architecture_label", "Unknown")}, '
            f'{profile.short_summary()} {pressure_payload.get("process_pressure_detail", "")}'
        ),
        phase='OpenCode benchmark (headless)',
        completed=0,
        total=total_steps,
    )

    current: Optional[Tuple[str, str, ModelConfig]] = None
    completed_steps = 0
    observed_context_floor = max(0, int(observed_opencode_context_floor(model) or 0))
    try:
        for attempt, (preset, tier, candidate, tune_msg) in enumerate(candidates, start=1):
            check_cancelled(cancel_token)
            runtime_profile = None
            if tier == 'measured':
                _measured_candidate, runtime_profile = model_and_runtime_profile_from_measured_profile(model, preset)
            if observed_context_floor and ctx_per_slot(candidate) < observed_context_floor:
                record = {
                    'preset': preset,
                    'tier': tier,
                    'status': 'context too small',
                    'score': 0.0,
                    'seconds': 0.0,
                    'passed': 0,
                    'tasks': len(OPENCODE_WORKFLOW_TASKS),
                    'detail': f'skipped ctx/slot={ctx_per_slot(candidate)} below observed OpenCode floor {observed_context_floor}',
                    'ctx': int(getattr(candidate, 'ctx', 0) or 0),
                    'ctx_per_slot': ctx_per_slot(candidate),
                    'parallel': int(getattr(candidate, 'parallel', 0) or 0),
                    'threads': int(getattr(candidate, 'threads', 0) or 0),
                    'ngl': int(getattr(candidate, 'ngl', 0) or 0),
                    'context_required': observed_context_floor,
                    'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                }
                record.update(benchmark_record_context(candidate))
                records.append(record)
                completed_steps += len(OPENCODE_WORKFLOW_TASKS)
                if progress:
                    progress(f'opencode candidate {attempt}/{len(candidates)} skipped: {record["detail"]}')
                emit_benchmark_event(
                    progress,
                    'benchmark_result',
                    model,
                    'opencode',
                    message=f'opencode candidate {attempt}/{len(candidates)} skipped below observed context floor',
                    phase='OpenCode benchmark (headless)',
                    completed=completed_steps,
                    total=total_steps,
                    candidate=f'{preset}/{tier}',
                    record=record,
                )
                continue
            guardrail_state = MemoryGuardrailState()
            guardrail_profile = app.hardware_profile(refresh=True)
            estimated_safe_ctx = candidate_safe_context_estimate(candidate, guardrail_profile)
            try:
                pressure_score = float(current_process_pressure_payload().get('process_pressure_score', 0.0) or 0.0)
            except Exception:
                pressure_score = 0.0
            admission = guardrail_state.observe(
                guardrail_profile,
                phase='admission',
                candidate_ctx=int(getattr(candidate, 'ctx', 0) or 0),
                safe_ctx=estimated_safe_ctx,
                observed_floor=observed_context_floor,
                required_for_floor=observed_context_floor > 0 and ctx_per_slot(candidate) >= observed_context_floor,
                pressure_score=pressure_score,
            )
            if admission.should_skip:
                record = {
                    'preset': preset,
                    'tier': tier,
                    'status': 'memory guardrail skipped',
                    'score': 0.0,
                    'seconds': 0.0,
                    'passed': 0,
                    'tasks': len(OPENCODE_WORKFLOW_TASKS),
                    'detail': f'candidate skipped by memory guardrail: {admission.reason}',
                    'ctx': int(getattr(candidate, 'ctx', 0) or 0),
                    'ctx_per_slot': ctx_per_slot(candidate),
                    'parallel': int(getattr(candidate, 'parallel', 0) or 0),
                    'threads': int(getattr(candidate, 'threads', 0) or 0),
                    'ngl': int(getattr(candidate, 'ngl', 0) or 0),
                    'estimated_safe_ctx': estimated_safe_ctx,
                    'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                    'failure_category': 'MEMORY_GUARDRAIL',
                    'failure_reason': admission.reason,
                }
                record.update(memory_guardrail_record_fields(admission))
                record.update(benchmark_record_context(candidate))
                records.append(record)
                completed_steps += len(OPENCODE_WORKFLOW_TASKS)
                if progress:
                    progress(f'opencode candidate {attempt}/{len(candidates)} skipped by memory guardrail: {admission.reason}')
                emit_benchmark_event(
                    progress,
                    'benchmark_result',
                    model,
                    'opencode',
                    message=f'opencode candidate {attempt}/{len(candidates)} skipped by memory guardrail',
                    phase='OpenCode benchmark (headless)',
                    completed=completed_steps,
                    total=total_steps,
                    candidate=f'{preset}/{tier}',
                    record=record,
                )
                continue
            current = (preset, tier, candidate)
            if progress:
                progress(
                    f'headless opencode candidate {attempt}/{len(candidates)} {preset}/{tier}: '
                    f'ctx={candidate.ctx} slot={ctx_per_slot(candidate)} parallel={candidate.parallel} | {tune_msg}'
                )
            emit_benchmark_event(
                progress,
                'benchmark_candidate',
                model,
                'opencode',
                message=(
                    f'headless opencode candidate {attempt}/{len(candidates)} {preset}/{tier}: '
                    f'ctx={candidate.ctx} slot={ctx_per_slot(candidate)} parallel={candidate.parallel}'
                ),
                phase='OpenCode benchmark (headless)',
                completed=completed_steps,
                total=total_steps,
                candidate=f'{preset}/{tier}',
            )
            try:
                ok, msg = app.start(candidate, runtime_profile=runtime_profile)
            except TypeError:
                ok, msg = app.start(candidate)
            if not ok:
                record = {
                    'preset': preset,
                    'tier': tier,
                    'status': 'start failed',
                    'score': 0.0,
                    'seconds': 0.0,
                    'passed': 0,
                    'tasks': len(OPENCODE_WORKFLOW_TASKS),
                    'detail': concise_failure(msg, limit=500),
                    'ctx': int(getattr(candidate, 'ctx', 0) or 0),
                    'ctx_per_slot': ctx_per_slot(candidate),
                    'parallel': int(getattr(candidate, 'parallel', 0) or 0),
                    'threads': int(getattr(candidate, 'threads', 0) or 0),
                    'ngl': int(getattr(candidate, 'ngl', 0) or 0),
                    'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                }
                record.update(benchmark_record_context(candidate))
                records.append(record)
                if progress:
                    progress(f'opencode candidate {attempt}/{len(candidates)} failed to start: {concise_failure(msg)}')
                completed_steps += len(OPENCODE_WORKFLOW_TASKS)
                emit_benchmark_event(
                    progress,
                    'benchmark_result',
                    model,
                    'opencode',
                    message=f'opencode candidate {attempt}/{len(candidates)} failed to start',
                    phase='OpenCode benchmark (headless)',
                    completed=completed_steps,
                    total=total_steps,
                    candidate=f'{preset}/{tier}',
                    record=record,
                )
                continue

            samples: List[Dict[str, object]] = []
            watchdog_stop = None
            watchdog_thread = None
            try:
                watchdog_stop, watchdog_thread = start_memory_guardrail_watchdog(
                    lambda: app.hardware_profile(refresh=True),
                    lambda: app.stop(candidate, managed_only=True),
                    guardrail_state,
                    candidate_ctx=int(getattr(candidate, 'ctx', 0) or 0),
                    safe_ctx=estimated_safe_ctx,
                    observed_floor=observed_context_floor,
                    required_for_floor=observed_context_floor > 0 and ctx_per_slot(candidate) >= observed_context_floor,
                    pressure_score=pressure_score,
                    phase='runtime',
                )
                ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=180, cancel_token=cancel_token)
                if not ready_ok:
                    if guardrail_state.stop_decision is not None:
                        ready_msg = guardrail_state.stop_decision.reason
                    record = {
                        'preset': preset,
                        'tier': tier,
                        'status': 'memory guardrail stopped' if guardrail_state.stop_decision is not None else 'not ready',
                        'score': 0.0,
                        'seconds': 0.0,
                        'passed': 0,
                        'tasks': len(OPENCODE_WORKFLOW_TASKS),
                        'detail': concise_failure(ready_msg, limit=500),
                        'ctx': int(getattr(candidate, 'ctx', 0) or 0),
                        'ctx_per_slot': ctx_per_slot(candidate),
                        'parallel': int(getattr(candidate, 'parallel', 0) or 0),
                        'threads': int(getattr(candidate, 'threads', 0) or 0),
                        'ngl': int(getattr(candidate, 'ngl', 0) or 0),
                        'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                    }
                    if guardrail_state.stop_decision is not None:
                        record['failure_category'] = 'MEMORY_GUARDRAIL'
                        record['failure_reason'] = guardrail_state.stop_decision.reason
                        record.update(guardrail_state.record_fields())
                    record.update(benchmark_record_context(candidate))
                    records.append(record)
                    if progress:
                        progress(f'opencode candidate {attempt}/{len(candidates)} not ready: {concise_failure(ready_msg)}')
                    completed_steps += len(OPENCODE_WORKFLOW_TASKS)
                    emit_benchmark_event(
                        progress,
                        'benchmark_result',
                        model,
                        'opencode',
                        message=f'opencode candidate {attempt}/{len(candidates)} not ready',
                        phase='OpenCode benchmark (headless)',
                        completed=completed_steps,
                        total=total_steps,
                        candidate=f'{preset}/{tier}',
                        record=record,
                    )
                    continue

                provider_ok, provider_msg = opencode_provider_preflight(app, candidate)
                if not provider_ok:
                    record = {
                        'preset': preset,
                        'tier': tier,
                        'status': 'opencode command failed',
                        'score': 0.0,
                        'seconds': 0.0,
                        'passed': 0,
                        'tasks': len(OPENCODE_WORKFLOW_TASKS),
                        'detail': concise_failure(provider_msg, limit=500),
                        'ctx': int(getattr(candidate, 'ctx', 0) or 0),
                        'ctx_per_slot': ctx_per_slot(candidate),
                        'parallel': int(getattr(candidate, 'parallel', 0) or 0),
                        'threads': int(getattr(candidate, 'threads', 0) or 0),
                        'ngl': int(getattr(candidate, 'ngl', 0) or 0),
                        'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                    }
                    record.update(benchmark_record_context(candidate))
                    records.append(record)
                    if progress:
                        progress(f'opencode provider check failed: {concise_failure(provider_msg)}')
                    completed_steps += len(OPENCODE_WORKFLOW_TASKS)
                    emit_benchmark_event(
                        progress,
                        'benchmark_result',
                        model,
                        'opencode',
                        message='opencode provider check failed',
                        phase='OpenCode benchmark (headless)',
                        completed=completed_steps,
                        total=total_steps,
                        candidate=f'{preset}/{tier}',
                        record=record,
                    )
                    continue
                if progress:
                    progress(provider_msg)

                for task_idx, task in enumerate(OPENCODE_WORKFLOW_TASKS, start=1):
                    check_cancelled(cancel_token)
                    if progress:
                        progress(
                            f'headless opencode candidate {attempt}/{len(candidates)} running task {task.name} '
                            f'ctx/slot={ctx_per_slot(candidate)}...'
                        )
                    emit_benchmark_event(
                        progress,
                        'benchmark_phase',
                        model,
                        'opencode',
                        message=f'headless opencode candidate {attempt}/{len(candidates)} task {task.name}',
                        phase='OpenCode benchmark (headless)',
                        completed=completed_steps,
                        total=total_steps,
                        candidate=f'{preset}/{tier} task {task_idx}/{len(OPENCODE_WORKFLOW_TASKS)}',
                    )
                    sample = run_opencode_task(app, candidate, task, cancel_token=cancel_token, progress=progress)
                    samples.append(sample)
                    completed_steps += 1
                    required = int(sample.get('context_required', 0) or 0)
                    if required:
                        observed_context_floor = max(observed_context_floor, required)
                    check_cancelled(cancel_token)
                    if progress:
                        state = str(sample.get('status', 'passed' if sample.get('ok') else 'failed'))
                        progress(
                            f'opencode task {task.name} {state} in {float(sample.get("elapsed", 0.0)):.1f}s '
                            f'exit={int(sample.get("exit_code", -1))} '
                            f'timeout={bool(sample.get("timed_out"))} abort={bool(sample.get("aborted"))}'
                        )
                        if not sample.get('ok') and sample.get('detail'):
                            progress(f'opencode task {task.name} detail: {concise_failure(str(sample.get("detail")), limit=500)}')
                    if required and not sample.get('ok'):
                        if progress:
                            progress(
                                f'opencode candidate {attempt}/{len(candidates)} stopped early: '
                                f'ctx/slot={ctx_per_slot(candidate)} below observed request {required}'
                            )
                        completed_steps += max(0, len(OPENCODE_WORKFLOW_TASKS) - task_idx)
                        break
                    if sample.get('memory_guardrail_status') == 'memory_guardrail_stopped':
                        if progress:
                            progress(
                                f'opencode candidate {attempt}/{len(candidates)} stopped early by memory guardrail: '
                                f'{sample.get("memory_guardrail_reason", "")}'
                            )
                        completed_steps += max(0, len(OPENCODE_WORKFLOW_TASKS) - task_idx)
                        break

                score = score_opencode_samples(samples)
                passed = sum(1 for sample in samples if sample.get('ok'))
                elapsed = sum(float(sample.get('elapsed', 0.0) or 0.0) for sample in samples)
                status_text = summarize_sample_status(samples)
                detail = '; '.join(str(sample.get('detail', '')) for sample in samples if not sample.get('ok')) or 'all tasks passed'
                failing_sample = next((sample for sample in samples if not sample.get('ok')), samples[-1] if samples else {})
                timeout_types = [sample_timeout_type(sample) for sample in samples if sample_timeout_type(sample)]
                exit_codes = [int(sample.get('exit_code', -1) or -1) for sample in samples]
                context_required = max([int(sample.get('context_required', 0) or 0) for sample in samples] or [0])
                timeout_policies = [str(sample.get('timeout_policy', '') or '') for sample in samples if sample.get('timeout_policy')]
                timeout_pressure_levels = [
                    str(sample.get('timeout_pressure_level', '') or '') for sample in samples if sample.get('timeout_pressure_level')
                ]
                resolved_no_output_timeouts = [
                    int(sample.get('resolved_no_output_timeout', 0) or 0) for sample in samples
                    if int(sample.get('resolved_no_output_timeout', 0) or 0) > 0
                ]
                resolved_idle_timeouts = [
                    int(sample.get('resolved_idle_output_timeout', 0) or 0) for sample in samples
                    if int(sample.get('resolved_idle_output_timeout', 0) or 0) > 0
                ]
                record = {
                    'preset': preset,
                    'tier': tier,
                    'status': status_text,
                    'score': score,
                    'seconds': round(elapsed, 2),
                    'first_output': round(statistics.median([float(sample.get('first_output', 0.0) or 0.0) for sample in samples]), 2),
                    'passed': passed,
                    'tasks': len(samples),
                    'ctx': int(getattr(candidate, 'ctx', 0) or 0),
                    'ctx_per_slot': ctx_per_slot(candidate),
                    'parallel': int(getattr(candidate, 'parallel', 0) or 0),
                    'threads': int(getattr(candidate, 'threads', 0) or 0),
                    'ngl': int(getattr(candidate, 'ngl', 0) or 0),
                    'vscode_processes': vscode['processes'],
                    'vscode_rss_mib': vscode['rss_mib'],
                    'exit_code': next((code for code in exit_codes if code != 0), exit_codes[-1] if exit_codes else -1),
                    'timeout_type': timeout_types[0] if timeout_types else '',
                    'timeout_policy': timeout_policies[0] if timeout_policies else '',
                    'timeout_pressure_level': timeout_pressure_levels[0] if timeout_pressure_levels else '',
                    'resolved_no_output_timeout': max(resolved_no_output_timeouts) if resolved_no_output_timeouts else 0,
                    'resolved_idle_output_timeout': max(resolved_idle_timeouts) if resolved_idle_timeouts else 0,
                    'startup_output_seen': any(bool(sample.get('startup_output_seen')) for sample in samples),
                    'first_meaningful_output': round(statistics.median([
                        float(sample.get('first_meaningful_output', 0.0) or 0.0) for sample in samples
                    ]), 2),
                    'first_process_output': round(statistics.median([
                        float(sample.get('first_process_output', 0.0) or 0.0) for sample in samples
                    ]), 2),
                    'unittest_command_seen': any(bool(sample.get('unittest_command_seen')) for sample in samples),
                    'context_required': context_required,
                    'stdout_tail': list(failing_sample.get('stdout_tail', []) or [])[-8:],
                    'stderr_tail': list(failing_sample.get('stderr_tail', []) or [])[-8:],
                    'task_details': compact_sample_details(samples),
                    'detail': concise_failure(detail, limit=500),
                    'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                }
                record.update(benchmark_record_context(candidate))
                record.update(guardrail_state.record_fields())
                sample_guardrail = next((sample for sample in samples if sample.get('memory_guardrail_status')), None)
                if sample_guardrail:
                    for key, value in sample_guardrail.items():
                        if key.startswith('memory_guardrail_'):
                            record[key] = value
                records.append(record)
                if passed == len(OPENCODE_WORKFLOW_TASKS) and len(samples) == len(OPENCODE_WORKFLOW_TASKS):
                    results.append({
                        'score': score,
                        'preset': preset,
                        'tier': tier,
                        'model': candidate,
                        'elapsed': elapsed,
                        'record': record,
                        'tune_msg': tune_msg,
                    })
                if progress:
                    progress(f'opencode candidate {attempt}/{len(candidates)} scored {score:.2f} ({passed}/{len(samples)} tasks)')
                emit_benchmark_event(
                    progress,
                    'benchmark_result',
                    model,
                    'opencode',
                    message=f'opencode candidate {attempt}/{len(candidates)} scored {score:.2f} ({passed}/{len(samples)} tasks)',
                    phase='OpenCode benchmark (headless)',
                    completed=completed_steps,
                    total=total_steps,
                    candidate=f'{preset}/{tier}',
                    record=record,
                )
            finally:
                if watchdog_stop is not None:
                    watchdog_stop.set()
                if watchdog_thread is not None:
                    watchdog_thread.join(timeout=1.0)
                app.stop(candidate, managed_only=True)
                if progress:
                    progress(f'opencode candidate {attempt}/{len(candidates)} stopped.')
                sleep_with_cancel(0.5, cancel_token)
    except CancelledError:
        if current is not None:
            preset, tier, candidate = current
            app.stop(candidate, managed_only=True)
            record = {
                'preset': preset,
                'tier': tier,
                'status': 'aborted',
                'score': 0.0,
                'seconds': 0.0,
                'passed': 0,
                'tasks': len(OPENCODE_WORKFLOW_TASKS),
                'detail': 'user requested abort',
                'ctx': int(getattr(candidate, 'ctx', 0) or 0),
                'ctx_per_slot': ctx_per_slot(candidate),
                'parallel': int(getattr(candidate, 'parallel', 0) or 0),
                'threads': int(getattr(candidate, 'threads', 0) or 0),
                'ngl': int(getattr(candidate, 'ngl', 0) or 0),
                'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
            }
            record.update(benchmark_record_context(candidate))
            records.append(record)
        recorded_model = clone_model_config(model)
        recorded_model.last_opencode_benchmark_results = records
        app.add_or_update(recorded_model)
        msg = '⚠ aborted; managed processes stopped'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_aborted',
            model,
            'opencode',
            message=msg,
            phase='aborted',
            records=records,
        )
        return False, msg

    recorded_model = clone_model_config(model)
    recorded_model.last_opencode_benchmark_results = records
    if not results:
        summary = opencode_failure_summary(records)
        recorded_model.last_opencode_benchmark_score = 0.0
        recorded_model.last_opencode_benchmark_seconds = 0.0
        recorded_model.last_opencode_benchmark_profile = f'failed: {summary}'
        app.add_or_update(recorded_model)
        msg = f'❌ opencode workflow benchmark failed: {summary}'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_error',
            model,
            'opencode',
            message=msg,
            phase='failed',
            records=records,
        )
        return False, msg

    best = max(results, key=lambda item: workflow_cache_ram_selection_key(item['record']))
    cache_ram_profile = workflow_cache_ram_profile_from_record('opencode', best['record'])
    best_model = clone_model_config(best['model'])
    best_model.cache_ram = int(getattr(model, 'cache_ram', 0) or 0)
    best_model.measured_profiles = dict(getattr(model, 'measured_profiles', {}) or {})
    best_model.measured_profiles['opencode_cache_ram'] = cache_ram_profile
    best_model.last_opencode_benchmark_score = round(float(best['score']), 2)
    best_model.last_opencode_benchmark_seconds = round(float(best['elapsed']), 2)
    best_model.last_opencode_benchmark_profile = (
        f'{best["preset"]}/{best["tier"]} '
        f'{float(best["score"]):.2f} score '
        f'{profile.short_summary()}'
    )
    best_model.last_opencode_benchmark_results = records
    app.add_or_update(best_model)
    sync_msg = sync_opencode_after_tuning(app)
    msg = (
        f'✅ opencode workflow winner: {best_model.id} {best["preset"]}/{best["tier"]} '
        f'score={float(best["score"]):.2f} ctx={best_model.ctx} parallel={best_model.parallel} '
        f'threads={best_model.threads} ngl={best_model.ngl} '
        f'cache_ram_rec={int(cache_ram_profile.get("cache_ram_mib", 0) or 0)} MiB | {sync_msg}'
    )
    if progress:
        progress(msg)
    emit_benchmark_event(
        progress,
        'benchmark_done',
        best_model,
        'opencode',
        message=msg,
        phase='complete',
        completed=total_steps,
        total=total_steps,
        records=records,
    )
    return True, msg
