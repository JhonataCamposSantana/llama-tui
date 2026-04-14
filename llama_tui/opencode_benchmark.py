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
    configure_adaptive_candidate,
    clone_model_config,
    concise_failure,
    ctx_per_slot,
    emit_benchmark_event,
    model_from_measured_profile,
    parse_context_requirement,
    sync_opencode_after_tuning,
)
from .control import CancelToken, CancelledError, check_cancelled, sleep_with_cancel
from .hardware import read_meminfo_bytes
from .models import ModelConfig
from .textutil import compact_message

OPENCODE_PREFLIGHT_TIMEOUT = 20
OPENCODE_TASK_TIMEOUT = 300
OPENCODE_NO_OUTPUT_TIMEOUT = 45
OPENCODE_IDLE_OUTPUT_TIMEOUT = 90
OPENCODE_BENCHMARK_CANDIDATES = 4
OPENCODE_LOG_LEVEL = 'WARN'
OPENCODE_DYNAMIC_CANDIDATE_OBJECTIVE = 'opencode_ready'


@dataclass
class WorkflowTask:
    name: str
    prompt: str
    files: Dict[str, str]


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
                            'context': model.ctx,
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

    def add(label: str, tier: str, candidate: ModelConfig, detail: str):
        key = (int(getattr(candidate, 'ctx', 0) or 0), int(getattr(candidate, 'parallel', 1) or 1), tuple(getattr(candidate, 'extra_args', []) or []))
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, tier, candidate, detail))

    for key in ('opencode_ready', 'long_context', 'auto', 'fast_chat'):
        measured = model_from_measured_profile(model, key)
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
        points = sorted(set([
            ctx_min,
            max(ctx_min, upper // 2),
            upper,
        ]))
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
        'gpu_memory_free': int(profile.gpu_memory_free or 0),
    }


def run_process_with_metrics(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout: int,
    app,
    cancel_token: Optional[CancelToken] = None,
    no_output_timeout: int = OPENCODE_NO_OUTPUT_TIMEOUT,
    idle_output_timeout: int = OPENCODE_IDLE_OUTPUT_TIMEOUT,
) -> Dict[str, object]:
    check_cancelled(cancel_token)
    started = time.monotonic()
    first_output: Optional[float] = None
    last_output_at = started
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    min_ram = 0
    min_vram = 0
    timed_out = False
    no_output_timed_out = False
    idle_timed_out = False
    aborted = False
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
        nonlocal min_ram, min_vram
        snap = sample_memory(app)
        ram = int(snap.get('ram_available', 0) or 0)
        vram = int(snap.get('gpu_memory_free', 0) or 0)
        min_ram = ram if min_ram <= 0 else min(min_ram, ram)
        if vram > 0:
            min_vram = vram if min_vram <= 0 else min(min_vram, vram)

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
                and time.monotonic() - last_output_at > idle_output_timeout
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
                if first_output is None:
                    first_output = time.monotonic() - started
                last_output_at = time.monotonic()
                if key.data == 'stdout':
                    stdout_lines.append(line.rstrip())
                else:
                    stderr_lines.append(line.rstrip())
            remember_memory()
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
        'elapsed': elapsed,
        'first_output': first_output if first_output is not None else elapsed,
        'stdout': stdout_lines[-40:],
        'stderr': stderr_lines[-40:],
        'json_event_tail': json_event_tail(stdout_lines + stderr_lines),
        'raw_event_tail': raw_event_tail(stdout_lines + stderr_lines),
        'min_ram_available': min_ram,
        'min_gpu_memory_free': min_vram,
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
            app.append_log(model.id, f'OpenCode benchmark command: {preview}')
            if progress:
                progress(f'opencode command: {preview}')
            run = run_process_with_metrics(command, workspace, env, timeout, app, cancel_token=cancel_token)
            all_output = list(run.get('stdout', []) or []) + list(run.get('stderr', []) or [])
            unittest_seen = detected_unittest_command(all_output)
            if run.get('aborted'):
                return {
                    'task': task.name,
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
                }
            check_cancelled(cancel_token)
            tests_ok, test_detail = verify_fixture(workspace)
            stderr = ' | '.join(str(line) for line in run.get('stderr', [])[-8:])
            stdout = ' | '.join(str(line) for line in run.get('stdout', [])[-8:])
            detail = test_detail or stderr or stdout
            context_required = parse_context_requirement(' | '.join([detail, stderr, stdout]))
            if context_required:
                status = 'context too small'
                detail = f'OpenCode requested about {context_required} tokens; {detail}'
            elif run.get('no_output_timeout'):
                status = 'opencode no output timeout'
                detail = f'no OpenCode output for {OPENCODE_NO_OUTPUT_TIMEOUT}s'
            elif run.get('idle_output_timeout'):
                status = 'opencode idle timeout'
                detail = f'no OpenCode output for {OPENCODE_IDLE_OUTPUT_TIMEOUT}s after initial output'
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
                'ok': bool(status == 'tests passed'),
                'tests_ok': tests_ok,
                'status': status,
                'exit_code': int(run.get('returncode', -1)),
                'timed_out': bool(run.get('timed_out')),
                'no_output_timeout': bool(run.get('no_output_timeout')),
                'idle_output_timeout': bool(run.get('idle_output_timeout')),
                'aborted': bool(run.get('aborted')),
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


def compact_sample_details(samples: List[Dict[str, object]]) -> List[Dict[str, object]]:
    details: List[Dict[str, object]] = []
    for sample in samples:
        details.append({
            'task': sample.get('task', ''),
            'status': sample.get('status', ''),
            'ok': bool(sample.get('ok')),
            'tests_ok': bool(sample.get('tests_ok')),
            'exit_code': int(sample.get('exit_code', -1) or -1),
            'timed_out': bool(sample.get('timed_out')),
            'no_output_timeout': bool(sample.get('no_output_timeout')),
            'idle_output_timeout': bool(sample.get('idle_output_timeout')),
            'unittest_command_seen': bool(sample.get('unittest_command_seen')),
            'context_required': int(sample.get('context_required', 0) or 0),
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
    status, _detail = app.health(model)
    if status in ('READY', 'LOADING', 'STARTING') or app.get_pid(model):
        return False, '❌ Stop the model before running opencode workflow benchmark.'

    profile = app.hardware_profile(refresh=True)
    candidates = opencode_candidate_models(model, profile)
    vscode = detect_vscode_pressure()
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
            f'opencode workflow benchmark started: {len(candidates)} candidate(s), '
            f'vscode={vscode["processes"]} proc/{vscode["rss_mib"]} MiB, {profile.short_summary()}'
        )
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'opencode',
        message=(
            f'opencode workflow benchmark started: {len(candidates)} candidate(s), '
            f'vscode={vscode["processes"]} proc/{vscode["rss_mib"]} MiB, {profile.short_summary()}'
        ),
        phase='starting',
        completed=0,
        total=total_steps,
    )

    current: Optional[Tuple[str, str, ModelConfig]] = None
    completed_steps = 0
    try:
        for attempt, (preset, tier, candidate, tune_msg) in enumerate(candidates, start=1):
            check_cancelled(cancel_token)
            current = (preset, tier, candidate)
            if progress:
                progress(
                    f'opencode candidate {attempt}/{len(candidates)} {preset}/{tier}: '
                    f'ctx={candidate.ctx} slot={ctx_per_slot(candidate)} parallel={candidate.parallel} | {tune_msg}'
                )
            emit_benchmark_event(
                progress,
                'benchmark_candidate',
                model,
                'opencode',
                message=(
                    f'opencode candidate {attempt}/{len(candidates)} {preset}/{tier}: '
                    f'ctx={candidate.ctx} slot={ctx_per_slot(candidate)} parallel={candidate.parallel}'
                ),
                phase='launching candidate',
                completed=completed_steps,
                total=total_steps,
                candidate=f'{preset}/{tier}',
            )
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
                    phase='candidate failed',
                    completed=completed_steps,
                    total=total_steps,
                    candidate=f'{preset}/{tier}',
                    record=record,
                )
                continue

            samples: List[Dict[str, object]] = []
            try:
                ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=180, cancel_token=cancel_token)
                if not ready_ok:
                    record = {
                        'preset': preset,
                        'tier': tier,
                        'status': 'not ready',
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
                        phase='candidate failed',
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
                        phase='candidate failed',
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
                            f'opencode candidate {attempt}/{len(candidates)} running task {task.name} '
                            f'ctx/slot={ctx_per_slot(candidate)}...'
                        )
                    emit_benchmark_event(
                        progress,
                        'benchmark_phase',
                        model,
                        'opencode',
                        message=f'opencode candidate {attempt}/{len(candidates)} task {task.name}',
                        phase='running workflow tasks',
                        completed=completed_steps,
                        total=total_steps,
                        candidate=f'{preset}/{tier} task {task_idx}/{len(OPENCODE_WORKFLOW_TASKS)}',
                    )
                    sample = run_opencode_task(app, candidate, task, cancel_token=cancel_token, progress=progress)
                    samples.append(sample)
                    completed_steps += 1
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

                score = score_opencode_samples(samples)
                passed = sum(1 for sample in samples if sample.get('ok'))
                elapsed = sum(float(sample.get('elapsed', 0.0) or 0.0) for sample in samples)
                status_text = summarize_sample_status(samples)
                detail = '; '.join(str(sample.get('detail', '')) for sample in samples if not sample.get('ok')) or 'all tasks passed'
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
                    'task_details': compact_sample_details(samples),
                    'detail': concise_failure(detail, limit=500),
                    'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                }
                records.append(record)
                if score > 0:
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
                    phase='candidate complete',
                    completed=completed_steps,
                    total=total_steps,
                    candidate=f'{preset}/{tier}',
                    record=record,
                )
            finally:
                app.stop(candidate, managed_only=True)
                if progress:
                    progress(f'opencode candidate {attempt}/{len(candidates)} stopped.')
                sleep_with_cancel(0.5, cancel_token)
    except CancelledError:
        if current is not None:
            preset, tier, candidate = current
            app.stop(candidate, managed_only=True)
            records.append({
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
            })
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
        app.add_or_update(recorded_model)
        msg = '❌ opencode workflow benchmark failed: no candidate completed a task'
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

    best = max(results, key=lambda item: float(item['score']))
    best_model = best['model']
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
        f'threads={best_model.threads} ngl={best_model.ngl} | {sync_msg}'
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
