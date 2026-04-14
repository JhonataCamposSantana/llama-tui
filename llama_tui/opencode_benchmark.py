import json
import os
import selectors
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
    benchmark_candidate_models,
    clone_model_config,
    concise_failure,
    sync_opencode_after_tuning,
)
from .control import CancelToken, CancelledError, check_cancelled, sleep_with_cancel
from .hardware import read_meminfo_bytes
from .models import ModelConfig
from .textutil import compact_message

OPENCODE_PREFLIGHT_TIMEOUT = 8
OPENCODE_TASK_TIMEOUT = 300
OPENCODE_BENCHMARK_CANDIDATES = 4


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
            'build': {'model': model_ref},
            'plan': {'model': model_ref},
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return config_path


def shlex_python() -> str:
    return Path(sys.executable).name


def isolated_opencode_env(home: Path, config_path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env['HOME'] = str(home)
    env['XDG_CONFIG_HOME'] = str(home / '.config')
    env['XDG_DATA_HOME'] = str(home / '.local' / 'share')
    env['XDG_STATE_HOME'] = str(home / '.local' / 'state')
    env['OPENCODE_CONFIG'] = str(config_path)
    env['OPENCODE_DISABLE_AUTOUPDATE'] = 'true'
    env['OPENCODE_DISABLE_PRUNE'] = 'true'
    env['OPENCODE_DISABLE_MODELS_FETCH'] = 'true'
    env['OPENCODE_CLIENT'] = 'llama-tui-benchmark'
    return env


def opencode_cli_preflight(timeout: int = OPENCODE_PREFLIGHT_TIMEOUT) -> Tuple[bool, str]:
    checks = [
        ['opencode', '--version'],
        ['opencode', 'run', '--help'],
    ]
    details = []
    for command in checks:
        try:
            result = subprocess.run(
                command,
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


def build_opencode_run_command(app, model: ModelConfig, workspace: Path, prompt: str) -> List[str]:
    return [
        'opencode',
        'run',
        '--pure',
        '--model', app.opencode_model_ref(model),
        '--agent', 'build',
        '--format', 'json',
        '--dir', str(workspace),
        prompt,
    ]


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
) -> Dict[str, object]:
    check_cancelled(cancel_token)
    started = time.monotonic()
    first_output: Optional[float] = None
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    min_ram = 0
    min_vram = 0
    timed_out = False
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
        'aborted': aborted,
        'elapsed': elapsed,
        'first_output': first_output if first_output is not None else elapsed,
        'stdout': stdout_lines[-40:],
        'stderr': stderr_lines[-40:],
        'min_ram_available': min_ram,
        'min_gpu_memory_free': min_vram,
    }


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
            run = run_process_with_metrics(command, workspace, env, timeout, app, cancel_token=cancel_token)
            if run.get('aborted'):
                return {
                    'task': task.name,
                    'ok': False,
                    'tests_ok': False,
                    'exit_code': int(run.get('returncode', -1)),
                    'timed_out': bool(run.get('timed_out')),
                    'aborted': True,
                    'elapsed': float(run.get('elapsed', 0.0) or 0.0),
                    'first_output': float(run.get('first_output', 0.0) or 0.0),
                    'min_ram_available': int(run.get('min_ram_available', 0) or 0),
                    'min_gpu_memory_free': int(run.get('min_gpu_memory_free', 0) or 0),
                    'detail': 'user requested abort',
                }
            check_cancelled(cancel_token)
            tests_ok, test_detail = verify_fixture(workspace)
            stderr = ' | '.join(str(line) for line in run.get('stderr', [])[-8:])
            stdout = ' | '.join(str(line) for line in run.get('stdout', [])[-8:])
            detail = test_detail or stderr or stdout
            return {
                'task': task.name,
                'ok': bool(run.get('returncode') == 0 and tests_ok),
                'tests_ok': tests_ok,
                'exit_code': int(run.get('returncode', -1)),
                'timed_out': bool(run.get('timed_out')),
                'aborted': bool(run.get('aborted')),
                'elapsed': float(run.get('elapsed', 0.0) or 0.0),
                'first_output': float(run.get('first_output', 0.0) or 0.0),
                'min_ram_available': int(run.get('min_ram_available', 0) or 0),
                'min_gpu_memory_free': int(run.get('min_gpu_memory_free', 0) or 0),
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
    candidates = benchmark_candidate_models(model, profile)[:OPENCODE_BENCHMARK_CANDIDATES]
    vscode = detect_vscode_pressure()
    records: List[Dict[str, object]] = []
    results: List[Dict[str, object]] = []
    if progress:
        progress(
            f'opencode workflow benchmark started: {len(candidates)} candidate(s), '
            f'vscode={vscode["processes"]} proc/{vscode["rss_mib"]} MiB, {profile.short_summary()}'
        )

    current: Optional[Tuple[str, str, ModelConfig]] = None
    try:
        for attempt, (preset, tier, candidate, tune_msg) in enumerate(candidates, start=1):
            check_cancelled(cancel_token)
            current = (preset, tier, candidate)
            if progress:
                progress(f'opencode candidate {attempt}/{len(candidates)} {preset}/{tier}: {tune_msg}')
            ok, msg = app.start(candidate)
            if not ok:
                records.append({
                    'preset': preset,
                    'tier': tier,
                    'status': 'start failed',
                    'score': 0.0,
                    'seconds': 0.0,
                    'passed': 0,
                    'tasks': len(OPENCODE_WORKFLOW_TASKS),
                    'detail': concise_failure(msg, limit=500),
                    'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                })
                if progress:
                    progress(f'opencode candidate {attempt}/{len(candidates)} failed to start: {concise_failure(msg)}')
                continue

            samples: List[Dict[str, object]] = []
            try:
                ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=180, cancel_token=cancel_token)
                if not ready_ok:
                    records.append({
                        'preset': preset,
                        'tier': tier,
                        'status': 'not ready',
                        'score': 0.0,
                        'seconds': 0.0,
                        'passed': 0,
                        'tasks': len(OPENCODE_WORKFLOW_TASKS),
                        'detail': concise_failure(ready_msg, limit=500),
                        'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
                    })
                    if progress:
                        progress(f'opencode candidate {attempt}/{len(candidates)} not ready: {concise_failure(ready_msg)}')
                    continue

                for task in OPENCODE_WORKFLOW_TASKS:
                    check_cancelled(cancel_token)
                    if progress:
                        progress(f'opencode candidate {attempt}/{len(candidates)} running task {task.name}...')
                    sample = run_opencode_task(app, candidate, task, cancel_token=cancel_token)
                    samples.append(sample)
                    check_cancelled(cancel_token)
                    if progress:
                        state = 'passed' if sample.get('ok') else 'failed'
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
                status_text = 'ok' if passed == len(samples) else ('partial' if passed else 'failed')
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
                    'parallel': int(getattr(candidate, 'parallel', 0) or 0),
                    'threads': int(getattr(candidate, 'threads', 0) or 0),
                    'ngl': int(getattr(candidate, 'ngl', 0) or 0),
                    'vscode_processes': vscode['processes'],
                    'vscode_rss_mib': vscode['rss_mib'],
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
                'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
            })
        recorded_model = clone_model_config(model)
        recorded_model.last_opencode_benchmark_results = records
        app.add_or_update(recorded_model)
        msg = '⚠ aborted; managed processes stopped'
        if progress:
            progress(msg)
        return False, msg

    recorded_model = clone_model_config(model)
    recorded_model.last_opencode_benchmark_results = records
    if not results:
        app.add_or_update(recorded_model)
        msg = '❌ opencode workflow benchmark failed: no candidate completed a task'
        if progress:
            progress(msg)
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
    return True, msg
