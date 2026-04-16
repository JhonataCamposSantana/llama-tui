from dataclasses import dataclass, field
from typing import Dict, List

from .constants import DEFAULT_HOST


@dataclass
class ModelConfig:
    id: str
    name: str
    path: str
    alias: str
    port: int
    host: str = DEFAULT_HOST
    ctx: int = 8192
    threads: int = 6
    ngl: int = 999
    temp: float = 0.7
    parallel: int = 1
    cache_ram: int = 0
    flash_attn: bool = True
    jinja: bool = True
    output: int = 4096
    optimize_mode: str = 'max_context_safe'
    optimize_tier: str = 'moderate'
    ctx_min: int = 2048
    ctx_max: int = 131072
    memory_reserve_percent: int = 25
    last_good_ctx: int = 0
    last_good_parallel: int = 0
    last_benchmark_tokens_per_sec: float = 0.0
    last_benchmark_seconds: float = 0.0
    last_benchmark_profile: str = ''
    last_benchmark_results: List[Dict[str, object]] = field(default_factory=list)
    measured_profiles: Dict[str, Dict[str, object]] = field(default_factory=dict)
    benchmark_runs: List[Dict[str, object]] = field(default_factory=list)
    benchmark_fingerprint: str = ''
    default_benchmark_status: str = ''
    default_benchmark_at: str = ''
    last_opencode_benchmark_score: float = 0.0
    last_opencode_benchmark_seconds: float = 0.0
    last_opencode_benchmark_profile: str = ''
    last_opencode_benchmark_results: List[Dict[str, object]] = field(default_factory=list)
    last_hermes_benchmark_score: float = 0.0
    last_hermes_benchmark_seconds: float = 0.0
    last_hermes_benchmark_profile: str = ''
    last_hermes_benchmark_results: List[Dict[str, object]] = field(default_factory=list)
    enabled: bool = True
    runtime: str = 'llama.cpp'
    source: str = 'manual'
    extra_args: List[str] = field(default_factory=list)

@dataclass
class OpencodeSettings:
    path: str = ''
    backup_dir: str = ''
    timeout: int = 600000
    chunk_timeout: int = 60000
    instructions: List[str] = field(default_factory=lambda: ['./instructions/warm.md'])
    build_prompt: str = '{file:./prompts/friendly-build.txt}'
    plan_prompt: str = '{file:./prompts/friendly-build.txt}'
    default_model_id: str = ''
    small_model_id: str = ''
    build_model_id: str = ''
    plan_model_id: str = ''
    terminal_command: str = ''
    last_workspace_path: str = ''

@dataclass
class HermesSettings:
    command: str = 'hermes'
    home_root: str = ''
    default_model_id: str = ''
    code_model_id: str = ''
    terminal_command: str = ''
    last_workspace_path: str = ''
    toolsets: List[str] = field(default_factory=lambda: ['terminal', 'file', 'todo'])
    max_turns: int = 20
    quiet: bool = True
    min_context_tokens: int = 64000
    experimental_context_override_tokens: int = 0
    allow_experimental_context_override: bool = False
