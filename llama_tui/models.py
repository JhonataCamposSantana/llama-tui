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
    last_opencode_benchmark_score: float = 0.0
    last_opencode_benchmark_seconds: float = 0.0
    last_opencode_benchmark_profile: str = ''
    last_opencode_benchmark_results: List[Dict[str, object]] = field(default_factory=list)
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
