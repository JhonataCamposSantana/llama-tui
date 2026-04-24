import os
from dataclasses import dataclass
from typing import List, Optional

BUUN_KV_MODES = ('turbo4', 'turbo3_tcq', 'turbo2_tcq', 'turbo3', 'turbo2')


@dataclass(frozen=True)
class RuntimeProfile:
    engine: str
    display_name: str
    server_command: str
    kv_mode: str = ''
    context_override: Optional[int] = None
    experimental: bool = False

    @property
    def is_buun(self) -> bool:
        return self.engine == 'buun'

    def llama_extra_args(self) -> List[str]:
        if not self.is_buun:
            return []
        kv = (self.kv_mode or 'turbo4').strip() or 'turbo4'
        return ['-fa', '-ctk', kv, '-ctv', kv]

    def header_indicator(self) -> str:
        kv = (self.kv_mode or '-').strip() or '-'
        ctx = str(self.context_override) if self.context_override is not None else 'model default'
        suffix = ' | Experimental' if self.experimental else ''
        return f'Engine: {self.display_name} | KV: {kv} | Context: {ctx}{suffix}'


def make_runtime_profile(
    engine: str,
    default_llama_server: str,
    ctx_override: Optional[int] = None,
    kv_mode: str = '',
) -> RuntimeProfile:
    normalized = (engine or 'llama.cpp').strip().lower()
    if normalized == 'buun':
        command = os.environ.get('BUUN_LLAMA_SERVER_BIN') or default_llama_server
        kv = (kv_mode or 'turbo4').strip() or 'turbo4'
        return RuntimeProfile(
            engine='buun',
            display_name='buun-llama-cpp',
            server_command=command,
            kv_mode=kv,
            context_override=ctx_override,
            experimental=True,
        )
    return RuntimeProfile(
        engine='llama.cpp',
        display_name='llama.cpp',
        server_command=default_llama_server,
        kv_mode=(kv_mode or '').strip(),
        context_override=ctx_override,
        experimental=False,
    )
