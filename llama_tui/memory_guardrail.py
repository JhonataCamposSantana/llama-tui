import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .hardware import HardwareProfile

GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(frozen=True)
class MemoryGuardrailPolicy:
    ram_critical_floor_bytes: int = 1 * GIB
    ram_critical_fraction: float = 0.07
    ram_warning_floor_bytes: int = 2 * GIB
    ram_warning_fraction: float = 0.12
    vram_critical_floor_bytes: int = 512 * MIB
    vram_critical_fraction: float = 0.08
    vram_warning_floor_bytes: int = 1 * GIB
    vram_warning_fraction: float = 0.12
    pressure_skip_threshold: float = 0.70
    sample_interval_seconds: float = 1.0

    @classmethod
    def balanced(cls) -> 'MemoryGuardrailPolicy':
        return cls()

    def ram_critical_floor(self, total: int) -> int:
        return _dynamic_floor(total, self.ram_critical_floor_bytes, self.ram_critical_fraction)

    def ram_warning_floor(self, total: int) -> int:
        return _dynamic_floor(total, self.ram_warning_floor_bytes, self.ram_warning_fraction)

    def vram_critical_floor(self, total: int) -> int:
        return _dynamic_floor(total, self.vram_critical_floor_bytes, self.vram_critical_fraction)

    def vram_warning_floor(self, total: int) -> int:
        return _dynamic_floor(total, self.vram_warning_floor_bytes, self.vram_warning_fraction)


@dataclass(frozen=True)
class MemoryGuardrailDecision:
    action: str = 'allow'
    status: str = ''
    reason: str = ''
    snapshot: Dict[str, object] = field(default_factory=dict)
    critical: bool = False
    warning: bool = False

    @property
    def should_stop(self) -> bool:
        return self.action == 'stop'

    @property
    def should_skip(self) -> bool:
        return self.action == 'skip'


@dataclass
class MemoryGuardrailState:
    policy: MemoryGuardrailPolicy = field(default_factory=MemoryGuardrailPolicy.balanced)
    min_ram_available: int = 0
    min_gpu_memory_free: int = 0
    warning_decision: Optional[MemoryGuardrailDecision] = None
    stop_decision: Optional[MemoryGuardrailDecision] = None
    skip_decision: Optional[MemoryGuardrailDecision] = None
    observations: List[Dict[str, object]] = field(default_factory=list)

    def observe(
        self,
        profile: HardwareProfile,
        phase: str = 'runtime',
        candidate_ctx: int = 0,
        safe_ctx: int = 0,
        observed_floor: int = 0,
        required_for_floor: bool = False,
        pressure_score: float = 0.0,
    ) -> MemoryGuardrailDecision:
        decision = memory_guardrail_decision(
            profile,
            policy=self.policy,
            phase=phase,
            candidate_ctx=candidate_ctx,
            safe_ctx=safe_ctx,
            observed_floor=observed_floor,
            required_for_floor=required_for_floor,
            pressure_score=pressure_score,
        )
        snapshot = dict(decision.snapshot)
        if snapshot:
            self.observations.append(snapshot)
            ram = int(snapshot.get('ram_available', 0) or 0)
            vram = int(snapshot.get('gpu_memory_free', 0) or 0)
            if ram > 0:
                self.min_ram_available = ram if self.min_ram_available <= 0 else min(self.min_ram_available, ram)
            if vram > 0:
                self.min_gpu_memory_free = vram if self.min_gpu_memory_free <= 0 else min(self.min_gpu_memory_free, vram)
        if decision.should_stop:
            self.stop_decision = decision
        elif decision.should_skip:
            self.skip_decision = decision
        elif decision.warning:
            self.warning_decision = decision
        return decision

    def record_fields(self) -> Dict[str, object]:
        decision = self.stop_decision or self.skip_decision or self.warning_decision
        if decision is None:
            fields: Dict[str, object] = {}
        else:
            fields = memory_guardrail_record_fields(decision)
        if self.min_ram_available > 0:
            fields['memory_guardrail_min_ram_available'] = self.min_ram_available
        if self.min_gpu_memory_free > 0:
            fields['memory_guardrail_min_gpu_memory_free'] = self.min_gpu_memory_free
        if self.observations:
            fields['memory_guardrail_observations'] = self.observations[-5:]
        return fields


def _dynamic_floor(total: int, absolute: int, fraction: float) -> int:
    total = int(total or 0)
    if total <= 0:
        return int(absolute or 0)
    return max(int(absolute or 0), int(total * float(fraction or 0.0)))


def _gib(value: int) -> float:
    return float(value or 0) / float(GIB)


def _snapshot(
    profile: HardwareProfile,
    policy: MemoryGuardrailPolicy,
    phase: str,
    candidate_ctx: int,
    safe_ctx: int,
    observed_floor: int,
    required_for_floor: bool,
    pressure_score: float,
) -> Dict[str, object]:
    ram_total = int(getattr(profile, 'memory_total', 0) or 0)
    ram_available = int(getattr(profile, 'memory_available', 0) or 0)
    vram_total = int(getattr(profile, 'gpu_memory_total', 0) or 0)
    vram_free = int(getattr(profile, 'gpu_memory_free', 0) or 0)
    return {
        'phase': phase,
        'ram_available': ram_available,
        'ram_total': ram_total,
        'ram_available_gib': round(_gib(ram_available), 2),
        'ram_warning_floor': policy.ram_warning_floor(ram_total),
        'ram_critical_floor': policy.ram_critical_floor(ram_total),
        'gpu_memory_free': vram_free,
        'gpu_memory_total': vram_total,
        'gpu_memory_free_gib': round(_gib(vram_free), 2),
        'vram_warning_floor': policy.vram_warning_floor(vram_total),
        'vram_critical_floor': policy.vram_critical_floor(vram_total),
        'candidate_ctx': int(candidate_ctx or 0),
        'safe_ctx': int(safe_ctx or 0),
        'observed_floor': int(observed_floor or 0),
        'required_for_floor': bool(required_for_floor),
        'pressure_score': round(max(0.0, min(1.0, float(pressure_score or 0.0))), 3),
        'sampled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def memory_guardrail_decision(
    profile: HardwareProfile,
    policy: Optional[MemoryGuardrailPolicy] = None,
    phase: str = 'runtime',
    candidate_ctx: int = 0,
    safe_ctx: int = 0,
    observed_floor: int = 0,
    required_for_floor: bool = False,
    pressure_score: float = 0.0,
) -> MemoryGuardrailDecision:
    policy = policy or MemoryGuardrailPolicy.balanced()
    profile = profile or HardwareProfile()
    snapshot = _snapshot(
        profile,
        policy,
        phase,
        candidate_ctx,
        safe_ctx,
        observed_floor,
        required_for_floor,
        pressure_score,
    )
    critical_reasons: List[str] = []
    warning_reasons: List[str] = []

    ram_available = int(snapshot['ram_available'])
    ram_total = int(snapshot['ram_total'])
    if ram_available > 0:
        if ram_available < int(snapshot['ram_critical_floor']):
            critical_reasons.append(
                f'RAM available {_gib(ram_available):.1f}GiB below critical floor {_gib(int(snapshot["ram_critical_floor"])):.1f}GiB'
            )
        elif ram_available < int(snapshot['ram_warning_floor']):
            warning_reasons.append(
                f'RAM available {_gib(ram_available):.1f}GiB below warning floor {_gib(int(snapshot["ram_warning_floor"])):.1f}GiB'
            )

    vram_free = int(snapshot['gpu_memory_free'])
    vram_total = int(snapshot['gpu_memory_total'])
    if vram_total > 0:
        if vram_free < int(snapshot['vram_critical_floor']):
            critical_reasons.append(
                f'VRAM free {_gib(vram_free):.1f}GiB below critical floor {_gib(int(snapshot["vram_critical_floor"])):.1f}GiB'
            )
        elif vram_free < int(snapshot['vram_warning_floor']):
            warning_reasons.append(
                f'VRAM free {_gib(vram_free):.1f}GiB below warning floor {_gib(int(snapshot["vram_warning_floor"])):.1f}GiB'
            )

    admission = str(phase or '').strip().lower() == 'admission'
    if critical_reasons:
        action = 'skip' if admission else 'stop'
        status = 'memory_guardrail_skipped' if admission else 'memory_guardrail_stopped'
        return MemoryGuardrailDecision(
            action=action,
            status=status,
            reason='; '.join(critical_reasons),
            snapshot=snapshot,
            critical=True,
            warning=True,
        )

    candidate_ctx = int(candidate_ctx or 0)
    safe_ctx = int(safe_ctx or 0)
    pressure_score = max(0.0, min(1.0, float(pressure_score or 0.0)))
    exceeds_safe = candidate_ctx > 0 and safe_ctx > 0 and candidate_ctx > safe_ctx
    pressure_guarded = pressure_score >= policy.pressure_skip_threshold or bool(warning_reasons)
    if admission and exceeds_safe and pressure_guarded and not required_for_floor:
        reason = (
            f'candidate ctx={candidate_ctx} exceeds current safe estimate ctx={safe_ctx} '
            f'under pressure={pressure_score:.2f}'
        )
        if warning_reasons:
            reason += f'; {"; ".join(warning_reasons)}'
        return MemoryGuardrailDecision(
            action='skip',
            status='memory_guardrail_skipped',
            reason=reason,
            snapshot=snapshot,
            warning=bool(warning_reasons),
        )

    if warning_reasons or (admission and exceeds_safe):
        reason = '; '.join(warning_reasons) if warning_reasons else (
            f'candidate ctx={candidate_ctx} exceeds current safe estimate ctx={safe_ctx}; allowing under healthy pressure'
        )
        return MemoryGuardrailDecision(
            action='warn',
            status='memory_guardrail_warning',
            reason=reason,
            snapshot=snapshot,
            warning=True,
        )

    return MemoryGuardrailDecision(action='allow', snapshot=snapshot)


def memory_guardrail_record_fields(decision: MemoryGuardrailDecision) -> Dict[str, object]:
    if not decision.status:
        return {}
    return {
        'memory_guardrail_status': decision.status,
        'memory_guardrail_reason': decision.reason,
        'memory_guardrail_action': decision.action,
        'memory_guardrail_snapshot': dict(decision.snapshot),
    }


def start_memory_guardrail_watchdog(
    sample_profile: Callable[[], HardwareProfile],
    stop_candidate: Callable[[], object],
    state: MemoryGuardrailState,
    candidate_ctx: int = 0,
    safe_ctx: int = 0,
    observed_floor: int = 0,
    required_for_floor: bool = False,
    pressure_score: float = 0.0,
    phase: str = 'runtime',
) -> Tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def run() -> None:
        while not stop_event.wait(max(0.1, float(state.policy.sample_interval_seconds or 1.0))):
            try:
                profile = sample_profile()
            except Exception:
                continue
            decision = state.observe(
                profile,
                phase=phase,
                candidate_ctx=candidate_ctx,
                safe_ctx=safe_ctx,
                observed_floor=observed_floor,
                required_for_floor=required_for_floor,
                pressure_score=pressure_score,
            )
            if decision.should_stop:
                try:
                    stop_candidate()
                except Exception:
                    pass
                break

    thread = threading.Thread(target=run, name='llama-tui-memory-guardrail', daemon=True)
    thread.start()
    return stop_event, thread
