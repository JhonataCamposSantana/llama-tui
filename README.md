# llama-tui

`llama-tui` is a zero-dependency terminal control plane for local LLM servers. It keeps a registry of local models, starts and stops `llama.cpp`, TurboQuant+, llama.cpp-tq3, or vLLM OpenAI-compatible servers, tunes launch settings for the current machine, benchmarks candidate profiles, and can export tool configs for OpenCode, Continue, and Hermes.

The project is intentionally small: it uses only the Python standard library, stores state as JSON, and runs from a terminal.

## What It Does

- Start, stop, and inspect local model servers.
- Manage GGUF models with `llama.cpp`-compatible engines and vLLM/Hugging Face model references.
- Detect `.gguf` files from Hugging Face, `llmfit`, LM Studio, and local model caches.
- Track server PID files, logs, and process groups under `~/.cache/llama-tui`.
- Clean up llama-tui-managed servers on stop, benchmark completion, and TUI exit.
- Probe CPU, RAM, NVIDIA VRAM, and current process pressure with `/proc` and `nvidia-smi`.
- Read GGUF metadata to estimate KV cache memory and safe context sizes.
- Auto-tune context size, CPU threads, GPU layer offload, KV cache type, batch size, and vLLM scheduler limits.
- Benchmark real serving launch profiles and persist measured Fast Chat, Long Context, OpenCode-ready, and Auto results.
- Run a separate Raw Speed Benchmark for deterministic engine-speed checks without changing saved serving recommendations.
- Run Deep Benchmark All across managed models and show machine-wide winners.
- Benchmark OpenCode coding workflows against disposable fixture projects.
- Try a model from inside the TUI with a temporary streaming chat console.
- Assign OpenCode roles: main, small, build, and plan.
- Assign independent Continue chat, edit/apply, and autocomplete roles.
- Verify model entries with static GGUF checks, benchmark proof, and cap diagnostics.
- Generate `opencode.json` with backups.
- Generate Continue `config.yaml` with backups while preserving user sections.
- Launch a model with OpenCode, or a full OpenCode + VS Code stack, from model details.

## Project Layout

```text
llama_tui.py              compatibility launcher
llama_tui/
  app.py                  config, model registry, server lifecycle
  benchmark.py            adaptive profile search, benchmark prompts, scoring
  chat.py                 OpenAI-compatible streaming chat helper
  constants.py            paths and defaults
  discovery.py            model detection and naming helpers
  engines.py              built-in engine registry and path/capability health helpers
  gguf.py                 GGUF metadata reader and cache-size estimates
  hardware.py             CPU/RAM/GPU probes
  launch_profiles.py      benchmark launch/request profile builder
  main.py                 entrypoint and shutdown cleanup
  models.py               dataclasses
  opencode_benchmark.py   OpenCode workflow benchmark fixtures and scoring
  optimize.py             tuning heuristics
  textutil.py             display/text helpers
  ui.py                   curses interface
examples/models.sample.json
```

## Requirements

- Python 3.10 or newer.
- A terminal with curses support.
- For GGUF models: a built `llama-server` binary.
- Optional TurboQuant+: a built `TheTom/llama-cpp-turboquant` server binary selected with `--engine turboquant`.
- Optional llama.cpp-tq3: a built `turbo-tan/llama.cpp-tq3` server binary selected with `--engine tq3`.
- Optional llama.cpp MTP: an experimental MTP branch server binary selected with `--engine llama.cpp-mtp`.
- For vLLM models: a working `vllm` command.
- Optional NVIDIA GPU: `nvidia-smi` in `PATH` lets llama-tui detect VRAM.

No Python packages are required.

## Install

Clone or copy this repository somewhere permanent, then create a command shim:

```bash
mkdir -p ~/.local/share ~/.local/bin ~/.config/llama-tui
cp -a /path/to/llama-tui-repo ~/.local/share/llama-tui
ln -sf ~/.local/share/llama-tui/llama_tui.py ~/.local/bin/llama-tui
chmod +x ~/.local/share/llama-tui/llama_tui.py
```

Make sure `~/.local/bin` is on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

The first run creates `~/.config/llama-tui/models.json` if it does not exist. You can also seed it manually:

```bash
cp ~/.local/share/llama-tui/examples/models.sample.json ~/.config/llama-tui/models.json
```

Run:

```bash
llama-tui
```

From a checkout, you can also run:

```bash
python3 llama_tui.py
python3 -m llama_tui
```

## Configuration

Default config path:

```text
~/.config/llama-tui/models.json
```

Override it for one run:

```bash
LLAMA_TUI_CONFIG=/path/to/models.json llama-tui
```

CLI help exits before curses starts:

```bash
llama-tui --help
llama-tui --engine turboquant --kv-key q8_0 --kv-value turbo4
llama-tui --engine tq3
llama-tui --engine buun --kill-existing
```

Useful top-level settings:

- `llama_server`: path or command for `llama-server`.
- `vllm_command`: command used for vLLM, default `vllm`.
- `hf_cache_root`: Hugging Face cache root.
- `llmfit_cache_root`: llmfit model cache root.
- `llm_models_cache_root`: additional local model cache root.
- `lm_studio_model_roots`: comma-separated LM Studio user model roots.
- `opencode`: export settings and role assignments.
- `continue`: Continue export settings.
- `hermes`: Hermes export and launch settings.
- `ui`: saved UI preferences, including `browser_view` (`compact` Fleet View or `advanced`) and detail density.
- `models`: registered model entries.

Useful per-model fields:

- `runtime`: model serving family, `llama.cpp` or `vllm`. The active GGUF engine can be switched at startup with `--engine llama.cpp`, `--engine buun`, `--engine turboquant`, or `--engine tq3`.
- `path`: GGUF path for `llama.cpp`, local path or repo id for vLLM.
- `alias`: served model name used by OpenAI-compatible requests.
- `host` and `port`: bind address. New local models default to `127.0.0.1:18080` so generated tool configs stay on one stable local endpoint.
- `ctx`: requested context size.
- `threads`: CPU generation threads.
- `ngl`: `llama.cpp` GPU layer offload count.
- `parallel`: llama.cpp parallel slots.
- `cache_ram`: llama.cpp prompt cache RAM value.
- `flash_attn`, `jinja`, `extra_args`: runtime flags. Continue tool-capable llama.cpp, buun, TurboQuant+, and llama.cpp-tq3 exports force `--jinja` at launch; add `--chat-template-file ...` in `extra_args` when a GGUF needs a tool-use template override.
- `top_p`, `top_k`, `repeat_penalty`, `presence_penalty`, `no_context_shift`, `preserve_thinking`: optional generation/runtime defaults used by Try-It-Out, normal serving when the engine supports server-side flags, and benchmark request payloads.
- `launch_overrides`: advanced config-only launch/benchmark overrides. Supported keys include `top_p`, `top_k`, `min_p`, `seed`, `samplers`, `repeat_penalty`, `presence_penalty`, `no_context_shift`, `preserve_thinking`, `reasoning`, `reasoning_budget`, `cache_prompt`, `cache_reuse`, `fit_target`, `measurement_output`, and `extra_args`.
- `optimize_mode`: `max_context_safe` or `manual`.
- `optimize_tier`: `safe`, `moderate`, or `extreme`.
- `ctx_min`, `ctx_max`, `memory_reserve_percent`: guardrails for auto tuning.
- `tags`: browser labels such as `coding`, `autocomplete`, `long-context`, `fast-chat`, or your own tags.
- `measured_profiles`: adaptive benchmark winners used by Auto, Fast Chat, Long Context, and OpenCode-ready launches.
- `verification_status`, `verification_summary`, `verification_results`: model proof and cap diagnosis state.

## Controls

- `Up / Down` or `j / k`: move in the model list.
- `Enter` on the list: open model details.
- `Enter` or `l` on details: open model actions.
- `Esc` on details: return to the model list.
- `Tab` or `]`: switch the right pane to the next tab.
- `Shift+Tab` or `[`: switch the right pane to the previous tab.
- `T`: open Try it out from model details.
- `B`: run the quality-first smart benchmark.
- `F`: run the faster benchmark.
- `D`: run the safer adaptive Deep Benchmark All for missing, stale, failed, or aborted managed models; the modal also offers a force refresh.
- `R` on the model list or `M` from the main views: open Machine Rankings.
- `O`: benchmark the selected model with an OpenCode workflow from details.
- `Y`: verify the selected model entry.
- `y`: queue benchmark-proof verification for enabled models with missing or stale proof.
- `A`: abort the active launch or benchmark action and clean up managed processes.
- `z`: apply the Auto profile without benchmarking.
- `S`: stop all known models.
- `x`: detect new GGUF models from configured cache roots.
- `X`: prune missing models.
- `a`: add a model.
- `e`: edit a model.
- `d`: delete a model from the registry and refresh generated tool configs.
- `m`, `s`, `b`, `p`: assign OpenCode main, small, build, or plan role.
- `g`: generate `opencode.json`.
- `c`: generate Continue `config.yaml`.
- `G`: generate Hermes `config.yaml` for the selected model.
- `o`: edit settings.
- `:`: open Actions, including Raw Speed Benchmark, stateful browser/detail toggles, export actions, and Config Doctor.
- `r`: sync inventory.
- `q`: quit.

The model browser defaults to compact Fleet View: model name, server state, recommended pick, effective context, measured tok/s, active engine, and health. Actions can toggle Advanced View when you need the denser legacy columns; toggling from another screen returns to Models so the change is visible immediately.

The top dashboard and selected-model Overview show a prominent active-engine badge such as `ENGINE: TurboQuant+ | KV key=q8_0 value=turbo4 | binary ok` or `ENGINE: TQ3 | KV key=q8_0 value=q8_0 | binary ok`. Missing binaries or engine warnings use the existing warning/error styling.

The details screen is split into Overview, Launch, Tuning, Benchmarks, Logs, Command, and Exports tabs so launch decisions stay separate from lower-level settings and generated-config status. Benchmark tables live in the Benchmarks tab to keep the main detail pane readable.

Settings are split into Runtime, Model Roots, OpenCode, Continue, Hermes, and UI sections. Config Doctor is available from Actions and summarizes runtime commands, active code path, active engine resolution, terminal launcher detection, VS Code, export paths, generated-config state, and model proof status.

## Running Servers

To start a server, select a model, open details with `Enter`, then press `Enter` or `l`.

If the model is stopped, llama-tui asks how to launch it:

- Start server now.
- Auto profile.
- Balanced chat.
- Fast chat.
- Long context.
- Advanced profiles.
- Try it out.
- Launch model + OpenCode.
- Launch full-stack: OpenCode + VS Code.

If a model is already running, the details action menu offers stop, Try it out, OpenCode launch, full-stack launch, or cancel.

`Start server now` uses the saved model settings guarded by safe launch checks. It does not require benchmark data. `Try it out` is available from the action menu and as `T` on model details. It opens an integrated chat console inside llama-tui. The left pane is a temporary transcript plus a five-row wrapped prompt editor; the right pane keeps the active launch profile, server logs, and live stats visible. Press `Enter` to send, `Up / Down` to scroll long prompt text, `Ctrl+U` to clear the input, `Ctrl+P/N/B/F/A/E` to scroll the conversation transcript, and `Esc` to leave. Streamed reasoning is shown inline above the final answer when the model provides it. Leaving Try-It-Out stops only a server launched by that Try-It-Out session; a server that was already running before entry is left running.

For `llama.cpp`, llama-tui builds a command like:

```bash
llama-server \
  -m /path/to/model.gguf \
  --alias my-model \
  --host 127.0.0.1 \
  --port 18080 \
  --ctx-size 8192 \
  --threads 6 \
  --n-gpu-layers 12 \
  --parallel 1 \
  --cache-ram 0 \
  --temp 0.65 \
  --flash-attn on \
  --jinja
```

Experimental GGUF engines are selected for the whole TUI session:

```bash
python -m llama_tui.main --engine turboquant
python -m llama_tui.main --engine turboquant --kv-key q8_0 --kv-value turbo4
python -m llama_tui.main --engine tq3
python -m llama_tui.main --engine llama.cpp-mtp
python -m llama_tui.main --engine buun
```

Engine path resolution is centralized but remains backward compatible: llama.cpp uses `LLAMA_SERVER` / `llama_server`, TurboQuant+ uses `TURBOQUANT_LLAMA_SERVER_BIN` or its existing defaults, llama.cpp-tq3 uses `TQ3_LLAMA_SERVER_BIN` or its existing defaults, llama.cpp MTP uses `LLAMA_CPP_MTP_PATH` or `~/src/llama.cpp-mtp/build-mtp/bin/llama-server`, Buun uses `BUUN_LLAMA_SERVER_BIN` or `buun-llama-server`, and vLLM uses `VLLM_COMMAND` / `vllm_command`.

TurboQuant+ uses `TURBOQUANT_LLAMA_SERVER_BIN` when set, otherwise it looks for `~/llama-cpp-turboquant/build/bin/llama-server` and then `turboquant-llama-server` in `PATH`. v1 does not download or install binaries. The `tqp-v0.1.1` release documents prebuilts for macOS arm64 Metal and Windows x64 CUDA 12.4; Linux CUDA users should build `TheTom/llama-cpp-turboquant` from source first.

llama.cpp-tq3 uses `TQ3_LLAMA_SERVER_BIN` when set, otherwise it looks for `~/llama.cpp-tq3/build/bin/llama-server` and then `tq3-llama-server` in `PATH`. It is treated as a TQ3-native engine: the default browser compatibility filter shows only GGUFs detected as `TQ3_1S` or `TQ3_4S`, and launch/benchmark preflight blocks regular GGUFs with a clear advisory. `q8_0/q8_0` is the default KV cache choice; `tq3_0/tq3_0` is available only as a manual experimental KV mode until local benchmarks prove it is better on this machine.

llama.cpp MTP is separate from stable upstream llama.cpp. It is marked Experimental, uses `LLAMA_CPP_MTP_PATH` when set, and verifies that the selected binary advertises `--spec-type mtp` and `--spec-draft-n-max` before MTP launches are allowed. Model configs can keep `supports_mtp` as `auto` or set it to `yes`/`no`; `mtp_enabled` defaults off and `mtp_draft_n_max` is clamped to 1, 2, or 3. When MTP is enabled, llama-tui emits `--spec-type mtp --spec-draft-n-max N`, forces a single parallel slot, and blocks mmproj/vision launches with a clear error. The helper `./llama-update-engines` can build the experimental branch into `~/src/llama.cpp-mtp/build-mtp`.

The default TurboQuant+ profile is `q8_0/q8_0`. If GGUF metadata reports `head_dim=64`, llama-tui keeps `q8_0/q8_0`, disables automatic turbo V compression, and shows a high-severity advisory. If head dim is unknown, `q8_0/turbo4` is still available manually but automatic benchmark planning stays conservative. For `head_dim >= 128`, benchmarks can try baseline, safe V-only, balanced V-only, extreme V-only, and symmetric turbo profiles. Symmetric turbo is auto-planned only for Q8_0/F16/FP-style weights or explicitly validated family and quantization pairs; low-bit and unknown quantizations keep symmetric profiles manual-only.

If the resolved TurboQuant+ command looks like vanilla llama.cpp or its `--help` output does not advertise `turbo2`, `turbo3`, or `turbo4` cache types, the command preview and launch log show a binary warning.

For Continue Agent Mode tool use, llama-tui forces `--jinja` for enabled Continue-exported llama.cpp, buun, TurboQuant+, llama.cpp-tq3, and llama.cpp MTP models even when the saved model setting has `jinja` off. It does not add a fallback chat template; use model `extra_args` such as `--chat-template-file /path/to/tool-template.jinja` for GGUFs whose embedded template is not tool-use compatible.

For vLLM, llama-tui builds:

```bash
vllm serve MODEL_REF \
  --host 127.0.0.1 \
  --port 18080 \
  --served-model-name my-model \
  --max-model-len 8192
```

Each managed server writes:

```text
~/.cache/llama-tui/<model-id>.log
~/.cache/llama-tui/<model-id>.pid
~/.cache/llama-tui/<model-id>.pid.json
```

The PID metadata records the process group. Stop commands and shutdown cleanup terminate the managed process group, which prevents child server processes from keeping VRAM allocated after a benchmark or TUI exit.

llama-tui still detects matching external processes for display and manual stop attempts, but benchmark and shutdown cleanup use managed-only process tracking so a separate manual `llmfit run` is not accidentally swept up.

## Optimization Logic

The default mode is `max_context_safe`. In that mode, the configured `ctx` is treated as a request, not a promise. At launch time llama-tui probes current hardware and may cap:

- context size,
- `parallel`,
- `ngl`,
- CPU thread count,
- batch and micro-batch sizes,
- KV cache type,
- vLLM GPU utilization and scheduler limits.

The goal is to start reliably first, then pick the fastest stable profile.

### Hardware Probes

llama-tui reads:

- CPU logical and physical-ish core counts from `/proc/cpuinfo`.
- RAM availability from `/proc/meminfo`.
- NVIDIA GPU name, total VRAM, and free VRAM from `nvidia-smi`.
- GGUF architecture metadata directly from the model file, including Dense vs MoE and expert metadata when available.

GGUF metadata is used to estimate KV cache bytes per token from layer count, KV heads, key/value dimensions, and cache type. This is much better than estimating from file size alone.

### Launch Profiles

The normal model action menu uses intent labels:

- `Start server now`: start from saved settings with safe launch checks; no benchmark required.
- `Auto profile`: use the best saved/measured behavior for this PC, with failsafe fallback.
- `Balanced chat`: tune for responsive chat without being too aggressive.
- `Fast chat`: push throughput harder.
- `Long context`: favor the largest stable context window.
- `Advanced profiles`: expose the underlying intent and aggression controls.

Internally those choices still map to the persisted `optimize_mode` and `optimize_tier` values so existing configs remain compatible.

### Aggression

`safe`, `moderate`, and `extreme` are memory headroom policies. The UI labels them as Safe, Balanced, and Aggressive:

- `safe`: higher reserve, smaller batches, conservative GPU usage.
- `moderate`: balanced defaults.
- `extreme`: lower reserve, larger batches, more aggressive GPU usage.

If a launch optimization fails, the failsafe path walks downward:

```text
extreme -> moderate -> safe
```

### Presets

`max_context` is shown as Long Context. Before a benchmark exists it uses conservative estimates; after pressing `B`, llama-tui prefers the measured Long Context profile:

- `parallel = 1`
- conservative batch sizes
- q8 KV cache for llama.cpp
- lower vLLM sequence concurrency

`tokens_per_sec` is shown as Fast Chat. Before a benchmark exists it uses estimated throughput settings; after pressing `B`, llama-tui prefers the measured Fast Chat profile:

- smaller target context
- higher `parallel`
- larger batch and micro-batch sizes
- regular f16 KV cache by default
- an extra q8 KV candidate when benchmarking on NVIDIA GPUs

### GPU Offload

Older behavior tried `ngl=999` whenever a GPU looked usable. That can fail on laptop GPUs when the full model does not fit in VRAM.

Current behavior estimates how many layers can fit after reserving memory for:

- model weights,
- KV cache floor,
- runtime workspace,
- configured headroom.

If the full model fits, `ngl=999` is still used. If not, llama-tui chooses a partial layer count or CPU-only launch.

## Benchmark Logic

Press `B` on a stopped model to run the adaptive benchmark. It can take a while by design; the target budget is about 20 minutes per model, and `A` can abort it. Press `F` for the faster bounded benchmark.

`B` and `F` now run explicit `serve_default` profile benchmarks. That means benchmark server launches go through the same canonical command builder as normal serving, including the selected engine, context, KV cache profile, flash attention, fit behavior, context-shift setting, template kwargs, and supported sampling flags. The HTTP chat-completion requests also use the profile sampling values.

Actions includes `Raw Speed Benchmark...`. Raw speed uses a separate `raw_speed` profile with deterministic/near-deterministic sampling, a fixed 512-token measurement, and no quality scoring. It writes a benchmark-history run and record rows, but it does not update `measured_profiles`, `last_benchmark_tokens_per_sec`, machine rankings, or benchmark-proof freshness.

Press `D` to run Deep Benchmark All. The default batch walks the registered models, skips fresh benchmark results, skips disabled entries, skips unmanaged external servers, and benchmarks pending, stale, failed, aborted, or missing measured profiles. If a model is already running under llama-tui management, the batch stops it for the benchmark and restores it on normal completion. The force-refresh option reruns every enabled managed model. This uses a safer adaptive batch benchmark instead of the full single-model quality-first run behind `B`, so it is better suited to large model libraries and lower-memory machines.

The benchmark runner:

1. Probes current hardware.
2. Estimates a model-specific context ceiling from RAM, VRAM, current process pressure, GGUF metadata, runtime settings, and KV-cache size.
3. Probes context with exponential growth and binary refinement.
4. Tests dynamic context, parallel, batch, and KV-cache variants one server at a time.
5. Waits for `/v1/models` to become ready.
6. Warms the model with a short completion.
7. Runs a two-prompt chat-completions suite.
8. Scores measured candidates by generated tokens per second, context per slot, process-aware headroom, and stability.
9. Stops the managed server process group after every candidate.
10. Saves measured `Fast Chat`, `Long Context`, `OpenCode-ready`, and `Auto` profiles back to `models.json`.

The prompt suite is intentionally short and stable. It is not a model quality benchmark. It measures local serving throughput for the selected runtime and hardware.

Output policy is intentionally bounded for this first profile milestone:

- `raw_speed`: measures 512 tokens.
- `serve_default` fast runs: store the intended model output, but measure up to 512 tokens.
- `serve_default` full runs: store the intended model output, but measure up to 1024 tokens.

Benchmark details show the compact launch profile for each row: benchmark profile, engine, context, intended output, measurement cap, KV key/value, flash attention, fit state, no-context-shift state, sampling, template kwargs, and unsupported optional launch flags. The main Fleet View stays uncluttered.

`preserve_thinking` defaults to `auto`. Explicit `on` or `off` wins. In `auto`, llama-tui looks for GGUF template/thinking/reasoning metadata and broad reasoning-family signals such as QwQ, DeepSeek-R1, GPT-OSS, or Qwen3 plus thinking/reasoning markers. If uncertain, it resolves off. When resolved on and the active server supports it, llama-tui emits `--chat-template-kwargs '{"preserve_thinking": true}'`.

`launch_overrides` is intentionally config-only for now. It lets advanced users tune request/server profile details without adding more fields to the model edit form:

```json
"launch_overrides": {
  "top_p": 0.95,
  "top_k": 20,
  "repeat_penalty": 1.05,
  "presence_penalty": 0.0,
  "no_context_shift": true,
  "preserve_thinking": "on",
  "min_p": 0.05,
  "seed": 123,
  "samplers": "top_k;top_p;min_p;temperature",
  "reasoning": "auto",
  "reasoning_budget": 8192,
  "cache_prompt": true,
  "cache_reuse": 256,
  "fit_target": "0.85",
  "extra_args": []
}
```

Pressing `A` while a launch or benchmark is running requests cancellation. The active candidate server is stopped after the current blocking operation unwinds, and benchmark-launched OpenCode subprocess groups are terminated.

Saved benchmark rows include:

- measured objective,
- architecture type and detection source,
- tokens/sec,
- elapsed seconds,
- context,
- context per slot,
- parallel,
- GPU layers,
- engine, cache key/value modes, detected head dim, model quant/family, binary path, and help-supported cache types,
- benchmark profile, intended output, measurement cap, sampling values, fit/context-shift/template kwargs, command preview, and unsupported optional launch flags,
- RAM/VRAM/process-pressure headroom,
- status and detail.

Benchmarking is optional for normal server launches. `Auto profile`, `Fast Chat`, `Long Context`, `Try it out`, and OpenCode stack launches use measured profiles when they exist. If no measured profile exists yet, llama-tui falls back to the estimated safe launch path and says so in the action log.

Benchmarks intentionally observe the current workload. If VS Code, browsers, Docker, OpenCode, Hermes, or other heavy processes are running, llama-tui treats that as part of the target environment and favors profiles with enough RAM/VRAM/CPU headroom for that real machine state. Closing or opening major apps can intentionally change the recommended profile.

Machine Rankings are computed from fresh measured profiles. The overview shows `Fastest Chat`, `Longest Context`, `OpenCode-ready`, and one `Machine Pick`. The Machine Pick uses a weighted score for speed, context per slot, RAM/VRAM/process headroom, and stability.

If all candidates fail, the failure details are saved and shown in the model details screen.

## Model Verification

Press `Y` to verify the selected model entry. Verification is offline-first:

- `llama.cpp` entries check the file path, `.gguf` suffix, GGUF magic header, metadata parse health, native context, KV-cache estimate inputs, file size, and projection-file mistakes such as `mmproj`.
- vLLM entries check that the target is a local path or repo-shaped model reference. llama-tui does not contact Hugging Face or any network service.
- Fresh measured benchmark proof marks a model `passed`; stale or missing proof is `needs_benchmark`, not failed.
- Static problems such as missing files, bad GGUF magic, or unsupported targets mark a model `failed`.

Press `y` to queue benchmark-proof verification for enabled models with missing or stale proof. This reuses the existing Deep Benchmark All path and never runs automatically on startup.

Cap diagnostics answer why the requested context may be reduced. The details view and Config Doctor show configured context, `ctx / parallel` per-slot context, native GGUF context, estimated safe hardware context, measured max context, and the active limiting factor: user `ctx_max`, model-native context, safe hardware estimate, parallel split, benchmark proof, or the configured request itself. So when context is lower than expected, it is not treated as a mysterious fixed cap; the UI names the cap that actually applied.

## Model Detection

Press `x` to scan configured roots for `.gguf` files:

```text
hf_cache_root
llmfit_cache_root
llm_models_cache_root
lm_studio_model_roots
```

LM Studio defaults are read from `LM_STUDIO_HOME`, then `~/.lmstudio-home-pointer`, then `~/.lmstudio`. llama-tui scans only the user model folders, `models` and `hub/models`, by default; internal bundled models are skipped unless you add that path manually.

Files containing `mmproj` are ignored. New models get generated ids, aliases, the stable local endpoint `127.0.0.1:18080`, architecture labels, and a generic safe profile: small context, CPU-first launch, safe memory reserve, and `default_benchmark_status=pending`. That pending marker means “unbenchmarked,” not blocked: you can start the server immediately, and nothing benchmarks automatically in the background. Open the model details and press `B` when you want measured settings.

Dense vs MoE detection is metadata-first. llama-tui reads GGUF `general.architecture` and expert metadata such as `{arch}.expert_count` and `{arch}.expert_used_count`; if metadata is incomplete, it can inspect tensor descriptors by name without reading tensor data; filename patterns such as `30B-A3B` are only a weak fallback. MoE benchmark mode keeps memory estimates based on the full loaded GGUF, keeps KV-cache estimates attention/layer-driven, and scores OpenCode-style profiles by stable context before raw tokens/sec.

Press `X` to prune registry entries whose model files disappeared. Model add, edit, delete, detect, prune, role, settings, and benchmark tuning actions refresh generated tool configs so removed models do not linger in OpenCode, Continue, or llama-tui-generated Hermes homes.

## OpenCode Export

Set `opencode.path` in settings, then press `g` to generate the config.

llama-tui writes OpenAI-compatible providers for enabled local models and maps:

- main model,
- small model,
- build model,
- plan model.

Existing config files are backed up under `opencode.backup_dir` before writing.

The OpenCode launch actions use `opencode.terminal_command` if set. The command is a template with `{title}`, `{cwd}`, and `{cmd}` placeholders. If unset, llama-tui tries common terminal launchers such as `konsole`, `gnome-terminal`, `kgx`, `kitty`, `alacritty`, `wezterm`, `foot`, and `xterm`.

The `O` benchmark is different from those launch actions: it runs `opencode run --agent build --model ... --dir ...` headlessly inside disposable fixture repos so timing, test success, and cleanup can be measured. Benchmark-launched OpenCode process groups are terminated at the end of each task, on timeout, or when aborted.

For full-stack launches, llama-tui opens VS Code with:

```bash
code --new-window WORKSPACE
```

If VS Code is unavailable, it still launches the model + OpenCode path and reports a warning.

## Continue Export

Press `c` to generate the config. By default `continue.path` is `~/.continue/config.yaml`; you can override it in settings if your Continue install uses a different YAML path.

llama-tui writes a local Continue `config.yaml` using the OpenAI-compatible provider format. Every enabled llama-tui-managed model is exported with `chat`, `edit`, `apply`, and `autocomplete` roles, autocomplete options, and the `tool_use` capability so Continue can use any exported model for interactive coding roles and Agent Mode tool calls.

Continue MCP tools only execute in Agent Mode. In plain Chat mode, Continue may still read rules and context, but it will not run MCP tools.

`continue.default_model_id`, `continue.edit_model_id`, and `continue.autocomplete_model_id` still influence the order of the generated model list. If a Continue role is blank, llama-tui falls back to the matching OpenCode role; if those are also blank, it uses the first enabled model for chat/edit and the second enabled model for autocomplete when available.

Existing Continue config files are backed up under `continue.backup_dir` before writing.

The default `continue.merge_mode` is `preserve_sections`. In that mode llama-tui keeps top-level user sections such as `rules`, `context`, `prompts`, `mcpServers`, `docs`, and `data`, preserves unmarked user models, sanitizes older duplicate or unterminated managed blocks, and replaces the managed block between:

```yaml
  # BEGIN llama-tui managed models
  # END llama-tui managed models
```

Set `continue.merge_mode` to `managed_file` if you want llama-tui to rewrite the whole file from its generated template.

Continue `contextLength` is exported as per-slot context, `ctx // parallel`, so autocomplete and chat tools see the same effective window the server exposes for each simultaneous request.

llama-tui also refreshes the Continue export after registry changes, settings/role changes, benchmark-driven tuning, and Auto-profile updates so the exported model list, context, and token limits stay aligned with the latest saved model settings. If no enabled models remain, the managed block is written empty instead of leaving stale entries behind.

## Safety Notes

- `manual` optimization mode uses the configured values exactly.
- `max_context_safe` may lower context or GPU layer count at launch time.
- Benchmarking starts and stops real local servers and can consume CPU, RAM, and VRAM.
- Stop-all (`S`) targets known model entries; shutdown cleanup targets llama-tui-managed PID metadata.
- If you start servers outside llama-tui, check `nvidia-smi` before benchmarking large models.
- Personal config files usually contain machine-specific paths. Avoid committing your real `~/.config/llama-tui/models.json`.

## Development

Run syntax checks:

```bash
python3 -m py_compile llama_tui.py llama_tui/*.py
```

Run tests:

```bash
python3 -m unittest discover -s tests
# or, from the repository root:
python3 -m unittest discover
```

Run import smoke checks:

```bash
python3 - <<'PY'
import importlib

for name in [
    'constants', 'models', 'hardware', 'gguf', 'discovery',
    'optimize', 'app', 'benchmark', 'chat', 'textutil', 'ui', 'main',
]:
    importlib.import_module(f'llama_tui.{name}')

print('ok')
PY
```

Run from the repository:

```bash
python3 llama_tui.py
```

or:

```bash
python3 -m llama_tui
```
