# llama-tui

`llama-tui` is a zero-dependency terminal control plane for local LLM servers. It keeps a registry of local models, starts and stops `llama.cpp` or vLLM OpenAI-compatible servers, tunes launch settings for the current machine, benchmarks candidate profiles, and can export an `opencode.json` provider config.

The project is intentionally small: it uses only the Python standard library, stores state as JSON, and runs from a terminal.

## What It Does

- Start, stop, and inspect local model servers.
- Manage `llama.cpp` GGUF models and vLLM/Hugging Face model references.
- Detect `.gguf` files from Hugging Face, `llmfit`, LM Studio, and local model caches.
- Track server PID files, logs, and process groups under `~/.cache/llama-tui`.
- Clean up llama-tui-managed servers on stop, benchmark completion, and TUI exit.
- Probe CPU, RAM, and NVIDIA VRAM with `/proc` and `nvidia-smi`.
- Read GGUF metadata to estimate KV cache memory and safe context sizes.
- Auto-tune context size, CPU threads, GPU layer offload, KV cache type, batch size, and vLLM scheduler limits.
- Benchmark adaptive profiles and persist measured Fast Chat, Long Context, OpenCode-ready, and Auto results.
- Run Deep Benchmark All across managed models and show machine-wide winners.
- Benchmark OpenCode coding workflows against disposable fixture projects.
- Try a model from inside the TUI with a temporary streaming chat console.
- Assign OpenCode roles: main, small, build, and plan.
- Generate `opencode.json` with backups.
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
  gguf.py                 GGUF metadata reader and cache-size estimates
  hardware.py             CPU/RAM/GPU probes
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
- For `llama.cpp` models: a built `llama-server` binary.
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

Useful top-level settings:

- `llama_server`: path or command for `llama-server`.
- `vllm_command`: command used for vLLM, default `vllm`.
- `hf_cache_root`: Hugging Face cache root.
- `llmfit_cache_root`: llmfit model cache root.
- `llm_models_cache_root`: additional local model cache root.
- `lm_studio_model_roots`: comma-separated LM Studio user model roots.
- `opencode`: export settings and role assignments.
- `models`: registered model entries.

Useful per-model fields:

- `runtime`: `llama.cpp` or `vllm`.
- `path`: GGUF path for `llama.cpp`, local path or repo id for vLLM.
- `alias`: served model name used by OpenAI-compatible requests.
- `host` and `port`: bind address.
- `ctx`: requested context size.
- `threads`: CPU generation threads.
- `ngl`: `llama.cpp` GPU layer offload count.
- `parallel`: llama.cpp parallel slots.
- `cache_ram`: llama.cpp prompt cache RAM value.
- `flash_attn`, `jinja`, `extra_args`: runtime flags.
- `optimize_mode`: `max_context_safe` or `manual`.
- `optimize_tier`: `safe`, `moderate`, or `extreme`.
- `ctx_min`, `ctx_max`, `memory_reserve_percent`: guardrails for auto tuning.
- `measured_profiles`: adaptive benchmark winners used by Auto, Fast Chat, Long Context, and OpenCode-ready launches.

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
- `A`: abort the active launch or benchmark action and clean up managed processes.
- `z`: apply the Auto profile without benchmarking.
- `S`: stop all known models.
- `x`: detect new GGUF models from configured cache roots.
- `X`: prune missing models.
- `a`: add a model.
- `e`: edit a model.
- `d`: delete a model from the registry.
- `m`, `s`, `b`, `p`: assign OpenCode main, small, build, or plan role.
- `g`: generate `opencode.json`.
- `o`: edit settings.
- `r`: sync inventory.
- `q`: quit.

The details screen shows the command preview, hardware profile, server status, PID, roles, recent logs, and the saved benchmark table.

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

`Start server now` uses the saved model settings guarded by safe launch checks. It does not require benchmark data. `Try it out` is available from the action menu and as `T` on model details. It opens an integrated chat console inside llama-tui. The left pane is a temporary transcript plus a five-row wrapped prompt editor; the right pane keeps the active launch profile, server logs, and live stats visible. Press `Enter` to send, `Up / Down` to scroll long prompt text, `Ctrl+U` to clear the input, `Ctrl+P/N/B/F/A/E` to scroll the conversation transcript, and `Esc` to leave. Streamed reasoning is shown inline above the final answer when the model provides it. Leaving Try-It-Out always stops the selected model, even if it was already running before you entered the console.

For `llama.cpp`, llama-tui builds a command like:

```bash
llama-server \
  -m /path/to/model.gguf \
  --alias my-model \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 8192 \
  --threads 6 \
  --n-gpu-layers 12 \
  --parallel 1 \
  --cache-ram 0 \
  --temp 0.65 \
  --flash-attn on \
  --jinja
```

For vLLM, llama-tui builds:

```bash
vllm serve MODEL_REF \
  --host 127.0.0.1 \
  --port 8080 \
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
- GGUF architecture metadata directly from the model file.

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

Press `B` on a stopped model to run the adaptive benchmark. It can take a while by design; the target budget is about 20 minutes per model, and `A` can abort it.

Press `D` to run Deep Benchmark All. The default batch walks the registered models, skips fresh benchmark results, skips disabled entries, skips unmanaged external servers, and benchmarks pending, stale, failed, aborted, or missing measured profiles. If a model is already running under llama-tui management, the batch stops it for the benchmark and restores it on normal completion. The force-refresh option reruns every enabled managed model. This uses a safer adaptive batch benchmark instead of the full single-model quality-first run behind `B`, so it is better suited to large model libraries and lower-memory machines.

The benchmark runner:

1. Probes current hardware.
2. Estimates a model-specific context ceiling from RAM, VRAM, GGUF metadata, runtime settings, and KV-cache size.
3. Probes context with exponential growth and binary refinement.
4. Tests dynamic context, parallel, batch, and KV-cache variants one server at a time.
5. Waits for `/v1/models` to become ready.
6. Warms the model with a short completion.
7. Runs a two-prompt chat-completions suite.
8. Scores measured candidates by generated tokens per second, context per slot, and headroom.
9. Stops the managed server process group after every candidate.
10. Saves measured `Fast Chat`, `Long Context`, `OpenCode-ready`, and `Auto` profiles back to `models.json`.

The prompt suite is intentionally short and stable. It is not a model quality benchmark. It measures local serving throughput for the selected runtime and hardware.

Pressing `A` while a launch or benchmark is running requests cancellation. The active candidate server is stopped after the current blocking operation unwinds, and benchmark-launched OpenCode subprocess groups are terminated.

Saved benchmark rows include:

- measured objective,
- tokens/sec,
- elapsed seconds,
- context,
- context per slot,
- parallel,
- GPU layers,
- RAM/VRAM headroom,
- status and detail.

Benchmarking is optional for normal server launches. `Auto profile`, `Fast Chat`, `Long Context`, `Try it out`, and OpenCode stack launches use measured profiles when they exist. If no measured profile exists yet, llama-tui falls back to the estimated safe launch path and says so in the action log.

Machine Rankings are computed from fresh measured profiles without changing the config schema. The overview shows `Fastest Chat`, `Longest Context`, `OpenCode-ready`, and one `Machine Pick`. The Machine Pick uses a weighted score for speed, context per slot, RAM/VRAM headroom, and stability.

If all candidates fail, the failure details are saved and shown in the model details screen.

## Model Detection

Press `x` to scan configured roots for `.gguf` files:

```text
hf_cache_root
llmfit_cache_root
llm_models_cache_root
lm_studio_model_roots
```

LM Studio defaults are read from `LM_STUDIO_HOME`, then `~/.lmstudio-home-pointer`, then `~/.lmstudio`. llama-tui scans only the user model folders, `models` and `hub/models`, by default; internal bundled models are skipped unless you add that path manually.

Files containing `mmproj` are ignored. New models get generated ids, aliases, ports, and a generic safe profile: small context, CPU-first launch, safe memory reserve, and `default_benchmark_status=pending`. That pending marker means “unbenchmarked,” not blocked: you can start the server immediately, and nothing benchmarks automatically in the background. Open the model details and press `B` when you want measured settings. Discovery does not special-case model families or filenames.

Press `X` to prune registry entries whose model files disappeared.

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
