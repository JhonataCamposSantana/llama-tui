# llama-tui

A small terminal UI for managing local `llama-server` models, scanning GGUFs from the Hugging Face cache, and generating `opencode.json` from a model registry.

## Features

- Start and stop local `llama-server` models
- Live status view: `READY`, `LOADING`, `STARTING`, `STOPPED`
- Detect GGUF models from the Hugging Face cache
- Prune missing models from the registry
- Assign OpenCode roles:
  - main
  - small
  - build
  - plan
- Generate `opencode.json` and archive the old one first
- View logs and command previews from inside the TUI
- Safe context auto-tuning mode (`max_context_safe`) to maximize context while reserving system memory
- Hardware-aware best optimization using CPU cores, available RAM, and usable NVIDIA VRAM
- Zero-dependency Python: uses only the standard library

## Recommended local layout

```text
~/.local/share/llama-tui/llama_tui.py
~/.config/llama-tui/models.json
~/.local/bin/llama-tui
~/.cache/llama-tui/
```

## Install

```bash
mkdir -p ~/.local/share/llama-tui ~/.config/llama-tui ~/.local/bin

cp llama_tui.py ~/.local/share/llama-tui/llama_tui.py
cp examples/models.sample.json ~/.config/llama-tui/models.json

chmod +x ~/.local/share/llama-tui/llama_tui.py
ln -sf ~/.local/share/llama-tui/llama_tui.py ~/.local/bin/llama-tui
```

Then run:

```bash
llama-tui
```

## Controls

- `↑ / ↓` or `j / k`: move
- `Enter`: start or stop selected model
- `Enter` on a stopped model: choose launch mode (`best optimization`, `max context`, `tokens/sec`, or `keep current`)
- `a`: add model
- `e`: edit model
- `d`: delete model
- `z`: apply best optimization for this PC and sync `opencode.json`
- `B`: benchmark candidate optimizations, keep the fastest stable profile, and sync `opencode.json`
- `x`: detect GGUF models from HF cache
- `X`: prune missing models
- `m`: mark selected model as OpenCode main model
- `s`: mark selected model as OpenCode small model
- `b`: mark selected model as OpenCode build model
- `p`: mark selected model as OpenCode plan model
- `g`: generate `opencode.json`
- `o`: edit settings
- `r`: refresh
- `q`: quit

Long-running launches and benchmarks close the chooser immediately. Progress is written to the selected model log and appears in the right-side log tail while the action runs.

### Context optimization modes

Each model can choose:

- `max_context_safe` (default): tries to use the requested context window, but auto-caps it at launch time based on current available system memory and a configurable reserve percentage.
- `manual`: uses the exact configured context and parallel values.

Per-model tuning fields:

- `optimize_tier` (`safe` / `moderate` / `extreme`)
- `ctx_min`
- `ctx_max`
- `memory_reserve_percent`

Launch-time failsafe:

- When launching with an optimization mode, llama-tui can step down tiers automatically (`extreme -> moderate -> safe`) if startup/readiness fails.

Best optimization:

- Probes the current PC profile each time it runs.
- Uses CPU core count to set `threads`.
- Uses available RAM and model size to cap context safely.
- Uses NVIDIA VRAM when `nvidia-smi` reports a working GPU; otherwise it avoids GPU offload for llama.cpp.
- Chooses `safe`, `moderate`, or `extreme` automatically for the selected model.

Benchmark optimization:

- Starts candidate profiles one at a time for the selected stopped model.
- Waits for the local OpenAI-compatible endpoint to become ready.
- Sends a fixed chat-completions prompt and measures generated tokens per second.
- Persists the fastest successful profile back into the model registry.

## Notes

- The sample config is intentionally just a starter.
- The recommended flow is to detect models from your local HF cache and then assign roles in the TUI.
- Avoid committing your personal `~/.config/llama-tui/models.json` if it contains machine-specific paths.

## GitHub quick start

Inside this folder:

```bash
git init
git add .
git commit -m "Initial commit"
```

With GitHub CLI:

```bash
gh repo create llama-tui --public --source=. --remote=origin --push
```

Without GitHub CLI:

```bash
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/llama-tui.git
git push -u origin main
```
