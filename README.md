# autoresearch — WSL2 / RTX Edition

> Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for **NVIDIA RTX GPUs** (Ampere, Ada, Blackwell) running on **WSL2**.

The upstream repo targets H100 GPUs with Flash Attention 3 and torch.compile. This fork makes three targeted changes to run on consumer RTX hardware under Windows Subsystem for Linux:

1. **Flash Attention 3 → PyTorch SDPA** — FA3 requires SM 9.0 (Hopper). We use `F.scaled_dot_product_attention` which works on all CUDA GPUs.
2. **torch.compile disabled** — The Inductor/Triton backend currently fails on Ampere with this model's architecture. Eager mode works reliably.
3. **DEVICE_BATCH_SIZE scaled for 24GB** — Default tuned for RTX 3090 (24GB VRAM).

Everything else is identical to upstream: same model architecture, same optimizer (Muon + AdamW), same 5-minute time budget, same val_bpb metric.

## Baseline Results

| Metric | Value |
|--------|-------|
| val_bpb | **1.413** |
| MFU | 3.2% |
| Peak VRAM | 11.8 GB / 24 GB |
| Training time | 302s |
| Steps | 85 |
| Tokens processed | ~44M |

Tested on RTX 3090 (24GB), Ubuntu 24.04 on WSL2, PyTorch 2.9.1+cu128, Driver 581.29.

## Quick Start

**Requirements:** NVIDIA RTX GPU (≥10GB VRAM recommended), WSL2, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (~2 min)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py
```

## WSL2 Setup Notes

If `nvidia-smi` isn't found in your WSL2 instance, add the driver path:

```bash
export PATH=/usr/lib/wsl/lib:$PATH
```

Make sure your distro is running as WSL2 (not WSL1):

```bash
# Check — should show "microsoft-standard-WSL2" in the version string
cat /proc/version

# If it shows WSL1 (4.4.0-19041-Microsoft), convert from PowerShell:
# wsl --set-version <distro-name> 2
```

## Tuning for Your GPU

If you have less than 24GB VRAM, reduce `DEVICE_BATCH_SIZE` in `train.py`. If you OOM, halve it until it fits:

| VRAM | Suggested DEVICE_BATCH_SIZE |
|------|-----------------------------|
| 24GB | 16 |
| 16GB | 8 |
| 10-12GB | 4 |

For significantly smaller GPUs, also follow Karpathy's upstream recommendations: lower `DEPTH`, switch to [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean), reduce `MAX_SEQ_LEN` in `prepare.py`.

## What's Different from Upstream

| Feature | Upstream (H100) | This Fork (RTX/WSL2) |
|---------|-----------------|----------------------|
| Attention | Flash Attention 3 | PyTorch SDPA |
| Compilation | torch.compile | Eager mode |
| Window attention | Native FA3 windowed (SSSL) | Full causal (all layers) |
| Default batch size | 128 | 16 |
| MFU | ~40% | ~3.2% |

## Known Limitations & Findings

- **No window attention** — Full causal attention on all layers. We tested explicit SDPA masks for windowed attention (`SSSL` pattern) and found it **hurt** both performance and quality at this scale — val_bpb went from 1.413 to 1.665 and MFU dropped from 3.2% to 1.7%. The mask construction overhead on SDPA outweighs any benefit from the attention pattern at this model size.

- **Lower MFU** — Without torch.compile, GPU utilization is ~3% vs ~40% on H100 with compiled mode. The model still trains correctly, just slower per FLOP. Your 3090 is fast enough that the 5-minute budget produces good results regardless.

- **torch.compile fails on Ampere** — The Inductor backend crashes during Triton codegen with this specific model architecture on SM 8.6 (RTX 3090). Both full model compilation and optimizer kernel compilation fail. We tested: model-only compile (fails), optimizer-only compile (fails), all compile (fails). This appears to be an Inductor/SDPA interaction issue, not a fundamental Triton-on-WSL2 problem. PRs welcome — `TORCHDYNAMO_VERBOSE=1` will show the exact failure point.

- **Batch size 32 is worse than 16** — Counterintuitively, doubling batch size (32) reduced MFU from 3.2% to 1.85% in eager mode. The larger batch increases per-step time without proportional throughput gains. Stick with 16 for best results on 24GB.

## Running the Agent

Same as upstream — point Claude/Codex at `program.md` and go:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Project Structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
README.md       — you are here
```

## Upstream

All credit to [@karpathy](https://github.com/karpathy) for the original [autoresearch](https://github.com/karpathy/autoresearch) concept and implementation. See the upstream repo for full context on the project's design philosophy and goals.
