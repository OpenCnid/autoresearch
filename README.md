```markdown
# autoresearch — WSL2 / RTX Edition

> Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for **NVIDIA RTX GPUs** (Ampere, Ada, Blackwell) running on **WSL2**.

The upstream repo targets H100 GPUs with Flash Attention 3 and torch.compile. This fork makes three targeted changes to run on consumer RTX hardware under Windows Subsystem for Linux:

1. **Flash Attention 3 → PyTorch SDPA** — FA3 requires SM 9.0 (Hopper). We use `F.scaled_dot_product_attention` which works on all CUDA GPUs.
2. **torch.compile disabled** — The Inductor/Triton backend currently fails on Ampere with this model's architecture. Eager mode works reliably.
3. **DEVICE_BATCH_SIZE scaled for 24GB** — Default tuned for RTX 3090 (24GB VRAM).

Everything else is identical to upstream: same model architecture, same optimizer (Muon + AdamW), same 5-minute time budget, same val_bpb metric.

## Tested On

| GPU | VRAM | OS | Status |
|-----|------|----|--------|
| RTX 3090 | 24GB | Ubuntu 24.04 (WSL2) | ✅ Working |

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

## Tuning for Your GPU

If you have less than 24GB VRAM, reduce `DEVICE_BATCH_SIZE` in `train.py`. If you OOM, halve it until it fits:

| VRAM | Suggested DEVICE_BATCH_SIZE |
|------|-----------------------------|
| 24GB | 32 |
| 16GB | 16 |
| 10-12GB | 8 |

For significantly smaller GPUs, also follow Karpathy's upstream recommendations: lower `DEPTH`, switch to TinyStories dataset, reduce `MAX_SEQ_LEN` in `prepare.py`.

## What's Different from Upstream

| Feature | Upstream (H100) | This Fork (RTX/WSL2) |
|---------|-----------------|----------------------|
| Attention | Flash Attention 3 | PyTorch SDPA |
| Compilation | torch.compile | Eager mode |
| Window attention | Native FA3 windowed | Full causal (no window) |
| Default batch size | 128 | 32 |
| MFU | ~40% | ~3-5% (eager penalty) |

## Known Limitations

- **No window attention** — SDPA doesn't natively support windowed attention. All layers use full causal attention. This may slightly affect val_bpb compared to upstream results.
- **Lower MFU** — Without torch.compile, GPU utilization is significantly lower. We're exploring Inductor fixes to re-enable compilation on Ampere.
- **torch.compile investigation ongoing** — The Inductor backend fails during Triton codegen with this specific model architecture on SM 8.6. PRs welcome.

## Running the Agent

Same as upstream — point Claude/Codex at `program.md` and go:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Upstream

All credit to [@karpathy](https://github.com/karpathy) for the original [autoresearch](https://github.com/karpathy/autoresearch) concept and implementation.
```