# Guide: Converting Llama-3.3-70B-Instruct-MXFP4 to MLIR and IRPA

This guide explains how to download, import, and convert the AMD Llama-3.3-70B-Instruct-MXFP4-Preview model for use with amd-shark-ai.

## Overview

The [Llama-3.3-70B-Instruct-MXFP4-Preview](https://huggingface.co/amd/Llama-3.3-70B-Instruct-MXFP4-Preview) model is a quantized version of Meta's Llama-3.3-70B-Instruct model using:
- **Weight quantization:** OCP MXFP4, Static
- **Activation quantization:** OCP MXFP4, Dynamic  
- **KV cache quantization:** OCP FP8, Static
- **Quantization tool:** AMD-Quark (V0.9) with AutoSmoothQuant algorithm

## Target Hardware

| Target | GPU Architecture | FP8 Type |
|--------|-----------------|----------|
| MI300X | gfx942 | `float8_e4m3fnuz` |
| MI350/MI355 | gfx950 | `float8_e4m3fn` |

## Prerequisites

1. **amd-shark-ai installed** - Ensure you have the repository cloned and dependencies installed
2. **Hugging Face access** - You may need a Hugging Face token for model downloads
3. **Sufficient disk space** - ~50GB for the model and converted artifacts
4. **Python environment** - With required dependencies (torch, transformers, safetensors, etc.)
5. **IREE and iree-compile** - For compilation to VMFB

## Quick Start

See [LLAMA33_MXFP4_QUICKSTART.md](../LLAMA33_MXFP4_QUICKSTART.md) for a quick reference.

## Step-by-Step Process

### Step 1: Download the Model

Download from Hugging Face using the Python script:

```bash
python3 scripts/download_llama33_mxfp4.py \
    --output-dir ./Llama-3.3-70B-Instruct-MXFP4 \
    --hf-token YOUR_HF_TOKEN  # Optional
```

Or manually using `huggingface-cli`:

```bash
huggingface-cli download \
    amd/Llama-3.3-70B-Instruct-MXFP4-Preview \
    --local-dir ./Llama-3.3-70B-Instruct-MXFP4
```

This downloads:
- `config.json` - Model configuration (includes `num_hidden_layers: 80`)
- `*.safetensors` - Model weights (quantized, may be sharded)
- `tokenizer.json` and `tokenizer_config.json` - Tokenizer files
- Other metadata files

### Step 2: Merge Safetensors (if sharded)

If the model has multiple safetensors files, merge them first:

```bash
python3 scripts/merge_safetensors.py ./Llama-3.3-70B-Instruct-MXFP4
```

This creates `./Llama-3.3-70B-Instruct-MXFP4/merged.safetensors`.

### Step 3: Convert to IRPA Format

IRPA (IREE Parameter Archive) is the native format for amd-shark-ai.

**For MI300X (gfx942):**
```bash
python3 -m amdsharktank.models.llama.tools.import_quark_dataset \
    --params ./Llama-3.3-70B-Instruct-MXFP4/merged.safetensors \
    --output-irpa-file=./Llama-3.3-70B-Instruct-MXFP4/model.irpa \
    --config-json ./Llama-3.3-70B-Instruct-MXFP4/config.json \
    --model-base="70b" \
    --weight-dtype-override=float16 \
    --fp4-block-size=32 \
    --fp4-scale-format=fe8m0 \
    --quantizer-dtype=float8_e4m3fnuz \
    --apply-shuffle
```

**For MI350/MI355 (gfx950):**
```bash
python3 -m amdsharktank.models.llama.tools.import_quark_dataset \
    --params ./Llama-3.3-70B-Instruct-MXFP4/merged.safetensors \
    --output-irpa-file=./Llama-3.3-70B-Instruct-MXFP4/model_mi355.irpa \
    --config-json ./Llama-3.3-70B-Instruct-MXFP4/config.json \
    --model-base="70b" \
    --weight-dtype-override=float16 \
    --fp4-block-size=32 \
    --fp4-scale-format=fe8m0 \
    --quantizer-dtype=float8_e4m3fn \
    --apply-shuffle
```

**Key parameters:**
- `--model-base="70b"` - Model size (for QKV tensor splitting pattern)
- `--weight-dtype-override=float16` - Casts certain weights to FP16
- `--fp4-block-size=32` - Block size for FP4 quantization
- `--fp4-scale-format=fe8m0` - Scale format for FP4 quantization
- `--quantizer-dtype` - FP8 type for quantizer (fnuz for MI300X, fn for MI355)
- `--apply-shuffle` - **Required** for the ASM matmul kernel

### Step 4: Export to Torch-MLIR

Export the IRPA model to MLIR format suitable for IREE compilation:

**For MI300X:**
```bash
python3 -m amdsharktank.examples.export_paged_llm_v1 \
    --irpa-file=./Llama-3.3-70B-Instruct-MXFP4/model.irpa \
    --output-mlir=./Llama-3.3-70B-Instruct-MXFP4/model.mlir \
    --output-config=./Llama-3.3-70B-Instruct-MXFP4/config_export.json \
    --bs-prefill=4 \
    --bs-decode=4 \
    --activation-dtype=float16 \
    --attention-dtype=float16 \
    --attention-kernel=torch \
    --use-hf \
    --matmul-kernel="sharktank.asm;*"
```

**For MI350/MI355:**
```bash
python3 -m amdsharktank.examples.export_paged_llm_v1 \
    --irpa-file=./Llama-3.3-70B-Instruct-MXFP4/model_mi355.irpa \
    --output-mlir=./Llama-3.3-70B-Instruct-MXFP4/model_mi355.mlir \
    --output-config=./Llama-3.3-70B-Instruct-MXFP4/config_mi355.json \
    --bs-prefill=4 \
    --bs-decode=4 \
    --activation-dtype=float16 \
    --attention-dtype=float16 \
    --attention-kernel=torch \
    --kv-cache-dtype=float8_e4m3fn \
    --use-hf \
    --matmul-kernel="sharktank.asm;*"
```

**Key parameters:**
- `--bs-prefill` - Batch sizes for prefill phase
- `--bs-decode` - Batch sizes for decode phase
- `--activation-dtype` - Data type for activations
- `--attention-dtype` - Data type for attention computations
- `--attention-kernel=torch` - Use PyTorch attention kernel
- `--kv-cache-dtype` - FP8 type for KV cache (fn for MI355)
- `--use-hf` - Use Hugging Face model format
- `--matmul-kernel="sharktank.asm;*"` - Use optimized ASM kernel for matmul

> **Note:** Do NOT use `--top-k=1` if you plan to evaluate perplexity, as it exports only top-1 token indices instead of full logits.

### Step 5: Compile to VMFB

Compile the MLIR to IREE VMFB format for execution:

**For MI300X (gfx942):**
```bash
iree-compile ./Llama-3.3-70B-Instruct-MXFP4/model.mlir \
    -o ./Llama-3.3-70B-Instruct-MXFP4/model.vmfb \
    --iree-hal-target-device=hip \
    --iree-hip-target=gfx942
```

**For MI350/MI355 (gfx950):**
```bash
iree-compile ./Llama-3.3-70B-Instruct-MXFP4/model_mi355.mlir \
    -o ./Llama-3.3-70B-Instruct-MXFP4/model_mi355.vmfb \
    --iree-hal-target-device=hip \
    --iree-hip-target=gfx950
```

## Output Files

After successful conversion, you'll have:

1. **`model.irpa` / `model_mi355.irpa`** - Model weights in IRPA format (with shuffled FP4 weights)
2. **`model.mlir` / `model_mi355.mlir`** - Model in MLIR format with paged attention support
3. **`config_export.json` / `config_mi355.json`** - Export configuration for the serving runtime
4. **`model.vmfb` / `model_mi355.vmfb`** - Compiled model ready for deployment

## Perplexity Evaluation

Evaluate the model's perplexity to verify conversion quality:

**With pre-compiled VMFB (recommended):**
```bash
python3 -m amdsharktank.evaluate.perplexity_iree \
    --irpa-file=./Llama-3.3-70B-Instruct-MXFP4/model_mi355.irpa \
    --tokenizer-config-json=./Llama-3.3-70B-Instruct-MXFP4/tokenizer_config.json \
    --num-prompts=4 \
    --iree-device='hip://0' \
    --iree-hal-target-device=hip \
    --iree-hip-target=gfx950 \
    --kv-cache-dtype=float8_e4m3fn \
    --input-vmfb=./Llama-3.3-70B-Instruct-MXFP4/model_mi355.vmfb
```

**Expected results:**
```
Mean perplexity: ~8.4
Prefill time: ~105 ms
Decode time per token: ~56 ms
```

## Advanced Options

### Tensor Parallelism

For multi-GPU deployment, use tensor parallelism:

```bash
# First, shard the model
python3 -m amdsharktank.examples.sharding.shard_llm_dataset \
    --irpa-file=model.irpa \
    --output-irpa=model.sharded.irpa \
    --tensor-parallelism-size=8

# Then export with TP flag
python3 -m amdsharktank.examples.export_paged_llm_v1 \
    --irpa-file=model.sharded.irpa \
    --output-mlir=model.mlir \
    --output-config=config.json \
    --bs-prefill=4 \
    --bs-decode=4 \
    --activation-dtype=float16 \
    --attention-dtype=float16 \
    --attention-kernel=torch \
    --kv-cache-dtype=float8_e4m3fn \
    --use-hf \
    --matmul-kernel="sharktank.asm;*" \
    --tensor-parallelism-size=8
```

This creates sharded IRPA files: `model.rank0.irpa`, `model.rank1.irpa`, etc.

### Custom Batch Sizes

Adjust batch sizes based on your use case:

```bash
# For high-throughput serving
--bs-prefill=1,2,4,8,16 \
--bs-decode=1,4,8,16,32

# For low-latency serving (single request)
--bs-prefill=1 \
--bs-decode=1
```

## Serving the Model

Once you have the compiled artifacts, serve the model using shortfin:

```bash
python3 -m shortfin_apps.llm.server \
    --model_config=./Llama-3.3-70B-Instruct-MXFP4/config_mi355.json \
    --vmfb=./Llama-3.3-70B-Instruct-MXFP4/model_mi355.vmfb \
    --parameters=./Llama-3.3-70B-Instruct-MXFP4/model_mi355.irpa \
    --tokenizer=./Llama-3.3-70B-Instruct-MXFP4/tokenizer.json
```

See [docs/shortfin/llm/user/llama_serving.md](./shortfin/llm/user/llama_serving.md) for detailed serving documentation.

## Troubleshooting

### FP8 Type Errors

If you see errors about `F8E5M2FNUZ` or `F8E4M3FNUZ` not being supported:
- You're targeting gfx950 (MI355) with fnuz types (only supported on gfx942/MI300X)
- Re-convert IRPA with `--quantizer-dtype=float8_e4m3fn`
- Re-export MLIR with `--kv-cache-dtype=float8_e4m3fn`

### ROCm/PyTorch Compatibility

If you encounter `hipErrorNoDevice` errors:

1. Set ROCm environment variables:
   ```bash
   export ROCM_PATH=/opt/rocm
   export HIP_PATH=/opt/rocm
   export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
   ```

2. **PyTorch/ROCm Version Conflict:** If PyTorch is compiled with a different ROCm version than your system (e.g., PyTorch for ROCm 6.2 with ROCm 6.5 installed), it can corrupt the HIP context for IREE. Install CPU-only PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. Check GPU visibility with `rocm-smi` before running

### High Perplexity Values

If perplexity is extremely high (>1000):
- Ensure `--apply-shuffle` was used during IRPA creation
- Ensure `--matmul-kernel="sharktank.asm;*"` is used during export
- Do NOT use `--top-k=1` as it breaks perplexity calculation

### Out of Memory During Conversion

If you run out of memory during IRPA conversion:
- Use a machine with more RAM (70B models need ~100GB+ for conversion)
- Use swap space (slower but works)

### Layer Count Issues

The Llama-3.3-70B model has 80 layers. The conversion script reads this from `config.json` automatically. If you encounter layer count mismatches:
- Ensure you're using the correct `config.json` from the model directory
- The `--model-base="70b"` parameter affects QKV tensor splitting, not layer count

## Model Evaluation

According to the model card, this MXFP4 quantized model achieves:

| Benchmark | Original FP | MXFP4 | Recovery |
|-----------|-------------|-------|----------|
| MMLU (5-shot) | 83.29 | 80.99 | 97.24% |
| GSM8K_COT (8-shot) | 93.18 | 92.12 | 98.86% |
| ARC Challenge (0-shot) | 94.25 | 93.05 | 98.73% |
| IFEVAL (0-shot) | 89.8 | 88.00 | 98.00% |

## References

- Model: https://huggingface.co/amd/Llama-3.3-70B-Instruct-MXFP4-Preview
- AMD-Quark: https://github.com/amd/quark
- amd-shark-ai Documentation: [docs/](../docs/)
- Serving Guide: [docs/shortfin/llm/user/llama_serving.md](./shortfin/llm/user/llama_serving.md)

## License

This model inherits the Llama 3 license. See the model card for full details.

Modifications Copyright(c) 2025 Advanced Micro Devices, Inc. All rights reserved.
