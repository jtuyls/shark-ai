# Llama-3.3-70B-Instruct-MXFP4 Quick Start

Quick reference for converting the [AMD Llama-3.3-70B-Instruct-MXFP4-Preview](https://huggingface.co/amd/Llama-3.3-70B-Instruct-MXFP4-Preview) model.

## Target Hardware

| Target | GPU Architecture | FP8 Quantizer Type |
|--------|-----------------|-------------------|
| MI300X | gfx942 | `float8_e4m3fnuz` (default) |
| MI350/MI355 | gfx950 | `float8_e4m3fn` |

## Manual Step-by-Step

### 1. Download
```bash
python3 scripts/download_llama33_mxfp4.py \
    --output-dir ./Llama-3.3-70B-Instruct-MXFP4 \
    --hf-token YOUR_HF_TOKEN
```

### 2. Merge Safetensors (if multiple files)
```bash
python3 scripts/merge_safetensors.py ./Llama-3.3-70B-Instruct-MXFP4
```
This creates `./Llama-3.3-70B-Instruct-MXFP4/merged.safetensors`.

### 3. Convert to IRPA

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

> **Note:** The `--apply-shuffle` flag is required for the optimized ASM matmul kernel. Omit for IREE kernel.

### 4. Export to MLIR

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
    --matmul-kernel="amdsharktank.asm;*"
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
    --matmul-kernel="amdsharktank.asm;*"
```

### 5. Compile to VMFB

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

## Alternative: IREE Kernel (No Shuffle Required)

If you prefer not to shuffle weights, use the IREE kernel:

**3. Convert to IRPA (no shuffle):**
```bash
python3 -m amdsharktank.models.llama.tools.import_quark_dataset \
    --params ./Llama-3.3-70B-Instruct-MXFP4/merged.safetensors \
    --output-irpa-file=./Llama-3.3-70B-Instruct-MXFP4/model_mi355_no_shuffle.irpa \
    --config-json ./Llama-3.3-70B-Instruct-MXFP4/config.json \
    --model-base="70b" \
    --weight-dtype-override=float16 \
    --fp4-block-size=32 \
    --fp4-scale-format=fe8m0 \
    --quantizer-dtype=float8_e4m3fn
```

**4. Export with IREE kernel:**
```bash
python3 -m amdsharktank.examples.export_paged_llm_v1 \
    --irpa-file=./Llama-3.3-70B-Instruct-MXFP4/model_mi355_no_shuffle.irpa \
    --output-mlir=./Llama-3.3-70B-Instruct-MXFP4/model_mi355_iree.mlir \
    --output-config=./Llama-3.3-70B-Instruct-MXFP4/config_mi355_iree.json \
    --bs-prefill=4 \
    --bs-decode=4 \
    --activation-dtype=float16 \
    --attention-dtype=float16 \
    --attention-kernel=torch \
    --kv-cache-dtype=float8_e4m3fn \
    --use-hf \
    --matmul-kernel="amdsharktank.iree;*"
```

**5. Compile (requires extra flag):**
```bash
iree-compile ./Llama-3.3-70B-Instruct-MXFP4/model_mi355_iree.mlir \
    -o ./Llama-3.3-70B-Instruct-MXFP4/model_mi355_iree.vmfb \
    --iree-hal-target-device=hip \
    --iree-hip-target=gfx950 \
    --iree-stream-affinity-solver-max-iterations=1024
```

## Output Files

- ✅ `model.irpa` / `model_mi355.irpa` - Weights in IRPA format (with shuffled FP4 weights)
- ✅ `model.mlir` / `model_mi355.mlir` - Model in MLIR/Torch-MLIR format  
- ✅ `config_export.json` / `config_mi355.json` - Export configuration
- ✅ `model.vmfb` / `model_mi355.vmfb` - Compiled model

## Perplexity Evaluation

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

**With on-the-fly compilation:**
```bash
python3 -m amdsharktank.evaluate.perplexity_iree \
    --irpa-file=./Llama-3.3-70B-Instruct-MXFP4/model_mi355.irpa \
    --tokenizer-config-json=./Llama-3.3-70B-Instruct-MXFP4/tokenizer_config.json \
    --num-prompts=4 \
    --iree-device='hip://0' \
    --iree-hal-target-device=hip \
    --iree-hip-target=gfx950 \
    --kv-cache-dtype=float8_e4m3fn \
    --attention-kernel=torch \
    --matmul-kernel="amdsharktank.asm;*"
```

## Full Documentation

See [docs/LLAMA33_MXFP4_GUIDE.md](docs/LLAMA33_MXFP4_GUIDE.md) for:
- Detailed explanations
- Advanced options (tensor parallelism, custom batch sizes, etc.)
- Troubleshooting
- Serving instructions

## Important Notes

- **FP8 Types:** MI300X uses `float8_e4m3fnuz`, MI350/MI355 uses `float8_e4m3fn`
- **Layer Count:** Llama-3.3-70B has 80 layers (read from `config.json`)
- **Model Size:** ~40GB for IRPA, ~3-400MB for compiled VMFB
- **ASM Kernel:** Use `--apply-shuffle` during IRPA creation and `--matmul-kernel="amdsharktank.asm;*"` during export
- **IREE Kernel:** Omit `--apply-shuffle` and use `--matmul-kernel="amdsharktank.iree;*"` during export
- **Pre-compiled VMFB:** Use `--input-vmfb` to skip on-the-fly compilation during perplexity evaluation

## Troubleshooting

### ROCm/PyTorch Compatibility
If you encounter `hipErrorNoDevice` errors when running perplexity evaluation:

1. Set ROCm environment variables:
   ```bash
   export ROCM_PATH=/opt/rocm
   export HIP_PATH=/opt/rocm
   export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
   ```

2. **PyTorch/ROCm Version Conflict:** If PyTorch is compiled with a different ROCm version than your system (e.g., PyTorch for ROCm 6.2 with ROCm 6.5 installed), it can corrupt the HIP context for IREE. Solution: Install CPU-only PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. Check GPU visibility with `rocm-smi` before running

### FP8 Type Errors
If you see errors about `F8E5M2FNUZ` or `F8E4M3FNUZ` not being supported:
- You're targeting gfx950 (MI355) with fnuz types
- Re-convert IRPA with `--quantizer-dtype=float8_e4m3fn`
- Re-export MLIR with `--kv-cache-dtype=float8_e4m3fn`

### High Perplexity Values
If perplexity is extremely high (>1000):
- Ensure `--apply-shuffle` was used during IRPA creation when using ASM kernel
- Ensure `--matmul-kernel="amdsharktank.asm;*"` is used during export with shuffled IRPA
- Do NOT use `--top-k=1` as it breaks perplexity calculation
- If using IREE kernel, ensure IRPA was created WITHOUT `--apply-shuffle`

### Affinity Analysis Failure
If you see "failed to solve for affinity analysis" during compilation:
- This occurs with IREE kernel MLIR
- Add `--iree-stream-affinity-solver-max-iterations=1024` to your iree-compile command
