#!/usr/bin/env python3
"""
Script to save inputs to prefill and decode phases for IREE model benchmarking.
Matches the naming convention in iree-model-benchmark/llama3/inputs.
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
import random

# Add amdsharktank to path
sys.path.insert(0, str(Path(__file__).parent.parent / "amdsharktank"))

from amdsharktank.utils.evaluate import get_prompts, pad_tokens
from amdsharktank.utils.tokenizer import load_tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Save prefill and decode inputs for IREE benchmarking")
    
    # Required args
    parser.add_argument("--irpa-file", required=True, help="Path to IRPA file")
    parser.add_argument("--tokenizer-config-json", required=True, help="Path to tokenizer config")
    parser.add_argument("--output-dir", default="./saved_inputs", help="Directory to save inputs")
    
    # Model options
    parser.add_argument("--prefill-length", type=int, default=2048, help="Prefill length (sequence length)")
    parser.add_argument("--num-prompts", type=int, default=4, help="Number of prompts (batch size)")
    parser.add_argument("--block-seq-stride", type=int, default=16, help="Block sequence stride")
    
    # Optional: use a custom long prompt
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt to use (for long sequences)")
    
    # KV cache options (for creating empty cache state)
    parser.add_argument("--num-kv-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--num-layers", type=int, default=80, help="Number of transformer layers")
    parser.add_argument("--page-cache-size", type=int, default=512, help="Number of pages in KV cache")
    parser.add_argument("--kv-cache-dtype", type=str, default="float8_e4m3fn", 
                        help="KV cache dtype (float16, float8_e4m3fn, etc.)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer_config = Path(args.tokenizer_config_json)
    tokenizer_dir = tokenizer_config.parent
    tokenizer = load_tokenizer(str(tokenizer_dir))
    
    # Get or create prompts
    if args.prompt:
        # Use custom prompt, replicated for batch
        test_prompts = [args.prompt] * args.num_prompts
    else:
        # Get prompts from wikitext - concatenate multiple to get longer sequences
        raw_prompts = get_prompts(num_prompts=50)
        # Concatenate multiple prompts to create longer sequences
        long_prompt = " ".join(raw_prompts[:20])  # Join 20 prompts for longer text
        test_prompts = [long_prompt] * args.num_prompts
    
    print(f"Batch size: {args.num_prompts}")
    print(f"Target prefill length: {args.prefill_length}")
    print(f"Prompt preview (first 200 chars): {test_prompts[0][:200]}...")
    
    # Tokenize
    token_ids, seq_lens = tokenizer.encode(
        test_prompts,
        pad_to_multiple_of=args.block_seq_stride,
    )
    
    print(f"Tokenized sequences: {len(token_ids)}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Max sequence length: {max(seq_lens)}")
    
    # Pad tokens for batching
    token_ids_padded, seq_lens_padded = pad_tokens(
        token_ids,
        pad_to_multiple_of=args.block_seq_stride,
    )
    
    token_ids_tensor = torch.tensor(token_ids_padded, dtype=torch.int64)
    seq_lens_tensor = torch.tensor(seq_lens_padded, dtype=torch.int64)
    
    print(f"Padded token shape: {token_ids_tensor.shape}")
    
    # Calculate effective prefill length (can't exceed sequence length)
    effective_prefill = min(args.prefill_length, max(seq_lens))
    print(f"Effective prefill length: {effective_prefill}")
    
    # Calculate block IDs for paged attention
    num_blocks_prefill = (effective_prefill + args.block_seq_stride - 1) // args.block_seq_stride
    num_blocks_decode = num_blocks_prefill + 1  # Decode may need one more block
    
    # Create sequential block IDs for each sequence in batch
    # Each sequence gets its own set of blocks
    seq_block_ids_prefill = torch.zeros((args.num_prompts, num_blocks_prefill), dtype=torch.int64)
    seq_block_ids_decode = torch.zeros((args.num_prompts, num_blocks_decode), dtype=torch.int64)
    
    for b in range(args.num_prompts):
        base_block = b * num_blocks_decode  # Each batch item gets separate blocks
        seq_block_ids_prefill[b] = torch.arange(base_block, base_block + num_blocks_prefill)
        seq_block_ids_decode[b] = torch.arange(base_block, base_block + num_blocks_decode)
    
    # ============ PREFILL INPUTS ============
    # Following naming: prefill_input{N}_{name}.npy
    prefill_tokens = token_ids_tensor[:, :effective_prefill]
    prefill_seq_lens = torch.minimum(seq_lens_tensor, torch.tensor(effective_prefill))
    
    # Save prefill inputs
    np.save(output_dir / "prefill_input0_tokens.npy", prefill_tokens.numpy().astype(np.int64))
    np.save(output_dir / "prefill_input1_seq_lens.npy", prefill_seq_lens.numpy().astype(np.int64))
    np.save(output_dir / "prefill_input2_seq_block_ids.npy", seq_block_ids_prefill.numpy().astype(np.int64))
    
    # Create empty KV cache state
    # Shape: (page_cache_size, page_size_elements) where page contains K and V for all layers
    # Page size = num_layers * 2 (K+V) * num_kv_heads * block_seq_stride * head_dim
    page_size_elements = args.num_layers * 2 * args.num_kv_heads * args.block_seq_stride * args.head_dim
    
    # Use void dtype (|V1) for FP8 to match iree-model-benchmark format
    if args.kv_cache_dtype in ["float8_e4m3fn", "float8_e4m3fnuz"]:
        # Create as void type with 1 byte per element (|V1)
        kv_cache_state = np.zeros((args.page_cache_size, page_size_elements), dtype=np.dtype('V1'))
    elif args.kv_cache_dtype == "float16":
        kv_cache_state = np.zeros((args.page_cache_size, page_size_elements), dtype=np.float16)
    else:
        kv_cache_state = np.zeros((args.page_cache_size, page_size_elements), dtype=np.float16)
    
    np.save(output_dir / "prefill_input3_kv_cache_state.npy", kv_cache_state)
    
    print(f"\nSaved prefill inputs:")
    print(f"  prefill_input0_tokens.npy: shape={prefill_tokens.shape}")
    print(f"  prefill_input1_seq_lens.npy: shape={prefill_seq_lens.shape}, values={prefill_seq_lens.tolist()}")
    print(f"  prefill_input2_seq_block_ids.npy: shape={seq_block_ids_prefill.shape}")
    print(f"  prefill_input3_kv_cache_state.npy: shape={kv_cache_state.shape}")
    
    # ============ DECODE INPUTS ============
    # Following naming: decode_input{N}_{name}.npy
    # Decode uses the next token after prefill
    decode_position = effective_prefill  # Position of decode token
    decode_tokens = token_ids_tensor[:, decode_position:decode_position+1]
    decode_seq_lens = torch.minimum(seq_lens_tensor, torch.tensor(decode_position + 1))
    decode_start_positions = torch.full((args.num_prompts,), decode_position, dtype=torch.int64)
    
    # Save decode inputs
    np.save(output_dir / "decode_input0_tokens.npy", decode_tokens.numpy().astype(np.int64))
    np.save(output_dir / "decode_input1_seq_lens.npy", decode_seq_lens.numpy().astype(np.int64))
    np.save(output_dir / "decode_input2_start_positions.npy", decode_start_positions.numpy().astype(np.int64))
    np.save(output_dir / "decode_input3_seq_block_ids.npy", seq_block_ids_decode.numpy().astype(np.int64))
    np.save(output_dir / "decode_input4_kv_cache_state.npy", kv_cache_state)
    
    print(f"\nSaved decode inputs:")
    print(f"  decode_input0_tokens.npy: shape={decode_tokens.shape}")
    print(f"  decode_input1_seq_lens.npy: shape={decode_seq_lens.shape}, values={decode_seq_lens.tolist()}")
    print(f"  decode_input2_start_positions.npy: shape={decode_start_positions.shape}, values={decode_start_positions.tolist()}")
    print(f"  decode_input3_seq_block_ids.npy: shape={seq_block_ids_decode.shape}")
    print(f"  decode_input4_kv_cache_state.npy: shape={kv_cache_state.shape}")
    
    # Save metadata
    metadata = {
        "irpa_file": str(args.irpa_file),
        "prefill_length": effective_prefill,
        "batch_size": args.num_prompts,
        "sequence_lengths": seq_lens,
        "max_sequence_length": max(seq_lens),
        "block_seq_stride": args.block_seq_stride,
        "num_blocks_prefill": num_blocks_prefill,
        "num_blocks_decode": num_blocks_decode,
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_cache_shape": list(kv_cache_state.shape),
        "prompt_preview": test_prompts[0][:500],
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save full token sequence for reference
    np.save(output_dir / "full_token_ids.npy", token_ids_tensor.numpy())
    
    # Save decoded tokens for inspection
    with open(output_dir / "tokens_decoded.txt", "w") as f:
        for i, token_row in enumerate(token_ids):
            f.write(f"=== Sequence {i} (length {seq_lens[i]}) ===\n")
            decoded = tokenizer.decode([token_row[:seq_lens[i]]])
            f.write(decoded[0] if decoded else "")
            f.write("\n\n")
    
    print(f"\n✓ All inputs saved to {output_dir}/")
    print(f"\nFiles created:")
    for f in sorted(output_dir.glob("*.npy")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
