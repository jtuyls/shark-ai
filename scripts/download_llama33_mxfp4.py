#!/usr/bin/env python3
# Copyright 2025 Advanced Micro Devices, Inc.
#
# Script to download Llama-3.3-70B-Instruct-MXFP4-Preview model from Hugging Face

from huggingface_hub import snapshot_download
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download Llama-3.3-70B-Instruct-MXFP4 model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Llama-3.3-70B-Instruct-MXFP4",
        help="Directory to save the downloaded model"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (if required for gated models)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading amd/Llama-3.3-70B-Instruct-MXFP4-Preview to {output_dir}")
    
    try:
        model_dir = snapshot_download(
            repo_id="amd/Llama-3.3-70B-Instruct-MXFP4-Preview",
            local_dir=output_dir,
            token=args.hf_token,
        )
        print(f"Model downloaded successfully to: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if __name__ == "__main__":
    main()


