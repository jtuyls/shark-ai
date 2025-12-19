#!/bin/bash
# Copyright 2025 Advanced Micro Devices, Inc.
#
# Complete pipeline to download Llama-3.3-70B-Instruct-MXFP4, convert to IRPA, and export to MLIR

set -e  # Exit on error

SCRIPT_DIR=$(dirname $(realpath "$0"))
REPO_ROOT=$(dirname "$SCRIPT_DIR")

# Default values
MODEL_NAME="Llama-3.3-70B-Instruct-MXFP4"
MODEL_DIR="${REPO_ROOT}/${MODEL_NAME}"
OUTPUT_IRPA="${MODEL_DIR}/${MODEL_NAME}.irpa"
OUTPUT_MLIR="${MODEL_DIR}/model.mlir"
OUTPUT_CONFIG="${MODEL_DIR}/config_export.json"
HF_TOKEN=""
BATCH_SIZES_PREFILL="1,4"
BATCH_SIZES_DECODE="1,4,8"
BLOCK_SEQ_STRIDE=16
TENSOR_PARALLELISM_SIZE=""

function print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model-dir DIR              Directory to save/load model (default: ${MODEL_DIR})"
    echo "  --output-irpa FILE           Output IRPA file path (default: ${OUTPUT_IRPA})"
    echo "  --output-mlir FILE           Output MLIR file path (default: ${OUTPUT_MLIR})"
    echo "  --output-config FILE         Output config file path (default: ${OUTPUT_CONFIG})"
    echo "  --hf-token TOKEN             Hugging Face token (required for gated models)"
    echo "  --skip-download              Skip downloading model (use existing files)"
    echo "  --skip-irpa                  Skip IRPA conversion (use existing IRPA)"
    echo "  --bs-prefill BS              Batch sizes for prefill (default: ${BATCH_SIZES_PREFILL})"
    echo "  --bs-decode BS               Batch sizes for decode (default: ${BATCH_SIZES_DECODE})"
    echo "  --block-seq-stride N         Block sequence stride (default: ${BLOCK_SEQ_STRIDE})"
    echo "  --tensor-parallelism-size N  Tensor parallelism size (optional)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --hf-token YOUR_HF_TOKEN"
}

SKIP_DOWNLOAD=false
SKIP_IRPA=false

# Parse arguments
while [[ "$1" != "" ]]; do
    case "$1" in
        --model-dir)
            shift
            MODEL_DIR=$1
            ;;
        --output-irpa)
            shift
            OUTPUT_IRPA=$1
            ;;
        --output-mlir)
            shift
            OUTPUT_MLIR=$1
            ;;
        --output-config)
            shift
            OUTPUT_CONFIG=$1
            ;;
        --hf-token)
            shift
            HF_TOKEN=$1
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            ;;
        --skip-irpa)
            SKIP_IRPA=true
            ;;
        --bs-prefill)
            shift
            BATCH_SIZES_PREFILL=$1
            ;;
        --bs-decode)
            shift
            BATCH_SIZES_DECODE=$1
            ;;
        --block-seq-stride)
            shift
            BLOCK_SEQ_STRIDE=$1
            ;;
        --tensor-parallelism-size)
            shift
            TENSOR_PARALLELISM_SIZE=$1
            ;;
        -h | --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Invalid argument: $1"
            print_usage
            exit 1
            ;;
    esac
    shift
done

echo "=========================================="
echo "Llama-3.3-70B-Instruct-MXFP4 Conversion Pipeline"
echo "=========================================="
echo ""

# Step 1: Download model
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "Step 1: Downloading model from Hugging Face..."
    echo "Model directory: ${MODEL_DIR}"
    
    if [ -z "${HF_TOKEN}" ]; then
        echo "Warning: No HF token provided. This may fail if the model is gated."
        python3 "${SCRIPT_DIR}/download_llama33_mxfp4.py" --output-dir "${MODEL_DIR}"
    else
        python3 "${SCRIPT_DIR}/download_llama33_mxfp4.py" --output-dir "${MODEL_DIR}" --hf-token "${HF_TOKEN}"
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Model download failed"
        exit 1
    fi
    echo "Model downloaded successfully!"
    echo ""
else
    echo "Step 1: Skipping download (using existing model at ${MODEL_DIR})"
    echo ""
fi

# Step 2: Merge safetensors files if there are multiple
echo "Step 2: Checking for safetensors files..."
SAFETENSORS_FILES=(${MODEL_DIR}/*.safetensors)
NUM_SAFETENSORS=${#SAFETENSORS_FILES[@]}

if [ ${NUM_SAFETENSORS} -eq 0 ]; then
    echo "Error: No safetensors files found in ${MODEL_DIR}"
    exit 1
elif [ ${NUM_SAFETENSORS} -eq 1 ]; then
    echo "Found single safetensors file: ${SAFETENSORS_FILES[0]}"
    MERGED_SAFETENSORS="${SAFETENSORS_FILES[0]}"
else
    echo "Found ${NUM_SAFETENSORS} safetensors files, merging..."
    MERGED_SAFETENSORS="${MODEL_DIR}/merged.safetensors"
    
    # Check if merge script exists, if not we'll use the first file
    if [ -f "${SCRIPT_DIR}/merge_safetensors.py" ]; then
        python3 "${SCRIPT_DIR}/merge_safetensors.py" "${MODEL_DIR}"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to merge safetensors files"
            exit 1
        fi
    else
        echo "Warning: merge_safetensors.py not found, using first safetensors file"
        MERGED_SAFETENSORS="${SAFETENSORS_FILES[0]}"
    fi
fi
echo ""

# Step 3: Convert to IRPA using import_quark_dataset
if [ "$SKIP_IRPA" = false ]; then
    echo "Step 3: Converting to IRPA format..."
    echo "Input: ${MERGED_SAFETENSORS}"
    echo "Output: ${OUTPUT_IRPA}"
    
    CONFIG_JSON="${MODEL_DIR}/config.json"
    if [ ! -f "${CONFIG_JSON}" ]; then
        echo "Error: config.json not found at ${CONFIG_JSON}"
        exit 1
    fi
    
    python3 -m amdsharktank.models.llama.tools.import_quark_dataset \
        --params "${MERGED_SAFETENSORS}" \
        --output-irpa-file="${OUTPUT_IRPA}" \
        --config-json "${CONFIG_JSON}" \
        --model-base="70b" \
        --weight-dtype-override=float16 \
        --fp4-block-size=32 \
        --fp4-scale-format=fe8m0
    
    if [ $? -ne 0 ]; then
        echo "Error: IRPA conversion failed"
        exit 1
    fi
    echo "IRPA file created successfully: ${OUTPUT_IRPA}"
    echo ""
else
    echo "Step 3: Skipping IRPA conversion (using existing IRPA at ${OUTPUT_IRPA})"
    echo ""
fi

# Step 4: Export to MLIR
echo "Step 4: Exporting to MLIR format..."
echo "Input IRPA: ${OUTPUT_IRPA}"
echo "Output MLIR: ${OUTPUT_MLIR}"
echo "Output Config: ${OUTPUT_CONFIG}"
echo "Batch sizes prefill: ${BATCH_SIZES_PREFILL}"
echo "Batch sizes decode: ${BATCH_SIZES_DECODE}"
echo "Block seq stride: ${BLOCK_SEQ_STRIDE}"

EXPORT_CMD="python3 -m amdsharktank.examples.export_paged_llm_v1 \
    --irpa-file=${OUTPUT_IRPA} \
    --output-mlir=${OUTPUT_MLIR} \
    --output-config=${OUTPUT_CONFIG} \
    --bs-prefill=${BATCH_SIZES_PREFILL} \
    --bs-decode=${BATCH_SIZES_DECODE} \
    --block-seq-stride=${BLOCK_SEQ_STRIDE}"

if [ -n "${TENSOR_PARALLELISM_SIZE}" ]; then
    echo "Tensor parallelism size: ${TENSOR_PARALLELISM_SIZE}"
    EXPORT_CMD="${EXPORT_CMD} --tensor-parallelism-size=${TENSOR_PARALLELISM_SIZE}"
fi

eval ${EXPORT_CMD}

if [ $? -ne 0 ]; then
    echo "Error: MLIR export failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Conversion completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - IRPA weights: ${OUTPUT_IRPA}"
echo "  - MLIR model: ${OUTPUT_MLIR}"
echo "  - Config: ${OUTPUT_CONFIG}"
echo ""
echo "Next steps:"
echo "  1. To compile the MLIR to VMFB:"
echo "     iree-compile ${OUTPUT_MLIR} -o model.vmfb [compile flags]"
echo ""
echo "  2. To serve the model with shortfin:"
echo "     See docs/shortfin/llm/user/llama_serving.md"
echo ""


