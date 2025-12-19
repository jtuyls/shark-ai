---
license: llama3
base_model:
- meta-llama/Llama-3.3-70B-Instruct
---


# Model Overview

- **Model Architecture:** Llama-3.3
  - **Input:** Text
  - **Output:** Text
- **Supported Hardware Microarchitecture:** AMD MI350/MI355
- **ROCm**: 7.0
- **PyTorch**: 2.8.0
- **Transformers**: 4.53.0
- **Operating System(s):** Linux
- **Inference Engine:** [vLLM](https://docs.vllm.ai/en/latest/)
- **Model Optimizer:** [AMD-Quark](https://quark.docs.amd.com/latest/index.html) (V0.9)
  - **Weight quantization:** OCP MXFP4, Static
  - **Activation quantization:** OCP MXFP4, Dynamic
  - **KV cache quantization:** OCP FP8, Static
- **Calibration Dataset:** [Pile](https://huggingface.co/datasets/mit-han-lab/pile-val-backup)

This model was built with Meta Llama by applying [AMD-Quark](https://quark.docs.amd.com/latest/index.html) for MXFP4 quantization.


# Model Quantization

This model was obtained by quantizing [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)'s weights and activations to MXFP4 and KV caches to FP8, using AutoSmoothQuant algorithm in [AMD-Quark](https://quark.docs.amd.com/latest/index.html).

**Quantization scripts:**
```
cd Quark/examples/torch/language_modeling/llm_ptq/
python3 quantize_quark.py --model_dir meta-llama/Llama-3.3-70B-Instruct \
                          --quant_scheme w_mxfp4_a_mxfp4 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 128 \
                          --multi_gpu \
                          --quant_algo autosmoothquant 
                          --model_export hf_format \
                          --output_dir amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-ASQ
```

# Deployment
### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend.

## Evaluation

The model was evaluated on MMLU, GSM8K_COT, ARC Challenge and IFEVAL.
Evaluation was conducted using the framework [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the vLLM engine.

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Llama-3.3-70B-Instruct </strong>
   </td>
   <td><strong>Llama-3.3-70B-Instruct-MXFP4(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>83.29
   </td>
   <td>80.99
   </td>
   <td>97.24%
   </td>
  </tr>
  <tr>
   <td>GSM8K_COT (8-shot, strict-match)
   </td>
   <td>93.18
   </td>
   <td>92.12
   </td>
   <td>98.86%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (0-shot)
   </td>
   <td>94.25
   </td>
   <td>93.05
   </td>
   <td>98.73%
   </td>
  </tr>  
  <tr>
   <td>IFEVAL (0-shot, (inst_level_strict_acc+prompt_level_strict_acc)/2)
   </td>
   <td>89.8
   </td>
   <td>88.00
   </td>
   <td>98.00%
   </td>
  </tr>
</table>


### Reproduction

The results were obtained using the following commands:

#### MMLU
```
lm_eval \
    --model vllm \
    --model_args pretrained=amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-ASQ,gpu_memory_utilization=0.8,tensor_parallel_size=4,kv_cache_dtype='fp8' \
    --tasks mmlu_llama \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --num_fewshot 5 \
    --batch_size auto
```

#### GSM8K_COT
```
lm_eval \
    --model vllm \
    --model_args pretrained=amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-ASQ,gpu_memory_utilization=0.8,tensor_parallel_size=4,kv_cache_dtype='fp8' \
    --tasks gsm8k_llama \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --num_fewshot 8 \
    --batch_size auto
```

#### ARC Challenge
```
lm_eval \
    --model vllm \
    --model_args pretrained=amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-ASQ,gpu_memory_utilization=0.8,tensor_parallel_size=4,kv_cache_dtype='fp8' \
    --tasks arc_challenge_llama \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto
```

#### IFEVAL
```
lm_eval \
    --model vllm \
    --model_args pretrained=amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-ASQ,gpu_memory_utilization=0.8,tensor_parallel_size=4,kv_cache_dtype='fp8' \
    --tasks ifeval \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto
```

# License
Modifications Copyright(c) 2025 Advanced Micro Devices, Inc. All rights reserved.