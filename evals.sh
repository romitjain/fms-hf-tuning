set -xe

# Base model evaluation
lm-eval run \
    --model vllm \
    --model_args pretrained=ibm-granite/granite-4.0-1b,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8 \
    --apply_chat_template \
    --batch_size 4 \
    --output_path "./eval/base/" \
    --log_samples \
    --tasks ifeval \
    -w

lm-eval run \
    --model vllm \
    --model_args pretrained=ibm-granite/granite-4.0-1b,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8 \
    --apply_chat_template \
    --batch_size 4 \
    --output_path "./eval/base/" \
    --log_samples \
    --num_fewshot 5 \
    --tasks mmlu \
    -w


# BF16
lm-eval run \
    --model vllm \
    --model_args pretrained=/workspace/fms-hf-tuning/fp8/bf16/final,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8 \
    --apply_chat_template \
    --batch_size 4 \
    --output_path "./eval/bf16/" \
    --log_samples \
    --tasks ifeval \
    -w

lm-eval run \
    --model vllm \
    --model_args pretrained=/workspace/fms-hf-tuning/fp8/bf16/final,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8 \
    --apply_chat_template \
    --batch_size 4 \
    --output_path "./eval/bf16/" \
    --log_samples \
    --num_fewshot 5 \
    --tasks mmlu \
    -w

# FP8
lm-eval run \
    --model vllm \
    --model_args pretrained=/workspace/fms-hf-tuning/fp8/fp8/final,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8 \
    --apply_chat_template \
    --batch_size 4 \
    --output_path "./eval/fp8/" \
    --log_samples \
    --tasks ifeval \
    -w

lm-eval run \
    --model vllm \
    --model_args pretrained=/workspace/fms-hf-tuning/fp8/fp8/final,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8 \
    --apply_chat_template \
    --batch_size 4 \
    --output_path "./eval/fp8/" \
    --log_samples \
    --num_fewshot 5 \
    --tasks mmlu \
    -w
