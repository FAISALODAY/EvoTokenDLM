#!/bin/bash

# Configuration variables
GPU_IDS=(0 1 2 3)

MASTER_PORT=29412


model_path="GSAI-ML/LLaDA-8B-Instruct"
checkpoint_path="zhongzero/EvoToken_LLaDA_Instruct_8B_Lora"
output_dir_base="outputs/eval_results"


TASKS=("countdown" "math" "gsm8k" "svamp")
GEN_LENGTHS=(128 256)
GENERATE_MODES=("generate_soft_token")
STEPS_FACTORS=(0.5)
ALPHA_SOFT_MASKS=(0.9 0.8 0.7 0.6 0.5)
k_soft=3

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for gen_length in "${GEN_LENGTHS[@]}"; do
  for step_factor in "${STEPS_FACTORS[@]}"; do
    for task in "${TASKS[@]}"; do
      for alpha_soft_mask in "${ALPHA_SOFT_MASKS[@]}"; do
      steps=$(echo "$gen_length * $step_factor" | bc | cut -d. -f1)

      output_dir="${output_dir_base}_kSoft_${k_soft}_alphaSoftMask_${alpha_soft_mask}"

      echo "Running evaluation on $task with gen_length=$gen_length, steps=$steps, k_soft=$k_soft, alpha_soft_mask=$alpha_soft_mask"

      CUDA_VISIBLE_DEVICES=$GPU_LIST PYTHONPATH=.:$PYTHONPATH torchrun \
          --nproc_per_node $NUM_GPUS \
          --master_port $MASTER_PORT \
          eval/evaluate.py \
          --dataset $task \
          --batch_size 1 \
          --gen_length $gen_length \
          --diffusion_steps $steps \
          --output_dir $output_dir \
          --model_path $model_path \
          --checkpoint_path $checkpoint_path \
          --temperature 0.5 \
          --k_soft $k_soft \
          --alpha_soft_mask $alpha_soft_mask
      done
    done
  done
done
