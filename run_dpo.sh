#!/usr/bin/env bash
set -e

# -----------------------------
# 1. Parse model argument
# -----------------------------
MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash run_dpo.sh --model <model_name_or_path>"
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required"
  echo "Usage: bash run_dpo.sh --model <model_name_or_path>"
  exit 1
fi

# -----------------------------
# 2. Generate output_dir based on model
# -----------------------------
MODEL_NAME=$(basename "$MODEL")
OUTPUT_DIR="./outcomes/${MODEL_NAME}"

# -----------------------------
# 2.1 Determine if it is LLaMA2 from the path string
# -----------------------------
MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')

MODEL_TYPE_ARGS=""
if [[ "$MODEL_LOWER" == *"llama2"* || "$MODEL_LOWER" == *"llama-2"* ]]; then
  MODEL_TYPE_ARGS="--model_type llama"
fi

# -----------------------------
# 3. Distributed Environment
# -----------------------------
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=$((29500 + RANDOM % 1000))
export NPROC_PER_NODE=4

# -----------------------------
# 4. Print Key messages
# -----------------------------
echo "=============================="
echo "MODEL        = $MODEL"
echo "MODEL_NAME   = $MODEL_NAME"
echo "OUTPUT_DIR   = $OUTPUT_DIR"
echo "MODEL_TYPE   = ${MODEL_TYPE_ARGS:-default}"
echo "GPUS         = $CUDA_VISIBLE_DEVICES"
echo "NPROC        = $NPROC_PER_NODE"
echo "=============================="

mkdir -p "$OUTPUT_DIR"

# -----------------------------
# 5. Start DPO training
# -----------------------------
swift rlhf \
  --rlhf_type dpo \
  --model "$MODEL" \
  $MODEL_TYPE_ARGS \
  --output_dir "$OUTPUT_DIR" \
  --dataset ./data/paraphrase_data.jsonl \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --system 'You are a helpful assistant.' \
  --train_type lora \
  --load_from_cache_file true \
  --save_strategy steps \
  --save_steps 78 \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --save_total_limit 14 \
  --logging_steps 5 \
  --max_length 1024 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --rpo_alpha 0.0 \
  --dataset_num_proc 4 \
  --deepspeed zero2
