#!/usr/bin/env bash
set -e

# =========================================================
# 0. The script's own directory (to ensure path stability)
# =========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================================================
# 1. Parse command-line arguments
# =========================================================
TASK=""
PRETRAINED=""
PEFT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --pretrained)
      PRETRAINED="$2"
      shift 2
      ;;
    --peft)
      PEFT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage:"
      echo "  bash generation_eval.sh --task <task> --pretrained <path> [--peft <path>]"
      exit 1
      ;;
  esac
done

if [[ -z "$TASK" || -z "$PRETRAINED" ]]; then
  echo "Error: --task and --pretrained are required"
  exit 1
fi

# =========================================================
# 2. task -> num_fewshot 
# =========================================================
case "$TASK" in
  drop)
    NUM_FEWSHOT=3
    ;;
  gsm8k_cot)
    NUM_FEWSHOT=8
    ;;
  truthfulqa_mc1|hellaswag)
    NUM_FEWSHOT=0
    ;;
  mmlu)
    NUM_FEWSHOT=5
    ;;
  *)
    echo "Error: unsupported task: $TASK"
    echo "Supported tasks: drop, gsm8k_cot, truthfulqa_mc1, hellaswag, mmlu"
    exit 1
    ;;
esac

# =========================================================
# 3. Construct output path
# =========================================================
MODEL_DIR="$(basename "$PRETRAINED")"

if [[ -n "$PEFT" ]]; then
  MODEL_DIR="${MODEL_DIR}+lora"
  PEFT_ARG=",peft=$PEFT"
else
  PEFT_ARG=""
fi

OUTPUT_PATH="${SCRIPT_DIR}/generation_results/${MODEL_DIR}/${TASK}"
mkdir -p "$OUTPUT_PATH"

# =========================================================
# 4. Print key information
# =========================================================
echo "=============================="
echo "TASK        = $TASK"
echo "NUM_FEWSHOT = $NUM_FEWSHOT"
echo "PRETRAINED = $PRETRAINED"
echo "PEFT       = ${PEFT:-none}"
echo "OUTPUT     = $OUTPUT_PATH"
echo "=============================="

# =========================================================
# 5. start lm-eval
# =========================================================
lm-eval \
  --model hf \
  --model_args pretrained=$PRETRAINED${PEFT_ARG},dtype=float16 \
  --tasks $TASK \
  --num_fewshot $NUM_FEWSHOT \
  --batch_size auto \
  --device cuda:0 \
  --output_path "$OUTPUT_PATH" \
  --log_samples
