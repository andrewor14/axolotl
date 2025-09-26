LOG_DIR_BASE="/home/andrewor/local/logs/axolotl"

MODEL="${MODEL:-Llama3.2-3B}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
ENABLE_QAT="${ENABLE_QAT:-true}"
MAX_STEPS="${MAX_STEPS:--1}"
QAT_SCHEME="${QAT_SCHEME:-nvfp4}"
EVAL_TASKS="wikitext,bbh,mmlu_pro"

if [[ "$QAT_SCHEME" == "nvfp4" ]]; then
    ACTIVATION_DTYPE="nvfp4"
    WEIGHT_DTYPE="nvfp4"
    GROUP_SIZE="16"
elif [[ "$QAT_SCHEME" == "int8-int4" ]]; then
    ACTIVATION_DTYPE="int8"
    WEIGHT_DTYPE="int4"
    GROUP_SIZE="32"
else
    echo "Unknown QAT_SCHEME $QAT_SCHEME"
    exit 1
fi
QAT_ARG="{'activation_dtype':'$ACTIVATION_DTYPE', 'weight_dtype':'$WEIGHT_DTYPE', 'group_size':$GROUP_SIZE}"

if [[ "$ENABLE_QAT" == "true" ]]; then
    LOG_DIR="${LOG_DIR_BASE}/${MODEL}_qat"
    TRAIN_QAT_ARG="$QAT_ARG"
else
    LOG_DIR="${LOG_DIR_BASE}/${MODEL}_baseline"
    TRAIN_QAT_ARG="None"
fi
if [[ -n "$RUN_TAG" ]]; then
    LOG_DIR="${LOG_DIR}_${RUN_TAG}"
fi

if [[ "$MODEL" == "Llama3.2-3B" ]]; then
    BASE_MODEL="meta-llama/Llama-3.2-3B"
    CONFIG="examples/llama-3/3b-qat-fsdp2-nvfp4.yaml"
elif [[ "$MODEL" == "Llama3.1-8B" ]]; then
    BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    CONFIG="examples/llama-3/3b-qat-fsdp2-nvfp4.yaml"  # is this right?
elif [[ "$MODEL" == "Qwen3-8B" ]]; then
    BASE_MODEL="Qwen/Qwen3-8B"
    CONFIG="examples/qwen3/salman-qwen3-8b.yml"
elif [[ "$MODEL" == "Gemma3-12B" ]]; then
    BASE_MODEL="google/gemma-3-12b-it"
    CONFIG="examples/gemma3/salman-gemma3-12b.yml"
else
    echo "Unknown MODEL $MODEL"
    exit 1
fi

if [[ "$SKIP_FINETUNE" != "true" ]]; then
    rm -rf "$LOG_DIR"
    mkdir -p "$LOG_DIR"

    # Record torchao and axolotl commits
    cd /home/andrewor/local/ao
    echo "torchao commit: " >> "${LOG_DIR}/run.log" 2>&1
    git log --oneline | head -n 1 >> "${LOG_DIR}/run.log" 2>&1
    git branch >> "${LOG_DIR}/run.log" 2>&1
    git diff >> "${LOG_DIR}/run.log" 2>&1
    cd /home/andrewor/local/axolotl
    echo "axolotl commit: " >> "${LOG_DIR}/run.log" 2>&1
    git log --oneline | head -n 1 >> "${LOG_DIR}/run.log" 2>&1
    git branch >> "${LOG_DIR}/run.log" 2>&1
    git diff >> "${LOG_DIR}/run.log" 2>&1

    # Finetune
    axolotl train "$CONFIG" \
        --base-model "$BASE_MODEL" \
        --output-dir "$LOG_DIR" \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --max-steps "$MAX_STEPS" \
        --qat "$TRAIN_QAT_ARG" \
        --evals-per-epoch 0 \
        >> "${LOG_DIR}/run.log" 2>&1
fi

if [[ "$SKIP_EVAL" != "true" ]]; then
    axolotl quantize "$CONFIG" \
        --base-model "$LOG_DIR" \
        --output-dir "$LOG_DIR" \
        --activation-dtype "$ACTIVATION_DTYPE" \
        --weight-dtype "$WEIGHT_DTYPE" \
        --group-size "$GROUP_SIZE" \
        > "${LOG_DIR}/quantize.log" 2>&1
    accelerate launch -m lm_eval --model hf --model_args pretrained="${LOG_DIR}",weights_only=False --tasks "$EVAL_TASKS" --batch_size auto > "${LOG_DIR}/eval_float.log" 2>&1
    accelerate launch -m lm_eval --model hf --model_args pretrained="${LOG_DIR}/quantized",weights_only=False --tasks "$EVAL_TASKS" --batch_size auto > "${LOG_DIR}/eval_quantized.log" 2>&1
fi
