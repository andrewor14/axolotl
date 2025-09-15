LOG_DIR_BASE="/home/andrewor/local/logs/axolotl"

MODEL="Llama3.2-3B"
CONFIG="${CONFIG:-examples/llama-3/3b-qat-fsdp2-nvfp4.yaml}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
ENABLE_QAT="${ENABLE_QAT:-true}"
MAX_STEPS="${MAX_STEPS:--1}"
QAT_SCHEME="${QAT_SCHEME:-nvfp4}"

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

if [[ "$SKIP_FINETUNE" != "true" ]]; then
    rm -rf "$LOG_DIR"
    mkdir -p "$LOG_DIR"
    axolotl train "$CONFIG" \
        --output-dir "$LOG_DIR" \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --max-steps "$MAX_STEPS" \
        --qat "$TRAIN_QAT_ARG" \
        --evals-per-epoch 0 \
        > "${LOG_DIR}/run.log" 2>&1
fi

if [[ "$SKIP_EVAL" != "true" ]]; then
    axolotl quantize "$CONFIG" \
        --base-model "$LOG_DIR" \
        --output-dir "$LOG_DIR" \
        --activation-dtype "$ACTIVATION_DTYPE" \
        --weight-dtype "$WEIGHT_DTYPE" \
        --group-size "$GROUP_SIZE" \
        > "${LOG_DIR}/quantize.log" 2>&1
    CUDA_VISIBLE_DEVICES=2 accelerate launch -m lm_eval --model hf --model_args pretrained="${LOG_DIR}",weights_only=False --tasks wikitext --batch_size 2 > "${LOG_DIR}/eval_float.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 accelerate launch -m lm_eval --model hf --model_args pretrained="${LOG_DIR}/quantized",weights_only=False --tasks wikitext --batch_size 2 > "${LOG_DIR}/eval_quantized.log" 2>&1 &
    wait
fi
