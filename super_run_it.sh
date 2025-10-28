export SKIP_FINETUNE="true"

#export MICRO_BATCH_SIZE=16
#export MODEL="Gemma3-12B"
#export RUN_TAG="gemma3_12b_bs128"
#ENABLE_QAT="true" ./run_it.sh
#ENABLE_QAT="false" ./run_it.sh
#
#export MICRO_BATCH_SIZE=4
#export MODEL="Gemma3-12B"
#export RUN_TAG="gemma3_12b_bs32"
#ENABLE_QAT="true" ./run_it.sh
#ENABLE_QAT="false" ./run_it.sh
#
#export MICRO_BATCH_SIZE=16
#export MODEL="Gemma3-12B"
#export LEARNING_RATE="4e-5"
#export RUN_TAG="gemma3_12b_bs128_lr4e-5"
#ENABLE_QAT="true" ./run_it.sh
#ENABLE_QAT="false" ./run_it.sh
#
#export MICRO_BATCH_SIZE=4
#export MODEL="Gemma3-12B"
#export LEARNING_RATE="4e-5"
#export RUN_TAG="gemma3_12b_bs32_lr4e-5"
#ENABLE_QAT="true" ./run_it.sh
#ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=32
export MODEL="Qwen3-8B"
export RUN_TAG="qwen3_8b_bs256"
SKIP_EVAL_FLOAT="true" ENABLE_QAT="false" ./run_it.sh
SKIP_EVAL_FLOAT="false" ENABLE_QAT="true" ./run_it.sh

export SKIP_EVAL_FLOAT="true"

export MICRO_BATCH_SIZE=4
export MODEL="Qwen3-8B"
export RUN_TAG="qwen3_8b_bs32"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=32
export MODEL="Qwen3-8B"
export LEARNING_RATE="4e-5"
export RUN_TAG="qwen3_8b_bs256_lr4e-5"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=4
export MODEL="Qwen3-8B"
export LEARNING_RATE="4e-5"
export RUN_TAG="qwen3_8b_bs32_lr4e-5"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh
