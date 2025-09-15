export MICRO_BATCH_SIZE=64
export RUN_TAG="bs512"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=4
export RUN_TAG="bs32"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=64
export LEARNING_RATE=4e-5
export RUN_TAG="bs512_lr_4e-5"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=4
export LEARNING_RATE=4e-5
export RUN_TAG="bs32_lr_4e-5"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

cd /home/andrewor/local/ao
git checkout nvfp4-temp
cd -

export MICRO_BATCH_SIZE=64
export RUN_TAG="bs512_before_#3050"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=4
export RUN_TAG="bs32_before_#3050"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=64
export LEARNING_RATE=4e-5
export RUN_TAG="bs512_lr_4e-5_before_#3050"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh

export MICRO_BATCH_SIZE=4
export LEARNING_RATE=4e-5
export RUN_TAG="bs32_lr_4e-5_before_#3050"
ENABLE_QAT="true" ./run_it.sh
ENABLE_QAT="false" ./run_it.sh
