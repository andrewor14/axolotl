# How to reproduce NVFP4 QAT results

## Setup

You will need:
- 8x B200 GPUs
- This axolotl branch: https://github.com/andrewor14/axolotl/tree/andrew-testing
- The latest torchao commit: https://github.com/pytorch/ao (nightly)
- PyTorch 2.8.0+

```
# Install axolotl first, otherwise you may override your torch and torchao installations

# Install torch for cuda 12.8, probably also works for other cuda versions
pip install torch --index-url https://download.pytorch.org/whl/cu128/

# Either build torchao from source (takes <1 minute)
git clone git@github.com:pytorch/ao
USE_CPP=0 python setup.py develop

# Or get the nightly torchao wheel for cuda 12.8
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128
```

## High-level usage

1. Change all hard-coded paths manually

```
git grep andrewor
```

2. Launch the experiments

```
# This calls run_it.sh a bunch of times
# Each call to run_it.sh does a full run of fine-tuning + quantize + eval
./super_run_it.sh

# For each experiment, you can set RUN_TAG to annotate the experiment
# For example, RUN_TAG="alpaca_lr2e-4_bs128" means your logs will be at
# /home/andrewor/local/logs/axolotl/Gemma3-12B_qat_alpaca_lr2e-4_bs128

# You can configure each experiment using these env vars:
#   MODEL: "Llama3.2-3B" (default)
#       also supports "Llama3.1-8B", "Qwen3-8B", and "Gemma3-12B"
#   EVAL_TASKS: "wikitext,bbh,mmlu_pro" (default)
#       for hyperparameter tuning, change to "wikitext,mmlu" (much faster)
#   QAT_SCHEME: "nvfp4" (default) or "int8-int4"
#   ENABLE_QAT: "true" (default) or "false"
#   MICRO_BATCH_SIZE: 64 (default), per GPU
#   LEARNING_RATE: 2e-5 (default), scale with batch size generally
#   MAX_STEPS: -1 (default), this means 1 epoch
```

3. Parse the results

```
# Once you finish running the experiments, you can run this script
# to parse the QAT improvements automatically:
find /home/andrewor/local/logs/axolotl -maxdepth 1 -type d | xargs python parse_it.py

# Example output:
# bs256: 10.7462 (baseline) -> 12.0319 (quant) -> 11.9486 (qat), recovered 6.479% (wikitext)
# bs256: 0.779 (baseline) -> 0.7291 (quant) -> 0.7334 (qat), recovered 8.617% (bbh)
# bs256: 0.4956 (baseline) -> 0.4506 (quant) -> 0.4688 (qat), recovered 40.444% (mmlu_pro)
```

## Adding new models or datasets

For new models, you will need to add an if case in `run_it.sh` that specifies
the `BASE_MODEL` and the `CONFIG` to pass to axolotl, for example:

```
# In run_it.sh
...
elif [[ "$MODEL" == "Gemma3-12B" ]]; then
    BASE_MODEL="google/gemma-3-12b-it"
    CONFIG="examples/gemma3/salman-gemma3-12b.yml"
else
...

# Later in the script we call this
axolotl train "$CONFIG" --base-model "$BASE_MODEL" ...
```

For new datasets, either specify it in your config or add code in `run_it.sh`
to pass a new flag to axolotl. Probably easier to just specify in the config.
