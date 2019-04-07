#!/bin/bash

CONFIG_FILE=submission1.json
SUBMISSION_DIR=output/submission1
NUM_MODELS=100

# Step 1: Train the models.

python hyperparam_search.py \
       --config_space $CONFIG_FILE \
       --algorithm rand \
       --max_trials $NUM_MODELS \
       --output $SUBMISSION_DIR

# Step 2: Generate the predictions.

for i in $(seq 1 $NUM_MODELS); do
    python predict.py --model $SUBMISSION_DIR/trial_$i
done

# Step 3: Ensemble the predictions.

python ensemble.py --models $SUBMISSION_DIR
