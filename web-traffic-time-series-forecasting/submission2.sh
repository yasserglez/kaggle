#!/usr/bin/env bash

python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --num-days-before 30 --random-seed 2766306141
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --num-days-before 90 --random-seed 2517504021
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --num-days-before 180 --random-seed 1532850297
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --num-days-before 270 --random-seed 1406174211
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --num-days-before 365 --random-seed 4221485063

python3 -m pipelines.submissions Submission2
