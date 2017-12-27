#!/usr/bin/env bash

python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --stage 3 --deploy-date 2017-09-10 --num-days-before 30 --random-seed 4208396372
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --stage 3 --deploy-date 2017-09-10 --num-days-before 30 --random-seed 535029318
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --stage 3 --deploy-date 2017-09-10 --num-days-before 30 --random-seed 1226457454
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --stage 3 --deploy-date 2017-09-10 --num-days-before 30 --random-seed 96302290
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --stage 3 --deploy-date 2017-09-10 --num-days-before 30 --random-seed 3391346194
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --stage 3 --deploy-date 2017-09-10 --num-days-before 30 --random-seed 2693560142
python3 -u -m pipelines.models.rnn_v1 PredictRNNv1 --stage 3 --deploy-date 2017-09-10 --num-days-before 30 --random-seed 196017792

python3 -m pipelines.submissions Submission4
