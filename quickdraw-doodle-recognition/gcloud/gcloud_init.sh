#!/bin/bash

export ZONE="us-west1-a"
export INSTANCE_NAME="kaggle"

gcloud compute scp --zone $ZONE requirements.txt $INSTANCE_NAME:
gcloud compute ssh --zone $ZONE $INSTANCE_NAME --command="pip3 install -r requirements.txt"

gcloud compute ssh --zone $ZONE $INSTANCE_NAME --command="mkdir -p ~/input"
gcloud compute scp --zone $ZONE input/test_simplified.csv $INSTANCE_NAME:~/input/

gcloud compute ssh --zone $ZONE $INSTANCE_NAME --command="mkdir -p ~/output"
gcloud compute scp --zone $ZONE output/train_simplified.tar.gz $INSTANCE_NAME:~/output/
gcloud compute ssh --zone $ZONE $INSTANCE_NAME --command="cd ~/output; tar -xzvf train_simplified.tar.gz"

gcloud compute ssh --zone $ZONE $INSTANCE_NAME --command="mkdir -p ~/output/convnet"

gcloud compute scp --zone $ZONE common.py $INSTANCE_NAME:
gcloud compute scp --zone $ZONE drawing.py $INSTANCE_NAME:
gcloud compute scp --zone $ZONE convnet.py $INSTANCE_NAME:
gcloud compute scp --zone $ZONE train_convnet.py $INSTANCE_NAME:
gcloud compute scp --zone $ZONE onecycle.py $INSTANCE_NAME:
gcloud compute scp --zone $ZONE predict_convnet.py $INSTANCE_NAME:
