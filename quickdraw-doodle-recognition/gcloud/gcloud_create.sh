#!/bin/bash

export ZONE="us-west1-a"
export IMAGE_FAMILY="tf-1-12-cu100"
export MACHINE_TYPE="n1-standard-4"
export INSTANCE_NAME="kaggle"

gcloud compute instances create $INSTANCE_NAME \
       --zone $ZONE \
       --image-project=deeplearning-platform-release \
       --image-family=$IMAGE_FAMILY \
       --machine-type=$MACHINE_TYPE \
       --accelerator="type=nvidia-tesla-v100,count=1" \
       --boot-disk-type=pd-ssd \
       --boot-disk-size=256GB \
       --maintenance-policy=TERMINATE \
       --metadata="install-nvidia-driver=True"
