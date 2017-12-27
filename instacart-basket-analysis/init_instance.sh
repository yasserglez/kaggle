#!/usr/bin/env bash

source instance.sh

ssh -i "$KEY" "$USER@$HOST" "sudo apt-get update"
ssh -i "$KEY" "$USER@$HOST" "sudo apt-get install htop python3 python3-pip python3-dev"

ssh -i "$KEY" "$USER@$HOST" "mkdir -p $WORKDIR/input"
ssh -i "$KEY" "$USER@$HOST" "mkdir -p $WORKDIR/output"
ssh -i "$KEY" "$USER@$HOST" "mkdir -p $WORKDIR/output/models/evaluation"
ssh -i "$KEY" "$USER@$HOST" "mkdir -p $WORKDIR/output/models/submission"

scp -i "$KEY" -r requirements.txt "$USER@$HOST:$WORKDIR"
ssh -i "$KEY" "$USER@$HOST" "sudo pip3 install --upgrade pip"
ssh -i "$KEY" "$USER@$HOST" "sudo pip3 install -r $WORKDIR/requirements.txt"

scp -i "$KEY" -r input/products.csv "$USER@$HOST:$WORKDIR/input/products.csv"

scp -i "$KEY" -r output/*.gz "$USER@$HOST:$WORKDIR/output"
ssh -i "$KEY" "$USER@$HOST" "cd $WORKDIR/output; gunzip -k *.gz"

ssh -i "$KEY" "$USER@$HOST" "rm -r $WORKDIR/pipelines"
scp -i "$KEY" -r pipelines "$USER@$HOST:$WORKDIR"

# TENSORFLOW="tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl"
# scp -i "$KEY" "$TENSORFLOW" "$USER@$HOST:"
# ssh -i "$KEY" "$USER@$HOST" "sudo pip3 install --upgrade $TENSORFLOW"

scp -i "$KEY" -r run.sh "$USER@$HOST:$WORKDIR"
