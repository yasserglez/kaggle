#!/usr/bin/env bash

source instance.sh

RESULTSDIR="results_$(date +%Y-%m-%d-%H-%M-%S)"

mkdir -p "$RESULTSDIR/models"
scp -i "$KEY" -r "$USER@$HOST:$WORKDIR/output/models" "$RESULTSDIR/"
