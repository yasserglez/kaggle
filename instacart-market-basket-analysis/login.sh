#!/usr/bin/env bash

source instance.sh

ssh -i "$KEY" "$USER@$HOST"
