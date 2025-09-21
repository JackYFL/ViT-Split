#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
# PORT=${PORT:-35422}
PORT=15300

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./tools/new_train.py $CONFIG --launcher pytorch --deterministic ${@:3}
