#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

set -x

PARTITION=$1
CONFIG=$2
GPUS=${GPUS:-1}
CPUS=${CPUS:-8}
MEM=${MEM:-128G}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --gres=gpu:${GPUS} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --cpus-per-task=${CPUS} \
    --kill-on-bad-exit=1 \
    --container-image=/netscratch/$USER/enroot/mmpose.sqsh \
    --container-mounts=/home/$USER:/home/$USER,/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`,/ds-av:/ds-av" \
    --container-workdir="`pwd`" \
    --mem=${MEM} \
    ${SRUN_ARGS} \
    python -u tools/train_continual.py ${CONFIG} --launcher="slurm" ${@:3}
