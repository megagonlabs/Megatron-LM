#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1
TOKENIZER_MODEL=$2
LOGS_PATH=$3
DATA_PATH=${@:4:($#-3)}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BERT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 8 \
    --global-batch-size 192 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16
"

DATA_ARGS="
    --num-workers 16 \
    --data-path $DATA_PATH \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-size 44544 \
    --data-impl mmap \
    --split 998,1,1
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 40
"

torchrun $DISTRIBUTED_ARGS pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH &> $LOGS_PATH/jawze_pretrain_`date +%Y%m%d-%H%M%S`
