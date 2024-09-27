#!/bin/bash

set -e

MEGATRON_CHECKPOINTS_ROOT_DIR=$1
HUGGINGFACE_TOKENIZER_DIR=$2
HUGGINGFACE_MODEL_DIR=$3

MEGATRON_CHECKPOINT_TP1_PP1=$1.tp1-pp1

python tools/checkpoint_util.py --model-type BERT \
  --load-dir $MEGATRON_CHECKPOINTS_ROOT_DIR \
  --save-dir $MEGATRON_CHECKPOINTS_TP1_PP1 \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1

pip install transformers

python -m transformers.models.megatron_bert.convert_megatron_bert_checkpoint \
  --print-checkpoint-structure \
  $MEGATRON_CHECKPOINTS_TP1_PP1

mkdir -p $HUGGINGFACE_MODEL_DIR
cp $HUGGINGFACE_MODEL_DIR/iter_*/mp_rank_00/config.json $HUGGINGFACE_MODEL_DIR/
cp $HUGGINGFACE_MODEL_DIR/iter_*/mp_rank_00/pytorch_model.bin $HUGGINGFACE_MODEL_DIR/
cp $HUGGINGFACE_TOKENIZER_DIR/* $HUGGINGFACE_MODEL_DIR/
rm -r $MEGATRON_CHECKPOINT_TP1_PP1
