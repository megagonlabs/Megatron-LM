#!/bin/bash

set -e

TOKENIZER_MODEL=$1
INPUT_JSON_GZ_PREFIX=$2
PARTITION_ZERO_DIGITS=$3
OUTPUT_PREFIX=$4

PARTITIONS=`ls ${INPUT_JSON_GZ_PREFIX}_*.jsonl.gz | wc -l` 

python tools/preprocess_data.py \
  --input ${INPUT_JSON_GZ_PREFIX}.jsonl.gz \
  --partitions ${PARTITIONS} \
  --partition-zero-digits ${PARTITION_ZERO_DIGITS} \
  --workers ${PARTITIONS} \
  --dataset-impl mmap \
  --output-prefix ${OUTPUT_PREFIX} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --tokenizer-type SentencePieceTokenizer \
  --split-sentences

rm ${INPUT_JSON_GZ_PREFIX}_ss_*.jsonl
rm ${OUTPUT_PREFIX}_*_text_sentence.*
