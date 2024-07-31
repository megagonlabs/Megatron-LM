#!/bin/bash

set -e

TOKENIZER_MODEL=$1
INPUT_JSON_GZ_PREFIX=$2
OUTPUT_PREFIX=$3

PARTITIONS=`ls ${INPUT_JSON_GZ_PREFIX}_*.jsonl.gz | wc -l` 

python tools/preprocess_data.py \
  --input ${INPUT_JSON_GZ_PREFIX}.jsonl.gz \
  --partitions ${PARTITIONS} \
  --workers ${PARTITIONS} \
  --dataset-impl mmap \
  --output-prefix ${OUTPUT_PREFIX} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --tokenizer-type SentencePieceTokenizer \
  --split-sentences

rm ${INPUT_JSON_GZ_PREFIX}_ss_*.jsonl
rm ${OUTPUT_PREFIX}_*_text_sentence.*
