#!/bin/bash

set -e

TOKENIZER_MODEL=$1
JA_CC_DIR=$2
LEVEL=$3
PARTITION_ZERO_DIGITS=$4
OUTPUT_BASE_DIR=$5

for SUBSET in $(ls ${JA_CC_DIR}/${LEVEL})
do
  jawze_bert/data_build.sh ${TOKENIZER_MODEL} ${JA_CC_DIR}/${LEVEL}/${SUBSET}/${SUBSET} ${PARTITION_ZERO_DIGITS} ${OUTPUT_BASE_DIR}/${LEVEL}-${SUBSET}
done