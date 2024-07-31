#!/bin/bash

set -e

TOKENIZER_MODEL=$1
JA_CC_DIR=$2
LEVEL=$3
OUTPUT_BASE_DIR=$4

for SUBSET in $(ls ${JA_CC_DIR}/${LEVEL})
do
  ./jawze_data_build.sh ${TOKENIZER_MODEL} ${JA_CC_DIR}/${LEVEL}/${SUBSET}/${SUBSET} ${OUTPUT_BASE_DIR}/${LEVEL}-${SUBSET}
done
