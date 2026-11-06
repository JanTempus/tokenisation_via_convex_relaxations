#!/bin/bash

# Downloads English FineWeb once, then trains two LP tokenizer families on it:
#   compression: PENTLY_RHO=0, the plain compression objective.
#   morph:       PENTLY_RHO=MORPH_RHO, adds the CELEX morphology and Unicode penalties.
#
# Run on a GPU node, inside tmux/screen so a dropped connection does not kill it:
#   bash training_files/english_fineweb_run_lp.sh
# Override: TARGET_ROWS=60000 MORPH_RHO=1 bash training_files/english_fineweb_run_lp.sh

set -euo pipefail

# Identifies this run; uses the job ID when started inside a Slurm allocation.
RUN_ID="${SLURM_JOB_ID:-$$}"

# User-tunable settings
NAME="${NAME:-FineWebEnglish}"
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_via_convex_relaxations}"
IOPS_ROOT="${IOPS_ROOT:-/iopsstor/scratch/cscs/jtempus/tokenisation-lp}"
TARGET_ROWS="${TARGET_ROWS:-8000}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="${VOCAB_SIZES:-8192,16384,32768,65536,131072,262144}"
# Shared CELEX/Unicode penalty weight for the morph tokenizers; must be positive.
MORPH_RHO="${MORPH_RHO:-0.5}"
# Must contain english/eml/eml.cd and english/emw/emw.cd.
CELEX_DIR="${CELEX_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/lp-tokeniser-experiments/CELEX_V2}"
NUM_PROC="${NUM_PROC:-256}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
# Number of GPUs visible to the solve process to use together for PDLP.
LP_NUM_GPUS="${LP_NUM_GPUS:-4}"
if [[ ! "${LP_NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: LP_NUM_GPUS must be a positive integer" >&2
    exit 1
fi
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

# Mirror all output to a log file, since no scheduler captures it here.
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/english_fineweb_output_$(date +%Y-%m-%d_%H-%M-%S)_${RUN_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Nothing reserves GPUs for this script, so check before starting the download.
VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
if (( VISIBLE_GPUS < LP_NUM_GPUS )); then
    echo "ERROR: need ${LP_NUM_GPUS} GPUs but found ${VISIBLE_GPUS}; run this on a GPU node" >&2
    exit 1
fi

# Check morphology data before downloading or training anything.
for CELEX_FILE in "${CELEX_DIR}/english/eml/eml.cd" "${CELEX_DIR}/english/emw/emw.cd"; do
    if [[ ! -r "${CELEX_FILE}" ]]; then
        echo "ERROR: CELEX file is missing or unreadable: ${CELEX_FILE}" >&2
        exit 1
    fi
done

DATASET_TAG="${NAME}_n${TARGET_ROWS}"
RUN_DIR="${IOPS_ROOT}/runs/${DATASET_TAG}_${RUN_ID}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-${IOPS_ROOT}/datasets/${DATASET_TAG}}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${OUTPUT_DATASET_DIR}}"

TMPDIR="/iopsstor/scratch/cscs/jtempus/tmp/${RUN_ID}"
TMP="${TMPDIR}"
TEMP="${TMPDIR}"
PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
XDG_CACHE_HOME="${IOPS_ROOT}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

mkdir -p \
    "${RUN_DIR}" \
    "${TMPDIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${IOPS_ROOT}/datasets" \
    "${IOPS_ROOT}/raw_vocabs" \
    "${HF_DATASETS_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" \
    "${MPLCONFIGDIR}"

source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${RUN_DIR}"

# Training runs from RUN_DIR, so pin imports to this checkout.
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export NAME
export REPO_DIR
export IOPS_ROOT
export RUN_DIR
export TARGET_ROWS
export OUTPUT_DATASET_DIR
export TRAIN_DATASET_PATH
export PRETOKENIZER_MODE
export VOCAB_SIZES
export CELEX_DIR
export NUM_PROC
export BATCH_SIZE
export LP_NUM_GPUS
export TMPDIR
export TMP
export TEMP
export PYTHONPYCACHEPREFIX
export XDG_CACHE_HOME
export HF_HOME
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE
export CUDA_CACHE_PATH
export NUMBA_CACHE_DIR
export MPLCONFIGDIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR
export VOCAB_UTILISATION_WEIGHT=0

echo "Run tag:              ${DATASET_TAG}"
echo "Repository:           ${REPO_DIR}"
echo "Job working dir:      ${RUN_DIR}"
echo "Log file:             ${LOG_FILE}"
echo "Training dataset:     ${TRAIN_DATASET_PATH}"
echo "Vocabulary sizes:     ${VOCAB_SIZES}"
echo "Morph penalty rho:    ${MORPH_RHO}"
echo "CELEX directory:      ${CELEX_DIR}"
echo "Map workers:          ${NUM_PROC}"
echo "Map batch size:       ${BATCH_SIZE}"
echo "GPUs per PDLP solve:  ${LP_NUM_GPUS}"

if [[ -f "${TRAIN_DATASET_PATH}/sampling_manifest.json" ]]; then
    echo "Step 1/3: reusing existing dataset at ${TRAIN_DATASET_PATH}"
elif [[ "${TRAIN_DATASET_PATH}" != "${OUTPUT_DATASET_DIR}" ]]; then
    echo "ERROR: overridden TRAIN_DATASET_PATH has no sampling manifest: ${TRAIN_DATASET_PATH}" >&2
    exit 1
else
    echo "Step 1/3: downloading ordered English FineWeb prefix -> ${OUTPUT_DATASET_DIR}"
    python3 -u "${REPO_DIR}/sample_tokenizer_data.py" \
        --source fineweb \
        --name "${NAME}" \
        --target-rows "${TARGET_ROWS}" \
        --output-dataset-dir "${OUTPUT_DATASET_DIR}"
fi

# Each variant gets its own outputs and LP cache, so the two never share Arrow files.
train_variant() {
    local step="$1"
    local variant="$2"
    export PENTLY_RHO="$3"
    export RAW_VOCAB_PATH="${IOPS_ROOT}/raw_vocabs/${DATASET_TAG}_${variant}"
    export SAVE_TOKENIZER_DIR="${REPO_DIR}/rounded_tokenizers/${DATASET_TAG}_${variant}"
    export LP_GRAPH_CACHE_DIR="${RUN_DIR}/lp_graph/${variant}"
    mkdir -p "${LP_GRAPH_CACHE_DIR}"

    echo "Step ${step}/3: training ${variant} LP vocab(s), rho=${PENTLY_RHO} -> ${RAW_VOCAB_PATH}"
    python3 -u -m lp_tokenizer.training

    echo "Step ${step}/3: rounding/exporting ${variant} tokenizer(s) -> ${SAVE_TOKENIZER_DIR}"
    python3 -u -m lp_tokenizer.export
}

train_variant 2 compression 0
train_variant 3 "morph_rho${MORPH_RHO}" "${MORPH_RHO}"
