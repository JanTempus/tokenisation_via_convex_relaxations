#!/bin/bash
#SBATCH -A a0229
#SBATCH -p normal
#SBATCH -t 2:30:00
#SBATCH --output=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_via_convex_relaxations/logs/english_fineweb_output_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_via_convex_relaxations/logs/english_fineweb_output_%j.err

# Downloads the first NUM_SHARDS shards of Karpathy's FineWeb-Edu
# (karpathy/fineweb-edu-100b-shuffle) once, then trains plain-compression LP tokenizers
# (PENTLY_RHO=0, no CELEX/Unicode penalties) for each size in VOCAB_SIZES.
#
# Submit:   sbatch training_files/english_fineweb_run_lp.sbatch
# Override: NUM_SHARDS=10 VOCAB_SIZES=65536 sbatch training_files/english_fineweb_run_lp.sbatch

set -euo pipefail

# Rename the open Slurm logs at startup; SBATCH directives do not expand dates.
LOG_DIR="/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_via_convex_relaxations/logs"
LOG_DATE="$(date +%Y-%m-%d_%H-%M-%S)"
for LOG_EXT in out err; do
    mv -- "${LOG_DIR}/english_fineweb_output_${SLURM_JOB_ID}.${LOG_EXT}" \
        "${LOG_DIR}/english_fineweb_output_${LOG_DATE}_${SLURM_JOB_ID}.${LOG_EXT}"
done

# User-tunable settings
NAME="${NAME:-KarpathyFineWeb}"
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_via_convex_relaxations}"
IOPS_ROOT="${IOPS_ROOT:-/iopsstor/scratch/cscs/jtempus/tokenisation-lp}"
# Every non-empty document in these shards is used for training.
NUM_SHARDS="${NUM_SHARDS:-7}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="${VOCAB_SIZES:-32768,65536}"
NUM_PROC="${NUM_PROC:-64}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
# Number of GPUs visible to the solve process to use together for PDLP.
LP_NUM_GPUS="${LP_NUM_GPUS:-4}"
if [[ ! "${LP_NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: LP_NUM_GPUS must be a positive integer" >&2
    exit 1
fi
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

# Fail before the download if the allocation has fewer GPUs than LP_NUM_GPUS.
VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
if (( VISIBLE_GPUS < LP_NUM_GPUS )); then
    echo "ERROR: need ${LP_NUM_GPUS} GPUs but found ${VISIBLE_GPUS}" >&2
    exit 1
fi

DATASET_TAG="${NAME}_shards${NUM_SHARDS}"
RUN_DIR="${IOPS_ROOT}/runs/${DATASET_TAG}_${SLURM_JOB_ID}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-${IOPS_ROOT}/datasets/${DATASET_TAG}}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${OUTPUT_DATASET_DIR}}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-${IOPS_ROOT}/raw_vocabs/${DATASET_TAG}_compression}"
LP_GRAPH_CACHE_DIR="${LP_GRAPH_CACHE_DIR:-${RUN_DIR}/lp_graph}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-${REPO_DIR}/rounded_tokenizers/${DATASET_TAG}_compression}"

TMPDIR="/iopsstor/scratch/cscs/jtempus/tmp/${SLURM_JOB_ID}"
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
    "${LP_GRAPH_CACHE_DIR}" \
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
export NUM_SHARDS
export OUTPUT_DATASET_DIR
export TRAIN_DATASET_PATH
export PRETOKENIZER_MODE
export VOCAB_SIZES
export NUM_PROC
export BATCH_SIZE
export LP_NUM_GPUS
export LP_GRAPH_CACHE_DIR
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
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
export PENTLY_RHO=0

echo "Run tag:              ${DATASET_TAG}"
echo "Repository:           ${REPO_DIR}"
echo "Job working dir:      ${RUN_DIR}"
echo "Training dataset:     ${TRAIN_DATASET_PATH}"
echo "Vocabulary sizes:     ${VOCAB_SIZES}"
echo "Raw vocabulary:       ${RAW_VOCAB_PATH}"
echo "LP graph cache:       ${LP_GRAPH_CACHE_DIR}"
echo "Final tokenizers:     ${SAVE_TOKENIZER_DIR}"
echo "Map workers:          ${NUM_PROC}"
echo "Map batch size:       ${BATCH_SIZE}"
echo "GPUs per PDLP solve:  ${LP_NUM_GPUS}"

if [[ -f "${TRAIN_DATASET_PATH}/sampling_manifest.json" ]]; then
    echo "Step 1/3: reusing existing dataset at ${TRAIN_DATASET_PATH}"
elif [[ "${TRAIN_DATASET_PATH}" != "${OUTPUT_DATASET_DIR}" ]]; then
    echo "ERROR: overridden TRAIN_DATASET_PATH has no sampling manifest: ${TRAIN_DATASET_PATH}" >&2
    exit 1
else
    echo "Step 1/3: downloading first ${NUM_SHARDS} Karpathy FineWeb-Edu shards -> ${OUTPUT_DATASET_DIR}"
    python3 -u "${REPO_DIR}/sample_tokenizer_data.py" \
        --source karpathy_fineweb \
        --name "${NAME}" \
        --num-shards "${NUM_SHARDS}" \
        --output-dataset-dir "${OUTPUT_DATASET_DIR}"
fi

echo "Step 2/3: training LP vocab(s) -> ${RAW_VOCAB_PATH}"
python3 -u -m lp_tokenizer.training

echo "Step 3/3: rounding/exporting LP tokenizer(s) -> ${SAVE_TOKENIZER_DIR}"
python3 -u -m lp_tokenizer.export
